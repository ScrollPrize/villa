from __future__ import annotations

import unittest
import tempfile
import random
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from dinovol_2.loss.ibot_patch_loss import iBOTPatchLoss
from dinovol_2.model.model import DinoVitStudentTeacher
from dinovol_2.ops.point_embeddings import sample_grid_points, sample_selected_grid_points


def tiny_model_config():
    return {
        "model_type": "v3", "embed_dim": 36, "num_heads": 3, "depth": 1,
        "global_crops_size": [16] * 3, "local_crops_size": [8] * 3, "patch_size": [4] * 3,
        "drop_path_rate": 0., "rope_rescale_coords": None,
        "dino_out_dim": 19, "ibot_out_dim": 19,
        "dino_head_hidden_dim": 16, "ibot_head_hidden_dim": 16,
        "dino_head_bottleneck_dim": 7, "ibot_head_bottleneck_dim": 7,
    }


class RNGIsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def tearDown(self):
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)


class SelectedInterpolationTests(RNGIsolatedTestCase):
    def test_model_projects_only_selected_teacher_tokens(self):
        torch.manual_seed(3)
        model = DinoVitStudentTeacher(tiny_model_config()).eval()
        x = torch.randn(4, 1, 16, 16, 16)
        coordinates = torch.rand(4, 64, 3) * 16
        rows = torch.tensor([2, 3, 0, 1])
        indices = torch.tensor([3, 65, 7, 191, 1])
        with torch.no_grad():
            for fused in (False, True):
                outputs = model(
                    x, mask_indices_list=indices, n_masked_patches=3, masked_patch_features=fused,
                    teacher_target_coordinates=coordinates, teacher_target_rows=rows,
                )
                teacher = outputs["teacher"]
                full = model._cross_view_tokens(
                    model.teacher, teacher["global"]["patch_tokens"], coordinates, rows, (16,) * 3
                )
                selected = model.select_masked_tokens(full, indices, 3)
                expected = model.teacher.ibot_head(selected, apply_last_layer=not fused)
                key = "global_masked_patch_features" if fused else "global_masked_patch_projections"
                torch.testing.assert_close(teacher[key], expected)

    def check_selection(self, device, dtype, channels_last=False, coordinate_dtype=torch.float32):
        torch.manual_seed(5)
        grid = torch.randn(4, 5, 6, 7, 37, device=device, dtype=dtype)
        if channels_last:
            grid = grid.permute(0, 4, 1, 2, 3).contiguous().permute(0, 2, 3, 4, 1)
        coordinates = torch.rand(4, 53, 3, device=device, dtype=coordinate_dtype) * 60 - 10
        coordinates[:, :3] = torch.tensor([[-0.5, -0.5, -0.5], [39.5, 47.5, 55.5], [3.5, 3.5, 3.5]], device=device)
        rows = torch.tensor([2, 3, 0, 1], device=device)
        # Includes duplicates and unsorted indices, plus tail channels and out-of-bounds coordinates.
        indices = torch.tensor([211, 0, 108, 2, 54, 0, 17, 109], device=device)
        expected = sample_grid_points(grid[rows], coordinates, (40, 48, 56)).to(dtype).flatten(0, 1)[indices]
        actual = sample_selected_grid_points(grid, coordinates, (40, 48, 56), indices, rows)
        torch.testing.assert_close(actual, expected)
        empty = sample_selected_grid_points(grid, coordinates, (40, 48, 56), indices[:0], rows)
        self.assertEqual(empty.shape, (0, 37))
        self.assertEqual(empty.dtype, dtype)

    def test_cpu_selection_and_gradients(self):
        self.check_selection("cpu", torch.float32)
        grid = torch.randn(2, 3, 4, 5, 7, requires_grad=True)
        coordinates = (torch.rand(2, 11, 3) * 20).requires_grad_()
        rows, indices = torch.tensor([1, 0]), torch.tensor([3, 0, 17, 3])
        expected = sample_grid_points(grid[rows], coordinates, (24, 32, 40)).flatten(0, 1)[indices]
        eg = torch.autograd.grad(expected.square().sum(), (grid, coordinates))
        actual = sample_selected_grid_points(grid, coordinates, (24, 32, 40), indices, rows)
        ag = torch.autograd.grad(actual.square().sum(), (grid, coordinates))
        for a, e in zip(ag, eg):
            torch.testing.assert_close(a, e)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_selection_layouts_and_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for layout in (False, True):
                with self.subTest(dtype=dtype, layout=layout):
                    self.check_selection("cuda", dtype, layout)
        # Preserve higher coordinate precision through the PyTorch fallback.
        self.check_selection("cuda", torch.float32, coordinate_dtype=torch.float64)


class TiledIBOTTests(RNGIsolatedTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_pretrainer_cross_view_backward(self):
        from dinovol_2.pretrain import DinoIBOTPretrainer

        torch.manual_seed(5)
        masks = torch.zeros(4, 64, dtype=torch.bool)
        masks[:2, ::3] = True
        indices = masks.flatten().nonzero().flatten()
        axes = [torch.arange(4) * 4 + 1.5] * 3
        centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, 64, 3)
        batch = {
            "collated_global_crops": torch.randn(4, 1, 16, 16, 16),
            "collated_local_crops": torch.randn(4, 1, 8, 8, 8),
            "collated_masks": masks, "mask_indices_list": indices,
            "n_masked_patches": torch.tensor(indices.numel()),
            "masks_weight": (1. / masks.sum(1).clamp(min=1)).unsqueeze(1).expand_as(masks)[masks],
            "n_global_views": 2, "n_local_views": 2,
            "global_token_targets": centers.expand(4, -1, -1).clone() + .25,
        }
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "device": "cuda", "use_amp": True, "output_dir": directory,
                "max_iterations": 2, "warmup_steps": 0, "batch_size": 2,
                "ibot_fused_last_layer": True, "ibot_masked_loss_chunk_size": 23,
                "ibot_targets": "cross_view", "model": tiny_model_config(),
                "koleo_loss_weight": 0.,
            }
            reference = DinoIBOTPretrainer(config)
            actual = DinoIBOTPretrainer({**config, "ibot_kernel_backend": "triton"})
            # The zero-initialized mask token with 1e-5 LayerScale produces
            # near-zero bottlenecks in this tiny head and overflows its FP16
            # normalization backward. Use a nonzero token as in a trained model.
            with torch.no_grad():
                reference.model_module.student.backbone.mask_token.normal_(std=.02)
            actual.model_module.load_state_dict(reference.model_module.state_dict())
            results = []
            for trainer in (reference, actual):
                # Tiny randomly initialized normalized heads overflow with large initial
                # scales. Large upstream scales are covered by the isolated loss test.
                trainer.scaler = torch.amp.GradScaler("cuda", init_scale=1)
                torch.manual_seed(8)
                values = trainer._accumulate_micro_batch(batch, .07, loss_scale=.5)
                gradients = torch.cat([
                    p.grad.flatten().float() for p in trainer.model_module.student.parameters() if p.grad is not None
                ])
                results.append((values, gradients))
            torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=2e-6)
            a, e = results[1][1], results[0][1]
            self.assertTrue(bool(a.isfinite().all()) and bool(e.isfinite().all()))
            self.assertLess(float((a - e).norm() / e.norm()), 5e-4)

    def test_cpu_fallback(self):
        torch.manual_seed(4)
        t, w = torch.randn(11, 7), torch.randn(19, 7) * .01
        for backend in ("torch", "triton"):
            loss = iBOTPatchLoss(19, kernel_backend=backend)
            scales = loss.sinkhorn_knopp_prototype_scales(t, w, .07, torch.tensor([11]))
            if backend == "torch":
                expected = scales
            else:
                torch.testing.assert_close(scales, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            iBOTPatchLoss(19, kernel_backend="invalid")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tiled_loss_and_gradients(self):
        torch.manual_seed(5)
        # Non-multiple tiles/features, multiple row chunks, AMP-scaled upstream gradients.
        for dtype in (torch.float16, torch.bfloat16):
            for iterations in (1, 3):
                with self.subTest(dtype=dtype, iterations=iterations):
                    m, k, d = 73, 259, 70
                    s = F.normalize(torch.randn(m, d, device="cuda"), dim=-1).to(dtype)
                    t = F.normalize(torch.randn_like(s).float(), dim=-1).to(dtype)
                    sw = F.normalize(torch.randn(k, d, device="cuda"), dim=-1).to(dtype)
                    tw = F.normalize(torch.randn_like(sw).float(), dim=-1).to(dtype)
                    weights = torch.rand(m, device="cuda") / m
                    results = []
                    count = torch.tensor([m], device="cuda")
                    for backend in ("torch", "triton"):
                        f = iBOTPatchLoss(k, masked_loss_chunk_size=41, kernel_backend=backend)
                        sf, wf = s.clone().requires_grad_(), sw.clone().requires_grad_()
                        scales = f.sinkhorn_knopp_prototype_scales(t, tw, .07, count, n_iterations=iterations)
                        value = f.forward_masked_from_features(
                            sf, wf, t, tw, scales, .07, torch.empty(4, 20, device="cuda"), weights
                        )
                        (value * 1024).backward()
                        results.append((scales, value.detach(), sf.grad, wf.grad))
                    torch.testing.assert_close(count, torch.tensor([m], device="cuda"))
                    for actual, expected in zip(results[1][:2], results[0][:2]):
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
                    for actual, expected in zip(results[1][2:], results[0][2:]):
                        self.assertTrue(bool(actual.isfinite().all()))
                        relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
                        self.assertLess(float(relative_error), 5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_distributed_prototype_reduction_contract(self):
        """Emulate collectives from two unequal shards against the full-batch reference."""
        torch.manual_seed(7)
        t = F.normalize(torch.randn(73, 64, device="cuda"), dim=-1).half()
        w = F.normalize(torch.randn(259, 64, device="cuda"), dim=-1).half()
        e = torch.exp((t @ w.T).float() / .07)
        sums = e.sum(0)
        global_sums = [sums]
        b = 1 / (sums * 259)
        for _ in range(2):
            a = 1 / ((e * b).sum(1, keepdim=True) * 73)
            sums = (e * a).sum(0)
            global_sums.append(sums)
            b = b / (sums * b * 259)
        for shard in (t[:31], t[31:]):
            collective_values = [torch.tensor([73], device="cuda"), *global_sums]
            calls = []

            def all_reduce(value):
                calls.append(value.shape)
                value.copy_(collective_values[len(calls) - 1])

            count = torch.tensor([len(shard)], device="cuda")
            with patch("torch.distributed.is_initialized", return_value=True), patch(
                "torch.distributed.all_reduce", side_effect=all_reduce
            ):
                actual = iBOTPatchLoss(259, kernel_backend="triton").sinkhorn_knopp_prototype_scales(
                    shard, w, .07, count
                )
            torch.testing.assert_close(actual, b.log())
            self.assertEqual(len(calls), 4)
            self.assertEqual(int(count), len(shard))


if __name__ == "__main__":
    unittest.main()
