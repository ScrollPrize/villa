"""JEM views, processing, model, objective integration, EMA, config and checkpoint contracts."""
import contextlib
from contextlib import nullcontext
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_cuda_pipeline import write_array
from dinovol_2.dataset.cuda_pipeline import BatchProcessor
from dinovol_2.dataset.jem_pipeline import JEM_EXCLUDED_OPERATIONS, patch_masks
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
import torch.nn.functional as F

from dinovol_2.loss.jem_losses import (
    alignment_from_means, entropy, fp32_reductions, kl_divergence, nonaffine_target, structural_uniform_contraction,
    subset_means,
)
from dinovol_2.methods.jem import KINDS, JemMethod
from vc_reader import views as vc_views
from dinovol_2.model.model import DINOHead
from dinovol_2.training.config import normalize_config

CUDA = torch.cuda.is_available()


def tiny_config(root, *, chunk=None, drop_path=0.0, batch=3, masking=None, scope="crop_type",
                weighting="uniform", backend="contraction", device="cpu", normalization="robust",
                compile=False, marginal=None, anisotropic=False):
    config = _tiny_config(root, chunk=chunk, drop_path=drop_path, batch=batch, masking=masking, scope=scope,
                          weighting=weighting, backend=backend, device=device, normalization=normalization,
                          compile=compile, marginal=marginal)
    if anisotropic:
        # 8x4x4 (ZYX) patches with a plain patchify embedding, z-preserving orientations and
        # crops with equal y and x: teacher/global 2x4x4 tokens, local 1x3x3 tokens.
        backbone = {key: value for key, value in config["backbone"].items() if not key.startswith("deeper_embed_")}
        backbone.update(patch_size=[8, 4, 4], embedding_type="default", global_crops_size=[16, 16, 16],
                        local_crops_size=[8, 12, 12],
                        rope_kwargs={**backbone.get("rope_kwargs", {}), "normalize_coords": "max"})
        config["backbone"] = backbone
        config["method"]["views"].update(local_size=[8, 12, 12], orientations="z_preserving")
        config = normalize_config(config)
    return config


def _tiny_config(root, *, chunk, drop_path, batch, masking, scope, weighting, backend, device, normalization,
                 compile, marginal):
    return normalize_config({
        "backbone": dict(embed_dim=64, depth=3, num_heads=4, patch_size=[4] * 3, embedding_type="deeper",
                         deeper_embed_stem_type="residual", deeper_embed_stem_channels=8,
                         deeper_embed_stage_channels=[8, 16], deeper_embed_stage_strides=[[2, 2, 2]] * 2,
                         deeper_embed_depth_per_level=[1, 1], global_crops_size=[16] * 3,
                         local_crops_size=[12] * 3, qk_norm=True, mlp_ratio=2.0, scale_mlp=False,
                         drop_path_rate=drop_path, class_token=False, untie_global_and_local_cls_norm=False,
                         rope_type="compass", attention_backend=None),
        "method": {"name": "jem", "global_batch_size": batch, "source_chunk_size": chunk, "categories": 16,
                   "head_hidden_dim": 64, "head_bottleneck_dim": 32, "structural_dim": 64,
                   "alignment_depths": [1, 2, 3], "structural_depths": [0, 1, 2],
                   "depth_weights": {"1": .25, "2": .25, "3": 1}, "regularizer_warmup_steps": 2,
                   "global_collapse_schedule": {"start": 1.0, "end": 0.1},
                   "structural_schedule": {"warm": 0.5, "final": 1.0},
                   "entropy_view_scope": scope, "structural_pair_weighting": weighting,
                   "structural_backend": backend, "structural_tile": 16, "masking": masking or {},
                   "views": {"source_size": 32, "teacher_size": 16, "global_size": 16, "local_size": 12},
                   **({"global_marginal": marginal} if marginal is not None else {})},
        "data": {"batch_size": batch, "num_workers": 0, "dataset": {
            "datasets": [{"volume_path": str(root / "volume"), "volume_scale": 0, "pixel_size_um": 2.0}],
            "vol_trim_pct": 1, "epoch_length": 8, "augmentation": "full", "normalization_scheme": normalization}},
        "runtime": {"max_iterations": 10, "use_amp": device == "cuda", "device": device, "compile": compile,
                    "output_dir": str(root / "output"), "warmup_steps": 0},
        "optimizer": {"lr": 1e-3, "warmup_steps": 0},
    })


def build(root, *, seed=0, augment=True, device="cpu", **kwargs):
    config = tiny_config(root, device=device, **kwargs)
    torch.manual_seed(seed)
    method = JemMethod(config, device=device)
    dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                             do_augmentations=augment)
    return method, dataset, config


def batch_for(method, dataset, device, count=3, seed=0):
    np.random.seed(seed)
    samples = [dataset[i] for i in range(count)]
    return method.build_processor(dataset, device)(samples), samples


def run(method, batch, device="cpu", amp=None):
    amp = device == "cuda" if amp is None else amp
    autocast = lambda: torch.autocast(device, dtype=torch.bfloat16, enabled=amp)
    method.model_module.student.zero_grad(set_to_none=True)
    output = method.forward_backward(batch, 0, autocast=autocast, sync_context=lambda sync=True: nullcontext(),
                                     backward=lambda loss: loss.backward())
    grads = {name: p.grad.detach().clone() for name, p in method.model_module.student.named_parameters()}
    return output, grads


def reference_alignment(method, logits, teacher_logits, entry):
    """Per-row alignment from an explicit gather of the overlapping tokens (independent of the method)."""
    recipe = method.recipe
    log_q = method._log_q(logits)
    overlap = entry["targets"] >= 0
    flat_rows, flat_positions = overlap.nonzero(as_tuple=True)
    teacher_flat = (entry["teacher_rows"][flat_rows] * teacher_logits.shape[1]
                    + entry["targets"][flat_rows, flat_positions])
    kl = kl_divergence(teacher_logits.reshape(-1, teacher_logits.shape[-1])[teacher_flat],
                       log_q[flat_rows, flat_positions], float(recipe["teacher_temperature"]))
    visible, masked, _, _ = subset_means(kl, flat_rows, entry["masks"][flat_rows, flat_positions], log_q.shape[0])
    return alignment_from_means(visible, masked, visible_weight=float(recipe["visible_weight"]),
                                masked_weight=float(recipe["masked_weight"]))


def reference_structural(a, targets, entry):
    """Per-row uniform structural loss over each row's overlapping tokens, one row at a time."""
    overlap = entry["targets"] >= 0
    losses = []
    for row in range(a.shape[0]):
        selected = overlap[row].nonzero(as_tuple=True)[0]
        student = F.normalize(a[row, selected].float(), dim=-1)
        teacher = targets[entry["teacher_rows"][row], entry["targets"][row, selected]]
        losses.append(structural_uniform_contraction(student, teacher))
    return torch.stack(losses)


class JemTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        write_array(self.root / "volume", np.random.default_rng(0).integers(1, 255, (48,) * 3, dtype=np.uint8))

    def tearDown(self):
        self._directory.cleanup()


class ViewTests(JemTestCase):
    def _check_ids(self, device, anisotropic=False):
        ids = np.arange(48 ** 3, dtype=np.float32).reshape((48,) * 3)
        write_array(self.root / "ids", ids)
        config = tiny_config(self.root, device=device, normalization="none", anisotropic=anisotropic,
                             masking={"global": {"probability": 0.0}, "local": {"probability": 0.0}})
        config["data"]["dataset"]["datasets"][0]["volume_path"] = str(self.root / "ids")
        config["runtime"]["use_amp"] = False  # FP32 crop storage keeps voxel IDs exact
        method = JemMethod(config, device=device)
        dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                 do_augmentations=False)
        batch, _ = batch_for(method, dataset, device)
        patch = tuple(batch["patch"])
        if anisotropic:
            for kind in ("teacher", "global", "local"):
                ids = batch[f"{kind}_orientations"]
                self.assertTrue(bool((ids < 16).all()), kind)  # z-preserving orientations only
                self.assertTrue(all(vc_views.orientation(int(i))[0][0] == 0 for i in ids))

        def cells(image, grid):
            (a, b, c), (p, q, r) = grid, patch
            return image.reshape(a, p, b, q, c, r).permute(0, 2, 4, 1, 3, 5).reshape(a * b * c, -1)

        teacher = [cells(crop[0].float().cpu(), batch["teacher_grid"]) for crop in batch["teacher_crops"]]
        for kind in ("global", "local"):
            targets = batch[f"{kind}_targets"].cpu()
            counts = (targets >= 0).sum(-1)
            # Valid overlap after orientation equals the overlap accepted before orientation.
            self.assertTrue(torch.equal(counts, batch[f"{kind}_overlap"].cpu()))
            for row, crop in enumerate(batch[f"{kind}_crops"]):
                student = cells(crop[0].float().cpu(), batch[f"{kind}_grid"])
                matched = targets[row] >= 0
                t = teacher[int(batch[f"{kind}_teacher_rows"][row])][targets[row][matched]]
                s = student[matched]
                # Same source voxels in each matched cell (orders differ between orientations).
                self.assertTrue(torch.equal(s.sort(-1).values, t.sort(-1).values))
                # Unmatched student cells share no voxel with the teacher crop.
                teacher_ids = set(teacher[int(batch[f"{kind}_teacher_rows"][row])].flatten().tolist())
                outside = set(student[~matched].flatten().tolist())
                self.assertFalse(outside & teacher_ids)

    def test_exact_correspondence_cpu(self):
        self._check_ids("cpu")

    @unittest.skipUnless(CUDA, "CUDA")
    def test_exact_correspondence_cuda(self):
        self._check_ids("cuda")

    def test_exact_correspondence_anisotropic_cpu(self):
        self._check_ids("cpu", anisotropic=True)

    @unittest.skipUnless(CUDA, "CUDA")
    def test_exact_correspondence_anisotropic_cuda(self):
        self._check_ids("cuda", anisotropic=True)

    def _check_scaled(self, device):
        """Scaled students: each matched student token's center lies in its teacher token.

        On a linear ramp along one axis, the mean of a token's voxels is the ramp at the token
        center (trilinear resizing reproduces a linear function; orientation does not change a
        cell's mean), so the student/teacher center offset along that axis is read directly.
        """
        coords = np.indices((48,) * 3, dtype=np.float32)
        offsets = []
        for axis in range(3):
            name = f"ramp{axis}"
            write_array(self.root / name, coords[axis] + 1.0)  # nonzero everywhere
            config = tiny_config(self.root, device=device, normalization="none",
                                 masking={"global": {"probability": 0.0}, "local": {"probability": 0.0}})
            config["method"]["views"]["student_scale"] = [0.8, 1.25]
            config["data"]["dataset"]["datasets"][0]["volume_path"] = str(self.root / name)
            config["runtime"]["use_amp"] = False
            method = JemMethod(config, device=device)
            dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                     do_augmentations=False)
            batch, samples = batch_for(method, dataset, device, seed=axis)
            patch = tuple(batch["patch"])
            extents = torch.cat([plan["jem_plan"][f"{kind}_boxes"][:, 3:] for plan in samples
                                 for kind in ("global", "local")])
            self.assertTrue(bool((extents != torch.tensor([16, 16, 16])).any()))  # resized views occur

            def means(image, grid):
                (a, b, c), (p, q, r) = grid, patch
                return image.reshape(a, p, b, q, c, r).mean((1, 3, 5)).reshape(-1)

            teacher = [means(crop[0].float().cpu(), batch["teacher_grid"]) for crop in batch["teacher_crops"]]
            for kind in ("global", "local"):
                targets = batch[f"{kind}_targets"].cpu()
                self.assertTrue(bool((targets >= 0).any()))
                for row, crop in enumerate(batch[f"{kind}_crops"]):
                    self.assertEqual(tuple(crop.shape[1:]), (16,) * 3 if kind == "global" else (12,) * 3)
                    student = means(crop[0].float().cpu(), batch[f"{kind}_grid"])
                    matched = targets[row] >= 0
                    if not matched.any():
                        continue
                    t = teacher[int(batch[f"{kind}_teacher_rows"][row])][targets[row][matched]]
                    offsets.append((student[matched] - t).abs().max().item() / patch[axis])
        # Within half a token, plus the edge clamp of resizing (under 0.15 voxel at scale 0.8).
        self.assertLessEqual(max(offsets), 0.5 + 0.15 / 4 + 1e-4)
        self.assertGreater(max(offsets), 0.05)  # tokens are no longer voxel-aligned

    def test_scaled_correspondence_cpu(self):
        self._check_scaled("cpu")

    @unittest.skipUnless(CUDA, "CUDA")
    def test_scaled_correspondence_cuda(self):
        self._check_scaled("cuda")

    def test_scaled_views_validation_and_step(self):
        from dinovol_2.dataset.jem_views import JemViewSpec
        base = {"source_size": 32, "teacher_size": 16, "global_size": 16, "local_size": 12}
        self.assertFalse(JemViewSpec.from_config(base, patch_size=4).scaled)
        self.assertTrue(JemViewSpec.from_config({**base, "student_scale": [0.8, 1.25]}, patch_size=4).scaled)
        for scale in ([1.25, 0.8], [0.0, 1.0], [1.0, 2.5], [1.0]):
            with self.assertRaises(ValueError):
                JemViewSpec.from_config({**base, "student_scale": scale}, patch_size=4)
        method, dataset, config = build(self.root, masking={"global": {"probability": 1.0}})
        config["method"]["views"]["student_scale"] = [0.8, 1.25]
        method = JemMethod(config, device="cpu")
        dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                 do_augmentations=True)
        batch, _ = batch_for(method, dataset, "cpu")
        method.successful_updates = 1
        output, grads = run(method, batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads.values()))

    def test_anisotropic_views_validation_and_step(self):
        from dinovol_2.dataset.jem_views import JemViewSpec
        spec = JemViewSpec.from_config({"source_size": 32, "teacher_size": 16, "global_size": 16,
                                        "local_size": [8, 12, 12], "orientations": "z_preserving"},
                                       patch_size=[8, 4, 4])
        self.assertEqual(spec.grid(spec.global_size), (2, 4, 4))
        self.assertEqual(spec.tokens(spec.local_size), 9)
        for settings, patch in (({"local_size": [8, 12, 12]}, [8, 4, 4]),  # 'all' would transpose z
                                ({"local_size": [8, 8, 12], "orientations": "z_preserving"}, [8, 4, 4]),
                                ({"orientations": "z_preserving"}, [8, 4, 8])):
            with self.assertRaises(ValueError):
                JemViewSpec.from_config({"source_size": 32, "teacher_size": 16, "global_size": 16, **settings},
                                        patch_size=patch)
        method, dataset, _ = build(self.root, anisotropic=True, masking={"global": {"probability": 1.0}})
        self.assertEqual(method.patch, (8, 4, 4))
        batch, _ = batch_for(method, dataset, "cpu")
        self.assertEqual(tuple(batch["global_crops"].shape[2:]), (16, 16, 16))
        self.assertEqual(tuple(batch["local_crops"].shape[2:]), (8, 12, 12))
        self.assertEqual(tuple(batch["global_grid"]), (2, 4, 4))
        self.assertEqual(tuple(batch["local_grid"]), (1, 3, 3))
        method.successful_updates = 1
        output, grads = run(method, batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads.values()))

    def test_orientations_are_independent_and_crops_native(self):
        method, dataset, _ = build(self.root)
        np.random.seed(1)
        orientations = []
        for _ in range(20):
            sample = dataset[0]
            plan = sample["jem_plan"]
            orientations += plan["teacher_orientations"].tolist() + plan["local_orientations"].tolist()
            for view in sample["teacher_views"] + sample["global_views"] + sample["local_views"]:
                self.assertEqual(tuple(view.raw.shape), tuple(view.target))  # no resampling
            self.assertEqual(int(plan["source_resamples"]), 0)
        self.assertGreater(len(set(orientations)), 30)

    def test_masks_and_withheld_values(self):
        method, dataset, _ = build(self.root)
        batch, _ = batch_for(method, dataset, "cpu")
        for kind, fraction, n in (("global", 0.55, 64), ("local", 0.40, 27)):
            masks = batch[f"{kind}_masks"]
            applied = batch[f"{kind}_mask_applied"]
            expected = math.floor(fraction * n + 0.5)
            self.assertTrue(torch.equal(masks.sum(-1)[applied], torch.full((int(applied.sum()),), expected)))
            self.assertEqual(int(masks[~applied].sum()), 0)
            grid = batch[f"{kind}_grid"]
            crops = batch[f"{kind}_crops"][:, 0]
            cells = crops.reshape(-1, grid[0], 4, grid[1], 4, grid[2], 4).permute(0, 1, 3, 5, 2, 4, 6)
            hidden = cells.reshape(len(crops), -1, 64)[masks.view(len(crops), -1)]
            self.assertEqual(float(hidden.abs().max()) if hidden.numel() else 0.0, 0.0)
        # Probability per view approaches 0.7, independent of the source.
        draws = patch_masks((8, 8, 8), torch.arange(4000), {"mode": "random", "probability": 0.7, "fraction": 0.4})
        self.assertAlmostEqual(float(draws["applied"].float().mean()), 0.7, delta=0.03)
        # Changing withheld voxel values before the same cell masking cannot change outputs.
        from vc_reader import views
        entry = method._chunk(batch, torch.arange(1))["global"]
        masks = entry["masks"]
        if not bool(masks.any()):
            return
        cells = masks.view(-1, 4, 4, 4).to(torch.uint8)
        voxel = masks.view(-1, 1, 4, 4, 4)
        for axis in (2, 3, 4):
            voxel = voxel.repeat_interleave(4, axis)
        withheld = entry["crops"].clone()
        withheld[voxel] = 123.0
        method.model.eval()
        with torch.no_grad():
            a = method.model_module.student_forward(views.mask_patch_cells(entry["crops"], cells, (4, 4, 4)),
                                                    masks, view_kind="global")
            b = method.model_module.student_forward(views.mask_patch_cells(withheld, cells, (4, 4, 4)),
                                                    masks, view_kind="global")
        for depth in a["logits"]:
            torch.testing.assert_close(a["logits"][depth], b["logits"][depth], rtol=0, atol=0)

    def test_jem_recipe_excludes_degradations_only_for_jem(self):
        method, dataset, _ = build(self.root)
        processor = method.build_processor(dataset, "cpu")
        recipe = processor.recipe((16,) * 3)
        for name in JEM_EXCLUDED_OPERATIONS:
            self.assertNotIn(name, recipe["intensity"])
        self.assertIn("MedianFilter", recipe["intensity"])
        self.assertIn("Sharpen", recipe["intensity"])
        dino = [name for name, _ in BatchProcessor({}, device="cpu").intensity_recipe((16,) * 3)]
        for name in JEM_EXCLUDED_OPERATIONS:
            self.assertIn(name, dino)

    def test_views_replay_alone_and_reordered(self):
        method, dataset, _ = build(self.root)
        np.random.seed(4)
        samples = [dataset[i] for i in range(3)]
        processor = method.build_processor(dataset, "cpu")
        together = processor(samples)
        reordered = processor([samples[2], samples[0], samples[1]])
        alone = processor([samples[1]])
        processor.group_size = 1  # grouped processing gives the same crops
        grouped = processor(samples)
        for kind in ("teacher", "global", "local"):
            torch.testing.assert_close(grouped[f"{kind}_crops"], together[f"{kind}_crops"], rtol=0, atol=0)
        for kind in ("teacher", "global", "local"):
            crops = together[f"{kind}_crops"]
            n = len(crops) // 3
            for slot in range(n):
                torch.testing.assert_close(crops[slot * 3 + 1], alone[f"{kind}_crops"][slot], rtol=0, atol=0)
                torch.testing.assert_close(crops[slot * 3 + 1], reordered[f"{kind}_crops"][slot * 3 + 2], rtol=0, atol=0)


class ObjectiveTests(JemTestCase):
    def dense_reference(self, method, batch):
        """Full-batch objective with autograd through the marginals (no surrogate)."""
        recipe = method.recipe
        chunk = method._chunk(batch, torch.arange(batch["batch_size"]))
        teacher = method._teacher(chunk, nullcontext)
        outputs = method._student(chunk, structural=True)
        w_global, w_struct = method.regularizer_weights(method.successful_updates)
        n_views = method.views_per_source * method.local_batch_size
        loss = 0
        with fp32_reductions("cpu"):
            for kind in ("global", "local"):
                entry = chunk[kind]
                for depth in method.alignment_depths:
                    weight = method.depth_weights[depth]
                    alignment = reference_alignment(method, outputs[kind]["logits"][depth],
                                                    teacher["logits"][depth].float(), entry)
                    loss = loss + weight * alignment.sum() / n_views
                    log_q = method._log_q(outputs[kind]["logits"][depth])
                    for group in method.groups:
                        if group.view_type != kind:
                            continue
                        rows = torch.isin(entry["slots"], torch.tensor(group.slots))
                        a = entry["valid"][rows].float()
                        q = log_q[rows].exp()
                        M = a.sum(0)
                        mu = (q * a[..., None]).sum(0) / M.clamp_min(1)[:, None]
                        values = (entropy(log_q[rows]) * a).sum(0) / M.clamp_min(1) + torch.special.xlogy(mu, mu).sum(-1)
                        present = M > 0
                        loss = loss + w_global * weight * group.weight * values[present].mean()
                for depth, head in outputs[kind]["structural"].items():
                    targets = nonaffine_target(teacher["structural"][depth])
                    structural = reference_structural(head, targets, entry)
                    loss = loss + w_struct * structural.sum() / (len(method.structural_depths) * n_views)
        method.model_module.student.zero_grad(set_to_none=True)
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in method.model_module.student.named_parameters()}

    def test_single_and_chunked_match_dense_full_batch_gradient(self):
        for scope in ("crop_type", "slot"):
            reference_method, dataset, _ = build(self.root, seed=3, scope=scope)
            reference_method.successful_updates = 1  # nonzero structural weight
            batch, _ = batch_for(reference_method, dataset, "cpu")
            reference_loss, reference_grads = self.dense_reference(reference_method, batch)
            for chunk in (None, 1, 2):
                with self.subTest(scope=scope, chunk=chunk):
                    method, _, _ = build(self.root, seed=3, scope=scope, chunk=chunk)
                    method.successful_updates = 1
                    output, grads = run(method, batch)
                    torch.testing.assert_close(output.loss, reference_loss, rtol=1e-5, atol=1e-6)
                    for name, grad in reference_grads.items():
                        torch.testing.assert_close(grads[name], grad, rtol=2e-4, atol=1e-6, msg=name)

    def test_running_marginal_single_pass(self):
        # The first update has no running marginal and runs the exact two passes. Once r equals
        # the batch's own marginal (same batch, unchanged parameters), the single pass gives the
        # exact gradient without a statistics pass.
        method, dataset, _ = build(self.root, seed=3, chunk=1, marginal={"mode": "running", "momentum": 0.0})
        method.successful_updates = 1
        batch, _ = batch_for(method, dataset, "cpu")
        calls = []
        student = method._student
        method._student = lambda chunk, **kw: (calls.append(kw.get("model") is not None), student(chunk, **kw))[1]
        first, first_grads = run(method, batch)
        self.assertEqual(calls.count(True), 3)  # statistics pass over three chunks
        self.assertNotIn("marginal_lag_kl", first.metrics)
        calls.clear()
        second, second_grads = run(method, batch)
        self.assertEqual(calls, [False] * 3)  # gradient pass only
        self.assertLess(float(second.metrics["marginal_lag_kl"]), 1e-6)
        torch.testing.assert_close(second.loss, first.loss, rtol=1e-6, atol=1e-7)
        for name, grad in first_grads.items():
            torch.testing.assert_close(second_grads[name], grad, rtol=1e-4, atol=1e-7, msg=name)

    def test_running_marginal_single_chunk(self):
        # With one chunk per rank, d also comes from the running marginal once it exists, not
        # from the current batch's marginal.
        method, dataset, _ = build(self.root, seed=3, chunk=None, marginal={"mode": "running", "momentum": 0.0})
        exact, _, _ = build(self.root, seed=3, chunk=None, marginal={"mode": "exact", "momentum": 0.0})
        method.successful_updates = exact.successful_updates = 1
        batches = [batch_for(method, dataset, "cpu", seed=seed)[0] for seed in (0, 1)]
        first, first_grads = run(method, batches[0])  # no running marginal yet: exact
        self.assertNotIn("marginal_lag_kl", first.metrics)
        same, same_grads = run(method, batches[0])  # r is this batch's own marginal
        self.assertLess(float(same.metrics["marginal_lag_kl"]), 1e-6)
        for name, grad in first_grads.items():
            torch.testing.assert_close(same_grads[name], grad, rtol=1e-4, atol=1e-7, msg=name)
        stale, stale_grads = run(method, batches[1])  # r is batch 0's marginal
        self.assertGreater(float(stale.metrics["marginal_lag_kl"]), 1e-6)
        _, exact_grads = run(exact, batches[1])
        self.assertFalse(all(torch.allclose(stale_grads[name], grad, rtol=1e-4, atol=1e-7)
                             for name, grad in exact_grads.items()))

    def test_mixed_marginal_endpoints_and_exact_phase(self):
        # mu_hat = w * mu + (1 - w) * r: w = 1 is the exact gradient, w = 0 the running one (d from r).
        # r comes from batch 0 (momentum 0); batch 1 is then compared across modes at equal
        # parameters, update counts (schedule weights) and chunking.
        def method_for(marginal, chunk=None, updates=1):
            method, dataset, _ = build(self.root, seed=3, chunk=chunk, marginal=marginal)
            method.successful_updates = updates
            return method, dataset

        def grads_after_r(marginal, chunk=None, updates=1):
            method, _ = method_for(marginal, chunk, updates)
            first, _ = run(method, batches[0])
            output, grads = run(method, batches[1])
            return first, output, grads

        exact, dataset = method_for({"mode": "exact", "momentum": 0.0})
        batches = [batch_for(exact, dataset, "cpu", seed=seed)[0] for seed in (0, 1)]
        for chunk in (None, 1):
            _, _, exact_grads = grads_after_r({"mode": "exact", "momentum": 0.0}, chunk)
            _, _, running_grads = grads_after_r({"mode": "running", "momentum": 0.0}, chunk)
            for weight, reference in ((1.0, exact_grads), (0.0, running_grads)):
                with self.subTest(chunk=chunk, weight=weight):
                    first, output, grads = grads_after_r({"mode": "mixed", "momentum": 0.0, "batch_weight": weight},
                                                         chunk)
                    self.assertNotIn("marginal_lag_kl", first.metrics)  # no r yet: exact
                    self.assertGreater(float(output.metrics["marginal_lag_kl"]), 1e-6)
                    for name, grad in reference.items():
                        torch.testing.assert_close(grads[name], grad, rtol=2e-4, atol=1e-6, msg=name)
            if chunk is None:
                # Halfway differs from the exact gradient.
                _, _, half_grads = grads_after_r({"mode": "mixed", "momentum": 0.0, "batch_weight": 0.5})
                self.assertFalse(all(torch.allclose(half_grads[n], exact_grads[n], rtol=1e-4, atol=1e-7)
                                     for n in exact_grads))
        # exact_updates keeps the exact scheme (r still updated) until that many updates ran.
        for updates, reference_mode in ((2, "exact"), (3, "running")):
            with self.subTest(updates=updates):
                _, output, grads = grads_after_r({"mode": "mixed", "momentum": 0.0, "batch_weight": 0.0,
                                                  "exact_updates": 3}, updates=updates)
                self.assertEqual("marginal_lag_kl" in output.metrics, reference_mode == "running")
                _, _, reference = grads_after_r({"mode": reference_mode, "momentum": 0.0}, updates=updates)
                for name, grad in reference.items():
                    torch.testing.assert_close(grads[name], grad, rtol=2e-4, atol=1e-6, msg=name)
        for bad in ({"mode": "mixed", "momentum": 0.0, "batch_weight": 1.5},
                    {"mode": "mixed", "momentum": 0.0, "exact_updates": -1}):
            with self.assertRaises(ValueError):
                build(self.root, marginal=bad)

    def test_exact_mode_logs_marginal_diagnostics(self):
        method, dataset, _ = build(self.root, seed=3, marginal={"mode": "exact", "momentum": 0.0,
                                                                "batch_weight": 0.5})
        method.successful_updates = 1
        batches = [batch_for(method, dataset, "cpu", seed=seed)[0] for seed in (0, 1)]
        first, _ = run(method, batches[0])
        self.assertFalse([name for name in first.metrics if name.startswith(("marginal_lag_kl", "mixed_d_rms",
                                                                              "marginal_step_kl"))])
        r = {key: method._running_reference(key)[0].clone() for key in method.running_marginal["S"]}
        second, _ = run(method, batches[1])
        self.assertNotIn("marginal_lag_kl", second.metrics)  # the gradient does not use r in exact mode
        depth = method.alignment_depths[-1]
        for group in method.groups:
            for prefix in ("marginal_entropy_ema", "marginal_lag_kl", "marginal_step_kl", "mixed_d_rms"):
                self.assertIn(f"{prefix}_d{depth}_{group.name}", second.metrics)
        # momentum 0: r is batch 0's marginal, so lag and step KL agree; the EMA entropy is H(r).
        for group in method.groups:
            key = (depth, group.name)
            lag, step = (float(second.metrics[f"{p}_d{depth}_{group.name}"]) for p in ("marginal_lag_kl",
                                                                                        "marginal_step_kl"))
            self.assertGreater(lag, 0.0)
            self.assertAlmostEqual(lag, step, places=5)
            present = method.running_marginal["M"][key] > 0  # batch 1's statistics now
            self.assertTrue(bool(present.any()))
            h = -torch.special.xlogy(r[key], r[key]).sum(-1)
            self.assertLess(abs(float(second.metrics[f"marginal_entropy_ema_d{depth}_{group.name}"])
                                - float(h[present].mean())), 1e-4)

    def test_running_marginal_momentum_and_checkpoint(self):
        reference, dataset, _ = build(self.root, seed=3, chunk=1, marginal={"mode": "running", "momentum": 0.0})
        batches = [batch_for(reference, dataset, "cpu", seed=seed)[0] for seed in (0, 1)]
        run(reference, batches[1])  # momentum 0: the running marginal is batch 1's statistics
        method, _, _ = build(self.root, seed=3, chunk=1, marginal={"mode": "running", "momentum": 0.25})
        run(method, batches[0])
        first = {name: {k: v.clone() for k, v in table.items()} for name, table in method.running_marginal.items()}
        method.load_checkpoint_state({**method.checkpoint_state()}, 0)  # round trip keeps the state
        # The second update runs one pass with the parameters unchanged, so batch 1's statistics
        # equal the reference's.
        run(method, batches[1])
        for name in ("S", "M"):
            for key, value in method.running_marginal[name].items():
                expected = 0.25 * first[name][key] + 0.75 * reference.running_marginal[name][key]
                torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6, msg=f"{name}{key}")
        restored, _, _ = build(self.root, seed=4, chunk=1)
        restored.load_checkpoint_state(method.checkpoint_state(), 0)
        for name in ("S", "M"):
            for key, value in method.running_marginal[name].items():
                self.assertTrue(torch.equal(restored.running_marginal[name][key], value))
        for bad in ({"mode": "stale", "momentum": 0.0}, {"mode": "running", "momentum": 1.0}):
            with self.assertRaises(ValueError):
                build(self.root, marginal=bad)

    def test_drop_path_replay_in_chunks(self):
        method, dataset, _ = build(self.root, drop_path=0.5, chunk=1, marginal={"mode": "exact", "momentum": 0.0})
        method.verify_replay = True
        batch, _ = batch_for(method, dataset, "cpu")
        method.train()
        output, grads = run(method, batch)
        self.assertEqual(float(output.metrics["replay_max_abs_diff"]), 0.0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))

    @unittest.skipUnless(CUDA, "CUDA")
    def test_host_and_device_batch_storage_agree(self):
        results = []
        for storage in ("device", "host"):
            config = tiny_config(self.root, device="cuda", drop_path=0.5, chunk=1)
            config["method"]["batch_storage"] = storage
            torch.manual_seed(0)
            method = JemMethod(config, device="cuda")
            dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                     do_augmentations=True)
            batch, _ = batch_for(method, dataset, "cuda")
            self.assertEqual(batch["global_crops"].is_pinned(), storage == "host")
            # Chunks upload the same crops (host: pinned slices copied asynchronously).
            crops = [{key: chunk[key]["crops"] if key in KINDS else chunk[key] for key in (*KINDS, "teacher_crops")}
                     for chunk in (method._chunk(batch, torch.arange(start, start + 1)) for start in range(3))]
            method.train()
            torch.manual_seed(1)
            results.append((crops, *run(method, batch, device="cuda")))
        (device_crops, device, device_grads), (host_crops, host, host_grads) = results
        for device_chunk, host_chunk in zip(device_crops, host_crops):
            for key, value in device_chunk.items():
                self.assertTrue(host_chunk[key].is_cuda and torch.equal(host_chunk[key], value), msg=key)
        self.assertTrue(torch.equal(host.loss, device.loss))
        for name, grad in device_grads.items():  # cuDNN weight gradients are not bitwise deterministic
            torch.testing.assert_close(host_grads[name], grad, rtol=1e-3, atol=1e-5, msg=name)

    @unittest.skipUnless(CUDA, "CUDA")
    def test_compiled_replay_and_eager_agreement(self):
        import torch._dynamo
        self.addCleanup(torch._dynamo.reset)
        # BF16: the compiled statistics pass draws the same drop-path masks and produces
        # bitwise the same logits as the compiled gradient pass (an inference-graph
        # statistics pass differs in the last bits, caught here without drop-path).
        for drop_path in (0.0, 0.5):
            torch._dynamo.reset()
            method, dataset, _ = build(self.root, device="cuda", drop_path=drop_path, chunk=1, compile=True)
            method.verify_replay = True
            batch, _ = batch_for(method, dataset, "cuda")
            method.train()
            output, _ = run(method, batch, device="cuda")
            self.assertEqual(float(output.metrics["replay_max_abs_diff"]), 0.0, msg=f"drop_path={drop_path}")
        # FP32: compiled and eager updates agree (same drop-path draws from the same seed).
        results = []
        for compile in (False, True):
            torch._dynamo.reset()
            config = tiny_config(self.root, device="cuda", drop_path=0.5, chunk=1, compile=compile)
            config["runtime"]["use_amp"] = False
            torch.manual_seed(0)
            method = JemMethod(config, device="cuda")
            method.successful_updates = 1  # nonzero structural weight
            dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                     do_augmentations=True)
            batch, _ = batch_for(method, dataset, "cuda")
            method.train()
            torch.manual_seed(1)
            results.append(run(method, batch, device="cuda", amp=False))
        (eager, eager_grads), (compiled, compiled_grads) = results
        torch.testing.assert_close(compiled.loss, eager.loss, rtol=1e-4, atol=1e-6)
        for name, grad in eager_grads.items():
            torch.testing.assert_close(compiled_grads[name], grad, rtol=1e-3, atol=1e-5, msg=name)

    def test_distance_stratified_and_tiled_backends_run(self):
        for weighting, backend in (("distance_stratified", "contraction"), ("uniform", "tiled")):
            method, dataset, _ = build(self.root, weighting=weighting, backend=backend)
            method.successful_updates = 1
            batch, _ = batch_for(method, dataset, "cpu")
            if weighting == "distance_stratified":
                self.assertIn("global_lattice", batch)
            output, _ = run(method, batch)
            self.assertTrue(math.isfinite(float(output.loss)))
            self.assertGreater(float(output.metrics["structural_loss"]), 0.0)

    def test_mask_status_contributions_add_up(self):
        method, dataset, _ = build(self.root)
        method.successful_updates = 1
        batch, _ = batch_for(method, dataset, "cpu")
        output, _ = run(method, batch)
        m = output.metrics
        w_global, w_struct = method.regularizer_weights(1)
        for name, total in (("align", m["alignment_loss"]), ("global", w_global * m["global_collapse_loss"]),
                            ("struct", w_struct * m["structural_loss"])):
            torch.testing.assert_close(m[f"contrib_{name}_masked"] + m[f"contrib_{name}_unmasked"], total,
                                       rtol=1e-4, atol=1e-6)

    def test_masked_marginal_baseline_metric(self):
        method, dataset, _ = build(self.root, masking={"global": {"probability": 1.0}, "local": {"probability": 1.0}})
        method.successful_updates = 1
        batch, _ = batch_for(method, dataset, "cpu")
        output, _ = run(method, batch)
        chunk = method._chunk(batch, torch.arange(int(batch["batch_size"])))
        teacher = method._teacher(chunk, nullcontext)
        temperature = float(method.recipe["teacher_temperature"])
        for depth in method.alignment_depths:
            log_p = F.log_softmax(teacher["logits"][depth].float() / temperature, -1)
            mu = log_p.exp().mean((0, 1))
            for kind in ("global", "local"):
                entry = chunk[kind]
                rows, positions = (entry["masks"] & (entry["targets"] >= 0)).nonzero(as_tuple=True)
                p = log_p[entry["teacher_rows"][rows], entry["targets"][rows, positions]].exp()
                expected = (p * (p.log() - mu.log())).sum(-1).mean()
                self.assertGreater(len(rows), 0)
                torch.testing.assert_close(output.metrics[f"kl_masked_marginal_d{depth}_{kind}"], expected,
                                           rtol=1e-4, atol=1e-6)

    def test_linear_tokenizer_comparison(self):
        config = tiny_config(self.root)
        backbone = dict(config["backbone"], embedding_type="default")
        config = normalize_config({**config, "backbone": backbone})
        torch.manual_seed(0)
        method = JemMethod(config, device="cpu")
        embed = method.model_module.student.backbone.down_projection.proj
        self.assertEqual((tuple(embed.kernel_size), tuple(embed.stride), embed.bias is not None), ((4,) * 3, (4,) * 3, True))
        dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True), do_augmentations=True)
        batch, _ = batch_for(method, dataset, "cpu")
        output, grads = run(method, batch)
        self.assertTrue(math.isfinite(float(output.loss)))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))

    @unittest.skipUnless(CUDA, "CUDA")
    def test_cuda_bf16_step(self):
        method, dataset, _ = build(self.root, device="cuda", chunk=2)
        batch, _ = batch_for(method, dataset, "cuda")
        self.assertTrue(batch["global_crops"].is_cuda)
        output, grads = run(method, batch, device="cuda")
        self.assertTrue(math.isfinite(float(output.loss)))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))


class ModelTests(JemTestCase):
    def test_heads_and_parameter_groups(self):
        method, _, _ = build(self.root)
        student = method.model_module.student
        self.assertEqual(set(student.keys()), {"backbone", "categorical_heads", "structural_heads"})
        self.assertEqual(set(method.model_module.teacher.keys()), {"backbone", "categorical_heads"})
        head = student.categorical_heads[0]
        self.assertEqual([n for n, _ in head.last_layer.named_parameters()], ["weight"])
        self.assertFalse(hasattr(head.last_layer, "parametrizations"))
        groups = method.parameter_groups()
        last = [g for g in groups if any(p is head.last_layer.weight for p in g["params"])]
        self.assertEqual(last[0]["wd_multiplier"], 1.0)
        big = DINOHead(256, 512, hidden_dim=1024, bottleneck_dim=256, last_layer_type="linear")
        for module in big.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertAlmostEqual(float(module.weight.std()), 0.02, delta=0.002)
                if module.bias is not None:
                    self.assertEqual(float(module.bias.abs().max()), 0.0)
        # DINO keeps weight normalization; norm_last_layer only freezes its scale.
        for frozen in (False, True):
            dino = DINOHead(16, 8, hidden_dim=16, bottleneck_dim=4, norm_last_layer=frozen)
            scale = dino.last_layer.parametrizations.weight.original0
            self.assertEqual(scale.requires_grad, not frozen)

    def test_direct_weight_ema(self):
        method, _, _ = build(self.root)
        module = method.model_module
        student = module.student.categorical_heads[0].last_layer.weight
        teacher = module.teacher.categorical_heads[0].last_layer.weight
        with torch.no_grad():
            student.zero_(); student[0, :2] = torch.tensor([1.0, 0.0])
            teacher.zero_(); teacher[0, :2] = torch.tensor([0.0, 1.0])
        method.recipe["momentum_teacher"] = 0.5
        method.after_optimizer_step(0)
        torch.testing.assert_close(teacher[0, :2], torch.tensor([0.5, 0.5]))  # no row renormalization
        self.assertEqual(method.successful_updates, 1)
        for momentum, expected in ((1.0, [0.5, 0.5]), (0.0, [1.0, 0.0])):
            method.recipe["momentum_teacher"] = momentum
            method.after_optimizer_step(0)
            torch.testing.assert_close(teacher[0, :2], torch.tensor(expected))
        before = teacher.detach().clone()
        method.optimizer_step_skipped(0)
        torch.testing.assert_close(teacher, before)
        self.assertEqual(method.skipped_updates, 1)

    def test_scratch_initialization_copies_student(self):
        method, _, _ = build(self.root)
        module = method.model_module
        student = dict(module.student.named_parameters())
        for name, parameter in module.teacher.named_parameters():
            self.assertTrue(torch.equal(parameter, student[name]))
            self.assertFalse(parameter.requires_grad)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "DINO settings"):
            normalize_config({"method": {"name": "jem", "teacher_temp": 0.07},
                              "backbone": {"class_token": False, "untie_global_and_local_cls_norm": False}})
        with self.assertRaisesRegex(ValueError, "class_token"):
            normalize_config({"method": {"name": "jem"}, "backbone": {"untie_global_and_local_cls_norm": False}})
        with self.assertRaisesRegex(ValueError, "axial RoPE"):
            normalize_config({"method": {"name": "jem"}, "backbone": {
                "class_token": False, "untie_global_and_local_cls_norm": False, "embed_dim": 512,
                "num_heads": 8, "rope_type": "axial"}})
        with self.assertRaisesRegex(ValueError, "RoPE coordinate augmentation"):
            normalize_config({"method": {"name": "jem"}, "backbone": {
                "class_token": False, "untie_global_and_local_cls_norm": False,
                "rope_kwargs": {"rescale_coords": 2.0}}})
        resolved = normalize_config({"method": {"name": "jem"}, "backbone": {
            "class_token": False, "untie_global_and_local_cls_norm": False}})["backbone"]["rope_kwargs"]
        self.assertEqual([resolved[k] for k in ("shift_coords", "jitter_coords", "rescale_coords")], [None] * 3)
        with self.assertRaises(NotImplementedError):
            normalize_config({"method": {"name": "jem", "auxiliaries": {"ink_weight": 0.1}},
                              "backbone": {"class_token": False, "untie_global_and_local_cls_norm": False}})
        config = tiny_config(self.root)
        config["data"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            JemMethod(config, device="cpu")


class CheckpointTests(JemTestCase):
    def test_fit_save_and_resume(self):
        from dinovol_2.training.trainer import Trainer
        config = tiny_config(self.root)
        config["runtime"].update(max_iterations=3, save_every_n=1, train_monitor_every_n=1, log_every=1,
                                 log_metrics=["mi_d3_global", "perplexity_d3_local", "not_a_metric"],
                                 metrics_jsonl=True)
        trainer = Trainer(config)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            trainer.fit()
        steps = [line for line in printed.getvalue().splitlines() if line.startswith("step=")]
        self.assertTrue(steps)
        for line in steps:
            self.assertIn(" mi_d3_global=", line)
            self.assertIn(" perplexity_d3_local=", line)
            self.assertNotIn("not_a_metric", line)
        records = [json.loads(line) for line in (self.root / "output" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([record["step"] for record in records], [0, 1, 2])
        self.assertIn("marginal_lag_kl_d3_global", records[-1])
        checkpoints = sorted((self.root / "output").glob("checkpoint_step_*.pt"))
        self.assertTrue(checkpoints)
        self.assertTrue(list((self.root / "output" / "monitor").glob("jem_monitor_step_*.png")))
        saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        self.assertEqual(saved["method"]["name"], "jem")
        state = saved["method"]["state"]
        self.assertEqual(state["recipe"]["categories"], 16)
        resumed = Trainer({**config, "runtime": {**config["runtime"], "resume_from": str(checkpoints[-1])}})
        step = resumed.load_checkpoint(checkpoints[-1])
        self.assertEqual(resumed.method.successful_updates, state["successful_updates"])
        for name, value in resumed.method.model_module.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, state["teacher"][name]), name)
        self.assertEqual(step, int(saved["step"]))
        # The exported evaluation backbone is the CLS-free teacher backbone.
        from dinovol_2.model.backbone import build_backbone
        from dinovol_2.training.checkpoint import backbone_state
        from dinovol_2.training.config import as_legacy_config
        backbone = build_backbone(as_legacy_config(saved["config"])["model"])
        weights, branch = backbone_state(saved)
        backbone.load_state_dict(weights, strict=True)
        self.assertEqual(branch, "teacher")
        self.assertEqual(backbone.num_class_tokens, 0)


if __name__ == "__main__":
    unittest.main()
