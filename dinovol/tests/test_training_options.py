from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from dinovol_2.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from dinovol_2.augmentation.transforms.spatial.mirroring import MirrorTransform
from dinovol_2.augmentation.transforms.spatial.rot90 import Rot90Transform
from dinovol_2.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from dinovol_2.augmentation.transforms.utils.compose import ComposeTransforms
from dinovol_2.augmentation.transforms.utils.random import RandomTransform
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.loss import KoLeoLoss
from dinovol_2.model.model import DinoVitStudentTeacher
from dinovol_2.model.rope import MixedRopePositionEmbedding, RopePositionEmbedding
from dinovol_2.pretrain import DinoIBOTPretrainer


def tiny_model_config(**overrides):
    return {
        "model_type": "v2", "embed_dim": 36, "depth": 4, "num_heads": 3,
        "global_crops_size": [16] * 3, "local_crops_size": [8] * 3,
        "patch_size": [8] * 3, "drop_path_rate": 0.0,
        "dino_out_dim": 16, "ibot_out_dim": 24,
        "dino_head_hidden_dim": 24, "dino_head_bottleneck_dim": 12,
        "ibot_head_hidden_dim": 24, "ibot_head_bottleneck_dim": 12,
        **overrides,
    }


class _AddIntensity(ImageOnlyTransform):
    def _apply_to_image(self, img, **params):
        # Consume a shape-dependent number of random values between geometry
        # transforms: replaying a full pipeline at a different resolution fails.
        torch.rand_like(img)
        return img + 100


def _koleo_worker(rank, init_path, result_path):
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        whole = torch.tensor([[1., 0., 0.], [0.4, 0.8, 0.2]])
        local = whole[rank:rank + 1].clone().requires_grad_()
        loss = KoLeoLoss()(local)
        loss.backward()
        expected_input = whole.clone().requires_grad_()
        expected = KoLeoLoss(distributed=False)(expected_input)
        expected.backward()
        # Each rank backpropagates its local mean; DDP then averages gradients.
        torch.testing.assert_close(local.grad / 2, expected_input.grad[rank:rank + 1])
        torch.testing.assert_close(loss.detach(), expected.detach())
        assert torch.isfinite(local.grad).all() and local.grad.abs().max() > 0
        for count, group_size in ((32, 16), (24, 8)):
            whole = torch.randn(count, 12, generator=torch.Generator().manual_seed(13))
            local_count = count // 2
            local = whole[rank * local_count:(rank + 1) * local_count].clone().requires_grad_()
            grouped_loss = KoLeoLoss(loss_group_size=group_size)(local)
            grouped_loss.backward()
            reference = whole.clone().requires_grad_()
            expected = torch.stack([KoLeoLoss(distributed=False)(chunk) for chunk in reference.split(group_size)]).mean()
            expected.backward()
            torch.testing.assert_close(local.grad / 2, reference.grad[rank * local_count:(rank + 1) * local_count])
            mean_loss = grouped_loss.detach().clone()
            dist.all_reduce(mean_loss)
            torch.testing.assert_close(mean_loss / 2, expected.detach())
        if rank == 0:
            torch.save(True, result_path)
    finally:
        dist.destroy_process_group()


class TrainingConfigTests(unittest.TestCase):
    def test_v3_defaults_and_legacy_v2_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"device": "cpu", "use_amp": False, "output_dir": directory,
                      "max_iterations": 10, "warmup_steps": 2,
                      "model": tiny_model_config(model_type="v3")}
            modern = DinoIBOTPretrainer(config)
            self.assertEqual(modern.model_config["rope_type"], "axial")
            self.assertFalse(modern.model_config["untie_cls_and_patch_norms"])
            self.assertTrue(modern.model_config["untie_global_and_local_cls_norm"])
            self.assertEqual(modern.lr_schedule_type, "constant")
            self.assertEqual(modern.koleo_loss.loss_group_size, 16)
            self.assertTrue(modern.koleo_loss.distributed)
            self.assertEqual(modern.final_weight_decay, modern.base_weight_decay)
            self.assertEqual(modern.momentum_teacher, .994)
            self.assertEqual(modern.final_momentum_teacher, modern.momentum_teacher)
            config["model"]["model_type"] = "v2"
            legacy = DinoIBOTPretrainer(config)
            self.assertEqual(legacy.model_config["rope_type"], "mixed")
            self.assertFalse(legacy.model_config["untie_global_and_local_cls_norm"])
            self.assertEqual(legacy.lr_schedule_type, "cosine")
            self.assertEqual(legacy.final_weight_decay, .4)
            self.assertEqual(legacy.final_momentum_teacher, 1.)
            self.assertIsNone(legacy.koleo_loss.loss_group_size)
            config.update(lr_schedule="cosine", final_momentum_teacher=.999,
                          weight_decay_end=.2, koleo_loss_group_size=None)
            config["model"].update(model_type="v3", rope_type="mixed", untie_global_and_local_cls_norm=False)
            overridden = DinoIBOTPretrainer(config)
            self.assertEqual(overridden.model_config["rope_type"], "mixed")
            self.assertFalse(overridden.model_config["untie_global_and_local_cls_norm"])
            self.assertEqual(overridden.lr_schedule_type, "cosine")
            self.assertEqual(overridden.final_weight_decay, .2)
            self.assertEqual(overridden.final_momentum_teacher, .999)
            self.assertIsNone(overridden.koleo_loss.loss_group_size)

    def test_checkpoint_and_head_settings_reach_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"device": "cpu", "use_amp": False, "output_dir": directory,
                      "max_iterations": 5, "model": tiny_model_config()}
            original = DinoIBOTPretrainer(config)
            with torch.no_grad():
                original.model_module.teacher.backbone.norm.weight.fill_(2.5)
            path = original.save_checkpoint(0)
            config["model"]["pretrained_weights"] = str(path)
            config["model"]["pretrained_backbone_only"] = False
            loaded = DinoIBOTPretrainer(config)
            self.assertEqual(loaded.model_config["pretrained_weights"], str(path))
            self.assertEqual(loaded.model_module.student.dino_head.last_layer.out_features, 16)
            self.assertEqual(loaded.model_module.student.ibot_head.last_layer.out_features, 24)
            self.assertEqual(loaded.dino_loss.center.shape[-1], 16)
            self.assertEqual(loaded.ibot_patch_loss.center.shape[-1], 24)
            for key, value in original.model_module.teacher.state_dict().items():
                torch.testing.assert_close(loaded.model_module.student.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(loaded.model_module.teacher.state_dict()[key], value, rtol=0, atol=0)

    def test_constant_lr_warmup_freeze_and_cosine_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"device": "cpu", "use_amp": False, "output_dir": directory,
                      "max_iterations": 8, "lr": 0.01, "min_lr": 0.001,
                      "warmup_steps": 3, "freeze_last_layer_steps": 2,
                      "model": tiny_model_config()}
            cosine = DinoIBOTPretrainer(config)
            self.assertLess(cosine.lr_schedule[7], cosine.lr_schedule[3])
            config["lr_schedule"] = "constant"
            constant = DinoIBOTPretrainer(config)
            self.assertEqual([constant.lr_schedule[i] for i in range(3)], [0, .005, .01])
            for step in (3, 7, 8, 50):
                self.assertEqual(constant.lr_schedule[step], .01)
                self.assertEqual(constant.last_layer_lr_schedule[step], .01)
            self.assertEqual(constant.last_layer_lr_schedule[1], 0)
            for step in range(8):
                self.assertEqual(constant.wd_schedule[step], cosine.wd_schedule[step])
                self.assertEqual(constant.momentum_schedule[step], cosine.momentum_schedule[step])
            config["warmup_steps"] = 0
            self.assertEqual(DinoIBOTPretrainer(config).lr_schedule[0], .01)
            config["lr_schedule"] = "typo"
            with self.assertRaisesRegex(ValueError, "lr_schedule"):
                DinoIBOTPretrainer(config)


class BackboneOptionsTests(unittest.TestCase):
    def test_norm_routing_and_legacy_checkpoint_initialization(self):
        base = DinoVitStudentTeacher._build_backbone(tiny_model_config())
        with torch.no_grad():
            base.norm.weight.fill_(2)
            base.norm.bias.fill_(3)
        for separate_cls in (False, True):
            for separate_local in (False, True):
                with self.subTest(cls=separate_cls, local=separate_local):
                    model = DinoVitStudentTeacher._build_backbone(tiny_model_config(
                        untie_cls_and_patch_norms=separate_cls,
                        untie_global_and_local_cls_norm=separate_local))
                    model.load_state_dict(base.state_dict(), strict=True)
                    x = torch.randn(2, 13, 36)
                    for norm in (model.cls_norm, model.local_cls_norm):
                        if norm is not None:
                            torch.testing.assert_close(norm.weight, base.norm.weight)
                            torch.testing.assert_close(norm.bias, base.norm.bias)
                    with torch.no_grad():
                        if model.cls_norm is not None:
                            model.cls_norm.bias.fill_(5)
                        if model.local_cls_norm is not None:
                            model.local_cls_norm.bias.fill_(7)
                    for training in (False, True):
                        model.train(training)
                        for view in ("global", "local"):
                            output = model._format_features(x, None, view_kind=view)
                            bias = 7 if training and view == "local" and separate_local else (5 if separate_cls else 3)
                            self.assertAlmostEqual(float(output["x_norm_clstoken"].mean().detach()), bias, places=5)
                            self.assertAlmostEqual(float(output["x_norm_regtokens"].mean().detach()), bias, places=5)
                            torch.testing.assert_close(output["x_norm_patchtokens"], base.norm(x[:, model.num_prefix_tokens:]))
                            self.assertIs(output["x_prenorm"], x)

    def test_axial_and_mixed_forward_backward_at_multiple_resolutions(self):
        for rope_type in ("axial", "mixed"):
            for chunks in (0, 2):
                with self.subTest(rope=rope_type, chunks=chunks):
                    model = DinoVitStudentTeacher._build_backbone(tiny_model_config(
                        rope_type=rope_type, block_chunks=chunks,
                        untie_cls_and_patch_norms=True, untie_global_and_local_cls_norm=True))
                    self.assertEqual(model.use_per_block_rope, rope_type == "mixed")
                    if rope_type == "axial":
                        self.assertIsInstance(model.rope_embed, RopePositionEmbedding)
                    else:
                        self.assertTrue(any(isinstance(m, MixedRopePositionEmbedding) for m in model.modules()))
                    loss = 0
                    for shape, view in (((16, 16, 24), "global"), ((8, 8, 8), "local")):
                        outputs = model.forward_features(torch.randn(2, 1, *shape), view_kind=view)
                        self.assertEqual(outputs["x_norm_patchtokens"].shape[1], (shape[0] // 8) * (shape[1] // 8) * (shape[2] // 8))
                        for key in ("x_norm_clstoken", "x_norm_patchtokens"):
                            loss = loss + (outputs[key] * torch.randn_like(outputs[key])).sum()
                    loss.backward()
                    for norm in (model.norm, model.cls_norm, model.local_cls_norm):
                        self.assertTrue(torch.isfinite(norm.weight.grad).all())
                        self.assertGreater(float(norm.weight.grad.abs().sum()), 0)
                    restored = copy.deepcopy(model)
                    restored.load_state_dict(model.state_dict(), strict=True)
                    model.eval()
                    restored.eval()
                    image = torch.randn(1, 1, 16, 16, 16)
                    torch.testing.assert_close(model(image)["x_norm_patchtokens"], restored(image)["x_norm_patchtokens"])


class GramAlignmentTests(unittest.TestCase):
    def test_geometry_is_shared_at_different_resolutions_with_points(self):
        dataset = object.__new__(SSLZarrDataset)
        dataset.gram_teacher_no_augmentations = True
        dataset.global_view_size = (3, 3, 3)
        pipeline = ComposeTransforms([
            RandomTransform(Rot90Transform(allowed_axes={0, 1, 2}), 1),
            _AddIntensity(),
            RandomTransform(MirrorTransform((0, 1, 2)), 1),
            RandomTransform(TransposeAxesTransform({0, 1, 2}), 1),
        ])
        student = torch.arange(27.).reshape(1, 3, 3, 3)
        teacher = F.interpolate(student[None], scale_factor=2, mode="nearest")[0]
        for seed in range(5):
            torch.manual_seed(seed)
            points = torch.tensor([[0., 1., 2.]])
            augmented, gram, transformed_points = dataset._apply_paired_global_transform_with_points(
                pipeline, student.clone(), teacher.clone(), points)
            torch.testing.assert_close(augmented - 100, gram[:, ::2, ::2, ::2])
            index = tuple(transformed_points[0].long().tolist())
            self.assertEqual(float(augmented[(0, *index)] - 100), float(student[0, 0, 1, 2]))
            augmented, gram = dataset._apply_paired_global_transform(pipeline, student.clone(), teacher.clone())
            torch.testing.assert_close(augmented - 100, gram[:, ::2, ::2, ::2])


class KoLeoTests(unittest.TestCase):
    def test_grouping_and_small_batch_fallback(self):
        for count in (1, 4, 8, 16, 32):
            with self.subTest(count=count):
                x = torch.randn(count, 12, requires_grad=True)
                actual = KoLeoLoss(distributed=False, loss_group_size=16)(x)
                expected = torch.stack([KoLeoLoss(distributed=False)(chunk) for chunk in x.split(16)]).mean()
                torch.testing.assert_close(actual, expected)
                actual_grad, = torch.autograd.grad(actual, x, retain_graph=True)
                expected_grad, = torch.autograd.grad(expected, x)
                torch.testing.assert_close(actual_grad, expected_grad)
        with self.assertRaisesRegex(ValueError, "divisible"):
            KoLeoLoss(loss_group_size=16)(torch.randn(20, 12))
        for invalid in (0, 1, -1, 2.5, True):
            with self.assertRaises(ValueError):
                KoLeoLoss(loss_group_size=invalid)

    def test_singleton_zero_and_antipodal_neighbour(self):
        singleton = torch.randn(1, 12, requires_grad=True)
        loss = KoLeoLoss()(singleton)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(float(singleton.grad.abs().sum()), 0)
        loss_fn = KoLeoLoss()
        self.assertEqual(loss_fn.pairwise_NNs_inner(torch.tensor([[1., 0.], [-1., 0.]])).tolist(), [1, 0])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_distributed_singletons_match_global_batch_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(_koleo_worker, args=(str(root / "init"), str(root / "result.pt")), nprocs=2, join=True)
            self.assertTrue(torch.load(root / "result.pt", weights_only=True))


if __name__ == "__main__":
    unittest.main()
