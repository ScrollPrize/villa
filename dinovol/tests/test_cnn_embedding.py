from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
import weakref

import torch
from torch import nn

from dinovol_2.model.dinov2_eva import EvaBlock
from dinovol_2.model.model import DinoVitStudentTeacher, _materialize_backbone_config
from dinovol_2.model.patch_encode_decode import LayerNormNd, PatchEmbedDeeper
from dinovol_2.ops.point_embeddings import sample_grid_points


class CNNEmbeddingTests(unittest.TestCase):
    def setUp(self):
        # Exact gradient comparisons need deterministic cuDNN weight-gradient kernels.
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

    def tearDown(self):
        torch.backends.cudnn.deterministic = self.cudnn_deterministic

    def test_no_grad_embedding_retains_stage_outputs_only_when_requested(self):
        encoder = PatchEmbedDeeper(
            patch_size=(4, 4, 4), input_channels=1, embed_dim=12,
            stage_channels=(4, 8, 12), stage_strides=(1, 2, 2),
            stem_channels=4, stem_type="conv",
        ).eval()
        refs, alive_at_projection = [], []
        handles = [stage.register_forward_hook(lambda module, args, output: refs.append(weakref.ref(output)))
                   for stage in encoder.stages]
        handles.append(encoder.final_proj.register_forward_pre_hook(
            lambda module, args: alive_at_projection.extend(ref() is not None for ref in refs)))
        x = torch.randn(1, 1, 8, 8, 8)
        try:
            with torch.no_grad():
                tokens = encoder(x, return_skips=False)
                self.assertEqual(alive_at_projection, [False, False, True])
                refs.clear()
                alive_at_projection.clear()
                tokens_with_skips, skips = encoder(x, return_skips=True)
                self.assertEqual(alive_at_projection, [True, True, True])
                self.assertEqual(len(skips), 2)
                torch.testing.assert_close(tokens, tokens_with_skips, rtol=0, atol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_embedding_skips_avoid_single_chunk_copies_and_preserve_gradients(self):
        for chunk_size in (None, 1, 2, 4):
            with self.subTest(chunk_size=chunk_size):
                torch.manual_seed(23)
                backbone = DinoVitStudentTeacher._build_backbone({
                    "model_type": "v2", "embed_dim": 12, "num_heads": 2, "depth": 1,
                    "global_crops_size": [8] * 3, "local_crops_size": [8] * 3,
                    "patch_size": [4] * 3, "embedding_type": "deeper",
                    "deeper_embed_stage_channels": [4, 8, 12],
                    "deeper_embed_stage_strides": [1, 2, 2],
                    "deeper_embed_depth_per_level": [1, 1, 1],
                    "deeper_embed_final_projection": True,
                    "deeper_embed_batch_chunk_size": chunk_size,
                })
                reference = copy.deepcopy(backbone.down_projection)
                x = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                raw_outputs = []
                handle = backbone.down_projection.register_forward_hook(
                    lambda module, args, output: raw_outputs.append(output))
                skips = []
                try:
                    tokens = backbone._down_project_with_skips(x, skips)
                finally:
                    handle.remove()
                if chunk_size != 1:
                    self.assertEqual(len(raw_outputs), 1)
                    self.assertIs(tokens, raw_outputs[0][0])
                    for skip, original in zip(skips, raw_outputs[0][1]):
                        self.assertIs(skip, original)
                chunks = ref_x.split(chunk_size) if chunk_size is not None and chunk_size < 2 else (ref_x,)
                ref_outputs = [reference(chunk, return_skips=True) for chunk in chunks]
                ref_tokens = torch.cat([output[0] for output in ref_outputs])
                ref_skips = [torch.cat(level) for level in zip(*(output[1] for output in ref_outputs))]
                for actual, expected in zip([tokens, *skips], [ref_tokens, *ref_skips]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                sum(t.square().mean() for t in [tokens, *skips]).backward()
                sum(t.square().mean() for t in [ref_tokens, *ref_skips]).backward()
                torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
                for actual, expected in zip(backbone.down_projection.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_rotary_attention_checkpointing_matches_direct_gradients(self):
        for chunks in (0, 2, 4):
            with self.subTest(block_chunks=chunks):
                torch.manual_seed(29)
                model = DinoVitStudentTeacher._build_backbone({
                    "model_type": "v3", "embed_dim": 36, "num_heads": 3, "depth": 4,
                    "global_crops_size": [8] * 3, "local_crops_size": [8] * 3,
                    "patch_size": [4] * 3, "grad_checkpointing": True,
                    "block_chunks": chunks, "drop_path_rate": 0.0,
                })
                reference = copy.deepcopy(model)
                reference.grad_checkpointing = False
                x = torch.randn(1, 1, 8, 8, 8)
                # Both paths must see the same training-time RoPE augmentation.
                rng_state = torch.get_rng_state()
                actual = model(x)["x_norm_patchtokens"]
                torch.set_rng_state(rng_state)
                expected = reference(x)["x_norm_patchtokens"]
                torch.testing.assert_close(actual, expected)
                weights = torch.randn_like(actual)
                (actual * weights).sum().backward()
                (expected * weights).sum().backward()
                for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                    if ref_parameter.grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, ref_parameter.grad)

    def test_recipe_architecture_forward_backward_and_reload(self):
        torch.manual_seed(23)
        recipe = Path(__file__).resolve().parents[1] / "dinovol_2/configs/pretrain_8_9um.json"
        recipe_config = json.loads(recipe.read_text())
        config = dict(recipe_config["model"])
        self.assertEqual(config["global_crops_size"], [128] * 3)
        self.assertEqual(config["local_crops_size"], [64] * 3)
        self.assertEqual(recipe_config["dataset"]["global_crop_size"], config["global_crops_size"])
        self.assertEqual(recipe_config["dataset"]["local_crop_size"], config["local_crops_size"])
        self.assertEqual(recipe_config["dataset"]["num_global_crops"], 2)
        self.assertEqual(recipe_config["dataset"]["num_local_crops"], 4)
        self.assertEqual(recipe_config["batch_size"], 12)
        self.assertEqual(recipe_config["grad_accumulation_steps"], 2)
        resolved = _materialize_backbone_config(config)
        self.assertEqual(resolved["embed_dim"], 480)
        self.assertEqual(resolved["num_heads"], 10)
        self.assertEqual(resolved["depth"], 10)
        self.assertEqual(resolved["attention_type"], "full")
        self.assertEqual(resolved["rope_type"], "mixed")
        self.assertEqual(resolved["mlp_ratio"], 7.2)
        self.assertTrue(resolved["grad_checkpointing"])
        self.assertEqual(resolved["patch_size"], (8, 8, 8))
        self.assertEqual(resolved["deeper_embed_stage_channels"], [32, 64, 128, 256])
        self.assertEqual(resolved["deeper_embed_stage_strides"], [1, 2, 2, 2])
        self.assertEqual(resolved["deeper_embed_depth_per_level"], [1, 2, 2, 1])
        self.assertTrue(resolved["deeper_embed_use_stem"])
        self.assertEqual(resolved["deeper_embed_stem_channels"], 32)
        self.assertTrue(resolved["deeper_embed_final_projection"])
        # Keep the recipe's architecture, but use a small crop for CPU validation.
        config.update(global_crops_size=[32] * 3, local_crops_size=[32] * 3)
        backbone = DinoVitStudentTeacher._build_backbone(config).eval()
        blocks = [module for module in backbone.modules() if isinstance(module, EvaBlock)]
        self.assertEqual(len(blocks), 10)
        for block in blocks:
            self.assertEqual(block.attention_type, "full")
            self.assertEqual(len(block.axial_attns), 0)
            self.assertEqual(block.attn.num_heads, 10)
            self.assertEqual(block.mlp.fc1.in_features, 480)
            self.assertEqual(block.mlp.fc1.out_features, 6912)  # Two 3456-wide SwiGLU branches.
            self.assertEqual(block.mlp.fc2.in_features, 3456)
            self.assertEqual(block.mlp.fc2.out_features, 480)
        encoder = backbone.down_projection
        stem_convs = [module for module in encoder.stem.modules() if isinstance(module, nn.Conv3d)]
        self.assertEqual(len(stem_convs), 1)
        self.assertEqual((stem_convs[0].in_channels, stem_convs[0].out_channels), (1, 32))
        self.assertEqual(stem_convs[0].kernel_size, (3, 3, 3))
        self.assertEqual(stem_convs[0].stride, (1, 1, 1))
        # The last stage is projected to the embedding width by a 1x1x1 convolution.
        self.assertIsInstance(encoder.final_proj, nn.Conv3d)
        self.assertEqual((encoder.final_proj.in_channels, encoder.final_proj.out_channels), (256, 480))
        self.assertEqual(encoder.final_proj.kernel_size, (1, 1, 1))
        self.assertEqual([len(stage.blocks) for stage in encoder.stages], [1, 2, 2, 1])
        for module in encoder.modules():
            if isinstance(module, nn.Conv3d):
                self.assertEqual(module.padding_mode, "zeros")

        shapes = []
        handles = [stage.register_forward_hook(lambda module, args, output: shapes.append(tuple(output.shape)))
                   for stage in [encoder.stem, *encoder.stages]]
        try:
            x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
            masks = torch.zeros(1, 64, dtype=torch.bool)
            masks[:, ::3] = True
            features = backbone(x, masks=masks)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes, [(1, 32, 32, 32, 32), (1, 32, 32, 32, 32),
                                  (1, 64, 16, 16, 16), (1, 128, 8, 8, 8),
                                  (1, 256, 4, 4, 4)])
        tokens = features["x_norm_patchtokens"]
        self.assertEqual(tuple(tokens.shape), (1, 64, 480))
        self.assertEqual(tuple(features["x_norm_clstoken"].shape), (1, 480))
        (tokens * torch.randn_like(tokens)).sum().backward()
        self.assertIsNotNone(stem_convs[0].weight.grad)
        self.assertTrue(torch.isfinite(stem_convs[0].weight.grad).all())
        self.assertGreater(stem_convs[0].weight.grad.abs().sum().item(), 0)
        for stage in encoder.stages:
            gradient = stage.blocks[0].conv1.conv.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertEqual(x.grad[:, :, :8, :8, :8].count_nonzero().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

        # Rebuild through the same serialized configuration used by checkpoints.
        serialized_config = json.loads(json.dumps(_materialize_backbone_config(config)))
        restored = DinoVitStudentTeacher._build_backbone(serialized_config).eval()
        restored.load_state_dict(backbone.state_dict(), strict=True)
        with torch.no_grad():
            restored_features = restored(x.detach(), masks=masks)
        torch.testing.assert_close(restored_features["x_norm_patchtokens"], tokens.detach())

    def test_explicit_strides_preserve_legacy_layout_and_support_anisotropic_grids(self):
        kwargs = dict(patch_size=(4, 4, 4), input_channels=1, embed_dim=24, base_features=4)
        legacy = PatchEmbedDeeper(**kwargs).eval()
        explicit = PatchEmbedDeeper(**kwargs, stage_strides=(2, 2), stem_channels=4,
                                   stem_type="residual").eval()
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(legacy(x), explicit(x), rtol=0, atol=0)
            anisotropic = PatchEmbedDeeper(
                patch_size=(2, 4, 8), input_channels=1, embed_dim=24,
                stage_channels=(4, 8, 16, 24),
                stage_strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (1, 1, 2)),
                stem_channels=4, stem_type="conv", final_projection=False,
            ).eval()
            self.assertEqual(tuple(anisotropic(torch.randn(1, 1, 8, 8, 16)).shape), (1, 24, 4, 2, 2))

    def test_projection_free_configuration_validation(self):
        for kwargs, message in (
            ({"stage_strides": (1, 2)}, "product must equal patch_size"),
            ({"stage_strides": ((2, 2), 2)}, "each spatial axis"),
            ({"stage_strides": (0, 2)}, "each spatial axis"),
            ({"stem_channels": 0}, "positive integer"),
            ({"stem_type": "unknown"}, "stem_type"),
            ({"stage_channels": (8, 16), "final_projection": False}, "must equal embed_dim"),
            ({"stage_channels": (8,)}, "positive channel counts"),
            ({"stage_channels": (8, 0)}, "positive channel counts"),
            ({"depth_per_level": (2, 0)}, "must be positive"),
            ({"stage_channels": (8, 24), "final_projection": False, "embed_proj_3x3x3": True},
             "requires final_projection"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                PatchEmbedDeeper(patch_size=(4, 4, 4), embed_dim=24, **kwargs)


    @unittest.skipUnless(torch.cuda.is_available(), "autocast behaviour is tested on CUDA")
    def test_layer_norm_keeps_input_dtype_under_autocast(self):
        torch.manual_seed(31)
        norm = LayerNormNd(8).cuda()
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
        x = 4 * torch.randn(2, 8, 5, 6, 7, device="cuda") + 3
        idx = (None, slice(None), None, None, None)
        u = x.mean(1, keepdim=True)
        normalized = (x - u) / torch.sqrt((x - u).pow(2).mean(1, keepdim=True) + norm.eps)
        expected = norm.weight[idx] * normalized + norm.bias[idx]
        # Float32 without autocast is the previous computation exactly.
        torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
        with torch.autocast("cuda"):
            actual = norm(x.half())
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, norm(x.half().float()).half(), rtol=0, atol=0)

    def _dense_backbone(self, **overrides):
        model = DinoVitStudentTeacher({
            "model_type": "v3", "embed_dim": 24, "num_heads": 2, "depth": 1,
            "global_crops_size": [16] * 3, "local_crops_size": [16] * 3, "patch_size": [4] * 3,
            "embedding_type": "deeper", "deeper_embed_stage_channels": [4, 8, 16],
            "deeper_embed_stage_strides": [1, 2, 2], "deeper_embed_depth_per_level": [1, 2, 1],
            "deeper_embed_final_projection": True, "deeper_embed_channels_last": True, "drop_path_rate": 0.0,
            "rope_shift_coords": None, "rope_jitter_coords": None, "rope_rescale_coords": None,
            "dino_out_dim": 16, "ibot_out_dim": 16, "dino_head_hidden_dim": 16, "ibot_head_hidden_dim": 16,
            "dino_head_bottleneck_dim": 8, "ibot_head_bottleneck_dim": 8,
            "dense_head": {"enabled": True, "hidden_dim": 16, "out_dim": 8, "ink_dim": 4},
            **overrides,
        })
        return model.student

    @unittest.skipUnless(torch.cuda.is_available(), "autocast behaviour is tested on CUDA")
    def test_selective_cnn_checkpointing_matches_direct_gradients(self):
        torch.manual_seed(37)
        student = self._dense_backbone().cuda()
        backbone = student.backbone
        reference = copy.deepcopy(backbone)
        backbone.grad_checkpointing = True
        reference.grad_checkpointing = False
        x = torch.randn(2, 1, 16, 16, 16, device="cuda")
        results = []
        for model in (backbone, reference):
            with torch.autocast("cuda"):
                features = model(x)
                outputs = [features["x_norm_patchtokens"], *features["embed_skips"]]
            loss = sum((output.float() * torch.linspace(-1, 1, output.numel(), device="cuda").view_as(output)).sum()
                       for output in outputs)
            loss.backward()
            results.append((outputs, [p.grad for p in model.parameters()]))
        (outputs, grads), (ref_outputs, ref_grads) = results
        for actual, expected in zip(outputs, ref_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(grads, ref_grads):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "memory is measured on CUDA")
    def test_selective_cnn_checkpointing_saves_one_tensor_per_convolution(self):
        torch.manual_seed(41)
        cnn = PatchEmbedDeeper(
            patch_size=(4, 4, 4), input_channels=1, embed_dim=16, stage_channels=(16, 16, 16),
            stage_strides=(1, 2, 2), depth_per_level=(1, 1, 1), stem_type="conv", stem_channels=16,
            channels_last=True,
        ).cuda()
        cnn.to(memory_format=cnn.memory_format)
        backbone = self._dense_backbone().backbone.cuda()
        backbone.down_projection = cnn
        x = torch.randn(1, 1, 64, 64, 64, device="cuda", requires_grad=True)
        retained = {}
        for checkpointing in (False, True):
            backbone.grad_checkpointing = checkpointing
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            with torch.autocast("cuda"):
                tokens = backbone._run_down_projection(x)
            retained[checkpointing] = torch.cuda.memory_allocated() - start - tokens.untyped_storage().nbytes()
            tokens.float().sum().backward()
        # Full-resolution fp16 tensors: stem conv, block conv1, block conv2, one each.
        full_resolution = 16 * 64 ** 3 * 2
        self.assertLess(retained[True], 3.5 * full_resolution)
        self.assertGreater(retained[False], 2 * retained[True])

    def test_point_sampled_skips_match_dense_head_sampling(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(43)
        student = self._dense_backbone().to(device)
        reference = copy.deepcopy(student)
        backbone, head = student.backbone, student.dense_head
        self.assertEqual(backbone.embed_skip_grid_start, head._fpn_skip_start)
        x = torch.randn(2, 1, 16, 16, 16, device=device)
        coordinates = torch.rand(2, 9, 3, device=device) * 16 - 0.5
        features = backbone(x, point_coordinates=coordinates)
        grids, points = features["embed_skips"], features["embed_skip_points"]
        self.assertEqual([grid is None for grid in grids],
                         [index < head._fpn_skip_start for index in range(len(head.skip_channels))])
        ref_features = reference.backbone(x)
        ref_skips = ref_features["embed_skips"]
        for actual, level in zip(points, ref_skips):
            expected = sample_grid_points(level.permute(0, 2, 3, 4, 1), coordinates, (16, 16, 16))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def head_outputs(features, dense_head, skip_points):
            tokens = features["x_norm_patchtokens"]
            grid = tokens.reshape(tokens.shape[0], 4, 4, 4, tokens.shape[-1])
            return dense_head.forward_points(grid, features["embed_skips"], coordinates, (16, 16, 16), skip_points)

        actual = head_outputs(features, head, points)
        expected = head_outputs(ref_features, reference.dense_head, None)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
        weights = {key: torch.randn_like(value) for key, value in expected.items()}
        sum((actual[key] * weights[key]).sum() for key in actual).backward()
        sum((expected[key] * weights[key]).sum() for key in expected).backward()
        for (name, parameter), ref_parameter in zip(student.named_parameters(), reference.parameters()):
            if ref_parameter.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, msg=name)


if __name__ == "__main__":
    unittest.main()
