from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from dinovol_2.model.dinov2_eva import EvaBlock
from dinovol_2.model.model import DinoVitStudentTeacher, _materialize_backbone_config
from dinovol_2.model.patch_encode_decode import PatchEmbedDeeper


class CNNEmbeddingTests(unittest.TestCase):
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
        config = recipe_config["model"]
        self.assertEqual(config["global_crops_size"], [128] * 3)
        self.assertEqual(config["local_crops_size"], [64] * 3)
        self.assertEqual(recipe_config["dataset"]["global_crop_size"], config["global_crops_size"])
        self.assertEqual(recipe_config["dataset"]["local_crop_size"], config["local_crops_size"])
        resolved = _materialize_backbone_config(config)
        self.assertEqual(resolved["embed_dim"], 288)
        self.assertEqual(resolved["num_heads"], 6)
        self.assertEqual(resolved["depth"], 8)
        self.assertEqual(resolved["attention_type"], "full")
        self.assertEqual(resolved["patch_size"], (8, 8, 8))
        self.assertEqual(resolved["deeper_embed_stage_channels"], [32, 64, 288])
        self.assertEqual(resolved["deeper_embed_depth_per_level"], [2, 2, 2])
        self.assertFalse(resolved["deeper_embed_use_stem"])
        self.assertFalse(resolved["deeper_embed_final_projection"])
        # Keep the recipe's architecture, but use a small crop for CPU validation.
        config.update(global_crops_size=[32] * 3, local_crops_size=[32] * 3)
        backbone = DinoVitStudentTeacher._build_backbone(config).eval()
        blocks = [module for module in backbone.modules() if isinstance(module, EvaBlock)]
        self.assertEqual(len(blocks), 8)
        for block in blocks:
            self.assertEqual(block.attention_type, "full")
            self.assertEqual(len(block.axial_attns), 0)
            self.assertEqual(block.mlp.fc1.in_features, 288)
            self.assertEqual(block.mlp.fc1.out_features, 4096)  # Two 2048-wide SwiGLU branches.
            self.assertEqual(block.mlp.fc2.in_features, 2048)
            self.assertEqual(block.mlp.fc2.out_features, 288)
        encoder = backbone.down_projection
        self.assertIsInstance(encoder.stem, nn.Identity)
        self.assertIsInstance(encoder.final_proj, nn.Identity)
        self.assertEqual([len(stage.blocks) for stage in encoder.stages], [2] * 3)
        self.assertFalse(any(name.startswith(("stem.", "final_proj.")) for name in encoder.state_dict()))
        for module in encoder.modules():
            if isinstance(module, nn.Conv3d):
                self.assertEqual(module.padding_mode, "zeros")

        shapes = []
        handles = [stage.register_forward_hook(lambda module, args, output: shapes.append(tuple(output.shape)))
                   for stage in encoder.stages]
        try:
            x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
            masks = torch.zeros(1, 64, dtype=torch.bool)
            masks[:, ::3] = True
            features = backbone(x, masks=masks)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes, [(1, 32, 16, 16, 16), (1, 64, 8, 8, 8),
                                  (1, 288, 4, 4, 4)])
        tokens = features["x_norm_patchtokens"]
        self.assertEqual(tuple(tokens.shape), (1, 64, 288))
        self.assertEqual(tuple(features["x_norm_clstoken"].shape), (1, 288))
        (tokens * torch.randn_like(tokens)).sum().backward()
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

    def test_projection_free_configuration_validation(self):
        for kwargs, message in (
            ({"stage_channels": (8, 16), "final_projection": False}, "must equal embed_dim"),
            ({"stage_channels": (8,)}, "positive channel counts"),
            ({"stage_channels": (8, 0)}, "positive channel counts"),
            ({"depth_per_level": (2, 0)}, "must be positive"),
            ({"stage_channels": (8, 24), "final_projection": False, "embed_proj_3x3x3": True},
             "requires final_projection"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                PatchEmbedDeeper(patch_size=(4, 4, 4), embed_dim=24, **kwargs)


if __name__ == "__main__":
    unittest.main()
