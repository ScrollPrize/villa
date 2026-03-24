from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ink.core import DataBundle
from ink.recipes.models import ResNet3D, ResNet3DSegmentationModel


def _bundle() -> DataBundle:
    return DataBundle(
        train_loader=list(range(7)),
        eval_loader="eval_loader",
        in_channels=62,
        extras={},
    )


def _count_modules(module: nn.Module, kind: type[nn.Module]) -> int:
    return sum(1 for child in module.modules() if isinstance(child, kind))


class ResNet3DRecipeContractTests(unittest.TestCase):
    def test_recipe_build_returns_segmentation_model_with_requested_norm(self):
        recipe = ResNet3D(
            depth=50,
            norm="group",
            group_norm_groups=16,
            pretrained=False,
            backbone_pretrained_path="unused_when_pretrained_false.pth",
        )

        model = recipe.build(data=_bundle(), runtime=SimpleNamespace(steps_per_epoch=7))

        self.assertIsInstance(model, ResNet3DSegmentationModel)
        self.assertGreater(_count_modules(model.backbone, nn.GroupNorm), 0)
        self.assertEqual(_count_modules(model.backbone, nn.BatchNorm3d), 0)
        self.assertGreater(model.decoder.logit.in_channels, 0)

    def test_forward_is_deterministic_for_fixed_seed(self):
        recipe = ResNet3D(pretrained=False, backbone_pretrained_path="unused_when_pretrained_false.pth")

        torch.manual_seed(1234)
        model_a = recipe.build(data=_bundle(), runtime=SimpleNamespace(steps_per_epoch=7))
        torch.manual_seed(1234)
        model_b = recipe.build(data=_bundle(), runtime=SimpleNamespace(steps_per_epoch=7))

        x = torch.randn(2, 62, 64, 64)
        logits_a = model_a(x)
        logits_b = model_b(x)

        self.assertEqual(logits_a.shape, logits_b.shape)
        self.assertEqual(logits_a.shape[0], 2)
        self.assertEqual(logits_a.shape[1], 1)
        torch.testing.assert_close(logits_a, logits_b)

    def test_invalid_depth_validation_raises_clear_error(self):
        recipe = ResNet3D(
            depth=34,
            pretrained=False,
            backbone_pretrained_path="unused_when_pretrained_false.pth",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported resnet3d_model_depth=34"):
            recipe.build(data=_bundle(), runtime=SimpleNamespace(steps_per_epoch=7))

    def test_missing_pretrained_weights_raises_clear_error(self):
        missing_path = str(Path(__file__).resolve().parent / "missing_r3d50_weights.pth")
        recipe = ResNet3D(pretrained=True, backbone_pretrained_path=missing_path)

        with self.assertRaisesRegex(FileNotFoundError, "Missing backbone pretrained weights"):
            recipe.build(data=_bundle(), runtime=SimpleNamespace(steps_per_epoch=7))

    def test_default_pretrained_path_is_standalone_not_villa_layout(self):
        recipe = ResNet3D()

        self.assertNotIn("villa", recipe.backbone_pretrained_path)
        self.assertIn("ink-detection", recipe.backbone_pretrained_path)
        self.assertIn("weights", recipe.backbone_pretrained_path)


if __name__ == "__main__":
    unittest.main()
