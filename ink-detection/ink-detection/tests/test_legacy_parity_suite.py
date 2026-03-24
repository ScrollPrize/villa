from __future__ import annotations

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ink.recipes.losses.bce import BCEBatch, BCEPerSample, build_bce_targets
from ink.recipes.losses.dice import DiceBatch, DicePerSample
from ink.recipes.losses.dice_bce import DiceBCEBatch, DiceBCEPerSample
from ink.recipes.losses.reporting import loss_values, train_components
from ink.recipes.objectives import compute_group_avg


_PARITY_ENABLED = os.getenv("INK_ENABLE_LEGACY_PARITY", "0") == "1"


@unittest.skipUnless(_PARITY_ENABLED, "set INK_ENABLE_LEGACY_PARITY=1 to run legacy parity suite")
class LegacyParitySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        villa_root = repo_root / "villa" / "ink-detection"
        losses_path = villa_root / "train_resnet3d_lib" / "modeling" / "losses.py"
        model_path = villa_root / "train_resnet3d_lib" / "model.py"

        if not losses_path.exists() or not model_path.exists():
            raise unittest.SkipTest("legacy parity sources are unavailable in this workspace")

        cls._villa_losses = cls._load_module("villa_losses_for_optional_parity", losses_path)
        cls._villa_compute_group_avg = cls._load_model_function(model_path, "compute_group_avg")

    @staticmethod
    def _load_module(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_model_function(path: Path, name: str):
        module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in module_ast.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {"torch": torch, "CFG": SimpleNamespace(print_freq=10**9)}
                compiled = compile(ast.Module(body=[node], type_ignores=[]), filename=str(path), mode="exec")
                exec(compiled, namespace)
                return namespace[name]
        raise AssertionError(f"function {name!r} not found in {path}")

    def test_build_bce_targets_matches_legacy(self):
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

        standalone = build_bce_targets(
            targets,
            smooth_factor=0.15,
            soft_label_positive=0.9,
            soft_label_negative=0.1,
        )
        legacy = self._villa_losses.build_bce_targets(
            targets,
            smooth_factor=0.15,
            soft_label_positive=0.9,
            soft_label_negative=0.1,
        )

        torch.testing.assert_close(standalone, legacy)

    def test_bce_recipe_matches_legacy_bce_path(self):
        logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        valid_mask = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )

        standalone_batch = BCEBatch(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )(logits, targets, valid_mask=valid_mask)
        standalone_per_sample = BCEPerSample(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )(logits, targets, valid_mask=valid_mask)
        standalone_metrics = BCEBatch(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )
        legacy_per_sample = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            loss_recipe="bce_only",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )
        legacy_batch = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            reduction_dims=(0, 1, 2, 3),
            loss_recipe="bce_only",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )

        torch.testing.assert_close(standalone_batch, legacy_batch[0])
        torch.testing.assert_close(standalone_per_sample, legacy_per_sample[0])
        torch.testing.assert_close(
            loss_values(standalone_metrics, logits, targets, valid_mask=valid_mask),
            legacy_per_sample[0],
        )

    def test_dice_recipe_matches_legacy_dice_component(self):
        logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        valid_mask = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )

        standalone_batch = DiceBatch()(logits, targets, valid_mask=valid_mask)
        standalone_per_sample = DicePerSample()(logits, targets, valid_mask=valid_mask)
        standalone_metrics = DiceBatch()
        legacy_per_sample = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            loss_recipe="dice_bce",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )
        legacy_batch = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            reduction_dims=(0, 1, 2, 3),
            loss_recipe="dice_bce",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )

        torch.testing.assert_close(standalone_batch, legacy_batch[3])
        torch.testing.assert_close(standalone_per_sample, legacy_per_sample[3])
        torch.testing.assert_close(
            loss_values(standalone_metrics, logits, targets, valid_mask=valid_mask),
            legacy_per_sample[3],
        )

    def test_dice_bce_recipe_matches_legacy_combined_path(self):
        logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        valid_mask = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )

        standalone_batch = DiceBCEBatch(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )(logits, targets, valid_mask=valid_mask)
        standalone_per_sample = DiceBCEPerSample(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )(logits, targets, valid_mask=valid_mask)
        standalone_metrics = DiceBCEBatch(
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )
        legacy_per_sample = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            loss_recipe="dice_bce",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )
        legacy_batch = self._villa_losses.compute_region_loss_and_dice(
            logits,
            targets,
            valid_mask=valid_mask,
            reduction_dims=(0, 1, 2, 3),
            loss_recipe="dice_bce",
            smooth_factor=0.1,
            soft_label_positive=0.95,
            soft_label_negative=0.05,
        )

        torch.testing.assert_close(standalone_batch, legacy_batch[0])
        torch.testing.assert_close(standalone_per_sample, legacy_per_sample[0])
        torch.testing.assert_close(
            loss_values(standalone_metrics, logits, targets, valid_mask=valid_mask),
            legacy_per_sample[0],
        )
        components = train_components(standalone_metrics, logits, targets, valid_mask=valid_mask)
        torch.testing.assert_close(components["bce_loss"], legacy_per_sample[2].mean())
        torch.testing.assert_close(components["dice_loss"], legacy_per_sample[3].mean())

    def test_compute_group_avg_matches_legacy(self):
        values = torch.tensor([0.4, 1.3, 0.8, 0.2], dtype=torch.float32)
        group_idx = torch.tensor([0, 2, 2, 1], dtype=torch.long)

        standalone = compute_group_avg(values, group_idx, n_groups=4)
        legacy = type(self)._villa_compute_group_avg(SimpleNamespace(n_groups=4), values, group_idx)

        for standalone_tensor, legacy_tensor in zip(standalone, legacy):
            torch.testing.assert_close(standalone_tensor, legacy_tensor)


if __name__ == "__main__":
    unittest.main()
