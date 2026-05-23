import unittest

import torch

from batchgeneratorsv2.benchmarks.decohesion_ablation import (
    _aggregate_runs,
    _parse_seeds,
    _train_variant,
)


class DecohesionAblationTests(unittest.TestCase):
    def test_parse_seeds_rejects_empty_values(self):
        self.assertEqual(_parse_seeds("201, 202,203"), [201, 202, 203])
        with self.assertRaisesRegex(ValueError, "seeds"):
            _parse_seeds("")

    def test_aggregate_reports_mean_delta(self):
        runs = [
            {"name": "baseline", "final": {"train_loss": 2.0, "clean_val_dice": 0.7, "decohesion_val_dice": 0.5}},
            {"name": "baseline", "final": {"train_loss": 1.0, "clean_val_dice": 0.8, "decohesion_val_dice": 0.6}},
            {"name": "decohesion", "final": {"train_loss": 1.5, "clean_val_dice": 0.6, "decohesion_val_dice": 0.7}},
            {"name": "decohesion", "final": {"train_loss": 1.0, "clean_val_dice": 0.7, "decohesion_val_dice": 0.8}},
        ]

        aggregate = _aggregate_runs(runs)

        self.assertAlmostEqual(aggregate["baseline"]["clean_val_dice_mean"], 0.75)
        self.assertAlmostEqual(aggregate["decohesion"]["decohesion_val_dice_mean"], 0.75)
        self.assertAlmostEqual(
            aggregate["delta_decohesion_minus_baseline"]["decohesion_val_dice_mean"],
            0.20,
        )

    def test_train_variant_smoke_runs_on_cpu(self):
        train_image = torch.rand((1, 1, 4, 8, 8), dtype=torch.float32)
        train_label = (train_image > 0.5).float()
        train_mask = torch.ones_like(train_label)
        val_image = torch.rand((1, 1, 4, 8, 8), dtype=torch.float32)
        val_label = (val_image > 0.5).float()
        val_mask = torch.ones_like(val_label)

        result = _train_variant(
            name="baseline",
            train_image=train_image,
            train_label=train_label,
            train_mask=train_mask,
            val_image=val_image,
            val_label=val_label,
            val_mask=val_mask,
            epochs=1,
            lr=1e-3,
            seed=201,
            device=torch.device("cpu"),
            use_decohesion=False,
        )

        self.assertEqual(result["final"]["epoch"], 1)
        self.assertGreaterEqual(result["final"]["clean_val_dice"], 0.0)
        self.assertLessEqual(result["final"]["clean_val_dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
