from __future__ import annotations

import unittest

import numpy as np
import torch

from ink.recipes.losses.bce import BCEBatch, BCEPerSample, build_bce_targets
from ink.recipes.losses.boundary import (
    StitchBoundaryLoss,
    binary_mask_to_signed_distance_map,
    compute_binary_boundary_loss,
)
from ink.recipes.losses.cldice import StitchCLDiceLoss, compute_binary_soft_cldice_loss
from ink.recipes.losses.composer import LossComposer, LossTerm
from ink.recipes.losses.dice import DiceBatch, DicePerSample
from ink.recipes.losses.dice_bce import DiceBCEBatch, DiceBCEPerSample
from ink.recipes.losses.reporting import (
    loss_values,
    resolve_train_output,
    train_components,
)
from ink.recipes.losses.stitch_region import StitchRegionLoss
from ink.recipes.objectives import (
    ERMBatch,
    ERMGroupTopK,
    ERMPerSample,
    GroupDROComputer,
    compute_group_avg,
    reduce_group_topk_loss,
)
from ink.recipes.stitch import (
    StitchLossBatch,
    accumulate_to_buffers,
    compose_segment_from_roi_buffers,
    compute_stitched_loss_components,
    gaussian_weights,
    resolve_buffer_crop,
    stitch_prob_map,
)


def _manual_bce_targets(targets, *, smooth_factor, soft_label_positive, soft_label_negative):
    targets = targets.float()
    soft_targets = targets * float(soft_label_positive) + (1.0 - targets) * float(soft_label_negative)
    smooth_factor = float(smooth_factor)
    if smooth_factor != 0.0:
        soft_targets = (1.0 - soft_targets) * smooth_factor + soft_targets * (1.0 - smooth_factor)
    return soft_targets


def _manual_region_terms(
    logits,
    targets,
    *,
    valid_mask=None,
    smooth_factor=0.25,
    soft_label_positive=1.0,
    soft_label_negative=0.0,
    eps=1e-7,
):
    targets = targets.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.float32)
    else:
        valid_mask = valid_mask.to(device=targets.device, dtype=torch.float32)
    probs = torch.sigmoid(logits)
    bce_targets = _manual_bce_targets(
        targets,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, bce_targets, reduction="none")

    batch_size = int(logits.shape[0])
    per_sample_valid = valid_mask.reshape(batch_size, -1).sum(dim=1).clamp_min(1.0)
    per_sample_bce = (bce * valid_mask).reshape(batch_size, -1).sum(dim=1) / per_sample_valid
    per_sample_intersection = (probs * targets * valid_mask).reshape(batch_size, -1).sum(dim=1)
    per_sample_union = (probs * valid_mask).reshape(batch_size, -1).sum(dim=1) + (targets * valid_mask).reshape(
        batch_size, -1
    ).sum(dim=1)
    per_sample_dice = (2.0 * per_sample_intersection + float(eps)) / (per_sample_union + float(eps))
    per_sample_dice_loss = 1.0 - per_sample_dice

    batch_valid = valid_mask.sum().clamp_min(1.0)
    batch_bce = (bce * valid_mask).sum() / batch_valid
    batch_intersection = (probs * targets * valid_mask).sum()
    batch_union = (probs * valid_mask).sum() + (targets * valid_mask).sum()
    batch_dice = (2.0 * batch_intersection + float(eps)) / (batch_union + float(eps))
    batch_dice_loss = 1.0 - batch_dice
    return {
        "batch_bce": batch_bce,
        "batch_dice_loss": batch_dice_loss,
        "per_sample_bce": per_sample_bce,
        "per_sample_dice": per_sample_dice,
        "per_sample_dice_loss": per_sample_dice_loss,
    }


class LossRecipeContractTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        self.valid_mask = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )

    def test_build_bce_targets_applies_label_smoothing_contract(self):
        out = build_bce_targets(
            self.targets,
            smooth_factor=0.15,
            soft_label_positive=0.9,
            soft_label_negative=0.1,
        )

        expected = self.targets * 0.9 + (1.0 - self.targets) * 0.1
        expected = (1.0 - expected) * 0.15 + expected * (1.0 - 0.15)
        torch.testing.assert_close(out, expected)

    def _assert_recipe_output(self, recipe, *, mode: str, reduction: str, valid_mask, smooth_factor, positive, negative):
        primary_loss = recipe(self.logits, self.targets, valid_mask=valid_mask)
        expected = _manual_region_terms(
            self.logits,
            self.targets,
            valid_mask=valid_mask,
            smooth_factor=smooth_factor,
            soft_label_positive=positive,
            soft_label_negative=negative,
            eps=getattr(recipe, "eps", 1e-7),
        )
        resolved_output = resolve_train_output(recipe, self.logits, self.targets, valid_mask=valid_mask)
        components = train_components(recipe, self.logits, self.targets, valid_mask=valid_mask)
        if mode == "dice_bce":
            if reduction == "batch":
                expected_train_bce_loss = expected["batch_bce"]
                expected_train_dice_loss = expected["batch_dice_loss"]
            else:
                expected_train_bce_loss = expected["per_sample_bce"].mean()
                expected_train_dice_loss = expected["per_sample_dice_loss"].mean()

            self.assertEqual(set(resolved_output.components), {"dice_loss", "bce_loss"})
            self.assertEqual(set(components), {"dice_loss", "bce_loss"})
            torch.testing.assert_close(components["bce_loss"], expected_train_bce_loss)
            torch.testing.assert_close(components["dice_loss"], expected_train_dice_loss)
        else:
            self.assertEqual(resolved_output.components, {})
            self.assertEqual(components, {})

        if mode == "bce":
            expected_batch_loss = expected["batch_bce"]
            expected_per_sample_loss = expected["per_sample_bce"]
        elif mode == "dice":
            expected_batch_loss = expected["batch_dice_loss"]
            expected_per_sample_loss = expected["per_sample_dice_loss"]
        elif mode == "dice_bce":
            expected_batch_loss = 0.5 * expected["batch_dice_loss"] + 0.5 * expected["batch_bce"]
            expected_per_sample_loss = 0.5 * expected["per_sample_dice_loss"] + 0.5 * expected["per_sample_bce"]
        else:
            raise AssertionError(f"unknown test mode {mode!r}")

        torch.testing.assert_close(
            loss_values(recipe, self.logits, self.targets, valid_mask=valid_mask),
            expected_per_sample_loss,
        )
        if reduction == "batch":
            torch.testing.assert_close(primary_loss, expected_batch_loss)
        elif reduction == "per_sample":
            torch.testing.assert_close(primary_loss, expected_per_sample_loss)
        else:
            raise AssertionError(f"unknown reduction {reduction!r}")

    def test_batch_patch_loss_recipes_match_manual_contract_with_valid_mask(self):
        cases = (
            ("bce", BCEBatch(smooth_factor=0.2, soft_label_positive=0.9, soft_label_negative=0.1), 0.2, 0.9, 0.1),
            ("dice", DiceBatch(), 0.1, 0.95, 0.05),
            ("dice_bce", DiceBCEBatch(smooth_factor=0.1, soft_label_positive=0.95, soft_label_negative=0.05), 0.1, 0.95, 0.05),
        )
        for mode, recipe, smooth_factor, positive, negative in cases:
            with self.subTest(mode=mode):
                self._assert_recipe_output(
                    recipe,
                    mode=mode,
                    reduction="batch",
                    valid_mask=self.valid_mask,
                    smooth_factor=smooth_factor,
                    positive=positive,
                    negative=negative,
                )

    def test_per_sample_patch_loss_recipes_match_manual_contract_with_valid_mask(self):
        cases = (
            ("bce", BCEPerSample(smooth_factor=0.2, soft_label_positive=0.9, soft_label_negative=0.1), 0.2, 0.9, 0.1),
            ("dice", DicePerSample(), 0.1, 0.95, 0.05),
            ("dice_bce", DiceBCEPerSample(smooth_factor=0.1, soft_label_positive=0.95, soft_label_negative=0.05), 0.1, 0.95, 0.05),
        )
        for mode, recipe, smooth_factor, positive, negative in cases:
            with self.subTest(mode=mode):
                self._assert_recipe_output(
                    recipe,
                    mode=mode,
                    reduction="per_sample",
                    valid_mask=self.valid_mask,
                    smooth_factor=smooth_factor,
                    positive=positive,
                    negative=negative,
                )

    def test_patch_loss_recipes_match_manual_contract_without_valid_mask(self):
        cases = (
            ("bce", "batch", BCEBatch(smooth_factor=0.2, soft_label_positive=0.85, soft_label_negative=0.05), 0.2, 0.85, 0.05),
            ("bce", "per_sample", BCEPerSample(smooth_factor=0.2, soft_label_positive=0.85, soft_label_negative=0.05), 0.2, 0.85, 0.05),
            ("dice", "batch", DiceBatch(), 0.2, 0.85, 0.05),
            ("dice", "per_sample", DicePerSample(), 0.2, 0.85, 0.05),
            ("dice_bce", "batch", DiceBCEBatch(smooth_factor=0.2, soft_label_positive=0.85, soft_label_negative=0.05), 0.2, 0.85, 0.05),
            ("dice_bce", "per_sample", DiceBCEPerSample(smooth_factor=0.2, soft_label_positive=0.85, soft_label_negative=0.05), 0.2, 0.85, 0.05),
        )
        for mode, reduction, recipe, smooth_factor, positive, negative in cases:
            with self.subTest(mode=mode, reduction=reduction):
                self._assert_recipe_output(
                    recipe,
                    mode=mode,
                    reduction=reduction,
                    valid_mask=None,
                    smooth_factor=smooth_factor,
                    positive=positive,
                    negative=negative,
                )

    def test_loss_composer_matches_weighted_batch_terms(self):
        recipe = LossComposer(
            terms=(
                LossTerm(loss=DiceBatch(), weight=0.7, name="dice"),
                LossTerm(loss=BCEBatch(), weight=0.3, name="bce"),
            )
        )

        output = resolve_train_output(recipe, self.logits, self.targets, valid_mask=self.valid_mask)
        terms = _manual_region_terms(self.logits, self.targets, valid_mask=self.valid_mask)
        expected = 0.7 * terms["batch_dice_loss"] + 0.3 * terms["batch_bce"]

        torch.testing.assert_close(output.loss, expected)
        torch.testing.assert_close(output.components["dice_loss"], terms["batch_dice_loss"])
        torch.testing.assert_close(output.components["bce_loss"], terms["batch_bce"])

    def test_loss_composer_matches_weighted_per_sample_terms(self):
        recipe = LossComposer(
            terms=(
                LossTerm(loss=DicePerSample(), weight=0.7, name="dice"),
                LossTerm(loss=BCEPerSample(), weight=0.3, name="bce"),
            )
        )

        values = loss_values(recipe, self.logits, self.targets, valid_mask=self.valid_mask)
        terms = _manual_region_terms(self.logits, self.targets, valid_mask=self.valid_mask)
        expected = 0.7 * terms["per_sample_dice_loss"] + 0.3 * terms["per_sample_bce"]

        torch.testing.assert_close(values, expected)

    def test_loss_composer_can_express_dice_bce_batch(self):
        recipe = LossComposer(
            terms=(
                LossTerm(loss=DiceBatch(), weight=0.5, name="dice"),
                LossTerm(loss=BCEBatch(), weight=0.5, name="bce"),
            )
        )

        composed = resolve_train_output(recipe, self.logits, self.targets, valid_mask=self.valid_mask)
        baseline = resolve_train_output(DiceBCEBatch(), self.logits, self.targets, valid_mask=self.valid_mask)

        torch.testing.assert_close(composed.loss, baseline.loss)
        torch.testing.assert_close(composed.components["dice_loss"], baseline.components["dice_loss"])
        torch.testing.assert_close(composed.components["bce_loss"], baseline.components["bce_loss"])

    def test_build_bce_targets_matches_manual_contract(self):
        self._assert_recipe_output(
            BCEBatch(smooth_factor=0.2, soft_label_positive=0.9, soft_label_negative=0.1),
            mode="bce",
            reduction="batch",
            valid_mask=self.valid_mask,
            smooth_factor=0.2,
            positive=0.9,
            negative=0.1,
        )

    def test_binary_boundary_and_cldice_helpers_return_scalar_tensor(self):
        dist_map_np = binary_mask_to_signed_distance_map(np.array([[0, 1], [1, 0]], dtype=np.uint8))
        dist_map = torch.tensor(dist_map_np, dtype=torch.float32).view(1, 1, 2, 2).repeat(2, 1, 1, 1)

        boundary = compute_binary_boundary_loss(
            self.logits,
            dist_map,
            valid_mask=self.valid_mask,
        )
        cldice = compute_binary_soft_cldice_loss(
            self.logits,
            self.targets,
            valid_mask=self.valid_mask,
            mask_mode="pre_skeleton",
            num_iter=5,
            smooth=0.5,
        )

        self.assertGreaterEqual(boundary.numel(), 1)
        self.assertGreaterEqual(cldice.numel(), 1)

class ERMObjectiveContractTests(unittest.TestCase):
    def test_compute_group_avg_matches_manual_average(self):
        values = torch.tensor([0.4, 1.3, 0.8, 0.2], dtype=torch.float32)
        group_idx = torch.tensor([0, 2, 2, 1], dtype=torch.long)

        group_avg, group_count = compute_group_avg(values, group_idx, n_groups=4)

        expected_avg = torch.tensor([0.4, 0.2, 1.05, 0.0], dtype=torch.float32)
        expected_count = torch.tensor([1.0, 1.0, 2.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(group_avg, expected_avg)
        torch.testing.assert_close(group_count, expected_count)

    def test_erm_batch_returns_scalar_loss(self):
        batch_loss = torch.tensor(0.73, dtype=torch.float32)

        out = ERMBatch()(batch_loss)

        torch.testing.assert_close(out, batch_loss)

    def test_erm_per_sample_returns_mean_loss(self):
        per_sample_loss = torch.tensor([0.2, 1.5, 0.9, 0.7, 0.3], dtype=torch.float32)

        out = ERMPerSample()(per_sample_loss)

        torch.testing.assert_close(out, per_sample_loss.mean())

    def test_reduce_group_topk_loss_matches_manual_topk_average(self):
        per_sample_loss = torch.tensor([0.2, 1.5, 0.9, 0.7, 0.3], dtype=torch.float32)
        group_idx = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)

        out = reduce_group_topk_loss(
            per_sample_loss,
            group_idx=group_idx,
            n_groups=4,
            group_topk=2,
        )

        # Group means: g0=0.2, g1=1.2, g2=0.5 (g3 absent). Top-2 mean = (1.2 + 0.5) / 2.
        expected = torch.tensor((1.2 + 0.5) / 2.0, dtype=torch.float32)
        torch.testing.assert_close(out, expected)

    def test_erm_group_topk_delegates_to_reducer(self):
        objective = ERMGroupTopK(group_topk=2)
        per_sample_loss = torch.tensor([0.2, 1.5, 0.9, 0.7, 0.3], dtype=torch.float32)
        group_idx = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)

        out = objective.reduce(
            per_sample_loss,
            group_idx=group_idx,
            n_groups=4,
        )
        expected = reduce_group_topk_loss(
            per_sample_loss,
            group_idx=group_idx,
            n_groups=4,
            group_topk=2,
        )

        torch.testing.assert_close(out, expected)

    def test_erm_group_topk_requires_per_sample_losses(self):
        with self.assertRaisesRegex(ValueError, "per-sample"):
            ERMGroupTopK(group_topk=1)(torch.tensor(0.4), group_idx=torch.tensor([0]), n_groups=1)


class GroupDROContractTests(unittest.TestCase):
    def test_group_dro_loss_updates_state_consistently(self):
        kwargs = {
            "n_groups": 4,
            "group_counts": [8, 4, 2, 1],
            "gamma": 0.2,
            "adj": [0.5, 0.5, 0.5, 0.5],
            "step_size": 0.3,
            "normalize_loss": True,
        }
        per_sample_loss = torch.tensor([0.3, 1.2, 0.7, 0.1, 0.6], dtype=torch.float32)
        group_idx = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)

        module = GroupDROComputer(**kwargs)
        actual_loss, group_loss, group_count, weights = module.loss(per_sample_loss, group_idx)

        self.assertEqual(actual_loss.ndim, 0)
        self.assertEqual(group_loss.shape[0], kwargs["n_groups"])
        self.assertEqual(group_count.shape[0], kwargs["n_groups"])
        self.assertEqual(weights.shape[0], kwargs["n_groups"])
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=5)
        self.assertTrue(bool(module.exp_avg_initialized[group_count > 0].all().item()))

    def test_group_dro_reduce_requires_per_sample_losses(self):
        module = GroupDROComputer(n_groups=2, group_counts=[3, 7], step_size=0.1)

        with self.assertRaisesRegex(ValueError, "per-sample"):
            module.reduce(None, group_idx=torch.tensor([0, 1], dtype=torch.long))

    def test_group_dro_btl_path_produces_valid_weights(self):
        kwargs = {
            "n_groups": 3,
            "group_counts": [3, 2, 5],
            "alpha": 0.55,
            "gamma": 0.1,
            "adj": [0.2, 0.2, 0.2],
            "min_var_weight": 0.15,
            "step_size": 0.05,
            "btl": True,
        }
        per_sample_loss = torch.tensor([1.0, 0.4, 0.9, 0.1, 0.3], dtype=torch.float32)
        group_idx = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)

        module = GroupDROComputer(**kwargs)
        _actual_loss, _group_loss, _group_count, weights = module.loss(per_sample_loss, group_idx)

        self.assertEqual(weights.shape[0], kwargs["n_groups"])
        self.assertGreaterEqual(float(weights.min().item()), 0.0)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=5)


class StitchedLossContractTests(unittest.TestCase):
    def test_buffer_ops_contract(self):
        cache = {}
        weights = gaussian_weights(cache, h=3, w=4, sigma_scale=0.125, min_weight=0.01)
        self.assertEqual(weights.shape, (3, 4))
        self.assertGreaterEqual(float(weights.min()), 0.01)

        crop = resolve_buffer_crop(
            xyxy=(2, 4, 6, 8),
            downsample=2,
            offset=(1, 1),
            buffer_shape=(5, 5),
        )
        self.assertEqual(crop, {"x1": 0, "x2": 2, "y1": 1, "y2": 3, "px0": 0, "px1": 2, "py0": 0, "py1": 2, "target_h": 2, "target_w": 2})

        pred_buf = np.array([[0.0, 1.0], [0.75, 0.0]], dtype=np.float32)
        count_buf = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        probs, has = stitch_prob_map(pred_buf, count_buf)
        np.testing.assert_allclose(
            probs,
            np.array(
                [
                    [0.0, 1.0 / (1.0 + np.exp(-0.5))],
                    [1.0 / (1.0 + np.exp(-0.75)), 0.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(has, np.array([[False, True], [True, False]]))

    def test_compose_segment_and_accumulate_to_buffers(self):
        roi_buffers = [
            (
                np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
                np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32),
                (0, 0),
            ),
            (
                np.array([[1.5]], dtype=np.float32),
                np.array([[1.0]], dtype=np.float32),
                (1, 1),
            ),
        ]
        compose_probs, compose_has = compose_segment_from_roi_buffers(roi_buffers, full_shape=(3, 3))
        self.assertEqual(compose_probs.shape, (3, 3))
        self.assertEqual(compose_has.shape, (3, 3))

        pred_buf = np.zeros((4, 4), dtype=np.float32)
        count_buf = np.zeros((4, 4), dtype=np.float32)
        outputs = torch.tensor(
            [
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.5, 0.6], [0.7, 0.8]]],
            ],
            dtype=torch.float32,
        )
        xyxys = [(0, 0, 2, 2), (2, 2, 4, 4)]
        accumulate_to_buffers(
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            downsample=1,
            offset=(0, 0),
            gaussian_cache={},
            gaussian_sigma_scale=0.125,
            gaussian_min_weight=0.01,
        )
        self.assertGreater(float(pred_buf.sum()), 0.0)
        self.assertGreater(float(count_buf.sum()), 0.0)

    def test_compute_stitched_loss_components_accepts_tuple_of_terms(self):
        boundary = StitchBoundaryLoss(weight=0.3)
        cldice = StitchCLDiceLoss(weight=0.4, mask_mode="post_skeleton")
        stitch_loss = (boundary, cldice)
        stitched_logits = torch.tensor([[0.5, -1.0], [0.7, -0.2]], dtype=torch.float32)
        stitched_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        valid_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32).bool()
        boundary_dist_map = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)

        batch = StitchLossBatch(
            logits=stitched_logits,
            targets=stitched_targets,
            valid_mask=valid_mask,
            boundary_dist_map=boundary_dist_map,
        )

        reduced = compute_stitched_loss_components(
            stitch_loss,
            batch,
        )

        boundary_metrics = boundary.compute(batch)
        cldice_metrics = cldice.compute(batch)

        torch.testing.assert_close(reduced["loss"], boundary_metrics["loss"] + cldice_metrics["loss"])
        torch.testing.assert_close(reduced["boundary_loss"], boundary_metrics["boundary_loss"])
        torch.testing.assert_close(reduced["cldice_loss"], cldice_metrics["cldice_loss"])
        self.assertEqual(reduced["covered_px"], int(valid_mask.sum().item()))
        self.assertNotIn("region_loss", reduced)

    def test_compute_stitched_loss_components_with_region_term_reports_expected_keys(self):
        stitched_logits = torch.tensor([[0.5, -1.0], [0.7, -0.2]], dtype=torch.float32)
        stitched_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        valid_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32).bool()
        boundary_dist_map = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
        stitch_loss = (
            StitchRegionLoss(
                patch_loss=DiceBCEBatch(
                    smooth_factor=0.2,
                    soft_label_positive=1.0,
                    soft_label_negative=0.0,
                )
            ),
            StitchBoundaryLoss(weight=0.3),
            StitchCLDiceLoss(weight=0.4, mask_mode="post_skeleton"),
        )

        out = compute_stitched_loss_components(
            stitch_loss,
            StitchLossBatch(
                logits=stitched_logits,
                targets=stitched_targets,
                valid_mask=valid_mask,
                boundary_dist_map=boundary_dist_map,
            ),
        )

        self.assertIn("loss", out)
        self.assertIn("region_loss", out)
        self.assertIn("boundary_loss", out)
        self.assertIn("cldice_loss", out)
        self.assertIn("dice", out)
        self.assertIn("bce", out)
        self.assertIn("dice_loss", out)
        self.assertEqual(out["covered_px"], int(valid_mask.sum().item()))


if __name__ == "__main__":
    unittest.main()
