"""Numerical witnesses for smoothing, masks, weighting, metrics, and clDice."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.losses import (
    CompositeLoss,
    LabelSmoothedDCAndBCELoss,
    WeightedLossTerm,
    compute_binary_soft_cldice_loss,
    create_loss,
)

from .test_model_foundation import _config_mapping


def _smoothed_bce_loss(*, weight_dice: float = 0.0):
    return LabelSmoothedDCAndBCELoss(
        bce_kwargs={},
        soft_dice_kwargs={"label_smoothing": 0.0},
        weight_dice=weight_dice,
        weight_ce=1.0,
        use_ignore_label=True,
        bce_label_smoothing=0.5,
    )


def test_half_epsilon_maps_binary_bce_targets_to_exact_quarters():
    loss = _smoothed_bce_loss()

    smoothed = loss._smooth_bce_targets(torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(
        smoothed,
        torch.tensor([0.25, 0.75]),
        rtol=0.0,
        atol=0.0,
    )


def test_smoothed_bce_matches_hand_calculated_quarter_target_value():
    loss = _smoothed_bce_loss()
    logit = math.log(3.0)
    logits_BCHW = torch.tensor([[[[logit]]]])
    target_with_ignore_BCHW = torch.tensor([[[[0.0]], [[0.0]]]])

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    expected = math.log(4.0) - 0.25 * math.log(3.0)

    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=1e-7,
    )


def test_ignore_channel_removes_value_and_gradient_at_ignored_pixels():
    loss = _smoothed_bce_loss()
    logits_BCHW = torch.zeros((1, 1, 1, 2), requires_grad=True)
    target_with_ignore_BCHW = torch.tensor(
        [[[[0.0, 1.0]], [[0.0, 1.0]]]]
    )

    value = loss(logits_BCHW, target_with_ignore_BCHW)
    value.backward()

    torch.testing.assert_close(
        value,
        torch.tensor(math.log(2.0)),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        logits_BCHW.grad,
        torch.tensor([[[[0.25, 0.0]]]]),
        rtol=0.0,
        atol=0.0,
    )


def test_combined_smoothed_bce_and_dice_matches_all_valid_literal():
    loss = _smoothed_bce_loss(weight_dice=0.25)
    logits_BCHW = torch.tensor(
        [[[[math.log(3.0), -math.log(3.0)]]]],
        requires_grad=True,
    )
    target_with_ignore_BCHW = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 0.0]]]]
    )

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    expected = math.log(4.0 / 3.0) + 0.25 * math.log(3.0) - 5.0 / 24.0

    assert expected == pytest.approx(0.3540018113, abs=5e-11)
    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=5e-8,
    )


def test_combined_smoothed_bce_and_dice_excludes_ignored_second_pixel():
    loss = _smoothed_bce_loss(weight_dice=0.25)
    logits_BCHW = torch.tensor(
        [[[[math.log(3.0), -math.log(3.0)]]]],
        requires_grad=True,
    )
    target_with_ignore_BCHW = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]]
    )

    actual = loss(logits_BCHW, target_with_ignore_BCHW)
    actual.backward()
    expected = math.log(4.0 / 3.0) + 0.25 * math.log(3.0) - 5.0 / 22.0

    assert expected == pytest.approx(0.3350624173, abs=5e-11)
    torch.testing.assert_close(
        actual,
        torch.tensor(expected),
        rtol=1e-6,
        atol=5e-8,
    )
    assert logits_BCHW.grad[0, 0, 0, 1].item() == 0.0


class _LiteralLoss(nn.Module):
    def __init__(self, value: float, auxiliary: float | None = None) -> None:
        super().__init__()
        self.value = value
        self.auxiliary = auxiliary

    def forward(self, net_output, target):
        value = net_output.sum() * 0.0 + self.value
        if self.auxiliary is None:
            return value
        return value, {"Edge Score": value * 0.0 + self.auxiliary}


def test_composite_applies_ordered_weights_and_records_literal_metrics():
    loss = CompositeLoss(
        [
            WeightedLossTerm("first", 2.0, _LiteralLoss(1.5), "Same Name"),
            WeightedLossTerm("second", -0.5, _LiteralLoss(4.0, 7.0), "Same Name"),
        ]
    )

    total = loss(torch.ones(1), torch.zeros(1))

    torch.testing.assert_close(total, torch.tensor(1.0))
    assert loss.latest_metrics == {
        "loss_terms/same_name_raw": 1.5,
        "loss_terms/same_name_weighted": 3.0,
        "loss_terms/same_name_1_raw": 4.0,
        "loss_terms/same_name_1_weighted": -2.0,
        "loss_aux/same_name_1/edge_score": 7.0,
        "loss/total": 1.0,
    }


def test_configured_composite_retains_term_weights_and_smoothing():
    authored = _config_mapping()
    authored["loss"] = {
        "terms": [
            {
                "name": "LabelSmoothedDCAndBCELoss",
                "metric_name": "ink base",
                "weight": 1.75,
                "weight_dice": 0.0,
                "weight_ce": 0.5,
                "bce_label_smoothing": 0.5,
            }
        ]
    }

    loss = create_loss(InkConfig.from_mapping(authored))

    assert loss.terms[0].weight == 1.75
    assert loss.terms[0].metric_name == "ink base"
    assert loss.terms[0].module.weight_ce == 0.5
    assert loss.terms[0].module.bce_label_smoothing == 0.5


@pytest.mark.parametrize("mask_mode", ["pre_skeleton", "post_skeleton"])
def test_cldice_all_invalid_mask_has_zero_value_and_gradient(mask_mode):
    logits_BCHW = torch.randn((1, 1, 5, 5), requires_grad=True)
    targets_BCHW = torch.ones_like(logits_BCHW)
    valid_BCHW = torch.zeros_like(logits_BCHW)

    loss = compute_binary_soft_cldice_loss(
        logits_BCHW,
        targets_BCHW,
        valid_mask=valid_BCHW,
        mask_mode=mask_mode,
    ).sum()
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        logits_BCHW.grad,
        torch.zeros_like(logits_BCHW),
        rtol=0.0,
        atol=0.0,
    )


def test_cldice_center_pixel_has_nonzero_one_fifteenth_loss():
    probabilities_BCHW = torch.full((1, 1, 3, 3), 0.25)
    probabilities_BCHW[0, 0, 1, 1] = 0.75
    targets_BCHW = torch.zeros_like(probabilities_BCHW)
    targets_BCHW[0, 0, 1, 1] = 1.0

    actual = compute_binary_soft_cldice_loss(
        torch.logit(probabilities_BCHW),
        targets_BCHW,
        num_iter=0,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([1.0 / 15.0]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_cldice_pre_and_post_skeleton_masks_are_numerically_distinct():
    probabilities_BCHW = torch.full((1, 1, 3, 3), 0.25)
    probabilities_BCHW[0, 0, 1, 1] = 0.75
    targets_BCHW = torch.zeros_like(probabilities_BCHW)
    valid_BCHW = torch.zeros_like(probabilities_BCHW)
    valid_BCHW[0, 0, 1, 1] = 1.0
    logits_BCHW = torch.logit(probabilities_BCHW)

    pre = compute_binary_soft_cldice_loss(
        logits_BCHW,
        targets_BCHW,
        valid_mask=valid_BCHW,
        mask_mode="pre_skeleton",
        num_iter=0,
    )
    post = compute_binary_soft_cldice_loss(
        logits_BCHW,
        targets_BCHW,
        valid_mask=valid_BCHW,
        mask_mode="post_skeleton",
        num_iter=0,
    )

    torch.testing.assert_close(
        pre,
        torch.tensor([3.0 / 11.0]),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        post,
        torch.tensor([1.0 / 5.0]),
        rtol=1e-6,
        atol=1e-7,
    )
