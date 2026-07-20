"""Tests for empty-input masking of inference outputs (issue #1114).

Covers the head-semantics edge case raised in the #1173 review: only
sigmoid/softmax (classification) heads have a background class to force,
so continuous heads (activation "none", e.g. the 9-channel surface_frame
target) and targets whose activation is unknown must pass through
unmasked. The blend-normalization edge case from the same review is
covered in test_blending_normalization.py.
"""

import inspect
import math
import sys

import torch

from vesuvius.models.run import inference
from vesuvius.models.run.inference import (
    Inferer,
    MASKABLE_ACTIVATIONS,
    apply_empty_input_mask,
)

# Probability threshold used by the affected surface-prediction runs.
# finalize_outputs converts it to a logit cutoff: sigmoid(x) > T <=> x > log(T/(1-T)),
# and for 2-class softmax: p_fg > T <=> logit_fg - logit_bg > log(T/(1-T)).
THRESHOLD = 0.2
LOGIT_CUTOFF = math.log(THRESHOLD / (1.0 - THRESHOLD))

SATURATION = 20.0


def test_mask_empty_input_is_opt_in():
    sig = inspect.signature(Inferer.__init__)
    assert sig.parameters["mask_empty_input"].default is False


def test_main_forwards_mask_empty_input(monkeypatch, tmp_path):
    """The CLI flag must actually reach Inferer, and default to off without it
    (mirrors test_main_accepts_input_anon in test_inference_legacy_checkpoint)."""
    captured = {}
    logits_path = tmp_path / "logits_part_0.zarr"
    coords_path = tmp_path / "coordinates_part_0.zarr"
    logits_path.mkdir()
    coords_path.mkdir()

    class FakeInferer:
        skip_empty_patches = False
        dataset = None

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def infer(self):
            return str(logits_path), str(coords_path)

    monkeypatch.setattr(inference, "Inferer", FakeInferer)
    base_argv = [
        "vesuvius.predict",
        "--model_path", str(tmp_path / "model.pth"),
        "--input_dir", str(tmp_path / "in.zarr"),
        "--output_dir", str(tmp_path / "out"),
        "--max_patches", "1",
        "--device", "cpu",
    ]

    monkeypatch.setattr(sys, "argv", base_argv + ["--mask_empty_input"])
    assert inference.main() == 0
    assert captured["mask_empty_input"] is True

    captured.clear()
    monkeypatch.setattr(sys, "argv", list(base_argv))
    assert inference.main() == 0
    assert captured["mask_empty_input"] is False


def _make_target_info(specs):
    """Build a target_info dict from (name, out_channels, activation) tuples."""
    info = {}
    start = 0
    for name, channels, activation in specs:
        info[name] = {
            "out_channels": channels,
            "start_channel": start,
            "end_channel": start + channels,
            "activation": activation,
        }
        start += channels
    return info


def test_continuous_head_is_not_masked():
    """A surface_frame style head (activation "none") passes through unchanged."""
    torch.manual_seed(0)
    output = torch.randn(2, 9, 4, 4, 4)
    original = output.clone()
    zero_mask = torch.ones(2, 1, 4, 4, 4, dtype=torch.bool)

    target_info = _make_target_info([("surface_frame", 9, "none")])
    apply_empty_input_mask(
        output, zero_mask, is_multi_task=True, target_info=target_info, num_classes=9
    )

    assert torch.equal(output, original)


def test_unknown_activation_is_not_masked():
    """Targets whose config was lost (activation None) are skipped, not guessed."""
    torch.manual_seed(1)
    output = torch.randn(2, 2, 4, 4, 4)
    original = output.clone()
    zero_mask = torch.ones(2, 1, 4, 4, 4, dtype=torch.bool)

    target_info = _make_target_info([("mystery", 2, None)])
    apply_empty_input_mask(
        output, zero_mask, is_multi_task=True, target_info=target_info, num_classes=2
    )

    assert torch.equal(output, original)


def test_mixed_targets_mask_only_classification_heads():
    """Multi-task: sigmoid/softmax heads are masked, the continuous head is not."""
    torch.manual_seed(2)
    specs = [("ink", 1, "sigmoid"), ("frame", 9, "none"), ("damage", 2, "softmax")]
    target_info = _make_target_info(specs)
    num_classes = 12

    output = torch.randn(2, num_classes, 4, 4, 4)
    original = output.clone()
    zero_mask = torch.zeros(2, 1, 4, 4, 4, dtype=torch.bool)
    zero_mask[:, :, :2] = True
    masked = zero_mask[:, 0]

    apply_empty_input_mask(
        output, zero_mask, is_multi_task=True, target_info=target_info,
        num_classes=num_classes,
    )

    # ink (sigmoid, channel 0): masked voxels forced to -20
    assert (output[:, 0][masked] == -SATURATION).all()
    # frame (continuous, channels 1:10): untouched everywhere
    assert torch.equal(output[:, 1:10], original[:, 1:10])
    # damage (softmax, channels 10:12): background up, foreground down
    assert (output[:, 10][masked] == SATURATION).all()
    assert (output[:, 11][masked] == -SATURATION).all()
    # Unmasked voxels untouched for all targets
    unmasked = ~masked
    assert torch.equal(output[:, :, 2:], original[:, :, 2:])
    assert (output[:, 0][unmasked] == original[:, 0][unmasked]).all()

    # The masked logits decide background under finalize_outputs semantics.
    assert (output[:, 0][masked] < LOGIT_CUTOFF).all()
    assert ((output[:, 11] - output[:, 10])[masked] < LOGIT_CUTOFF).all()


def test_single_task_conventions_without_target_info():
    """Without target configs (nnUNet/ResNet loaders) channel count decides."""
    zero_mask = torch.ones(2, 1, 4, 4, 4, dtype=torch.bool)

    # Single channel: sigmoid convention, strongly negative logit.
    output = torch.randn(2, 1, 4, 4, 4)
    apply_empty_input_mask(output, zero_mask, num_classes=1)
    assert (output == -SATURATION).all()

    # Multi channel: softmax convention, class 0 is background.
    output = torch.randn(2, 3, 4, 4, 4)
    apply_empty_input_mask(output, zero_mask, num_classes=3)
    assert (output[:, 0] == SATURATION).all()
    assert (output[:, 1:] == -SATURATION).all()


def test_activation_matching_is_case_insensitive():
    """Cased activation strings appear in configs and are lowercased elsewhere
    in villa (cf. utils/plotting.py); the masking gate must tolerate them."""
    torch.manual_seed(3)
    output = torch.randn(2, 3, 4, 4, 4)
    zero_mask = torch.ones(2, 1, 4, 4, 4, dtype=torch.bool)
    target_info = _make_target_info([("ink", 1, "Sigmoid"), ("damage", 2, "SOFTMAX")])

    apply_empty_input_mask(
        output, zero_mask, is_multi_task=True, target_info=target_info, num_classes=3
    )

    assert (output[:, 0] == -SATURATION).all()
    assert (output[:, 1] == SATURATION).all()
    assert (output[:, 2] == -SATURATION).all()


def test_maskable_activations_are_classification_only():
    assert set(MASKABLE_ACTIVATIONS) == {"sigmoid", "softmax"}
