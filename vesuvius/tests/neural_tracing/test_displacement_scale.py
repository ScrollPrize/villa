import warnings

import pytest

from vesuvius.neural_tracing.inference.displacement_helpers import (
    is_auto_displacement_scale,
    resolve_displacement_scale,
)


def test_no_flag_with_matching_voxel_size_is_silent_and_unscaled():
    config = {"training_voxel_size_um": 4.8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(None, config, 4.8)
    assert scale == pytest.approx(1.0)
    assert source == "default"


def test_no_flag_with_mismatched_voxel_size_stays_unscaled_but_warns():
    # Issue #1149 asks that scaling stay opt-in until the unit contract is
    # settled, so a mismatch warns without changing geometry.
    config = {"training_voxel_size_um": 4.8}
    with pytest.warns(UserWarning, match="applied unscaled"):
        scale, source = resolve_displacement_scale(None, config, 7.91)
    assert scale == pytest.approx(1.0)
    assert source == "default"


def test_auto_derives_scale_from_checkpoint_metadata():
    # The issue #1149 case: checkpoint trained at 4.8 um, Scroll 3 volume at 7.91 um.
    config = {"training_voxel_size_um": 4.8}
    scale, source = resolve_displacement_scale("auto", config, 7.91)
    assert scale == pytest.approx(4.8 / 7.91)
    assert source == "auto"


@pytest.mark.parametrize("config,voxel_size", [({}, 7.91), (None, 7.91), ({"training_voxel_size_um": 4.8}, None)])
def test_auto_without_enough_information_raises(config, voxel_size):
    with pytest.raises(ValueError, match="cannot derive a scale"):
        resolve_displacement_scale("auto", config, voxel_size)


def test_missing_metadata_keeps_legacy_scale_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(None, {}, 7.91)
    assert scale == pytest.approx(1.0)
    assert source == "default"


def test_missing_config_keeps_legacy_scale_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(None, None, 7.91)
    assert scale == pytest.approx(1.0)
    assert source == "default"


def test_metadata_without_inference_voxel_size_keeps_legacy_scale_silently():
    config = {"training_voxel_size_um": 4.8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(None, config, None)
    assert scale == pytest.approx(1.0)
    assert source == "default"


def test_explicit_scale_wins_without_warning_when_consistent():
    config = {"training_voxel_size_um": 4.8}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(4.8 / 7.91, config, 7.91)
    assert scale == pytest.approx(4.8 / 7.91)
    assert source == "cli"


def test_explicit_scale_wins_but_warns_on_metadata_mismatch():
    config = {"training_voxel_size_um": 4.8}
    with pytest.warns(UserWarning, match="differs from the checkpoint-derived value"):
        scale, source = resolve_displacement_scale(1.0, config, 7.91)
    assert scale == pytest.approx(1.0)
    assert source == "cli"


def test_explicit_scale_without_metadata_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(0.6068, {}, None)
    assert scale == pytest.approx(0.6068)
    assert source == "cli"


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_explicit_scale_raises(bad_scale):
    with pytest.raises(ValueError, match="displacement scale"):
        resolve_displacement_scale(bad_scale, {}, None)


@pytest.mark.parametrize("bad_voxel_size", [0.0, -4.8, float("nan"), "not-a-number"])
def test_invalid_metadata_voxel_size_is_ignored(bad_voxel_size):
    # Unusable metadata is treated as absent: unchanged behavior, no noise.
    config = {"training_voxel_size_um": bad_voxel_size}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scale, source = resolve_displacement_scale(None, config, 7.91)
    assert scale == pytest.approx(1.0)
    assert source == "default"


@pytest.mark.parametrize("bad_voxel_size", [0.0, -4.8, float("nan"), "not-a-number"])
def test_auto_rejects_invalid_metadata_voxel_size(bad_voxel_size):
    config = {"training_voxel_size_um": bad_voxel_size}
    with pytest.raises(ValueError, match="cannot derive a scale"):
        resolve_displacement_scale("auto", config, 7.91)


@pytest.mark.parametrize("value,expected", [("auto", True), ("AUTO", True), ("  auto  ", True),
                                            ("0.6", False), (0.6, False), (None, False)])
def test_is_auto_displacement_scale(value, expected):
    assert is_auto_displacement_scale(value) is expected
