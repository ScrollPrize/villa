"""Logic tests for the flat-inference input contract (tiny synthetic arguments).

These test the rules only. The real-data behaviour (released checkpoints on a
labelled segment) is shown in the pull request, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.ink_detection.inference.input_contract import (
    ERROR,
    FORMAT_TAG,
    InputContractError,
    check_input_contract,
    enforce_input_contract,
    expected_source_depth,
    require_finite_probabilities,
)

CENTRED_17_OF_21 = np.arange(2, 19)


def _check(**overrides):
    kwargs = dict(
        depth=21,
        dtype=np.uint8,
        attrs=None,
        layer_indices=CENTRED_17_OF_21,
        direction="forward",
        window_depth=17,
        preprocessing="tifxyz_robust",
        max_z_offset=2,
    )
    kwargs.update(overrides)
    return check_input_contract(**kwargs)


def _codes(findings, severity=None):
    return {f.code for f in findings if severity is None or f.severity == severity}


def test_expected_source_depth_is_window_plus_jitter():
    assert expected_source_depth(17, 2) == 21
    assert expected_source_depth(17, 0) is None


def test_upstream_form_has_no_errors_or_warnings():
    findings = _check()
    assert _codes(findings, ERROR) == set()
    assert {f.severity for f in findings} == {"info"}


def test_17_slice_input_is_refused_as_not_the_trained_form():
    findings = _check(depth=17, layer_indices=np.arange(17))
    assert "DEPTH_NOT_TRAINED_FORM" in _codes(findings, ERROR)


def test_too_shallow_input_is_refused():
    assert "DEPTH_TOO_SHALLOW" in _codes(
        _check(depth=9, layer_indices=np.arange(9)), ERROR
    )


def test_recipe_without_declared_depth_skips_depth_mismatch():
    findings = _check(depth=17, layer_indices=np.arange(17), max_z_offset=0)
    assert _codes(findings, ERROR) == set()


def test_deeper_volume_warns_only():
    findings = _check(depth=28, layer_indices=np.arange(5, 22))
    assert _codes(findings, ERROR) == set()
    assert "DEPTH_DEEPER_THAN_TRAINED" in _codes(findings)


@pytest.mark.parametrize(
    ("attrs", "code"),
    [
        ({"format": FORMAT_TAG, "source_level": "0"}, "SCALE_MISMATCH"),
        ({"format": FORMAT_TAG, "source_z_slice": [10, 90]}, "Z_POOL_MISMATCH"),
    ],
)
def test_prepared_input_provenance_errors(attrs, code):
    assert code in _codes(_check(attrs=attrs), ERROR)


def test_prepared_input_with_matching_provenance_is_clean():
    attrs = {
        "format": FORMAT_TAG,
        "source_level": "2",
        "source_shape_zyx": [109, 100, 100],
        "source_z_slice": [13, 97],
    }
    findings = _check(attrs=attrs)
    assert {f.severity for f in findings} == {"info"}


def test_off_centre_pooled_window_warns():
    attrs = {
        "format": FORMAT_TAG,
        "source_level": "2",
        "source_shape_zyx": [109, 100, 100],
        "source_z_slice": [12, 96],
    }
    assert "Z_WINDOW_OFF_CENTRE" in _codes(_check(attrs=attrs), "warn")


def test_unknown_format_warns_and_missing_format_is_info():
    assert "UNKNOWN_FORMAT" in _codes(_check(attrs={"format": "other"}), "warn")
    assert "NO_PROVENANCE" in _codes(_check(), "info")


def test_dtype_rules():
    assert "DTYPE" in _codes(
        _check(dtype=np.float32, preprocessing="divide_255"), ERROR
    )
    assert "DTYPE" in _codes(
        _check(dtype=np.float32, attrs={"format": FORMAT_TAG}), "warn"
    )


def test_window_size_and_off_centre_window():
    assert "WINDOW_SIZE" in _codes(_check(layer_indices=np.arange(10)), ERROR)
    shifted = _check(layer_indices=np.arange(0, 17), max_z_offset=1)
    assert "Z_WINDOW_OUTSIDE_TRAINED_RANGE" in _codes(shifted, "warn")
    assert "Z_WINDOW_OUTSIDE_TRAINED_RANGE" not in _codes(_check(layer_indices=np.arange(0, 17)))


def test_direction_is_reported_and_reverse_warns():
    forward = [f for f in _check() if f.code == "DIRECTION"][0]
    assert forward.severity == "info" and "2..18" in forward.message
    reverse = [
        f
        for f in _check(direction="reverse", layer_indices=CENTRED_17_OF_21[::-1])
        if f.code == "DIRECTION"
    ][0]
    assert reverse.severity == "warn" and "18..2" in reverse.message
    assert "descending" in reverse.message
    both = [f for f in _check(direction="both") if f.code == "DIRECTION"][0]
    assert both.severity == "info"


def test_enforce_modes(caplog):
    findings = _check(depth=17, layer_indices=np.arange(17))
    with pytest.raises(InputContractError, match="DEPTH_NOT_TRAINED_FORM"):
        enforce_input_contract(findings, mode="error", label="unit")
    enforce_input_contract(findings, mode="warn", label="unit")
    assert "DEPTH_NOT_TRAINED_FORM" in caplog.text
    caplog.clear()
    enforce_input_contract(findings, mode="off", label="unit")
    assert caplog.text == ""
    with pytest.raises(ValueError, match="mode"):
        enforce_input_contract(findings, mode="strict")


def test_non_finite_probabilities_raise():
    require_finite_probabilities(np.zeros((2, 4, 4), dtype=np.float32))
    for bad in (np.nan, np.inf):
        values = np.zeros((2, 4, 4), dtype=np.float32)
        values[0, 1, 1] = bad
        with pytest.raises(FloatingPointError, match="non-finite"):
            require_finite_probabilities(values)


def test_default_mode_is_warn_and_continues(caplog):
    from vesuvius.ink_detection.inference.infer import parse_args

    assert parse_args(["in.zarr", "ck.pth", "out.tif"]).input_contract == "warn"
    findings = _check(depth=17, layer_indices=np.arange(17))
    enforce_input_contract(findings, mode="warn", label="unit")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("WARNING [DEPTH_NOT_TRAINED_FORM]" in r.getMessage() for r in warnings)


def test_error_mode_refuses_and_off_is_silent(caplog):
    findings = _check(depth=17, layer_indices=np.arange(17))
    with pytest.raises(InputContractError, match="DEPTH_NOT_TRAINED_FORM"):
        enforce_input_contract(findings, mode="error", label="unit")
    caplog.clear()
    enforce_input_contract(findings, mode="off", label="unit")
    assert caplog.records == []


@pytest.mark.parametrize("mode", ["warn", "error", "off"])
def test_non_finite_raises_in_every_mode(mode):
    values = np.zeros((1, 4, 4), dtype=np.float32)
    values[0, 0, 0] = np.nan
    # The finite guard takes no mode: the contract mode cannot switch it off.
    enforce_input_contract([], mode=mode, label="unit")
    with pytest.raises(FloatingPointError):
        require_finite_probabilities(values)
