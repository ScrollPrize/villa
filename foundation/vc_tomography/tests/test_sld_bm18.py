"""BM18 calibration import, pinned upstream numerics, and reconstruction integration."""
import json
from pathlib import Path

import numpy as np
import pytest

import helirec
from helirec import _core
from helirec.sld import correct, sld_config
from test_sld import scan


def test_pinned_nabuxx_full_resolution():
    reference = json.loads((Path(__file__).parent / "reference" / "sld_bm18.json").read_text())
    frames = np.array(reference["frames"], np.float32)
    for case in reference["cases"]:
        got = correct(frames, case["model"])
        np.testing.assert_allclose(got, case["corrected"], rtol=3e-6, atol=8e-7)
    # A second component changes the PSF, and the illuminated boundary matters.
    model = reference["cases"][0]["model"]
    single = dict(model, fraction=[.2], scale_l=[4.])
    assert np.max(np.abs(correct(frames, model) - correct(frames, single))) > .01
    dark = dict(model, border_x=0, border_y=0)
    assert np.max(np.abs(correct(frames, model) - correct(frames, dark))) > .01


def test_json_and_inline_model(tmp_path):
    model = dict(fraction=[.12, .08], scale_l=[40., 200.], replica_factor=0., replica_shift_x=0,
                 replica_shift_y=0, save_binning=1, source_files=["calibration.edf"], roi=[0, 5])
    path = tmp_path / "BM18 calibration.json"
    path.write_text(json.dumps(model))
    expected = sld_config("profile=bm18,fraction=0.12:0.08,length=40:200")
    assert sld_config(path) == expected
    assert sld_config(f"json={path}") == expected
    assert expected["sld_bin"] == expected["sld_stride"] == 1
    assert expected["sld_border_x"] == expected["sld_border_y"] == 100
    overrides = sld_config(dict(json=path, border_x=0, stride=4))
    assert overrides["sld_border_x"] == 0 and overrides["sld_stride"] == 4
    model["save_binning"] = 4
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="full detector pixels"):
        sld_config(path)


@pytest.mark.parametrize("model", [
    dict(fraction=[.1, .2], scale_l=[100]), dict(fraction=[-.1], scale_l=[100]),
    dict(fraction=[float("nan")], scale_l=[100]), dict(fraction=[.1], scale_l=[float("inf")]),
    dict(fraction=[.1], scale_l=[0]), dict(fraction=[], scale_l=[]),
    dict(fraction=.1, profile="bm18", bin=8), dict(fraction=.1, stride=1.5),
    dict(fraction=.1, profile="bm18", border_x=-1), dict(fraction=.1, replica_factor=.01),
    dict(fraction=.1, replica_shift_x=3), dict(fraction=.1, profile="unknown"),
    dict(fraction=.1, scale_l=100, length=100), dict(fraction=.1, surprise=1),
])
def test_invalid_calibrations_fail_early(model):
    with pytest.raises(ValueError):
        sld_config(model)


def test_native_validation_and_disabled():
    frames = np.ones((1, 9, 13), np.float32)
    np.testing.assert_array_equal(correct(frames, ""), frames)
    cfg = sld_config(dict(profile="bm18", fraction=.1, border_x=0, border_y=0))
    for key, value, message in [("sld_bin", 8, "bin=1"), ("sld_border_x", -1, "non-negative"),
                                ("sld_lengths", [0.], "positive"), ("sld_fractions", [-1.], "non-negative")]:
        with pytest.raises(Exception, match=message):
            _core.sld_correct_config(frames, dict(cfg, **{key: value}))
    # A short, excessively strong tail can make the discrete PSF non-invertible.
    with pytest.raises(Exception, match="invertible"):
        correct(frames, dict(profile="bm18", fraction=10, length=.001, border_x=0, border_y=0))


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("stride", [1, 3])
def test_bm18_pipeline_against_corrected_input(tmp_path, device, stride):
    """Reference frames are corrected independently, then gridded with SLD off.

    Also checks interpolation of current-normalized maps, cropped map strips,
    non-integer detector row interpolation and independence from chunk height.
    """
    nr, nc, n = 24, 48, 81
    p = np.arange(n)
    angles, z = p * 360 / 32 + 1., p * .14
    current = .2 * (1 + .1 * np.sin(p / 10))
    y, x = np.mgrid[:nr, :nc]
    flat = (25000 + 5000 * np.exp(-((y - nr / 2) / 8) ** 2)) * np.ones((1, nc))
    signal = flat[None] * (1 - .35 * np.exp(-((x[None] - 20 - 5 * np.sin(p[:, None, None] / 12)) / 6) ** 2))
    dark = np.full((nr, nc), 100., np.float32)
    raw = np.rint(signal * (current / .2)[:, None, None] + dark).astype(np.uint16)
    model = dict(fraction=[.12, .04], scale_l=[2., 8.], border_x=5, border_y=2, stride=stride)
    cfg = dict(device=device, pixel_size_m=.01, rot_center=(nc - 1) / 2, grid_size=nc + 1, n_grid=32,
               backprojector="fbp", luminosity_repartition=0., filters=[], block_angles=8,
               energy_kev=100., distance_m=.1, z_interp="lanczos3")
    seen = scan(tmp_path, "seen", raw, flat + dark, dark, angles, z, current)
    rec = _core.SliceReconstructor([seen], dict(cfg, **sld_config(model)))
    # Reproduce only temporal map interpolation independently in NumPy.
    normalized = (raw.astype(np.float32) - dark) / (current / .2).astype(np.float32)[:, None, None]
    indices = np.unique(np.minimum(np.arange(0, n + stride, stride), n - 1))
    corrected_samples = correct(normalized[indices], model)
    glare = normalized[indices] - corrected_samples
    corrections = np.empty_like(normalized)
    for j in range(nr):
        for k in range(nc):
            corrections[:, j, k] = np.interp(p, indices, glare[:, j, k])
    fixed = normalized - corrections
    fixed_flat = correct(flat[None], model)[0]
    # Quantize the independent corrected input once; tolerance includes rounding.
    clean = scan(tmp_path, "clean", np.rint(fixed * (current / .2)[:, None, None] + dark).astype(np.uint16),
                 fixed_flat + dark, dark, angles, z, current)
    reference = _core.SliceReconstructor([clean], cfg)
    lo, hi = rec.info["doable"]
    h = (lo + hi) // 2
    got = rec.reconstruct(h, h + 3, debug=True)["sino"]
    want = reference.reconstruct(h, h + 3, debug=True)["sino"]
    np.testing.assert_allclose(got, want, rtol=0, atol=9e-5)
    small = rec.reconstruct(h + 1, h + 2, debug=True)["sino"]
    np.testing.assert_allclose(small, got[:, 1:2], rtol=0, atol=2e-6)
