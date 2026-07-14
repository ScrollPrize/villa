"""M2.5 ink bake — synthetic 64x64 grading checks (see docs/QUALITY-GATES.md).

Asserts on compose_overlay/grade_overlay: alpha untouched, amber tint applied
where ink is high, base preserved exactly where ink is 0 (beyond glow reach),
screen glow strictly non-darkening, plus polarity verification and resampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from scrollkit.ink.bake import (
    DEFAULT_PARAMS,
    block_mean,
    choose_polarity,
    compose_overlay,
    grade_overlay,
    registration_check,
    resample_ink,
    smoothstep,
)

N = 64
TINT = np.array(DEFAULT_PARAMS["tint_rgb"], dtype=np.float32)


@pytest.fixture()
def base() -> np.ndarray:
    """Known RGBA base: random papyrus-ish RGB, alpha with 0 / 200 / 255 values."""
    rng = np.random.default_rng(7)
    rgba = np.empty((N, N, 4), np.uint8)
    rgba[..., :3] = rng.integers(60, 200, (N, N, 3))
    rgba[..., 3] = 0
    rgba[4:60, 4:60, 3] = 255
    rgba[4:8, 4:8, 3] = 200  # mid alpha values must survive verbatim too
    return rgba


@pytest.fixture()
def ink_blob() -> np.ndarray:
    """Synthetic ink: zero everywhere, a value-1.0 disc (r=5) centred at (16,16)."""
    yy, xx = np.mgrid[0:N, 0:N]
    return ((yy - 16) ** 2 + (xx - 16) ** 2 <= 5**2).astype(np.float32)


def test_alpha_untouched(base, ink_blob):
    out, _ = compose_overlay(base, ink_blob, DEFAULT_PARAMS)
    assert out.dtype == np.uint8 and out.shape == base.shape
    assert np.array_equal(out[..., 3], base[..., 3])


def test_base_preserved_where_ink_zero(base, ink_blob):
    """Pixels beyond ink AND beyond gaussian reach (4*sigma truncation) are byte-equal."""
    out, op = compose_overlay(base, ink_blob, DEFAULT_PARAMS)
    assert np.array_equal(op == 0, ink_blob < DEFAULT_PARAMS["opacity_lo"])
    far = np.s_[40:, 40:]  # blob edge at ~21 px + glow radius 12 px << 40
    assert np.array_equal(out[far][..., :3], base[far][..., :3])


def test_tint_applied_where_ink_high(base, ink_blob):
    p0 = dict(DEFAULT_PARAMS, glow_strength=0.0)
    out0, op = grade_overlay(base[..., :3], ink_blob, p0)
    c = (16, 16)
    # full smoothstep saturation at ink=1.0 -> opacity == scale
    assert op[c] == pytest.approx(DEFAULT_PARAMS["opacity_scale"])
    # glow-free output == the lerp formula exactly (incl. round-half-even), everywhere
    expected = np.clip(
        np.rint(base[..., :3].astype(np.float32) * (1 - op[..., None]) + TINT * op[..., None]), 0, 255
    ).astype(np.uint8)
    assert np.array_equal(out0, expected)
    # the lerp moves every channel toward the amber tint
    assert np.all(np.abs(out0[c] - TINT) <= np.abs(base[c][:3].astype(np.float32) - TINT))
    # full pipeline: screen glow only brightens on top of the lerp
    out, _ = compose_overlay(base, ink_blob, DEFAULT_PARAMS)
    assert np.all(out[c][:3].astype(int) >= out0[c].astype(int))
    # amber channel ordering R > G > B at a saturated ink pixel
    got = out[c][:3].astype(np.float32)
    assert got[0] > got[1] > got[2]


def test_screen_glow_monotone(base, ink_blob):
    """screen(out, tint*G) never darkens any channel anywhere, and grows with strength."""
    p0 = dict(DEFAULT_PARAMS, glow_strength=0.0)
    p1 = dict(DEFAULT_PARAMS, glow_strength=0.30)
    p2 = dict(DEFAULT_PARAMS, glow_strength=0.60)
    out0, _ = grade_overlay(base[..., :3], ink_blob, p0)
    out1, _ = grade_overlay(base[..., :3], ink_blob, p1)
    out2, _ = grade_overlay(base[..., :3], ink_blob, p2)
    assert np.all(out1.astype(int) >= out0.astype(int))
    assert np.all(out2.astype(int) >= out1.astype(int))
    assert out1.sum() > out0.sum()  # glow visibly adds light near the blob


def test_smoothstep_edges():
    lo, hi = DEFAULT_PARAMS["opacity_lo"], DEFAULT_PARAMS["opacity_hi"]
    x = np.array([0.0, lo, (lo + hi) / 2, hi, 1.0], np.float32)
    s = smoothstep(x, lo, hi)
    assert s[0] == 0 and s[1] == 0 and s[3] == 1 and s[4] == 1
    assert 0.49 < s[2] < 0.51  # midpoint of the band -> 0.5: fibers visible through midtones


def test_polarity_flip_on_mislabel():
    """Dark-ink raster labelled 'positive' must be flipped by the sparse-tail check."""
    rng = np.random.default_rng(1)
    ink255 = np.full((N, N), 220.0, np.float32) + rng.normal(0, 3, (N, N)).astype(np.float32)
    ink255[10:14, 10:40] = 40.0  # dark stroke, ~3% of the frame
    mask = np.ones((N, N), bool)
    thr_mid = (DEFAULT_PARAMS["opacity_lo"] + DEFAULT_PARAMS["opacity_hi"]) / 2
    n, info = choose_polarity(ink255, mask, "positive (ink bright)", thr_mid)
    assert info["flipped_vs_audit"] and info["used"].startswith("inverted") and info["in_band"]
    assert 0.005 <= info["coverage_used"] <= 0.35
    assert n[12, 20] > 0.8 and n[40, 40] < 0.2  # ink=high after fix
    # and a correctly-labelled inverted raster is kept as-is
    _, info2 = choose_polarity(ink255, mask, "inverted (ink dark)", thr_mid)
    assert not info2["flipped_vs_audit"] and info2["in_band"]


def test_registration_identity_and_flip():
    # asymmetric mask (top-left block) so the flip variants are distinguishable
    mask = np.zeros((N, N), bool)
    mask[:32, :32] = True
    ink = np.zeros((N, N), np.float32)
    ink[4:12, 8:24] = 1.0  # ink inside the top-left block
    kept, reg = registration_check(ink, mask)
    assert reg["variant_used"] == "identity" and reg["anomaly"] is None
    assert reg["outside_frac"] == 0.0 and kept is ink
    # mirrored ink (all on the wrong side) must be caught and decisively flipped back
    bad = ink[:, ::-1].copy()
    kept2, reg2 = registration_check(bad, mask)
    assert reg2["anomaly"] is not None and reg2["anomaly"]["decisive"]
    assert reg2["variant_used"] == "fliplr" and reg2["outside_frac"] == 0.0
    assert np.array_equal(kept2, ink)


def test_block_mean_and_resample_two_step():
    rng = np.random.default_rng(2)
    src = rng.integers(0, 256, (130, 97), np.uint8)
    ds, crop = block_mean(src, 4)
    assert ds.shape == (32, 24) and crop == (2, 1)
    assert ds[0, 0] == pytest.approx(src[:4, :4].mean(), abs=1e-4)
    # 5x source -> two-step lands on exact target dims, values stay in range
    big = np.tile(src, (5, 5))[: 64 * 5, : 32 * 5]
    out, steps = resample_ink(big, (32, 64), two_step=True)
    assert out.shape == (64, 32) and out.dtype == np.float32
    assert [s["op"] for s in steps] == ["block_mean", "lanczos"]
    assert steps[0]["factor"] == 3 and float(out.min()) >= 0.0 and float(out.max()) <= 255.0
    # constant raster survives the whole chain exactly
    flat, _ = resample_ink(np.full((300, 200), 130, np.uint8), (40, 60), two_step=True)
    assert np.allclose(flat, 130.0, atol=0.51)
