"""Grey-level anchors: debris takes no part, and a picture without both peaks falls back."""

import numpy as np

from volreg import render


def histogram(debris=0.0, seed=0):
    rng = np.random.default_rng(seed)
    air = rng.normal(50, 6, 40_000)
    papyrus = rng.normal(130, 18, 60_000)
    blobs = rng.normal(240, 5, int(debris * 100_000))
    values = np.clip(np.concatenate([air, papyrus, blobs]), 1, 254).astype(int)
    return np.bincount(values, minlength=256)[:256]


def test_anchors_find_air_and_the_papyrus_edge():
    found = render.anchors(histogram())
    assert abs(found["air"] - 50) <= 2 and abs(found["papyrus_mode"] - 130) <= 3
    # a gaussian falls to a tenth of its height at 2.15 sigma
    assert abs(found["papyrus_e10"] - (130 + 2.15 * 18)) <= 3


def test_debris_does_not_move_the_anchors():
    clean, dirty = render.anchors(histogram()), render.anchors(histogram(debris=0.08))
    for key in clean:
        assert abs(clean[key] - dirty[key]) <= 1.0


def test_both_sides_land_on_the_same_grey_levels():
    anchor = render.anchors(histogram())
    mapped = render.grey(np.array([anchor["air"], anchor["papyrus_e10"], 255.0]), anchor)
    assert abs(int(mapped[0]) - render.AIR_GREY) <= 1 and abs(int(mapped[1]) - render.PAPYRUS_GREY) <= 1
    assert mapped[2] == 255          # debris saturates


def test_no_two_peaks_falls_back_to_the_stretch():
    one_peak = np.bincount(np.clip(np.random.default_rng(1).normal(120, 20, 50_000), 1, 254).astype(int),
                           minlength=256)[:256]
    assert render.anchors(one_peak) is None
    assert render.grey(np.arange(256, dtype=float), None).dtype == np.uint8
