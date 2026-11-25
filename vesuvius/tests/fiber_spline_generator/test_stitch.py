import numpy as np

from vesuvius.fiber_spline_generator.cli import GAP_MODEL
from vesuvius.fiber_spline_generator.extend.ct_support import CTSupport
from vesuvius.fiber_spline_generator.extend.gap_model import Predictor, geometry_features
from vesuvius.fiber_spline_generator.extend.stitch import Stitcher, trim_cut_ends

SHAPE_XYZ = (320, 96, 96)


def line(x0, x1, y=48.0, z=48.0):
    """ZYX points of a straight vertical-family spline along x."""
    x = np.arange(x0, x1 + 0.25, 0.5)
    return np.stack([np.full_like(x, z), np.full_like(x, y), x], axis=1)


def scorer():
    model = Predictor(GAP_MODEL)
    return lambda proposals, row: model.score(np.stack([
        geometry_features(p["a"], p["b"], p["context"], family=row["family"], voxel_microns=8.64) for p in proposals]))


def measure_with(ct):
    support = CTSupport(lambda low, high: ct[low[2]:high[2], low[1]:high[1], low[0]:high[0]], SHAPE_XYZ)

    def measure(a, b):
        return {"state": "available" if support.features(a, b)["valid"] else "unavailable"}
    return measure


def confident(proposals, row):
    return np.full(len(proposals), 0.95)


def whole(tmp_path, traces, ct, score=None):
    stitcher = Stitcher(tmp_path)
    try:
        stitcher.add_block("block-0", [0, 0, 0], list(SHAPE_XYZ), traces, (np.zeros(3), np.asarray(SHAPE_XYZ)))
        return stitcher.expand(score or scorer(), measure_with(ct))
    finally:
        stitcher.close()


def test_overlapping_splines_of_neighbouring_blocks_become_one_fiber(tmp_path):
    stitcher = Stitcher(tmp_path)
    try:
        # Block 0 owns x < 160, block 1 x >= 160; each sees 32 voxels past the border.
        stitcher.add_block("block-0", [0, 0, 0], [192, 96, 96], [("V", line(20, 190))], (np.zeros(3), np.array([160, 96, 96])))
        stitcher.add_block("block-1", [128, 0, 0], [192, 96, 96], [("V", line(1, 170))], (np.array([160, 0, 0]), np.array([320, 96, 96])))
        fibers = list(stitcher.chains())
    finally:
        stitcher.close()
    assert len(fibers) == 1
    points = fibers[0]["points"]
    assert np.isclose(points[:, 0].min(), 20) and np.isclose(points[:, 0].max(), 298)
    assert np.abs(points[:, 1:] - 48).max() < 1e-3


def hooked(points, x, offset=1.6, over=4.):
    """``points`` bent sideways over their last ``over`` voxels before ``x``, as a block border bends a spline."""
    bent = points.copy()
    bent[:, 1] += offset * np.clip(1 - np.abs(bent[:, 2] - x) / over, 0, None) ** 2
    return bent


def test_a_fiber_bent_at_both_block_borders_becomes_one_fiber_once_the_cut_ends_are_trimmed(tmp_path):
    blocks = [("block-0", [0, 0, 0], hooked(line(20, 191), 191), (np.zeros(3), np.array([160, 96, 96]))),
              ("block-1", [128, 0, 0], hooked(line(1, 170), 1), (np.array([160, 0, 0]), np.array([320, 96, 96])))]
    for trim, count in ((False, 2), (True, 1)):
        stitcher = Stitcher(tmp_path / str(trim))
        try:
            for name, low, spline, core in blocks:
                traces = [("V", spline)]
                if trim:
                    traces = trim_cut_ends(traces, low[::-1], np.add(low, [192, 96, 96])[::-1],
                                           (0, 0, 0), SHAPE_XYZ[::-1])
                stitcher.add_block(name, low, [192, 96, 96], traces, core)
            fibers = list(stitcher.chains())
        finally:
            stitcher.close()
        assert len(fibers) == count
    assert np.isclose(fibers[0]["points"][:, 0].min(), 20) and np.isclose(fibers[0]["points"][:, 0].max(), 298)


def test_only_ends_cut_by_a_border_another_block_reads_past_are_trimmed():
    # The second block of a zone filling a 320-voxel volume reads x 128..320: another block reads past its low x border.
    low, high, reach = (0, 0, 128), (96, 96, 320), ((0, 0, 0), (96, 96, 320))
    (_, kept), = trim_cut_ends([("V", line(1, 191))], low, high, *reach)
    assert np.isclose(kept[:, 2].min(), 16) and np.isclose(kept[:, 2].max(), 191)
    # The end at the volume's border stays; so does a spline whose ends are away from every border.
    (_, kept), = trim_cut_ends([("V", line(40, 191.5))], low, high, *reach)
    assert np.isclose(kept[:, 2].min(), 40) and np.isclose(kept[:, 2].max(), 191.5)
    # Nothing is left of a spline lying along the cut border.
    assert trim_cut_ends([("V", line(0, 10))], low, high, *reach) == []


def test_a_zone_of_one_block_inside_the_volume_is_not_trimmed():
    # A one-block zone at x 64..192 of a 320-voxel volume reads x 32..224: no other block reads past either border.
    low, high = (0, 0, 32), (96, 96, 224)
    (_, kept), = trim_cut_ends([("V", line(1, 191))], low, high, reach_low_zyx=low, reach_high_zyx=high)
    assert np.array_equal(kept, line(1, 191))
    # The second block of a zone reading y 32..128 and x 0..320: an end beside the zone's own y border stays, as one
    # beside the volume's border does, though another block reads past its low x border.
    low, high, reach = (0, 32, 128), (96, 128, 320), ((0, 32, 0), (96, 128, 320))
    (_, kept), = trim_cut_ends([("V", line(1, 191, y=1.0))], low, high, *reach)
    assert np.isclose(kept[:, 2].min(), 1)
    # And at the far borders: the first block, x 0..192, with a spline beside the zone's high y border.
    (_, kept), = trim_cut_ends([("V", line(1, 191, y=94.0))], (0, 32, 0), (96, 128, 192), *reach)
    assert np.isclose(kept[:, 2].max(), 191)


def test_a_short_aligned_gap_is_joined_by_the_catalogue(tmp_path):
    fibers = whole(tmp_path, [("V", line(10, 130)), ("V", line(146, 290))], np.zeros(SHAPE_XYZ[::-1], np.uint8))
    assert len(fibers) == 1 and fibers[0]["gaps"] == [[240, 241]]
    assert np.isclose(fibers[0]["points"][:, 0].min(), 10) and np.isclose(fibers[0]["points"][:, 0].max(), 290)


def test_a_longer_gap_is_joined_only_with_ct_at_the_join(tmp_path):
    # 36 voxels: beyond the catalogue's gaps, the Expander decides.
    traces = [("V", line(10, 130)), ("V", line(166, 290))]
    joined = whole(tmp_path / "ct", traces, np.full(SHAPE_XYZ[::-1], 100, np.uint8), confident)
    assert len(joined) == 1 and len(joined[0]["chains"]) == 2 and joined[0]["gaps"]
    points = joined[0]["points"]
    assert np.isclose(points[:, 0].min(), 10) and np.isclose(points[:, 0].max(), 290)
    unmeasured = whole(tmp_path / "empty", traces, np.zeros(SHAPE_XYZ[::-1], np.uint8), confident)
    assert len(unmeasured) == 2
    assert {f["stops"]["0"]["code"] for f in unmeasured} | {f["stops"]["1"]["code"] for f in unmeasured} >= {"ct_unavailable"}


def test_diverging_splines_are_not_joined(tmp_path):
    turned = line(166, 290)
    turned[:, 1] = 48 + (turned[:, 2] - 166)  # 45 degrees away from the first spline
    turned = turned[turned[:, 1] < 95]
    fibers = whole(tmp_path, [("V", line(10, 130)), ("V", turned)], np.full(SHAPE_XYZ[::-1], 100, np.uint8))
    assert len(fibers) == 2


def test_short_splines_are_kept_once_by_the_block_owning_them(tmp_path):
    stitcher = Stitcher(tmp_path)
    try:
        short = [("H", line(150, 162))]
        stitcher.add_block("block-0", [0, 0, 0], [192, 96, 96], short, (np.zeros(3), np.array([160, 96, 96])))
        stitcher.add_block("block-1", [128, 0, 0], [192, 96, 96], [("H", line(22, 34))], (np.array([160, 0, 0]), np.array([320, 96, 96])))
        fibers = stitcher.expand(scorer(), lambda a, b: {"state": "available"})
    finally:
        stitcher.close()
    assert len(fibers) == 1 and fibers[0]["family"] == 1
