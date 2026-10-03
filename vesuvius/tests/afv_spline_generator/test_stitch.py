import numpy as np

from vesuvius.afv_spline_generator.cli import GAP_MODEL
from vesuvius.afv_spline_generator.extend.ct_support import CTSupport
from vesuvius.afv_spline_generator.extend.gap_model import Predictor, geometry_features
from vesuvius.afv_spline_generator.extend.stitch import Stitcher

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
