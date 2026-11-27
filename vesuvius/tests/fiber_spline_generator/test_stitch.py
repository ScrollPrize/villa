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


# Two blocks of a zone filling the 320-voxel volume along x: block 0 owns x < 160 and reads x 0..192, block 1 owns
# x >= 160 and reads x 128..320. Each block's splines are block-local ZYX; neighbours are absolute.
READS = {"block-0": ((0, 0, 0), (96, 96, 192)), "block-1": ((0, 0, 128), (96, 96, 320))}
REACH = ((0, 0, 0), (96, 96, 320))


def trimmed(name, spline, other_name, other_spline, reads=READS, reach=REACH):
    """``spline`` (absolute ZYX) of block ``name``, trimmed against ``other_spline`` of the other block."""
    low, high = reads[name]
    other_low, other_high = reads[other_name]
    (family, kept), = trim_cut_ends([("V", spline - np.asarray(low))], low, high, *reach,
                                    [("V", other_spline, other_low, other_high)])
    return kept + np.asarray(low)


def stitch(tmp_path, block0, block1):
    stitcher = Stitcher(tmp_path)
    try:
        stitcher.add_block("block-0", [0, 0, 0], [192, 96, 96], [("V", block0)], (np.zeros(3), np.array([160, 96, 96])))
        stitcher.add_block("block-1", [128, 0, 0], [192, 96, 96], [("V", block1 - [0, 0, 128])],
                           (np.array([160, 0, 0]), np.array([320, 96, 96])))
        return list(stitcher.chains())
    finally:
        stitcher.close()


def test_a_fiber_bent_at_both_block_borders_becomes_one_fiber_once_the_cut_ends_are_trimmed(tmp_path):
    a, b = hooked(line(20, 191), 191), hooked(line(129, 298), 129)
    assert len(stitch(tmp_path / "main", a, b)) == 2
    a2, b2 = trimmed("block-0", a, "block-1", b), trimmed("block-1", b, "block-0", a)
    assert np.isclose(a2[:, 2].max(), 175) and np.isclose(b2[:, 2].min(), 144)
    fibers = stitch(tmp_path / "trimmed", a2, b2)
    assert len(fibers) == 1
    assert np.isclose(fibers[0]["points"][:, 0].min(), 20) and np.isclose(fibers[0]["points"][:, 0].max(), 298)


def test_an_end_is_kept_when_trimming_would_leave_the_copies_too_little_overlap(tmp_path):
    # Block 1's copy starts late, at x 160: untrimmed, the copies share 28 voxels and join, as on main. Trimming
    # block 0's end at x 176 would leave them 15.5, under the catalogue's 24, so the end is kept.
    a, b = line(20, 188), line(160, 298)
    assert np.array_equal(trimmed("block-0", a, "block-1", b), a)
    assert np.array_equal(trimmed("block-1", b, "block-0", a), b)
    assert len(stitch(tmp_path, a, b)) == 1


def test_an_end_no_neighbouring_block_detects_is_kept():
    # Only block 0 sees this fiber's end in the cut band: block 1's spline runs 20 voxels away.
    a, elsewhere = line(20, 190), line(130, 298, y=68.0)
    assert np.array_equal(trimmed("block-0", a, "block-1", elsewhere), a)
    # A trace lying wholly in the cut band, along the block face, is kept whatever the neighbour holds.
    along = np.stack([np.full(125, 48.0), np.linspace(-0.5, 61.5, 125) + 17, np.full(125, 185.0)], axis=1)
    assert np.array_equal(trimmed("block-0", along, "block-1", line(130, 298)), along)


# Two default 512-voxel blocks of a 1024-voxel zone: block 0 owns x < 512 and reads x 0..544, block 1 owns x >= 512
# and reads x 480..1024.
DEFAULT_READS = {"block-0": ((0, 0, 0), (160, 160, 544)), "block-1": ((0, 0, 480), (160, 160, 1024))}
DEFAULT_REACH = ((0, 0, 0), (160, 160, 1024))


def test_main_s_join_at_the_default_block_size_is_kept(tmp_path):
    # Main joins the copies at x 100..537 and 507..770. Trimming block 0's end would leave them 19.5 voxels of overlap.
    a, b = line(100, 537), line(507, 770)
    assert np.array_equal(trimmed("block-0", a, "block-1", b, DEFAULT_READS, DEFAULT_REACH), a)
    assert np.array_equal(trimmed("block-1", b, "block-0", a, DEFAULT_READS, DEFAULT_REACH), b)
    stitcher = Stitcher(tmp_path)
    try:
        stitcher.add_block("block-0", [0, 0, 0], [544, 160, 160], [("V", a)], (np.zeros(3), np.array([512, 160, 160])))
        stitcher.add_block("block-1", [480, 0, 0], [544, 160, 160], [("V", b - [0, 0, 480])],
                           (np.array([512, 0, 0]), np.array([1024, 160, 160])))
        assert len(list(stitcher.chains())) == 1
    finally:
        stitcher.close()


def test_a_trace_only_one_block_detects_along_its_face_is_kept():
    # Block 0 alone predicts a 125-voxel trace parallel to its face at x 535; block 1's nearest spline is far away.
    along = np.stack([np.full(251, 48.0), np.linspace(10, 135, 251), np.full(251, 535.0)], axis=1)
    kept = trimmed("block-0", along, "block-1", line(600, 900), DEFAULT_READS, DEFAULT_REACH)
    assert np.array_equal(kept, along)


def test_only_ends_cut_by_a_border_another_block_reads_past_are_trimmed():
    # Block 1's low x end has block 0's copy beside it: trimmed. Its high x end lies at the volume's border: kept.
    b = trimmed("block-1", line(129, 319), "block-0", line(20, 191))
    assert np.isclose(b[:, 2].min(), 144) and np.isclose(b[:, 2].max(), 319)
    # A one-block zone at x 64..192 of a 320-voxel volume reads x 32..224: no other block reads past either border.
    low, high = (0, 0, 32), (96, 96, 224)
    (_, kept), = trim_cut_ends([("V", line(1, 191))], low, high, reach_low_zyx=low, reach_high_zyx=high)
    assert np.array_equal(kept, line(1, 191))


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
