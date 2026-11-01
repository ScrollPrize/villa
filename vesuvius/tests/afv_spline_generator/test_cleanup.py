import numpy as np
import pytest

from vesuvius.afv_spline_generator import cleanup


def line(start, end, step=0.5):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = int(np.ceil(np.linalg.norm(end - start) / step)) + 1
    return np.linspace(start, end, count)


def joined(*pieces):
    """One fiber of observed pieces; consecutive pieces are joined by an inferred straight bridge."""
    points, gaps = [], []
    for piece in pieces:
        if points:
            gaps.append([len(points) - 1, len(points)])
        points.extend(piece.tolist())
    return dict(family=0, points=np.asarray(points), gaps=gaps)


def test_joins_turning_too_much_are_cut_and_the_bridge_dropped():
    first, second = line([0, 0, 0], [40, 0, 0]), line([42, 2, 0], [42, 42, 0])
    fiber = joined(first, second)
    assert cleanup.join_angle(fiber["points"], cleanup.arclength(fiber["points"]), *fiber["gaps"][0]) == pytest.approx(90)
    assert cleanup.cut_sharp_joins([fiber], 120) == [fiber]
    parts = cleanup.cut_sharp_joins([fiber], 45)
    assert [len(p["points"]) for p in parts] == [len(first), len(second)]
    assert np.array_equal(parts[0]["points"], first) and np.array_equal(parts[1]["points"], second)
    assert all(p["gaps"] == [] for p in parts)


def test_remaining_joins_follow_their_piece():
    straight = joined(line([0, 0, 0], [40, 0, 0]), line([43, 0, 0], [80, 0, 0]), line([82, 2, 0], [82, 40, 0]))
    kept, cut = straight["gaps"]
    parts = cleanup.cut_sharp_joins([straight], 45)
    assert len(parts) == 2 and parts[0]["gaps"] == [kept] and parts[1]["gaps"] == []
    assert len(parts[0]["points"]) == cut[0] + 1


def test_short_fibers_are_removed_unless_the_minimum_is_zero():
    fibers = [dict(family=0, points=line([0, 0, 0], [length, 0, 0]), gaps=[]) for length in (10, 31.9, 32, 100)]
    assert [len(f["points"]) for f in cleanup.clean(fibers, 45, 32)] == [2, 2]
    assert len(cleanup.clean(fibers, 45, 0)) == 4


def segment_distances(points, polyline):
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    t = np.clip(np.einsum("pij,ij->pi", points[:, None] - a, d) / np.einsum("ij,ij->i", d, d), 0, 1)
    return np.linalg.norm(points[:, None] - (a + t[..., None] * d), axis=2).min(axis=1)


def test_thinning_keeps_ends_joins_and_shape():
    s = np.linspace(0, 6 * np.pi, 4000)
    points = np.stack([20 * np.cos(s), 20 * np.sin(s), 3 * s], axis=1)
    kept = cleanup.simplify(points, pins=[1234, 1235])
    assert kept[0] == 0 and kept[-1] == len(points) - 1 and {1234, 1235} <= set(kept.tolist())
    assert len(kept) < len(points) / 4
    assert segment_distances(points, points[kept]).max() <= cleanup.TOLERANCE + 1e-9
    full, thinned = cleanup.arclength(points)[-1], cleanup.arclength(points[kept])[-1]
    assert full - thinned <= cleanup.LENGTH_LOSS * full
    assert np.array_equal(cleanup.simplify(line([0, 0, 0], [100, 0, 0])), [0, 200])


def test_thinning_keeps_join_indices_on_their_points():
    fiber = joined(line([0, 0, 0], [40, 0, 0]), line([43, 0, 0], [80, 0, 0]))
    thin = cleanup.thin(fiber)
    (low, high), = fiber["gaps"]
    (new_low, new_high), = thin["gaps"]
    assert np.array_equal(thin["points"][[new_low, new_high]], fiber["points"][[low, high]])


@pytest.fixture
def scroll_edge():
    """CT of 64³ voxels, XYZ order reversed: zero outside for x < 20, papyrus
    elsewhere, with a single zero voxel and an enclosed zero pocket inside."""
    ct = np.full((64, 64, 64), 100, dtype=np.uint8)
    ct[:, :, :20] = 0
    ct[32, 32, 50] = 0
    ct[40:48, 40:48, 36:44] = 0
    return ct


def fiber_at(x, y=None):
    return dict(family=1, points=line([x, 0, 32], [x, 63, 32]) if y is None else line([x, y, 0], [x, y, 63]), gaps=[])


def test_only_black_connected_to_the_outside_removes_fibers(scroll_edge):
    black = cleanup.ExteriorBlack([0, 0, 0], [64, 64, 64])
    black.add(scroll_edge, [0, 0, 0])
    fibers = [fiber_at(30), fiber_at(10), fiber_at(40), fiber_at(50, 32), fiber_at(40, 44)]
    # 12 voxels from the outside, inside it, 20 voxels away, through the single zero voxel, through the pocket.
    assert black.near(fibers, 16).tolist() == [True, True, False, False, False]
    assert black.near(fibers, 8).tolist() == [False, True, False, False, False]


def test_black_regions_can_be_recorded_in_overlapping_pieces(scroll_edge):
    whole = cleanup.ExteriorBlack([0, 0, 0], [64, 64, 64])
    whole.add(scroll_edge, [0, 0, 0])
    pieces = cleanup.ExteriorBlack([0, 0, 0], [64, 64, 64])
    for low, high in (([0, 0, 0], [37, 64, 64]), ([27, 0, 0], [64, 30, 64]), ([27, 21, 0], [64, 64, 64])):
        pieces.add(scroll_edge[low[2]:high[2], low[1]:high[1], low[0]:high[0]], low)
    assert np.array_equal(whole.any_zero, pieces.any_zero) and np.array_equal(whole.all_zero, pieces.all_zero)
