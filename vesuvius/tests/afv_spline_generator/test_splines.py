import numpy as np
import pytest
from scipy.spatial import cKDTree

from vesuvius.afv_spline_generator.splines import MAX_DEVIATION, chain_order, extract_splines, family_probability, threshold_to_u8


SHAPE = (40, 64, 112)


def tube(probabilities, channel, centerline, radius=2.2, value=230):
    """Mark voxels near a ZYX centerline in one channel."""
    grid = np.indices(SHAPE).reshape(3, -1).T
    distance, _ = cKDTree(centerline).query(grid, distance_upper_bound=radius + 1)
    inside = (distance <= radius).reshape(SHAPE)
    probabilities[inside, channel] = value


def wavy_centerline(z, y, x0, x1):
    x = np.linspace(x0, x1, 1000)
    return np.stack([np.full_like(x, z), y + 6 * np.sin(x / 15), x], axis=1)


def arc_length(line):
    return np.linalg.norm(np.diff(line, axis=0), axis=1).sum()


def test_threshold_is_a_percentage():
    assert threshold_to_u8(60) == 153
    assert threshold_to_u8(100) == 255
    for bad in (0, -1, 101):
        with pytest.raises(ValueError):
            threshold_to_u8(bad)


def test_family_probability_adds_half_the_intersection():
    probabilities = np.zeros((1, 1, 2, 3), dtype=np.uint8)
    probabilities[0, 0, 0] = (100, 0, 50)
    probabilities[0, 0, 1] = (200, 0, 200)
    assert family_probability(probabilities, 0).ravel().tolist() == [125, 255]
    assert family_probability(probabilities, 1).ravel().tolist() == [25, 100]


def test_chain_order():
    chain = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 1], [1, 3, 2]])
    ordered = chain_order(chain[[2, 0, 3, 1]])
    assert ordered is not None
    assert np.array_equal(ordered, chain) or np.array_equal(ordered, chain[::-1])
    ring = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
    assert chain_order(ring) is None


def test_a_tube_gives_one_smooth_polyline_on_its_centerline():
    probabilities = np.zeros(SHAPE + (3,), dtype=np.uint8)
    centerline = wavy_centerline(20, 30, 10, 100)
    tube(probabilities, 0, centerline)
    traces = extract_splines(probabilities, threshold_to_u8(60))
    assert [family for family, _ in traces] == ["V"]
    line = traces[0][1]
    # The skeleton may reach into the rounded ends of the tube.
    assert 0.85 * arc_length(centerline) < arc_length(line) <= arc_length(centerline) + 2 * 2.2
    assert cKDTree(centerline).query(line)[0].max() < MAX_DEVIATION
    assert np.linalg.norm(np.diff(line, axis=0), axis=1).max() <= 0.6


def test_separate_tubes_are_never_bridged_and_longest_comes_first():
    probabilities = np.zeros(SHAPE + (3,), dtype=np.uint8)
    tube(probabilities, 0, wavy_centerline(10, 20, 5, 50))
    tube(probabilities, 1, wavy_centerline(30, 40, 20, 105))
    traces = extract_splines(probabilities, threshold_to_u8(60))
    assert [family for family, _ in traces] == ["H", "V"]
    assert traces[0][1][:, 0].round().tolist() == [30] * len(traces[0][1])
    assert traces[1][1][:, 0].round().tolist() == [10] * len(traces[1][1])


def test_below_threshold_gives_nothing():
    probabilities = np.zeros(SHAPE + (3,), dtype=np.uint8)
    tube(probabilities, 0, wavy_centerline(20, 30, 10, 100), value=150)
    assert extract_splines(probabilities, threshold_to_u8(60)) == []
