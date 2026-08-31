from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vc_spiral import surface_index


def make_surface(surface_id, rows=3, cols=3, z_fn=lambda row, col: 0.0):
    zyx = np.empty((rows, cols, 3), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            zyx[row, col] = (z_fn(row, col), row, col)
    return zyx, surface_index.QuadSurface(surface_id, zyx, 1.0, 1.0)


def query(index, xyzs, tolerance):
    return index.locate_all_xyz_batch(
        np.ascontiguousarray(xyzs, dtype=np.float32), tolerance
    )


def test_flat_and_tilted_geometry_return_grid_ij():
    _, flat = make_surface("flat")
    _, tilted = make_surface("tilted", z_fn=lambda _row, col: 10.0 + col)
    index = surface_index.SurfacePatchIndex()
    index.rebuild([flat, tilted])

    offsets, surfaces, distances, ijs = query(
        index,
        [[0.25, 1.5, 0.2], [0.5, 0.25, 10.5]],
        0.25,
    )

    np.testing.assert_array_equal(offsets, [0, 1, 2])
    np.testing.assert_array_equal(surfaces, [0, 1])
    np.testing.assert_allclose(distances, [0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(ijs, [[1.5, 0.25], [0.25, 0.5]], atol=1e-6)


def test_invalid_vertices_remove_their_adjacent_quads():
    zyx, surface = make_surface("hole")
    zyx[1, 1] = -1.0
    surface = surface_index.QuadSurface("hole", zyx, 1.0, 1.0)
    index = surface_index.SurfacePatchIndex()
    index.rebuild([surface])

    offsets, surfaces, distances, ijs = query(index, [[1.0, 1.0, 0.0]], 0.1)
    np.testing.assert_array_equal(offsets, [0, 0])
    assert surfaces.shape == (0,)
    assert distances.shape == (0,)
    assert ijs.shape == (0, 2)


def test_tolerance_boundary_is_inclusive():
    _, surface = make_surface("flat")
    index = surface_index.SurfacePatchIndex()
    index.rebuild([surface])
    point = [[0.5, 0.5, 0.5]]

    at_boundary = query(index, point, 0.5)
    np.testing.assert_array_equal(at_boundary[0], [0, 1])
    np.testing.assert_allclose(at_boundary[2], [0.5])

    just_inside = float(np.nextafter(np.float32(0.5), np.float32(0.0)))
    outside = query(index, point, just_inside)
    np.testing.assert_array_equal(outside[0], [0, 0])


def test_zero_tolerance_returns_exact_hits():
    _, surface = make_surface("flat")
    index = surface_index.SurfacePatchIndex()
    index.rebuild([surface])

    exact = query(index, [[0.5, 0.5, 0.0], [0.5, 0.5, 0.1]], 0.0)

    np.testing.assert_array_equal(exact[0], [0, 1, 1])
    np.testing.assert_array_equal(exact[1], [0])
    np.testing.assert_array_equal(exact[2], [0.0])
    np.testing.assert_allclose(exact[3], [[0.5, 0.5]], atol=1e-6)


def test_multiple_surfaces_and_subset_filtering():
    _, low = make_surface("low", z_fn=lambda _row, _col: 0.0)
    _, high = make_surface("high", z_fn=lambda _row, _col: 0.2)
    index = surface_index.SurfacePatchIndex()
    index.rebuild([low, high])

    assert index.surface_ids() == ["low", "high"]
    all_hits = query(index, [[0.5, 0.5, 0.1]], 0.11)
    np.testing.assert_array_equal(all_hits[0], [0, 2])
    np.testing.assert_array_equal(all_hits[1], [0, 1])
    np.testing.assert_allclose(all_hits[2], [0.1, 0.1], atol=1e-6)

    subset_hits = index.locate_all_xyz_batch_in(
        np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        np.asarray([1, 1, -1, 999], dtype=np.int32),
        0.11,
    )
    np.testing.assert_array_equal(subset_hits[0], [0, 1])
    np.testing.assert_array_equal(subset_hits[1], [1])

    empty_hits = index.locate_all_xyz_batch_in(
        np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        np.asarray([], dtype=np.int32),
        0.11,
    )
    np.testing.assert_array_equal(empty_hits[0], [0, 0])


def test_sampling_stride_controls_the_evaluated_quads():
    zyx, _ = make_surface("stride", rows=5, cols=5)
    zyx[1, 1] = -1.0
    surface = surface_index.QuadSurface("stride", zyx, 1.0, 1.0)
    point = [[0.5, 0.5, 0.0]]

    fine = surface_index.SurfacePatchIndex()
    fine.rebuild([surface], sampling_stride=1)
    np.testing.assert_array_equal(query(fine, point, 0.1)[0], [0, 0])

    coarse = surface_index.SurfacePatchIndex()
    coarse.rebuild([surface], sampling_stride=2)
    coarse_hit = query(coarse, point, 0.1)
    np.testing.assert_array_equal(coarse_hit[0], [0, 1])
    np.testing.assert_allclose(coarse_hit[3], [[0.5, 0.5]], atol=1e-6)

    with pytest.raises(RuntimeError, match="sampling_stride must be >= 1"):
        coarse.rebuild([surface], sampling_stride=0)


def test_output_dtypes_shapes_and_one_hit_per_surface():
    _, first = make_surface("first")
    _, second = make_surface("second")
    index = surface_index.SurfacePatchIndex()
    index.rebuild([first, second])

    offsets, surfaces, distances, ijs = query(
        index, [[0.25, 0.25, 0.0], [50.0, 50.0, 50.0]], 0.1
    )
    assert offsets.dtype == np.int64
    assert surfaces.dtype == np.int32
    assert distances.dtype == np.float32
    assert ijs.dtype == np.float32
    assert offsets.shape == (3,)
    assert surfaces.shape == (2,)
    assert distances.shape == (2,)
    assert ijs.shape == (2, 2)
    np.testing.assert_array_equal(offsets, [0, 2, 2])
    np.testing.assert_array_equal(surfaces, [0, 1])


def test_equal_distance_ties_are_deterministic():
    _, first = make_surface("first", rows=12, cols=12)
    _, second = make_surface("second", rows=12, cols=12)
    index = surface_index.SurfacePatchIndex()
    index.rebuild([first, second])
    points = np.asarray([[8.0, 8.0, 0.05], [4.0, 4.0, 0.05]], dtype=np.float32)

    expected = query(index, points, 0.1)
    np.testing.assert_array_equal(expected[1], [0, 1, 0, 1])
    for _ in range(20):
        actual = query(index, points, 0.1)
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_array_equal(actual_array, expected_array)


def test_concurrent_queries_are_read_only_and_repeatable():
    _, first = make_surface("first", rows=16, cols=16)
    _, second = make_surface("second", rows=16, cols=16, z_fn=lambda _row, _col: 0.1)
    index = surface_index.SurfacePatchIndex()
    index.rebuild([first, second])
    points = np.asarray(
        [[col + 0.25, row + 0.75, 0.05] for row in range(12) for col in range(12)],
        dtype=np.float32,
    )
    expected = query(index, points, 0.1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: query(index, points, 0.1), range(32)))

    for result in results:
        for expected_array, actual_array in zip(expected, result):
            np.testing.assert_array_equal(actual_array, expected_array)
