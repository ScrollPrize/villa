"""Tests for process_slice_points_label's triangle-plane intersection logic.

Covers a real bug: a plane can intersect a non-degenerate triangle's
boundary in at most 2 points, but the code could compute 3 when one
vertex sits very close to (without being exactly on) the slicing plane --
a realistic case for floating-point mesh coordinates. The two edges
sharing that near-degenerate vertex each compute their own independent,
slightly different point near it; both survive the (tight) dedup check,
producing a spurious near-duplicate pair alongside the one genuine
crossing. The previous code connected every pair of found points,
drawing a spurious extra line segment between the two near-duplicates.
"""
import numpy as np
import pytest

from vesuvius.image_proc.run.voxelize_objs import process_slice_points_label


def _triangle(v0, v1, v2):
    return (
        np.array([v0, v1, v2], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
        np.array([7], dtype=np.uint16),
    )


def test_ordinary_triangle_draws_a_single_line():
    """No regression: a normal, non-degenerate triangle still works."""
    vertices, triangles, mesh_labels = _triangle(
        [10.0, 10.0, 5.0], [10.0, 20.0, 0.0], [20.0, 10.0, 10.0]
    )
    img = process_slice_points_label(vertices, triangles, mesh_labels, 5.0, w=40, h=40)
    assert (img > 0).any()
    assert set(np.unique(img[img > 0])) == {7}


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4, 1e-3])
def test_near_degenerate_vertex_matches_exact_on_plane_case(eps):
    """A vertex offset by `eps` from the slicing plane -- realistic
    floating-point noise -- must not draw more pixels than the same
    triangle with that vertex placed exactly on the plane.

    Reproduces the bug directly: the previous code could draw a spurious
    extra line segment for a near-degenerate vertex that the exact-on-
    -plane case never produces.
    """
    zslice = 5.0
    vertices, triangles, mesh_labels = _triangle(
        [10.0, 10.0, zslice + eps], [10.0, 20.0, 0.0], [20.0, 10.0, 10.0]
    )
    img = process_slice_points_label(vertices, triangles, mesh_labels, zslice, w=400, h=400)

    vertices_exact = vertices.copy()
    vertices_exact[0, 2] = zslice
    img_exact = process_slice_points_label(
        vertices_exact, triangles, mesh_labels, zslice, w=400, h=400
    )

    assert (img > 0).sum() == (img_exact > 0).sum(), (
        f"near-degenerate vertex (eps={eps}) drew a different pixel count "
        f"than the same triangle with that vertex exactly on the plane -- "
        f"the near-degenerate case must not draw a spurious extra segment"
    )


def test_three_intersection_points_draws_exactly_one_line_not_three():
    """Directly reproduces the reported mechanism: construct a triangle
    where the near-degenerate-vertex effect is large enough (eps=1e-3,
    scaled-up triangle) to draw a materially different number of pixels
    under the old all-pairs behaviour, and confirm the fix draws only the
    genuine single line instead.
    """
    zslice = 5.0
    scale = 10.0
    vertices, triangles, mesh_labels = _triangle(
        [10.0 * scale, 10.0 * scale, zslice + 1e-3],
        [10.0 * scale, 20.0 * scale, 0.0],
        [20.0 * scale, 10.0 * scale, 10.0],
    )
    img = process_slice_points_label(vertices, triangles, mesh_labels, zslice, w=400, h=400)

    vertices_exact = vertices.copy()
    vertices_exact[0, 2] = zslice
    img_exact = process_slice_points_label(
        vertices_exact, triangles, mesh_labels, zslice, w=400, h=400
    )

    assert (img > 0).sum() == (img_exact > 0).sum()


def test_random_near_degenerate_triangles_never_exceed_exact_case_pixel_count():
    """Broader statistical check: across many random near-degenerate
    triangles, the patched function's pixel count must always match the
    corresponding exact-on-plane case -- never draw extra, spurious
    pixels from a phantom third intersection point.

    eps is kept to genuine floating-point-noise scale (1e-6 to 1e-4). A
    larger eps represents a real geometric displacement of the vertex,
    which legitimately produces a slightly different (not necessarily
    pixel-identical) intersection line -- that's correct behaviour, not
    a bug, and demanding exact equality at that scale would be testing
    the wrong thing.
    """
    rng = np.random.default_rng(7)
    zslice = 5.0
    checked = 0
    for _ in range(60):
        eps = rng.uniform(1e-6, 1e-4) * rng.choice([1, -1])
        v0 = np.array([rng.uniform(0, 100), rng.uniform(0, 100), zslice + eps])
        v1 = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(-10, zslice - 1)])
        v2 = np.array([rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(zslice + 1, 20)])
        vertices, triangles, mesh_labels = _triangle(v0, v1, v2)

        img = process_slice_points_label(vertices, triangles, mesh_labels, zslice, w=200, h=200)
        vertices_exact = vertices.copy()
        vertices_exact[0, 2] = zslice
        img_exact = process_slice_points_label(
            vertices_exact, triangles, mesh_labels, zslice, w=200, h=200
        )
        assert (img > 0).sum() == (img_exact > 0).sum(), f"mismatch at eps={eps}"
        checked += 1

    assert checked == 60
