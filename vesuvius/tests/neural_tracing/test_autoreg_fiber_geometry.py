from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_fiber import fiber_geometry as geom
from vesuvius.neural_tracing.autoreg_fiber.dataset import _apply_cubic_spatial_augmentation


@dataclass(frozen=True)
class _Node:
    id: int
    position: tuple[float, float, float]


class _Tree:
    def __init__(self, *, tree_id: int = 1, nodes=None, edges=None) -> None:
        self.id = tree_id
        self.name = f"tree-{tree_id}"
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])


def test_order_tree_path_follows_single_path_between_endpoints() -> None:
    nodes = [
        _Node(20, (2.0, 0.0, 0.0)),
        _Node(10, (1.0, 0.0, 0.0)),
        _Node(30, (3.0, 0.0, 0.0)),
    ]
    tree = _Tree(nodes=nodes, edges=[(nodes[0], nodes[1]), (nodes[0], nodes[2])])

    ordered = geom.order_tree_path_xyz(tree)

    # Endpoint ids are 10 and 30; deterministic traversal starts at 10.
    assert ordered.node_keys == ["id:10", "id:20", "id:30"]
    np.testing.assert_allclose(
        ordered.points_xyz,
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )


def test_order_tree_path_rejects_branchy_tree() -> None:
    nodes = [
        _Node(1, (0.0, 0.0, 0.0)),
        _Node(2, (1.0, 0.0, 0.0)),
        _Node(3, (0.0, 1.0, 0.0)),
        _Node(4, (0.0, 0.0, 1.0)),
    ]
    tree = _Tree(nodes=nodes, edges=[(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[0], nodes[3])])
    with pytest.raises(geom.FiberGeometryError, match="single path"):
        geom.order_tree_path_xyz(tree)


def test_coordinate_conversion_and_inverse_transform() -> None:
    old_xyz = np.asarray([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    new_to_old = np.eye(4)
    new_to_old[:3, 3] = [1.0, 2.0, 3.0]

    got_new_zyx = geom.old_xyz_to_new_zyx(old_xyz, new_to_old_matrix_xyz=new_to_old)

    expected_new_xyz = old_xyz - np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(got_new_zyx, expected_new_xyz[:, ::-1])
    np.testing.assert_allclose(geom.zyx_to_xyz(got_new_zyx), expected_new_xyz)


def test_coordinate_conversion_inverts_full_affine_not_axes_or_translation_only() -> None:
    new_xyz = np.asarray(
        [
            [3.0, 4.0, 5.0],
            [7.0, -2.0, 11.0],
        ],
        dtype=np.float64,
    )
    new_to_old = np.eye(4, dtype=np.float64)
    new_to_old[:3, :3] = np.asarray(
        [
            [1.5, 0.2, 0.0],
            [-0.1, 2.0, 0.3],
            [0.05, 0.0, -1.25],
        ],
        dtype=np.float64,
    )
    new_to_old[:3, 3] = [10.0, -5.0, 2.5]
    old_xyz = geom.apply_affine_xyz(new_to_old, new_xyz)

    got_new_zyx = geom.old_xyz_to_new_zyx(old_xyz, new_to_old_matrix_xyz=new_to_old)

    np.testing.assert_allclose(geom.zyx_to_xyz(got_new_zyx), new_xyz, atol=1e-10)


def test_densify_polyline_preserves_original_endpoints() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.0, 0.0]])
    dense = geom.densify_polyline(points, max_step=2.0)

    np.testing.assert_allclose(dense[0], points[0])
    np.testing.assert_allclose(dense[-1], points[-1])
    assert dense.shape[0] == 5
    assert np.max(np.linalg.norm(np.diff(dense, axis=0), axis=1)) <= 2.0


def test_resample_polyline_uniform_straight_line() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    out = geom.resample_polyline_uniform(points, target_spacing=20.0)
    assert out.shape == (6, 3)
    np.testing.assert_allclose(out[:, 0], [0.0, 20.0, 40.0, 60.0, 80.0, 100.0], atol=1e-9)
    distances = np.linalg.norm(np.diff(out, axis=0), axis=1)
    np.testing.assert_allclose(distances, np.full(5, 20.0), atol=1e-9)


def test_resample_polyline_uniform_l_shape_hits_corner() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0]])
    out = geom.resample_polyline_uniform(points, target_spacing=10.0)
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
        ]
    )
    np.testing.assert_allclose(out, expected, atol=1e-9)
    distances = np.linalg.norm(np.diff(out, axis=0), axis=1)
    np.testing.assert_allclose(distances, np.full(4, 10.0), atol=1e-9)


def test_resample_polyline_uniform_helix_arc_uniform_even_when_3d_varies() -> None:
    t = np.linspace(0.0, 4.0 * np.pi, 200)
    points = np.stack([np.cos(t) * 10.0, np.sin(t) * 10.0, t * 5.0], axis=1)
    out = geom.resample_polyline_uniform(points, target_spacing=20.0)

    seg_lengths_in = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_in = np.concatenate([[0.0], np.cumsum(seg_lengths_in)])

    def arc_position(point: np.ndarray) -> float:
        diffs = points - point[None, :]
        idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        return float(arc_in[idx])

    arc_at_resampled = np.asarray([arc_position(p) for p in out])
    arc_diffs = np.diff(arc_at_resampled)
    assert arc_diffs.std() < 0.5
    distances_3d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert distances_3d.std() > 0


def test_cubic_augmentation_preserves_point_volume_correspondence() -> None:
    n = 8
    vol = (
        np.arange(n).reshape(n, 1, 1) * 100
        + np.arange(n).reshape(1, n, 1) * 10
        + np.arange(n).reshape(1, 1, n)
    ).astype(np.float32)
    pts = np.array(
        [[1.0, 2.0, 3.0], [5.0, 1.0, 6.0], [0.0, 7.0, 4.0]],
        dtype=np.float32,
    )

    def _lookup(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.array(
            [volume[int(round(z)), int(round(y)), int(round(x))] for (z, y, x) in points]
        )

    orig_vals = _lookup(vol, pts)
    seen = set()
    for trial in range(200):
        np.random.seed(trial)
        aug_vol, aug_pts = _apply_cubic_spatial_augmentation(
            volume_zyx=vol.copy(),
            points_local_zyx=pts.copy(),
            crop_size_zyx=(n, n, n),
        )
        np.testing.assert_allclose(_lookup(aug_vol, aug_pts), orig_vals)
        seen.add(tuple(aug_pts.flatten().tolist()))
    assert len(seen) > 30, f"expected >30 unique cubic-group elements, saw {len(seen)}"


def test_cubic_augmentation_rejects_anisotropic_crop() -> None:
    vol = np.zeros((4, 4, 8), dtype=np.float32)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="isotropic"):
        _apply_cubic_spatial_augmentation(
            volume_zyx=vol,
            points_local_zyx=pts,
            crop_size_zyx=(4, 4, 8),
        )


def test_resample_polyline_uniform_edge_cases() -> None:
    single = np.asarray([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(geom.resample_polyline_uniform(single, target_spacing=5.0), single)

    short = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    short_out = geom.resample_polyline_uniform(short, target_spacing=20.0)
    assert short_out.shape == (2, 3)
    np.testing.assert_allclose(short_out[0], short[0])
    np.testing.assert_allclose(short_out[-1], short[-1])

    coincident = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    coincident_out = geom.resample_polyline_uniform(coincident, target_spacing=5.0)
    assert coincident_out.shape == (1, 3)
    np.testing.assert_allclose(coincident_out[0], coincident[0])

    np.testing.assert_allclose(
        geom.resample_polyline_uniform(short, target_spacing=0.0), short
    )


def test_fiber_cache_key_and_npz_roundtrip(tmp_path: Path) -> None:
    nodes = [
        _Node(1, (0.0, 0.0, 0.0)),
        _Node(2, (2.0, 0.0, 0.0)),
    ]
    matrix = np.eye(4)
    fiber = geom.tree_to_fiber_path(
        _Tree(tree_id=42, nodes=nodes, edges=[(nodes[0], nodes[1])]),
        annotation_id="ann",
        marker="fibers_s3",
        target_volume="PHerc0332",
        new_to_old_matrix_xyz=matrix,
        densify_step=1.0,
    )

    key_a = geom.fiber_cache_key(
        annotation_id=fiber.annotation_id,
        tree_id=fiber.tree_id,
        transform_checksum=fiber.transform_checksum,
        densify_step=fiber.densify_step,
    )
    key_b = geom.fiber_cache_key(
        annotation_id=fiber.annotation_id,
        tree_id=fiber.tree_id,
        transform_checksum=fiber.transform_checksum,
        densify_step=2.0,
    )
    assert key_a != key_b

    path = geom.write_fiber_cache(fiber, tmp_path)
    points_zyx, metadata = geom.load_fiber_cache(path)
    assert path.suffix == ".npz"
    assert metadata["annotation_id"] == "ann"
    assert metadata["tree_id"] == "42"
    assert metadata["coordinate_convention"] == "wk_xyz_to_new_zyx"
    np.testing.assert_allclose(points_zyx, fiber.points_zyx.astype(np.float32))
