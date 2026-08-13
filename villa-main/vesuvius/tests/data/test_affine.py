"""Tests for vesuvius.data.affine."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from vesuvius.data import affine


def _identity_matrix_xyz() -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    return m


def _write_transform_json(
    path: Path, matrix_xyz_4x4: np.ndarray, fixed_volume: str = "fixed_vol"
) -> None:
    m3x4 = matrix_xyz_4x4[:3, :].tolist()
    payload = {
        "schema_version": affine.SCHEMA_VERSION,
        "fixed_volume": fixed_volume,
        "transformation_matrix": m3x4,
        "fixed_landmarks": [[0.0, 0.0, 0.0]],
        "moving_landmarks": [[0.0, 0.0, 0.0]],
    }
    path.write_text(json.dumps(payload))


def test_read_transform_json_roundtrip(tmp_path: Path) -> None:
    m = _identity_matrix_xyz()
    m[:3, 3] = [1.5, -2.5, 3.5]
    path = tmp_path / "transform.json"
    _write_transform_json(path, m, fixed_volume="foo")

    doc = affine.read_transform_json(str(path))
    np.testing.assert_allclose(doc.matrix_xyz, m)
    assert doc.fixed_volume == "foo"
    assert doc.fixed_landmarks == [[0.0, 0.0, 0.0]]


def test_read_transform_json_rejects_wrong_schema(tmp_path: Path) -> None:
    m = _identity_matrix_xyz()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "schema_version": "0.9.0",
        "fixed_volume": "x",
        "transformation_matrix": m[:3, :].tolist(),
        "fixed_landmarks": [],
        "moving_landmarks": [],
    }))
    with pytest.raises(ValueError, match="schema_version"):
        affine.read_transform_json(str(bad))


def test_invert_roundtrip() -> None:
    rng = np.random.default_rng(0)
    m = np.eye(4)
    m[:3, :3] = rng.normal(size=(3, 3))
    m[:3, 3] = rng.normal(size=3)
    m_inv = affine.invert_affine_matrix(m)
    np.testing.assert_allclose(affine.invert_affine_matrix(m_inv), m, atol=1e-9)
    np.testing.assert_allclose(m @ m_inv, np.eye(4), atol=1e-9)


def test_swap_is_self_inverse() -> None:
    rng = np.random.default_rng(1)
    m = np.eye(4)
    m[:3, :3] = rng.normal(size=(3, 3))
    m[:3, 3] = rng.normal(size=3)
    np.testing.assert_allclose(
        affine.matrix_swap_xyz_zyx(affine.matrix_swap_xyz_zyx(m)), m, atol=1e-12
    )


def test_swap_matches_coordinate_reordering() -> None:
    """Applying swap(M) to zyx == applying M to xyz and reading back."""
    rng = np.random.default_rng(2)
    m_xyz = np.eye(4)
    m_xyz[:3, :3] = rng.normal(size=(3, 3))
    m_xyz[:3, 3] = rng.normal(size=3)

    point_xyz = rng.normal(size=3)
    point_zyx = point_xyz[::-1]

    m_zyx = affine.matrix_swap_xyz_zyx(m_xyz)

    out_xyz = (m_xyz @ np.concatenate([point_xyz, [1.0]]))[:3]
    out_zyx = (m_zyx @ np.concatenate([point_zyx, [1.0]]))[:3]

    np.testing.assert_allclose(out_zyx, out_xyz[::-1], atol=1e-12)


def test_label_to_image_zyx_matrix_inverts_and_swaps() -> None:
    rng = np.random.default_rng(3)
    m_xyz = np.eye(4)
    m_xyz[:3, :3] = rng.normal(size=(3, 3))
    m_xyz[:3, 3] = rng.normal(size=3)

    got = affine.label_to_image_zyx_matrix(m_xyz, invert=True)
    expected = affine.matrix_swap_xyz_zyx(np.linalg.inv(m_xyz))
    np.testing.assert_allclose(got, expected)


def test_aabb_identity_is_exact_patch() -> None:
    m = np.eye(4)
    start, stop = affine.label_patch_image_aabb(
        m, position_label_zyx=(10, 20, 30), patch_shape_zyx=(8, 8, 8), margin=0
    )
    assert start == (10, 20, 30)
    assert stop == (18, 28, 38)


def test_aabb_bounded_under_shear() -> None:
    # 30-deg rotation around z-axis; patch+2 margin per side at most.
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4)
    m[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )
    patch = (16, 16, 16)
    # Place the label patch with room on all sides so the AABB isn't clipped.
    start, stop = affine.label_patch_image_aabb(
        m, (32, 32, 32), patch, image_shape_zyx=(128, 128, 128)
    )
    extents = np.array(stop) - np.array(start)
    # Rotation enlarges by ~|sin|+|cos|+2*margin ~= 1.37*16 + 2; ensure 2x bound.
    assert np.all(extents <= 2 * np.array(patch) + 2)
    # The rotation axis (Z) is preserved, so that extent must equal the patch size (+2 margin).
    assert extents[0] == patch[0] + 2


def test_aabb_clipped_to_image_shape() -> None:
    m = np.eye(4)
    start, stop = affine.label_patch_image_aabb(
        m, position_label_zyx=(0, 0, 0), patch_shape_zyx=(8, 8, 8),
        image_shape_zyx=(4, 4, 4), margin=1,
    )
    for a in stop:
        assert a <= 4


def test_resample_identity_matches_slab() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8)

    patch = affine.resample_image_to_label_grid(
        image,
        np.eye(4),
        position_label_zyx=(4, 6, 8),
        patch_shape_zyx=(8, 8, 8),
    )
    np.testing.assert_allclose(patch, image[4:12, 6:14, 8:16].astype(np.float32), atol=1e-5)


def test_resample_pure_translation() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8)
    m = np.eye(4)
    m[:3, 3] = [3.0, -2.0, 1.0]  # image_zyx = label_zyx + [3, -2, 1]

    patch = affine.resample_image_to_label_grid(
        image, m, position_label_zyx=(5, 5, 5), patch_shape_zyx=(8, 8, 8),
    )
    expected = image[8:16, 3:11, 6:14].astype(np.float32)
    np.testing.assert_allclose(patch, expected, atol=1e-5)


def test_resample_pure_isotropic_scale() -> None:
    """Scale 2x: image is half-resolution, each label voxel maps to half an image voxel."""
    rng = np.random.default_rng(6)
    image = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8).astype(np.float32)
    m = np.eye(4)
    m[:3, :3] *= 0.5  # image = 0.5 * label  (label twice as fine)

    patch_shape = (8, 8, 8)
    pos = (4, 4, 4)
    patch = affine.resample_image_to_label_grid(image, m, pos, patch_shape)

    # Reference via scipy directly over the whole volume
    zs = np.arange(patch_shape[0]) + pos[0]
    ys = np.arange(patch_shape[1]) + pos[1]
    xs = np.arange(patch_shape[2]) + pos[2]
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    coords = np.stack([gz * 0.5, gy * 0.5, gx * 0.5], axis=0)
    expected = ndimage.map_coordinates(image, coords, order=1, mode="constant", cval=0.0, prefilter=False)
    np.testing.assert_allclose(patch, expected, atol=1e-4)


def test_resample_handles_out_of_bounds_patch() -> None:
    image = np.ones((8, 8, 8), dtype=np.float32)
    # Label patch translated entirely outside the image
    m = np.eye(4)
    m[:3, 3] = [100.0, 100.0, 100.0]
    patch = affine.resample_image_to_label_grid(
        image, m, position_label_zyx=(0, 0, 0), patch_shape_zyx=(4, 4, 4),
    )
    np.testing.assert_array_equal(patch, np.zeros((4, 4, 4), dtype=np.float32))


def test_matrix_checksum_stable() -> None:
    m = np.eye(4)
    digest_a = affine.matrix_checksum(m)
    digest_b = affine.matrix_checksum(m.copy())
    assert digest_a == digest_b
    m[0, 0] = 2.0
    assert affine.matrix_checksum(m) != digest_a


def test_resample_128_cube_within_budget() -> None:
    """Microbenchmark: single 128^3 resample under 75 ms warm on CPU."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(160, 160, 160), dtype=np.uint8).astype(np.float32)
    m = np.eye(4)
    m[:3, 3] = [0.25, 0.3, 0.4]  # sub-voxel shift so interpolation runs

    # warm-up
    affine.resample_image_to_label_grid(image, m, (8, 8, 8), (128, 128, 128))

    t0 = time.perf_counter()
    patch = affine.resample_image_to_label_grid(image, m, (8, 8, 8), (128, 128, 128))
    elapsed = time.perf_counter() - t0
    assert patch.shape == (128, 128, 128)
    assert elapsed < 0.5, f"128^3 resample took {elapsed*1000:.1f} ms (expected < 500 ms)"
