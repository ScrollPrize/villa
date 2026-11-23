"""Tests for vesuvius.data.affine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_matrix_checksum_stable() -> None:
    m = np.eye(4)
    digest_a = affine.matrix_checksum(m)
    digest_b = affine.matrix_checksum(m.copy())
    assert digest_a == digest_b
    m[0, 0] = 2.0
    assert affine.matrix_checksum(m) != digest_a
