"""Tests for the WK I/O helpers and prompt round-trip checks.

The streaming tracer's WK upload path is exercised with the live network in
the integration smoke test; here we verify the pure helpers and the swap-only
invariant of the prompt loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.streaming.wk_io import (
    DEFAULT_WK_SERVER_URL,
    PromptPayload,
    build_annotation,
    build_skeleton,
    load_prompt_npz,
    save_annotation,
)


def _write_cache_with_swap_only(tmp_path: Path, points_zyx: np.ndarray) -> Path:
    fiber = FiberPath(
        annotation_id="ann-roundtrip",
        tree_id="tree-roundtrip",
        target_volume="PHercParis4",
        marker="fibers_s1a",
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32, copy=False),
        points_zyx=points_zyx.astype(np.float32, copy=False),
        transform_checksum="identity",
        densify_step=1.0,
    )
    return write_fiber_cache(fiber, tmp_path)


def _write_cache_with_corrupted_source(tmp_path: Path, points_zyx: np.ndarray) -> Path:
    bad_xyz = points_zyx[:, ::-1].astype(np.float32, copy=True)
    bad_xyz[0, 0] += 5.0  # introduce a 5-voxel mismatch
    fiber = FiberPath(
        annotation_id="ann-corrupt",
        tree_id="tree-corrupt",
        target_volume="PHercParis4",
        marker="fibers_s1a",
        source_points_xyz=bad_xyz,
        points_zyx=points_zyx.astype(np.float32, copy=False),
        transform_checksum="non-identity",
        densify_step=1.0,
    )
    return write_fiber_cache(fiber, tmp_path)


def test_load_prompt_npz_returns_swap_only_data(tmp_path: Path) -> None:
    points = np.array(
        [[10, 20, 30], [11, 20, 30], [12, 20, 30]],
        dtype=np.float32,
    )
    npz_path = _write_cache_with_swap_only(tmp_path, points)
    payload = load_prompt_npz(npz_path)
    assert isinstance(payload, PromptPayload)
    np.testing.assert_array_equal(payload.points_zyx, points)
    assert payload.source_points_xyz is not None
    np.testing.assert_array_equal(payload.source_points_xyz, points[:, ::-1])
    assert payload.metadata["annotation_id"] == "ann-roundtrip"


def test_load_prompt_npz_rejects_non_identity_affine(tmp_path: Path) -> None:
    points = np.array([[10, 20, 30], [11, 20, 30]], dtype=np.float32)
    npz_path = _write_cache_with_corrupted_source(tmp_path, points)
    with pytest.raises(ValueError, match="swap-only invariant"):
        load_prompt_npz(npz_path)


def test_load_prompt_npz_missing_points_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, something=np.array([1.0]))
    with pytest.raises(ValueError, match="points_zyx"):
        load_prompt_npz(path)


def test_build_skeleton_emits_one_tree_with_chained_edges() -> None:
    polyline_zyx = np.array(
        [[10, 11, 12], [11, 11, 12], [12, 11, 12], [13, 11, 12]],
        dtype=np.float32,
    )
    skeleton = build_skeleton(
        polyline_zyx,
        dataset_name="PHercParis4-69da9fa9010000c20022c400",
        voxel_size=(2400, 2400, 2400),
        tree_name="autoreg_fiber_smoke",
    )
    trees = list(skeleton.flattened_trees())
    assert len(trees) == 1
    tree = trees[0]
    nodes = list(tree.nodes)
    assert len(nodes) == polyline_zyx.shape[0]
    # Positions stored in XYZ voxel order.
    for node, expected_zyx in zip(nodes, polyline_zyx, strict=True):
        actual_xyz = (int(node.position[0]), int(node.position[1]), int(node.position[2]))
        expected_xyz = (int(round(expected_zyx[2])), int(round(expected_zyx[1])), int(round(expected_zyx[0])))
        assert actual_xyz == expected_xyz
    assert tree.number_of_edges() == polyline_zyx.shape[0] - 1


def test_build_annotation_and_save_writes_zip_and_nml(tmp_path: Path) -> None:
    polyline_zyx = np.array(
        [[5, 5, 5], [6, 5, 5], [7, 5, 5]],
        dtype=np.float32,
    )
    skeleton = build_skeleton(
        polyline_zyx,
        dataset_name="PHercParis4-69da9fa9010000c20022c400",
        voxel_size=(2400, 2400, 2400),
    )
    annotation = build_annotation(skeleton, name="autoreg_fiber_test")
    out_dir = tmp_path / "out"
    paths = save_annotation(annotation, out_dir, basename="trace")
    assert Path(paths["zip"]).exists()
    assert Path(paths["nml"]).exists()
    assert paths["zip"].endswith("trace.zip")
    assert paths["nml"].endswith("trace.nml")


def test_default_wk_server_url_is_aws_ash2txt() -> None:
    # Hardcoded default for the user's instance — guard against accidental edit.
    assert DEFAULT_WK_SERVER_URL == "https://wk.aws.ash2txt.org"
