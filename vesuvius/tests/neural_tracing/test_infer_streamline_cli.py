import json

import numpy as np
import pytest

from vesuvius.neural_tracing.inference import infer_streamline
from vesuvius.neural_tracing.inference.view_streamline import (
    _crop_bounds_from_bbox,
    points_to_crop_mask,
    surface_grid_to_mesh,
    vector_field_crop_to_vectors,
)


def test_sanitize_surface_validity_rejects_sentinels_and_nonfinite():
    zyx = np.zeros((2, 3, 3), dtype=np.float32)
    zyx[..., 0] = 1.0
    zyx[..., 1] = 2.0
    zyx[..., 2] = 3.0
    zyx[0, 1] = -1.0
    zyx[1, 0, 2] = np.nan
    reader_valid = np.ones((2, 3), dtype=bool)
    reader_valid[1, 2] = False

    valid = infer_streamline._sanitize_surface_validity(zyx, reader_valid)

    assert valid.tolist() == [
        [True, False, True],
        [False, True, False],
    ]


def test_resolve_auto_volume_path_matches_tifxyz_parent(tmp_path):
    tifxyz_path = tmp_path / "datasets" / "neural_tracing" / "050826" / "PhercParis4" / "w00_0805261021"
    tifxyz_path.mkdir(parents=True)
    (tifxyz_path / "meta.json").write_text(
        json.dumps({"format": "tifxyz", "scale": [0.05, 0.05], "uuid": "w00_0805261021"}),
        encoding="utf-8",
    )
    model_config = {
        "datasets": [
            {
                "volume_path": "s3://example/PHerc0139/volumes/a.zarr/",
                "volume_scale": 0,
                "segments_path": "/ephemeral/nt_ds/PHerc0139",
            },
            {
                "volume_path": "s3://vesuvius-challenge-open-data/PHercParis4/volumes/paris4.zarr/",
                "volume_scale": 0,
                "segments_path": "/ephemeral/nt_ds/PhercParis4",
            },
        ],
    }

    volume_path, volume_scale, matched = infer_streamline.resolve_volume_path_from_config(
        tifxyz_path,
        model_config,
        requested_volume_path="auto",
    )

    assert volume_path == "s3://vesuvius-challenge-open-data/PHercParis4/volumes/paris4.zarr/"
    assert volume_scale == 0
    assert matched["segments_path"].endswith("PhercParis4")


def test_derive_tifxyz_voxel_step_from_isotropic_scale(tmp_path):
    tifxyz_path = tmp_path / "segment"
    tifxyz_path.mkdir()
    (tifxyz_path / "meta.json").write_text(
        json.dumps({"format": "tifxyz", "scale": [0.05, 0.05]}),
        encoding="utf-8",
    )

    assert infer_streamline.derive_tifxyz_voxel_step(tifxyz_path) == 20.0


def test_cond_edge_bboxes_preserve_full_edge_for_merge():
    zyx = np.zeros((4, 3, 3), dtype=np.float32)
    for row in range(4):
        for col in range(3):
            zyx[row, col] = [row, col, row + col]
    zyx[1, :] = -1.0
    valid = infer_streamline._sanitize_surface_validity(zyx)

    bboxes, edge = infer_streamline.get_cond_edge_bboxes(
        zyx,
        "right",
        crop_size=(8, 8, 8),
        cond_valid=valid,
    )

    assert bboxes
    assert edge.shape == (4, 3)
    assert (edge[1] == -1.0).all()
    assert infer_streamline._valid_surface_mask(edge).sum() == 3


def test_surface_grid_to_mesh_builds_triangles_from_valid_quads():
    rows, cols = 3, 3
    zyx = np.zeros((rows, cols, 3), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            zyx[row, col] = [row, col, row + col]

    vertices, faces = surface_grid_to_mesh(zyx, np.ones((rows, cols), dtype=bool), stride=1)

    assert vertices.shape == (9, 3)
    assert faces.shape == (8, 3)
    assert faces.max() < vertices.shape[0]


def test_slice_view_crop_mask_uses_global_zyx_coordinates():
    bbox = (10, 13, 20, 23, 30, 33)
    crop_min, crop_max = _crop_bounds_from_bbox(bbox, padding=1, volume_shape=(100, 100, 100))
    points = np.asarray(
        [
            [10.1, 20.2, 30.3],
            [13.0, 23.0, 33.0],
            [99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )

    mask = points_to_crop_mask(points, crop_min, crop_max - crop_min)

    assert crop_min.tolist() == [9, 19, 29]
    assert crop_max.tolist() == [15, 25, 35]
    assert int(mask.sum()) == 2
    assert mask[1, 1, 1] == 1
    assert mask[4, 4, 4] == 1


def test_prediction_vector_helper_uses_global_origins_and_scale():
    field = np.zeros((3, 2, 2, 2), dtype=np.float32)
    field[2, 1, 1, 1] = 2.0

    vectors = vector_field_crop_to_vectors(
        field,
        origin_zyx=np.asarray([10, 20, 30]),
        stride=4,
        vector_scale=8.0,
    )

    assert vectors.shape == (1, 2, 3)
    np.testing.assert_allclose(vectors[0, 0], [14.0, 24.0, 34.0])
    np.testing.assert_allclose(vectors[0, 1], [0.0, 0.0, 8.0])


def test_distributed_context_parses_torchrun_env_and_selects_local_cuda_device():
    context = infer_streamline._distributed_context_from_env({
        "WORLD_SIZE": "4",
        "RANK": "2",
        "LOCAL_RANK": "1",
    })

    assert context.world_size == 4
    assert context.rank == 2
    assert context.local_rank == 1
    assert context.is_distributed
    assert not context.is_rank0
    assert infer_streamline._distributed_device_for_args("cuda", context) == "cuda:1"
    assert infer_streamline._distributed_device_for_args("cpu", context) == "cpu"
    assert infer_streamline._resolve_distributed_backend("auto", "cuda:1") == "nccl"
    assert infer_streamline._resolve_distributed_backend("auto", "cpu") == "gloo"


def test_distributed_batch_assignment_is_static_round_robin():
    batch_specs = infer_streamline._batch_specs_for_count(item_count=10, batch_size=3)

    assert batch_specs == [
        {"batch_index": 0, "start": 0, "stop": 3},
        {"batch_index": 1, "start": 3, "stop": 6},
        {"batch_index": 2, "start": 6, "stop": 9},
        {"batch_index": 3, "start": 9, "stop": 10},
    ]
    assert [s["batch_index"] for s in infer_streamline._assigned_batch_specs(batch_specs, 0, 3)] == [0, 3]
    assert [s["batch_index"] for s in infer_streamline._assigned_batch_specs(batch_specs, 1, 3)] == [1]
    assert [s["batch_index"] for s in infer_streamline._assigned_batch_specs(batch_specs, 2, 3)] == [2]
    assert infer_streamline._rank_batch_assignment_summary(total_batches=4, world_size=3) == {
        "0": [0, 3],
        "1": [1],
        "2": [2],
    }


def test_distributed_mode_rejects_show_napari():
    args = infer_streamline._parse_args([
        "--tifxyz-path",
        "input.tifxyz",
        "--checkpoint-path",
        "ckpt.pth",
        "--output-dir",
        "out",
        "--show-napari",
    ])
    context = infer_streamline._DistributedContext(world_size=2, rank=0, local_rank=0)

    with pytest.raises(ValueError, match="show-napari"):
        infer_streamline._validate_distributed_args(args, context)


def test_rank0_only_output_guard():
    assert infer_streamline._should_write_outputs(infer_streamline._DistributedContext(world_size=2, rank=0))
    assert not infer_streamline._should_write_outputs(infer_streamline._DistributedContext(world_size=2, rank=1))
