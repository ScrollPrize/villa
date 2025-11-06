import numpy as np
import torch
from pathlib import Path
import trimesh

from vesuvius.models.datasets.mesh.handles import MeshHandle
from vesuvius.models.datasets.mesh.types import MeshMetadata, MeshPayload
from vesuvius.models.datasets.mesh.voxelize import mesh_to_binary_voxels
from vesuvius.models.datasets.slicers.chunk import (
    ChunkPatch,
    ChunkSliceConfig,
    ChunkSlicer,
    ChunkVolume,
)
from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms


def _make_mesh_handle(mesh_id: str, volume_id: str) -> MeshHandle:
    metadata = MeshMetadata(
        mesh_id=mesh_id,
        path=Path(f"{mesh_id}.obj"),
        source_volume_id=volume_id,
        transform=None,
        attributes={},
    )
    payload = MeshPayload(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.int64),
        normals=None,
        uv=None,
        uv_faces=None,
    )
    return MeshHandle(
        path=metadata.path,
        metadata=metadata,
        loader=lambda _: payload,
    )


def test_chunk_slicer_attaches_mesh_payloads():
    config = ChunkSliceConfig(
        patch_size=(2, 2, 2),
        stride=None,
        min_labeled_ratio=0.0,
        min_bbox_percent=0.0,
        allow_unlabeled=True,
        downsample_level=1,
        num_workers=0,
        cache_enabled=False,
        cache_dir=None,
    )

    slicer = ChunkSlicer(config=config, target_names=["ink"])

    image = np.zeros((4, 4, 4), dtype=np.float32)
    labels = {"ink": np.zeros((4, 4, 4), dtype=np.float32)}
    mesh_handle = _make_mesh_handle("mesh0", "vol0")

    slicer.register_volume(
        ChunkVolume(
            index=0,
            name="vol0",
            image=image,
            labels=labels,
            label_source=None,
            cache_key_path=None,
            meshes={"mesh0": mesh_handle},
        )
    )

    patch = ChunkPatch(volume_index=0, volume_name="vol0", position=(0, 0, 0), patch_size=(2, 2, 2))
    result = slicer.extract(patch)

    assert "mesh0" in result.meshes
    mesh_entry = result.meshes["mesh0"]
    assert mesh_entry["payload"].vertices.shape == (3, 3)
    assert mesh_entry["metadata"].mesh_id == "mesh0"
    assert "meshes" in result.patch_info
    assert result.patch_info["meshes"]["mesh0"]["path"].endswith("mesh0.obj")

class _SpatialProbe(BasicTransform):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self._skip_when_vector = True

    def apply(self, data_dict, **params):
        self.calls += 1
        data_dict["spatial_tag"] = True
        return data_dict


class _IntensityProbe(BasicTransform):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def apply(self, data_dict, **params):
        self.calls += 1
        data_dict["intensity_tag"] = True
        return data_dict


def test_compose_skips_spatial_transforms_when_flagged():
    spatial = _SpatialProbe()
    intensity = _IntensityProbe()
    compose = ComposeTransforms([spatial, intensity])

    sample = {
        "image": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        "_skip_spatial_transforms": True,
    }

    result = compose(**sample)

    assert spatial.calls == 0
    assert intensity.calls == 1
    assert "intensity_tag" in result
    assert "spatial_tag" not in result


def test_compose_runs_spatial_transforms_without_flag():
    spatial = _SpatialProbe()
    intensity = _IntensityProbe()
    compose = ComposeTransforms([spatial, intensity])

    sample = {"image": torch.zeros((1, 1, 1, 1), dtype=torch.float32)}
    result = compose(**sample)

    assert spatial.calls == 1
    assert intensity.calls == 1
    assert "spatial_tag" in result
    assert "intensity_tag" in result


def test_mesh_to_binary_voxels_produces_filled_volume():
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    payload = MeshPayload(
        vertices=np.asarray(cube.vertices, dtype=np.float32),
        faces=np.asarray(cube.faces, dtype=np.int64),
    )

    result = mesh_to_binary_voxels(payload, voxel_size=0.25)

    assert result.voxels.dtype == np.bool_
    assert result.voxels.shape[0] > 0 and result.voxels.shape[1] > 0 and result.voxels.shape[2] > 0
    assert result.voxels.sum() > 0
    assert result.voxel_size == 0.25
    assert result.origin.shape == (3,)
