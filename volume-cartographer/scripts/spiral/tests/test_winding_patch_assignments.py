import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from point_collection import locate_points_on_patches
from theta_crossing_map import ThetaCrossingMap
import winding_patch_assignments as assignments


class _FakePatch:
    def __init__(self, *, invalid=False, displacement=0.0):
        self.invalid = invalid
        self.displacement = float(displacement)

    def ij_to_zyx(self, ij):
        result = torch.stack([
            torch.zeros(len(ij)),
            torch.zeros(len(ij)),
            ij[:, 1] + self.displacement,
        ], dim=-1)
        valid = torch.full((len(ij),), not self.invalid, dtype=torch.bool)
        if self.invalid:
            result[:] = -1
        return result, valid


def test_verified_patches_load_in_worker_processes(tmp_path):
    root = tmp_path / 'patches'
    for patch_id, offset in [('b', 10), ('a', 0)]:
        patch = root / patch_id
        patch.mkdir(parents=True)
        (patch / 'meta.json').write_text(json.dumps({
            'format': 'tifxyz',
            'scale': [1, 1],
            'uuid': patch_id,
        }))
        for axis in range(3):
            grid = np.arange(4, dtype=np.float32).reshape(2, 2)
            Image.fromarray(grid + offset + axis).save(
                patch / f"{'zyx'[axis]}.tif")
        # Geometry-only preprocessing must not try to decode this optional
        # file, which is deliberately not a TIFF.
        (patch / 'winding.tif').write_bytes(b'not needed')

    loaded = assignments._load_verified_patches(root, workers=2)
    assert list(loaded) == ['a', 'b']
    assert loaded['a'].winding is None
    assert loaded['b'].zyxs.shape == (2, 2, 3)


def test_compact_linker_uses_zyx_input_and_largest_area_policy():
    class IndexedPatch:
        def __init__(self, area):
            self.area = area

    class Index:
        def locate_all_xyz_batch(self, xyz, tolerance):
            assert tolerance == 2.5
            assert xyz.tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]
            return (
                np.array([0, 2, 3], dtype=np.int64),
                np.array([0, 1, 1], dtype=np.int32),
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32),
            )

    patches = {'small': IndexedPatch(1), 'large': IndexedPatch(2)}
    patch_index, ij, distance, patch_ids = locate_points_on_patches(
        patches,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        2.5,
        built_index=(Index(), ['small', 'large']),
        general_hit_policy='largest_area',
    )
    assert patch_ids == ['small', 'large']
    assert patch_index.tolist() == [1, 1]
    assert ij.tolist() == [[2, 2], [3, 3]]
    assert distance == pytest.approx([0.2, 0.3])


class _FakeWindingStore:
    device = torch.device('cpu')

    def __init__(self, fingerprint='source-fingerprint'):
        self.fingerprint = {'fingerprint': fingerprint}
        self.offset = torch.tensor([0, 8], dtype=torch.int64)
        self.crossing_level = torch.arange(8, dtype=torch.int32)
        self.points = torch.stack([
            torch.zeros(8), torch.zeros(8), torch.arange(8)
        ], dim=-1).to(torch.float32)

    def materialize_flat(self, indices):
        return self.points[indices]


def _write_array(root, name, value):
    path = root / f'{name}.npy'
    np.save(path, value, allow_pickle=False)
    return assignments._array_description(path, root, value)


def _write_assignment_artifact(root, *, source_fingerprint='source-fingerprint'):
    shard = root / 'shard-00000'
    shard.mkdir(parents=True)
    crossing = np.arange(7, dtype=np.int64)
    patch_index = np.arange(7, dtype=np.int32)
    patch_ij = np.stack([
        np.zeros(7, dtype=np.float32), np.arange(7, dtype=np.float32)
    ], axis=-1)
    distance = np.zeros(7, dtype=np.float32)
    arrays = {
        name: _write_array(shard, name, value)
        for name, value in (
            ('crossing_index', crossing),
            ('patch_index', patch_index),
            ('patch_ij', patch_ij),
            ('distance', distance),
        )
    }
    manifest = {
        'artifact_type': assignments.ARTIFACT_TYPE,
        'format_version': assignments.FORMAT_VERSION,
        'coordinate_order': 'zyx',
        'source_winding_inference_fingerprint': source_fingerprint,
        'attachment_tolerance': 2.5,
        'hit_policy': 'largest_area',
        'patch_ids': [
            'a', 'removed', 'changed', 'b', 'invalid', 'ambiguous', 'ambiguous'
        ],
        'num_source_crossings': 8,
        'num_attached': 7,
        'shards': [{
            'name': 'shard-00000',
            'source_shard': 'source-0',
            'source_crossing_base': 0,
            'num_source_rays': 1,
            'num_source_crossings': 8,
            'num_attached': 7,
            'arrays': arrays,
        }],
    }
    manifest['fingerprint'] = assignments._canonical_digest(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_loader_skips_only_stale_patch_assignments(tmp_path):
    root = tmp_path / 'assignments'
    _write_assignment_artifact(root)
    patches = {
        'a': _FakePatch(),
        'changed': _FakePatch(displacement=100),
        'b': _FakePatch(),
        'invalid': _FakePatch(invalid=True),
        # Both rows attach to the same current patch at different levels and
        # are removed as one ambiguous ray/patch group.
        'ambiguous': _FakePatch(),
    }
    store = _FakeWindingStore()
    prepared = assignments.load_winding_patch_assignments(
        root, store, patches)

    assert prepared.stats == {
        'stored': 7,
        'missing_patch': 1,
        'invalid_ij': 1,
        'geometry_mismatch': 1,
        'ambiguous_level': 2,
        'retained': 2,
        'eligible_rays': 1,
    }
    assert prepared.crossing_index.tolist() == [0, 3]
    assert prepared.num_eligible_rays == 1

    crossing_map = ThetaCrossingMap('cpu')
    prepared.register_theta_topology(crossing_map, store)
    cfg = {
        'sample_count_winding_model_patch_relative_rays': 1,
        'sample_count_winding_model_patch_pairs_per_ray': 1,
        'winding_model_relative_pair_delta': [3, 15],
        'pcl_rel_winding_adjacent_patches_only': True,
    }
    request, = prepared.sample_pair_requests(cfg)
    assert request[2:5] == ('a', 'b', 3.0)
    assert request[5].tolist() == [0, 1, 2, 3]
    assert request[6:8] == (0, 3)


def test_loader_rejects_a_different_winding_source(tmp_path):
    root = tmp_path / 'assignments'
    _write_assignment_artifact(root, source_fingerprint='old-source')
    with pytest.raises(ValueError, match='source fingerprint'):
        assignments.load_winding_patch_assignments(
            root, _FakeWindingStore('new-source'), {'a': _FakePatch()})


def test_pair_sampler_preserves_signed_winding_deltas():
    prepared = assignments._prepare_compact_groups(
        crossing_index=np.array([0, 3], dtype=np.int64),
        patch_index=np.array([0, 1], dtype=np.int32),
        patch_ij=np.array([[0, 0], [0, 3]], dtype=np.float32),
        level=np.array([5, 2], dtype=np.int32),
        ray=np.array([0, 0], dtype=np.int64),
        source_local=np.array([0, 3], dtype=np.int64),
        patch_ids=['a', 'b'],
        stats={},
        fingerprint={},
    )
    prepared.register_theta_topology(
        ThetaCrossingMap('cpu'), _FakeWindingStore())
    request, = prepared.sample_pair_requests({
        'sample_count_winding_model_patch_relative_rays': 1,
        'sample_count_winding_model_patch_pairs_per_ray': 1,
        'winding_model_relative_pair_delta': [3, 15],
        'pcl_rel_winding_adjacent_patches_only': True,
    })
    assert request[4] == -3.0


def _write_source_inference(root):
    shard = root / 'source-0'
    shard.mkdir(parents=True)
    values = {
        'ray_origin_zyx': np.array([[0, 0, 0]], dtype=np.float32),
        'ray_step_zyx': np.array([[0, 0, 1]], dtype=np.float32),
        'crossing_t': np.arange(4, dtype=np.float32),
        'crossing_offsets': np.array([0, 4], dtype=np.int64),
    }
    arrays = {
        name: _write_array(shard, name, value)
        for name, value in values.items()
    }
    manifest = {
        'artifact_type': assignments.INFERENCE_ARTIFACT_TYPE,
        'format_version': assignments.INFERENCE_FORMAT_VERSION,
        'coordinate_order': 'zyx',
        'num_rays': 1,
        'num_crossings': 4,
        'shards': [{'name': 'source-0', 'arrays': arrays}],
    }
    manifest['fingerprint'] = assignments._canonical_digest(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_builder_writes_sparse_atomic_artifact(tmp_path, monkeypatch):
    source = tmp_path / 'inference'
    _write_source_inference(source)
    patches = tmp_path / 'patches'
    patches.mkdir()
    output = tmp_path / 'winding_patch_assignments'

    loaded_with = []

    def load_patches(_path, *, workers):
        loaded_with.append(workers)
        return {'p1': object()}

    monkeypatch.setattr(assignments, '_load_verified_patches', load_patches)
    built_index = (object(), ['p1'])
    monkeypatch.setattr(
        assignments, 'build_surface_patch_index',
        lambda _patches, _tolerance: built_index)

    def locate(_patches, points, _tolerance, **_kwargs):
        attached = points[:, 2].astype(np.int32) % 2 == 0
        patch_index = np.where(attached, 0, -1).astype(np.int32)
        ij = np.stack([np.zeros(len(points)), points[:, 2]], axis=-1).astype(np.float32)
        distance = np.where(attached, 0.25, np.inf).astype(np.float32)
        return patch_index, ij, distance, ['p1']

    monkeypatch.setattr(assignments, 'locate_points_on_patches', locate)
    assignments.build_winding_patch_assignments(
        source, patches, output, chunk_size=2, patch_workers=3)
    assert loaded_with == [3]

    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['num_source_crossings'] == 4
    assert manifest['num_attached'] == 2
    shard = manifest['shards'][0]
    stored = np.load(
        output / shard['name'] / shard['arrays']['crossing_index']['file'])
    assert stored.tolist() == [0, 2]
    with pytest.raises(FileExistsError):
        assignments.build_winding_patch_assignments(
            source, patches, output, chunk_size=2, patch_workers=3)
    assignments.build_winding_patch_assignments(
        source, patches, output, chunk_size=2, patch_workers=3, force=True)
    assert loaded_with == [3, 3]
