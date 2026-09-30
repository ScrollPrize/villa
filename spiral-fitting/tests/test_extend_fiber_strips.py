from pathlib import Path

import numpy as np
import torch

import extend_fiber_strips as extension
from connect_overlapping_patches import build_surface_index
from tifxyz import Patch


def planar_patch(x_offset):
    rows, cols = np.mgrid[:4, :4]
    zyx = np.stack((np.zeros_like(rows), rows, cols + x_offset), axis=-1)
    return Patch(torch.tensor(zyx, dtype=torch.float32), torch.ones(2), None, None)


def test_shared_index_matches_separate_source_and_strip_indexes(monkeypatch):
    source = planar_patch(0)
    strip = planar_patch(0)
    candidate = planar_patch(2)
    distant = planar_patch(100)
    paths = {Path('/patches/source'): source,
             Path('/patches/candidate'): candidate,
             Path('/patches/distant'): distant,
             Path('/outputs/fiber_merge_all_strips/strips_flattened/strip'): strip}
    monkeypatch.setattr(extension, 'load_tifxyz', lambda path, **_kwargs: paths[path])
    available = {'source': (Path('/patches/source'), None),
                 'candidate': (Path('/patches/candidate'), None),
                 'distant': (Path('/patches/distant'), None)}
    index, positions, empty = extension.build_shared_index(
        available, [{'name': 'strip'}], Path('/outputs'), 0.1)
    assert not empty
    nearby = set(index.nearby_surfaces(positions['graph', 'source']))
    assert positions['graph', 'candidate'] in nearby
    assert positions['graph', 'distant'] not in nearby
    source_subset = np.asarray([positions['graph', 'source']], dtype=np.int32)
    strip_subset = np.asarray([positions['strip', 'strip']], dtype=np.int32)
    standalone_source = build_surface_index({'source': source}, 0.1, 1)
    standalone_strip = build_surface_index({'strip': strip}, 0.1, 1)

    for stride in (1, 2):
        shared = extension.overlap_counts(index, positions['graph', 'candidate'],
                                          source_subset, strip_subset, 0.1, stride)
        xyz = extension.patch_query_points(candidate, stride)
        np.testing.assert_array_equal(
            index.sample_xyz(positions['graph', 'candidate'], stride), xyz)
        source_hits = extension.hits(standalone_source, xyz, 0.1)
        strip_hits = extension.hits(standalone_strip, xyz, 0.1)
        assert shared == {'samples': len(xyz), 'source_hits': int(source_hits.sum()),
                          'strip_hits': int(strip_hits.sum()),
                          'joint_hits': int((source_hits & strip_hits).sum())}


def test_merged_strip_metadata_gets_fresh_uuid(tmp_path):
    import json
    import uuid
    original = tmp_path / 'original'
    merged = tmp_path / 'merged'
    original.mkdir()
    merged.mkdir()
    source_uuid = str(uuid.uuid4())
    (original / 'meta.json').write_text(json.dumps(
        {'type': 'seg', 'format': 'tifxyz', 'uuid': source_uuid, 'extra': 1}))
    (merged / 'meta.json').write_text(json.dumps({'bbox': [[0, 0, 0], [1, 1, 1]],
                                                  'constraint_graph': {}}))
    extension.write_merged_metadata(
        original, merged, {'accepted': [{'patch': 'p'}]},
        {'area_vx2': 4.0, 'original_retained_fraction': 1.0, 'original_distance_p95': 0.0})
    result = json.loads((merged / 'meta.json').read_text())
    assert result['uuid'] != source_uuid
    assert uuid.UUID(result['uuid']).version == 4
    assert result['extra'] == 1 and result['source'] == 'anchored_strip_merge'
