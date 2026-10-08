import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts' / 'split_fiber_collection.py'
spec = importlib.util.spec_from_file_location('split_fiber_collection', path)
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def test_length_and_whole_fiber_round_up():
    assert split.fiber_length([(0, 0, 0), (3, 4, 0), (3, 4, 12)]) == 17
    records = [{'path': str(i), 'length_base_voxels': length}
               for i, length in enumerate([1, 1, 1, 1, 96])]
    stats = split.assign_source(records, seed=1, source='x', fraction=.05)
    assert stats['test_length'] >= 5
    assert stats['test_prefix_before_last_length'] < 5
    assert next(r for r in records if r['length_base_voxels'] == 96)['split'] == 'test'
    other = list(reversed(copy.deepcopy(records)))
    split.assign_source(other, seed=1, source='x', fraction=.05)
    assert sorted(records, key=lambda r: r['path']) == sorted(other, key=lambda r: r['path'])


@pytest.mark.parametrize('points', [[(0, 0, 0)], [(1, 2, 3), (1, 2, 3)], [(0, 0, 0), (float('nan'), 1, 0)]])
def test_invalid_length(points):
    with pytest.raises(ValueError):
        split.fiber_length(points)


def test_publish_preserves_bytes_and_refuses_overwrite(tmp_path):
    root = tmp_path / 'all'
    (root / 'source').mkdir(parents=True)
    (root / 'SOURCES.md').write_text('provenance')
    (root / 'SHA256SUMS').write_text('fixture inventory')
    records = []
    for i, side in enumerate(['train', 'test']):
        source = root / 'source' / f'{i}.json'
        source.write_text(json.dumps({'fiber': i}))
        records.append({'path': f'source/{i}.json', 'split': side, 'sha256': split.digest(source.read_bytes())})
    plan = {'source_root': str(root), 'files': records, 'seed': 1, 'test_fraction': .05, 'sources': [], 'cross_split_branch_references': [], 'cross_split_duplicate_geometry_groups': []}
    plan['snapshot_inventory_sha256'] = split.digest((root / 'SHA256SUMS').read_bytes())
    plan['source_documentation_sha256'] = split.digest((root / 'SOURCES.md').read_bytes())
    output = tmp_path / 'split'
    split.publish(plan, output)
    for r in records:
        link = output / r['split'] / r['path']
        assert link.is_symlink() and link.read_bytes() == (root / r['path']).read_bytes()
    with pytest.raises(FileExistsError):
        split.publish(plan, output)
    (root / records[0]['path']).write_text('changed')
    with pytest.raises(ValueError, match='Source changed'):
        split.publish(plan, tmp_path / 'other_split')
    assert not (tmp_path / 'other_split').exists()


def test_source_order_is_independent():
    a = [{'path': f'a/{i}', 'length_base_voxels': 1} for i in range(50)]
    first = copy.deepcopy(a)
    split.assign_source(first, seed=20260910, source='a', fraction=.05)
    b = [{'path': f'b/{i}', 'length_base_voxels': 1} for i in range(100)]
    split.assign_source(b, seed=20260910, source='b', fraction=.05)
    split.assign_source(a, seed=20260910, source='a', fraction=.05)
    assert a == first


def test_full_plan_inventory_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(split, 'SOURCES', ('source',))
    root = tmp_path / 'all'
    (root / 'source').mkdir(parents=True)
    (root / 'SOURCES.md').write_text('original provenance')
    inventory = []
    for i in range(10):
        path = root / 'source' / f'{i}.json'
        path.write_text(json.dumps({'type': 'vc3d_fiber', 'version': 1,
            'line_points': [[0, 0, 0], [i + 1, 0, 0]], 'control_points': [[0, 0, 0], [i + 1, 0, 0]]}))
        inventory.append(f'{split.digest(path.read_bytes())}  source/{i}.json')
    (root / 'SHA256SUMS').write_text('\n'.join(inventory))
    plan = split.plan_split(root, fraction=.2)
    assert len(plan['files']) == 10
    assert 'at least 20%' in split.summary_markdown(plan)
    assert split.plan_split(root, fraction=.2) == plan
    (root / 'SOURCES.md').write_text('modified provenance')
    with pytest.raises(ValueError, match='provenance changed'):
        split.publish(plan, tmp_path / 'output')
    (root / 'SOURCES.md').write_text('original provenance')
    (root / 'SHA256SUMS').write_text('\n'.join(inventory[:-1]))
    with pytest.raises(ValueError, match='provenance changed'):
        split.publish(plan, tmp_path / 'output')
    with pytest.raises(ValueError, match='checksum mismatch'):
        split.plan_split(root)
