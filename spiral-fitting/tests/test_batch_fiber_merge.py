import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

import batch_fiber_merge as batch


def test_batch_exports_only_continuous_surface_and_resumes(tmp_path, monkeypatch):
    links = tmp_path / 'links.json'
    links.write_text(json.dumps({'type': 'vc_fiber_patch_links', 'version': 1, 'fibers': {
        'a.json': {'linked': [{'patch': 'a'}]}, 'b.json': {'linked': []}}}))
    executable = tmp_path / 'fitter'
    executable.write_text('test executable fingerprint')
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[-1] == '--continuous-only'
        attempt = Path(command[3])
        surface = attempt / 'continuous/refined'
        surface.mkdir(parents=True)
        y, x = np.mgrid[:3, :3]
        for axis, values in zip('xyz', [100+x, 200+y, np.full((3, 3), 300)]):
            tifffile.imwrite(surface / f'{axis}.tif', values.astype(np.float32))
        (surface / 'meta.json').write_text(json.dumps({'type': 'seg', 'format': 'tifxyz', 'uuid': 'refined', 'constraint_graph': {'includes_fiber_ribbons': True}}))
        component = {'valid': True, 'path': 'continuous/refined', 'origin': [0, 0], 'spacing': 10}
        info = dict(component, components=[component], component_count=1,
                    dropped_single_quad_components=2, dropped_other_components=1)
        (attempt / 'report.json').write_text(json.dumps({'complete': True, 'continuous': {'refined': info}, 'fits': [{'valid': True}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, 'run', run)
    args = SimpleNamespace(links=links, executable=executable, output=tmp_path/'output',
                           workers=2, spacing=10, sweeps=1, retry_failed=False)
    assert batch.run_batch(args) == 0
    status = json.loads((args.output / 'status.json').read_text())
    assert status['complete']
    assert status['counts'] == {'exported': 1, 'no_assignments': 1}
    assert list((args.output / 'strips').iterdir()) == [args.output / 'strips/a.tifxyz']
    assert len(list(args.output.rglob('x.tif'))) == 1
    published_uuid = json.loads((args.output / 'strips/a.tifxyz/meta.json').read_text())['uuid']
    assert uuid.UUID(published_uuid).version == 4 and published_uuid not in ('a', 'refined')
    assert batch.run_batch(args) == 0
    assert len(commands) == 1
    # Resumption must refuse silently changed assignments.
    links.write_text(links.read_text() + '\n')
    import pytest
    with pytest.raises(ValueError, match='differ'):
        batch.run_batch(args)


def test_failed_fitter_is_recorded_without_publishing_surface(tmp_path, monkeypatch):
    (tmp_path / 'strips').mkdir()
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    row = batch.run_one('failed.json', 2, {'output': str(tmp_path), 'executable': '/unused', 'spacing': 10, 'sweeps': 1})
    assert row['status'] == 'failed'
    assert row['returncode'] == 1
    assert not list((tmp_path / 'strips').iterdir())
    assert json.loads((tmp_path / 'jobs/failed/status.json').read_text()) == row


def test_fiber_only_batch_includes_unassigned_horizontal_fiber(tmp_path, monkeypatch):
    links = tmp_path / 'links.json'
    links.write_text(json.dumps({'type': 'vc_fiber_patch_links', 'version': 1, 'fibers': {
        'horizontal.json': {'direction': 'H', 'linked': []}}}))
    executable = tmp_path / 'fitter'
    executable.write_text('fiber-only test executable')

    def run(command, **kwargs):
        assert command[-1] == '--fiber-only'
        assert command[-2] == '0'
        attempt = Path(command[3])
        surface = attempt / 'continuous/refined'
        surface.mkdir(parents=True)
        y, x = np.mgrid[:3, :3]
        for axis, values in zip('xyz', [100+x, 200+y, np.full((3, 3), 300)]):
            tifffile.imwrite(surface / f'{axis}.tif', values.astype(np.float32))
        (surface / 'meta.json').write_text(json.dumps({'type': 'seg', 'format': 'tifxyz', 'uuid': 'refined',
            'constraint_graph': {'includes_fiber_ribbons': True, 'fiber_only': True}}))
        component = {'valid': True, 'path': 'continuous/refined', 'origin': [0, 0], 'spacing': 10}
        info = dict(component, components=[component], component_count=1,
                    dropped_single_quad_components=0, dropped_other_components=0)
        (attempt / 'report.json').write_text(json.dumps({'complete': True, 'continuous': {'refined': info}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, 'run', run)
    args = SimpleNamespace(links=links, executable=executable, output=tmp_path/'output',
                           workers=1, spacing=10, sweeps=1, retry_failed=False, fiber_only=True)
    assert batch.run_batch(args) == 0
    row = json.loads((args.output / 'status.json').read_text())['fibers'][0]
    assert row['status'] == 'exported' and row['patch_count'] == 0 and row['export_kind'] == 'fiber_only'
    meta = json.loads((args.output / row['surface'] / 'meta.json').read_text())
    assert uuid.UUID(meta['uuid']).version == 4 and meta['constraint_graph']['fiber_only']


def test_id_repair_waits_for_completion_and_preserves_geometry_and_other_metadata(tmp_path):
    import pytest
    rows = []
    original = {'type': 'seg', 'format': 'tifxyz', 'uuid': 'refined', 'scale': [.1, .1], 'extra': {'preserve': True}}
    for name in ('a', 'b'):
        folder = tmp_path / 'strips' / f'{name}.tifxyz'
        folder.mkdir(parents=True)
        (folder / 'meta.json').write_text(json.dumps(original))
        (folder / 'x.tif').write_bytes(b'unchanged geometry')
        rows.append({'status': 'exported', 'surface': f'strips/{name}.tifxyz'})
    status = {'complete': False, 'fibers': rows}
    batch.write_json(tmp_path / 'status.json', status)
    with pytest.raises(ValueError, match='not finished'):
        batch.finalize_segment_ids_locked(tmp_path)
    assert not (tmp_path / 'metadata_before_unique_ids').exists()
    assert all(json.loads(p.read_text()) == original for p in (tmp_path / 'strips').glob('*/meta.json'))
    status['complete'] = True
    batch.write_json(tmp_path / 'status.json', status)
    first = batch.finalize_segment_ids_locked(tmp_path)
    assert first['unique_ids'] == 2
    assigned = set()
    for name in ('a', 'b'):
        folder = tmp_path / 'strips' / f'{name}.tifxyz'
        repaired = json.loads((folder / 'meta.json').read_text())
        assert uuid.UUID(repaired['uuid']).version == 4
        assert repaired == dict(original, uuid=repaired['uuid'])
        assigned.add(repaired['uuid'])
        assert (folder / 'x.tif').read_bytes() == b'unchanged geometry'
        assert json.loads((tmp_path / 'metadata_before_unique_ids' / folder.name / 'meta.json').read_text()) == original
    assert len(assigned) == 2
    assert batch.finalize_segment_ids_locked(tmp_path) == first


def test_id_repair_keeps_unique_fitter_uuids_and_replaces_duplicates(tmp_path):
    kept = str(uuid.uuid4())
    rows = []
    for name in ('a', 'b', 'c'):
        folder = tmp_path / 'strips' / f'{name}.tifxyz'
        folder.mkdir(parents=True)
        # a and b share one uuid (a copied surface); c is legacy name-derived.
        meta_uuid = 'c' if name == 'c' else kept
        (folder / 'meta.json').write_text(json.dumps({'type': 'seg', 'format': 'tifxyz', 'uuid': meta_uuid}))
        rows.append({'status': 'exported', 'surface': f'strips/{name}.tifxyz'})
    batch.write_json(tmp_path / 'status.json', {'complete': True, 'fibers': rows})
    batch.finalize_segment_ids_locked(tmp_path)
    ids = {name: json.loads((tmp_path / 'strips' / f'{name}.tifxyz' / 'meta.json').read_text())['uuid']
           for name in ('a', 'b', 'c')}
    assert ids['a'] == kept
    assert len(set(ids.values())) == 3
    assert all(uuid.UUID(value).version == 4 for value in ids.values())
