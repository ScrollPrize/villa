import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from vc_object_store import ObjectStore

spec = importlib.util.spec_from_file_location('upgrade_vc3d_fibers', Path(__file__).parents[2] / 'scripts/upgrade_vc3d_fibers.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

NAMESPACE = uuid.UUID('e14f6968-9145-494d-8b06-ab2f2fac7f3b')
FILES = ['kb_20260821T164641854_000031.json', 'lt_20261016T073323561_000068.json', 'dj_20260902T151037604_000001.json']


def link(target, own, other, direction, branch_direction):
    return dict(branch_file=FILES[target], branch_fiber_id=target, control_point_index=own,
                branch_control_point_index=other, control_point_position=[0, 0, 0],
                branch_control_point_position=[0, 0, 0], control_point_direction=direction,
                branch_control_point_direction=branch_direction, pending=True)


def write_vc3d_folder(directory, noise=1e-16):
    """Files as VC3D saved them before the object store: version 4, links by file only."""
    directory.mkdir()
    documents = [dict(type='vc3d_fiber', version=4, filename=name, sequence=i,
                      control_points=[{'position': [i, 0, 0]}, {'position': [i, 1, 0]}],
                      line_points=[[i, 0, 0], [i, 1, 0]], branches=[])
                 for i, name in enumerate(FILES)]
    documents[1]['uuid'] = str(uuid.uuid4()).upper()
    documents[0].update(vc_open_data_coordinate_space='Sample/20260411134726@L0',
                        vc_open_data_source_coordinate_scale_factor=1,
                        vc_open_data_source_path='https://example.invalid/volume.zarr')
    documents[2]['version'] = 3
    # Each link declared from both ends; directions differ by float noise.
    documents[0]['branches'].append(link(1, 1, 0, [1, 0, 0], [0, 1, 0]))
    documents[1]['branches'].append(link(0, 0, 1, [0, 1 + noise, 0], [1, 0, 0]))
    # A version 4 file can link to a version 3 one.
    documents[1]['branches'].append(link(2, 1, 0, [0, 0, 1], [1, 0, 0]))
    documents[2]['branches'].append(link(1, 0, 1, [1, 0, 0], [0, 0, 1]))
    for name, document in zip(FILES, documents):
        (directory / name).write_text(json.dumps(document))
    return documents


def test_vc3d_version4_folder_ingests_after_upgrade(tmp_path):
    documents = write_vc3d_folder(tmp_path / 'vc3d')
    before = {name: (tmp_path / 'vc3d' / name).read_bytes() for name in FILES}
    with ObjectStore.create(str(tmp_path / 'strict.vcobj'), author='test') as store:
        assert store.ingest_geometry_source('fiber', tmp_path / 'vc3d')['problems']
        assert store.object_count('fiber') == 0
    assert module.upgrade(tmp_path / 'vc3d', tmp_path / 'upgraded') == (3, 2, 2)
    assert {name: (tmp_path / 'vc3d' / name).read_bytes() for name in FILES} == before
    upgraded = {name: json.loads((tmp_path / 'upgraded' / name).read_text()) for name in FILES}
    ids = [str(uuid.uuid5(NAMESPACE, 'vc3d_fiber/' + FILES[0])), documents[1]['uuid'].lower(),
           str(uuid.uuid5(NAMESPACE, 'vc3d_fiber/' + FILES[2]))]
    assert upgraded[FILES[0]]['branches'][0]['target_uuid'] == ids[1]
    assert upgraded[FILES[0]]['coordinates'] == dict(volume_uuid='20260411134726', scale_factor=1)
    assert not any(key.startswith('vc_open_data_') for key in upgraded[FILES[0]])
    assert [entry['version'] for entry in upgraded.values()] == [4, 4, 3]
    # Files are read in name order (dj, kb, lt): lt's two declarations repeat links
    # already declared from dj and kb, so they are dropped and each link reaches ingest once.
    assert upgraded[FILES[2]]['branches'] == [dict(documents[2]['branches'][0], target_uuid=ids[1])]
    assert upgraded[FILES[1]]['branches'] == []
    with ObjectStore.create(str(tmp_path / 'upgraded.vcobj'), author='test') as store:
        assert store.ingest_geometry_source('fiber', tmp_path / 'upgraded') == dict(
            ingested=3, edges_imported=2, problems=[])
        assert {frozenset((edge['a'], edge['b'])) for edge in store.edges_by_kind('fiber_branch')} == {
            frozenset(ids[:2]), frozenset(ids[1:])}
        assert store.object(ids[0])['meta']['coordinates'] == dict(volume_uuid='20260411134726', scale_factor=1)


def test_upgrade_matches_version3_ingest(tmp_path):
    write_vc3d_folder(tmp_path / 'vc3d')
    (tmp_path / 'as3').mkdir()
    for name in FILES:
        document = json.loads((tmp_path / 'vc3d' / name).read_text())
        document['version'] = 3
        (tmp_path / 'as3' / name).write_text(json.dumps(document))
    module.upgrade(tmp_path / 'vc3d', tmp_path / 'upgraded')
    stores = {}
    for folder in ('as3', 'upgraded'):
        with ObjectStore.create(str(tmp_path / (folder + '.vcobj')), author='test') as store:
            assert store.ingest_geometry_source('fiber', tmp_path / folder)['problems'] == []
            keep = ('a', 'b', 'a_anchor', 'b_anchor', 'approval_state', 'meta', 'kind')
            stores[folder] = (
                sorted(json.dumps({k: edge[k] for k in keep}, sort_keys=True) for edge in store.edges_by_kind('fiber_branch')),
                sorted(json.dumps(store.object(uid)['meta'], sort_keys=True)
                       for uid in (o['uuid'] if isinstance(o, dict) else o for o in store.objects_by_kind('fiber'))))
    assert stores['as3'] == stores['upgraded']


def test_upgrade_refuses_disagreeing_or_missing_links(tmp_path):
    write_vc3d_folder(tmp_path / 'conflict', noise=1e-3)
    with pytest.raises(SystemExit, match='disagrees'):
        module.upgrade(tmp_path / 'conflict', tmp_path / 'out')
    write_vc3d_folder(tmp_path / 'missing')
    (tmp_path / 'missing' / FILES[2]).unlink()
    with pytest.raises(SystemExit, match='not in'):
        module.upgrade(tmp_path / 'missing', tmp_path / 'out')
    assert not (tmp_path / 'out').exists() and not list(tmp_path.glob('.out.*'))
    write_vc3d_folder(tmp_path / 'ok')
    (tmp_path / 'taken').mkdir()
    with pytest.raises(SystemExit):
        module.main([str(tmp_path / 'ok'), str(tmp_path / 'taken')])
