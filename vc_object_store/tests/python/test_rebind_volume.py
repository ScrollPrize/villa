import importlib.util
from pathlib import Path
from vc_object_store import ObjectStore, Fiber

spec = importlib.util.spec_from_file_location('rebind_volume', Path(__file__).parents[2] / 'scripts/rebind_volume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rebind_preserves_geometry_and_history(tmp_path):
    path = tmp_path / 'binding.vcobj'
    old = '2d2a9b21-9c82-5343-9443-57838a5fa5a1'
    new = '20260411134726'
    with ObjectStore.create(str(path), author='test') as store:
        identity = store.upsert_object({'kind': 'fiber', 'meta': {'coordinates': {'volume_uuid': old, 'scale_factor': 4}}})
        geometry = Fiber([[1., 2., 3.], [4., 5., 6.]], [])
        revision = store.put_geometry(identity, geometry)
        prior = store.current_object_revision(identity)
        raw = store.read_geometry_array(revision, 'control_points')
    assert module.rebind(path, old, new)['matching'] == {'fiber': 1}
    with ObjectStore.open(str(path), read_only=True) as store:
        assert store.object(identity)['meta']['coordinates']['volume_uuid'] == old
    assert module.rebind(path, old, new, apply=True)['updated'] == 1
    assert module.rebind(path, old, new, apply=True)['updated'] == 0
    with ObjectStore.open(str(path), read_only=True) as store:
        assert store.object(identity)['meta']['coordinates'] == {'volume_uuid': new, 'scale_factor': 4}
        assert store.current_geometry(identity)['uuid'] == revision
        assert store.read_geometry_array(revision, 'control_points') == raw
        assert store.object_revision(prior['uuid'])['record']['meta']['coordinates']['volume_uuid'] == old
