"""Fixtures shared by the object-store and service tests (none is autouse).

Stores are written with object_store_fixture; tests that need a CPU-only
fit request the ``cpu`` fixture (``pytestmark = pytest.mark.usefixtures('cpu')``).
"""
import pytest


@pytest.fixture()
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


@pytest.fixture()
def store_path(tmp_path):
    """A closed store with two patches and a PCL touching each of them."""
    from object_store_fixture import write_walk_store
    return write_walk_store(tmp_path / 'fit.vcobj')


@pytest.fixture()
def shared(tmp_path):
    """The same objects in one store file the service and an editor (VC3D on
    the service's host) write in turn."""
    from object_store_fixture import write_walk_store
    return write_walk_store(tmp_path / 'shared.vcobj')


@pytest.fixture()
def device_stubs(monkeypatch):
    """Stub the CUDA-only stages of fit_spiral.FitContext and count the
    object-store input loads (each one's ``object_store_refresh_links``)."""
    import fit_spiral
    import object_store_inputs as osi
    from cpu_service import stub_device_stages
    for name in ('build_device_state', 'rebuild_model_state',
                 'release_setup_only_tracks'):
        monkeypatch.setattr(fit_spiral.FitContext, name,
                            getattr(fit_spiral.FitContext, name))
    stub_device_stages(fit_spiral)
    loads = []
    load = osi.load_fit_inputs

    def counted(config, **kwargs):
        loads.append(config['object_store_refresh_links'])
        return load(config, **kwargs)
    monkeypatch.setattr(osi, 'load_fit_inputs', counted)
    return loads


@pytest.fixture()
def db(tmp_path):
    """An empty store, open for writing."""
    from vc_object_store import ObjectStore
    with ObjectStore.create(str(tmp_path / 'fit.vcobj'), author='test',
                            hostname='host') as store:
        yield store
