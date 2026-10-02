"""AFV catalog rounding must not invalidate replay at geometry endpoints."""
import json
import pickle
import sqlite3

import numpy as np
import pytest

from test_data_and_scoring import make_states, sample_config
from test_mixed_datasets import afv_fixture
from vesuvius.neural_tracing.fiber_follow.shared.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


@pytest.fixture(params=[-np.inf, np.inf])
def rounded_catalog(tmp_path, request):
    path = tmp_path/'rounded.afv'
    afv_fixture(path)
    # Either rounding direction is possible when summing densified segments.
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE fibers SET length=? WHERE id=1',
                           (float(np.nextafter(80., request.param)),))
    return AFVFibers(path, grid_scale=2.)


def endpoint_states(fibers):
    states = make_states(fibers[0], z=0)
    states.manifest = fiber_manifest(fibers)
    states.provenance = dict(states.provenance, volume={'grid_scale': fibers.grid_scale})
    states.t = np.array([fibers[0].s[-1]])
    return states


def test_afv_length_uses_geometry_with_lazy_catalog_sampling(rounded_catalog):
    fibers = rounded_catalog
    dataset = FollowDataset(fibers, FiberVolumeSpec('unused', grid_scale=2.),
                            sample_config(), None)
    assert not fibers._cache  # Constructing sampling weights decodes no geometry.
    np.testing.assert_array_equal(dataset.weights, fibers.lengths/fibers.lengths.sum())
    assert fibers[0].length == fibers[0].s[-1] == 40.
    assert fibers[0].length != fibers.lengths[0]


def test_afv_endpoint_replay_refresh_and_worker_roundtrip(tmp_path, rounded_catalog):
    fibers = rounded_catalog
    states = endpoint_states(fibers)
    archive = tmp_path/'endpoint.npz'
    states.save(archive)
    index = tmp_path/'replay.json'
    index.write_text(json.dumps([str(archive)]))
    # A worker starts with a cold geometry cache and discovers new replay later.
    worker_fibers = pickle.loads(pickle.dumps(fibers))
    dataset = FollowDataset(worker_fibers, FiberVolumeSpec('unused', grid_scale=2.),
                            sample_config(), None, replay_index=index)
    dataset.refresh_replay()
    loaded, = dataset.onpolicy
    assert loaded.t[0] == states.t[0]
    assert loaded.t.dtype == np.float64
    assert set(worker_fibers._cache) == {0}  # Only the referenced fiber is decoded.
    pickle.loads(pickle.dumps(loaded)).validate_fibers(worker_fibers)


@pytest.mark.parametrize('arc', [-1., np.nan, np.inf, np.nextafter(40., np.inf)])
def test_afv_replay_still_rejects_invalid_geometry_arcs(rounded_catalog, arc):
    states = endpoint_states(rounded_catalog)
    states.t[0] = arc
    with pytest.raises(ValueError, match='arc positions outside controlled spans'):
        states.validate_fibers(rounded_catalog)
