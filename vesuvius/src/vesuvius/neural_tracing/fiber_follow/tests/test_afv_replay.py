"""AFV catalog rounding must not invalidate replay at geometry endpoints; length-weighted fiber draws."""
import json
import pickle
import sqlite3

import numpy as np
import pytest

from test_data_and_scoring import make_states, sample_config
from test_mixed_datasets import afv_fixture
from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.data.data import FollowDataset, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


@pytest.fixture
def rounded_catalog(tmp_path):
    path = tmp_path/'rounded.afv'
    afv_fixture(path)
    # Summing densified segments can round the catalog length below the geometry's own length.
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE fibers SET length=? WHERE id=1', (float(np.nextafter(80., -np.inf)),))
    return AFVFibers(path, grid_scale=2.)


def endpoint_states(fibers):
    states = make_states(fibers[0], z=0)
    states.manifest = fiber_manifest(fibers)
    states.provenance = dict(states.provenance, volume={'grid_scale': fibers.grid_scale})
    states.t = np.array([fibers[0].s[-1]])
    return states


def test_afv_endpoint_replay_refresh_and_worker_roundtrip(tmp_path, rounded_catalog):
    fibers = rounded_catalog
    spec = FiberVolumeSpec('unused', grid_scale=2.)
    # Sampling weights are catalog length**power and decode no geometry.
    lengths = fibers.lengths
    for power in (1., 3.):
        dataset = FollowDataset(fibers, spec, sample_config(), None, length_power=power)
        np.testing.assert_allclose(dataset.weights, lengths**power/np.sum(lengths**power))
    assert not fibers._cache
    for power in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='length power'):
            FollowDataset(fibers, spec, sample_config(), None, length_power=power)
    assert fibers[0].length == fibers[0].s[-1] == 40. and fibers[0].length != fibers.lengths[0]
    states = endpoint_states(fibers)
    archive = tmp_path/'endpoint.npz'
    states.save(archive)
    index = tmp_path/'replay.json'
    index.write_text(json.dumps([str(archive)]))
    # A worker starts with a cold geometry cache and discovers new replay later.
    worker_fibers = pickle.loads(pickle.dumps(AFVFibers(fibers.path, grid_scale=2.)))
    dataset = FollowDataset(worker_fibers, spec, sample_config(), None, replay_index=index)
    dataset.refresh_replay()
    loaded, = dataset.onpolicy
    assert loaded.t[0] == states.t[0]
    assert loaded.t.dtype == np.float64
    assert set(worker_fibers._cache) == {0}  # Only the referenced fiber is decoded.
    pickle.loads(pickle.dumps(loaded)).validate_fibers(worker_fibers)


def test_afv_replay_still_rejects_invalid_geometry_arcs(rounded_catalog):
    for arc in (-1., np.nan, np.nextafter(40., np.inf)):
        states = endpoint_states(rounded_catalog)
        states.t[0] = arc
        with pytest.raises(ValueError, match='arc positions outside controlled spans'):
            states.validate_fibers(rounded_catalog)
