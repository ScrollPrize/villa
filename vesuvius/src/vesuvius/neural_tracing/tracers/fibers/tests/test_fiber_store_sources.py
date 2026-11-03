"""Fiber-store dataset sources (data/datasets.open_fiber_source, data/fiber_bank.py) and the traced-fiber export
(tracing/volume_store.py)."""
import json
import uuid

import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines, write_fibers

CT = 's3://vesuvius-challenge-open-data/PHercTest/volumes/20260101000000-2.400um-0.2m-78keV-masked.zarr'


def test_fiber_source_holds_out_by_uuid_and_its_bank_draws_other_store_fibers(tmp_path):
    from vesuvius.neural_tracing.tracers.fibers.data.datasets import open_fiber_source
    from vesuvius.neural_tracing.tracers.fibers.data.fiber_bank import FiberBank
    x = np.arange(0., 600., 1.)
    write_fibers(tmp_path/'f.vcobj', [(dict(id=f'00000000-0000-4000-8000-00000000000{k}', name=f'f{k}', family='H'),
                                       np.c_[x, np.full_like(x, 100.+4*k), 100.+0*x]) for k in range(4)],
                 volume='20260101000000')
    source = dict(name='test', kind='fibers', path=str(tmp_path/'f.vcobj'), coordinate_space='PHercTest/20260101000000',
                  ct=CT, grid_scale=2., validation=dict(strategy='fiber_hash', seed=1, count=1))
    train, validation, spec = open_fiber_source(source, None)
    assert len(train) == 3 and len(validation) == 1 and not train.ids & validation.ids
    assert spec.grid_scale == 2.
    with pytest.raises(ValueError, match='not the open-data volume'):
        open_fiber_source(dict(source, coordinate_space='PHercTest/20260101000001'), None)
    bank = FiberBank(train)
    assert bank.provenance()['kind'] == 'fibers'
    rng = np.random.default_rng(0)
    drawn = [d for d in (bank.draw_path(rng, min_length=32.) for _ in range(4)) if d is not None]
    assert drawn and all(bank.clear_of_target(fi, curve).all() and lo < hi for fi, curve, (lo, hi) in drawn)
    own = train.catalog[0]['id']
    records = bank.spatial_records(0, train[0].points[:50], radius=5.)  # the next two fibers (2 and 4 away)
    assert records and all(r['shard'].startswith('fiber:') and r['shard'] != f'fiber:{own}' for r in records)


def test_volume_export_gives_each_traced_fiber_a_stable_uuid_and_links_by_uuid(tmp_path):
    from vesuvius.neural_tracing.tracers.fibers.tracing.volume_store import export_store, fiber_uuid
    namespace = str(uuid.uuid4())
    (tmp_path/'run.json').write_text(json.dumps(dict(grid_scale=2., fiber_namespace=namespace, ct_zarr=CT)))
    (tmp_path/'tiles').mkdir()
    points = np.r_[np.c_[np.arange(10.), np.zeros(10), np.zeros(10)], np.c_[np.arange(10.), np.ones(10), np.zeros(10)]]
    np.savez(tmp_path/'tiles'/'z00000_y00000_x00000.npz', points=points.astype(np.float32), offsets=[0, 10, 20],
             id=[3, 7], seed=np.zeros((2, 3)), family=['H', 'V'], plane=['x', 'z'],
             reasons=[['box', 'owned'], ['box', 'box']], links=[[-1, 7], [3, -1]])
    first = export_store(tmp_path)
    assert first['fibers'] == 2 and first['volume'] == '20260101000000'
    export_store(tmp_path)  # a re-export replaces the store and keeps the UUIDs
    rows = {row['name']: (row, xyz) for row, xyz in read_polylines(tmp_path/'fibers.vcobj')}
    a, b = fiber_uuid(namespace, 3), fiber_uuid(namespace, 7)
    (row, xyz), (other, _) = rows['fiber_follow 00000003'], rows['fiber_follow 00000007']
    assert (row['id'], row['family'], other['id']) == (a, 'H', b)
    assert np.allclose(xyz, points[:10]*2.)
    assert row['meta']['fiber_follow']['links'] == [None, b] and other['meta']['fiber_follow']['links'] == [a, None]
    assert row['meta']['fiber_follow']['run'] == namespace and row['meta']['fiber_follow']['run_id'] == 3
