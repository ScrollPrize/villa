"""Opt-in end-to-end validation on an existing fiber store and normal inference."""
import json
import os
from pathlib import Path
import sqlite3

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.constraint_extraction import run_native_constraints
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_catalog, open_store


def test_existing_store_constraint_export(tmp_path):
    names = ('VC_CONSTRAINT_TEST_FIBERS', 'VC_CONSTRAINT_TEST_NORMALS',
             'VC_CONSTRAINT_TEST_PROVENANCE', 'VC_CONSTRAINT_TEST_BINARY')
    if not all(os.environ.get(name) for name in names):
        pytest.skip('set VC_CONSTRAINT_TEST_FIBERS/NORMALS/PROVENANCE/BINARY for real-data export')
    fibers, normals, provenance, binary = (Path(os.environ[name]).resolve() for name in names)
    # The inference provenance names the scan; the exporter requires the fibers be bound to that volume.
    source = json.loads(provenance.read_text())['source']
    catalog = fiber_catalog(fibers, source['volume_id'])
    assert catalog and catalog[0]['bbox'] is not None
    bbox = catalog[0]['bbox']
    row = [(bbox[k]+bbox[k+3])/2 for k in range(3)]
    config = dict(fibers=str(fibers), coordinate_space=f"{source['sample_id']}/{source['volume_id']}",
                  normal_manifest=str(normals), normal_provenance=str(provenance),
                  remote_cache=str(tmp_path / 'cache'), output=str(tmp_path / 'export.json'),
                  database=str(tmp_path / 'constraints.sqlite'), threads=4, repetitions=1,
                  minimum_xyz=[value-128 for value in row], maximum_xyz=[value+128 for value in row])
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    run_native_constraints(path, dict(native_binary=str(binary)))
    result = json.loads(Path(config['output']).read_text())
    assert result['selected_fibers'] > 0 and result['points'] >= 2
    assert result['normal_alignment'] is False
    with open_store(fibers) as store:
        assert result['fibers_store_uuid'] == store.uuid
    assert catalog[0]['id'] in result['selected_ids'] and result['selected_ids'] == sorted(result['selected_ids'])
    with sqlite3.connect(Path(config['database']).as_uri() + '?mode=ro', uri=True) as database:
        metadata = {key: json.loads(value) for key, value in database.execute('SELECT key,value FROM metadata')}
        assert metadata['complete'] is True
        ids = [r[0] for r in database.execute('SELECT id FROM fibers ORDER BY source_index')]
        assert ids == result['selected_ids']
    print('REAL_STORE_EXPORT', json.dumps({key: result[key] for key in
          ('selected_fibers', 'points', 'arc_base', 'query_seconds', 'load_seconds')}))
