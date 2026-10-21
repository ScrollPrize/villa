"""Opt-in end-to-end validation on an existing AFV and normal inference."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.constraint_extraction import run_native_constraints


def test_existing_afv_constraint_export(tmp_path):
    names = ('VC_CONSTRAINT_TEST_AFV', 'VC_CONSTRAINT_TEST_NORMALS',
             'VC_CONSTRAINT_TEST_PROVENANCE', 'VC_CONSTRAINT_TEST_BINARY')
    if not all(os.environ.get(name) for name in names):
        pytest.skip('set VC_CONSTRAINT_TEST_AFV/NORMALS/PROVENANCE/BINARY for real-data export')
    afv, normals, provenance, binary = (Path(os.environ[name]).resolve() for name in names)
    with sqlite3.connect(afv.as_uri() + '?mode=ro', uri=True) as source:
        row = source.execute('SELECT (min_x+max_x)/2, (min_y+max_y)/2, '
                             '(min_z+max_z)/2 FROM fibers ORDER BY id LIMIT 1').fetchone()
        frame = json.loads(source.execute('SELECT value FROM metadata WHERE key=?', ('frame',)).fetchone()[0])
    assert row is not None
    config = dict(afv=str(afv), coordinate_space=frame['vc_open_data_coordinate_space'],
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
    with sqlite3.connect(Path(config['database']).as_uri() + '?mode=ro', uri=True) as database:
        metadata = {key: json.loads(value) for key, value in database.execute('SELECT key,value FROM metadata')}
        assert metadata['complete'] is True
        assert database.execute('SELECT count(*) FROM fibers').fetchone()[0] == result['selected_fibers']
    print('REAL_AFV_EXPORT', json.dumps({key: result[key] for key in
          ('selected_fibers', 'points', 'arc_base', 'query_seconds', 'load_seconds')}))
