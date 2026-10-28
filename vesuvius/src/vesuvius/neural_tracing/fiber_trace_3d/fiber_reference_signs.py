"""Fit new whole-fiber signs to a frozen, spatial signed-normal reference.

Independent binary sign fits have an exact solution: sign(sum of dot products).
No fitted windings, radial prior, reference updates, or silent unsupported signs.
"""
import json
from pathlib import Path
import sqlite3
import numpy as np
from .fiber_normal_convention import inward_solution


def validate_reference(config, target_afv):
    """Do not transfer a signed gauge from a different acquisition or AFV."""
    def metadata(path):
        with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
            return {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    reference=metadata(config['afv']); target=metadata(target_afv); database=metadata(config['database'])
    if reference['frame'] != target['frame']:
        raise ValueError('Reference normal field coordinate frame differs')
    if database['provenance']['afv_uuid'] != reference['uuid']:
        raise ValueError('Frozen normals belong to another reference AFV')
    if database['schema_version'] != 2 or not database['complete']:
        raise ValueError('A complete locally oriented normal database is required')


def fit_sign(dots):
    dots = np.asarray(dots, float)
    if not len(dots): return 0, 0., 0.
    total = float(dots.sum()); weight = float(np.abs(dots).sum())
    sign = int(np.sign(total))
    return sign, abs(total) / max(weight, 1e-30), float(np.mean(dots * sign < 0))
