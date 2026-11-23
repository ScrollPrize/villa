"""Fit new whole-fiber signs to a frozen, spatial signed-normal reference.

Independent binary sign fits have an exact solution: sign(sum of dot products).
No fitted windings, radial prior, reference updates, or silent unsupported signs.
"""
import json
from pathlib import Path
import numpy as np
from .fiber_normal_convention import inward_solution


def validate_reference(config, target_fibers):
    """Bind the frozen gauge to the same volume and unchanged fiber versions."""
    from vc_object_store import CollectionReader
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume, fibers_digest
    if fiber_volume(config['fibers']) != fiber_volume(target_fibers):
        raise ValueError('Reference normal field coordinate frame differs')
    if fibers_digest(config['database']) != fibers_digest(config['fibers']):
        raise ValueError('Frozen normals belong to another fiber snapshot')
    with CollectionReader(config['database']) as store:
        runs=store.extension_rows('ext_fiber_graph_solve_runs')
        if not any(row[1]=='normal_orientation' for row in runs):
            raise ValueError('A locally oriented normal graph is required')


def fit_sign(dots):
    dots = np.asarray(dots, float)
    if not len(dots): return 0, 0., 0.
    total = float(dots.sum()); weight = float(np.abs(dots).sum())
    sign = int(np.sign(total))
    return sign, abs(total) / max(weight, 1e-30), float(np.mean(dots * sign < 0))
