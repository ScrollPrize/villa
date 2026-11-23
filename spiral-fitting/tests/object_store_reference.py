"""Synthetic fit inputs and canonical summaries of the patches they produce.

``PATCHES`` describes a small scroll: four patches on an Archimedean spiral.
``constraint_summary`` reduces a loaded FitContext's patches to
JSON-comparable values. tests/data/object_store_reference.json holds them as
produced by the former file loader, captured before that path was removed;
the object-store adapter must reproduce them. The reference cannot be
regenerated: change it only for a deliberate change to the patches.
"""
import json
import math
import os
import sys

import numpy as np

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)

REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data',
    'object_store_reference.json')

DR = 12.0
CENTRE_YX = (150.0, 150.0)
BASE_SHAPE_ZYX = (64, 300, 300)
GRID_STEP = 4.0
SEED = 1234


def spiral_zyx(theta, winding, z):
    """Volume ZYX of the spiral sheet ``winding`` at ``theta`` and ``z``."""
    radius = DR * (winding + theta / (2 * math.pi))
    return [float(z), CENTRE_YX[0] + math.sin(theta) * radius,
            CENTRE_YX[1] + math.cos(theta) * radius]


def _patch(name, uuid, winding, theta_begin, theta_end):
    zs = np.arange(4.0, 60.0 + 1e-6, GRID_STEP)
    radius = DR * (winding + (theta_begin + theta_end) / (4 * math.pi))
    thetas = np.arange(theta_begin, theta_end + 1e-9, GRID_STEP / radius)
    grid = np.asarray([[spiral_zyx(theta, winding, z) for theta in thetas]
                       for z in zs], dtype=np.float32)
    return {'name': name, 'uuid': uuid, 'winding': winding,
            'thetas': thetas, 'zs': zs, 'zyxs': grid,
            'scale': [1 / GRID_STEP, 1 / GRID_STEP]}


PATCHES = [
    _patch('patch_a', '00000000-0000-4000-8000-0000000000a1', 8, 0.3, 1.3),
    _patch('patch_b', '00000000-0000-4000-8000-0000000000a2', 8, 1.1, 2.1),
    _patch('patch_c', '00000000-0000-4000-8000-0000000000a3', 9, 0.5, 1.5),
    _patch('patch_d', '00000000-0000-4000-8000-0000000000a4', 10, 2.0, 3.0),
]

CONFIG_OVERRIDES = {
    'z_begin': 0,
    'z_end': 64,
    'optimizer_random_seed': SEED,
    'input_use_normals': False,
    'input_use_gradient_magnitude': False,
    'input_use_winding_inference': False,
    'input_use_outer_shell': False,
    'input_use_tracks': False,
    'input_use_fiber_directions': False,
    'loss_weight_fiber_directions': 0.0,
    'sample_count_points_per_patch': 16,
}


def fit_config():
    from config import Config
    config = Config().as_dict()
    config.update(CONFIG_OVERRIDES)
    return config


def scroll_spec():
    from fit_session import ScrollSpec
    return ScrollSpec(name='synthetic', voxel_size_um=1.0,
                      spiral_outward_sense='CW',
                      base_shape_zyx=BASE_SHAPE_ZYX)


def write_umbilicus(path):
    with open(path, 'w') as stream:
        json.dump({'control_points': [
            {'z': z, 'y': CENTRE_YX[0], 'x': CENTRE_YX[1]} for z in (0, 64)]},
            stream)


def _r(values, digits=4):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def constraint_summary(context, patch_names=None):
    """JSON-comparable runtime patches; ``patch_names`` maps runtime patch
    ids to fixture names (identity when None)."""
    name_of = (lambda pid: pid) if patch_names is None else patch_names.__getitem__
    patches = []
    for pid, patch in context.verified_patches.items():
        valid = patch.valid_vertex_mask.numpy()
        patches.append({
            'name': name_of(pid),
            'shape': list(patch.zyxs.shape),
            'valid_vertices': int(valid.sum()),
            'valid_sum': _r(patch.zyxs.numpy()[valid].sum(axis=0), 2),
            'scale': _r(patch.scale.numpy()),
            'area': round(float(patch.area), 3),
            'sampling_area': round(float(patch._sampling_area), 3),
        })
    return {'patches': patches}


def load_reference():
    with open(REFERENCE_PATH) as stream:
        return json.load(stream)
