"""A published preview generation on disk, as lasagna_publish leaves it: a
schema-3 manifest, the flattened tifxyz with its meta.json, and the float32
winding map. What a commit reads, and what VC3D's preview loader checks.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lasagna_publish import flatten_profile

MODEL_STATE = 'a' * 64
BASE_SHAPE = [64, 64, 64]
STEP = flatten_profile()['output_step_vx']


def grid(rows=5, columns=7):
    """A flattened grid of windings 4 (columns 0-2) and 5 (columns 3-6) with a
    NaN vertex and a -1 vertex, both invalid and in no winding; winding 4
    still spans row 0 through column 1."""
    r, c = np.mgrid[:rows, :columns].astype(np.float32)
    xyz = np.stack([10.0 + c * 2.5, 20.0 + r * 1.5, 30.0 + r + c], axis=-1)
    xyz[0, 0] = np.nan
    xyz[1, 2] = -1.0
    windings = np.where(c < 3, 4.0, 5.0).astype(np.float32)
    windings[0, 0] = windings[1, 2] = -1.0
    return xyz, windings


def winding_bounds(windings):
    """lasagna_publish's winding_bounds of a winding map."""
    bounds = []
    for winding in sorted(int(w) for w in np.unique(windings) if w >= 0):
        rows, columns = np.nonzero(windings == winding)
        bounds.append({'winding': winding, 'row_begin': int(rows.min()), 'row_end': int(rows.max()) + 1,
                       'column_begin': int(columns.min()), 'column_end': int(columns.max()) + 1})
    return bounds


def publish(root, *, input_digest, input_position=None, xyz=None, windings=None,
            surface_meta=None, run_diff=False, **manifest):
    """Write one published generation under ``root``; returns ``root``.
    ``manifest`` entries replace the manifest's (None removes one)."""
    root = Path(root)
    surface_id = 'preview-lasagna.tifxyz'
    surface = root / surface_id
    surface.mkdir(parents=True)
    if xyz is None:
        xyz, windings = grid()
    for axis, name in enumerate(('x.tif', 'y.tif', 'z.tif')):
        Image.fromarray(np.ascontiguousarray(xyz[..., axis]), mode='F').save(surface / name)
    Image.fromarray(windings, mode='F').save(root / 'winding-ids.tif')
    bounds = winding_bounds(windings)
    ids = [item['winding'] for item in bounds]
    shape = list(xyz.shape[:2])
    meta = {'scale': [1.0 / STEP, 1.0 / STEP], 'export_coordinate_scale': 1.0,
            'base_shape_zyx': BASE_SHAPE, 'winding_id_map': 'winding-ids.tif',
            'winding_bounds': bounds, 'component_winding_ids': ids, 'grid_shape': shape,
            'output_step_vx': STEP}
    meta.update(surface_meta or {})
    meta = {key: value for key, value in meta.items() if value is not None}
    (surface / 'meta.json').write_text(json.dumps(meta))
    document = {
        'schema_version': 3, 'kind': 'spiral_combined_preview', 'surface_id': surface_id,
        'surface_path': str(surface), 'output_step_vx': STEP,
        'flatten_profile': flatten_profile(), 'grid_shape': shape,
        'winding_ids': ids, 'winding_id_map': 'winding-ids.tif',
        'winding_id_dtype': 'float32_integer', 'winding_bounds': bounds,
        'loss_maps': [], 'source_fit_iteration': 120,
        'model_state_sha256': MODEL_STATE, 'input_digest': input_digest,
        'input_position': input_position, 'base_shape_zyx': BASE_SHAPE}
    if run_diff:
        Image.fromarray(np.zeros((*xyz.shape[:2], 4), np.uint8), mode='RGBA').save(
            root / 'run-diff.png')
        document['run_diff'] = {'path': 'run-diff.png', 'changed_source_pixels': 3,
                                'supported_pixels': 3}
    document.update(manifest)
    document = {key: value for key, value in document.items() if value is not None}
    (root / 'manifest.json').write_text(json.dumps(document))
    return root
