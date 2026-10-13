"""The unmocked CPU preview/export path no longer allocates on CUDA.

``save_combined_preview`` (and through it the same ``get_spiral_yxs`` call
shape used by ``save_mesh``) builds its theta grid via ``get_spiral_yxs``
with the caller's ``dr_per_winding`` tensor and no explicit device. On a
CPU-only run that tensor is on CPU; before the device-inference fix the grid
silently went to CUDA and the preview died concatenating it with the CPU z
values (or, on a machine with CUDA, allocated on the GPU without the caller
asking). This test runs the real consumer path on CPU with a real published
umbilicus and a real ``SpiralAndTransform``, with nothing mocked.

Input: the official PHerc0139 umbilicus from the open data bucket
(``PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json``,
391 control points, sha256 8123543f1792ab307a6f6d426469710899aaa0d36a2dc3bc8160ddf7a57a7edb),
reproducing the input of upstream issue #2029. The file is checked in at
``tests/data/umbilicus.json`` (27 KB) and is required test data: the class
fails rather than skips if it is missing or altered.
"""
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import torch

import spiral_helpers
import umbilicus
from transforms import SpiralAndTransform

UMBILICUS_SHA256 = '8123543f1792ab307a6f6d426469710899aaa0d36a2dc3bc8160ddf7a57a7edb'

UMBILICUS_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'umbilicus.json')


def _build_model(z_begin, z_end):
    raw = umbilicus.json_umbilicus_z_to_yx(UMBILICUS_PATH)
    zs = np.arange(z_begin, z_end)
    umb_zyx = torch.from_numpy(
        np.concatenate([zs[:, None], raw(zs)], axis=-1).astype(np.float32))
    cfg = {
        'model_initial_dr_per_winding': 10.0,
        'model_flow_field_type': 'cartesian',
        'model_flow_voxel_resolution': 16,
        'model_linear_z_resolution': 32,
        'model_gap_expander_num_windings': 14,
        'model_gap_expander_logit_resolution': 24,
        'model_gap_expander_min_gap': 1.0,
        'model_gap_expander_softplus_bias': 4.0,
        'model_gap_expander_lr_scale': 100.0,
        'model_flow_bounds_z_margin': 0,
        'model_flow_bounds_radius': 400,
        'model_num_flow_integration_steps': 24,
        'model_flow_integration_solver': 'rk4',
        'shell_outer_winding_idx': 12,
        'output_step_size': 8,
        'output_first_winding': 10,
    }
    model = SpiralAndTransform(
        flow_integration_steps=24,
        flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([z_begin, -400, -400], dtype=torch.int64),
        flow_max_corner_zyx=torch.tensor([z_end, 400, 400], dtype=torch.int64),
        umbilicus_zyx=umb_zyx,
        config=cfg,
        spiral_outward_sense='CW',
    )
    model.eval()
    return model.get_slice_to_spiral_transform(), cfg

class CpuPreviewExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The checked-in real published umbilicus is required test data:
        # the regression is only meaningful against real data, and a silent
        # skip would hide the very bug it guards against.
        cls.umbilicus_path = UMBILICUS_PATH
        digest = hashlib.sha256(open(cls.umbilicus_path, 'rb').read()).hexdigest()
        assert digest == UMBILICUS_SHA256, (
            f'umbilicus.json sha256 mismatch: {digest}')
    def test_cpu_save_combined_preview_stays_on_cpu(self):
        z_begin, z_end = 3100, 3360
        transform, cfg = _build_model(z_begin, z_end)
        dr_per_winding = torch.tensor(10.0)  # CPU tensor, like a CPU fit
        with tempfile.TemporaryDirectory() as tmp_dir:
            # save_combined_tifxyz requires the destination itself to be new;
            # point it at a fresh child of the temporary directory.
            out_dir = os.path.join(tmp_dir, 'generation')
            with torch.inference_mode():
                manifest = spiral_helpers.save_combined_preview(
                    transform, dr_per_winding, [], [], out_dir, cfg,
                    z_begin=z_begin, z_end=z_end, voxel_size_um=2.399,
                    get_or_build_unattached_pcl_flat=lambda *_: None,
                    surface_id='surface',
                    z_direction_is_top_to_bottom=None,
                )
            surface_path = manifest['surface_path']
            self.assertTrue(os.path.isfile(os.path.join(surface_path, 'x.tif')))
            self.assertTrue(os.path.isfile(os.path.join(surface_path, 'y.tif')))
            self.assertTrue(os.path.isfile(os.path.join(surface_path, 'z.tif')))
            meta = json.load(open(os.path.join(surface_path, 'meta.json')))
        # Real published axis: the surface sits around the PHerc0139 umbilicus
        # yx at these z, i.e. tens of thousands of voxels from the origin.
        lo, hi = meta['bbox']
        self.assertGreater(lo[1], 1e4)
        self.assertGreater(lo[0], 0)
        self.assertLess(hi[2], z_end)
        self.assertGreaterEqual(lo[2], z_begin - cfg['output_step_size'])


if __name__ == '__main__':
    unittest.main()
