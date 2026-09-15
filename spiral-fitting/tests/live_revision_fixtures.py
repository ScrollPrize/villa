"""Real-scroll session setup shared by the opt-in test and client driver."""

import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig, parse_scroll_spec
from input_publication import fingerprint
from spiral_runtime import create_session
from tifxyz import load_tifxyz


def make_real_revision_session(tmp_path, influence=False):
    source = Path(os.environ['SPIRAL_REVISION_LIVE_DATASET'])
    patch_name = os.environ['SPIRAL_REVISION_PATCH']
    source_patch = source / 'verified_patches' / patch_name
    source_digest = fingerprint(source_patch)
    dataset = tmp_path / 'dataset'
    baseline = dataset / 'verified_patches' / 'baseline'
    shutil.copytree(source_patch, baseline)
    shutil.copyfile(source / 'umbilicus.json', dataset / 'umbilicus.json')
    metadata = json.loads((baseline / 'meta.json').read_text())
    metadata['spiral_patch_erode_cells'] = 0
    (baseline / 'meta.json').write_text(json.dumps(metadata))
    replacement = tmp_path / 'replacement'
    shutil.copytree(baseline, replacement)
    with Image.open(replacement / 'x.tif') as raster:
        values = np.array(raster)
    values[values != -1] += 2
    Image.fromarray(values).save(replacement / 'x.tif')
    patch = load_tifxyz(str(baseline))
    zs = patch.zyxs[..., 0][patch.valid_vertex_mask]
    spec = json.loads((source / 'spiral-scroll.json').read_text())
    config = {
        'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
        'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0,
        'loss_weight_shell_patch_radius': 0,
        'model_flow_voxel_resolution': 64,
        'sample_count_patches_per_step': 8, 'sample_count_patches_per_step_for_dt': 8,
        'sample_count_points_per_patch': 32, 'sample_count_regularisation_points': 64,
        'sample_count_dense_spacing_pairs': 64,
        'sample_count_dense_spacing_density_extra_pairs': 64,
        'sample_count_shell_samples': 64,
        'sample_count_minimum_spacing_independent_samples': 64,
        'output_save_png_visualizations': False, 'influence_enabled': influence,
    }
    paths = SpiralInputPaths(dataset_root=str(dataset),
                             umbilicus=str(dataset / 'umbilicus.json'),
                             verified_patches=str(baseline.parent),
                             output_directory=str(tmp_path / 'output'),
                             cache_directory=str(tmp_path / 'cache'))
    session = create_session(paths, SpiralRunConfig(
        z_begin=int(zs.min()) - 1, z_end=int(zs.max()) + 2, config=config),
        SpiralPreviewConfig(), parse_scroll_spec(spec, dataset))
    return source_patch, source_digest, dataset, baseline, replacement, patch, config, session
