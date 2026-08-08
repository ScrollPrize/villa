"""Golden-run driver: one short headless fit, executed the same way the
fit_spiral CLI drives it, with dataset locations supplied by a JSON spec
instead of edits to fit_spiral module globals.

This is the measurement half of the golden-run characterization harness
(REFACTOR_PLAN.md, PR 1 commit 0). It is meant to run as a subprocess so
each run gets a fresh interpreter, CUDA context, and RNG state:

    python tests/golden_run_driver.py <spec.json> <result.json> <out_dir>

It records, into <result.json>:
  - the fully resolved (z-range-scaled) config actually used;
  - every wandb.log payload from the training loop (the per-loss-family
    traces emitted every 200 iterations), tensors converted to floats;
  - the structure of the final checkpoint: top-level keys, model state
    key -> (shape, dtype), optimiser/scheduler structure, durable metadata;
  - hashes of the host RNG states stored in the checkpoint (numpy / torch
    CPU), which are bit-stable iff host-side RNG consumption order is
    preserved;
  - the parsed satisfied_fitted.json metrics.

The assertion half lives in tests/test_golden_run.py; tolerance bands are
calibrated by tests/record_golden_run.py from repeated runs of unmodified
code.
"""

import glob
import hashlib
import json
import os
import pickle
import sys

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)


# Keys the CLI scales with the z-range; must match fit_spiral.__main__.
Z_RANGE_SCALED_COUNT_KEYS = (
    'sample_count_patches_per_step',
    'sample_count_patches_per_step_for_dt',
    'sample_count_unverified_patches_per_step',
    'sample_count_unverified_patches_per_step_for_dt',
    'sample_count_relative_winding_pcls',
    'sample_count_absolute_winding_pcls',
    'sample_count_unattached_pcls_per_step',
    'sample_count_tracks_per_step',
    'sample_count_dense_normal_points',
    'sample_count_dense_spacing_pairs',
    'sample_count_dense_spacing_density_extra_pairs',
    'sample_count_dense_attachment_points',
    'sample_count_regularisation_points',
    'sample_count_shell_samples',
)
REFERENCE_Z_RANGE_NUM_SLICES = 9500  # mirrors spiral_helpers.REFERENCE_Z_RANGE_NUM_SLICES


def derive_module_globals(spec):
    """Dataset-derived module globals, mirroring fit_spiral's conventions."""
    dataset = spec['dataset_path']
    return {
        'dataset_path': dataset,
        'scroll_zarr_path': None,
        # The golden run intentionally runs without the Lasagna input stores
        # (dense losses are zero-weighted in the spec's config overrides).
        'normal_nx_zarr_path': None,
        'normal_ny_zarr_path': None,
        'grad_mag_zarr_path': None,
        'normal_zarr_group': '4',
        'surf_sdt_zarr_path': f'{dataset}/lasagna_inputs/las_008_surf_sdt.ome.zarr',
        'surf_sdt_zarr_group': '1',
        'pcl_json_paths': [
            f'{dataset}/abs_winding.json',
            f'{dataset}/patch-overlap-pcls.json',
            f'{dataset}/relative_windings.json',
            f'{dataset}/same_windings.json',
            f'{dataset}/drawn_control_points.json',
        ],
        'pcl_input_specs': None,
        'fibers_path': f'{dataset}/fibers',
        'verified_patches_path': f'{dataset}/verified_patches',
        'unverified_patches_path': None,
        'shell_path': f'{dataset}/outer_shell',
        'tracks_dbm_path': f'{dataset}/tracks/2um_ds2_ps256_surf_v2.dbm',
        'spiral_outward_sense': 'CW',
        'scroll_name': 's1',
        'z_begin': spec['z_begin'],
        'z_end': spec['z_end'],
        'voxel_size_um': 9.6,
        'lasagna_scale': 4,
        'lasagna_storage_backend': 'sparse_cuda',
    }


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _rng_state_hashes(checkpoint):
    import numpy as np
    import torch

    hashes = {}
    numpy_state = checkpoint.get('numpy_rng_state')
    if numpy_state is not None:
        hashes['numpy'] = _hash_bytes(pickle.dumps((
            numpy_state[0],
            np.asarray(numpy_state[1]).tobytes(),
            *numpy_state[2:],
        )))
    cpu_state = checkpoint.get('torch_cpu_rng_state')
    if cpu_state is not None:
        hashes['torch_cpu'] = _hash_bytes(bytes(cpu_state.numpy().tobytes()))
    cuda_states = checkpoint.get('torch_cuda_rng_states')
    if cuda_states is not None:
        hashes['torch_cuda'] = [
            _hash_bytes(bytes(state.numpy().tobytes())) for state in cuda_states]
    return hashes


def _tensor_structure(state_dict):
    import torch

    structure = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            structure[key] = [list(value.shape), str(value.dtype)]
        else:
            structure[key] = ['non-tensor', type(value).__name__]
    return structure


def _optimiser_structure(optimiser_state):
    import torch

    per_param_state = {}
    for index, state in sorted(optimiser_state.get('state', {}).items()):
        per_param_state[str(index)] = {
            key: ([list(value.shape), str(value.dtype)]
                  if torch.is_tensor(value) else value)
            for key, value in sorted(state.items())
        }
    return {
        'state': per_param_state,
        'param_groups': optimiser_state.get('param_groups'),
    }


def _checkpoint_structure(checkpoint):
    return {
        'top_level_keys': sorted(checkpoint.keys()),
        'schema_version': checkpoint.get('schema_version'),
        'completed_iterations': checkpoint.get('completed_iterations'),
        'z_begin': checkpoint.get('z_begin'),
        'z_end': checkpoint.get('z_end'),
        'spiral_outward_sense': checkpoint.get('spiral_outward_sense'),
        'lasagna_scale': checkpoint.get('lasagna_scale'),
        'lasagna_group': checkpoint.get('lasagna_group'),
        'surf_sdt_fingerprint': checkpoint.get('surf_sdt_fingerprint'),
        'input_manifest': checkpoint.get('input_manifest'),
        'preview_first_winding': checkpoint.get('preview_first_winding'),
        'cfg_keys': sorted(checkpoint.get('cfg', {}).keys()),
        'requested_config_keys': sorted(checkpoint.get('requested_config', {}).keys()),
        'resolved_config_keys': sorted(checkpoint.get('resolved_config', {}).keys()),
        'model_state': _tensor_structure(checkpoint['spiral_and_transform']),
        'optimiser': _optimiser_structure(checkpoint['optimiser']),
        'scheduler': checkpoint.get('scheduler'),
    }


def _canonicalize_satisfied(satisfied):
    """Sort entry lists by identity: patch/PCL load order is not
    deterministic run-to-run, only the set of entries is."""
    canonical = dict(satisfied)
    for key, identity in (('patches', 'id'), ('unverified_patches', 'id'),
                          ('pcls', 'name')):
        entries = canonical.get(key)
        if isinstance(entries, list):
            canonical[key] = sorted(
                entries, key=lambda entry: str(entry.get(identity)))
    return canonical


def _satisfied_aggregates(satisfied):
    """Aggregate satisfaction metrics: per-entry values are noisy (patch
    areas are quantized in coarse cells), but sums over hundreds of entries
    carry a tight regression signal."""
    aggregates = {}
    for key, satisfied_field, total_field in (
            ('patches', 'satisfied_area', 'total_area'),
            ('unverified_patches', 'satisfied_area', 'total_area'),
            ('pcls', 'satisfied_points', 'total_points')):
        entries = satisfied.get(key) or []
        total_satisfied = sum(entry[satisfied_field] for entry in entries)
        total = sum(entry[total_field] for entry in entries)
        aggregates[key] = {
            'num_entries': len(entries),
            'sum_satisfied': total_satisfied,
            'sum_total': total,
            'overall_fraction': (total_satisfied / total) if total else None,
        }
    return aggregates


def run(spec_path, result_path, out_dir):
    with open(spec_path) as spec_file:
        spec = json.load(spec_file)

    os.makedirs(out_dir, exist_ok=True)
    # Mesh export is slow and exercises no fitting behavior; keep the golden
    # run focused on the training loop and checkpoint.
    os.environ.setdefault('FIT_SPIRAL_SKIP_SAVE_MESH', '1')
    if spec.get('cache_path'):
        os.environ['FIT_SPIRAL_CACHE_DIR'] = spec['cache_path']

    import torch
    import wandb

    import fit_spiral as fs
    from config import Config, FitConfig
    from spiral_helpers import SAMPLING_COUNT_FLOORS, scale_counts_for_z_range

    module_globals = derive_module_globals(spec)
    for name, value in module_globals.items():
        assert hasattr(fs, name), f'fit_spiral no longer defines {name}'
        setattr(fs, name, value)
    dataset = spec['dataset_path']
    fs.umbilicus_z_to_yx = lambda: fs.json_umbilicus_z_to_yx(
        f'{dataset}/umbilicus.json', coordinate_scale=1.0)

    # Mirror fit_spiral.__main__ (single process, no DDP): resolve config,
    # scale per-step counts for the z-range, then hand an explicit FitConfig
    # to main(). wandb stays exactly as the CLI uses it: an optional
    # (disabled) logging sink, never a source of configuration.
    config = Config().as_dict()
    config.update(spec.get('config_overrides', {}))
    config['optimizer_num_training_steps'] = spec['iterations']
    scale_counts_for_z_range(
        config, fs.z_begin, fs.z_end,
        REFERENCE_Z_RANGE_NUM_SLICES, Z_RANGE_SCALED_COUNT_KEYS,
        floors=SAMPLING_COUNT_FLOORS,
    )
    fit_config = FitConfig(config)

    wandb.init(project='scrolls', config=config, mode='disabled')

    # The training loop reports per-loss-family values through wandb.log every
    # 200 iterations; capture those payloads instead of parsing stdout.
    logged = []

    def capture_log(payload, *args, **kwargs):
        logged.append({
            key: (float(value) if hasattr(value, 'item') or isinstance(value, (int, float, bool))
                  else repr(value))
            for key, value in payload.items()
        })

    wandb.log = capture_log

    fs.main(
        fit_config,
        progress=None,
        out_base_dir=out_dir,
        run_name=wandb.run.name if wandb.run is not None else None,
    )

    checkpoints = glob.glob(f'{out_dir}/*/checkpoint_fitted.ckpt')
    assert len(checkpoints) == 1, f'expected one final checkpoint, found {checkpoints}'
    checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)

    satisfied_files = glob.glob(f'{out_dir}/*/satisfied_fitted.json')
    assert len(satisfied_files) == 1, f'expected one satisfied_fitted.json, found {satisfied_files}'
    with open(satisfied_files[0]) as satisfied_file:
        satisfied = _canonicalize_satisfied(json.load(satisfied_file))

    result = {
        'spec': spec,
        'resolved_config': dict(fit_config),
        'metrics': [
            {'iteration': index * 200, 'values': values}
            for index, values in enumerate(logged)
        ],
        'checkpoint_structure': _checkpoint_structure(checkpoint),
        'rng_state_hashes': _rng_state_hashes(checkpoint),
        'satisfied': satisfied,
        'satisfied_aggregates': _satisfied_aggregates(satisfied),
    }
    with open(result_path, 'w') as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
    print(f'golden-run driver: wrote {result_path}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
