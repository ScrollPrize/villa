"""Rebuild-equivalence driver: is a model-stage rebuild the session a full
build would have produced?

Builds a resident-style FitContext from the golden run spec, rebuilds its
model stage with one MODEL_STAGE_KEYS setting changed, and records the
structure of the checkpoint that session would write. Then builds a second
context from scratch with that same value and records its structure too. The
assertion half (tests/test_rebuild_equivalence.py) compares the two.

    python tests/rebuild_equivalence_driver.py <spec.json> <result.json> <out_dir>

Structural equality is exact and is what this measures: the checkpoint
payload's key set, every model tensor's shape and dtype, the optimiser and
scheduler structure, and the durable cfg. Parameter *values* are deliberately
not compared — model construction draws from the global torch RNG, and a
rebuild starts from wherever the session's training left that stream, so the
two models are structurally identical and numerically unrelated by
construction.

The setting changed is model_num_flow_stages, which is on the allowlist and
changes the model's parameter set, so a rebuild that quietly kept the old
model would fail rather than pass vacuously.
"""

import gc
import json
import os
import sys

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)
sys.path.insert(0, os.path.join(SPIRAL_DIR, 'tests'))

from golden_run_driver import (  # noqa: E402
    REFERENCE_Z_RANGE_NUM_SLICES, Z_RANGE_SCALED_COUNT_KEYS,
    _checkpoint_structure)

REBUILD_KEY = 'model_num_flow_stages'
REBUILD_VALUE = 2


class _ResidentDriver:
    """The little of an interactive driver a FitContext actually reads."""

    def __init__(self, requested_config):
        self.requested_config = dict(requested_config)


def _resolved_config(spec, extra=None):
    from config import Config
    from spiral_helpers import SAMPLING_COUNT_FLOORS, scale_counts_for_z_range

    config = Config().as_dict()
    config.update(spec.get('config_overrides', {}))
    config.update(extra or {})
    config['z_begin'] = spec['z_begin']
    config['z_end'] = spec['z_end']
    config['optimizer_num_training_steps'] = spec['iterations']
    scale_counts_for_z_range(
        config, config['z_begin'], config['z_end'],
        REFERENCE_Z_RANGE_NUM_SLICES, Z_RANGE_SCALED_COUNT_KEYS,
        floors=SAMPLING_COUNT_FLOORS,
    )
    return config


def _build(spec, out_dir, config, cache_dir):
    """One resident-style session, built exactly as spiral_runtime builds it."""
    import fit_spiral as fs
    from config import FitConfig
    from fit_session import conventional_input_paths, load_scroll_spec

    scroll = load_scroll_spec(spec['dataset_path'])
    context = fs.FitContext(
        FitConfig(config),
        scroll=scroll,
        paths=conventional_input_paths(spec['dataset_path'], scroll),
        interactive_driver=_ResidentDriver(config),
        out_base_dir=out_dir,
        cache_dir=cache_dir)
    context.load_host_inputs()
    context.resolve_output_path()
    context.build_device_state()
    # The runtime drops the per-track input arrays here, which is exactly the
    # state a later model-stage rebuild has to cope with.
    context.release_setup_only_tracks()
    return context


def _structure(context):
    return {
        **_checkpoint_structure(context._checkpoint_payload(0)),
        'cfg': dict(context._checkpoint_payload(0)['cfg']),
    }


def run(spec_path, result_path, out_dir):
    with open(spec_path) as spec_file:
        spec = json.load(spec_file)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = spec.get('cache_path') or '../cache'

    import torch

    rebuilt_config = _resolved_config(spec, {REBUILD_KEY: REBUILD_VALUE})

    context = _build(spec, out_dir, _resolved_config(spec), cache_dir)
    before = _structure(context)
    context.config.update({REBUILD_KEY: REBUILD_VALUE})
    context.rebuild_model_state()
    rebuilt = _structure(context)
    context.close()
    del context
    gc.collect()
    torch.cuda.empty_cache()

    fresh_context = _build(spec, out_dir, rebuilt_config, cache_dir)
    fresh = _structure(fresh_context)
    fresh_context.close()

    result = {
        'spec': spec,
        'rebuild_key': REBUILD_KEY,
        'rebuild_value': REBUILD_VALUE,
        'before': before,
        'rebuilt': rebuilt,
        'fresh': fresh,
    }
    with open(result_path, 'w') as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
    print(f'rebuild-equivalence driver: wrote {result_path}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
