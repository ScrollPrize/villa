"""Fork a checkpoint with explicit path memory and paired identity training."""
import argparse
import copy
import json
import math
import io

import numpy as np

import torch

from .checkpoint_migration import fork_checkpoint
from .model import DirectConfig, build_model
from .train import checkpoint_config


def expand_memory_checkpoint(checkpoint, *, pair_rank_weight=.25, live_steps=(12, 32), zscore=False):
    cfg = checkpoint_config(checkpoint)
    if cfg.history_encoder != 'fine' or cfg.history_path_tokens:
        raise ValueError('Require a fine history checkpoint without explicit path tokens')
    if not math.isfinite(pair_rank_weight) or pair_rank_weight <= 0:
        raise ValueError('Pair ranking weight must be finite and positive')
    if len(live_steps) != 2 or any(type(s) is not int for s in live_steps) or not 1 <= live_steps[0] <= live_steps[1]:
        raise ValueError('Invalid live chain range')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable checkpoint: missing {key}')
    options = checkpoint['training_options']
    if not options['live_continuation'] or not options['decision_fraction']:
        raise ValueError('Source must already use live continuation and matched pairs')
    with torch.random.fork_rng(devices=[]):
        old = build_model(cfg)
        old.load_state_dict(checkpoint['model'], strict=True)
        old.load_state_dict(checkpoint['ema'], strict=True)
        old_names = [name for name, _ in old.named_parameters()]
        groups = checkpoint['optimizer']['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(old_names):
            raise ValueError('Expected standard complete single-group AdamW layout')
        old_ids = dict(zip(old_names, groups[0]['params']))
        migrated = copy.deepcopy(checkpoint)
        migrated['model_cfg']['history_path_tokens'] = True
        new_cfg = DirectConfig(**migrated['model_cfg'])
        migrated['architecture'] = new_cfg.architecture
        # Reproducible new parameters without consuming the saved training RNG.
        torch.manual_seed(20261001)
        new = build_model(new_cfg)
        initialized = new.state_dict()
        added = set(initialized)-set(checkpoint['model'])
        if not added or any(not name.startswith('history_encoder.path_projection.') for name in added):
            raise ValueError('Unexpected parameters in memory migration')
        for section in ('model', 'ema'):
            for name in added:
                migrated[section][name] = initialized[name].clone()
            new.load_state_dict(migrated[section], strict=True)
        # New modules change parameter enumeration order. Map moments by name,
        # leaving only new parameters without moments (Adam initializes these).
        new_names = [name for name, _ in new.named_parameters()]
        state = migrated['optimizer']['state']
        migrated['optimizer']['state'] = {i: state[old_ids[name]] for i, name in enumerate(new_names)
                                          if name in old_ids and old_ids[name] in state}
        migrated['optimizer']['param_groups'][0]['params'] = list(range(len(new_names)))
        opt = torch.optim.AdamW(new.parameters())
        opt.load_state_dict(migrated['optimizer'])
        for param, values in opt.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in values and values[key].shape != param.shape:
                    raise ValueError(f'Optimizer tensor shape mismatch: {key}')
    migrated['training_options'].update(history_path_tokens=True, pair_rank_weight=pair_rank_weight,
        live_continuation_steps=list(live_steps), live_continuation_stratified=True)
    if zscore:
        from ..shared.ct_normalization import ZSCORE_METHOD, ZSCORE_EPSILON, volume_key
        from ..shared.volume import FiberVolumeSpec
        document = dict(method=ZSCORE_METHOD, volumes={})
        for key, record in checkpoint['ct_normalization']['volumes'].items():
            document['volumes'][key] = dict(method=ZSCORE_METHOD, volume=key, epsilon=ZSCORE_EPSILON,
                **{k: record[k] for k in ('shape', 'chunks', 'dtype')})
        migrated['ct_normalization'] = document
        key = volume_key(FiberVolumeSpec(**migrated['vol_spec']))
        migrated['vol_spec']['ct_normalization'] = copy.deepcopy(document['volumes'][key])
    return migrated


def migrate_fixture_normalization(payload, migrated):
    """Change only preprocessing provenance; preserve every frozen geometry array."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads(str(arrays['__metadata__']))
    metadata['provenance']['volume']['ct_normalization'] = migrated['vol_spec']['ct_normalization']
    arrays['__metadata__'] = json.dumps(metadata)
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def fork_run(source, name, *, pair_rank_weight=.25, live_steps=(12, 32), zscore=False):
    return fork_checkpoint(source, name,
        lambda ck: expand_memory_checkpoint(ck, pair_rank_weight=pair_rank_weight, live_steps=live_steps, zscore=zscore),
        transform_fixture=migrate_fixture_normalization if zscore else None,
        allowed_changes={'history_path_tokens', 'pair_rank_weight', 'live_continuation_steps',
                         'live_continuation_stratified'}, migration_key='memory_migration',
        describe=lambda original, migrated: dict(
            initialization='preserve all existing model/EMA tensors and named Adam moments; '
                           'new path projection seed 20261001, zero final layer, fresh Adam state',
            model_changes={'history_path_tokens': dict(before=False, after=True)},
            normalization_change=(dict(before=original['ct_normalization']['method'],
                                       after=migrated['ct_normalization']['method']) if zscore else None),
            pair_rank_weight=pair_rank_weight, live_steps=list(live_steps),
            live_strata='three balanced contiguous bands; uniform integer limits within each band'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--pair-rank-weight', type=float, default=.25)
    parser.add_argument('--live-steps', type=int, nargs=2, default=(12, 32))
    parser.add_argument('--zscore', action='store_true', help='Migrate current and historical CT to ordinary per-crop z-scores')
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, pair_rank_weight=args.pair_rank_weight,
                              live_steps=tuple(args.live_steps), zscore=args.zscore), indent=2))


if __name__ == '__main__':
    main()
