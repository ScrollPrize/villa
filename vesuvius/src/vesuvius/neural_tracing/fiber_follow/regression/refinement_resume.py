"""Fork a resumable checkpoint with additional learned refinement stages."""
import argparse
import copy
import json

import torch

from .model import build_model
from .train import checkpoint_config
from .checkpoint_migration import fork_checkpoint


STAGE = 'refinement_stage.weight'


def expand_refinement_checkpoint(checkpoint, steps):
    """Keep existing weights/moments; copy the last stage into added stage rows.

    New rows get zero Adam moments, retaining the tensor's existing Adam step.
    Only the standard single-group optimizer layout is supported.
    """
    cfg = checkpoint_config(checkpoint)
    old = cfg.recurrent_refinement_steps
    if type(steps) is not int or not 1 <= old < steps:
        raise ValueError('Require an existing refinement stage and a larger integer stage count')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable training checkpoint: missing {key}')
    if checkpoint['training_options']['recurrent_refinement_steps'] != old:
        raise ValueError('Checkpoint model and training refinement settings disagree')
    # Building validation models must not consume the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        model = build_model(cfg)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.load_state_dict(checkpoint['ema'], strict=True)
        names = [name for name, _ in model.named_parameters()]
        groups = checkpoint['optimizer']['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(names):
            raise ValueError('Expected the standard complete single-group AdamW layout')
        stage_id = groups[0]['params'][names.index(STAGE)]
        migrated = copy.deepcopy(checkpoint)
        for section in ('model', 'ema'):
            value = checkpoint[section][STAGE]
            if tuple(value.shape) != (old, cfg.hidden):
                raise ValueError('Unexpected stage embedding shape')
            migrated[section][STAGE] = torch.cat((value, value[-1:].expand(steps-old, -1)), 0).clone()
        for key, value in migrated['optimizer']['state'].get(stage_id, {}).items():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if tuple(value.shape) != (old, cfg.hidden):
                    raise ValueError(f'Unexpected stage optimizer shape: {key}')
                migrated['optimizer']['state'][stage_id][key] = torch.cat(
                    (value, value.new_zeros(steps-old, cfg.hidden)), 0)
            elif key != 'step':
                raise ValueError(f'Unsupported stage optimizer field: {key}')
        migrated['model_cfg']['recurrent_refinement_steps'] = steps
        migrated['training_options']['recurrent_refinement_steps'] = steps
        expanded = build_model(checkpoint_config(migrated))
        expanded.load_state_dict(migrated['model'], strict=True)
        expanded.load_state_dict(migrated['ema'], strict=True)
        optimizer = torch.optim.AdamW(expanded.parameters())
        optimizer.load_state_dict(migrated['optimizer'])
        for param, state in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and state[key].shape != param.shape:
                    raise ValueError(f'Optimizer tensor shape mismatch: {key}')
    return migrated


def fork_run(source, name, steps):
    return fork_checkpoint(source, name, lambda ck: expand_refinement_checkpoint(ck, steps),
        allowed_changes={'recurrent_refinement_steps'}, migration_key='refinement_migration',
        describe=lambda original, migrated: dict(
            source_refinements=original['model_cfg']['recurrent_refinement_steps'],
            refinement_steps=steps,
            initialization='copy last learned stage; zero new Adam moment rows; retain Adam step'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--refinement-steps', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, args.refinement_steps), indent=2))


if __name__ == '__main__':
    main()
