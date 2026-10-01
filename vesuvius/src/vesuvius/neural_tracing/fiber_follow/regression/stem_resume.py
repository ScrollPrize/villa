"""Initialize a BasicBlockD stem run with fresh history encoder and optimizer."""
import argparse
import copy
from dataclasses import replace
import json
import math

import numpy as np
import torch

from .checkpoint_migration import fork_checkpoint
from .model import DirectConfig, TOKEN_ARCHITECTURE, build_model


def expand_stem_checkpoint(checkpoint, channels=32, blocks=2, *, additional_steps=100000,
                           lr=1e-4, warmup=5000, fresh_fraction=.9, bank_following_probability=0.):
    """Transfer weights/EMA outside history, with fresh AdamW and LR warmup.

    The checkpoint already contains the fresh optimizer. Normal resumes of the
    fork retain its subsequent moments and schedule instead of resetting again.
    """
    if checkpoint['architecture'] not in ('axial_patch4_overlap_tokens_fiber_slabs_v11', TOKEN_ARCHITECTURE):
        raise ValueError('Require a token-only patch4 checkpoint without a residual stem')
    config = copy.deepcopy(checkpoint['model_cfg'])
    # Sample complete cells rather than synthesizing voxels at crop boundaries.
    for key in ('depth', 'width'):
        config['fine'][key] = 4*((config['fine'][key]+3)//4)
    cfg = DirectConfig(**config)
    if cfg.encoder != 'patch4' or not cfg.token_only or cfg.stem_channels:
        raise ValueError('Require a token-only patch4 checkpoint without a residual stem')
    if type(channels) is not int or channels < 1:
        raise ValueError('Stem channels must be a positive integer')
    if type(additional_steps) is not int or additional_steps < 1:
        raise ValueError('Additional steps must be a positive integer')
    if not math.isfinite(lr) or lr <= 0 or type(warmup) is not int or not 0 <= warmup <= additional_steps:
        raise ValueError('Require a positive finite LR and warmup within the continuation')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable training checkpoint: missing {key}')
    decision = checkpoint['training_options']['decision_fraction']
    if not math.isfinite(fresh_fraction) or not 0 <= fresh_fraction <= 1:
        raise ValueError('Fresh fraction must lie in [0, 1]')
    if not math.isfinite(bank_following_probability) or not 0 <= bank_following_probability <= 1-decision:
        raise ValueError('Bank-following fraction must fit the remaining decision budget')
    expanded_cfg = replace(cfg, stem_channels=channels, stem_blocks=blocks)
    with torch.random.fork_rng(devices=[]):
        # Only CPU model construction uses randomness. Do not alter CUDA streams.
        torch.random.default_generator.manual_seed(checkpoint['training_options']['seed'])
        expanded = build_model(expanded_cfg)
        initialized = expanded.state_dict()
        fresh_prefixes = ('encoder.stem.', 'history_encoder.')
        retained = {name for name in initialized if not name.startswith(fresh_prefixes)}
        migrated = copy.deepcopy(checkpoint)
        for section in ('model', 'ema'):
            source = checkpoint[section]
            if {name for name in source if not name.startswith('history_encoder.')} != retained:
                raise ValueError(f'Unexpected pretrained parameter names in {section}')
            migrated[section] = {name: value.clone() for name, value in initialized.items()}
            migrated[section].update({name: source[name].clone() for name in retained})
            expanded.load_state_dict(migrated[section], strict=True)
        optimizer = torch.optim.AdamW(expanded.parameters(), lr=lr, weight_decay=1e-4)
        migrated['optimizer'] = optimizer.state_dict()
    migrated['architecture'] = expanded_cfg.architecture
    migrated['model_cfg'] = expanded_cfg.to_dict()
    migrated['crop'] = migrated['model_cfg']['fine']
    if 'sample_cfg' in migrated:
        migrated['sample_cfg']['crop'] = copy.deepcopy(migrated['crop'])
    migrated['lr_restart_step'] = checkpoint['step']
    migrated['training_options'].update(stem_channels=channels, stem_blocks=blocks,
        steps=checkpoint['step']+additional_steps, lr=lr, warmup=warmup, reset_optimizer=False,
        fresh_fraction=fresh_fraction, bank_following_probability=bank_following_probability)
    return migrated


def update_fixture_crop(path, original, migrated):
    """Retain frozen recovery geometry while changing its image sampling crop."""
    if original['model_cfg']['fine'] == migrated['model_cfg']['fine']:
        return
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays['__metadata__']))
    if metadata['provenance']['sample_cfg']['crop'] != original['model_cfg']['fine']:
        raise ValueError('Recovery fixture crop does not match the source model')
    metadata['provenance']['sample_cfg']['crop'] = copy.deepcopy(migrated['model_cfg']['fine'])
    arrays['__metadata__'] = json.dumps(metadata)
    np.savez(path, **arrays)


def fork_run(source, name, channels=32, blocks=2, *, additional_steps=100000, lr=1e-4, warmup=5000, fresh_fraction=.9, bank_following_probability=0.):
    return fork_checkpoint(source, name,
        lambda ck: expand_stem_checkpoint(ck, channels, blocks,
            additional_steps=additional_steps, lr=lr, warmup=warmup,
            fresh_fraction=fresh_fraction, bank_following_probability=bank_following_probability),
        allowed_changes={'stem_channels', 'stem_blocks', 'steps', 'lr', 'warmup', 'reset_optimizer',
                         'fresh_fraction', 'bank_following_probability'},
        migration_key='stem_migration', transform_fixture=update_fixture_crop,
        describe=lambda original, migrated: dict(
            stem_channels=channels, stem_blocks=blocks, additional_steps=additional_steps,
            stem_design='BasicBlockD; InstanceNorm/ReLU; full-resolution block; two downsampling stages',
            history_initialization='fresh encoder; BasicBlockD with InstanceNorm/LeakyReLU',
            crop_before=original['model_cfg']['fine'], crop_after=migrated['model_cfg']['fine'],
            recovery_fixture='preserve fixed states; update crop metadata and digest',
            lr_restart_step=migrated['lr_restart_step'],
            initialization='zero stem output; fresh history encoder; transfer remaining weights/EMA; fresh AdamW'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=2, help='BasicBlockD blocks per downsampling stage')
    parser.add_argument('--additional-steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--fresh-fraction', type=float, default=.9)
    parser.add_argument('--bank-following-probability', type=float, default=0.)
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, args.channels, args.blocks,
        additional_steps=args.additional_steps, lr=args.lr, warmup=args.warmup,
        fresh_fraction=args.fresh_fraction, bank_following_probability=args.bank_following_probability), indent=2))


if __name__ == '__main__':
    main()
