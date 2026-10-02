"""Fork a path-token checkpoint with observed-path geometry tokens and a fresh optimizer.

Preserves every existing model/EMA tensor, RNG and update count. New geometry-token
parameters use a fixed seed with a zero final layer. AdamW starts without moments
and the LR schedule restarts its warmup at the fork step. The dataset config may
change only source sampling weights; the rest-gradient clip is replaced.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from .checkpoint_migration import fork_checkpoint
from .datasets import read_dataset_config
from .model import DirectConfig, build_model
from .train import checkpoint_config

SEED = 20261002
PREFIX = 'path_geometry.'


def without_weights(document):
    document = copy.deepcopy(document)
    for source in document['sources']:
        source.pop('weight')
    return document


def expand_geometry_checkpoint(checkpoint, *, dataset_config, rest_grad_clip):
    cfg = checkpoint_config(checkpoint)
    if not cfg.history_path_tokens or cfg.path_geometry_tokens:
        raise ValueError('Require a path-token checkpoint without geometry tokens')
    if not math.isfinite(rest_grad_clip) or rest_grad_clip <= 0:
        raise ValueError('Rest gradient clip must be finite and positive')
    for key in ('optimizer', 'rng', 'ema', 'step', 'samples_seen', 'training_options', 'dataset_config'):
        if key not in checkpoint:
            raise ValueError(f'Not a resumable mixed-dataset checkpoint: missing {key}')
    document, digest = read_dataset_config(dataset_config)
    if without_weights(document) != without_weights(checkpoint['dataset_config']):
        raise ValueError('Only dataset sampling weights may change')
    options = checkpoint['training_options']
    with torch.random.fork_rng(devices=[]):
        old = build_model(cfg)
        old.load_state_dict(checkpoint['model'], strict=True)
        old.load_state_dict(checkpoint['ema'], strict=True)
        migrated = copy.deepcopy(checkpoint)
        migrated['model_cfg']['path_geometry_tokens'] = True
        new_cfg = DirectConfig(**migrated['model_cfg'])
        migrated['architecture'] = new_cfg.architecture
        # Reproducible new parameters without consuming the saved training RNG.
        torch.manual_seed(SEED)
        new = build_model(new_cfg)
        initialized = new.state_dict()
        added = set(initialized)-set(checkpoint['model'])
        if not added or set(checkpoint['model'])-set(initialized) or any(not n.startswith(PREFIX) for n in added):
            raise ValueError('Unexpected parameters in geometry migration')
        for section in ('model', 'ema'):
            for name in added:
                migrated[section][name] = initialized[name].clone()
            new.load_state_dict(migrated[section], strict=True)
        # The trainer's AdamW layout with no moments; resume loads it unchanged.
        migrated['optimizer'] = torch.optim.AdamW(new.parameters(), lr=options['lr'], weight_decay=1e-4).state_dict()
    migrated['lr_restart_step'] = int(checkpoint['step'])
    migrated['dataset_config'], migrated['dataset_config_sha256'] = document, digest
    migrated['training_options'].update(path_geometry_tokens=True, rest_grad_clip=float(rest_grad_clip),
                                        dataset_config=str(Path(dataset_config).resolve()))
    return migrated


def fork_run(source, name, *, dataset_config, rest_grad_clip):
    def weights(document):
        return {s['name']: s['weight'] for s in document['sources']}
    return fork_checkpoint(source, name,
        lambda ck: expand_geometry_checkpoint(ck, dataset_config=dataset_config, rest_grad_clip=rest_grad_clip),
        allowed_changes={'path_geometry_tokens', 'rest_grad_clip', 'dataset_config'}, migration_key='geometry_migration',
        describe=lambda original, migrated: dict(
            initialization='preserve all existing model/EMA tensors, RNG and update count; new path geometry '
                           f'tokens seed {SEED}, zero final layer; fresh AdamW (no moments); LR warmup restarts '
                           f"at step {original['step']}",
            model_changes={'path_geometry_tokens': dict(before=False, after=True)},
            dataset_weights=dict(before=weights(original['dataset_config']), after=weights(migrated['dataset_config'])),
            rest_grad_clip=dict(before=original['training_options']['rest_grad_clip'], after=rest_grad_clip),
            lr_restart_step=migrated['lr_restart_step']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset-config', required=True, help='Same sources as the checkpoint; weights may differ')
    parser.add_argument('--rest-grad-clip', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(fork_run(args.checkpoint, args.name, dataset_config=args.dataset_config,
                              rest_grad_clip=args.rest_grad_clip), indent=2))


if __name__ == '__main__':
    main()
