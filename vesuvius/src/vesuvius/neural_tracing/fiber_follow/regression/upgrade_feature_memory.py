"""Migrate a stopped revision-1 run to revision 2 without resetting its optimizer."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .model import build_model
from .train import checkpoint_config, build_parser, FEATURE_OPTIONS
from .upgrade_refinement import remap_optimizer, options_argv


def upgrade_checkpoint(ck):
    old_cfg = checkpoint_config(ck)
    if old_cfg.memory_version != 4 or old_cfg.feature_memory_revision != 1:
        raise ValueError('Expected feature-memory revision 1')
    if not all(k in ck for k in ('optimizer', 'rng', 'model', 'ema', 'step')):
        raise ValueError('A resumable checkpoint is required')
    cfg = replace(old_cfg, feature_memory_revision=2, feature_detail_tokens=16,
                  feature_stream_steps=max(128, old_cfg.memory_steps*2), feature_replay_weight=.5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        old, model = build_model(old_cfg), build_model(cfg)
    old.load_state_dict(ck['model'], strict=True)
    fresh = model.state_dict()
    added = set(fresh)-set(ck['model'])
    if set(ck['model'])-set(fresh):
        raise ValueError('Migration removed existing parameters')
    result = dict(ck, model_cfg=cfg.to_dict())
    for kind in ('model', 'ema'):
        result[kind] = {**{k: fresh[k].clone() for k in added}, **ck[kind]}
        model.load_state_dict(result[kind], strict=True)
    result['optimizer'] = remap_optimizer(ck['optimizer'], old, model)
    result['feature_memory_upgrade'] = dict(source_step=ck['step'], new_parameters=sorted(added),
        feature_memory_revision=2, feature_stream_steps=cfg.feature_stream_steps,
        replay='three age-stratified historical encoders; all writer transitions; detached stale features elsewhere',
        optimizer='existing moments preserved by parameter name; new parameters have no moments')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = args.checkpoint.resolve()
    ck = torch.load(source, map_location='cpu', weights_only=False)
    upgraded = upgrade_checkpoint(ck)
    cfg = checkpoint_config(upgraded)
    # Fill options added after the source checkpoint with their parser defaults.
    options = vars(build_parser().parse_args(options_argv(ck['training_options'])))
    destination = args.out.resolve()
    cp = destination/f'ckpt_{ck["step"]:06d}.pt'
    options.update(name=destination.name, out_root=str(destination.parent), resume=str(cp), init_tracer=None,
                   **{k: getattr(cfg, k) for k in FEATURE_OPTIONS})
    upgraded['training_options'] = options
    upgraded['feature_memory_upgrade'].update(source_checkpoint=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    fixture = source.parent/'monitor_recovery.npz'
    if options['recovery_every'] and hashlib.sha256(fixture.read_bytes()).hexdigest() != ck['monitor_recovery_sha256']:
        raise ValueError('Monitor fixture differs from checkpoint')
    replay = json.loads((source.parent/'dagger/replay.json').read_text())
    if any(not Path(p).exists() for p in replay):
        raise FileNotFoundError('Published replay bank missing')
    destination.mkdir(parents=True)
    torch.save(upgraded, cp)
    shutil.copy2(cp, destination/'last.pt')
    if options['recovery_every']:
        shutil.copy2(fixture, destination/fixture.name)
    (destination/'dagger').mkdir()
    (destination/'dagger/replay.json').write_text(json.dumps(replay, indent=2))
    config = json.loads((source.parent/'config.json').read_text())
    config.update(options, model_cfg=cfg.to_dict(), feature_memory_upgrade=upgraded['feature_memory_upgrade'])
    config['parameter_count'] = sum(p.numel() for p in build_model(cfg).parameters())
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    (destination/'upgrade.json').write_text(json.dumps(upgraded['feature_memory_upgrade'], indent=2))
    (destination/'resume_argv.json').write_text(json.dumps(options_argv(options), indent=2))
    print(json.dumps(dict(checkpoint=str(cp), step=ck['step'], changes=upgraded['feature_memory_upgrade']), indent=2))


if __name__ == '__main__':
    main()
