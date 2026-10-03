"""Extend a heading+normal checkpoint with H/V conditioning, preserving optimizer and step.

python scripts/convert_heading_frame.py SOURCE/last.pt FRAME_RUN/last.pt
Resume the destination using a config with model.predict_frames=true and a larger total steps value.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingNet, load_heading_model, save_heading_model
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import normal_target_policy


def convert(source, destination, seed=0, roll_loss_weight=.25):
    old, checkpoint = load_heading_model(source)
    if not old.cfg.predict_normals or old.cfg.predict_frames:
        raise ValueError('Expected a heading+normal checkpoint without H/V conditioning')
    if 'optimizer' not in checkpoint or 'step' not in checkpoint:
        raise ValueError('Use last.pt so optimizer and training step can be continued')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HeadingNet(replace(old.cfg, predict_frames=True)).eval()
        missing = model.load_state_dict(old.state_dict(), strict=False)
        if missing.unexpected_keys or missing.missing_keys != ['family_embedding.weight']:
            raise ValueError(f'Unexpected model parameters: {missing}')
        patch = torch.randn(8, 1, old.cfg.patch.depth, old.cfg.patch.width, old.cfg.patch.width)
        path = torch.randn(8, 4*old.cfg.window)
        with torch.no_grad():
            expected = old.forward_outputs(patch, path)
            for family in (0, 1):
                actual = model.forward_outputs(patch, path, torch.full((8,), family, dtype=torch.long))
                for key in ('heading', 'normal'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    old_names, new_names = list(dict(old.named_parameters())), list(dict(model.named_parameters()))
    if new_names != old_names+['family_embedding.weight']:
        raise ValueError('Parameter ordering changed; optimizer conversion needs updating')
    optimizer = copy.deepcopy(checkpoint['optimizer'])
    groups = optimizer['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(old_names):
        raise ValueError('Expected the heading trainer single-group optimizer')
    groups[0]['params'].append(max(groups[0]['params'])+1)
    # Existing moments and counters are unchanged; the new embedding gets fresh Adam state on its first update.
    check_optimizer = torch.optim.AdamW(model.parameters())
    check_optimizer.load_state_dict(optimizer)
    extra = {k: v for k, v in checkpoint.items() if k not in ('architecture', 'config', 'state')}
    policy = normal_target_policy(model.cfg)
    history = list(checkpoint.get('normal_target_history', []))
    history.append(dict(step=checkpoint['step'], old=checkpoint.get('normal_target_policy'), new=policy))
    extra.update(optimizer=optimizer, best=float('inf'), best_normal=float('inf'), best_frame=float('inf'),
                 normal_target_policy=policy, normal_target_history=history, roll_loss_weight=roll_loss_weight,
                 family_encoding=dict(H=0, V=1),
                 initialized_from=dict(path=str(Path(source).resolve()), step=checkpoint['step'],
                     sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), seed=seed))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        save_heading_model(stream, model, **extra)
    print(f'{destination}: continued step {checkpoint["step"]}; heading/normal outputs exactly preserved for H and V; '
          'optimizer preserved; frame targets use fiber-corrected normals')
    return model, extra


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('destination')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--roll-loss-weight', type=float, default=.25)
    args = ap.parse_args()
    convert(args.source, args.destination, args.seed, args.roll_loss_weight)
