"""Add a fresh sheet-normal head while preserving all trained heading weights.

Usage: python scripts/convert_heading_normals.py SOURCE.pt DEST.pt [--seed 0]
The result initializes a new run; it intentionally has no optimizer/resume state.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingNet, load_heading_model, save_heading_model
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY


def convert(source, destination, seed=0):
    old, checkpoint = load_heading_model(source)
    if old.cfg.predict_normals:
        raise ValueError('Source already has a normal head')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HeadingNet(replace(old.cfg, predict_normals=True)).eval()
        result = model.load_state_dict(old.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {k for k in model.state_dict() if k.startswith('normal_head.')}:
            raise ValueError(f'Unexpected conversion keys: {result}')
        patch = torch.randn(8, 1, old.cfg.patch.depth, old.cfg.patch.width, old.cfg.patch.width)
        path = torch.randn(8, 4*old.cfg.window)
        with torch.no_grad():
            torch.testing.assert_close(old(patch, path), model(patch, path), rtol=0, atol=0)
    provenance = {k: checkpoint[k] for k in ('sampling', 'dataset_config', 'dataset_config_sha256', 'ct_normalization', 'sources')
                  if k in checkpoint}
    # Exclusive creation protects an existing checkpoint, including the source itself.
    with Path(destination).open('xb') as stream:
        save_heading_model(stream, model, **provenance, normal_target_policy=NORMAL_TARGET_POLICY,
                           initialized_from=dict(path=str(Path(source).resolve()), step=checkpoint.get('step'), seed=seed))
    print(f'Converted {source} (step {checkpoint.get("step")}) -> {destination}; heading outputs exactly equal; fresh normal head')
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('destination')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    convert(args.source, args.destination, args.seed)
