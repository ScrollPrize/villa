"""One-off crop-heading v1 -> v2 conversion (no runtime legacy loader).

v1's encoder used 3-wide stride-2 kernels; v2 uses 4-wide ones with the same padding, whose taps cover inputs
2o-1..2o+2. A v1 kernel is the v2 kernel with a zero last tap on each axis, so the converted model computes the
same function. The converter checks that on random patches against the v1 computation before writing.

Keeps weights and data/inference provenance; drops optimizer and run progress. The result supports inference and
evaluation, not --resume. The source checkpoint is never modified.

usage (from fiber_follow/):
  python scripts/convert_heading_v1.py output/heading_model_l0_w16/best.pt output/heading_model_l0_w16/best_v2.pt
"""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from vesuvius.neural_tracing.fiber_follow.heading_model.model import ARCHITECTURE, HeadingConfig, HeadingNet

SOURCE_ARCHITECTURE = 'crop_heading_ct_path_v1'
CONVS = ('features.0', 'features.2', 'features.4', 'features.6')


def v1_forward(state, patch, path):
    """The v1 network's computation, from its state dict (verification only)."""
    x = patch
    for name in CONVS:
        x = F.silu(F.conv3d(x, state[f'{name}.weight'], state[f'{name}.bias'], stride=2, padding=1))
    hidden = torch.cat((F.adaptive_avg_pool3d(x, 2).flatten(1),
                        F.silu(F.linear(path, state['path.0.weight'], state['path.0.bias']))), -1)
    hidden = F.silu(F.linear(hidden, state['head.0.weight'], state['head.0.bias']))
    out = F.linear(hidden, state['head.2.weight'], state['head.2.bias'])
    return F.normalize(out+patch.new_tensor([0., 0., 1.]), dim=-1)


def convert(source, destination, *, checks=64, tolerance=1e-5):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    ck = torch.load(source, map_location='cpu', weights_only=False)
    if ck.get('architecture') != SOURCE_ARCHITECTURE:
        raise ValueError(f'Expected a {SOURCE_ARCHITECTURE} checkpoint, got {ck.get("architecture")}: {source}')
    old = dict(ck['config'])
    # CropSpec fields retired since v1; none affects CT patch sampling (history rendering is follower-only).
    patch = dict(old['patch'])
    if patch.pop('gate_direction', False):
        raise ValueError('Unsupported source configuration: gate_direction')
    patch.pop('history_render', None)
    patch.pop('history_sigma', None)
    cfg = HeadingConfig(**dict(old, patch=patch))
    model = HeadingNet(cfg)
    state = {}
    for key, value in ck['state'].items():
        if key.endswith('.weight') and key.rsplit('.', 1)[0] in CONVS:
            if value.shape[2:] != (3, 3, 3):
                raise ValueError(f'{key}: expected a 3x3x3 v1 kernel, got {tuple(value.shape)}')
            value = F.pad(value, (0, 1, 0, 1, 0, 1))  # zero tap at offset +2 on each axis
        state[key] = value
    model.load_state_dict(state, strict=True)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    patch = torch.randn(checks, 1, cfg.patch.depth, cfg.patch.width, cfg.patch.width, generator=generator)
    path = torch.randn(checks, 4*cfg.window, generator=generator)
    with torch.no_grad():
        error = (model(patch, path)-v1_forward(ck['state'], patch, path)).abs().max().item()
    if error > tolerance:
        raise RuntimeError(f'Converted model differs from the v1 computation by {error:g}')
    converted = {key: value for key, value in ck.items() if key not in ('state', 'optimizer', 'best')}
    converted.update(architecture=ARCHITECTURE, config=cfg.to_dict(), state=model.state_dict(),
                     conversion=dict(source=str(source.resolve()), source_architecture=SOURCE_ARCHITECTURE,
                                     source_step=ck.get('step'), max_abs_output_difference=error))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(converted, stream)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    converted = convert(args.source, args.destination)
    print(f"{args.destination}: step {converted.get('step')}, max output difference from v1 "
          f"{converted['conversion']['max_abs_output_difference']:.2e}")


if __name__ == '__main__':
    main()
