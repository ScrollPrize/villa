"""Heading alignment on held-out fibers: does a heading keep the upcoming fiber in crop? No tracing.

For each history bin: angle to the in-crop target for the prior (the tracer's heading) and the model, the share of
states where the model is closer, and the largest lateral distance of the annotated fiber from the crop axis over
the 16-voxel forecast and over the follower crop's forward extent (prior, model and the target's floor).

Usage: python -m vesuvius.neural_tracing.fiber_follow.heading_model.evaluate --config CONFIG.json --checkpoint MODEL.pt
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.data import HISTORY_BINS
from vesuvius.neural_tracing.fiber_follow.heading_model.targets import lateral_extent


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def quantiles(values):
    return np.round(np.quantile(values, [.5, .9, .99]), 2).tolist() if len(values) else None


@torch.no_grad()
def model_headings(model, patch, path, frames, device='cpu', batch=512):
    local = torch.cat([model(patch[i:i+batch].to(device), path[i:i+batch].to(device)).double().cpu()
                       for i in range(0, len(patch), batch)]).numpy()
    return [f @ d for f, d in zip(frames, local)]


def alignment_report(states, headings, forward):
    report = {}
    for name, lo, hi in HISTORY_BINS:
        idx = [i for i, s in enumerate(states) if lo <= s['history'] < hi]
        if not idx:
            continue
        rows = [states[i] for i in idx]
        heads = dict(prior=[s['prior'] for s in rows], model=[headings[i] for i in idx], target=[s['target'] for s in rows])
        prior = np.array([angle(h, s['target']) for h, s in zip(heads['prior'], rows)])
        model = np.array([angle(h, s['target']) for h, s in zip(heads['model'], rows)])
        report[name] = dict(n=len(rows), angle_prior=quantiles(prior), angle_model=quantiles(model),
                            model_closer=float(np.mean(model < prior)),
                            **{f'off_axis16_{k}': quantiles([lateral_extent(s['future'][:16], h) for h, s in zip(v, rows)])
                               for k, v in heads.items()},
                            **{f'off_axis_crop_{k}': quantiles([lateral_extent(s['future'], h, forward)
                                                                for h, s in zip(v, rows)]) for k, v in heads.items()})
    return report


def summary_metric(reports):
    """Selection score: mean p90 crop-extent off-axis distance of the model's headings over sources and bins."""
    values = [b['off_axis_crop_model'][1] for r in reports.values() for b in r.values()]
    return float(np.mean(values)) if values else float('inf')


def format_report(name, report):
    lines = [f'== {name}']
    for bin_name, b in report.items():
        lines.append(f"  history {bin_name:5s} n {b['n']:4d} | angle p50/p90/p99 prior {b['angle_prior']} -> model "
                     f"{b['angle_model']} (closer {b['model_closer']:.0%}) | crop off-axis p50/p90 prior "
                     f"{b['off_axis_crop_prior'][:2]} -> model {b['off_axis_crop_model'][:2]} (floor {b['off_axis_crop_target'][:2]})")
    return '\n'.join(lines)


def main():
    from vesuvius.neural_tracing.fiber_follow.heading_model.data import load_sources, sampling_from_dict, validation_states
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import load_heading_model
    from vesuvius.neural_tracing.fiber_follow.heading_model.train import DEFAULT_CONFIG, read_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(DEFAULT_CONFIG), help='config JSON (default: configs/heading_model.json)')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--states', type=int, help='held-out states per source (default: config val_states_per_source)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', help='optional JSON report path')
    args = ap.parse_args()
    from vesuvius.neural_tracing.fiber_follow.shared.runloop import raise_open_file_limit
    raise_open_file_limit()  # memory-mapped CT chunks
    config = read_config(args.config)
    model, checkpoint = load_heading_model(args.checkpoint, args.device)
    # Bind the same CT records the model was trained with; a scratch directory receives the normalization JSON.
    scratch = Path(args.out).parent if args.out else Path(args.checkpoint).parent/'evaluation'
    _, _, sources, _ = load_sources(config['dataset_config'], scratch, ct_normalization=config.get('ct_normalization'),
                                    ct_downsample_levels=model.cfg.ct_downsample_levels)
    sampling = sampling_from_dict(checkpoint.get('sampling', config.get('sampling')))
    with ThreadPoolExecutor(16) as pool:
        held_out = validation_states(sources, model.cfg, sampling, args.states or config['val_states_per_source'],
                                     config['seed'], pool)
    reports = {}
    for name, (states, patch, path, _) in held_out.items():
        reports[name] = alignment_report(states, model_headings(model, patch, path, [s['frame'] for s in states],
                                                                args.device), model.cfg.forward)
        print(format_report(name, reports[name]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))


if __name__ == '__main__':
    main()
