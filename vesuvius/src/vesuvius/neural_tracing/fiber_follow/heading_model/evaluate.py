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
from vesuvius.neural_tracing.fiber_follow.heading_model.frames import orthonormal_frame, roll_supervision, frame_angles


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def quantiles(values):
    return np.round(np.quantile(values, [.5, .9, .99]), 2).tolist() if len(values) else None


@torch.no_grad()
def model_headings(model, patch, path, frames, device='cpu', batch=512, family=None):
    local = torch.cat([model(patch[i:i+batch].to(device), path[i:i+batch].to(device),
                            family[i:i+batch].to(device) if family is not None else None).double().cpu()
                       for i in range(0, len(patch), batch)]).numpy()
    return [f @ d for f, d in zip(frames, local)]


@torch.no_grad()
def evaluate_states(model, states, patch, path, device='cpu', batch=512):
    family = torch.tensor([s['family'] for s in states], dtype=torch.long, device=device) if model.cfg.predict_frames else None
    outputs = [model.forward_outputs(patch[i:i+batch].to(device), path[i:i+batch].to(device),
                                     family[i:i+batch] if family is not None else None)
               for i in range(0, len(patch), batch)]
    local = torch.cat([o['heading'].cpu() for o in outputs]).double().numpy()
    report = alignment_report(states, [s['frame'] @ h for s, h in zip(states, local)], model.cfg.forward)
    if model.cfg.predict_normals:
        normals = torch.cat([o['normal'].cpu() for o in outputs]).double().numpy()
        weights = np.array([s['normal_weight'] for s in states])
        cosine = np.clip(np.abs(np.sum(normals*np.stack([s['normal_target'] for s in states]), axis=-1)), 0, 1)
        for name, lo, hi in HISTORY_BINS:
            if name not in report:
                continue
            idx = np.array([lo <= s['history'] < hi for s in states]) & (weights > 0)
            report[name]['normal'] = dict(n=int(idx.sum()), angle_deg=quantiles(np.degrees(np.arccos(cosine[idx]))),
                loss=float(np.average(1-cosine[idx]**2, weights=weights[idx])) if idx.any() else None)
    if model.cfg.predict_frames:
        target_h = torch.from_numpy(np.stack([s['target'] @ s['frame'] for s in states])).double()
        target_n = torch.from_numpy(np.stack([s['normal_target'] for s in states])).double()
        _, roll_cos, roll_weights = roll_supervision(torch.from_numpy(normals), target_n, target_h,
                                                    torch.from_numpy(weights))
        predicted = torch.cat([o['frame'].cpu() for o in outputs]).double()
        errors = frame_angles(predicted, orthonormal_frame(target_h, target_n)).numpy()
        roll_errors = torch.rad2deg(torch.acos(roll_cos.clamp(0, 1))).numpy()
        for name, lo, hi in HISTORY_BINS:
            if name not in report:
                continue
            idx = np.array([lo <= s['history'] < hi for s in states]) & (roll_weights.numpy() > 0)
            report[name]['frame'] = dict(n=int(idx.sum()), angle_deg=quantiles(errors[idx]),
                mean_angle_deg=float(errors[idx].mean()) if idx.any() else None,
                roll_angle_deg=quantiles(roll_errors[idx]))
    return report


def normal_summary_metric(reports):
    values = [b['normal']['loss'] for r in reports.values() for b in r.values()
              if b.get('normal', {}).get('loss') is not None]
    return float(np.mean(values)) if values else float('inf')


def frame_summary_metric(reports):
    values = [b['frame']['mean_angle_deg'] for r in reports.values() for b in r.values()
              if b.get('frame', {}).get('mean_angle_deg') is not None]
    return float(np.mean(values)) if values else float('inf')


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
        if 'normal' in b:
            lines.append(f"    normal: n {b['normal']['n']} | unsigned angle p50/p90/p99 {b['normal']['angle_deg']} deg")
        if 'frame' in b:
            lines.append(f"    frame: n {b['frame']['n']} | rotation p50/p90/p99 {b['frame']['angle_deg']} deg | "
                         f"roll about target heading {b['frame']['roll_angle_deg']} deg")
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
    from vesuvius.neural_tracing.fiber_follow.train.runloop import raise_open_file_limit
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
        reports[name] = evaluate_states(model, states, patch, path, args.device)
        print(format_report(name, reports[name]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))


if __name__ == '__main__':
    main()
