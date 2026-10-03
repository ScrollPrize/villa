"""Compare crop frames against annotated future fibers, without running a follower.

OMP_NUM_THREADS=1 python evaluation/compare_future_crop_frames.py --checkpoint RUN/ckpt_STEP.pt --out OUTPUT
Uses the trainer's deterministic held-out states, simulated path/prior inputs and shared 2/8 CT stencil.
Future annotations only score predictions. Outputs retain geometry for testing other crop dimensions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.heading_model.data import (
    HISTORY_BINS, load_sources, sampling_from_dict, fiber_weights, plan_states,
)
from vesuvius.neural_tracing.fiber_follow.heading_model.model import load_heading_model, model_inputs
from vesuvius.neural_tracing.fiber_follow.heading_model.frames import orthonormal_frame
from vesuvius.neural_tracing.fiber_follow.heading_model.train import read_config
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.tracing.heading import sheet_heading, SeedHeadingError
from vesuvius.neural_tracing.fiber_follow.train.runloop import raise_open_file_limit


def containment(future, frame, crop):
    """Score every future point, including points behind/beyond the crop; distances in trace voxels."""
    local = np.asarray(future) @ np.asarray(frame)
    half = (crop.width-1)*crop.spacing/2
    axial = (local[:, 2] >= -crop.behind*crop.spacing) & (local[:, 2] <= (crop.depth-1-crop.behind)*crop.spacing)
    lateral = np.abs(local[:, :2]).max(axis=1)
    inside = axial & (lateral <= half)
    return dict(max_u=float(np.abs(local[:, 0]).max()), max_v=float(np.abs(local[:, 1]).max()),
                max_radial=float(np.linalg.norm(local[:, :2], axis=1).max()),
                required_half_width=float(lateral.max()), all_inside=bool(inside.all()),
                point_fraction=float(inside.mean()), axial_all_inside=bool(axial.all()),
                width_sweep={str(w): bool((axial & (lateral <= w)).all()) for w in (2., 4., 8., 12., 16.)})


def distribution(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                p99=float(np.percentile(values, 99)))


def summarize(rows):
    if not rows:
        return dict(n=0)
    out = dict(n=len(rows), methods={})
    for method in rows[0]['metrics']:
        out['methods'][method] = {}
        for horizon in rows[0]['metrics'][method]:
            values = [r['metrics'][method][horizon] for r in rows]
            out['methods'][method][horizon] = dict(
                **{k: distribution([v[k] for v in values]) for k in
                   ('max_u', 'max_v', 'max_radial', 'required_half_width')},
                **{k: float(np.mean([v[k] for v in values])) for k in
                   ('all_inside', 'point_fraction', 'axial_all_inside')},
                width_sweep={w: float(np.mean([v['width_sweep'][w] for v in values]))
                             for w in values[0]['width_sweep']})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/heading_model_l0_w16_centered_frame.json')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--states', type=int, default=512, help='Held-out states per source')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    raise_open_file_limit()
    args.out.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    model, checkpoint = load_heading_model(args.checkpoint)
    if not model.cfg.predict_frames or model.cfg.ct_downsample_levels:
        raise ValueError('Expected the level-0 frame model')
    print('Loading annotated sources', flush=True)
    _, _, sources, _ = load_sources(config['dataset_config'], args.out,
                                    ct_normalization=config['ct_normalization'])
    sampling = sampling_from_dict(checkpoint['sampling'])
    crop = CoordinateRegressionConfig().fine
    rows, excluded = [], []
    with ThreadPoolExecutor(8) as pool:
        for source_index, source in enumerate(sources):
            print('Evaluating', source.name, flush=True)
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 7349, source_index]))
            states = plan_states(source.validation, fiber_weights(source.validation, model.cfg.forward+8),
                                 args.states, rng, model.cfg, sampling)
            vol = FiberVolume(source.spec, cache_bytes=128 << 20, cache_only=True)
            for start in range(0, len(states), 32):
                batch = states[start:start+32]
                patch, path, normals, weights = model_inputs(vol, model.cfg, [s['pos'] for s in batch],
                    [s['frame'] for s in batch], [s['path'] for s in batch], pool, normal_targets=True)
                with torch.no_grad():
                    output = model.forward_outputs(patch, path, torch.tensor([s['family'] for s in batch]))
                for j, state in enumerate(batch):
                    identity = dict(source=source.name, index=start+j, family='HV'[state['family']])
                    if weights[j] <= 0:
                        excluded.append(dict(**identity, reason='weak tensor normal'))
                        continue
                    tensor_n = state['frame'] @ normals[j].numpy()
                    try:
                        tensor_h = sheet_heading(tensor_n, identity['family'])
                    except SeedHeadingError as error:
                        excluded.append(dict(**identity, reason=str(error)))
                        continue
                    if tensor_h @ state['prior'] < 0:
                        tensor_h = -tensor_h
                    model_h = state['frame'] @ output['heading'][j].numpy()
                    if model_h @ state['prior'] < 0:
                        model_h = -model_h
                    model_n = state['frame'] @ output['normal'][j].numpy()
                    frames = {name: orthonormal_frame(torch.from_numpy(h), torch.from_numpy(n)).numpy()
                              for name, h, n in (
                                  ('tensor_hv', tensor_h, tensor_n),
                                  ('prior_tensor_roll', state['prior'], tensor_n),
                                  ('model_tensor_roll', model_h, tensor_n),
                                  ('model_frame', model_h, model_n))}
                    future = state['future']
                    rows.append(dict(**identity, history=state['history'], position=state['pos'].tolist(),
                        prior=state['prior'].tolist(), future=future.tolist(),
                        frames={k: v.tolist() for k, v in frames.items()},
                        metrics={k: {str(n): containment(future[:n], f, crop) for n in (16, len(future))}
                                 for k, f in frames.items()}))
                print(source.name, min(start+32, len(states)), '/', len(states), flush=True)
    report = dict(checkpoint=str(args.checkpoint), step=checkpoint['step'], crop=asdict(crop),
        seed=config['seed'], requested_states_per_source=args.states, excluded=excluded,
        protocol='Held-out fibers, trainer validation-state distribution and deterministic input roll. '
                 'Simulated observed history; noisy seed priors as during training, not actual trace outcomes. '
                 '2/8 tensor uses shared raw CT stencil without fiber-tangent correction. '
                 'Tensor H/V heading uses the existing global-Z sheet-heading rule, signed along the same input prior. '
                 'Model inputs are CT, prior, observed path and H/V; future annotations used only for scoring. '
                 'All methods score the same 16 or 35 annotated future arclength samples, with no axial filtering. '
                 'Distances in trace voxels; half-width sweep retains the follower axial bounds. CPU inference, cached CT.',
        overall=summarize(rows),
        by_source={s.name: summarize([r for r in rows if r['source']==s.name]) for s in sources},
        by_history={name: summarize([r for r in rows if lo <= r['history'] < hi]) for name, lo, hi in HISTORY_BINS},
        by_family={family: summarize([r for r in rows if r['family']==family]) for family in ('H', 'V')}, rows=rows)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2)+'\n')
    print(json.dumps(report['overall']), flush=True)


if __name__ == '__main__':
    main()
