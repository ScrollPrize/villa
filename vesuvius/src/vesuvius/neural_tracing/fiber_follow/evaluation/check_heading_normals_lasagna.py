"""Small CPU-only seed test of heading checkpoints against cached Lasagna normals.

python evaluation/check_heading_normals_lasagna.py --checkpoints RUN/ckpt_010000.pt RUN/ckpt_012000.pt --out OUTPUT
Uses cached Paris 4 / AFV 1447 regions, both fiber directions and both signs, with
the trainer's noisy seed priors and empty path features. No downloads or training changes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
from vesuvius.neural_tracing.fiber_follow.data.datasets import ct_source_spec, read_dataset_config
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.evaluation.compare_sheet_normals import angle, stats
from vesuvius.neural_tracing.fiber_follow.heading_model.data import cone, sampling_from_dict
from vesuvius.neural_tracing.fiber_follow.heading_model.model import load_heading_model, model_inputs, prior_frames
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('output/sheet_normals_20261002'))
    ap.add_argument('--references', type=Path, default=Path('output/lasagna_tensor_sweep_20261002'))
    ap.add_argument('--seed', type=int, default=7431)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    models = [load_heading_model(p) for p in args.checkpoints]
    cfg = models[0][0].cfg
    if not cfg.predict_normals or cfg.ct_downsample_levels:
        raise ValueError('Expected a level-0 heading + normal model')
    if any(m.cfg.to_dict() != cfg.to_dict() for m, _ in models):
        raise ValueError('Checkpoints must share input configuration')
    checkpoint = models[0][1]
    sampling = sampling_from_dict(checkpoint['sampling'])
    sources, _ = read_dataset_config('configs/mixed_ct_datasets_paris50.json')
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                  target_policy=NORMAL_TARGET_POLICY, checkpoints=[], sources={},
                  protocol='32 cached regions/volume; H+, H-, V+, V- noisy seed priors; random roll; '
                           'exact normalized 32-cube inputs and empty observed path; CPU only; '
                           'unsigned angles; locations not guaranteed excluded from model training; '
                           'four orientations per region are correlated; Lasagna is a prediction, not ground truth')
    for path, (_, ckpt) in zip(args.checkpoints, models):
        report['checkpoints'].append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                          step=ckpt['step'], normal_target_policy=ckpt.get('normal_target_policy')))
    for source_index, source in enumerate(sources['sources']):
        name = source['name']
        if name not in ('paris4', '1447_5mm_v1'):
            continue
        spec = ct_source_spec(source, str(args.cache/'cache'))
        spec.ct_normalization = checkpoint['ct_normalization']['volumes'][volume_key(spec)]
        vol = FiberVolume(spec, cache_bytes=128 << 20, cache_only=True)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, source_index]))
        patches, paths, frames, references, teachers, weights, regions, modes, positions = ([] for _ in range(9))
        missing = []
        for region, file in enumerate(sorted((args.references/name).glob('chunk_*.npz'))):
            with np.load(file) as saved:
                pos_ct, reference = saved['positions'][0].copy(), saved['normals'][0].copy()
            with np.load(args.cache/name/file.name) as saved:
                row = json.loads(str(saved['row_json']))
            if not np.isfinite(reference).all():
                continue
            pos = pos_ct/vol.input_scale
            for mode, tangent in zip(('H+', 'H-', 'V+', 'V-'),
                                      [np.array(row['tangents']['12'][i])*sign for i in (0, 1) for sign in (1, -1)]):
                prior = cone(tangent, rng, sampling.cone_deg, sampling.cone_cap_deg)
                frame = prior_frames([prior], rng)[0]
                try:
                    patch, path, target, weight = model_inputs(vol, cfg, [pos], [frame], [pos[None]], normal_targets=True)
                except FileNotFoundError as e:
                    if 'not prefetched' not in str(e):
                        raise
                    missing.append(dict(region=region, mode=mode, reason=str(e)))
                    continue
                assert not path.any()  # real seed condition: no synthetic path or normal features
                patches.append(patch); paths.append(path); frames.append(frame); references.append(reference)
                teachers.append(frame @ target[0].numpy()); weights.append(float(weight[0]))
                regions.append(region); modes.append(mode); positions.append(pos)
        if not patches:
            raise ValueError(f'No cached evaluation samples for {name}')
        patch, path = torch.cat(patches), torch.cat(paths)
        frames, references, teachers = np.stack(frames), np.stack(references), np.stack(teachers)
        tensor_angles = angle(teachers, references)
        summary = dict(n=len(patch), regions=len(set(regions)), missing_cached_samples=missing,
                       tensor_vs_lasagna=stats(tensor_angles), models={})
        arrays = dict(regions=regions, modes=modes, positions=positions, frames=frames, lasagna=references,
                      tensor=teachers, tensor_weight=weights, tensor_vs_lasagna=tensor_angles)
        for model, ckpt in models:
            predictions = []
            with torch.no_grad():
                for i in range(0, len(patch), 32):
                    predictions.append(model.forward_outputs(patch[i:i+32], path[i:i+32])['normal'].numpy())
            normals = np.einsum('bij,bj->bi', frames, np.concatenate(predictions))
            error, teacher_error = angle(normals, references), angle(normals, teachers)
            key = str(ckpt['step'])
            summary['models'][key] = dict(vs_lasagna=stats(error), vs_tensor=stats(teacher_error),
                closer_to_lasagna_than_tensor=float(np.mean(error < tensor_angles)),
                by_direction={mode: stats(error[np.array(modes) == mode]) for mode in sorted(set(modes))})
            arrays[f'model_{key}'], arrays[f'angle_{key}'] = normals, error
        np.savez_compressed(args.out/(name+'.npz'), **arrays)
        report['sources'][name] = summary
        print(name, json.dumps(summary), flush=True)
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(args.out, report)


def plot(out, report):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fiber_follow_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, summary) in zip(axes, report['sources'].items()):
        data = np.load(out/(name+'.npz'))
        for key, label in [('tensor_vs_lasagna', '2/8 tensor')]+[(f'angle_{s}', f'Model step {s}') for s in summary['models']]:
            values = np.sort(data[key])
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
        ax.set(title=f'{name}: {summary["regions"]} regions, {summary["n"]} seed views',
               xlabel='Unsigned difference from Lasagna (degrees)', ylabel='Fraction of samples', xlim=(0, 40), ylim=(0, 1))
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
