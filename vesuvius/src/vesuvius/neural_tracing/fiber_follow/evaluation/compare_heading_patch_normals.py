"""Compare normals from the EXACT heading-model input with native 65³ CT normals.

Uses heading_model.data.plan_states and heading_model.model.model_inputs,
including the trainer's state distribution, random roll, sampler and z-score.
No separate or resized patch is substituted for the model's tensor.

python evaluation/compare_heading_patch_normals.py --out output/heading_patch_normals
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.data.data import load_fibers
from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config, ct_source_spec
from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.heading_model.data import plan_states, fiber_weights, sampling_from_dict
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingConfig, model_inputs
from vesuvius.neural_tracing.fiber_follow.heading_model.train import read_config
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor, _ct_context
from vesuvius.neural_tracing.fiber_follow.shared.geometry import normalize


def normal(image, center, spacing=1.):
    values, vectors = np.linalg.eigh(ct_structure_tensor(image, center, sample_spacing=spacing))
    gap = (values[-1]-values[-2])/max(values[-1], 1e-12)
    return vectors[:, -1], float(gap)


def angles(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(normalize(a)*normalize(b), axis=-1)), 0, 1)))


def summary(values):
    return dict(n=len(values), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                p99=float(np.percentile(values, 99)), maximum=float(np.max(values)),
                over20_count=int(np.sum(values > 20)), over30_count=int(np.sum(values > 30)))


def sample_source(source, config, cfg, out, count, seed, threads):
    path = out/(source['name']+'.npz')
    if path.exists():
        return path
    spec = ct_source_spec(source, str(out/'ct_cache'))
    records = json.loads(Path(config['ct_normalization']).read_text())['volumes']
    spec.ct_normalization = records[volume_key(spec)]
    print('Loading fibers:', source['name'], flush=True)
    fibers = (load_fibers(source['fibers'], grid_scale=spec.grid_scale) if source['kind'] == 'paris4' else
              AFVFibers(source['path'], grid_scale=spec.grid_scale, validation=source['validation'], split='train'))
    rng = np.random.default_rng(seed)
    states = plan_states(fibers, fiber_weights(fibers, cfg.forward+8), count, rng, cfg,
                         sampling_from_dict(config['sampling']), roll_rng=rng)
    vol = FiberVolume(spec, cache_bytes=256 << 20)
    positions = np.stack([s['pos'] for s in states])
    frames = np.stack([s['frame'] for s in states])
    started = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        patch, _ = model_inputs(vol, cfg, positions, frames, [s['path'] for s in states], pool)
        contexts = list(pool.map(lambda pos: _ct_context(vol, pos), positions))
    assert tuple(patch.shape) == (count, 1, 32, 32, 32)
    centers = np.stack([center for _, center in contexts])
    np.savez_compressed(path, patches=patch.numpy()[:, 0], frames=frames, positions=positions,
                        contexts=np.stack([image for image, _ in contexts]), context_centers=centers,
                        history=np.array([s['history'] for s in states]), input_scale=vol.input_scale,
                        config=json.dumps(config), fetch_seconds=time.perf_counter()-started)
    print('Saved exact model tensors:', path, flush=True)
    return path


def evaluate(path, cfg, repeats):
    data = np.load(path)
    patches, frames, contexts = data['patches'], data['frames'], data['contexts']
    centers = data['context_centers']
    center = np.array([cfg.patch.behind, (cfg.patch.width-1)/2, (cfg.patch.width-1)/2])
    spacing = cfg.patch.spacing*float(data['input_scale'])
    # Control: same patch, same native physical kernel widths. Alternative:
    # original sigma=1 / integration=4 in patch samples (=2.5 / 10 CT voxels).
    results = {}
    variants = dict(native65=(contexts, centers, None, 1.),
                    model_patch_matched_scale=(patches, [center]*len(patches), frames, spacing),
                    model_patch_sigma1_sigma4=(patches, [center]*len(patches), frames, 1.))
    for key, (images, at, rotations, step) in variants.items():
        normal(images[0], at[0], step)  # warmup
        durations, normals, gaps = [], [], []
        for repeat in range(repeats):
            started = time.perf_counter()
            estimates = [normal(image, c, step) for image, c in zip(images, at)]
            durations.append((time.perf_counter()-started)/len(images)*1000)
            if repeat == 0:
                normals = np.stack([n for n, _ in estimates])
                gaps = np.array([gap for _, gap in estimates])
                if rotations is not None:
                    normals = np.einsum('bij,bj->bi', rotations, normals)
        results[key] = dict(normals=normals, gaps=gaps, ms_per_sample=float(np.median(durations)),
                            timing_repeats=durations)
    report = {}
    for key, value in results.items():
        difference = angles(value['normals'], results['native65']['normals'])
        report[key] = dict(angle_vs_native=summary(difference), ms_per_sample=value['ms_per_sample'],
                          timing_repeats=value['timing_repeats'],
                          seed_angles=summary(difference[data['history'] < 1e-9]) if np.any(data['history'] < 1e-9) else None,
                          trace_angles=summary(difference[data['history'] >= 1e-9]) if np.any(data['history'] >= 1e-9) else None,
                          weak_tensor_count=int(np.sum(value['gaps'] < .05)))
    np.savez_compressed(path.with_name(path.stem+'_normals.npz'),
                        **{k:v['normals'] for k,v in results.items()})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/heading_model_l0_w16_centered.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--count', type=int, default=64)
    ap.add_argument('--seed', type=int, default=7351)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'protocol.json').write_text(json.dumps(dict(
        command=sys.argv, platform=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
        python=platform.python_version(), omp_threads=os.environ.get('OMP_NUM_THREADS'),
        openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'), count_per_source=args.count,
        seed=args.seed, timing_repeats=args.repeats,
        patch='Exact float32 z-scored model_inputs output; no resizing, intensity thresholding, or additional CT',
        reference='65^3 native CT at the same head; derivative sigma 1, integration sigma 4',
        variants={'model_patch_matched_scale': 'Kernel widths 1/2.5 and 4/2.5 patch pixels',
                  'model_patch_sigma1_sigma4': 'Kernel widths 1 and 4 patch pixels (2.5 and 10 native CT voxels)'},
        timing='Single-thread tensor calculation plus eigendecomposition; excludes loading, crop sampling, and training',
        sampling='Same plan_states sampler as training, including seeds, noisy/off-fiber traces, prior headings, and random roll; Paris4 controlled fibers and AFV training split'), indent=2))
    config = read_config(args.config)
    cfg = HeadingConfig(**config['model'])
    if cfg.ct_downsample_levels != 0:
        raise ValueError('This experiment expects the selected level-0 heading model')
    document, _ = read_dataset_config(config['dataset_config'])
    report = {}
    for i, source in enumerate(document['sources']):
        path = sample_source(source, config, cfg, args.out, args.count, args.seed+i, args.threads)
        report[source['name']] = evaluate(path, cfg, args.repeats)
        (args.out/'summary.json').write_text(json.dumps(report, indent=2))
        print(source['name'], json.dumps(report[source['name']], indent=2), flush=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fiber_follow_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(report), figsize=(15, 4))
    for ax, name in zip(axes, report):
        normals = np.load(args.out/(name+'_normals.npz'))
        for key, label in [('model_patch_matched_scale', 'Original CT smoothing widths'),
                           ('model_patch_sigma1_sigma4', 'Sigma 1 / 4 in patch pixels')]:
            values = np.sort(angles(normals[key], normals['native65']))
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
        ax.set(title=name, xlabel='Difference from 65³ native-CT normal (degrees)', ylabel='Fraction of samples',
               xlim=(0, 90), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Exact normalized 32³ heading-model inputs — 64 samples per source')
    fig.tight_layout()
    fig.savefig(args.out/'comparison.png', dpi=150)


if __name__ == '__main__':
    main()
