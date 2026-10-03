"""Benchmark shared heading/normal sampling on cached real Paris 4 and AFV states.

Run after compare_heading_patch_normals.py; uses its saved positions, frames and native CT references.
Cache-only unless --fetch is passed. Timings include CT reads, sampling and targets;
exclude initial downloads, path/heading label generation and GPU.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
from vesuvius.neural_tracing.fiber_follow.data.datasets import ct_source_spec, read_dataset_config
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingConfig, model_inputs
from vesuvius.neural_tracing.fiber_follow.heading_model.normals import NORMAL_TARGET_POLICY, normal_target
from vesuvius.neural_tracing.fiber_follow.heading_model.train import read_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', default='output/heading_patch_normals_20261002')
    ap.add_argument('--config', default='configs/heading_model_l0_w16_centered_normals.json')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--fetch', action='store_true', help='fetch missing public CT chunks into the experiment cache')
    ap.add_argument('--out', default='output/heading_normal_sampling_benchmark.json')
    args = ap.parse_args()
    config = read_config(args.config)
    cfg = HeadingConfig(**config['model'])
    document, _ = read_dataset_config(config['dataset_config'])
    records = json.loads(Path(config['ct_normalization']).read_text())['volumes']
    report = dict(platform=platform.platform(), repeats=args.repeats, threads=1,
                  normal_target_policy=NORMAL_TARGET_POLICY, sources={})
    for source in document['sources']:
        data = np.load(Path(args.samples)/(source['name']+'.npz'))
        spec = ct_source_spec(source, str(Path(args.samples)/'ct_cache'))
        spec.ct_normalization = records[volume_key(spec)]
        vol = FiberVolume(spec, cache_bytes=256 << 20, cache_only=not args.fetch)
        positions, frames = data['positions'], data['frames']
        paths = [p[None] for p in positions]  # timing only; original histories were not saved
        def sample(normals):
            return model_inputs(vol, cfg, positions, frames, paths, normal_targets=normals)
        old, shared = sample(False), sample(True)  # warm cache and JIT
        np.testing.assert_array_equal(shared[0].numpy(), old[0].numpy())
        np.testing.assert_array_equal(shared[0].numpy()[:, 0], data['patches'])
        reference = np.stack([normal_target(c, p, 1.)[0] for c, p in zip(data['contexts'], data['context_centers'])])
        world = np.einsum('nij,nj->ni', frames, shared[2].numpy())
        angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(world*reference, axis=-1)), 0, 1)))
        timings = {k: [] for k in ('heading_only', 'shared_heading_normal', 'two_pass_heading_normal')}
        for _ in range(args.repeats):
            for name in timings:
                start = time.perf_counter()
                if name == 'two_pass_heading_normal':
                    sample(False); sample(True)
                else:
                    sample(name == 'shared_heading_normal')
                timings[name].append(1000*(time.perf_counter()-start)/len(positions))
        row = dict(n=len(positions), input_max_abs_difference=0., normal_valid_fraction=float((shared[3] > 0).float().mean()),
                   normal_angle_to_native_p50_p90_max=np.quantile(angles, [.5, .9, 1]).tolist(),
                   ms_per_sample={k: dict(mean=float(np.mean(v)), p50=float(np.median(v)), min=min(v), max=max(v))
                                  for k, v in timings.items()})
        report['sources'][source['name']] = row
        print(source['name'], json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
