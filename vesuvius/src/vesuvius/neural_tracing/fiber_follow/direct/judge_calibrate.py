"""Run the frozen paired judge calibration protocol or its locked final comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .judge_evaluation import (FORECAST_THRESHOLDS, ACCEPT_THRESHOLDS, ALARM_THRESHOLDS,
                              freeze_protocol, paired_row, metrics, select, load_selection, bootstrap)
from .judge_options import load_judge
from .judge_policy import JudgePolicyConfig
from .train import load_checkpoint, validate_volume_source
from .data import DirectTracer
from ..data import load_fibers, split_fibers, ZBand, fiber_manifest
from ..events import label_path
from ..experiment import read_manifest, rollout_summary
from ..evaluate import score_trace
from ..trace import TraceParams
from ..volume import FiberVolume


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('calibrate','final'))
    ap.add_argument('--judge', action='store_true')
    ap.add_argument('--checkpoints', nargs='+')
    ap.add_argument('--selection')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fibers', default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    manifest = read_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    selection = None
    if args.mode == 'final':
        if not args.selection or args.checkpoints:
            ap.error('Final requires --selection and forbids checkpoint overrides')
        selection = load_selection(args.selection)
        if selection['manifest_sha256'] != manifest['sha256']:
            raise ValueError('Final manifest differs from locked calibration')
        checkpoints = [selection['checkpoint']]
        thresholds = [selection['threshold']]
        policies = [JudgePolicyConfig(**selection['judge_policy'])]
        protocol = None
    else:
        if not args.checkpoints:
            ap.error('Calibration requires --checkpoints')
        checkpoints, thresholds = args.checkpoints, FORECAST_THRESHOLDS
        policies = [JudgePolicyConfig(accept=a, alarm=b) for a in ACCEPT_THRESHOLDS for b in ALARM_THRESHOLDS if b < a]
        protocol = freeze_protocol(args.out, checkpoints, manifest['sha256'], max_len=6000, sampling_seed=0)
    reports = []
    for checkpoint in checkpoints:
        model, crop, nh, spec, ck = load_checkpoint(checkpoint, args.device)
        validate_volume_source(spec, manifest)
        _, fibers = split_fibers(load_fibers(args.fibers, spec.grid_scale), ZBand(45000/spec.grid_scale,48500/spec.grid_scale))
        if fiber_manifest(fibers) != manifest['fibers']:
            raise ValueError('Frozen annotation geometry changed')
        judge_args = load_judge(ck, args.device)
        reader = judge_args['judge_slices'].open()
        if selection is not None and reader.identity != selection['judge_source_sha256']:
            raise ValueError('Selected native source changed')
        for threshold in thresholds:
            params = TraceParams(max_len=6000, confidence=threshold, n_commit=ck.get('n_commit', 4), seed=0)
            baseline = DirectTracer(model, FiberVolume(spec), crop, nh, params, device=args.device)
            baseline_paths, event_labels, base_legacy = [], [], []
            initialized_legacy = []
            initialized_path = ck.get('init_tracer_path') or ck.get('training_options', {}).get('init_tracer')
            initialized_tracer = None
            if initialized_path:
                if hashlib.sha256(Path(initialized_path).read_bytes()).hexdigest() != ck['init_tracer_sha256']:
                    raise ValueError('Initialized follower checkpoint changed')
                initial, initial_crop, initial_nh, initial_spec, _ = load_checkpoint(initialized_path, args.device)
                initialized_tracer = DirectTracer(initial, FiberVolume(initial_spec), initial_crop, initial_nh,
                                                  params, device=args.device)
            try:
                for seed in manifest[args.mode if args.mode == 'final' else 'calibration']:
                    paths, reasons = baseline.trace(np.asarray([seed['pos']]), np.asarray([seed['heading']]))
                    path = paths[0]
                    f = fibers[seed['fiber']]
                    reverse = seed['sign'] < 0
                    event_labels.append(label_path(path, f.points[::-1] if reverse else f.points,
                                        f.length-seed['t'] if reverse else seed['t'], f.endpoint_stop[0 if reverse else 1]))
                    baseline_paths.append(path)
                    base_legacy.append(score_trace(path, f, seed['t'], seed['sign']))
                    if initialized_tracer is not None:
                        initial_paths, _ = initialized_tracer.trace(np.asarray([seed['pos']]), np.asarray([seed['heading']]))
                        initialized_legacy.append(score_trace(initial_paths[0], f, seed['t'], seed['sign']))
            finally:
                baseline.close()
                if initialized_tracer is not None:
                    initialized_tracer.close()
            for policy in policies:
                kwargs = dict(judge_args, judge_policy=policy)
                tracer = DirectTracer(model, FiberVolume(spec), crop, nh, params, device=args.device, **kwargs)
                tracer.judge_reader = reader
                rows, legacy = [], []
                began = time.monotonic()
                try:
                    for seed, baseline_path, event in zip(manifest['final' if selection else 'calibration'], baseline_paths, event_labels):
                        paths, reasons = tracer.trace(np.asarray([seed['pos']]), np.asarray([seed['heading']]))
                        state = tracer.observed_states[0]
                        audit = state['policy'].audit[-1]
                        induced = reasons[0].startswith('judge_')
                        rows.append(paired_row(baseline_path, paths[0], event, audit['endpoint'], seed['fiber'], induced,
                                              any(not all(a['support']) for a in state['policy'].audit)))
                        legacy.append(score_trace(paths[0], fibers[seed['fiber']], seed['t'], seed['sign']))
                finally:
                    tracer.close()
                report = dict(checkpoint=str(Path(checkpoint).resolve()),
                              checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                              threshold=threshold, judge_policy=policy.to_dict(), judge_source_sha256=reader.identity,
                              metrics=metrics(rows), rows=rows, seconds=time.monotonic()-began,
                              legacy_baseline=base_legacy, legacy_judge=legacy, n_commit=params.n_commit,
                              initialized_follower=initialized_legacy,
                              sampling_seed=0, max_len=6000)
                if selection:
                    report['bootstrap'] = bootstrap(rows)
                reports.append(report)
                name = f'{Path(checkpoint).stem}_{threshold}_{policy.accept}_{policy.alarm}.json'
                (args.out/name).write_text(json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(), allow_nan=True))
                print(json.dumps(report['metrics']), flush=True)
    if protocol is not None:
        return select(reports, protocol, args.out)
    return reports
