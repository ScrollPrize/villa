"""Resumable, deterministic sweep of every annotation for native-traced negatives."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, ZBand, fiber_manifest, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at, tangent_at
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import (
    MiningConfig, MiningResources, PolylineIndex, prediction_manifest, review_image,
    seed_pairs, trace_controls, validated_path,
)
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_dedup import PathCoverage


BANK_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def anchor_positions(length, stride, cfg):
    # End margins are the same as the reviewed sweep; validation independently
    # checks the entire result against controlled annotation boundaries.
    margin = cfg.block_size/2
    return np.arange(margin, length-margin+1e-9, stride, dtype=np.float64)


def training_eligible(is_training_fiber, origin, cfg, band):
    lo, hi = origin[2]-2, origin[2]+cfg.block_size+2
    return bool(is_training_fiber and not (lo < band.hi and hi > band.lo))


def pack_paths(paths, ranges, eligible, anchors):
    return dict(points=np.concatenate(paths) if paths else np.empty((0, 3), np.float64),
                offsets=np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64),
                arc_ranges=np.asarray(ranges, np.float64).reshape(-1, 2),
                train_eligible=np.asarray(eligible, bool), anchors=np.asarray(anchors, np.float64))


_WORK = None


def worker_init(root):
    global _WORK
    import torch
    torch.set_num_threads(1)
    root = Path(root)
    run = json.loads((root/'run.json').read_text())
    cfg = MiningConfig(**run['mining'])
    resources = MiningResources(root/'predictions.lasagna.json', run['native_build_python'], run['ct'],
                                run['ct_grid_scale'], cfg.grid_scale)
    _WORK = dict(root=root, run=run, cfg=cfg, resources=resources, fibers=OrderedDict(),
                 band=ZBand(*run['excluded_z']))


def worker_fiber(fi):
    cache = _WORK['fibers']
    if fi not in cache:
        record = _WORK['run']['fibers'][fi]
        with np.load(_WORK['root']/record['geometry'], allow_pickle=False) as data:
            fiber = TracedFiber(record['name'], data['points'], data['s'], record['tag'], source_hash=record['source_hash'])
        cache[fi] = (fiber, PolylineIndex(fiber.points))
        if len(cache) > 2:
            cache.popitem(last=False)
    cache.move_to_end(fi)
    return cache[fi]


def shard_job(job):
    fi, begin, end = job
    root, run, cfg, res = (_WORK[k] for k in ('root', 'run', 'cfg', 'resources'))
    fiber, index = worker_fiber(fi)
    record = run['fibers'][fi]
    # Workers propose; only the coordinator may publish globally deduplicated
    # shards. Unpublished proposals can be reused after an interruption.
    relative = Path('shards')/f'{fi:04d}'/f'{begin:06d}'
    directory = root/'pending'/relative.relative_to('shards')
    directory.mkdir(parents=True, exist_ok=True)
    done = directory/'proposal.json'
    if done.exists():
        result = json.loads(done.read_text())
        if result['run_digest'] != run['digest']:
            raise ValueError(f'Incompatible completed shard: {directory}')
        if hashlib.sha256((directory/'bank.npz').read_bytes()).hexdigest() != result['bank_sha256']:
            raise ValueError(f'Damaged proposal: {directory}')
        return result
    positions = anchor_positions(fiber.length, run['stride'], cfg)[begin:end]
    paths, ranges, eligible, anchors, rows, timings = [], [], [], [], [], []
    rejected = Counter()
    started = time.perf_counter()
    for ai, t in enumerate(positions, begin):
        center = interp_at(fiber.points, fiber.s, [t])[0]
        origin = np.floor(center).astype(int)-cfg.block_size//2
        presence, directions = res.block(origin, cfg)
        pairs, support = seed_pairs(presence, directions, origin, center, tangent_at(fiber.points, fiber.s, t),
                                    fiber.points, cfg, target_index=index)
        if not pairs:
            rejected['no_confident_seed_pair'] += 1
        for _, controls, component, support_id in pairs:
            before = time.perf_counter()
            path, native_detail = trace_controls(res.native, res.field, controls, res.native_config, cfg)
            elapsed = time.perf_counter()-before
            timings.append(elapsed)
            detail = native_detail
            if path is not None:
                path, detail = validated_path(path, fiber.points, fiber.s, origin, presence, directions, support,
                                              support_id, cfg, target_index=index,
                                              seed_center_arc=native_detail.get('seed_center_arc'))
            if path is None:
                rejected[detail['reason']] += 1
                continue
            name = f'a{ai:06d}_c{component:05d}'
            train = training_eligible(record['training_fiber'], origin, cfg, _WORK['band'])
            row = dict(id=name, status='auto_validated', target=fiber.name, target_hash=fiber.source_hash,
                       target_type=fiber.tag, target_arc=float(t), grid_scale=cfg.grid_scale,
                       training_eligible=train, negative_xyz=path.tolist(), seeds_xyz=controls.tolist(),
                       metrics=detail, native=native_detail, native_seconds=elapsed,
                       target_geometry='../../../'+record['geometry'])
            review_image(directory/(name+'.png'), fiber, path, controls, detail,
                         res.volumes['presence'], res.ct, res.ct_scale)
            write_json(directory/(name+'.json'), row)
            rows.append(row)
            paths.append(path)
            ranges.append(detail['target_arc_range'])
            eligible.append(train)
            anchors.append(float(t))
    bank = directory/'bank.npz'
    with bank.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **pack_paths(paths, ranges, eligible, anchors))
    bank.with_suffix('.tmp').replace(bank)
    write_json(directory/'candidates.json', rows)
    result = dict(run_digest=run['digest'], fiber=fi, begin=begin, end=end, path=str(relative),
                  anchor_range=[float(positions[0]), float(positions[-1])], anchors=len(positions),
                  candidates=len(paths), training_candidates=sum(eligible), attempted_traces=len(timings),
                  rejected=dict(rejected), seconds=time.perf_counter()-started,
                  native_seconds=timings, bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest())
    write_json(done, result)  # Proposal marker is last: incomplete work reruns.
    return result


def restore_coverage(root, completed, options):
    """Rebuild from committed geometry only, never from in-flight proposals."""
    coverage = PathCoverage(**options)
    for row in sorted(completed,key=lambda r:(r['fiber'],r['begin'])):
        bank = root/row['path']/'bank.npz'
        if hashlib.sha256(bank.read_bytes()).hexdigest() != row['bank_sha256']:
            raise ValueError(f'Damaged shard {bank}')
        with np.load(bank,allow_pickle=False) as data:
            for a,b in zip(data['offsets'][:-1],data['offsets'][1:]):
                coverage.add(data['points'][a:b])
    coverage.flush()
    return coverage


def commit_shard(root, run, proposal, coverage):
    """The single coordinator accepts proposals in stable annotation/arc order.

    A committed shard is immutable. This ordering and restoring its geometry
    on resume make duplicate decisions independent of worker completion order.
    """
    if proposal['run_digest'] != run['digest']:
        raise ValueError('Proposal run differs')
    relative = Path(proposal['path'])
    source = root/'pending'/relative.relative_to('shards')
    destination = root/relative
    if (destination/'done.json').exists():
        raise FileExistsError(f'Shard already committed: {destination}')
    if hashlib.sha256((source/'bank.npz').read_bytes()).hexdigest() != proposal['bank_sha256']:
        raise ValueError(f'Damaged proposal {source}')
    rows = json.loads((source/'candidates.json').read_text())
    paths, ranges, eligible, anchors, kept = [], [], [], [], []
    with np.load(source/'bank.npz',allow_pickle=False) as data:
        for i,row in enumerate(rows):
            path = data['points'][data['offsets'][i]:data['offsets'][i+1]]
            if coverage.duplicate(path):
                continue
            coverage.add(path)
            paths.append(path.copy())
            ranges.append(data['arc_ranges'][i])
            eligible.append(bool(data['train_eligible'][i]))
            anchors.append(float(data['anchors'][i]))
            kept.append(row)
    destination.mkdir(parents=True,exist_ok=True)
    for row in kept:
        for suffix in ('.json','.png'):
            shutil.copyfile(source/(row['id']+suffix),destination/(row['id']+suffix))
    bank = destination/'bank.npz'
    with bank.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream,**pack_paths(paths,ranges,eligible,anchors))
    bank.with_suffix('.tmp').replace(bank)
    write_json(destination/'candidates.json',kept)
    rejected = Counter(proposal['rejected'])
    rejected['duplicate'] += len(rows)-len(kept)
    result = dict(proposal,candidates=len(kept),training_candidates=sum(eligible),rejected=dict(rejected),
                  bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest())
    write_json(destination/'done.json',result)  # Atomic commit after all outputs.
    shutil.rmtree(source)
    return result


def publish(root, run, completed, total_jobs, elapsed):
    ordered = sorted(completed, key=lambda r: (r['fiber'], r['begin']))
    complete = len(ordered) == total_jobs
    counts = Counter()
    for row in ordered:
        counts.update(row['rejected'])
    timings = [t for row in ordered for t in row['native_seconds']]
    report = dict(complete=complete, completed_shards=len(ordered), total_shards=total_jobs,
                  completed_anchors=sum(r['anchors'] for r in ordered), total_anchors=run['total_anchors'],
                  candidates=sum(r['candidates'] for r in ordered),
                  training_candidates=sum(r['training_candidates'] for r in ordered),
                  evaluation_candidates=sum(r['candidates']-r['training_candidates'] for r in ordered),
                  annotations_with_candidates=len({r['fiber'] for r in ordered if r['candidates']}),
                  rejected=dict(counts), wall_seconds_this_session=elapsed,
                  native_ms=({k: float(v*1000) for k,v in zip(('mean','p50','p95'),
                              (np.mean(timings),np.median(timings),np.quantile(timings,.95)))} if timings else {}))
    write_json(root/'report.json', report)
    shards = [{k:r[k] for k in ('fiber','begin','end','path','anchor_range','candidates','training_candidates','bank_sha256')} for r in ordered]
    manifest = dict(version=BANK_VERSION, complete=complete, run_digest=run['digest'], mining=run['mining'],
                    excluded_z=run['excluded_z'], fibers=run['fibers'], shards=shards)
    manifest['sha256'] = digest(manifest)
    write_json(root/'bank.json', manifest)
    links = []
    by_fiber = {}
    for row in ordered:
        by_fiber.setdefault(row['fiber'], []).append(row)
    for fi, fiber in enumerate(run['fibers']):
        members = by_fiber.get(fi, [])
        count = sum(r['candidates'] for r in members)
        train = sum(r['training_candidates'] for r in members)
        links.append(f'<tr><td><a href="fibers/{fi:04d}/index.html">{html.escape(fiber["name"])}</a></td>'
                     f'<td>{html.escape(fiber["tag"])}</td><td>{count}</td><td>{train}</td></tr>')
    page = '<!doctype html><meta charset="utf-8"><title>Bulk neighbor fibers</title><style>body{font:16px sans-serif;margin:24px}td,th{padding:6px;text-align:left}</style>'
    page += f'<h1>Bulk neighbor fibers</h1><p>{report["candidates"]:,} accepted paths; {report["training_candidates"]:,} training eligible. '
    page += f'{report["completed_anchors"]:,} / {run["total_anchors"]:,} anchor locations completed.</p>'
    page += '<p>Cyan: target annotation. Orange: proposed negative. Every accepted path has a comparison image and coordinates. Evaluation-only paths are excluded from training.</p>'
    page += '<table><tr><th>Annotation</th><th>Type</th><th>Accepted</th><th>Training eligible</th></tr>'+''.join(links)+'</table>'
    (root/'index.html').write_text(page)
    return report


def fiber_gallery(root, run, fi, completed):
    directory = root/'fibers'/f'{fi:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for shard in sorted((r for r in completed if r['fiber'] == fi), key=lambda r:r['begin']):
        for row in json.loads((root/shard['path']/'candidates.json').read_text()):
            relative = '../../'+shard['path']+'/'+row['id']
            partition = 'training eligible' if row['training_eligible'] else 'evaluation only'
            entries.append(f'<article><h2>{row["id"]} · {partition}</h2><a href="{relative}.json">Coordinates and checks</a>'
                           f'<img loading="lazy" src="{relative}.png"></article>')
    title = html.escape(run['fibers'][fi]['name'])
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title>'
        '<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;background:#eee}img{width:100%}article{background:white;padding:12px;margin:24px 0}</style>'
        f'<a href="../../index.html">All annotations</a><h1>{title}</h1><p>Both large panels show the same pair. Cyan: target annotation. Orange: proposed negative. Bottom: 3D separation.</p>'+''.join(entries))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--ct', required=True, help='CT array including level')
    ap.add_argument('--ct-grid-scale', type=float, default=8.)
    ap.add_argument('--native-build-python', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stride', type=float, default=8.)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--anchors-per-shard', type=int, default=64)
    ap.add_argument('--seed-spacing', type=float, default=20., help='Distance between tracing controls, in fiber-grid voxels')
    ap.add_argument('--extrapolation', type=float, default=10., help='Trace length beyond each control, in fiber-grid voxels')
    ap.add_argument('--min-path-length', type=float, help='Minimum retained path arclength; requires --max-path-length')
    ap.add_argument('--max-path-length', type=float, help='Maximum retained path arclength; overrides --extrapolation')
    ap.add_argument('--block-size', type=int, default=80, help='Validation cube side; at least maximum path length + 8')
    ap.add_argument('--min-distance', type=float, default=0., help='Inner search radius from the target, in trace-grid voxels')
    ap.add_argument('--max-distance', type=float, default=12., help='Outer search radius from the target, in trace-grid voxels')
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000.,48500.))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-shards', type=int, help='Bound a pilot; resume without this flag for the full sweep')
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not np.isfinite(args.stride) or args.stride <= 0 or min(args.workers,args.anchors_per_shard) < 1:
        ap.error('Stride and worker/shard sizes must be positive')
    if args.max_shards is not None and args.max_shards < 1:
        ap.error('max-shards must be positive')
    cfg = MiningConfig(seed_spacing=args.seed_spacing, extrapolation=args.extrapolation, block_size=args.block_size,
                       min_path_length=args.min_path_length, max_path_length=args.max_path_length,
                       min_distance=args.min_distance, max_distance=args.max_distance)
    root = args.output.resolve()
    fibers = load_fibers(args.fibers, grid_scale=cfg.grid_scale)
    band = ZBand(*(np.asarray(args.val_z)/cfg.grid_scale))
    train, _ = split_fibers(fibers, band)
    train_names = {f.name for f in train}
    records = [dict(entry, length=f.length, tag=f.tag, training_fiber=f.name in train_names,
                    geometry=f'fibers/{i:04d}/geometry.npz') for i,(entry,f) in enumerate(zip(fiber_manifest(fibers), fibers))]
    pred = prediction_manifest(args.fiber_zarrs)
    policy_files = [Path(__file__), Path(__file__).with_name('neighbor_mining.py'),
                    Path(__file__).with_name('neighbor_dedup.py')]
    run = dict(version=BANK_VERSION, mining=asdict(cfg), stride=args.stride, anchors_per_shard=args.anchors_per_shard,
               excluded_z=[band.lo, band.hi], fibers=records, prediction_manifest=pred,
               ct=str(Path(args.ct).resolve()), ct_grid_scale=args.ct_grid_scale,
               native_build_python=str(Path(args.native_build_python).resolve()),
               implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in policy_files},
               deduplication=dict(distance=2.,overlap=.8,sample_step=1.,max_angle=25.),
               total_anchors=sum(len(anchor_positions(f.length,args.stride,cfg)) for f in fibers))
    run['digest'] = digest(run)
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous = json.loads((root/'run.json').read_text())
        if previous != run:
            raise ValueError('Resume input geometry, mining policy, source paths or implementation changed')
    else:
        if any(root.iterdir()):
            raise FileExistsError('Output is nonempty; use --resume for an identical run')
        write_json(root/'run.json', run)
        write_json(root/'predictions.lasagna.json', pred)
        for record, fiber in zip(records, fibers):
            destination = root/record['geometry']
            destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez(destination, points=fiber.points, s=fiber.s)
    jobs, completed = [], []
    missing = False
    for fi, fiber in enumerate(fibers):
        count = len(anchor_positions(fiber.length,args.stride,cfg))
        for begin in range(0,count,args.anchors_per_shard):
            done = root/'shards'/f'{fi:04d}'/f'{begin:06d}'/'done.json'
            if done.exists():
                if missing:
                    raise ValueError('Committed shards must form an ordered prefix for deterministic deduplication')
                row = json.loads(done.read_text())
                if row['run_digest'] != run['digest']:
                    raise ValueError(f'Incompatible shard {done}')
                bank = done.parent/'bank.npz'
                if hashlib.sha256(bank.read_bytes()).hexdigest() != row['bank_sha256']:
                    raise ValueError(f'Damaged shard {bank}')
                completed.append(row)
            else:
                missing = True
                jobs.append((fi,begin,min(count,begin+args.anchors_per_shard)))
    total_jobs = len(jobs)+len(completed)
    if args.max_shards:
        jobs = jobs[:args.max_shards]
    started = last_report = time.perf_counter()
    publish(root,run,completed,total_jobs,0.)
    print(f'{len(fibers)} annotations, {run["total_anchors"]:,} anchors; {len(jobs)} pending shards, {args.workers} workers',flush=True)
    changed = set()
    coverage = restore_coverage(root,completed,run['deduplication'])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(root),)) as pool:
        futures = [pool.submit(shard_job,job) for job in jobs]
        for future in futures:
            row = commit_shard(root,run,future.result(),coverage)
            completed.append(row)
            changed.add(row['fiber'])
            now = time.perf_counter()
            if now-last_report >= 15 or len(completed) == total_jobs:
                report = publish(root,run,completed,total_jobs,now-started)
                for fi in changed:
                    fiber_gallery(root,run,fi,completed)
                changed.clear()
                print(f'{report["completed_anchors"]:,}/{report["total_anchors"]:,} anchors; '
                      f'{report["candidates"]:,} paths ({report["training_candidates"]:,} train); '
                      f'{now-started:.0f}s',flush=True)
                last_report = now
    for fi in changed:
        fiber_gallery(root,run,fi,completed)
    report = publish(root,run,completed,total_jobs,time.perf_counter()-started)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
