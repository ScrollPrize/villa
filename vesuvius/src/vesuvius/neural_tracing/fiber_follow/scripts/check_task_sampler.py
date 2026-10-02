"""Pre-launch sampler check: delivered task shares, replay supply and fresh-trace geometry.

Builds each dataset source exactly as the trainer does (fibers, banks, replay caches,
task budget) and draws batch plans without reading CT. Reports requested and delivered
shares, fallbacks, replay classes, distinct fibers/episodes/events, startup draws and
realized seed ages, and excursion head offsets, heading error and path curvature. Exits
nonzero if no correct DAgger replay (pre-excursion, recoverable, premature stop or
ordinary) is delivered while replay is supplied.

  python scripts/check_task_sampler.py --checkpoint RUN/ckpt.pt \\
      --replay paris4=RUN/dagger/decisions_001000.npz --out output/DIR/sampler_check.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.data import FALLBACKS, SEED_AGE_STRATA, SOURCE, STARTUP_CATEGORIES, TASKS, FollowDataset, OnPolicyStates, SampleConfig, TaskBudget, seed_age_stratum, traversal_curve
from vesuvius.neural_tracing.fiber_follow.shared.geometry import tangent_at
from vesuvius.neural_tracing.fiber_follow.data.state_labels import REPLAY_CLASSES, SUPERVISION

CORRECT_REPLAY = ('dagger_pre_excursion', 'dagger_recoverable', 'dagger_premature_stop', 'dagger_ordinary')


def path_curvature(path, span=32):
    """Largest turn per voxel over the last ``span`` voxels of the observed path, degrees."""
    tail = np.asarray(path)[-span-2:]
    if len(tail) < 3:
        return 0.
    steps = np.diff(tail, axis=0)
    steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-9)
    return float(np.degrees(np.arccos(np.clip((steps[1:]*steps[:-1]).sum(-1), -1, 1))).max())


def heading_error(item, fiber):
    p, s = traversal_curve(fiber, item['fiber_ref'][2])
    tangent = tangent_at(p, s, item['fiber_ref'][1])
    return float(np.degrees(np.arccos(np.clip(abs(tangent @ np.asarray(item['frame'])[:, 2]), -1, 1))))


def summarize(items, fibers):
    rows = len(items)
    count = lambda values: dict(Counter(values))
    fresh = [i for i in items if i['source'] == SOURCE['fresh']]
    excursion = [i for i in fresh if i.get('excursion')]
    plain = [i for i in fresh if not i.get('excursion') and i['startup'] == STARTUP_CATEGORIES.index('established')]
    stats = lambda values: dict(n=len(values), **({} if not values else dict(zip(
        ('p10', 'p50', 'p90', 'max'), np.quantile(values, [.1, .5, .9, 1.]).round(3).tolist()))))
    replay = [i for i in items if i['source'] == SOURCE['replay']]
    return dict(
        rows=rows,
        requested={TASKS[k]: v/rows for k, v in count(i['task_requested'] for i in items).items()},
        delivered={TASKS[k]: v/rows for k, v in count(i['task_delivered'] for i in items).items()},
        fallbacks={f'{TASKS[i["task_requested"]]}->{FALLBACKS[i["task_fallback"]]}': 1 for i in items
                   if i['task_fallback']} and dict(Counter(f'{TASKS[i["task_requested"]]}->{FALLBACKS[i["task_fallback"]]}'
                                                          for i in items if i['task_fallback'])),
        supervision=dict(Counter(SUPERVISION[int(i['supervision'])] for i in items)),
        replay_classes=dict(Counter(REPLAY_CLASSES[i['replay_class']] for i in replay)),
        replay_distinct=dict(fibers=len({i['fiber_ref'][0] for i in replay}),
                             episodes=len({i['replay_episode'] for i in replay}),
                             events=len({i['replay_event'] for i in replay})),
        correct_replay_delivered=sum(TASKS[i['task_delivered']] in CORRECT_REPLAY for i in items),
        startup_requested=dict(Counter(STARTUP_CATEGORIES[i['startup']] for i in fresh)),
        seed_age=dict(Counter(SEED_AGE_STRATA[seed_age_stratum(i['seed_age'])][0] for i in fresh)),
        excursion_fraction_of_established=len(excursion)/max(1, len(excursion)+len(plain)),
        excursion_head_offset=stats([i['excursion_head_offset'] for i in excursion]),
        excursion_match_distance=stats([float(i['match_distance']) for i in excursion]),
        heading_error_deg=dict(excursion=stats([heading_error(i, fibers[i['fiber_ref'][0]]) for i in excursion]),
                               established=stats([heading_error(i, fibers[i['fiber_ref'][0]]) for i in plain])),
        curvature_deg_per_voxel=dict(excursion=stats([path_curvature(i['observed_path']) for i in excursion]),
                                     established=stats([path_curvature(i['observed_path']) for i in plain])))


def main(argv=None):
    from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling
    from vesuvius.neural_tracing.fiber_follow.data.datasets import AFVBank, HoldoutFilteredBank, load_primary_dataset, open_afv_source, read_dataset_config
    from vesuvius.neural_tracing.fiber_follow.train.train import checkpoint_config
    from vesuvius.neural_tracing.fiber_follow.data.data import ZBand
    from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint
    from vesuvius.neural_tracing.fiber_follow.train.train import MODEL_TYPES
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='Supplies the model crop/horizon and label tolerance')
    ap.add_argument('--dataset-config', default=str(Path(__file__).parents[1]/'configs'/'mixed_ct_datasets_paris50.json'))
    ap.add_argument('--replay', action='append', default=[], metavar='SOURCE=CACHE', help='Replay cache per source')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--task-share', action='append', default=[])
    ap.add_argument('--replay-event-cap', type=int, default=TaskBudget.replay_event_cap)
    ap.add_argument('--batches', type=int, default=100)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    ck = read_checkpoint(args.checkpoint, MODEL_TYPES, 'cpu')
    cfg = checkpoint_config(ck)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future, future_step=cfg.future_step,
                          recent_history_points=cfg.n_history, label_tolerance=float(ck.get('tolerance', 1.5)),
                          max_recovery_distance=cfg.max_recovery_distance)
    budget = TaskBudget.parse(args.task_share, replay_event_cap=args.replay_event_cap)
    document, _ = read_dataset_config(args.dataset_config)
    replay = dict(value.split('=', 1) for value in args.replay)
    sampling = IdentitySampling(bank_coverage_probability=.2)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), budget=budget.to_dict(), sources={})
    failed = False
    for source in document['sources']:
        if args.sources and source['name'] not in args.sources:
            continue
        if source['kind'] == 'paris4':
            spec = FiberVolumeSpec(**ck['vol_spec'])
            _, fibers, heldout, _ = load_primary_dataset(document, spec)
            band = ZBand(*(v/spec.grid_scale for v in source['val_z']))
            bank = HoldoutFilteredBank(source['negative_bank'], fibers, band, grid_scale=spec.grid_scale, heldout=heldout)
        else:
            fibers, _, spec, _ = open_afv_source(source, document['cache_dir'])
            bank = AFVBank(fibers)
        caches = [OnPolicyStates.load(replay[source['name']])] if source['name'] in replay else []
        builder = IdentityObservationBuilder(cfg, fibers, sampling, augment=True, negative_bank=bank)
        dataset = FollowDataset(fibers, spec, sample, None, chunk=args.batch, seed=11, budget=budget,
                                batch_builder=builder, onpolicy=caches)
        if caches:
            dataset.set_step(int(caches[0].provenance['step']))
        plans = dataset._iter_plans(None, [])
        items = [item for _ in range(args.batches) for item in next(plans)]
        summary = summarize(items, fibers)
        if caches:
            summary['cache'] = dict(path=replay[source['name']], supply=caches[0].provenance['supply'],
                                    index=dataset.index.counts(), max_event_reuse=dataset.max_event_reuse())
            if not summary['correct_replay_delivered']:
                failed = True
        report['sources'][source['name']] = summary
        print(json.dumps({source['name']: {k: summary[k] for k in ('requested', 'delivered', 'fallbacks',
                                                                   'correct_replay_delivered')}}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    if failed:
        print('No correct DAgger replay was delivered from a supplied cache', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
