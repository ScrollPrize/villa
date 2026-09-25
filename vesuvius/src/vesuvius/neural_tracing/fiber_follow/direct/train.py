"""Train a direct curve follower from scratch, with original-fiber online replay."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import (
    DATA_POLICY, FollowDataset, OnPolicyStates, SampleConfig, ZBand, fiber_manifest, load_fibers, split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.runloop import (
    RunLog, lr_at, prepare_run_dir, read_checkpoint, save_checkpoint as write_checkpoint,
    update_ema, training_rng_state, resume_training,
)
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.direct.model import ARCHITECTURE, DirectConfig, DirectFollower
from vesuvius.neural_tracing.fiber_follow.direct.data import ObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.direct.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.direct.diagnostics import decision_rows, summarize_decisions
from vesuvius.neural_tracing.fiber_follow.direct.recovery import monitor_fixture, evaluate_monitor


def validate_volume_source(spec, manifest):
    """Allow a different CT pyramid level, retaining frozen physical data/seeds."""
    for key in ('fiber_zarr_dir', 'ct_zarr', 'fiber_level', 'grid_scale', 'inputs'):
        if spec.to_dict()[key] != manifest['volume'][key]:
            raise ValueError(f'Volume source {key} differs from frozen manifest')


def save_checkpoint(path, model, ema, spec, sample, extra=None):
    # Atomic publication: collectors must never open a partial checkpoint.
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    write_checkpoint(temporary, model, spec, sample.crop, sample.n_history, ARCHITECTURE,
                     dict(extra or {}, ema=ema.state_dict(), sample_cfg=asdict(sample),
                          coarse_ct_level=1, coarse_ct_grid_scale=8.))
    temporary.replace(path)


def load_checkpoint(path, device='cuda'):
    ck = read_checkpoint(path, ARCHITECTURE, device)
    # Older one-pass checkpoints have no correction module or config field.
    cfg = DirectConfig(**{'correction': False, **ck['model_cfg']})
    model = DirectFollower(cfg).to(device)
    model.load_state_dict(ck['ema'])
    model.eval()
    if ck.get('coarse_ct_level') != 1 or ck.get('coarse_ct_grid_scale') != 8.:
        raise ValueError('Unsupported coarse image source')
    return model, cfg.fine, cfg.n_history, FiberVolumeSpec(**ck['vol_spec']), ck


def move_batch(batch, device):
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


def optimizer_update(model, ema, opt, batches, step, lr, *, device='cpu', tolerance=1.5,
                     confidence_weight=.5, ema_decay=.999, n_commit=None, compute_metrics=True):
    """Equal weight per observed state, independent of microbatch boundaries.

    Within a state each loss averages over its known points; fully unknown
    states contribute zero. Geometry and confidence are evaluated in one pass.
    """
    total = sum(len(b['hist']) for b in batches)
    if total < 1:
        raise ValueError('An update needs at least one state')
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    sums = dict(loss=0., geometry=0., confidence_loss=0., error_sum=0., geometry_count=0.,
                correct_count=0., confidence_count=0.)
    sources = np.zeros(3, dtype=np.int64)
    decisions = []
    model.train()
    for cpu in batches:
        batch = move_batch(cpu, device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            output = model(batch['x'], batch['hist'], batch['hmask'])
            terms = loss_terms(output, batch, model.cfg, tolerance, n_commit=n_commit)
            geometry = terms['geometry_per_state'].sum()/total
            confidence = terms['confidence_per_state'].sum()/total
            loss = geometry + confidence_weight*confidence
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        if compute_metrics:
            decisions.extend(decision_rows(output, batch, model.cfg, n_commit, tolerance))
        for key, value in (('loss', loss), ('geometry', geometry), ('confidence_loss', confidence)):
            sums[key] += value.detach().item()
        for key in ('error_sum', 'geometry_count', 'correct_count', 'confidence_count'):
            sums[key] += terms[key].detach().item()
        if 'source' in cpu:
            for source in range(3):
                sources[source] += int((cpu['source'] == source).sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    update_ema(ema, model, step, ema_decay)
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']),
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']),
                fresh_fraction=float(sources[0]/total), fixed_fraction=float(sources[1]/total),
                recent_fraction=float(sources[2]/total))
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, min(8, model.cfg.n_future) if n_commit is None else n_commit)
    return sums


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--fixed-bank', help='Optional existing v5 original-fiber recovery bank')
    ap.add_argument('--onpolicy', nargs='*', default=[])
    ap.add_argument('--out-root', default=str(Path(__file__).parents[1]/'output'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--microbatch', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--worker-cache-gb', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--ema-decay', type=float, default=.999)
    ap.add_argument('--confidence-weight', type=float, default=.5)
    ap.add_argument('--tolerance', type=float, default=1.5)
    ap.add_argument('--n-commit', type=int, default=8)
    ap.add_argument('--correction', action=argparse.BooleanOptionalAction, default=True,
                    help='One bounded fine-image correction; disable for the paired one-pass baseline')
    ap.add_argument('--correction-limit', type=float, default=1., help='Maximum lateral correction in trace voxels')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--diag-every', type=int, default=1000)
    ap.add_argument('--diag-max-len', type=float, default=1200.)
    ap.add_argument('--recovery-every', type=int, default=1000, help='Fixed monitor recovery diagnostic cadence; 0 disables')
    ap.add_argument('--recovery-seeds', type=int, default=8, help='First N frozen monitor seeds, four drift bands each')
    ap.add_argument('--recovery-length', type=float, default=32.)
    ap.add_argument('--dagger-every', type=int, default=1000)
    ap.add_argument('--dagger-seeds', type=int, default=64)
    ap.add_argument('--dagger-device')
    ap.add_argument('--dagger-trace-len', type=float, default=6000.)
    ap.add_argument('--replay-keep', type=int, default=4)
    ap.add_argument('--resume', help='Resume last.pt inside this run with the same training options')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.steps, args.batch, args.microbatch, args.log_every, args.ckpt_every,
           args.threads, args.replay_keep, args.dagger_seeds, args.recovery_seeds) < 1 or args.batch % args.microbatch:
        raise ValueError('Positive counts required; microbatch must divide effective batch')
    if min(args.workers, args.warmup, args.diag_every, args.dagger_every, args.recovery_every, args.confidence_weight) < 0:
        raise ValueError('Invalid training settings')
    if not 0 <= args.ema_decay < 1 or min(args.lr, args.tolerance, args.worker_cache_gb,
                                       args.diag_max_len, args.dagger_trace_len, args.recovery_length) <= 0:
        raise ValueError('Invalid loss, learning rate, cache, or rollout settings')
    if args.val_z[0] >= args.val_z[1]:
        raise ValueError('Holdout interval must be increasing')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DirectConfig(correction=args.correction, correction_limit=args.correction_limit)
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Native fine imagery, independently read coarse level-1 imagery.
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                          future_step=cfg.future_step, recent_history_points=cfg.n_history)
    manifest = read_manifest(args.manifest)
    validate_volume_source(spec, manifest)
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(*(v/spec.grid_scale for v in args.val_z))
    train_f, val_f = split_fibers(fibers, band)
    if fiber_manifest(val_f) != manifest['fibers']:
        raise ValueError('Frozen validation geometry differs from dataset/holdout')
    resume = read_checkpoint(args.resume, ARCHITECTURE, args.device) if args.resume else None
    if resume:
        ignored = {'resume', 'device', 'workers', 'threads', 'worker_cache_gb', 'log_every',
                   'ckpt_every', 'diag_every', 'dagger_device'}
        for key, value in vars(args).items():
            if key not in ignored and resume['training_options'].get(key) != value:
                raise ValueError(f'Resume option differs: {key}')
        if resume['seed_manifest_sha256'] != manifest['sha256'] or resume['fiber_manifest'] != fiber_manifest(fibers):
            raise ValueError('Resume data/manifest changed')
    out = prepare_run_dir(args.out_root, args.name, resume is not None)
    if resume and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError('Resume checkpoint must be inside the named run')
    recovery_states = recovery_hash = None
    if args.recovery_every:
        if resume and not (out/'monitor_recovery.npz').exists():
            raise ValueError('Resume requires the original monitor recovery fixture')
        recovery_states, recovery_hash = monitor_fixture(out/'monitor_recovery.npz', val_f, manifest,
                                                         sample, spec, args.recovery_seeds)
        if resume and resume.get('monitor_recovery_sha256') != recovery_hash:
            raise ValueError('Monitor recovery fixture changed since checkpoint')
    model = DirectFollower(cfg).to(args.device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    done = resume_training(resume, model, ema, opt)[0] if resume else 0
    if done >= args.steps:
        raise ValueError('Run has already reached its requested update count')
    replay_index = out/'dagger'/'replay.json'
    replay_paths = json.loads(replay_index.read_text()) if resume and replay_index.exists() else args.onpolicy
    caches = [OnPolicyStates.load(p) for p in replay_paths]
    fixed = [OnPolicyStates.load(args.fixed_bank)] if args.fixed_bank else []
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.dagger_device or args.device,
        every=args.dagger_every, max_seeds=args.dagger_seeds, seed=args.seed, replay_keep=args.replay_keep,
        initial=[c._dir for c in caches], trace_len=args.dagger_trace_len, n_commit=args.n_commit,
        collector_module='vesuvius.neural_tracing.fiber_follow.direct.collect')
    dataset = FollowDataset(train_f, spec, sample, band, chunk=args.microbatch, seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), fixed=fixed, onpolicy=caches,
        replay_index=str(collector.index), batch_builder=ObservationBuilder(cfg), additional_crops=(cfg.coarse,))
    loader_args = dict(batch_size=None, num_workers=args.workers)
    if args.workers:
        loader_args.update(prefetch_factor=2, persistent_workers=True)
    loader = torch.utils.data.DataLoader(dataset, **loader_args)
    if not resume:
        (out/'config.json').write_text(json.dumps(dict(vars(args), architecture=ARCHITECTURE,
            model_cfg=cfg.to_dict(), sample_cfg=asdict(sample), vol_spec=spec.to_dict(),
            coarse_ct_level=1, coarse_ct_grid_scale=8., data_policy=DATA_POLICY,
            monitor_recovery_sha256=recovery_hash,
            seed_manifest_sha256=manifest['sha256'], fiber_manifest=fiber_manifest(fibers),
            parameter_count=sum(p.numel() for p in model.parameters())), indent=2))
    log = RunLog(out/'log.jsonl')
    tracer = None
    recovery_vol = FiberVolume(spec) if recovery_states is not None else None
    started = time.monotonic()
    try:
        if args.diag_every:
            from vesuvius.neural_tracing.fiber_follow.trace import TraceParams
            tracer = DirectTracer(ema, FiberVolume(spec), cfg.fine, cfg.n_history,
                TraceParams(n_commit=args.n_commit, max_len=args.diag_max_len), device=args.device)
        iterator = iter(loader)
        for step in range(done+1, args.steps+1):
            event = collector.poll()
            if event:
                log.record(dict(step=step, **event))
            batches = [next(iterator) for _ in range(args.batch//args.microbatch)]
            lr = lr_at(step, args.lr, args.warmup, args.steps)
            metrics = optimizer_update(model, ema, opt, batches, step, lr, device=args.device,
                tolerance=args.tolerance, confidence_weight=args.confidence_weight, ema_decay=args.ema_decay,
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps)
            if step % args.log_every == 0 or step == args.steps:
                log.record(dict(step=step, lr=lr, **metrics,
                    samples_per_second=(step-done)*args.batch/(time.monotonic()-started)))

            def save(path, resumable=False):
                extra = dict(step=step, tolerance=args.tolerance, n_commit=args.n_commit,
                    seed_manifest_sha256=manifest['sha256'], training_options=vars(args),
                    monitor_recovery_sha256=recovery_hash,
                    fiber_manifest=fiber_manifest(fibers))
                if resumable:
                    extra.update(optimizer=opt.state_dict(), rng=training_rng_state())
                save_checkpoint(path, model, ema, spec, sample, extra)

            if step < args.steps and collector.launch(step, save):
                log.record(dict(step=step, dagger_launched=True))
            if tracer is not None and step % args.diag_every == 0:
                from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate
                from vesuvius.neural_tracing.fiber_follow.experiment import rollout_summary
                for threshold in (.5, .85):
                    tracer.p.confidence = threshold
                    rows, _ = evaluate(tracer, val_f, manifest['monitor'], batch=1)
                    log.record(dict(step=step, split='monitor', threshold=threshold, **rollout_summary(rows)))
            if recovery_states is not None and step % args.recovery_every == 0:
                report = evaluate_monitor(ema, recovery_vol, recovery_states, val_f, sample, device=args.device,
                    n_commit=args.n_commit, tolerance=args.tolerance, recovery_length=args.recovery_length)
                report.update(step=step, split='monitor', fixture_sha256=recovery_hash)
                folder = out/'recovery'
                folder.mkdir(exist_ok=True)
                (folder/f'monitor_{step:06d}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                log.record(dict(step=step, split='monitor', recovery={k: v for k, v in report.items() if k != 'rows'}))
            if step % args.ckpt_every == 0 or step == args.steps:
                save(out/f'ckpt_{step:06d}.pt', resumable=True)
                save(out/'last.pt', resumable=True)
    finally:
        event = collector.close()
        if event:
            log.record(event)
        if tracer is not None:
            tracer.close()
        log.close()
    return str(out/'last.pt')


if __name__ == '__main__':
    main()
