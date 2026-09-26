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
from vesuvius.neural_tracing.fiber_follow.direct.supervision import commit_window, loss_terms
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
                     dict({'n_commit': commit_window(model.cfg, None), **(extra or {})},
                          ema=ema.state_dict(), sample_cfg=asdict(sample),
                          coarse_ct_level=1, coarse_ct_grid_scale=8.))
    temporary.replace(path)


def conv_memory_format(device):
    """cuDNN's 3-D convolutions are much faster on channels-last activations.

    Returns the memory format the model's parameters should use on ``device``.
    With contiguous (NCDHW) tensors cuDNN falls back to a slow direct backward
    kernel for these small channel counts.
    """
    return torch.channels_last_3d if torch.device(device).type == 'cuda' else torch.contiguous_format


def match_optimizer_layout(opt):
    """Resumed moment estimates take their parameter's memory format."""
    for param, state in opt.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == param.shape:
                state[key] = torch.empty_like(param).copy_(value)


def compile_training_judge(judge):
    """Compile CNN batches and sequence decoding after EMA copy/resume.

    Leave the view-batch loop outside the graph: growing histories change its
    iteration count. Dynamic shapes cover the final batch and query count.
    Bound methods share the original parameters and preserve checkpoint keys.
    """
    import torch._functorch.config
    # optimizer_update performs backward outside its forward autocast context.
    torch._functorch.config.backward_pass_autocast = 'off'
    for name in ('encode_views', 'decode'):
        setattr(judge, name, torch.compile(getattr(judge, name), dynamic=True))
    return judge


def load_checkpoint(path, device='cuda'):
    ck = read_checkpoint(path, ARCHITECTURE, device)
    # Historical checkpoints retain their exact one-pass/one-correction model,
    # including the old path heads. Active legacy collectors still load these.
    cfg = DirectConfig(**{'correction': False, 'correction_steps': 1,
                          'rich_path_context': False, **ck['model_cfg']})
    model = DirectFollower(cfg).to(device, memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    if ck.get('coarse_ct_level') != 1 or ck.get('coarse_ct_grid_scale') != 8.:
        raise ValueError('Unsupported coarse image source')
    return model, cfg.fine, cfg.n_history, FiberVolumeSpec(**ck['vol_spec']), ck


def raise_open_file_limit():
    """Many loader workers share tensors through file descriptors; use the hard limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else [move_batch(s, device) for s in v] if isinstance(v, list) else v.to(device, non_blocking=True)
            for k, v in batch.items()}


@torch.no_grad()
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device,
                         judge=None, judge_slices=None, judge_policy=None):
    """Plot EMA proposals and the same monitor rollouts used for logged metrics."""
    from vesuvius.neural_tracing.fiber_follow.diag import plot_batch, plot_curves, plot_refinement, plot_rollouts
    from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate
    from vesuvius.neural_tracing.fiber_follow.experiment import rollout_summary

    images = Path(out)/'images'
    images.mkdir(exist_ok=True)
    # Bound image size even when training with larger microbatches.
    def take(value):
        return {k: take(v) for k, v in value.items() if k != 'judge'} if isinstance(value, dict) else value[:6]
    batch = move_batch(take(cpu_batch), device)
    was_training = model.training
    threshold_before = tracer.p.confidence
    model.eval()
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            prediction = model(batch['x'], batch['hist'], batch['hmask'])
        points = prediction['points']
        target = torch.cat((batch['plane_ab'], points[..., 2:]), -1)
        for scale, crop in (('fine', model.cfg.fine), ('coarse', model.cfg.coarse)):
            filename = f'batch_{step:06d}.png' if scale == 'fine' else f'batch_coarse_{step:06d}.png'
            plot_batch(batch['x'][scale], points, target, batch['plane_mask'], crop, images/filename,
                       batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                       source=batch['source'], offtrack=batch['offtrack'], confidence=prediction['confidence'],
                       history_channel=None)
        curves = prediction['refinement_points']
        labels = (['initial proposal']+[f'correction {i}' for i in range(1, curves.shape[1]-1)]+
                  ['corrected proposal']) if model.cfg.correction else ['proposal']
        plot_refinement(curves, batch['hist'], batch['hmask'],
                        images/f'correction_{step:06d}.png',
                        labels=labels,
                        target=target, target_mask=batch['plane_mask'])
        for threshold in (.5,):
            if not seeds:
                break
            tracer.p.confidence = threshold
            traces = []
            rows, _ = evaluate(tracer, fibers, seeds, batch=1,
                               coverage_max_len=tracer.p.max_len,
                               on_trace=lambda seed, path, reason: traces.append((path, reason)))
            paths, reasons = zip(*traces)
            plot_rollouts(tracer.vol, fibers, seeds, paths, reasons,
                          images/f'rollout_{step:06d}_c{threshold}.png', tracer.p.max_len, rows=rows)
            log.record(dict(step=step, split='monitor', threshold=threshold,
                            coverage_max_len=tracer.p.max_len, **rollout_summary(rows)))
            if judge is not None:
                from .judge_evaluation import paired_monitor
                from .judge_model import sequence_tensors
                from .diagnostics import plot_judge_sequence, plot_judge_path
                judged = DirectTracer(model, tracer.vol, tracer.crop, tracer.n_history, tracer.p,
                                      device=device, judge=judge, judge_slices=judge_slices, judge_policy=judge_policy)
                try:
                    report, audits = paired_monitor(judged, fibers, seeds, paths)
                    log.record(dict(step=step, split='monitor', threshold=threshold, judge=report['metrics']))
                    (images/f'judge_rollouts_{step:06d}_c{threshold}.json').write_text(json.dumps(report, default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist()))
                    for index, state in enumerate(audits[:3]):
                        sequence = sequence_tensors(state['records'], device)
                        logits = judge(**sequence)
                        target, known = state['events'].labels([r['arc'] for r in state['records']])
                        sequence.update(target=torch.as_tensor(target)[None], known=torch.as_tensor(known)[None])
                        plot_judge_sequence(sequence, logits, images/f'judge_rollout_{step:06d}_c{threshold}_{index}.png',
                                            audit=state['policy'].audit[-1])
                        plot_judge_path(state['observed_path'], state['policy'].accepted, state['events'],
                                        images/f'judge_path_{step:06d}_c{threshold}_{index}.png')
                finally:
                    judged.close()
        plot_curves(Path(out)/'log.jsonl', Path(out)/'curves.png', loss_key='geometry')
    finally:
        model.train(was_training)
        tracer.p.confidence = threshold_before


def optimizer_update(model, ema, opt, batches, step, lr, *, device='cpu', tolerance=1.5,
                     confidence_weight=.5, ema_decay=.999, n_commit=None, compute_metrics=True,
                     judge=None, judge_ema=None, judge_weight=.5):
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
    judge_total = sum(len(b.get('judge', [])) for b in batches)
    sums.update(judge_loss=0., judge_sequences=judge_total, judge_positive=0, judge_departed=0, judge_unknown=0)
    judge_sources = np.zeros(5, dtype=int)
    allocation = sum((b.get('judge_allocation', torch.zeros(3)).numpy() for b in batches), np.zeros(3))
    sums['judge_synthetic_attempts'], sums['judge_synthetic_fallback'] = map(int, allocation[1:])
    if judge is not None:
        judge.train()
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
        if judge is not None:
            from .judge_supervision import masked_bce
            for sequence in batch.get('judge', []):
                judge_sources[int(sequence.get('source', torch.tensor([0]))[0])] += 1
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                    logits = judge(**{k: sequence[k] for k in ('images', 'metadata', 'valid', 'queries')})
                    jl = masked_bce(logits, sequence['target'], sequence['known'], sequence['eligible']).sum()/max(1, judge_total)
                (judge_weight*jl).backward()
                sums['judge_loss'] += jl.detach().item()
                mask = sequence['known'] & sequence['eligible']
                sums['judge_positive'] += int((mask & (sequence['target'] > .5)).sum())
                sums['judge_departed'] += int((mask & (sequence['target'] <= .5)).sum())
                sums['judge_unknown'] += int((~mask).sum())
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
    if judge is not None:
        torch.nn.utils.clip_grad_norm_(judge.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    update_ema(ema, model, step, ema_decay)
    if judge is not None:
        update_ema(judge_ema, judge, step, ema_decay)
        sums['loss'] += judge_weight*sums['judge_loss']
        sums['judge_source_fractions'] = dict(zip(('fresh','fixed','recent','synthetic_switch','matched_contact'),
                                                  (judge_sources/max(1,judge_total)).tolist()))
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']),
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']),
                fresh_fraction=float(sources[0]/total), fixed_fraction=float(sources[1]/total),
                recent_fraction=float(sources[2]/total))
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, commit_window(model.cfg, n_commit))
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
    ap.add_argument('--n-commit', type=int, default=4)
    ap.add_argument('--channels', type=int, default=24, help='Base image encoder width')
    ap.add_argument('--decoder-layers', type=int, default=4)
    ap.add_argument('--correction', action=argparse.BooleanOptionalAction, default=True,
                    help='Refine the curve using refreshed local and deep image evidence')
    ap.add_argument('--correction-steps', type=int, default=2)
    ap.add_argument('--correction-limit', type=float, default=1., help='Maximum lateral correction per step in trace voxels')
    ap.add_argument('--no-history-prob', type=float, default=.15, help='Fresh-state probability of absent observed history')
    ap.add_argument('--short-history-prob', type=float, default=.4,
                    help='Given history is present, probability of a balanced 1-8/9-32 point startup history')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                    help='Compile follower and judge training on CUDA (EMA, diagnostics and collection stay eager)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--diag-every', type=int, default=1000)
    ap.add_argument('--diag-max-len', type=float, default=400.)
    ap.add_argument('--recovery-every', type=int, default=1000, help='Fixed monitor recovery diagnostic cadence; 0 disables')
    ap.add_argument('--recovery-seeds', type=int, default=8, help='First N frozen monitor seeds, four drift bands each')
    ap.add_argument('--recovery-length', type=float, default=32.)
    ap.add_argument('--dagger-every', type=int, default=1000)
    ap.add_argument('--dagger-seeds', type=int, default=64)
    ap.add_argument('--dagger-device')
    ap.add_argument('--dagger-trace-len', type=float, default=6000.)
    ap.add_argument('--replay-keep', type=int, default=4)
    ap.add_argument('--resume', help='Resume last.pt inside this run with the same training options')
    ap.add_argument('--init-tracer', help='Initialize a new run from saved EMA follower weights')
    from .judge_options import add_judge_options
    add_judge_options(ap)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    launch_started = time.monotonic()

    def progress(message):
        print(f'[{args.name} +{time.monotonic()-launch_started:.1f}s] {message}', flush=True)

    progress(f'Starting training: device={args.device}, workers={args.workers}, judge={args.judge}')
    if args.resume and args.init_tracer:
        raise ValueError('--init-tracer starts a new run and cannot be combined with --resume')
    if min(args.judge_loss_weight, args.judge_synthetic_fraction) < 0:
        raise ValueError('Judge allocations and loss weight must be nonnegative')
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
    if not all(0 <= p <= 1 for p in (args.no_history_prob, args.short_history_prob)):
        raise ValueError('History probabilities must be in [0, 1]')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    raise_open_file_limit()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DirectConfig(channels=args.channels, layers=args.decoder_layers,
                       correction=args.correction, correction_limit=args.correction_limit,
                       correction_steps=args.correction_steps)
    initialized = None
    if args.init_tracer:
        progress(f'Loading follower weights from {args.init_tracer}')
        initialized, _, _, _, _ = load_checkpoint(args.init_tracer, args.device)
        cfg = initialized.cfg
    if args.resume:
        progress(f'Loading resume checkpoint from {args.resume}')
        resume_config = read_checkpoint(args.resume, ARCHITECTURE, args.device)
        cfg = DirectConfig(**resume_config['model_cfg'])
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Native fine imagery, independently read coarse level-1 imagery.
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                          future_step=cfg.future_step, recent_history_points=cfg.n_history,
                          no_history_prob=args.no_history_prob, short_history_prob=args.short_history_prob)
    progress('Loading manifest and fiber annotations')
    manifest = read_manifest(args.manifest)
    validate_volume_source(spec, manifest)
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(*(v/spec.grid_scale for v in args.val_z))
    train_f, val_f = split_fibers(fibers, band)
    progress(f'Loaded {len(train_f)} training fibers and {len(val_f)} validation fibers')
    if fiber_manifest(val_f) != manifest['fibers']:
        raise ValueError('Frozen validation geometry differs from dataset/holdout')
    resume = read_checkpoint(args.resume, ARCHITECTURE, args.device) if args.resume else None
    if resume:
        # The run directory may move; the checkpoint must still sit inside the named run.
        ignored = {'resume', 'out_root', 'device', 'batch', 'microbatch', 'workers', 'threads', 'worker_cache_gb',
                   'log_every', 'ckpt_every', 'diag_every', 'dagger_device', 'compile', 'init_tracer'}
        for key, value in vars(args).items():
            if key.startswith('judge') and key not in resume['training_options'] and not args.judge:
                continue
            if key not in ignored and resume['training_options'].get(key) != value:
                raise ValueError(f'Resume option differs: {key}')
        if resume['seed_manifest_sha256'] != manifest['sha256'] or resume['fiber_manifest'] != fiber_manifest(fibers):
            raise ValueError('Resume data/manifest changed')
    out = prepare_run_dir(args.out_root, args.name, resume is not None)
    if resume and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError('Resume checkpoint must be inside the named run')
    recovery_states = recovery_hash = None
    if args.recovery_every:
        progress('Preparing monitor recovery fixture')
        if resume and not (out/'monitor_recovery.npz').exists():
            raise ValueError('Resume requires the original monitor recovery fixture')
        recovery_states, recovery_hash = monitor_fixture(out/'monitor_recovery.npz', val_f, manifest,
                                                         sample, spec, args.recovery_seeds)
        if resume and resume.get('monitor_recovery_sha256') != recovery_hash:
            raise ValueError('Monitor recovery fixture changed since checkpoint')
    progress('Initializing models and optimizer')
    model = initialized if initialized is not None else DirectFollower(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    judge = judge_ema = slices = policy = None
    if args.judge:
        from .judge_options import configs
        from .judge_model import CTJudge
        slices, judge_cfg, policy = configs(args, spec)
        progress(f'Opening judge CT: {slices.source}, level={slices.level}, pixel spacing={slices.spacing:g}')
        judge_source = slices.open().identity
        if resume and resume.get('judge_source_sha256') != judge_source:
            raise ValueError('Resume native CT source metadata changed')
        judge = CTJudge(judge_cfg).to(args.device)
        judge_ema = copy.deepcopy(judge).requires_grad_(False).eval()
        if resume:
            judge.load_state_dict(resume['judge'])
            judge_ema.load_state_dict(resume['judge_ema'])
    groups = [dict(params=model.parameters())]
    if judge is not None:
        groups.append(dict(params=judge.parameters()))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    done = resume_training(resume, model, ema, opt)[0] if resume else 0
    match_optimizer_layout(opt)
    # The compiled wrapper shares the module's parameters, so EMA updates, gradient
    # clipping and checkpoints keep using ``model``; only the training forward is compiled.
    trainable = torch.compile(model) if args.compile and torch.device(args.device).type == 'cuda' else model
    if judge is not None and args.compile and torch.device(args.device).type == 'cuda':
        compile_training_judge(judge)
    if args.compile and torch.device(args.device).type == 'cuda':
        progress('Compilation enabled; first forward/backward passes will compile lazily and may take several minutes')
    if done >= args.steps:
        raise ValueError('Run has already reached its requested update count')
    replay_index = out/'dagger'/'replay.json'
    replay_paths = json.loads(replay_index.read_text()) if resume and replay_index.exists() else args.onpolicy
    progress('Loading replay banks and preparing data loader')
    caches = [OnPolicyStates.load(p) for p in replay_paths]
    fixed = [OnPolicyStates.load(args.fixed_bank)] if args.fixed_bank else []
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.dagger_device or args.device,
        every=args.dagger_every, max_seeds=args.dagger_seeds, seed=args.seed, replay_keep=args.replay_keep,
        initial=[c._dir for c in caches], trace_len=args.dagger_trace_len, n_commit=args.n_commit,
        collector_module='vesuvius.neural_tracing.fiber_follow.direct.collect')
    builder = ObservationBuilder(cfg)
    if args.judge:
        from .judge_supervision import JointObservationBuilder
        builder = JointObservationBuilder(builder, slices, band, args.judge_synthetic_fraction, train_f)
    dataset = FollowDataset(train_f, spec, sample, band, chunk=args.microbatch, seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), fixed=fixed, onpolicy=caches,
        replay_index=str(collector.index), batch_builder=builder, additional_crops=(cfg.coarse,))
    loader_args = dict(batch_size=None, num_workers=args.workers,
                       pin_memory=torch.device(args.device).type == 'cuda')
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
        progress(f'Starting data loader; waiting for {args.batch//args.microbatch} microbatches for update {done+1}')
        iterator = iter(loader)
        for step in range(done+1, args.steps+1):
            event = collector.poll()
            if event:
                log.record(dict(step=step, **event))
            early = step <= done+5
            batch_started = time.monotonic()
            if early and step != done+1:
                progress(f'Update {step}: waiting for data')
            batches = []
            for index in range(args.batch//args.microbatch):
                batches.append(next(iterator))
                if step == done+1:
                    progress(f'Update {step}: received microbatch {index+1}/{args.batch//args.microbatch}')
            data_seconds = time.monotonic()-batch_started
            update_started = time.monotonic()
            if early:
                progress(f'Update {step}: data ready in {data_seconds:.1f}s; running forward/backward and optimizer')
            lr = lr_at(step, args.lr, args.warmup, args.steps)
            metrics = optimizer_update(trainable, ema, opt, batches, step, lr, device=args.device,
                tolerance=args.tolerance, confidence_weight=args.confidence_weight, ema_decay=args.ema_decay,
                judge=judge, judge_ema=judge_ema, judge_weight=args.judge_loss_weight,
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps)
            if early:
                progress(f'Update {step} complete in {time.monotonic()-update_started:.1f}s; loss={metrics["loss"]:.5f}')
                if step == done+5:
                    progress(f'Startup progress complete; regular metrics every {args.log_every} updates')
            if step % args.log_every == 0 or step == args.steps:
                log.record(dict(step=step, lr=lr, **metrics,
                    samples_per_second=(step-done)*args.batch/(time.monotonic()-started)))

            def save(path, resumable=False):
                extra = dict(step=step, tolerance=args.tolerance, n_commit=args.n_commit,
                    seed_manifest_sha256=manifest['sha256'], training_options=vars(args),
                    monitor_recovery_sha256=recovery_hash,
                    fiber_manifest=fiber_manifest(fibers))
                if args.init_tracer:
                    import hashlib
                    extra['init_tracer_sha256'] = hashlib.sha256(Path(args.init_tracer).read_bytes()).hexdigest()
                    extra['init_tracer_path'] = str(Path(args.init_tracer).resolve())
                elif resume and 'init_tracer_sha256' in resume:
                    extra['init_tracer_sha256'] = resume['init_tracer_sha256']
                    extra['init_tracer_path'] = resume.get('init_tracer_path')
                if judge is not None:
                    from .judge_model import JUDGE_ARCHITECTURE
                    from ..events import EVENT_VERSION
                    from .judge_slices import SLICE_VERSION
                    extra.update(judge=judge.state_dict(), judge_ema=judge_ema.state_dict(),
                                 judge_architecture=JUDGE_ARCHITECTURE, judge_cfg=judge.cfg.to_dict(),
                                 judge_slices=slices.to_dict(), judge_policy=policy.to_dict(),
                                 judge_source_sha256=judge_source,
                                 judge_event_version=EVENT_VERSION, judge_slice_version=SLICE_VERSION)
                if resumable:
                    extra.update(optimizer=opt.state_dict(), rng=training_rng_state())
                save_checkpoint(path, model, ema, spec, sample, extra)

            if step < args.steps and collector.launch(step, save):
                log.record(dict(step=step, dagger_launched=True))
            periodic = {}
            if tracer is not None and step % args.diag_every == 0:
                began = time.monotonic()
                training_diagnostics(ema, batches[-1], tracer, val_f, manifest['monitor'], out, step, log,
                                     device=args.device, judge=judge_ema, judge_slices=slices, judge_policy=policy)
                if judge_ema is not None:
                    from .diagnostics import plot_judge_sequence
                    with torch.no_grad():
                        for index, sequence in enumerate(batches[-1].get('judge', [])[:3]):
                            device_sequence = move_batch(sequence, args.device)
                            logits = judge_ema(**{k: device_sequence[k] for k in ('images','metadata','valid','queries')})
                            plot_judge_sequence(sequence, logits, out/'images'/f'judge_{step:06d}_{index}.png')
                periodic['diagnostics_seconds'] = time.monotonic()-began
            if recovery_states is not None and step % args.recovery_every == 0:
                began = time.monotonic()
                report = evaluate_monitor(ema, recovery_vol, recovery_states, val_f, sample, device=args.device,
                    n_commit=args.n_commit, tolerance=args.tolerance, recovery_length=args.recovery_length)
                report.update(step=step, split='monitor', fixture_sha256=recovery_hash)
                folder = out/'recovery'
                folder.mkdir(exist_ok=True)
                (folder/f'monitor_{step:06d}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                log.record(dict(step=step, split='monitor', recovery={k: v for k, v in report.items() if k != 'rows'}))
                periodic['recovery_seconds'] = time.monotonic()-began
            if periodic:
                # Wall time spent outside optimizer updates, so throughput can be read from the log.
                log.record(dict(step=step, **periodic))
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
