"""Train a direct curve follower from scratch, with original-fiber online replay."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import (
    DATA_POLICY, FollowDataset, OnPolicyStates, SampleConfig, ZBand, fiber_manifest, load_fibers, split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.shared.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.shared.runloop import (
    RunLog, lr_at, prepare_run_dir, read_checkpoint, save_checkpoint as write_checkpoint,
    update_ema, training_rng_state, resume_training, raise_open_file_limit,
)
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE, IDENTITY_ARCHITECTURE, DirectConfig, DirectFollower, IdentityConfig, IdentityFollower,
    config_class, follower_class,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, ObservationBuilder, DirectTracer, LOCATION_SOURCES,
    load_contacts, load_hard_spans,
)
from vesuvius.neural_tracing.fiber_follow.shared.components import PAIR_SAMPLING_VERSION, ComponentRule
from vesuvius.neural_tracing.fiber_follow.regression.supervision import commit_window, loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import (
    decision_rows, summarize_decisions, identity_ranking, summarize_ranking,
)
from vesuvius.neural_tracing.fiber_follow.regression.recovery import monitor_fixture, evaluate_monitor
from vesuvius.neural_tracing.fiber_follow.shared.training_log import format_training_log


def validate_volume_source(spec, manifest):
    """Allow a different CT pyramid level, retaining frozen physical data/seeds."""
    for key in ('fiber_zarr_dir', 'ct_zarr', 'fiber_level', 'grid_scale', 'inputs'):
        if spec.to_dict()[key] != manifest['volume'][key]:
            raise ValueError(f'Volume source {key} differs from frozen manifest')


def save_checkpoint(path, model, ema, spec, sample, extra=None):
    # Atomic publication: collectors must never open a partial checkpoint.
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    write_checkpoint(temporary, model, spec, sample.crop, sample.n_history, model.architecture,
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


ARCHITECTURES = (ARCHITECTURE, IDENTITY_ARCHITECTURE)


def checkpoint_config(ck):
    """Model configuration of either direct architecture.

    ``rich_path_context`` was retired with its ``True`` behavior kept; older
    checkpoints recording that value load unchanged.
    """
    values = dict(ck['model_cfg'])
    if values.pop('rich_path_context', True) is not True:
        raise ValueError('Checkpoints without rich path context are no longer supported')
    if ck['architecture'] == IDENTITY_ARCHITECTURE:
        values.setdefault('appearance_version', 1)
    return config_class(ck['architecture'])(**values)


def load_checkpoint(path, device='cuda'):
    ck = read_checkpoint(path, ARCHITECTURES, device)
    cfg = checkpoint_config(ck)
    model = follower_class(ck['architecture'])(cfg).to(device, memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    if ck.get('coarse_ct_level') != 1 or ck.get('coarse_ct_grid_scale') != 8.:
        raise ValueError('Unsupported coarse image source')
    return model, cfg.fine, cfg.n_history, FiberVolumeSpec(**ck['vol_spec']), ck


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items()}


@torch.no_grad()
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device):
    """Plot EMA proposals and the same monitor rollouts used for logged metrics."""
    from vesuvius.neural_tracing.fiber_follow.shared.diag import plot_batch, plot_curves, plot_refinement, plot_rollouts
    from vesuvius.neural_tracing.fiber_follow.shared.evaluate import evaluate
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary

    images = Path(out)/'images'
    images.mkdir(exist_ok=True)
    # Bound image size even when training with larger microbatches.
    def take(value):
        return {k: take(v) for k, v in value.items()} if isinstance(value, dict) else value[:6]
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
        plot_curves(Path(out)/'log.jsonl', Path(out)/'curves.png', loss_key='geometry')
    finally:
        model.train(was_training)
        tracer.p.confidence = threshold_before


IDENTITY_SUMS = ('identity_count', 'identity_states', 'identity_rank_correct', 'identity_correct_count',
                 'identity_flipped_count')


def optimizer_update(model, ema, opt, batches, step, lr, *, device='cpu', tolerance=1.5,
                     confidence_weight=.5, ema_decay=.999, n_commit=None, compute_metrics=True,
                     identity_weight=.5, identity_temperature=.1):
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
    sources = np.zeros(4, dtype=np.int64)
    identity = {}
    decisions, rankings = [], []
    model.train()
    for cpu in batches:
        batch = move_batch(cpu, device)
        queries = dict(queries=batch['identity_points']) if 'identity_points' in batch else {}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            output = model(batch['x'], batch['hist'], batch['hmask'], **queries)
            terms = loss_terms(output, batch, model.cfg, tolerance, n_commit=n_commit,
                               identity_temperature=identity_temperature)
            geometry = terms['geometry_per_state'].sum()/total
            confidence = terms['confidence_per_state'].sum()/total
            loss = geometry + confidence_weight*confidence
            if 'identity_per_state' in terms:
                identity_loss = terms['identity_per_state'].sum()/total
                loss = loss + identity_weight*identity_loss
                identity['identity_loss'] = identity.get('identity_loss', 0.)+identity_loss.detach().item()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        if compute_metrics:
            decisions.extend(decision_rows(output, batch, model.cfg, n_commit, tolerance))
            if 'query_embedding' in output:
                rankings.append(identity_ranking(output, batch, model.cfg))
        for key in IDENTITY_SUMS:
            if key in terms:
                identity[key] = identity.get(key, 0.)+terms[key].detach().item()
        for key in ('presence_dropped', 'foreign_components'):
            if key in cpu:
                identity[key] = identity.get(key, 0.)+float((cpu[key] > 0).sum())
        if 'negative_bank_shards' in cpu:
            low,high = int(cpu['negative_bank_shards'].min()),int(cpu['negative_bank_shards'].max())
            identity['negative_bank_shards_min'] = min(identity.get('negative_bank_shards_min',low),low)
            identity['negative_bank_shards_max'] = max(identity.get('negative_bank_shards_max',high),high)
        if 'location_source' in cpu:
            for index, name in enumerate(LOCATION_SOURCES):
                key = f'location_{name}'
                identity[key] = identity.get(key, 0.)+float((cpu['location_source'] == index).sum())
        for key, value in (('loss', loss), ('geometry', geometry), ('confidence_loss', confidence)):
            sums[key] += value.detach().item()
        for key in ('error_sum', 'geometry_count', 'correct_count', 'confidence_count'):
            sums[key] += terms[key].detach().item()
        if 'source' in cpu:
            for source in range(4):
                sources[source] += int((cpu['source'] == source).sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    update_ema(ema, model, step, ema_decay)
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']),
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']),
                fresh_fraction=float(sources[0]/total), fixed_fraction=float(sources[1]/total),
                recent_fraction=float(sources[2]/total),bank_wrong_continuation_fraction=float(sources[3]/total))
    if identity:
        # Versioned separately from the distance metrics above.
        identity.update(identity_version=1, pair_sampling_version=PAIR_SAMPLING_VERSION,
                        identity_rank_accuracy=identity.get('identity_rank_correct', 0.)/max(1., identity.get('identity_count', 0.)),
                        identity_prefix_correct_fraction=identity.get('identity_correct_count', 0.)/max(1., sums['confidence_count']))
        for key in ('presence_dropped', 'foreign_components', *(f'location_{n}' for n in LOCATION_SOURCES)):
            if key in identity:
                identity[key+'_fraction'] = identity.pop(key)/total
        sums['identity'] = identity
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, commit_window(model.cfg, n_commit))
        if rankings:
            sums['identity']['ranking'] = summarize_ranking(rankings)
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
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--microbatch', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6)
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
    ap.add_argument('--identity', action=argparse.BooleanOptionalAction, default=False,
                    help=f'Train {IDENTITY_ARCHITECTURE}: CT-only visual history and identity objective')
    ap.add_argument('--identity-weight', type=float, default=.5, help='InfoNCE coefficient')
    ap.add_argument('--identity-temperature', type=float, default=.1)
    ap.add_argument('--appearance-channels', type=int, default=32)
    ap.add_argument('--embedding', type=int, default=32)
    ap.add_argument('--negative-threshold', type=float, default=ComponentRule().threshold, help='Presence threshold for components')
    ap.add_argument('--negative-bank', help='Live native-traced negative bank directory (or bank.json); use its validated paths for identity negatives')
    ap.add_argument('--negative-bank-refresh-seconds', type=float, default=30., help='Each loader worker checks for completed new negative shards at this interval')
    ap.add_argument('--negative-bank-cache-mb', type=float, default=64., help='Maximum cached negative geometry per loader worker')
    ap.add_argument('--bank-wrong-continuation-probability', type=float, default=.75,
                    help='Fraction of recent DAgger departure draws to replace with safe bank continuations when available')
    ap.add_argument('--presence-dropout', type=float, default=.25, help='Probability of zeroing both presence crops')
    ap.add_argument('--anchor-prob', type=float, default=.75, help='Training states given seed-segment anchors')
    ap.add_argument('--contacts', help='Mined contact episodes of the training fibers (oversampled)')
    ap.add_argument('--hard-spans', help='Hard controlled spans by fiber name (oversampled)')
    ap.add_argument('--contact-fraction', type=float, default=.2, help='Fresh draws near contact episodes')
    ap.add_argument('--hard-span-fraction', type=float, default=.1)
    ap.add_argument('--lateral-fraction', type=float, default=.1,
                    help='Fresh draws near earlier states with lateral presence components')
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                    help='Compile follower training on CUDA (EMA, diagnostics and collection stay eager)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--diag-every', type=int, default=1000)
    ap.add_argument('--diag-max-len', type=float, default=400.)
    ap.add_argument('--long-diag-every', type=int, default=0, help='Additional long monitor rollouts; 0 disables')
    ap.add_argument('--long-diag-max-len', type=float, default=1200.)
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
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    launch_started = time.monotonic()

    def progress(message):
        print(f'[{args.name} +{time.monotonic()-launch_started:.1f}s] {message}', flush=True)

    progress(f'Starting training: device={args.device}, workers={args.workers}')
    if args.resume and args.init_tracer:
        raise ValueError('--init-tracer starts a new run and cannot be combined with --resume')
    if min(args.steps, args.batch, args.microbatch, args.log_every, args.ckpt_every,
           args.threads, args.replay_keep, args.dagger_seeds, args.recovery_seeds) < 1 or args.batch % args.microbatch:
        raise ValueError('Positive counts required; microbatch must divide effective batch')
    if min(args.workers, args.warmup, args.diag_every, args.long_diag_every, args.dagger_every, args.recovery_every, args.confidence_weight) < 0:
        raise ValueError('Invalid training settings')
    if not 0 <= args.ema_decay < 1 or min(args.lr, args.tolerance, args.worker_cache_gb,
                                       args.diag_max_len, args.long_diag_max_len, args.dagger_trace_len, args.recovery_length) <= 0:
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
    options = dict(channels=args.channels, layers=args.decoder_layers, correction=args.correction,
                   correction_limit=args.correction_limit, correction_steps=args.correction_steps)
    cfg = (IdentityConfig(**options, appearance_channels=args.appearance_channels, embedding=args.embedding)
           if args.identity else DirectConfig(**options))
    identity_sampling = IdentitySampling(
        rule=ComponentRule(threshold=args.negative_threshold), anchor_prob=args.anchor_prob,
        presence_dropout=args.presence_dropout, contact_fraction=args.contact_fraction,
        hard_span_fraction=args.hard_span_fraction, lateral_fraction=args.lateral_fraction,
        bank_wrong_continuation_probability=args.bank_wrong_continuation_probability)
    initialized = None
    if args.init_tracer:
        progress(f'Loading follower weights from {args.init_tracer}')
        initialized, _, _, _, _ = load_checkpoint(args.init_tracer, args.device)
        cfg = initialized.cfg
    if args.resume:
        progress(f'Loading resume checkpoint from {args.resume}')
        cfg = checkpoint_config(read_checkpoint(args.resume, ARCHITECTURES, args.device))
    identity_enabled = isinstance(cfg,IdentityConfig)
    if min(args.identity_weight,args.identity_temperature) <= 0 and identity_enabled:
        raise ValueError('Identity weight and temperature must be positive')
    if args.negative_bank and not identity_enabled:
        raise ValueError('--negative-bank requires an identity follower')
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Native fine imagery, independently read coarse level-1 imagery.
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4.,
                           inputs='ct+presence')
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
    resume = read_checkpoint(args.resume, ARCHITECTURES, args.device) if args.resume else None
    negative_bank = None
    if args.negative_bank:
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
        negative_bank = NeighborBank(args.negative_bank,train_f,band,grid_scale=spec.grid_scale,
            refresh_seconds=args.negative_bank_refresh_seconds,cache_bytes=int(args.negative_bank_cache_mb*(1 << 20)))
        negative_bank.validate_volume(spec)
        if resume and (resume.get('negative_bank_provenance') is not None or resume['training_options'].get('negative_bank')):
            negative_bank.validate_resume(resume.get('negative_bank_provenance'))
        progress(f'Live negative bank: {negative_bank.shard_count} published shards, refresh every {args.negative_bank_refresh_seconds:g}s per worker')
    if resume:
        # The run directory may move; the checkpoint must still sit inside the named run.
        ignored = {'resume', 'out_root', 'device', 'batch', 'microbatch', 'workers', 'threads', 'worker_cache_gb',
                   'log_every', 'ckpt_every', 'diag_every', 'dagger_device', 'compile', 'init_tracer',
                   'negative_bank_refresh_seconds','negative_bank_cache_mb'}
        parser = build_parser()
        for key, value in vars(args).items():
            if key in ('long_diag_every', 'long_diag_max_len') and key not in resume['training_options'] and not args.long_diag_every:
                continue
            # Options added after a run started resume at their defaults.
            if key not in resume['training_options'] and value == parser.get_default(key):
                continue
            # Explicitly attaching a bank to an older run is permitted. Once
            # attached, its immutable run identity is checked on every resume.
            if key == 'negative_bank' and not resume['training_options'].get(key) and value:
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
    follower = IdentityFollower if isinstance(cfg, IdentityConfig) else DirectFollower
    model = initialized if initialized is not None else follower(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    done = resume_training(resume, model, ema, opt)[0] if resume else 0
    match_optimizer_layout(opt)
    # The compiled wrapper shares the module's parameters, so EMA updates, gradient
    # clipping and checkpoints keep using ``model``; only the training forward is compiled.
    trainable = torch.compile(model) if args.compile and torch.device(args.device).type == 'cuda' else model
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
        collector_module='vesuvius.neural_tracing.fiber_follow.regression.collect')
    if identity_enabled:
        contacts = load_contacts(args.contacts, train_f, band) if args.contacts else ()
        hard_spans = load_hard_spans(args.hard_spans, train_f) if args.hard_spans else ()
        progress(f'Identity sampling: {len(contacts)} contact episodes, {len(hard_spans)} hard spans')
        builder = IdentityObservationBuilder(cfg, train_f, identity_sampling, contacts=contacts,
                                             hard_spans=hard_spans, augment=True,negative_bank=negative_bank)
    else:
        builder = ObservationBuilder(cfg)
    dataset = FollowDataset(train_f, spec, sample, band, chunk=args.microbatch, seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), fixed=fixed, onpolicy=caches,
        replay_index=str(collector.index), batch_builder=builder, additional_crops=(cfg.coarse,))
    loader_args = dict(batch_size=None, num_workers=args.workers,
                       pin_memory=torch.device(args.device).type == 'cuda')
    if args.workers:
        loader_args.update(prefetch_factor=2, persistent_workers=True)
    loader = torch.utils.data.DataLoader(dataset, **loader_args)
    if not resume:
        (out/'config.json').write_text(json.dumps(dict(vars(args), architecture=model.architecture,
            identity_sampling=asdict(identity_sampling) if identity_enabled else None,
            negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
            model_cfg=cfg.to_dict(), sample_cfg=asdict(sample), vol_spec=spec.to_dict(),
            coarse_ct_level=1, coarse_ct_grid_scale=8., data_policy=DATA_POLICY,
            monitor_recovery_sha256=recovery_hash,
            seed_manifest_sha256=manifest['sha256'], fiber_manifest=fiber_manifest(fibers),
            parameter_count=sum(p.numel() for p in model.parameters())), indent=2))
    log = RunLog(out/'log.jsonl', formatter=format_training_log)
    if identity_enabled:
        log.record(dict(step=done, event='identity_sampling', pair_sampling_version=PAIR_SAMPLING_VERSION,
                        resume=args.resume, history_policy='full_shared_target_history',
                        negative_source='native_path_bank_v1' if negative_bank else 'crop_components',
                        negative_bank_provenance=negative_bank.provenance() if negative_bank else None))
    tracer = None
    recovery_vol = FiberVolume(spec) if recovery_states is not None else None
    started = time.monotonic()
    try:
        if args.diag_every or args.long_diag_every:
            from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
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
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps,
                identity_weight=args.identity_weight, identity_temperature=args.identity_temperature)
            if early:
                progress(f'Update {step} complete in {time.monotonic()-update_started:.1f}s; loss={metrics["loss"]:.5f}')
                if step == done+5:
                    progress(f'Startup progress complete; regular metrics every {args.log_every} updates')
            if step % args.log_every == 0 or step == args.steps:
                log.record(dict(step=step, lr=lr, **metrics, samples_seen=step*args.batch,
                    train_seconds=time.monotonic()-started,
                    samples_per_second=(step-done)*args.batch/(time.monotonic()-started)))

            def save(path, resumable=False):
                extra = dict(step=step, tolerance=args.tolerance, n_commit=args.n_commit,
                    identity_sampling=asdict(identity_sampling) if identity_enabled else None,
                    negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
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
                if resumable:
                    extra.update(optimizer=opt.state_dict(), rng=training_rng_state())
                save_checkpoint(path, model, ema, spec, sample, extra)

            if step < args.steps and collector.launch(step, save):
                log.record(dict(step=step, dagger_launched=True))
            periodic = {}
            if tracer is not None and args.diag_every and step % args.diag_every == 0:
                began = time.monotonic()
                training_diagnostics(ema, batches[-1], tracer, val_f, manifest['monitor'], out, step, log,
                                     device=args.device)
                periodic['diagnostics_seconds'] = time.monotonic()-began
            if tracer is not None and args.long_diag_every and step % args.long_diag_every == 0:
                from vesuvius.neural_tracing.fiber_follow.shared.evaluate import evaluate
                from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary
                from vesuvius.neural_tracing.fiber_follow.shared.diag import plot_rollouts
                began = time.monotonic()
                original_length, original_threshold = tracer.p.max_len, tracer.p.confidence
                tracer.p.max_len, tracer.p.confidence = args.long_diag_max_len, .5
                traces = []
                try:
                    rows, _ = evaluate(tracer, val_f, manifest['monitor'], batch=1,
                        coverage_max_len=args.long_diag_max_len,
                        on_trace=lambda seed, path, reason: traces.append((path, reason)))
                    log.record(dict(step=step, split='monitor_long', threshold=.5,
                                    coverage_max_len=args.long_diag_max_len, **rollout_summary(rows)))
                    if traces:
                        paths, reasons = zip(*traces)
                        (out/'images').mkdir(exist_ok=True)
                        plot_rollouts(tracer.vol, val_f, manifest['monitor'], paths, reasons,
                            out/'images'/f'rollout_long_{step:06d}_c0.5.png', args.long_diag_max_len, rows=rows)
                finally:
                    tracer.p.max_len, tracer.p.confidence = original_length, original_threshold
                periodic['long_diagnostics_seconds'] = time.monotonic()-began
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
