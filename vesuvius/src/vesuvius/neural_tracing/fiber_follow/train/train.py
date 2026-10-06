"""Train the fiber followers (regression, flow, sequence) with shared online replay.

    python train/train.py --config run.json [--resume CHECKPOINT]

The run configuration (train/run_config.py) holds the dataset, model, training and runtime settings."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import (
    TASKS, FollowDataset, SampleConfig, TaskBudget, fiber_manifest, load_replay, usable_replay)
from vesuvius.neural_tracing.fiber_follow.train.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.tracing.policy import (DEFAULT_CONFIDENCE, DEFAULT_GATE, gate_horizon,
    OperatingPolicy, selection_window)
from vesuvius.neural_tracing.fiber_follow.train.runloop import RunLog, lr_at, prepare_run_dir, read_checkpoint, save_checkpoint as write_checkpoint, update_ema, training_rng_state, resume_training, raise_open_file_limit
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling, FiberTracer, LOCATION_SOURCES
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.supervision import commit_window, loss_terms
from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows, summarize_decisions
from vesuvius.neural_tracing.fiber_follow.evaluation.recovery import monitor_fixture, evaluate_monitor
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import configured_frame_policy, frame_predictor, bind_frame_checkpoint
from vesuvius.neural_tracing.fiber_follow.train.training_log import format_training_log, TrainingInterval, SamplingLedger


def save_checkpoint(path, model, ema, spec, sample, extra=None):
    # Atomic publication: collectors must never open a partial checkpoint.
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    write_checkpoint(temporary, model, spec, sample.crop, sample.n_history, model.model_type,
                     dict({'n_commit': commit_window(model.cfg, None), **(extra or {})},
                          ema=ema.state_dict(), sample_cfg=asdict(sample)))
    temporary.replace(path)


def collector_settings(collector):
    return {name: member.settings() for name, member in collector.collectors}


def training_policy(cfg, n_commit, confidence=DEFAULT_CONFIDENCE, gate=DEFAULT_GATE):
    """The operating policy used for live feedback and recorded with every checkpoint."""
    return OperatingPolicy(confidence=confidence, n_commit=n_commit, max_recovery_distance=cfg.max_recovery_distance,
                           refinement_steps=cfg.recurrent_refinement_steps, gate=gate)


def conv_memory_format(device):
    """Use the shared encoder convolution layout on CUDA."""
    return torch.channels_last_3d if torch.device(device).type == 'cuda' else torch.contiguous_format


def prepare_training(model, batch_size=2, *, backend=None):
    """Compile the training operations in place, keeping one model and parameter set.

    There is no eager training mode. ``backend`` lets tests inspect graph capture.
    Inference uses independently constructed models/EMA, without training setup.
    """
    if hasattr(model, 'training_batch_size'):
        if batch_size != model.training_batch_size:
            raise ValueError('Training batch size cannot change after setup')
        return model
    if batch_size < 1:
        raise ValueError('Positive training batch size required')
    options = dict(dynamic=False, fullgraph=True)
    # Keep eager BF16 rounding without changing global compiler configuration.
    if backend is None or backend == 'inductor':
        options['options'] = dict(emulate_precision_casts=True)
    if backend is not None:
        options['backend'] = backend
    if model.model_type == 'sequence':
        # Episode batches (train/sequence.py) run eagerly; their CNN, which encodes every step's crop, is compiled
        # and fed fixed-size chunks. Compiling forward (not the module) keeps the parameter names.
        model.training_batch_size, model._training_refined, model.training_loss = batch_size, None, loss_terms
        model.cnn.forward = torch.compile(model.cnn.forward, **options)
        return model
    model.training_batch_size = batch_size
    model._training_refined = None
    model.training_forward = torch.compile(model.training_forward, **options)
    model.training_loss = torch.compile(loss_terms, **options)
    return model


def fixed_rows(value, size):
    """Pad with finite copies; the caller discards the extra output rows."""
    if not 0 < len(value) <= size:
        raise ValueError(f'Expected 1..{size} rows, got {len(value)}')
    if len(value) < size:
        value = torch.cat((value, value[:1].expand(size-len(value), *value.shape[1:])))
    stride = 1
    canonical = []
    for extent in reversed(value.shape):
        canonical.append(stride)
        stride *= max(1, extent)
    if value.stride() == tuple(reversed(canonical)):
        return value
    # Singleton dimensions can otherwise retain the stride of an indexed view.
    return torch.empty_like(value, memory_format=torch.contiguous_format).copy_(value)


def training_inputs(x, hist, hmask, size):
    b = len(hist)
    defaults = dict(seed=hist.new_zeros(b, 1, 3), seed_mask=hist.new_zeros(b, 1),
                    seed_tangent=hist.new_zeros(b, 3), seed_age=hist.new_zeros(b))
    names = ('fine', 'seed', 'seed_mask', 'seed_tangent', 'seed_age')
    # Optional observed-path geometry exists only for models that consume it.
    names += tuple(name for name in ('path_geometry', 'path_geometry_valid') if name in x)
    image = {name: fixed_rows(x[name] if name in x else defaults[name], size) for name in names}
    return image, fixed_rows(hist, size), fixed_rows(hmask, size)


def begin_training_update(model):
    model._training_refined = torch.zeros((), device=next(model.parameters()).device, dtype=torch.bool)


def finish_training_update(model):
    # Masked, unexecuted retries must preserve AdamW's grad=None semantics.
    if model.cfg.recurrent_refinement_steps and not bool(model._training_refined):
        for module in (model.refinement_fusion, model.refinement_stage):
            for parameter in module.parameters():
                parameter.grad = None


REFINE_ALL = 2.  # retry threshold above any confidence: every refinement pass runs (--refinement-loss all)


def training_prediction(model, x, hist, hmask, confidence_threshold=.5, n_commit=None, targets=None, retry_threshold=None):
    """Training forward and proposal selection. Selection follows the operating ``confidence_threshold``; refinement
    passes run while a proposal's confidence is below ``retry_threshold`` (default: the same threshold; REFINE_ALL:
    every pass)."""
    actual = len(hist)
    # A single decision needs no duplicate encoder, proposal or scoring work.
    # Keep only two batch specializations: one row or the configured batch.
    size = 1 if actual == 1 else model.training_batch_size
    image, history, mask = training_inputs(x, hist, hmask, size)
    threshold = hist.new_full((), confidence_threshold if retry_threshold is None else retry_threshold)
    padded_targets = None if targets is None else {
        key: fixed_rows(value, size) for key, value in targets.items()
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == actual}
    output = model.training_forward(image, history, mask, threshold, padded_targets)
    output = {name: value[:actual] for name, value in output.items()}
    if model._training_refined is None:
        begin_training_update(model)
    model._training_refined = model._training_refined | output['refinement_mask'][:, 1:].any()
    return model.select_prediction(output, confidence_threshold, n_commit)


def match_optimizer_layout(opt):
    """Resumed moment estimates take their parameter's memory format."""
    for param, state in opt.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == param.shape:
                state[key] = torch.empty_like(param).copy_(value)


def initialize_training_optimizer(model, ema, args, resume=None):
    """Restore training, or explicitly restart AdamW and its LR schedule."""
    reset = args.reset_optimizer
    if reset and resume is None:
        raise ValueError('--reset-optimizer requires --resume')
    parameters = model.parameters()
    if reset:
        model.requires_grad_(True)
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    done = resume_training(resume, model, ema, opt, reset_optimizer=reset) if resume else 0
    origin = done if reset else int((resume or {}).get('lr_restart_step', 0))
    match_optimizer_layout(opt)
    return opt, done, origin


GRAD_CLIP = 60.


def load_matching_weights(model, state, exclude=()):
    """Load tensors whose names and shapes match, except names starting with an ``exclude`` prefix.

    Returns a report: tensors left at initialization (``fresh``), checkpoint tensors with no counterpart
    (``unexpected``) and same-name tensors of another shape (``mismatched``)."""
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape
               and not any(k.startswith(prefix) for prefix in exclude)}
    model.load_state_dict(matched, strict=False)
    return dict(fresh=sorted(set(own)-set(matched)), unexpected=sorted(set(state)-set(own)),
                mismatched=sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape))


def initialize_model_weights(cfg, device, initial=None, exclude=()):
    """Weight-only initialization (model and EMA tensors matched by name and shape); never inherits optimizer,
    replay or run settings."""
    model = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    report = None
    if initial is not None:
        report = load_matching_weights(model, initial['model'], exclude)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    if initial is not None:
        load_matching_weights(ema, initial.get('ema', initial['model']), exclude)
    model.initialization_report = report
    return model, ema


def load_checkpoint(path,device='cuda'):
    # Collection shares a GPU with training. Keep duplicate weights and any
    # optimizer state on the CPU; only the EMA inference model needs VRAM.
    from vesuvius.neural_tracing.fiber_follow.shared.paths import relocate_checkpoint
    ck = relocate_checkpoint(read_checkpoint(path,'cpu'))
    cfg = config_from_checkpoint(ck)
    model = build_model(cfg).to(device,memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    return model,cfg.fine,cfg.n_history,FiberVolumeSpec.from_dict(ck['vol_spec']),ck


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items() if not k.startswith('_')}


class DecisionBatchPrefetch:
    """Assemble one CPU update ahead while the current update uses the GPU.

    Only this thread consumes the loader iterator. Chunks remain ordered and
    exactly grad_steps whole batches form each update.
    The complete denominator is therefore known before any task backward.
    With ``merge``, each batch is ``merge`` of ``per_chunk`` consecutive loader items (sequence episodes).
    """
    def __init__(self, iterator, grad_steps, per_chunk=1, merge=None):
        if grad_steps < 1 or per_chunk < 1:
            raise ValueError('Positive gradient accumulation steps and loader items per batch required')
        self.iterator, self.grad_steps = iterator, grad_steps
        self.per_chunk, self.merge = per_chunk, merge
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='decision-batch')
        self.pending = self.executor.submit(self.collect)
        self.closed = False

    def collect(self):
        chunks = []
        for _ in range(self.grad_steps):
            if self.merge is None:
                chunks.append(next(self.iterator))
            else:
                chunks.append(self.merge([next(self.iterator) for _ in range(self.per_chunk)]))
        return chunks

    def __next__(self):
        if self.closed:
            raise StopIteration
        chunks = self.pending.result()
        self.pending = self.executor.submit(self.collect)
        return chunks

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)


@torch.no_grad()
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device, dataset_name=None):
    """Plot EMA proposals and the same monitor rollouts used for logged metrics."""
    from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_batch, plot_curves, plot_refinement, plot_rollouts
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import evaluate
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary

    images = Path(out)/'images'
    images.mkdir(exist_ok=True)
    # Bound image size even when training with larger batches.
    def take(value):
        return {k: take(v) for k, v in value.items()} if isinstance(value, dict) else value[:6]
    batch = move_batch(take(cpu_batch), device)
    was_training = model.training
    threshold_before = tracer.p.confidence
    model.eval()
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            prediction = model(batch['x'], batch['hist'], batch['hmask'],
                               n_commit=tracer.p.n_commit)
        points = prediction['points']
        target = torch.cat((batch['plane_ab'], points[..., 2:]), -1)
        for scale, crop in (('fine', model.cfg.fine),):
            filename = f'batch_{step:06d}.png'
            plot_batch(batch['x'][scale], points, target, batch['plane_mask'], crop, images/filename,
                       batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                       source=batch['source'], terminal=batch['terminal'], confidence=prediction['confidence'])
        curves = prediction.get('solver_points', prediction['refinement_points'])
        labels = ([f'integration stage {i}' for i in range(curves.shape[1])] if 'solver_points' in prediction
                  else ['initial proposal']+[f'feedback proposal {i}' for i in range(1, curves.shape[1])])
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
                            **({'dataset':dataset_name} if dataset_name else {}),
                            coverage_max_len=tracer.p.max_len, **rollout_summary(rows)))
        plot_curves(Path(out)/'log.jsonl', Path(out)/'curves.png', loss_key='flow' if model.cfg.model_type == 'flow' else 'geometry')
    finally:
        model.train(was_training)
        tracer.p.confidence = threshold_before


def accumulate(table, key, value):
    """Add a device scalar without synchronizing; ``resolve_device_sums`` reads them once."""
    table[key] = table.get(key, 0.)+value.detach().double()


def resolve_device_sums(*tables):
    """Replace accumulated device scalars by Python floats with one transfer."""
    entries = [(table, key) for table in tables for key, value in table.items() if torch.is_tensor(value)]
    if entries:
        for (table, key), value in zip(entries, torch.stack([t[k] for t, k in entries]).tolist()):
            table[key] = value


IDENTITY_SUMS = ('identity_correct_count', 'identity_flipped_count')


def clip_training_gradients(model, max_norm=GRAD_CLIP):
    """Clip the global gradient norm; checked before modifying gradients."""
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError('The gradient clipping limit must be finite and nonnegative (0 disables clipping)')
    params = list(model.parameters())
    norm = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None], error_if_nonfinite=True)
    if max_norm > 0:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, norm)
    norm = float(norm)
    return dict(grad_norm=norm, grad_clip_scale=min(1., max_norm/(norm+1e-6)) if max_norm else 1.)


def optimizer_update(model, ema, opt, batches, step, lr, *, device='cpu', tolerance=1.5,
                     confidence_weight=.5, ema_decay=.999, ema_ramp=True, n_commit=None, compute_metrics=True,
                     grad_clip=GRAD_CLIP, diagnostic=None, live_continuation=None, ledger=None,
                     confidence_threshold=.5, gate='prefix', refinement_loss='policy',
                     refinement_weight=1.):
    """One equally weighted task loss per independent supervised decision."""
    prepare_training(model, getattr(model, 'training_batch_size', 2))
    # Episode batches (model 'sequence') supervise their trailing decisions; earlier steps only build history.
    total = sum(int(batch['episode_supervised'].sum()) if 'episode_supervised' in batch else len(batch['hist'])
                for batch in batches)
    observed = sum(len(batch['hist']) for batch in batches)
    if total < 1:
        raise ValueError('An update needs at least one supervised decision')
    denominator = max(1, total)
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    begin_training_update(model)
    sums = dict(loss=0., geometry=0., confidence_loss=0., error_sum=0., geometry_count=0.,
                correct_count=0., confidence_count=0.)
    identity = {}
    decisions = []
    per_state = []
    ledger_batches = []
    model.train()
    for cpu in batches:
        if 'ct_frame_rejected_batches' in cpu:
            sums['ct_frame_rejected_batches'] = sums.get('ct_frame_rejected_batches', 0)+int(cpu['ct_frame_rejected_batches'].sum())
        if 'dataset_id' in cpu:
            counts = sums.setdefault('dataset_counts', {})
            ids, sizes = torch.unique(cpu['dataset_id'], return_counts=True)
            for source_id, size in zip(ids.tolist(), sizes.tolist()):
                counts[str(source_id)] = counts.get(str(source_id), 0)+size
        for prefix in ('ct_frame',):
            if prefix+'_source' not in cpu['x']:
                continue
            source = cpu['x'][prefix+'_source']
            known = source >= 0
            for suffix, value in dict(count=known.sum(), transported=(source == 1).sum(),
                    deterministic=(source == 2).sum(), learned=(source == 3).sum(),
                    energy_sum=cpu['x'][prefix+'_energy'][known].sum(),
                    gap_sum=cpu['x'][prefix+'_gap'][known].sum()).items():
                name = prefix+'_'+suffix
                sums[name] = sums.get(name, 0.)+float(value)
        batch = move_batch(cpu, device)
        if diagnostic is not None:
            diagnostic.update(cpu_batch=cpu)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            window = selection_window(commit_window(model.cfg, n_commit), gate_horizon(model.cfg), gate)
            if model.model_type == 'sequence':  # an episode batch: outputs and rows of its supervised decisions
                from vesuvius.neural_tracing.fiber_follow.train.sequence import episode_forward, select_rows
                output, batch = episode_forward(model, batch, confidence_threshold, window)
                ledger_batches.append(select_rows(cpu, cpu['episode_supervised'].nonzero().flatten(), len(cpu['hist'])))
            else:
                ledger_batches.append(cpu)
                # Proposal selection follows the operating policy, exactly as in tracing; so do retries, unless every
                # refinement pass is trained (refinement_loss 'all').
                output = training_prediction(model, batch['x'], batch['hist'], batch['hmask'], confidence_threshold,
                                             n_commit=window, targets=batch,
                                             retry_threshold=REFINE_ALL if refinement_loss == 'all' else None)
            terms = model.training_loss(output, batch, model.cfg, tolerance, n_commit=n_commit, refinement_loss=refinement_loss,
                                        refinement_weight=refinement_weight, policy_threshold=confidence_threshold)
            geometry = terms['geometry_per_state'].sum()/denominator
            confidence = terms['confidence_per_state'].sum()/denominator
            loss = geometry + confidence_weight*confidence
        loss.backward()
        if live_continuation is not None:
            live_continuation.feedback(cpu, output, step)
        if compute_metrics:
            decisions.extend(decision_rows(output, batch, model.cfg, n_commit, tolerance))
        per_state.append(torch.stack((terms['positive_targets_per_state'].detach().float(),
                                      terms['negative_targets_per_state'].detach().float(),
                                      terms['geometry_states_per_state'].detach().float()), -1))
        for key in IDENTITY_SUMS:
            if key in terms:
                accumulate(identity, key, terms[key])
        for key in ('blurred', 'foreign_components', 'seed_present', 'identity_observable'):
            if key in cpu:
                identity[key] = identity.get(key, 0.)+float((cpu[key] > 0).sum())
        if 'live_continuation' in cpu:
            live = cpu['live_continuation']
            sums['live_rows'] = sums.get('live_rows', 0)+int(live.sum())
            sums['live_terminal_rows'] = sums.get('live_terminal_rows', 0)+int(cpu['live_terminal'].sum())
        if 'live_depth' in cpu:
            sums['live_depth_sum'] = sums.get('live_depth_sum', 0.)+float(cpu['live_depth'].sum())
            sums['live_travelled_sum'] = sums.get('live_travelled_sum', 0.)+float(cpu['live_travelled'].sum())
            for name, selected in (('live_depth_counts', cpu['live_depth'] > 0),
                                   ('live_start_limit_counts', cpu['live_depth'] == 1)):
                values = cpu['live_depth' if name == 'live_depth_counts' else 'live_limit'][selected]
                counts = sums.setdefault(name, {})
                for value, count in zip(*torch.unique(values, return_counts=True)):
                    key = str(int(value))
                    counts[key] = counts.get(key, 0)+int(count)
        if 'location_source' in cpu:
            for index, name in enumerate(LOCATION_SOURCES):
                key = f'location_{name}'
                identity[key] = identity.get(key, 0.)+float((cpu['location_source'] == index).sum())
        for key, value in (('loss', loss), ('geometry', geometry), ('confidence_loss', confidence)):
            accumulate(sums, key, value)
        for key in ('error_sum', 'geometry_count', 'correct_count', 'confidence_count',
                    'point_correct_count', 'point_wrong_count', 'point_unknown_count',
                    'confidence_labeled_states', 'confidence_terminal_states', 'confidence_recoverable_states',
                    'refinement_attempts_sum', *(('connector_rejected_targets',) if 'connector_rejected_targets' in terms else ())):
            accumulate(sums, key, terms[key])
    resolve_device_sums(sums, identity)
    per_state = torch.cat(per_state).cpu()
    if ledger is not None:
        offset = 0
        for cpu in ledger_batches:
            rows = per_state[offset:offset+len(cpu['hist'])]
            offset += len(cpu['hist'])
            ledger.add(cpu, step, rows.T.tolist())
    # The summed loss is finite only if every batch loss was; checked before any update.
    if not math.isfinite(sums['loss']):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    if total:
        finish_training_update(model)
        sums.update(clip_training_gradients(model, grad_clip))
        opt.step()
        update_ema(ema, model, step, ema_decay, ramp=ema_ramp)
    sums.update(observed_states=observed, supervised_states=total,
                observation_only_states=observed-total, optimizer_applied=bool(total),
                positive_confidence_targets=float(per_state[:, 0].sum()),
                negative_confidence_targets=float(per_state[:, 1].sum()))
    sums['refinement_attempts_mean'] = sums.get('refinement_attempts_sum', 0.)/denominator
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']) if sums['geometry_count'] else None,
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']))
    if identity:
        identity['identity_prefix_correct_fraction'] = identity.get('identity_correct_count', 0.)/max(1., sums['confidence_count'])
        for key in ('blurred', 'foreign_components', 'seed_present', 'identity_observable', *(f'location_{n}' for n in LOCATION_SOURCES)):
            if key in identity:
                identity[key+'_fraction'] = identity.pop(key)/denominator
        sums['identity'] = identity
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, commit_window(model.cfg, n_commit))
    sums['prediction_loss_type'] = 'flow' if model.cfg.model_type == 'flow' else 'geometry'
    sums['prediction_loss'] = sums['geometry']
    if model.cfg.model_type == 'flow':
        sums.setdefault('flow', sums['geometry'])
    return sums


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Run configuration JSON (train/run_config.py)')
    ap.add_argument('--resume', help='Checkpoint to continue (model, EMA, optimizer, step); its run folder is the '
                                     "configuration's name under out_root")
    return ap


def configured_model(run, resume, progress):
    """The model configuration: the checkpoint's when resuming (fitted flow scales included), else the run's."""
    cfg = run_config.model_config(run)
    if resume is None:
        return cfg
    recorded = config_from_checkpoint(resume)
    ignored = ('flow_sigma', 'frame_checkpoint', 'frame_checkpoint_sha256')
    differs = sorted(k for k, v in cfg.to_dict().items() if k not in ignored and recorded.to_dict().get(k) != v)
    if differs:
        progress(f'Resuming with the checkpoint model configuration; ignoring configured {", ".join(differs)}')
    return recorded


def main(argv=None):
    cli = build_parser().parse_args(argv)
    run = run_config.load(cli.config)
    args = run_config.namespace(run, cli.resume)
    dataset_document = run['dataset']
    dataset_digest = hashlib.sha256(json.dumps(dataset_document, sort_keys=True).encode()).hexdigest()
    launch_started = time.monotonic()

    def progress(message):
        print(f'[{args.name} +{time.monotonic()-launch_started:.1f}s] {message}', flush=True)

    progress(f'Starting training ({args.model}): device={args.device}, workers={args.workers}')
    if args.reset_optimizer and not args.resume:
        raise ValueError('reset_optimizer applies to --resume')
    budget = TaskBudget(tuple(args.task_shares[name] for name in TASKS), terminal_fallback_cap=args.terminal_fallback_cap,
                        replay_max_age=args.replay_max_age, replay_event_cap=args.replay_event_cap)
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    raise_open_file_limit()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    resume = read_checkpoint(args.resume, args.device) if args.resume else None
    if resume is not None and args.init_weights:
        progress('Resuming: init_weights only applies to a new run and is ignored')
    initial = read_checkpoint(args.init_weights, 'cpu') if args.init_weights and resume is None else None
    initialization = resume.get('initialization') if resume else args.init_weights
    cfg = configured_model(run, resume, progress)
    # The configured frame model; a resumed run records the file it now uses.
    bind_frame_checkpoint(cfg, args.frame_checkpoint)
    frame_predictor(cfg)  # validate once before dataset loading; workers reuse a frozen CPU predictor
    # The recorded configuration holds the model actually trained; fitted flow scales stay with the checkpoint.
    configured_sigma = run['model'].get('flow_sigma')
    run['model'] = dict(type=cfg.model_type, **{k: v for k, v in json.loads(json.dumps(cfg.to_dict())).items()
                                                if k not in run_config.DERIVED_MODEL_FIELDS})
    if cfg.model_type == 'flow':
        run['model']['flow_sigma'] = configured_sigma
    from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import BankSwitchDetector
    BankSwitchDetector([], args.bank_switch_tolerance, args.bank_own_tolerance)
    identity_sampling = IdentitySampling(
        blur_probability=args.blur_probability,blur_sigma=tuple(args.blur_sigma),
        lateral_fraction=args.lateral_fraction,bank_hard_fraction=args.bank_hard_fraction,
        bank_coverage_probability=args.bank_coverage_probability,synthetic_tail=tuple(args.synthetic_tail))
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    primary_source = next(s for s in dataset_document['sources'] if s['kind'] == 'paris4')
    from vesuvius.neural_tracing.fiber_follow.data.datasets import primary_source_spec
    spec = primary_source_spec(dataset_document)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                          future_step=cfg.future_step, recent_history_points=cfg.n_history,
                          startup_shares=tuple(args.startup_shares), excursion_probability=args.excursion_probability,
                          excursion_amplitude=tuple(args.excursion_amplitude), excursion_rise=tuple(args.excursion_rise),
                          label_tolerance=args.tolerance, max_recovery_distance=cfg.max_recovery_distance)
    policy = training_policy(cfg, args.n_commit, args.trace_confidence, args.gate)
    progress('Loading manifest and fiber annotations')
    from vesuvius.neural_tracing.fiber_follow.data.datasets import load_primary_dataset
    fibers,train_f,val_f,manifest = load_primary_dataset(dataset_document,spec)
    progress(f'Loaded {len(train_f)} training fibers and {len(val_f)} validation fibers')
    if fiber_manifest(val_f) != manifest['fibers']:
        raise ValueError('Frozen validation geometry differs from dataset/holdout')
    if initial and initial.get('dataset_config'):
        known = {s['name'] for s in initial['dataset_config'].get('sources', [])}
        added = [s['name'] for s in dataset_document['sources'] if s['name'] not in known]
        if added:
            progress(f'Warm start adds dataset sources the initialization never saw: {", ".join(added)}')
    out = prepare_run_dir(args.out_root, args.name, resume is not None)
    (out/'trainer.pid').write_text(f'{os.getpid()}\n')
    (out/'validation_paris4.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import prepare_normalization
    from vesuvius.neural_tracing.fiber_follow.data.datasets import ct_source_spec
    calibration_specs = [spec, *(ct_source_spec(s, dataset_document['cache_dir'])
                                 for s in dataset_document['sources'] if s['kind'] != 'paris4')]
    progress('Preparing per-crop CT z-score normalization')
    ct_normalization = prepare_normalization(out, calibration_specs,
        known=resume['ct_normalization'] if resume is not None else None)  # a relocated CT adds its own record
    recovery_states = recovery_hash = None
    if args.recovery_every:
        progress('Preparing monitor recovery fixture')
        recovery_states, recovery_hash = monitor_fixture(out/'monitor_recovery.npz', val_f, manifest,
                                                         sample, spec, args.recovery_seeds)
    progress('Initializing models and optimizer')
    progress(f'Gradient clipping at global norm {args.grad_clip:g} (0 disables clipping)')
    if args.refinement_loss == 'all':
        progress(f'Refinement loss: every pass for every decision; initial proposal full weight, passes x{args.refinement_weight:g} '
                 f'of their mean; selection and the attempt metric follow confidence {args.trace_confidence:g}')
    # Flow scales are fitted after constructing the common training dataset below, unless the configuration came
    # with them (a resumed checkpoint, or flow_sigma in the run configuration).
    fit_flow_scales = cfg.model_type == 'flow' and not cfg.flow_sigma
    if fit_flow_scales:
        cfg.flow_sigma = tuple((cfg.flow_sigma_floor,)*2 for _ in range(len(cfg.path_plane_values)))
    model, ema = initialize_model_weights(cfg, args.device, initial, exclude=tuple(args.init_exclude))
    if model.initialization_report is not None:
        report = model.initialization_report
        progress(f'Initialized from {args.init_weights}: {len(report["fresh"])} tensors left at initialization, '
                 f'{len(report["unexpected"])} checkpoint tensors unused, {len(report["mismatched"])} of another shape')
        for name in ('fresh', 'unexpected', 'mismatched'):
            if report[name]:
                progress(f'  {name}: {", ".join(report[name][:20])}{" ..." if len(report[name]) > 20 else ""}')
    opt, done, lr_restart_step = initialize_training_optimizer(model, ema, args, resume)
    if args.reset_optimizer:
        progress(f'Fresh AdamW: one parameter group, all parameters trainable; LR restarts at update {done+1} '
                 f'with {args.warmup} warmup updates to {args.lr:g}')
    # Episode batches present their supervised decisions to the compiled decision graph.
    prepare_training(model, args.batch*args.episode_supervised if cfg.model_type == 'sequence' else args.batch)
    progress('Compiling training operations; first forward/backward passes may take several minutes')
    if done >= args.steps:
        raise ValueError('Run has already reached its requested update count')
    replay_index = out/'dagger'/'replay.json'
    replay_paths = json.loads(replay_index.read_text()) if resume and replay_index.exists() else []
    progress('Loading replay banks and preparing data loader')
    caches = usable_replay(load_replay(replay_paths), train_f, cfg.n_history, spec.grid_scale)
    collection = dict(every=args.dagger_every, fibers_per_collection=args.dagger_fibers, batch=args.dagger_batch,
                      forward_chunk=args.dagger_forward_chunk, seed=args.seed, replay_keep=args.replay_keep,
                      trace_len=args.dagger_trace_len, before=args.dagger_before, after=args.dagger_after,
                      # Sequence episodes need every decision of a collected trace (consecutive history steps).
                      stride=0. if cfg.model_type == 'sequence' else args.dagger_stride,
                      max_states=1 << 20 if cfg.model_type == 'sequence' else 192,
                      confidence=policy.confidence, n_commit=policy.n_commit, gate=policy.gate,
                      collector_module='vesuvius.neural_tracing.fiber_follow.tracing.collect', threads=args.dagger_threads)
    bank_args = ('--bank-switch-tolerance', args.bank_switch_tolerance, '--bank-own-tolerance', args.bank_own_tolerance)
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.device,
        initial=[c._dir for c in caches], **collection, extra_args=bank_args)
    progress(f'{args.batch} independent decisions per batch')
    # Paris 4 has no neighbor paths: no foreign masks, bank-covered locations or synthetic wrong continuations.
    builder = IdentityObservationBuilder(cfg,train_f,identity_sampling,augment=True)
    from vesuvius.neural_tracing.fiber_follow.data.data import loader_chunk
    dataset = FollowDataset(train_f, spec, sample, chunk=loader_chunk(cfg, args.batch), seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), onpolicy=caches,
        replay_index=str(collector.index), batch_builder=builder, budget=budget)
    from vesuvius.neural_tracing.fiber_follow.data.datasets import build_mixed_dataset
    progress('Checking mixed-source datasets and AFV checksums')
    dataset, dataset_provenance = build_mixed_dataset(dataset, dataset_document, cfg, sample,
        identity_sampling, args, seed=args.seed+done, out=out, resume=resume is not None,
        normalization=ct_normalization, budget=budget)
    from vesuvius.neural_tracing.fiber_follow.train.online import MultiSourceCollector
    collectors = []
    for source, source_dataset in zip(dataset_document['sources'], dataset.datasets):
        if source['kind'] == 'paris4':
            collector.extra_args += ['--dataset-name',source['name']]
            collectors.append((source['name'],collector))
            continue
        (out/f'validation_{source["name"]}.json').write_text(json.dumps(source_dataset.validation_manifest,indent=2)+'\n')
        afv_collection = dict(collection, length_power=args.afv_length_power,
                              fibers_per_collection=args.afv_dagger_fibers or args.dagger_fibers)
        collectors.append((source['name'],OnlineCollector(out/'dagger'/source['name'],
            source['path'], (0,1), args.device,
            initial=[c._dir for c in source_dataset.onpolicy], **afv_collection,
            extra_args=('--dataset-name',source['name'],*bank_args))))
    collector = MultiSourceCollector(collectors)
    progress('Dataset sampling: '+', '.join(f'{name}={weight:.1%}' for name,weight in zip(dataset.names,dataset.weights))
             +f'; AFV training draws and AFV collection order proportional to length**{args.afv_length_power:g}')
    progress('Task budget per source: '+', '.join(f'{k} {v:.0%}' for k, v in budget.to_dict()['shares'].items())
             +f'; replay age ceiling {budget.replay_max_age}, event cap {budget.replay_event_cap}, '
             f'synthetic terminal fallback cap {budget.terminal_fallback_cap:.0%}')
    for source_dataset in dataset.datasets:
        source_dataset.set_step(done)
        if cfg.model_type == 'sequence':
            from vesuvius.neural_tracing.fiber_follow.data.data import EpisodeSpec
            source_dataset.episodes = EpisodeSpec(steps=args.episode_steps, supervised=args.episode_supervised,
                                                  commit=args.episode_commit or args.n_commit)
    from vesuvius.neural_tracing.fiber_follow.train.live_continuation import LiveContinuation, preserve_live_metadata
    live_continuation = LiveContinuation(dataset, policy=policy, steps=args.live_continuation_steps,
        switch_tolerance=args.bank_switch_tolerance, own_tolerance=args.bank_own_tolerance, horizon=gate_horizon(cfg))
    loader_args = dict(batch_size=None, num_workers=args.workers,
                       pin_memory=torch.device(args.device).type == 'cuda', collate_fn=preserve_live_metadata)
    if args.workers:
        loader_args.update(prefetch_factor=2, persistent_workers=True)
    loader = torch.utils.data.DataLoader(dataset, **loader_args)
    if fit_flow_scales:
        from vesuvius.neural_tracing.fiber_follow.models.flow import fit_flow_sigma
        progress('Fitting flow residual scales from training states')
        cfg.flow_sigma = fit_flow_sigma(iter(loader), cfg, args.flow_calibration_states)
        for follower in (model, ema):
            follower.sigma.copy_(torch.tensor(cfg.flow_sigma, device=args.device))
            follower.cfg.flow_sigma = cfg.flow_sigma
    (out/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    if not resume:
        (out/'config.json').write_text(json.dumps(dict(run_config=run, model_type=model.model_type,
            dataset_config_sha256=dataset_digest, dataset_provenance=dataset_provenance,
            ct_normalization=ct_normalization, identity_sampling=asdict(identity_sampling),
            model_cfg=cfg.to_dict(), sample_cfg=asdict(sample), vol_spec=spec.to_dict(),
            task_budget=budget.to_dict(), operating_policy=policy.to_dict(),
            collection=collector_settings(collector),
            initialization=dict(checkpoint=str(Path(args.init_weights).resolve()), optimizer='fresh AdamW',
                                warmup=args.warmup, excluded=list(args.init_exclude),
                                **model.initialization_report) if args.init_weights else None,
            frame_policy=configured_frame_policy(cfg), monitor_recovery_sha256=recovery_hash,
            seed_manifest_sha256=manifest['sha256'], fiber_manifest=fiber_manifest(fibers),
            parameter_count=sum(p.numel() for p in model.parameters())), indent=2))
    log = RunLog(out/'log.jsonl', formatter=format_training_log)
    log.record(dict(step=done, event='ct_normalization', calibration=ct_normalization))
    log.record(dict(step=done, event='dataset_configuration', datasets=dataset_provenance,
                    names=dataset.names, probabilities=dataset.weights.tolist(),
                    dataset_config_sha256=dataset_digest, input_mode='ct',
                    evaluation_scope='Source-specific held-out fibers; Paris 4 recovery fixture'))
    if resume:
        log.record(dict(step=done, event='resume_configuration', checkpoint=str(args.resume),
                        run_config=run, model_cfg=cfg.to_dict()))
    log.record(dict(step=done, event='optimizer_configuration', reset=args.reset_optimizer,
                    lr_restart_step=lr_restart_step, optimizer_state_entries=len(opt.state),
                    groups=[dict(parameters=len(g['params'])) for g in opt.param_groups],
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    log.record(dict(step=done,event='identity_sampling',model_type=model.model_type,
        task_budget=budget.to_dict(), operating_policy=policy.to_dict(), collection=collector_settings(collector),
        fresh_geometry=('simulated traces: annotated seed, smooth lateral tracing error (OU) plus optional smooth '
                        'excursions, tracer heading/history/CT seed heading; labels from the shared state contract'),
        simulated_traces=dict(trace_noise='commit_v1', trace_noise_sigma=sample.trace_noise_sigma,
                              trace_noise_length=sample.trace_noise_length, trace_noise_bulge=sample.trace_noise_bulge,
                              trace_noise_bias=sample.trace_noise_bias, trace_noise_smoothing=sample.trace_noise_smoothing,
                              excursion_rise_distribution='log-uniform',
                              startup_shares=sample.startup_shares, excursion_probability=sample.excursion_probability,
                              excursion_amplitude=sample.excursion_amplitude, excursion_rise=sample.excursion_rise),
        label_contract=dict(tolerance=sample.label_tolerance, max_recovery_distance=sample.max_recovery_distance),
        live_continuation_steps=args.live_continuation_steps,
        frame_policy=configured_frame_policy(cfg), sampling=asdict(identity_sampling)))
    tracer = None
    recovery_vol = FiberVolume(spec) if recovery_states is not None else None
    started = time.monotonic()
    interval_started, interval_step = started, done
    interval_data_seconds = interval_update_seconds = 0.
    interval_metrics = TrainingInterval()
    updates = None
    remote_prefetch = None
    try:
        if args.remote_prefetch_connections:
            from vesuvius.neural_tracing.fiber_follow.data.remote_prefetch import attach_remote_prefetch
            remote_prefetch, remote_sources = attach_remote_prefetch(dataset.datasets,
                args.remote_prefetch_connections, args.remote_prefetch_queue_size, args.remote_prefetch_timeout,
                args.remote_prefetch_lookahead, args.workers)
            if remote_prefetch is not None:
                progress(f'Remote CT prefetch: {args.remote_prefetch_connections} concurrent fetches, '
                         f'{args.remote_prefetch_queue_size} requests per priority queue, '
                         f'{args.remote_prefetch_lookahead} future batch plans per remote source/worker, separate async process')
            log.record(dict(step=done,event='remote_prefetch_configuration',
                enabled=remote_prefetch is not None,connections=args.remote_prefetch_connections,
                queue_size=args.remote_prefetch_queue_size,lookahead=args.remote_prefetch_lookahead,
                timeout=args.remote_prefetch_timeout,sources=len(remote_sources)))
        if args.diag_every:
            from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
            tracer = FiberTracer(ema, FiberVolume(spec), cfg.fine, cfg.n_history,
                TraceParams.from_policy(policy, max_len=args.diag_max_len), device=args.device)
        progress(f'Starting data loader; batch {args.batch} × grad steps {args.grad_steps} = '
                 f'{args.batch * args.grad_steps} decisions per update; starting update {done+1}')
        iterator = iter(loader)
        if cfg.model_type == 'sequence':  # single-episode loader items, merged per microbatch
            from vesuvius.neural_tracing.fiber_follow.train.sequence import merge_episodes
            updates = DecisionBatchPrefetch(iterator, args.grad_steps, per_chunk=args.batch, merge=merge_episodes)
        else:
            updates = DecisionBatchPrefetch(iterator, args.grad_steps)
        observed_states = interval_states = 0
        prior_samples = int(resume.get('samples_seen', 0)) if resume else 0
        ledger = SamplingLedger()
        for step in range(done+1, args.steps+1):
            for source_dataset in dataset.datasets:
                source_dataset.set_step(step)
            event = collector.poll(step)
            if event:
                log.record(dict(step=step, **event))
            early = step <= done+5
            batch_started = time.monotonic()
            if early and step != done+1:
                progress(f'Update {step}: waiting for data')
            batches = next(updates)
            data_seconds = time.monotonic()-batch_started
            update_started = time.monotonic()
            if early:
                progress(f'Update {step}: data ready in {data_seconds:.1f}s; running forward/backward and optimizer')
            lr = lr_at(step-lr_restart_step, args.lr, args.warmup, args.steps-lr_restart_step)
            diagnostic = {} if tracer is not None and args.diag_every and step % args.diag_every == 0 else None
            metrics = optimizer_update(model, ema, opt, batches, step, lr, device=args.device,
                tolerance=args.tolerance, confidence_weight=args.confidence_weight, ema_decay=args.ema_decay,
                ema_ramp=not initialization,
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps,
                grad_clip=args.grad_clip, confidence_threshold=policy.confidence, gate=policy.gate,
                refinement_loss=args.refinement_loss, refinement_weight=args.refinement_weight,
                diagnostic=diagnostic, live_continuation=live_continuation, ledger=ledger)
            observed_states += metrics['observed_states']
            interval_states += metrics['observed_states']
            update_seconds = time.monotonic()-update_started
            interval_data_seconds += data_seconds
            interval_update_seconds += update_seconds
            interval_metrics.add(metrics)
            if early:
                progress(f'Update {step} complete in {time.monotonic()-update_started:.1f}s; loss={metrics["loss"]:.5f}')
                if step == done+5:
                    progress(f'Startup progress complete; regular metrics every {args.log_every} updates')
            if step % args.log_every == 0 or step == args.steps:
                now = time.monotonic()
                log.record(dict(step=step, lr=lr, **metrics, samples_seen=prior_samples+observed_states,
                    **({'remote_prefetch':remote_prefetch.snapshot()} if remote_prefetch else {}),
                    train_seconds=now-started,
                    samples_per_second=observed_states/(now-started),
                    interval_samples_per_second=interval_states/(now-interval_started),
                    interval_data_seconds=interval_data_seconds, interval_update_seconds=interval_update_seconds,
                    interval=interval_metrics.summary(), sampling=ledger.summary(), dataset_names=list(dataset.names),
                    n_future=cfg.n_future, tolerance=args.tolerance,
                    interval_updates=step-interval_step,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30
                        if torch.device(args.device).type == 'cuda' else None))
                from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_curves
                plot_curves(out/'log.jsonl', out/'curves.png', loss_key='flow' if model.cfg.model_type == 'flow' else 'geometry')
                interval_metrics = TrainingInterval()
                ledger = SamplingLedger()
                interval_started, interval_step = now, step
                interval_data_seconds = interval_update_seconds = 0.
                interval_states = 0

            def save(path, resumable=False):
                extra = dict(step=step, lr_restart_step=lr_restart_step, tolerance=args.tolerance, n_commit=args.n_commit,
                    operating_policy=policy.to_dict(), task_budget=budget.to_dict(),
                    initialization=str(Path(initialization).resolve()) if initialization else None,
                    dataset_config=dataset_document, dataset_provenance=dataset_provenance,
                    ct_normalization=ct_normalization, frame_policy=configured_frame_policy(cfg),
                    samples_seen=prior_samples+observed_states, identity_sampling=asdict(identity_sampling),
                    run_config=run, training_options=vars(args), fiber_manifest=fiber_manifest(fibers))
                if resumable:
                    extra.update(optimizer=opt.state_dict(), rng=training_rng_state())
                save_checkpoint(path, model, ema, spec, sample, extra)

            # Preserve the completed optimizer update before optional collectors
            # or diagnostics can fail. These checkpoints include optimizer/RNG.
            if step % args.ckpt_every == 0 or step == args.steps:
                save(out/f'ckpt_{step:06d}.pt', resumable=True)
                save(out/'last.pt', resumable=True)
            if step < args.steps:
                if collector.launch(step, save):
                    log.record(dict(step=step, dagger_launched=True))
                elif step % args.dagger_every == 0 if args.dagger_every else False:
                    log.record(dict(step=step, dagger_skipped_busy=True))
            periodic = {}
            # The batch renderer reads the patch-token models' internals; sequence models use the monitor rollouts.
            if args.batch_diag_every and (step % args.batch_diag_every == 0
                                         or (resume is not None and step == done+1)):
                from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import render_microbatch
                began = time.monotonic()
                report = render_microbatch(ema, batches[-1], out, step, device=args.device,
                    n_commit=args.n_commit, tolerance=args.tolerance, dataset_names=dataset.names,
                    training_metrics=dict(metrics, lr=lr, data_seconds=data_seconds,
                        update_seconds=update_seconds,
                        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30
                            if torch.device(args.device).type == 'cuda' else None))
                log.record(dict(step=step, split='current_training_microbatch', diagnostic_images=report))
                periodic['batch_diagnostics_seconds'] = time.monotonic()-began
            if tracer is not None and args.diag_every and step % args.diag_every == 0:
                began = time.monotonic()
                training_diagnostics(ema, diagnostic['cpu_batch'], tracer, val_f, manifest['monitor'], out, step, log,
                                     device=args.device,dataset_name=primary_source['name'])
                from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import evaluate
                from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary
                from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
                for source, source_dataset in zip(dataset_document['sources'],dataset.datasets):
                    if source['kind'] != 'afv':
                        continue
                    source_tracer = FiberTracer(ema,FiberVolume(source_dataset.vol_spec),cfg.fine,cfg.n_history,
                        TraceParams.from_policy(policy,max_len=args.diag_max_len),device=args.device)
                    try:
                        rows,_ = evaluate(source_tracer,source_dataset.validation_fibers,
                            source_dataset.validation_manifest['monitor'],batch=1,coverage_max_len=args.diag_max_len)
                        log.record(dict(step=step,split='monitor',dataset=source['name'],
                            threshold=.5,coverage_max_len=args.diag_max_len,**rollout_summary(rows)))
                    finally:
                        source_tracer.close()
                from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_curves
                plot_curves(out/'log.jsonl',out/'curves.png',loss_key='flow' if model.cfg.model_type == 'flow' else 'geometry')
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
    finally:
        if remote_prefetch is not None:
            remote_prefetch.close()
            log.record(dict(event='remote_prefetch_shutdown',remote_prefetch=remote_prefetch.snapshot()))
        if updates is not None:
            updates.close()
        if live_continuation is not None:
            live_continuation.close()
        event = collector.close()
        if event:
            log.record(event)
        if tracer is not None:
            tracer.close()
        log.close()
    return str(out/'last.pt')


if __name__ == '__main__':
    main()
