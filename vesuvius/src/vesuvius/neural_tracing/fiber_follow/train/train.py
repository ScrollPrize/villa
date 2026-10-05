"""Train coordinate regression or flow-matching fiber followers with shared online replay."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import DATA_POLICY, FollowDataset, OnPolicyStates, SampleConfig, TaskBudget, ZBand, fiber_identities, fiber_manifest, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.tracing.policy import OperatingPolicy
from vesuvius.neural_tracing.fiber_follow.train.runloop import RunLog, lr_at, prepare_run_dir, read_checkpoint, save_checkpoint as write_checkpoint, update_ema, training_rng_state, resume_training, raise_open_file_limit
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig, build_model
from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling, FiberTracer, LOCATION_SOURCES
from vesuvius.neural_tracing.fiber_follow.train.supervision import commit_window, loss_terms
from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows, summarize_decisions
from vesuvius.neural_tracing.fiber_follow.evaluation.recovery import monitor_fixture, evaluate_monitor
from vesuvius.neural_tracing.fiber_follow.data.history_slabs import SAMPLING_REVISION
from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import DEFAULT_FRAME_CHECKPOINT, configured_frame_policy, frame_predictor, bind_frame_checkpoint
from vesuvius.neural_tracing.fiber_follow.train.training_log import format_training_log, TrainingInterval, SamplingLedger


def validate_volume_source(spec, manifest):
    """Allow a different CT pyramid level, retaining frozen physical data/seeds."""
    from vesuvius.neural_tracing.fiber_follow.shared.paths import recorded_path
    for key in ('fiber_zarr_dir', 'ct_zarr', 'fiber_level', 'grid_scale'):
        recorded = manifest['volume'][key]
        if key in ('fiber_zarr_dir', 'ct_zarr') and recorded is not None:
            recorded = recorded_path(recorded)
        if spec.to_dict()[key] != recorded:
            raise ValueError(f'Volume source {key} differs from frozen manifest')


def save_checkpoint(path, model, ema, spec, sample, extra=None):
    # Atomic publication: collectors must never open a partial checkpoint.
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    write_checkpoint(temporary, model, spec, sample.crop, sample.n_history, model.model_type,
                     dict({'n_commit': commit_window(model.cfg, None), **(extra or {})},
                          ema=ema.state_dict(), sample_cfg=asdict(sample)))
    temporary.replace(path)


def collector_settings(collector):
    members = getattr(collector, 'collectors', [(None, collector)])
    return {name or 'primary': member.settings() for name, member in members}


def training_policy(cfg, n_commit, confidence=.5):
    """The operating policy used for live feedback and recorded with every checkpoint."""
    return OperatingPolicy(confidence=confidence, n_commit=n_commit, max_recovery_distance=cfg.max_recovery_distance,
                           refinement_steps=cfg.recurrent_refinement_steps)


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


IDENTITY_INPUTS = ('identity_points', 'identity_point_mask', 'identity_anchor_mask', 'identity_anchor_features')
VERIFY_INPUTS = ('identity_candidates', 'identity_samples')  # positions only; labels stay with the loss


def training_inputs(x, hist, hmask, size):
    b = len(hist)
    defaults = dict(seed=hist.new_zeros(b, 1, 3), seed_mask=hist.new_zeros(b, 1),
                    seed_tangent=hist.new_zeros(b, 3), seed_age=hist.new_zeros(b))
    names = ('fine', 'seed', 'seed_mask', 'seed_tangent', 'seed_age')
    if 'history_tokens' in x:  # absent without memory
        names += ('history_tokens', 'history_padding')
    # Optional observed-path geometry exists only for models that consume it; identity
    # points/anchors only for the memory-identity loss.
    from vesuvius.neural_tracing.fiber_follow.models.model import READOUT_MEMORY
    names += tuple(name for name in ('path_geometry', 'path_geometry_valid', *IDENTITY_INPUTS, *VERIFY_INPUTS,
                                     *READOUT_MEMORY) if name in x)
    image = {name: fixed_rows(x[name] if name in x else defaults[name], size) for name in names}
    return image, fixed_rows(hist, size), fixed_rows(hmask, size)


def begin_training_update(model):
    model._training_refined = torch.zeros((), device=next(model.parameters()).device, dtype=torch.bool)
    model._history_timings = []


def finish_training_update(model):
    # Masked, unexecuted retries must preserve AdamW's grad=None semantics.
    if model.cfg.recurrent_refinement_steps and not bool(model._training_refined):
        for module in (model.refinement_fusion, model.refinement_stage):
            for parameter in module.parameters():
                parameter.grad = None


def training_prediction(model, x, hist, hmask, confidence_threshold=.5, n_commit=None, targets=None):
    actual = len(hist)
    # A single decision needs no duplicate encoder, proposal or scoring work.
    # Keep only two batch specializations: one row or the configured batch.
    size = 1 if actual == 1 else model.training_batch_size
    # Compact live slabs before the fixed-shape compiled decision graph.
    # Convolutions see valid slots only; gradients remain attached across heads.
    timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if hist.is_cuda else time.perf_counter()
    if hist.is_cuda:
        timing[0].record()
    identity = {}
    if getattr(model, 'identity_projection', None) is not None and targets is not None and 'identity_points' in targets:
        tokens, padding, anchors = model.encode_history(x, return_anchors=True)
        identity = {key: targets[key] for key in IDENTITY_INPUTS[:3]}
        identity['identity_anchor_features'] = anchors.detach()
    else:
        # Readout identity memory travels with the history tokens (empty for other objectives).
        tokens, padding, readout = model.encode_history(x, return_identity=True)
        identity = dict(readout)
        if model.cfg.identity_mode == 'verify' and targets is not None and 'identity_candidates' in targets:
            identity.update({key: targets[key] for key in VERIFY_INPUTS})
    if hist.is_cuda:
        timing[1].record()
    if hasattr(model, '_history_timings') and tokens is not None:
        model._history_timings.append(timing if hist.is_cuda else time.perf_counter()-timing)
    memory = {} if tokens is None else dict(history_tokens=tokens, history_padding=padding)
    image, history, mask = training_inputs(dict(x, **memory, **identity), hist, hmask, size)
    if any(key in identity for key in (*IDENTITY_INPUTS, *VERIFY_INPUTS)):
        # Padding repeats row 0; the identity loss must see which rows are real and their fibers.
        image['identity_row'] = torch.arange(size, device=hist.device) < actual
        image['identity_fiber'] = fixed_rows(targets['fiber_id'], size)
    threshold = hist.new_full((), confidence_threshold)
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
    done = resume_training(resume, model, ema, opt, reset_optimizer=reset)[0] if resume else 0
    origin = done if reset else int((resume or {}).get('lr_restart_step', 0))
    match_optimizer_layout(opt)
    return opt, done, origin


from vesuvius.neural_tracing.fiber_follow.models.flow import FlowConfig
MODEL_TYPES = ('coordinate_regression', 'flow_matching')
HISTORY_GRAD_CLIP = 5.
REST_GRAD_CLIP = 100.


def sampling_revision(cfg):
    if cfg.memory == 'decisions':
        from vesuvius.neural_tracing.fiber_follow.data.decision_memory import SAMPLING_REVISION as DECISIONS
        return DECISIONS
    return SAMPLING_REVISION


def checkpoint_config(ck):
    if ck.get('model_type') not in MODEL_TYPES:
        raise ValueError('Unsupported checkpoint model type')
    cls = FlowConfig if ck.get('model_type') == 'flow_matching' else CoordinateRegressionConfig
    cfg = cls(**ck['model_cfg'])
    if cfg.model_type != ck['model_type']:
        raise ValueError('Checkpoint model type does not match configuration')
    return cfg


def model_config_from_args(args, checkpoint=None):
    if checkpoint is not None:
        cfg = checkpoint_config(checkpoint)
        if args.model is not None and args.model != cfg.model_type:
            raise ValueError('Model type must match checkpoint')
        return cfg
    cls = FlowConfig if args.model == 'flow_matching' else CoordinateRegressionConfig
    options = {key: getattr(args, key) for key in ('stem_channels', 'stem_blocks', 'stem', 'memory', 'path_planes', 'tube_head',
        'identity_dim', 'identity_temperature', 'identity_objective', 'identity_map', 'identity_feedback', 'hidden',
        'encoder_ffn', 'decoder_layers', 'decoder_ffn', 'scorer_layers', 'activation_checkpointing')}
    options['layers'] = args.axial_layers
    options['fine'] = CropSpec(depth=args.crop_depth, width=args.crop_width, behind=args.crop_behind,
                               spacing=args.crop_spacing)
    if cls is FlowConfig:
        options.update({key: getattr(args, key) for key in (
            'flow_steps', 'flow_draws', 'flow_samples', 'flow_sample_scale', 'flow_time_conditioning',
            'flow_sigma_floor', 'flow_unknown_planes')})
    else:
        options['recurrent_refinement_steps'] = args.recurrent_refinement_steps
    return cls(**options)


def load_matching_weights(model, state, exclude=()):
    """Load tensors whose names and shapes match; new residual branches start at zero.

    Tensors whose names start with an ``exclude`` prefix keep their initialization.

    Returns the parameter/buffer names left at initialization. An encoder axial block or
    image stem without loaded weights gets zero residual outputs, so the warm-started
    network initially computes what the loaded tensors compute without them.
    """
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape
               and not any(k.startswith(prefix) for prefix in exclude)}
    model.load_state_dict(matched, strict=False)
    fresh = sorted(set(own)-set(matched))
    with torch.no_grad():
        for index, block in enumerate(model.encoder.blocks):
            if f'encoder.blocks.{index}.mlp.2.weight' in fresh:
                for layer in [axis.projection for axis in block.axes]+[block.mlp[-1]]:
                    layer.weight.zero_()
                    layer.bias.zero_()
        if 'encoder.stem.projection.weight' in fresh:
            model.encoder.stem.projection.weight.zero_()
            model.encoder.stem.projection.bias.zero_()
    return fresh


def initialize_model_weights(cfg, device, initial=None, partial=False, exclude=()):
    """Weight-only initialization never inherits optimizer, replay or run settings."""
    model = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    fresh = []
    if initial is not None:
        if partial:
            fresh = load_matching_weights(model, initial['model'], exclude)
        else:
            model.load_state_dict(initial['model'], strict=True)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    if initial is not None:
        if partial:
            load_matching_weights(ema, initial['ema'], exclude)
        else:
            ema.load_state_dict(initial['ema'], strict=True)
    model.reinitialized_tensors = fresh
    return model, ema


def load_checkpoint(path,device='cuda'):
    # Collection shares a GPU with training. Keep duplicate weights and any
    # optimizer state on the CPU; only the EMA inference model needs VRAM.
    from vesuvius.neural_tracing.fiber_follow.shared.paths import relocate_checkpoint
    ck = relocate_checkpoint(read_checkpoint(path,MODEL_TYPES,'cpu'))
    cfg = checkpoint_config(ck)
    model = build_model(cfg).to(device,memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    return model,cfg.fine,cfg.n_history,FiberVolumeSpec(**ck['vol_spec']),ck


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items() if not k.startswith('_')}


class DecisionBatchPrefetch:
    """Assemble one CPU update ahead while the current update uses the GPU.

    Only this thread consumes the loader iterator. Chunks remain ordered and
    exactly grad_steps whole batches form each update.
    The complete denominator is therefore known before any task backward.
    """
    def __init__(self, iterator, grad_steps):
        if grad_steps < 1:
            raise ValueError('Positive gradient accumulation steps required')
        self.iterator, self.grad_steps = iterator, grad_steps
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='decision-batch')
        self.pending = self.executor.submit(self.collect)
        self.closed = False

    def collect(self):
        chunks = []
        for _ in range(self.grad_steps):
            chunks.append(next(self.iterator))
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
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device, dataset_name=None, memory=None):
    """Plot EMA proposals and the same monitor rollouts used for logged metrics."""
    from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_batch, plot_curves, plot_refinement, plot_rollouts
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import evaluate
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary

    images = Path(out)/'images'
    images.mkdir(exist_ok=True)
    # Bound image size even when training with larger batches.
    def take(value):
        return {k: take(v) for k, v in value.items()} if isinstance(value, dict) else value[:6]
    if getattr(model.cfg, 'memory', 'slabs') == 'decisions':
        # Memory crops are stacked per batch, not per row: resolve them before taking rows.
        from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import decision_memory_rows
        with torch.no_grad():
            cpu_batch = decision_memory_rows(model, cpu_batch, device, memory)
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
                       source=batch['source'], terminal=batch['terminal'], confidence=prediction['confidence'],
                       history_channel=None, ct_range=(-4, 4))
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
        plot_curves(Path(out)/'log.jsonl', Path(out)/'curves.png', loss_key='flow' if model.cfg.model_type == 'flow_matching' else 'geometry')
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


def clip_training_gradients(model, history_max_norm=HISTORY_GRAD_CLIP, rest_max_norm=REST_GRAD_CLIP):
    """Clip the slab encoder separately; check both groups before modifying gradients."""
    limits = dict(history=history_max_norm, rest=rest_max_norm)
    if any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Gradient clipping limits must be finite and nonnegative (0 disables clipping)')
    parameters = list(model.parameters())
    memory = getattr(model, 'history_encoder', None)
    memory_ids = {id(p) for p in memory.parameters()} if memory is not None else set()
    groups = dict(history=[p for p in parameters if id(p) in memory_ids],
                  rest=[p for p in parameters if id(p) not in memory_ids])
    if memory is None:  # no memory encoder: a single clipping group
        del groups['history']
    norms = {name: torch.nn.utils.get_total_norm(
        [p.grad for p in params if p.grad is not None], error_if_nonfinite=True)
        for name, params in groups.items()}
    for name, params in groups.items():
        if limits[name] > 0:
            torch.nn.utils.clip_grads_with_norm_(params, limits[name], norms[name])
    values = {name: float(norm) for name, norm in norms.items()}
    metrics = dict(grad_norm=math.hypot(*values.values()))
    for name, norm in values.items():
        metrics[name+'_grad_norm'] = norm
        metrics[name+'_grad_clip_scale'] = min(1., limits[name]/(norm+1e-6)) if limits[name] else 1.
    return metrics


def optimizer_update(model, ema, opt, batches, step, lr, *, device='cpu', tolerance=1.5,
                     confidence_weight=.5, ema_decay=.999, ema_ramp=True, n_commit=None, compute_metrics=True,
                     history_grad_clip=HISTORY_GRAD_CLIP, rest_grad_clip=REST_GRAD_CLIP,
                     diagnostic=None, live_continuation=None, ledger=None, identity_weight=0.,
                     identity_exclude_synthetic=False, tube_weight=1., tube_sigma=1.5):
    """One equally weighted task loss per independent supervised decision."""
    prepare_training(model, getattr(model, 'training_batch_size', 2))
    total = observed = sum(len(batch['hist']) for batch in batches)
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
    model.train()
    for cpu in batches:
        if 'ct_frame_rejected_batches' in cpu:
            sums['ct_frame_rejected_batches'] = sums.get('ct_frame_rejected_batches', 0)+int(cpu['ct_frame_rejected_batches'].sum())
        if 'dataset_id' in cpu:
            counts = sums.setdefault('dataset_counts', {})
            ids, sizes = torch.unique(cpu['dataset_id'], return_counts=True)
            for source_id, size in zip(ids.tolist(), sizes.tolist()):
                counts[str(source_id)] = counts.get(str(source_id), 0)+size
        for prefix in ('ct_frame', 'history_frame'):
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
        if model.cfg.memory == 'decisions':
            # Entries recorded by earlier training forwards of the same live chain.
            recorded = batch['x']['history_keys'] >= 0
            if bool(recorded.any()):
                if live_continuation is None:
                    raise ValueError('Recorded decision memory requires the live continuation store')
                features, found = live_continuation.attach_memory(batch['x'], model.cfg.memory_entry_shape)
                missing = recorded & ~found
                batch['x']['history_features'] = features
                batch['x']['history_valid'] = batch['x']['history_valid'] & ~missing
                sums['memory_recorded'] = sums.get('memory_recorded', 0)+int(found.sum())
                sums['memory_missing'] = sums.get('memory_missing', 0)+int(missing.sum())
            sums['memory_encoded'] = sums.get('memory_encoded', 0)+int(cpu['x']['history_encode'].sum())
        if 'history_valid' in cpu['x']:  # memory inputs (absent without memory)
            valid = cpu['x']['history_valid']
            for name, value in dict(history_valid_slabs=valid.sum(),
                    history_age_sum=cpu['x']['history_ages'][valid].sum(),
                    history_overlap_sum=cpu['x']['history_overlap'][valid].sum(),
                    history_load_seconds=cpu['x']['history_load_seconds'].sum()).items():
                sums[name] = sums.get(name, 0.)+float(value)
        if diagnostic is not None:
            diagnostic.update(cpu_batch=cpu)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            output = training_prediction(model, batch['x'], batch['hist'], batch['hmask'],
                                         n_commit=commit_window(model.cfg, n_commit), targets=batch)
            terms = model.training_loss(output, batch, model.cfg, tolerance, n_commit=n_commit)
            geometry = terms['geometry_per_state'].sum()/denominator
            confidence = terms['confidence_per_state'].sum()/denominator
            loss = geometry + confidence_weight*confidence
            if 'tube_logits' in output:
                from vesuvius.neural_tracing.fiber_follow.models.whole_crop import tube_loss
                tube, tube_counts = tube_loss(output['tube_logits'], batch, model.cfg.fine, tube_sigma)
                loss = loss + tube_weight*tube.sum()/denominator
                accumulate(sums, 'tube_loss_sum', tube.detach().sum())
                accumulate(sums, 'tube_states', torch.ones_like(tube).sum())
            if 'identity_loss_per_state' in output:
                loss = loss + identity_weight*output['identity_loss_per_state'].sum()/denominator
            if 'identity_logits' in output:
                from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import verification_loss, verification_targets
                verify = verification_targets(output, batch, identity_exclude_synthetic)
                if getattr(model, '_identity_balance', None) is None:
                    model._identity_balance = torch.ones(2, 2, device=verify['known'].device)
                verify_loss, verify_planes = verification_loss(output['identity_logits'], verify, model._identity_balance)
                loss = loss + identity_weight*verify_loss.sum()/denominator
        loss.backward()
        if 'identity_logits' in output:
            from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import update_balance, verification_metrics
            model._identity_balance = update_balance(model._identity_balance, verify['label'],
                                                     verify['on_path'], verify['train'])
            for key, value in verification_metrics(output, batch, verify, verify_loss, verify_planes).items():
                accumulate(sums, key, value)
        if 'identity_pairs' in output:
            from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_DISTANCE
            pairs, correct = output['identity_pairs'], output['identity_correct']
            departed = batch['match_distance'] > DEPARTURE_DISTANCE
            age = batch['identity_departure_age']
            # Departures the crop can still see (<=24 voxels) vs older ones only memory can reveal.
            groups = dict(departed_recent=departed & (age <= 24), departed_old=departed & (age > 24),
                          departed_old_afv=departed & (age > 24) & (batch['dataset_id'] != 0 if 'dataset_id' in batch
                                                                    else torch.zeros_like(departed)))
            values = [('memory_identity_loss_sum', output['identity_loss_per_state'].detach().sum()),
                      ('memory_identity_states', (pairs.any(-1) | output['identity_anchor_valid']).sum()),
                      ('memory_identity_pairs', pairs.sum()), ('memory_identity_correct', correct.sum()),
                      ('memory_identity_anchor_pairs', output['identity_anchor_valid'].sum()),
                      ('memory_identity_anchor_correct', output['identity_anchor_correct'].sum()),
                      ('memory_identity_control_pairs', output['identity_control_pairs'].sum()),
                      ('memory_identity_control_correct', output['identity_control_correct'].sum())]
            for name, rows in groups.items():
                values += [(f'memory_identity_{name}_pairs', (pairs & rows[:, None]).sum()),
                           (f'memory_identity_{name}_correct', (correct & rows[:, None]).sum())]
            for key, value in values:
                accumulate(sums, key, value)
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
        for cpu in batches:
            rows = per_state[offset:offset+len(cpu['hist'])]
            offset += len(cpu['hist'])
            ledger.add(cpu, step, rows.T.tolist())
    if model._history_timings:
        sums['history_encode_seconds'] = sum(t[0].elapsed_time(t[1])/1000 if isinstance(t, tuple) else t
                                             for t in model._history_timings)
    # The summed loss is finite only if every batch loss was; checked before any update.
    if not math.isfinite(sums['loss']):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    if total:
        finish_training_update(model)
        sums.update(clip_training_gradients(model, history_grad_clip, rest_grad_clip))
        opt.step()
        update_ema(ema, model, step, ema_decay, ramp=ema_ramp)
    sums.update(observed_states=observed, supervised_states=total,
                observation_only_states=observed-total, optimizer_applied=bool(total),
                positive_confidence_targets=float(per_state[:, 0].sum()),
                negative_confidence_targets=float(per_state[:, 1].sum()))
    if 'history_valid_slabs' in sums:
        slab_count = max(1., sums['history_valid_slabs'])
        sums.update(history_age_mean=sums['history_age_sum']/slab_count,
                    history_overlap_mean=sums['history_overlap_sum']/slab_count,
                    history_valid_slabs_mean=sums['history_valid_slabs']/denominator)
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
    sums['prediction_loss_type'] = ('flow' if model.cfg.model_type == 'flow_matching' else
                                    'whole-crop geometry' if model.cfg.path_planes == 'crop' else 'geometry')
    if 'tube_states' in sums:
        sums['tube_loss'] = sums['tube_loss_sum']/max(1., sums['tube_states'])
    sums['prediction_loss'] = sums['geometry']
    if model.cfg.model_type == 'flow_matching':
        sums['flow'] = sums['geometry']
    return sums


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('coordinate_regression', 'flow_matching'), default=None,
                    help='Follower type; default coordinate_regression, inferred from checkpoint on resume/init')
    ap.add_argument('--flow-steps', type=int, default=4)
    ap.add_argument('--flow-draws', type=int, default=64)
    ap.add_argument('--flow-calibration-states', type=int, default=2048)
    ap.add_argument('--flow-samples', type=int, default=FlowConfig.flow_samples,
                    help='Flow: Gaussian-start proposals after the zero-start path, scored in training and '
                         'used as retries in tracing')
    ap.add_argument('--flow-sample-scale', type=float, default=FlowConfig.flow_sample_scale,
                    help='Flow: standard deviation of those starts, in residual-scale units')
    ap.add_argument('--flow-time-conditioning', choices=('input', 'adaln'), default=FlowConfig.flow_time_conditioning,
                    help="Flow: 'adaln' also modulates every decoder branch and the output norm by the time")
    ap.add_argument('--flow-sigma-floor', type=float, default=FlowConfig.flow_sigma_floor,
                    help='Flow: lower bound (voxels) of the fitted residual scales, the width of the noise prior')
    ap.add_argument('--flow-unknown-planes', choices=('padded', 'own_path'), default=FlowConfig.flow_unknown_planes,
                    help="Flow: 'own_path' keeps target-less planes in self-attention, following the model's own path")
    ap.add_argument('--dagger-threads', type=int, default=4)
    ap.add_argument('--name', required=True)
    ap.add_argument('--fiber-zarrs')
    ap.add_argument('--fibers')
    ap.add_argument('--ct')
    ap.add_argument('--manifest', help='Frozen Paris 4 monitor/calibration/test seeds; also configurable in --dataset-config')
    ap.add_argument('--dataset-config', help='JSON source paths, fiber holdouts, cache directory and sampling weights')
    ap.add_argument('--onpolicy', nargs='*', default=[])
    ap.add_argument('--out-root', default=str(Path(__file__).parents[1]/'output'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--frame-checkpoint', help='Frozen learned heading/normal checkpoint; new runs default to the tested 64k model')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=4, help='Decisions per forward/backward pass')
    ap.add_argument('--grad-steps', type=int, default=2,
                    help='Batches accumulated per optimizer update; effective batch = batch * grad steps')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--worker-cache-gb', type=float, default=.5)
    ap.add_argument('--remote-prefetch-connections', type=int, default=0,
                    help='Concurrent async chunk fetches and S3 pool limit per client in a separate process; 0 disables')
    ap.add_argument('--remote-prefetch-queue-size', type=int, default=512,
                    help='Bound per priority queue; required batches override speculative fetches')
    ap.add_argument('--remote-prefetch-lookahead', type=int, default=16,
                    help='Future batch plans per remote source per worker; 0 disables deeper lookahead')
    ap.add_argument('--remote-prefetch-timeout', type=float, default=120.,
                    help='Maximum seconds waiting for batch cache readiness; no foreground network fallback')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--ema-decay', type=float, default=.999)
    ap.add_argument('--history-grad-clip', type=float, default=HISTORY_GRAD_CLIP,
                    help='Gradient-norm limit for historical slab encoder; 0 disables clipping')
    ap.add_argument('--rest-grad-clip', type=float, default=REST_GRAD_CLIP,
                    help='Independent gradient-norm limit for all other parameters; 0 disables clipping')
    ap.add_argument('--confidence-weight', type=float, default=.5)
    ap.add_argument('--tolerance', type=float, default=1.5)
    ap.add_argument('--n-commit', type=int, default=16)
    ap.add_argument('--stem-channels', type=int, default=CoordinateRegressionConfig.stem_channels,
                    help='Parallel residual patch stem width; 0 disables (token-only patch4 models)')
    ap.add_argument('--stem-blocks', type=int, default=CoordinateRegressionConfig.stem_blocks,
                    help='BasicBlockD blocks per downsampling stage in the optional patch stem')
    ap.add_argument('--decoder-layers', type=int, default=CoordinateRegressionConfig.decoder_layers)
    ap.add_argument('--decoder-ffn', type=int, default=CoordinateRegressionConfig.decoder_ffn,
                    help='Path decoder feed-forward width; scorer stays at 1024')
    ap.add_argument('--scorer-layers', type=int, default=CoordinateRegressionConfig.scorer_layers,
                    help='Causal segment survival decoder depth')
    ap.add_argument('--axial-layers', type=int, default=CoordinateRegressionConfig.layers)
    crop = CoordinateRegressionConfig.__dataclass_fields__['fine'].default_factory()
    ap.add_argument('--crop-depth', type=int, default=crop.depth,
                    help='Model crop samples along the heading (multiple of four)')
    ap.add_argument('--crop-width', type=int, default=crop.width,
                    help='Model crop samples across the heading, both lateral axes (multiple of four)')
    ap.add_argument('--crop-behind', type=int, default=crop.behind,
                    help='Model crop samples behind the head along the heading')
    ap.add_argument('--crop-spacing', type=float, default=crop.spacing,
                    help='Trace voxels per crop sample. The CT level read is set per source in the dataset config; '
                         'with the current sources 0.5 is one sample per level-0 CT voxel')
    ap.add_argument('--stem', choices=('residual', 'stride2'), default=CoordinateRegressionConfig.stem,
                    help="Image stem: 'residual' full-resolution stem, or 'stride2' light stem")
    ap.add_argument('--memory', choices=('slabs', 'decisions', 'none'), default=CoordinateRegressionConfig.memory,
                    help="History memory: 'slabs' separate slab CNN, 'decisions' main-encoder entries "
                         "of earlier decisions (live chains start at seeds and reuse recorded entries), or 'none' "
                         "(no memory inputs, tokens or attention; incompatible with the identity objectives)")
    ap.add_argument('--chain-seed-fraction', type=float, default=0.,
                    help='With --memory decisions: start live chains at a seed within this leading fraction '
                         'of the traversal (0 keeps the ordinary fresh location)')
    ap.add_argument('--identity-dim', type=int, default=CoordinateRegressionConfig.identity_dim,
                    help='With --memory decisions: width of the linear projection for the memory-identity '
                         'InfoNCE loss (0 disables it)')
    ap.add_argument('--identity-temperature', type=float, default=CoordinateRegressionConfig.identity_temperature)
    ap.add_argument('--path-planes', choices=('future', 'crop'), default=CoordinateRegressionConfig.path_planes,
                    help="Path head: 'future' planes 1..n_future, or 'crop' one point per crop plane behind and ahead "
                         "of the head, trained toward the original fiber wherever it is (planes 1..n_future stay the "
                         "committed, scored and measured proposal)")
    ap.add_argument('--tube-head', action=argparse.BooleanOptionalAction, default=CoordinateRegressionConfig.tube_head,
                    help='Dense Gaussian-tube head around the original fiber over the whole crop (auxiliary loss)')
    ap.add_argument('--tube-sigma', type=float, default=1.5, help='Tube target width (trace voxels)')
    ap.add_argument('--tube-weight', type=float, default=1., help='Tube loss coefficient')
    ap.add_argument('--identity-weight', type=float, default=.1,
                    help='Memory-identity loss coefficient (InfoNCE or verification)')
    ap.add_argument('--identity-objective', choices=('infonce', 'verify', 'readout'),
                    default=CoordinateRegressionConfig.identity_objective,
                    help="With --memory decisions: 'infonce' (needs --identity-dim); 'verify', a verifier that "
                         "reads decision-memory tokens and labels current-crop locations on/off the original fiber; "
                         "or 'readout', path-blind appearance keys stored per decision and read out per location "
                         "(same labels and loss as verify)")
    ap.add_argument('--identity-map', action=argparse.BooleanOptionalAction,
                    default=CoordinateRegressionConfig.identity_map,
                    help='With --identity-objective verify/readout: add the dense identity field to decoder/scorer image tokens')
    ap.add_argument('--identity-feedback', action=argparse.BooleanOptionalAction,
                    default=CoordinateRegressionConfig.identity_feedback,
                    help='With --identity-objective verify/readout: feed identity-field samples to scorer segments and retries')
    ap.add_argument('--identity-exclude-synthetic', action=argparse.BooleanOptionalAction, default=False,
                    help='With --identity-objective verify: synthetic switch states (synthetic_terminal/'
                         'synthetic_identity) only evaluate the verifier; its loss uses real states')
    ap.add_argument('--identity-switch-tail', type=float, nargs=2, default=IdentitySampling.identity_switch_tail,
                    metavar=('MIN', 'MAX'), help='Neighbor tail of the synthetic_identity task, trace voxels')
    ap.add_argument('--memory-augmentation', choices=('shared', 'independent'),
                    default=IdentitySampling.memory_augmentation,
                    help="Photometric/blur draw of memory crops encoded in training: the current crop's, or per crop")
    ap.add_argument('--lr-step-offset', type=int, default=0,
                    help='Continue another schedule: the cosine is evaluated at step+offset over steps+offset '
                         '(warmup still counts this run\'s own updates)')
    ap.add_argument('--init-partial', action='store_true',
                    help='With --init-weights: build the model from these options and load only tensors whose '
                         'names and shapes match; new axial blocks and the new stem start as identity residuals')
    ap.add_argument('--init-exclude', action='append', default=[], metavar='PREFIX',
                    help='With --init-partial: keep the initialization of tensors whose names start with PREFIX '
                         '(repeatable), e.g. zero-initialized heads whose input changed meaning')
    ap.add_argument('--hidden', type=int, default=CoordinateRegressionConfig.hidden)
    ap.add_argument('--encoder-ffn', type=int, default=CoordinateRegressionConfig.encoder_ffn,
                    help='Axial image encoder feed-forward width')
    ap.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--task-share', action='append', default=[], metavar='TASK=SHARE',
                    help='Override one task share (fresh, live, dagger_pre_excursion, dagger_recoverable, '
                         'dagger_terminal, dagger_premature_stop, dagger_ordinary, synthetic_terminal, '
                         'synthetic_identity); shares sum to one')
    ap.add_argument('--terminal-fallback-cap', type=float, default=TaskBudget.terminal_fallback_cap,
                    help='Largest fraction of terminal replay slots filled by certified synthetic failures')
    ap.add_argument('--replay-max-age', type=int, default=TaskBudget.replay_max_age,
                    help='Updates after which a replay cache is no longer drawn')
    ap.add_argument('--replay-event-cap', type=int, default=TaskBudget.replay_event_cap,
                    help='Draws per replay event, shared by all loader workers of a source')
    ap.add_argument('--startup-shares', type=float, nargs=4, default=SampleConfig.startup_shares,
                    metavar=('SEED_ONLY', 'SHORT', 'EARLY', 'ESTABLISHED'),
                    help='Fresh trace starts: seed only, 1-8 and 9-32 requested voxels, uniform available history')
    ap.add_argument('--excursion-probability', type=float, default=SampleConfig.excursion_probability,
                    help='Established fresh traces given a smooth lateral excursion')
    ap.add_argument('--excursion-amplitude', type=float, nargs=2, default=SampleConfig.excursion_amplitude)
    ap.add_argument('--excursion-rise', type=float, nargs=2, default=SampleConfig.excursion_rise)
    ap.add_argument('--synthetic-tail', type=float, nargs=2, default=(4., 16.), metavar=('MIN', 'MAX'),
                    help='Neighbor tail of certified synthetic failures, trace voxels')
    ap.add_argument('--live-continuation-steps', type=int, nargs=2, default=(12, 32), metavar=('MIN', 'MAX'),
                    help='Live chain decision limits, balanced over three contiguous bands')
    ap.add_argument('--negative-bank', default=str(Path(__file__).parents[1]/'output'/'neighbor_samples_r0_32_l80_160_v2'), help='Shared live bank for foreign-fiber masks, wrong continuations and following supervision')
    ap.add_argument('--near-negative-bank', help='Additional bank of validated nearby negative relationships')
    ap.add_argument('--continuation-bank', help='Synthetic failure path source (default: negative-bank)')
    ap.add_argument('--bank-coverage-probability', type=float, default=.2, help='Fresh locations on covered parents with usable history')
    ap.add_argument('--negative-bank-refresh-seconds', type=float, default=30., help='Each loader worker checks for completed new negative shards at this interval')
    ap.add_argument('--negative-bank-cache-mb', type=float, default=64., help='Maximum cached negative geometry per loader worker')
    ap.add_argument('--bank-hard-fraction', type=float, default=.5,
                    help='Fraction of bank draws ranked by nearby similar, curved, or converging geometry')
    ap.add_argument('--bank-switch-tolerance', type=float, default=.75,
                    help='Foreign centerline contact radius for DAgger labels, in trace voxels')
    ap.add_argument('--bank-own-tolerance', type=float, default=1.5,
                    help='Annotation tube excluded from confirmed foreign contact')
    ap.add_argument('--blur-probability', type=float, default=.25,
                    help='Probability of shared CT/presence Gaussian blur; may be changed on resume')
    ap.add_argument('--blur-sigma', type=float, nargs=2, default=(.5, 1.25), metavar=('MIN', 'MAX'),
                    help='Gaussian blur sigma range in sampled crop voxels; may be changed on resume')
    ap.add_argument('--lateral-fraction', type=float, default=.1,
                    help='Fresh draws near earlier states with bank negatives')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000., 48500.))
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--diag-every', type=int, default=5000)
    ap.add_argument('--batch-diag-every', type=int, default=1000,
                    help='Current microbatch prediction, orientation and model-layer contact sheets; 0 disables')
    ap.add_argument('--diag-max-len', type=float, default=400.)
    ap.add_argument('--long-diag-every', type=int, default=0, help='Additional long monitor rollouts; 0 disables')
    ap.add_argument('--long-diag-max-len', type=float, default=1200.)
    ap.add_argument('--recovery-every', type=int, default=1000, help='Fixed monitor recovery diagnostic cadence; 0 disables')
    ap.add_argument('--recovery-seeds', type=int, default=8, help='First N frozen monitor seeds, four displacement strata each')
    ap.add_argument('--recovery-length', type=float, default=32.)
    ap.add_argument('--dagger-every', type=int, default=1000,
                    help='Global collection cadence; sources take turns and a busy launch is skipped')
    ap.add_argument('--dagger-fibers', type=int, default=64,
                    help='Distinct fibers per collection, one directed episode each')
    ap.add_argument('--afv-dagger-fibers', type=int,
                    help='Distinct fibers per AFV collection (default: --dagger-fibers); AFV coverage order '
                         'favors fibers by length**--afv-length-power, like AFV training draws')
    ap.add_argument('--dagger-batch', type=int, default=8, help='Traces per collector forward batch')
    ap.add_argument('--dagger-forward-chunk', type=int, default=0, help='Rows per collector model call; 0 = whole batch')
    ap.add_argument('--afv-length-power', type=float, default=1.,
                    help='AFV training fiber draws, and AFV collection order, proportional to length**p')
    ap.add_argument('--dagger-device')
    ap.add_argument('--dagger-trace-len', type=float, default=768.)
    ap.add_argument('--dagger-before', type=float, default=48., help='Dense decisions kept before an excursion')
    ap.add_argument('--dagger-after', type=float, default=64., help='Voxels kept after the first terminal failure')
    ap.add_argument('--dagger-stride', type=float, default=16., help='Travel between kept ordinary decisions')
    ap.add_argument('--replay-keep', type=int, default=4)
    ap.add_argument('--init-weights', help='Start a new run from matching model and EMA tensors of this checkpoint; '
                                           'fresh optimizer, sampler and options')
    ap.add_argument('--resume', help='Resume last.pt inside this run with the same training options')
    ap.add_argument('--reset-optimizer', action='store_true',
                    help='With --resume: fresh AdamW, one LR for all parameters, no transfer freeze, and restart LR warmup/decay')
    ap.add_argument('--recurrent-refinement-steps', type=int, default=CoordinateRegressionConfig.recurrent_refinement_steps,
                    help='Maximum additional absolute-coordinate attempts; stop early when the full path is accepted')
    return ap


def options_argv(options):
    """Serialize effective trainer settings for a bounded benchmark run."""
    argv = []
    for action in build_parser()._actions:
        if action.dest == 'help' or action.dest not in options:
            continue
        value = options[action.dest]
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(action.option_strings[0 if value else 1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
        else:
            argv.append(action.option_strings[0])
            argv.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return argv


def validate_resume_options(args, recorded_options):
    # Allow a new base LR without resetting AdamW or the schedule origin.
    # Extending the endpoint also changes the cosine factor; callers preserving
    # the current effective LR must rescale the base LR for the new endpoint.
    # The frame selection comes from model_cfg unless explicitly overridden;
    # an omitted CLI flag must not prevent resuming that recorded selection.
    # Initialization is historical provenance, not a resume-time input.
    # Operational settings may change on resume; sampling and label semantics may not.
    ignored = {'resume','init_weights','init_partial','init_exclude','reset_optimizer','lr','steps','frame_checkpoint','out_root','device','batch','grad_steps','workers','threads','dagger_threads','worker_cache_gb','dataset_config',
               'remote_prefetch_connections','remote_prefetch_queue_size','remote_prefetch_timeout','remote_prefetch_lookahead',
               'log_every','ckpt_every','diag_every','batch_diag_every','dagger_device','dagger_batch','dagger_forward_chunk',
               'negative_bank_refresh_seconds','negative_bank_cache_mb','activation_checkpointing',
               'history_grad_clip','rest_grad_clip','blur_probability','blur_sigma',
               'tolerance'}  # confidence-label tolerance; the departure threshold is a separate constant
    defaults = build_parser()
    for key,value in vars(args).items():
        # Options added after a run started were recorded implicitly at their default.
        recorded = recorded_options[key] if key in recorded_options else defaults.get_default(key)
        if key not in ignored and json.dumps(recorded,sort_keys=True) != json.dumps(value,sort_keys=True):
            raise ValueError(f'Resume option differs: {key}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset_document = dataset_digest = None
    if args.dataset_config:
        from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config, apply_primary_source
        dataset_document, dataset_digest = read_dataset_config(args.dataset_config)
        apply_primary_source(args, dataset_document)
    if any(not getattr(args, key) for key in ('manifest', 'fiber_zarrs', 'fibers', 'ct')):
        raise ValueError('Provide source paths and --manifest, or --dataset-config')
    launch_started = time.monotonic()

    def progress(message):
        print(f'[{args.name} +{time.monotonic()-launch_started:.1f}s] {message}', flush=True)

    progress(f'Starting training: device={args.device}, workers={args.workers}')
    if args.reset_optimizer and not args.resume:
        raise ValueError('--reset-optimizer requires --resume')
    if args.init_weights and args.resume:
        raise ValueError('--init-weights starts a new run; it cannot be combined with --resume')
    if any(not math.isfinite(v) or v < 0 for v in (args.history_grad_clip, args.rest_grad_clip)):
        raise ValueError('Gradient clipping limits must be finite and nonnegative (0 disables clipping)')
    budget = TaskBudget.parse(args.task_share, terminal_fallback_cap=args.terminal_fallback_cap,
                              replay_max_age=args.replay_max_age, replay_event_cap=args.replay_event_cap)
    if (args.remote_prefetch_connections < 0 or args.remote_prefetch_queue_size < 1 or args.remote_prefetch_lookahead < 0
            or not math.isfinite(args.remote_prefetch_timeout) or args.remote_prefetch_timeout <= 0):
        raise ValueError('Prefetch connections and lookahead must be nonnegative; queue size and timeout must be positive')
    if min(args.steps, args.batch, args.grad_steps, args.log_every, args.ckpt_every,
           args.threads, args.dagger_threads, args.flow_calibration_states, args.replay_keep, args.dagger_fibers, args.dagger_batch, args.recovery_seeds) < 1:
        raise ValueError('Positive counts required, including batch and grad steps')
    if args.afv_dagger_fibers is not None and args.afv_dagger_fibers < 1:
        raise ValueError('AFV collection fiber count must be positive')
    if min(args.workers, args.warmup, args.diag_every, args.batch_diag_every,
           args.long_diag_every, args.dagger_every, args.recovery_every, args.confidence_weight) < 0:
        raise ValueError('Invalid training settings')
    if not 0 <= args.ema_decay < 1 or min(args.lr, args.tolerance, args.worker_cache_gb, args.dagger_after,
                                       args.diag_max_len, args.long_diag_max_len, args.dagger_trace_len, args.recovery_length) <= 0:
        raise ValueError('Invalid loss, learning rate, cache, or rollout settings')
    if args.val_z[0] >= args.val_z[1]:
        raise ValueError('Holdout interval must be increasing')
    if not math.isfinite(args.afv_length_power) or args.afv_length_power < 0:
        raise ValueError('AFV length power must be finite and nonnegative')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    raise_open_file_limit()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    resume = read_checkpoint(args.resume,MODEL_TYPES,args.device) if args.resume else None
    if resume is not None and resume.get('kind') == 'weights':
        raise ValueError('Weight-only files require --init-weights, not --resume')
    initial = read_checkpoint(args.init_weights,MODEL_TYPES,'cpu') if args.init_weights else None
    if args.init_exclude and not args.init_partial:
        raise ValueError('--init-exclude requires --init-partial')
    if args.init_partial and not args.init_weights:
        raise ValueError('--init-partial requires --init-weights')
    if not 0 <= args.chain_seed_fraction <= 1:
        raise ValueError('Chain seed fraction must lie in [0, 1]')
    initialization = resume.get('initialization') if resume else args.init_weights
    # A partial warm start builds the requested architecture; a full one copies the source's.
    cfg = model_config_from_args(args, resume or (None if args.init_partial else initial))
    inherited_frame = initial['model_cfg'].get('frame_checkpoint') if args.init_partial else None
    if args.frame_checkpoint or not resume:
        bind_frame_checkpoint(cfg, args.frame_checkpoint or cfg.frame_checkpoint or inherited_frame
                              or DEFAULT_FRAME_CHECKPOINT)
    frame_predictor(cfg)  # validate once before dataset loading; workers reuse a frozen CPU predictor
    args.model = cfg.model_type
    if resume:
        from vesuvius.neural_tracing.fiber_follow.data.datasets import validate_dataset_resume
        validate_dataset_resume(resume,dataset_document,dataset_digest)
    if not 1 <= args.live_continuation_steps[0] <= args.live_continuation_steps[1]:
        raise ValueError('Live continuation requires 1 <= minimum steps <= maximum steps')
    from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import BankSwitchDetector
    BankSwitchDetector([], args.bank_switch_tolerance, args.bank_own_tolerance)
    identity_sampling = IdentitySampling(
        blur_probability=args.blur_probability,blur_sigma=args.blur_sigma,
        lateral_fraction=args.lateral_fraction,bank_hard_fraction=args.bank_hard_fraction,
        bank_coverage_probability=args.bank_coverage_probability,synthetic_tail=args.synthetic_tail,
        identity_switch_tail=args.identity_switch_tail,memory_augmentation=args.memory_augmentation)
    if not args.negative_bank:
        raise ValueError('Training requires --negative-bank')
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Finest-level CT in a single enlarged crop.
    primary_source = (next(s for s in dataset_document['sources'] if s['kind'] == 'paris4')
                      if dataset_document else {})
    from vesuvius.neural_tracing.fiber_follow.data.datasets import primary_source_spec
    spec = primary_source_spec(dataset_document, fiber_zarrs=args.fiber_zarrs, ct=args.ct)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                          future_step=cfg.future_step, recent_history_points=cfg.n_history,
                          startup_shares=tuple(args.startup_shares), excursion_probability=args.excursion_probability,
                          excursion_amplitude=tuple(args.excursion_amplitude), excursion_rise=tuple(args.excursion_rise),
                          label_tolerance=args.tolerance, max_recovery_distance=cfg.max_recovery_distance,
                          crop_targets=getattr(cfg, 'path_planes', 'future') == 'crop' or getattr(cfg, 'tube_head', False))
    policy = training_policy(cfg, args.n_commit)
    progress('Loading manifest and fiber annotations')
    bank_band = ZBand(*(v/spec.grid_scale for v in args.val_z))
    if dataset_document:
        from vesuvius.neural_tracing.fiber_follow.data.datasets import load_primary_dataset, HoldoutFilteredBank
        fibers,train_f,val_f,manifest = load_primary_dataset(dataset_document,spec)
        band = None
    else:
        manifest = read_manifest(args.manifest)
        validate_volume_source(spec, manifest)
        fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
        band = bank_band
        train_f, val_f = split_fibers(fibers, band)
    progress(f'Loaded {len(train_f)} training fibers and {len(val_f)} validation fibers')
    if fiber_manifest(val_f) != manifest['fibers']:
        raise ValueError('Frozen validation geometry differs from dataset/holdout')
    negative_bank = None
    role_banks = {}
    if args.negative_bank:
        from vesuvius.neural_tracing.fiber_follow.data.neighbor_bank import NeighborBank
        bank_class = HoldoutFilteredBank if dataset_document else NeighborBank
        bank_kwargs = dict(heldout=val_f) if dataset_document else {}
        negative_bank = bank_class(args.negative_bank,train_f,bank_band,grid_scale=spec.grid_scale,**bank_kwargs,
            refresh_seconds=args.negative_bank_refresh_seconds,cache_bytes=int(args.negative_bank_cache_mb*(1 << 20)))
        negative_bank.validate_volume(spec)
        if resume and (resume.get('negative_bank_provenance') is not None or resume['training_options'].get('negative_bank')):
            negative_bank.validate_resume(resume.get('negative_bank_provenance'))
        progress(f'Live negative bank: {negative_bank.shard_count} published shards, refresh every {args.negative_bank_refresh_seconds:g}s per worker')
        by_path = {negative_bank.root:negative_bank}
        for role in ('near_negative_bank','continuation_bank'):
            path = getattr(args,role)
            if path is None:
                continue
            root = Path(path).resolve()
            if root.name == 'bank.json':
                root = root.parent
            if root not in by_path:
                by_path[root] = bank_class(root,train_f,bank_band,grid_scale=spec.grid_scale,**bank_kwargs,
                    refresh_seconds=args.negative_bank_refresh_seconds,cache_bytes=int(args.negative_bank_cache_mb*(1 << 20)))
                by_path[root].validate_volume(spec)
            role_banks[role] = by_path[root]
            if resume:
                role_banks[role].validate_resume((resume.get('bank_role_provenance') or {}).get(role))
    def role_provenance():
        return {role:bank.provenance() for role,bank in role_banks.items()}
    if resume:
        validate_resume_options(args, resume['training_options'])
        if resume['seed_manifest_sha256'] != manifest['sha256'] or resume['fiber_manifest'] != fiber_manifest(fibers):
            raise ValueError('Resume data/manifest changed')
    if initial:
        # Fiber identities and source holdouts must match the weights' training data, so no
        # held-out fiber was trained on. Geometry (load-time annotation repair) and the seeds
        # placed on it may differ; the new run evaluates both checkpoints on its own seeds.
        if fiber_identities(initial['fiber_manifest']) != fiber_identities(fiber_manifest(fibers)):
            raise ValueError('Initialization checkpoint used different fibers or evaluation splits')
        from vesuvius.neural_tracing.fiber_follow.data.datasets import same_sources
        if not same_sources(initial.get('dataset_config'), dataset_document):
            raise ValueError('Initialization checkpoint used different dataset sources or holdouts')
    out = prepare_run_dir(args.out_root, args.name, resume is not None)
    if dataset_document:
        (out/'validation_paris4.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if resume and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError('Resume checkpoint must be inside the named run')
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import prepare_normalization
    calibration_specs = [spec]
    if dataset_document:
        from vesuvius.neural_tracing.fiber_follow.data.datasets import ct_source_spec
        calibration_specs.extend(ct_source_spec(s, dataset_document['cache_dir'])
                                 for s in dataset_document['sources'] if s['kind'] != 'paris4')
    progress('Preparing per-crop CT z-score normalization')
    ct_normalization = prepare_normalization(out, calibration_specs,
        resume=resume['ct_normalization'] if resume is not None else None)
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
    progress(f'Independent gradient clipping: history={args.history_grad_clip:g}, rest={args.rest_grad_clip:g} (0 disables clipping)')
    # Flow scales are fitted after constructing the common training dataset below, unless the
    # configuration came with them (resume or a full flow warm start). A partial warm start builds
    # the configuration from arguments and therefore fits them, even from a flow checkpoint.
    fit_flow_scales = isinstance(cfg, FlowConfig) and not cfg.flow_sigma
    if fit_flow_scales:
        cfg.flow_sigma = tuple((cfg.flow_sigma_floor,)*2 for _ in range(cfg.n_future))
    model, ema = initialize_model_weights(cfg, args.device, initial, partial=args.init_partial,
                                          exclude=tuple(args.init_exclude))
    if args.init_partial:
        progress(f'Partial warm start: {len(model.reinitialized_tensors)} tensors left at initialization')
    opt, done, lr_restart_step = initialize_training_optimizer(model, ema, args, resume)
    if args.reset_optimizer:
        progress(f'Fresh AdamW: one parameter group, all parameters trainable; LR restarts at update {done+1} '
                 f'with {args.warmup} warmup updates to {args.lr:g}')
    prepare_training(model, args.batch)
    progress('Compiling training operations; first forward/backward passes may take several minutes')
    if done >= args.steps:
        raise ValueError('Run has already reached its requested update count')
    replay_index = out/'dagger'/'replay.json'
    replay_paths = json.loads(replay_index.read_text()) if resume and replay_index.exists() else args.onpolicy
    progress('Loading replay banks and preparing data loader')
    caches = [OnPolicyStates.load(p) for p in replay_paths]
    collection = dict(every=args.dagger_every, fibers_per_collection=args.dagger_fibers, batch=args.dagger_batch,
                      forward_chunk=args.dagger_forward_chunk, seed=args.seed, replay_keep=args.replay_keep,
                      trace_len=args.dagger_trace_len, before=args.dagger_before, after=args.dagger_after,
                      stride=args.dagger_stride, confidence=policy.confidence, n_commit=policy.n_commit,
                      collector_module='vesuvius.neural_tracing.fiber_follow.tracing.collect', threads=args.dagger_threads)
    bank_args = ('--bank-switch-tolerance', args.bank_switch_tolerance, '--bank-own-tolerance', args.bank_own_tolerance)
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.dagger_device or args.device,
        initial=[c._dir for c in caches], **collection,
        extra_args=(*bank_args, *[v for path in (args.negative_bank, args.near_negative_bank) if path for v in ('--failure-bank', path)]))
    progress(f'Live historical slabs: eight slots; {args.batch} independent decisions per batch')
    builder = IdentityObservationBuilder(cfg,train_f,identity_sampling,
        augment=True,negative_bank=negative_bank,**role_banks)
    dataset = FollowDataset(train_f, spec, sample, band, chunk=args.batch, seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), onpolicy=caches,
        replay_index=str(collector.index), batch_builder=builder, additional_crops=(), budget=budget)
    dataset_provenance = None
    if dataset_document:
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
                source['path'], (0,1), args.dagger_device or args.device,
                initial=[c._dir for c in source_dataset.onpolicy], **afv_collection,
                extra_args=('--dataset-name',source['name'],*bank_args))))
        collector = MultiSourceCollector(collectors)
        progress('Dataset sampling: '+', '.join(f'{name}={weight:.1%}' for name,weight in zip(dataset.names,dataset.weights))
                 +f'; AFV training draws and AFV collection order proportional to length**{args.afv_length_power:g}')
    progress('Task budget per source: '+', '.join(f'{k} {v:.0%}' for k, v in budget.to_dict()['shares'].items())
             +f'; replay age ceiling {budget.replay_max_age}, event cap {budget.replay_event_cap}, '
             f'synthetic terminal fallback cap {budget.terminal_fallback_cap:.0%}')
    sources = getattr(dataset, 'datasets', [dataset])
    for source_dataset in sources:
        source_dataset.set_step(done)
        source_dataset.chain_seed_only = cfg.memory == 'decisions'
        source_dataset.chain_seed_fraction = args.chain_seed_fraction if cfg.memory == 'decisions' else 0.
    from vesuvius.neural_tracing.fiber_follow.train.live_continuation import LiveContinuation, preserve_live_metadata
    live_continuation = LiveContinuation(dataset, policy=policy, steps=args.live_continuation_steps,
        switch_tolerance=args.bank_switch_tolerance, own_tolerance=args.bank_own_tolerance)
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
    if not resume:
        (out/'config.json').write_text(json.dumps(dict(vars(args), model_type=model.model_type,
            resolved_dataset_config=dataset_document, dataset_config_sha256=dataset_digest,
            dataset_provenance=dataset_provenance,
            ct_normalization=ct_normalization,
            identity_sampling=asdict(identity_sampling),
            negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
            bank_role_provenance=role_provenance(),
            model_cfg=cfg.to_dict(), sample_cfg=asdict(sample), vol_spec=spec.to_dict(),
            task_budget=budget.to_dict(), operating_policy=policy.to_dict(),
            collection=collector_settings(collector),
            initialization=dict(checkpoint=str(Path(args.init_weights).resolve()), optimizer='fresh AdamW',
                                warmup=args.warmup, partial=args.init_partial, excluded=list(args.init_exclude),
                                reinitialized=model.reinitialized_tensors) if args.init_weights else None,
            data_policy=DATA_POLICY, frame_policy=configured_frame_policy(cfg),
            monitor_recovery_sha256=recovery_hash,
            seed_manifest_sha256=manifest['sha256'], fiber_manifest=fiber_manifest(fibers),
            parameter_count=sum(p.numel() for p in model.parameters())), indent=2))
    log = RunLog(out/'log.jsonl', formatter=format_training_log)
    log.record(dict(step=done, event='ct_normalization', calibration=ct_normalization))
    if dataset_document:
        log.record(dict(step=done, event='dataset_configuration', datasets=dataset_provenance,
                        names=dataset.names, probabilities=dataset.weights.tolist(),
                        dataset_config_sha256=dataset_digest, input_mode='ct',
                        evaluation_scope='Source-specific held-out fibers; Paris 4 recovery fixture'))
    if resume:
        log.record(dict(step=done, event='resume_configuration', checkpoint=str(args.resume),
                        training_options=vars(args), model_cfg=cfg.to_dict()))
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
        history_sampling_revision=sampling_revision(cfg), frame_policy=configured_frame_policy(cfg),
        history_policy='live_observed_slabs',sampling=asdict(identity_sampling),
        negative_bank_path=str(negative_bank.root),negative_bank_provenance=negative_bank.provenance(),
        bank_role_provenance=role_provenance()))
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
            remote_prefetch, remote_sources = attach_remote_prefetch(getattr(dataset,'datasets',[dataset]),
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
        if args.diag_every or args.long_diag_every:
            from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
            tracer = FiberTracer(ema, FiberVolume(spec), cfg.fine, cfg.n_history,
                TraceParams.from_policy(policy, max_len=args.diag_max_len), device=args.device)
        progress(f'Starting data loader; batch {args.batch} × grad steps {args.grad_steps} = '
                 f'{args.batch * args.grad_steps} decisions per update; starting update {done+1}')
        iterator = iter(loader)
        updates = DecisionBatchPrefetch(iterator, args.grad_steps)
        observed_states = interval_states = 0
        prior_samples = int(resume['samples_seen']) if resume else 0
        ledger = SamplingLedger()
        for step in range(done+1, args.steps+1):
            for source_dataset in sources:
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
            lr = lr_at(step-lr_restart_step, args.lr, args.warmup, args.steps-lr_restart_step, args.lr_step_offset)
            diagnostic = {} if tracer is not None and args.diag_every and step % args.diag_every == 0 else None
            metrics = optimizer_update(model, ema, opt, batches, step, lr, device=args.device,
                tolerance=args.tolerance, confidence_weight=args.confidence_weight, ema_decay=args.ema_decay,
                ema_ramp=not initialization,
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps,
                history_grad_clip=args.history_grad_clip, rest_grad_clip=args.rest_grad_clip,
                identity_weight=args.identity_weight, identity_exclude_synthetic=args.identity_exclude_synthetic,
                tube_weight=args.tube_weight, tube_sigma=args.tube_sigma,
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
                    interval=interval_metrics.summary(), sampling=ledger.summary(),
                    n_future=cfg.n_future, tolerance=args.tolerance,
                    interval_updates=step-interval_step,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30
                        if torch.device(args.device).type == 'cuda' else None))
                from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_curves
                plot_curves(out/'log.jsonl', out/'curves.png', loss_key='flow' if model.cfg.model_type == 'flow_matching' else 'geometry')
                interval_metrics = TrainingInterval()
                ledger = SamplingLedger()
                interval_started, interval_step = now, step
                interval_data_seconds = interval_update_seconds = 0.
                interval_states = 0

            def save(path, resumable=False):
                extra = dict(step=step, lr_restart_step=lr_restart_step, tolerance=args.tolerance, n_commit=args.n_commit,
                    operating_policy=policy.to_dict(), task_budget=budget.to_dict(), sample_contract=asdict(sample),
                    initialization=str(Path(initialization).resolve()) if initialization else None,
                    dataset_config=dataset_document, dataset_config_sha256=dataset_digest,
                    dataset_provenance=dataset_provenance,
                    ct_normalization=ct_normalization,
                    history_sampling_revision=sampling_revision(cfg), frame_policy=configured_frame_policy(cfg),
                    samples_seen=prior_samples+observed_states,
                    identity_sampling=asdict(identity_sampling),
                    negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
                    bank_role_provenance=role_provenance(),
                    seed_manifest_sha256=manifest['sha256'], training_options=vars(args),
                    monitor_recovery_sha256=recovery_hash,
                    fiber_manifest=fiber_manifest(fibers))
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
            if args.batch_diag_every and (step % args.batch_diag_every == 0
                                         or (resume is not None and step == done+1)):
                from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import render_microbatch
                began = time.monotonic()
                names = dataset.names if dataset_document else [primary_source.get('name', 'paris4')]
                report = render_microbatch(ema, batches[-1], out, step, device=args.device,
                    n_commit=args.n_commit, tolerance=args.tolerance, dataset_names=names,
                    memory=live_continuation,
                    training_metrics=dict(metrics, lr=lr, data_seconds=data_seconds,
                        update_seconds=update_seconds,
                        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30
                            if torch.device(args.device).type == 'cuda' else None))
                log.record(dict(step=step, split='current_training_microbatch', diagnostic_images=report))
                periodic['batch_diagnostics_seconds'] = time.monotonic()-began
            if tracer is not None and args.diag_every and step % args.diag_every == 0:
                began = time.monotonic()
                training_diagnostics(ema, diagnostic['cpu_batch'], tracer, val_f, manifest['monitor'], out, step, log,
                    memory=live_continuation,
                                     device=args.device,dataset_name=next((s['name'] for s in dataset_document['sources']
                                         if s['kind']=='paris4'),None) if dataset_document else None)
                if dataset_document:
                    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import evaluate
                    from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary
                    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
                    for source, source_dataset in zip(dataset_document['sources'],dataset.datasets):
                        if source['kind'] != 'afv':
                            continue
                        source_tracer = FiberTracer(ema,FiberVolume(source_dataset.vol_spec),cfg.fine,cfg.n_history,
                            TraceParams(n_commit=args.n_commit,max_len=args.diag_max_len),device=args.device)
                        try:
                            rows,_ = evaluate(source_tracer,source_dataset.validation_fibers,
                                source_dataset.validation_manifest['monitor'],batch=1,coverage_max_len=args.diag_max_len)
                            log.record(dict(step=step,split='monitor',dataset=source['name'],
                                threshold=.5,coverage_max_len=args.diag_max_len,**rollout_summary(rows)))
                        finally:
                            source_tracer.close()
                    from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_curves
                    plot_curves(out/'log.jsonl',out/'curves.png',loss_key='flow' if model.cfg.model_type == 'flow_matching' else 'geometry')
                periodic['diagnostics_seconds'] = time.monotonic()-began
            if tracer is not None and args.long_diag_every and step % args.long_diag_every == 0:
                from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import evaluate
                from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary
                from vesuvius.neural_tracing.fiber_follow.evaluation.diag import plot_rollouts
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
