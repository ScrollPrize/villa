"""Train a direct curve follower from scratch, with original-fiber online replay."""
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

from vesuvius.neural_tracing.fiber_follow.shared.data import (
    DATA_POLICY, FollowDataset, OnPolicyStates, SampleConfig, ZBand, fiber_manifest, load_fibers, split_fibers, REPLAY_FAILURES,
)
from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.shared.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.shared.runloop import (
    RunLog, lr_at, prepare_run_dir, read_checkpoint, save_checkpoint as write_checkpoint,
    update_ema, training_rng_state, resume_training, raise_open_file_limit,
)
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    ARCHITECTURE, PATCH_ARCHITECTURE, TOKEN_ARCHITECTURE, DirectConfig, build_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, DirectTracer, LOCATION_SOURCES,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import commit_window, loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import (
    decision_rows, summarize_decisions,
    candidate_decisions, summarize_candidates,
)
from vesuvius.neural_tracing.fiber_follow.regression.recovery import monitor_fixture, evaluate_monitor
from vesuvius.neural_tracing.fiber_follow.regression.history_slabs import SAMPLING_REVISION
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import CANDIDATE_COUNT
from vesuvius.neural_tracing.fiber_follow.shared.training_log import format_training_log, DirectTrainingInterval


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
                          ema=ema.state_dict(), sample_cfg=asdict(sample)))
    temporary.replace(path)


def conv_memory_format(device):
    """Channels-last is faster for the ordinary stem/decoder convolutions.

    DepthwiseConv3d selects contiguous inputs and weights locally: its CUDA
    kernels have the opposite layout preference on the measured hardware.
    """
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
    model.score_candidates = torch.compile(model.score_candidates, **options)
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
    names = ('fine', 'seed', 'seed_mask', 'seed_tangent', 'seed_age',
             'history_tokens', 'history_padding')
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


def training_prediction(model, x, hist, hmask, candidates=None, confidence_threshold=.5, n_commit=None):
    actual = len(hist)
    # A single decision needs no duplicate encoder, proposal or scoring work.
    # Keep only two batch specializations: one row or the configured batch.
    size = 1 if actual == 1 else model.training_batch_size
    # Compact live slabs before the fixed-shape compiled decision graph.
    # Convolutions see valid slots only; gradients remain attached across heads.
    timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if hist.is_cuda else time.perf_counter()
    if hist.is_cuda:
        timing[0].record()
    tokens, padding = model.encode_history(x)
    if hist.is_cuda:
        timing[1].record()
    if hasattr(model, '_history_timings'):
        model._history_timings.append(timing if hist.is_cuda else time.perf_counter()-timing)
    image, history, mask = training_inputs(dict(x, history_tokens=tokens, history_padding=padding), hist, hmask, size)
    threshold = hist.new_full((), confidence_threshold)
    output, context = model.training_forward(image, history, mask, threshold)
    if candidates is not None:
        count = candidates.shape[1]
        if not 0 < count <= CANDIDATE_COUNT:
            raise ValueError(f'Expected 1..{CANDIDATE_COUNT} candidate paths')
        curves = candidates.detach()
        if count < CANDIDATE_COUNT:
            curves = torch.cat((curves, curves[:, :1].expand(-1, CANDIDATE_COUNT-count, -1, -1)), 1)
        scored = model.score_candidates(context, fixed_rows(curves, size))
        output.update({name: value[:, :count] for name, value in scored.items()})
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


ARCHITECTURES = (ARCHITECTURE, PATCH_ARCHITECTURE, TOKEN_ARCHITECTURE)
HISTORY_GRAD_CLIP = 5.
REST_GRAD_CLIP = 100.


def checkpoint_config(ck):
    if ck['architecture'] not in ARCHITECTURES:
        raise ValueError('Historical slabs require fresh v10 training and replay v6')
    cfg = DirectConfig(**ck['model_cfg'])
    if ck['architecture'] != cfg.architecture:
        raise ValueError('Checkpoint architecture does not match its encoder configuration')
    return cfg


def resolve_encoder(requested, checkpoint=None):
    if checkpoint is None:
        return requested or DirectConfig.encoder
    saved = checkpoint_config(checkpoint).encoder
    if requested is not None and requested != saved:
        raise ValueError('Encoder must match the resumed checkpoint; start a new run to change it')
    return saved


def resolve_token_only(requested, checkpoint=None):
    if checkpoint is None:
        return bool(requested)
    saved = checkpoint_config(checkpoint).token_only
    if requested is not None and requested != saved:
        raise ValueError('Token-only mode must match the resumed checkpoint; start a new run to change it')
    return saved


def load_checkpoint(path,device='cuda'):
    ck = read_checkpoint(path,ARCHITECTURES,device)
    cfg = checkpoint_config(ck)
    model = build_model(cfg).to(device,memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    return model,cfg.fine,cfg.n_history,FiberVolumeSpec(**ck['vol_spec']),ck


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items()}


class DecisionBatchPrefetch:
    """Assemble one CPU update ahead while the current update uses the GPU.

    Only this thread consumes the loader iterator. Chunks remain ordered and
    the same first whole chunk reaching the decision budget ends each update.
    The complete denominator is therefore known before any task backward.
    """
    def __init__(self, iterator, decisions):
        if decisions < 1:
            raise ValueError('Positive decision budget required')
        self.iterator, self.decisions = iterator, decisions
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='decision-batch')
        self.pending = self.executor.submit(self.collect)
        self.closed = False

    def collect(self):
        chunks, count = [], 0
        while count < self.decisions:
            chunk = next(self.iterator)
            chunks.append(chunk)
            count += len(chunk['hist'])
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
            prediction = model(batch['x'], batch['hist'], batch['hmask'],
                               n_commit=tracer.p.n_commit)
        points = prediction['points']
        target = torch.cat((batch['plane_ab'], points[..., 2:]), -1)
        for scale, crop in (('fine', model.cfg.fine),):
            filename = f'batch_{step:06d}.png'
            plot_batch(batch['x'][scale], points, target, batch['plane_mask'], crop, images/filename,
                       batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                       source=batch['source'], offtrack=batch['offtrack'], confidence=prediction['confidence'],
                       history_channel=None)
        curves = prediction['refinement_points']
        labels = ['initial proposal']+[f'feedback proposal {i}' for i in range(1, curves.shape[1])]
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


def accumulate(table, key, value):
    """Add a device scalar without synchronizing; ``resolve_device_sums`` reads them once."""
    table[key] = table.get(key, 0.)+value.detach().double()


def resolve_device_sums(*tables):
    """Replace accumulated device scalars by Python floats with one transfer."""
    entries = [(table, key) for table in tables for key, value in table.items() if torch.is_tensor(value)]
    if entries:
        for (table, key), value in zip(entries, torch.stack([t[k] for t, k in entries]).tolist()):
            table[key] = value


IDENTITY_SUMS = ('identity_correct_count',
                 'identity_flipped_count', 'candidate_states', 'candidate_intervals',
                 'candidate_late_failures', 'candidate_first_failures', 'candidate_supervision_weight')


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
                     confidence_weight=.5, ema_decay=.999, n_commit=None, compute_metrics=True,
                     candidate_weight=1.,
                     history_grad_clip=HISTORY_GRAD_CLIP, rest_grad_clip=REST_GRAD_CLIP,
                     diagnostic=None):
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
    sources = np.zeros(6, dtype=np.int64)
    bank_tails = []
    identity = {}
    candidate_groups = {}
    requested_decisions = 0.
    decisions = []
    model.train()
    for cpu in batches:
        batch = move_batch(cpu, device)
        valid = cpu['x']['history_valid']
        for name, value in dict(history_valid_slabs=valid.sum(),
                history_age_sum=cpu['x']['history_ages'][valid].sum(),
                history_overlap_sum=cpu['x']['history_overlap'][valid].sum(),
                history_load_seconds=cpu['x']['history_load_seconds'].sum()).items():
            sums[name] = sums.get(name, 0.)+float(value)
        if diagnostic is not None:
            diagnostic.update(cpu_batch=cpu)
        scoring = {}
        if 'candidate_points' in batch and cpu['candidate_mask'].any():
            scoring['candidates'] = batch['candidate_points']
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            output = training_prediction(model, batch['x'], batch['hist'], batch['hmask'],
                           n_commit=commit_window(model.cfg, n_commit), **scoring)
            terms = model.training_loss(output, batch, model.cfg, tolerance, n_commit=n_commit)
            geometry = terms['geometry_per_state'].sum()/denominator
            confidence = terms['confidence_per_state'].sum()/denominator
            loss = geometry + confidence_weight*confidence
            if 'candidate_per_state' in terms:
                candidate_loss = terms['candidate_per_state'].sum()/denominator
                loss = loss+candidate_weight*candidate_loss
                accumulate(identity, 'candidate_loss', candidate_loss)
        loss.backward()
        if compute_metrics:
            decisions.extend(decision_rows(output, batch, model.cfg, n_commit, tolerance))
            if 'candidate_confidence_logits' in output:
                for name, row in candidate_decisions(output, batch, model.cfg, n_commit).items():
                    group = candidate_groups.setdefault(name, {})
                    for key, value in row.items():
                        group[key] = group.get(key, 0)+value
        for key in IDENTITY_SUMS:
            if key in terms:
                accumulate(identity, key, terms[key])
        for key in ('presence_dropped', 'blurred', 'foreign_components', 'seed_present', 'identity_observable'):
            if key in cpu:
                identity[key] = identity.get(key, 0.)+float((cpu[key] > 0).sum())
        weights = torch.ones(len(cpu['hist']))
        endpoint = torch.ones(len(weights), dtype=torch.bool)
        matched = endpoint & (cpu.get('decision_kind', torch.zeros(len(weights))) > 0)
        choice = endpoint & (cpu.get('decision_kind', torch.zeros(len(weights))) == 1)
        for name, select in (('supervision', torch.ones_like(endpoint)), ('endpoint', endpoint),
                             ('matched_endpoint', matched), ('choice_endpoint', choice)):
            sums[name+'_weight'] = sums.get(name+'_weight', 0.)+float(weights[select].sum())
        sums['endpoint_states'] = sums.get('endpoint_states', 0)+int(endpoint.sum())
        sums['matched_endpoint_states'] = sums.get('matched_endpoint_states', 0)+int(matched.sum())
        sums['choice_endpoint_states'] = sums.get('choice_endpoint_states', 0)+int(choice.sum())
        if 'failure_kind' in cpu:
            for kind, name in enumerate(REPLAY_FAILURES[1:], 1):
                key = 'replay_'+name+'_endpoints'
                sums[key] = sums.get(key, 0)+int(((cpu['failure_kind'] == kind) & endpoint).sum())
        if 'decision_requested' in cpu:
            requested_decisions += float(cpu['decision_requested'].sum())
        if 'negative_bank_shards' in cpu:
            low,high = int(cpu['negative_bank_shards'].min()),int(cpu['negative_bank_shards'].max())
            identity['negative_bank_shards_min'] = min(identity.get('negative_bank_shards_min',low),low)
            identity['negative_bank_shards_max'] = max(identity.get('negative_bank_shards_max',high),high)
        if 'location_source' in cpu:
            for index, name in enumerate(LOCATION_SOURCES):
                key = f'location_{name}'
                identity[key] = identity.get(key, 0.)+float((cpu['location_source'] == index).sum())
        for key, value in (('loss', loss), ('geometry', geometry), ('confidence_loss', confidence)):
            accumulate(sums, key, value)
        for key in ('error_sum', 'geometry_count', 'correct_count', 'confidence_count',
                    'point_correct_count', 'point_wrong_count', 'point_unknown_count',
                    'confidence_labeled_states', 'confidence_departed_states', 'refinement_attempts_sum'):
            accumulate(sums, key, terms[key])
        if 'source' in cpu:
            for source in range(len(sources)):
                sources[source] += int((cpu['source'] == source).sum())
            if 'bank_tail_length' in cpu:
                bank_tails.extend(cpu['bank_tail_length'][cpu['source'] == 3].tolist())
    resolve_device_sums(sums, identity)
    sums['history_encode_seconds'] = sum(t[0].elapsed_time(t[1])/1000 if isinstance(t, tuple) else t
                                         for t in model._history_timings)
    # The summed loss is finite only if every microbatch loss was; checked before any update.
    if not math.isfinite(sums['loss']):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    if total:
        finish_training_update(model)
        sums.update(clip_training_gradients(model, history_grad_clip, rest_grad_clip))
        opt.step()
        update_ema(ema, model, step, ema_decay)
    sums.update(observed_states=observed, supervised_states=total,
                observation_only_states=observed-total, optimizer_applied=bool(total))
    slab_count = max(1., sums['history_valid_slabs'])
    sums.update(history_age_mean=sums['history_age_sum']/slab_count,
                history_overlap_mean=sums['history_overlap_sum']/slab_count,
                history_valid_slabs_mean=sums['history_valid_slabs']/denominator)
    sums['refinement_attempts_mean'] = sums.get('refinement_attempts_sum', 0.)/denominator
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']),
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']),
                fresh_fraction=float(sources[0]/denominator),
                recent_fraction=float(sources[2]/denominator),bank_wrong_continuation_fraction=float(sources[3]/denominator),
                bank_following_fraction=float(sources[4]/denominator),
                decision_pair_fraction=float(sources[5]/denominator), decision_requested_fraction=requested_decisions/denominator)
    sums.update(bank_wrong_continuation_tail_mean=float(np.mean(bank_tails)) if bank_tails else None,
                bank_wrong_continuation_tail_min=min(bank_tails) if bank_tails else None,
                bank_wrong_continuation_tail_max=max(bank_tails) if bank_tails else None)
    if identity:
        # Versioned separately from the distance metrics above.
        identity.update(identity_version=1,
                        identity_prefix_correct_fraction=identity.get('identity_correct_count', 0.)/max(1., sums['confidence_count']))
        if 'candidate_loss' in identity:
            identity['candidate_loss_eligible'] = identity['candidate_loss']*total/max(1e-12, identity.get('candidate_supervision_weight', 0.))
        for key in ('presence_dropped', 'blurred', 'foreign_components', 'seed_present', 'identity_observable', *(f'location_{n}' for n in LOCATION_SOURCES)):
            if key in identity:
                identity[key+'_fraction'] = identity.pop(key)/denominator
        sums['identity'] = identity
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, commit_window(model.cfg, n_commit))
        if candidate_groups:
            sums['identity']['candidate_decisions'] = summarize_candidates(candidate_groups)
    return sums


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--onpolicy', nargs='*', default=[])
    ap.add_argument('--out-root', default=str(Path(__file__).parents[1]/'output'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=8, help='Target supervised decisions per optimizer update')
    ap.add_argument('--microbatch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--worker-cache-gb', type=float, default=.5)
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
    ap.add_argument('--channels', type=int, default=DirectConfig.channels, help='Base image encoder width')
    ap.add_argument('--encoder', choices=('conv', 'patch4'), default=None,
                    help='Image encoder: conv (default) or convolution-free 4x4x4 patches; inferred on resume')
    ap.add_argument('--token-only', action=argparse.BooleanOptionalAction, default=None,
                    help='Use only patch4 tokens throughout; no reconstructed fine features or output planes')
    ap.add_argument('--direction-inputs', action=argparse.BooleanOptionalAction, default=True,
                    help='Add six sign-invariant direction channels to main crops; no image augmentations on these channels')
    ap.add_argument('--decoder-layers', type=int, default=4)
    ap.add_argument('--axial-layers', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--memory-switch-probability', type=float, default=.3,
                    help='Fresh draws replaced by original-then-neighbor memory sequences')
    ap.add_argument('--memory-switch-tail', type=float, nargs=2, default=(16.,96.), metavar=('MIN', 'MAX'),
                    help='Neighbor tail length of memory-switch sequences in trace voxels')
    ap.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--no-history-prob', type=float, default=.15, help='Fresh-state probability of absent observed history')
    ap.add_argument('--short-history-prob', type=float, default=.4,
                    help='Given history is present, probability of a balanced 1-8/9-32 point startup history')
    ap.add_argument('--decision-fraction', type=float, default=.3,
                    help='Fraction of endpoint proposals reserved for matched pairs with remote observed seeds')
    ap.add_argument('--decision-choice-fraction', type=float, default=.75,
                    help='Requested fraction of matched pairs teaching recoverable geometry choices')
    ap.add_argument('--candidate-weight', type=float, default=1., help='Weight of candidate first-failure survival likelihood')
    ap.add_argument('--fresh-fraction', type=float, default=.7,
                    help='Annotation-fresh share after decision and bank-following budgets; remainder uses replay')
    ap.add_argument('--negative-bank', default=str(Path(__file__).parents[1]/'output'/'neighbor_samples_r0_32_l80_160_v2'), help='Shared live bank for foreign-fiber masks, wrong continuations and following supervision')
    ap.add_argument('--near-negative-bank', help='Additional bank of validated nearby negative relationships')
    ap.add_argument('--following-bank', help='Following path source (default: negative-bank)')
    ap.add_argument('--continuation-bank', help='Wrong-continuation path source (default: negative-bank)')
    ap.add_argument('--bank-coverage-probability', type=float, default=.2, help='Fresh slots reserved for covered parents with usable history (new runs: .2)')
    ap.add_argument('--prefer-long-continuations', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--negative-bank-refresh-seconds', type=float, default=30., help='Each loader worker checks for completed new negative shards at this interval')
    ap.add_argument('--negative-bank-cache-mb', type=float, default=64., help='Maximum cached negative geometry per loader worker')
    ap.add_argument('--bank-wrong-continuation-probability', type=float, default=0.,
                    help='Fraction of recent DAgger departure draws to replace with safe bank continuations when available')
    ap.add_argument('--bank-wrong-continuation-tail', type=float, nargs=2, default=(4.,16.), metavar=('MIN', 'MAX'),
                    help='Wrong-fiber tail range in trace voxels (default: 4 16)')
    ap.add_argument('--bank-following-probability', type=float, default=.2,
                    help='Independent fraction of endpoint proposals following validated bank paths (default: .2)')
    ap.add_argument('--bank-hard-fraction', type=float, default=.5,
                    help='Fraction of bank draws ranked by nearby similar, curved, or converging geometry')
    ap.add_argument('--replay-failure-fraction', type=float, default=.5,
                    help='Replay share balanced across available failure kinds; remainder uses drift bands')
    ap.add_argument('--bank-switch-tolerance', type=float, default=.75,
                    help='Foreign centerline contact radius for DAgger labels, in trace voxels')
    ap.add_argument('--bank-own-tolerance', type=float, default=1.5,
                    help='Annotation tube excluded from confirmed foreign contact')
    ap.add_argument('--presence-dropout', type=float, default=0.,
                    help='Probability of zeroing the presence crop; may be changed on resume')
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
    ap.add_argument('--dagger-after', type=float, default=96.,
                    help='Replay states recorded after a confirmed departure, in trace voxels')
    ap.add_argument('--replay-keep', type=int, default=4)
    ap.add_argument('--resume', help='Resume last.pt inside this run with the same training options')
    ap.add_argument('--reset-optimizer', action='store_true',
                    help='With --resume: fresh AdamW, one LR for all parameters, no transfer freeze, and restart LR warmup/decay')
    ap.add_argument('--recurrent-refinement-steps', type=int, default=DirectConfig.recurrent_refinement_steps,
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    launch_started = time.monotonic()

    def progress(message):
        print(f'[{args.name} +{time.monotonic()-launch_started:.1f}s] {message}', flush=True)

    progress(f'Starting training: device={args.device}, workers={args.workers}')
    if args.reset_optimizer and not args.resume:
        raise ValueError('--reset-optimizer requires --resume')
    if any(not math.isfinite(v) or v < 0 for v in (args.history_grad_clip, args.rest_grad_clip)):
        raise ValueError('Gradient clipping limits must be finite and nonnegative (0 disables clipping)')
    if not math.isfinite(args.fresh_fraction) or not 0 <= args.fresh_fraction <= 1:
        raise ValueError('Fresh fraction must be finite and in [0, 1]')
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
    resume = read_checkpoint(args.resume,ARCHITECTURES,args.device) if args.resume else None
    cfg = DirectConfig(encoder=resolve_encoder(args.encoder, resume),
                       token_only=resolve_token_only(args.token_only, resume),
                       direction_inputs=args.direction_inputs,channels=args.channels,hidden=args.hidden,layers=args.axial_layers,
                       decoder_layers=args.decoder_layers,
                       activation_checkpointing=args.activation_checkpointing,
                       recurrent_refinement_steps=args.recurrent_refinement_steps)
    if resume:
        cfg = checkpoint_config(resume)
        if args.direction_inputs != cfg.direction_inputs:
            raise ValueError('Direction inputs must match the resumed checkpoint; start a new run to change them')
    args.encoder = cfg.encoder
    args.token_only = cfg.token_only
    if args.decision_fraction and args.microbatch % 2:
        raise ValueError('Matched decisions require an even microbatch')
    if not np.isfinite(args.candidate_weight) or args.candidate_weight <= 0:
        raise ValueError('Candidate weight must be finite and positive')
    if args.dagger_after <= 0:
        raise ValueError('--dagger-after must be positive')
    from vesuvius.neural_tracing.fiber_follow.regression.bank_geometry import BankSwitchDetector
    BankSwitchDetector([], args.bank_switch_tolerance, args.bank_own_tolerance)
    identity_sampling = IdentitySampling(
        presence_dropout=args.presence_dropout,
        blur_probability=args.blur_probability,blur_sigma=args.blur_sigma,
        lateral_fraction=args.lateral_fraction,
        bank_wrong_continuation_probability=args.bank_wrong_continuation_probability,
        bank_wrong_continuation_tail=args.bank_wrong_continuation_tail,
        bank_following_probability=args.bank_following_probability,
        bank_hard_fraction=args.bank_hard_fraction,replay_failure_fraction=args.replay_failure_fraction,
        decision_fraction=args.decision_fraction,decision_choice_fraction=args.decision_choice_fraction,
        candidate_tolerance=args.tolerance,bank_coverage_probability=args.bank_coverage_probability,
        prefer_long_continuations=args.prefer_long_continuations,
        memory_switch_probability=args.memory_switch_probability,memory_switch_tail=args.memory_switch_tail)
    if not args.negative_bank:
        raise ValueError('Training requires --negative-bank')
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Finest-level CT in a single enlarged crop.
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4.,
                           inputs='ct+presence')
    if cfg.direction_inputs:
        # Validate sibling paths and grids before starting loaders or collectors.
        FiberVolume(spec, cache_bytes=1 << 20).direction_fields()
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future, full_observed_history=True,
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
    negative_bank = None
    role_banks = {}
    if args.negative_bank:
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
        negative_bank = NeighborBank(args.negative_bank,train_f,band,grid_scale=spec.grid_scale,
            refresh_seconds=args.negative_bank_refresh_seconds,cache_bytes=int(args.negative_bank_cache_mb*(1 << 20)))
        negative_bank.validate_volume(spec)
        if resume and (resume.get('negative_bank_provenance') is not None or resume['training_options'].get('negative_bank')):
            negative_bank.validate_resume(resume.get('negative_bank_provenance'))
        progress(f'Live negative bank: {negative_bank.shard_count} published shards, refresh every {args.negative_bank_refresh_seconds:g}s per worker')
        by_path = {negative_bank.root:negative_bank}
        for role in ('near_negative_bank','following_bank','continuation_bank'):
            path = getattr(args,role)
            if path is None:
                continue
            root = Path(path).resolve()
            if root.name == 'bank.json':
                root = root.parent
            if root not in by_path:
                by_path[root] = NeighborBank(root,train_f,band,grid_scale=spec.grid_scale,
                    refresh_seconds=args.negative_bank_refresh_seconds,cache_bytes=int(args.negative_bank_cache_mb*(1 << 20)))
                by_path[root].validate_volume(spec)
            role_banks[role] = by_path[root]
            if resume:
                role_banks[role].validate_resume((resume.get('bank_role_provenance') or {}).get(role))
    def role_provenance():
        return {role:bank.provenance() for role,bank in role_banks.items()}
    if resume:
        ignored = {'resume','reset_optimizer','out_root','device','batch','microbatch','workers','threads','worker_cache_gb',
                   'log_every','ckpt_every','diag_every','dagger_device',
                   'negative_bank_refresh_seconds','negative_bank_cache_mb','activation_checkpointing',
                   'history_grad_clip','rest_grad_clip','presence_dropout','blur_probability','blur_sigma',
                   'decision_fraction','decision_choice_fraction','bank_following_probability','fresh_fraction',
                   'bank_hard_fraction','replay_failure_fraction','bank_switch_tolerance','bank_own_tolerance',
                   'n_commit'}
        for key,value in vars(args).items():
            recorded = resume['training_options'][key]
            if key not in ignored and json.dumps(recorded,sort_keys=True) != json.dumps(value,sort_keys=True):
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
    progress(f'Independent gradient clipping: history={args.history_grad_clip:g}, rest={args.rest_grad_clip:g} (0 disables clipping)')
    model = build_model(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt, done, lr_restart_step = initialize_training_optimizer(model, ema, args, resume)
    if args.reset_optimizer:
        progress(f'Fresh AdamW: one parameter group, all parameters trainable; LR restarts at update {done+1} '
                 f'with {args.warmup} warmup updates to {args.lr:g}')
    prepare_training(model, args.microbatch)
    progress('Compiling training operations; first forward/backward passes may take several minutes')
    if done >= args.steps:
        raise ValueError('Run has already reached its requested update count')
    replay_index = out/'dagger'/'replay.json'
    replay_paths = json.loads(replay_index.read_text()) if resume and replay_index.exists() else args.onpolicy
    progress('Loading replay banks and preparing data loader')
    caches = [OnPolicyStates.load(p) for p in replay_paths]
    collector = OnlineCollector(out/'dagger', args.fibers, args.val_z, args.dagger_device or args.device,
        every=args.dagger_every, max_seeds=args.dagger_seeds, seed=args.seed, replay_keep=args.replay_keep,
        initial=[c._dir for c in caches], trace_len=args.dagger_trace_len, n_commit=args.n_commit,
        collector_module='vesuvius.neural_tracing.fiber_follow.regression.collect',
        extra_args=('--after', args.dagger_after, '--bank-switch-tolerance', args.bank_switch_tolerance,
                    '--bank-own-tolerance', args.bank_own_tolerance,
                    *[v for path in (args.negative_bank, args.near_negative_bank) if path for v in ('--failure-bank', path)]))
    progress(f'Live historical slabs: eight slots; {args.microbatch} independent decisions per microbatch')
    builder = IdentityObservationBuilder(cfg,train_f,identity_sampling,
        augment=True,negative_bank=negative_bank,**role_banks)
    dataset = FollowDataset(train_f, spec, sample, band, chunk=args.microbatch, seed=args.seed+done,
        cache_bytes=int(args.worker_cache_gb*(1 << 30)), onpolicy=caches,
        replay_index=str(collector.index), batch_builder=builder, additional_crops=(), fresh_fraction=args.fresh_fraction)
    loader_args = dict(batch_size=None, num_workers=args.workers,
                       pin_memory=torch.device(args.device).type == 'cuda')
    if args.workers:
        loader_args.update(prefetch_factor=2, persistent_workers=True)
    loader = torch.utils.data.DataLoader(dataset, **loader_args)
    if not resume:
        (out/'config.json').write_text(json.dumps(dict(vars(args), architecture=model.architecture,
            identity_sampling=asdict(identity_sampling),
            negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
            bank_role_provenance=role_provenance(),
            model_cfg=cfg.to_dict(), sample_cfg=asdict(sample), vol_spec=spec.to_dict(),
            data_policy=DATA_POLICY,
            monitor_recovery_sha256=recovery_hash,
            seed_manifest_sha256=manifest['sha256'], fiber_manifest=fiber_manifest(fibers),
            parameter_count=sum(p.numel() for p in model.parameters())), indent=2))
    log = RunLog(out/'log.jsonl', formatter=format_training_log)
    if resume:
        log.record(dict(step=done, event='resume_configuration', checkpoint=str(args.resume),
                        training_options=vars(args), model_cfg=cfg.to_dict()))
    log.record(dict(step=done, event='optimizer_configuration', reset=args.reset_optimizer,
                    lr_restart_step=lr_restart_step, optimizer_state_entries=len(opt.state),
                    groups=[dict(parameters=len(g['params'])) for g in opt.param_groups],
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    log.record(dict(step=done,event='identity_sampling',architecture=model.architecture,
        source_sampling=dict(decision=args.decision_fraction,bank_following=args.bank_following_probability,
            fresh=(1-args.decision_fraction-args.bank_following_probability)*args.fresh_fraction,
            recent=(1-args.decision_fraction-args.bank_following_probability)*(1-args.fresh_fraction)),
        history_sampling_revision=SAMPLING_REVISION,
        history_policy='live_observed_slabs',sampling=asdict(identity_sampling),
        negative_bank_path=str(negative_bank.root),negative_bank_provenance=negative_bank.provenance(),
        bank_role_provenance=role_provenance()))
    tracer = None
    recovery_vol = FiberVolume(spec) if recovery_states is not None else None
    started = time.monotonic()
    interval_started, interval_step = started, done
    interval_data_seconds = interval_update_seconds = 0.
    interval_metrics = DirectTrainingInterval()
    updates = None
    try:
        if args.diag_every or args.long_diag_every:
            from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
            tracer = DirectTracer(ema, FiberVolume(spec), cfg.fine, cfg.n_history,
                TraceParams(n_commit=args.n_commit, max_len=args.diag_max_len), device=args.device)
        progress(f'Starting data loader; collecting {args.batch} supervised decisions for update {done+1}')
        iterator = iter(loader)
        updates = DecisionBatchPrefetch(iterator, args.batch)
        observed_states = interval_states = 0
        prior_samples = int(resume['samples_seen']) if resume else 0
        for step in range(done+1, args.steps+1):
            event = collector.poll()
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
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps,
                candidate_weight=args.candidate_weight,
                history_grad_clip=args.history_grad_clip, rest_grad_clip=args.rest_grad_clip,
                diagnostic=diagnostic)
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
                    train_seconds=now-started,
                    samples_per_second=observed_states/(now-started),
                    interval_samples_per_second=interval_states/(now-interval_started),
                    interval_data_seconds=interval_data_seconds, interval_update_seconds=interval_update_seconds,
                    interval=interval_metrics.summary(), n_future=cfg.n_future, tolerance=args.tolerance,
                    interval_updates=step-interval_step,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30
                        if torch.device(args.device).type == 'cuda' else None))
                from vesuvius.neural_tracing.fiber_follow.shared.diag import plot_curves
                plot_curves(out/'log.jsonl', out/'curves.png', loss_key='geometry')
                interval_metrics = DirectTrainingInterval()
                interval_started, interval_step = now, step
                interval_data_seconds = interval_update_seconds = 0.
                interval_states = 0

            def save(path, resumable=False):
                extra = dict(step=step, lr_restart_step=lr_restart_step, tolerance=args.tolerance, n_commit=args.n_commit,
                    history_sampling_revision=SAMPLING_REVISION,
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
            if step < args.steps and collector.launch(step, save):
                log.record(dict(step=step, dagger_launched=True))
            periodic = {}
            if tracer is not None and args.diag_every and step % args.diag_every == 0:
                began = time.monotonic()
                training_diagnostics(ema, diagnostic['cpu_batch'], tracer, val_f, manifest['monitor'], out, step, log,
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
    finally:
        if updates is not None:
            updates.close()
        event = collector.close()
        if event:
            log.record(event)
        if tracer is not None:
            tracer.close()
        log.close()
    return str(out/'last.pt')


if __name__ == '__main__':
    main()
