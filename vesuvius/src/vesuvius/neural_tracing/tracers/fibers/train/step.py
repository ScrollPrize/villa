"""One training update: the compiled training forward, proposal selection, loss accumulation over microbatches,
gradient clipping and the guarded AdamW step."""
import math

import torch

from vesuvius.neural_tracing.tracers.shared.policy import gate_horizon, selection_window
from vesuvius.neural_tracing.tracers.fibers.train.runloop import update_ema, resume_training
from vesuvius.neural_tracing.tracers.shared.tensors import fixed_rows, move_batch
from vesuvius.neural_tracing.tracers.fibers.data.observations import LOCATION_SOURCES
from vesuvius.neural_tracing.tracers.fibers.train.supervision import commit_window, loss_terms
from vesuvius.neural_tracing.tracers.fibers.evaluation.monitor.diagnostics import decision_rows, summarize_decisions


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
    if reset:
        model.requires_grad_(True)
    scales = [p for name, p in model.named_parameters() if name.endswith('.logit_scale')]
    opt = torch.optim.AdamW([p for name, p in model.named_parameters() if not name.endswith('.logit_scale')],
                            lr=args.lr, weight_decay=1e-4)
    # Attention logit scales (log multipliers) train in their own group: no decay toward zero, a faster LR.
    scale_group = dict(params=scales, weight_decay=0., lr_scale=LOGIT_SCALE_LR_SCALE)
    if scales and not (resume and not reset and len(resume['optimizer']['param_groups']) == 1):
        opt.add_param_group(scale_group)
    done = resume_training(resume, model, ema, opt, reset_optimizer=reset) if resume else 0
    if scales and len(opt.param_groups) == 1:  # resumed from a checkpoint trained without them: they start fresh
        opt.add_param_group(scale_group)
    origin = done if reset else int((resume or {}).get('lr_restart_step', 0))
    match_optimizer_layout(opt)
    return opt, done, origin


GRAD_CLIP = 60.
LOGIT_SCALE_LR_SCALE = 10.  # LR multiple of the attention logit scales' group


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
    denominator = total  # >= 1, checked above
    for group in opt.param_groups:
        group['lr'] = lr*group.get('lr_scale', 1.)
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
        if 'ct_frame_source' in cpu['x']:
            source = cpu['x']['ct_frame_source']
            known = source >= 0
            for suffix, value in dict(count=known.sum(), transported=(source == 1).sum(),
                    deterministic=(source == 2).sum(), learned=(source == 3).sum(), prediction=(source == 4).sum(),
                    energy_sum=cpu['x']['ct_frame_energy'][known].sum(),
                    gap_sum=cpu['x']['ct_frame_gap'][known].sum()).items():
                name = 'ct_frame_'+suffix
                sums[name] = sums.get(name, 0.)+float(value)
        batch = move_batch(cpu, device)
        if diagnostic is not None:
            diagnostic.update(cpu_batch=cpu)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
            window = selection_window(commit_window(model.cfg, n_commit), gate_horizon(model.cfg), gate)
            if model.model_type == 'sequence':  # an episode batch: outputs and rows of its supervised decisions
                from vesuvius.neural_tracing.tracers.fibers.train.sequence import episode_forward, select_rows
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
    finish_training_update(model)
    sums.update(clip_training_gradients(model, grad_clip))
    opt.step()
    update_ema(ema, model, step, ema_decay, ramp=ema_ramp)
    sums.update(observed_states=observed, supervised_states=total,
                observation_only_states=observed-total, optimizer_applied=True,
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
