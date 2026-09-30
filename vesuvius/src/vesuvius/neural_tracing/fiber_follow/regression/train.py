"""Train a direct curve follower from scratch, with original-fiber online replay."""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
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
    ARCHITECTURE, MEMORY_ARCHITECTURE, MEMORY_ARCHITECTURE_V1, SPATIAL_MEMORY_ARCHITECTURE,
    TRAJECTORY_MEMORY_ARCHITECTURE, DirectConfig, DirectFollower, build_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, DirectTracer, LOCATION_SOURCES,
    load_contacts, load_hard_spans,
)
from vesuvius.neural_tracing.fiber_follow.shared.components import PAIR_SAMPLING_VERSION, ComponentRule
from vesuvius.neural_tracing.fiber_follow.regression.supervision import commit_window, loss_terms, memory_probe_terms
from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import (
    decision_rows, summarize_decisions, identity_ranking, summarize_ranking,
    identity_training_groups, summarize_identity_groups,
    candidate_decisions, summarize_candidates,
)
from vesuvius.neural_tracing.fiber_follow.regression.recovery import monitor_fixture, evaluate_monitor
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import FEATURE_SAMPLING_REVISION
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


def compile_training_model(model):
    """Compile the training forward; the module keeps its own parameters.

    Inductor (torch 2.12) otherwise returns corrupted encoder gradients for this
    graph whenever candidate curves are scored: 30-10,000x the eager magnitude and
    uncorrelated between identical repeats, with forward values unchanged. Emulating
    eager bf16 rounding restores eager-matching gradients (aot_eager was already
    correct, which isolates the fault to inductor code generation).

    Each input variant (batch size, candidates, replay) is a static graph.
    Allow enough variants to avoid dropping later ones to eager execution.
    """
    import torch._dynamo.config
    import torch._inductor.config
    torch._inductor.config.emulate_precision_casts = True
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    return torch.compile(model)


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


ARCHITECTURES = (ARCHITECTURE, MEMORY_ARCHITECTURE, MEMORY_ARCHITECTURE_V1,
                 SPATIAL_MEMORY_ARCHITECTURE, TRAJECTORY_MEMORY_ARCHITECTURE)
ROUTE_OPTIONS = ('route_grid_step','route_transition_radius','route_transition_cost','route_loss_weight','route_sequence_weight')
MEMORY_OPTIONS = ('memory_slots','memory_steps','memory_stride','memory_patch_size','memory_grad_steps')
FEATURE_OPTIONS = ('feature_memory_revision', 'feature_detail_tokens', 'feature_stream_steps', 'feature_replay_weight',
                   'feature_switch_crop_fraction')
MEMORY_GRAD_CLIP = 5.
REST_GRAD_CLIP = 20.


def checkpoint_config(ck):
    if ck['architecture'] not in ARCHITECTURES:
        raise ValueError('Unsupported checkpoint architecture')
    # Older shared configs serialized unused proposal options for every model.
    model_cfg = {k: v for k, v in ck['model_cfg'].items() if not k.startswith('proposal_')}
    if model_cfg.get('memory_version') == 4 and 'feature_memory_revision' not in model_cfg:
        raise ValueError('This checkpoint is the superseded patch-memory v4; start feature-memory v4 from v2/v3 weights')
    if ck['architecture'] == MEMORY_ARCHITECTURE_V1:
        model_cfg.setdefault('memory_version', 1)  # saved before versions were recorded
    cfg = DirectConfig(**model_cfg)
    expected = ({1: MEMORY_ARCHITECTURE_V1, 2: MEMORY_ARCHITECTURE, 3: SPATIAL_MEMORY_ARCHITECTURE,
                 4: TRAJECTORY_MEMORY_ARCHITECTURE}[cfg.memory_version]
                if cfg.memory_slots else ARCHITECTURE)
    if ck['architecture'] != expected:
        raise ValueError('Checkpoint architecture and memory configuration disagree')
    return cfg


def resolve_refinement_config(cfg, radius):
    """An omitted override preserves the checkpoint's refinement behavior."""
    if radius is None:
        return cfg
    if cfg.memory_version != 3 or not cfg.memory_slots:
        raise ValueError('Route refinement radius requires spatial memory v3')
    return replace(cfg, route_refinement_radius=radius)


def load_checkpoint(path,device='cuda'):
    ck = read_checkpoint(path,ARCHITECTURES,device)
    cfg = checkpoint_config(ck)
    model = build_model(cfg).to(device,memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    return model,cfg.fine,cfg.n_history,FiberVolumeSpec(**ck['vol_spec']),ck


def add_direction_inputs(model):
    """Expand CT/presence input weights with zeros for six direction channels.

    All existing weights, including recurrent memory and its seed encoder, are
    retained. This initializes a new run; optimizer/step state is not resumed.
    """
    if model.cfg.direction_inputs:
        return model
    cfg = replace(model.cfg, direction_inputs=True)
    device = next(model.parameters()).device
    expanded = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    state = model.state_dict()
    keys = ['encoder.stem.0.weight']
    if cfg.memory_slots and not cfg.feature_memory:
        keys.append('recurrent_memory.patch_encoder.0.weight')
    target = expanded.state_dict()
    for key in keys:
        old, new = state[key], target[key]
        if old.shape[1] != 2 or new.shape[1] != 8 or old.shape[:1]+old.shape[2:] != new.shape[:1]+new.shape[2:]:
            raise ValueError(f'Unexpected input weight shape: {key}')
        state[key] = torch.zeros_like(new)
        state[key][:, :2].copy_(old)
    expanded.load_state_dict(state, strict=True)
    expanded.train(model.training)
    return expanded


def initialize_spatial_model(model, cfg):
    """Explicit new-run migration; retain shared EMA weights, never optimizer state."""
    if cfg.memory_version != 3:
        raise ValueError('Spatial initialization requires memory version 3')
    upgraded = build_model(cfg).to(next(model.parameters()).device,
                                   memory_format=conv_memory_format(next(model.parameters()).device))
    incompatible = upgraded.load_state_dict(model.state_dict(),strict=False)
    allowed = ('route_',) if model.cfg.memory_slots else ('route_','recurrent_memory.')
    if incompatible.unexpected_keys or any(not key.startswith(allowed) for key in incompatible.missing_keys):
        raise ValueError(f'Unexpected spatial initialization mismatch: {incompatible}')
    upgraded.train(model.training)
    return upgraded


def initialize_trajectory_model(model, cfg):
    """Start a v4 run from compatible weights, excluding lattice/refinement heads.

This is an architectural migration, never an optimizer or RNG resume. Direct
decoder attention to identity memory changes behavior even with shared weights.
"""
    if cfg.memory_version != 4:
        raise ValueError('Trajectory initialization requires memory version 4')
    if model.cfg.memory_slots and model.cfg.memory_version == 1:
        raise ValueError('Legacy v1 memory checkpoints cannot initialize new runs')
    device = next(model.parameters()).device
    upgraded = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    # V3 never used its inherited continuous head; preserve the fresh v4
    # initialization instead. V2's trained continuous head remains transferable.
    fresh = ('recurrent_memory.',) + (('coordinates.',) if model.cfg.memory_version == 3 else ())
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith(('route_', 'correction_head.') + fresh)}
    incompatible = upgraded.load_state_dict(state, strict=False)
    allowed = fresh
    if incompatible.unexpected_keys or any(not k.startswith(allowed) for k in incompatible.missing_keys):
        raise ValueError(f'Unexpected trajectory initialization mismatch: {incompatible}')
    upgraded.train(model.training)
    return upgraded


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items()}


@torch.no_grad()
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device, memory=None):
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
                               memory=None if memory is None else take(memory))
        points = prediction['points']
        target = torch.cat((batch['plane_ab'], points[..., 2:]), -1)
        for scale, crop in (('fine', model.cfg.fine),):
            filename = f'batch_{step:06d}.png'
            plot_batch(batch['x'][scale], points, target, batch['plane_mask'], crop, images/filename,
                       batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                       source=batch['source'], offtrack=batch['offtrack'], confidence=prediction['confidence'],
                       history_channel=None)
        curves = prediction['refinement_points']
        labels = (['initial proposal']+[f'correction {i}' for i in range(1, curves.shape[1]-1)]+
                  ['corrected proposal']) if curves.shape[1] > 1 else ['proposal']
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


IDENTITY_SUMS = ('identity_count', 'identity_states', 'identity_rank_correct', 'identity_correct_count',
                 'identity_flipped_count', 'candidate_states')


def clip_training_gradients(model, memory_max_norm=MEMORY_GRAD_CLIP, rest_max_norm=REST_GRAD_CLIP):
    """Clip memory separately so its recurrent spikes cannot scale image gradients.

    Zero disables clipping for a group, but never disables finite-gradient checks.
    Identify parameters by object identity: compiled wrappers prefix their names.
    Check both groups before modifying either group's gradients.
    """
    limits = dict(memory=memory_max_norm, rest=rest_max_norm)
    if any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Gradient clipping limits must be finite and nonnegative (0 disables clipping)')
    parameters = list(model.parameters())
    memory = getattr(model, 'recurrent_memory', None)
    memory_ids = {id(p) for p in memory.parameters()} if memory is not None else set()
    groups = dict(memory=[p for p in parameters if id(p) in memory_ids],
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
                     identity_weight=.5, identity_temperature=.1, candidate_weight=1., memory_probe_weight=.5,
                     memory_grad_clip=MEMORY_GRAD_CLIP, rest_grad_clip=REST_GRAD_CLIP, stream_states=None,
                     diagnostic=None):
    """Equal weight per observed state, independent of microbatch boundaries.

    Within a state each loss averages over its known points; fully unknown
    states contribute zero. Geometry and confidence are evaluated in one pass.
    Loss sums stay on the device until every microbatch is queued, so the host
    synchronizes once per update rather than once per microbatch.
    """
    from .feature_sequences import sequence_steps, FeatureStreamStates
    stream_states = FeatureStreamStates() if stream_states is None else stream_states
    total = sum(len(b['hist']) for chunk in batches for b in sequence_steps(chunk))
    if total < 1:
        raise ValueError('An update needs at least one state')
    for group in opt.param_groups:
        group['lr'] = lr
    opt.zero_grad(set_to_none=True)
    sums = dict(loss=0., geometry=0., confidence_loss=0., error_sum=0., geometry_count=0.,
                correct_count=0., confidence_count=0.)
    sources = np.zeros(6, dtype=np.int64)
    bank_tails = []
    identity = {}
    identity_groups = {}
    candidate_groups = {}
    requested_decisions = 0.
    pair_version = PAIR_SAMPLING_VERSION
    decisions, rankings = [], []
    memory = {}
    model.train()
    for chunk in batches:
        sequence = sequence_steps(chunk)
        chunk_losses = []
        for cpu in sequence:
            if 'pair_sampling_version' in cpu:
                pair_version = int(cpu['pair_sampling_version'][0])
            batch = move_batch(cpu, device)
            if model.cfg.memory_slots:
                sums['memory_observations_mean'] = sums.get('memory_observations_mean', 0.)+float(cpu['x']['memory_mask'].sum())/total
                if model.cfg.memory_probe and 'memory_target_identity_mask' in cpu:
                    labeled = cpu['memory_target_identity_mask']
                    memory['labeled_writes'] = memory.get('labeled_writes', 0.)+float(labeled.sum())
                    memory['labeled_states'] = memory.get('labeled_states', 0.)+float(labeled.any(-1).sum())
                    memory['departed_states'] = memory.get('departed_states', 0.)+float(
                        (labeled & (cpu['memory_target_identity'] < .5)).any(-1).sum())
                sums['memory_anchor_fraction'] = sums.get('memory_anchor_fraction', 0.)+float(cpu['x']['memory_seed_valid'].sum())/total
            carried = stream_states.incoming(model, cpu, device) if 'stream_id' in cpu else None
            if diagnostic is not None:
                diagnostic.update(cpu_batch=cpu, memory=None if carried is None else
                                  {k: v.detach() for k, v in carried.items()})
            queries = dict(queries=batch['identity_points']) if 'identity_points' in batch else {}
            if 'candidate_points' in batch:
                queries['candidates'] = batch['candidate_points']
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                output = model(batch['x'], batch['hist'], batch['hmask'], memory=carried, **queries)
                if carried is not None:
                    stream_states.update(cpu, output, carried)
                    if model.cfg.feature_memory_revision == 2 and model.cfg.feature_replay_weight:
                        stream_states.replay.record(cpu, output)
                terms = loss_terms(output, batch, model.cfg, tolerance, n_commit=n_commit,
                                   identity_temperature=identity_temperature)
                geometry = terms['geometry_per_state'].sum()/total
                confidence = terms['confidence_per_state'].sum()/total
                loss = geometry + confidence_weight*confidence
                if 'route_per_state' in terms:
                    route_loss = terms['route_per_state'].sum()/total
                    loss = loss+model.cfg.route_loss_weight*route_loss
                    accumulate(memory, 'route_loss', route_loss)
                    for key in ('route_count','route_error_sum'):
                        accumulate(memory,key,terms[key])
                if 'identity_per_state' in terms:
                    identity_loss = terms['identity_per_state'].sum()/total
                    loss = loss + identity_weight*identity_loss
                    accumulate(identity, 'identity_loss', identity_loss)
                if 'candidate_per_state' in terms:
                    candidate_loss = terms['candidate_per_state'].sum()/total
                    loss = loss+candidate_weight*candidate_loss
                    accumulate(identity, 'candidate_loss', candidate_loss)
                if 'memory_probe' in output and 'memory_target_identity' in batch:
                    probe = memory_probe_terms(output, batch, departed_weight=model.cfg.memory_departed_weight)
                    identity_probe = probe['memory_identity_per_state'].sum()/total
                    offset_probe = probe['memory_offset_per_state'].sum()/total
                    loss = loss+memory_probe_weight*(identity_probe+offset_probe)
                    for key, value in (('probe_identity_loss', identity_probe), ('probe_offset_loss', offset_probe),
                                       *((k.removeprefix('memory_'), v) for k, v in probe.items() if not k.endswith('_per_state'))):
                        accumulate(memory, key, value)
            if 'feature_sequence' in chunk:
                chunk_losses.append(loss)
            else:
                loss.backward()
            if model.cfg.memory_version == 3 and model.cfg.sequence_key in batch:
                earlier = batch[model.cfg.sequence_key]
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.device(device).type == 'cuda'):
                    sequence_output = model(earlier['x'],earlier['hist'],earlier['hmask'])
                    sequence_terms = loss_terms(sequence_output,earlier,model.cfg,tolerance,n_commit=n_commit)
                    sequence_loss = model.cfg.sequence_weight*(
                        model.cfg.route_loss_weight*sequence_terms.get('route_per_state', 0.)+
                        sequence_terms['geometry_per_state']+confidence_weight*sequence_terms['confidence_per_state']).sum()/total
                sequence_loss.backward()
                accumulate(memory,'sequence_loss',sequence_loss)
                memory['sequence_states'] = memory.get('sequence_states',0)+len(earlier['hist'])
                loss = loss.detach()+sequence_loss.detach()
            if compute_metrics:
                decisions.extend(decision_rows(output, batch, model.cfg, n_commit, tolerance))
                if 'candidate_confidence_logits' in output:
                    for name, row in candidate_decisions(output, batch, model.cfg, n_commit).items():
                        group = candidate_groups.setdefault(name, {})
                        for key, value in row.items():
                            group[key] = group.get(key, 0)+value
                if 'query_embedding' in output:
                    rankings.append(identity_ranking(output, batch, model.cfg))
                    for name,row in identity_training_groups(output,batch,terms,model.cfg,identity_temperature).items():
                        group = identity_groups.setdefault(name,{})
                        for key,value in row.items():
                            group[key] = group.get(key,0)+value
            for key in IDENTITY_SUMS:
                if key in terms:
                    accumulate(identity, key, terms[key])
            for key in ('presence_dropped', 'blurred', 'foreign_components', 'seed_present', 'identity_observable'):
                if key in cpu:
                    identity[key] = identity.get(key, 0.)+float((cpu[key] > 0).sum())
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
                        'confidence_labeled_states', 'confidence_departed_states'):
                accumulate(sums, key, terms[key])
            if 'source' in cpu:
                for source in range(len(sources)):
                    sources[source] += int((cpu['source'] == source).sum())
                if 'bank_tail_length' in cpu:
                    bank_tails.extend(cpu['bank_tail_length'][cpu['source'] == 3].tolist())
        if chunk_losses:
            torch.stack(chunk_losses).sum().backward()
            stream_states.detach()
            if model.cfg.feature_memory_revision == 2 and model.cfg.feature_replay_weight:
                replay = stream_states.replay.backward(model, total, device=device, tolerance=tolerance,
                    n_commit=n_commit, confidence_weight=confidence_weight,
                    candidate_weight=candidate_weight, memory_probe_weight=memory_probe_weight)
                for key, value in replay.items():
                    sums[key] = sums.get(key, 0.)+value
                sums['loss'] = sums['loss']+replay['replay_loss']
    resolve_device_sums(sums, identity, memory)
    # The summed loss is finite only if every microbatch loss was; checked before any update.
    if not math.isfinite(sums['loss']):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    sums.update(clip_training_gradients(model, memory_grad_clip, rest_grad_clip))
    opt.step()
    update_ema(ema, model, step, ema_decay)
    sums['observed_states'] = total
    sums.update(error_mean=sums['error_sum']/max(1., sums['geometry_count']),
                prefix_correct_fraction=sums['correct_count']/max(1., sums['confidence_count']),
                fresh_fraction=float(sources[0]/total),
                recent_fraction=float(sources[2]/total),bank_wrong_continuation_fraction=float(sources[3]/total),
                bank_following_fraction=float(sources[4]/total),
                decision_pair_fraction=float(sources[5]/total), decision_requested_fraction=requested_decisions/total)
    sums.update(bank_wrong_continuation_tail_mean=float(np.mean(bank_tails)) if bank_tails else None,
                bank_wrong_continuation_tail_min=min(bank_tails) if bank_tails else None,
                bank_wrong_continuation_tail_max=max(bank_tails) if bank_tails else None)
    if memory:
        memory.update(probe_identity_accuracy=memory.get('identity_correct', 0.)/max(1., memory.get('identity_count', 0.)),
                      probe_departed_recall=memory.get('departed_correct', 0.)/max(1., memory.get('departed_count', 0.)),
                      probe_offset_error_mean=memory.get('offset_error_sum', 0.)/max(1., memory.get('offset_count', 0.)),
                      labeled_writes_per_state=memory.get('labeled_writes', 0.)/total,
                      labeled_state_fraction=memory.get('labeled_states', 0.)/total,
                      departed_state_fraction=memory.get('departed_states', 0.)/total)
        sums['memory'] = memory
    if identity:
        # Versioned separately from the distance metrics above.
        identity.update(identity_version=1, pair_sampling_version=pair_version,
                        eligible_fraction=identity.get('identity_states',0.)/total,
                        identity_rank_accuracy=identity.get('identity_rank_correct', 0.)/max(1., identity.get('identity_count', 0.)),
                        identity_prefix_correct_fraction=identity.get('identity_correct_count', 0.)/max(1., sums['confidence_count']))
        identity['identity_loss_eligible'] = identity.get('identity_loss', 0.)*total/max(1., identity.get('identity_states', 0.))
        if 'candidate_loss' in identity:
            identity['candidate_loss_eligible'] = identity['candidate_loss']*total/max(1., identity.get('candidate_states', 0.))
        for key in ('presence_dropped', 'blurred', 'foreign_components', 'seed_present', 'identity_observable', *(f'location_{n}' for n in LOCATION_SOURCES)):
            if key in identity:
                identity[key+'_fraction'] = identity.pop(key)/total
        sums['identity'] = identity
    if compute_metrics:
        sums['decisions'] = summarize_decisions(decisions, commit_window(model.cfg, n_commit))
        if rankings:
            sums['identity']['ranking'] = summarize_ranking(rankings)
            sums['identity']['training_groups'] = summarize_identity_groups(identity_groups)
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
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--microbatch', type=int, default=24)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--worker-cache-gb', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--ema-decay', type=float, default=.999)
    ap.add_argument('--memory-grad-clip', type=float, default=MEMORY_GRAD_CLIP,
                    help='Gradient-norm limit for recurrent memory only; 0 disables clipping')
    ap.add_argument('--rest-grad-clip', type=float, default=REST_GRAD_CLIP,
                    help='Independent gradient-norm limit for all other parameters; 0 disables clipping')
    ap.add_argument('--confidence-weight', type=float, default=.5)
    ap.add_argument('--tolerance', type=float, default=1.5)
    ap.add_argument('--n-commit', type=int, default=4)
    ap.add_argument('--channels', type=int, default=DirectConfig.channels, help='Base image encoder width')
    ap.add_argument('--direction-inputs', action=argparse.BooleanOptionalAction, default=False,
                    help='Add six sign-invariant direction channels to main, memory and seed crops; no image augmentations on these channels')
    ap.add_argument('--decoder-layers', type=int, default=4)
    ap.add_argument('--axial-layers', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--memory-version', type=int, choices=(2,3,4), default=2,
                    help='2: recurrent follower; 3: spatial lattice route; 4: continuous regression (requires --no-correction)')
    ap.add_argument('--trajectory-sequence-weight', type=float, default=DirectConfig.trajectory_sequence_weight,
                    help='Obsolete v4 prefix loss; must be zero for feature-memory streams')
    ap.add_argument('--feature-sequence-length', type=int, default=DirectConfig.feature_sequence_length,
                    help='V4 decisions per gradient chunk; microbatch counts crops across time and traces')
    for name in FEATURE_OPTIONS:
        default = getattr(DirectConfig, name)
        ap.add_argument('--'+name.replace('_', '-'), type=type(default), default=default)
    ap.add_argument('--feature-memory-grid', type=int, nargs=3, default=DirectConfig.feature_memory_grid,
                    help='V4 pooled main-encoder spatial grid, depth height width')
    ap.add_argument('--route-grid-step', type=float, default=DirectConfig.route_grid_step)
    ap.add_argument('--route-transition-radius', type=int, default=DirectConfig.route_transition_radius)
    ap.add_argument('--route-transition-cost', type=float, default=DirectConfig.route_transition_cost)
    ap.add_argument('--route-loss-weight', type=float, default=DirectConfig.route_loss_weight)
    ap.add_argument('--route-sequence-weight', type=float, default=DirectConfig.route_sequence_weight)
    ap.add_argument('--route-refinement-radius', type=float,
                    help='V3 total per-axis refinement bound in trace voxels; omitted preserves checkpoint value or legacy half-cell bound; may change on resume')
    ap.add_argument('--memory-slots', type=int, default=16,
                    help='Learned recurrent memory slots; 0 preserves the crop-only model')
    ap.add_argument('--memory-steps', type=int, default=64,
                    help='Past observed patches unrolled before the supervised current decision')
    ap.add_argument('--memory-stride', type=int, default=4,
                    help='Spacing of reconstructed historical observations in trace voxels; may change on resume')
    ap.add_argument('--memory-patch-size', type=int, default=17,
                    help='V2/v3 odd raw memory-patch size; v4 reuses main-encoder features')
    ap.add_argument('--memory-grad-steps', type=int, default=32,
                    help='Newest observations that backpropagate; older ones are a no-grad burn-in')
    ap.add_argument('--memory-probe-weight', type=float, default=.5,
                    help='Per-write departure/offset probe coefficient (memory models)')
    ap.add_argument('--memory-departed-weight', type=float, default=1.,
                    help='BCE multiplier for labeled departed observations only; may change on resume')
    ap.add_argument('--memory-switch-probability', type=float, default=0.,
                    help='Fresh draws replaced by original-then-neighbor memory sequences')
    ap.add_argument('--memory-switch-tail', type=float, nargs=2, default=(16.,96.), metavar=('MIN', 'MAX'),
                    help='Neighbor tail length of memory-switch sequences in trace voxels')
    ap.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--correction', action=argparse.BooleanOptionalAction, default=True,
                    help='Refine the curve using refreshed local and deep image evidence')
    ap.add_argument('--correction-steps', type=int, default=2)
    ap.add_argument('--correction-limit', type=float, default=1., help='Maximum lateral correction per step in trace voxels')
    ap.add_argument('--no-history-prob', type=float, default=.15, help='Fresh-state probability of absent observed history')
    ap.add_argument('--short-history-prob', type=float, default=.4,
                    help='Given history is present, probability of a balanced 1-8/9-32 point startup history')
    ap.add_argument('--identity-weight', type=float, default=.5, help='InfoNCE coefficient')
    ap.add_argument('--identity-temperature', type=float, default=.1)
    ap.add_argument('--decision-fraction', type=float, default=.25,
                    help='Fraction reserved for matched pairs with visible reference seeds')
    ap.add_argument('--candidate-weight', type=float, default=1., help='Weight of candidate prefix BCE through the existing confidence head')
    ap.add_argument('--fresh-fraction', type=float, default=.7,
                    help='Fresh share of non-pair draws; remainder uses current replay; may change on resume')
    ap.add_argument('--embedding', type=int, default=32)
    ap.add_argument('--negative-bank', help='Shared live bank for InfoNCE negatives, wrong continuations and following supervision')
    ap.add_argument('--near-negative-bank', help='Additional bank of validated nearby negative relationships')
    ap.add_argument('--following-bank', help='Following path source (default: negative-bank)')
    ap.add_argument('--continuation-bank', help='Wrong-continuation path source (default: negative-bank)')
    ap.add_argument('--negative-near-fraction', type=float, default=.5, help='Negative slots reserved for nearby paths (new runs: .5)')
    ap.add_argument('--negative-near-distance', type=float, default=12., help='Near/outer split in trace voxels (default: 12)')
    ap.add_argument('--bank-coverage-probability', type=float, default=.2, help='Fresh slots reserved for covered parents with usable history (new runs: .2)')
    ap.add_argument('--prefer-long-continuations', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--negative-lateral-max', type=float, default=32.,
                    help='Maximum neighbor distance; every sampled point must also lie inside the CT crop')
    ap.add_argument('--negative-bank-refresh-seconds', type=float, default=30., help='Each loader worker checks for completed new negative shards at this interval')
    ap.add_argument('--negative-bank-cache-mb', type=float, default=64., help='Maximum cached negative geometry per loader worker')
    ap.add_argument('--bank-wrong-continuation-probability', type=float, default=.75,
                    help='Fraction of recent DAgger departure draws to replace with safe bank continuations when available')
    ap.add_argument('--bank-wrong-continuation-tail', type=float, nargs=2, default=(4.,16.), metavar=('MIN', 'MAX'),
                    help='Wrong-fiber tail range in trace voxels (default: 4 16)')
    ap.add_argument('--bank-following-probability', type=float, default=.1,
                    help='Fraction of fresh draws following validated bank paths (default: .1)')
    ap.add_argument('--presence-dropout', type=float, default=.25,
                    help='Probability of zeroing the presence crop; may be changed on resume')
    ap.add_argument('--blur-probability', type=float, default=.25,
                    help='Probability of shared CT/presence Gaussian blur; may be changed on resume')
    ap.add_argument('--blur-sigma', type=float, nargs=2, default=(.5, 1.25), metavar=('MIN', 'MAX'),
                    help='Gaussian blur sigma range in sampled crop voxels; may be changed on resume')
    ap.add_argument('--contacts', help='Mined contact episodes of the training fibers (oversampled)')
    ap.add_argument('--hard-spans', help='Hard controlled spans by fiber name (oversampled)')
    ap.add_argument('--contact-fraction', type=float, default=.2, help='Fresh draws near contact episodes')
    ap.add_argument('--hard-span-fraction', type=float, default=.1)
    ap.add_argument('--lateral-fraction', type=float, default=.1,
                    help='Fresh draws near earlier states with bank negatives')
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
    ap.add_argument('--dagger-after', type=float, default=24.,
                    help='Replay states recorded after a confirmed departure, in trace voxels')
    ap.add_argument('--replay-keep', type=int, default=4)
    ap.add_argument('--resume', help='Resume last.pt inside this run with the same training options')
    ap.add_argument('--reset-optimizer', action='store_true',
                    help='With --resume: fresh AdamW, one LR for all parameters, no transfer freeze, and restart LR warmup/decay')
    ap.add_argument('--recurrent-refinement-steps', type=int, default=0,
                    help='V4 shared-decoder refinement passes; upgrade existing runs with upgrade_refinement')
    ap.add_argument('--recurrent-refinement-limit', type=float, default=None,
                    help='V4 maximum lateral displacement norm per refinement pass, in trace voxels; '
                         'omitted keeps the checkpoint value (1 for new models); may change on resume')
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
    if args.reset_optimizer and not args.resume:
        raise ValueError('--reset-optimizer requires --resume')
    if any(not math.isfinite(v) or v < 0 for v in (args.memory_grad_clip, args.rest_grad_clip)):
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
    # Omitted: new models use the config default; checkpoints keep their own value.
    refinement_limit = ({} if args.recurrent_refinement_limit is None else
                        dict(recurrent_refinement_limit=args.recurrent_refinement_limit))
    cfg = DirectConfig(direction_inputs=args.direction_inputs,channels=args.channels,hidden=args.hidden,layers=args.axial_layers,
                       decoder_layers=args.decoder_layers,embedding=args.embedding,
                       activation_checkpointing=args.activation_checkpointing,correction=args.correction,
                       correction_limit=args.correction_limit,correction_steps=args.correction_steps,
                       memory_slots=args.memory_slots,memory_steps=args.memory_steps,
                       memory_stride=args.memory_stride,memory_patch_size=args.memory_patch_size,
                       memory_grad_steps=args.memory_grad_steps,memory_version=args.memory_version,
                       memory_departed_weight=args.memory_departed_weight,
                       trajectory_sequence_weight=args.trajectory_sequence_weight,
                       feature_sequence_length=args.feature_sequence_length, feature_memory_grid=args.feature_memory_grid,
                       recurrent_refinement_steps=args.recurrent_refinement_steps,
                       **refinement_limit,
                       **{k:getattr(args,k) for k in (*ROUTE_OPTIONS, *FEATURE_OPTIONS)})
    args.recurrent_refinement_limit = cfg.recurrent_refinement_limit
    initialized = None
    resume = None
    if args.init_tracer:
        initialized,_,_,_,_ = load_checkpoint(args.init_tracer,args.device)
        if args.memory_version == 4 and not initialized.cfg.feature_memory:
            cfg = replace(initialized.cfg, memory_version=4, correction=False, route_refinement_radius=None,
                          trajectory_sequence_weight=args.trajectory_sequence_weight,
                          feature_sequence_length=args.feature_sequence_length, feature_memory_grid=args.feature_memory_grid,
                          **{k: getattr(args,k) for k in MEMORY_OPTIONS})
            reset_coordinates = initialized.cfg.memory_version == 3
            initialized = initialize_trajectory_model(initialized, cfg)
            progress('Initialized feature-memory v4 from shared EMA encoder/decoder weights; new memory initialized'
                     + ('; continuous coordinate head starts fresh' if reset_coordinates else ''))
        elif args.memory_version == 3 and initialized.cfg.memory_version != 3:
            cfg = replace(initialized.cfg,memory_version=3,
                          **{k:getattr(args,k) for k in (*MEMORY_OPTIONS,*ROUTE_OPTIONS)})
            initialized = initialize_spatial_model(initialized,cfg)
            progress('Initialized spatial memory v3 from shared EMA weights; route head starts fresh')
        elif args.memory_slots and not initialized.cfg.memory_slots:
            cfg = replace(initialized.cfg, **{k: getattr(args, k) for k in MEMORY_OPTIONS})
            upgraded = build_model(cfg).to(args.device, memory_format=conv_memory_format(args.device))
            incompatible = upgraded.load_state_dict(initialized.state_dict(), strict=False)
            if incompatible.unexpected_keys or any(not k.startswith('recurrent_memory.') for k in incompatible.missing_keys):
                raise ValueError('Unexpected parameter mismatch initializing memory model')
            initialized = upgraded
        else:
            if args.memory_slots and any(getattr(args,k) != getattr(initialized.cfg,k) for k in MEMORY_OPTIONS):
                raise ValueError('--init-tracer memory configuration differs from existing memory checkpoint')
            if initialized.cfg.memory_slots and initialized.cfg.memory_version == 1:
                raise ValueError('Legacy v1 memory checkpoints cannot initialize new runs')
            cfg = initialized.cfg
        if args.recurrent_refinement_steps and args.recurrent_refinement_steps != cfg.recurrent_refinement_steps:
            raise ValueError('Adding recurrent refinement to a checkpoint requires upgrade_refinement')
        for key in (*MEMORY_OPTIONS,'memory_version',*ROUTE_OPTIONS,'trajectory_sequence_weight',
                    'feature_sequence_length','feature_memory_grid','recurrent_refinement_steps','recurrent_refinement_limit',*FEATURE_OPTIONS):
            setattr(args, key, getattr(cfg, key))
        if args.direction_inputs and not cfg.direction_inputs:
            initialized = add_direction_inputs(initialized)
            cfg = initialized.cfg
            progress('Initialized six direction inputs with zero weights; existing EMA weights retained')
        args.direction_inputs = cfg.direction_inputs
    if args.resume:
        resume = read_checkpoint(args.resume,ARCHITECTURES,args.device)
        # Sampling spacing changes no parameter shapes; use the requested value.
        cfg = replace(checkpoint_config(resume), memory_stride=args.memory_stride,
                      memory_departed_weight=args.memory_departed_weight,
                      feature_switch_crop_fraction=args.feature_switch_crop_fraction, **refinement_limit)
        args.recurrent_refinement_limit = cfg.recurrent_refinement_limit
        if args.direction_inputs != cfg.direction_inputs:
            raise ValueError('Resume direction inputs differ; use --init-tracer for a new direction-enabled run')
    cfg = resolve_refinement_config(cfg, args.route_refinement_radius)
    args.route_refinement_radius = cfg.route_refinement_radius
    if initialized is not None:
        initialized.cfg = cfg
    if args.decision_fraction and args.microbatch % 2:
        raise ValueError('Matched decisions require an even microbatch')
    stream_batch = args.microbatch
    if cfg.feature_memory:
        if args.microbatch % cfg.feature_sequence_length:
            raise ValueError('V4 microbatch must divide into feature_sequence_length decisions')
        stream_batch = args.microbatch//cfg.feature_sequence_length
        if args.decision_fraction and stream_batch % 2:
            raise ValueError('V4 matched decisions need an even number of traces: microbatch / feature_sequence_length')
    if not np.isfinite(args.candidate_weight) or args.candidate_weight <= 0:
        raise ValueError('Candidate weight must be finite and positive')
    if not np.isfinite(args.memory_probe_weight) or args.memory_probe_weight < 0 or args.dagger_after <= 0:
        raise ValueError('Memory probe weight must be nonnegative; --dagger-after positive')
    if args.memory_switch_probability and not cfg.memory_slots:
        raise ValueError('Memory-switch sequences require --memory-slots')
    identity_sampling = IdentitySampling(
        rule=ComponentRule(lateral_max=args.negative_lateral_max),
        presence_dropout=args.presence_dropout,contact_fraction=args.contact_fraction,
        blur_probability=args.blur_probability,blur_sigma=args.blur_sigma,
        hard_span_fraction=args.hard_span_fraction,lateral_fraction=args.lateral_fraction,
        bank_wrong_continuation_probability=args.bank_wrong_continuation_probability,
        bank_wrong_continuation_tail=args.bank_wrong_continuation_tail,
        bank_following_probability=args.bank_following_probability,
        decision_fraction=args.decision_fraction,negative_near_fraction=args.negative_near_fraction,
        negative_near_distance=args.negative_near_distance,bank_coverage_probability=args.bank_coverage_probability,
        prefer_long_continuations=args.prefer_long_continuations,
        memory_switch_probability=args.memory_switch_probability,memory_switch_tail=args.memory_switch_tail)
    if not args.negative_bank:
        raise ValueError('Training requires --negative-bank')
    if min(args.identity_weight,args.identity_temperature) <= 0:
        raise ValueError('Identity weight and temperature must be positive')
    if not 1 <= args.n_commit <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    # Finest-level CT in a single enlarged crop.
    spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4.,
                           inputs='ct+presence')
    if cfg.direction_inputs:
        # Validate sibling paths and grids before starting loaders or collectors.
        FiberVolume(spec, cache_bytes=1 << 20).direction_fields()
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
                   'log_every','ckpt_every','diag_every','dagger_device','compile','init_tracer',
                   'negative_bank_refresh_seconds','negative_bank_cache_mb','activation_checkpointing',
                   'memory_grad_clip','rest_grad_clip','presence_dropout','blur_probability','blur_sigma',
                   'decision_fraction','bank_following_probability','fresh_fraction','route_refinement_radius',
                   'n_commit','memory_stride','memory_departed_weight','feature_switch_crop_fraction',
                   'recurrent_refinement_limit'}
        if not cfg.feature_memory:
            ignored.update(('trajectory_sequence_weight', 'feature_sequence_length', 'feature_memory_grid'))
        defaults = build_parser()
        for key,value in vars(args).items():
            # Options added after a run started had their default behavior.
            recorded = resume['training_options'].get(key, getattr(DirectConfig, key, defaults.get_default(key))
                                                       if key.startswith('memory_') else defaults.get_default(key))
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
    progress(f'Independent gradient clipping: memory={args.memory_grad_clip:g}, rest={args.rest_grad_clip:g} (0 disables clipping)')
    model = initialized if initialized is not None else build_model(cfg).to(args.device, memory_format=conv_memory_format(args.device))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt, done, lr_restart_step = initialize_training_optimizer(model, ema, args, resume)
    if args.reset_optimizer:
        progress(f'Fresh AdamW: one parameter group, all parameters trainable; LR restarts at update {done+1} '
                 f'with {args.warmup} warmup updates to {args.lr:g}')
    # The compiled wrapper shares the module's parameters, so EMA updates, gradient
    # clipping and checkpoints keep using ``model``; only the training forward is compiled.
    trainable = compile_training_model(model) if args.compile and torch.device(args.device).type == 'cuda' else model
    if args.compile and torch.device(args.device).type == 'cuda':
        progress('Compilation enabled; first forward/backward passes will compile lazily and may take several minutes')
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
        extra_args=('--after', args.dagger_after))
    contacts = load_contacts(args.contacts,train_f,band) if args.contacts else ()
    hard_spans = load_hard_spans(args.hard_spans,train_f) if args.hard_spans else ()
    progress(f'Sampling: {len(contacts)} contacts, {len(hard_spans)} hard spans; memory slots={cfg.memory_slots}')
    if cfg.feature_memory:
        progress(f'Feature-memory streams: {stream_batch} traces x {cfg.feature_sequence_length} decisions per full chunk; state persists across chunks')
    builder = IdentityObservationBuilder(cfg,train_f,identity_sampling,contacts=contacts,
        hard_spans=hard_spans,augment=True,negative_bank=negative_bank,**role_banks)
    dataset = FollowDataset(train_f, spec, sample, band, chunk=stream_batch, seed=args.seed+done,
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
        source_sampling=dict(fresh=args.fresh_fraction,recent=1-args.fresh_fraction),
        pair_sampling_version=identity_sampling.pair_sampling_version,
        feature_sampling_revision=FEATURE_SAMPLING_REVISION if cfg.feature_memory else None,
        history_policy=('main_encoder_feature_memory' if cfg.feature_memory else
                        'learned_observation_memory' if cfg.memory_slots else 'visible_crop_only'),sampling=asdict(identity_sampling),
        negative_bank_path=str(negative_bank.root),negative_bank_provenance=negative_bank.provenance(),
        bank_role_provenance=role_provenance()))
    tracer = None
    recovery_vol = FiberVolume(spec) if recovery_states is not None else None
    started = time.monotonic()
    interval_started, interval_step = started, done
    interval_data_seconds = interval_update_seconds = 0.
    interval_metrics = DirectTrainingInterval()
    try:
        if args.diag_every or args.long_diag_every:
            from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
            tracer = DirectTracer(ema, FiberVolume(spec), cfg.fine, cfg.n_history,
                TraceParams(n_commit=args.n_commit, max_len=args.diag_max_len), device=args.device)
        progress(f'Starting data loader; waiting for {args.batch//args.microbatch} microbatches for update {done+1}')
        iterator = iter(loader)
        from .feature_sequences import FeatureStreamStates, sequence_steps
        stream_states = FeatureStreamStates()
        observed_states = interval_states = 0
        prior_samples = int(resume.get('samples_seen', done*args.batch)) if resume else 0
        for step in range(done+1, args.steps+1):
            event = collector.poll()
            if event:
                log.record(dict(step=step, **event))
            early = step <= done+5
            batch_started = time.monotonic()
            if early and step != done+1:
                progress(f'Update {step}: waiting for data')
            batches = []
            batch_states = 0
            while batch_states < args.batch:
                chunk = next(iterator)
                batches.append(chunk)
                batch_states += sum(len(b['hist']) for b in sequence_steps(chunk))
                if step == done+1:
                    progress(f'Update {step}: received {batch_states}/{args.batch} crop states')
            data_seconds = time.monotonic()-batch_started
            update_started = time.monotonic()
            if early:
                progress(f'Update {step}: data ready in {data_seconds:.1f}s; running forward/backward and optimizer')
            lr = lr_at(step-lr_restart_step, args.lr, args.warmup, args.steps-lr_restart_step)
            diagnostic = {} if tracer is not None and args.diag_every and step % args.diag_every == 0 else None
            metrics = optimizer_update(trainable, ema, opt, batches, step, lr, device=args.device,
                tolerance=args.tolerance, confidence_weight=args.confidence_weight, ema_decay=args.ema_decay,
                n_commit=args.n_commit, compute_metrics=step % args.log_every == 0 or step == args.steps,
                identity_weight=args.identity_weight, identity_temperature=args.identity_temperature,
                candidate_weight=args.candidate_weight, memory_probe_weight=args.memory_probe_weight,
                memory_grad_clip=args.memory_grad_clip, rest_grad_clip=args.rest_grad_clip,
                stream_states=stream_states, diagnostic=diagnostic)
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
                    memory_departed_weight=cfg.memory_departed_weight,
                    feature_switch_crop_fraction=cfg.feature_switch_crop_fraction,
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
                    feature_sampling_revision=FEATURE_SAMPLING_REVISION if cfg.feature_memory else None,
                    samples_seen=prior_samples+observed_states,
                    identity_sampling=asdict(identity_sampling),
                    negative_bank_provenance=negative_bank.provenance() if negative_bank else None,
                    bank_role_provenance=role_provenance(),
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
                for upgrade in ('refinement_upgrade', 'feature_memory_upgrade'):
                    if resume and upgrade in resume:
                        extra[upgrade] = resume[upgrade]
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
                                     device=args.device, memory=diagnostic['memory'])
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
        event = collector.close()
        if event:
            log.record(event)
        if tracer is not None:
            tracer.close()
        log.close()
    return str(out/'last.pt')


if __name__ == '__main__':
    main()
