"""Current-microbatch contact sheets and cheap measurements; PNG/JSON only."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ..shared.labels import prefix_labels
from ..shared.policy import commit_prefix
from .supervision import geometry_mask, foreign_failures


def slice_batch(batch, selection):
    return {k: slice_batch(v, selection) if isinstance(v, dict) else v[selection]
            for k, v in batch.items()}


def cpu_values(value):
    if isinstance(value, dict):
        return {k: cpu_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_values(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
    return value


def tensor_stats(value):
    value = value.detach().float()
    keys = ('finite_fraction', 'rms', 'mean', 'std', 'abs_max', 'zero_fraction')
    if not value.numel():
        return dict(count=0, **dict.fromkeys(keys))
    finite = torch.isfinite(value)
    clean = torch.where(finite, value, 0.)
    count = finite.sum().clamp_min(1)
    mean = clean.sum()/count
    variance = torch.where(finite, (clean-mean).square(), 0.).sum()/count
    # One small host transfer, avoiding per-statistic CUDA synchronization and
    # dynamic boolean indexing of full spatial activation banks.
    values = torch.stack((finite.float().mean(), (clean.square().sum()/count).sqrt(),
                          mean, variance.sqrt(), clean.abs().max(),
                          ((clean == 0) & finite).sum()/count)).cpu().tolist()
    if values[0] == 0:
        values[1:] = [None]*5
    return dict(count=value.numel(), **dict(zip(keys, values)))


@contextmanager
def layer_capture(model):
    """Capture real intermediate activations on singleton eager EMA inference.

    Cached decoder methods bypass module hooks, so wrap those methods directly.
    Store reduced spatial maps, query activations and per-slot attention, never
    full spatial feature banks. Restore everything even after a failed forward.
    """
    record = dict(encoder={}, decoder={}, scorer={}, history={}, statistics={})
    handles, methods = [], []

    def spatial(name, value, channels_last=False):
        value = value.detach().float()
        if not channels_last:
            value = value.movedim(1, -1)
        record['statistics'][name] = tensor_stats(value)
        # RMS deviation from each channel's spatial mean survives LayerNorm;
        # unlike post-LayerNorm RMS it can show spatially varying information.
        contrast = (value-value.mean((1, 2, 3), keepdim=True)).square().mean(-1).sqrt()[0]
        record['encoder'][name] = contrast[:, contrast.shape[1]//2, :].cpu()

    def patch(name, module, method_name, wrapper):
        prior = module.__dict__.get(method_name)
        methods.append((module, method_name, prior))
        setattr(module, method_name, wrapper(getattr(module, method_name)))

    encoder = model.encoder
    try:
        if hasattr(encoder, 'patch_projection'):
            handles.append(encoder.patch_projection.register_forward_hook(
                lambda m, a, y: spatial('patch embedding', y)))
        if encoder.stem is not None:
            handles.append(encoder.stem.register_forward_hook(lambda m, a, y: spatial('CT stem', y)))
        for index in sorted({0, len(encoder.blocks)//2, len(encoder.blocks)-1}):
            if index >= 0:
                handles.append(encoder.blocks[index].register_forward_hook(
                    lambda m, a, y, index=index: spatial(f'encoder block {index+1}', y, True)))
        handles.append(encoder.norm.register_forward_hook(lambda m, a, y: spatial('encoder output', y, True)))

        for head, modules in (('decoder', model.decoder.layers), ('scorer', model.confidence_scorer.layers)):
            for index, module in enumerate(modules):
                def wrap(original, head=head, index=index):
                    def forward(*args, **kwargs):
                        if index == 0:
                            record[head].setdefault('input', []).append(args[0].detach().float()[0].cpu())
                        result = original(*args, **kwargs)
                        record[head].setdefault(f'block {index+1}', []).append(result.detach().float()[0].cpu())
                        return result
                    return forward
                patch(head, module, 'forward_cached', wrap)
        handles.append(model.decoder.norm.register_forward_hook(
            lambda m, a, y: record['decoder'].setdefault('output', []).append(y.detach().float()[0].cpu())))

        def history_conv(m, a, y):
            record['statistics']['history convolution'] = tensor_stats(y)
            record['history']['convolution'] = y.detach().float().square().mean((1, 2)).sqrt().cpu()
        handles.append(model.history_encoder.convolution.register_forward_hook(history_conv))
        def history_tokens(m, a, y):
            tokens, padding = y
            slots = a[1].shape[1]
            record['statistics']['history tokens'] = tensor_stats(tokens[~padding])
            # Center spatial features within each slot before reducing channels.
            values = tokens.detach().float().reshape(1, slots, 2, 9, 9, -1)
            contrast = (values-values.mean((2, 3, 4), keepdim=True)).square().mean(-1).sqrt().mean(2)
            record['history']['tokens'] = contrast[0].cpu()
        handles.append(model.history_encoder.register_forward_hook(history_tokens))

        for name, module in (('generator', model.history_attention),
                             ('scorer', model.confidence_scorer.history_attention)):
            def wrap(original, module=module, name=name):
                def forward(query, k, v, allowed, empty):
                    attn = module.attention
                    width, heads = attn.embed_dim, attn.num_heads
                    q = F.linear(module.norm(query), attn.in_proj_weight[:width], attn.in_proj_bias[:width])
                    q = q.reshape(len(query), -1, heads, width//heads).transpose(1, 2)
                    scores = (q.float() @ k.float().transpose(-1, -2))/(width//heads)**.5
                    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
                    probabilities = probabilities.masked_fill(empty[:, None, None, None], 0.)
                    slots = probabilities.shape[-1]//162
                    weights = probabilities.reshape(*probabilities.shape[:-1], slots, 162).sum(-1).mean(1)
                    record['history'].setdefault(name+'_attention', []).append(weights[0].detach().cpu())
                    return original(query, k, v, allowed, empty)
                return forward
            patch(name, module, 'forward_cached', wrap)
        yield record
    finally:
        for handle in handles:
            handle.remove()
        for module, name, prior in reversed(methods):
            if prior is None:
                delattr(module, name)
            else:
                setattr(module, name, prior)
@torch.no_grad()
def decision_details(output, batch, cfg, n_commit, tolerance):
    """Use training's identity-aware labels and the actual deployed commit policy."""
    curve = output['points']
    mask = geometry_mask(batch, cfg)
    def errors(points):
        dense = F.interpolate(points[..., :2].transpose(1, 2), size=mask.shape[1],
                              mode='linear', align_corners=True).transpose(1, 2)
        return (dense-batch['dense_ab']).norm(dim=-1).masked_fill(~mask, float('nan'))
    foreign = foreign_failures(curve, batch, cfg, mask.shape[1]) if 'foreign' in batch else None
    labels, known, _ = prefix_labels(curve, batch, tolerance, cfg.max_recovery_distance, extra_failure=foreign)
    if 'identity_observable' in batch:
        known *= batch['identity_observable'][:, None]
    counts, allowed = commit_prefix(curve, output['confidence'], .5, n_commit, cfg.max_recovery_distance)
    attempts = []
    for i, proposal in enumerate(output['refinement_points'].unbind(1)):
        count, _ = commit_prefix(proposal, output['refinement_confidence'][:, i], .5, n_commit, cfg.max_recovery_distance)
        error = errors(proposal)[0]
        valid = torch.isfinite(error)
        attempts.append(dict(attempt=i, attempted=bool(output['refinement_mask'][0, i]),
                             selected=int(output['selected_refinement'][0]) == i,
                             error=float(error[valid].mean()) if valid.any() else None,
                             commit=int(count[0])))
    frame_quality = {key: value[0] for key, value in batch['x'].items()
                     if key.startswith(('ct_frame_', 'history_frame_'))}
    return cpu_values(dict(labels=labels[0], known=known[0], error=errors(curve)[0],
        initial_error=errors(output['initial_points'])[0], commit=int(counts[0]),
        connection_allowed=bool(allowed[0]), attempts=attempts, frame_quality=frame_quality))


def json_values(value):
    """Unknown or nonfinite measurements stay null in strict JSON."""
    import math
    if torch.is_tensor(value):
        return json_values(value.tolist())
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def measured_geometry(batch):
    """Geometry is in the same local frame as the sampled CT, without warping."""
    annotation = batch.get('diagnostic_annotation')
    valid = batch.get('diagnostic_annotation_mask')
    angle = None
    if annotation is not None:
        points = annotation[0][valid[0]].float()
        if len(points) >= 2:
            near = int(points.square().sum(-1).argmin())
            tangent = points[min(len(points)-1, near+2)]-points[max(0, near-2)]
            if tangent.norm() > 1e-8:
                angle = float(torch.rad2deg(torch.atan2(tangent[:2].norm(), tangent[2])))
    return dict(heading_annotation_angle_degrees=angle,
                position_xyz=cpu_values(batch['crop_pos'][0]) if 'crop_pos' in batch else None,
                frame_columns_uv_heading_xyz=cpu_values(batch['crop_frame'][0]) if 'crop_frame' in batch else None)


@torch.no_grad()
def render_microbatch(model, cpu_batch, out, step, *, device, n_commit, tolerance,
                      dataset_names=(), training_metrics=None):
    """One sheet per image type; every current-microbatch row, in loader order."""
    from .train import move_batch
    from .diagnostic_plots import plot_sheets
    folder = Path(out)/'diagnostic_images'/str(step)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    examples, rows = [], []
    inference_seconds = 0.
    try:
        for i in range(len(cpu_batch['hist'])):
            batch = move_batch(slice_batch(cpu_batch, slice(i, i+1)), device)
            # Singleton inference keeps each adaptive refinement's row identity
            # unambiguous, and bounds peak diagnostic GPU memory.
            begin = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                with layer_capture(model) as layers:
                    output = model(batch['x'], batch['hist'], batch['hmask'], n_commit=n_commit)
            details = decision_details(output, batch, model.cfg, n_commit, tolerance)
            output, batch = cpu_values(output), cpu_values(batch)
            inference_seconds += time.perf_counter()-begin  # CPU transfer above synchronizes CUDA.
            dataset_id = int(batch['dataset_id'][0]) if 'dataset_id' in batch else 0
            name = dataset_names[dataset_id] if 0 <= dataset_id < len(dataset_names) else str(dataset_id)
            label = f'{i:02d} | {name}'
            selected = int(output['selected_refinement'][0])
            known = details['known'].bool()
            errors = details['error'][torch.isfinite(details['error'])]
            confidence = output['confidence'][0]
            stats = dict(layers['statistics'])
            for head in ('decoder', 'scorer'):
                for stage, attempts in layers[head].items():
                    stats[f'{head} {stage}'] = [tensor_stats(a) for a in attempts]
            row = dict(example=i, dataset=name, dataset_id=dataset_id,
                source=int(batch['source'][0]) if 'source' in batch else None,
                offtrack=bool(batch['offtrack'][0]),
                identity_observable=bool(batch['identity_observable'][0]) if 'identity_observable' in batch else None,
                selected_attempt=selected, **details, **measured_geometry(batch),
                confidence=confidence, hazard_probability=output['hazard_logits'][0].sigmoid(),
                mean_error=float(errors.mean()) if len(errors) else None,
                p95_error=float(errors.quantile(.95)) if len(errors) else None,
                max_error=float(errors.max()) if len(errors) else None,
                known_prefixes=int(known.sum()),
                prefix_brier=float((confidence[known]-details['labels'][known]).square().mean()) if known.any() else None,
                history_valid=batch['x']['history_valid'][0], history_ages=batch['x']['history_ages'][0],
                ct_statistics=tensor_stats(batch['x']['fine'][:, 0]), activation_statistics=stats)
            for key in ('generator_attention', 'scorer_attention'):
                row[key] = layers['history'].get(key, [])
            rows.append(row)
            # Retain only display-sized CT sections and reduced features on CPU.
            from .diagnostic_plots import display_example
            examples.append(display_example(batch, output, details, layers, model.cfg, label, row))
        plot_sheets(examples, model.cfg, folder, step)
        errors = [r['mean_error'] for r in rows if r['mean_error'] is not None]
        report = dict(step=step, split='current_training_microbatch', model='EMA', examples=len(rows),
            n_commit=n_commit, tolerance=tolerance,
            summary=dict(committed_points=sum(r['commit'] for r in rows),
                stopped=sum(r['commit'] == 0 for r in rows),
                mean_example_error=sum(errors)/len(errors) if errors else None,
                geometry_examples=len(errors)),
            timing=dict(inference_and_measurement_seconds=inference_seconds,
                        total_diagnostic_seconds=time.perf_counter()-started),
            training_update=training_metrics or {}, rows=rows,
            interpretation=dict(encoder='Channel RMS deviation from spatial mean, fixed central v token section.',
                decoder='Actual query activations at the selected attempt; signed values, not spatial images.',
                history='CT inputs, convolution/token features and attention mass; attention is not attribution.',
                orientation='Fixed CT sections and annotation in the actual crop frame; no GT-following reslicing.',
                confidence='Prefix survival; unknown labels excluded from error and calibration metrics.',
                sampling='Current augmented training microbatch; not held-out validation.'))
        (folder/'metrics.json').write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+'\n')
        report['timing']['total_diagnostic_seconds'] = time.perf_counter()-started
        return report['summary'] | dict(examples=len(rows), **report['timing'])
    finally:
        model.train(was_training)
