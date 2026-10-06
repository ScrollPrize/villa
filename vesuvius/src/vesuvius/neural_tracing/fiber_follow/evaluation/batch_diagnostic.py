"""Current-microbatch contact sheets and cheap measurements; PNG/JSON only."""
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix
from vesuvius.neural_tracing.fiber_follow.train.supervision import geometry_mask, foreign_failures


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
                     if key.startswith('ct_frame_')}
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


SEQUENCE_EPISODES, SEQUENCE_DECISIONS = 4, 4  # sequence sheets: the last decisions of the first episodes


@torch.no_grad()
def sequence_rows(model, cpu_batch, device, n_commit):
    """A sequence model's episode batch reduced to the rows the sheets show: the last ``SEQUENCE_DECISIONS``
    supervised decisions of the first ``SEQUENCE_EPISODES`` episodes, each predicted exactly as in training (its own
    crop with its episode's earlier steps as history, train/sequence.py). Returns those rows and their outputs."""
    from vesuvius.neural_tracing.fiber_follow.train.sequence import episode_forward, select_rows
    from vesuvius.neural_tracing.fiber_follow.train.train import move_batch
    from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
        output, _ = episode_forward(model, move_batch(cpu_batch, device), DEFAULT_CONFIDENCE, n_commit)
    supervised = cpu_batch['episode_supervised'].bool().nonzero().flatten()
    episodes = cpu_batch['episode_index'][supervised]
    picks = [k for episode in episodes.unique(sorted=True)[:SEQUENCE_EPISODES].tolist()
             for k in (episodes == episode).nonzero().flatten()[-SEQUENCE_DECISIONS:].tolist()]
    picks = torch.tensor(picks, dtype=torch.long)
    count = len(supervised)
    outputs = {key: value[picks.to(value.device)] for key, value in output.items()
               if torch.is_tensor(value) and value.ndim and len(value) == count}
    return select_rows(cpu_batch, supervised[picks], len(cpu_batch['hist'])), outputs


@torch.no_grad()
def render_microbatch(model, cpu_batch, out, step, *, device, n_commit, tolerance,
                      dataset_names=(), training_metrics=None):
    """One sheet per image type; every current-microbatch row, in loader order."""
    from vesuvius.neural_tracing.fiber_follow.train.train import move_batch
    from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostic_plots import plot_sheets
    folder = Path(out)/'diagnostic_images'/str(step)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    examples, rows = [], []
    inference_seconds = 0.
    try:
        episode_outputs = None
        if model.model_type == 'sequence':
            cpu_batch, episode_outputs = sequence_rows(model, cpu_batch, device, n_commit)
        for i in range(len(cpu_batch['hist'])):
            batch = move_batch(slice_batch(cpu_batch, slice(i, i+1)), device)
            # Singleton inference keeps each adaptive refinement's row identity
            # unambiguous, and bounds peak diagnostic GPU memory.
            begin = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                if episode_outputs is None:
                    output = model(batch['x'], batch['hist'], batch['hmask'], n_commit=n_commit)
                else:  # a sequence decision, predicted with its episode's history (sequence_rows)
                    output = {key: value[i:i+1] for key, value in episode_outputs.items()}
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
            row = dict(example=i, dataset=name, dataset_id=dataset_id,
                source=int(batch['source'][0]) if 'source' in batch else None,
                terminal=bool(batch['terminal'][0]), supervision=int(batch['supervision'][0]),
                identity_observable=bool(batch['identity_observable'][0]) if 'identity_observable' in batch else None,
                selected_attempt=selected, **details, **measured_geometry(batch),
                confidence=confidence, hazard_probability=output['hazard_logits'][0].sigmoid(),
                mean_error=float(errors.mean()) if len(errors) else None,
                p95_error=float(errors.quantile(.95)) if len(errors) else None,
                max_error=float(errors.max()) if len(errors) else None,
                known_prefixes=int(known.sum()),
                prefix_brier=float((confidence[known]-details['labels'][known]).square().mean()) if known.any() else None,
                ct_statistics=tensor_stats(batch['x']['fine'][:, 0]))
            rows.append(row)
            # Retain only display-sized CT sections and reduced features on CPU.
            from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostic_plots import display_example
            examples.append(display_example(batch, output, details, model.cfg, label, row))
        plot_sheets(examples, model.cfg, folder, step)
        errors = [r['mean_error'] for r in rows if r['mean_error'] is not None]
        report = dict(step=step, split='current_training_microbatch', model='EMA', model_type=model.model_type, examples=len(rows),
            n_commit=n_commit, tolerance=tolerance,
            summary=dict(committed_points=sum(r['commit'] for r in rows),
                stopped=sum(r['commit'] == 0 for r in rows),
                mean_example_error=sum(errors)/len(errors) if errors else None,
                geometry_examples=len(errors)),
            timing=dict(inference_and_measurement_seconds=inference_seconds,
                        total_diagnostic_seconds=time.perf_counter()-started),
            training_update=training_metrics or {}, rows=rows,
            interpretation=dict(orientation='Fixed CT sections and annotation in the actual crop frame; no GT-following reslicing.',
                confidence='Prefix survival; unknown labels excluded from error and calibration metrics.',
                sampling='Current augmented training microbatch; not held-out validation.'))
        (folder/'metrics.json').write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+'\n')
        report['timing']['total_diagnostic_seconds'] = time.perf_counter()-started
        return report['summary'] | dict(examples=len(rows), **report['timing'])
    finally:
        model.train(was_training)
