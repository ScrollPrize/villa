"""Monitor rollouts during training: EMA proposal and refinement images, the primary source's monitor rollouts
with their images, and each AFV source's monitor rollouts."""
from pathlib import Path

import torch

from vesuvius.neural_tracing.tracers.fibers.evaluation.monitor.plots import plot_batch, plot_refinement, plot_rollouts, plot_run_curves
from vesuvius.neural_tracing.tracers.fibers.evaluation.scoring import evaluate, rollout_summary
from vesuvius.neural_tracing.tracers.fibers.tracing.trace import FiberTracer, TraceParams
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.policy import DIAGNOSTIC_THRESHOLDS
from vesuvius.neural_tracing.tracers.shared.tensors import move_batch


@torch.no_grad()
def training_diagnostics(model, cpu_batch, tracer, fibers, seeds, out, step, log, *, device, dataset_name=None):
    """Plot EMA proposals and the same monitor rollouts used for logged metrics."""
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
        plot_batch(batch['x']['fine'], points, target, batch['plane_mask'], model.cfg.fine, images/f'batch_{step:06d}.png',
                   batch['hist'], batch['hmask'], batch['gt_history'], batch['gt_history_mask'],
                   source=batch['source'], terminal=batch['terminal'], confidence=prediction['confidence'])
        curves = prediction.get('solver_points', prediction['refinement_points'])
        labels = ([f'integration stage {i}' for i in range(curves.shape[1])] if 'solver_points' in prediction
                  else ['initial proposal']+[f'feedback proposal {i}' for i in range(1, curves.shape[1])])
        plot_refinement(curves, batch['hist'], batch['hmask'],
                        images/f'correction_{step:06d}.png',
                        labels=labels,
                        target=target, target_mask=batch['plane_mask'])
        for threshold in DIAGNOSTIC_THRESHOLDS:
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
        plot_run_curves(out, model.cfg.model_type)
    finally:
        model.train(was_training)
        tracer.p.confidence = threshold_before


def monitor_afv_sources(model, sources, datasets, cfg, policy, step, log, *, max_len, device):
    """Monitor rollouts of each AFV source's validation fibers, recorded per dataset."""
    for source, source_dataset in zip(sources, datasets):
        if source['kind'] != 'afv':
            continue
        source_tracer = FiberTracer(model,CTVolume(source_dataset.vol_spec),cfg.fine,cfg.n_history,
            TraceParams.from_policy(policy,max_len=max_len),device=device)
        try:
            rows,_ = evaluate(source_tracer,source_dataset.validation_fibers,
                source_dataset.validation_manifest['monitor'],batch=1,coverage_max_len=max_len)
            log.record(dict(step=step,split='monitor',dataset=source['name'],
                threshold=.5,coverage_max_len=max_len,**rollout_summary(rows)))
        finally:
            source_tracer.close()
