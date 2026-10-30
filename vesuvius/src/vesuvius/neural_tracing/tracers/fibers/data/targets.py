"""Plane targets and state labels for a decision on an original fiber, and the annotation-only batch tensors."""
from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import frame_from_heading, interp_at, traversal_curve
from vesuvius.neural_tracing.tracers.fibers.frames.ct import oriented_seed_heading, reframe_item
from vesuvius.neural_tracing.tracers.fibers.data.state_labels import supervise


def plane_targets(p, s, t, t_end, pos, frame, planes):
    """Where the GT curve (traversal arc ``s``, from ``t`` up to ``t_end``)
    first crosses each local forward plane c = planes[k]. Returns lateral
    (a, b) per plane (K, 2) and a validity mask (K,)."""
    K = len(planes)
    ab = np.zeros((K, 2), np.float32)
    m = np.zeros(K, np.float32)
    span = min(t_end, t + 2.5 * float(planes[-1])) - t
    if span <= 0:
        return ab, m
    arc = np.r_[t, s[(s > t) & (s < t+span)], t+span]
    loc = (interp_at(p, s, arc) - pos) @ frame
    c = loc[:, 2]
    for k, ck in enumerate(planes):
        hit = np.nonzero((c[:-1] < ck) & (c[1:] >= ck))[0]
        if len(hit):
            i = hit[0]
            w = (ck - c[i]) / max(c[i + 1] - c[i], 1e-9)
            ab[k] = (1 - w) * loc[i, :2] + w * loc[i + 1, :2]
            m[k] = 1.0
    return ab, m


def resolve_trace_seed(item, vol):
    """Give a simulated trace the tracer's CT seed heading before its image is built.

    Sets the seed tangent. A path still holding the seed heading (< 12 voxels) also
    takes it as crop heading: local geometry is re-expressed and labels recomputed.
    Missing CT orientation raises ``SeedHeadingError``: training rejects and resamples
    that seed, exactly as collection and evaluation skip it. There is no annotation
    fallback. Synthetic sources mark themselves with ``seed_heading_family``.
    """
    family = item.pop('seed_heading_family', None)
    pending = item.pop('_pending_seed_heading', None)
    if family is None:
        return item
    item['fiber_family'] = family
    heading = oriented_seed_heading(vol, item['seed_pos'], family, item['seed_tangent'])
    item['seed_tangent'] = heading
    if pending is not None:
        fiber, t, reverse, cfg = pending
        frame = frame_from_heading(heading)
        reframe_item(item, frame)
        item.update(continuation_targets(fiber, t, reverse, item['pos'], frame, cfg))
        supervise(item)
    return item


def continuation_targets(fiber, t, reverse, pos, frame, cfg):
    """Dense supervised geometry; prediction-support gaps never remove GT labels.

    t is traversal arclength. Missing forward crossings are unknown, except
    continuation beyond an explicitly tagged physical endpoint. Whether the
    geometry is supervised at all is the state contract's decision.
    """
    p, s = traversal_curve(fiber, reverse)
    tf = t + cfg.future_s
    fut = interp_at(p, s, np.clip(tf, 0, s[-1]))
    history_arc = t - np.arange(cfg.recent_history_points + 1) * cfg.history_step
    gt_history = (interp_at(p, s, np.clip(history_arc, 0, s[-1])) - pos) @ frame
    gt_history_mask = (history_arc >= 0).astype(np.float32)
    dense_planes = np.linspace(cfg.future_step, cfg.future_s[-1],
                               (cfg.n_future-1)*cfg.dense_substeps+1)
    ab, mask = plane_targets(p, s, t, s[-1], pos, frame, cfg.future_s)
    dense_ab, dense_mask = plane_targets(p, s, t, s[-1], pos, frame, dense_planes)
    fmask = (tf <= s[-1]).astype(np.float32)
    end = (p[-1]-pos) @ frame
    # Only expose endpoint labels when it is within the local traversal window.
    known = fiber.endpoint_stop[0 if reverse else 1] and s[-1]-t <= 2.5*cfg.future_s[-1]
    out = dict(gt_history=gt_history.astype(np.float32), gt_history_mask=gt_history_mask,
               fut_local=(fut-pos) @ frame, fmask=fmask,
               plane_ab=ab, plane_mask=mask, planes=cfg.future_s,
               dense_ab=dense_ab, dense_mask=dense_mask, dense_planes=dense_planes,
               end_local=end, endpoint_known=float(known))
    return out


def refresh_frame_targets(item):
    """Rebuild plane crossings and supervision after a learned heading change; labels only."""
    context = item.get('_frame_label_context')
    if context is None:
        return
    fiber, t, reverse, cfg = context
    item.update(continuation_targets(fiber, t, reverse, item['pos'], item['frame'], cfg))
    supervise_matched(item)


def collate_targets(items):
    """Pack annotation-only tensors, independently of a model's image inputs."""
    st = lambda k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32))
    out = {}
    if "fut_local" in items[0]:
        out["gt_history"] = st("gt_history")
        out["gt_history_mask"] = st("gt_history_mask")
        out["plane_ab"] = st("plane_ab")
        out["plane_mask"] = st("plane_mask")
        for key in ("dense_ab", "dense_mask", "end_local", "endpoint_known", "terminal", "match_distance"):
            out[key] = st(key)
        for key in ('geometry_valid', 'confidence_valid'):
            out[key] = torch.tensor([bool(it[key]) for it in items], dtype=torch.bool)
        for key in ('supervision', 'supervision_reason'):
            out[key] = torch.tensor([int(it[key]) for it in items], dtype=torch.long)
    if 'source' in items[0]:
        out['source'] = torch.tensor([it['source'] for it in items], dtype=torch.long)
        out['source_step'] = torch.tensor([it.get('source_step', -1) for it in items], dtype=torch.long)
        out['fiber_id'] = torch.tensor([it['fiber_ref'][0] for it in items], dtype=torch.long)
        out['seed_age'] = torch.tensor([float(it.get('seed_age', 0.)) for it in items], dtype=torch.float32)
        out['travelled'] = torch.tensor([float(it.get('travelled', 0.)) for it in items], dtype=torch.float32)
        for key, default in (('task_requested', -1), ('task_delivered', -1), ('task_fallback', 0),
                             ('startup', -1), ('replay_class', -1), ('replay_episode', -1), ('replay_event', -1)):
            out[key] = torch.tensor([int(it.get(key, default)) for it in items], dtype=torch.long)
        for key in ('excursion', 'live_requested', 'live_continuation', 'live_terminal'):
            out[key] = torch.tensor([bool(it.get(key, False)) for it in items], dtype=torch.bool)
    return out


def label_state(fiber, pos, frame, hist_world, hmask, cfg, *, t, reverse, trace):
    """Relabel an exact observed frame/history against its original fiber.

    ``t`` is the fiber's own arclength of the matched correspondence and ``trace`` the
    state's trace facts (``state_labels.facts``). This is the one labeling call used by
    the collector, replay, live continuation and synthetic sources.
    """
    if len(hist_world) != cfg.n_history or len(hmask) != cfg.n_history:
        raise ValueError('Replay history must equal n_history')
    traversal_t = fiber.length-t if reverse else t
    item = dict(pos=pos, frame=frame, hist_local=(hist_world-pos) @ frame,
                _frame_label_context=(fiber, traversal_t, reverse, cfg), fiber_family=fiber.tag,
                hmask=np.asarray(hmask, np.float32), trace_facts=dict(trace),
                **continuation_targets(fiber, traversal_t, reverse, pos, frame, cfg))
    return supervise_matched(item)


def supervise_matched(item):
    """Supervise ``item``; its ground-truth history counts only when the trace matches the fiber unambiguously."""
    facts = item['trace_facts']
    if not (facts['match_valid'] and not facts['match_ambiguous']):
        item['gt_history_mask'] = np.zeros_like(item['gt_history_mask'])
    return supervise(item)
