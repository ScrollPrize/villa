"""Frozen learned crop frames shared by training, collection, and tracing.

Only CT, H/V family, prior headings and observed paths enter the frame model.
CPU inference is cached per process and batched; data-loader workers never use CUDA.

Continuation crops (``cfg.crop_axis == 'prediction'``, the default): the forward axis is the straight axis through the
head that keeps the previous decision's prediction, beyond what it committed, closest to the axis over the gate horizon
(``predicted_axis``); the roll comes from the frame model. Seeds and restarts, which have no previous prediction, keep the
frame model's heading. Training gives fresh continuation crops a simulated previous prediction
(data.data.simulated_prediction) and live chains their real one (item 'prev_prediction').
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading
from vesuvius.neural_tracing.fiber_follow.tracing.heading import (
    FRAME_POLICY, LEARNED_FRAME_POLICY, FRAME_POLICIES, PREDICTION_FRAME_POLICY, fiber_family, orient_item, reframe_item,
    frame_prefetch_bounds, heading_free_bounds, normal_context,
)
from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path

DEFAULT_FRAME_CHECKPOINT = str(Path(__file__).resolve().parents[1]/'output'/'heading_model_l0_w16_centered_frame'/'ckpt_064000.pt')


def bind_frame_checkpoint(cfg, path):
    path = Path(path).expanduser().resolve()
    cfg.frame_checkpoint = str(path)
    cfg.frame_checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame_predictor(cfg)


def configured_frame_policy(cfg):
    return LEARNED_FRAME_POLICY if getattr(cfg, 'frame_checkpoint', None) else FRAME_POLICY


def prediction_window(cfg):
    """Horizon (vox) of the previous-prediction crop axis, or 0 when crops use the frame model's heading."""
    if getattr(cfg, 'crop_axis', 'heading_model') != 'prediction':
        return 0
    return float(cfg.gate_horizon*cfg.future_step)


def _cone(centre, max_angle, step):
    """Unit directions within max_angle (deg) of centre on rings step (deg) apart, centre first."""
    e1 = np.cross(centre, np.eye(3)[np.argmin(np.abs(centre))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    out = [centre]
    for a in np.radians(np.arange(step, max_angle+1e-9, step)):
        n = max(6, int(np.ceil(360*np.sin(a)/step)))
        b = np.linspace(0., 2*np.pi, n, endpoint=False)
        out.append(np.cos(a)*centre+np.sin(a)*(np.cos(b)[:, None]*e1+np.sin(b)[:, None]*e2))
    return np.vstack([np.atleast_2d(o) for o in out])


def predicted_axis(future, head, window):
    """Straight axis through ``head`` minimizing the largest lateral distance of the predicted curve ``future``
    ((K, 3) world points ahead of the head, nearest first) over ``window`` vox: the prediction is cut at ``window`` vox of
    arclength from the head, or extended along its end direction to reach it; coarse cone search (75 deg, 2.5 deg rings)
    around the head-to-end chord, then a fine one (3 deg, 0.25 deg) around the best."""
    head = np.asarray(head, np.float64)
    pts = np.vstack([head[None], np.asarray(future, np.float64)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.], np.cumsum(seg)])
    if arc[-1] < 1e-6:
        return None
    if arc[-1] >= window:
        t = np.arange(1., window+1e-9, 1.)
        curve = np.stack([np.interp(t, arc, pts[:, k]) for k in range(3)], -1)
    else:
        end = pts[-1]-pts[max(0, len(pts)-4)]
        end = end/max(np.linalg.norm(end), 1e-9)
        t = np.arange(1., arc[-1]+1e-9, 1.)
        inside = np.stack([np.interp(t, arc, pts[:, k]) for k in range(3)], -1) if len(t) else np.zeros((0, 3))
        extra = pts[-1]+np.arange(1., window-arc[-1]+1e-9, 1.)[:, None]*end
        curve = np.vstack([inside, pts[-1][None], extra])
    rel = curve-head
    chord = rel[-1]/max(np.linalg.norm(rel[-1]), 1e-9)

    def best(dirs):
        along = rel @ dirs.T
        lateral = np.sqrt(np.maximum((rel**2).sum(1)[:, None]-along**2, 0.)).max(0)
        return dirs[int(np.argmin(lateral))]
    coarse = best(_cone(chord, 75., 2.5))
    return best(_cone(coarse, 3., .25))


def apply_prediction_axis(item, window):
    """Re-orient a continuation item's crop along its previous prediction's axis, keeping the current roll; labels
    follow the new frame. Items without 'prev_prediction' (seeds, restarts) are left unchanged."""
    future = item.get('prev_prediction')
    if not window or future is None or not len(future):
        return False
    axis = predicted_axis(future, item['pos'], window)
    if axis is None:
        return False
    reframe_item(item, frame_from_heading(axis, np.asarray(item['frame'])[:, 0]))
    from vesuvius.neural_tracing.fiber_follow.data.data import refresh_frame_targets
    refresh_frame_targets(item)
    item['frame_policy'] = PREDICTION_FRAME_POLICY
    return True


@lru_cache(maxsize=4)
def _load_predictor(path, digest):
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
    if digest and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        print(f'Warning: frame checkpoint {path} differs from the one recorded in the follower configuration', flush=True)
    predictor = HeadingPredictor.load(path, 'cpu')
    if not predictor.model.cfg.predict_frames:
        raise ValueError('Crop generation requires an H/V heading-and-normal model')
    predictor.model.requires_grad_(False)
    return predictor


def frame_predictor(cfg):
    from vesuvius.neural_tracing.fiber_follow.shared.paths import recorded_path
    path = getattr(cfg, 'frame_checkpoint', None)
    if not path:
        return None
    # A checkpoint trained elsewhere records that machine's path; the sha256 check still pins the file.
    return _load_predictor(str(Path(recorded_path(path)).expanduser().resolve()), getattr(cfg, 'frame_checkpoint_sha256', None))


def trace_family_kwargs(tracer, families):
    """Supply explicit families whenever the tracer uses learned frames."""
    return dict(families=families) if getattr(tracer, 'frame_predictor', None) is not None else {}


def predict_frames(predictor, vol, positions, priors, paths, families, previous=None, *, pool=None):
    results = []
    previous = previous if previous is not None else [None]*len(positions)
    for start in range(0, len(positions), 64):
        part = slice(start, start+64)
        results.extend(predictor.predict_frames(vol, positions[part], priors[part], paths[part], families[part],
                                                previous=previous[part], pool=pool))
    return results


def orient_items(items, vol, predictor=None, *, previous=None, pool=None, window=0):
    """Crop frames for items without a recorded frame policy: the frame model (or the CT frame), then for continuation
    items with a previous prediction the prediction axis over ``window`` vox (prediction_window)."""
    recorded = ['frame_policy' in item for item in items]  # recorded observation geometry stays immutable
    _orient_items(items, vol, predictor, previous=previous, pool=pool)
    for item, fixed in zip(items, recorded):
        if not fixed:
            apply_prediction_axis(item, window)
        item.pop('prev_prediction', None)  # used only to orient the crop


def _orient_items(items, vol, predictor=None, *, previous=None, pool=None):
    pending = []
    for i, item in enumerate(items):
        if 'frame_policy' in item:
            if item['frame_policy'] not in FRAME_POLICIES:
                raise ValueError('Unsupported crop frame policy; recollect replay')
            continue  # recorded observation geometry stays immutable
        if predictor is None or not item.get('fiber_family'):
            orient_item(item, vol)
        else:
            pending.append(i)
    if not pending:
        return
    selected = [items[i] for i in pending]
    frames = predict_frames(predictor, vol, [i['pos'] for i in selected], [i['frame'][:, 2] for i in selected],
        [observed_path(i) for i in selected], [fiber_family(i['fiber_family']) for i in selected],
        previous=[None if previous is None else previous[i] for i in pending], pool=pool)
    for item, frame in zip(selected, frames):
        reframe_item(item, frame)
        from vesuvius.neural_tracing.fiber_follow.data.data import refresh_frame_targets
        refresh_frame_targets(item)
        item.update(frame_policy=LEARNED_FRAME_POLICY, ct_frame_diagnostics=dict(source=3, energy=0., gap=0.))


def crop_frame_bounds(item, crop, vol, predictor=None):
    """Reads of ``vol`` for one state's crop and frame."""
    if predictor is None or item.get('frame_policy') in FRAME_POLICIES:
        yield from frame_prefetch_bounds(item, crop, vol.input_scale)
    else:
        # Learned headings can change as well as roll; enclose every possible orientation.
        yield heading_free_bounds(item['pos'], crop, vol.input_scale)
        yield heading_free_bounds(item['pos'], predictor.model.cfg.patch, vol.input_scale)
        if not item.get('fiber_family'):
            yield normal_context(item['pos'], vol.input_scale)
    if item.get('seed_heading_family') is not None:
        yield normal_context(item['seed_pos'], vol.input_scale)
