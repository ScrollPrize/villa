"""Frozen learned crop frames shared by training, collection, and tracing.

Only CT, H/V family, prior headings and observed paths enter the frame model.
CPU inference is cached per process and batched; data-loader workers never use CUDA.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.tracing.heading import (
    FRAME_POLICY, LEARNED_FRAME_POLICY, FRAME_POLICIES, fiber_family, orient_item, reframe_item,
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


@lru_cache(maxsize=4)
def _load_predictor(path, digest):
    from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
    if digest and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        print(f'Warning: frame checkpoint {path} differs from the one recorded in the follower configuration', flush=True)
    predictor = HeadingPredictor.load(path, 'cpu')
    if not predictor.model.cfg.predict_frames or predictor.model.cfg.ct_downsample_levels:
        raise ValueError('Crop generation requires a level-0 H/V heading-and-normal model')
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


def orient_items(items, vol, predictor=None, *, previous=None, pool=None):
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


def crop_frame_bounds(item, crop, vol, predictor=None, *, include_crop=True):
    """Reads of ``vol`` for one state's crop and frame. ``include_crop=False`` when the model crop itself is read
    from another CT level (the crop view prefetches it); the frame/seed contexts stay."""
    if predictor is None:
        if include_crop:
            yield from frame_prefetch_bounds(item, crop, vol.input_scale)
        elif item.get('frame_policy') not in FRAME_POLICIES:
            yield normal_context(item['pos'], vol.input_scale)  # the CT frame still reads this level
    else:
        if item.get('frame_policy') in FRAME_POLICIES:
            if include_crop:
                yield from frame_prefetch_bounds(item, crop, vol.input_scale)
        else:
            # Learned headings can change as well as roll; enclose every possible orientation.
            if include_crop:
                yield heading_free_bounds(item['pos'], crop, vol.input_scale)
            yield heading_free_bounds(item['pos'], predictor.model.cfg.patch, vol.input_scale)
            if not item.get('fiber_family'):
                yield normal_context(item['pos'], vol.input_scale)
    if item.get('seed_heading_family') is not None:
        yield normal_context(item['seed_pos'], vol.input_scale)
