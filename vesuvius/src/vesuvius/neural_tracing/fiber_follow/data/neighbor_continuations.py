"""Synthetic wrong-fiber history: noised original prefix (trace_noise), smooth bridge, traced tail."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.neighbor_mining import exact_nearest
from vesuvius.neural_tracing.fiber_follow.data.data import SOURCE, label_state, trace_noise
from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    arclength, interp_at, frame_from_heading, normalize,
)
from vesuvius.neural_tracing.fiber_follow.tracing.heading import trace_heading
from vesuvius.neural_tracing.fiber_follow.data.state_labels import constructed_facts


def validate_tail_range(lengths):
    if len(lengths) != 2 or not np.isfinite(lengths).all() or not 0 < lengths[0] <= lengths[1]:
        raise ValueError('Wrong-continuation tail lengths must be finite, positive and ordered')
    return tuple(float(v) for v in lengths)


def wrong_continuation(bank, cfg, rng, *, tail_length_range=(4., 16.), prefix_length=0.):
    """No tracing or volume I/O; abstain if a safe connected history cannot fit.

    The head and recent tail lie on a verified neighbor: a certified committed switch,
    labeled terminal by the shared state contract. The artificial bridge is observed
    history, never a positive future target. The original-fiber prefix carries the
    simulated tracing error of fresh traces (zero at its annotated seed), tapering to
    zero across the bridge so the certified tail stays exact. An undersized path is
    rejected, rather than shortening the requested tail. ``prefix_length`` extends the
    original-fiber prefix (and moves the seed back).
    """
    tail_length_range = validate_tail_range(tail_length_range)
    requested_tail = float(rng.uniform(*tail_length_range))
    minimum = requested_tail+max(16.,2*bank.run['mining'].get('min_distance',0.))+4
    draw = bank.draw_path(rng,min_length=minimum,unique=False)
    if draw is None:
        return None
    fi,line,arc_range = draw
    fiber = bank.fibers[fi]
    # Limit projection to the already validated target correspondence window.
    lo = max(0,int(np.searchsorted(fiber.s,arc_range[0]))-2)
    hi = min(len(fiber.s),int(np.searchsorted(fiber.s,arc_range[1]))+3)
    target = fiber.points[lo:hi]
    if len(target) < 2:
        return None
    separation,_,segment,u = exact_nearest(line,target)
    matched = fiber.s[lo+segment]+u*np.diff(fiber.s[lo:hi])[segment]
    if matched[-1] < matched[0]:
        line,matched = line[::-1],matched[::-1]
    if np.any(np.diff(matched) < -1e-5):
        return None
    reverse = bool(rng.integers(2))
    if reverse:
        line,matched = line[::-1],matched[::-1]
    own_points = fiber.points[::-1] if reverse else fiber.points
    own_arc = fiber.length-fiber.s[::-1] if reverse else fiber.s
    matched = fiber.length-matched if reverse else matched
    s = arclength(line)
    bridge_length = float(rng.uniform(16.,24.))
    # Preserve the established bridge for nearby paths. Outer-band neighbors
    # need more forward travel to make a smooth lateral transition.
    if separation.max() > 12.:
        bridge_length = max(bridge_length,2*float(separation.max()))
    tail_length = requested_tail
    if s[-1] < bridge_length+tail_length+4:
        return None
    start = float(rng.uniform(0.,s[-1]-bridge_length-tail_length-4))
    finish,head = start+bridge_length,start+bridge_length+tail_length
    # Preserve the exact bridge/tail join when lengths are not multiples of .25.
    samples = np.unique(np.r_[np.arange(start,head,.25),finish,head])
    own_t = np.interp(samples,s,matched)
    own = interp_at(own_points,own_arc,own_t)
    neighbor = interp_at(line,s,samples)
    phase = np.clip((samples-start)/bridge_length,0,1)
    weight = phase**3*(10+phase*(-15+6*phase))  # zero first/second derivatives at both joins
    transition = own*(1-weight[:,None])+neighbor*weight[:,None]
    prefix_t = np.arange(max(0.,own_t[0]-max(cfg.n_history*cfg.history_step+4,prefix_length)),own_t[0],.25)
    path = np.concatenate((interp_at(own_points,own_arc,prefix_t),transition))
    # Tracing error on the original prefix at unit arclength, removed along the bridge.
    distance = arclength(path)
    units = np.arange(0., distance[-1]+1.)
    own_units = np.interp(units, distance, np.r_[prefix_t, own_t])
    noise, sigma = trace_noise(own_units, own_points, own_arc, cfg, rng)
    keep = 1-np.r_[np.zeros(len(prefix_t)), weight]
    path = path+np.stack([np.interp(distance, units, noise[:, k]) for k in range(3)], -1)*keep[:, None]
    # A clipped prefix/bridge join can repeat the first point (notably when
    # a reversed AFV path's catalog length differs by roundoff). Preserve the
    # join indices below, but derive the seed heading from actual movement.
    moving = np.flatnonzero(np.linalg.norm(path-path[0],axis=1) > 1e-6)
    if not len(moving):
        return None
    seed_tangent = normalize(path[moving[0]]-path[0])
    distance = arclength(path)
    back = distance[-1]-np.arange(1,cfg.n_history+1)*cfg.history_step
    mask = (back >= 0).astype(np.float32)
    history = interp_at(path,distance,np.clip(back,0,distance[-1]))
    pos = path[-1]
    # The head and its recent tail must still be on the validated wrong fiber.
    tail = interp_at(line,s,np.arange(finish,head+1e-9,.25))
    if not bank.clear_of_target(fi,np.concatenate((tail,pos[None]))).all():
        return None
    chord = pos-interp_at(path,distance,np.array([distance[-1]-2]))[0]
    if np.linalg.norm(chord) < 1e-6:
        return None
    # The tracer's heading for this observed path (its trusted 12-voxel fit).
    frame = frame_from_heading(trace_heading(path,0,chord))
    original_t = fiber.length-own_t[-1] if reverse else own_t[-1]
    trace = constructed_facts(fiber,float(own_t[-1]),reverse,pos,cfg,switched=True)
    item = label_state(fiber,pos,frame,history,mask,cfg,t=original_t,reverse=reverse,trace=trace)
    item.update(fiber_ref=(fi,float(own_t[-1]),reverse),source=SOURCE['synthetic'],source_step=-1,
                bank_transition_length=bridge_length,bank_tail_length=tail_length,trace_noise_sigma=sigma,
                bank_prefix_end_t=float(own_t[0]), seed_pos=path[0].copy(),
                seed_tangent=seed_tangent,seed_age=float(distance[-1]),seed_valid=True,
                seed_heading_family=fiber.tag,heading_start=0,travelled=float(distance[-1]))
    # Certification metadata only: never enters the model.
    item.update(_seed_original_certified=True, _constructed_path=path,
                _constructed_arc=distance, _leave_arc=distance[len(prefix_t)],
                _reach_arc=distance[len(prefix_t)+int(np.searchsorted(samples,finish))])
    item['observed_path'] = path
    return item
