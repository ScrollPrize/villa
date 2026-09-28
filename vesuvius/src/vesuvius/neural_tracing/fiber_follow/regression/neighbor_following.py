"""Ordinary following targets from a live bank's training-eligible paths."""
import hashlib
import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength


def bank_fiber(parent, points):
    """A validated bank path is a separate, conservatively censored target."""
    identity = hashlib.sha256(points.tobytes()).hexdigest()
    return TracedFiber('bank:'+identity,points,arclength(points),parent.tag,
                       endpoint_stop=(False,False),source_hash=identity)


def following_sample(bank, cfg, rng):
    draw = bank.draw_path(rng)
    if draw is None:
        return None
    fi, points, arc_range = draw
    s = arclength(points)
    end = s[-1]-cfg.future_s[-1]-2.
    if end <= 8.:
        return None
    parent = bank.fibers[fi]
    fiber = bank_fiber(parent, points)
    reverse = bool(rng.integers(2))
    t = float(rng.uniform(8.,end))
    item = make_sample(fiber,t,reverse,cfg,rng)
    item.update(fiber_ref=(fi,t,reverse),source=4,source_step=-1,stratum=-1,location_source=4,
                supervision_fiber=fiber,bank_parent_arc_range=arc_range)
    return item
