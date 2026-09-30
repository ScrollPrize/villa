"""Matched choices and departures distinguished by earlier observed fibers."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.regression.data import LOCATION_SOURCES

from vesuvius.neural_tracing.fiber_follow.shared.data import continuation_targets, label_state
from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    arclength, interp_at, tangent_at, frame_from_heading, random_rotation_about, normalize,
)
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import bank_fiber
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import exact_nearest


CANDIDATE_COUNT = 4


def path_candidates(curves, supported, rng):
    """Two continuations and two smooth switches with identical paired inputs.

    Switch onset varies over the forecast. Labels are deliberately not assigned
    here: dense geometry, foreign masks and censoring determine the first failure.
    """
    count = curves.shape[1]
    mixed, masks = [], []
    for origin in (0, 1):
        if count < 4:
            mixed.append(curves[origin].copy())
            masks.append(np.zeros(count, bool))
            continue
        start = int(rng.integers(1, count-2))  # this many endpoints remain correct
        width = min(int(rng.integers(2, 5)), count-start)
        phase = np.clip((np.arange(count)-start+1)/width, 0., 1.)
        blend = phase**3*(10+phase*(-15+6*phase))
        mixed.append(curves[origin]*(1-blend[:, None])+curves[1-origin]*blend[:, None])
        # Both source curves must be supported once mixing begins.
        masks.append(supported[origin] & ((blend == 0) | supported[1-origin]))
    order = rng.permutation(CANDIDATE_COUNT)
    return (np.concatenate((curves, mixed)).astype(np.float32)[order],
            np.minimum.accumulate(np.concatenate((supported, masks)), axis=-1)[order],
            np.array([0, 0, 1, 1], np.int64)[order])


def decision_pair(bank, sample, model, rng, *, attempts=32, choice=None, hard_fraction=0.):
    """Return two states sharing crops, candidate order and observed history.

    Choice pairs share a short uncertain path between nearby fibers; either
    original is recoverable and supplies its own geometry target. Departure
    pairs share a real B tail: B's seed permits following, A's seed requires
    stopping. References precede the shared history and are actual seed
    observations of the simulated paths, not replacements of contaminated
    recent observations. Only certified bank relationships are used.
    """
    choice = bool(rng.integers(2)) if choice is None else choice
    tails = [v for v in (4.,8.,12.) if v+8 < model.fine.behind*model.fine.spacing]
    if not tails:
        return None
    tail = float(rng.choice(tails))
    reference_gap = (max(8.,model.fine.behind*model.fine.spacing+
                         17*model.fine.spacing/2+2.))
    horizon = sample.future_s[-1]
    for _ in range(attempts):
        draw = bank.draw_path(rng, unique=False, min_length=tail+horizon+12, hard_fraction=hard_fraction)
        if draw is None:
            continue
        fi, line, arc_range = draw
        parent = bank.fibers[fi]
        lo = max(0, int(np.searchsorted(parent.s, arc_range[0]))-2)
        hi = min(len(parent.s), int(np.searchsorted(parent.s, arc_range[1]))+3)
        if hi-lo < 2:
            continue
        separation, _, segment, u = exact_nearest(line, parent.points[lo:hi])
        matched = parent.s[lo+segment]+u*np.diff(parent.s[lo:hi])[segment]
        if matched[-1] < matched[0]:
            line, matched, separation = line[::-1], matched[::-1], separation[::-1]
        if np.any(np.diff(matched) < -1e-5):
            continue
        reverse = bool(rng.integers(2))
        if reverse:
            line, matched, separation = line[::-1], matched[::-1], separation[::-1]
        s = arclength(line)
        heads = np.flatnonzero((s >= tail+reference_gap) & (s <= s[-1]-horizon-2))
        if choice:
            heads = heads[separation[heads] <= min(6., 2*(model.max_recovery_distance-1.))]
        if not len(heads):
            continue
        head = float(s[int(rng.choice(heads))])
        own_t = float(np.interp(head, s, matched))
        neighbor = bank_fiber(parent, line)
        bpos = interp_at(line, s, np.array([head]))[0]
        apos = interp_at(parent.points, parent.s, np.array([own_t]))[0]
        heading = tangent_at(line, s, head)
        if choice:
            heading = normalize(heading+tangent_at(parent.points, parent.s, own_t)*(-1 if reverse else 1))
        frame = random_rotation_about(frame_from_heading(heading), rng.uniform(0, 2*np.pi))
        pos = (apos+bpos)/2 if choice else bpos
        back = head-np.arange(1, sample.n_history+1)*sample.history_step
        mask = (np.arange(1, sample.n_history+1)*sample.history_step <= tail).astype(np.float32)
        history = interp_at(line, s, np.clip(back, 0, s[-1]))
        if choice:
            original = interp_at(parent.points, parent.s, np.interp(np.clip(back, 0, s[-1]), s, matched))
            history = (history+original)/2
        ta = parent.length-own_t if reverse else own_t
        targets = [continuation_targets(parent, ta, reverse, pos, frame, sample),
                   continuation_targets(neighbor, head, False, pos, frame, sample)]
        curves = np.stack([np.c_[t['plane_ab'], sample.future_s] for t in targets]).astype(np.float32)
        # Candidate evidence must fit within the appearance crop. Unknown
        # prefixes and invalid connections are never positive training labels.
        supported = np.stack([t['plane_mask'] for t in targets]).astype(bool)
        supported &= np.abs(curves[..., :2]).max(-1) <= model.lateral_limit
        supported &= np.isfinite(curves).all(-1)
        supported = np.minimum.accumulate(supported, axis=-1)
        allowed = np.linalg.norm(curves[:, 0], axis=-1) <= model.max_recovery_distance
        supported &= allowed[:, None]
        if not supported[1].all() or (choice and not supported[0].all()):
            continue
        # Recheck the exact interpolated B history against the original target.
        if not choice and not bank.clear_of_target(fi, np.concatenate((history[mask > 0], bpos[None]))).all():
            continue
        curves, supported, candidate_kind = path_candidates(curves, supported, rng)
        reference_b = head-tail-reference_gap
        reference_a = float(np.interp(reference_b, s, matched))
        refs = [interp_at(parent.points, parent.s, np.array([reference_a]))[0],
                interp_at(line, s, np.array([reference_b]))[0]]
        tangents = [tangent_at(parent.points, parent.s, reference_a)*(-1 if reverse else 1),
                    tangent_at(line, s, reference_b)]
        from vesuvius.neural_tracing.fiber_follow.regression.data import visible_points
        seed_visible = visible_points((np.asarray(refs)-pos) @ frame,model.fine)
        if seed_visible.any():
            continue
        rows = []
        pair_observation_seed = int(rng.integers(2**63))
        for target in range(2):
            offtrack = not choice and target == 0
            fiber, t, rev = (parent, own_t, reverse) if target == 0 else (neighbor, head, False)
            row = label_state(fiber, pos.copy(), frame.copy(), history.copy(), mask.copy(), sample,
                              t=t, reverse=rev, offtrack=offtrack)
            row['_seed_original_certified'] = True
            row.update(fiber_ref=(fi, ta if target == 0 else head, rev), source=5, source_step=-1,
                       stratum=4 if offtrack else -1, location_source=LOCATION_SOURCES.index('decision_pair'),
                       seed_pos=refs[target], seed_tangent=tangents[target], seed_valid=True,
                       seed_age=float(arclength(np.concatenate((refs[target][None], history[mask > 0][::-1], pos[None])))[-1]),
                       candidate_points=curves.copy(), candidate_mask=supported.copy(),
                       candidate_kind=candidate_kind.copy(),
                       decision_kind=1 if choice else (2 if offtrack else 3),
                       decision_tail=tail, bank_tail_length=tail if offtrack else 0.)
            if target == 1:
                row.update(supervision_fiber=neighbor, bank_parent_arc_range=arc_range)
            row['pair_observation_seed'] = pair_observation_seed
            # Observed synthetic paths differ before the shared local tail.
            # Labels are kept in memory_track, never in image/model inputs.
            prefix_s = np.arange(reference_b,head-tail,.5)
            prefix_b = interp_at(line,s,prefix_s)
            prefix_a = interp_at(parent.points,parent.s,np.interp(prefix_s,s,matched))
            prefix = prefix_a if target == 0 else prefix_b
            common = (prefix_a+prefix_b)/2 if choice else prefix_b
            blend = np.clip((prefix_s-(head-tail-8))/8,0,1)
            blend = blend*blend*(3-2*blend)
            prefix = prefix*(1-blend[:,None])+common*blend[:,None]
            observed = np.concatenate((prefix,history[mask > 0][::-1],pos[None]))
            observed_arc = arclength(observed)
            arcs = np.arange(0,observed_arc[-1],model.memory_stride)
            track_pos = interp_at(observed,observed_arc,arcs)
            behind = interp_at(observed,observed_arc,np.maximum(0,arcs-.5))
            heading0 = tangents[target]
            track_frame = np.stack([frame_from_heading(normalize(p-q) if np.linalg.norm(p-q)>1e-6 else heading0)
                                    for p,q in zip(track_pos,behind)])
            # Only certain prefix and shared-tail membership is supervised;
            # the interpolated bridge remains unknown.
            pure_index = min(len(prefix_s)-1,int(np.searchsorted(prefix_s,head-tail-8.)))
            pure = observed_arc[max(0,pure_index)]
            on_tail = arcs >= observed_arc[-1]-tail+1e-5
            off = np.where(arcs < pure,0.,np.nan)
            if not choice:
                off[on_tail] = float(target == 0)
            row['memory_track'] = dict(pos=track_pos,frame=track_frame,offtrack=off,
                offset=np.where((off == 0)[:,None],np.zeros((len(off),3)),np.nan))
            row['seed_age'] = float(observed_arc[-1])
            rows.append(row)
        if rng.integers(2):
            rows.reverse()
        return rows
    return None
