"""Observed-prefix targets and sequence-normalized binary supervision."""
import numpy as np
import torch
import torch.nn.functional as F
from ..events import label_path
from ..geometry import arclength, frame_from_heading
from .judge_slices import SliceStream, footprint_allowed
from .judge_model import sequence_tensors


def masked_bce(logits, target, known, eligible):
    mask = known.bool() & eligible.bool()
    safe = torch.where(mask, target, torch.zeros_like(target))
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    loss = F.binary_cross_entropy_with_logits(safe_logits, safe, reduction='none')
    return torch.where(mask, loss, torch.zeros_like(loss)).sum(-1)/mask.sum(-1).clamp_min(1)


def make_sequence(item, reader, cfg, band=None):
    context = item.get('judge_context')
    if context is None:
        return None  # legacy offtrack alone is never a judge target
    path = np.asarray(context['path'], float)
    if band is not None and path[:, 2].min() < band.hi and path[:, 2].max() >= band.lo:
        return None
    stream = SliceStream(reader, cfg, context['seed_frame'])
    records = stream.update(path)
    if not footprint_allowed(records, cfg, band):
        return None
    events = label_path(context.get('complete_path', path), context['annotation'], context['q0'], context['physical_end'])
    if band is not None and events.samples:
        from ..geometry import interp_at
        matched = interp_at(events.annotation, events.q, np.array([s[1] for s in events.samples]))
        if matched[:, 2].min() < band.hi and matched[:, 2].max() >= band.lo:
            return None
    arcs = np.array([r['arc'] for r in records])
    target, known = events.labels(arcs)
    if context.get('construction_interval') is not None:
        lo, hi = context['construction_interval']
        known[(arcs > lo) & (arcs < hi)] = False
    audit = context.get('policy_audit')
    accepted = 0.
    if audit is not None:
        accepted = audit['previous_accepted'] if abs(arcs[-1]-audit['endpoint']) < 1e-7 else audit['accepted']
        acquired = dict(zip(audit['arcs'], audit['support']))
        for record in records:
            if acquired.get(record['arc']) is False:
                record['support'] = False
                record['images'][:, 2] = 0
    # Moving overlap is supplied only by a causally recorded policy ledger.
    query = np.array([r['query'] for r in records])
    support = np.array([r['support'] for r in records])
    candidates = [i for i,r in enumerate(records) if r['query'] and r['regular'] and r['arc'] <= accepted+1e-7]
    anchor = candidates[0] if candidates else None
    eligible = np.zeros(len(records), bool)
    if anchor is not None and support[0] and len(arcs[query]) >= 4 and np.ptp(arcs[query]) >= 12:
        eligible[anchor:] = query[anchor:] & np.logical_and.accumulate(support[anchor:])
    batch = sequence_tensors(records, anchor=arcs[anchor] if anchor is not None else 0.)
    batch.update(target=torch.as_tensor(target)[None], known=torch.as_tensor(known)[None],
                 eligible=torch.as_tensor(eligible)[None],
                 source=torch.tensor([int(context.get('source', item.get('source', 0)))]))
    return batch


def fresh_context(item, fiber, t, reverse):
    """Same augmented geometry as the follower; annotation used only for labels."""
    hist = item['pos']+item['hist_local'] @ item['frame'].T
    path = np.concatenate((hist[np.asarray(item['hmask']) > 0][::-1], item['pos'][None]))
    # Choose a real reached seed with preceding context short enough for direct eligibility.
    arc = arclength(path)
    first = int(np.searchsorted(arc, max(0., arc[-1]-120.)))
    path = path[first:]
    annotation = fiber.points[::-1] if reverse else fiber.points
    q0 = max(0., t-(len(hist[np.asarray(item['hmask']) > 0])-first))
    heading = path[1]-path[0] if len(path) > 1 else item['frame'][:, 2]
    return dict(path=path, annotation=annotation, q0=q0,
                physical_end=fiber.endpoint_stop[0 if reverse else 1], seed_frame=frame_from_heading(heading))


class JointObservationBuilder:
    def __init__(self, follower, slices, band=None, synthetic_fraction=.25, fibers=()):
        self.follower, self.slices, self.band = follower, slices, band
        self.synthetic_fraction, self.fibers = synthetic_fraction, fibers
        self.reader = None
        self.extra_accumulator = 0.
        self.rng = np.random.default_rng(0)
        self.contact_cache = {}

    def __getstate__(self):
        return dict(self.__dict__, reader=None, contact_cache={})

    def __call__(self, items, vol):
        if self.reader is None:
            self.reader = self.slices.open()
        batch = self.follower(items, vol)
        sequences = [make_sequence(i, self.reader, self.slices, self.band) for i in items]
        sequences = [s for s in sequences if s is not None]
        attempted = fallback = 0
        self.extra_accumulator += len(items)*self.synthetic_fraction
        while self.extra_accumulator >= 1:
            self.extra_accumulator -= 1
            attempted += 1
            context = synthetic_contact(self.fibers, self.rng, cache=self.contact_cache)
            if context is not None:
                extra = make_sequence(dict(judge_context=context), self.reader, self.slices, self.band)
                if extra is not None:
                    sequences.append(extra)
                else:
                    fallback += 1
            else:
                fallback += 1
        batch['judge'] = sequences
        for sequence in sequences:
            gain = self.rng.uniform(1-self.slices.photometric_gain, 1+self.slices.photometric_gain)
            bias = self.rng.uniform(-self.slices.photometric_bias, self.slices.photometric_bias)
            # One transform for the entire observed sequence, never an event-dependent seam.
            sequence['images'][:, :, :, 0].mul_(gain).add_(bias).clamp_(0, 1)
        batch['judge_allocation'] = torch.tensor([len(items), attempted, fallback])
        return batch


def synthetic_contact(fibers, rng, attempts=32, cache=None):
    """Construct real-geometry contact switches; reject duplicate/ambiguous pairs.

    ``cache`` (a dict owned by the caller for these fibers) keeps per-fiber
    bounds and KD-trees between calls; results do not depend on it.
    """
    if len(fibers) < 2:
        return None
    from scipy.spatial import cKDTree
    from ..geometry import interp_at
    cache = {} if cache is None else cache
    def bounds(i):
        if ('bounds', i) not in cache:
            cache['bounds', i] = fibers[i].points.min(0), fibers[i].points.max(0)
        return cache['bounds', i]
    for _ in range(attempts):
        ia, ib = rng.choice(len(fibers), 2, replace=False)
        a, b = fibers[ia], fibers[ib]
        if a.name == b.name or a.source_hash and a.source_hash == b.source_hash:
            continue
        (alo, ahi), (blo, bhi) = bounds(ia), bounds(ib)
        # Fibers whose boxes are 6 apart have no contact below 6 (small float margin).
        if np.linalg.norm(np.maximum(0, np.maximum(blo-ahi, alo-bhi))) >= 6.+1e-6:
            continue
        if ('tree', ib) not in cache:
            cache['tree', ib] = cKDTree(b.points)
        # Only distances below 6 are used; farther points report inf, never a contact.
        dist, idx = cache['tree', ib].query(a.points, distance_upper_bound=6.)
        contacts = np.flatnonzero((dist > 1.) & (dist < 6.) & (a.s > 24) & (a.s < a.length-24))
        # Large overlap is likely duplicate annotation, not a confirmed second identity.
        if not len(contacts) or np.mean(dist < 1.) > .2:
            continue
        i = int(rng.choice(contacts)); j = int(idx[i])
        direction = 1 if np.dot(a.points[min(i+1,len(a.points)-1)]-a.points[i-1],
                               b.points[min(j+1,len(b.points)-1)]-b.points[max(0,j-1)]) >= 0 else -1
        age = float(rng.uniform(8, 40))
        if not 0 <= b.s[j]+direction*age <= b.length:
            continue
        before = interp_at(a.points, a.s, np.arange(max(0, a.s[i]-48), a.s[i], 1.))
        after = interp_at(b.points, b.s, b.s[j]+direction*np.arange(0, age, 1.))
        # Cubic join with observed tangents; sample real CT only after constructing geometry.
        u = np.linspace(0,1,9)[1:-1,None]
        chord = max(1., float(np.linalg.norm(after[0]-before[-1])))
        ta, tb = before[-1]-before[-2], after[1]-after[0]
        ta *= chord/max(np.linalg.norm(ta), 1e-9)
        tb *= chord/max(np.linalg.norm(tb), 1e-9)
        bridge = (2*u**3-3*u**2+1)*before[-1]+(u**3-2*u**2+u)*ta+(-2*u**3+3*u**2)*after[0]+(u**3-u**2)*tb
        path = np.concatenate((before, bridge, after))
        context = dict(path=path, annotation=a.points, q0=max(0., a.s[i]-48), source=3,
                       physical_end=a.endpoint_stop[1], seed_frame=frame_from_heading(before[1]-before[0]),
                       construction_interval=(float(arclength(before)[-1]), float(arclength(path[:len(before)+len(bridge)+1])[-1])))
        event = label_path(path, a.points, context['q0'], a.endpoint_stop[1])
        if event.onset is not None and event.kind == 'departure':
            # Matched positive at the same contact, with similar prefix and age.
            if rng.random() < .5:
                q = np.arange(context['q0'], min(a.length,a.s[i]+age), 1.)
                context.update(path=interp_at(a.points,a.s,q), source=4)
                context.pop('construction_interval')
            return context
    return None
