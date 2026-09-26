"""Deterministic contact indexing of equally validated controlled fiber spans."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from ..data import (fiber_manifest, make_sample, plane_targets, continuation_targets,
                    training_state_allowed)
from ..geometry import interp_at, tangent_at, frame_from_heading
from .data import ObservationBuilder


class ContactIndex:
    """Index nearby *different* validated curves; never infer identity from CT.

    A contact episode is a contiguous run of proximity samples, not a collection
    of independent augmented views. Cached indices are tied to geometry/settings.
    """
    def __init__(self, fibers, band=None, path=None, spacing=4., radius=6.):
        self.fibers, self.band = fibers, band
        self.by_name = {f.name: i for i, f in enumerate(fibers)}
        self.trees = {}
        identity = dict(version=1, fibers=fiber_manifest(fibers), spacing=spacing, radius=radius,
                        band=asdict(band) if band is not None else None)
        self.sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if path is not None and Path(path).exists():
            data = json.loads(Path(path).read_text())
            if data['sha256'] != self.sha256:
                raise ValueError('Contact index geometry or settings changed')
            self.episodes = data['episodes']
        else:
            points, identities, arcs = [], [], []
            for i, f in enumerate(fibers):
                s = np.arange(0, f.length, spacing)
                p = interp_at(f.points, f.s, s)
                eligible = np.ones(len(s), bool) if band is None else ~band.contains(p[:, 2])
                points.append(p[eligible]); arcs.append(s[eligible])
                identities.append(np.full(eligible.sum(), i, np.int32))
            points, ids, arcs = np.concatenate(points), np.concatenate(identities), np.concatenate(arcs)
            tree = cKDTree(points)
            pairs = {}
            for start in range(0, len(points), 16384):
                stop = min(len(points), start+16384)
                distances, neighbors = tree.query(points[start:stop], k=min(16, len(points)), distance_upper_bound=radius)
                for row in range(stop-start):
                    a = int(ids[start+row])
                    seen = set()
                    for d, index in zip(distances[row], neighbors[row]):
                        if not np.isfinite(d):
                            continue
                        b = int(ids[index])
                        if a >= b or b in seen or fibers[a].source_hash and fibers[a].source_hash == fibers[b].source_hash:
                            continue
                        seen.add(b)
                        pairs.setdefault((a, b), []).append((float(arcs[start+row]), float(arcs[index]), float(d)))
            self.episodes = []
            for (a, b), rows in sorted(pairs.items()):
                rows.sort()
                # Exclude near-identical overlapping annotations as competing
                # identities; this changes contact sampling, never target quality.
                if len(rows)*spacing > .5*min(fibers[a].length, fibers[b].length) and np.mean(np.array(rows)[:, 2] < .5) > .8:
                    continue
                groups = [[]]
                for row in rows:
                    if groups[-1] and (row[0]-groups[-1][-1][0] > 3*spacing or abs(row[1]-groups[-1][-1][1]) > 6*spacing):
                        groups.append([])
                    groups[-1].append(row)
                for group in groups:
                    best = min(group, key=lambda r: r[2])
                    self.episodes.append(dict(a=a, b=b, a0=group[0][0], a1=group[-1][0],
                        b0=min(r[1] for r in group), b1=max(r[1] for r in group),
                        sa=best[0], sb=best[1], distance=best[2]))
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(dict(sha256=self.sha256, episodes=self.episodes), indent=2))
        self.by_fiber = {i: [] for i in range(len(fibers))}
        for episode in self.episodes:
            self.by_fiber[episode['a']].append(episode)
            self.by_fiber[episode['b']].append(episode)

    def nearest(self, fiber_index, point):
        if fiber_index not in self.trees:
            self.trees[fiber_index] = cKDTree(self.fibers[fiber_index].points)
        distance, index = self.trees[fiber_index].query(point)
        return float(self.fibers[fiber_index].s[index]), float(distance)

    def summary(self, horizon):
        lengths = np.array([max(e['a1']-e['a0'], e['b1']-e['b0'])+24 for e in self.episodes])
        return dict(episodes=len(lengths), sha256=self.sha256, proximity_spacing=4., proximity_radius=6.,
                    passage_length_quantiles=np.quantile(lengths, [.25, .5, .75, .9, .95]).tolist() if len(lengths) else [],
                    nominal_horizon_fraction=float(np.mean(lengths <= horizon)) if len(lengths) else 0.)


class SpatialObservationBuilder(ObservationBuilder):
    def __init__(self, cfg, sample, fibers, band=None, contacts=None, contact_fraction=.35):
        super().__init__(cfg)
        self.sample, self.fibers, self.band = sample, fibers, band
        self.contacts = contacts or ContactIndex(fibers, band)
        self.contact_fraction = contact_fraction
        self.contact_sample = replace(sample, no_history_prob=0., short_history_prob=0.)
        self.sampling_counts = dict(attempts=0, horizon=0, geometry=0, footprint=0, accepted=0)

    def _allowed(self, item):
        if not all(training_state_allowed(item, crop, self.band) for crop in (self.cfg.fine, self.cfg.coarse)):
            return False
        if item.get('seed_valid', False):
            seed = dict(pos=item['seed_pos'], frame=item['seed_frame'])
            if not training_state_allowed(seed, self.cfg.seed_crop, self.band):
                return False
        return True

    def prepare_sample(self, item, fiber, original_t, reverse, rng, replay=None, replay_row=None):
        item = dict(item)
        item['frontier'] = np.asarray(item.get('frontier', np.zeros(3)), np.float32)
        item['frontier_direction'] = np.asarray(item.get('frontier_direction', [0., 0., 1.]), np.float32)
        item['contact'] = float(item.get('contact', 0))
        item['seed_valid'] = False
        item['seed_pos'], item['seed_frame'] = item['pos'].copy(), item['frame'].copy()
        if replay is not None:
            # The archive's first point is the actual externally supplied seed.
            # Legacy fixed banks without this provenance get a missing-seed mask.
            if replay.provenance.get('judge_archive') and replay.judge_trace[replay_row] >= 0:
                from .judge_archive import PathArchive
                trace = PathArchive.from_replay(replay).traces[int(replay.judge_trace[replay_row])]
                item.update(seed_pos=trace['path'][0].copy(), seed_frame=trace['seed_frame'].copy(), seed_valid=True)
        else:
            available = fiber.length-original_t if reverse else original_t
            if item['hmask'].sum() == 0:
                item.update(seed_pos=item['pos']+item['frontier'] @ item['frame'].T,
                            seed_frame=frame_from_heading(item['frame'] @ item['frontier_direction']), seed_valid=True)
            else:
                distance = min(available, float(rng.uniform(16., 256.)))
                seed_t = original_t+distance if reverse else original_t-distance
                item.update(seed_pos=interp_at(fiber.points, fiber.s, [seed_t])[0],
                            seed_frame=frame_from_heading(tangent_at(fiber.points, fiber.s, seed_t)*(-1 if reverse else 1)),
                            seed_valid=True)
        # Label geometry only. Neighbor identities/features never enter x.
        fi = self.contacts.by_name[fiber.name]
        nearby = []
        horizon = self.cfg.n_future*self.cfg.future_step
        for e in self.contacts.by_fiber[fi]:
            lo, hi = (e['a0'], e['a1']) if e['a'] == fi else (e['b0'], e['b1'])
            if lo-horizon-8 <= original_t <= hi+horizon+8:
                other = e['b'] if e['a'] == fi else e['a']
                if other not in nearby:
                    nearby.append(other)
        q = len(item['dense_ab'])
        item['neighbor_ab'] = np.zeros((2, q, 2), np.float32)
        item['neighbor_mask'] = np.zeros((2, q), np.float32)
        dense_planes = np.linspace(self.cfg.future_step, horizon, q)+item['frontier'][2]
        footprints = []
        for j, other in enumerate(nearby[:2]):
            f = self.fibers[other]
            t, _ = self.contacts.nearest(other, item['pos']+item['frontier'] @ item['frame'].T)
            rev = np.dot(tangent_at(f.points, f.s, t), item['frame'][:, 2]) < 0
            p, s = (f.points[::-1], f.length-f.s[::-1]) if rev else (f.points, f.s)
            t = f.length-t if rev else t
            ab, mask = plane_targets(p, s, max(0., t-8), s[-1], item['pos'], item['frame'], dense_planes, True)
            item['neighbor_ab'][j], item['neighbor_mask'][j] = ab, mask
            footprints.append(np.c_[ab, dense_planes][mask > 0] @ item['frame'].T+item['pos'])
        if footprints:
            item['extra_world'] = np.concatenate(footprints)
        if not self._allowed(item):
            # Dataset state_allowed will reject this item before reading CT.
            item['spatial_excluded'] = True
        return item

    def contact_batch(self, rng, count):
        if not self.contacts.episodes or rng.random() >= self.contact_fraction:
            return []
        result = []
        while len(result) < count:
            pair = self._contact_pair(rng, min(2, count-len(result)))
            if not pair:
                break
            result.extend(pair)
        return result

    def _contact_pair(self, rng, count):
        for _ in range(64):
            self.sampling_counts['attempts'] += 1
            episode_id = int(rng.integers(len(self.contacts.episodes)))
            episode = self.contacts.episodes[episode_id]
            reverse = bool(rng.integers(2))
            a, b = self.fibers[episode['a']], self.fibers[episode['b']]
            approach = float(rng.uniform(8., 20.))
            if max(episode['a1']-episode['a0'], episode['b1']-episode['b0'])+approach+8 > self.cfg.n_future*self.cfg.future_step:
                self.sampling_counts['horizon'] += 1
                continue
            ta = episode['a1']+approach if reverse else episode['a0']-approach
            if not 8 < ta < a.length-8:
                continue
            first = make_sample(a, a.length-ta if reverse else ta, reverse, self.contact_sample, rng)
            first.update(source=0, source_step=-1, stratum=-1, contact=1., contact_reverse=reverse, contact_episode=episode_id)
            first = self.prepare_sample(first, a, ta, reverse, rng)
            items = [first]
            if count > 1:
                tb, distance = self.contacts.nearest(episode['b'], first['pos'])
                revb = np.dot(tangent_at(b.points, b.s, tb), first['frame'][:, 2]) < 0
                second = make_sample(b, b.length-tb if revb else tb, revb, self.contact_sample, rng)
                # Both prompts see identical CT coordinates. B's true frontier
                # and history are explicitly expressed in A's crop frame.
                world_hist = second['hist_local'] @ second['frame'].T+second['pos']
                frontier = (second['pos']-first['pos']) @ first['frame']
                direction = first['frame'].T @ second['frame'][:, 2]
                second.update(pos=first['pos'].copy(), frame=first['frame'].copy(), frontier=frontier,
                              frontier_direction=direction, hist_local=(world_hist-first['pos']) @ first['frame'],
                              source=0, source_step=-1, stratum=-1, contact=1., contact_reverse=reverse, contact_episode=episode_id)
                # Targets use common-frame planes displaced by B's frontier z.
                targets = continuation_targets(b, b.length-tb if revb else tb, revb,
                    first['pos']+first['frame'][:, 2]*frontier[2], first['frame'], self.sample)
                for key in ('gt_history', 'fut_local', 'end_local'):
                    targets[key][..., 2] += frontier[2]
                targets['planes'] = targets['planes']+frontier[2]
                targets['dense_planes'] = targets['dense_planes']+frontier[2]
                second.update(targets)
                second = self.prepare_sample(second, b, tb, revb, rng)
                if distance > self.cfg.lateral_limit/2 or np.linalg.norm(frontier[:2]) > self.cfg.lateral_limit-3:
                    self.sampling_counts['geometry'] += 1
                    continue
                items.append(second)
            # Complete-contact examples require a representable, fully covered
            # passage in both prompts' shared frame. Ordinary/replay examples
            # still include partial/censored geometry with their normal masks.
            def complete(item):
                z = item['frontier'][2]+self.sample.future_s
                return (item['plane_mask'].all() and item['dense_mask'].all()
                        and np.abs(item['dense_ab']).max() <= self.cfg.lateral_limit
                        and z[-1] <= (self.cfg.fine.depth-self.cfg.fine.behind-1)*self.cfg.fine.spacing
                        and np.linalg.norm(np.diff(item['plane_ab'], axis=0), axis=-1).max() <= self.cfg.lateral_step)
            if not all(complete(i) for i in items):
                self.sampling_counts['geometry'] += 1
                continue
            if all(not i.get('spatial_excluded', False) for i in items):
                self.sampling_counts['accepted'] += 1
                return items[:count]
            self.sampling_counts['footprint'] += 1
        return []

    def __call__(self, items, vol):
        # Diagnostic callers can provide already-labeled states without fiber IDs.
        batch = super().__call__(items, vol)
        for key in ('neighbor_ab', 'neighbor_mask', 'contact'):
            batch[key] = torch.as_tensor(np.stack([i[key] for i in items]), dtype=torch.float32)
        batch['contact_episode'] = torch.tensor([i.get('contact_episode', -1) for i in items])
        return batch


def contact_monitor(cfg, sample, fibers, band, path, count=16):
    """Frozen monitor-only contacts, with every evidence footprint inside holdout."""
    from ..data import crop_corners
    index = ContactIndex(fibers, path=path)
    builder = SpatialObservationBuilder(cfg, sample, fibers, contacts=index, contact_fraction=1.)
    rng = np.random.default_rng(20260926)
    items, seen = [], set()
    def contained(points):
        return not len(points) or (np.min(points[:, 2]) >= band.lo+2 and np.max(points[:, 2]) < band.hi-2)
    for _ in range(128):
        pair = builder.contact_batch(rng, 2)
        valid = bool(pair)
        identity = (pair[0]['contact_episode'], pair[0]['contact_reverse']) if pair else None
        valid &= identity not in seen
        for item in pair:
            for crop in (cfg.fine, cfg.coarse):
                valid &= contained(crop_corners(crop) @ item['frame'].T+item['pos'])
            valid &= contained(item['hist_local'][item['hmask'] > 0] @ item['frame'].T+item['pos'])
            valid &= contained(item.get('extra_world', np.empty((0, 3))))
            if item['seed_valid']:
                valid &= contained(crop_corners(cfg.seed_crop) @ item['seed_frame'].T+item['seed_pos'])
        if valid:
            items.extend(pair)
            seen.add(identity)
        if len(items) >= count:
            break
    return builder, items[:count]
