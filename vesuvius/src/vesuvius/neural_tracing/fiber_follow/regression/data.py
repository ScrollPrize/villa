"""Shared training/rollout observation builder for the direct follower."""
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.components import (
    PAIR_SAMPLING_VERSION, ComponentRule, sample_pairs,
)
from vesuvius.neural_tracing.fiber_follow.shared.data import collate_targets, crop_corners, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, normalize, tangent_at
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.regression.model import IdentityConfig


def image_crop(items, vol, crop, pool=None):
    """CT and presence only. Reuse the existing physical-coordinate sampler.

    The scalar sampler normalizes both channels to [0,1]. An empty history
    skips rendering. Sampling is identical in training
    and tracing, including the independently resolved presence grid.
    Each item reads only the axis-aligned block its own oriented crop needs.
    """
    return torch.stack([scalar_crops(items, vol, crop, pool, presence=presence)[:, 0]
                        for presence in (False, True)], 1)


class ObservationBuilder:
    """Lazy worker-local coarse-volume reader; fine CT uses the supplied volume."""
    def __init__(self, cfg, coarse_level=1, coarse_grid_scale=8.):
        self.cfg, self.coarse_level, self.coarse_grid_scale = cfg, coarse_level, coarse_grid_scale
        self._coarse = None

    def __getstate__(self):
        return dict(self.__dict__, _coarse=None)

    def images(self, items, vol, pool=None):
        if self._coarse is None:
            spec = replace(vol.spec, ct_level=self.coarse_level, ct_grid_scale=self.coarse_grid_scale)
            self._coarse = FiberVolume(spec, cache_bytes=vol.ct.cache_bytes)
            # Both scales sample the same presence array. One reader with the
            # combined budget lets the wider coarse footprint serve the fine crop.
            if vol.presence is not None:
                vol.presence.cache_bytes += self._coarse.presence.cache_bytes
                self._coarse.presence = vol.presence
        return dict(fine=image_crop(items, vol, self.cfg.fine, pool),
                    coarse=image_crop(items, self._coarse, self.cfg.coarse, pool))

    def __call__(self, items, vol):
        return dict(x=self.images(items, vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]), dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]), dtype=torch.float32),
                    **collate_targets(items))


# ---------------------------------------------------------------- visual identity


@dataclass(frozen=True)
class IdentitySampling:
    """Training-only identity targets, augmentation and ambiguous-state oversampling."""
    rule: ComponentRule = ComponentRule()
    pair_sampling_version: int = PAIR_SAMPLING_VERSION
    positives: int = 4
    negatives: int = 8
    on_fiber_tolerance: float = 1.5  # history patch counts toward the anchor within this of GT
    anchor_prob: float = .75  # fresh/replay states given seed-segment anchor patches
    presence_dropout: float = .25
    contrast: float = 1.4  # log-uniform contrast factor in [1/c, c]
    brightness: float = .1
    noise: float = .03  # maximum Gaussian noise standard deviation
    contact_fraction: float = .2  # of fresh draws, near mined contact episodes
    hard_span_fraction: float = .1
    lateral_fraction: float = .1  # near earlier fresh states that had bank negatives
    lateral_memory: int = 1024
    bank_wrong_continuation_probability: float = .75
    bank_wrong_continuation_tail: tuple = (4., 128.)
    bank_following_probability: float = 0.  # fraction of fresh draws; trainer opts in
    query_patches: bool = False  # legacy dense-map queries remain checkpoint compatible

    def __post_init__(self):
        if isinstance(self.rule, dict):
            object.__setattr__(self, 'rule', ComponentRule(**self.rule))
        fractions = (self.anchor_prob, self.presence_dropout, self.contact_fraction,
                     self.hard_span_fraction, self.lateral_fraction)
        if not all(0 <= f <= 1 for f in fractions) or sum(fractions[2:]) > 1:
            raise ValueError('Identity sampling probabilities must lie in [0, 1]; oversampling at most 1')
        if min(self.positives, self.negatives, self.lateral_memory) < 1 or self.contrast < 1:
            raise ValueError('Invalid identity sample counts or augmentation')
        if not np.isfinite(self.rule.lateral_max) or self.rule.lateral_max <= self.rule.own_radius:
            raise ValueError('Identity negative radius must exceed own-fiber radius')
        if not 0 <= self.bank_wrong_continuation_probability <= 1:
            raise ValueError('Bank wrong-continuation probability must be in [0,1]')
        if not 0 <= self.bank_following_probability <= 1:
            raise ValueError('Bank following probability must be in [0,1]')
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import validate_tail_range
        object.__setattr__(self, 'bank_wrong_continuation_tail', validate_tail_range(self.bank_wrong_continuation_tail))


# Oversampled fresh locations, recorded per state.
LOCATION_SOURCES = ('uniform', 'contact', 'hard_span', 'lateral', 'bank_following')


def contact_index_sha256(fibers, band, spacing=4., radius=6.):
    identity = dict(version=1, fibers=fiber_manifest(fibers), spacing=spacing, radius=radius,
                    band=asdict(band) if band is not None else None)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_contacts(path, fibers, band):
    """Mined contact episodes; indices refer to ``fibers``, checked by geometry hash."""
    data = json.loads(Path(path).read_text())
    if data['sha256'] != contact_index_sha256(fibers, band):
        raise ValueError(f'{path} was mined from different fibers or holdout')
    return data['episodes']


def load_hard_spans(path, fibers):
    """(fiber index, start arc, end arc) of listed controlled spans, checked by geometry."""
    data = json.loads(Path(path).read_text())
    manifest = {entry['name']: entry for entry in data['fiber_manifest']}
    current = {entry['name']: entry for entry in fiber_manifest(fibers)}
    index = {f.name: i for i, f in enumerate(fibers)}
    spans = []
    for name, ids in sorted(data['hard_spans'].items()):
        if name not in index:
            continue
        if manifest.get(name) != current[name]:
            raise ValueError(f'{path}: {name} geometry changed')
        spans.extend((index[name], fibers[index[name]].spans[i].start, fibers[index[name]].spans[i].end) for i in ids)
    return spans


def traversal(fiber, reverse):
    return (fiber.points[::-1], fiber.length-fiber.s[::-1]) if reverse else (fiber.points, fiber.s)


def local_frames(tangent):
    """Head-local patch frames (columns u, v, tangent), u parallel to the head's u."""
    t = normalize(np.asarray(tangent, np.float64))
    u = np.array([1., 0., 0.])-t[:, :1]*t
    fallback = np.linalg.norm(u, axis=-1) < 1e-3
    u[fallback] = np.cross(np.array([0., 0., 1.]), t[fallback])
    u = normalize(u)
    return np.stack((u, np.cross(t, u), t), -1)


def path_anchor(segment, travelled, cfg):
    """Seed-segment anchors of a trace, once the head has left them behind the recent span."""
    segment = np.asarray(segment, np.float64).reshape(-1, 3)
    offsets = np.arange(cfg.anchor_patches)*cfg.anchor_every
    if not cfg.anchor_patches or travelled < cfg.anchor_offset or len(segment) < 2:
        return {}
    arc = arclength(segment)
    if arc[-1] < offsets[-1]+2:
        return {}
    tangent = interp_at(segment, arc, offsets+2)-interp_at(segment, arc, np.maximum(offsets-2, 0))
    return dict(anchor_world=interp_at(segment, arc, offsets), anchor_tangent=normalize(tangent),
                anchor_age=travelled-offsets)


def patch_layout(item, cfg):
    """Recent path patches every ``patch_every`` voxels, then any seed anchors.

    Tangents come from the observed path itself (toward the head); each patch
    frame keeps the head's u axis as far as possible. Geometry is head-local.
    """
    if 'patch_centers' in item:
        return item
    P, R, H = cfg.n_patches, cfg.recent_patches, cfg.n_history
    centers = np.zeros((P, 3))
    tangents = np.tile([0., 0., 1.], (P, 1))
    ages, mask, anchor = np.zeros(P), np.zeros(P, np.float32), np.zeros(P, np.float32)
    hist, hmask = item.get('hist_local'), item.get('hmask')
    if hist is not None:
        hist, hmask = np.asarray(hist, np.float64), np.asarray(hmask)
        k = np.arange(R)*cfg.patch_every+cfg.patch_every-1
        ahead = np.where((k >= 2)[:, None], hist[np.maximum(k-2, 0)], 0.)
        back = np.minimum(k+2, H-1)
        behind = np.where((hmask[back] > 0)[:, None], hist[back], hist[k])
        centers[:R], tangents[:R], ages[:R] = hist[k], ahead-behind, k+1
        mask[:R] = hmask[k] > 0
    if 'anchor_world' in item:
        frame, pos = np.asarray(item['frame']), np.asarray(item['pos'])
        centers[R:] = (item['anchor_world']-pos) @ frame
        tangents[R:] = item['anchor_tangent'] @ frame
        ages[R:], mask[R:], anchor[R:] = item['anchor_age'], 1., 1.
    tangents[np.linalg.norm(tangents, axis=-1) < 1e-6] = (0., 0., 1.)
    frames = local_frames(tangents)
    age = np.log1p(np.minimum(ages, cfg.max_patch_age))/np.log1p(cfg.max_patch_age)
    geometry = np.concatenate((centers/16, frames[..., 2], frames[..., 0], age[:, None], anchor[:, None]), -1)
    item.update(patch_centers=centers, patch_frames=frames, patch_mask=mask,
                patch_geometry=np.where(mask[:, None] > 0, geometry, 0.).astype(np.float32))
    return item


def photometric(image, params, rng):
    """Contrast about the mean, brightness offset, then Gaussian noise; clipped to [0, 1]."""
    contrast, brightness, noise = params
    mean = float(image.mean())
    out = (image-mean)*contrast+mean+brightness
    if noise > 0:
        out = out+rng.normal(0., noise, image.shape).astype(np.float32)
    return np.clip(out, 0., 1.).astype(np.float32)


def contact_location(fibers, episode, side, reverse, rng, approach=24.):
    """A fresh state leading into or passing through one side of a contact episode."""
    fi, lo, hi = ((episode['a'], episode['a0'], episode['a1']) if side == 0
                  else (episode['b'], episode['b0'], episode['b1']))
    original = rng.uniform(lo, hi+approach) if reverse else rng.uniform(lo-approach, hi)
    length = fibers[fi].length
    original = float(np.clip(original, 0, length))
    return dict(fiber=int(fi), t=length-original if reverse else original, reverse=reverse, source=1)


class IdentityObservationBuilder(ObservationBuilder):
    """Adds CT patches along the committed path to the direct observations.

    Tracing and training call the same ``images``. Training additionally
    ``prepare``s each state (seed-segment anchors drawn from annotation,
    augmentation draws) and receives identity targets from a live validated
    path bank. Targets are sampled before any presence dropout.
    """
    def __init__(self, cfg: IdentityConfig, fibers=None, sampling=IdentitySampling(), *,
                 contacts=(), hard_spans=(), augment=False, negative_bank=None):
        super().__init__(cfg)
        self.fibers, self.sampling, self.augment = fibers, sampling, augment
        self.contacts, self.hard_spans = list(contacts), list(hard_spans)
        self.negative_bank = negative_bank
        self.lateral = deque(maxlen=sampling.lateral_memory)
        self.stats = dict(patch_seconds=0., patches=0, calls=0)

    def patch_inputs(self, items, vol, pool=None):
        cfg = self.cfg
        P, crop = cfg.n_patches, cfg.patch_crop
        patches = np.zeros((len(items), P, crop.depth, crop.width, crop.width), np.float32)
        reads, where = [], []
        for j, item in enumerate(items):
            patch_layout(item, cfg)
            frame, pos = np.asarray(item['frame']), np.asarray(item['pos'])
            for p in np.flatnonzero(item['patch_mask']):
                reads.append(dict(pos=pos+frame @ item['patch_centers'][p], frame=frame @ item['patch_frames'][p]))
                where.append((j, p))
        started = time.perf_counter()
        if reads:
            values = scalar_crops(reads, vol, crop, pool).numpy()[:, 0]
            for (j, p), value in zip(where, values):
                patches[j, p] = value
        self.stats['patch_seconds'] += time.perf_counter()-started
        self.stats['patches'] += len(reads)
        self.stats['calls'] += 1
        stack = lambda key: torch.from_numpy(np.stack([item[key] for item in items]).astype(np.float32))
        return dict(patches=torch.from_numpy(patches), patch_geometry=stack('patch_geometry'),
                    patch_mask=stack('patch_mask'))

    def images(self, items, vol, pool=None):
        x = super().images(items, vol, pool)
        x.update(self.patch_inputs(items, vol, pool))
        return x

    @property
    def pair_crop(self):
        """Presence-only search area; CT is read in identical per-query patches."""
        half = max((self.cfg.fine.width-1)*self.cfg.fine.spacing/2,
                   self.sampling.rule.lateral_max+self.cfg.max_recovery_distance)
        return replace(self.cfg.fine,width=2*int(np.ceil(half/self.cfg.fine.spacing))+1)

    def query_inputs(self, items, vol, points, valid):
        crop = self.cfg.patch_crop
        points,valid = points.numpy(),valid.numpy()
        patches = np.zeros((*points.shape[:2],crop.depth,crop.width,crop.width),np.float32)
        reads,where = [],[]
        for j,item in enumerate(items):
            for q in np.flatnonzero(valid[j]):
                reads.append(dict(pos=item['pos']+item['frame'] @ points[j,q],frame=item['frame']))
                where.append((j,q))
        if reads:
            values = scalar_crops(reads,vol,crop).numpy()[:,0]
            for index,value in zip(where,values):
                patches[index] = value
        return dict(identity_query_patches=torch.from_numpy(patches),identity_query_mask=torch.from_numpy(valid))

    # -- training

    def replace_fresh(self, sample_cfg, rng):
        """Occasional ordinary following supervision from the same live bank."""
        if (self.negative_bank is None or self.sampling.bank_following_probability == 0
                or rng.random() >= self.sampling.bank_following_probability):
            return None
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import following_sample
        return following_sample(self.negative_bank,sample_cfg,rng)

    def replace_replay(self, source, stratum, sample_cfg, rng):
        """Prefer bank departures only within the recent DAgger departure slot."""
        if (source != 2 or stratum != 4 or self.negative_bank is None
                or rng.random() >= self.sampling.bank_wrong_continuation_probability):
            return None
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
        for _ in range(3):
            item = wrong_continuation(self.negative_bank,sample_cfg,rng,
                                      tail_length_range=self.sampling.bank_wrong_continuation_tail)
            if item is not None:
                return item
        return None
    def fresh_location(self, rng):
        s = self.sampling
        u = rng.random()
        location = None
        if u < s.contact_fraction:
            if self.contacts:
                e = self.contacts[int(rng.integers(len(self.contacts)))]
                return contact_location(self.fibers, e, int(rng.integers(2)), bool(rng.integers(2)), rng)
        elif u < s.contact_fraction+s.hard_span_fraction:
            if self.hard_spans:
                fi, lo, hi = self.hard_spans[int(rng.integers(len(self.hard_spans)))]
                location = (fi, rng.uniform(lo-16, hi+16), bool(rng.integers(2)), 2)
        elif u < s.contact_fraction+s.hard_span_fraction+s.lateral_fraction:
            if self.lateral:
                fi, t, reverse = self.lateral[int(rng.integers(len(self.lateral)))]
                length = self.fibers[fi].length
                t = float(np.clip(t+rng.uniform(-16, 16), 0, length))
                return dict(fiber=fi, t=t, reverse=reverse, source=3)
        if location is None:
            return None
        fi, original, reverse, source = location
        length = self.fibers[fi].length
        original = float(np.clip(original, 0, length))
        return dict(fiber=int(fi), t=length-original if reverse else original, reverse=reverse, source=source)

    def prepare(self, item, fiber, rng):
        """Annotation-derived anchors and label geometry; augmentation draws. No volume I/O."""
        cfg, s = self.cfg, self.sampling
        _, t, reverse = item['fiber_ref']
        p, arc = traversal(fiber, reverse)
        anchor_offset = cfg.anchor_offset
        if 'bank_prefix_end_t' in item:
            # Seed anchors must precede the synthetic switch, even when its tail
            # has displaced the entire recent-history window onto the neighbor.
            anchor_offset = max(anchor_offset, t-item['bank_prefix_end_t']+(cfg.anchor_patches-1)*cfg.anchor_every)
        if cfg.anchor_patches and anchor_offset <= min(t, cfg.max_patch_age) and rng.random() < s.anchor_prob:
            back = rng.uniform(anchor_offset, min(t, cfg.max_patch_age))
            arcs = t-back+np.arange(cfg.anchor_patches)*cfg.anchor_every
            item.update(anchor_world=interp_at(p, arc, arcs), anchor_age=t-arcs,
                        anchor_tangent=np.stack([tangent_at(p, arc, a) for a in arcs]))
        patch_layout(item, cfg)
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        local = lambda arcs: (interp_at(p, arc, np.clip(arcs, 0, fiber.length))-pos) @ frame
        # Annotation through the fine crop (quarter-voxel), and behind it for history flags.
        ahead = np.arange(max(0., t-24), min(fiber.length, t+40)+1e-9, .25)
        behind = np.arange(max(0., t-cfg.patch_span-8), min(fiber.length, t+2)+1e-9, .5)
        item['identity_curve'] = local(ahead) if len(ahead) > 2 else np.zeros((0, 3))
        history_curve = local(behind) if len(behind) else np.zeros((1, 3))+np.inf
        centers = item['patch_centers']
        distance = np.linalg.norm(centers[:, None]-history_curve[None], axis=-1).min(1)
        item['patch_on_fiber'] = ((distance <= s.on_fiber_tolerance) & (item['patch_mask'] > 0)).astype(np.float32)
        item['patch_on_fiber'][cfg.recent_patches:] = item['patch_mask'][cfg.recent_patches:]
        labels = np.concatenate((item['identity_curve'], history_curve[np.isfinite(history_curve).all(-1)]))
        item['identity_label_z'] = ((labels @ frame.T+pos)[:, 2] if len(labels) else pos[2:3]).astype(np.float64)
        if self.augment:
            draw = lambda: (float(np.exp(rng.uniform(-np.log(s.contrast), np.log(s.contrast)))),
                            float(rng.uniform(-s.brightness, s.brightness)), float(rng.uniform(0, s.noise)))
            item.update(photometric=(draw(), draw()), drop_presence=bool(rng.random() < s.presence_dropout))
        item['identity_seed'] = int(rng.integers(2**63))
        item.setdefault('location_source', 0)
        return item

    def footprint_allowed(self, item, band):
        """Holdout check of every patch read footprint and of identity label geometry."""
        if band is None:
            return True
        patch_layout(item, self.cfg)
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        corners = crop_corners(self.cfg.patch_crop)
        zs = [np.asarray(item.get('identity_label_z', pos[2:3]))]
        if self.sampling.query_patches:
            # Include presence search and the full CT receptive field around
            # every possible query before reading any volume data.
            pair,patch = self.pair_crop,self.cfg.patch_crop
            footprint = replace(pair,width=pair.width+patch.width-1,
                                depth=pair.depth+patch.depth-1,behind=pair.behind+patch.behind)
            zs.append((pos+crop_corners(footprint) @ frame.T)[:,2])
        for p in np.flatnonzero(item['patch_mask']):
            world = pos+frame @ item['patch_centers'][p]+corners @ (frame @ item['patch_frames'][p]).T
            zs.append(world[:, 2])
        z = np.concatenate(zs)
        return not (z.min()-2 < band.hi and z.max()+2 >= band.lo)

    def identity_targets(self, items, x, pair_presence=None):
        if self.negative_bank is None:
            raise ValueError('Identity supervision requires a negative bank')
        cfg, s = self.cfg, self.sampling
        B, K, M, P = len(items), s.positives, s.negatives, cfg.n_patches
        shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        out = dict(positive_mask=np.zeros((B, K), np.float32), negative_mask=np.zeros((B, K, M), np.float32),
                   identity_points=np.zeros((B, K*(1+M), 3), np.float32), patch_on_fiber=np.zeros((B, P), np.float32),
                   foreign=np.zeros((B, *shape), np.uint8), presence_dropped=np.zeros(B, np.float32),
                   location_source=np.zeros(B, np.float32), foreign_components=np.zeros(B, np.float32))
        if self.negative_bank is not None:
            out['negative_bank_shards'] = np.zeros(B, np.int64)
        presence = x['fine'][:, 1].numpy() if pair_presence is None else pair_presence
        crop = cfg.fine if pair_presence is None else self.pair_crop
        for j, item in enumerate(items):
            if 'identity_curve' not in item:
                continue
            rng = np.random.default_rng(item['identity_seed'])
            bank = self.negative_bank
            found = bank.candidates(item,crop,presence[j],s.rule,mask_crop=cfg.fine) if pair_presence is not None else bank.candidates(item,crop,presence[j],s.rule)
            pos, pos_mask, neg, neg_mask = sample_pairs(
                item['identity_curve'], presence[j], crop, found['local'], found['nearest'], rng,
                positives=K, negatives=M, margin=0. if s.query_patches else (cfg.patch_crop.width-1)*cfg.fine.spacing/2,
                rule=s.rule,
                appearance_crop=crop if s.query_patches else cfg.appearance_crop,
                along_margin=0. if s.query_patches else (cfg.patch_crop.depth-1)*cfg.fine.spacing/2)
            if bank is not None:
                # Recheck float32 query coordinates against the whole target,
                # including geometry outside this crop.
                valid = neg_mask > 0
                if valid.any():
                    world = neg[valid] @ np.asarray(item['frame']).T+item['pos']
                    neg_mask[valid] *= bank.clear_of_state(item,world)
                out['negative_bank_shards'][j] = bank.shard_count
            out['positive_mask'][j], out['negative_mask'][j] = pos_mask, neg_mask
            out['identity_points'][j] = np.concatenate((pos, neg.reshape(-1, 3)))
            out['patch_on_fiber'][j] = item['patch_on_fiber']
            out['foreign'][j] = found['foreign']
            out['foreign_components'][j] = found['counts']['foreign_components']
            out['location_source'][j] = item.get('location_source', 0)
            if self.fibers is not None and item.get('source') == 0 and (neg_mask.sum(-1)*pos_mask).any():
                self.lateral.append(item['fiber_ref'])
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        # Rollout/monitor observation-only builders produce no identity labels.
        # Training and embedding evaluation provide fibers and prepared curves.
        if self.fibers is None and not self.augment and not any('identity_curve' in i for i in items):
            return batch
        x = batch['x']
        if self.sampling.query_patches:
            presence = scalar_crops(items,vol,self.pair_crop,presence=True).numpy()[:,0]
            batch.update(self.identity_targets(items,x,presence))
            valid = torch.cat((batch['positive_mask'],batch['negative_mask'].flatten(1)),1)
            x.update(self.query_inputs(items,vol,batch['identity_points'],valid))
        else:
            batch.update(self.identity_targets(items, x))
        batch['bank_tail_length'] = torch.tensor([i.get('bank_tail_length', 0.) for i in items], dtype=torch.float32)
        if self.augment:
            x['appearance'] = x['fine'][:, :1].clone()
            for j, item in enumerate(items):
                if 'photometric' not in item:
                    continue
                rng = np.random.default_rng(item['identity_seed']+1)
                crop_params, patch_params = item['photometric']
                x['appearance'][j, 0] = torch.from_numpy(photometric(x['appearance'][j, 0].numpy(), crop_params, rng))
                if 'identity_query_patches' in x:
                    queries = x['identity_query_mask'][j] > 0
                    if queries.any():
                        x['identity_query_patches'][j,queries] = torch.from_numpy(
                            photometric(x['identity_query_patches'][j,queries].numpy(),crop_params,rng))
                valid = x['patch_mask'][j] > 0
                if valid.any():
                    x['patches'][j, valid] = torch.from_numpy(photometric(x['patches'][j, valid].numpy(), patch_params, rng))
                if item['drop_presence']:
                    for scale in ('fine', 'coarse'):
                        x[scale][j, 1] = 0
                    batch['presence_dropped'][j] = 1
        return batch


def observation_builder(cfg, **kwargs):
    """Inference observation builder for either direct architecture."""
    return IdentityObservationBuilder(cfg, **kwargs) if isinstance(cfg, IdentityConfig) else ObservationBuilder(cfg)


class DirectTracer(ModelTracer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.additional_crops = (model.cfg.coarse,)
        self.observations = observation_builder(model.cfg)
        self.path_context = isinstance(model.cfg, IdentityConfig)

    def build_inputs(self, pos, frames, hist, hmask, paths=None):
        items = [dict(pos=p, frame=f) for p, f in zip(pos, frames)]
        if self.path_context:
            for j, item in enumerate(items):
                item.update(hist_local=hist[j], hmask=hmask[j])
                if paths is not None:
                    item.update(path_anchor(paths[j]['seed_segment'], paths[j]['travelled'], self.model.cfg))
        return {k: v.to(self.device) for k, v in self.observations.images(items, self.vol, self.pool).items()}
