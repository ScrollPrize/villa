"""Shared training/rollout observation builder for the direct follower."""
from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import collate_targets, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_seed
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig


def image_crop(items, vol, crop, pool=None, *, directions=False):
    """CT/presence, optionally followed by six unsigned local direction moments.

    The scalar sampler normalizes both channels to [0,1]. An empty history
    skips rendering. Sampling is identical in training
    and tracing, including the independently resolved presence grid.
    Each item reads only the axis-aligned block its own oriented crop needs.
    """
    if not directions:
        return torch.stack([scalar_crops(items, vol, crop, pool, presence=presence)[:, 0]
                            for presence in (False, True)], 1)
    from vesuvius.neural_tracing.fiber_follow.shared.direction_fields import direction_crops
    # Write every channel directly into its final storage: concatenating eight
    # full-resolution channels was an avoidable copy on every observation.
    output = np.empty((len(items), 8, crop.depth, crop.width, crop.width), np.float32)
    scalar_crops(items, vol, crop, pool, presence=False, out=output[:, :1])
    scalar_crops(items, vol, crop, pool, presence=True, out=output[:, 1:2])
    direction_crops(items, vol, crop, pool, out=output[:, 2:])
    return torch.from_numpy(output)


class ObservationBuilder:
    """One crop and only visible observed references, shared by training/tracing."""
    def __init__(self,cfg):
        self.cfg = cfg

    def images(self,items,vol,pool=None):
        for item in items:
            reference_layout(item,self.cfg)
        stack = lambda key: torch.from_numpy(np.stack([item[key] for item in items]).astype(np.float32))
        crop_images = partial(image_crop, directions=self.cfg.direction_inputs)
        x = dict(fine=crop_images(items,vol,self.cfg.fine,pool),seed=stack('visible_seed'),
                 seed_mask=stack('visible_seed_mask'),seed_age=stack('visible_seed_age'),
                 seed_tangent=stack('visible_seed_tangent'))
        x['query_frame'] = stack('frame')
        x['query_position'] = stack('pos')
        here = [bool(i.get('seed_valid', False)) and
                np.linalg.norm(np.asarray(i['seed_pos'])-i['pos']) < 1e-4 for i in items]
        x['feature_seed_here'] = torch.tensor(here)
        x['memory_mask'] = torch.ones(len(items), 1, dtype=torch.bool)
        x['memory_seed_valid'] = torch.tensor([bool(i.get('seed_valid', False)) for i in items])
        remote = [bool(i.get('seed_valid', False)) and not i.get('memory_warm', False) and not h
                  for i, h in zip(items, here)]
        if any(remote):
            from .feature_sequences import seed_observation
            seeds = [seed_observation(i, self.cfg) if r else dict(i, memory_warm=True)
                     for i, r in zip(items, remote)]
            x['feature_seed_x'] = self.images(seeds, vol, pool)
            x['feature_seed_x']['feature_active'] = torch.tensor(remote)
        return x

    def __call__(self,items,vol):
        return dict(x=self.images(items,vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]),dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]),dtype=torch.float32),
                    **collate_targets(items))


@dataclass(frozen=True)
class IdentitySampling:
    """Training-only identity targets, augmentation and ambiguous-state oversampling."""
    rule: ComponentRule = ComponentRule()
    on_fiber_tolerance: float = 1.5  # visible reference counts toward the anchor within this of GT
    presence_dropout: float = .25
    contrast: float = 1.4  # log-uniform contrast factor in [1/c, c]
    brightness: float = .1
    noise: float = .03  # maximum Gaussian noise standard deviation
    blur_probability: float = .25
    blur_sigma: tuple = (.5, 1.25)  # Gaussian sigma in sampled crop voxels
    contact_fraction: float = .2  # of fresh draws, near mined contact episodes
    hard_span_fraction: float = .1
    lateral_fraction: float = .1  # near earlier fresh states that had bank negatives
    lateral_memory: int = 1024
    bank_wrong_continuation_probability: float = .75
    bank_wrong_continuation_tail: tuple = (4., 16.)
    bank_following_probability: float = 0.  # fraction of fresh draws; trainer opts in
    bank_coverage_probability: float = 0.  # reserved fresh slots on covered parent spans
    prefer_long_continuations: bool = False
    decision_fraction: float = 0.  # fraction of endpoint proposals reserved for matched pairs
    decision_choice_fraction: float = .75  # requested choice share; remaining pairs teach departure
    candidate_tolerance: float = 1.5
    # Fresh draws replaced by long original-then-neighbor memory sequences.
    memory_switch_probability: float = 0.
    memory_switch_tail: tuple = (16., 96.)

    def __post_init__(self):
        if isinstance(self.rule, dict):
            # Checkpoints before pair sampling v5 stored a presence threshold.
            rule = {k: v for k, v in self.rule.items() if k != 'threshold'}
            object.__setattr__(self, 'rule', ComponentRule(**rule))
        fractions = (self.presence_dropout, self.contact_fraction,
                     self.hard_span_fraction, self.lateral_fraction)
        if not all(0 <= f <= 1 for f in fractions) or sum(fractions[1:]) > 1:
            raise ValueError('Identity sampling probabilities must lie in [0, 1]; oversampling at most 1')
        if self.lateral_memory < 1 or self.contrast < 1:
            raise ValueError('Invalid identity sample counts or augmentation')
        if not np.isfinite(self.blur_probability) or not 0 <= self.blur_probability <= 1:
            raise ValueError('Blur probability must be in [0, 1]')
        if (len(self.blur_sigma) != 2 or not all(np.isfinite(v) for v in self.blur_sigma)
                or not 0 <= self.blur_sigma[0] <= self.blur_sigma[1]):
            raise ValueError('Blur sigma must be a finite, nonnegative MIN MAX range')
        object.__setattr__(self, 'blur_sigma', tuple(self.blur_sigma))
        if not np.isfinite(self.rule.lateral_max) or self.rule.lateral_max <= self.rule.own_radius:
            raise ValueError('Identity negative radius must exceed own-fiber radius')
        if not 0 <= self.bank_wrong_continuation_probability <= 1:
            raise ValueError('Bank wrong-continuation probability must be in [0,1]')
        if not 0 <= self.bank_following_probability <= 1:
            raise ValueError('Bank following probability must be in [0,1]')
        if not 0 <= self.decision_fraction <= 1:
            raise ValueError('Decision fraction must be in [0,1]')
        if not 0 <= self.decision_choice_fraction <= 1:
            raise ValueError('Decision choice fraction must be in [0,1]')
        if not np.isfinite(self.candidate_tolerance) or self.candidate_tolerance <= 0:
            raise ValueError('Candidate tolerance must be finite and positive')
        if not 0 <= self.bank_coverage_probability <= 1-self.bank_following_probability:
            raise ValueError('Following and covered fresh probabilities must sum to at most one')
        if not 0 <= self.memory_switch_probability <= 1-self.bank_following_probability-self.bank_coverage_probability:
            raise ValueError('Following, covered and memory-switch fresh probabilities must sum to at most one')
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import validate_tail_range
        object.__setattr__(self, 'bank_wrong_continuation_tail', validate_tail_range(self.bank_wrong_continuation_tail))
        object.__setattr__(self, 'memory_switch_tail', validate_tail_range(self.memory_switch_tail))


# Oversampled fresh locations, recorded per state.
LOCATION_SOURCES = ('uniform', 'contact', 'hard_span', 'lateral', 'bank_following', 'bank_covered', 'decision_pair',
                    'memory_switch')


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


def visible_points(points,crop,margin=0.):
    points = np.asarray(points)
    half = (crop.width-1)*crop.spacing/2-margin
    lo = np.array([-half,-half,-crop.behind*crop.spacing+margin])
    hi = np.array([half,half,(crop.depth-1-crop.behind)*crop.spacing-margin])
    return np.isfinite(points).all(-1) & (points >= lo).all(-1) & (points <= hi).all(-1)


def reference_layout(item,cfg):
    """All reference features come from the current crop; never read remote CT."""
    points = np.zeros((cfg.n_history+1,3),np.float32)
    mask = np.zeros(cfg.n_history+1,bool)
    points[:-1] = item.get('hist_local',points[:-1])
    mask[:-1] = np.asarray(item.get('hmask',mask[:-1])).astype(bool)
    tangent = np.zeros(3,np.float32)
    age = 0.
    if item.get('seed_valid',False):
        points[-1] = (np.asarray(item['seed_pos'])-item['pos']) @ item['frame']
        mask[-1] = True
        tangent = np.asarray(item['seed_tangent']) @ item['frame']
        age = float(item['seed_age'])
    mask &= visible_points(points,cfg.fine)
    points[~mask] = 0.
    if not mask[-1]:
        tangent,age = np.zeros(3,np.float32),0.
    item.update(reference_points=points,reference_mask=mask.astype(np.float32),
                visible_seed=points[-1:],visible_seed_mask=mask[-1:].astype(np.float32),
                visible_seed_tangent=tangent.astype(np.float32),visible_seed_age=np.float32(age))
    return item


def photometric(image, params, rng):
    """Contrast about the mean, brightness offset, then Gaussian noise; clipped to [0, 1]."""
    contrast, brightness, noise = params
    mean = float(image.mean())
    out = (image-mean)*contrast+mean+brightness
    if noise > 0:
        out = out+rng.normal(0., noise, image.shape).astype(np.float32)
    return np.clip(out, 0., 1.).astype(np.float32)


def augment_image_pair(image, params, rng, *, blur_sigma=0., drop_presence=False):
    """Augment only CT/presence in place; extra direction channels are untouched.

    Blur both channels before adding CT intensity noise. Reflect padding keeps
    constant inputs constant; channel dropout remains exactly zero after blur.
    """
    values = image[:2].numpy()
    if blur_sigma > 0:
        gaussian_filter(values, sigma=(0., blur_sigma, blur_sigma, blur_sigma),
                        mode='reflect', output=values)
    values[0] = photometric(values[0], params, rng)
    if drop_presence:
        values[1] = 0


def contact_location(fibers, episode, side, reverse, rng, approach=24.):
    """A fresh state leading into or passing through one side of a contact episode."""
    fi, lo, hi = ((episode['a'], episode['a0'], episode['a1']) if side == 0
                  else (episode['b'], episode['b0'], episode['b1']))
    original = rng.uniform(lo, hi+approach) if reverse else rng.uniform(lo-approach, hi)
    length = fibers[fi].length
    original = float(np.clip(original, 0, length))
    return dict(fiber=int(fi), t=length-original if reverse else original, reverse=reverse, source=1)


class IdentityObservationBuilder(ObservationBuilder):
    """Bank-derived following, foreign masks, and memory-dependent path decisions."""
    def __init__(self,cfg: DirectConfig,fibers=None,sampling=IdentitySampling(),*,
                 contacts=(),hard_spans=(),augment=False,negative_bank=None,
                 near_negative_bank=None,following_bank=None,continuation_bank=None):
        super().__init__(cfg)
        self.fibers,self.sampling,self.augment = fibers,sampling,augment
        self.contacts,self.hard_spans = list(contacts),list(hard_spans)
        self.negative_bank,self.near_negative_bank = negative_bank,near_negative_bank
        self.following_bank,self.continuation_bank = following_bank,continuation_bank
        self.lateral = deque(maxlen=sampling.lateral_memory)

    def decision_pair(self, sample_cfg, rng):
        from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
        if self.negative_bank is None:
            raise ValueError('Matched decisions require a bank')
        return decision_pair(self.near_negative_bank or self.negative_bank, sample_cfg, self.cfg, rng,
                             choice=rng.random() < self.sampling.decision_choice_fraction)

    def replace_fresh(self, sample_cfg, rng):
        """Reserve fresh slots for following and covered annotation locations."""
        s = self.sampling
        reserved = s.bank_following_probability+s.bank_coverage_probability
        if self.negative_bank is None or reserved+s.memory_switch_probability == 0:
            return None
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import following_sample
        u = rng.random()
        if u < s.bank_following_probability:
            return following_sample(self.following_bank or self.negative_bank,sample_cfg,rng)
        if u >= reserved:
            if u < reserved+s.memory_switch_probability:
                return self.memory_switch(sample_cfg,rng)
            return None
        from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
        banks = [self.negative_bank]+([self.near_negative_bank]
            if self.near_negative_bank is not None and self.near_negative_bank is not self.negative_bank else [])
        for _ in range(3):
            bank = banks[int(rng.integers(len(banks)))]
            draw = bank.draw_path(rng,unique=False)
            if draw is None:
                continue
            fi,_,(a,b) = draw
            reverse = bool(rng.integers(2))
            length = bank.fibers[fi].length
            lo,hi = (length-b,length-a) if reverse else (a,b)
            hi -= sample_cfg.future_s[-1]+2
            if hi <= lo:
                continue
            t = float(rng.uniform(lo,hi))
            item = make_sample(bank.fibers[fi],t,reverse,sample_cfg,rng)
            item.update(fiber_ref=(fi,t,reverse),source=0,source_step=-1,stratum=-1,location_source=5)
            self.prepare(item,bank.fibers[fi],rng)
            if item['reference_on_fiber'][:-1].sum() >= 2:
                item['_identity_prepared'] = True
                return item
        return None

    def memory_switch(self, sample_cfg, rng):
        """Original fiber, bridge, then a long neighbor tail, observed along the way."""
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
        if not self.cfg.memory_slots:
            raise ValueError('Memory-switch sequences require a memory model')
        for _ in range(3):
            item = wrong_continuation(self.continuation_bank or self.negative_bank,sample_cfg,rng,
                                      tail_length_range=self.sampling.memory_switch_tail,prefer_long=True,
                                      prefix_length=(self.cfg.feature_stream_steps)*self.cfg.memory_stride,
                                      track_stride=self.cfg.memory_stride)
            if item is not None:
                item['location_source'] = LOCATION_SOURCES.index('memory_switch')
                return item
        return None

    def replace_replay(self, source, stratum, sample_cfg, rng):
        """Prefer bank departures only within the recent DAgger departure slot."""
        if (source != 2 or stratum != 4 or self.negative_bank is None
                or rng.random() >= self.sampling.bank_wrong_continuation_probability):
            return None
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
        for _ in range(3):
            item = wrong_continuation(self.continuation_bank or self.negative_bank,sample_cfg,rng,
                                      tail_length_range=self.sampling.bank_wrong_continuation_tail,
                                      prefer_long=self.sampling.prefer_long_continuations)
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

    def prepare(self,item,fiber,rng):
        """Build observable references and labels; annotations never become inputs."""
        if item.pop('_identity_prepared',False):
            return item
        cfg,s = self.cfg,self.sampling
        _,t,reverse = item['fiber_ref']
        p,arc = traversal(fiber,reverse)
        pos,frame = np.asarray(item['pos']),np.asarray(item['frame'])
        local = lambda arcs: (interp_at(p,arc,np.clip(arcs,0,fiber.length))-pos) @ frame
        if 'seed_valid' not in item and item.get('source',0) in (0,4):
            item.update(observed_seed(pos,frame,item['hist_local'],item['hmask']))
        reference_layout(item,cfg)
        # Sample the label curve densely enough for reference membership, including
        # recovery offsets. Only visible references are eligible for supervision.
        extent = max(cfg.n_history,cfg.fine.depth*cfg.fine.spacing)+16
        curve = local(np.arange(max(0.,t-extent),min(fiber.length,t+extent)+1e-9,.25))
        if len(curve):
            from scipy.spatial import cKDTree
            distance = cKDTree(curve).query(item['reference_points'])[0]
            on = (distance <= s.on_fiber_tolerance) & item['reference_mask'].astype(bool)
        else:
            on = np.zeros(cfg.n_history+1,bool)
        item['reference_on_fiber'] = on.astype(np.float32)
        item['identity_reference_valid'] = bool(on[:-1].sum() >= 2 or on[-1])
        # Confirmed old departures with no visible original-fiber evidence cannot
        # be distinguished from ordinary following of the neighboring fiber.
        item['identity_observable'] = bool(not item.get('offtrack',False) or item['identity_reference_valid'])
        if cfg.memory_slots and item.get('offtrack', False):
            from .memory_data import memory_layout
            observations, seed = memory_layout(item, cfg)
            observations = observations[:-1]  # head alone is not original-fiber history
            # Annotation membership controls supervision only. The memory
            # writer receives every observed patch, including contaminated ones.
            if seed is not None:
                observations = observations+[seed]
            if observations:
                from scipy.spatial import cKDTree
                distance = cKDTree(fiber.points).query(np.stack([o['pos'] for o in observations]))[0]
                item['identity_observable'] |= bool((distance <= s.on_fiber_tolerance).any())
        item['identity_curve'] = curve
        visible = visible_points(curve,cfg.fine)
        item['identity_label_z'] = (curve[visible] @ frame.T+pos)[:,2] if visible.any() else pos[2:3]
        if 'pair_observation_seed' in item:
            # Matched local inputs stay identical after augmentation as well;
            # the earlier observations must supply the distinguishing evidence.
            rng = np.random.default_rng(item['pair_observation_seed'])
        if self.augment:
            draw = (float(np.exp(rng.uniform(-np.log(s.contrast),np.log(s.contrast)))),
                    float(rng.uniform(-s.brightness,s.brightness)),float(rng.uniform(0,s.noise)))
            item.update(photometric=draw,drop_presence=bool(rng.random() < s.presence_dropout))
            item['blur_sigma'] = (float(rng.uniform(*s.blur_sigma))
                                  if s.blur_probability and rng.random() < s.blur_probability else 0.)
        item['identity_seed'] = int(rng.integers(2**63))
        item.setdefault('location_source',0)
        return item

    def footprint_allowed(self,item,band):
        # CT is restricted to the main crop, whose footprint FollowDataset checks.
        # Also reject matched labels if their distinguishing seed is not visible.
        if self.cfg.memory_slots:
            from .memory_data import memory_allowed
            if not memory_allowed(item, self.cfg, band):
                return False
        if band is None:
            return True
        z = np.asarray(item.get('identity_label_z',np.asarray(item['pos'])[2:3]))
        return not (z.min()-2 < band.hi and z.max()+2 >= band.lo)

    def bank_targets(self, items):
        """Foreign-path masks and coverage feedback; no contrastive point queries."""
        if self.negative_bank is None:
            raise ValueError('Bank supervision requires a negative bank')
        cfg, bank = self.cfg, self.negative_bank
        shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        out = dict(foreign=np.zeros((len(items), *shape), np.uint8),
                   presence_dropped=np.zeros(len(items), np.float32),
                   location_source=np.zeros(len(items), np.float32),
                   foreign_components=np.zeros(len(items), np.float32),
                   negative_bank_shards=np.zeros(len(items), np.int64))
        for j, item in enumerate(items):
            out['location_source'][j] = item.get('location_source', 0)
            if 'identity_curve' not in item:
                continue
            found = bank.candidates(item, cfg.fine, self.sampling.rule, mask_crop=cfg.fine,
                additional_banks=([self.near_negative_bank] if self.near_negative_bank is not None
                                  and self.near_negative_bank is not bank else ()))
            out['foreign'][j] = found['foreign']
            out['foreign_components'][j] = found['counts']['foreign_components']
            out['negative_bank_shards'][j] = bank.shard_count+(self.near_negative_bank.shard_count
                if self.near_negative_bank is not None and self.near_negative_bank is not bank else 0)
            # Remember covered locations directly, independent of randomly selected
            # contrastive queries. Missing coverage remains unknown.
            ahead = found['local']
            if (self.fibers is not None and item.get('source') == 0 and len(ahead)
                    and ((ahead[:, 2] >= cfg.future_step)
                         & (ahead[:, 2] <= cfg.n_future*cfg.future_step)).any()):
                self.lateral.append(item['fiber_ref'])
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def __call__(self,items,vol):
        batch = super().__call__(items,vol)
        if self.fibers is None and not self.augment and not any('identity_curve' in i for i in items):
            return batch
        batch.update(self.bank_targets(items))
        batch['identity_observable'] = torch.tensor([i.get('identity_observable',True) for i in items])
        batch['bank_tail_length'] = torch.tensor([i.get('bank_tail_length',0.) for i in items],dtype=torch.float32)
        if self.sampling.decision_fraction:
            from .identity_decisions import CANDIDATE_COUNT
            from .supervision import candidate_targets
            shape = (CANDIDATE_COUNT,self.cfg.n_future)
            for key,trailing in (('candidate_points',(3,)),('candidate_mask',())):
                batch[key] = torch.from_numpy(np.stack([i.get(key,np.zeros((*shape,*trailing),np.float32)) for i in items]))
            batch['candidate_kind'] = torch.from_numpy(np.stack([i.get('candidate_kind',
                np.full(CANDIDATE_COUNT, -1, np.int64)) for i in items]))
            batch['candidate_labels'], batch['candidate_mask'] = candidate_targets(
                batch, self.cfg, self.sampling.candidate_tolerance)
            for key in ('decision_kind','decision_tail'):
                batch[key] = torch.tensor([i.get(key,0) for i in items],dtype=torch.float32)
        batch['seed_present'] = batch['x']['seed_mask'].flatten()
        if self.augment:
            batch['blurred'] = torch.zeros(len(items))
            for j,item in enumerate(items):
                if 'photometric' not in item:
                    continue
                rng = np.random.default_rng(item['identity_seed']+1)
                augmentation = dict(blur_sigma=item.get('blur_sigma', 0.), drop_presence=item['drop_presence'])
                batch['blurred'][j] = augmentation['blur_sigma'] > 0
                augment_image_pair(batch['x']['fine'][j], item['photometric'], rng, **augmentation)
                if item['drop_presence']:
                    batch['presence_dropped'][j] = 1
        return batch

    @property
    def streaming(self):
        return True

    def sequence_batches(self, items, vol, **kwargs):
        from .feature_sequences import sequence_batches
        return sequence_batches(self, items, vol, **kwargs)


def observation_builder(cfg,**kwargs):
    return IdentityObservationBuilder(cfg,**kwargs)


class DirectTracer(ModelTracer):
    path_context = True

    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.additional_crops = ()
        self.observations = observation_builder(model.cfg)

    def build_inputs(self,pos,frames,hist,hmask,paths=None):
        items = [dict(pos=p,frame=f,hist_local=h,hmask=m) for p,f,h,m in zip(pos,frames,hist,hmask)]
        if paths is not None:
            for item,path in zip(items,paths):
                item.update({k:path[k] for k in SEED_FIELDS if k in path})
                item['memory_warm'] = path.get('memory_warm', False)
        def move(value):
            return {k: move(v) for k, v in value.items()} if isinstance(value, dict) else value.to(self.device)
        return move(self.observations.images(items,self.vol,self.pool))
