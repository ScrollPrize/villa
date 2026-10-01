"""Shared training/rollout observation builder for the direct follower."""
from collections import deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import collate_targets
from vesuvius.neural_tracing.fiber_follow.shared.crop_sampling import scalar_crops, empty_image_batch
from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import BACKGROUND, LIMIT
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_seed
from vesuvius.neural_tracing.fiber_follow.shared.heading import orient_item, frame_prefetch_bounds, FRAME_POLICY
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig


def image_crop(items, vol, crop, pool=None, *, directions=False, input_mode='ct+presence'):
    """CT/presence, optionally followed by six unsigned local direction moments.

    CT uses foreground-only robust z-scores; presence remains [0,1]. An empty history
    skips rendering. Sampling is identical in training
    and tracing, including the independently resolved presence grid.
    Each item reads only the axis-aligned block its own oriented crop needs.
    """
    # Write every channel directly into its final storage: concatenating eight
    # full-resolution channels and copying them for IPC are both avoidable.
    if input_mode == 'ct' and directions:
        raise ValueError('CT-only image inputs cannot include direction fields')
    channels = 1 if input_mode == 'ct' else (8 if directions else 2)
    tensor = empty_image_batch((len(items), channels, crop.depth, crop.width, crop.width))
    output = tensor.numpy()
    scalar_crops(items, vol, crop, pool, presence=False, out=output[:, :1])
    if input_mode != 'ct':
        if vol.presence is None:
            raise ValueError('Presence inputs require a presence prediction volume')
        scalar_crops(items, vol, crop, pool, presence=True, out=output[:, 1:2])
    if directions:
        from vesuvius.neural_tracing.fiber_follow.shared.direction_fields import direction_crops
        direction_crops(items, vol, crop, pool, out=output[:, 2:])
    return tensor


class ObservationBuilder:
    """One crop and only visible observed references, shared by training/tracing."""
    def __init__(self,cfg):
        self.cfg = cfg

    def prefetch_bounds(self,item,vol):
        """CT footprints covering unresolved roll and normal-estimation context."""
        from .history_slabs import slab_layout, SLAB
        yield from frame_prefetch_bounds(item,self.cfg.fine,vol.input_scale)
        for slab in slab_layout(item):
            yield from frame_prefetch_bounds(slab,SLAB,vol.input_scale)

    def finalize_frames(self,items,vol):
        for item in items:
            orient_item(item,vol)

    def images(self,items,vol,pool=None):
        self.finalize_frames(items,vol)
        for item in items:
            reference_layout(item,self.cfg)
        stack = lambda key: torch.from_numpy(np.stack([item[key] for item in items]).astype(np.float32))
        crop_images = partial(image_crop, directions=self.cfg.direction_inputs, input_mode=self.cfg.input_mode)
        x = dict(fine=crop_images(items,vol,self.cfg.fine,pool),seed=stack('visible_seed'),
                 seed_mask=stack('visible_seed_mask'),seed_age=stack('visible_seed_age'),
                 seed_tangent=stack('visible_seed_tangent'))
        from .history_slabs import load_slabs
        x.update(load_slabs(items, vol, self.cfg, pool))
        return x

    def observations(self, items, vol):
        return dict(x=self.images(items,vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]),dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]),dtype=torch.float32))

    def __call__(self,items,vol):
        return dict(self.observations(items, vol), **collate_targets(items))


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
    lateral_fraction: float = .1  # near earlier fresh states that had bank negatives
    lateral_memory: int = 1024
    bank_wrong_continuation_probability: float = .75
    bank_wrong_continuation_tail: tuple = (4., 16.)
    bank_following_probability: float = 0.  # independent fraction of endpoint proposals
    bank_hard_fraction: float = .5  # mix geometry-ranked proposals with uniform bank draws
    replay_failure_fraction: float = .5  # remainder preserves recoverable drift bands
    bank_coverage_probability: float = 0.  # reserved fresh slots on covered parent spans
    prefer_long_continuations: bool = False
    decision_fraction: float = 0.  # fraction of endpoint proposals reserved for matched pairs
    decision_choice_fraction: float = .75  # requested choice share; remaining pairs teach departure
    candidate_tolerance: float = 1.5
    # Fresh draws replaced by long original-then-neighbor observed paths.
    memory_switch_probability: float = 0.
    memory_switch_tail: tuple = (16., 96.)

    def __post_init__(self):
        if isinstance(self.rule, dict):
            # Checkpoints before pair sampling v5 stored a presence threshold.
            rule = {k: v for k, v in self.rule.items() if k != 'threshold'}
            object.__setattr__(self, 'rule', ComponentRule(**rule))
        if not all(0 <= f <= 1 for f in (self.presence_dropout, self.lateral_fraction)):
            raise ValueError('Identity sampling probabilities must lie in [0, 1]')
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
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in (self.bank_hard_fraction, self.replay_failure_fraction)):
            raise ValueError('Hard-bank and replay-failure fractions must be in [0,1]')
        if not 0 <= self.decision_fraction <= 1:
            raise ValueError('Decision fraction must be in [0,1]')
        if self.decision_fraction+self.bank_following_probability > 1:
            raise ValueError('Decision and bank-following endpoint fractions must sum to at most one')
        if not 0 <= self.decision_choice_fraction <= 1:
            raise ValueError('Decision choice fraction must be in [0,1]')
        if not np.isfinite(self.candidate_tolerance) or self.candidate_tolerance <= 0:
            raise ValueError('Candidate tolerance must be finite and positive')
        if not 0 <= self.bank_coverage_probability <= 1:
            raise ValueError('Covered fresh probability must be in [0,1]')
        if not 0 <= self.memory_switch_probability <= 1-self.bank_coverage_probability:
            raise ValueError('Covered and memory-switch fresh probabilities must sum to at most one')
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import validate_tail_range
        object.__setattr__(self, 'bank_wrong_continuation_tail', validate_tail_range(self.bank_wrong_continuation_tail))
        object.__setattr__(self, 'memory_switch_tail', validate_tail_range(self.memory_switch_tail))


# Oversampled fresh locations, recorded per state.
LOCATION_SOURCES = ('uniform', 'lateral', 'bank_following', 'bank_covered', 'decision_pair', 'memory_switch')


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
    """Augment normalized material only; keep masked background exactly -4."""
    contrast, brightness, noise = params
    material = image > BACKGROUND
    mean = float(image[material].mean()) if material.any() else 0.
    out = (image-mean)*contrast+mean+brightness*(2*LIMIT)
    if noise > 0:
        out = out+rng.normal(0., noise*(2*LIMIT), image.shape).astype(np.float32)
    np.clip(out, BACKGROUND, LIMIT, out=out)
    out[~material] = BACKGROUND
    return out.astype(np.float32)


def augment_ct(image, params, rng, blur_sigma=0.):
    """Normalized convolution avoids bleeding black background into material."""
    if blur_sigma > 0:
        material = image > BACKGROUND
        weights = gaussian_filter(material.astype(np.float32), blur_sigma, mode='reflect')
        blurred = gaussian_filter(np.where(material, image, 0.), blur_sigma, mode='reflect')
        np.divide(blurred, weights, out=blurred, where=weights > 0)
        image[:] = np.where(material, blurred, BACKGROUND)
    image[:] = photometric(image, params, rng)


def augment_image_pair(image, params, rng, *, blur_sigma=0., drop_presence=False):
    """Augment only CT/presence in place; extra direction channels are untouched.

    Blur both channels before adding CT intensity noise. Reflect padding keeps
    constant inputs constant; channel dropout remains exactly zero after blur.
    """
    values = image[:2].numpy()
    augment_ct(values[0], params, rng, blur_sigma)
    if blur_sigma > 0 and len(values) > 1:
        gaussian_filter(values[1], sigma=blur_sigma, mode='reflect', output=values[1])
    if drop_presence and len(values) > 1:
        values[1] = 0


class IdentityObservationBuilder(ObservationBuilder):
    """Bank-derived following, foreign masks, and history-dependent path decisions."""

    def finalize_frames(self,items,vol):
        super().finalize_frames(items,vol)
        from scipy.spatial import cKDTree
        for item in items:
            reference_layout(item,self.cfg)
            if 'identity_curve' in item:
                curve = item['identity_curve']
                distance = (cKDTree(curve).query(item['reference_points'])[0] if len(curve)
                            else np.full(len(item['reference_points']),np.inf))
                on = (distance <= self.sampling.on_fiber_tolerance) & item['reference_mask'].astype(bool)
                item['reference_on_fiber'] = on.astype(np.float32)
                item['identity_reference_valid'] = bool(on[:-1].sum() >= 2 or on[-1])
                item['identity_observable'] = bool(not item.get('offtrack',False)
                    or item['identity_reference_valid'] or item.get('slab_identity_observable',False))
            if 'candidate_points' in item:
                inside = visible_points(item['candidate_points'],self.cfg.fine)
                item['candidate_mask'] = np.minimum.accumulate(item['candidate_mask']*inside,axis=-1)

    def __init__(self,cfg: DirectConfig,fibers=None,sampling=IdentitySampling(),*,
                 augment=False,negative_bank=None,
                 near_negative_bank=None,following_bank=None,continuation_bank=None):
        super().__init__(cfg)
        self.fibers,self.sampling,self.augment = fibers,sampling,augment
        self.negative_bank,self.near_negative_bank = negative_bank,near_negative_bank
        self.following_bank,self.continuation_bank = following_bank,continuation_bank
        self.lateral = deque(maxlen=sampling.lateral_memory)

    def decision_pair(self, sample_cfg, rng):
        from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
        if self.negative_bank is None:
            raise ValueError('Matched decisions require a bank')
        return decision_pair(self.near_negative_bank or self.negative_bank, sample_cfg, self.cfg, rng,
                             choice=rng.random() < self.sampling.decision_choice_fraction,
                             hard_fraction=self.sampling.bank_hard_fraction)

    def bank_following(self, sample_cfg, rng):
        """Draw a generated target from the dedicated endpoint budget."""
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import following_sample
        bank = self.following_bank or self.negative_bank
        return None if bank is None else following_sample(bank, sample_cfg, rng,
                                                        hard_fraction=self.sampling.bank_hard_fraction)

    def replace_fresh(self, sample_cfg, rng):
        """Oversample covered annotations or switch histories with annotated targets."""
        s = self.sampling
        reserved = s.bank_coverage_probability
        if self.negative_bank is None or reserved+s.memory_switch_probability == 0:
            return None
        u = rng.random()
        if u >= reserved:
            if u < reserved+s.memory_switch_probability:
                return self.memory_switch(sample_cfg,rng)
            return None
        from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
        banks = [self.negative_bank]+([self.near_negative_bank]
            if self.near_negative_bank is not None and self.near_negative_bank is not self.negative_bank else [])
        for _ in range(3):
            bank = banks[int(rng.integers(len(banks)))]
            draw = bank.draw_path(rng,unique=False,hard_fraction=s.bank_hard_fraction)
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
            item.update(fiber_ref=(fi,t,reverse),source=0,source_step=-1,stratum=-1,location_source=LOCATION_SOURCES.index('bank_covered'))
            self.prepare(item,bank.fibers[fi],rng)
            if item['reference_on_fiber'][:-1].sum() >= 2:
                item['_identity_prepared'] = True
                return item
        return None

    def memory_switch(self, sample_cfg, rng):
        """Original fiber, bridge, then a long neighbor tail in one observed prefix."""
        from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
        for _ in range(3):
            item = wrong_continuation(self.continuation_bank or self.negative_bank,sample_cfg,rng,
                                      tail_length_range=self.sampling.memory_switch_tail,prefer_long=True,
                                      prefix_length=float(rng.uniform(128., 1024.)))
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
        if rng.random() < self.sampling.lateral_fraction and self.lateral:
            fi, t, reverse = self.lateral[int(rng.integers(len(self.lateral)))]
            length = self.fibers[fi].length
            t = float(np.clip(t+rng.uniform(-16, 16), 0, length))
            return dict(fiber=fi, t=t, reverse=reverse, source=LOCATION_SOURCES.index('lateral'))
        return None

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
        if item.get('offtrack', False):
            from .history_slabs import slab_layout
            observations = slab_layout(item)
            # Membership affects labels only; every observed slab is still input.
            from scipy.spatial import cKDTree
            distance = cKDTree(fiber.points).query(np.stack([o['pos'] for o in observations]))[0]
            item['slab_identity_observable'] = bool((distance <= s.on_fiber_tolerance).any())
            item['identity_observable'] |= item['slab_identity_observable']
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
        from .history_slabs import slabs_allowed
        if not slabs_allowed(item, band):
            return False
        if band is None:
            return True
        z = np.asarray(item.get('identity_label_z',np.asarray(item['pos'])[2:3]))
        return not (z.min()-2 < band.hi and z.max()+2 >= band.lo)

    def prepare_sampling_feedback(self, items):
        """Advance geometry-only sampling feedback before planning another batch."""
        self.bank_targets(items,np.zeros(len(items),bool))
        for item in items:
            item['_sampling_feedback_prepared'] = True

    def bank_targets(self, items, decision_mask=None):
        """Foreign-path masks and coverage feedback; no contrastive point queries."""
        if self.negative_bank is None:
            raise ValueError('Bank supervision requires a negative bank')
        cfg, bank = self.cfg, self.negative_bank
        selected = np.ones(len(items), bool) if decision_mask is None else np.asarray(decision_mask, dtype=bool)
        shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        out = dict(presence_dropped=np.zeros(len(items), np.float32),
                   location_source=np.zeros(len(items), np.float32),
                   foreign_components=np.zeros(len(items), np.float32),
                   negative_bank_shards=np.zeros(len(items), np.int64))
        if selected.any() or not len(items):
            out['foreign'] = np.zeros((len(items), *shape), np.uint8)
        for j, item in enumerate(items):
            out['location_source'][j] = item.get('location_source', 0)
            if 'identity_curve' not in item:
                continue
            feedback = (self.fibers is not None and item.get('source') == 0
                        and not item.get('_sampling_feedback_prepared',False))
            if not selected[j] and not feedback:
                continue
            found = bank.candidates(item, cfg.fine, self.sampling.rule, mask_crop=cfg.fine,
                additional_banks=([self.near_negative_bank] if self.near_negative_bank is not None
                                  and self.near_negative_bank is not bank else ()), rasterize=bool(selected[j]))
            if selected[j]:
                out['foreign'][j] = found['foreign']
            out['foreign_components'][j] = found['counts']['foreign_components']
            out['negative_bank_shards'][j] = bank.shard_count+(self.near_negative_bank.shard_count
                if self.near_negative_bank is not None and self.near_negative_bank is not bank else 0)
            # Remember covered locations directly, independent of randomly selected
            # contrastive queries. Missing coverage remains unknown.
            ahead = found['local']
            if (feedback and len(ahead)
                    and ((ahead[:, 2] >= cfg.future_step)
                         & (ahead[:, 2] <= cfg.n_future*cfg.future_step)).any()):
                self.lateral.append(item['fiber_ref'])
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def __call__(self,items,vol, *, decision_mask=None):
        """Images for every observation; expensive targets only for decisions.

        Mixed batches retain row-aligned tensors with unused rows zero-filled.
        Observation-only batches omit annotation and dense-mask tensors entirely.
        Preparation/holdout checks and per-item augmentation RNG are unchanged.
        """
        selected = (torch.ones(len(items), dtype=torch.bool) if decision_mask is None
                    else torch.as_tensor(decision_mask, dtype=torch.bool))
        if selected.shape != (len(items),):
            raise ValueError('Decision mask must have one entry per observation')
        indices = selected.nonzero().flatten()
        def scatter(values):
            if len(indices) == len(items):
                return values
            return {k: v.new_zeros((len(items), *v.shape[1:])).index_copy_(0, indices, v)
                    for k, v in values.items()}
        batch = self.observations(items, vol)
        if len(indices):
            batch.update(scatter(collate_targets([items[j] for j in indices.tolist()])))
        if self.fibers is None and not self.augment and not any('identity_curve' in i for i in items):
            return batch
        batch.update(self.bank_targets(items, selected))
        batch['identity_observable'] = torch.tensor([i.get('identity_observable',True) for i in items])
        batch['bank_tail_length'] = torch.tensor([i.get('bank_tail_length',0.) for i in items],dtype=torch.float32)
        if self.sampling.decision_fraction and len(indices):
            from .identity_decisions import CANDIDATE_COUNT
            from .supervision import candidate_targets
            shape = (CANDIDATE_COUNT,self.cfg.n_future)
            # CT frame rotation promotes local coordinates to NumPy float64.
            # Match the other FP32 model inputs before candidate scoring/autocast.
            for key,trailing in (('candidate_points',(3,)),('candidate_mask',())):
                batch[key] = torch.as_tensor(np.stack([
                    i.get(key,np.zeros((*shape,*trailing),np.float32)) for i in items]), dtype=torch.float32)
            batch['candidate_kind'] = torch.from_numpy(np.stack([i.get('candidate_kind',
                np.full(CANDIDATE_COUNT, -1, np.int64)) for i in items]))
            # Most rows have no supplied candidate paths, even at decisions.
            # Avoid sampling their full foreign volume for four all-masked paths.
            eligible = (selected & batch['candidate_mask'].bool().flatten(1).any(1)
                        & batch['identity_observable']).nonzero().flatten()
            labels = torch.zeros_like(batch['candidate_mask'], dtype=torch.float32)
            known = torch.zeros_like(labels)
            if len(eligible):
                target_batch = batch if len(eligible) == len(items) else {
                    k: v[eligible] for k, v in batch.items() if torch.is_tensor(v)}
                target, mask = candidate_targets(target_batch, self.cfg, self.sampling.candidate_tolerance)
                labels[eligible], known[eligible] = target, mask
            batch['candidate_labels'], batch['candidate_mask'] = labels, known
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
                for slot in batch['x']['history_valid'][j].nonzero().flatten().tolist():
                    ct = batch['x']['history_slabs'][j, slot, 0].numpy()
                    augment_ct(ct, item['photometric'], rng, augmentation['blur_sigma'])
                if item['drop_presence']:
                    batch['presence_dropped'][j] = 1
        return batch



def observation_builder(cfg,**kwargs):
    return IdentityObservationBuilder(cfg,**kwargs)


class DirectTracer(ModelTracer):
    path_context = True

    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.additional_crops = ()
        self.observations = observation_builder(model.cfg)

    def build_inputs(self,pos,frames,hist,hmask,paths=None):
        items = [dict(pos=p,frame=f,hist_local=h,hmask=m,frame_policy=FRAME_POLICY) for p,f,h,m in zip(pos,frames,hist,hmask)]
        if paths is not None:
            for item,path in zip(items,paths):
                item.update({k:path[k] for k in SEED_FIELDS if k in path})
                item['observed_path'] = path['observed_path']
        def move(value):
            return {k: move(v) for k, v in value.items()} if isinstance(value, dict) else value.to(self.device)
        return move(self.observations.images(items,self.vol,self.pool))
