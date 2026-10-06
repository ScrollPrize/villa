"""Shared training/rollout observation builder for both fiber followers."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from vesuvius.neural_tracing.fiber_follow.data.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.data.data import SOURCE, collate_targets, resolve_trace_seed
from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops, empty_image_batch
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_seed
from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_DISTANCE, supervise
from vesuvius.neural_tracing.fiber_follow.tracing.heading import heading_free_bounds, reframe_item, FRAME_POLICY, FRAME_POLICIES
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import frame_predictor, orient_items, crop_frame_bounds
from vesuvius.neural_tracing.fiber_follow.models.model import FollowerConfig


def image_crop(items, vol, crop, pool=None):
    """CT-only crops, normalized identically for training and tracing."""
    tensor = empty_image_batch((len(items), 1, crop.depth, crop.width, crop.width))
    scalar_crops(items, vol, crop, pool, out=tensor.numpy())
    return tensor


class ObservationBuilder:
    """One crop and only visible observed references, shared by training/tracing."""
    def __init__(self,cfg):
        self.cfg = cfg

    def prefetch_bounds(self,item,vol):
        """CT footprints covering unresolved roll and normal-estimation context."""
        if '_pending_seed_heading' in item:
            # A trace start takes its CT seed heading only once the image is built.
            yield heading_free_bounds(item['pos'],self.cfg.fine,vol.input_scale)
        predictor = frame_predictor(self.cfg)
        yield from crop_frame_bounds(item,self.cfg.fine,vol,predictor)

    def finalize_frames(self,items,vol):
        for item in items:
            resolve_trace_seed(item,vol)
        orient_items(items,vol,frame_predictor(self.cfg))

    def images(self,items,vol,pool=None):
        self.finalize_frames(items,vol)
        for item in items:
            reference_layout(item,self.cfg)
        stack = lambda key: torch.from_numpy(np.stack([item[key] for item in items]).astype(np.float32))
        crop_images = image_crop
        x = dict(fine=crop_images(items,vol,self.cfg.fine,pool),seed=stack('visible_seed'),
                 seed_mask=stack('visible_seed_mask'),seed_age=stack('visible_seed_age'),
                 seed_tangent=stack('visible_seed_tangent'))
        from vesuvius.neural_tracing.fiber_follow.models.path_geometry import path_geometry_inputs
        x.update(path_geometry_inputs(items))
        # Recorded replay frames have unknown quality unless it was supplied;
        # do not count them as newly successful CT estimates.
        quality = [item.get('ct_frame_diagnostics', {}) for item in items]
        for key, default, dtype in (('source', -1, torch.int64), ('energy', 0., torch.float32),
                                    ('gap', 0., torch.float32)):
            x['ct_frame_'+key] = torch.tensor([q.get(key, default) for q in quality], dtype=dtype)
        return x

    def observations(self, items, vol):
        batch = dict(x=self.images(items,vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]),dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]),dtype=torch.float32))
        # Small, CPU-side geometry for truthful crop diagnostics. It is never
        # passed to the model and does not require another CT read.
        annotation = np.zeros((len(items), 257, 3), dtype=np.float32)
        valid = np.zeros((len(items), 257), dtype=bool)
        for row, item in enumerate(items):
            curve = item.get('identity_curve')
            if curve is None and 'fut_local' in item:
                curve = np.concatenate((item['gt_history'][::-1], item['fut_local']))
            if curve is not None and len(curve):
                curve = np.asarray(curve)
                index = np.linspace(0, len(curve)-1, min(257, len(curve))).round().astype(int)
                annotation[row, :len(index)] = curve[index]
                valid[row, :len(index)] = np.isfinite(curve[index]).all(-1)
        batch.update(diagnostic_annotation=torch.from_numpy(annotation),
                     diagnostic_annotation_mask=torch.from_numpy(valid),
                     crop_frame=torch.as_tensor(np.stack([i['frame'] for i in items]), dtype=torch.float64),
                     crop_pos=torch.as_tensor(np.stack([i['pos'] for i in items]), dtype=torch.float64))
        return batch

    def __call__(self,items,vol):
        return dict(self.observations(items, vol), **collate_targets(items))


@dataclass(frozen=True)
class IdentitySampling:
    """Training-only identity targets, augmentation and fresh-location oversampling."""
    rule: ComponentRule = ComponentRule()
    on_fiber_tolerance: float = 1.5  # visible reference counts toward the anchor within this of GT
    contrast: float = 1.4  # log-uniform contrast factor in [1/c, c]
    brightness: float = .1
    noise: float = .03  # maximum Gaussian noise standard deviation
    blur_probability: float = .25
    blur_sigma: tuple = (.5, 1.25)  # Gaussian sigma in sampled crop voxels
    lateral_fraction: float = .1  # fresh locations near earlier fresh states that had bank negatives
    lateral_memory: int = 1024
    bank_hard_fraction: float = .5  # mix geometry-ranked proposals with uniform bank draws
    bank_coverage_probability: float = 0.  # fresh locations on covered parent spans
    # Certified synthetic failures: OU-noised original prefix, bridge, short neighbor tail.
    synthetic_tail: tuple = (4., 16.)
    synthetic_prefix: tuple = (128., 1024.)
    # Crop roll about the heading: exact 180-degree flips make the tracer's roll-sign
    # convention irrelevant; jitter covers CT roll-estimate noise (p99 ~5 degrees).
    roll_flip_probability: float = .5
    roll_jitter_deg: float = 5.
    roll_jitter_max_deg: float = 15.

    def __post_init__(self):
        if isinstance(self.rule, dict):
            object.__setattr__(self, 'rule', ComponentRule(**self.rule))
        if not all(0 <= f <= 1 for f in (self.lateral_fraction,)):
            raise ValueError('Identity sampling probabilities must lie in [0, 1]')
        if self.lateral_memory < 1 or self.contrast < 1:
            raise ValueError('Invalid identity sample counts or augmentation')
        if not np.isfinite(self.blur_probability) or not 0 <= self.blur_probability <= 1:
            raise ValueError('Blur probability must be in [0, 1]')
        if (not 0 <= self.roll_flip_probability <= 1 or not np.isfinite(self.roll_jitter_deg) or self.roll_jitter_deg < 0
                or not np.isfinite(self.roll_jitter_max_deg) or self.roll_jitter_max_deg < 0):
            raise ValueError('Roll augmentation needs a flip probability in [0, 1] and finite nonnegative jitter')
        if (len(self.blur_sigma) != 2 or not all(np.isfinite(v) for v in self.blur_sigma)
                or not 0 <= self.blur_sigma[0] <= self.blur_sigma[1]):
            raise ValueError('Blur sigma must be a finite, nonnegative MIN MAX range')
        object.__setattr__(self, 'blur_sigma', tuple(self.blur_sigma))
        if not np.isfinite(self.rule.lateral_max) or self.rule.lateral_max <= self.rule.own_radius:
            raise ValueError('Identity negative radius must exceed own-fiber radius')
        if not np.isfinite(self.bank_hard_fraction) or not 0 <= self.bank_hard_fraction <= 1:
            raise ValueError('Hard-bank fraction must be in [0,1]')
        if not 0 <= self.bank_coverage_probability <= 1:
            raise ValueError('Covered fresh probability must be in [0,1]')
        from vesuvius.neural_tracing.fiber_follow.data.neighbor_continuations import validate_tail_range
        object.__setattr__(self, 'synthetic_tail', validate_tail_range(self.synthetic_tail))
        object.__setattr__(self, 'synthetic_prefix', validate_tail_range(self.synthetic_prefix))


# Oversampled fresh locations, recorded per state.
LOCATION_SOURCES = ('uniform', 'lateral', 'bank_covered')


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


def photometric_draw(sampling, rng):
    """Contrast/brightness/noise and blur sigma (0: none) for one crop."""
    s = sampling
    draw = (float(np.exp(rng.uniform(-np.log(s.contrast),np.log(s.contrast)))),
            float(rng.uniform(-s.brightness,s.brightness)),float(rng.uniform(0,s.noise)))
    blur = float(rng.uniform(*s.blur_sigma)) if s.blur_probability and rng.random() < s.blur_probability else 0.
    return draw, blur


def augment_ct(image, params, rng, blur_sigma=0.):
    """Augment every z-scored CT value without a mask or clipping."""
    if blur_sigma > 0:
        image[:] = gaussian_filter(image, blur_sigma, mode='reflect')
    contrast, brightness, noise = params
    mean = float(image.mean())
    image[:] = (image-mean)*contrast+mean+brightness*8.
    if noise > 0:
        image += rng.normal(0., noise*8., image.shape).astype(np.float32)


def augment_image_pair(image, params, rng, *, blur_sigma=0.):
    """Augment CT in place; auxiliary observed-path channels are unchanged."""
    augment_ct(image[0].numpy(), params, rng, blur_sigma)


def roll_frame(frame, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(frame) @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class IdentityObservationBuilder(ObservationBuilder):
    """Foreign masks, identity evidence and augmentation on top of the shared observation."""

    def __init__(self,cfg: FollowerConfig,fibers=None,sampling=IdentitySampling(),*,
                 augment=False,neighbors=None):
        super().__init__(cfg)
        self.fibers,self.sampling,self.augment = fibers,sampling,augment
        self.neighbors = neighbors  # neighboring annotations (AFV sources); None: no known neighbors
        self.lateral = deque(maxlen=sampling.lateral_memory)

    def apply_roll(self, item):
        """Apply a drawn roll once the frame is final, before any read is planned.

        Every local label, reference and crop footprint then shares the rolled frame;
        world geometry is unchanged. Unresolved CT frames keep the roll pending: their
        planned footprint already covers every roll about the heading.
        """
        if item.get('frame_policy') not in FRAME_POLICIES or 'roll_augmentation' not in item:
            return item
        angle = item.pop('roll_augmentation')
        if angle:
            reframe_item(item, roll_frame(item['frame'], angle))
        return item

    def identity_evidence(self, item, curve):
        """Visible original-fiber evidence; it gates every displaced state's supervision."""
        from scipy.spatial import cKDTree
        reference_layout(item,self.cfg)
        distance = (cKDTree(curve).query(item['reference_points'])[0] if len(curve)
                    else np.full(len(item['reference_points']),np.inf))
        on = (distance <= self.sampling.on_fiber_tolerance) & item['reference_mask'].astype(bool)
        item['reference_on_fiber'] = on.astype(np.float32)
        item['identity_evidence'] = bool(on[:-1].sum() >= 2 or on[-1])
        item['identity_observable'] = bool(item['match_distance'] <= DEPARTURE_DISTANCE or item['identity_evidence'])
        if 'trace_facts' in item:
            item['trace_facts']['identity_observable'] = item['identity_evidence']
            supervise(item)
        return item

    def finalize_frames(self,items,vol):
        super().finalize_frames(items,vol)
        for item in items:
            if self.augment:
                self.apply_roll(item)
            else:
                item.pop('roll_augmentation', None)
            if 'identity_curve' in item:
                self.identity_evidence(item, item['identity_curve'])
            else:
                reference_layout(item,self.cfg)

    def replace_fresh(self, sample_cfg, rng, **options):
        """Oversample covered annotations with annotated targets."""
        s = self.sampling
        if self.neighbors is None or not s.bank_coverage_probability or rng.random() >= s.bank_coverage_probability:
            return None
        from vesuvius.neural_tracing.fiber_follow.data.data import make_sample
        bank = self.neighbors
        for _ in range(3):
            draw = bank.draw_path(rng,hard_fraction=s.bank_hard_fraction)
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
            item = make_sample(bank.fibers[fi],t,reverse,sample_cfg,rng,**options)
            item.update(fiber_ref=(fi,t,reverse),location_source=LOCATION_SOURCES.index('bank_covered'))
            self.prepare(item,bank.fibers[fi],rng)
            if item['reference_on_fiber'][:-1].sum() >= 2:
                return item
        return None

    def synthetic_terminal(self, sample_cfg, rng):
        """Certified wrong continuation after an OU-noised original prefix; labeled terminal."""
        from vesuvius.neural_tracing.fiber_follow.data.neighbor_continuations import wrong_continuation
        if self.neighbors is None:
            return None
        tail = self.sampling.synthetic_tail
        for _ in range(3):
            item = wrong_continuation(self.neighbors,sample_cfg,rng,tail_length_range=tail,
                                      prefix_length=float(rng.uniform(*self.sampling.synthetic_prefix)))
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
        """Build observable references, identity evidence and augmentation draws.

        Annotations never become inputs. A recorded (final) frame takes its roll now,
        before footprint checks and read planning.
        """
        cfg,s = self.cfg,self.sampling
        item['fiber_family'] = fiber.tag
        _,t,reverse = item['fiber_ref']
        p,arc = traversal(fiber,reverse)
        pos,frame = np.asarray(item['pos']),np.asarray(item['frame'])
        local = lambda arcs: (interp_at(p,arc,np.clip(arcs,0,fiber.length))-pos) @ frame
        if 'seed_valid' not in item:
            item.update(observed_seed(pos,frame,item['hist_local'],item['hmask']))
        # Sample the label curve densely enough for reference membership, including
        # recovery offsets. Only visible references are eligible for supervision.
        extent = max(cfg.n_history,cfg.fine.depth*cfg.fine.spacing)+16
        curve = local(np.arange(max(0.,t-extent),min(fiber.length,t+extent)+1e-9,.25))
        item['identity_curve'] = curve
        self.identity_evidence(item, curve)
        if self.augment:
            item['photometric'], item['blur_sigma'] = photometric_draw(s, rng)
            jitter = float(np.clip(rng.normal(0., s.roll_jitter_deg), -s.roll_jitter_max_deg, s.roll_jitter_max_deg))
            item['roll_augmentation'] = float(np.deg2rad(jitter)+(np.pi if rng.random() < s.roll_flip_probability else 0.))
            self.apply_roll(item)
        item['identity_seed'] = int(rng.integers(2**63))
        item.setdefault('location_source',0)
        return item

    def prepare_sampling_feedback(self, items):
        """Advance geometry-only sampling feedback before planning another batch."""
        self.bank_targets(items,np.zeros(len(items),bool))
        for item in items:
            item['_sampling_feedback_prepared'] = True

    def bank_targets(self, items, decision_mask=None):
        """Foreign-path masks and coverage feedback; no contrastive point queries.

        Without neighbors (Paris 4) every foreign mask is empty: no known neighbor, as at an isolated AFV location.
        """
        cfg, bank = self.cfg, self.neighbors
        selected = np.ones(len(items), bool) if decision_mask is None else np.asarray(decision_mask, dtype=bool)
        shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        out = dict(location_source=np.zeros(len(items), np.float32),
                   foreign_components=np.zeros(len(items), np.float32))
        if selected.any() or not len(items):
            out['foreign'] = np.zeros((len(items), *shape), np.uint8)
        for j, item in enumerate(items):
            out['location_source'][j] = item.get('location_source', 0)
            if bank is None or 'identity_curve' not in item:
                continue
            feedback = (self.fibers is not None and item.get('source') == SOURCE['fresh']
                        and not item.get('_sampling_feedback_prepared',False))
            if not selected[j] and not feedback:
                continue
            found = bank.candidates(item, cfg.fine, self.sampling.rule, mask_crop=cfg.fine, rasterize=bool(selected[j]))
            if selected[j]:
                out['foreign'][j] = found['foreign']
            out['foreign_components'][j] = found['counts']['foreign_components']
            # Remember covered locations directly, independent of randomly selected
            # contrastive queries. Missing coverage remains unknown.
            ahead = found['local']
            if (feedback and len(ahead)
                    and ((ahead[:, 2] >= cfg.future_step)
                         & (ahead[:, 2] <= cfg.n_future*cfg.future_step)).any()):
                self.lateral.append(item['fiber_ref'])
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def __call__(self,items,vol):
        """Images and targets for every observation; per-item augmentation RNG is unchanged."""
        batch = self.observations(items, vol)
        if items:
            batch.update(collate_targets(items))
        if self.fibers is None and not self.augment and not any('identity_curve' in i for i in items):
            return batch
        batch.update(self.bank_targets(items))
        batch['identity_observable'] = torch.tensor([i.get('identity_observable',True) for i in items])
        batch['seed_present'] = batch['x']['seed_mask'].flatten()
        if self.augment:
            batch['blurred'] = torch.zeros(len(items))
            for j,item in enumerate(items):
                if 'photometric' not in item:
                    continue
                rng = np.random.default_rng(item['identity_seed']+1)
                augmentation = dict(blur_sigma=item.get('blur_sigma', 0.))
                batch['blurred'][j] = augmentation['blur_sigma'] > 0
                augment_image_pair(batch['x']['fine'][j], item['photometric'], rng, **augmentation)
        return batch


def observation_builder(cfg,**kwargs):
    return IdentityObservationBuilder(cfg,**kwargs)


class FiberTracer(ModelTracer):
    path_context = True

    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.observations = observation_builder(model.cfg)

    def build_inputs(self,pos,frames,hist,hmask,paths=None):
        items = [dict(pos=p,frame=f,hist_local=h,hmask=m,frame_policy=getattr(self, 'frame_policy', FRAME_POLICY))
                 for p,f,h,m in zip(pos,frames,hist,hmask)]
        if paths is not None:
            for item,path in zip(items,paths):
                item.update({k:path[k] for k in SEED_FIELDS if k in path})
                item['observed_path'] = path['observed_path']
                if 'fiber_family' in path:
                    item['fiber_family'] = path['fiber_family']
        def move(value):
            return {k: move(v) for k, v in value.items()} if isinstance(value, dict) else value.to(self.device)
        return move(self.observations.images(items,self.vol,self.pool))
