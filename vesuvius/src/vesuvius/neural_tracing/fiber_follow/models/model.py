"""The follower contract shared by the three model types, the model registry and shared tensor helpers.

Model types: 'regression' and 'flow' (models/crop_transformer.py) and 'sequence' (models/sequence.py).
"""
from dataclasses import asdict, dataclass, field
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.retired import retire
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE, commit_prefix

MODEL_TYPES = ('regression', 'flow', 'sequence')


@dataclass
class FollowerConfig:
    """The crop, horizon and label contract read by data, losses and tracing; each model type adds its own sizes."""
    model_type: str = ''
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=144, width=104, behind=72, spacing=.5))
    hidden: int = 256
    n_future: int = 16
    future_step: float = 1.
    gate_plane: int | None = None  # plane whose confidence accepts a proposal (full gate, retries); None: the last plane
    n_history: int = 128
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    recurrent_refinement_steps: int = 0
    frame_checkpoint: str | None = None  # frozen heading/normal model used by crop builders
    frame_checkpoint_sha256: str | None = None

    def __post_init__(self):
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
        if min(self.hidden, self.n_future, self.n_history) < 1:
            raise ValueError('Positive dimensions required')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (c.width-1)*c.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if self.n_future*self.future_step > (c.depth-c.behind-1)*c.spacing:
            raise ValueError('Future horizon exceeds fine image')
        if self.gate_plane is not None and (type(self.gate_plane) is not int or not 1 <= self.gate_plane <= self.n_future):
            raise ValueError('Gate plane must be an integer in [1, n_future]')
        if not isinstance(self.recurrent_refinement_steps, int) or self.recurrent_refinement_steps < 0:
            raise ValueError('Recurrent refinement steps must be a nonnegative integer')

    @property
    def gate_horizon(self):
        """Planes whose confidence decides acceptance (full gate) and retries; the rest are predicted and supervised."""
        return self.n_future if self.gate_plane is None else self.gate_plane

    @property
    def path_plane_values(self):
        """Forward coordinates of the path planes (planes 1..n_future)."""
        import numpy as np
        return self.future_step*np.arange(1, self.n_future+1, dtype=np.float64)

    @property
    def input_channels(self):
        return 1

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2-self.patch_radius

    @property
    def token_shape(self):
        return tuple(math.ceil(n/s) for n,s in zip((self.fine.depth,self.fine.width,self.fine.width),self.token_stride))

    def to_dict(self):
        return asdict(self)


def config_class(model_type):
    if model_type == 'sequence':
        from .sequence import SequenceConfig
        return SequenceConfig
    if model_type == 'regression':
        from .crop_transformer import RegressionConfig
        return RegressionConfig
    if model_type == 'flow':
        from .crop_transformer import FlowConfig
        return FlowConfig
    raise ValueError(f'Unsupported model type: {model_type!r} (supported: {", ".join(MODEL_TYPES)})')


# Removed configuration fields and their only supported value: configurations recorded before the removal (checkpoints,
# run files) still hold them and load when they hold that value.
RETIRED_FIELDS = dict(activation_checkpointing=False, query_scale=None, flow_time_conditioning='adaln_zero',
                      flow_unknown_planes='own_path', flow_selection='retry', flow_zero_start=True,
                      flow_sample_threshold=0., flow_loss='pseudo_huber', flow_geometry_weight=0.)


def config_from_checkpoint(ck):
    """The model configuration recorded in a checkpoint."""
    return config_class(ck.get('model_type'))(**retire(ck['model_cfg'], RETIRED_FIELDS, 'model'))


def build_model(cfg):
    if cfg.model_type == 'sequence':
        from .sequence import SequenceFollower
        return SequenceFollower(cfg)
    if cfg.model_type == 'regression':
        from .crop_transformer import RegressionFollower
        return RegressionFollower(cfg)
    if cfg.model_type == 'flow':
        from .crop_transformer import FlowFollower
        return FlowFollower(cfg)
    raise ValueError(f'Unsupported model type: {cfg.model_type!r}')


def select_refinement(output, cfg, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
    """Longest acceptable prefix, then confidence at its end, then earlier pass.

    If every proposal stops, select the best first-point confidence among valid
    connections; the unchanged commit gate still stops. Selection never splices
    paths or transfers one proposal's confidence to another proposal.
    Proposals after the first accepted one are ignored, so later proposals act only as retries.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must lie in [0, 1]')
    window = cfg.n_future if n_commit is None else n_commit
    curves, confidence = output['refinement_points'], output['refinement_confidence'].detach()
    counts, allowed = commit_prefix(curves, confidence, confidence_threshold, window, cfg.max_recovery_distance)
    gate = cfg.gate_horizon  # acceptance through the gate plane, as in tracing
    accepted = (confidence[..., gate-1] >= confidence_threshold) & allowed & output['refinement_mask']
    prior_accept = torch.cat((torch.zeros_like(accepted[:, :1]), accepted[:, :-1]), 1).long().cumsum(1) > 0
    valid = output['refinement_mask'] & ~prior_accept
    longest = counts.masked_fill(~valid, -1).max(-1, keepdim=True).values
    score = confidence.gather(-1, (counts-1).clamp_min(0)[..., None]).squeeze(-1)
    score = score.nan_to_num(nan=-torch.inf).masked_fill((counts != longest) | ~allowed | ~valid, -torch.inf)
    selected = score.argmax(-1)
    index = torch.arange(len(curves), device=curves.device)
    result = dict(output, selected_refinement=selected)
    for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
        result[name] = output['refinement_'+name][index, selected]
    return result


def device_vector(like, values):
    """``like.new_tensor(values)`` without a host-to-device copy, which would
    synchronize with the queued GPU work (these run in eager losses and tracing)."""
    return torch.stack([like.new_full((), v) for v in values])


def feature_grid(points, crop, shape, stride=1):
    """Exact physical coordinates for a feature lattice (stride in zyx); cell i is centred on input sample stride*i."""
    if isinstance(stride, (int, float)):
        stride = (stride,)*3
    size = device_vector(points, tuple(reversed(shape)))
    scale = device_vector(points, tuple(reversed(stride)))*crop.spacing
    origin = device_vector(points, (-(crop.width-1)*crop.spacing/2,
                                    -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    return 2*(points-origin)/(scale*(size-1).clamp_min(1))-1


def crop_support(points, crop):
    """Input support, independent of the coarser contextual-feature lattice."""
    lo = device_vector(points, (-(crop.width-1)*crop.spacing/2,)*2+(-crop.behind*crop.spacing,))
    hi = device_vector(points, ((crop.width-1)*crop.spacing/2,)*2+((crop.depth-1-crop.behind)*crop.spacing,))
    return torch.isfinite(points).all(-1) & (points >= lo).all(-1) & (points <= hi).all(-1)


def sample_features(features, points, crop, stride=1):
    supported = crop_support(points, crop)
    points = torch.where(supported[...,None], points.float(), 0.)
    grid = feature_grid(points, crop, features.shape[-3:], stride)
    # Border extrapolation covers the half-token margins of the input crop.
    values = F.grid_sample(features.float(), grid[:,:,None,None], padding_mode='border', align_corners=True)
    values = values[:,:,:,0,0].transpose(1,2)
    return torch.where(supported[...,None],values,0.), supported


def reference_points(x, hist, hmask, crop):
    """Recent observed-path points and the seed (B, n_history+1, 3), zeroed and masked outside the crop."""
    seed = x.get('seed',hist.new_zeros(len(hist),1,3))
    seed_mask = x.get('seed_mask',hmask.new_zeros(len(hist),1)).bool()
    references = torch.cat((hist,seed),1)
    mask = torch.cat((hmask.bool(),seed_mask),1) & crop_support(references,crop)
    references = torch.where(mask[...,None],references.float(),0.)
    return references, mask


REFERENCE_METADATA = 8  # position (3), tangent (3), log age, seed role


def reference_metadata(x, hist, references, mask):
    """Per-reference metadata (B, n_history+1, REFERENCE_METADATA): position, seed tangent, age and seed role."""
    ages = torch.arange(1,references.shape[1]+1,device=hist.device).float()[None].expand(len(hist),-1).clone()
    ages[:,-1] = x.get('seed_age',hist.new_zeros(len(hist))).reshape(-1)
    anchor = torch.zeros_like(ages)
    anchor[:,-1] = 1.
    tangent = torch.zeros_like(references)
    tangent[:,-1] = x.get('seed_tangent',hist.new_zeros(len(hist),3))
    metadata = torch.cat((references/16,tangent,torch.log1p(ages.clamp(0,2048))[...,None]/math.log(2049),anchor[...,None]),-1)
    return torch.where(mask[...,None],metadata,0.)


def future_points(raw, planes, cfg):
    """Plane points (B, P, 3) from raw lateral outputs (B, P, 2): bounded to the crop, the first connection to the
    recovery distance, at forward coordinates ``planes`` (P,)."""
    lateral = cfg.lateral_limit*torch.tanh(raw.float())
    first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
    first = lateral[:, :1]
    first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    lateral = torch.cat((first, lateral[:, 1:]), 1)
    return torch.cat((lateral, planes[None, :, None].expand(len(raw), -1, -1)), -1)


def proposal_output(initial, refinements, scores, valid):
    """The shared all-proposal output contract: (B, A, ...) attempts, their survival scores and validity."""
    out = dict(initial_points=initial, refinement_points=torch.stack(refinements, 1),
               refinement_mask=torch.stack(valid, 1))
    out.update({'refinement_'+name: torch.stack([score[i] for score in scores], 1)
                for i, name in enumerate(('hazard_logits', 'confidence_logits', 'confidence'))})
    return out


def token_coordinates(cfg):
    """Token centers in crop-local XYZ."""
    d,y,x = torch.meshgrid(*(torch.arange(n).float() for n in cfg.token_shape),indexing='ij')
    xyz = torch.stack((x,y,d),-1)
    xyz = xyz*xyz.new_tensor(tuple(reversed(cfg.token_stride)))*cfg.fine.spacing
    return xyz-xyz.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))
