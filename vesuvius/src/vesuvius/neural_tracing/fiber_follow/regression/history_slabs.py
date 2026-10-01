"""Live causal observations. No annotation, cached feature, or writer inputs."""
import time
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vesuvius.models.build.resblocks import BasicBlockD

from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec, arclength, interp_at, frame_from_heading, normalize, crop_local_grid,
    render_history,
)

from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path
from vesuvius.neural_tracing.fiber_follow.shared.heading import ct_frame, reframe_item, FRAME_POLICY

SLOTS = 8
SLAB = CropSpec(depth=8, width=65, behind=4, spacing=.5,
                history_render='segments', history_sigma=1.)
SAMPLING_REVISION = 'live_observed_slabs_ct_transverse_v3'




def selected_arcs(length):
    """Seed, up to four older samples, then available 96/64/32 anchors."""
    older = max(0., length-96.)
    count = min(4, max(0, int(np.floor(older/32.))-1))
    values = [0.]
    if count:
        values.extend(older*np.arange(1, count+1)/(count+1))
    for age in (96., 64., 32.):
        at = length-age
        if at-values[-1] >= 32.-1e-8:
            values.append(at)
    return np.asarray(values)


def fitted_heading(path, arc, at, seed_heading):
    """Four points, two voxels apart, regressed against traversal arclength."""
    if arc[-1] >= 6.:
        start = np.clip(at-3., 0., arc[-1]-6.)
        points = interp_at(path, arc, start+np.arange(4)*2.)
        heading = (points*np.array([-3., -1., 1., 3.])[:, None]).sum(0)
        if np.isfinite(heading).all() and np.linalg.norm(heading) > 1e-8:
            return normalize(heading)
    return normalize(np.asarray(seed_heading, dtype=np.float64))


def slab_layout(item):
    path = observed_path(item)
    arc = arclength(path)
    heading = item['seed_tangent'] if item.get('seed_valid', False) else np.asarray(item['frame'])[:, 2]
    if not np.isfinite(heading).all() or np.linalg.norm(heading) < 1e-8:
        raise ValueError('A finite nonzero seed heading is required')
    slabs = []
    for slot, at in enumerate(selected_arcs(arc[-1])):
        pos = interp_at(path, arc, np.array([at]))[0]
        frame = frame_from_heading(fitted_heading(path, arc, at, heading))
        lo, hi = max(0., at-8.), min(arc[-1], at+8.)
        # Include exact boundaries and every committed vertex between them.
        local_arc = np.unique(np.r_[lo, arc[(arc > lo) & (arc < hi)], at, hi])
        local = (interp_at(path, arc, local_arc)-pos) @ frame
        # Renderer joins origin to its first valid point. A masked leading
        # point breaks that artificial chord; real segments remain connected.
        local = np.concatenate((np.zeros((1, 3)), local))
        slabs.append(dict(pos=pos, frame=frame, hist_local=local,
                          hmask=np.r_[0., np.ones(len(local)-1)],
                          age=arc[-1]-at, seed=slot == 0))
    return slabs


def slabs_allowed(item, band):
    from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed
    return all(training_state_allowed(slab, SLAB, band) for slab in slab_layout(item))


def load_slabs(items, vol, cfg, pool=None):
    from .data import visible_points
    from vesuvius.neural_tracing.fiber_follow.shared.crop_sampling import scalar_crops, empty_image_batch
    started = time.perf_counter()
    layouts = [slab_layout(item) for item in items]
    for item, layout in zip(items, layouts):
        previous = None
        for slab in layout:
            diagnostics = {}
            fallback = item['frame'] if item.get('frame_policy') == FRAME_POLICY else None
            frame = ct_frame(vol, slab['pos'], slab['frame'][:, 2], previous,
                             fallback=fallback, diagnostics=diagnostics)
            reframe_item(slab, frame)
            slab['frame_policy'] = FRAME_POLICY
            slab['ct_frame_diagnostics'] = diagnostics
            previous = frame
        item['_sampled_slabs'] = layout
    flat = [slab for layout in layouts for slab in layout]
    images = scalar_crops(flat, vol, SLAB, pool, presence=False)
    grid = torch.from_numpy(crop_local_grid(SLAB)).float()
    output = empty_image_batch((len(items), SLOTS, 2, 8, 65, 65)).zero_()
    valid = torch.zeros(len(items), SLOTS, dtype=torch.bool)
    # Translation (3), relative rotation (9), log age (1), seed role (1).
    pose = torch.zeros(len(items), SLOTS, 14)
    ages, overlap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    frame_source = torch.full((len(items), SLOTS), -1, dtype=torch.int64)
    frame_energy, frame_gap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    cursor = 0
    for row, (item, layout) in enumerate(zip(items, layouts)):
        for slot, slab in enumerate(layout):
            heat = render_history(torch.as_tensor(slab['hist_local'])[None].float(),
                                  torch.as_tensor(slab['hmask'])[None], grid, 1., 'segments')[0]
            output[row, slot] = torch.cat((images[cursor], heat), 0)
            valid[row, slot] = True
            quality = slab['ct_frame_diagnostics']
            frame_source[row, slot] = quality['source']
            frame_energy[row, slot], frame_gap[row, slot] = quality['energy'], quality['gap']
            relative = (slab['pos']-item['pos']) @ item['frame']
            rotation = np.asarray(item['frame']).T @ slab['frame']
            pose[row, slot] = torch.tensor(np.r_[relative/128., rotation.ravel(),
                                                   np.log1p(slab['age'])/8., float(slab['seed'])])
            ages[row, slot] = slab['age']
            world = grid.numpy().reshape(-1, 3) @ slab['frame'].T+slab['pos']
            overlap[row, slot] = float(visible_points((world-item['pos']) @ item['frame'], cfg.fine).mean())
            cursor += 1
    elapsed = torch.full((len(items),), (time.perf_counter()-started)/max(1, len(items)))
    return dict(history_slabs=output, history_valid=valid, history_pose=pose,
                history_ages=ages, history_overlap=overlap, history_load_seconds=elapsed,
                history_frame_source=frame_source, history_frame_energy=frame_energy, history_frame_gap=frame_gap)


class SlabInstanceNorm(nn.InstanceNorm3d):
    """Affine instance normalization that also supports zero valid slabs.

    One group per channel computes the same per-instance spatial statistics
    without native InstanceNorm's empty-batch failure or running statistics.
    This also avoids branching on a dynamic batch size inside torch.compile.
    """
    def __init__(self, channels, eps=1e-5, affine=True):
        super().__init__(channels, eps=eps, affine=affine, track_running_stats=False)

    def forward(self, x):
        return F.group_norm(x, self.num_features, self.weight, self.bias, self.eps)


class HistoryEncoder(nn.Module):
    """The sole history interface: spatial feature tokens and padding mask."""
    def __init__(self, cfg):
        super().__init__()
        stages, previous = [], 2  # CT and observed-path heatmap
        fine = cfg.history_encoder == 'fine'
        widths = (32, 64, 128) if fine else (8, 16, 32)
        strides = ((1, 1, 1), (2, 2, 2), (2, 2, 2)) if fine else (
            (1, 2, 2), (2, 2, 2), (2, 2, 2))
        shape = (SLAB.depth, SLAB.width, SLAB.width)
        for channels, stride in zip(widths, strides):
            stages.extend((nn.Conv3d(previous, channels, 3, stride=stride, padding=1),
                BasicBlockD(nn.Conv3d, channels, channels, 3, 1,
                    norm_op=SlabInstanceNorm, norm_op_kwargs=dict(eps=1e-5, affine=True),
                    nonlin=nn.LeakyReLU, nonlin_kwargs=dict(negative_slope=.01, inplace=True))))
            previous = channels
            shape = tuple(math.ceil(n/s) for n, s in zip(shape, stride))
        self.token_shape = shape
        self.tokens_per_slab = math.prod(shape)
        self.convolution = nn.Sequential(*stages)
        self.projection = nn.Linear(previous, cfg.hidden)
        self.position = nn.Linear(3, cfg.hidden)
        self.pose = nn.Sequential(nn.Linear(14, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.slot = nn.Embedding(SLOTS, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden)
        z, y, x = torch.meshgrid(*(torch.arange(n) for n in shape), indexing='ij')
        xyz = torch.stack((x/(shape[2]-1), y/(shape[1]-1), z/(shape[0]-1)), -1)
        self.register_buffer('xyz', xyz.reshape(self.tokens_per_slab, 3), persistent=False)

    def forward(self, slabs, valid, pose):
        b, slots = valid.shape
        indices = valid.flatten().nonzero().flatten()
        selected = slabs.flatten(0, 1).index_select(0, indices)
        # SlabInstanceNorm supports empty batches: no dummy slab work.
        features = self.convolution(selected).flatten(2).transpose(1, 2)
        projected = self.projection(features)
        tokens = projected.new_zeros(b*slots, self.tokens_per_slab, projected.shape[-1])
        tokens = tokens.index_copy(0, indices, projected).reshape(b, slots, self.tokens_per_slab, -1)
        metadata = self.pose(torch.where(valid[..., None], pose, 0.))
        tokens = self.norm(tokens+self.position(self.xyz)[None, None]
                           +metadata[:, :, None]+self.slot.weight[None, :, None])
        padding = (~valid[:, :, None]).expand(-1, -1, self.tokens_per_slab).reshape(b, -1)
        tokens = tokens.flatten(1, 2).masked_fill(padding[..., None], 0.)
        return tokens, padding


class HistoryAttention(nn.Module):
    """Residual attention shared across one head's decoder layers, without a gate."""
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)

    def project_memory(self, tokens, padding):
        """Attached K/V shared by all layers and attempts within one decision."""
        from .model import project_attention_memory
        k, v = project_attention_memory(self.attention, tokens)
        # Supply one finite key for empty rows and zero their contribution,
        # including the output bias. No NaN softmax or suppressive learned gate.
        empty = padding.all(-1)
        allowed = (~padding | empty[:, None])[:, None, None, :]
        return k, v, allowed, empty

    def forward_cached(self, query, k, v, allowed, empty):
        attn = self.attention
        width, heads = attn.embed_dim, attn.num_heads
        q = F.linear(self.norm(query), attn.in_proj_weight[:width], attn.in_proj_bias[:width])
        q = q.reshape(len(query), -1, heads, width//heads).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        value = value.transpose(1, 2).reshape(len(query), -1, width)
        value = attn.out_proj(value)
        return query+torch.where(empty[:, None, None], 0., value)

    def forward(self, query, tokens, padding):
        return self.forward_cached(query, *self.project_memory(tokens, padding))
