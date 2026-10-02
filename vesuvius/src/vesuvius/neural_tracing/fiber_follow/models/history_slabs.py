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
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_frame, reframe_item, FRAME_POLICY

SLOTS = 8
SLAB = CropSpec(depth=8, width=65, behind=4, spacing=.5,
                history_render='segments', history_sigma=1.)
SAMPLING_REVISION = 'live_observed_slabs_ct_transverse_v3'
PATH_SAMPLES = 3


def slab_path_samples(item, slab):
    """Sample the actual committed polyline near this observation, never GT."""
    path = observed_path(item)
    arc = arclength(path)
    at = arc[-1]-slab['age']+np.array([-1., 0., 1.])
    points = interp_at(path, arc, np.clip(at, 0., arc[-1]))
    before = interp_at(path, arc, np.clip(at-.25, 0., arc[-1]))
    after = interp_at(path, arc, np.clip(at+.25, 0., arc[-1]))
    tangent = (after-before) @ slab['frame']
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent/np.maximum(norm, 1e-8)
    points = (points-slab['pos']) @ slab['frame']
    lo = np.array([-16., -16., -2.])
    hi = np.array([16., 16., 1.5])
    valid = (at >= -1e-8) & (at <= arc[-1]+1e-8) & (norm[:, 0] > 1e-8)
    valid &= ((points >= lo-1e-6) & (points <= hi+1e-6)).all(-1)
    return points, tangent, valid




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
    from vesuvius.neural_tracing.fiber_follow.data.data import training_state_allowed
    return all(training_state_allowed(slab, SLAB, band) for slab in slab_layout(item))


def load_slabs(items, vol, cfg, pool=None):
    from vesuvius.neural_tracing.fiber_follow.data.observations import visible_points
    from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops, empty_image_batch
    started = time.perf_counter()
    def orient(item):
        # Frames chain within an item (sign continuity), never across items.
        layout = slab_layout(item)
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
        return layout
    layouts = list(map(orient, items) if pool is None else pool.map(orient, items))
    flat = [slab for layout in layouts for slab in layout]
    images = scalar_crops(flat, vol, SLAB, pool, presence=False)
    grid = torch.from_numpy(crop_local_grid(SLAB)).float()
    output = empty_image_batch((len(items), SLOTS, 2, 8, 65, 65)).zero_()
    valid = torch.zeros(len(items), SLOTS, dtype=torch.bool)
    # Translation (3), relative rotation (9), log age (1), seed role (1).
    pose = torch.zeros(len(items), SLOTS, 14)
    ages, overlap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    path_inputs = {}
    if cfg.history_path_tokens:
        path_inputs = dict(history_path_points=torch.zeros(len(items), SLOTS, PATH_SAMPLES, 3),
                           history_path_tangents=torch.zeros(len(items), SLOTS, PATH_SAMPLES, 3),
                           history_path_valid=torch.zeros(len(items), SLOTS, PATH_SAMPLES, dtype=torch.bool))
    frame_source = torch.full((len(items), SLOTS), -1, dtype=torch.int64)
    frame_energy, frame_gap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    offsets = np.cumsum([0]+[len(layout) for layout in layouts])

    inference = torch.is_inference_mode_enabled()

    def fill(row):
        # Writes only this row of each preallocated tensor. Pool threads do not inherit
        # the caller's inference mode, which in-place writes to its tensors require.
        with torch.inference_mode(inference):
            item, layout = items[row], layouts[row]
            for slot, slab in enumerate(layout):
                cursor = int(offsets[row])+slot
                if cfg.history_path_tokens:
                    for key, value in zip(path_inputs, slab_path_samples(item, slab)):
                        path_inputs[key][row, slot] = torch.as_tensor(value)
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
    if pool is None:
        for row in range(len(items)):
            fill(row)
    else:
        list(pool.map(fill, range(len(items))))
    elapsed = torch.full((len(items),), (time.perf_counter()-started)/max(1, len(items)))
    return dict(history_slabs=output, history_valid=valid, history_pose=pose,
                history_ages=ages, history_overlap=overlap, history_load_seconds=elapsed,
                history_frame_source=frame_source, history_frame_energy=frame_energy, history_frame_gap=frame_gap,
                **path_inputs)


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
        widths = (32, 64, 128)
        strides = ((1, 1, 1), (2, 2, 2), (2, 2, 2))
        shape = (SLAB.depth, SLAB.width, SLAB.width)
        for channels, stride in zip(widths, strides):
            stages.extend((nn.Conv3d(previous, channels, 3, stride=stride, padding=1),
                BasicBlockD(nn.Conv3d, channels, channels, 3, 1,
                    norm_op=SlabInstanceNorm, norm_op_kwargs=dict(eps=1e-5, affine=True),
                    nonlin=nn.LeakyReLU, nonlin_kwargs=dict(negative_slope=.01, inplace=True))))
            previous = channels
            shape = tuple(math.ceil(n/s) for n, s in zip(shape, stride))
        self.token_shape = shape
        self.spatial_tokens_per_slab = math.prod(shape)
        self.path_tokens = PATH_SAMPLES if cfg.history_path_tokens else 0
        self.tokens_per_slab = self.spatial_tokens_per_slab+self.path_tokens
        self.convolution = nn.Sequential(*stages)
        self.projection = nn.Linear(previous, cfg.hidden)
        self.position = nn.Linear(3, cfg.hidden)
        self.pose = nn.Sequential(nn.Linear(14, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.slot = nn.Embedding(SLOTS, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden)
        z, y, x = torch.meshgrid(*(torch.arange(n) for n in shape), indexing='ij')
        xyz = torch.stack((x/(shape[2]-1), y/(shape[1]-1), z/(shape[0]-1)), -1)
        self.register_buffer('xyz', xyz.reshape(self.spatial_tokens_per_slab, 3), persistent=False)
        if self.path_tokens:
            # Center and four lateral neighbors, position and observed tangent.
            self.path_projection = nn.Sequential(nn.Linear(5*previous+6, cfg.hidden), nn.SiLU(),
                                                 nn.Linear(cfg.hidden, cfg.hidden))
            nn.init.zeros_(self.path_projection[-1].weight)
            nn.init.zeros_(self.path_projection[-1].bias)
            self.register_buffer('path_stencil', torch.tensor([[0.,0.,0.],[-1.,0.,0.],
                [1.,0.,0.],[0.,-1.,0.],[0.,1.,0.]]), persistent=False)

    def forward(self, slabs, valid, pose, *, path_points=None, path_tangents=None, path_valid=None):
        b, slots = valid.shape
        indices = valid.flatten().nonzero().flatten()
        selected = slabs.flatten(0, 1).index_select(0, indices)
        # SlabInstanceNorm supports empty batches: no dummy slab work.
        volume = self.convolution(selected)
        features = volume.flatten(2).transpose(1, 2)
        projected = self.projection(features)
        positions = self.position(self.xyz)[None].expand(len(indices), -1, -1)
        token_valid = valid[:, :, None].expand(-1, -1, self.spatial_tokens_per_slab)
        if self.path_tokens:
            from vesuvius.neural_tracing.fiber_follow.models.model import feature_grid
            points = path_points.flatten(0, 1).index_select(0, indices)
            tangents = path_tangents.flatten(0, 1).index_select(0, indices)
            supported = path_valid.flatten(0, 1).index_select(0, indices)
            points = torch.where(supported[..., None], points, 0.).float()
            tangents = torch.where(supported[..., None], tangents, 0.).float()
            grid = feature_grid(points[:, :, None]+self.path_stencil, SLAB, self.token_shape, stride=4)
            # Border sampling only extends the feature lattice; path validity is
            # separately checked against the physical CT crop by the loader.
            local = F.grid_sample(volume.float(), grid[:, :, :, None], padding_mode='border',
                                  align_corners=True).squeeze(-1).permute(0, 2, 3, 1).to(projected.dtype)
            extra = self.projection(local[:, :, 0])+self.path_projection(torch.cat(
                (local.flatten(2), (points/16.).to(local.dtype), tangents.to(local.dtype)), -1))
            projected = torch.cat((projected, extra), 1)
            xyz = (points+points.new_tensor([16.,16.,2.]))/points.new_tensor([32.,32.,3.5])
            positions = torch.cat((positions, self.position(xyz)), 1)
            token_valid = torch.cat((token_valid, valid[..., None] & path_valid), -1)
        projected = projected+positions
        tokens = projected.new_zeros(b*slots, self.tokens_per_slab, projected.shape[-1])
        tokens = tokens.index_copy(0, indices, projected).reshape(b, slots, self.tokens_per_slab, -1)
        metadata = self.pose(torch.where(valid[..., None], pose, 0.))
        tokens = self.norm(tokens+metadata[:, :, None]+self.slot.weight[None, :, None])
        padding = (~token_valid).reshape(b, -1)
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
        from vesuvius.neural_tracing.fiber_follow.models.model import project_attention_memory
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
