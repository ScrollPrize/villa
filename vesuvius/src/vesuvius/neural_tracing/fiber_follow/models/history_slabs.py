"""Shared historical slab encoder and attention."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from vesuvius.models.build.resblocks import BasicBlockD
from ..data.history_slabs import SLAB, SLOTS, PATH_SAMPLES

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
        self.path_tokens = PATH_SAMPLES
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
        # Center and four lateral neighbors, position and observed tangent.
        self.path_projection = nn.Sequential(nn.Linear(5*previous+6, cfg.hidden), nn.SiLU(),
                                             nn.Linear(cfg.hidden, cfg.hidden))
        nn.init.zeros_(self.path_projection[-1].weight)
        nn.init.zeros_(self.path_projection[-1].bias)
        self.register_buffer('path_stencil', torch.tensor([[0.,0.,0.],[-1.,0.,0.],
            [1.,0.,0.],[0.,-1.,0.],[0.,1.,0.]]), persistent=False)

    def forward(self, slabs, valid, pose, *, path_points, path_tangents, path_valid):
        b, slots = valid.shape
        indices = valid.flatten().nonzero().flatten()
        selected = slabs.flatten(0, 1).index_select(0, indices)
        # SlabInstanceNorm supports empty batches: no dummy slab work.
        volume = self.convolution(selected)
        features = volume.flatten(2).transpose(1, 2)
        projected = self.projection(features)
        positions = self.position(self.xyz)[None].expand(len(indices), -1, -1)
        token_valid = valid[:, :, None].expand(-1, -1, self.spatial_tokens_per_slab)
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
        return query+self.attend(self.norm(query), k, v, allowed, empty)

    def attend(self, normed, k, v, allowed, empty):
        """The residual branch for already normalized queries; empty rows contribute zero."""
        attn = self.attention
        width, heads = attn.embed_dim, attn.num_heads
        q = F.linear(normed, attn.in_proj_weight[:width], attn.in_proj_bias[:width])
        q = q.reshape(len(normed), -1, heads, width//heads).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        value = value.transpose(1, 2).reshape(len(normed), -1, width)
        value = attn.out_proj(value)
        return torch.where(empty[:, None, None], 0., value)

    def forward(self, query, tokens, padding):
        return self.forward_cached(query, *self.project_memory(tokens, padding))
