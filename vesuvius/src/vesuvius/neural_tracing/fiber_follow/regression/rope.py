"""Dinovol's fixed 3D RoPE applied to the lines of patch-grid axial attention."""
import torch
from torch import nn

from vesuvius.models.build.pretrained_backbones.rope import (
    RopePositionEmbedding, apply_rotary_embedding,
)


class AxialRoPE3D(nn.Module):
    """Rotate complete XYZ channel groups, retaining any remainder unchanged.

    The patch grid is isotropic. Normalize all axes by its longest dimension,
    preserving relative distances across axes. Coordinates constant along a
    line cancel in Q/K dot products, so only its varying coordinate is needed.
    This avoids constructing a full-volume sine/cosine table for each layer.
    """
    def __init__(self, head_dim):
        super().__init__()
        self.rotary_dim = 6*(head_dim//6)
        if self.rotary_dim == 0:
            raise ValueError('3D RoPE requires at least six channels per attention head')
        self.embedding = RopePositionEmbedding(self.rotary_dim, ndim=3, base=100.,
            normalize_coords='max', shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=torch.float32)

    def forward(self, q, k, spatial_shape, axis):
        coordinate = 2*(torch.arange(spatial_shape[axis], device=q.device,
            dtype=self.embedding.periods.dtype)+.5)/max(spatial_shape)-1
        zero = torch.zeros_like(coordinate)
        coords = torch.stack([coordinate if dim == axis else zero for dim in range(3)], -1)
        embedding = self.embedding.get_embed_from_coords(coords)
        rotated = []
        for value in (q, k):
            prefix = apply_rotary_embedding(value[..., :self.rotary_dim], embedding)
            rotated.append(torch.cat((prefix, value[..., self.rotary_dim:]), -1))
        return rotated[0], rotated[1]
