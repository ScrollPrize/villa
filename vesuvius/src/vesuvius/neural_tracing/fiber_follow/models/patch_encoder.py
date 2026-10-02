"""Overlapping 6x6x6 patch encoding on a stride-four spatial grid."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vesuvius.models.build.resblocks import BasicBlockD, StackedResidualBlocks

from vesuvius.neural_tracing.fiber_follow.models.model import AxialBlock, token_coordinates, device_vector


class ResidualPatchStem(nn.Module):
    """BasicBlockD patch embedding, adapted to the existing stride-four grid.

    Following PatchEmbed_deeper: one full-resolution block, then two residual
    downsampling stages at twice and four times the base width, with InstanceNorm
    and ReLU. stem_blocks specifies the
    depth of each downsampling stage. Both paths consume the sampled image
    directly on a complete stride-four grid. BasicBlockD mixes
    odd-kernel convolution and average-pool skip footprints, so its receptive
    fields are not identical to those of the original patch projection.
    """
    def __init__(self, cfg):
        super().__init__()
        c = cfg.stem_channels
        block_options = dict(conv_op=nn.Conv3d, kernel_size=3,
            norm_op=nn.InstanceNorm3d, norm_op_kwargs=dict(eps=1e-5, affine=True),
            nonlin=nn.ReLU, nonlin_kwargs=dict(inplace=True), block=BasicBlockD)
        self.input = StackedResidualBlocks(n_blocks=1, input_channels=cfg.input_channels,
            output_channels=c, initial_stride=1, conv_bias=True, **block_options)
        self.blocks = nn.Sequential(
            StackedResidualBlocks(n_blocks=cfg.stem_blocks, input_channels=c,
                output_channels=2*c, initial_stride=2, conv_bias=False, **block_options),
            StackedResidualBlocks(n_blocks=cfg.stem_blocks, input_channels=2*c,
                output_channels=4*c, initial_stride=2, conv_bias=False, **block_options))
        self.projection = nn.Conv3d(4*c, cfg.hidden, 1)

    def forward(self, image):
        return self.projection(self.blocks(self.input(image)))


class PatchEncoder(nn.Module):
    """Patch tokens with observed-path occupancy and a residual stem."""
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        # A one-voxel halo around each 4-cube gives two-voxel overlap.
        # Kernel center 2.5 minus padding 1 retains the old 1.5 offset.
        self.patch_projection = nn.Conv3d(cfg.input_channels, cfg.hidden, 6, stride=4, padding=1)
        self.stem = ResidualPatchStem(cfg)
        self.position = nn.Linear(3, cfg.hidden)
        self.condition = nn.Linear(3, cfg.hidden, bias=False)
        self.blocks = nn.ModuleList(AxialBlock(cfg.hidden, cfg.heads, ffn=cfg.encoder_ffn, rotary=True)
                                    for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.hidden)
        self.register_buffer('token_xyz', token_coordinates(cfg).reshape(-1,3), persistent=False)

    def encode(self, image, references, mask):
        tokens = self.patch_projection(image)
        if self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
            tokens = tokens+checkpoint(self.stem, image, use_reentrant=False)
        else:
            tokens = tokens+self.stem(image)
        tokens = tokens.permute(0,2,3,4,1)
        tokens = tokens+self.position(self.token_xyz/16).reshape(*self.cfg.token_shape,self.cfg.hidden).to(tokens.dtype)
        tokens = tokens+self.condition(self.conditioning(references,mask)).to(tokens.dtype)
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block,tokens,use_reentrant=False)
            else:
                tokens = block(tokens)
        deep = self.norm(tokens).permute(0,4,1,2,3)
        # Both path queries and scoring sample the same contextual token lattice.
        return deep, deep

    def forward(self, image, references, mask):
        fine, deep = self.encode(image, references, mask)
        return fine, deep, deep.flatten(2).transpose(1,2)

    def conditioning(self, references, mask):
        cfg = self.cfg
        points = torch.where(mask[...,None],references,0.).float()
        origin = device_vector(points, (-(cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(-cfg.fine.behind*cfg.fine.spacing,))
        offset = device_vector(points, tuple(reversed(cfg.token_offset)))*cfg.fine.spacing
        index = torch.round((points-origin-offset)/(device_vector(points, tuple(reversed(cfg.token_stride)))*cfg.fine.spacing)).long()
        d,y,x = cfg.token_shape
        index = torch.stack((index[...,0].clamp(0,x-1),index[...,1].clamp(0,y-1),index[...,2].clamp(0,d-1)),-1)
        flat = index[...,0]+x*(index[...,1]+y*index[...,2])
        ages = torch.arange(1,cfg.n_history+2,device=points.device).float()/cfg.n_history
        history = mask.clone()
        history[:,-1] = False
        seed = mask & ~history
        values = torch.stack((history.float(),history*ages[None],seed.float()),-1)
        rendered = points.new_zeros(len(points),d*y*x,3).scatter_add(1,flat[...,None].expand(-1,-1,3),values)
        count = rendered[...,:1]
        rendered = torch.cat((count.clamp_max(1),rendered[...,1:2]/count.clamp_min(1),rendered[...,2:].clamp_max(1)),-1)
        return rendered.reshape(len(points),d,y,x,3)
