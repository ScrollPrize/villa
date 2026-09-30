"""Overlapping 6x6x6 patch encoding on a stride-four spatial grid."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import AxialEncoder, AxialBlock, token_coordinates


def pad_to_patch_grid(image):
    """Complete the last stride-four cell on the high edge of each axis."""
    d,y,x = image.shape[-3:]
    return F.pad(image, (0,(-x)%4,0,(-y)%4,0,(-d)%4))


def patchify(image):
    """Flatten ordered 4x4x4 samples; pad only the high edge of each axis."""
    b,c,d,y,x = image.shape
    image = pad_to_patch_grid(image)
    nd,ny,nx = ((n+3)//4 for n in (d,y,x))
    return image.reshape(b,c,nd,4,ny,4,nx,4).permute(
        0,2,4,6,3,5,7,1).reshape(b,nd,ny,nx,64*c)


def unpatchify(tokens, shape):
    """3D pixel shuffle, restoring the original crop extent after padding."""
    b,d,y,x,cp = tokens.shape
    result = tokens.reshape(b,d,y,x,4,4,4,cp//64).permute(
        0,7,1,4,2,5,3,6).reshape(b,cp//64,4*d,4*y,4*x)
    return result[:, :, :shape[0], :shape[1], :shape[2]]


class PatchShuffleEncoder(AxialEncoder):
    """Share physical history conditioning with the convolutional encoder."""
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
        # A one-voxel halo around each 4-cube gives two-voxel overlap.
        # Kernel center 2.5 minus padding 1 retains the old 1.5 offset.
        self.patch_projection = nn.Conv3d(cfg.input_channels, cfg.hidden, 6, stride=4, padding=1)
        self.position = nn.Linear(3, cfg.hidden)
        self.condition = nn.Linear(3, cfg.hidden, bias=False)
        self.blocks = nn.ModuleList(AxialBlock(cfg.hidden, cfg.heads, local_convolution=False)
                                    for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.hidden)
        self.reconstruction = None if cfg.token_only else nn.Linear(cfg.hidden, 64*cfg.channels)
        self.register_buffer('token_xyz', token_coordinates(cfg).reshape(-1,3), persistent=False)

    def encode(self, image, references, mask):
        tokens = self.patch_projection(pad_to_patch_grid(image)).permute(0,2,3,4,1)
        tokens = tokens+self.position(self.token_xyz/16).reshape(*self.cfg.token_shape,self.cfg.hidden).to(tokens.dtype)
        tokens = tokens+self.condition(self.conditioning(references,mask)).to(tokens.dtype)
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block,tokens,use_reentrant=False)
            else:
                tokens = block(tokens)
        deep = self.norm(tokens).permute(0,4,1,2,3)
        # Token-only mode returns this same coarse lattice for both consumers.
        return self.decode(None, deep), deep

    def decode(self, fine, deep):
        if self.cfg.token_only:
            return deep
        return unpatchify(self.reconstruction(deep.permute(0,2,3,4,1)), self.shape)

    def forward(self, image, references, mask):
        fine, deep = self.encode(image, references, mask)
        return fine, deep, deep.flatten(2).transpose(1,2)
