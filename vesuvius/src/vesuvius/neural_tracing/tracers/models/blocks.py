"""Building blocks shared by the tracers' models: the residual crop CNN (feature map, per-cell tokens or one token per
crop), the two-layer MLP of the token embeddings, the pre-norm transformer layer's weights and head split/merge, and
plain self-attention over a token set (the sheet tracer). The crop transformer (models/crop_transformer.py) and the
sequence follower (models/sequence.py) subclass the layer to define which tokens attend to which."""
import math

import torch
from torch import nn
import torch.nn.functional as F

LOGIT_SCALE_LR_SCALE = 10.  # LR multiple of the attention logit scales' optimizer group (no weight decay)

from vesuvius.models.build.resblocks import BasicBlockD, StackedResidualBlocks


def _group_norm(channels: int, **kwargs) -> nn.GroupNorm:
    """GroupNorm with 4 groups (1 when the channels do not divide by 4)."""
    return nn.GroupNorm(4 if channels % 4 == 0 else 1, channels, **kwargs)


CNN_NORMS = dict(instance=(nn.InstanceNorm3d, dict(eps=1e-5, affine=True)), group=(_group_norm, {}))
# Encoder activations: name -> (module, keyword arguments); 'leaky_relu' has nnU-Net's slope.
CNN_ACTIVATIONS = dict(relu=(nn.ReLU, dict(inplace=True)), leaky_relu=(nn.LeakyReLU, dict(negative_slope=1e-2, inplace=True)),
                       gelu=(nn.GELU, {}))


class CropCNN(nn.Module):
    """Residual CNN stages of ResNet-D basic blocks (channels, initial stride, blocks per stage) over a crop, and
    optionally a head that turns its feature map into tokens of width ``dim``.

    ``stem``: first one full-resolution convolution (norm, activation) to the first stage's channels, as nnU-Net's
    residual encoder. ``norm``: 'instance' (InstanceNorm3d, affine) or 'group' (GroupNorm, 4 groups); ``activation``:
    a CNN_ACTIVATIONS name; ``conv_bias``: 'first' (only the first stage's convolutions carry biases) or 'all'.
    ``head``: None (``forward`` returns the feature map), 'cells' (one token per feature-map cell, (B, cells, dim), a
    Linear of the channels) or 'crop' (one token per crop, (B, dim): an optional 1x1 conv to ``bottleneck`` channels
    with the same norm and activation, then flatten and a Linear; needs the ``crop`` size, which must divide by 2 per
    stride-2 stage). ``features`` and ``tokens`` run the two halves separately, for models that read the feature map
    as well. ``in_channels``: input channels of the crop (default one, e.g. CT). ``context`` > 0: a stride-2 residual
    block of that many channels on the last stage's map, upsampled (trilinear) to it and added through a 1x1
    convolution that starts at zero: about twice the receptive field for an eighth of a stage's positions."""
    def __init__(self, channels, strides, blocks, *, stem=False, norm='instance', activation='relu', conv_bias='first',
                 head=None, dim=None, crop=None, bottleneck=0, in_channels=1, context=0):
        super().__init__()
        if conv_bias not in ('first', 'all') or head not in (None, 'cells', 'crop') or (head and not dim):
            raise ValueError(f'CropCNN: conv_bias {conv_bias!r}, head {head!r}, dim {dim!r}')
        norm_op, norm_kwargs = CNN_NORMS[norm]
        nonlin_op, nonlin_kwargs = CNN_ACTIVATIONS[activation]
        options = dict(conv_op=nn.Conv3d, kernel_size=3, norm_op=norm_op, norm_op_kwargs=norm_kwargs, nonlin=nonlin_op,
                       nonlin_kwargs=nonlin_kwargs, block=BasicBlockD)
        if stem:
            self.stem = nn.Sequential(nn.Conv3d(in_channels, channels[0], 3, padding=1),
                                      norm_op(channels[0], **norm_kwargs), nonlin_op(**nonlin_kwargs))
        inputs = (channels[0] if stem else in_channels, *channels[:-1])
        self.stages = nn.ModuleList(
            StackedResidualBlocks(n_blocks=n, input_channels=i, output_channels=c, initial_stride=s,
                                  conv_bias=conv_bias == 'all' or not index, **options)
            for index, (i, c, s, n) in enumerate(zip(inputs, channels, strides, blocks)))
        self.head_kind = head
        width = channels[-1]
        if context:
            self.context_stage = StackedResidualBlocks(n_blocks=1, input_channels=width, output_channels=context,
                                                       initial_stride=2, conv_bias=conv_bias == 'all', **options)
            self.context_fuse = nn.Conv3d(context, width, 1)
            nn.init.zeros_(self.context_fuse.weight)
            nn.init.zeros_(self.context_fuse.bias)
        if head == 'cells':
            self.head = nn.Linear(width, dim)
        elif head == 'crop':
            if bottleneck:
                self.bottleneck = nn.Sequential(nn.Conv3d(width, bottleneck, 1), norm_op(bottleneck, **norm_kwargs),
                                                nonlin_op(**nonlin_kwargs))
                width = bottleneck
            shape = [int(c)//2**sum(int(s) == 2 for s in strides) for c in crop]
            self.head = nn.Linear(width*math.prod(shape), dim)

    def features(self, image):
        """The feature map (B, C, D', H', W') of crops (B, in_channels, D, H, W)."""
        x = self.stem(image) if hasattr(self, 'stem') else image
        for stage in self.stages:
            x = stage(x)
        if hasattr(self, 'context_stage'):
            context = F.interpolate(self.context_stage(x), size=x.shape[2:], mode='trilinear', align_corners=False)
            x = x+self.context_fuse(context)
        return x

    def tokens(self, features):
        """The head's tokens from a feature map of ``features``."""
        if self.head_kind == 'cells':
            return self.head(features.flatten(2).transpose(1, 2))
        if hasattr(self, 'bottleneck'):
            features = self.bottleneck(features)
        return self.head(features.flatten(1))

    def forward(self, image):
        x = self.features(image)
        return x if self.head_kind is None else self.tokens(x)


class MLP(nn.Sequential):
    """Linear, nonlinearity, Linear: ``inputs`` -> ``hidden`` (default ``outputs``) -> ``outputs``; ``zero_last``:
    the last Linear starts at zero (the branch starts switched off). Layer names ``0`` and ``2`` as nn.Sequential."""
    def __init__(self, inputs, outputs, nonlin=nn.SiLU, hidden=None, zero_last=False):
        hidden = hidden or outputs
        super().__init__(nn.Linear(inputs, hidden), nonlin(), nn.Linear(hidden, outputs))
        if zero_last:
            nn.init.zeros_(self[-1].weight)
            nn.init.zeros_(self[-1].bias)


class TransformerLayer(nn.Module):
    """One pre-norm transformer layer's weights (attention and FFN branches) and its head split/merge. ``qk_norm``:
    queries and keys are RMS-normalized per head (learned gain) before attention, which bounds the attention logits.
    ``logit_scale``: a learned per-head log multiplier of the attention logits (zero: the standard 1/sqrt(d)), so
    normalized attention can sharpen (Henry et al. 2020, Query-Key Normalization for Transformers)."""
    def __init__(self, width, heads, ffn, qk_norm=False, logit_scale=False):
        super().__init__()
        if width % heads:
            raise ValueError('Transformer width must divide by its heads')
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(width//heads, eps=1e-6), nn.RMSNorm(width//heads, eps=1e-6)
        if logit_scale:
            self.logit_scale = nn.Parameter(torch.zeros(heads))
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ffn = MLP(width, width, nn.GELU, hidden=ffn)

    def split(self, x):
        b, n, width = x.shape
        shape = lambda t: t.reshape(b, n, self.heads, width//self.heads).transpose(1, 2)
        q, k, v = map(shape, self.qkv(x).chunk(3, -1))
        if hasattr(self, 'q_norm'):
            q, k = self.q_norm(q), self.k_norm(k)
        if hasattr(self, 'logit_scale'):
            q = q*self.logit_scale.exp().to(q.dtype)[:, None, None]
        return q, k, v

    def merge(self, value):
        b, heads, n, d = value.shape
        return self.out(value.transpose(1, 2).reshape(b, n, heads*d))


class SelfAttentionLayer(TransformerLayer):
    """A pre-norm transformer layer in which every token attends to every token; ``padding`` (B, L) bool marks the
    tokens no token attends to."""
    def forward(self, x, padding=None):
        q, k, v = self.split(self.norm1(x))
        mask = None if padding is None else ~padding[:, None, None, :]
        x = x+self.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        return x+self.ffn(self.norm2(x))
