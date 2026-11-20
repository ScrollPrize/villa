"""Run 3D convolutions as sums of 2D convolutions.

A convolution with a depth-``Kd`` kernel is the sum, over its ``Kd`` depth taps,
of 2D convolutions of depth-shifted slices. The weights and the products are
the same; only the order of the additions changes.

On ROCm, MIOpen runs 3D convolutions much slower than 2D ones, so this rewrite
makes models with 3D layers faster there (for example the depth-fusion stem of
the 3D-stem/2D-UNet ink models). On CUDA, cuDNN's 3D kernels are faster than
the rewrite, which is why inference only turns it on under ROCm by default,
and then only for those depth-fusion stems.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.ink_detection.models.hybrid_3d2d import LocalDepthFusionStem


CONV3D_AS_2D_CHOICES = ("auto", "on", "off")


def conv3d_as_conv2d(
    input_NCDHW: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    stride: Sequence[int],
    padding: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """Compute ``F.conv3d`` (zero padding, no dilation) with 2D convolutions."""

    stride_d, stride_h, stride_w = (int(value) for value in stride)
    pad_d, pad_h, pad_w = (int(value) for value in padding)
    kernel_d = int(weight.shape[2])
    if pad_d:
        input_NCDHW = F.pad(input_NCDHW, (0, 0, 0, 0, pad_d, pad_d))
    batch, channels, depth, height, width = input_NCDHW.shape
    out_depth = (depth - kernel_d) // stride_d + 1
    if out_depth <= 0:
        # Let conv3d report the size error in its own words.
        return F.conv3d(
            input_NCDHW, weight, bias, tuple(stride), (0, pad_h, pad_w), 1, groups
        )

    output = None
    for tap in range(kernel_d):
        # The output depths that read input depth ``tap``, ``tap + stride``, ...
        # become a batch of 2D images.
        taps = input_NCDHW[:, :, tap : tap + stride_d * (out_depth - 1) + 1 : stride_d]
        planes = taps.transpose(1, 2).reshape(batch * out_depth, channels, height, width)
        partial = F.conv2d(
            planes, weight[:, :, tap], None, (stride_h, stride_w), (pad_h, pad_w), 1, groups
        )
        output = partial if output is None else output.add_(partial)
    output = output.reshape(batch, out_depth, *output.shape[1:]).transpose(1, 2)
    if bias is not None:
        output = output + bias.to(output.dtype).view(1, -1, 1, 1, 1)
    return output.contiguous()


class Conv3dAsConv2d(nn.Conv3d):
    """An ``nn.Conv3d`` whose forward runs as a sum of 2D convolutions.

    It keeps the parameter names, so state dicts load as before.
    """

    @classmethod
    def from_conv3d(cls, conv: nn.Conv3d) -> "Conv3dAsConv2d":
        replacement = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device="meta",
        )
        replacement.weight = conv.weight
        replacement.bias = conv.bias
        replacement.train(conv.training)
        return replacement

    def _conv_forward(self, input, weight, bias):
        if input.dim() != 5:
            return super()._conv_forward(input, weight, bias)
        return conv3d_as_conv2d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


def _can_run_as_conv2d(module: nn.Module) -> bool:
    return (
        type(module) is nn.Conv3d
        and module.padding_mode == "zeros"
        and not isinstance(module.padding, str)
        and tuple(module.dilation) == (1, 1, 1)
    )


def split_conv3d_layers(model: nn.Module) -> int:
    """Swap every plain ``nn.Conv3d`` in ``model`` for ``Conv3dAsConv2d``.

    Layers with other padding modes, string padding or dilation are left as
    they are. Returns the number of layers swapped.
    """

    swapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if _can_run_as_conv2d(child):
                setattr(parent, name, Conv3dAsConv2d.from_conv3d(child))
                swapped += 1
    return swapped


def resolve_conv3d_as_2d(mode: str, device: torch.device) -> bool:
    """Decide whether to run 3D convolutions as 2D ones on ``device``.

    ``auto`` means only on a ROCm (HIP) build of PyTorch running on the GPU.
    """

    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        return device.type == "cuda" and torch.version.hip is not None
    raise ValueError(
        f"conv3d_as_2d must be one of {', '.join(CONV3D_AS_2D_CHOICES)}, got {mode!r}"
    )


def split_conv3d_for_inference(
    model: nn.Module, mode: str, device: torch.device
) -> int:
    """Apply an inference ``conv3d_as_2d`` mode to ``model``.

    ``on`` swaps every plain ``nn.Conv3d``. ``auto`` swaps, on ROCm only, the
    layers of the depth-fusion stems (``LocalDepthFusionStem``) of the
    3D-stem/2D-UNet models; fully 3D models keep stock ``Conv3d``. Returns the
    number of layers swapped.
    """

    if not resolve_conv3d_as_2d(mode, device):
        return 0
    if mode == "on":
        return split_conv3d_layers(model)
    stems = [
        module for module in model.modules() if isinstance(module, LocalDepthFusionStem)
    ]
    return sum(split_conv3d_layers(stem) for stem in stems)
