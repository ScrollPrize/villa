from itertools import product
from typing import Literal, Optional, Tuple, List, Union, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd

block_type = Literal["basic", "bottleneck"]
block_style = Literal["residual", "conv"]


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    if dimension == 1:
        return nn.Conv1d
    if dimension == 2:
        return nn.Conv2d
    if dimension == 3:
        return nn.Conv3d
    raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


def maybe_convert_scalar_to_list(conv_op: Type[_ConvNd], scalar):
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv1d:
            return [scalar]
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        if conv_op == nn.Conv3d:
            return [scalar] * 3
        raise RuntimeError(f"Invalid conv op: {conv_op}")
    return scalar


def get_matching_pool_op(conv_op: Type[_ConvNd], *, pool_type: str = "avg") -> Type[nn.Module]:
    if pool_type != "avg":
        raise ValueError(f"unsupported pool_type={pool_type!r}")
    mapping = {
        nn.Conv1d: nn.AvgPool1d,
        nn.Conv2d: nn.AvgPool2d,
        nn.Conv3d: nn.AvgPool3d,
    }
    return mapping[conv_op]


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        return self.weight[idx] * x + self.bias[idx]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class ConvDropoutNormReLU(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        padding_mode: str = "zeros",
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(k - 1) // 2 for k in kernel_size],
            dilation=1,
            padding_mode=padding_mode,
            bias=conv_bias,
        )
        ops.append(self.conv)
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)
        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)
        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)
        if nonlin_first and norm_op is not None and nonlin is not None:
            ops[-1], ops[-2] = ops[-2], ops[-1]
        self.all_modules = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        padding_mode: str = "zeros",
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs
        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op,
                input_channels,
                output_channels[0],
                kernel_size,
                initial_stride,
                padding_mode,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op,
                    output_channels[i - 1],
                    output_channels[i],
                    kernel_size,
                    1,
                    padding_mode,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
                for i in range(1, num_convs)
            ],
        )
        self.output_channels = output_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class BasicBlockD(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        padding_mode: str = "zeros",
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(
            conv_op,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding_mode,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op,
            output_channels,
            output_channels,
            kernel_size,
            1,
            padding_mode,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            None,
            None,
        )
        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

        has_stride = any(step != 1 for step in stride)
        requires_projection = input_channels != output_channels
        if has_stride or requires_projection:
            ops = []
            if has_stride:
                pool_op = get_matching_pool_op(conv_op, pool_type="avg")
                ops.append(pool_op(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op,
                        input_channels,
                        output_channels,
                        1,
                        1,
                        "zeros",
                        False,
                        norm_op,
                        norm_op_kwargs,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out = out + residual
        return self.nonlin2(out)


class BottleneckD(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        bottleneck_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        padding_mode: str = "zeros",
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        self.conv1 = ConvDropoutNormReLU(
            conv_op,
            input_channels,
            bottleneck_channels,
            1,
            1,
            "zeros",
            conv_bias,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            nonlin,
            nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op,
            bottleneck_channels,
            bottleneck_channels,
            kernel_size,
            stride,
            padding_mode,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
        )
        self.conv3 = ConvDropoutNormReLU(
            conv_op,
            bottleneck_channels,
            output_channels,
            1,
            1,
            "zeros",
            conv_bias,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            None,
            None,
        )
        self.nonlin3 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()

        has_stride = any(step != 1 for step in stride)
        requires_projection = input_channels != output_channels
        if has_stride or requires_projection:
            ops = []
            if has_stride:
                pool_op = get_matching_pool_op(conv_op, pool_type="avg")
                ops.append(pool_op(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op,
                        input_channels,
                        output_channels,
                        1,
                        1,
                        "zeros",
                        False,
                        norm_op,
                        norm_op_kwargs,
                        None,
                        None,
                        None,
                        None,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out = out + residual
        return self.nonlin3(out)


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        padding_mode: str = "zeros",
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...], None] = None,
    ) -> None:
        super().__init__()
        assert n_blocks > 0
        assert block in [BasicBlockD, BottleneckD]
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        if block == BasicBlockD:
            blocks = nn.Sequential(
                block(
                    conv_op,
                    input_channels,
                    output_channels[0],
                    kernel_size,
                    initial_stride,
                    padding_mode,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                ),
                *[
                    block(
                        conv_op,
                        output_channels[n - 1],
                        output_channels[n],
                        kernel_size,
                        1,
                        padding_mode,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                    )
                    for n in range(1, n_blocks)
                ],
            )
        else:
            blocks = nn.Sequential(
                block(
                    conv_op,
                    input_channels,
                    bottleneck_channels[0],
                    output_channels[0],
                    kernel_size,
                    initial_stride,
                    padding_mode,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                ),
                *[
                    block(
                        conv_op,
                        output_channels[n - 1],
                        bottleneck_channels[n],
                        output_channels[n],
                        kernel_size,
                        1,
                        padding_mode,
                        conv_bias,
                        norm_op,
                        norm_op_kwargs,
                        dropout_op,
                        dropout_op_kwargs,
                        nonlin,
                        nonlin_kwargs,
                    )
                    for n in range(1, n_blocks)
                ],
            )
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a patch grid of shape (B, embed_dim, D, H, W) for 3D or
        (B, embed_dim, H, W) for 2D, where (D, H, W) = input_shape / patch_size.
        This output will need to be rearranged to whatever your transformer expects.
        """
        x = self.proj(x)
        return x


class PatchEmbedDeeper(nn.Module):
    """Dynamic-network-architectures-style patch embed with patch-size-derived stage strides."""

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
        base_features: int = 32,
        depth_per_level: Optional[Tuple[int, ...]] = None,
        embed_proj_3x3x3: bool = False,
        embed_block_type: block_type = "basic",
        embed_block_style: block_style = "residual",
    ) -> None:
        super().__init__()
        self.patch_size = tuple(int(p) for p in patch_size)
        self.ndim = len(self.patch_size)
        if self.ndim not in (2, 3):
            raise ValueError(f"PatchEmbedDeeper only supports 2D or 3D, got ndim={self.ndim}")
        if any(not _is_power_of_two(size) for size in self.patch_size):
            raise ValueError(
                f"PatchEmbedDeeper requires power-of-two patch sizes, got patch_size={self.patch_size}"
            )

        max_patch = max(self.patch_size)
        num_stages = int(np.log2(max_patch)) if max_patch > 1 else 0
        if depth_per_level is None:
            depth_per_level = (1,) * num_stages
        if len(depth_per_level) != num_stages:
            raise ValueError(
                f"depth_per_level must match the number of downsampling stages ({num_stages}), got {depth_per_level}"
            )

        conv_op = convert_dim_to_conv_op(self.ndim)
        norm_op = LayerNormNd
        kernel_size = [3] * self.ndim
        block = BottleneckD if embed_block_type == "bottleneck" else BasicBlockD
        nonlin = nn.LeakyReLU if embed_block_type == "bottleneck" else nn.ReLU
        norm_op_kwargs = {"eps": 1e-5}
        nonlin_kwargs = {"inplace": True}

        if embed_block_style == "residual":
            self.stem = StackedResidualBlocks(
                1,
                conv_op,
                input_channels,
                base_features,
                kernel_size,
                1,
                padding_mode="reflect",
                conv_bias=True,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=block,
            )
        elif embed_block_style == "conv":
            self.stem = StackedConvBlocks(
                1,
                conv_op,
                input_channels,
                base_features,
                kernel_size,
                1,
                padding_mode="reflect",
                conv_bias=True,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown embed_block_style: {embed_block_style}. Must be 'residual' or 'conv'."
            )

        stage_strides = [
            tuple(2 if (patch / (2 ** stage_index)) % 2 == 0 else 1 for patch in self.patch_size)
            for stage_index in range(num_stages)
        ]

        self.stages = nn.ModuleList()
        stage_in_channels = base_features
        for stage_index, (depth, stride) in enumerate(zip(depth_per_level, stage_strides)):
            stage_out_channels = base_features * (2 ** stage_index)
            bottleneck_channels = stage_out_channels // 4 if embed_block_type == "bottleneck" else None
            if embed_block_style == "residual":
                stage = StackedResidualBlocks(
                    n_blocks=depth,
                    conv_op=conv_op,
                    input_channels=stage_in_channels,
                    output_channels=stage_out_channels,
                    kernel_size=kernel_size,
                    initial_stride=stride,
                    padding_mode="reflect",
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    block=block,
                    bottleneck_channels=bottleneck_channels,
                )
            else:
                stage = StackedConvBlocks(
                    num_convs=depth,
                    conv_op=conv_op,
                    input_channels=stage_in_channels,
                    output_channels=stage_out_channels,
                    kernel_size=kernel_size,
                    initial_stride=stride,
                    padding_mode="reflect",
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            self.stages.append(stage)
            stage_in_channels = stage_out_channels

        final_proj_kernel = [3] * self.ndim if embed_proj_3x3x3 else [1] * self.ndim
        final_pad = [1] * self.ndim if embed_proj_3x3x3 else [0] * self.ndim
        self.final_proj = conv_op(
            stage_in_channels,
            embed_dim,
            kernel_size=final_proj_kernel,
            stride=[1] * self.ndim,
            padding=final_pad,
            padding_mode="reflect",
        )
        self._patch_halo: tuple[int, ...] | None = None

    def _normalize_patch_chunk_size(
        self,
        patch_chunk_size: Optional[Union[int, tuple[int, ...], list[int]]],
        *,
        full_patch_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        if patch_chunk_size is None:
            return full_patch_shape
        if isinstance(patch_chunk_size, int):
            if patch_chunk_size <= 0:
                raise ValueError(f"patch_chunk_size must be positive, got {patch_chunk_size}")
            normalized = (int(patch_chunk_size),) * self.ndim
        else:
            normalized = tuple(int(v) for v in patch_chunk_size)
            if len(normalized) != self.ndim:
                raise ValueError(
                    f"patch_chunk_size must provide {self.ndim} values, got {normalized}"
                )
            if any(v <= 0 for v in normalized):
                raise ValueError(f"patch_chunk_size must be positive, got {normalized}")
        return tuple(min(chunk, full) for chunk, full in zip(normalized, full_patch_shape))

    def _avg_pool_support(
        self,
        support: torch.Tensor,
        *,
        kernel_size: Union[int, tuple[int, ...], list[int]],
        stride: Union[int, tuple[int, ...], list[int]],
        padding: Union[int, tuple[int, ...], list[int]],
    ) -> torch.Tensor:
        pool = F.avg_pool2d if self.ndim == 2 else F.avg_pool3d
        return pool(
            support,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True,
        )

    def _support_from_conv(self, support: torch.Tensor, conv: _ConvNd) -> torch.Tensor:
        return self._avg_pool_support(
            support,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
        )

    def _support_from_module(self, module: nn.Module, support: torch.Tensor) -> torch.Tensor:
        if isinstance(module, ConvDropoutNormReLU):
            return self._support_from_conv(support, module.conv)
        if isinstance(module, StackedConvBlocks):
            for block in module.convs:
                support = self._support_from_module(block, support)
            return support
        if isinstance(module, BasicBlockD):
            residual = self._support_from_module(module.skip, support)
            main = self._support_from_module(module.conv1, support)
            main = self._support_from_module(module.conv2, main)
            return 0.5 * (main + residual)
        if isinstance(module, BottleneckD):
            residual = self._support_from_module(module.skip, support)
            main = self._support_from_module(module.conv1, support)
            main = self._support_from_module(module.conv2, main)
            main = self._support_from_module(module.conv3, main)
            return 0.5 * (main + residual)
        if isinstance(module, StackedResidualBlocks):
            for block in module.blocks:
                support = self._support_from_module(block, support)
            return support
        if isinstance(module, (nn.AvgPool2d, nn.AvgPool3d)):
            return self._avg_pool_support(
                support,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
            )
        if isinstance(module, (nn.Identity, LayerNormNd, nn.ReLU, nn.LeakyReLU)):
            return support
        if isinstance(module, nn.Sequential):
            for submodule in module:
                support = self._support_from_module(submodule, support)
            return support
        raise TypeError(f"Unsupported support propagation module: {type(module).__name__}")

    @torch.no_grad()
    def patch_halo(self) -> tuple[int, ...]:
        if self._patch_halo is None:
            probe_patch_shape = tuple(max(13, 2 * (len(self.stages) + 2) + 1) for _ in range(self.ndim))
            probe_spatial_shape = tuple(
                patch_count * patch_size for patch_count, patch_size in zip(probe_patch_shape, self.patch_size)
            )
            support = torch.ones((1, 1, *probe_spatial_shape), dtype=torch.float32)
            support = self._support_from_module(self.stem, support)
            for stage in self.stages:
                support = self._support_from_module(stage, support)
            support = self._support_from_conv(support, self.final_proj)
            support = support / support.amax().clamp(min=torch.finfo(support.dtype).eps)
            full_support = support[0, 0] >= (1.0 - 1e-6)
            indices = torch.nonzero(full_support, as_tuple=False)
            if indices.numel() == 0:
                raise RuntimeError("PatchEmbedDeeper could not determine a full-support interior patch region.")

            halo: list[int] = []
            for axis, axis_size in enumerate(full_support.shape):
                axis_min = int(indices[:, axis].min().item())
                axis_max = int(indices[:, axis].max().item())
                halo.append(min(axis_min, axis_size - axis_max - 1))
            self._patch_halo = tuple(halo)
        return self._patch_halo

    def forward_tiled(
        self,
        x: torch.Tensor,
        *,
        target_patch_shape: Optional[Union[int, tuple[int, ...], list[int]]] = None,
        patch_chunk_size: Optional[Union[int, tuple[int, ...], list[int]]] = None,
        batch_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        full_patch_shape = tuple(int(dim) // int(patch) for dim, patch in zip(x.shape[2:], self.patch_size))
        if any(int(dim) % int(patch) != 0 for dim, patch in zip(x.shape[2:], self.patch_size)):
            raise ValueError(
                f"PatchEmbedDeeper input shape must be divisible by patch_size, got shape={tuple(x.shape[2:])} "
                f"and patch_size={self.patch_size}"
            )

        if batch_chunk_size is None:
            batch_chunk_size = x.shape[0]
        batch_chunk_size = int(batch_chunk_size)
        if batch_chunk_size <= 0:
            raise ValueError(f"batch_chunk_size must be positive, got {batch_chunk_size}")

        if target_patch_shape is None:
            target_patch_shape_normalized = full_patch_shape
        elif isinstance(target_patch_shape, int):
            if target_patch_shape <= 0:
                raise ValueError(f"target_patch_shape must be positive, got {target_patch_shape}")
            target_patch_shape_normalized = (int(target_patch_shape),) * self.ndim
        else:
            target_patch_shape_normalized = tuple(int(v) for v in target_patch_shape)
            if len(target_patch_shape_normalized) != self.ndim:
                raise ValueError(
                    f"target_patch_shape must provide {self.ndim} values, got {target_patch_shape_normalized}"
                )
            if any(v <= 0 for v in target_patch_shape_normalized):
                raise ValueError(
                    f"target_patch_shape must be positive, got {target_patch_shape_normalized}"
                )
        if any(target > full for target, full in zip(target_patch_shape_normalized, full_patch_shape)):
            raise ValueError(
                f"target_patch_shape={target_patch_shape_normalized} cannot exceed full patch shape {full_patch_shape}"
            )

        patch_chunk_shape = self._normalize_patch_chunk_size(
            patch_chunk_size,
            full_patch_shape=target_patch_shape_normalized,
        )
        if (
            batch_chunk_size >= x.shape[0]
            and patch_chunk_shape == target_patch_shape_normalized
            and target_patch_shape_normalized == full_patch_shape
        ):
            return self.forward(x)

        crop_starts = []
        for full, target in zip(full_patch_shape, target_patch_shape_normalized):
            delta = full - target
            if delta < 0 or delta % 2 != 0:
                raise ValueError(
                    f"cannot center-crop patch grid from {full_patch_shape} to {target_patch_shape_normalized}"
                )
            crop_starts.append(delta // 2)

        crop_starts = tuple(crop_starts)
        halo = self.patch_halo()
        if patch_chunk_size is not None and target_patch_shape_normalized == full_patch_shape:
            raise ValueError(
                "PatchEmbedDeeper spatial chunking requires a smaller centered target_patch_shape; "
                "set patch_chunk_size=None to use batch-only chunking."
            )
        if patch_chunk_size is not None:
            required_halo = tuple(min(start, full - (start + target)) for start, target, full in zip(
                crop_starts,
                target_patch_shape_normalized,
                full_patch_shape,
            ))
            if any(available < needed for available, needed in zip(required_halo, halo)):
                raise ValueError(
                    "PatchEmbedDeeper spatial chunking requires centered overscan at least as large as patch_halo; "
                    f"got available={required_halo} and required={halo}"
                )

        if patch_chunk_size is None:
            outputs = []
            crop_stops = tuple(start + size for start, size in zip(crop_starts, target_patch_shape_normalized))
            output_slices = tuple(slice(start, stop) for start, stop in zip(crop_starts, crop_stops))
            for batch_start in range(0, x.shape[0], batch_chunk_size):
                batch_end = min(batch_start + batch_chunk_size, x.shape[0])
                batch_y = self.forward(x[batch_start:batch_end])
                outputs.append(batch_y[(slice(None), slice(None), *output_slices)])
            return torch.cat(outputs, dim=0)

        outputs: list[torch.Tensor] = []
        tile_starts = [range(0, size, chunk) for size, chunk in zip(target_patch_shape_normalized, patch_chunk_shape)]
        for batch_start in range(0, x.shape[0], batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, x.shape[0])
            batch_x = x[batch_start:batch_end]
            batch_output: torch.Tensor | None = None

            for starts in product(*tile_starts):
                tile_shape = tuple(
                    min(chunk, full - start)
                    for start, chunk, full in zip(starts, patch_chunk_shape, target_patch_shape_normalized)
                )
                target_starts = tuple(crop_start + start for crop_start, start in zip(crop_starts, starts))
                context_starts = tuple(target_start - axis_halo for target_start, axis_halo in zip(target_starts, halo))
                context_stops = tuple(
                    target_start + size + axis_halo
                    for target_start, size, axis_halo in zip(target_starts, tile_shape, halo)
                )
                input_slices = tuple(
                    slice(start * patch, stop * patch)
                    for start, stop, patch in zip(context_starts, context_stops, self.patch_size)
                )
                tile_x = batch_x[(slice(None), slice(None), *input_slices)]
                tile_y = self.forward(tile_x)

                output_slices = tuple(
                    slice(target_start - context_start, target_start - context_start + size)
                    for target_start, context_start, size in zip(target_starts, context_starts, tile_shape)
                )
                tile_output = tile_y[(slice(None), slice(None), *output_slices)]
                if batch_output is None:
                    batch_output = torch.empty(
                        (batch_x.shape[0], tile_output.shape[1], *target_patch_shape_normalized),
                        device=tile_output.device,
                        dtype=tile_output.dtype,
                    )
                destination_slices = tuple(slice(start, start + size) for start, size in zip(starts, tile_shape))
                batch_output[(slice(None), slice(None), *destination_slices)] = tile_output

            if batch_output is None:
                raise RuntimeError("PatchEmbedDeeper.forward_tiled produced no output tiles.")
            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53

    Supports both 2D and 3D inputs based on patch_size dimensionality.
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die

        Args:
            patch_size: Tuple of (H, W) for 2D or (D, H, W) for 3D
        """
        super().__init__()

        # Determine dimensionality from patch_size
        self.ndim = len(patch_size)
        conv_transpose_op = nn.ConvTranspose2d if self.ndim == 2 else nn.ConvTranspose3d

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    conv_transpose_op(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                    norm(channels[s + 1]),
                    activation(),
                )
            )
        stages.append(conv_transpose_op(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, D, H, W) for 3D or (B, embed_dim, H, W) for 2D.
        This will require you to reshape the output of your transformer.
        """
        return self.decode(x)
