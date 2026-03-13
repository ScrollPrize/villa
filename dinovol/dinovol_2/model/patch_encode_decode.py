from typing import Literal, Optional, Tuple, List, Union, Type

import numpy as np
import torch
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
        norm_op = nn.InstanceNorm2d if self.ndim == 2 else nn.InstanceNorm3d
        kernel_size = [3] * self.ndim
        block = BottleneckD if embed_block_type == "bottleneck" else BasicBlockD
        nonlin = nn.LeakyReLU if embed_block_type == "bottleneck" else nn.ReLU
        norm_op_kwargs = {"eps": 1e-5, "affine": True}
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
