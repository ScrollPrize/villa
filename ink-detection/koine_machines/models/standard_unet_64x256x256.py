from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd


def maybe_convert_scalar_to_list(
    conv_op: Type[_ConvNd],
    value: Union[int, Sequence[int]],
) -> Union[List[int], Sequence[int]]:
    if isinstance(value, (tuple, list)):
        return value
    if conv_op == nn.Conv1d:
        return [value]
    if conv_op == nn.Conv2d:
        return [value, value]
    if conv_op == nn.Conv3d:
        return [value, value, value]


def get_matching_convtransp(
    conv_op: Type[_ConvNd],
) -> Type[_ConvTransposeNd]:
    if conv_op == nn.Conv1d:
        return nn.ConvTranspose1d
    if conv_op == nn.Conv2d:
        return nn.ConvTranspose2d
    if conv_op == nn.Conv3d:
        return nn.ConvTranspose3d


def get_matching_pool_op(conv_op: Type[_ConvNd]) -> Type[nn.Module]:
    if conv_op == nn.Conv1d:
        return nn.AvgPool1d
    if conv_op == nn.Conv2d:
        return nn.AvgPool2d
    if conv_op == nn.Conv3d:
        return nn.AvgPool3d


class ConvDropoutNormReLU(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        conv_bias: bool = False,
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] | None = None,
        nonlin_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        norm_op_kwargs = {} if norm_op_kwargs is None else dict(norm_op_kwargs)
        nonlin_kwargs = {} if nonlin_kwargs is None else dict(nonlin_kwargs)

        ops: list[nn.Module] = []
        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            self.stride,
            padding=[(k - 1) // 2 for k in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: Union[int, Sequence[int]],
        kernel_size: Union[int, Sequence[int]],
        initial_stride: Union[int, Sequence[int]],
        conv_bias: bool,
        norm_op: Type[nn.Module] | None,
        norm_op_kwargs: dict | None,
        nonlin: Type[nn.Module] | None,
        nonlin_kwargs: dict | None,
    ) -> None:
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=output_channels[0],
                kernel_size=kernel_size,
                stride=initial_stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op=conv_op,
                    input_channels=output_channels[i - 1],
                    output_channels=output_channels[i],
                    kernel_size=kernel_size,
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for i in range(1, num_convs)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class BasicBlockD(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        conv_bias: bool,
        norm_op: Type[nn.Module] | None,
        norm_op_kwargs: dict | None,
        nonlin: Type[nn.Module] | None,
        nonlin_kwargs: dict | None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)

        self.conv1 = ConvDropoutNormReLU(
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op=conv_op,
            input_channels=output_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=None,
            nonlin_kwargs=None,
        )
        self.nonlin2 = nn.Identity() if nonlin is None else nonlin(**(nonlin_kwargs or {}))

        has_stride = any(int(s) != 1 for s in self.stride)
        requires_projection = int(input_channels) != int(output_channels)
        if has_stride or requires_projection:
            ops: list[nn.Module] = []
            if has_stride:
                pool_op = get_matching_pool_op(conv_op)
                ops.append(pool_op(self.stride, self.stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op=conv_op,
                        input_channels=input_channels,
                        output_channels=output_channels,
                        kernel_size=1,
                        stride=1,
                        conv_bias=False,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=None,
                        nonlin_kwargs=None,
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


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Sequence[int]],
        initial_stride: Union[int, Sequence[int]],
        conv_bias: bool,
        norm_op: Type[nn.Module] | None,
        norm_op_kwargs: dict | None,
        nonlin: Type[nn.Module] | None,
        nonlin_kwargs: dict | None,
    ) -> None:
        super().__init__()

        blocks = [
            BasicBlockD(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=initial_stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        ]
        blocks.extend(
            BasicBlockD(
                conv_op=conv_op,
                input_channels=output_channels,
                output_channels=output_channels,
                kernel_size=kernel_size,
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            for _ in range(1, n_blocks)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        features_per_stage: Sequence[int],
        n_blocks_per_stage: Sequence[int],
        conv_op: Type[_ConvNd],
        strides: Sequence[Sequence[int]],
        kernel_sizes: Sequence[Sequence[int]],
        conv_bias: bool,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict,
    ) -> None:
        super().__init__()
        self.stem = StackedConvBlocks(
            num_convs=1,
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=features_per_stage[0],
            kernel_size=kernel_sizes[0],
            initial_stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        stages: list[nn.Module] = []
        current_channels = int(features_per_stage[0])
        for stage_idx, output_channels in enumerate(features_per_stage):
            stages.append(
                StackedResidualBlocks(
                    n_blocks=int(n_blocks_per_stage[stage_idx]),
                    conv_op=conv_op,
                    input_channels=current_channels,
                    output_channels=int(output_channels),
                    kernel_size=kernel_sizes[stage_idx],
                    initial_stride=strides[stage_idx],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            current_channels = int(output_channels)

        self.stages = nn.Sequential(*stages)
        self.output_channels = [int(v) for v in features_per_stage]
        self.strides = [list(maybe_convert_scalar_to_list(conv_op, s)) for s in strides]
        self.kernel_sizes = [list(k) for k in kernel_sizes]
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = dict(norm_op_kwargs)
        self.nonlin = nonlin
        self.nonlin_kwargs = dict(nonlin_kwargs)
        self.conv_bias = bool(conv_bias)
        self.return_skips = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        n_conv_per_stage: Sequence[int],
        deep_supervision: bool,
    ) -> None:
        super().__init__()
        if len(n_conv_per_stage) != len(encoder.output_channels) - 1:
            raise ValueError(
                "n_conv_per_stage must have one entry per decoder stage, "
                f"got {len(n_conv_per_stage)} for {len(encoder.output_channels) - 1} stages"
            )

        self.deep_supervision = bool(deep_supervision)
        self.num_classes = int(num_classes)
        object.__setattr__(self, "_encoder_ref", encoder)

        transpconv_op = get_matching_convtransp(encoder.conv_op)
        stages: list[nn.Module] = []
        transpconvs: list[nn.Module] = []
        seg_layers: list[nn.Module] = []

        for stage_idx in range(1, len(encoder.output_channels)):
            input_features_below = encoder.output_channels[-stage_idx]
            input_features_skip = encoder.output_channels[-(stage_idx + 1)]
            stride = encoder.strides[-stage_idx]

            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride,
                    stride,
                    bias=encoder.conv_bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    num_convs=int(n_conv_per_stage[stage_idx - 1]),
                    conv_op=encoder.conv_op,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(stage_idx + 1)],
                    initial_stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
            )
            seg_layers.append(
                encoder.conv_op(
                    input_features_skip,
                    self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    @property
    def encoder(self) -> Encoder:
        return self._encoder_ref

    def forward(self, skips: Sequence[torch.Tensor]) -> Union[torch.Tensor, list[torch.Tensor]]:
        low_res = skips[-1]
        seg_outputs: list[torch.Tensor] = []
        for stage_idx in range(len(self.stages)):
            x = self.transpconvs[stage_idx](low_res)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.stages[stage_idx](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[stage_idx](x))
            elif stage_idx == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[stage_idx](x))
            low_res = x

        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]


class StandardUNet64x256x256DeepSupervision(nn.Module):
    patch_size: Tuple[int, int, int] = (64, 256, 256)
    features_per_stage: Tuple[int, ...] = (32, 64, 128, 256, 320, 320, 320)
    n_blocks_per_stage: Tuple[int, ...] = (1, 3, 4, 6, 6, 6, 6)
    n_conv_per_stage_decoder: Tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    kernel_sizes: Tuple[Tuple[int, int, int], ...] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    )
    strides: Tuple[Tuple[int, int, int], ...] = (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
    )
    num_pool_per_axis: Tuple[int, int, int] = (4, 6, 6)
    must_be_divisible_by: Tuple[int, int, int] = (16, 64, 64)

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {"affine": True, "eps": 1e-5}
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {"inplace": True, "negative_slope": 0.01}
        self.conv_bias = True

        self.encoder = Encoder(
            input_channels=self.in_channels,
            features_per_stage=self.features_per_stage,
            n_blocks_per_stage=self.n_blocks_per_stage,
            conv_op=self.conv_op,
            strides=self.strides,
            kernel_sizes=self.kernel_sizes,
            conv_bias=self.conv_bias,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )
        self.decoder = Decoder(
            encoder=self.encoder,
            num_classes=self.out_channels,
            n_conv_per_stage=self.n_conv_per_stage_decoder,
            deep_supervision=True,
        )
        self.final_config = {
            "patch_size": self.patch_size,
            "features_per_stage": self.features_per_stage,
            "n_blocks_per_stage": self.n_blocks_per_stage,
            "n_conv_per_stage_decoder": self.n_conv_per_stage_decoder,
            "kernel_sizes": self.kernel_sizes,
            "strides": self.strides,
            "num_pool_per_axis": self.num_pool_per_axis,
            "must_be_divisible_by": self.must_be_divisible_by,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "deep_supervision": True,
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = self.encoder(x)
        outputs = self.decoder(skips)
        return outputs

    @staticmethod
    def remap_network_from_config_state_dict(
        state_dict: dict[str, torch.Tensor],
        *,
        target_name: str = "surface",
    ) -> "OrderedDict[str, torch.Tensor]":
        mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        decoder_prefix = f"task_decoders.{target_name}."
        for key, value in state_dict.items():
            normalized = key[7:] if key.startswith("module.") else key
            if normalized.startswith("shared_encoder."):
                mapped["encoder." + normalized[len("shared_encoder."):]] = value
            elif normalized.startswith(decoder_prefix):
                mapped["decoder." + normalized[len(decoder_prefix):]] = value
        return mapped

    def load_from_network_from_config(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        target_name: str = "surface",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        remapped = self.remap_network_from_config_state_dict(
            state_dict,
            target_name=target_name,
        )
        incompatible = self.load_state_dict(remapped, strict=strict)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_standard_unet_64x256x256_from_config(
    config_dict: dict,
    *,
    target_name: str = "surface",
) -> StandardUNet64x256x256DeepSupervision:
    targets = dict(config_dict.get("targets") or {})
    target_cfg = dict(targets.get(target_name) or {})
    out_channels = int(
        target_cfg.get(
            "out_channels",
            target_cfg.get("channels", config_dict.get("in_channels", 1)),
        )
    )
    return StandardUNet64x256x256DeepSupervision(
        in_channels=int(config_dict.get("in_channels", 1)),
        out_channels=out_channels,
    )
