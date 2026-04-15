"""
Wrapper classes to integrate MedNeXt with NetworkFromConfig.

The wrappers keep the shared-encoder / task-decoder contract used by the rest
of vesuvius while preserving MedNeXt's additive skip behavior.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .mednext.blocks import MedNeXtBlock3D, MedNeXtDownBlock3D, MedNeXtUpBlock3D, OutBlock3D


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        n_channels: int,
        exp_r: Sequence[int],
        block_counts: Sequence[int],
        kernel_size: int,
        checkpoint_style: str | None,
        norm_type: str,
        grn: bool,
        do_res: bool = True,
        do_res_up_down: bool = True,
    ):
        super().__init__()
        if checkpoint_style not in {None, "outside_block"}:
            raise ValueError(
                f"Unsupported mednext_checkpoint_style {checkpoint_style!r}. "
                "Expected None or 'outside_block'."
            )
        self.outside_block_checkpointing = checkpoint_style == "outside_block"
        self.n_channels = int(n_channels)
        self.exp_r = [int(v) for v in exp_r]
        self.block_counts = [int(v) for v in block_counts]
        self.kernel_size = int(kernel_size)
        self.checkpoint_style = checkpoint_style
        self.norm_type = str(norm_type)
        self.grn = bool(grn)
        self.do_res = bool(do_res)
        self.do_res_up_down = bool(do_res_up_down)
        self.ndim = 3
        self.output_channels = [
            self.n_channels,
            2 * self.n_channels,
            4 * self.n_channels,
            8 * self.n_channels,
            16 * self.n_channels,
        ]
        self.strides = [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ]
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d if self.norm_type == "instance" else None
        self.norm_op_kwargs = {"affine": True, "eps": 1e-5}
        self.nonlin = nn.GELU
        self.nonlin_kwargs = {}
        self.dropout_op = None
        self.dropout_op_kwargs = None
        self.conv_bias = True
        self.kernel_sizes = [[int(self.kernel_size)] * 3 for _ in self.output_channels]

        c = self.n_channels
        e = self.exp_r
        b = self.block_counts
        k = self.kernel_size
        self.stem = nn.Conv3d(input_channels, c, kernel_size=1)
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock3D(c, c, exp_r=e[0], kernel_size=k, do_res=self.do_res, norm_type=self.norm_type, grn=self.grn)
            for _ in range(b[0])
        ])
        self.down_0 = MedNeXtDownBlock3D(c, 2 * c, exp_r=e[1], kernel_size=k, do_res=self.do_res_up_down, norm_type=self.norm_type, grn=self.grn)
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock3D(2 * c, 2 * c, exp_r=e[1], kernel_size=k, do_res=self.do_res, norm_type=self.norm_type, grn=self.grn)
            for _ in range(b[1])
        ])
        self.down_1 = MedNeXtDownBlock3D(2 * c, 4 * c, exp_r=e[2], kernel_size=k, do_res=self.do_res_up_down, norm_type=self.norm_type, grn=self.grn)
        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock3D(4 * c, 4 * c, exp_r=e[2], kernel_size=k, do_res=self.do_res, norm_type=self.norm_type, grn=self.grn)
            for _ in range(b[2])
        ])
        self.down_2 = MedNeXtDownBlock3D(4 * c, 8 * c, exp_r=e[3], kernel_size=k, do_res=self.do_res_up_down, norm_type=self.norm_type, grn=self.grn)
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock3D(8 * c, 8 * c, exp_r=e[3], kernel_size=k, do_res=self.do_res, norm_type=self.norm_type, grn=self.grn)
            for _ in range(b[3])
        ])
        self.down_3 = MedNeXtDownBlock3D(8 * c, 16 * c, exp_r=e[4], kernel_size=k, do_res=self.do_res_up_down, norm_type=self.norm_type, grn=self.grn)
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock3D(16 * c, 16 * c, exp_r=e[4], kernel_size=k, do_res=self.do_res, norm_type=self.norm_type, grn=self.grn)
            for _ in range(b[4])
        ])
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def _run_sequential(self, block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not self.outside_block_checkpointing:
            return block(x)
        for layer in block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor)
        return x

    def _run_module(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.outside_block_checkpointing:
            return module(x)
        return checkpoint.checkpoint(module, x, self.dummy_tensor)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x_res_0 = self._run_sequential(self.enc_block_0, x)
        x = self._run_module(self.down_0, x_res_0)
        x_res_1 = self._run_sequential(self.enc_block_1, x)
        x = self._run_module(self.down_1, x_res_1)
        x_res_2 = self._run_sequential(self.enc_block_2, x)
        x = self._run_module(self.down_2, x_res_2)
        x_res_3 = self._run_sequential(self.enc_block_3, x)
        x = self._run_module(self.down_3, x_res_3)
        x = self._run_sequential(self.bottleneck, x)
        return [x_res_0, x_res_1, x_res_2, x_res_3, x]


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder: MedNeXtEncoder,
        num_classes: int | None,
        deep_supervision: bool,
    ):
        super().__init__()
        if num_classes is None and deep_supervision:
            raise ValueError("MedNeXt shared decoder cannot emit deep supervision outputs without logits heads")
        object.__setattr__(self, "_encoder_ref", encoder)
        self.num_classes = num_classes
        self.do_ds = bool(deep_supervision and num_classes is not None)
        c = encoder.n_channels
        e = encoder.exp_r
        b = encoder.block_counts
        k = encoder.kernel_size
        self.outside_block_checkpointing = encoder.outside_block_checkpointing
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.up_3 = MedNeXtUpBlock3D(16 * c, 8 * c, exp_r=e[5], kernel_size=k, do_res=encoder.do_res_up_down, norm_type=encoder.norm_type, grn=encoder.grn)
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock3D(8 * c, 8 * c, exp_r=e[5], kernel_size=k, do_res=encoder.do_res, norm_type=encoder.norm_type, grn=encoder.grn)
            for _ in range(b[5])
        ])
        self.up_2 = MedNeXtUpBlock3D(8 * c, 4 * c, exp_r=e[6], kernel_size=k, do_res=encoder.do_res_up_down, norm_type=encoder.norm_type, grn=encoder.grn)
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock3D(4 * c, 4 * c, exp_r=e[6], kernel_size=k, do_res=encoder.do_res, norm_type=encoder.norm_type, grn=encoder.grn)
            for _ in range(b[6])
        ])
        self.up_1 = MedNeXtUpBlock3D(4 * c, 2 * c, exp_r=e[7], kernel_size=k, do_res=encoder.do_res_up_down, norm_type=encoder.norm_type, grn=encoder.grn)
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock3D(2 * c, 2 * c, exp_r=e[7], kernel_size=k, do_res=encoder.do_res, norm_type=encoder.norm_type, grn=encoder.grn)
            for _ in range(b[7])
        ])
        self.up_0 = MedNeXtUpBlock3D(2 * c, c, exp_r=e[8], kernel_size=k, do_res=encoder.do_res_up_down, norm_type=encoder.norm_type, grn=encoder.grn)
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock3D(c, c, exp_r=e[8], kernel_size=k, do_res=encoder.do_res, norm_type=encoder.norm_type, grn=encoder.grn)
            for _ in range(b[8])
        ])

        if self.num_classes is None:
            self.out_0 = None
            self.out_1 = None
            self.out_2 = None
            self.out_3 = None
            self.out_4 = None
        else:
            self.out_0 = OutBlock3D(c, self.num_classes)
            if self.do_ds:
                self.out_1 = OutBlock3D(2 * c, self.num_classes)
                self.out_2 = OutBlock3D(4 * c, self.num_classes)
                self.out_3 = OutBlock3D(8 * c, self.num_classes)
                self.out_4 = OutBlock3D(16 * c, self.num_classes)
            else:
                self.out_1 = self.out_2 = self.out_3 = self.out_4 = None

    @property
    def encoder(self) -> MedNeXtEncoder:
        return self._encoder_ref

    def _run_sequential(self, block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not self.outside_block_checkpointing:
            return block(x)
        for layer in block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor)
        return x

    def _run_module(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.outside_block_checkpointing:
            return module(x)
        return checkpoint.checkpoint(module, x, self.dummy_tensor)

    def forward(self, features):
        x_res_0, x_res_1, x_res_2, x_res_3, x = features
        if self.do_ds:
            x_ds_4 = self._run_module(self.out_4, x)

        x = self._run_sequential(self.dec_block_3, x_res_3 + self._run_module(self.up_3, x))
        if self.do_ds:
            x_ds_3 = self._run_module(self.out_3, x)

        x = self._run_sequential(self.dec_block_2, x_res_2 + self._run_module(self.up_2, x))
        if self.do_ds:
            x_ds_2 = self._run_module(self.out_2, x)

        x = self._run_sequential(self.dec_block_1, x_res_1 + self._run_module(self.up_1, x))
        if self.do_ds:
            x_ds_1 = self._run_module(self.out_1, x)

        x = self._run_sequential(self.dec_block_0, x_res_0 + self._run_module(self.up_0, x))
        if self.num_classes is None:
            return x

        x = self._run_module(self.out_0, x)
        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x
