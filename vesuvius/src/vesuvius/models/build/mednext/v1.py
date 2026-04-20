"""
3D MedNeXt v1 architecture adapted from the official MIC-DKFZ MedNeXt repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .blocks import MedNeXtBlock3D, MedNeXtDownBlock3D, MedNeXtUpBlock3D, OutBlock3D


class MedNeXtV1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        *,
        exp_r: int | list[int] = 4,
        kernel_size: int = 7,
        enc_kernel_size: int | None = None,
        dec_kernel_size: int | None = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: str | None = None,
        block_counts: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2, 2),
        norm_type: str = "group",
        grn: bool = False,
    ):
        super().__init__()

        self.do_ds = bool(deep_supervision)
        if checkpoint_style not in {None, "outside_block"}:
            raise ValueError(
                f"Unsupported mednext_checkpoint_style {checkpoint_style!r}. "
                "Expected None or 'outside_block'."
            )
        self.outside_block_checkpointing = checkpoint_style == "outside_block"

        if kernel_size is not None:
            enc_kernel_size = int(kernel_size)
            dec_kernel_size = int(kernel_size)
        enc_kernel_size = int(enc_kernel_size or 3)
        dec_kernel_size = int(dec_kernel_size or 3)
        block_counts = list(int(v) for v in block_counts)
        if len(block_counts) != 9:
            raise ValueError(f"MedNeXt expects 9 block_counts entries, got {block_counts}")

        if isinstance(exp_r, int):
            exp_r = [int(exp_r)] * len(block_counts)
        else:
            exp_r = [int(v) for v in exp_r]
        if len(exp_r) != len(block_counts):
            raise ValueError(f"MedNeXt expects 9 exp_r entries, got {exp_r}")

        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1)

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[0])
        ])
        self.down_0 = MedNeXtDownBlock3D(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=2 * n_channels,
                out_channels=2 * n_channels,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[1])
        ])
        self.down_1 = MedNeXtDownBlock3D(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=4 * n_channels,
                out_channels=4 * n_channels,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[2])
        ])
        self.down_2 = MedNeXtDownBlock3D(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=8 * n_channels,
                out_channels=8 * n_channels,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[3])
        ])
        self.down_3 = MedNeXtDownBlock3D(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=16 * n_channels,
                out_channels=16 * n_channels,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[4])
        ])

        self.up_3 = MedNeXtUpBlock3D(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=8 * n_channels,
                out_channels=8 * n_channels,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock3D(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=4 * n_channels,
                out_channels=4 * n_channels,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock3D(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=2 * n_channels,
                out_channels=2 * n_channels,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock3D(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            grn=grn,
        )
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock3D(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                grn=grn,
            ) for _ in range(block_counts[8])
        ])

        self.out_0 = OutBlock3D(n_channels, n_classes)
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        if self.do_ds:
            self.out_1 = OutBlock3D(2 * n_channels, n_classes)
            self.out_2 = OutBlock3D(4 * n_channels, n_classes)
            self.out_3 = OutBlock3D(8 * n_channels, n_classes)
            self.out_4 = OutBlock3D(16 * n_channels, n_classes)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for layer in sequential_block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor)
        return x

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)
            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)
            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            x = self.iterative_checkpoint(self.dec_block_3, x_res_3 + x_up_3)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            x = self.iterative_checkpoint(self.dec_block_2, x_res_2 + x_up_2)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            x = self.iterative_checkpoint(self.dec_block_1, x_res_1 + x_up_1)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            x = self.iterative_checkpoint(self.dec_block_0, x_res_0 + x_up_0)
            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)
            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)
            x = self.dec_block_3(x_res_3 + self.up_3(x))
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            x = self.dec_block_2(x_res_2 + self.up_2(x))
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            x = self.dec_block_1(x_res_1 + self.up_1(x))
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            x = self.dec_block_0(x_res_0 + self.up_0(x))
            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x
