"""MedNeXt-L (kernel5 / kernel3) + bruniss's SkeletonRecall loss.

Targets the compressed / highly curved surface regions where the production
ResEnc-L surface model loses recall. See issue #191 ("Surface and Fiber
Predictions in Compressed or Highly Curved areas") and the benchmark comment
that motivated this trainer:

  https://github.com/ScrollPrize/villa/issues/191

On a held-out 5-cube benchmark over PHerc Paris 1 (S1) plus PHerc 1667 (S4),
MedNeXt-L kernel5 + SkeletonRecall on bruniss's Dataset059 reaches
high_compressed IoU 0.671 vs the d058 production model's 0.404 (+0.267
absolute, +66% relative). overall_macro IoU goes from 0.534 -> 0.685.

This trainer keeps everything in ``nnUNetTrainerSkeletonRecall`` (loss,
SkeletonTransform, custom data loaders, train/validation steps) and only
overrides ``build_network_architecture`` to swap the default ResEnc U-Net for
MedNeXt-L with the kernel size and channel/block schedule from
``nnUNetTrainerV2_MedNeXt_L_kernel5`` in the upstream MedNeXt repo
(https://github.com/MIC-DKFZ/MedNeXt). MedNeXt's deep_supervision output list
matches nnUNetv2's expectations (5 levels, full resolution first).

Requirements:
  - ``mednextv1`` (pip install mednextv1) for the ``MedNeXt`` class.
    Source: https://github.com/MIC-DKFZ/MedNeXt
  - Everything ``nnUNetTrainerSkeletonRecall`` already depends on.

Pretrained weights:
  https://huggingface.co/ciscoriordan/mednext-l-scroll-surface
  (best checkpoint: ``kernel5_skelrec_dataset059_ep33/``)

Usage:
  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5

Memory note:
  MedNeXt-L kernel5 at the default batch_size=2, patch 128^3 fits on a
  40 GB A100 with room to spare. On smaller cards or larger patches, fall
  back to ``nnUNetTrainerSkeletonRecall_MedNeXtL_kernel3``.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecall import (
    nnUNetTrainerSkeletonRecall,
)
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5(nnUNetTrainerSkeletonRecall):
    """MedNeXt-L kernel5 architecture + SkeletonRecall loss."""

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Match ``nnUNetTrainerV2_MedNeXt_L_kernel5.initialize_network()`` from mednextv1.

        nnUNetv2's deep_supervision_scales for a 5-pool 3D U-Net is 5 levels
        (full, /2, /4, /8, /16). MedNeXt with these settings returns exactly
        that list shape, with ``output[0]`` = full resolution.
        """
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=5,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style="outside_block",
        )


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel3(nnUNetTrainerSkeletonRecall):
    """Kernel3 fallback variant. Use when kernel5 doesn't fit in VRAM at the
    desired batch_size / patch size. Same channel/block schedule, kernel=3.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style="outside_block",
        )
