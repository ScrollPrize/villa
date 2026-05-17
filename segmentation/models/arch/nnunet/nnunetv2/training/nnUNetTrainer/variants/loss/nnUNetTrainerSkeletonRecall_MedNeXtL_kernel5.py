"""MedNeXt-L (kernel5 / kernel3) + bruniss's SkeletonRecall loss.

Targets the compressed / highly curved surface regions where the production
ResEnc-L surface model loses recall. See issue #191 ("Surface and Fiber
Predictions in Compressed or Highly Curved areas") and the re-pitch PR that
motivated this trainer:

  https://github.com/ScrollPrize/villa/issues/191
  https://github.com/ScrollPrize/villa/pull/975

On a held-out 7-cube benchmark over PHerc Paris 1 (S1), scored with the
official Kaggle Vesuvius Surface Detection metric
(0.30 * TopoScore + 0.35 * SurfaceDice@tau + 0.35 * VOI_score, threshold 0.5),
MedNeXt-L kernel5 + SkeletonRecall on bruniss's Dataset059 scores 0.4397 vs
the d058 production model's 0.3996 (+0.040, +10.0% relative) and wins on all
7 cubes. The recommended deployment is a voxel-wise max(d058, MN) ensemble
(0.4437), per the ensemble-not-swap design. Full methodology, per-cube
breakdown, and a downstream ink-AUC sanity check are in PR #975 and the
supporting writeup:

  https://github.com/ciscoriordan/mednext-vs-umamba-scroll/blob/main/docs/pr925_re_pitch.md

This trainer keeps everything in ``nnUNetTrainerSkeletonRecall`` (loss,
SkeletonTransform, custom data loaders, train/validation steps) and only
swaps the default ResEnc U-Net for MedNeXt-L, using the kernel size and
channel/block schedule from ``nnUNetTrainerV2_MedNeXt_L_kernel5`` in the
upstream MedNeXt repo (https://github.com/MIC-DKFZ/MedNeXt). MedNeXt's
deep_supervision output list matches nnUNetv2's expectations (5 levels, full
resolution first); see ``set_deep_supervision_enabled`` below for the one
nnU-Net hook that MedNeXt needs overridden.

Optional dependency:
  The MedNeXt architecture lives in the ``mednextv1`` package, which is not
  vendored here and is not on PyPI. It is imported lazily inside
  ``build_network_architecture`` (see ``_load_mednext``) so that nnU-Net's
  trainer discovery -- which imports every module in this folder -- does not
  fail for users who never build this trainer. Install the extra with:

    pip install -e ".[mednext]"

  (the ``mednext`` extra pulls ``mednextv1`` from its GitHub source).

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
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecall import (
    nnUNetTrainerSkeletonRecall,
)


def _load_mednext():
    """Import the MedNeXt class lazily.

    nnU-Net's ``recursive_find_python_class`` imports every trainer module in
    this folder during trainer discovery, even when the user is building a
    completely different trainer. A top-level ``import nnunet_mednext`` would
    therefore break discovery for everyone who has not installed this optional
    dependency, so the import is deferred to here and accompanied by an
    actionable error.
    """
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
    except ImportError as exc:
        raise ImportError(
            "The MedNeXt SkeletonRecall trainers require the optional 'mednext' "
            "extra (the `mednextv1` package from "
            "https://github.com/MIC-DKFZ/MedNeXt, which is not on PyPI). "
            'Install it with:\n    pip install -e ".[mednext]"'
        ) from exc
    return MedNeXt


class _MedNeXtSkeletonRecallTrainer(nnUNetTrainerSkeletonRecall):
    """Shared MedNeXt + SkeletonRecall behavior.

    Not selected directly; use one of the concrete ``..._kernel5`` /
    ``..._kernel3`` subclasses with ``-tr``.
    """

    def set_deep_supervision_enabled(self, enabled: bool):
        """Toggle deep supervision on a MedNeXt network.

        nnU-Net's default implementation sets ``mod.decoder.deep_supervision``,
        which only exists on nnU-Net's own decoder. MedNeXt exposes the same
        switch as ``mod.do_ds`` instead, so without this override the
        train/validation output-mode toggle would raise ``AttributeError``
        (MedNeXt has no ``decoder`` attribute).
        """
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5(_MedNeXtSkeletonRecallTrainer):
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
        MedNeXt = _load_mednext()
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


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel3(_MedNeXtSkeletonRecallTrainer):
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
        MedNeXt = _load_mednext()
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
