"""Standalone loss recipes and contracts."""

from ink.recipes.losses.bce import BCEBatch, BCEPerSample, build_bce_targets
from ink.recipes.losses.betti_matching import (
    StitchBettiMatchingLoss,
    compute_binary_betti_matching_loss,
)
from ink.recipes.losses.boundary import (
    StitchBoundaryLoss,
    binary_mask_to_signed_distance_map,
    compute_binary_boundary_loss,
)
from ink.recipes.losses.cldice import StitchCLDiceLoss, compute_binary_soft_cldice_loss
from ink.recipes.losses.composer import LossComposer, LossTerm
from ink.recipes.losses.dice import DiceBatch, DicePerSample
from ink.recipes.losses.dice_bce import DiceBCEBatch, DiceBCEPerSample
from ink.recipes.losses.reporting import (
    loss_values,
    resolve_train_output,
    train_components,
)
from ink.recipes.losses.stitch_region import StitchRegionLoss

__all__ = [
    "BCEBatch",
    "BCEPerSample",
    "DiceBatch",
    "DicePerSample",
    "DiceBCEBatch",
    "DiceBCEPerSample",
    "LossComposer",
    "LossTerm",
    "StitchBettiMatchingLoss",
    "StitchBoundaryLoss",
    "StitchCLDiceLoss",
    "StitchRegionLoss",
    "binary_mask_to_signed_distance_map",
    "build_bce_targets",
    "compute_binary_betti_matching_loss",
    "compute_binary_boundary_loss",
    "compute_binary_soft_cldice_loss",
    "loss_values",
    "resolve_train_output",
    "train_components",
]
