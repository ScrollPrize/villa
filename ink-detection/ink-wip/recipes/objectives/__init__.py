"""Standalone objective helpers."""

from ink.recipes.objectives.common import compute_group_avg
from ink.recipes.objectives.erm import (
    ERMBatch,
    ERMGroupTopK,
    ERMObjective,
    ERMPerSample,
    reduce_group_topk_loss,
)
from ink.recipes.objectives.group_dro import GroupDROComputer, GroupDROObjective

__all__ = [
    "ERMBatch",
    "ERMGroupTopK",
    "ERMObjective",
    "ERMPerSample",
    "GroupDROComputer",
    "GroupDROObjective",
    "compute_group_avg",
    "reduce_group_topk_loss",
]
