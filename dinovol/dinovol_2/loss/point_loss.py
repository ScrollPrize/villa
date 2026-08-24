from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PointLossResult:
    loss: torch.Tensor
    same_type: torch.Tensor
    different_type: torch.Tensor
    point_count: int
    same_pair_count: int
    different_pair_count: int


class PointCosineLoss(nn.Module):
    """Type-balanced positive and negative cosine-pair objective."""

    def __init__(self, different_type_margin: float = 0.0) -> None:
        super().__init__()
        self.different_type_margin = float(different_type_margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> PointLossResult:
        if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Point embeddings and labels must have matching leading dimensions.")
        zero = embeddings.sum() * 0.0
        normalized = F.normalize(embeddings.float(), dim=-1) if embeddings.shape[0] else embeddings.float()
        represented_types = torch.unique(labels, sorted=True).tolist()

        same_losses: list[torch.Tensor] = []
        same_pair_count = 0
        for type_id in represented_types:
            indices = torch.nonzero(labels == type_id, as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            pairs = torch.combinations(indices, r=2)
            similarities = (normalized[pairs[:, 0]] * normalized[pairs[:, 1]]).sum(dim=-1)
            same_losses.append((1.0 - similarities).mean())
            same_pair_count += int(pairs.shape[0])
        same_type = torch.stack(same_losses).mean() if same_losses else zero

        different_losses: list[torch.Tensor] = []
        different_pair_count = 0
        for first_type, second_type in combinations(represented_types, 2):
            first = normalized[labels == first_type]
            second = normalized[labels == second_type]
            if first.shape[0] == 0 or second.shape[0] == 0:
                continue
            similarities = first @ second.transpose(0, 1)
            different_losses.append(F.relu(similarities - self.different_type_margin).mean())
            different_pair_count += int(similarities.numel())
        different_type = torch.stack(different_losses).mean() if different_losses else zero

        return PointLossResult(
            loss=same_type + different_type,
            same_type=same_type,
            different_type=different_type,
            point_count=int(embeddings.shape[0]),
            same_pair_count=same_pair_count,
            different_pair_count=different_pair_count,
        )

