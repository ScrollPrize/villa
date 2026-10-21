"""Shared sufficient statistics; no dependency on trainer or data loading."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossStatistics:
    """Differentiable numerators and non-differentiable normalization counts."""

    direction_sum: torch.Tensor
    direction_count: torch.Tensor
    positive_presence_sum: torch.Tensor
    positive_presence_count: torch.Tensor
    negative_presence_sum: torch.Tensor
    negative_presence_count: torch.Tensor
    angle_sum: torch.Tensor
    angle_count: torch.Tensor
    branch0_sum: torch.Tensor
    branch1_sum: torch.Tensor
    branch_count: torch.Tensor
    selected_score_sum: torch.Tensor
    selected_score_count: torch.Tensor
    positive_embedding_sum: torch.Tensor | None = None
    positive_embedding_count: torch.Tensor | None = None
    negative_embedding_sum: torch.Tensor | None = None
    negative_embedding_count: torch.Tensor | None = None

    def __post_init__(self):
        for name in ("positive_embedding_sum", "positive_embedding_count",
                     "negative_embedding_sum", "negative_embedding_count"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, torch.zeros_like(self.direction_count))

    def __add__(self, other: "LossStatistics") -> "LossStatistics":
        return LossStatistics(**{field: getattr(self, field) + getattr(other, field)
                                 for field in self.__dataclass_fields__})

    def detached(self):
        return LossStatistics(**{name: value.detach().float() for name, value in vars(self).items()})

    def packed(self):
        return torch.stack([getattr(self, name).detach().float().reshape(())
                            for name in self.__dataclass_fields__])

    @classmethod
    def unpack(cls, values):
        if values.numel() != len(cls.__dataclass_fields__):
            raise ValueError("incorrect loss statistics size")
        return cls(**dict(zip(cls.__dataclass_fields__, values.unbind(), strict=True)))
