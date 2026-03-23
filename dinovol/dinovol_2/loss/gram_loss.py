from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """Mean-squared error between student and teacher Gram matrices."""

    def __init__(
        self,
        *,
        apply_norm: bool = True,
        remove_neg: bool = False,
        remove_only_teacher_neg: bool = False,
    ) -> None:
        super().__init__()
        if remove_neg and remove_only_teacher_neg:
            raise ValueError("remove_neg and remove_only_teacher_neg are mutually exclusive.")
        self.apply_norm = bool(apply_norm)
        self.remove_neg = bool(remove_neg)
        self.remove_only_teacher_neg = bool(remove_only_teacher_neg)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        *,
        img_level: bool = True,
    ) -> torch.Tensor:
        if img_level and (student_features.ndim != 3 or teacher_features.ndim != 3):
            raise ValueError("img_level Gram loss expects (B, N, D) student and teacher features.")
        if student_features.ndim != teacher_features.ndim:
            raise ValueError(
                f"student and teacher features must have the same rank, got {student_features.ndim} and {teacher_features.ndim}."
            )

        student = student_features.float()
        teacher = teacher_features.float()

        if self.apply_norm:
            student = F.normalize(student, dim=-1)
            teacher = F.normalize(teacher, dim=-1)

        if not img_level:
            if student.ndim == 3:
                # Preserve per-sample independence by forming feature-feature
                # Gram matrices for each sample instead of mixing tokens across
                # the whole batch.
                student = student.transpose(-1, -2)
                teacher = teacher.transpose(-1, -2)
            elif student.ndim != 2:
                raise ValueError("non-img-level Gram loss expects (N, D) or (B, N, D) features.")

        teacher_sim = torch.matmul(teacher, teacher.transpose(-1, -2))
        student_sim = torch.matmul(student, student.transpose(-1, -2))

        if self.remove_neg:
            teacher_sim = teacher_sim.clamp_min_(0.0)
            student_sim = student_sim.clamp_min_(0.0)
        elif self.remove_only_teacher_neg:
            teacher_neg = teacher_sim < 0
            teacher_sim = teacher_sim.masked_fill(teacher_neg, 0.0)
            student_sim = student_sim.masked_fill(teacher_neg & (student_sim < 0), 0.0)

        return self.mse_loss(student_sim, teacher_sim)
