from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatch
from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_ID,
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    NEGATIVE_ONLY_ID,
    POSITIVE_LABEL,
)


@dataclass(frozen=True)
class FiberTraceLoss:
    total: Tensor
    contrastive: Tensor


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _sample_mask_indices(mask: Tensor, max_count: int) -> Tensor:
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    max_count = int(max_count)
    if max_count > 0 and idx.numel() > max_count:
        keep = torch.linspace(
            0, idx.numel() - 1, steps=max_count, device=idx.device
        ).to(torch.long)
        idx = idx[keep]
    return idx


def _deterministic_positions(length: int, count: int, *, device: torch.device) -> Tensor:
    length = int(length)
    count = int(count)
    if length <= 0 or count <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if length == 1:
        return torch.zeros((count,), device=device, dtype=torch.long)
    stride = min(104729, length - 1)
    while math.gcd(stride, length) != 1:
        stride -= 1
        if stride <= 0:
            stride = 1
            break
    return (torch.arange(count, device=device, dtype=torch.long) * int(stride)) % length


def _take_deterministic(idx: Tensor, count: int) -> Tensor:
    positions = _deterministic_positions(
        int(idx.numel()), int(count), device=idx.device
    )
    if positions.numel() == 0:
        return idx[:0]
    return idx[positions]


def _sample_positive_pairs(pos_idx: Tensor, ids: Tensor, count: int) -> tuple[Tensor, Tensor]:
    count = int(count)
    if count <= 0 or pos_idx.numel() < 2:
        empty = pos_idx[:0]
        return empty, empty

    left_parts: list[Tensor] = []
    right_parts: list[Tensor] = []
    for target_id in torch.unique(ids[pos_idx]):
        group = pos_idx[ids[pos_idx] == target_id]
        if group.numel() < 2:
            continue
        left_parts.append(group)
        right_parts.append(torch.roll(group, shifts=-1, dims=0))
    if not left_parts:
        empty = pos_idx[:0]
        return empty, empty

    left = torch.cat(left_parts, dim=0)
    right = torch.cat(right_parts, dim=0)
    positions = _deterministic_positions(int(left.numel()), count, device=left.device)
    return left[positions], right[positions]


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    target_id: Tensor | None = None,
    *,
    temperature: float = 0.1,
    ignore_index: int = IGNORE_INDEX,
    max_samples: int = 4096,
) -> Tensor:
    """Sample positive-positive and positive-negative embedding pairs.

    Positives with the same target id are attraction pairs. Explicit negatives
    are only tested against positives; negative-negative relationships are not
    part of the objective.
    """
    if embeddings.ndim != 5:
        raise ValueError(
            f"embeddings must have shape [B, E, D, H, W], got {tuple(embeddings.shape)}"
        )
    if labels.shape != embeddings.shape[:1] + embeddings.shape[2:]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match embeddings spatial shape"
        )
    if target_id is None:
        target_id = torch.full_like(labels, int(IGNORE_ID))
        target_id = torch.where(
            labels == POSITIVE_LABEL,
            torch.ones_like(target_id),
            target_id,
        )
        target_id = torch.where(
            labels == NEGATIVE_LABEL,
            torch.full_like(target_id, int(NEGATIVE_ONLY_ID)),
            target_id,
        )
    elif target_id.shape != labels.shape:
        raise ValueError(
            f"target_id shape {tuple(target_id.shape)} does not match labels shape"
        )

    emb = embeddings.permute(0, 2, 3, 4, 1).reshape(-1, embeddings.shape[1])
    lab = labels.reshape(-1)
    ids = target_id.reshape(-1)
    is_positive = (lab == int(POSITIVE_LABEL)) & (ids != int(IGNORE_ID)) & (
        ids != int(NEGATIVE_ONLY_ID)
    )
    is_negative = lab == int(NEGATIVE_LABEL)

    pos_idx = _sample_mask_indices(is_positive, 0)
    neg_idx = _sample_mask_indices(is_negative, 0)
    if pos_idx.numel() == 0:
        return _zero_like_loss(embeddings)

    max_samples = int(max_samples)
    pair_budget = (
        max_samples
        if max_samples > 0
        else int(pos_idx.numel()) + int(neg_idx.numel())
    )
    if pair_budget <= 0:
        return _zero_like_loss(embeddings)

    emb = F.normalize(emb, dim=1)
    temp = max(float(temperature), 1e-6)
    pos_pair_budget = pair_budget if neg_idx.numel() == 0 else max(1, pair_budget // 2)
    neg_pair_budget = pair_budget - pos_pair_budget

    pos_a, pos_b = _sample_positive_pairs(pos_idx, ids, pos_pair_budget)
    if neg_idx.numel() > 0 and neg_pair_budget > 0:
        neg_a = _take_deterministic(pos_idx, neg_pair_budget)
        neg_b = _take_deterministic(neg_idx, neg_pair_budget)
    else:
        neg_a = pos_idx[:0]
        neg_b = neg_idx[:0]

    total = _zero_like_loss(embeddings)
    pair_count = 0
    if pos_a.numel() > 0:
        pos_cos = torch.sum(emb[pos_a] * emb[pos_b], dim=1).clamp(-1.0, 1.0)
        pos_loss = F.softplus(-pos_cos / temp)
        total = total + pos_loss.sum()
        pair_count += int(pos_loss.numel())
    if neg_a.numel() > 0:
        neg_cos = torch.sum(emb[neg_a] * emb[neg_b], dim=1).clamp(-1.0, 1.0)
        neg_loss = F.softplus(neg_cos / temp)
        total = total + neg_loss.sum()
        pair_count += int(neg_loss.numel())
    if pair_count == 0:
        return _zero_like_loss(embeddings)
    return total / float(pair_count)


def compute_fiber_trace_loss(
    outputs: dict[str, Tensor],
    batch: FiberTraceBatch,
    *,
    temperature: float = 0.1,
    contrastive_weight: float = 1.0,
    max_contrastive_samples: int = 4096,
) -> FiberTraceLoss:
    contrastive = supervised_contrastive_loss(
        outputs["embedding"],
        batch.labels,
        batch.target_id,
        temperature=temperature,
        max_samples=max_contrastive_samples,
    )
    total = float(contrastive_weight) * contrastive
    return FiberTraceLoss(total=total, contrastive=contrastive)
