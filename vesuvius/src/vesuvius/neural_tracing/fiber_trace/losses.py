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


@dataclass(frozen=True)
class ContrastivePairSamples:
    flat_indices: Tensor
    pos_a: Tensor
    pos_b: Tensor
    neg_a: Tensor
    neg_b: Tensor


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


def _target_ids_for_labels(labels: Tensor, target_id: Tensor | None) -> Tensor:
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
    return target_id


def sample_contrastive_pair_indices(
    labels: Tensor,
    target_id: Tensor | None = None,
    *,
    max_samples: int = 4096,
) -> ContrastivePairSamples:
    """Sample flattened voxel indices for positive-positive and positive-negative pairs.

    Positives with the same target id are attraction pairs. Explicit negatives
    are only tested against positives; negative-negative relationships are not
    part of the objective.
    """
    target_id = _target_ids_for_labels(labels, target_id)
    lab = labels.reshape(-1)
    ids = target_id.reshape(-1)
    is_positive = (lab == int(POSITIVE_LABEL)) & (ids != int(IGNORE_ID)) & (
        ids != int(NEGATIVE_ONLY_ID)
    )
    is_negative = lab == int(NEGATIVE_LABEL)

    pos_idx = _sample_mask_indices(is_positive, 0)
    neg_idx = _sample_mask_indices(is_negative, 0)
    empty = torch.empty((0,), device=labels.device, dtype=torch.long)
    if pos_idx.numel() == 0:
        return ContrastivePairSamples(empty, empty, empty, empty, empty)

    max_samples = int(max_samples)
    pair_budget = (
        max_samples
        if max_samples > 0
        else int(pos_idx.numel()) + int(neg_idx.numel())
    )
    if pair_budget <= 0:
        return ContrastivePairSamples(empty, empty, empty, empty, empty)

    pos_pair_budget = pair_budget if neg_idx.numel() == 0 else max(1, pair_budget // 2)
    neg_pair_budget = pair_budget - pos_pair_budget

    pos_a, pos_b = _sample_positive_pairs(pos_idx, ids, pos_pair_budget)
    if neg_idx.numel() > 0 and neg_pair_budget > 0:
        neg_a = _take_deterministic(pos_idx, neg_pair_budget)
        neg_b = _take_deterministic(neg_idx, neg_pair_budget)
    else:
        neg_a = pos_idx[:0]
        neg_b = neg_idx[:0]

    pair_indices = torch.cat([pos_a, pos_b, neg_a, neg_b], dim=0)
    if pair_indices.numel() == 0:
        return ContrastivePairSamples(empty, empty, empty, empty, empty)
    flat_indices, inverse = torch.unique(
        pair_indices, sorted=True, return_inverse=True
    )
    pos_count = int(pos_a.numel())
    neg_count = int(neg_a.numel())
    pos_a_local = inverse[:pos_count]
    pos_b_local = inverse[pos_count : 2 * pos_count]
    neg_a_local = inverse[2 * pos_count : 2 * pos_count + neg_count]
    neg_b_local = inverse[2 * pos_count + neg_count :]
    return ContrastivePairSamples(
        flat_indices=flat_indices,
        pos_a=pos_a_local,
        pos_b=pos_b_local,
        neg_a=neg_a_local,
        neg_b=neg_b_local,
    )


def sampled_supervised_contrastive_loss(
    embeddings: Tensor,
    samples: ContrastivePairSamples,
    *,
    temperature: float = 0.1,
) -> Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            f"sampled embeddings must have shape [N, E], got {tuple(embeddings.shape)}"
        )
    if samples.flat_indices.numel() != embeddings.shape[0]:
        raise ValueError(
            "sample count mismatch: "
            f"{samples.flat_indices.numel()} indices for {embeddings.shape[0]} embeddings"
        )

    emb = F.normalize(embeddings, dim=1)
    temp = max(float(temperature), 1e-6)

    total = _zero_like_loss(embeddings)
    pair_count = 0
    if samples.pos_a.numel() > 0:
        pos_cos = torch.sum(
            emb[samples.pos_a] * emb[samples.pos_b], dim=1
        ).clamp(-1.0, 1.0)
        pos_loss = F.softplus(-pos_cos / temp)
        total = total + pos_loss.sum()
        pair_count += int(pos_loss.numel())
    if samples.neg_a.numel() > 0:
        neg_cos = torch.sum(
            emb[samples.neg_a] * emb[samples.neg_b], dim=1
        ).clamp(-1.0, 1.0)
        neg_loss = F.softplus(neg_cos / temp)
        total = total + neg_loss.sum()
        pair_count += int(neg_loss.numel())
    if pair_count == 0:
        return _zero_like_loss(embeddings)
    return total / float(pair_count)


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
    del ignore_index
    if embeddings.ndim != 5:
        raise ValueError(
            f"embeddings must have shape [B, E, D, H, W], got {tuple(embeddings.shape)}"
        )
    if labels.shape != embeddings.shape[:1] + embeddings.shape[2:]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match embeddings spatial shape"
        )

    samples = sample_contrastive_pair_indices(
        labels, target_id, max_samples=max_samples
    )
    flat = embeddings.permute(0, 2, 3, 4, 1).reshape(-1, embeddings.shape[1])
    sampled = flat[samples.flat_indices] if samples.flat_indices.numel() else flat[:0]
    return sampled_supervised_contrastive_loss(
        sampled, samples, temperature=temperature
    )


def compute_fiber_trace_loss(
    outputs: dict[str, Tensor],
    batch: FiberTraceBatch,
    *,
    temperature: float = 0.1,
    contrastive_weight: float = 1.0,
    max_contrastive_samples: int = 4096,
    contrastive_samples: ContrastivePairSamples | None = None,
) -> FiberTraceLoss:
    if contrastive_samples is None:
        contrastive = supervised_contrastive_loss(
            outputs["embedding"],
            batch.labels,
            batch.target_id,
            temperature=temperature,
            max_samples=max_contrastive_samples,
        )
    else:
        contrastive = sampled_supervised_contrastive_loss(
            outputs["embedding"],
            contrastive_samples,
            temperature=temperature,
        )
    total = float(contrastive_weight) * contrastive
    return FiberTraceLoss(total=total, contrastive=contrastive)
