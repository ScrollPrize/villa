from __future__ import annotations

from dataclasses import dataclass

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
    fw: Tensor
    up: Tensor


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


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    target_id: Tensor | None = None,
    *,
    temperature: float = 0.1,
    ignore_index: int = IGNORE_INDEX,
    max_samples: int = 4096,
) -> Tensor:
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
    max_samples = int(max_samples)
    if max_samples > 0 and pos_idx.numel() + neg_idx.numel() > max_samples:
        if pos_idx.numel() == 0:
            pos_budget = 0
        elif neg_idx.numel() == 0:
            pos_budget = max_samples
        else:
            min_positive_budget = 2 if max_samples >= 2 else 1
            pos_budget = min(
                pos_idx.numel(), max(min_positive_budget, max_samples // 2)
            )
        neg_budget = min(neg_idx.numel(), max_samples - pos_budget)
        remaining = max_samples - pos_budget - neg_budget
        if remaining > 0:
            extra_pos = min(pos_idx.numel() - pos_budget, remaining)
            pos_budget += extra_pos
            remaining -= extra_pos
        if remaining > 0:
            neg_budget += min(neg_idx.numel() - neg_budget, remaining)
        pos_idx = _sample_mask_indices(is_positive, int(pos_budget))
        neg_idx = _sample_mask_indices(is_negative, int(neg_budget))

    sample_idx = torch.cat([pos_idx, neg_idx], dim=0)
    if sample_idx.numel() < 2:
        return _zero_like_loss(embeddings)

    emb = emb[sample_idx]
    lab = lab[sample_idx]
    ids = ids[sample_idx]
    positive_sample = (lab == int(POSITIVE_LABEL)) & (ids != int(IGNORE_ID)) & (
        ids != int(NEGATIVE_ONLY_ID)
    )
    if emb.shape[0] < 2:
        return _zero_like_loss(embeddings)

    emb = F.normalize(emb, dim=1)
    logits = emb @ emb.T / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive = (
        positive_sample[:, None]
        & positive_sample[None, :]
        & (ids[:, None] == ids[None, :])
        & ~eye
    )
    anchors = positive_sample & positive.any(dim=1)
    if not bool(anchors.any()):
        return _zero_like_loss(embeddings)

    logits_no_self = logits.masked_fill(eye, float("-inf"))
    log_den = torch.logsumexp(logits_no_self, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    return -(log_num[anchors] - log_den[anchors]).mean()


def masked_cosine_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return _zero_like_loss(pred)
    pred_n = F.normalize(pred, dim=1)
    target_n = F.normalize(target, dim=1)
    cosine = torch.sum(pred_n * target_n, dim=1).clamp(-1.0, 1.0)
    return (1.0 - cosine)[mask].mean()


def sign_ambiguous_up_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return _zero_like_loss(pred)
    pred_n = F.normalize(pred, dim=1)
    target_n = F.normalize(target, dim=1)
    cosine = torch.sum(pred_n * target_n, dim=1).clamp(-1.0, 1.0)
    return (1.0 - cosine.abs())[mask].mean()


def compute_fiber_trace_loss(
    outputs: dict[str, Tensor],
    batch: FiberTraceBatch,
    *,
    temperature: float = 0.1,
    contrastive_weight: float = 1.0,
    fw_weight: float = 1.0,
    up_weight: float = 1.0,
    max_contrastive_samples: int = 4096,
) -> FiberTraceLoss:
    contrastive = supervised_contrastive_loss(
        outputs["embedding"],
        batch.labels,
        batch.target_id,
        temperature=temperature,
        max_samples=max_contrastive_samples,
    )
    positive_mask = batch.labels == POSITIVE_LABEL
    fw = masked_cosine_loss(outputs["fw"], batch.target_fw_xyz, positive_mask)
    up = sign_ambiguous_up_loss(
        outputs["up"], batch.target_up_xyz, positive_mask & batch.target_up_valid
    )
    total = (
        float(contrastive_weight) * contrastive
        + float(fw_weight) * fw
        + float(up_weight) * up
    )
    return FiberTraceLoss(total=total, contrastive=contrastive, fw=fw, up=up)
