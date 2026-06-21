from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatch
from vesuvius.neural_tracing.fiber_trace.labels import IGNORE_INDEX, POSITIVE_LABEL


@dataclass(frozen=True)
class FiberTraceLoss:
    total: Tensor
    contrastive: Tensor
    fw: Tensor
    up: Tensor


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
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

    emb = embeddings.permute(0, 2, 3, 4, 1).reshape(-1, embeddings.shape[1])
    lab = labels.reshape(-1)
    valid = lab != int(ignore_index)
    emb = emb[valid]
    lab = lab[valid]
    if emb.shape[0] < 2:
        return _zero_like_loss(embeddings)

    max_samples = int(max_samples)
    if max_samples > 0 and emb.shape[0] > max_samples:
        idx = torch.linspace(
            0, emb.shape[0] - 1, steps=max_samples, device=emb.device
        ).to(torch.long)
        emb = emb[idx]
        lab = lab[idx]

    emb = F.normalize(emb, dim=1)
    logits = emb @ emb.T / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive = (lab[:, None] == lab[None, :]) & ~eye
    anchors = positive.any(dim=1)
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
        temperature=temperature,
        max_samples=max_contrastive_samples,
    )
    positive_mask = batch.labels == POSITIVE_LABEL
    fw = masked_cosine_loss(outputs["fw"], batch.target_fw, positive_mask)
    up = sign_ambiguous_up_loss(outputs["up"], batch.target_up, positive_mask)
    total = (
        float(contrastive_weight) * contrastive
        + float(fw_weight) * fw
        + float(up_weight) * up
    )
    return FiberTraceLoss(total=total, contrastive=contrastive, fw=fw, up=up)
