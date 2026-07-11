from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_2d.direction import DirectionSupervision
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStripSample
from vesuvius.neural_tracing.fiber_trace_2d.model import embedding_output


@dataclass(frozen=True)
class ContrastiveEmbeddingMetrics:
    loss: float
    positive_loss: float
    negative_loss: float
    positive_samples: int
    negative_samples: int


def _sample_fiber_ids(
    samples: Sequence[FiberStripSample],
    patch_indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    ids: dict[str, int] = {}
    rows: list[int] = []
    for patch_index in patch_indices.detach().cpu().numpy().astype(np.int64).tolist():
        sample = samples[int(patch_index)]
        key = sample.fiber_path or f"record:{sample.record_index}"
        value = ids.setdefault(key, len(ids))
        rows.append(value)
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def contrastive_embedding_loss(
    model_output: torch.Tensor,
    supervision: DirectionSupervision,
    samples: Sequence[FiberStripSample],
    valid_mask: np.ndarray | torch.Tensor,
    *,
    weight: float,
    negative_margin: float = 0.0,
) -> tuple[torch.Tensor, ContrastiveEmbeddingMetrics]:
    embeddings = embedding_output(model_output)
    if int(embeddings.shape[1]) <= 0:
        raise ValueError("contrastive embedding loss requires model embedding channels")
    if int(supervision.patch_indices.numel()) <= 1:
        raise ValueError("contrastive embedding loss requires at least two CP-local samples")
    if len(samples) != int(model_output.shape[0]):
        raise ValueError("samples length must match flattened model output patch count")

    device = model_output.device
    patch_indices = supervision.patch_indices.to(device=device)
    ys = supervision.y.to(device=device)
    xs = supervision.x.to(device=device)
    positive = embeddings[patch_indices, :, ys, xs]
    positive = F.normalize(positive, dim=1, eps=1.0e-12)
    fiber_ids = _sample_fiber_ids(samples, patch_indices, device=device)

    positive_losses: list[torch.Tensor] = []
    for fiber_id in torch.unique(fiber_ids).tolist():
        group = torch.nonzero(fiber_ids == int(fiber_id), as_tuple=False).flatten()
        if int(group.numel()) <= 1:
            continue
        group_embeddings = positive[group]
        similarity = group_embeddings @ group_embeddings.T
        mask = ~torch.eye(int(group.numel()), dtype=torch.bool, device=device)
        positive_losses.append((1.0 - similarity[mask]).mean())
    if not positive_losses:
        positive_loss = positive.new_tensor(0.0)
    else:
        positive_loss = torch.stack(positive_losses).mean()

    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if valid.ndim != 3:
        raise ValueError("valid_mask must have shape N,H,W")
    if valid.shape[0] != model_output.shape[0] or valid.shape[1:] != model_output.shape[2:]:
        raise ValueError("valid_mask shape must match model output spatial shape")
    cp_mask = torch.zeros_like(valid, dtype=torch.bool)
    cp_mask[patch_indices, ys, xs] = True
    negative_flat = torch.nonzero((valid & ~cp_mask).reshape(-1), as_tuple=False).flatten()
    if int(negative_flat.numel()) == 0:
        negative_loss = positive.new_tensor(0.0)
        negative_count = 0
    else:
        count = int(positive.shape[0])
        selected = negative_flat[
            torch.remainder(
                torch.arange(count, dtype=torch.long, device=device) * 104729 + 17,
                int(negative_flat.numel()),
            )
        ]
        height = int(valid.shape[1])
        width = int(valid.shape[2])
        patch = selected // (height * width)
        rem = selected - patch * height * width
        y = rem // width
        x = rem - y * width
        negative = F.normalize(embeddings[patch, :, y, x], dim=1, eps=1.0e-12)
        negative_similarity = torch.sum(positive * negative, dim=1)
        negative_loss = torch.square(torch.relu(negative_similarity - float(negative_margin))).mean()
        negative_count = count

    combined = 0.5 * (positive_loss + negative_loss)
    weighted = combined * float(weight)
    metrics = ContrastiveEmbeddingMetrics(
        loss=float(weighted.detach().cpu().item()),
        positive_loss=float(positive_loss.detach().cpu().item()),
        negative_loss=float(negative_loss.detach().cpu().item()),
        positive_samples=int(supervision.patch_indices.numel()),
        negative_samples=int(negative_count),
    )
    return weighted, metrics


def embedding_similarity_to_cp(
    model_output: torch.Tensor,
    *,
    patch_index: int,
    cp_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    embeddings = embedding_output(model_output.detach())
    if int(embeddings.shape[1]) <= 0:
        raise ValueError("embedding similarity visualization requires embedding channels")
    patch = int(patch_index)
    if patch < 0 or patch >= int(embeddings.shape[0]):
        raise IndexError(patch_index)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != tuple(int(v) for v in embeddings.shape[2:]):
        raise ValueError("valid_mask shape must match embedding patch spatial shape")
    cp = np.rint(np.asarray(cp_xy, dtype=np.float32)).astype(np.int64)
    if cp.shape != (2,):
        raise ValueError("cp_xy must have shape (2,)")
    x = int(np.clip(cp[0], 0, int(embeddings.shape[3]) - 1))
    y = int(np.clip(cp[1], 0, int(embeddings.shape[2]) - 1))
    emb = F.normalize(embeddings[patch], dim=0, eps=1.0e-12)
    cp_embedding = F.normalize(embeddings[patch, :, y, x], dim=0, eps=1.0e-12)
    similarity = torch.sum(emb * cp_embedding.view(-1, 1, 1), dim=0)
    out = (0.5 + 0.5 * similarity).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    return np.where(valid, out, 0.0).astype(np.float32)
