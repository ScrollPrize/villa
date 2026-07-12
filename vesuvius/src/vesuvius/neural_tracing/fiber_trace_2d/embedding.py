from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_2d.direction import DirectionSupervision
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStripSample
from vesuvius.neural_tracing.fiber_trace_2d.model import embedding_output


CONTRASTIVE_SIMILARITY_MEAN_TARGET = 0.1


@dataclass(frozen=True)
class ContrastiveEmbeddingMetrics:
    loss: float
    positive_loss: float
    negative_loss: float
    positive_samples: int
    negative_samples: int
    pixel_negative_loss: float = 0.0
    pixel_negative_samples: int = 0
    similarity_mean_loss: float = 0.0
    similarity_mean_value: float = 0.0
    similarity_mean_target: float = CONTRASTIVE_SIMILARITY_MEAN_TARGET
    similarity_mean_samples: int = 0


def contrastive_negative_reachable_mask(
    shape_hw: tuple[int, int],
    *,
    shift_x: float,
    shift_y: float,
    neighborhood_radius: int = 1,
) -> np.ndarray:
    height, width = (int(v) for v in shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"shape_hw must contain positive dimensions, got {shape_hw}")
    sx = float(shift_x)
    sy = float(shift_y)
    if not np.isfinite(sx) or not np.isfinite(sy):
        raise ValueError("contrastive negative reachable shift must be finite")
    radius = max(0, int(neighborhood_radius))
    center_x = (float(width) - 1.0) * 0.5
    center_y = (float(height) - 1.0) * 0.5
    x0 = max(0, int(np.floor(center_x - abs(sx) - float(radius))))
    x1 = min(width - 1, int(np.ceil(center_x + abs(sx) + float(radius))))
    y0 = max(0, int(np.floor(center_y - abs(sy) - float(radius))))
    y1 = min(height - 1, int(np.ceil(center_y + abs(sy) + float(radius))))
    mask = np.zeros((height, width), dtype=bool)
    if x0 <= x1 and y0 <= y1:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


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


def _similarity_mean_loss(
    embeddings: torch.Tensor,
    positive: torch.Tensor,
    patch_indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    patch_embeddings = F.normalize(embeddings[patch_indices], dim=1, eps=1.0e-12)
    similarity = torch.sum(patch_embeddings * positive[:, :, None, None], dim=1)
    similarity01 = (0.5 + 0.5 * similarity).clamp(0.0, 1.0)
    valid_for_cp = valid[patch_indices].to(dtype=similarity01.dtype)
    counts = valid_for_cp.flatten(1).sum(dim=1)
    has_valid = counts > 0.0
    if not bool(has_valid.any().detach().cpu().item()):
        return positive.new_tensor(0.0), positive.new_tensor(0.0), 0
    means = (
        (similarity01 * valid_for_cp).flatten(1).sum(dim=1)[has_valid]
        / counts[has_valid].clamp_min(1.0)
    )
    target_t = means.new_tensor(float(target))
    return torch.square(means - target_t).mean(), means.mean(), int(means.numel())


def contrastive_embedding_loss(
    model_output: torch.Tensor,
    supervision: DirectionSupervision,
    samples: Sequence[FiberStripSample],
    valid_mask: np.ndarray | torch.Tensor,
    *,
    weight: float,
    negative_margin: float = 0.0,
    negative_candidate_mask: np.ndarray | torch.Tensor | None = None,
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
    candidate = valid & ~cp_mask
    similarity_mean_mask = valid
    if negative_candidate_mask is not None:
        reachable = torch.as_tensor(negative_candidate_mask, dtype=torch.bool, device=device)
        if reachable.ndim == 2:
            if tuple(int(v) for v in reachable.shape) != tuple(int(v) for v in valid.shape[1:]):
                raise ValueError("2D negative_candidate_mask shape must match model output spatial shape")
            reachable = reachable.unsqueeze(0)
        elif reachable.ndim == 3:
            if tuple(int(v) for v in reachable.shape) != tuple(int(v) for v in valid.shape):
                raise ValueError("3D negative_candidate_mask shape must match valid_mask shape")
        else:
            raise ValueError("negative_candidate_mask must have shape H,W or N,H,W")
        candidate = candidate & reachable
        similarity_mean_mask = valid & reachable
    negative_flat = torch.nonzero(candidate.reshape(-1), as_tuple=False).flatten()
    if int(negative_flat.numel()) == 0:
        pixel_negative_loss = positive.new_tensor(0.0)
        pixel_negative_count = 0
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
        pixel_negative_loss = torch.square(torch.relu(negative_similarity - float(negative_margin))).mean()
        pixel_negative_count = count

    if pixel_negative_count == 0:
        negative_loss = positive.new_tensor(0.0)
    else:
        negative_loss = pixel_negative_loss
    similarity_mean_loss, similarity_mean_value, similarity_mean_count = _similarity_mean_loss(
        embeddings,
        positive,
        patch_indices,
        similarity_mean_mask,
        target=CONTRASTIVE_SIMILARITY_MEAN_TARGET,
    )
    combined = 0.5 * (positive_loss + negative_loss) + similarity_mean_loss
    weighted = combined * float(weight)
    metrics = ContrastiveEmbeddingMetrics(
        loss=float(weighted.detach().cpu().item()),
        positive_loss=float(positive_loss.detach().cpu().item()),
        negative_loss=float(negative_loss.detach().cpu().item()),
        positive_samples=int(supervision.patch_indices.numel()),
        negative_samples=int(pixel_negative_count),
        pixel_negative_loss=float(pixel_negative_loss.detach().cpu().item()),
        pixel_negative_samples=int(pixel_negative_count),
        similarity_mean_loss=float(similarity_mean_loss.detach().cpu().item()),
        similarity_mean_value=float(similarity_mean_value.detach().cpu().item()),
        similarity_mean_target=CONTRASTIVE_SIMILARITY_MEAN_TARGET,
        similarity_mean_samples=int(similarity_mean_count),
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
