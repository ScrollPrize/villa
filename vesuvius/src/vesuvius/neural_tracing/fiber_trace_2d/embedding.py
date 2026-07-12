from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_2d.direction import DirectionSupervision, line_cp_and_tangent_xy
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStripSample
from vesuvius.neural_tracing.fiber_trace_2d.model import embedding_output, presence_output


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


@dataclass(frozen=True)
class PresenceMetrics:
    loss: float
    positive_loss: float
    negative_loss: float
    positive_samples: int
    negative_samples: int


@dataclass(frozen=True)
class _CpPixelRows:
    patch_indices: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    fiber_ids: torch.Tensor
    control_point_ids: torch.Tensor


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


def _valid_mask_numpy(valid_mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(valid_mask, torch.Tensor):
        return valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(valid_mask, dtype=bool)


def _sample_fiber_key(sample: FiberStripSample, patch_index: int) -> str:
    fiber_path = getattr(sample, "fiber_path", None)
    if fiber_path:
        return str(fiber_path)
    return f"record:{getattr(sample, 'record_index', int(patch_index))}"


def _sample_control_point_key(sample: FiberStripSample, patch_index: int) -> str:
    cp_index = getattr(sample, "control_point_index", None)
    if cp_index is None:
        return f"patch:{int(patch_index)}"
    return str(int(cp_index))


def _fallback_supervision_positions(supervision: DirectionSupervision) -> dict[int, tuple[int, int]]:
    fallback: dict[int, tuple[int, int]] = {}
    patches = supervision.patch_indices.detach().cpu().numpy().astype(np.int64)
    ys = supervision.y.detach().cpu().numpy().astype(np.int64)
    xs = supervision.x.detach().cpu().numpy().astype(np.int64)
    for patch, y, x in zip(patches.tolist(), ys.tolist(), xs.tolist(), strict=True):
        fallback.setdefault(int(patch), (int(y), int(x)))
    return fallback


def _cp_pixel_rows(
    samples: Sequence[FiberStripSample],
    valid_mask: np.ndarray | torch.Tensor,
    supervision: DirectionSupervision,
    *,
    device: torch.device,
) -> _CpPixelRows:
    valid_np = _valid_mask_numpy(valid_mask)
    if valid_np.ndim != 3:
        raise ValueError("valid_mask must have shape N,H,W")
    if len(samples) != int(valid_np.shape[0]):
        raise ValueError("samples length must match valid_mask patch count")
    height, width = int(valid_np.shape[1]), int(valid_np.shape[2])
    fallback = _fallback_supervision_positions(supervision)
    fiber_map: dict[str, int] = {}
    cp_map: dict[tuple[str, str], int] = {}
    patch_rows: list[int] = []
    y_rows: list[int] = []
    x_rows: list[int] = []
    fiber_rows: list[int] = []
    cp_rows: list[int] = []
    for patch_index, sample in enumerate(samples):
        cp_xy = getattr(sample, "control_point_xy", None)
        if cp_xy is None:
            fallback_pos = fallback.get(int(patch_index))
            if fallback_pos is None:
                continue
            y, x = fallback_pos
        else:
            cp_and_tangent = line_cp_and_tangent_xy(
                getattr(sample, "line_xy", np.empty((0, 2), dtype=np.float32)),
                cp_xy,
            )
            cp = np.asarray(cp_xy if cp_and_tangent is None else cp_and_tangent[0], dtype=np.float32)
            if cp.shape != (2,) or not bool(np.isfinite(cp).all()):
                continue
            rounded = np.rint(cp).astype(np.int64)
            x = int(rounded[0])
            y = int(rounded[1])
        if not (0 <= y < height and 0 <= x < width):
            continue
        if not bool(valid_np[int(patch_index), y, x]):
            continue
        fiber_key = _sample_fiber_key(sample, patch_index)
        fiber_id = fiber_map.setdefault(fiber_key, len(fiber_map))
        cp_key = _sample_control_point_key(sample, patch_index)
        cp_id = cp_map.setdefault((fiber_key, cp_key), len(cp_map))
        patch_rows.append(int(patch_index))
        y_rows.append(int(y))
        x_rows.append(int(x))
        fiber_rows.append(int(fiber_id))
        cp_rows.append(int(cp_id))
    return _CpPixelRows(
        patch_indices=torch.as_tensor(patch_rows, dtype=torch.long, device=device),
        y=torch.as_tensor(y_rows, dtype=torch.long, device=device),
        x=torch.as_tensor(x_rows, dtype=torch.long, device=device),
        fiber_ids=torch.as_tensor(fiber_rows, dtype=torch.long, device=device),
        control_point_ids=torch.as_tensor(cp_rows, dtype=torch.long, device=device),
    )


def _candidate_mask_tensor(
    valid: torch.Tensor,
    negative_candidate_mask: np.ndarray | torch.Tensor | None,
) -> torch.Tensor:
    if negative_candidate_mask is None:
        return valid
    reachable = torch.as_tensor(negative_candidate_mask, dtype=torch.bool, device=valid.device)
    if reachable.ndim == 2:
        if tuple(int(v) for v in reachable.shape) != tuple(int(v) for v in valid.shape[1:]):
            raise ValueError("2D negative_candidate_mask shape must match model output spatial shape")
        reachable = reachable.unsqueeze(0)
    elif reachable.ndim == 3:
        if tuple(int(v) for v in reachable.shape) != tuple(int(v) for v in valid.shape):
            raise ValueError("3D negative_candidate_mask shape must match valid_mask shape")
    else:
        raise ValueError("negative_candidate_mask must have shape H,W or N,H,W")
    return valid & reachable


def presence_loss(
    model_output: torch.Tensor,
    supervision: DirectionSupervision,
    samples: Sequence[FiberStripSample],
    valid_mask: np.ndarray | torch.Tensor,
    *,
    weight: float,
    negative_candidate_mask: np.ndarray | torch.Tensor | None = None,
    presence_channels: int = 1,
) -> tuple[torch.Tensor, PresenceMetrics]:
    presence = presence_output(model_output, presence_channels=presence_channels)
    if int(presence.shape[1]) != 1:
        raise ValueError("presence loss requires exactly one model presence channel")
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=model_output.device)
    if valid.ndim != 3:
        raise ValueError("valid_mask must have shape N,H,W")
    if valid.shape[0] != model_output.shape[0] or valid.shape[1:] != model_output.shape[2:]:
        raise ValueError("valid_mask shape must match model output spatial shape")
    rows = _cp_pixel_rows(samples, valid_mask, supervision, device=model_output.device)
    if int(rows.patch_indices.numel()) == 0:
        raise ValueError("presence loss requires at least one valid transformed CP pixel")
    cp_mask = torch.zeros_like(valid, dtype=torch.bool)
    cp_mask[rows.patch_indices, rows.y, rows.x] = True
    candidate = _candidate_mask_tensor(valid, negative_candidate_mask)
    negative_mask = candidate & ~cp_mask
    probs = presence[:, 0].clamp(1.0e-6, 1.0 - 1.0e-6)
    positive_values = probs[cp_mask]
    negative_values = probs[negative_mask]
    if int(negative_values.numel()) == 0:
        negative_loss = probs.new_tensor(0.0)
    else:
        negative_loss = -torch.log1p(-negative_values).mean()
    positive_loss = -torch.log(positive_values).mean()
    loss = 0.5 * (positive_loss + negative_loss) * float(weight)
    metrics = PresenceMetrics(
        loss=float(loss.detach().cpu().item()),
        positive_loss=float(positive_loss.detach().cpu().item()),
        negative_loss=float(negative_loss.detach().cpu().item()),
        positive_samples=int(positive_values.numel()),
        negative_samples=int(negative_values.numel()),
    )
    return loss, metrics


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
    presence_channels: int = 0,
) -> tuple[torch.Tensor, ContrastiveEmbeddingMetrics]:
    embeddings = embedding_output(model_output, presence_channels=presence_channels)
    if int(embeddings.shape[1]) <= 0:
        raise ValueError("contrastive embedding loss requires model embedding channels")
    if len(samples) != int(model_output.shape[0]):
        raise ValueError("samples length must match flattened model output patch count")

    device = model_output.device
    rows = _cp_pixel_rows(samples, valid_mask, supervision, device=device)
    if int(rows.patch_indices.numel()) <= 1:
        raise ValueError("contrastive embedding loss requires at least two valid transformed CP pixels")
    patch_indices = rows.patch_indices
    ys = rows.y
    xs = rows.x
    positive = embeddings[patch_indices, :, ys, xs]
    positive = F.normalize(positive, dim=1, eps=1.0e-12)
    fiber_ids = rows.fiber_ids
    control_point_ids = rows.control_point_ids

    similarity = positive @ positive.T
    same_fiber = fiber_ids[:, None] == fiber_ids[None, :]
    different_control_point = control_point_ids[:, None] != control_point_ids[None, :]
    positive_candidate = same_fiber & different_control_point
    has_candidate = positive_candidate.any(dim=1)
    if not bool(has_candidate.any().detach().cpu().item()):
        positive_loss = positive.new_tensor(0.0)
        positive_pair_count = 0
    else:
        masked_similarity = similarity.masked_fill(~positive_candidate, -2.0)
        best_similarity = masked_similarity.max(dim=1).values[has_candidate]
        positive_loss = (1.0 - best_similarity).mean()
        positive_pair_count = int(best_similarity.numel())

    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if valid.ndim != 3:
        raise ValueError("valid_mask must have shape N,H,W")
    if valid.shape[0] != model_output.shape[0] or valid.shape[1:] != model_output.shape[2:]:
        raise ValueError("valid_mask shape must match model output spatial shape")
    cp_mask = torch.zeros_like(valid, dtype=torch.bool)
    cp_mask[patch_indices, ys, xs] = True
    similarity_mean_mask = _candidate_mask_tensor(valid, negative_candidate_mask)
    candidate = similarity_mean_mask & ~cp_mask
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
        positive_samples=int(positive_pair_count),
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
    presence_channels: int = 0,
) -> np.ndarray:
    embeddings = embedding_output(model_output.detach(), presence_channels=presence_channels)
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
