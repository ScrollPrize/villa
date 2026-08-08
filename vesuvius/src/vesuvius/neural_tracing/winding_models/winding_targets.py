"""Dense supervision targets, losses, and decoding for the winding model.

The dataset labels each ray with ordered winding crossings; training needs
dense per-ray-sample targets:

- ``phase``: relative winding coordinate, piecewise linear through the
  crossings (exactly k at crossing k). The sign is canonicalized so phase
  increases along the ray, keeping scroll chirality out of the learning
  problem; the fit_spiral consumer knows each ray's winding direction and
  flips per ray. The free offset is handled by a shift-invariant loss (the
  consumer's E-step registration likewise absorbs it).
- ``crossing``: a narrow Gaussian heatmap at the crossings whose nearest
  sample is pinned to exactly one, giving the penalty-reduced focal loss an
  exact positive set.

Negatives are only supervised where ``winding_valid`` holds: spans that may
contain unlabeled wraps must not be taught as "no crossing".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Below this heatmap value a sample no longer counts as crossing evidence and
# its supervision falls back to the winding-validity mask alone.
_CROSSING_SUPPORT = 0.05


def render_targets(
    sample: dict, *, crossing_sigma_wv: float
) -> dict[str, torch.Tensor]:
    """Densify one dataset sample's crossing labels along the ray."""
    ray_length = int(sample["ray_length"])
    spacing = float(sample["ray_extent"]) / (ray_length - 1)
    sample_ts = np.arange(ray_length, dtype=np.float64) * spacing
    crossing_t = np.asarray(sample["crossing_t"], dtype=np.float64)
    indices = np.asarray(sample["winding_indices"], dtype=np.float64)
    if indices[-1] < indices[0]:
        indices = -indices

    phase = np.interp(sample_ts, crossing_t, indices)

    deviation = np.abs(sample_ts[:, None] - crossing_t[None, :]).min(axis=1)
    heatmap = np.exp(-0.5 * (deviation / crossing_sigma_wv) ** 2)
    nearest = np.clip(np.rint(crossing_t / spacing).astype(int), 0, ray_length - 1)
    heatmap[nearest] = 1.0

    winding_valid = np.asarray(sample["winding_valid"], dtype=bool)
    return {
        "phase_target": torch.from_numpy(phase.astype(np.float32)),
        "phase_valid": torch.from_numpy(winding_valid.copy()),
        "crossing_target": torch.from_numpy(heatmap.astype(np.float32)),
        "crossing_valid": torch.from_numpy(
            winding_valid | (heatmap > _CROSSING_SUPPORT)
        ),
    }


def collate_winding_batch(batch: list[dict], *, crossing_sigma_wv: float) -> dict:
    """Stack dataset samples together with their rendered dense targets.

    Crossing counts vary per ray; positions are padded with NaN alongside a
    count tensor so metrics and visualization can recover them.
    """
    rendered = [
        render_targets(sample, crossing_sigma_wv=crossing_sigma_wv)
        for sample in batch
    ]
    max_crossings = max(len(sample["crossing_t"]) for sample in batch)
    crossing_t = torch.full((len(batch), max_crossings), float("nan"))
    for row, sample in enumerate(batch):
        crossing_t[row, : len(sample["crossing_t"])] = sample["crossing_t"]
    collated = {
        "plane_images": torch.stack([sample["plane_images"] for sample in batch]),
        "plane_valid": torch.stack([sample["plane_valid"] for sample in batch]),
        "crossing_t": crossing_t,
        "num_crossings": torch.tensor(
            [len(sample["crossing_t"]) for sample in batch], dtype=torch.int64
        ),
    }
    for key in ("phase_target", "phase_valid", "crossing_target", "crossing_valid"):
        collated[key] = torch.stack([targets[key] for targets in rendered])
    return collated


def phase_loss(
    phase_pred: torch.Tensor,
    phase_target: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    huber_delta: float = 0.25,
) -> torch.Tensor:
    """Shift-invariant masked Huber loss on the relative winding phase.

    Prediction and target are mean-centered over each ray's valid samples, so
    only phase differences are supervised. Rays with fewer than two valid
    samples contribute nothing.
    """
    weight = phase_valid.to(phase_pred.dtype)
    count = weight.sum(dim=-1)
    denominator = count.clamp_min(1.0)
    pred_mean = (phase_pred * weight).sum(dim=-1) / denominator
    target_mean = (phase_target * weight).sum(dim=-1) / denominator
    residual = (phase_pred - pred_mean[:, None]) - (
        phase_target - target_mean[:, None]
    )
    per_sample = F.huber_loss(
        residual, torch.zeros_like(residual), delta=huber_delta, reduction="none"
    )
    per_ray = (per_sample * weight).sum(dim=-1) / denominator
    active = (count >= 2).to(phase_pred.dtype)
    return (per_ray * active).sum() / active.sum().clamp_min(1.0)


def crossing_loss(
    crossing_logits: torch.Tensor,
    crossing_target: torch.Tensor,
    crossing_valid: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet) on the masked crossing heatmap."""
    prob = torch.sigmoid(crossing_logits.float()).clamp(1e-5, 1.0 - 1e-5)
    valid = crossing_valid.to(prob.dtype)
    positive = (crossing_target >= 1.0).to(prob.dtype) * valid
    negative = (1.0 - positive) * valid
    positive_loss = -torch.log(prob) * (1.0 - prob) ** alpha * positive
    negative_loss = (
        -torch.log(1.0 - prob)
        * prob**alpha
        * (1.0 - crossing_target) ** beta
        * negative
    )
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)


def extract_peaks(
    prob: np.ndarray, *, threshold: float = 0.3, min_distance: int = 2
) -> np.ndarray:
    """Sample indices of local maxima above ``threshold``, greedy NMS."""
    order = np.argsort(prob, kind="stable")[::-1]
    suppressed = np.zeros(len(prob), dtype=bool)
    kept = []
    for index in order:
        if prob[index] < threshold:
            break
        if suppressed[index]:
            continue
        kept.append(int(index))
        suppressed[max(0, index - min_distance) : index + min_distance + 1] = True
    return np.sort(np.asarray(kept, dtype=np.int64))


def match_crossings(
    predicted_ts: np.ndarray, target_ts: np.ndarray, *, tolerance: float
) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (true pos, false pos, false neg)."""
    remaining = [float(t) for t in target_ts]
    true_positives = 0
    for t in sorted(float(t) for t in predicted_ts):
        if not remaining:
            break
        nearest = min(range(len(remaining)), key=lambda i: abs(remaining[i] - t))
        if abs(remaining[nearest] - t) <= tolerance:
            remaining.pop(nearest)
            true_positives += 1
    return (
        true_positives,
        len(predicted_ts) - true_positives,
        len(remaining),
    )
