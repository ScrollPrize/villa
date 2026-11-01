"""Fiber probabilities for a zone of CT, predicted with a sliding window."""

from __future__ import annotations

import contextlib
from typing import Callable, Sequence

import numpy as np
import torch

from vesuvius.models.run.tta import infer_with_tta


# The fiber models were trained on zones z-scored with their own statistics,
# the deviation clamped to at least 10 grey levels.
MIN_STD = 10.0
FIBER_LABELS = {"background": 0, "vt-fiber": 1, "hz-fiber": 2, "intersection": 3}

Progress = Callable[[int, int], None]


def window_starts(size: int, patch: int) -> list[int]:
    """Origins of half-overlapping windows; the last one ends at the border."""
    starts = list(range(0, size - patch + 1, max(1, patch // 2)))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def gaussian_weight(patch: Sequence[int]) -> np.ndarray:
    axes = [np.exp(-0.5 * ((np.arange(p, dtype=np.float32) - (p - 1) / 2) / (p / 8)) ** 2) for p in patch]
    return np.maximum(axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :], 1e-4)


def normalize(ct: np.ndarray) -> np.ndarray:
    """A float32 copy of ``ct``, z-scored with its own statistics."""
    image = np.array(ct, dtype=np.float32)
    # Plane by plane in float64: whole-volume float64 temporaries would be
    # twice the size of the zone.
    mean = sum(float(plane.sum(dtype=np.float64)) for plane in image) / image.size
    variance = sum(float(np.square(plane - mean, dtype=np.float64).sum()) for plane in image) / image.size
    image -= mean
    image /= max(variance**0.5, MIN_STD)
    return image


def _autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def _accumulators(shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.zeros((3,) + shape, device=device), torch.zeros(shape, device=device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return torch.zeros((3,) + shape), torch.zeros(shape)


@torch.inference_mode()
def predict_fibers(
    network: torch.nn.Module,
    ct: np.ndarray,
    patch_size: Sequence[int],
    *,
    device: torch.device,
    mirror: bool = True,
    progress: Progress | None = None,
) -> np.ndarray:
    """Probabilities of shape ``ct.shape + (3,)``: vertical, horizontal, intersection.

    Values are uint8, 255 for certainty. ``network`` maps a ``(1, 1, *patch)``
    tensor to logits of the four ``FIBER_LABELS`` classes. Overlapping windows
    are blended with Gaussian weights; with ``mirror``, each window averages
    the logits of its eight flips, as nnU-Net test-time mirroring does.
    """
    patch = tuple(int(p) for p in patch_size)
    shape = ct.shape
    image = normalize(ct)
    # Zones smaller than a window get context from their edge values.
    pad = [((max(0, p - n)) // 2, (max(0, p - n) + 1) // 2) for n, p in zip(shape, patch)]
    if any(before or after for before, after in pad):
        image = np.pad(image, pad, mode="edge")
    depth, height, width = image.shape
    zs, ys, xs = (window_starts(n, p) for n, p in zip(image.shape, patch))
    total = len(zs) * len(ys) * len(xs)
    # Windows are visited in z order: planes before the current window row
    # are final, so the accumulators span one window depth only.
    sums, counts = _accumulators((patch[0], height, width), device)
    weight = torch.from_numpy(gaussian_weight(patch)).to(sums.device)
    output = np.empty(shape + (3,), dtype=np.uint8)
    rows = slice(pad[1][0], pad[1][0] + shape[1])
    columns = slice(pad[2][0], pad[2][0] + shape[2])
    base = 0

    def finish(end: int) -> None:
        nonlocal base
        n = end - base
        if n <= 0:
            return
        if not bool((counts[:n, rows, columns] > 0).all()):
            raise RuntimeError("A voxel was not covered by any window")
        probability = (sums[:, :n, rows, columns] / counts[:n, rows, columns]).cpu().numpy()
        first, last = max(base, pad[0][0]), min(end, pad[0][0] + shape[0])
        if first < last:
            selected = probability[:, first - base : last - base]
            output[first - pad[0][0] : last - pad[0][0]] = np.moveaxis(np.rint(selected * 255).clip(0, 255).astype(np.uint8), 0, -1)
        sums[:, : patch[0] - n] = sums[:, n:].clone()
        sums[:, patch[0] - n :] = 0
        counts[: patch[0] - n] = counts[n:].clone()
        counts[patch[0] - n :] = 0
        base = end

    def forward(x: torch.Tensor) -> torch.Tensor:
        logits = network(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits.float()

    done = 0
    for z in zs:
        finish(z)
        for y in ys:
            for x in xs:
                window = np.ascontiguousarray(image[z : z + patch[0], y : y + patch[1], x : x + patch[2]])
                tile = torch.from_numpy(window)[None, None].to(device)
                with _autocast(device):
                    logits = infer_with_tta(forward, tile, "mirroring") if mirror else forward(tile)
                # Channel 0 is the background.
                probability = logits[0].float().softmax(0)[1:].to(sums.device)
                region = (slice(z - base, z - base + patch[0]), slice(y, y + patch[1]), slice(x, x + patch[2]))
                sums[(slice(None),) + region] += probability * weight
                counts[region] += weight
                done += 1
                if progress is not None:
                    progress(done, total)
    finish(depth)
    return output
