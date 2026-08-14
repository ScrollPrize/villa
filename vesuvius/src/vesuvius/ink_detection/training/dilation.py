"""Lazy CUDA label dilation and full-3D batch morphology."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from vesuvius.ink_detection.config import TrainingConfig
from vesuvius.ink_detection.data.geometry import native_volume_downsample_factor


def resolve_dilation_distances(
    config: TrainingConfig,
) -> tuple[float, float]:
    """Convert level-0 dilation distances to the one native dataset level."""

    if config.ink.data.mode not in {"full_3d", "full_3d_single_wrap"}:
        return 0.0, 0.0
    full_3d = config.ink.data.full_3d
    label_distance = float(full_3d.label_dilation_distance)
    supervision_distance = float(full_3d.supervision_dilation_distance)
    if label_distance <= 0.0 and supervision_distance <= 0.0:
        return label_distance, supervision_distance
    factors = {
        native_volume_downsample_factor(source.volume_scale)
        for source in config.ink.data.datasets
    }
    if len(factors) != 1:
        raise ValueError(
            "full_3d dilation distances require a single volume_scale across "
            f"datasets, got downsample factors {sorted(factors)!r}"
        )
    factor = float(factors.pop())
    return label_distance / factor, supervision_distance / factor


def dilate_label_batch_with_cucim(
    labels_BCZYX: torch.Tensor,
    valid_B1ZYX: torch.Tensor,
    distance: float | None,
) -> torch.Tensor:
    """Dilate CUDA binary labels within a validity mask via cuCIM EDT."""

    if distance in (None, 0):
        return labels_BCZYX
    if labels_BCZYX.device.type != "cuda":
        raise RuntimeError(
            "positive full_3d dilation requires CUDA with CuPy and cuCIM"
        )
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError(
            "positive full_3d dilation requires CUDA with CuPy and cuCIM"
        ) from exc

    cp.cuda.Device(labels_BCZYX.device.index).use()
    output_BCZYX = labels_BCZYX.clone()
    if valid_B1ZYX.ndim == labels_BCZYX.ndim - 1:
        valid_B1ZYX = valid_B1ZYX.unsqueeze(1)
    streams = [
        cp.cuda.Stream(non_blocking=True)
        for _ in range(output_BCZYX.shape[0])
    ]
    for batch_index in range(output_BCZYX.shape[0]):
        with streams[batch_index]:
            for channel_index in range(output_BCZYX.shape[1]):
                label_ZYX = cp.from_dlpack(
                    output_BCZYX[batch_index, channel_index].contiguous()
                )
                valid_ZYX = cp.from_dlpack(
                    valid_B1ZYX[batch_index, 0].contiguous()
                )
                source_ZYX = (label_ZYX == 1) & (valid_ZYX > 0)
                distances_ZYX = distance_transform_edt(
                    ~source_ZYX,
                    return_indices=False,
                    float64_distances=False,
                )
                fill_ZYX = (
                    (label_ZYX == 0)
                    & (valid_ZYX > 0)
                    & (distances_ZYX <= float(distance))
                )
                label_ZYX[fill_ZYX] = label_ZYX.dtype.type(1)
    for stream in streams:
        stream.synchronize()
    return output_BCZYX


def apply_label_dilation(
    batch: Mapping[str, torch.Tensor],
    label_distance: float,
    supervision_distance: float,
) -> dict[str, torch.Tensor]:
    """Return a batch with ink/background dilation-union semantics."""

    if label_distance <= 0.0 and supervision_distance <= 0.0:
        return batch if isinstance(batch, dict) else dict(batch)
    output = dict(batch)
    inklabels_BCZYX = output["inklabels"]
    supervision_BCZYX = output["supervision_mask"]
    valid_B1ZYX = torch.ones(
        inklabels_BCZYX.shape[0],
        1,
        *inklabels_BCZYX.shape[2:],
        device=inklabels_BCZYX.device,
        dtype=inklabels_BCZYX.dtype,
    )
    if label_distance > 0.0:
        inklabels_BCZYX = dilate_label_batch_with_cucim(
            inklabels_BCZYX, valid_B1ZYX, label_distance
        )
    if supervision_distance > 0.0:
        background_BCZYX = (
            (supervision_BCZYX > 0) & (inklabels_BCZYX <= 0)
        ).to(dtype=inklabels_BCZYX.dtype)
        background_BCZYX = dilate_label_batch_with_cucim(
            background_BCZYX, valid_B1ZYX, supervision_distance
        )
        background_BCZYX = background_BCZYX * (
            inklabels_BCZYX <= 0
        ).to(dtype=background_BCZYX.dtype)
        supervision_BCZYX = (
            (inklabels_BCZYX > 0) | (background_BCZYX > 0)
        ).to(dtype=supervision_BCZYX.dtype)
    elif label_distance > 0.0:
        supervision_BCZYX = (
            (inklabels_BCZYX > 0) | (supervision_BCZYX > 0)
        ).to(dtype=supervision_BCZYX.dtype)
    output["inklabels"] = inklabels_BCZYX
    output["supervision_mask"] = supervision_BCZYX
    return output
