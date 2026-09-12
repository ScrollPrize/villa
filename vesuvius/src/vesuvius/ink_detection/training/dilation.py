"""Lazy CUDA label dilation, a CPU fallback, and full-3D batch morphology."""

from __future__ import annotations

from collections.abc import Mapping
import logging

import edt
import numpy as np
import torch

from vesuvius.ink_detection.config import TrainingConfig
from vesuvius.ink_detection.data.geometry import native_volume_downsample_factor


logger = logging.getLogger(__name__)

_CUCIM_UNAVAILABLE_MESSAGE = (
    "positive full_3d dilation requires CUDA with CuPy and cuCIM"
)
# Process-wide memo: once a CUDA tensor could not use cuCIM, stop retrying the
# import on every batch. The warning is logged once per process.
_cucim_unavailable = False
_fallback_announced = False


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


class CucimUnavailableError(RuntimeError):
    """Raised when the cuCIM dilation path cannot run on this tensor or host."""


def dilate_label_batch_with_cucim(
    labels_BCZYX: torch.Tensor,
    valid_B1ZYX: torch.Tensor,
    distance: float | None,
) -> torch.Tensor:
    """Dilate CUDA binary labels within a validity mask via cuCIM EDT."""

    if distance in (None, 0):
        return labels_BCZYX
    if labels_BCZYX.device.type != "cuda":
        raise CucimUnavailableError(_CUCIM_UNAVAILABLE_MESSAGE)
    try:
        import cupy as cp
        from cucim.core.operations.morphology import distance_transform_edt
    except ImportError as exc:
        raise CucimUnavailableError(_CUCIM_UNAVAILABLE_MESSAGE) from exc

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


def dilate_label_batch_with_edt(
    labels_BCZYX: torch.Tensor,
    valid_B1ZYX: torch.Tensor,
    distance: float | None,
) -> torch.Tensor:
    """Dilate binary labels within a validity mask via the CPU ``edt`` package.

    Same semantics as :func:`dilate_label_batch_with_cucim`: a voxel with
    label 0 inside the validity mask becomes 1 when its Euclidean distance to
    the nearest label-1 voxel that is also inside the mask is at most
    ``distance``. Tensors on any device are accepted; the work happens in host
    memory and the result is returned on the input device with the input dtype.
    """

    if distance in (None, 0):
        return labels_BCZYX

    if valid_B1ZYX.ndim == labels_BCZYX.ndim - 1:
        valid_B1ZYX = valid_B1ZYX.unsqueeze(1)
    threshold = float(distance)
    source_dtype = labels_BCZYX.dtype
    host_dtype = (
        torch.float32
        if source_dtype in (torch.float16, torch.bfloat16)
        else source_dtype
    )
    labels_np = (
        labels_BCZYX.detach().to(device="cpu", dtype=host_dtype).numpy().copy()
    )
    valid_np = valid_B1ZYX.detach().to(device="cpu").numpy()
    for batch_index in range(labels_np.shape[0]):
        valid_ZYX = valid_np[batch_index, 0] > 0
        for channel_index in range(labels_np.shape[1]):
            label_ZYX = labels_np[batch_index, channel_index]
            source_ZYX = (label_ZYX == 1) & valid_ZYX
            if not source_ZYX.any():
                continue
            fill_ZYX = (label_ZYX == 0) & valid_ZYX
            if not fill_ZYX.any():
                continue
            distances_ZYX = edt.edt(
                np.ascontiguousarray(~source_ZYX), black_border=False
            )
            fill_ZYX &= distances_ZYX <= threshold
            label_ZYX[fill_ZYX] = 1
    return torch.from_numpy(labels_np).to(
        device=labels_BCZYX.device, dtype=source_dtype
    )


def dilate_label_batch(
    labels_BCZYX: torch.Tensor,
    valid_B1ZYX: torch.Tensor,
    distance: float | None,
) -> torch.Tensor:
    """Dilate with cuCIM when the tensor is on CUDA and cuCIM imports, else with edt."""

    global _cucim_unavailable, _fallback_announced
    if distance in (None, 0):
        return labels_BCZYX
    if labels_BCZYX.device.type == "cuda" and not _cucim_unavailable:
        try:
            return dilate_label_batch_with_cucim(
                labels_BCZYX, valid_B1ZYX, distance
            )
        except CucimUnavailableError:
            _cucim_unavailable = True
    if not _fallback_announced:
        _fallback_announced = True
        logger.warning(
            "cuCIM is unavailable for full_3d label dilation on %s; using the "
            "CPU edt fallback, which produces the same labels but is slower",
            labels_BCZYX.device,
        )
    return dilate_label_batch_with_edt(labels_BCZYX, valid_B1ZYX, distance)


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
        inklabels_BCZYX = dilate_label_batch(
            inklabels_BCZYX, valid_B1ZYX, label_distance
        )
    if supervision_distance > 0.0:
        background_BCZYX = (
            (supervision_BCZYX > 0) & (inklabels_BCZYX <= 0)
        ).to(dtype=inklabels_BCZYX.dtype)
        background_BCZYX = dilate_label_batch(
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
