"""Label dilation and full-3D batch morphology."""

from __future__ import annotations

from collections.abc import Mapping

import torch

import vc_reader.transforms as A

from vesuvius.ink_detection.config import TrainingConfig
from vesuvius.ink_detection.data.geometry import native_volume_downsample_factor

_OUTSIDE_DISTANCE = A.DistanceTransform(mode="outside", implicit_background=True)


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


def dilate_label_batch(
    labels_BCZYX: torch.Tensor,
    valid_B1ZYX: torch.Tensor,
    distance: float | None,
) -> torch.Tensor:
    """Set background voxels within ``distance`` of label 1 to 1, inside a validity mask.

    Distances are exact Euclidean (vc_reader DistanceTransform) on the tensor's
    device. A channel with no valid label 1 voxels is returned unchanged.
    """

    if distance in (None, 0):
        return labels_BCZYX
    if valid_B1ZYX.ndim == labels_BCZYX.ndim - 1:
        valid_B1ZYX = valid_B1ZYX.unsqueeze(1)
    batch, channels, *spatial = labels_BCZYX.shape
    valid_B1ZYX = valid_B1ZYX > 0
    source_BCZYX = (labels_BCZYX == 1) & valid_B1ZYX
    distances_BCZYX = _OUTSIDE_DISTANCE.batch(
        source_BCZYX.reshape(batch * channels, 1, *spatial).to(torch.uint8)
    ).reshape(labels_BCZYX.shape)
    has_source_BC111 = source_BCZYX.flatten(2).any(-1).reshape(batch, channels, 1, 1, 1)
    fill_BCZYX = (
        (labels_BCZYX == 0)
        & valid_B1ZYX
        & (distances_BCZYX <= float(distance))
        & has_source_BC111
    )
    return labels_BCZYX.masked_fill(fill_BCZYX, 1)


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
