"""nnU-Net-style target pyramids and weighted deep-supervision loss."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def deep_supervision_weights(output_count: int) -> list[float] | None:
    """Return decoder weights, including the one-stage zero edge case."""

    if output_count <= 0:
        return None
    weights = np.asarray(
        [1 / (2**index) for index in range(output_count)], dtype=np.float32
    )
    weights[-1] = 0.0
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.tolist()


def _resize_for_supervision(
    tensor: torch.Tensor,
    size: tuple[int, ...],
    *,
    mode: str,
    align_corners: bool | None,
) -> torch.Tensor:
    if tensor.shape[2:] == size:
        return tensor
    if align_corners is None:
        return F.interpolate(tensor.float(), size=size, mode=mode).to(
            tensor.dtype
        )
    return F.interpolate(
        tensor.float(),
        size=size,
        mode=mode,
        align_corners=align_corners,
    ).to(tensor.dtype)


def build_deep_supervision_targets(
    tensor: torch.Tensor,
    reference_outputs: torch.Tensor | Sequence[torch.Tensor],
    *,
    mode: str = "nearest",
    align_corners: bool | None = None,
):
    """Resize one BCHW/BCDHW tensor independently to every output scale."""

    if not isinstance(reference_outputs, (list, tuple)):
        return tensor
    return type(reference_outputs)(
        _resize_for_supervision(
            tensor,
            tuple(int(value) for value in output.shape[2:]),
            mode=mode,
            align_corners=align_corners,
        )
        for output in reference_outputs
    )


def concatenate_deep_supervision_ignore(
    targets: torch.Tensor,
    ignore_mask: torch.Tensor,
    reference_outputs: torch.Tensor | Sequence[torch.Tensor],
):
    """Build matching target/ignore pyramids and concatenate their channels."""

    target_pyramid = build_deep_supervision_targets(
        targets, reference_outputs, mode="nearest"
    )
    ignore_pyramid = build_deep_supervision_targets(
        ignore_mask, reference_outputs, mode="nearest"
    )
    if isinstance(target_pyramid, (list, tuple)):
        return type(target_pyramid)(
            torch.cat((target_level, ignore_level), dim=1)
            for target_level, ignore_level in zip(
                target_pyramid, ignore_pyramid, strict=True
            )
        )
    return torch.cat((target_pyramid, ignore_pyramid), dim=1)


class DeepSupervisionWrapper(nn.Module):
    """Apply one loss across a prediction pyramid with fixed scalar weights."""

    def __init__(self, loss: nn.Module, weights: Sequence[float]) -> None:
        super().__init__()
        self.loss = loss
        self.weights = tuple(float(weight) for weight in weights)
        self.latest_metrics: dict[str, float] = {}

    def _capture_metrics(self, scale_index: int) -> None:
        metrics = getattr(self.loss, "latest_metrics", None)
        if not isinstance(metrics, dict):
            return
        for key, value in metrics.items():
            metric_name = str(key)
            if scale_index == 0:
                self.latest_metrics[metric_name] = float(value)
            self.latest_metrics[f"{metric_name}/ds{scale_index}"] = float(value)

    def forward(self, net_output, target) -> torch.Tensor:
        self.latest_metrics = {}
        if isinstance(net_output, (list, tuple)):
            if not isinstance(target, (list, tuple)):
                raise TypeError("deep-supervision targets must match output type")
            if len(net_output) != len(target) or len(net_output) != len(
                self.weights
            ):
                raise ValueError(
                    "deep-supervision outputs, targets, and weights must match"
                )
            result = None
            for index, weight in enumerate(self.weights):
                if weight == 0:
                    continue
                scale_loss = self.loss(net_output[index], target[index])
                self._capture_metrics(index)
                weighted = weight * scale_loss.reshape(())
                result = weighted if result is None else result + weighted
            if result is None:
                return net_output[0].new_zeros(())
            return result

        result = self.loss(net_output, target)
        self._capture_metrics(0)
        return result.reshape(())
