from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_ds_weights(n: int):
    if n <= 0:
        return None
    weights = np.array([1 / (2 ** i) for i in range(n)], dtype=np.float32)
    weights[-1] = 0.0
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights.tolist()


def _resize_for_ds(tensor, size, *, mode, align_corners=None):
    if tensor.shape[2:] == size:
        return tensor
    if align_corners is None:
        return F.interpolate(tensor.float(), size=size, mode=mode).to(tensor.dtype)
    return F.interpolate(tensor.float(), size=size, mode=mode, align_corners=align_corners).to(tensor.dtype)


def build_deep_supervision_targets(
    tensor: torch.Tensor,
    reference_outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    *,
    mode: str = "nearest",
    align_corners=None,
):
    if not isinstance(reference_outputs, (list, tuple)):
        return tensor
    return type(reference_outputs)(
        _resize_for_ds(
            tensor,
            tuple(int(v) for v in output.shape[2:]),
            mode=mode,
            align_corners=align_corners,
        )
        for output in reference_outputs
    )


class DeepSupervisionWrapper(nn.Module):
    """Apply a base loss across a pyramid of outputs with nnUNet-style weights."""

    def __init__(self, loss, weights):
        super().__init__()
        self.loss = loss
        self.weights = weights
        self.latest_metrics = {}

    def _capture_metrics(self, metrics, scale_index: int):
        if not isinstance(metrics, dict):
            return
        for key, value in metrics.items():
            metrics_key = str(key)
            if scale_index == 0:
                self.latest_metrics[metrics_key] = float(value)
            self.latest_metrics[f"{metrics_key}/ds{scale_index}"] = float(value)

    def forward(self, net_output: Union[torch.Tensor, List[torch.Tensor]], target: Union[torch.Tensor, List[torch.Tensor]]):
        self.latest_metrics = {}
        if isinstance(net_output, (list, tuple)):
            assert isinstance(target, (list, tuple))
            assert len(net_output) == len(target) == len(self.weights)

            loss = None
            for i, weight in enumerate(self.weights):
                if weight == 0:
                    continue
                res = self.loss(net_output[i], target[i])
                metrics = getattr(self.loss, "latest_metrics", None)
                self._capture_metrics(metrics, i)
                res = res.reshape(())
                weighted = float(weight) * res
                loss = weighted if loss is None else loss + weighted

            if loss is None:
                ref = net_output[0]
                return ref.new_zeros(())
            return loss

        res = self.loss(net_output, target)
        metrics = getattr(self.loss, "latest_metrics", None)
        self._capture_metrics(metrics, 0)
        return res.reshape(())
