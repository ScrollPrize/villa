from __future__ import annotations

from dataclasses import dataclass

import torch


def compute_group_avg(values, group_idx, *, n_groups):
    n_groups = int(n_groups)
    group_idx = group_idx.long()
    group_map = (group_idx == torch.arange(n_groups, device=group_idx.device).unsqueeze(1).long()).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()
    group_avg = (group_map @ values.view(-1)) / group_denom
    return group_avg, group_count


def _require_scalar_loss(loss):
    if loss is None:
        raise ValueError("ERMBatch requires a scalar loss tensor")
    if not isinstance(loss, torch.Tensor):
        raise TypeError("ERM objectives require torch.Tensor losses")
    if loss.ndim != 0:
        raise ValueError(f"ERMBatch requires a scalar loss tensor, got shape={tuple(loss.shape)}")
    return loss


def _require_per_sample_losses(loss):
    if loss is None:
        raise ValueError("per-sample objective requires a per-sample loss tensor")
    if not isinstance(loss, torch.Tensor):
        raise TypeError("ERM objectives require torch.Tensor losses")
    if loss.ndim == 0:
        raise ValueError("per-sample objective requires a per-sample loss tensor")
    return loss.reshape(-1)


def reduce_group_topk_loss(
    per_sample_losses,
    *,
    meta=None,
    group_idx=None,
    n_groups=None,
    group_topk=1,
):
    if group_idx is None and meta is not None:
        group_idx = getattr(meta, "group_idx", None)
    if group_idx is None:
        raise ValueError("group_idx is required when group_topk > 0")
    if n_groups is None:
        flattened = group_idx.reshape(-1)
        if int(flattened.numel()) <= 0:
            raise ValueError("n_groups is required when group_idx is empty")
        n_groups = int(flattened.max().item()) + 1

    losses = _require_per_sample_losses(per_sample_losses)
    group_loss, group_count = compute_group_avg(losses, group_idx, n_groups=int(n_groups))
    present = group_count > 0
    if present.any():
        present_losses = group_loss[present]
        topk = min(int(group_topk), int(present_losses.numel()))
        topk_losses, _ = torch.topk(present_losses, topk, largest=True)
        return topk_losses.mean()
    return losses.mean()


@dataclass(frozen=True)
class ERMBatch:
    requires_group_idx = False

    def build(self, _bundle=None):
        return self

    def __call__(self, loss, *, meta=None, group_idx=None, n_groups=None):
        del meta, group_idx, n_groups
        return _require_scalar_loss(loss)

    def reduce(self, loss, *, meta=None, group_idx=None, n_groups=None):
        return self(loss, meta=meta, group_idx=group_idx, n_groups=n_groups)


@dataclass(frozen=True)
class ERMPerSample:
    requires_group_idx = False

    def build(self, _bundle=None):
        return self

    def __call__(self, loss, *, meta=None, group_idx=None, n_groups=None):
        del meta, group_idx, n_groups
        return _require_per_sample_losses(loss).mean()

    def reduce(self, loss, *, meta=None, group_idx=None, n_groups=None):
        return self(loss, meta=meta, group_idx=group_idx, n_groups=n_groups)


@dataclass(frozen=True)
class ERMGroupTopK:
    requires_group_idx = True

    group_topk: int = 1

    def __post_init__(self) -> None:
        if int(self.group_topk) <= 0:
            raise ValueError("ERMGroupTopK requires group_topk > 0")

    def build(self, _bundle=None):
        return self

    def __call__(self, loss, *, meta=None, group_idx=None, n_groups=None):
        return reduce_group_topk_loss(
            loss,
            meta=meta,
            group_idx=group_idx,
            n_groups=n_groups,
            group_topk=self.group_topk,
        )

    def reduce(self, loss, *, meta=None, group_idx=None, n_groups=None):
        return self(loss, meta=meta, group_idx=group_idx, n_groups=n_groups)


ERMObjective = ERMBatch
