"""Memory-bounded numerator gradients for a fixed effective optimizer batch.

No communication occurs in add(). reduce_and_assign() is an update boundary.
The caller retains ownership of autocast, GradScaler, clipping and optimizer step.
This utility does not enable cooperative training by itself.
"""

import math

import torch
import torch.distributed as dist

from .loss_statistics import LossStatistics


COMPONENTS = ("direction", "positive_presence", "negative_presence")


def validate_target_only_loss_config(raw):
    """Reject unsupported loss modes, without restricting inference models."""
    model = raw.get("model_3d", raw.get("model", {}))
    branches = int(model.get("direction_branch_count", int(model.get("output_channels", 7)) // 7))
    if raw.get("embedding", {}).get("enabled", False) or (
        not model.get("conditioned_decoder_enabled", False) and branches != 1
    ):
        raise ValueError("Embedding and ordinary multi-branch losses are temporarily unsupported. "
                         "TODO: fix prediction-dependent global normalization before re-enabling these modes.")


class NormalizedGradientAccumulator:
    """One backward per microbatch, using counts known before the first forward.

    Counts are SUM-reduced at construction. Numerators are already divided by
    global counts, so parameter gradients are SUM-reduced, never averaged.
    AMP's scale stays applied until the caller's usual unscale/step boundary.
    """

    def __init__(self, model, counts, *, direction_weight, presence_weight,
                 positive_query_weight=1.0, negative_query_weight=1.0,
                 distributed=False, process_group=None, loss_scale=1.0):
        validate_sample_independent_model(model)
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters or any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("requires FP32 master parameters; use autocast for mixed precision")
        self.distributed, self.process_group = distributed, process_group
        self.counts = counts.detach().to(device=self.parameters[0].device, dtype=torch.float32).clone()
        if self.counts.shape != (3,):
            raise ValueError("expected three target-derived normalization counts")
        self.local_counts = self.counts.clone()
        if distributed:
            dist.all_reduce(self.counts, group=process_group)
        self.weights = self.counts.new_tensor((direction_weight,
            presence_weight * positive_query_weight, presence_weight * negative_query_weight))
        self.loss_scale = float(loss_scale)
        if not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("invalid loss scale")
        self.statistics = self.counts.new_zeros(len(LossStatistics.__dataclass_fields__))
        self.finished = False

    def add(self, statistics):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        if statistics.positive_embedding_sum.requires_grad or statistics.negative_embedding_sum.requires_grad:
            raise ValueError("target-only accumulator does not support prediction-dependent embedding counts")
        self.statistics.add_(statistics.packed())
        numerators = torch.stack([getattr(statistics, f"{name}_sum").float() for name in COMPONENTS])
        loss = (numerators * (self.weights / self.counts.clamp_min(1))).sum()
        (loss * self.loss_scale).backward()

    def reduce_and_assign(self):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        self.finished = True
        statistics = LossStatistics.unpack(self.statistics)
        actual = torch.stack([getattr(statistics, f"{name}_count") for name in COMPONENTS])
        # Compare locally in identical microbatch order, before collectives.
        # Fractional Gaussian masses can round differently when NCCL reduces
        # the short count vector versus the longer reporting-stat vector.
        mismatch = (actual != self.local_counts).any().float().reshape(1)
        if self.distributed:
            # Piggyback failure consensus on the existing reporting collective:
            # every rank fails together, without weakening the exact check.
            packed = torch.cat((self.statistics, mismatch))
            dist.all_reduce(packed, group=self.process_group)
            self.statistics = packed[:-1]
            mismatch = packed[-1:]
            statistics = LossStatistics.unpack(self.statistics)
        if bool(mismatch.item()):
            raise RuntimeError(f"target normalization counts changed on one or more ranks: local prepared={self.local_counts.tolist()} actual={actual.tolist()}")
        if self.distributed:
            used = torch.tensor([p.grad is not None for p in self.parameters],
                                dtype=torch.int32, device=self.counts.device)
            dist.all_reduce(used, op=dist.ReduceOp.MAX, group=self.process_group)
            sizes = [p.numel() for p in self.parameters]
            flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                              for p in self.parameters])
            dist.all_reduce(flat, group=self.process_group)
            for p, value, present in zip(self.parameters, flat.split(sizes), used.cpu().tolist(), strict=True):
                p.grad = value.view_as(p) if present else None
        return statistics


def validate_sample_independent_model(model):
    """Reject known sample-/rank-dependent state and random model operations."""
    rejected = (torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.modules.dropout._DropoutNd,
                torch.nn.RNNBase)
    for name, module in model.named_modules():
        if isinstance(module, rejected):
            raise ValueError(f"bounded dynamic training rejects {type(module).__name__} at {name}")


def effective_batch_partition(step, effective_batch_size, rank, world_size, microbatch_size):
    """Contiguous canonical IDs; rank changes only physical placement, never membership."""
    values = (step, effective_batch_size, rank, world_size, microbatch_size)
    if any(type(v) is not int for v in values) or step < 0 or effective_batch_size < 1 or world_size < 1 or microbatch_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid effective batch partition")
    start = step * effective_batch_size + effective_batch_size * rank // world_size
    stop = step * effective_batch_size + effective_batch_size * (rank + 1) // world_size
    return [tuple(range(offset, min(offset + microbatch_size, stop)))
            for offset in range(start, stop, microbatch_size)]
