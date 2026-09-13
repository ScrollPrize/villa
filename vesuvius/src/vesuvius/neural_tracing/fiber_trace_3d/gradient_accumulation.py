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
EMBEDDING_COMPONENTS = ("positive_embedding", "negative_embedding")


def validate_target_only_loss_config(raw):
    """Reject unsupported loss modes, without restricting inference models."""
    model = raw.get("model_3d", raw.get("model", {}))
    branches = int(model.get("direction_branch_count", int(model.get("output_channels", 7)) // 7))
    if not model.get("conditioned_decoder_enabled", False) and branches != 1:
        raise ValueError("Ordinary multi-branch losses are temporarily unsupported. "
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
        self.components = COMPONENTS + (EMBEDDING_COMPONENTS if getattr(model, "embedding_channels", 0) else ())
        self.counts = counts.detach().to(device=self.parameters[0].device, dtype=torch.float32).clone()
        if self.counts.shape != (len(self.components),):
            raise ValueError(f"expected {len(self.components)} target-derived normalization counts")
        if distributed:
            dist.all_reduce(self.counts, group=process_group)
        self.weights = self.counts.new_tensor((direction_weight,
            presence_weight * positive_query_weight, presence_weight * negative_query_weight)
            + ((0.5, 0.5) if len(self.components) == 5 else ()))
        self.loss_scale = float(loss_scale)
        if not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("invalid loss scale")
        self.statistics = self.counts.new_zeros(len(LossStatistics.__dataclass_fields__))
        self.finished = False

    def add(self, statistics):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        if len(self.components) == 3 and (statistics.positive_embedding_sum.requires_grad or statistics.negative_embedding_sum.requires_grad):
            raise ValueError("embedding statistics require an embedding-enabled accumulator model")
        self.statistics.add_(statistics.packed())
        numerators = torch.stack([getattr(statistics, f"{name}_sum").float() for name in self.components])
        loss = (numerators * (self.weights / self.counts.clamp_min(1))).sum()
        (loss * self.loss_scale).backward()

    def reduce_and_assign(self):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        self.finished = True
        if self.distributed:
            dist.all_reduce(self.statistics, group=self.process_group)
        statistics = LossStatistics.unpack(self.statistics)
        actual = torch.stack([getattr(statistics, f"{name}_count") for name in self.components])
        # Refuse an update if prepared geometry and actual forward targets differ.
        if not torch.equal(actual, self.counts):
            raise RuntimeError(f"target normalization counts changed: prepared={self.counts.tolist()} actual={actual.tolist()}")
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


class ComponentGradientAccumulator:
    def __init__(self, model, *, loss_scale=1.0, protect_fp16=False, gradient_observer=None):
        validate_sample_independent_model(model)
        base_model = model
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod
        while hasattr(base_model, "module"):
            base_model = base_model.module
        self.components = COMPONENTS + (("positive_embedding", "negative_embedding")
                                       if getattr(base_model, "embedding_channels", 0) else ())
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters:
            raise ValueError("no trainable parameters")
        if any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("requires FP32 master parameters; use autocast for mixed precision")
        self.loss_scale = float(loss_scale)
        if not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("invalid reversible numerator scaling")
        self.protect_fp16 = bool(protect_fp16)
        self.gradient_observer = gradient_observer
        self.local_scale_retries = 0
        sizes = [p.numel() for p in self.parameters]
        self.flat_banks = [torch.zeros(sum(sizes), device=self.parameters[0].device,
                                      dtype=torch.float32) for _ in self.components]
        self.banks = [[view.view_as(p) for view, p in zip(bank.split(sizes), self.parameters, strict=True)]
                      for bank in self.flat_banks]
        self.used = torch.zeros(len(self.parameters), device=self.parameters[0].device, dtype=torch.int32)
        self.statistics = torch.zeros(len(LossStatistics.__dataclass_fields__), device=self.parameters[0].device)
        self.finished = False

    def add(self, statistics: LossStatistics):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        if len(self.components) == 3 and (statistics.positive_embedding_sum.requires_grad or statistics.negative_embedding_sum.requires_grad):
            raise ValueError("embedding statistics require an embedding-enabled accumulator model")
        self.statistics.add_(statistics.packed())
        numerators = [getattr(statistics, f"{name}_sum") for name in self.components]
        differentiable = [i for i, n in enumerate(numerators) if n.requires_grad]
        for index, component in enumerate(differentiable):
            count = getattr(statistics, f"{self.components[component]}_count").detach().float().clamp_min(1)
            # Normalize by actual local support, not a dense-volume upper bound:
            # FP16 cannot recover values that underflowed before the FP32 bank.
            # Locally normalized gradients can overflow while a full-batch mean
            # would not. Retry that VJP with reversible power-of-two scaling,
            # without changing the global GradScaler or skipping the update.
            attenuation = 1.0
            for attempt in range(25):
                scaled = numerators[component].float() * ((self.loss_scale * attenuation) / count)
                gradients = torch.autograd.grad(scaled, self.parameters,
                                                retain_graph=self.protect_fp16 or index + 1 < len(differentiable), allow_unused=True)
                if not self.protect_fp16:
                    break
                present = [g for g in gradients if g is not None]
                finite = torch.stack([torch.isfinite(g).all() for g in present]).all() if present else True
                if bool(finite) or not bool(torch.isfinite(numerators[component])) or attempt == 24:
                    break
                attenuation *= 0.5
                self.local_scale_retries += 1
            with torch.no_grad():
                for pi, (bank, grad) in enumerate(zip(self.banks[component], gradients, strict=True)):
                    if grad is not None:
                        contribution = grad.detach().float() * (count / attenuation)
                        bank.add_(contribution)
                        if self.gradient_observer is not None:
                            self.gradient_observer(component, pi, contribution)
                        self.used[pi] = 1

    def reduce_and_assign(self, *, direction_weight, presence_weight,
                          positive_query_weight=1.0, negative_query_weight=1.0,
                          process_group=None, distributed=False):
        if self.finished:
            raise RuntimeError("accumulator already finalized")
        self.finished = True
        # All ranks participate even when they processed no samples.
        if distributed:
            dist.all_reduce(self.statistics, group=process_group)
            dist.all_reduce(self.used, op=dist.ReduceOp.MAX, group=process_group)
            for bank in self.flat_banks:
                dist.all_reduce(bank, group=process_group)
        statistics = LossStatistics.unpack(self.statistics)
        weights = (direction_weight, presence_weight * positive_query_weight,
                   presence_weight * negative_query_weight, 0.5, 0.5)
        with torch.no_grad():
            # Keep GradScaler's scale applied. The existing optimizer boundary
            # unscales/checks/clips exactly once, after globally combining banks.
            used = self.used.cpu().tolist()
            for pi, parameter in enumerate(self.parameters):
                if not used[pi]:
                    parameter.grad = None
                    continue
                grad = torch.zeros_like(parameter)
                for ci, component in enumerate(self.components):
                    count = getattr(statistics, f"{component}_count").clamp_min(1)
                    grad.add_(self.banks[ci][pi] * (weights[ci] / count))
                parameter.grad = grad
        return statistics
