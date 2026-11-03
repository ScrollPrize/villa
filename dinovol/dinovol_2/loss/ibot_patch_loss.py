# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import logging


logger = logging.getLogger("dinov2")


def _soft_target_log_likelihood(t, s, temp: float):
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


def _soft_target_log_likelihood_grad(t, s, grad_rows, temp: float):
    # d/ds sum_j t_j log_softmax(s / temp)_j = (t - softmax(s / temp) * sum(t)) / temp
    t = t.float()
    probabilities = F.softmax(s.float() / temp, dim=-1)
    grad = (t - probabilities * t.sum(dim=-1, keepdim=True)) * (grad_rows.unsqueeze(-1) / temp)
    return grad.to(s.dtype)


class _SoftTargetLogLikelihood(torch.autograd.Function):
    """Row-wise sum(t * log_softmax(s / temp)) evaluated in float32 chunks.

    Autograd would keep a float32 log-softmax of every row for backward. This
    function saves only its inputs and recomputes the softmax one chunk at a time.
    """

    @staticmethod
    def forward(ctx, t, s, temp, chunk_size, forward_fn, backward_fn):
        rows = s.new_empty(s.shape[0], dtype=torch.float32)
        for start in range(0, s.shape[0], chunk_size):
            end = min(start + chunk_size, s.shape[0])
            rows[start:end] = forward_fn(t[start:end], s[start:end], temp)
        ctx.save_for_backward(t, s)
        ctx.temp, ctx.chunk_size, ctx.backward_fn = temp, chunk_size, backward_fn
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        t, s = ctx.saved_tensors
        grad_s = torch.empty_like(s)
        for start in range(0, s.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, s.shape[0])
            grad_s[start:end] = ctx.backward_fn(t[start:end], s[start:end], grad_rows[start:end], ctx.temp)
        return None, grad_s, None, None, None, None


# Fused last layer, Sinkhorn-Knopp targets, and masked cross-entropy.
#
# Sinkhorn-Knopp on E = exp(L / T) alternates prototype (column) and sample (row)
# normalizations. After the final row normalization each target row is
#     t_i = softmax(L_i / T + log b),
# where b holds the accumulated per-prototype scales. b needs one pass over E
# per iteration, so the [n_masked, K] logits, targets, and gradients are only
# formed one chunk of rows at a time from the bottleneck features.

def _logits(features, weight):
    return torch.mm(features, weight.t()).float()


def _prototype_sums(teacher_features, teacher_weight, teacher_temp: float):
    return torch.exp(_logits(teacher_features, teacher_weight) / teacher_temp).sum(dim=0)


def _row_normalized_prototype_sums(teacher_features, teacher_weight, teacher_temp: float, prototype_scales, n_samples):
    weights = torch.exp(_logits(teacher_features, teacher_weight) / teacher_temp)
    sample_scales = 1.0 / (torch.sum(weights * prototype_scales, dim=1, keepdim=True) * n_samples)
    return torch.sum(weights * sample_scales, dim=0)


def _fused_targets(teacher_features, teacher_weight, teacher_temp: float, log_prototype_scales):
    return torch.softmax(_logits(teacher_features, teacher_weight) / teacher_temp + log_prototype_scales, dim=-1)


def _fused_log_likelihood(student_features, student_weight, teacher_features, teacher_weight,
                          student_temp: float, teacher_temp: float, log_prototype_scales):
    targets = _fused_targets(teacher_features, teacher_weight, teacher_temp, log_prototype_scales)
    log_probabilities = F.log_softmax(_logits(student_features, student_weight) / student_temp, dim=-1)
    return torch.sum(targets * log_probabilities, dim=-1)


def _fused_log_likelihood_grad(student_features, student_weight, teacher_features, teacher_weight,
                               student_temp: float, teacher_temp: float, log_prototype_scales, grad_rows):
    targets = _fused_targets(teacher_features, teacher_weight, teacher_temp, log_prototype_scales)
    probabilities = F.softmax(_logits(student_features, student_weight) / student_temp, dim=-1)
    grad_logits = (targets - probabilities * targets.sum(dim=-1, keepdim=True)) * (grad_rows.unsqueeze(-1) / student_temp)
    grad_logits = grad_logits.to(student_features.dtype)
    grad_features = torch.mm(grad_logits, student_weight)
    grad_weight = torch.mm(grad_logits.t(), student_features).float()
    return grad_features, grad_weight


_FUSED_KERNELS = {
    "prototype_sums": _prototype_sums,
    "row_normalized_prototype_sums": _row_normalized_prototype_sums,
    "log_likelihood": _fused_log_likelihood,
    "log_likelihood_grad": _fused_log_likelihood_grad,
}


class _FusedProjectionLogLikelihood(torch.autograd.Function):
    """Row-wise sum(t * log_softmax(z_s W_s^T / T_s)) with targets recomputed per chunk."""

    @staticmethod
    def forward(ctx, student_features, student_weight, teacher_features, teacher_weight,
                log_prototype_scales, student_temp, teacher_temp, chunk_size, kernels):
        rows = student_features.new_empty(student_features.shape[0], dtype=torch.float32)
        for start in range(0, student_features.shape[0], chunk_size):
            end = min(start + chunk_size, student_features.shape[0])
            rows[start:end] = kernels["log_likelihood"](
                student_features[start:end], student_weight, teacher_features[start:end], teacher_weight,
                student_temp, teacher_temp, log_prototype_scales,
            )
        ctx.save_for_backward(student_features, student_weight, teacher_features, teacher_weight, log_prototype_scales)
        ctx.student_temp, ctx.teacher_temp, ctx.chunk_size, ctx.kernels = student_temp, teacher_temp, chunk_size, kernels
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        student_features, student_weight, teacher_features, teacher_weight, log_prototype_scales = ctx.saved_tensors
        grad_features = torch.empty_like(student_features)
        grad_weight = torch.zeros(student_weight.shape, dtype=torch.float32, device=student_weight.device)
        for start in range(0, student_features.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, student_features.shape[0])
            chunk_features, chunk_weight = ctx.kernels["log_likelihood_grad"](
                student_features[start:end], student_weight, teacher_features[start:end], teacher_weight,
                ctx.student_temp, ctx.teacher_temp, log_prototype_scales, grad_rows[start:end],
            )
            grad_features[start:end] = chunk_features
            grad_weight += chunk_weight
        return grad_features, grad_weight.to(student_weight.dtype), None, None, None, None, None, None, None


class iBOTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        masked_loss_chunk_size: int | None = None,
    ):
        super().__init__()
        if masked_loss_chunk_size is not None and masked_loss_chunk_size <= 0:
            raise ValueError("masked_loss_chunk_size must be positive when set.")
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.masked_loss_chunk_size = masked_loss_chunk_size
        self._log_likelihood_fn = _soft_target_log_likelihood
        self._log_likelihood_grad_fn = _soft_target_log_likelihood_grad
        self._fused_kernels = dict(_FUSED_KERNELS)
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        #
        # WARNING:
        #   as self.center is a float32, everything gets casted to float32 afterwards
        #
        # teacher_patch_tokens = teacher_patch_tokens.float()
        # return F.softmax((teacher_patch_tokens.sub_(self.center.to(teacher_patch_tokens.dtype))).mul_(1 / teacher_temp), dim=-1)

        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

        # this is experimental, keep everything in float16 and let's see what happens:
        # return F.softmax((teacher_patch_tokens.sub_(self.center)) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        # Q is B-by-K (samples by prototypes), the transpose of the paper's notation, so
        # the returned targets are contiguous. One float32 copy is exponentiated in place.
        Q = teacher_output.to(torch.float32, copy=True)
        Q.div_(teacher_temp).exp_()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # B = Q.shape[0] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        if dist.is_initialized():
            dist.all_reduce(B)
        K = Q.shape[1]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each prototype: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=0, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each sample: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # each sample's weights must sum to 1 so that Q is an assignment
        return Q

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens.float()
        s = student_patch_tokens.float()
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        if masks_weight is None:
            tokens_per_sample = student_masks_flat.shape[1]
            masked_indices = student_masks_flat.flatten().nonzero().flatten()
            masked_sample_indices = torch.div(masked_indices, tokens_per_sample, rounding_mode="floor")
            inverse_mask_counts = 1 / student_masks_flat.sum(-1).clamp(min=1.0)
            masks_weight = inverse_mask_counts.index_select(0, masked_sample_indices)
        if n_masked_patches is not None:
            t = t[:n_masked_patches]
            s = s[:n_masked_patches]
            masks_weight = masks_weight[:n_masked_patches]
        chunk_size = self.masked_loss_chunk_size or max(1, s.shape[0])
        loss = _SoftTargetLogLikelihood.apply(
            t, s, self.student_temp, chunk_size, self._log_likelihood_fn, self._log_likelihood_grad_fn
        )
        return -(loss * masks_weight).sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def sinkhorn_knopp_prototype_scales(self, teacher_features, teacher_weight, teacher_temp,
                                        n_masked_patches_tensor, n_iterations=3):
        """Log prototype scales b such that softmax(z_t W_t^T / T + log b) equals the
        Sinkhorn-Knopp targets of sinkhorn_knopp_teacher on the teacher logits."""
        chunk_size = self.masked_loss_chunk_size or max(1, teacher_features.shape[0])
        n_samples = n_masked_patches_tensor.to(device=teacher_features.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(n_samples)
        n_prototypes = teacher_weight.shape[0]
        chunks = [
            (start, min(start + chunk_size, teacher_features.shape[0]))
            for start in range(0, teacher_features.shape[0], chunk_size)
        ]

        def all_reduced(sums):
            if dist.is_initialized():
                dist.all_reduce(sums)
            return sums

        # The first normalization by the total sum cancels in the prototype normalization.
        prototype_sums = all_reduced(sum(
            self._fused_kernels["prototype_sums"](teacher_features[start:end], teacher_weight, teacher_temp)
            for start, end in chunks
        ))
        prototype_scales = 1.0 / (prototype_sums * n_prototypes)
        for _ in range(n_iterations - 1):
            prototype_sums = all_reduced(sum(
                self._fused_kernels["row_normalized_prototype_sums"](
                    teacher_features[start:end], teacher_weight, teacher_temp, prototype_scales, n_samples,
                )
                for start, end in chunks
            ))
            prototype_scales = prototype_scales / (prototype_sums * prototype_scales * n_prototypes)
        # The last sample normalization is the softmax applied when targets are formed.
        return torch.log(prototype_scales)

    def forward_masked_from_features(
        self,
        student_features,
        student_weight,
        teacher_features,
        teacher_weight,
        log_prototype_scales,
        teacher_temp,
        student_masks_flat,
        masks_weight,
    ):
        """forward_masked for logits z W^T and Sinkhorn-Knopp targets, without materializing them.

        Features and weights should already be in the logit compute dtype (the autocast
        dtype under mixed precision), matching what the head's last linear layer would use.
        """
        chunk_size = self.masked_loss_chunk_size or max(1, student_features.shape[0])
        loss = _FusedProjectionLogLikelihood.apply(
            student_features, student_weight, teacher_features, teacher_weight, log_prototype_scales,
            self.student_temp, teacher_temp, chunk_size, self._fused_kernels,
        )
        return -(loss * masks_weight).sum() / student_masks_flat.shape[0]

    def compile_kernels(self, **options) -> None:
        """Compile the per-chunk masked-loss kernels into fused reductions."""
        options = {**options, "dynamic": True}
        self._log_likelihood_fn = torch.compile(_soft_target_log_likelihood, **options)
        self._log_likelihood_grad_fn = torch.compile(_soft_target_log_likelihood_grad, **options)
        self._fused_kernels = {name: torch.compile(kernel, **options) for name, kernel in _FUSED_KERNELS.items()}

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens, token_weights=None):
        self.reduce_center_update(teacher_patch_tokens, token_weights=token_weights)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens, token_weights=None):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        if token_weights is None:
            batch_centers = teacher_patch_tokens.mean(1)
        else:
            weights = token_weights
            if weights.ndim == teacher_patch_tokens.ndim - 1:
                weights = weights.unsqueeze(-1)
            weights = weights.to(dtype=teacher_patch_tokens.dtype, device=teacher_patch_tokens.device)
            denom = weights.sum(dim=1).clamp(min=torch.finfo(teacher_patch_tokens.dtype).eps)
            batch_centers = (teacher_patch_tokens * weights).sum(dim=1) / denom
        self.async_batch_center = torch.sum(batch_centers, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
