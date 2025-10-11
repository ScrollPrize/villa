import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import sys
import os
from vesuvius.image_proc.geometry.structure_tensor import (
    StructureTensorComputer,
    components_to_matrix,
)

# Import nnUNet losses
from .nnunet_losses import (
    DC_and_CE_loss, DC_and_BCE_loss, MemoryEfficientSoftDiceLoss,
    DeepSupervisionWrapper
)




class MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)




class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


class MaskedMSELoss(nn.Module):
    """
    MSE Loss that computes loss only on labeled/masked regions.
    Can accept masks in multiple formats:
    - As a separate mask tensor (same shape as input/target)
    - Through ignore_index (computes where target != ignore_index)
    - As an extra channel in the target tensor (last channel is the mask)
    """
    def __init__(self, ignore_index=None, mask_channel=False):
        """
        Args:
            ignore_index: Value to ignore in target (creates mask where target != ignore_index)
            mask_channel: If True, expects last channel of target to be the mask
        """
        super(MaskedMSELoss, self).__init__()
        self.ignore_index = ignore_index
        self.mask_channel = mask_channel
        
    def forward(self, input, target, mask=None):
        # Handle different mask formats
        if mask is None:
            if self.mask_channel and target.size(1) > 1:
                # Last channel of target is the mask
                mask = target[:, -1:, ...]
                target = target[:, :-1, ...]
                # Ensure mask is binary (0 or 1)
                mask = (mask > 0).float()
            elif self.ignore_index is not None:
                # Create mask from ignore_index
                mask = (target != self.ignore_index).float()
            else:
                # No mask provided, compute regular MSE
                return F.mse_loss(input, target)
        
        # Ensure input and target have same shape
        if input.size() != target.size():
            if target.size(1) == 1 and input.size(1) > 1:
                # Expand target to match input channels if needed
                target = target.expand_as(input)
        
        # Ensure mask has same spatial dimensions
        if mask.dim() == input.dim() - 1:
            mask = mask.unsqueeze(1)
        
        # Expand mask to match input channels if needed
        if mask.size(1) == 1 and input.size(1) > 1:
            mask = mask.expand_as(input)
            
        # Compute masked MSE
        diff_squared = (input - target) ** 2
        
        # Apply mask
        masked_diff = diff_squared * mask
        
        # Compute mean only over masked elements
        num_masked = mask.sum()
        if num_masked > 0:
            return masked_diff.sum() / num_masked
        else:
            # If no valid pixels, return 0 to avoid NaN
            return torch.tensor(0.0, device=input.device, requires_grad=True)

import torch
from torch import nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss that computes loss only on labeled/masked regions.
    
    This loss computes 1 - cosine_similarity for each spatial location,
    then averages only over the labeled (non-masked) regions.
    """
    def __init__(self, dim=1, eps=1e-8, ignore_index=None):
        """
        Args:
            dim: Dimension along which to compute cosine similarity (default: 1 for channel dim)
            eps: Small value to avoid division by zero
            ignore_index: Value to ignore in target (creates mask where target != ignore_index)
        """
        super(CosineSimilarityLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.ignore_index = ignore_index
        
    def forward(self, input, target, mask=None, **kwargs):
        # Ensure input and target have same shape
        assert input.size() == target.size(), f"Input and target must have same shape, got {input.size()} vs {target.size()}"
        
        # Handle mask creation from ignore_index if no explicit mask provided
        if mask is None and self.ignore_index is not None:
            # Create mask from ignore_index
            mask = (target != self.ignore_index).float()
            # If target has multiple channels, take max across channels
            if mask.dim() > input.dim() - 1:
                mask = mask.max(dim=1, keepdim=True)[0]
        
        # Compute cosine similarity
        # Normalize along channel dimension
        input_norm = F.normalize(input, p=2, dim=self.dim, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=self.dim, eps=self.eps)
        
        # Compute dot product (cosine similarity)
        cosine_sim = (input_norm * target_norm).sum(dim=self.dim, keepdim=True)
        
        # Loss is 1 - cosine_similarity (so perfect match = 0 loss)
        loss = 1 - cosine_sim
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has same spatial dimensions
            if mask.dim() == input.dim() - 1:
                mask = mask.unsqueeze(1)
            
            # Expand mask to match loss dimensions if needed
            if mask.size(1) == 1 and loss.size(1) > 1:
                mask = mask.expand_as(loss)
            
            # Apply mask
            masked_loss = loss * mask
            
            # Compute mean only over masked elements
            num_masked = mask.sum()
            if num_masked > 0:
                return masked_loss.sum() / num_masked
            else:
                # If no valid pixels, return 0 to avoid NaN
                return torch.tensor(0.0, device=input.device, requires_grad=True)
        else:
            # No mask, compute regular mean
            return loss.mean()


class SignInvariantCosineLoss(nn.Module):
    """
    Sign-invariant cosine loss for vector fields with optional masking.

    Computes 1 - |cos(theta)| per voxel along the channel dimension, then
    averages only over masked (non-ignored) elements when a mask or ignore_index
    is provided.
    """
    def __init__(self, dim=1, eps=1e-8, ignore_index=None):
        super(SignInvariantCosineLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target, mask=None, **kwargs):
        # Ensure input and target have same shape
        assert input.size() == target.size(), f"Input and target must have same shape, got {input.size()} vs {target.size()}"

        # Derive mask from ignore_index if not explicitly given
        if mask is None and self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            if mask.dim() > input.dim() - 1:
                mask = mask.max(dim=1, keepdim=True)[0]

        # Normalize both vectors along channel dim
        input_norm = F.normalize(input, p=2, dim=self.dim, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=self.dim, eps=self.eps)

        # Cosine similarity and sign invariance via absolute value
        cosine_sim = (input_norm * target_norm).sum(dim=self.dim, keepdim=True)
        loss = 1 - cosine_sim.abs()

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == input.dim() - 1:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1 and loss.size(1) > 1:
                mask = mask.expand_as(loss)
            masked_loss = loss * mask
            num_masked = mask.sum()
            if num_masked > 0:
                return masked_loss.sum() / num_masked
            else:
                return torch.tensor(0.0, device=input.device, requires_grad=True)
        else:
            return loss.mean()


class ChannelSelectLoss(nn.Module):
    """
    Wrapper to apply a base loss to a channel subset along dim=1.

    channels can be:
    - 'last'            -> select last channel only
    - 'all_but_last'    -> select all except last
    - list/tuple of int -> explicit indices
    """
    def __init__(self, base_loss: nn.Module, channels):
        super().__init__()
        self.loss = base_loss
        self.channels = channels

    def _select(self, x):
        if self.channels == 'last':
            return x[:, -1:, ...]
        elif self.channels == 'all_but_last':
            return x[:, :-1, ...]
        elif isinstance(self.channels, (list, tuple)):
            return x[:, self.channels, ...]
        else:
            raise ValueError(f"Unsupported channels spec: {self.channels}")

    def forward(self, input, target, mask=None, **kwargs):
        x = self._select(input)
        y = self._select(target)
        # Some wrapped losses don't accept a mask kwarg. Try with mask, fallback without.
        try:
            return self.loss(x, y, mask=mask, **kwargs)
        except TypeError:
            return self.loss(x, y, **kwargs)
class EigenvalueLoss(nn.Module):
    """
    Loss for regressing a *set* of eigen-values that

      • treats the eigen-values as an unordered set
      • can use absolute or relative squared error
      • accepts per-eigen-value weights
      • honors an `ignore_index` 
    """

    def __init__(
        self,
        reduction: str = "mean",
        relative: bool = False,
        weight: torch.Tensor | None = None,
        ignore_index: float | int | None = None,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        reduction     {"mean","sum","none"}
        relative      If True ⇒ use squared relative error
        weight        1-D tensor of length k with per-eigen-value weights
        ignore_index  Scalar sentinel value in the target to mask out
        eps           Small value to stabilise relative error
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.relative = relative
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError(
                f"input and target must have the same shape, got {input.shape} vs {target.shape}"
            )

        # ── Sort eigen-values so their order is irrelevant ──────────────────────
        input_sorted, _ = torch.sort(input, dim=1)
        target_sorted, _ = torch.sort(target, dim=1)

        # ── Create mask for ignore_index (all ones if not used) ────────────────
        if self.ignore_index is None:
            mask = torch.ones_like(target_sorted, dtype=torch.bool)
            target_masked = target_sorted
        else:
            mask = target_sorted.ne(self.ignore_index)
            # Replace ignored entries by *something* that keeps the
            # arithmetic valid but will be masked out later.
            target_masked = torch.where(mask, target_sorted, torch.zeros_like(target_sorted))

        # ── Compute (relative) squared error ───────────────────────────────────
        if self.relative:
            diff = (input_sorted - target_masked) / (target_masked.abs() + self.eps)
        else:
            diff = input_sorted - target_masked

        sq_err = diff.pow(2)

        # ── Apply per-eigen-value weights ──────────────────────────────────────
        if self.weight is not None:
            w = self.weight.to(sq_err.device).view(1, -1)
            sq_err = sq_err * w

        # ── Zero-out ignored positions, then reduce ───────────────────────────
        sq_err = sq_err * mask

        if self.reduction == "none":
            return sq_err

        valid_elems = mask.sum()  # scalar
        if valid_elems == 0:
            # Nothing to optimise – return 0 so .backward() is safe
            return torch.zeros(
                (), dtype=sq_err.dtype, device=sq_err.device, requires_grad=input.requires_grad
            )

        if self.reduction == "sum":
            return sq_err.sum()

        # "mean" – average only over *valid* entries
        return sq_err.sum() / valid_elems


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)



import torch
import torch.nn.functional as F
from torch import nn


class SignedDistanceLoss(nn.Module):
    """
    Band-limited Smooth-L1 loss for signed-distance regression, with optional
    Eikonal term enforcing ‖∇d_pred‖ ≈ 1 near the surface.

    Parameters
    ----------
    rho            Width of the surface band in *voxels* (|d_gt| < rho)          (default: 4)
    beta           Huber transition point (see torch.nn.SmoothL1Loss)            (default: 1)
    eikonal        If True, add λ * (‖∇d_pred‖ − 1)^2 in the same band           (default: False)
    eikonal_weight λ – weight of the Eikonal term relative to data term          (default: 0.01)
    reduction      "mean" (default) | "sum" | "none"
    ignore_index   Sentinel value in target to be ignored                        (default: None)
    """
    def __init__(
        self,
        rho: float = 4.0,
        beta: float = 1.0,
        eikonal: bool = False,
        eikonal_weight: float = 0.01,
        reduction: str = "mean",
        ignore_index: float | int | None = None,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.rho = float(rho)
        self.beta = float(beta)
        self.eikonal = bool(eikonal)
        self.eik_w = float(eikonal_weight)
        self.reduction = reduction
        self.ignore_index = ignore_index

    @staticmethod
    def _gradient_3d(t: torch.Tensor) -> torch.Tensor:
        """Finite-difference ∇t (same shape as `t`, zero-padded borders)."""
        dz = F.pad(t[:, :, 2:] - t[:, :, :-2],  (0, 0, 0, 0, 1, 1))
        dy = F.pad(t[:, :, :, 2:] - t[:, :, :, :-2], (0, 0, 1, 1, 0, 0))
        dx = F.pad(t[:, :, :, :, 2:] - t[:, :, :, :, :-2], (1, 1, 0, 0, 0, 0))
        return torch.stack((dz, dy, dx), dim=1) * 0.5   # shape (B,3,D,H,W)

    def forward(self, d_pred: torch.Tensor, d_gt: torch.Tensor) -> torch.Tensor:
        if d_pred.shape != d_gt.shape:
            raise ValueError(f"Shape mismatch {d_pred.shape} vs {d_gt.shape}")

        # ── build validity mask ───────────────────────────────────────────────
        band_mask = (d_gt.abs() < self.rho)
        if self.ignore_index is not None:
            band_mask &= d_gt.ne(self.ignore_index)

        if band_mask.sum() == 0:
            # nothing to optimise (e.g. empty crop) – safe zero loss
            return torch.zeros(
                (), dtype=d_pred.dtype, device=d_pred.device,
                requires_grad=d_pred.requires_grad
            )

        # ── Smooth-L1 (Huber) inside the band ────────────────────────────────
        huber = F.smooth_l1_loss(
            d_pred[band_mask], d_gt[band_mask],
            beta=self.beta, reduction="none"
        )

        data_term = huber

        # ── optional Eikonal regulariser ─────────────────────────────────────
        if self.eikonal:
            grad = self._gradient_3d(d_pred)           # (B,3,D,H,W)
            grad_norm = grad.norm(dim=1)               # (B,D,H,W)
            eik = (grad_norm - 1.0) ** 2
            eik_data = eik[band_mask]
            data_term = data_term + self.eik_w * eik_data

        # ── reduction ────────────────────────────────────────────────────────
        if self.reduction == "sum":
            return data_term.sum()
        if self.reduction == "none":
            out = torch.zeros_like(d_pred)
            out[band_mask] = data_term
            return out
        # "mean"  – average only over valid voxels
        return data_term.mean()

# ======================================================================
#  PLANARITY  –  encourages each foreground voxel to live on a thin sheet
# ======================================================================

import torch, math
import torch.nn.functional as F
from torch import nn

class PlanarityLoss(nn.Module):
    """
    π-planarity loss  =  mean( mask * (1 – π)^q )

      π = (λ₂ – λ₁) / (λ₀+λ₁+λ₂+eps)         using eigen-values of
          the 3×3 structure tensor J_ρ = G_ρ * (∇p ∇pᵀ).

    Parameters
    ----------
    rho           Gaussian window radius (voxels) for J_ρ          (default 1.5)
    q             Exponent  (q = 1 → L1,  q ≈ 2 for stronger)      (default 1)
    prob_thresh   Only penalise voxels where p > prob_thresh       (default 0.5)
    eps           Numerical stabiliser                             (default 1e-8)
    reduction     "mean" | "sum" | "none"                          (default "mean")
    ignore_index  Target value to skip (like Dice etc.)            (default None)
    """

    def __init__(self,
                 rho: float = 1.5,
                 q: float = 1.0,
                 prob_thresh: float = .5,
                 eps: float = 1e-8,
                 reduction: str = 'mean',
                 ignore_index=None,
                 normalization: str = 'sigmoid'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('reduction must be mean|sum|none')
        self.rho, self.q, self.eps = rho, q, eps
        self.t = prob_thresh
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.normalization = nn.Sigmoid() if normalization == 'sigmoid' \
                             else (lambda x: x)
        self._st_computer = StructureTensorComputer(
            sigma=0.0,
            component_sigma=rho if rho > 0 else None,
            smooth_components=rho > 0,
            device='cpu',
            dtype=torch.float32,
        )

    def _compute_eigenvalues_3x3_batch(self, J: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenvalues for batched 3x3 symmetric matrices using analytical formula.
        More stable and faster than torch.linalg.eigvalsh for small matrices.
        
        Input: J of shape (..., 3, 3)
        Output: eigenvalues of shape (..., 3) in ascending order
        """
        # Extract unique elements of symmetric matrix
        a11 = J[..., 0, 0]
        a22 = J[..., 1, 1]
        a33 = J[..., 2, 2]
        a12 = J[..., 0, 1]
        a13 = J[..., 0, 2]
        a23 = J[..., 1, 2]
        
        # Compute invariants
        # Trace
        p1 = a11 + a22 + a33
        
        # Sum of minors
        p2 = (a11 * a22 - a12 * a12) + (a11 * a33 - a13 * a13) + (a22 * a33 - a23 * a23)
        
        # Determinant
        p3 = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a13 * a23) + a13 * (a12 * a23 - a13 * a22)
        
        # Compute eigenvalues using Cardano's method
        q = p1 * p1 / 9.0 - p2 / 3.0
        r = p1 * p1 * p1 / 27.0 - p1 * p2 / 6.0 + p3 / 2.0
        
        # Clamp to avoid numerical issues with arccos
        sqrt_q = torch.sqrt(torch.clamp(q, min=self.eps))
        theta = torch.acos(torch.clamp(r / (sqrt_q ** 3 + self.eps), min=-1.0, max=1.0))
        
        # Eigenvalues
        sqrt_q_2 = 2.0 * sqrt_q
        p1_3 = p1 / 3.0
        
        lambda1 = p1_3 - sqrt_q_2 * torch.cos(theta / 3.0)
        lambda2 = p1_3 - sqrt_q_2 * torch.cos((theta - 2.0 * math.pi) / 3.0)
        lambda3 = p1_3 - sqrt_q_2 * torch.cos((theta - 4.0 * math.pi) / 3.0)
        
        # Stack and sort
        eigenvalues = torch.stack([lambda1, lambda2, lambda3], dim=-1)
        eigenvalues, _ = torch.sort(eigenvalues, dim=-1)
        
        return eigenvalues

    # ------------------------------------------------------------------
    def forward(self, input: torch.Tensor, target: torch.Tensor | None = None, source_pred: torch.Tensor | None = None):
        """
        input  – logits or probabilities  (B,1,D,H,W)
        target – ground-truth mask, same shape or None
        source_pred – ignored for PlanarityLoss (accepted for API consistency)
        """
        p = self.normalization(input)

        if target is not None and self.ignore_index is not None:
            valid = target.ne(self.ignore_index)
        else:
            valid = torch.ones_like(p, dtype=torch.bool)

        # ---------- structure tensor via shared computer ---------------
        rho = float(self.rho)
        component_sigma = rho if rho > 0 else None
        st_device = p.device
        self._st_computer.device = st_device
        self._st_computer.smooth_components = component_sigma is not None
        self._st_computer.component_sigma = component_sigma
        J_components = self._st_computer.compute(
            p,
            sigma=0.0,
            component_sigma=component_sigma,
            smooth_components=self._st_computer.smooth_components,
            device=st_device,
            spatial_dims=3,
        )
        J = components_to_matrix(J_components)

        # Compute eigenvalues using optimized method
        try:
            # Try analytical method first (faster and more stable)
            eigenvalues = self._compute_eigenvalues_3x3_batch(J)
        except:
            # Fallback to torch.linalg.eigvalsh if analytical method fails
            # Convert to float32 for eigvalsh (doesn't support float16)
            J_float32 = J.to(torch.float32)
            
            # Add small epsilon to diagonal for numerical stability
            eps_diag = 1e-6
            eye = torch.eye(3, dtype=J_float32.dtype, device=J_float32.device)
            J_float32 = J_float32 + eps_diag * eye
            
            eigenvalues = torch.linalg.eigvalsh(J_float32)
            # Convert back to original dtype
            eigenvalues = eigenvalues.to(J.dtype)
        
        # Extract eigenvalues (already sorted in ascending order)
        lam0, lam1, lam2 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]

        pi = (lam1 - lam0) / (lam0 + lam1 + lam2 + self.eps)
        loss_vox = (1.0 - pi).clamp(min=0).pow(self.q)

        # ---------- masks & reduction -------------------------------
        mask = (p > self.t) & valid
        # Squeeze the channel dimension from mask to match loss_vox dimensions
        mask = mask.squeeze(1)  # From (B,1,D,H,W) to (B,D,H,W)
        
        if self.reduction == 'none':
            # For 'none' reduction, apply mask but keep spatial dimensions
            loss_vox = loss_vox * mask.float()
            return loss_vox.unsqueeze(1)  # Add channel dimension back for consistency
        else:
            # For 'mean' or 'sum' reduction, flatten and extract only masked values
            loss_vox_flat = loss_vox.flatten()
            mask_flat = mask.flatten()
            loss_vox_masked = loss_vox_flat[mask_flat]
            
            if loss_vox_masked.numel() == 0:
                return torch.zeros(
                    (), dtype=input.dtype, device=input.device,
                    requires_grad=input.requires_grad)
            
            if self.reduction == 'sum':
                return loss_vox_masked.sum()
            else:  # 'mean'
                return loss_vox_masked.mean()

# ======================================================================
#  NORMAL-SMOOTH  –  penalises sharp flips in surface normal field
# ======================================================================

class NormalSmoothnessLoss(nn.Module):
    """
    L_smooth = mean( mask * (1 - ⟨n, n̄⟩)^q )

    where n̄ is n blurred with a Gaussian (σ).

    Parameters
    ----------
    sigma          Gaussian σ (vox) for the reference normal n̄     (default 2)
    q              Exponent (q=2 gives stronger push)              (default 2)
    prob_thresh    Foreground mask: use voxels where p>prob_thresh (default 0.5)
    reduction      "mean" | "sum" | "none"                         (default "mean")
    """
    def __init__(self,
                 sigma: float = 2.0,
                 q: float = 2.0,
                 prob_thresh: float = .5,
                 reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError
        self.sigma, self.q = sigma, q
        self.t = prob_thresh
        self.reduction = reduction

    # ------------------------------------------------------------------
    def _gauss_blur(self, x, sig):
        """
        Apply 3D Gaussian blur using separable 1D convolutions for efficiency.
        """
        # Create 1D Gaussian kernel
        kernel_size = int(2 * math.ceil(3 * sig) + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Create 1D Gaussian kernel
        kernel_1d = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
        kernel_1d = kernel_1d - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sig) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable convolution
        B, C, D, H, W = x.shape
        
        # Reshape for 1D convolutions
        padding = (kernel_size - 1) // 2
        
        # Conv along Z axis
        kernel_z = kernel_1d.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
        x = F.conv3d(x, kernel_z, padding=(padding, 0, 0), groups=C)
        
        # Conv along Y axis
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
        x = F.conv3d(x, kernel_y, padding=(0, padding, 0), groups=C)
        
        # Conv along X axis
        kernel_x = kernel_1d.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
        x = F.conv3d(x, kernel_x, padding=(0, 0, padding), groups=C)
        
        return x

    # ------------------------------------------------------------------
    def forward(self,
                n_pred: torch.Tensor,
                n_gt: torch.Tensor | None = None,
                source_pred: torch.Tensor | None = None):
        """
        n_pred : (B,3,D,H,W) – predicted surface normals
        n_gt   : (B,3,D,H,W) – ground truth surface normals (ignored, for compatibility)
        source_pred : (B,1,D,H,W) – source segmentation predictions (optional, for masking)
        
        Note: This loss only uses predicted normals for self-consistency smoothness.
        """
        n_pred = F.normalize(n_pred, p=2, dim=1, eps=1e-6)
        n_bar = self._gauss_blur(n_pred, self.sigma)
        n_bar = F.normalize(n_bar, p=2, dim=1, eps=1e-6)

        dot = (n_pred * n_bar).sum(1).clamp(-1, 1)   # (B,D,H,W)
        loss_vox = (1.0 - dot).pow(self.q)

        # Apply masking if source predictions are provided
        if source_pred is not None:
            # Apply sigmoid to get probabilities
            prob = torch.sigmoid(source_pred).squeeze(1)  # (B,D,H,W)
            mask = (prob > self.t).float()
            
            if self.reduction == 'none':
                return loss_vox * mask
            else:
                # Apply mask and compute mean only over valid regions
                masked_loss = loss_vox * mask
                num_valid = mask.sum()
                if num_valid > 0:
                    if self.reduction == 'sum':
                        return masked_loss.sum()
                    else:  # 'mean'
                        return masked_loss.sum() / num_valid
                else:
                    return torch.zeros((), dtype=loss_vox.dtype, device=loss_vox.device, requires_grad=True)
        else:
            # No masking
            if self.reduction == 'sum':
                return loss_vox.sum()
            elif self.reduction == 'none':
                return loss_vox
            else:  # 'mean'
                return loss_vox.mean()
    
    # ======================================================================
#  NORMAL-GATED REPULSION  –  keeps separate sheets apart, but lets the
#                            two faces of *one* sheet stay together
# ======================================================================

class NormalGatedRepulsionLoss(nn.Module):
    """
    L_rep = Σ_{‖Δx‖≤τ}  w_d(Δx) · mean( w_theta )

      w_d     = exp(-‖Δx‖² / σ_d²)
      w_theta = exp(-θ²     / σ_θ²)   with θ = angle(n_i, n_j)

    Parameters
    ----------
    tau            neighbourhood radius (voxels)                  (default 2)
    sigma_d        if None ⇒ tau/1.5                              (default None)
    sigma_theta    (deg) normal gating width                      (default 20)
    reduction      "mean" | "sum"                                 (default "mean")
    """

    def __init__(self,
                 tau: int = 2,
                 sigma_d: float | None = None,
                 sigma_theta: float = 20.,
                 reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError
        self.tau = int(tau)
        self.sigma_d2 = (sigma_d if sigma_d is not None else tau / 1.5) ** 2
        self.sigma_th2 = math.radians(sigma_theta) ** 2
        self.reduction = reduction

        # pre-compute neighbour offsets (exclude 0,0,0)
        offs = range(-self.tau, self.tau + 1)
        self.offsets = [(dz, dy, dx) for dz in offs for dy in offs for dx in offs
                        if dz or dy or dx]
        self.dist2 = {o: float(o[0]**2 + o[1]**2 + o[2]**2) for o in self.offsets}
        
        # Pre-compute distance weights as a tensor for vectorized operations
        self.register_buffer('distance_weights', 
                           torch.tensor([math.exp(-self.dist2[o] / self.sigma_d2) 
                                       for o in self.offsets], dtype=torch.float32))

    def _create_shifted_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Create all shifted versions of the input tensor for neighborhood comparisons.
        Uses padding and slicing to handle boundaries efficiently.
        
        Returns tensor of shape (B, C, num_offsets, D, H, W)
        """
        B, C, D, H, W = tensor.shape
        num_offsets = len(self.offsets)
        
        # Pad the tensor to handle boundary cases
        pad_size = self.tau
        padded = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), 
                      mode='constant', value=0)
        
        # Pre-allocate output tensor
        shifted_list = []
        
        for dz, dy, dx in self.offsets:
            # Extract shifted version from padded tensor
            z_start = pad_size + dz
            y_start = pad_size + dy
            x_start = pad_size + dx
            
            shifted_view = padded[:, :, 
                                z_start:z_start+D,
                                y_start:y_start+H,
                                x_start:x_start+W]
            shifted_list.append(shifted_view)
        
        # Stack all shifted versions along a new dimension
        shifted = torch.stack(shifted_list, dim=2)  # (B, C, num_offsets, D, H, W)
        
        return shifted
    
    # ------------------------------------------------------------------
    def forward(self,
                n_pred: torch.Tensor,       # (B,3,D,H,W)  – predicted unit normals
                n_gt: torch.Tensor = None,  # (B,3,D,H,W)  – ground truth normals (ignored)
                source_pred: torch.Tensor | None = None): # (B,1,D,H,W) – source predictions
        """
        Compute normal-gated repulsion loss using predicted normals.
        Optionally uses source predictions for masking.
        """
        B, C, D, H, W = n_pred.shape
        
        # Normalize the normals
        n_pred = F.normalize(n_pred, p=2, dim=1, eps=1e-6)
        
        # Create probability mask if source predictions are provided
        if source_pred is not None:
            # Apply sigmoid to get probabilities
            prob = torch.sigmoid(source_pred)  # (B, 1, D, H, W)
            prob_mask = (prob > 0.5)  # Threshold at 50%
            
            # Create shifted versions of probability mask
            prob_shifted = self._create_shifted_tensors(prob)  # (B, 1, num_offsets, D, H, W)
            prob_central = prob.unsqueeze(2).expand(-1, -1, len(self.offsets), -1, -1, -1)
            
            # Compute masks for valid pairs (both central and neighbor > threshold)
            mask = (prob_central > 0.5) & (prob_shifted > 0.5)  # (B, 1, num_offsets, D, H, W)
            mask = mask.squeeze(1).float()  # (B, num_offsets, D, H, W)
        else:
            mask = None
        
        # Create shifted versions of normals
        n_pred_shifted = self._create_shifted_tensors(n_pred)  # (B, 3, num_offsets, D, H, W)
        
        # Central (unshifted) normals - expand to match shifted shape
        n_pred_central = n_pred.unsqueeze(2).expand(-1, -1, len(self.offsets), -1, -1, -1)
        
        # Compute dot products between central and shifted normals
        # Sum over channel dimension (dim=1)
        dot_products = (n_pred_central * n_pred_shifted).sum(dim=1).clamp(-1, 1)  # (B, num_offsets, D, H, W)
        
        # Compute angles and angular weights
        theta2 = torch.acos(dot_products).pow(2)
        w_theta = torch.exp(-theta2 / self.sigma_th2)
        
        # Apply distance weights (broadcast to match shape)
        w_dist = self.distance_weights.to(n_pred.device).to(n_pred.dtype)
        w_dist = w_dist.view(1, -1, 1, 1, 1)  # Shape for broadcasting
        
        # Compute loss values
        loss_vox = w_theta * w_dist  # (B, num_offsets, D, H, W)
        
        # Apply mask if available
        if mask is not None:
            loss_vox = loss_vox * mask
            total = loss_vox.sum()
            count = mask.sum()
            
            if count == 0:
                return torch.zeros((), dtype=n_pred.dtype, device=n_pred.device, requires_grad=True)
        else:
            total = loss_vox.sum()
            count = loss_vox.numel()
        
        if self.reduction == 'sum':
            return total
        return total / count



#######################################################################################################################

def _create_loss(name, loss_config, weight, ignore_index, pos_weight, mgr=None):
    # Define losses that don't natively support ignore_index
    losses_without_ignore_support = ['BCEWithLogitsLoss', 'MSELoss', 'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']
    
    # Create the base loss function
    if name == 'BCEWithLogitsLoss':
        base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'MSELoss':
        base_loss = MSELoss()
    elif name == 'MaskedMSELoss':
        mask_channel = loss_config.get('mask_channel', False)
        base_loss = MaskedMSELoss(ignore_index=ignore_index, mask_channel=mask_channel)
    elif name == 'SmoothL1Loss':
        base_loss = SmoothL1Loss()
    elif name == 'L1Loss':
        base_loss = L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        base_loss = WeightedSmoothL1Loss(threshold=loss_config['threshold'],
                                        initial_weight=loss_config['initial_weight'],
                                        apply_below_threshold=loss_config.get('apply_below_threshold', True))
    elif name == 'EigenvalueLoss':
        base_loss = EigenvalueLoss(
            reduction   = loss_config.get('reduction', 'mean'),
            relative    = loss_config.get('relative', False),
            weight      = weight,
            ignore_index= ignore_index, 
            eps         = loss_config.get('eps', 1e-8)
        )
    elif name == 'CosineSimilarityLoss':
        dim = int(loss_config.get('dim', 1))
        eps = float(loss_config.get('eps', 1e-8))
        base_loss = CosineSimilarityLoss(dim=dim, eps=eps, ignore_index=ignore_index)

    elif name == 'SignInvariantCosineLoss':
        dim = int(loss_config.get('dim', 1))
        eps = float(loss_config.get('eps', 1e-8))
        base_loss = SignInvariantCosineLoss(dim=dim, eps=eps, ignore_index=ignore_index)

    elif name == 'ChannelSelectLoss':
        base_cfg = loss_config.get('base', {})
        if not base_cfg or 'name' not in base_cfg:
            raise RuntimeError("ChannelSelectLoss requires 'base' config with a 'name'")
        base_name = base_cfg['name']
        base_kwargs = base_cfg.get('kwargs', {})
        # Recursively create the base loss
        inner = _create_loss(
            name=base_name,
            loss_config=base_kwargs,
            weight=weight,
            ignore_index=ignore_index,
            pos_weight=pos_weight,
            mgr=mgr,
        )
        channels = loss_config.get('channels', 'all_but_last')
        base_loss = ChannelSelectLoss(inner, channels)

    elif name == 'SignedDistanceLoss':
        # rho, beta, eikonal, eikonal_weight, reduction are read from the YAML / json
        base_loss = SignedDistanceLoss(
            rho             = loss_config.get('rho', 4.0),
            beta            = loss_config.get('beta', 1.0),
            eikonal         = loss_config.get('eikonal', True),
            eikonal_weight  = loss_config.get('eikonal_weight', 0.01),
            reduction       = loss_config.get('reduction', 'mean'),
            ignore_index    = ignore_index,
        )
        
    elif name == 'PlanarityLoss':
        base_loss = PlanarityLoss(
            rho           = loss_config.get('rho', 1.5),
            q             = loss_config.get('q', 1.0),
            prob_thresh   = loss_config.get('prob_thresh', 0.5),
            reduction     = loss_config.get('reduction', 'mean'),
            ignore_index  = ignore_index,
            normalization = loss_config.get('normalization', 'sigmoid')
        )

    elif name == 'NormalSmoothnessLoss':
        base_loss = NormalSmoothnessLoss(
            sigma        = loss_config.get('sigma', 2.0),
            q            = loss_config.get('q', 2.0),
            prob_thresh  = loss_config.get('prob_thresh', 0.5),
            reduction    = loss_config.get('reduction', 'mean')
        )

    elif name == 'NormalGatedRepulsionLoss':
        base_loss = NormalGatedRepulsionLoss(
            tau          = loss_config.get('tau', 2),
            sigma_d      = loss_config.get('sigma_d', None),
            sigma_theta  = loss_config.get('sigma_theta', 20.0),
            reduction    = loss_config.get('reduction', 'mean')
        )
    
    # nnUNet losses
    elif name == 'nnUNet_DC_and_CE_loss':
        # Default nnUNet loss: Dice + Cross Entropy
        # Use nnUNet's default parameters if not specified
        soft_dice_kwargs = {
            'batch_dice': loss_config.get('batch_dice', False),
            'smooth': loss_config.get('smooth', 1e-5),
            'do_bg': loss_config.get('do_bg', False),
            'ddp': loss_config.get('ddp', False)
        }
        
        # Allow override with soft_dice_kwargs if provided
        if 'soft_dice_kwargs' in loss_config:
            soft_dice_kwargs.update(loss_config['soft_dice_kwargs'])
        
        ce_kwargs = loss_config.get('ce_kwargs', {})
        weight_ce = loss_config.get('weight_ce', 1)
        weight_dice = loss_config.get('weight_dice', 1)
        
        base_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_index,
            dice_class=MemoryEfficientSoftDiceLoss
        )
    
    elif name == 'nnUNet_DC_and_BCE_loss':
        # nnUNet loss for region-based training: Dice + Binary Cross Entropy
        # Use nnUNet's default parameters if not specified
        soft_dice_kwargs = {
            'batch_dice': loss_config.get('batch_dice', False),
            'smooth': loss_config.get('smooth', 1e-5),
            'do_bg': loss_config.get('do_bg', True),  # Note: True for BCE variant
            'ddp': loss_config.get('ddp', False)
        }
        
        # Allow override with soft_dice_kwargs if provided
        if 'soft_dice_kwargs' in loss_config:
            soft_dice_kwargs.update(loss_config['soft_dice_kwargs'])
            
        bce_kwargs = loss_config.get('bce_kwargs', {})
        weight_ce = loss_config.get('weight_ce', 1)
        weight_dice = loss_config.get('weight_dice', 1)
        use_ignore_label = ignore_index is not None
        
        base_loss = DC_and_BCE_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            use_ignore_label=use_ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )
    
    elif name == 'MemoryEfficientSoftDiceLoss':
        # Standalone memory efficient dice loss
        base_loss = MemoryEfficientSoftDiceLoss(
            apply_nonlin=loss_config.get('apply_nonlin', None),
            batch_dice=loss_config.get('batch_dice', False),
            do_bg=loss_config.get('do_bg', True),
            smooth=loss_config.get('smooth', 1.),
            ddp=loss_config.get('ddp', False)
        )

    elif name == 'MedialSurfaceRecall':
        from vesuvius.models.training.loss.skeleton_recall import DC_SkelREC_and_CE_loss
        soft_dice_kwargs = {
            'batch_dice': loss_config.get('batch_dice', False),
            'smooth': loss_config.get('smooth', 1e-5),
            'do_bg': loss_config.get('do_bg', False),
            'ddp': loss_config.get('ddp', False)
        }

        # Allow override with soft_dice_kwargs if provided
        if 'soft_dice_kwargs' in loss_config:
            soft_dice_kwargs.update(loss_config['soft_dice_kwargs'])

        ce_kwargs = loss_config.get('ce_kwargs', {})
        weight_ce = loss_config.get('weight_ce', 1)
        weight_dice = loss_config.get('weight_dice', 1)
        use_ignore_label = ignore_index is not None

        base_loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs={
                'batch_dice': soft_dice_kwargs.get('batch_dice'),
                'smooth': soft_dice_kwargs.get('smooth'),
                'do_bg': soft_dice_kwargs.get('do_bg'),
                'ddp': soft_dice_kwargs.get('ddp')
            },
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            weight_srec=loss_config.get('weight_srec', 1),
            ignore_label=ignore_index,
            dice_class=MemoryEfficientSoftDiceLoss
        )

    elif name == 'BettiMatchingLoss':
        from .betti_losses import BettiMatchingLoss
        base_loss = BettiMatchingLoss(
            filtration=loss_config.get('filtration', 'superlevel')
        )

    
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
    
    
    # Wrap with MaskingLossWrapper if ignore_index is specified and loss doesn't support it natively
    if ignore_index is not None and name in losses_without_ignore_support:
        return MaskingLossWrapper(base_loss, ignore_index)
    
    return base_loss
