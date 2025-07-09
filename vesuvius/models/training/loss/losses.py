import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from .mae_loss import MaskedReconstructionLoss


def compute_per_channel_dice(input, target, epsilon=1e-5, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


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


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='softmax'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid', exclude_channels=None):
        super().__init__(weight, normalization)
        self.exclude_channels = exclude_channels if exclude_channels is not None else []

    def dice(self, input, target, weight):
        per_channel_dice = compute_per_channel_dice(input, target, weight=self.weight)
        
        # Create mask to exclude specified channels
        if len(self.exclude_channels) > 0:
            mask = torch.ones(per_channel_dice.shape[0], dtype=torch.bool, device=per_channel_dice.device)
            for ch in self.exclude_channels:
                if 0 <= ch < per_channel_dice.shape[0]:
                    mask[ch] = False
            # Return only the dice scores for non-excluded channels
            return per_channel_dice[mask]
        
        return per_channel_dice


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-5):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.bce(input, target) + self.alpha * self.dice(input, target)


class CEDiceLoss(nn.Module):
    """Linear combination of CrossEntropy and Dice losses for multi-class segmentation"""

    def __init__(self, alpha=0.5, weight=None, ignore_index=-100):
        """
        Args:
            alpha (float): Weight factor for Dice loss component (default: 1.0)
            weight (Tensor): Manual rescaling weight for each class (for CrossEntropyLoss)
            ignore_index (int): Specifies a target value that is ignored (default: -100)
        """
        super(CEDiceLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        # CrossEntropyLoss with weight and ignore_index support
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
        # DiceLoss with softmax normalization for multi-channel
        # Exclude channel 0 (background) from dice computation
        self.dice = DiceLoss(weight=weight, normalization='softmax', exclude_channels=[0])
        
        # For tracking individual components
        self.last_ce_loss = 0
        self.last_dc_loss = 0

    def forward(self, input, target):
        # Check if target is one-hot encoded (same dims as input and channel > 1)
        if target.dim() == input.dim() and target.shape[1] > 1:
            # Target is already one-hot encoded
            target_one_hot = target
            # Convert to class indices for CrossEntropyLoss
            target_ce = target.argmax(dim=1).long()
        else:
            # Target is class indices, handle as before
            # Ensure target has channel dimension for shape consistency check
            if target.dim() == input.dim() - 1:
                target = target.unsqueeze(1)
            
            # For CrossEntropyLoss: need to squeeze channel dimension and convert to long
            target_ce = target.squeeze(1).long()
            
            # For DiceLoss: need to convert target to one-hot encoding
            # Get number of classes from input
            num_classes = input.shape[1]
            
            # Create one-hot encoded target for Dice loss
            # First, handle ignore_index by creating a mask
            mask = (target_ce != self.ignore_index)
            
            # Create one-hot tensor
            target_shape = list(target_ce.shape)
            target_shape.insert(1, num_classes)  # Insert channel dimension
            target_one_hot = torch.zeros(target_shape, dtype=input.dtype, device=input.device)
            
            # Fill one-hot encoding only for valid pixels
            # Use scatter_ with dim=1 to fill the channel dimension
            valid_target = target_ce.clone()
            valid_target[~mask] = 0  # Set ignored pixels to class 0 temporarily
            
            # Expand target to have the same number of dimensions for scatter
            scatter_target = valid_target.unsqueeze(1)
            target_one_hot.scatter_(1, scatter_target, 1)
            
            # Zero out ignored pixels in one-hot encoding
            for c in range(num_classes):
                target_one_hot[:, c][~mask] = 0
        
        # Calculate CE loss
        ce_loss = self.ce(input, target_ce)
        self.last_ce_loss = ce_loss.item()
        
        # Handle ignore_index for Dice loss when target was one-hot encoded
        if target.dim() == input.dim() and target.shape[1] > 1 and self.ignore_index != -100:
            # Create mask for ignore_index
            mask = (target_ce != self.ignore_index)
            
            # Apply mask to both input and target for Dice calculation
            masked_input = input.clone()
            masked_target = target_one_hot.clone()
            for c in range(input.shape[1]):
                masked_input[:, c][~mask] = 0
                masked_target[:, c][~mask] = 0
            
            # Calculate Dice loss with masked tensors
            dice_loss = self.dice(masked_input, masked_target)
        else:
            # Calculate Dice loss normally
            dice_loss = self.dice(input, target_one_hot)
        
        self.last_dc_loss = dice_loss.item()
        
        # Combine losses
        total_loss = ce_loss + self.alpha * dice_loss
        
        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = nominator / denominator
        return class_weights.detach()


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
        
    def forward(self, input, target, mask=None):
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


class CrossEntropyLossWrapper(nn.Module):
    """
    Wrapper for CrossEntropyLoss that handles the channel dimension from BaseDataset.
    BaseDataset outputs labels with shape [B, 1, H, W] or [B, 1, D, H, W],
    but CrossEntropyLoss expects [B, H, W] or [B, D, H, W].
    """
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index
    
    def forward(self, input, target):
        # Remove channel dimension from target if present
        # From [B, 1, H, W] to [B, H, W] or [B, 1, D, H, W] to [B, D, H, W]
        if target.dim() == input.dim() and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # Convert to long for CrossEntropyLoss
        target = target.long()
        
        return self.loss(input, target)


class WeightedCrossEntropyLossWrapper(WeightedCrossEntropyLoss):
    """
    Wrapper for WeightedCrossEntropyLoss that handles the channel dimension from BaseDataset.
    """
    def forward(self, input, target):
        # Remove channel dimension from target if present
        # From [B, 1, H, W] to [B, H, W] or [B, 1, D, H, W] to [B, D, H, W]
        if target.dim() == input.dim() and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # Convert to long for CrossEntropyLoss
        target = target.long()
        
        # Call parent class forward method
        return super().forward(input, target)

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
        
        # Create Sobel kernels for 3D gradients
        self._create_sobel_kernels()

    def _create_sobel_kernels(self):
        """Create 3D Sobel kernels for gradient computation."""
        # Basic 1D kernels
        smooth = torch.tensor([1., 2., 1.], dtype=torch.float32)
        diff = torch.tensor([-1., 0., 1.], dtype=torch.float32)
        
        # Create 3D Sobel kernels for each direction
        # For dz (axis 0)
        kernel_z = diff.view(-1, 1, 1) * smooth.view(1, -1, 1) * smooth.view(1, 1, -1)
        kernel_z = kernel_z.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3, 3)
        
        # For dy (axis 1)
        kernel_y = smooth.view(-1, 1, 1) * diff.view(1, -1, 1) * smooth.view(1, 1, -1)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        
        # For dx (axis 2)
        kernel_x = smooth.view(-1, 1, 1) * smooth.view(1, -1, 1) * diff.view(1, 1, -1)
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        
        # Normalize by the sum of absolute values (similar to scipy's normalization)
        kernel_z = kernel_z / 8.0
        kernel_y = kernel_y / 8.0
        kernel_x = kernel_x / 8.0
        
        # Stack kernels for all three gradients
        self.register_buffer('sobel_kernels', torch.cat([kernel_z, kernel_y, kernel_x], dim=0))
    
    # ------------------------------------------------------------------
    def _sobel3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D Sobel gradients using conv3d.
        Input shape: (B, 1, D, H, W)
        Output shape: (B, 3, D, H, W) for (dz, dy, dx)
        """
        # Ensure kernels are on the same device and dtype as input
        kernels = self.sobel_kernels.to(x.device).to(x.dtype)
        
        # Apply convolution for all three gradients at once
        # Input: (B, 1, D, H, W), Kernel: (3, 1, 3, 3, 3), Output: (B, 3, D, H, W)
        gradients = F.conv3d(x, kernels, padding=1)
        
        return gradients

    def _gauss_blur(self, x: torch.Tensor, sig: float):
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

        # ---------- gradients & structure tensor ----------------------
        g = self._sobel3d(p)                           # B,3,D,H,W
        
        # Compute structure tensor components (outer products)
        # J = ∇p ∇p^T, which has 6 unique components for symmetric 3x3
        J_components = []
        for i in range(3):
            for j in range(i, 3):
                J_components.append(g[:, i:i+1] * g[:, j:j+1])
        
        J_components = torch.cat(J_components, dim=1)  # (B, 6, D, H, W)
        
        # Apply Gaussian blur to structure tensor components
        J_components = self._gauss_blur(J_components, self.rho)
        
        # Extract components
        Jxx = J_components[:, 0]
        Jxy = J_components[:, 1]
        Jxz = J_components[:, 2]
        Jyy = J_components[:, 3]
        Jyz = J_components[:, 4]
        Jzz = J_components[:, 5]
        
        # Reconstruct 3x3 structure tensor
        # Shape: (B, D, H, W, 3, 3)
        B, D, H, W = Jxx.shape
        J = torch.zeros(B, D, H, W, 3, 3, dtype=Jxx.dtype, device=Jxx.device)
        
        J[..., 0, 0] = Jxx
        J[..., 0, 1] = Jxy
        J[..., 0, 2] = Jxz
        J[..., 1, 0] = Jxy
        J[..., 1, 1] = Jyy
        J[..., 1, 2] = Jyz
        J[..., 2, 0] = Jxz
        J[..., 2, 1] = Jyz
        J[..., 2, 2] = Jzz

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

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    # Define losses that don't natively support ignore_index
    losses_without_ignore_support = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'DiceLoss', 'GeneralizedDiceLoss', 
                                   'MSELoss', 'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']
    
    # Create the base loss function
    if name == 'BCEWithLogitsLoss':
        base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        base_loss = BCEDiceLoss(alpha)
    elif name == 'CEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        # CEDiceLoss has built-in support for weight and ignore_index
        return CEDiceLoss(alpha=alpha, weight=weight, ignore_index=ignore_index)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        # Use wrapper to handle channel dimension
        return CrossEntropyLossWrapper(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        # Use wrapper to handle channel dimension
        return WeightedCrossEntropyLossWrapper(ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        base_loss = GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        exclude_channels = loss_config.get('exclude_channels', None)
        base_loss = DiceLoss(weight=weight, normalization=normalization, exclude_channels=exclude_channels)
    elif name == 'MSELoss':
        base_loss = MSELoss()
    elif name == 'MaskedMSELoss':
        mask_channel = loss_config.get('mask_channel', False)
        base_loss = MaskedMSELoss(ignore_index=ignore_index, mask_channel=mask_channel)
    elif name == 'MaskedReconstructionLoss':
        base_loss_type = loss_config.get('base_loss', 'mse')
        normalize_targets = loss_config.get('normalize_targets', True)
        reduction = loss_config.get('reduction', 'mean')
        eps = loss_config.get('eps', 1e-6)
        variance_threshold = loss_config.get('variance_threshold', 0.1)
        use_robust_norm = loss_config.get('use_robust_norm', True)
        max_loss_value = loss_config.get('max_loss_value', 100.0)
        log_high_losses = loss_config.get('log_high_losses', True)
        base_loss = MaskedReconstructionLoss(base_loss=base_loss_type, 
                                         normalize_targets=normalize_targets,
                                         reduction=reduction,
                                         eps=eps,
                                         variance_threshold=variance_threshold,
                                         use_robust_norm=use_robust_norm,
                                         max_loss_value=max_loss_value,
                                         log_high_losses=log_high_losses)
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
        dim = loss_config.get('dim', 1)
        eps = loss_config.get('eps', 1e-8)
        base_loss = CosineSimilarityLoss(dim=dim, eps=eps, ignore_index=ignore_index)

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

    
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
    
    
    # Wrap with MaskingLossWrapper if ignore_index is specified and loss doesn't support it natively
    if ignore_index is not None and name in losses_without_ignore_support:
        return MaskingLossWrapper(base_loss, ignore_index)
    
    return base_loss
