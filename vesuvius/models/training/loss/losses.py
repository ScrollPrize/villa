import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


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

    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
    
    
    # Wrap with MaskingLossWrapper if ignore_index is specified and loss doesn't support it natively
    if ignore_index is not None and name in losses_without_ignore_support:
        return MaskingLossWrapper(base_loss, ignore_index)
    
    return base_loss
