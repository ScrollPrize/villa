"""
nnUNet loss implementations adapted for vesuvius.
Based on nnUNetv2 but standalone without imports from nnunetv2.
"""
import torch
from torch import nn
from typing import Callable, Optional, Union, List, Tuple
import numpy as np


class AllGatherGrad(torch.autograd.Function):
    """
    All gather with gradient support for distributed training.
    """
    @staticmethod
    def forward(ctx, x):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # Gather tensors from all ranks
            x_list = [torch.zeros_like(x) for _ in range(world_size)]
            torch.distributed.all_gather(x_list, x)
            return torch.cat(x_list, dim=0)
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # Return only the gradient for this rank
            grad_per_rank = grad_output.shape[0] // world_size
            return grad_output[rank * grad_per_rank:(rank + 1) * grad_per_rank]
        else:
            return grad_output


def softmax_helper_dim1(x):
    """Helper function to apply softmax along dimension 1."""
    return torch.softmax(x, dim=1)


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    Calculate true positives, false positives, false negatives, true negatives.
    
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.ones_like(mask, dtype=torch.bool)
            mask_here[mask == 0] = 0
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if axes is not None:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class MemoryEfficientSoftDiceLoss(nn.Module):
    """
    Memory efficient implementation of soft dice loss.
    Based on nnUNetv2 implementation.
    """
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1., ddp: bool = True,
                 label_smoothing: float = 0.0, ignore_label: int = None):
        super(MemoryEfficientSoftDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.label_smoothing = label_smoothing
        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            # Handle ignore_label: create internal mask and combine with provided loss_mask
            if self.ignore_label is not None:
                ignore_mask = (y == self.ignore_label)
                if loss_mask is None:
                    loss_mask = ~ignore_mask
                else:
                    loss_mask = loss_mask & ~ignore_mask

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                num_classes = x.shape[1]
                # Clamp y to valid indices to prevent CUDA scatter_ crash
                # This is safe because ignored pixels are masked out anyway
                y_safe = torch.clamp(y.long(), min=0, max=num_classes - 1)

                if self.label_smoothing > 0:
                    # Soft one-hot encoding with label smoothing
                    y_onehot = torch.full(x.shape, self.label_smoothing / (num_classes - 1),
                                          device=x.device, dtype=x.dtype)
                    y_onehot.scatter_(1, y_safe, 1.0 - self.label_smoothing)
                else:
                    y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                    y_onehot.scatter_(1, y_safe, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Cross entropy loss that supports ignore_index and is more robust.
    Based on nnUNetv2 implementation.

    When label_smoothing > 0 and ignore_index is set, we handle this manually
    because PyTorch's CrossEntropyLoss can have issues with this combination.
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        target = target.long()

        # When using ignore_index, we need to handle manually
        # because PyTorch CE can crash if target values >= num_classes even with ignore_index set
        if self.ignore_index is not None and self.ignore_index >= 0:
            num_classes = input.shape[1]
            # Create mask of valid (non-ignored) pixels
            valid_mask = target != self.ignore_index
            # Replace ignore_index values with 0 to prevent CUDA crash
            # These will be masked out of the loss anyway
            target_safe = torch.where(valid_mask, target, torch.zeros_like(target))
            # Also clamp to valid range in case of other out-of-range values
            target_safe = torch.clamp(target_safe, min=0, max=num_classes - 1)

            # Compute loss with no reduction
            original_reduction = self.reduction
            self.reduction = 'none'
            loss = super().forward(input, target_safe)
            self.reduction = original_reduction

            # Apply mask and reduce
            loss = loss * valid_mask.float()
            if original_reduction == 'mean':
                num_valid = valid_mask.sum()
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0  # Return 0 with grad
            elif original_reduction == 'sum':
                return loss.sum()
            else:
                return loss

        return super().forward(input, target)


class DC_and_CE_loss(nn.Module):
    """
    Combined Dice and Cross Entropy loss as used in nnUNetv2.
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        # Pass ignore_label to dice loss so it handles masking internally
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, ignore_label=ignore_label, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            num_fg = (target != self.ignore_label).sum()
        else:
            num_fg = None

        # Dice loss handles ignore_label internally via clamping + masking
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0

        # CE loss uses ignore_index natively
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy loss as used in nnUNetv2 for region-based training.
    """
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, 
                 use_ignore_label: bool = False, dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None):
        # If use_ignore_label is set, derive the mask from the ignore channel
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = loss_mask

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            denom = torch.clip(mask.sum(), min=1e-8)
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / denom
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DeepSupervisionWrapper(nn.Module):
    """
    Wrapper for deep supervision as used in nnUNetv2.
    """
    def __init__(self, loss, weights):
        super(DeepSupervisionWrapper, self).__init__()
        self.loss = loss
        self.weights = weights

    def forward(self, net_output: Union[torch.Tensor, List[torch.Tensor]], 
                target: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        """
        net_output and target can be either Tensors or lists of Tensors.
        If lists, they represent outputs at different resolutions for deep supervision.
        """
        if isinstance(net_output, (list, tuple)):
            # Deep supervision case
            assert isinstance(target, (list, tuple))
            assert len(net_output) == len(target) == len(self.weights)
            
            loss = 0
            for i, weight in enumerate(self.weights):
                if weight != 0:
                    # Pass through any additional args/kwargs (e.g., skeleton_data). If an arg is a list/tuple,
                    # use the i-th element to match the current supervision scale.
                    if args:
                        f_args = [a[i] if isinstance(a, (list, tuple)) else a for a in args]
                    else:
                        f_args = []
                    if kwargs:
                        f_kwargs = {k: (v[i] if isinstance(v, (list, tuple)) else v) for k, v in kwargs.items()}
                    else:
                        f_kwargs = {}
                    res = self.loss(net_output[i], target[i], *f_args, **f_kwargs)
                    # Some losses (e.g., Betti) may return (loss, aux_dict); extract loss
                    if isinstance(res, tuple) and len(res) == 2:
                        res = res[0]
                    loss += weight * res
            return loss
        else:
            # Regular case, no deep supervision
            res = self.loss(net_output, target, *args, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                return res[0]
            return res
