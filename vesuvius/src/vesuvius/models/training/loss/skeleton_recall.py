from vesuvius.models.training.loss.nnunet_losses import MemoryEfficientSoftDiceLoss, SoftDiceLoss, AllGatherGrad, RobustCrossEntropyLoss
from vesuvius.models.training.loss.loss_helpers import softmax_helper_dim1, softmax_helper
import torch.nn as nn
import torch
from typing import Callable


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                num_classes = shp_x[1]
                # Clamp to valid indices to prevent CUDA scatter_ crash
                # This is safe because ignored pixels are masked out anyway
                gt = torch.clamp(y.long(), min=0, max=num_classes - 1)
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return -rec

class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_srec=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor, loss_mask=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :param skel: skeleton target for skeleton recall loss
        :param loss_mask: optional spatial mask to weight/exclude regions from loss (e.g., exclude conditioning region)
        :return:
        """
        import torch.nn.functional as F

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None

        # Combine external loss_mask with ignore_label mask
        combined_mask = mask
        if loss_mask is not None:
            combined_mask = loss_mask if combined_mask is None else combined_mask * loss_mask

        dc_loss = self.dc(net_output, target_dice, loss_mask=combined_mask) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, target_skel, loss_mask=combined_mask) \
            if self.weight_srec != 0 else 0

        # CE loss with optional masking
        if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0):
            if combined_mask is not None:
                # Use target_dice which has ignore_label values replaced with 0
                # to prevent CUDA assertion failures (values must be in [0, num_classes-1])
                ce_per_voxel = F.cross_entropy(
                    net_output, target_dice[:, 0].long(), reduction='none'
                )
                ce_loss = (ce_per_voxel * combined_mask[:, 0]).sum() / combined_mask.sum().clamp(min=1)
            else:
                ce_loss = self.ce(net_output, target[:, 0])
        else:
            ce_loss = 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result