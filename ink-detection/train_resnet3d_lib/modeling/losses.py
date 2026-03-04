import torch
import torch.nn.functional as F


def compute_per_sample_loss_and_dice(logits, targets, *, smooth_factor=0.25, eps=1e-7):
    targets = targets.float()
    soft_targets = (1.0 - targets) * float(smooth_factor) + targets * (1.0 - float(smooth_factor))

    bce = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + float(eps)) / (union + float(eps))

    dice_loss = 1.0 - dice
    per_sample_loss = 0.5 * dice_loss + 0.5 * bce
    return per_sample_loss, dice, bce, dice_loss
