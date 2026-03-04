import torch
import torch.nn.functional as F


def build_bce_targets(
    targets,
    *,
    smooth_factor=0.25,
    soft_label_positive=1.0,
    soft_label_negative=0.0,
):
    targets = targets.float()
    soft_label_positive = float(soft_label_positive)
    soft_label_negative = float(soft_label_negative)
    soft_targets = targets * soft_label_positive + (1.0 - targets) * soft_label_negative
    smooth_factor = float(smooth_factor)
    if smooth_factor != 0.0:
        soft_targets = (1.0 - soft_targets) * smooth_factor + soft_targets * (1.0 - smooth_factor)
    return soft_targets


def compute_per_sample_loss_and_dice(
    logits,
    targets,
    *,
    loss_recipe="dice_bce",
    smooth_factor=0.25,
    soft_label_positive=1.0,
    soft_label_negative=0.0,
    eps=1e-7,
):
    targets = targets.float()
    bce_targets = build_bce_targets(
        targets,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
    )

    bce = F.binary_cross_entropy_with_logits(logits, bce_targets, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + float(eps)) / (union + float(eps))

    dice_loss = 1.0 - dice
    loss_recipe = str(loss_recipe).strip().lower()
    if loss_recipe == "dice_bce":
        per_sample_loss = 0.5 * dice_loss + 0.5 * bce
    elif loss_recipe == "bce_only":
        per_sample_loss = bce
    else:
        raise ValueError(f"Unknown training.loss_recipe: {loss_recipe!r}")
    return per_sample_loss, dice, bce, dice_loss
