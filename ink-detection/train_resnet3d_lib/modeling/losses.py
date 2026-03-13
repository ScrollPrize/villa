import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

_BETTI_MATCHING_LOSS_CLASS = None
_BETTI_MATCHING_LOSS_CACHE = {}


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


def compute_region_loss_and_dice(
    logits,
    targets,
    *,
    valid_mask=None,
    reduction_dims=(1, 2, 3),
    loss_recipe="dice_bce",
    smooth_factor=0.25,
    soft_label_positive=1.0,
    soft_label_negative=0.0,
    eps=1e-7,
):
    targets = targets.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.float32)
    else:
        valid_mask = valid_mask.to(device=targets.device, dtype=torch.float32)
        if tuple(valid_mask.shape) != tuple(targets.shape):
            raise ValueError(
                f"valid_mask shape must match targets shape, got {tuple(valid_mask.shape)} vs {tuple(targets.shape)}"
            )
    bce_targets = build_bce_targets(
        targets,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
    )

    bce = F.binary_cross_entropy_with_logits(logits, bce_targets, reduction="none")
    reduce_dims = tuple(int(dim) for dim in reduction_dims)
    denom = valid_mask.sum(dim=reduce_dims).clamp_min(1.0)
    bce = (bce * valid_mask).sum(dim=reduce_dims) / denom

    probs = torch.sigmoid(logits)
    intersection = (probs * targets * valid_mask).sum(dim=reduce_dims)
    union = (probs * valid_mask).sum(dim=reduce_dims) + (targets * valid_mask).sum(dim=reduce_dims)
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


def binary_mask_to_signed_distance_map(mask, *, dtype=np.float32):
    mask_np = np.asarray(mask, dtype=np.bool_)
    if mask_np.ndim != 2:
        raise ValueError(f"binary_mask_to_signed_distance_map expects a 2D mask, got shape={tuple(mask_np.shape)}")
    if not bool(mask_np.any()):
        return np.zeros(mask_np.shape, dtype=dtype)

    negmask = ~mask_np
    dist_out = ndimage.distance_transform_edt(negmask)
    dist_in = ndimage.distance_transform_edt(mask_np)
    signed_dist = dist_out * negmask - (dist_in - 1.0) * mask_np
    return np.asarray(signed_dist, dtype=dtype)


def compute_binary_boundary_loss(
    logits,
    dist_map,
    *,
    valid_mask=None,
    reduction_dims=(1, 2, 3),
):
    probs = torch.sigmoid(logits)
    dist_map = dist_map.to(device=probs.device, dtype=probs.dtype)
    if tuple(dist_map.shape) != tuple(probs.shape):
        raise ValueError(f"dist_map shape must match logits shape, got {tuple(dist_map.shape)} vs {tuple(probs.shape)}")

    if valid_mask is None:
        valid_mask = torch.ones_like(probs, dtype=probs.dtype)
    else:
        valid_mask = valid_mask.to(device=probs.device, dtype=probs.dtype)
        if tuple(valid_mask.shape) != tuple(probs.shape):
            raise ValueError(
                f"valid_mask shape must match logits shape, got {tuple(valid_mask.shape)} vs {tuple(probs.shape)}"
            )

    reduce_dims = tuple(int(dim) for dim in reduction_dims)
    denom = valid_mask.sum(dim=reduce_dims).clamp_min(1.0)
    return (probs * dist_map * valid_mask).sum(dim=reduce_dims) / denom


def _soft_erode_2d(img):
    if img.ndim != 4:
        raise ValueError(f"_soft_erode_2d expects shape (N, C, H, W), got {tuple(img.shape)}")
    pooled_h = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
    pooled_w = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(pooled_h, pooled_w)


def _soft_dilate_2d(img):
    if img.ndim != 4:
        raise ValueError(f"_soft_dilate_2d expects shape (N, C, H, W), got {tuple(img.shape)}")
    return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


def _soft_open_2d(img):
    return _soft_dilate_2d(_soft_erode_2d(img))


def _soft_skeletonize_2d(img, *, num_iter):
    if int(num_iter) < 0:
        raise ValueError(f"num_iter must be >= 0, got {num_iter}")
    img_open = _soft_open_2d(img)
    skel = F.relu(img - img_open)
    for _ in range(int(num_iter)):
        img = _soft_erode_2d(img)
        img_open = _soft_open_2d(img)
        delta = F.relu(img - img_open)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def _prepare_binary_topology_inputs(logits, targets, *, valid_mask=None):
    if tuple(logits.shape) != tuple(targets.shape):
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if logits.ndim != 4:
        raise ValueError(f"binary topology helpers expect shape (N, C, H, W), got {tuple(logits.shape)}")

    probs = torch.sigmoid(logits)
    targets = targets.to(device=probs.device, dtype=probs.dtype)
    topology_mask = None
    if valid_mask is not None:
        topology_mask = valid_mask.to(device=probs.device, dtype=probs.dtype)
        if tuple(topology_mask.shape) != tuple(probs.shape):
            raise ValueError(
                f"valid_mask shape must match logits shape, got {tuple(topology_mask.shape)} vs {tuple(probs.shape)}"
            )
    return probs, targets, topology_mask


def _get_betti_matching_loss_class():
    global _BETTI_MATCHING_LOSS_CLASS
    if _BETTI_MATCHING_LOSS_CLASS is not None:
        return _BETTI_MATCHING_LOSS_CLASS

    try:
        from topolosses.losses import BettiMatchingLoss
    except Exception:
        try:
            from topolosses.losses.betti_matching import BettiMatchingLoss
        except Exception as exc:
            raise ImportError(
                "Betti matching training loss requires `topolosses`. "
                "Install `topolosses==0.2.0` on the training environment."
            ) from exc

    _BETTI_MATCHING_LOSS_CLASS = BettiMatchingLoss
    return _BETTI_MATCHING_LOSS_CLASS


def _resolve_betti_matching_loss_module(*, filtration_type, num_processes):
    key = (str(filtration_type).strip().lower(), int(num_processes))
    module = _BETTI_MATCHING_LOSS_CACHE.get(key)
    if module is not None:
        return module

    betti_matching_loss_cls = _get_betti_matching_loss_class()
    common_kwargs = {
        "filtration_type": key[0],
        "num_processes": key[1],
        "sigmoid": False,
        "softmax": False,
        "include_background": False,
    }
    try:
        module = betti_matching_loss_cls(use_base_loss=False, **common_kwargs)
    except TypeError:
        module = betti_matching_loss_cls(use_base_component=False, **common_kwargs)

    _BETTI_MATCHING_LOSS_CACHE[key] = module
    return module


def compute_binary_soft_cldice_loss(
    logits,
    targets,
    *,
    valid_mask=None,
    mask_mode="pre_skeleton",
    reduction_dims=(1, 2, 3),
    num_iter=10,
    smooth=1.0,
):
    probs, targets, topology_mask = _prepare_binary_topology_inputs(
        logits,
        targets,
        valid_mask=valid_mask,
    )
    mask_mode = str(mask_mode).strip().lower()
    if mask_mode not in {"pre_skeleton", "post_skeleton"}:
        raise ValueError(f"mask_mode must be 'pre_skeleton' or 'post_skeleton', got {mask_mode!r}")

    if topology_mask is not None and mask_mode == "pre_skeleton":
        probs = probs * topology_mask
        targets = targets * topology_mask
        topology_mask = None

    skel_pred = _soft_skeletonize_2d(probs, num_iter=int(num_iter))
    skel_true = _soft_skeletonize_2d(targets, num_iter=int(num_iter))

    probs_eval = probs
    targets_eval = targets
    if topology_mask is not None:
        probs_eval = probs_eval * topology_mask
        targets_eval = targets_eval * topology_mask
        skel_pred = skel_pred * topology_mask
        skel_true = skel_true * topology_mask

    reduce_dims = tuple(int(dim) for dim in reduction_dims)
    smooth = float(smooth)
    tprec = ((skel_pred * targets_eval).sum(dim=reduce_dims) + smooth) / (skel_pred.sum(dim=reduce_dims) + smooth)
    tsens = ((skel_true * probs_eval).sum(dim=reduce_dims) + smooth) / (skel_true.sum(dim=reduce_dims) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def compute_binary_betti_matching_loss(
    logits,
    targets,
    *,
    valid_mask=None,
    filtration_type="superlevel",
    num_processes=1,
):
    probs, targets, topology_mask = _prepare_binary_topology_inputs(
        logits,
        targets,
        valid_mask=valid_mask,
    )
    if topology_mask is not None:
        probs = probs * topology_mask
        targets = targets * topology_mask

    loss_module = _resolve_betti_matching_loss_module(
        filtration_type=filtration_type,
        num_processes=num_processes,
    )
    loss = loss_module(probs, targets)
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss, device=probs.device, dtype=probs.dtype)
    if loss.ndim == 0:
        loss = loss.unsqueeze(0)
    return loss


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
    return compute_region_loss_and_dice(
        logits,
        targets,
        reduction_dims=(1, 2, 3),
        loss_recipe=loss_recipe,
        smooth_factor=smooth_factor,
        soft_label_positive=soft_label_positive,
        soft_label_negative=soft_label_negative,
        eps=eps,
    )
