from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


@dataclass(frozen=True)
class StitchCLDiceLoss:
    weight: float = 1.0
    mask_mode: str = "pre_skeleton"

    def compute(self, stitched_logits, stitched_targets, *, valid_mask, **_kwargs):
        cldice_loss = compute_binary_soft_cldice_loss(
            stitched_logits[None, None],
            stitched_targets[None, None],
            valid_mask=valid_mask[None, None],
            mask_mode=self.mask_mode,
            reduction_dims=(1, 2, 3),
        )[0]
        return {
            "loss": float(self.weight) * cldice_loss,
            "cldice_loss": cldice_loss,
        }
