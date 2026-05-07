from __future__ import annotations

from typing import Any, Callable

import torch


def augment_triplet_payload(
    *,
    augmentations,
    crop_size,
    vol_crop: torch.Tensor,
    cond_seg_gt: torch.Tensor,
    behind_seg: torch.Tensor,
    front_seg: torch.Tensor,
    cond_surface_local,
    cond_surface_keypoints: torch.Tensor | None,
    cond_surface_shape,
    restore_cond_surface_fn: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    """Apply triplet-mode augmentations and unpack outputs."""
    if augmentations is None:
        return {
            "vol_crop": vol_crop,
            "cond_seg_gt": cond_seg_gt,
            "behind_seg": behind_seg,
            "front_seg": front_seg,
            "cond_surface_local": cond_surface_local,
        }

    aug_kwargs = {
        "image": vol_crop[None],
        "segmentation": torch.stack([cond_seg_gt, behind_seg, front_seg], dim=0),
        "crop_shape": crop_size,
    }
    if cond_surface_keypoints is not None:
        aug_kwargs["keypoints"] = cond_surface_keypoints

    augmented = augmentations(**aug_kwargs)
    vol_crop = augmented["image"].squeeze(0)
    cond_seg_gt = augmented["segmentation"][0]
    behind_seg = augmented["segmentation"][1]
    front_seg = augmented["segmentation"][2]

    if cond_surface_keypoints is not None:
        cond_surface_local = restore_cond_surface_fn(
            augmented=augmented,
            cond_surface_keypoints=cond_surface_keypoints,
            cond_surface_shape=cond_surface_shape,
            mode="triplet",
        )

    return {
        "vol_crop": vol_crop,
        "cond_seg_gt": cond_seg_gt,
        "behind_seg": behind_seg,
        "front_seg": front_seg,
        "cond_surface_local": cond_surface_local,
    }


def augment_split_payload(
    *,
    augmentations,
    crop_size,
    vol_crop: torch.Tensor,
    masked_seg: torch.Tensor,
    other_wraps_tensor: torch.Tensor,
    cond_seg_gt: torch.Tensor,
    cond_surface_local,
    cond_surface_keypoints: torch.Tensor | None,
    cond_surface_shape,
    restore_cond_surface_fn: Callable[..., torch.Tensor],
    masked_surface_local=None,
    masked_surface_keypoints: torch.Tensor | None = None,
    masked_surface_shape=None,
    neighbor_seg_tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Apply split-mode augmentations and unpack outputs."""
    if augmentations is None:
        return {
            "vol_crop": vol_crop,
            "masked_seg": masked_seg,
            "other_wraps_tensor": other_wraps_tensor,
            "cond_seg_gt": cond_seg_gt,
            "cond_surface_local": cond_surface_local,
            "masked_surface_local": masked_surface_local,
            "neighbor_seg_tensor": neighbor_seg_tensor,
        }

    seg_tensors = [masked_seg, other_wraps_tensor, cond_seg_gt]
    seg_keys = ["masked_seg", "other_wraps_tensor", "cond_seg_gt"]
    if neighbor_seg_tensor is not None:
        seg_tensors.append(neighbor_seg_tensor)
        seg_keys.append("neighbor_seg_tensor")

    aug_kwargs = {
        "image": vol_crop[None],
        "segmentation": torch.stack(seg_tensors, dim=0),
        "crop_shape": crop_size,
    }
    keypoint_parts = []
    cond_keypoint_count = 0
    if cond_surface_keypoints is not None:
        keypoint_parts.append(cond_surface_keypoints)
        cond_keypoint_count = int(cond_surface_keypoints.shape[0])
    if masked_surface_keypoints is not None:
        keypoint_parts.append(masked_surface_keypoints)
    if keypoint_parts:
        aug_kwargs["keypoints"] = torch.cat(keypoint_parts, dim=0)

    augmented = augmentations(**aug_kwargs)
    vol_crop = augmented["image"].squeeze(0)

    unpacked = {}
    for i, key in enumerate(seg_keys):
        unpacked[key] = augmented["segmentation"][i]

    if keypoint_parts:
        augmented_keypoints = augmented.get("keypoints")
        if augmented_keypoints is None:
            raise RuntimeError("split augmentation did not return keypoints")

    if cond_surface_keypoints is not None:
        cond_augmented = dict(augmented)
        cond_augmented["keypoints"] = augmented_keypoints[:cond_keypoint_count]
        cond_surface_local = restore_cond_surface_fn(
            augmented=cond_augmented,
            cond_surface_keypoints=cond_surface_keypoints,
            cond_surface_shape=cond_surface_shape,
            mode="split",
        )
    if masked_surface_keypoints is not None:
        masked_augmented = dict(augmented)
        masked_augmented["keypoints"] = augmented_keypoints[cond_keypoint_count:]
        masked_surface_local = restore_cond_surface_fn(
            augmented=masked_augmented,
            cond_surface_keypoints=masked_surface_keypoints,
            cond_surface_shape=masked_surface_shape,
            mode="split",
        )

    return {
        "vol_crop": vol_crop,
        "masked_seg": unpacked["masked_seg"],
        "other_wraps_tensor": unpacked["other_wraps_tensor"],
        "cond_seg_gt": unpacked["cond_seg_gt"],
        "cond_surface_local": cond_surface_local,
        "masked_surface_local": masked_surface_local,
        "neighbor_seg_tensor": unpacked.get("neighbor_seg_tensor", neighbor_seg_tensor),
    }
