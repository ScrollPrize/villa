from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_cdt, distance_transform_edt

from ink.recipes.components import label_components


def as_bool_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
    return array.astype(bool, copy=False)


def normalize_skeleton_method(method: str) -> str:
    method_name = str(method).strip().lower()
    if method_name not in {"guo_hall", "zhang_suen"}:
        raise ValueError(f"unsupported skeleton method: {method!r}")
    return method_name


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


def skeletonize_binary(mask: np.ndarray, *, method: str = "guo_hall") -> np.ndarray:
    import cv2

    method_name = normalize_skeleton_method(method)
    skeleton = as_bool_2d(mask).copy()
    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=bool)
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
        raise ImportError(
            "cv2.ximgproc.thinning is required for stitched PFMWeighted; "
            "install opencv-contrib-python-headless."
        )
    thinning_type = (
        int(cv2.ximgproc.THINNING_GUOHALL)
        if method_name == "guo_hall"
        else int(cv2.ximgproc.THINNING_ZHANGSUEN)
    )
    skeleton_u8 = cv2.ximgproc.thinning(
        (skeleton.astype(np.uint8, copy=False) * 255),
        thinningType=thinning_type,
    )
    return skeleton_u8 > 0


def _pseudo_contour_mask(gt: np.ndarray) -> np.ndarray:
    import cv2

    gt = as_bool_2d(gt)
    if not gt.any():
        return np.zeros_like(gt, dtype=bool)
    gt_u8 = gt.astype(np.uint8, copy=False) * 255
    blurred = cv2.GaussianBlur(gt_u8, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_REPLICATE)
    canny = cv2.Canny(blurred, threshold1=int(round(0.2 * 255.0)), threshold2=int(round(0.3 * 255.0)))
    contour = np.logical_and(canny > 0, gt)
    if not contour.any():
        raise ValueError("weighted pseudo-FMeasure contour extraction produced an empty contour for non-empty GT")
    return contour


def _pseudo_recall_normalizer(c_i: int) -> float:
    c_i = max(1, int(c_i))
    if (c_i % 2) == 1:
        value = ((float(c_i) - 1.0) / 2.0) ** 2
    else:
        half = float(c_i) / 2.0
        value = half * (half - 1.0)
    return float(max(1.0, value))


def pseudo_weight_maps(
    gt: np.ndarray,
    *,
    connectivity: int,
    skel_gt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gt = as_bool_2d(gt)
    gw = np.zeros(gt.shape, dtype=np.float64)
    pw = np.ones(gt.shape, dtype=np.float64)
    if not gt.any():
        return gw, pw

    gt_labels, n_gt = label_components(gt, connectivity=connectivity)
    if n_gt <= 0:
        return gw, pw

    contour = _pseudo_contour_mask(gt)
    d_cm = distance_transform_cdt((~contour).astype(np.uint8), metric="chessboard").astype(np.float64)

    contour_labels = np.where(contour, gt_labels, 0).astype(np.int32, copy=False)
    _d_edt, nearest_contour_indices = distance_transform_edt(
        (~contour).astype(np.uint8),
        return_indices=True,
    )
    nearest_contour_label = contour_labels[nearest_contour_indices[0], nearest_contour_indices[1]]

    c_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)
    n_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)
    sw_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)

    if skel_gt is None:
        skel_all = skeletonize_binary(gt)
    else:
        skel_all = as_bool_2d(skel_gt)
        if skel_all.shape != gt.shape:
            raise ValueError(f"skel_gt/gt shape mismatch: {tuple(skel_all.shape)} vs {tuple(gt.shape)}")
        if np.logical_and(skel_all, ~gt).any():
            raise ValueError("skel_gt must be a subset of gt foreground")

    for label_id in range(1, int(n_gt) + 1):
        component = gt_labels == label_id
        if not component.any():
            continue
        skel_component = np.logical_and(skel_all, component)
        sw_samples = d_cm[skel_component]
        if sw_samples.size == 0:
            sw_samples = d_cm[component]
        if sw_samples.size == 0:
            continue
        c_i = max(1, int(math.ceil(float(sw_samples.max()))))
        c_by_label[label_id] = float(c_i)
        n_by_label[label_id] = _pseudo_recall_normalizer(c_i)
        sw_by_label[label_id] = max(1.0, float(2.0 * sw_samples.mean()))

    inside = gt
    d_in = d_cm[inside]
    labels_in = gt_labels[inside]
    c_in = c_by_label[labels_in]
    n_in = n_by_label[labels_in]

    weights_in = np.empty_like(d_in, dtype=np.float64)
    on_contour = d_in <= 0.0
    in_core = np.logical_and(d_in > 0.0, d_in < c_in)
    in_plateau = np.logical_not(np.logical_or(on_contour, in_core))

    weights_in[on_contour] = 1.0
    weights_in[in_core] = d_in[in_core] / n_in[in_core]
    weights_in[in_plateau] = c_in[in_plateau] / n_in[in_plateau]
    gw[inside] = weights_in

    outside = np.logical_not(gt)
    sw_map = sw_by_label[nearest_contour_label]
    near = np.logical_and(outside, np.logical_and(nearest_contour_label > 0, d_cm <= sw_map))
    if near.any():
        pw[near] = 1.0 + (d_cm[near] / np.maximum(1.0, sw_map[near]))
        pw[near] = np.minimum(2.0, pw[near])

    return gw, pw


__all__ = [
    "as_bool_2d",
    "binary_mask_to_signed_distance_map",
    "normalize_skeleton_method",
    "pseudo_weight_maps",
    "skeletonize_binary",
]
