"""
Module: critical_components
Computes the numbers of positive and negative critical components for 3D images.
This version implements a multiclass GPU‐based detection.
"""

from typing import Dict, Any
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation, label as cp_label

# ---------------------------
# GPU Kernels and Multi-Class Detection
# ---------------------------

multi_class_false_negative_kernel = cp.ElementwiseKernel(
    'T target, T pred',
    'int8 out',
    '''
    out = ((target == (T)1) && (pred == (T)0)) ? 1 : 0;
    ''',
    'multi_class_false_negative_kernel'
)

multi_class_false_positive_kernel = cp.ElementwiseKernel(
    'T target, T pred',
    'int8 out',
    '''
    out = ((pred == (T)1) && (target == (T)0)) ? 1 : 0;
    ''',
    'multi_class_false_positive_kernel'
)

fused_external_boundary_kernel = cp.ElementwiseKernel(
    'bool dilated, bool region_mask, int8 mistake, T volume, T root_val',
    'bool out',
    '''
    bool boundary = dilated && (!region_mask);
    out = boundary && (mistake == 0) && (volume == root_val);
    ''',
    'fused_external_boundary_kernel'
)

def detect_critical_multi_class_gpu(y_target, y_pred):
    """
    Detect critical regions for each non-background class concurrently.
    Expects binary volumes per class with shape (num_classes, H, W, D).
    
    Returns:
       crit_masks_fn : cp.ndarray with shape (H, W, D)
         (For each channel, a mask of critical false negatives.)
       n_crit_fn   : list of int
         Number of critical false negative regions per non-background class.
       crit_masks_fp : cp.ndarray with shape (H, W, D)
         (For each channel, a mask of critical false positives.)
       n_crit_fp   : list of int
         Number of critical false positive regions per non-background class.
    """
    num_classes = y_target.shape[0]
    fn_masks = multi_class_false_negative_kernel(y_target, y_pred)
    fp_masks = multi_class_false_positive_kernel(y_target, y_pred)
    
    structure = cp.ones((3, 3, 3), dtype=cp.int8)
    dilation_structure = cp.ones((3, 3, 3), dtype=cp.bool_)
    
    def process_channel(target_c, mistakes):
        vol_minus_mistakes, _ = cp_label(target_c * (1 - mistakes), structure=structure)
        mistake_labels, _ = cp_label(mistakes, structure=structure)
        crit_mask = cp.zeros(target_c.shape, dtype=cp.bool_)
        n_regions = 0
        unique_ids = cp.unique(mistake_labels)
        for rid in unique_ids:
            if rid.item() == 0:
                continue
            region_mask = (mistake_labels == rid)
            indices = cp.argwhere(region_mask)
            if indices.shape[0] == 0:
                continue
            root_idx = tuple(int(x.item()) for x in indices[0])
            root_val = target_c[root_idx]
            dilated = binary_dilation(region_mask, structure=dilation_structure)
            external_boundary = fused_external_boundary_kernel(
                dilated, region_mask, mistakes, target_c, root_val
            )
            if cp.any(external_boundary):
                unique_vals = cp.unique(vol_minus_mistakes[external_boundary])
                is_critical = (unique_vals.size != 1)
            else:
                is_critical = True
            if is_critical:
                cp.logical_or(crit_mask, region_mask, out=crit_mask)
                n_regions += 1
        return crit_mask, n_regions

    crit_masks_fn_list = []
    n_crit_fn = []
    crit_masks_fp_list = []
    n_crit_fp = []
    
    # Loop over non-background classes (assume channel 0 is background).
    for c in range(1, num_classes):
        target_c = y_target[c]
        mistakes_fn = fn_masks[c]
        crit_mask_fn, n_fn = process_channel(target_c, mistakes_fn)
        crit_masks_fn_list.append(crit_mask_fn)
        n_crit_fn.append(n_fn)
        
        pred_c = y_pred[c]
        mistakes_fp = fp_masks[c]
        crit_mask_fp, n_fp = process_channel(pred_c, mistakes_fp)
        crit_masks_fp_list.append(crit_mask_fp)
        n_crit_fp.append(n_fp)
    
    crit_masks_fn = cp.stack(crit_masks_fn_list, axis=0).sum(axis=0)
    crit_masks_fp = cp.stack(crit_masks_fp_list, axis=0).sum(axis=0)
    return crit_masks_fn, n_crit_fn, crit_masks_fp, n_crit_fp

# ---------------------------
# Multiclass Compute Function
# ---------------------------
def compute(label: np.ndarray, prediction: np.ndarray, **hyperparams: Any) -> Dict[str, Any]:
    """
    Computes per-class critical components for multiclass 3D segmentation.
    
    Positive critical components (false negatives) and negative critical components
    (false positives) are both computed in a single call.
    
    Parameters
    ----------
    label : np.ndarray
        Groundtruth segmentation with integer labels.
    prediction : np.ndarray
        Predicted segmentation with integer labels.
    hyperparams : Any
        Should include "num_classes" (background is assumed to be 0).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
            - "critical_components_positive": dict mapping class index to count.
            - "critical_components_negative": dict mapping class index to count.
    """
    # Determine the number of classes (default: inferred from max label).
    num_classes = hyperparams.get("num_classes", int(np.max(label)) + 1)
    
    # Convert the label maps into one-hot (per-class binary) volumes.
    target_bin = np.stack([(label == c).astype(np.float32) for c in range(num_classes)], axis=0)
    pred_bin = np.stack([(prediction == c).astype(np.float32) for c in range(num_classes)], axis=0)
    
    # Transfer to GPU using half precision.
    target_cp = cp.array(target_bin).astype(cp.float16)
    pred_cp = cp.array(pred_bin).astype(cp.float16)
    
    # Call the multi-class detection once. It returns both false negatives and false positives.
    _, n_crit_fn, _, n_crit_fp = detect_critical_multi_class_gpu(target_cp, pred_cp)
    
    # For non-background classes (assume class 0 is background), assign counts.
    positive = np.asarray([float(n_crit_fn[c - 1]) for c in range(1, num_classes)]).sum()
    negative = np.asarray([float(n_crit_fp[c - 1]) for c in range(1, num_classes)]).sum()
    
    return {
        "critical_components_positive": positive,
        "critical_components_negative": negative
    }