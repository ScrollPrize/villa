import numpy as np
from typing import Dict
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def compute(label: np.ndarray, prediction: np.ndarray, tolerance_radius: float = 3.0, 
            method: str = 'lee', eps: float = 1e-8, surface: bool = False) -> Dict[str, float]:
    """
    Computes the CenterLine or CenterSurface Dice coefficient between label and prediction.
    
    This metric measures how well centerlines of segmented structures overlap, which is
    particularly useful for thin structures like blood vessels, fibers, etc.
    
    The implementation:
    1. Extracts centerlines (skeletons) from both label and prediction using 2D skeletonization
       applied to each slice independently
    2. Creates a tolerance region around the ground truth skeleton
    3. Calculates overlap between prediction skeleton and ground truth tolerance region
    
    Args:
        label (np.ndarray): Ground truth mask as a numpy array
        prediction (np.ndarray): Predicted mask as a numpy array
        tolerance_radius (float): Maximum allowed distance between centerlines in pixels
        method (str): Skeletonization method ('lee' or 'zhang')
        eps (float): Small value to avoid division by zero
        
    Returns:
        Dict[str, float]: Dictionary containing centerline_dice and related metrics
    """
    # Ensure inputs are numpy arrays and boolean
    label = np.asarray(label).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    
    # Initialize empty skeletons
    label_skeleton = np.zeros_like(label)
    prediction_skeleton = np.zeros_like(prediction)
    
    # Apply 2D skeletonization to each slice independently
    if surface is False:
        for z in range(label.shape[0]):
            if np.any(label[z]):  # Only process non-empty slices
                label_skeleton[z] = skeletonize(label[z], method=method)
            
            if np.any(prediction[z]):  # Only process non-empty slices
                prediction_skeleton[z] = skeletonize(prediction[z], method=method)
    else:
        # Surface Skeletonize Label
        if not np.sum(label) == 0:
            label_skeleton = skeletonize(label, surface=True)

            # initializing different axes
            label_skel_x = np.zeros_like(label_skeleton)
            label_skel_y = np.zeros_like(label_skeleton)
            label_skel_z = np.zeros_like(label_skeleton)

            for z in range(label_skeleton.shape[0]):
                if np.any(label[z]): 
                    label_skel_z[z] = skeletonize(label[z], surface=False)
            for y in range(label_skeleton.shape[1]):
                if np.any(label[:,y,:]): 
                    label_skel_y[:,y,:] = skeletonize(label[:,y,:], surface=False)
            for x in range(label_skeleton.shape[2]):
                if np.any(label[:,:,x]): 
                    label_skel_x[:,:,x] = skeletonize(label[:,:,x], surface=False)

            np.logical_or(label_skeleton, label_skel_z, out=label_skeleton)
            np.logical_or(label_skeleton, label_skel_y, out=label_skeleton)
            np.logical_or(label_skeleton, label_skel_x, out=label_skeleton)

        # Surface Skeletonize Prediction
        if not np.sum(prediction) == 0:
            prediction_skeleton = skeletonize(prediction, surface=True)

            # initializing different axes
            pred_skel_x = np.zeros_like(prediction_skeleton)
            pred_skel_y = np.zeros_like(prediction_skeleton)
            pred_skel_z = np.zeros_like(prediction_skeleton)

            for z in range(prediction_skeleton.shape[0]):
                if np.any(prediction[z]): 
                    pred_skel_z[z] = skeletonize(prediction[z], surface=False)
            for y in range(prediction_skeleton.shape[1]):
                if np.any(prediction[:,y,:]): 
                    pred_skel_y[:,y,:] = skeletonize(prediction[:,y,:], surface=False)
            for x in range(prediction_skeleton.shape[2]):
                if np.any(prediction[:,:,x]): 
                    pred_skel_x[:,:,x] = skeletonize(prediction[:,:,x], surface=False)

            np.logical_or(prediction_skeleton, pred_skel_z, out=prediction_skeleton)
            np.logical_or(prediction_skeleton, pred_skel_y, out=prediction_skeleton)
            np.logical_or(prediction_skeleton, pred_skel_x, out=prediction_skeleton)

    # Create tolerance region around ground truth skeleton
    # Distance transform gives distance to nearest foreground pixel
    dist_transform = np.zeros_like(label, dtype=float)
    for z in range(label.shape[0]):
        if np.any(label_skeleton[z]):
            # Get distance to nearest skeleton pixel
            dt = distance_transform_edt(~label_skeleton[z])
            dist_transform[z] = dt
    
    # Create tolerance mask (tube around ground truth skeleton)
    tolerance_mask = dist_transform <= tolerance_radius
    
    # Calculate centerline dice
    # Numerator: prediction skeleton pixels within tolerance of ground truth skeleton
    numerator = np.sum(prediction_skeleton & tolerance_mask)
    
    # Denominator: sum of both skeleton lengths
    denominator = np.sum(label_skeleton) + np.sum(prediction_skeleton)
    
    # Calculate the centerline dice
    centerline_dice = (2.0 * numerator) / (denominator + eps)
    
    # Additional useful metrics
    gt_skeleton_length = np.sum(label_skeleton)
    pred_skeleton_length = np.sum(prediction_skeleton)
    
    # True positive rate (sensitivity): what fraction of GT skeleton is captured
    sensitivity = numerator / (np.sum(label_skeleton) + eps)
    
    # Precision: what fraction of predicted skeleton is correct
    precision = numerator / (np.sum(prediction_skeleton) + eps)
    
    if surface:
        suffix = "surface"
    else:
        suffix = "line"
    return {
        f"center{suffix}_dice": float(centerline_dice),
        f"center{suffix}_length_gt": float(gt_skeleton_length),
        f"center{suffix}_length_pred": float(pred_skeleton_length),
        f"center{suffix}_sensitivity": float(sensitivity),
        f"center{suffix}_precision": float(precision)
    }