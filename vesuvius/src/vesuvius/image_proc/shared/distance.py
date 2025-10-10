from scipy.ndimage import distance_transform_edt
import numpy as np

def dilate_by_inverse_edt(binary_volume, dilation_distance):
    eps = 1e-6
    edt = distance_transform_edt(1 - binary_volume)
    inv_edt = 1.0 / (edt + eps)
    threshold = 1.0 / dilation_distance
    dilated = (inv_edt > threshold).astype(np.uint8)
    return dilated