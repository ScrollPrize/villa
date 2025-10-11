import numpy as np
from scipy.ndimage.filters import gaussian_filter
from vesuvius.image_proc.intensity.normalization import normalize_minmax


def hessian_curvature_2d(image, gauss_sigma=2, sigma=6):
    image_smoothed = gaussian_filter(image, sigma=gauss_sigma)
    image_smoothed = normalize_minmax(image_smoothed)

    joint_hessian = np.zeros((image.shape[0], image.shape[1], 2, 2), dtype=float)

    Dy = np.gradient(image_smoothed, axis=0, edge_order=2)
    joint_hessian[:, :, 1, 1] = np.gradient(Dy, axis=0, edge_order=2)
    joint_hessian[:, :, 0, 1] = np.gradient(Dy, axis=1, edge_order=2)
    del Dy

    Dx = np.gradient(image_smoothed, axis=1, edge_order=2)
    joint_hessian[:, :, 0, 0] = np.gradient(Dx, axis=1, edge_order=2)
    joint_hessian[:, :, 1, 0] = joint_hessian[:, :, 0, 1]
    del Dx

    joint_hessian = joint_hessian * (sigma ** 2)
    zero_mask = np.trace(joint_hessian, axis1=2, axis2=3) == 0
    return joint_hessian, zero_mask


def hessian_curvature_3d(volume, gauss_sigma=2, sigma=6):
    volume_smoothed = gaussian_filter(volume, sigma=gauss_sigma)
    volume_smoothed = normalize_minmax(volume_smoothed)

    joint_hessian = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3, 3), dtype=float)

    Dz = np.gradient(volume_smoothed, axis=0, edge_order=2)
    joint_hessian[:, :, :, 2, 2] = np.gradient(Dz, axis=0, edge_order=2)
    del Dz

    Dy = np.gradient(volume_smoothed, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 1] = np.gradient(Dy, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 2] = np.gradient(Dy, axis=0, edge_order=2)
    del Dy

    Dx = np.gradient(volume_smoothed, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 0] = np.gradient(Dx, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 1] = np.gradient(Dx, axis=1, edge_order=2)
    joint_hessian[:, :, :, 0, 2] = np.gradient(Dx, axis=0, edge_order=2)
    del Dx

    joint_hessian = joint_hessian * (sigma ** 2)
    zero_mask = np.trace(joint_hessian, axis1=3, axis2=4) == 0
    return joint_hessian, zero_mask
