import numpy as np
from numpy import linalg as LA


from vesuvius.image_proc.helpers import divide_nonzero
from vesuvius.image_proc.features.curvature import hessian_curvature_3d, hessian_curvature_2d

def detect_ridges_2d(image, gamma=1.5, beta=0.5, gauss_sigma=2, sigma=6):
    joint_hessian, zero_mask = hessian_curvature_2d(image, gauss_sigma, sigma)
    eigvals = LA.eigvalsh(joint_hessian, 'U')
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0

    L1 = np.abs(eigvals[:, :, 0])
    L2 = eigvals[:, :, 1]
    L2abs = np.abs(L2)

    S = np.sqrt(np.square(eigvals).sum(axis=-1))
    background_term = 1 - np.exp(-0.5 * np.square(S / gamma))

    Rb = divide_nonzero(L1, L2abs)
    blob_term = np.exp(-0.5 * np.square(Rb / beta))

    ridges = background_term * blob_term
    ridges[L2 > 0] = 0
    return ridges

def detect_ridges_3d(volume, gamma=1.5, beta1=0.5, beta2=0.5, gauss_sigma=2, sigma=6):
    joint_hessian, zero_mask = hessian_curvature_3d(volume, gauss_sigma, sigma)
    eigvals = LA.eigvalsh(joint_hessian, 'U')
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0

    L1 = np.abs(eigvals[:, :, :, 0])
    L2 = np.abs(eigvals[:, :, :, 1])
    L3 = eigvals[:, :, :, 2]
    L3abs = np.abs(L3)

    S = np.sqrt(np.square(eigvals).sum(axis=-1))
    background_term = 1 - np.exp(-0.5 * np.square(S / gamma))

    Ra = divide_nonzero(L2, L3abs)
    planar_term = np.exp(-0.5 * np.square(Ra / beta1))

    Rb = divide_nonzero(L1, np.sqrt(L2 * L3abs))
    blob_term = np.exp(-0.5 * np.square(Rb / beta2))

    ridges = background_term * planar_term * blob_term
    ridges[L3 > 0] = 0
    return ridges
