# Taken from a Brett Olsen's contribution to the Vesuvius Challenge and slightly modified

"""Contains various filters and tools for handling papyrus analysis.

TODO:  convert bottleneck code over to cupy for GPU speed up, along with other performance optimization.

Brett Olsen, March 2024
"""

import numpy as np
import math

from tqdm import tqdm
from scipy import ndimage

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import equalize_adapthist

# TODO: some cupy acceleration
xp = np
from numpy import linalg as LA
xndimage = ndimage

def divide_nonzero(array1, array2, eps=1e-10):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = eps
    return np.divide(array1, denominator)

def normalize(volume):
    minim = xp.min(volume)
    maxim = xp.max(volume)
    volume -= minim
    volume /= (maxim - minim)
    return volume

def nlm(volume: np.ndarray, h=0.03):
    sigma = estimate_sigma(volume)
    return denoise_nl_means(volume, patch_size=7, patch_distance=3, sigma=sigma, h=h)

def nms_3d(magnitude, grad, precision):
    """
    Applies Non-Maximum Suppression on a 3D volume using interpolation along gradient directions.

    Parameters:
    - magnitude: 3D numpy array representing the magnitude of gradients.
    - grad: 3D numpy array of shape (3, *magnitude.shape) representing gradient vectors.

    Returns:
    - nms_volume: 3D numpy array after applying NMS.
    """
    # Initialize the output volume
    nms_volume = np.zeros_like(magnitude)

    # Get the shape of the volume
    z_dim, y_dim, x_dim = magnitude.shape
    
    # Create meshgrid of indices
    Z, Y, X = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')

    # Calculate continuous indices for forward and backward positions based on gradients
    forward_indices = np.array([Z, Y, X]) + grad
    backward_indices = np.array([Z, Y, X]) - grad

    # Interpolate the magnitude values at these continuous indices
    forward_values = ndimage.map_coordinates(magnitude, forward_indices, order=1, mode='nearest')
    backward_values = ndimage.map_coordinates(magnitude, backward_indices, order=1, mode='nearest')

    # Apply conditions for NMS using NumPy logical functions
    condition1 = np.logical_and(magnitude >= forward_values, magnitude > backward_values)
    condition2 = np.logical_and(magnitude > forward_values, magnitude >= backward_values)
    mask = np.logical_or(condition1, condition2)

    # Apply mask to set NMS volume
    nms_volume[mask] = magnitude[mask]

    return nms_volume

def ms_3d(magnitude, grad, precision):
    """
    Applies Maximum Suppression on a 3D volume using interpolation along gradient directions.

    Parameters:
    - magnitude: 3D numpy array representing the magnitude of gradients.
    - grad: 3D numpy array of shape (3, *magnitude.shape) representing gradient vectors.

    Returns:
    - nms_volume: 3D numpy array after applying NMS.
    """
    # Initialize the output volume
    nms_volume = np.zeros_like(magnitude)

    # Get the shape of the volume
    z_dim, y_dim, x_dim = magnitude.shape
    
    # Create meshgrid of indices
    Z, Y, X = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')

    # Calculate continuous indices for forward and backward positions based on gradients
    forward_indices = np.array([Z, Y, X]) + grad
    backward_indices = np.array([Z, Y, X]) - grad

    # Interpolate the magnitude values at these continuous indices
    forward_values = ndimage.map_coordinates(magnitude, forward_indices, order=1, mode='nearest')
    backward_values = ndimage.map_coordinates(magnitude, backward_indices, order=1, mode='nearest')

    # Apply conditions for NMS using NumPy logical functions
    condition1 = np.logical_and(magnitude == forward_values, magnitude == backward_values)
    condition2 = np.logical_and(magnitude > forward_values, magnitude > backward_values)
    mask = np.logical_or(condition1, condition2)

    # Apply mask to set NMS volume
    nms_volume[mask] = magnitude[mask]

    return nms_volume

def denoise_3d(volume, h=0.03):
    """Uses a non-local means approach to denoise an input 3D volume.
    """
    precision = volume.dtype
    result = nlm(volume, h=h)
    result = normalize(result)
    result = nlm(np.log(result + np.finfo(precision).tiny), h=h)
    return np.exp(result) - np.finfo(precision).tiny

def adjust_contrast(volume, kernel_size=8):
    return equalize_adapthist(volume, kernel_size, clip_limit=0.01, nbins=256)

def hessian_of_normalized(volume, sigma=6):
    """Derivative half of `hessian`, on an ALREADY smoothed+normalized volume.

    Split out so it can be evaluated on a z-slab: everything here is local (two
    np.gradient passes), unlike the gaussian+normalize step above it, which is not.
    `hessian` keeps its old signature and result, so existing callers are untouched.
    """
    # N.B. this only returns the upper triangular matrix to save time
    joint_hessian = xp.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3, 3), dtype=float)
    
    Dz = xp.gradient(volume, axis=0, edge_order=2)
    joint_hessian[:, :, :, 2, 2] = xp.gradient(Dz, axis=0, edge_order=2)
    del Dz

    Dy = xp.gradient(volume, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 1] = xp.gradient(Dy, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 2] = xp.gradient(Dy, axis=0, edge_order=2)
    #joint_hessian[:, :, :, 2, 1] = joint_hessian[:, :, :, 1, 2]
    del Dy

    Dx = xp.gradient(volume, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 0] = xp.gradient(Dx, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 1] = xp.gradient(Dx, axis=1, edge_order=2)
    #joint_hessian[:, :, :, 1, 0] = joint_hessian[:, :, :, 0, 1]
    joint_hessian[:, :, :, 0, 2] = xp.gradient(Dx, axis=0, edge_order=2)
    #joint_hessian[:, :, :, 2, 0] = joint_hessian[:, :, :, 0, 2]
    del Dx

    joint_hessian = xp.multiply(sigma ** 2, joint_hessian)
    
    #zero_mask = (Dxx + Dyy + Dzz) == 0
    zero_mask = xp.trace(joint_hessian, axis1=3, axis2=4) == 0
    
    return joint_hessian, zero_mask

def hessian(volume, gauss_sigma=2, sigma=6):
    volume = xndimage.gaussian_filter(volume, sigma=gauss_sigma)
    volume = normalize(volume)
    return hessian_of_normalized(volume, sigma)

def detect_ridges(volume, gamma=1.5, beta1=0.5, beta2=0.5, gauss_sigma=2, sigma=6):
    joint_hessian, zero_mask = hessian(volume, gauss_sigma, sigma)
    eigvals = LA.eigvalsh(joint_hessian, "U")
    # Sort in increasing size of the absolute value of the eigenvalues
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0

    L1 = np.abs(eigvals[:, :, :, 0])
    L2 = np.abs(eigvals[:, :, :, 1])
    L3 = eigvals[:, :, :, 2]
    L3abs = np.abs(L3)
    
    S = np.sqrt(np.square(eigvals).sum(axis=-1))
    background_term = 1 - np.exp(-(.5 * np.square(S / gamma)))
    
    Ra = divide_nonzero(L2, L3abs)
    planar_term = np.exp(-(0.5 * np.square(Ra / beta1)))
    
    Rb = divide_nonzero(L1, np.sqrt(np.multiply(L2, L3abs)))
    blob_term = np.exp(-(0.5 * np.square(Rb / beta2)))
    
    ridges = background_term * planar_term * blob_term
    ridges[L3 > 0] = 0
 
    return ridges

def frangi_from_hessian(joint_hessian, zero_mask, gamma=1.5, beta1=0.5, beta2=0.5):
    """Frangi response for an already-built Hessian. Unchanged maths, split out so that
    `detect_vesselness` can apply it one z-slab at a time."""
    eigvals = LA.eigvalsh(joint_hessian, "U")
    # Sort eigenvalues by magnitude (ascending order)
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0  # Ignore zero regions

    # Extract eigenvalues
    L1 = eigvals[:, :, :, 0]
    L2 = eigvals[:, :, :, 1]
    L3 = eigvals[:, :, :, 2]

    # Compute terms for Frangi filter
    Ra = divide_nonzero(np.abs(L2), np.abs(L3))  # Tubularity ratio
    Rb = divide_nonzero(np.abs(L1), np.sqrt(np.abs(L2 * L3)))  # Blobness ratio
    S = np.sqrt(np.square(eigvals).sum(axis=-1))  # Frobenius norm

    # Frangi vesselness components
    planar_term = 1 - np.exp(-0.5 * np.square(Ra / beta1))
    blob_term = np.exp(-0.5 * np.square(Rb / beta2))
    background_term = 1 - np.exp(-0.5 * np.square(S / gamma))

    # Combine terms
    vesselness = background_term * planar_term * blob_term

    # Suppress areas where L2 or L3 are positive (non-tubular regions)
    vesselness[L2 > 0] = 0
    vesselness[L3 > 0] = 0

    return vesselness

def detect_vesselness(volume, gamma=1.5, beta1=0.5, beta2=0.5, gauss_sigma=2, sigma=6,
                      slab=64, halo=16):
    """
    Detect vesselness using the Frangi filter.

    Evaluated in z-slabs. The Hessian is float64 of shape (Z, Y, X, 3, 3) = 72 bytes per
    voxel, so holding it for a whole cube costs 9.7 GB at 512^3 before eigvalsh, the int64
    argsort index and the intermediate terms are added. Slabbing bounds that by the slab
    size; the output is unchanged.

    Parameters:
    - volume: 3D array representing the input volume.
    - gamma: Sensitivity to overall structure strength (controls suppression of background).
    - beta1: Controls sensitivity to tubular structures.
    - beta2: Controls sensitivity to blob-like structures.
    - gauss_sigma: Gaussian smoothing applied to the Hessian matrix.
    - sigma: Scale of differentiation for computing the Hessian.
    - slab: number of z-planes evaluated at a time.
    - halo: z-planes of context kept around each slab and then discarded. Only two operations
      are non-local: gaussian_filter (radius 4*gauss_sigma, scipy truncate=4.0) and the two
      np.gradient passes (2 planes each at a boundary), so the default covers them exactly
      rather than approximately, and the result is bit-for-bit the unslabbed one.

    Returns:
    - vesselness: 3D array representing vesselness probability at each voxel.
    """
    # Both of these are GLOBAL: normalize() takes min/max over the whole cube, so it has to
    # run once, here, before any slabbing. Doing it per slab would rescale each slab by its
    # own extrema and silently change the output.
    volume = xndimage.gaussian_filter(volume, sigma=gauss_sigma)
    volume = normalize(volume)

    n_z = volume.shape[0]
    vesselness = xp.empty(volume.shape, dtype=float)
    for z0 in range(0, n_z, slab):
        z1 = min(z0 + slab, n_z)
        a, b = max(0, z0 - halo), min(n_z, z1 + halo)
        joint_hessian, zero_mask = hessian_of_normalized(volume[a:b], sigma)
        chunk = frangi_from_hessian(joint_hessian, zero_mask, gamma, beta1, beta2)
        del joint_hessian, zero_mask
        vesselness[z0:z1] = chunk[z0 - a: z0 - a + (z1 - z0)]
        del chunk

    return vesselness

def proximity_boolean_filter(volume):
    # Define the 3x3x3 kernel
    kernel = np.ones((3, 3, 3)) * -1/26  # Each neighbor contributes equally when it is zero
    kernel[1, 1, 1] = 1        
    """Detect edges where the central voxel is 1 and at least three neighbors are 0."""
    # Apply the convolution
    filtered = ndimage.convolve(volume, kernel, mode='constant', cval=1)  # Assume boundary is 1 to prevent false edges
    # An edge is detected where the convolution result is 1 - 3*(-1/26) or less (i.e., 1 + 3/26)
    # We use a threshold of slightly more than three zeros (since 3/26 subtracted from 1)
    edges = filtered <= (1 - 3 * (1/26))
    return edges

def detect_edges(volume, filter):
    precision = volume.dtype
    # Define the 3D Scharr kernels for x, y, and z directions
    # Scharr operator values for derivative approximation and smoothing
    if filter == "scharr":
        scharr_1d = np.array([-1, 0, 1], dtype=precision)  # Derivative approximation
        scharr_1d_smooth = np.array([3, 10, 3], dtype=precision)  # Smoothing

        # Create 3D kernels by outer products and normalization
        kz = np.outer(np.outer(scharr_1d, scharr_1d_smooth), scharr_1d_smooth).reshape(3, 3, 3) / 32
        ky = np.outer(np.outer(scharr_1d_smooth, scharr_1d), scharr_1d_smooth).reshape(3, 3, 3) / 32
        kx = np.outer(scharr_1d_smooth, np.outer(scharr_1d_smooth, scharr_1d)).reshape(3, 3, 3) / 32
    elif filter == "pavel":
        pavel_1d = np.array([2,1,-16,-27,0,27,16,-1,-2], dtype=precision)  # Derivative approximation
        pavel_1d_smooth = np.array([1, 4, 6, 4, 1], dtype=precision)  # Smoothing
        pavel_1d_2nd = np.array([-7,12,52,-12,-90,-12,52,12,-7], dtype=precision)
        # Create 3D kernels by outer products and normalization
        kz = np.outer(np.outer(pavel_1d, pavel_1d_smooth), pavel_1d_smooth).reshape(9, 5, 5)/ (96*16*16)
        ky = np.outer(np.outer(pavel_1d_smooth, pavel_1d), pavel_1d_smooth).reshape(5, 9, 5)/ (96*16*16)
        kx = np.outer(pavel_1d_smooth, np.outer(pavel_1d_smooth, pavel_1d)).reshape(5, 5, 9)/ (96*16*16)
        kzz = np.outer(np.outer(pavel_1d_2nd, pavel_1d_smooth), pavel_1d_smooth).reshape(9, 5, 5)/ (192*16*16)
        kyy = np.outer(np.outer(pavel_1d_smooth, pavel_1d_2nd), pavel_1d_smooth).reshape(5, 9, 5)/ (192*16*16)
        kxx = np.outer(pavel_1d_smooth, np.outer(pavel_1d_smooth, pavel_1d_2nd)).reshape(5, 5, 9)/ (192*16*16)

    gradient = np.zeros((3, volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)
    # Apply the kernels to the volume
    gradient[2] = ndimage.convolve(volume, kx)
    gradient[1] = ndimage.convolve(volume, ky)
    gradient[0] = ndimage.convolve(volume, kz)
    
    first_derivative = np.sqrt(gradient[2]**2 + gradient[1]**2 + gradient[0]**2)
    gradient /= first_derivative
    
    nms = nms_3d(first_derivative, gradient, precision)

    #normalization
    first_derivative = nms / nms.max()
    

    hessian = np.zeros((3,3,volume.shape[0], volume.shape[1], volume.shape[2]), dtype=precision)

    if filter == "scharr":
        hessian[2,2] = ndimage.convolve(gradient[2], kx).astype(precision)
        hessian[1,1] = ndimage.convolve(gradient[1], ky).astype(precision)
        hessian[0,0] = ndimage.convolve(gradient[0], kz).astype(precision)

    elif filter == "pavel":
        hessian[2,2] = ndimage.convolve(volume, kxx).astype(precision)
        hessian[1,1] = ndimage.convolve(volume, kyy).astype(precision)
        hessian[0,0] = ndimage.convolve(volume, kzz).astype(precision)
        
    hessian[1,2] = ndimage.convolve(gradient[2], ky).astype(precision)
    hessian[0,2] = ndimage.convolve(gradient[2], kz).astype(precision)

    #print('Calculating Hessian 2')
    hessian[2,1] = ndimage.convolve(gradient[1], kx).astype(precision)
    hessian[0,1] = ndimage.convolve(gradient[1], kz).astype(precision)

    #print('Calculating Hessian 3')
    hessian[2,0] = ndimage.convolve(gradient[0], kx).astype(precision)
    hessian[1,0] = ndimage.convolve(gradient[0], ky).astype(precision)

    #print('Calculating Determinant')
    det = np.abs(hessian[0,0]*(hessian[1,1]*hessian[2,2]-hessian[1,2]*hessian[2,1])-hessian[0,1]*(hessian[1,0]*hessian[2,2]-hessian[1,2]*hessian[2,0])+hessian[0,2]*(hessian[1,0]*hessian[2,1]-hessian[1,1]*hessian[2,0]))

    
    return first_derivative, det, gradient