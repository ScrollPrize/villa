"""Point-based transform fits, from `transform_utils.py` without its volume and SimpleITK imports."""

import numpy as np
from scipy.optimize import least_squares


def fit_affine_transform_from_points(fixed_points, moving_points):
    """Fit an affine transform from corresponding point pairs.

    Args:
        fixed_points: List of points in fixed space (Nx3)
        moving_points: List of points in moving space (Nx3)

    Returns:
        4x4 affine transformation matrix or None if insufficient points
    """
    if len(fixed_points) != len(moving_points) or len(fixed_points) < 4:
        return None

    # Convert to numpy arrays
    fixed_array = np.array(fixed_points)  # Points in fixed space
    moving_array = np.array(moving_points)  # Points in moving space

    # Add homogeneous coordinate to moving points
    moving_homogeneous = np.column_stack([moving_array, np.ones(len(moving_array))])

    # Solve for transform: fixed = transform @ moving_homogeneous.T
    # We want: transform @ moving_homogeneous.T = fixed_array.T
    # So: transform = fixed_array.T @ pinv(moving_homogeneous.T)
    transform_3x4 = fixed_array.T @ np.linalg.pinv(moving_homogeneous.T)

    # Add bottom row to make it 4x4
    transform_4x4 = np.vstack([transform_3x4, [0, 0, 0, 1]])

    return transform_4x4


def fit_constrained_transform_from_points(fixed_points, moving_points):
    """Fit a constrained affine transform from corresponding point pairs.

    Fits 5 continuous parameters + 3 discrete sign choices:
    - Isotropic scale s
    - Rotation around z-axis (index 0 in ZYX) by angle theta
    - Translation tx, ty, tz
    - Independent axis flips: fz, fy, fx each ±1 (tries all 8 combinations)

    The transform applies flip, then rotation, then scale, then translation.
    The resulting 4x4 matrix in ZYX order is:
        [fz*s    0               0              tz]
        [0       fy*s*cos(θ)    -fx*s*sin(θ)    ty]
        [0       fy*s*sin(θ)     fx*s*cos(θ)    tx]
        [0       0               0               1]

    Args:
        fixed_points: List of points in fixed space (Nx3, ZYX order)
        moving_points: List of points in moving space (Nx3, ZYX order)

    Returns:
        4x4 affine transformation matrix or None if insufficient points
    """
    if len(fixed_points) != len(moving_points) or len(fixed_points) < 3:
        return None

    fixed_array = np.array(fixed_points)
    moving_array = np.array(moving_points)

    def _build_matrix(params, flips):
        s, theta, tx, ty, tz = params
        fz, fy, fx = flips
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            [fz * s, 0,              0,             tz],
            [0,      fy * s * cos_t, -fx * s * sin_t, ty],
            [0,      fy * s * sin_t,  fx * s * cos_t, tx],
            [0,      0,              0,              1],
        ])

    def _residuals(params, flips):
        mat = _build_matrix(params, flips)
        moving_h = np.column_stack([moving_array, np.ones(len(moving_array))])
        predicted = (mat @ moving_h.T).T[:, :3]
        return (predicted - fixed_array).ravel()

    # Initial scale estimate
    fixed_spread = np.linalg.norm(fixed_array - fixed_array.mean(axis=0), axis=1).mean()
    moving_spread = np.linalg.norm(moving_array - moving_array.mean(axis=0), axis=1).mean()
    s0 = fixed_spread / moving_spread if moving_spread > 0 else 1.0

    fixed_centroid = fixed_array.mean(axis=0)  # [z, y, x]
    moving_centroid = moving_array.mean(axis=0)

    best_result = None
    best_cost = np.inf
    best_flips = (1, 1, 1)

    for fz in [1, -1]:
        for fy in [1, -1]:
            for fx in [1, -1]:
                flips = (fz, fy, fx)
                # Initial translation: t = fixed_centroid - diag(fz,fy,fx)*s0 * moving_centroid
                tz0 = fixed_centroid[0] - fz * s0 * moving_centroid[0]
                ty0 = fixed_centroid[1] - fy * s0 * moving_centroid[1]
                tx0 = fixed_centroid[2] - fx * s0 * moving_centroid[2]
                init = [s0, 0.0, tx0, ty0, tz0]
                result = least_squares(_residuals, init, args=(flips,))
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    best_flips = flips

    return _build_matrix(best_result.x, best_flips)


