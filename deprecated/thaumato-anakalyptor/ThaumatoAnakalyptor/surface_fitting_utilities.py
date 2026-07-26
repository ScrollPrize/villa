### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from sklearn.linear_model import Ridge
import open3d as o3d
import os


def load_ply(filename):
    """
    Load point cloud data from a .ply file.
    """
    # Check that the file exists
    assert os.path.isfile(filename), f"File {filename} not found."

    # Load the file and extract the points and normals
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # Extract additional features
    colors = np.asarray(pcd.colors)

    return points, normals, colors


def rotation_matrix_to_align_z_with_v(v):
    # Normalize v
    v = v / np.linalg.norm(v)
    # z-axis vector
    z = np.array([0, 0, -1])
    # Calculate the rotation axis k
    k = np.cross(z, v)
    norm_k = np.linalg.norm(k)
    if norm_k == 0:
        # v is the same as z
        norm_k = 1.0
    k /= norm_k
    # Calculate the rotation angle theta
    theta = np.arccos(np.dot(z, v))
    # Calculate the skew-symmetric matrix K
    K = np.array([
    [0, -k[2], k[1]],
    [k[2], 0, -k[0]],
    [-k[1], k[0], 0]
    ])
    # Calculate the rotation matrix R using Rodrigues' formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    # Invers of R
    R = R.T
    return R

def rotate_points(points, R):
    return points.dot(R.T)

# Rotate back to initial coordinate system
def rotate_points_invers(points, R):
    return np.dot(points, R)

def fit_surface_to_points_n_regularized(points, R, n, alpha=0.1, slope_alpha=0.1):
    # Step 1: Rotate points to align with v
    rotated_points = rotate_points(points, R)

    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    z = rotated_points[:, 2]

    terms = []
    derivatives_x = []
    derivatives_y = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            terms.append((x ** i) * (y ** j))
            # Compute the derivative contributions for this term
            dx = i * (x ** (i - 1 if i > 0 else 0)) * (y ** j)
            dy = j * (x ** i) * (y ** (j - 1 if j > 0 else 0))
            derivatives_x.append(dx)
            derivatives_y.append(dy)
            
    A = np.vstack(terms).T
    Dx = np.vstack(derivatives_x).T
    Dy = np.vstack(derivatives_y).T

    # Ridge regression with slope regularization
    num_coeff = A.shape[1]
    
    A_reg = np.vstack([
        A,
        np.sqrt(alpha) * np.eye(num_coeff),
        np.sqrt(slope_alpha) * Dx,
        np.sqrt(slope_alpha) * Dy
    ])
    z_reg = np.hstack([z, np.zeros(A_reg.shape[0] - z.shape[0])])

    if np.any(np.isnan(A_reg)) or np.any(np.isinf(A_reg)):
        print("Found NaN or Inf values in the matrix 'A_reg'")
    if np.any(np.isnan(z_reg)) or np.any(np.isinf(z_reg)):
        print("Found NaN or Inf values in the matrix 'z_reg'")
    coeff, _, _, _ = lstsq(A_reg, z_reg)

    return coeff

def fit_surface_to_points_n_regularized_rotated(points, n, R, alpha=0.1):
    # Step 1: Rotate points to align with v
    rotated_points = rotate_points(points, R)

    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    z = rotated_points[:, 2]

    terms = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            terms.append((x ** i) * (y ** j))
    A = np.vstack(terms).T

    # Ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, z)
    
    return ridge.coef_

def fit_surface_to_points_n_regularized_sklearn(points, R, n, alpha=10000.0, slope_alpha=1.0):
    # Step 1: Rotate points to align with v
    rotated_points = rotate_points(points, R)

    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    z = rotated_points[:, 2]

    terms = []
    derivatives_x = []
    derivatives_y = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            terms.append((x ** i) * (y ** j))
            # Compute the derivative contributions for this term
            dx = i * (x ** (i - 1 if i > 0 else 0)) * (y ** j)
            dy = j * (x ** i) * (y ** (j - 1 if j > 0 else 0))
            derivatives_x.append(dx)
            derivatives_y.append(dy)
            
    A = np.vstack(terms).T
    Dx = np.vstack(derivatives_x).T
    Dy = np.vstack(derivatives_y).T

    # Ridge regression with slope regularization
    num_coeff = A.shape[1]
    
    A_reg = np.vstack([
        A,
        np.sqrt(alpha) * np.eye(num_coeff),
        np.sqrt(slope_alpha) * Dx,
        np.sqrt(slope_alpha) * Dy
    ])
    z_reg = np.hstack([z, np.zeros(A_reg.shape[0] - z.shape[0])])

    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(A_reg, z_reg)

    coeff = ridge.coef_

    return coeff[:num_coeff]

# Works in rotated coordinate system
def f_n(x, y, n, coeff):
    idx = 0
    z = np.zeros_like(x)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            z += coeff[idx] * (x ** i) * (y ** j)
            idx += 1
    return z

# Works in original coordinate system
def f_n_original_coords(points, R, n, coeff):
    # Step 1: Rotate points to align with v
    rotated_points = rotate_points(points, R)

    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    z = f_n(x, y, n, coeff)

    # Build the matrix of rotated points
    rotated_points[:, 2] = z

    # Rotate back to initial coordinate system
    points = rotate_points_invers(rotated_points, R)

    return points

# Visualization function
def plot_surface_and_points(points, R, v, n, coeff):
    v = v / np.linalg.norm(v)

    # Calculate centroid of the points
    centroid = np.mean(points, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Calculate the range for each axis
    x_range = np.ptp(points[:, 0])  # ptp computes the "peak-to-peak" value, i.e., max - min
    y_range = np.ptp(points[:, 1])
    z_range = np.ptp(points[:, 2])

    # Calculate the maximum range value
    max_range = max([x_range, y_range, z_range])

    # Set the axis limits based on the maximum range value
    ax.set_xlim([centroid[0] - max_range/2, centroid[0] + max_range/2])
    ax.set_ylim([centroid[1] - max_range/2, centroid[1] + max_range/2])
    ax.set_zlim([centroid[2] - max_range/2, centroid[2] + max_range/2])

    # Ensure that the aspect ratio is the same for all axes
    ax.set_box_aspect([1, 1, 1])

    
    # Plot original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', label='Original Points', s=1)

    #  Rotate points to align with v
    rotated_points = rotate_points(points, R)

    # Create a grid for plotting the surface
    x = np.linspace(min(rotated_points[:, 0]), max(rotated_points[:, 0]), 50)
    y = np.linspace(min(rotated_points[:, 1]), max(rotated_points[:, 1]), 50)
    X, Y = np.meshgrid(x, y)
    Z = f_n(X, Y, n, coeff)  # Passing coefficients to the quadratic function

    # Transform X Y Z back to original coordinate system
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            rotated_point = np.array([X[i, j], Y[i, j], Z[i, j]])
            original_point = rotate_points_invers(rotated_point, R)
            X[i, j], Y[i, j], Z[i, j] = original_point

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100)
    
    # Project the points onto the surface and plot
    projected = f_n_original_coords(points, R, n, coeff)
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], color='r', label='Projected Points', s=1)
    
    v *= 10
    print(f"Centroid: {centroid}, v: {v}")
    # Plot v starting from the centroid
    ax.quiver(centroid[0], centroid[1], centroid[2], 
              v[0], v[1], v[2], color='g', length=1, normalize=False, arrow_length_ratio=0.1, label='Vector v')
    
    ax.legend()
    plt.show()

def distance_from_surface_clipped(points, R, n, coeff, min_, max_, return_direction=False):
    """Compute the distance of multiple points from the surface"""
    
    # Project the points to the surface using your functions
    projected = f_n_original_coords(points, R, n, coeff)
    
    # Compute the distance for each point from its projection on the surface
    distances = np.linalg.norm(points - projected, axis=1)
    # Clip the distances
    distances = np.clip(distances, min_, max_)

    if return_direction:
        # Compute the direction from each point to its projection on the surface
        directions = points - projected
        # Normalize the directions
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        # get the mean direction
        direction = get_vector_mean(directions)
        return distances, direction
    
    return distances

def distance_from_surface(points, R, n, coeff, return_direction=False):
    """Compute the distance of multiple points from the surface"""
    
    # Project the points to the surface using your functions
    projected = f_n_original_coords(points, R, n, coeff)
    
    # Compute the distance for each point from its projection on the surface
    distances = np.linalg.norm(points - projected, axis=1)

    if return_direction:
        # Compute the direction from each point to its projection on the surface
        directions = points - projected
        # Normalize the directions
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        # get the mean direction
        direction = get_vector_mean(directions)
        return distances, direction
    
    return distances

def optimize_sheet(points, R, n, threshold=3.0, max_iters=5, alpha=0.1, slope_alpha=0.1):
    """Iteratively optimize the sheet fitting"""
    current_points = points
    inlier_mask = np.ones(len(points), dtype=bool)

    for iteration in range(max_iters):
        coeff = fit_surface_to_points_n_regularized(current_points, R, n, alpha=alpha, slope_alpha=slope_alpha)

        # Calculate distances from points to the fitted surface
        distances_points = distance_from_surface(points, R, n, coeff)

        distances = distances_points[inlier_mask]

        # Select subset of points close enough to the surface
        # median_distance = np.median(distances_points)
        # top 60%
        percentile = 50
        while True:
            sheet_distance = np.percentile(distances, percentile) * 3

            # print(f"Median distance: {np.median(distances)}")

            inlier_mask = np.array(distances_points) <= sheet_distance

            if np.sum(inlier_mask) > distances_points.shape[0] * 0.80:
                break
            else:
                percentile += 5

        current_points = points[inlier_mask]

        if sheet_distance < threshold:
            inlier_mask = np.array(distances_points) <= threshold
            current_points = points[inlier_mask]
            break

    return coeff, current_points, inlier_mask, sheet_distance

def get_vector_mean(vectors):
    # vectors = np.array([v / np.linalg.norm(v) for v in vectors])
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    vectors = vectors / norms

    vector = np.mean(vectors, axis=0)
    return vector / np.linalg.norm(vector)

if __name__ == '__main__':
    # Sample 3D points
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [2, 3, 2],
        [4, 1, 5]
    ])

    instance_nr = 8
    path= f"/media/julian/FastSSD/scroll1_surface_points/point_cloud_colorized_test_subvolume_blocks/000775_001450_000975/surface_{instance_nr}.ply"

    points, normals, colors = load_ply(path)

    # normalize normals
    normals = np.array([v / np.linalg.norm(v) for v in normals])

    normal = np.mean(normals, axis=0)

    # Set the degree
    n = 4  # or 3, 4, ...
    alpha = 1000.0
    slope_alpha = 0.1
    v = normal

    R = rotation_matrix_to_align_z_with_v(v)
    coeff, points, points_mask, sheet_distance = optimize_sheet(points, R, n, alpha=alpha, slope_alpha=slope_alpha)
    # Visualize the surface and the points
    plot_surface_and_points(points, R, v, n, coeff)
