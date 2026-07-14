### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

# python3 -m datasets.preprocessing.thaumato_preprocessing preprocess --data_dir="data/raw/thaumatoanakalyptor" --save_dir="data/processed/stpls3d"

from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
from .mask3d.utils.point_cloud_utils import write_point_cloud_in_ply
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process

MAX_NUM_POINTS_PER_SURFACE = 50000
MIN_NUM_POINTS_PER_SURFACE = 1000
CUBE_SIZE = 1.0
MIN_SURFACES = 4
MAX_SURFACES = 15

def random_unit_vector():
    """Generate a random 3D unit vector."""
    v = np.random.uniform(-1, 1, 3)
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else random_unit_vector()

def generate_smooth_random_heatmap(size=(3000, 3000), alpha=10.0, sigma=300): # sigma 250 - 300
    """Generate a heatmap with smoothed random values."""
    
    # Assign random values to the entire heatmap
    heatmap = np.random.rand(*size) * alpha
    
    # Apply Gaussian blur
    blurred_heatmap = gaussian_filter(heatmap, sigma)
    
    # Generate holes in the heatmap
    # Clip values to zero that fall in the lowest 1/6 of all values
    threshold = np.percentile(heatmap, 100.0/ 6.0)  # 1/6 corresponds to approximately 16.6666 percentile
    blurred_heatmap[heatmap < threshold] = 0

    # Normalize the heatmap to the range [0, 1]
    blurred_heatmap /= blurred_heatmap.max()

    return blurred_heatmap

def generate_heatmap(size=(10000, 10000), center=None, sigma=20):
    """Generate a Gaussian heatmap centered at 'center'."""
    if center is None:
        center = (size[0]//2, size[1]//2)
    y, x = np.ogrid[:size[0], :size[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    heatmap = np.exp(-dist_from_center**2 / (2*sigma**2))
    return heatmap

def sample_from_heatmap(heatmap, num_samples):
    """
    Directly sample coordinates from a 2D heatmap.
    Returns the sampled coordinates as integers.
    """
    x_shape, y_shape = heatmap.shape

    # Flatten the heatmap
    flat_heatmap = heatmap.ravel()
    
    # Normalize it
    probabilities = flat_heatmap / flat_heatmap.sum()
    
    # Sample flattened indices from the heatmap
    sampled_indices = np.random.choice(flat_heatmap.size, size=num_samples, p=probabilities)
    
    # Convert these indices back to 2D coordinates
    sampled_coords = np.column_stack(np.unravel_index(sampled_indices, (x_shape, y_shape)))

    return sampled_coords

def rotate_vector(vector, axis, theta):
    """
    Rotate a vector by theta degrees around a given axis.
    """
    theta = np.radians(theta)  # Convert angle to radians
    cross_prod = np.cross(axis, vector)
    dot_prod = np.dot(axis, vector)
    
    rotated_vector = np.cos(theta) * vector + np.sin(theta) * cross_prod + (1 - np.cos(theta)) * dot_prod * axis
    return rotated_vector

def adjust_normal_within_degree(normal, max_degree=5):
    """
    Slightly adjust the normal vector within the given max_degree.
    """
    random_direction = np.random.randn(3)
    random_direction /= np.linalg.norm(random_direction)
    
    random_angle = np.random.uniform(0, max_degree)
    
    adjusted_normal = rotate_vector(normal, random_direction, random_angle)
    adjusted_normal /= np.linalg.norm(adjusted_normal)  # Ensure the vector is normalized
    
    return adjusted_normal

def generate_flat_surface(normal, num_points=1000, cube_size=1.0):
    """
    Generate a flat surface with a random normal within the cube, biased by a heatmap.
    """
    adjusted_normal = adjust_normal_within_degree(normal, max_degree=5)

    point_on_plane = np.array([cube_size / 2, cube_size / 2, cube_size / 2])
    
    # Create an orthonormal basis for the plane
    basis_a = np.random.randn(3)
    basis_a -= basis_a.dot(adjusted_normal) * adjusted_normal
    basis_a /= np.linalg.norm(basis_a)
    
    basis_b = np.cross(adjusted_normal, basis_a)
    
    # Generate a heatmap
    heatmap = generate_smooth_random_heatmap((5000,5000))

    # Sample from the heatmap
    random_coefficients = sample_from_heatmap(heatmap, num_points)
    # Convert integer coordinates to float and normalize
    random_coefficients = (random_coefficients - np.array(heatmap.shape) / 2.0) / np.array(heatmap.shape)

    plane_points = point_on_plane + random_coefficients[:, 0:1] * basis_a + random_coefficients[:, 1:2] * basis_b
    
    # Remove points that fall outside the cube
    mask_inside = np.all((plane_points >= 0) & (plane_points <= cube_size), axis=1)
    plane_points = plane_points[mask_inside]
    
    return plane_points

def sinusoidal_distortion(points):
    """Sinusoidal distortion."""
    return points + 0.05 * np.sin(5 * points)

def generate_displacement_field(volume_shape, alpha, sigma):
    """Generate a dense displacement field for the given volume shape."""
    dz = gaussian_filter(np.random.randn(*volume_shape), sigma, mode="reflect") * alpha
    dy = gaussian_filter(np.random.randn(*volume_shape), sigma, mode="reflect") * alpha
    dx = gaussian_filter(np.random.randn(*volume_shape), sigma, mode="reflect") * alpha
    return dz, dy, dx

def sample_displacement(points, dz, dy, dx):
    """Sample from the displacement fields at the point cloud coordinates."""
    z_coords = np.clip(np.round(points[:, 0]).astype(int), 0, dz.shape[0]-1)
    y_coords = np.clip(np.round(points[:, 1]).astype(int), 0, dy.shape[1]-1)
    x_coords = np.clip(np.round(points[:, 2]).astype(int), 0, dx.shape[2]-1)

    sampled_dz = dz[z_coords, y_coords, x_coords]
    sampled_dy = dy[z_coords, y_coords, x_coords]
    sampled_dx = dx[z_coords, y_coords, x_coords]

    return np.stack([sampled_dz, sampled_dy, sampled_dx], axis=-1)

def elastic_deformation_on_volume(points, volume_shape, alpha=2000, sigma=50):
    """Apply elastic deformation on a point cloud by creating a dense displacement field over a volume."""
    
    # 1. Create a dense displacement field
    dz, dy, dx = generate_displacement_field(volume_shape, alpha, sigma)
    
    # 2. Sample from this field at the coordinates of the points
    displacements = sample_displacement(points, dz, dy, dx)

    # 3. Apply these displacements to the point cloud
    deformed_points = points + displacements

    return deformed_points

def compute_bounding_box_centroid(points):
    """Calculate the centroid of the bounding box formed by the given points."""
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    centroid = (min_values + max_values) / 2
    return centroid

def bend_volume(points, normal, radius=1.0):
    """
    Bend the whole volume in the shape of a circular arc.
    
    Parameters:
    - points: Numpy array containing all points from the volume.
    - normal: Normal direction to determine the center of the circular arc.
    - radius: Radius of the circular arc.
    
    Returns:
    - bent_points: Numpy array of points after the bend transformation.
    """
    
    # Compute the centroid of the points
    centroid = compute_bounding_box_centroid(points)

    # Compute the center of the circle in the direction of the normal from the centroid
    circle_center = centroid + radius * normal
    
    # To generate a random axis perpendicular to the normal, first choose a random vector
    random_vector = np.random.rand(3)

    # Determine an axis perpendicular to the normal (cross product with the random vector)
    axis = np.cross(normal, random_vector)
    
    # If the cross product is close to zero (the vectors are nearly parallel), choose a different random vector
    while np.linalg.norm(axis) < 1e-5:
        random_vector = np.random.rand(3)
        axis = np.cross(normal, random_vector)
    
    axis /= np.linalg.norm(axis)

    # For each point, find the closest point on the line defined by circle_center and axis
    t_values = np.dot(points - circle_center, axis) / np.dot(axis, axis)
    pc_points = np.outer(t_values, axis) + circle_center
    
    # Compute the vectors between pc and the points
    pvec = points - pc_points

    # Project pvec onto the normal
    projection_values = np.dot(pvec, normal)
    projections = projection_values[:, np.newaxis] * normal

    projected_lengths = np.linalg.norm(projections, axis=1, keepdims=True)

    # Adjust pvec to have the length of the projected vector
    adjusted_pvec = (pvec / np.linalg.norm(pvec, axis=1, keepdims=True)) * projected_lengths

    # Translate the points
    bent_points = pc_points + adjusted_pvec

    return bent_points


def merge_surfaces(surfaces, normal, lower_bound=0.025, upper_bound=0.1):
    """
    Merge multiple surfaces into a single volume by translating them in the direction of the normal.
    
    Parameters:
    - surfaces: List of point arrays representing each surface
    - normal: Normal direction for translation
    - lower_bound: Minimum distance to translate a surface
    - upper_bound: Maximum distance to translate a surface
    
    Returns:
    - merged_points: Numpy array containing all points from the merged surfaces
    """
    merged_points = surfaces[0]  # Start with the first surface
    translation_distance = 0

    for i in range(1, len(surfaces)):
        translation_distance += np.random.uniform(lower_bound, upper_bound)
        translated_surface = surfaces[i] + translation_distance * normal
        merged_points = np.vstack((merged_points, translated_surface))

    return merged_points

def add_noise_to_volume(points, labels, cube_size, num_noise_points):
    """
    Add noise points throughout the volume.
    
    Args:
    - points (numpy array): The array of points from the surface.
    - labels: Labels
    - cube_size (float): The size of the volume.
    - num_noise_points (int): The number of noisy points to generate.

    Returns:
    - combined_points (numpy array): The combined array of original and noisy points.
    - labels (numpy array): The corresponding labels for the combined points (0 for original, -1 for noisy).
    """
    # Generate random points within the cube
    noisy_points = np.random.rand(num_noise_points, 3) * cube_size
    
    # Combine original and noisy points
    combined_points = np.vstack([points, noisy_points])
    
    # Generate labels: 0 for original points, -1 for noisy points
    labels = np.concatenate([labels, -1 * np.ones(num_noise_points)])
    
    return combined_points, labels

def crop_points_within_volume(points, labels, volume_size):
    """
    Crop points that lie outside of a volume of the specified size, centered at the bounding box centroid of the points.
    
    Parameters:
    - points: Numpy array of 3D points (Nx3)
    - labels: Labels
    - volume_size: Size of the volume [x_size, y_size, z_size]
    
    Returns:
    - cropped_points: Points that lie within the specified volume
    """
    centroid = compute_bounding_box_centroid(points)
    half_size = 0.5 * np.array(volume_size)
    
    # Determine the bounds of the volume
    lower_bounds = centroid - half_size
    upper_bounds = centroid + half_size
    
    # Filter points that are within these bounds
    inside_mask = np.all((points >= lower_bounds) & (points <= upper_bounds), axis=1)
    
    cropped_points = points[inside_mask] - lower_bounds
    cropped_labels = labels[inside_mask]
    
    return cropped_points, cropped_labels

def visualize(points, labels):
    """Visualize using Matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels):
        ax.scatter(points[labels == label][:, 0], points[labels == label][:, 1], points[labels == label][:, 2], s=1)
    plt.show()

def compute_normals(points):
    """Compute normals for points."""
    # Placeholder. An accurate implementation might require neighborhood analysis or library calls.
    normals = np.ones_like(points)
    return normals

def save_point_cloud_(points, normals, labels, filename="point_cloud.ply"):
    """Save point cloud in .ply format."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(labels[:, np.newaxis], 3, axis=1) / float(MAX_SURFACES)) # pseudo-color
    o3d.io.write_point_cloud(filename, pcd)

def save_point_cloud__(points, normals, labels, filename="point_cloud.ply"):
    """
    Save point cloud in .ply format using the utility script's function.
    """

    # Convert labels into colors for saving.
    processed_labels = np.where(labels < 0, 0, 1)
    instance_ids = np.where(labels < 0, 0, labels)

    colors = np.zeros_like(points)
    colors[:, 0] = processed_labels
    colors[:, 1] = instance_ids

    # Convert filename to Path object for compatibility with the utility function.
    filepath = Path(filename)

    # Save using the utility function.
    write_point_cloud_in_ply(filepath, points, colors, labels)

def save_point_cloud(points, normals, labels, filename="point_cloud.txt"):
    """
    Save point cloud in .txt format for use with the process_file function.
    """

    # Convert labels into processed labels and instance ids.
    processed_labels = np.where(labels < 0, 0, 1)
    instance_ids = np.where(labels < 0, -100, labels)  # Changed -1 to -100 directly
    colors = np.random.rand(*points.shape)

    # Assuming the points array shape is (n, 3) and normals is (n, 3),
    # concatenate them along with labels to form the desired structure.
    # Here, we'll only use the first three columns of normals for now.
    data_to_save = np.hstack((points, colors, processed_labels[:, None], instance_ids[:, None]))

    # Convert filename to Path object for compatibility and save as .txt format.
    filepath = Path(filename)
    np.savetxt(filepath, data_to_save, delimiter=',')  # Using comma as delimiter


def process_sample(sample_nr):
    # Seed numpy's random number generator with the current process ID
    np.random.seed(sample_nr*current_process().pid)
    # Generate and distort surfaces
    num_surfaces = np.random.randint(MIN_SURFACES, MAX_SURFACES+1)
    surfaces = []
    labels_list = []
    normal = random_unit_vector()
    for i in range(num_surfaces):
        surface = generate_flat_surface(normal, num_points=np.random.randint(min_points_surface, max_points_surface+1), cube_size=CUBE_SIZE)
        surface = sinusoidal_distortion(surface)
        surfaces.append(surface)
        labels_list.append(np.ones(len(surface)) * i)
        print(f"Generated surface {i+1}/{num_surfaces}")

    # calculate spacings to fill out the space appropriately
    middle = 0.5 / num_surfaces
    lower_bound = 0.015
    upper_bound = 2* middle - lower_bound
    # looser bounds to give more randomness
    lower_bound *= 0.95
    upper_bound *= 1.05
    # Merge surfaces
    surfaces = merge_surfaces(surfaces, normal, lower_bound=lower_bound, upper_bound=upper_bound)
    # Merge labels
    labels = np.hstack(labels_list)
    
    # Distort further
    cube_size = 100
    surfaces *= cube_size
    volume_shape = (cube_size, cube_size, cube_size)  # Assuming a 100x100x100 cube
    surfaces = elastic_deformation_on_volume(surfaces, volume_shape, alpha=250, sigma=5.5) # sigma 7 -
    surfaces = bend_volume(surfaces, normal, radius=np.random.randint(50, 100)) # 75 original
    
    # Crop points within a volume
    volume_size = [50, 50, 50]
    surfaces, labels = crop_points_within_volume(surfaces, labels, volume_size=volume_size )
    
    # Noise
    surfaces, labels = add_noise_to_volume(surfaces, labels, volume_size[0], 1000)

    # Compute normals (placeholder)
    normals = compute_normals(surfaces)
    
    # Visualization
    # visualize(surfaces, labels)

    # Generate folder
    if not os.path.exists(f"point_clouds"):
        os.makedirs(f"point_clouds")
    # Save the generated data
    filename=f"point_clouds/point_cloud_{sample_nr}.txt"
    save_point_cloud(surfaces, normals, labels, filename=filename)

def main():
    global min_points_surface, max_points_surface
    min_points_surface = np.random.randint(MIN_NUM_POINTS_PER_SURFACE, MAX_NUM_POINTS_PER_SURFACE)
    max_points_surface = np.random.randint(min_points_surface, MAX_NUM_POINTS_PER_SURFACE+1)

    n = 500 # nr training samples to generate
    pool = Pool(processes=cpu_count())  # use all available CPUs
    pool.map(process_sample, range(0, n))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
