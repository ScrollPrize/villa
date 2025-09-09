### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import hdbscan
import os
import open3d as o3d
import trimesh
import threading
# Global variable to store the main_sheet data
main_sheet_data = None
# Signal to indicate when the data is updated
data_updated_event = threading.Event()

# surface points extraction
from multiprocessing import Pool

from tqdm import tqdm

from scipy.interpolate import interp1d

from sklearn.neighbors import KDTree
from copy import deepcopy

from .instances_to_sheets import load_main_sheet, surrounding_volumes_main_sheet, build_main_sheet_patches_list, build_main_sheet_from_patches_list, build_main_sheet_volume_from_patches_list, build_patch, make_unique, alpha_angles, angle_to_180
from .surface_fitting_utilities import get_vector_mean, rotation_matrix_to_align_z_with_v, f_n,  rotate_points, rotate_points_invers, fit_surface_to_points_n_regularized
from .fix_mesh import find_degenerated_triangles_and_delete

def load_xyz_from_file(filename='umbilicus.txt'):
    """
    Load a file with comma-separated xyz coordinates into a 2D numpy array.
    
    :param filename: The path to the file.
    :return: A 2D numpy array of shape (n, 3) where n is the number of lines/coordinates in the file.
    """
    return np.loadtxt(filename, delimiter=',')


def umbilicus(points_array):
    """
    Interpolate between points in the provided 2D array based on y values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :return: A 2D numpy array with interpolated points for each 0.1 step in the y direction.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and y based on z
    fx = interp1d(y, x, kind='linear', fill_value="extrapolate")
    fz = interp1d(y, z, kind='linear', fill_value="extrapolate")

    # Define new y values for interpolation
    y_new = np.arange(y.min(), y.max(), 1)

    # Calculate interpolated x and y values
    x_new = fx(y_new)
    z_new = fz(y_new)

    # Return the combined x, y, and z values as a 2D array
    return np.column_stack((x_new, y_new, z_new))

def umbilicus_xz_at_y(points_array, y_new):
    """
    Interpolate between points in the provided 2D array based on y values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :return: A 2D numpy array with interpolated points for each 0.1 step in the y direction.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and z based on y
    fx = interp1d(y, x, kind='linear', fill_value="extrapolate")
    fz = interp1d(y, z, kind='linear', fill_value="extrapolate")

    # Calculate interpolated x and z values
    x_new = fx(y_new)
    z_new = fz(y_new)

    # Return the combined x, y, and z values as a 2D array
    res = np.array([x_new, y_new, z_new]).T
    return res

def umbilicus_xy_at_z(points_array, z_new):
    """
    Interpolate between points in the provided 2D array based on z values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :param z_new: A 1D numpy array of y-values.
    :return: A 2D numpy array with interpolated points for each z value.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and z based on y
    fx = interp1d(z, x, kind='linear', fill_value="extrapolate")
    fy = interp1d(z, y, kind='linear', fill_value="extrapolate")

    # Calculate interpolated x and z values
    x_new = fx(z_new)
    y_new = fy(z_new)

    # Return the combined x, y, and z values as a 2D array
    return np.column_stack([x_new, y_new, z_new])

def generate_line(start_point, direction_vector, filename, length=500, step=0.1):
    """
    Generate a line in 3D space.
    
    :param start_point: Starting point of the line as a 1x3 numpy array.
    :param direction_vector: Direction of the line as a 1x3 numpy array. This will be normalized.
    :param length: Length of the line.
    :param step: Distance between consecutive points.
    :return: A 2D numpy array containing the points of the line.
    """
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Create the points
    num_points = int(length / step)
    points = np.array([start_point + i * step * direction_vector for i in range(num_points)])
    
    line = np.array(points)

    yellow = np.array([1, 1, 0])  # RGB for yellow
    colors = np.tile(yellow, (points.shape[0], 1))
    save_surface_ply(line, np.zeros_like(line), filename, color=colors)

def save_surface_ply(surface_points, normals, filename, colors=None):
    # Create an Open3D point cloud object and populate it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create folder if it doesn't exist
    print(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save as a PLY file
    o3d.io.write_point_cloud(filename, pcd)

def filter_patches_by_angle_offset(main_sheet_volume_patches, offset_angle_patch, winding_angle_threshold=120):
    angle_range = [offset_angle_patch - winding_angle_threshold, offset_angle_patch + winding_angle_threshold]
    new_main_sheet_volume_patches = []
    for patch in main_sheet_volume_patches:
        if angle_range[0] <= patch[2] <= angle_range[1]:
            new_main_sheet_volume_patches.append(patch)
    return new_main_sheet_volume_patches

def winding_surface(main_sheet_info, path, volume_id, patch_nr, volume_size=50, winding_angle_threshold=120, density = 5.0, sample_ratio=0.1, only_originals=False):
    """
    Extract the patches in main sheet of the same winding as the given patch, return fitted surface
    """
    # Meta data of patch
    patch_metadata = main_sheet_info[volume_id][patch_nr]
    # Get the patch
    offset_angle_patch = patch_metadata["offset_angle"]
    # Build main sheet in region around patch
    if only_originals:
        surrounding_volume_ids = [volume_id]
    else:
        surrounding_volume_ids = surrounding_volumes_main_sheet(volume_id, volume_size=volume_size) # build larger main sheet, also adjust tiling for combine patches
    main_sheet_volume_patches = build_main_sheet_patches_list(surrounding_volume_ids, main_sheet_info)
    # filter all main patches with similar angle offset (same winding)
    main_sheet_volume_patches = filter_patches_by_angle_offset(main_sheet_volume_patches, offset_angle_patch, winding_angle_threshold=winding_angle_threshold)
    main_sheet, _ = build_main_sheet_from_patches_list(main_sheet_volume_patches, volume_size, path, sample_ratio_score=sample_ratio, align_and_flip_normals=False)
    # fit surface to main sheet this winding
    if not only_originals:
        surface = calculate_surface(main_sheet)
    else:
        surface = None

    if only_originals:
        # make all points unique again
        main_sheet = make_unique(main_sheet)

    # Points of patch
    main_sheet_points_around_id = main_sheet["points"]
    main_sheet_normals_around_id = main_sheet["normals"]
    main_sheet_colors_around_id = main_sheet["colors"]
    main_sheet_winding_angles_around_id = main_sheet["angles"]

    main_sheet_points_around_id_org = (main_sheet_points_around_id, main_sheet_normals_around_id, main_sheet_colors_around_id, main_sheet_winding_angles_around_id)

    if only_originals:
        res_interpolated = None
    else:
        volume_id_np = np.array(volume_id)
        volume_end = volume_id_np + volume_size

        # Adjust volume_end for the density step
        volume_end_adj = [
            volume_end[0] - (volume_end[0] - volume_id_np[0]) % density,
            volume_end[1] - (volume_end[1] - volume_id_np[1]) % density,
            volume_end[2] - (volume_end[2] - volume_id_np[2]) % density
        ]

        # Compute middle points
        middle = (volume_id_np + volume_end_adj) / 2
        # Check if middle is reached when making density steps from start
        assert middle[0] in np.arange(volume_id_np[0], volume_end_adj[0], density)
        assert middle[1] in np.arange(volume_id_np[1], volume_end_adj[1], density)
        assert middle[2] in np.arange(volume_id_np[2], volume_end_adj[2], density)


        # Sample points on the surface
        y = np.arange(volume_id_np[1], volume_end_adj[1], density)
        z = np.arange(volume_id_np[2], volume_end_adj[2], density)
        x = np.arange(volume_id_np[0], volume_end_adj[0], density)
        # For the x faces
        X1, Y1, Z1 = np.meshgrid(volume_id_np[0], y, z) # one side
        X2, Y2, Z2 = np.meshgrid(volume_end_adj[0], y, z) # other side
        X_mid_x, Y_mid_x, Z_mid_x = np.meshgrid(middle[0], y, z) # middle in x direction

        # For the y faces
        X3, Y3, Z3 = np.meshgrid(x, volume_id_np[1], z) # one side
        X4, Y4, Z4 = np.meshgrid(x, volume_end_adj[1], z) # other side
        X_mid_y, Y_mid_y, Z_mid_y = np.meshgrid(x, middle[1], z) # middle in y direction

        # For the z faces
        X5, Y5, Z5 = np.meshgrid(x, y, volume_id_np[2]) # one side
        X6, Y6, Z6 = np.meshgrid(x, y, volume_end_adj[2]) # other side
        X_mid_z, Y_mid_z, Z_mid_z = np.meshgrid(x, y, middle[2]) # middle in z direction

        # Combine all the points
        X = np.concatenate((X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel(), X_mid_x.ravel(), X_mid_y.ravel(), X_mid_z.ravel()))
        Y = np.concatenate((Y1.ravel(), Y2.ravel(), Y3.ravel(), Y4.ravel(), Y5.ravel(), Y6.ravel(), Y_mid_x.ravel(), Y_mid_y.ravel(), Y_mid_z.ravel()))
        Z = np.concatenate((Z1.ravel(), Z2.ravel(), Z3.ravel(), Z4.ravel(), Z5.ravel(), Z6.ravel(), Z_mid_x.ravel(), Z_mid_y.ravel(), Z_mid_z.ravel()))


        # Reshape them into a list of 3D coordinates
        grid_points_3d = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        # Remove duplicate rows to get unique 3D points
        grid_points_3d = np.unique(grid_points_3d, axis=0)
        grid_normals_3d = np.tile(np.array([0, 0, 1]), (len(grid_points_3d), 1))
        grid_colors_3d = np.tile(np.array([0, 0, 0]), (len(grid_points_3d), 1))
        grid_winding_angles_3d = np.tile(offset_angle_patch, len(grid_points_3d))

        # Concatenate the points of the patch and the grid points
        main_sheet_points_around_id = np.concatenate([main_sheet_points_around_id, grid_points_3d])
        main_sheet_normals_around_id = np.concatenate([main_sheet_normals_around_id, grid_normals_3d])
        main_sheet_colors_around_id = np.concatenate([main_sheet_colors_around_id, grid_colors_3d])
        main_sheet_winding_angles_around_id = np.concatenate([main_sheet_winding_angles_around_id, grid_winding_angles_3d])

        # Get the points of the main sheet in the region around the patch
        mask = (main_sheet_points_around_id[:, 0] >= volume_id_np[0]) & \
                (main_sheet_points_around_id[:, 1] >= volume_id_np[1]) & \
                (main_sheet_points_around_id[:, 2] >= volume_id_np[2]) & \
                (main_sheet_points_around_id[:, 0] <= volume_end[0]) & \
                (main_sheet_points_around_id[:, 1] <= volume_end[1]) & \
                (main_sheet_points_around_id[:, 2] <= volume_end[2])
        
        res_interpolated = (main_sheet_points_around_id[mask], main_sheet_normals_around_id[mask], main_sheet_colors_around_id[mask], main_sheet_winding_angles_around_id[mask])
    
    return surface, offset_angle_patch, res_interpolated, main_sheet_points_around_id_org

def calculate_surface(main_sheet_surface_patch):
    # points and normals from patches
    points = main_sheet_surface_patch["points"]
    normals = main_sheet_surface_patch["normals"]
    normal_vector = get_vector_mean(normals)
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Sheet fitting params
    n = 8
    alpha = 0.0
    slope_alpha = 0.0

    # Fit other patch and main sheet together
    coeff_all = fit_surface_to_points_n_regularized(points, R, n, alpha=alpha, slope_alpha=slope_alpha)
    surface = (normal_vector, R, n, coeff_all)
    return surface

def points_to_surface(points, surface):
    """
    Each point on to the surface
    """
    normal_vector, R, n, coeff_all = surface

    # Rotate points to the 2.5D surface coordinates
    points_surface_coord = rotate_points(points, R)

    # Calculate the distance of each point to the surface
    x, y, z = points_surface_coord.T
    # Calculate z values using the surface function for given x and y
    z = f_n(x, y, n, coeff_all)

    # Rotate points back to the original volume coordinates
    points_in_original_coords = rotate_points_invers(np.vstack([x, y, z]).T, R)

    return points_in_original_coords


def cast_points_to_surface(patch_position, patch_angle, surface, raster_distance, subvolume_size):
    """
    Cast points to surface and return the surface points
    """
    normal_vector, R, n, coeff_all = surface

    # Cast the middle of the subvolume into the 2.5D surface coordinates
    half_size = subvolume_size / 2
    patch_position = np.array(patch_position)
    mid_point = patch_position + half_size
    mid_point_surface_coord = rotate_points(mid_point[np.newaxis, :], R)
    
    x_center, y_center, _ = mid_point_surface_coord[0]
    
    # Space points on surface
    x_range = np.arange(x_center - half_size, x_center + half_size + raster_distance, raster_distance)
    y_range = np.arange(y_center - half_size, y_center + half_size + raster_distance, raster_distance)
    
    x, y = np.meshgrid(x_range, y_range)
    x = x.ravel()
    y = y.ravel()

    # Calculate z values using the surface function for given x and y
    z = f_n(x, y, n, coeff_all)

    # Rotate points back to the original volume coordinates
    points = np.vstack([x, y, z]).T
    points_in_original_coords = rotate_points_invers(points, R)

    # Filter points that are inside the subvolume block
    patch_end_position = patch_position + subvolume_size
    mask = (points_in_original_coords[:, 0] > patch_position[0]) & \
           (points_in_original_coords[:, 1] > patch_position[1]) & \
           (points_in_original_coords[:, 2] > patch_position[2]) & \
           (points_in_original_coords[:, 0] < patch_end_position[0]) & \
           (points_in_original_coords[:, 1] < patch_end_position[1]) & \
           (points_in_original_coords[:, 2] < patch_end_position[2])
    
    filtered_points = points_in_original_coords[mask]
    
    # Calculate normals and winding angles
    num_points = filtered_points.shape[0]
    normals = np.tile(normal_vector, (num_points, 1))
    winding_angles = np.tile(patch_angle, (num_points, 1))

    return filtered_points, normals, winding_angles

def concat_patches_points(patches_points_list):
    points_list = []
    normals_list = []
    winding_angles_list = []
    print("Building points lists")
    for patch_points in patches_points_list:
        points_, normals_, _, angles_ = patch_points
        points_list.append(points_)
        normals_list.append(normals_)
        winding_angles_list.append(angles_)
    print("concatenate")
    # Concatenate all the points in the patches
    all_points = np.concatenate(points_list, axis=0)
    all_normals = np.concatenate(normals_list, axis=0)
    all_winding_angles = np.concatenate(winding_angles_list, axis=0)
    print("unique")
    # Get unique points and their indices
    unique_points, first_occurrence_indices, unique_indices = np.unique(all_points, axis=0, return_index=True, return_inverse=True)
    # Get the corresponding normals and winding angles for the unique points
    unique_normals = all_normals[first_occurrence_indices]
    unique_winding_angles = all_winding_angles[first_occurrence_indices]

    print("unique ranges")
    # Compute the start and end index for each points array in the combined array
    ranges = []
    curr_start = 0
    for points in points_list:
        length = len(points)
        curr_end = curr_start + length
        ranges.append((curr_start, curr_end))
        curr_start = curr_end

    print("index ranges")
    # Convert the ranges to index ranges in the unique points
    unique_ranges = [[unique_indices[u] for u in range(r[0], r[1])] for r in ranges]
    print("done")
    return unique_points, unique_normals, unique_winding_angles, unique_ranges

def update_points_in_combined(combined_points_update_count, patch_points_update, patch_points_weights, combined_points, combined_points_weight, combined_ranges, idx):
    indices_in_combined = combined_ranges[idx]
    combined_points_update_count[indices_in_combined] += 1
    combined_points[indices_in_combined] += patch_points_update * patch_points_weights[:, np.newaxis]
    combined_points_weight[indices_in_combined] += patch_points_weights

def update_points_from_weight(main_sheet_points_org, main_sheet_points_count, main_sheet_points, main_sheet_normals, main_sheet_winding_angles, main_sheet_points_weight, subvolume_size):
    # Update the combined points with the weighted average of the patches
    mask_0 = main_sheet_points_weight != 0
    # mask_count = main_sheet_points_count > 2 # for less badly overlapping patches
    mask_count = main_sheet_points_count > 0 # disable this feature
    mask = np.logical_and(mask_0, mask_count)
    print(f"Removing total {np.sum(~mask)} points from the combined pointcloud. {np.sum(~mask_0)} because they have weight 0 and {np.sum(~mask_count)} because they are only in zero patch, remaining points: {np.sum(mask)}")
    print(f"Total number of points in the combined pointcloud: {len(main_sheet_points)}, number of weights: {len(main_sheet_points_weight)}, number of normals: {len(main_sheet_normals)}, number of winding angles: {len(main_sheet_winding_angles)}")
    main_sheet_points_org = main_sheet_points_org[mask] # Remove points with weight 0
    main_sheet_points = main_sheet_points[mask] # Remove points with weight 0
    main_sheet_normals = main_sheet_normals[mask] # Remove points with weight 0
    main_sheet_winding_angles = main_sheet_winding_angles[mask] # Remove points with weight 0
    main_sheet_points_weight = main_sheet_points_weight[mask] # Remove points with weight 0

    # Divide by the weight to get the weighted average
    main_sheet_points /= main_sheet_points_weight[:, np.newaxis]

    # Check, that each point is still inside the subvolume / 2 block it was originally in. Mask and remove points that are not.
    mask = np.all(main_sheet_points_org // (subvolume_size / 2) == main_sheet_points // (subvolume_size / 2), axis=1)
    print(f"Removing {np.sum(~mask)} points from the combined pointcloud because they are not in the same subvolume / 2 block as they were originally, remaining points: {np.sum(mask)}")

    main_sheet_points = main_sheet_points[mask]
    main_sheet_normals = main_sheet_normals[mask]
    main_sheet_winding_angles = main_sheet_winding_angles[mask]
    main_sheet_points_weight = main_sheet_points_weight[mask]

    return main_sheet_points, main_sheet_normals, main_sheet_winding_angles
    
def patch_points_weight(patch_points, volume_id, subvolume_size):
    volume_id = np.array(volume_id)
    middle = volume_id + subvolume_size / 2

    # Calculate the distance of each point in each dimension to the middle of the subvolume
    dist = np.clip((0.01 + subvolume_size / 2 - np.abs(patch_points - middle)) / subvolume_size, a_min = 0.0, a_max=None)

    # Calculate the weight of each point
    weight = dist[:,0] * dist[:,1] * dist[:,2]

    return weight

def process_volume(args):
    # Unpack arguments 
    volume_id, main_sheet_info, subvolume_size, path, winding_angle_threshold, sample_ratio, only_originals = args
    
    points_dict = {}
    patches_points_list = []
    patches_points_list_org = []
    idx_patches = []
    points_dict[volume_id] = {}
    for patch_nr in main_sheet_info[volume_id]:
        surface, _, main_sheet_points_around_id, main_sheet_points_org = winding_surface(main_sheet_info, path, volume_id, patch_nr, volume_size=subvolume_size, winding_angle_threshold=winding_angle_threshold, sample_ratio=sample_ratio, only_originals=only_originals)

        # Points of patch
        volume_id_np = np.array(volume_id)
        volume_end = volume_id_np + subvolume_size
        
        # original points
        points_, normals_, colors_, winding_angles_ = main_sheet_points_org
        patch_points = points_
        patch_points_weights = patch_points_weight(patch_points, volume_id, subvolume_size)

        # Get the points of the main sheet in the region around the patch
        mask = (patch_points[:, 0] >= volume_id_np[0]) & \
                (patch_points[:, 1] >= volume_id_np[1]) & \
                (patch_points[:, 2] >= volume_id_np[2]) & \
                (patch_points[:, 0] <= volume_end[0]) & \
                (patch_points[:, 1] <= volume_end[1]) & \
                (patch_points[:, 2] <= volume_end[2])
        
        points_ = points_[mask]
        patch_points = patch_points[mask]
        normals_ = normals_[mask]
        colors_ = colors_[mask]
        winding_angles_ = winding_angles_[mask]
        patch_points_weights = patch_points_weights[mask]

        points_dict[volume_id][patch_nr] = (points_, patch_points, normals_, colors_, winding_angles_, patch_points_weights)

        main_sheet_points_around_id = (points_, normals_, colors_, winding_angles_) # update with masked points
        patches_points_list.append(main_sheet_points_around_id)
        patches_points_list_org.append(main_sheet_points_org)
        idx_patches.append((volume_id, patch_nr))

    return patches_points_list, patches_points_list_org, points_dict, idx_patches

def build_main_sheet_points_interpolted(main_sheet_info, subvolume_size, path, winding_angle_threshold=120, winding_angle_threshold_cut=120, min_num_points=1000, umbilicus_path=None, sample_ratio=0.1, overlap_y=20, only_originals=False):
    """
    Build main sheet points from the main sheet info patches
    """
    patches_points_list_org = []
    idx = 0
    points_dict = {}

    num_processes = 16  # Adjust based on your system's capabilities
    args_list = []
    for volume_id in main_sheet_info:
        if only_originals:
            main_info_patch_part = {volume_id: main_sheet_info[volume_id]}
        else:
            main_info_patch_part = main_sheet_info
        args_list.append((volume_id, main_info_patch_part, subvolume_size, path, winding_angle_threshold, sample_ratio, only_originals))

    with Pool(num_processes) as p:
        results = list(tqdm(p.imap(process_volume, args_list), total=len(args_list)))
        print(f"Finished processing {len(results)} volumes")

    # Aggregate results from all processes
    all_patches_points_list = []
    all_patches_points_list_org = []
    idx = 0
    for patches_points_list, patches_points_list_org, points_dict_, idx_patch in results:
        all_patches_points_list_org += patches_points_list_org
        for v in points_dict_:
            if v not in points_dict:
                points_dict[v] = {}
            for p in points_dict_[v]:
                points_dict[v][p] = points_dict_[v][p]
                
    main_sheet_points_org_, _, _, _ = concat_patches_points(all_patches_points_list_org)

    print("Cutting main sheet pointcloud...")
    points_dict_cuts = cut_main_dicts(main_sheet_info, points_dict, umbilicus_path, path, winding_angle_threshold=winding_angle_threshold_cut, subvolume_size=subvolume_size, min_num_points=min_num_points, overlap_y=overlap_y)
    print(f"Done with cutting, points_dict_cuts has length: {len(points_dict_cuts)}")
    cut_results = []
    for i, (points_dict, cut_normal) in enumerate(points_dict_cuts):
        main_sheet_info_ = deepcopy(main_sheet_info)
        # Aggregate results from all processes
        all_patches_points_list = []
        debug_scurface_points = []
        debug_scurface_points_whole = []
        idx = 0
        for v in points_dict:
            for p in points_dict[v]:
                points_, patch_points_, normals_, colors_, winding_angles_, _ = points_dict[v][p]
                all_patches_points_list.append((points_, normals_, colors_, winding_angles_))
                debug_scurface_points.append(patch_points_)
                debug_scurface_points_whole.append(points_)
                main_sheet_info_[v][p]["idx"] = idx
                idx += 1
        
        save_surface_ply(np.concatenate(debug_scurface_points, axis=0), np.zeros_like(np.concatenate(debug_scurface_points, axis=0)), path + f"/../debug_surface_points_{i}.ply")
        save_surface_ply(np.concatenate(debug_scurface_points_whole, axis=0), np.zeros_like(np.concatenate(debug_scurface_points_whole, axis=0)), path + f"/../debug_surface_whole_points_{i}.ply")
        
        main_sheet_points_org, main_sheet_normals, main_sheet_winding_angles, unique_ranges = concat_patches_points(all_patches_points_list)
        main_sheet_points = np.zeros_like(main_sheet_points_org)
        main_sheet_points_weight = np.zeros(len(main_sheet_points))
        main_sheet_points_count = np.zeros(len(main_sheet_points))

        for volume_id in tqdm(points_dict):
            for patch_nr in points_dict[volume_id]:
                # Update the main sheet points with the points of the patch
                idx = main_sheet_info_[volume_id][patch_nr]["idx"]
                points, patch_points, patch_normals, patch_colors, patch_winding_angles, patch_points_weights = points_dict[volume_id][patch_nr]
                update_points_in_combined(main_sheet_points_count, patch_points, patch_points_weights, main_sheet_points, main_sheet_points_weight, unique_ranges, idx)
        
        main_sheet_points, main_sheet_normals, main_sheet_winding_angles = update_points_from_weight(main_sheet_points_org, main_sheet_points_count, main_sheet_points, main_sheet_normals, main_sheet_winding_angles, main_sheet_points_weight, subvolume_size)
        cut_results.append(((main_sheet_points, main_sheet_normals, main_sheet_winding_angles), cut_normal))
    
    return main_sheet_points_org_, cut_results

def clean_main_sheet_points(main_sheet_points, normals, winding_angles, min_winding_distance, winding_angle_threshold=120):
    """
    Clean main sheet pointcloud.
    """
    tree = KDTree(main_sheet_points)
    to_remove = set()

    for idx, point in enumerate(main_sheet_points):
        # Query neighbors within a distance of `min_winding_distance`
        indices = tree.query_radius([point], r=min_winding_distance)[0]

        for neighbor_idx in indices:
            # Skip the point itself
            if neighbor_idx == idx:
                continue
            
            angle_diff = abs(winding_angles[idx] - winding_angles[neighbor_idx])
            
            if angle_diff > winding_angle_threshold:
                # Remove the point with the larger winding angle
                if winding_angles[idx] > winding_angles[neighbor_idx]:
                    to_remove.add(idx)
                else:
                    to_remove.add(neighbor_idx)

    # Create masks for filtering
    mask = np.ones(len(main_sheet_points), dtype=bool)
    mask[list(to_remove)] = False
    
    cleaned_points = main_sheet_points[mask]
    cleaned_normals = normals[mask]
    cleaned_winding_angles = winding_angles[mask]

    print(f"Removed {len(to_remove)} points from main sheet pointcloud during winding angle distance enforcement")

    return cleaned_points, cleaned_normals, cleaned_winding_angles

def extract_dict_points_winding(dict_points, start_piece_normal, angle_range, umbilicus_path, overlap_y=2):
    """
    Extract the points around the umbilicus with a given angle range and side of the umbilicus from the point dict. returns: (a new dict with the filtered points, the number of points)
    """
    # Load umbilicus points
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    start_piece_angle = alpha_angles(np.array([start_piece_normal]))

    # dict that stores the output
    output_dict = {}
    nr_total_points = 0
    # Loop over each patch
    for volume_id in tqdm(dict_points):
        for patch_id in dict_points[volume_id]:
            # extract the propper points, if any: add patch to output_dict
            main_sheet_points, points, normals, colors, winding_angles, weights = dict_points[volume_id][patch_id]
            # Filter points by winding angle
            mask = np.logical_and((winding_angles > angle_range[0]), (winding_angles < angle_range[1]))

            main_sheet_points_scaled = scale_points(main_sheet_points, 200.0 / 50.0, axis_offset=0.0) # not correct coordinate system. only needed axis is true
            umbilicus_points_main = umbilicus_xz_at_y(umbilicus_raw_points, main_sheet_points_scaled[:, 1])

            # Get the side of the umbilicus
            umbilicus_normals = umbilicus_points_main - main_sheet_points_scaled

            y_dist_umbilicus = umbilicus_normals[:,2] # carefull, only entry dim 2 is correct
            mask = (y_dist_umbilicus * start_piece_normal[0]) < overlap_y
            mask_other_side = np.logical_and(0.0 < (y_dist_umbilicus * start_piece_normal[0]), (y_dist_umbilicus * start_piece_normal[0]) < overlap_y)
            mask_other_side = mask_other_side[mask]

            # Filter points by umbilicus side
            main_sheet_points = main_sheet_points[mask]
            points = points[mask]
            normals = normals[mask]
            colors = colors[mask]
            winding_angles = winding_angles[mask]
            weights = weights[mask]

            angles = alpha_angles(umbilicus_normals[mask]) - start_piece_angle
            angles_other_side = angles[mask_other_side]
            angles[mask_other_side] = angles_other_side - 180
            angles = angle_to_180(angles)
            winding_angles = winding_angles - angles
            mask = np.logical_and((winding_angles > angle_range[0]), (winding_angles < angle_range[1]))
            main_sheet_points = main_sheet_points[mask]
            points = points[mask]
            normals = normals[mask]
            colors = colors[mask]
            winding_angles = winding_angles[mask]
            weights = weights[mask]

            

            if len(main_sheet_points) > 0:
                if not volume_id in output_dict:
                    output_dict[volume_id] = {}
                output_dict[volume_id][patch_id] = (main_sheet_points, points, normals, colors, winding_angles, weights)
                nr_total_points += len(main_sheet_points)

    return (output_dict, start_piece_normal.copy()), nr_total_points

def extract_points_winding(main_sheet_points, normals, winding_angles, start_piece_normal, angle_range, umbilicus_path, overlap_y=2):
    """
    Extract the points around the umbilicus with a given angle range and side of the umbilicus
    """
    # Filter points by winding angle
    mask = np.logical_and((winding_angles > angle_range[0]), (winding_angles < angle_range[1]))
    main_sheet_points = main_sheet_points[mask]
    normals = normals[mask]
    winding_angles = winding_angles[mask]

    # Load umbilicus points
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    # umbilicus_points = umbilicus(umbilicus_raw_points)

    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_raw_points, main_sheet_points[:, 1])

    # Get the side of the umbilicus
    y_dist_umbilicus = (umbilicus_points_main - main_sheet_points)[:,2]
    mask = (y_dist_umbilicus * start_piece_normal[0]) < overlap_y

    # Filter points by umbilicus side
    main_sheet_points = main_sheet_points[mask]
    normals = normals[mask]
    winding_angles = winding_angles[mask]

    return main_sheet_points, normals, winding_angles

def cut_main_dicts(main_sheet_info, points_dict, umbilicus_path, path, winding_angle_threshold=120, subvolume_size=50, min_num_points=1000, overlap_y=20):
    """
    Cuts the main points inside the dict into half rolled sheets around the umbilicus for surface fitting. returns a list of dicts containing those half rolls.
    """
    start_piece_normal = np.array([1, 0, 0]) # Normal of the start piece
    start_piece_normal_angle = alpha_angles(np.array([start_piece_normal]))[0]
    # get angle offset of patch with patch angle 0
    found = False
    offset_angle_patch_best = float("inf")
    for volume_id in main_sheet_info:
        for patch_nr in main_sheet_info[volume_id]:
            offset_angle_patch = main_sheet_info[volume_id][patch_nr]["offset_angle"]
            main_sheet_patch = (volume_id, patch_nr, offset_angle_patch)
            # Get the patch
            if abs(offset_angle_patch) < offset_angle_patch_best:
                found = True
                patch, _ = build_patch(main_sheet_patch, subvolume_size, path, align_and_flip_normals=False)
                patch_offset_angle = patch["anchor_angles"][0]
                offset_angle_patch_best = offset_angle_patch

    assert found, "Could not find patch with offset angle 0"
    
    # This next line is buggy!
    print(f"Start piece normal angle: {start_piece_normal_angle}, patch offset angle: {patch_offset_angle}")
    rotate_offset = + start_piece_normal_angle + patch_offset_angle + 180
    rotate_offset = angle_to_180(np.array([rotate_offset]))[0]
    print(f"Rotate offset: {rotate_offset}, start piece normal angle: {start_piece_normal_angle}, patch offset angle: {patch_offset_angle}")

    angle_range = rotate_offset + np.array([-winding_angle_threshold, winding_angle_threshold])
    
    main_dict_points_cuts = []
    print(f"Making cut number {len(main_dict_points_cuts)}")
    cut_start, nr_points_cut = extract_dict_points_winding(points_dict, start_piece_normal, angle_range, umbilicus_path, overlap_y=overlap_y)
    if nr_points_cut > min_num_points:
        main_dict_points_cuts.append(cut_start)
    else:
        print(f"Cut number {len(main_dict_points_cuts)} has less than {min_num_points} points ({nr_points_cut}), stopping")

    # Go along the umbilicus in negative angle direction until no more points are found
    angle_range_negative = angle_range.copy()
    piece_normal_negative = start_piece_normal.copy()
    while True:
        angle_range_negative = angle_range_negative - 180
        piece_normal_negative = piece_normal_negative * (-1)
        print(f"Making cut number {len(main_dict_points_cuts)}")
        cut, nr_points_cut = extract_dict_points_winding(points_dict, piece_normal_negative, angle_range_negative, umbilicus_path, overlap_y=overlap_y)
        if nr_points_cut > min_num_points:
            main_dict_points_cuts.append(cut)
        else:
            print(f"Cut number {len(main_dict_points_cuts)} has less than {min_num_points} points ({nr_points_cut}), stopping")
            break

    # Reverse the cuts
    main_dict_points_cuts = main_dict_points_cuts[::-1]

    # Go along the umbilicus in positive angle direction until no more points are found
    angle_range_positive = angle_range.copy()
    piece_normal_positive = start_piece_normal.copy()
    while True:
        angle_range_positive = angle_range_positive + 180
        piece_normal_positive = piece_normal_positive * (-1)
        print(f"Making cut number {len(main_dict_points_cuts)}")
        cut, nr_points_cut = extract_dict_points_winding(points_dict, piece_normal_positive, angle_range_positive, umbilicus_path, overlap_y=overlap_y)
        if nr_points_cut > min_num_points:
            main_dict_points_cuts.append(cut)
        else:
            print(f"Cut number {len(main_dict_points_cuts)} has less than {min_num_points} points ({nr_points_cut}), stopping")
            break

    return main_dict_points_cuts

def cut_main_points(main_sheet_info, main_sheet_points, normals, winding_angles, umbilicus_path, path, winding_angle_threshold=120, subvolume_size=50, min_num_points=1000, overlap_y=20):
    """
    Cuts the main points into half rolled sheets around the umbilicus for surface fitting
    """
    start_piece_normal = np.array([1, 0, 0]) # Normal of the start piece
    start_piece_normal_angle = alpha_angles(np.array([start_piece_normal]))[0]
    # get angle offset of patch with patch angle 0
    found = False
    for volume_id in main_sheet_info:
        for patch_nr in main_sheet_info[volume_id]:
            offset_angle_patch = main_sheet_info[volume_id][patch_nr]["offset_angle"]
            main_sheet_patch = (volume_id, patch_nr, offset_angle_patch)
            # Get the patch
            patch, _ = build_patch(main_sheet_patch, subvolume_size, path, align_and_flip_normals=False)
            patch_offset_angle = patch["anchor_angles"][0]
            if patch_offset_angle == offset_angle_patch:
                found = True
                break
        if found:
            break
    rotate_offset = start_piece_normal_angle - patch_offset_angle
    angle_range = rotate_offset + np.array([-winding_angle_threshold, winding_angle_threshold])
    
    main_points_cuts = []

    cut_start = extract_points_winding(main_sheet_points, normals, winding_angles, start_piece_normal, angle_range, umbilicus_path, overlap_y=overlap_y)
    if len(cut_start[0]) > min_num_points:
        main_points_cuts.append(cut_start)

    # Go along the umbilicus in negative angle direction until no more points are found
    angle_range_negative = angle_range.copy()
    piece_normal_negative = start_piece_normal.copy()
    while True:
        angle_range_negative -= 180
        piece_normal_negative *= -1
        cut = extract_points_winding(main_sheet_points, normals, winding_angles, piece_normal_negative, angle_range_negative, umbilicus_path, overlap_y=overlap_y)
        if len(cut[0]) > min_num_points:
            main_points_cuts.append(cut)
        else:
            break

    # Reverse the cuts
    main_points_cuts = main_points_cuts[::-1]

    # Go along the umbilicus in positive angle direction until no more points are found
    angle_range_positive = angle_range.copy()
    piece_normal_positive = start_piece_normal.copy()
    while True:
        angle_range_positive += 180
        piece_normal_positive *= -1
        cut = extract_points_winding(main_sheet_points, normals, winding_angles, piece_normal_positive, angle_range_positive, umbilicus_path, overlap_y=overlap_y)
        if len(cut[0]) > min_num_points:
            main_points_cuts.append(cut)
        else:
            break

    return main_points_cuts

def split_intersection_edges(mesh, intersection_edges, umbilicus_points, side, axis_indices):
    vertices = np.asarray(mesh.vertices)
    vertices = vertices[:, axis_indices]
    vertices = scale_points(vertices, 1.0, axis_offset=+500)

    intersection_edges_fronts = []
    intersection_edges_backs = []    
    for ie in intersection_edges:
        umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, vertices[ie][:, 1])
        # Get the side of the umbilicus
        y_dist_umbilicus = (umbilicus_points_main - vertices[ie])
        mask_x = y_dist_umbilicus[:,0] * side < 0
        mask = np.all(mask_x)

        if mask:
            # front direction sheet intersection_edges
            intersection_edges_fronts.append(ie)
        else:
            # back direction sheet intersection_edges
            intersection_edges_backs.append(ie)

    return intersection_edges_fronts, intersection_edges_backs


def create_new_triangles(mesh, triangle_case, triangles_on_both_sides, triangle_crossing_edges, intersection_points, bad_triangles):
    vertices = np.array(mesh.vertices).tolist()  # Convert vertices to list for easy appending
    all_triangles = np.array(mesh.triangles)

    # Index from which new vertices start
    start_idx = len(vertices)

    # Append intersection points to vertices list and get their indices
    intersection_indices = []
    for i in range(intersection_points.shape[0]):
        for j in range(intersection_points.shape[1]):
            vertices.append(intersection_points[i, j].tolist())
            intersection_indices.append(start_idx)
            start_idx += 1

    intersection_indices = np.array(intersection_indices).reshape(-1, 2)
    intersection_edges_indices = []
    # Construct the new triangles
    all_triangles_retriangulated = []
    for triangle in range(triangles_on_both_sides.shape[0]):
        edge1 = triangle_crossing_edges[triangle, 0]
        edge2 = triangle_crossing_edges[triangle, 1]

        p1 = edge1[0] if edge1[0] in edge2 else edge1[1]
        p2 = edge1[1] if edge1[0] in edge2 else edge1[0]
        p3 = edge2[1] if edge2[0] in edge1 else edge2[0]

        i12, i13 = intersection_indices[triangle]
        intersection_edges_indices.append((i12, i13))

        # Make new triangles
        if triangle_case[triangle]:
            new_triangles = [[p1, i13, i12]] # only triangle with p1 is on propper side
        else:
            new_triangles = [[i12, i13, p2], [p3, p2, i13]]
        all_triangles_retriangulated += new_triangles

    all_triangles_retriangulated = np.array(all_triangles_retriangulated)
    # Remove old triangles and add new ones
    updated_triangles = np.vstack([all_triangles, all_triangles_retriangulated])
    # Remove bad triangles
    to_remove = np.concatenate([triangles_on_both_sides, bad_triangles])
    updated_triangles = np.delete(updated_triangles, to_remove, axis=0)

    # Update the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(updated_triangles)

    return mesh, np.array(intersection_edges_indices)

def compute_intersection_points(mesh, triangle_crossing_edges, umbilicus_points, axis_indices):
    vertices = np.array(mesh.vertices)
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]
    mesh_points = scale_points(mesh_points, 1.0, axis_offset=+500)

    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])
    y_dist_umbilicus = umbilicus_points_main[:, 2] - mesh_points[:, 2]

    points_A = vertices[triangle_crossing_edges[:, :, 0]]
    points_B = vertices[triangle_crossing_edges[:, :, 1]]
    a_values = y_dist_umbilicus[triangle_crossing_edges[:, :, 0]]
    b_values = y_dist_umbilicus[triangle_crossing_edges[:, :, 1]]

    # Calculating t with reshaped values to avoid shape mismatch
    t = (np.abs(a_values)[:, :, np.newaxis] / 
         (np.abs(a_values)[:, :, np.newaxis] + np.abs(b_values)[:, :, np.newaxis]))
    intersection_points = points_A + t * (points_B - points_A)
    
    return intersection_points

def get_crossing_edges_for_triangles(mesh, triangles_on_both_sides, umbilicus_points, side, axis_indices):
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]
    mesh_points = scale_points(mesh_points, 1.0, axis_offset=+500)

    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])
    y_dist_umbilicus = umbilicus_points_main[:, 2] - mesh_points[:, 2]

    triangles = np.array(mesh.triangles)[triangles_on_both_sides]

    # Determine which vertices of the triangle are on the same side and which is on the other side
    vertex_mask_raw= (y_dist_umbilicus * side < 0)
    vertex_masks = vertex_mask_raw[triangles]
    # Identify the edges that cross the border
    edges_01_mask = vertex_masks[:, 0] != vertex_masks[:, 1]
    edges_12_mask = vertex_masks[:, 1] != vertex_masks[:, 2]
    edges_02_mask = vertex_masks[:, 0] != vertex_masks[:, 2]

    edges_01 = triangles[:, [0, 1]]
    edges_12 = triangles[:, [1, 2]]
    edges_02 = triangles[:, [0, 2]]

    print(f"Shape of edges_01_mask: {edges_01_mask.shape}, edges_01: {edges_01.shape}")

    # Create a boolean array to mask the selected edges
    edge_masks = np.stack([edges_01_mask, edges_12_mask, edges_02_mask], axis=-1)

    # Use the masks to select the edges
    edges = np.stack([edges_01, edges_12, edges_02], axis=1)

    print(f"Shape of edge_masks: {edge_masks.shape}, edges: {edges.shape}")

    # For each triangle, select the two edges that cross the border
    triangle_crossing_edges = edges[edge_masks].reshape(-1, 2, 2)

    print(f"Shape of triangle_crossing_edges: {triangle_crossing_edges.shape}, shape of edges masked {edges[edge_masks].shape}")

    # Extract the vertex mask values for each edge's vertices
    edge_mask_values = vertex_mask_raw[triangle_crossing_edges]
    print(f"Shape of edge_mask_values: {edge_mask_values.shape}")
    
    # Sum the mask values for each edge
    edge_mask_sums = edge_mask_values.sum(axis=2)

    # Check that the sum is 1 for each edge
    valid_edges = (edge_mask_sums == 1).all(axis=1)
    valid_computation = np.all(valid_edges)

    assert valid_computation, "Not all extracted 'side-crossing'edges have exactly one vertex on each side of the umbilicus"

    return triangle_crossing_edges


def triangles_on_both_sides_of_umbilicus(mesh, umbilicus_points, side, axis_indices):
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]
    mesh_points = scale_points(mesh_points, 1.0, axis_offset=+500)
    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])

    # Determine side of each vertex
    y_dist_umbilicus = umbilicus_points_main[:,2] - mesh_points[:,2]
    mask = (y_dist_umbilicus * side) < 0

    triangles = np.array(mesh.triangles)
    triangle_masks = mask[triangles]
    triangles_on_both_sides = np.where(np.logical_and(np.any(triangle_masks, axis=1), 
                                                      np.any(~triangle_masks, axis=1)))[0]
    triangle_case = (np.sum(triangle_masks, axis=1)==1)[triangles_on_both_sides]
    print(f"Shape of triangle_masks: {triangle_masks.shape}, shape of triangles_on_both_sides: {triangles_on_both_sides.shape}, shape of triangle_case: {triangle_case.shape}")
    return triangles_on_both_sides, triangle_case

def triangles_on_wrong_side_of_umbilicus(mesh, umbilicus_points, side, axis_indices):
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]
    mesh_points = scale_points(mesh_points, 1.0, axis_offset=+500)
    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])

    # Determine side of each vertex
    y_dist_umbilicus = umbilicus_points_main[:,2] - mesh_points[:,2]
    mask = (y_dist_umbilicus * side) < 0

    triangles = np.array(mesh.triangles)
    triangle_masks = ~mask[triangles]
    triangles_on_wrong_side = np.where(np.all(triangle_masks, axis=1))[0]
    
    return triangles_on_wrong_side

def trim_mesh(mesh, umbilicus_points, side, axis_indices):
    # extract the propper points, if any: add patch to output_dict
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]

    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])

    # Get the side of the umbilicus
    y_dist_umbilicus = umbilicus_points_main[:,2] - mesh_points[:,2]
    mask = (y_dist_umbilicus * side) < 0

    # Remove the unselected points from the mesh
    mesh.remove_vertices_by_mask(~mask)
    # Heal the mesh
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    print(f"Number of vertices after trimming: {len(mesh.vertices)} vs {len(mesh_points)} before trimming")
    return mesh

def extract_cut_boundaries(boundary_edges, mesh, umbilicus_points, side, axis_indices, umbilicus_distance_threshold=10):
    vertices = np.asarray(mesh.vertices)
    vertices = vertices[:, axis_indices]

    # filter edges with both vertices close enough to the umbilicus line on the indicated side
    filtered_edges = []
    for edge in boundary_edges:
        edge_vertices = vertices[edge]
        umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, edge_vertices[:, 1])
        # Get the side of the umbilicus
        y_dist_umbilicus = (umbilicus_points_main - edge_vertices)
        mask_y = np.abs(y_dist_umbilicus[:,2]) < umbilicus_distance_threshold
        mask_x = y_dist_umbilicus[:,0] * np.sign(y_dist_umbilicus[:,2]) * side < 0
        mask = np.logical_and(mask_x, mask_y)
        if np.sum(mask) == 2:
            filtered_edges.append(edge)

    print(f"Boundary of length {len(boundary_edges)} has a path of length {len(filtered_edges)} remaining close enough to the stitching side.")

    # Flip vertices if verteex 2 is higher than vertex 1
    for i, edge in enumerate(filtered_edges):
        edge_vertices = vertices[edge]
        if edge_vertices[1, 1] > edge_vertices[0, 1]:
            filtered_edges[i] = edge[[1, 0]]

    # order edges from highest z point to lowest z point with largest edge vertex z point as key
    sorted_edges = sorted(filtered_edges, key=lambda edge: -max(vertices[edge][:, 1]))

    # if two edges have overlap (edge1 vertex 2 is smaller than edge2 vertex 1) remove the edge with the larger dist to the umbilicus cut between edge1 vertex 2 and edge2 vertex 1. continue from highest z to lowest z
    if len(sorted_edges) < 2:
        print(f"The mesh does not reach to the boundary and will not be merged on this side. Is this message shown more than twice? if so, there is a problem with mesh subdevision and stitching! Please contact person responsible for ThaumatoAnakalyptor!")
        return []

    # process overlaps
    cleaned_edges = []
    prev_edge = sorted_edges[0]
    for current_edge in sorted_edges[1:]:
        if vertices[prev_edge][1, 1] < vertices[current_edge][0, 1]:  # Check for overlap
            # Calculate distance to umbilicus for both edges
            prev_dist = np.abs(umbilicus_xz_at_y(umbilicus_points, vertices[prev_edge][:, 1])[1, 2] - vertices[prev_edge][1, 2])
            current_dist = np.abs(umbilicus_xz_at_y(umbilicus_points, vertices[current_edge][:, 1])[0, 2] - vertices[current_edge][0, 2])
            
            # Keep the edge closer to the umbilicus
            if prev_dist < current_dist:
                cleaned_edges.append(prev_edge)
            else:
                prev_edge = current_edge
        else:
            cleaned_edges.append(prev_edge)
            prev_edge = current_edge

    # The last edge should be added outside the loop
    cleaned_edges.append(prev_edge)

    # make boundary continuous again
    final_edges = []
    for i in range(len(cleaned_edges)-1):
        final_edges.append(cleaned_edges[i])
        if np.any(cleaned_edges[i][1] != cleaned_edges[i+1][0]):
            final_edges.append(np.array([cleaned_edges[i][1], cleaned_edges[i+1][0]]))
    return final_edges

def visualize_boundary(mesh, boundary_edges, path):
    vertices = np.asarray(mesh.vertices)
    
    # Convert boundary edges to line segments for visualization
    lines = [[edge[0], edge[1]] for edge in boundary_edges]
    colors = [[1, 0, 0] for i in range(len(lines))]  # red
    
    # Create a LineSet from the boundary edges
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
def stitch_meshes(mesh0_tuple, mesh1_tuple, len_meshes):
    """
    Stitch two meshes together based on boundary edges.
    
    mesh0, mesh1: The two meshes to be stitched
    boundary_edges0, boundary_edges1: Touple of lists of boundary edges for mesh1 and mesh2. 
                                      Each edge is represented as a tuple of vertex indices.
    """    
    
    mesh0, boundary_tuple0 = mesh0_tuple
    mesh1, boundary_tuple1 = mesh1_tuple

    boundary_edges0 = boundary_tuple0[1]
    boundary_edges1 = boundary_tuple1[0]
    

    vertices0 = np.asarray(mesh0.vertices)
    vertices1 = np.asarray(mesh1.vertices)

    print(f"Shape of vertices0: {vertices0.shape}, shape of vertices1: {vertices1.shape}, shape of boundary_edges0: {len(boundary_edges0)}, shape of boundary_edges1: {len(boundary_edges1)}")
    if len(boundary_edges0) == 0:
        return mesh1, boundary_tuple1
    if len(boundary_edges1) == 0:
        return mesh0, boundary_tuple0

    # Sort, such that index 0 of each edge is larger in z
    boundary_edges0 = np.array([edge if vertices0[edge[0]][2] > vertices0[edge[1]][2] else [edge[1], edge[0]] for edge in boundary_edges0])
    boundary_edges1 = np.array([edge if vertices1[edge[0]][2] > vertices1[edge[1]][2] else [edge[1], edge[0]] for edge in boundary_edges1])

    # Sort boundary edges based on the y-coordinate of their topmost vertex
    sorted_edges0 = np.array(sorted(boundary_edges0, key=lambda edge: max(vertices0[edge, 2]), reverse=True))
    sorted_edges1 = np.array(sorted(boundary_edges1, key=lambda edge: max(vertices1[edge, 2]), reverse=True))

    # Merging the meshes
    all_vertices = np.vstack([vertices0, vertices1])
    all_triangles = np.vstack([np.asarray(mesh0.triangles), np.asarray(mesh1.triangles) + len(vertices0)])
    sorted_edges1 = sorted_edges1 + len(vertices0)
    return_boundary = (np.array(boundary_tuple0[0]), np.array(boundary_tuple1[1]) + len(vertices0))

    new_triangles = []
    sorted_edges0 = list(sorted_edges0)
    sorted_edges1 = list(sorted_edges1)

    stitch_direction = -1
    # Stitching the meshes
    while sorted_edges0 and sorted_edges1:
        if all_vertices[sorted_edges0[0][0], 2] > all_vertices[sorted_edges1[0][0], 2]:
            sorted_edges0, sorted_edges1 = sorted_edges1, sorted_edges0
            stitch_direction *= -1
        
        v1, v2 = sorted_edges0[0]
        u1, u2 = sorted_edges1[0]
        triangle = [v1, u1, u2] if stitch_direction == 1 else [v1, u2, u1]
        new_triangles.append(triangle)
        
        # Remove the processed edge from sorted_edges1
        sorted_edges1.pop(0)
    
    # Convert vertices and triangles to a single mesh
    all_triangles = np.vstack([all_triangles, new_triangles])
    
    stitched_mesh = o3d.geometry.TriangleMesh()
    stitched_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    stitched_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    
    return stitched_mesh, return_boundary

def remove_triangles_with_short_edges(mesh, threshold=1e-2):
    # Get vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # A function to compute the distance between two points given their indices
    def edge_length(i, j):
        return np.linalg.norm(vertices[i] - vertices[j])
    
    # Find triangles with edges below the threshold
    triangles_to_remove = []
    for idx, tri in enumerate(triangles):
        if edge_length(tri[0], tri[1]) < threshold or \
           edge_length(tri[1], tri[2]) < threshold or \
           edge_length(tri[2], tri[0]) < threshold:
            triangles_to_remove.append(idx)
    print(f"Removing {len(triangles_to_remove)} triangles with short edges")
    # Remove identified triangles
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh.orient_triangles()
    return mesh

def stitch_cuts(cuts_meshes, umbilicus_points, umbilicus_func, mesh_save_path="/home/julian/scroll1_surface_points/half_windings/", continue_meshing=False, winding_direction=1.0, include_boarder=False):
    """
    Stitch the cut meshes together
    """

    # Return early if there is only one mesh
    if len(cuts_meshes) == 1:
        mesh = cuts_meshes[0][0]
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh = remove_triangles_with_short_edges(mesh)
        return mesh
    
    # Axis swap mesh PC
    axis_indices = [1, 2, 0]
    meshes = []
    for i, (mesh, cut_normal) in enumerate(cuts_meshes):
        side = cut_normal[0]

        # test
        crossing_triangles, triangle_case = triangles_on_both_sides_of_umbilicus(mesh, umbilicus_points, side, axis_indices)
        bad_triangles = triangles_on_wrong_side_of_umbilicus(mesh, umbilicus_points, side, axis_indices)
        print(f"Number of crossing triangles: {len(crossing_triangles)}")
        crossing_edges = get_crossing_edges_for_triangles(mesh, crossing_triangles, umbilicus_points, side, axis_indices)
        print(f"Number of crossing edges: {len(crossing_edges)}")
        intersection_points = compute_intersection_points(mesh, crossing_edges, umbilicus_points, axis_indices)
        mesh, intersection_edges = create_new_triangles(mesh, triangle_case, crossing_triangles, crossing_edges, intersection_points, bad_triangles)
        # Get boundary
        boundary_tuple = split_intersection_edges(mesh, intersection_edges, umbilicus_points, side, axis_indices)
        
        mesh.compute_triangle_normals()
        if not continue_meshing:
            save_mesh(mesh, mesh_save_path + f"trimmed_mesh_{i}.obj")

        mesh_cut = mesh
        meshes.append((mesh_cut, boundary_tuple))

    if (len(meshes) > 2) and (not include_boarder):
        meshes = meshes[1:-1]
    if winding_direction < 0.0: # reverse mesh order if winding direction is opposite
        meshes = meshes[::-1]
    while len(meshes) > 1:
        # Stitch mesh 0 and mesh 1 together
        print(f"Stitching meshes {len(meshes)}")
        mesh0_tuple = meshes[0]
        mesh1_tuple = meshes.pop(1)
        meshes[0] = stitch_meshes(mesh0_tuple, mesh1_tuple, len(meshes))

    # Fix the mesh
    mesh, _ = meshes[0]
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh = remove_triangles_with_short_edges(mesh)

    return mesh

def trim_to_scroll(mesh, x_range, y_range, z_range):
    """
    Removes vertices outside of the scroll range in the mesh
    """
    vertices = np.asarray(mesh.vertices)
    mask_x = np.logical_and(vertices[:, 0] > x_range[0], vertices[:, 0] < x_range[1])
    mask_y = np.logical_and(vertices[:, 1] > y_range[0], vertices[:, 1] < y_range[1])
    mask_z = np.logical_and(vertices[:, 2] > z_range[0], vertices[:, 2] < z_range[1])
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

    mesh.remove_vertices_by_mask(~mask)
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh.orient_triangles()

    return mesh

def orient_normals_towards_umbilicus(mesh, umbilicus_points, side, axis_indices):
    # extract the propper points, if any: add patch to output_dict
    mesh_points = np.array(mesh.vertices)
    mesh_points = mesh_points[:, axis_indices]

    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, mesh_points[:, 1])

    # Get the side of the umbilicus
    mesh_umbilicus_vecs = umbilicus_points_main - mesh_points
    mesh_normals = np.array(mesh.vertex_normals)
    
    # Compute dot products for each point-normal pair
    dot_products = np.sum(mesh_normals * mesh_umbilicus_vecs, axis=1)

    # Identify which normals need to be reversed
    to_reverse = dot_products < 0

    # Count the number of normals that need to be reversed
    num_to_reverse = np.sum(to_reverse)
    num_not_to_reverse = len(to_reverse) - num_to_reverse

    # Reverse all normals if more than half need to be reversed
    if num_to_reverse > num_not_to_reverse:
        mesh_normals = -mesh_normals
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(mesh_normals)
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()

    return mesh


def orient_normals_towards_curve(pcd, umbilicus_func):
    # Convert points and normals to numpy arrays
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Get the curve points corresponding to the y-values of the points using umbilicus_func
    curve_points = umbilicus_func(points[:,2])

    print(f"Shape of curve points: {curve_points.shape}")

    # Compute vectors from points to curve points
    direction_to_curve = curve_points - points
    normalized_directions = direction_to_curve / np.linalg.norm(direction_to_curve, axis=1)[:, np.newaxis]

    # Compute dot products for each point-normal pair
    dot_products = np.sum(normalized_directions * normals, axis=1)

    # Identify which normals need to be reversed
    to_reverse = dot_products < 0

    print(f"Shape of normals: {normals.shape}, shape of to_reverse: {to_reverse.shape}")

    # Reverse the necessary normals
    normals[to_reverse] = -normals[to_reverse]

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

def orient_normals_towards_curve_mesh(mesh, umbilicus_func):
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    # Convert points and normals to numpy arrays
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # Get the curve points corresponding to the y-values of the points using umbilicus_func
    curve_points = umbilicus_func(points[:,2])

    print(f"Shape of curve points: {curve_points.shape}")

    # Compute vectors from points to curve points
    direction_to_curve = curve_points - points
    normalized_directions = direction_to_curve / np.linalg.norm(direction_to_curve, axis=1)[:, np.newaxis]

    # Compute dot products for each point-normal pair
    dot_products = np.sum(normalized_directions * normals, axis=1)

    # Identify which normals need to be reversed
    to_reverse = dot_products < 0

    print(f"Shape of normals: {normals.shape}, shape of to_reverse: {to_reverse.shape}")

    # Reverse the necessary normals
    normals[to_reverse] = -normals[to_reverse]

    # Assign the new normals to the mesh
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    return mesh

def generate_normals_toward_curve(mesh, umbilicus_func):
    # Convert points and normals to numpy arrays
    points = np.asarray(mesh.vertices)
    
    # Get the curve points corresponding to the y-values of the points using umbilicus_func
    curve_points = umbilicus_func(points[:,2])

    print(f"Shape of curve points: {curve_points.shape}")

    # Compute vectors from points to curve points
    direction_to_curve = curve_points - points
    normalized_directions = direction_to_curve / np.linalg.norm(direction_to_curve, axis=1)[:, np.newaxis]
    normals = normalized_directions

    # Assign the new normals to the mesh
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    return mesh

def cuda_to_cpu(cuda_mesh):
    device = o3d.core.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)

    mesh.vertex.positions = cuda_mesh.vertex.positions.cpu()
    mesh.vertex.normals = cuda_mesh.vertex.normals.cpu()
    mesh.triangle.indices = cuda_mesh.triangle.indices.cpu()
    mesh.triangle.normals = cuda_mesh.triangle.normals.cpu()

    return mesh

def tensor_to_legacy(mesh):
    legacy_mesh = o3d.geometry.TriangleMesh()
    
    # Transfer vertices
    legacy_mesh.vertices = mesh.vertices
    
    # Transfer triangles
    legacy_mesh.triangles = mesh.triangles
    
    # Transfer normals if they exist
    if mesh.has_vertex_normals():
        legacy_mesh.vertex_normals = mesh.vertex_normals
    
    return legacy_mesh

def o3d_mesh_to_trimesh(mesh):
    """Convert an open3d mesh to a trimesh one."""
    print(f"Length of mesh vertices: {len(mesh.vertices)}, length of mesh triangles: {len(mesh.triangles)}, length of mesh vertex normals: {len(mesh.vertex_normals)}")
    return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))

def trimesh_to_o3d_mesh(mesh):
    """Convert a trimesh mesh to an open3d one."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if len(mesh.vertex_normals) > 0:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals.copy())
    return o3d_mesh

def biggest_connected_component(mesh):
    # Split the mesh into connected components
    components = mesh.split(only_watertight=False)

    # Find the largest component by volume or area or nr of vertices
    largest_component = max(components, key=lambda x: len(x.vertices)) # area: x.area, volume: x.volume, nr of vertices: len(x.vertices)

    return largest_component

def remove_non_manifold_faces(mesh):
    """
    Remove non-manifold faces from a mesh.
    
    Parameters:
    - mesh
    """
    # Identify non manifold edges
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)

    # Remove faces with non manifold edges
    faces_to_remove = []
    for i, face in enumerate(mesh.triangles):
        if any(((edge[0] in face) and (edge[1] in face)) for edge in edges):
            faces_to_remove.append(i)

    print(f"Removing {len(faces_to_remove)} non-manifold faces")

    # Remove the identified faces
    mesh.remove_triangles_by_index(faces_to_remove)

    # Clean up the mesh to remove isolated vertices
    mesh = mesh.remove_unreferenced_vertices()

    return mesh

def get_longest_boundary(boundary_edges):
    print(f"Number of boundary edges: {len(boundary_edges)}")
    # Step 1: Construct the graph
    graph = {}
    for edge in boundary_edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    # Helper function to retrieve a boundary path
    def get_boundary_path(start_vertex):
        path = [start_vertex]
        current_vertex = start_vertex
        while True:
            # There should be 1 or 2 neighbors for each boundary vertex
            # If there's only one, it's an endpoint. Otherwise, we choose the one not already in the path
            next_vertex = None
            for neighbor in graph[current_vertex]:
                if neighbor not in path:
                    next_vertex = neighbor
                    break
            
            if next_vertex is None:
                break
            path.append(next_vertex)
            current_vertex = next_vertex

        return path

    # Step 2: Retrieve all boundary paths
    paths = []
    visited = set()
    for vertex in graph.keys():
        if vertex not in visited:
            path = get_boundary_path(vertex)
            paths.append(path)
            visited.update(path)

    # Step 3: Identify the longest boundary path
    longest_path = max(paths, key=lambda path: len(path))
    # Step 4: Back to edges
    longest_path_edges = []
    for i in range(len(longest_path)):
        longest_path_edges.append(np.array([longest_path[i], longest_path[(i+1)%len(longest_path)]]))
    return longest_path_edges

def get_boundary_edges(mesh):
    triangles = np.asarray(mesh.triangles)
    # Get edges
    edges = np.vstack((triangles[:, :2], 
                    triangles[:, 1:], 
                    triangles[:, [0, 2]]))
    sorted_edges = np.sort(edges, axis=1)

    print(f"Shape of sorted mesh edges: {sorted_edges.shape}")
    
    # Identify boundary edges
    unique_edges, counts = np.unique(sorted_edges, return_counts=True, axis=0)
    boundary_edges = unique_edges[counts == 1]

    return boundary_edges

def remove_branching_boundary_vertices(mesh):
    """
    Remove vertices on the boundary (including holes) with more than two boundary edges.
    
    Parameters:
    - mesh 
    """
    
    while True:
        triangles = np.asarray(mesh.triangles)
        boundary_edges = get_boundary_edges(mesh)
        
        # Identify boundary vertices with more than two boundary edges
        vertex_count = np.bincount(boundary_edges.ravel())
        branching_vertices = np.where(vertex_count > 2)[0]
        
        print(f"Building list of faces to remove for {len(branching_vertices)} branching vertices")

        # Use a mask to identify triangles with branching vertices
        mask = np.isin(triangles, branching_vertices).any(axis=1)

        # Find the indices of the triangles to remove
        faces_to_remove = np.where(mask)[0]

        print(f"Removing {len(faces_to_remove)} faces with branching vertices. Number of branching vertices: {len(branching_vertices)}. Current number of vertices: {len(mesh.vertices)}")
        
        # Remove the identified faces
        mesh.remove_triangles_by_index(faces_to_remove)
        
        # Clean up the mesh to remove isolated vertices
        mesh = mesh.remove_unreferenced_vertices()

        print(f"Number of vertices after removing faces and vertices: {len(mesh.vertices)}")

        if len(branching_vertices) == 0:
            break

    return mesh

def smooth_mesh(mesh, iterations=1):
    mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    return mesh

def subsample_mesh(mesh):
    area_factor_cm = ((0.00324 * 2) ** 2) / 100.0
    print(f"Surface Area is {mesh.get_surface_area() * area_factor_cm:.3f} cm^2")
    area = mesh.get_surface_area() * area_factor_cm
    nr_points = int(area * 100 * 60)
    print(f"Target Nr points is {nr_points}")
    mesh = mesh.simplify_quadric_decimation(nr_points)
    print(f"Number of triangles is {len(mesh.triangles)} and number of vertices is {len(mesh.vertices)}")
    return mesh

def repair_mesh(mesh, voxel_size):
    print(f"Computing vertex normals")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.orient_triangles()

    print(f"Cleaning mesh")
    print(f"Number of vertices after merging: {len(mesh.vertices)}")
    mesh = mesh.remove_duplicated_vertices()
    print(f"Number of vertices after removing duplicates: {len(mesh.vertices)}")
    mesh = mesh.remove_duplicated_triangles()
    print(f"Number of vertices after removing duplicate triangles: {len(mesh.vertices)}")
    mesh = mesh.remove_non_manifold_edges()
    print(f"Number of vertices after removing non-manifold edges: {len(mesh.vertices)}")
    print(f"Number of vertices after removing degenerate triangles: {len(mesh.vertices)}")
    mesh = mesh.remove_unreferenced_vertices()
    print(f"Number of vertices after cleaning: {len(mesh.vertices)}")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()

    mesh = mesh.merge_close_vertices(voxel_size/2)
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_non_manifold_edges()
    indices = mesh.get_self_intersecting_triangles()
    indices = list(np.asarray(indices).flatten())
    mesh.remove_triangles_by_index(indices)
    mesh = mesh.remove_unreferenced_vertices()
    print(f"Number of vertices after smoothing: {len(mesh.vertices)}")

    print(f"Mesh is manifold: {mesh.is_vertex_manifold()}")
    print(f"Number of vertices after hole filling: {len(mesh.vertices)}")

    mesh = remove_branching_boundary_vertices(mesh)

    print(f"Number of vertices after branching boundary removal: {len(mesh.vertices)}")
    print(f"Mesh is manifold: {mesh.is_vertex_manifold()}")

    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()

    return mesh

def points_to_mesh(points, normals, radii=[0.1, 0.2, 0.3, 0.4, 0.5], umbilicus_func=None):
    """
    Convert points to mesh using the Ball Pivoting Algorithm with Open3D.
    """
    # Convert points and normals to an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    voxel_size = 4.0
    print(f"Downsampling with voxel size {voxel_size}, original number of points: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # save pointcloud for debugging
    o3d.io.write_point_cloud("debugging_downsampled_point_cloud.ply", pcd)

    print(f"Resulting number of points after downsampling: {len(pcd.points)}")

    print(f"Estimating normals")
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * 3, max_nn=300))
    print(f"Orienting normals")

    # Orient normals
    pcd = orient_normals_towards_curve(pcd, umbilicus_func)

    print(f"Computing poisson mesh")
    # Use the Poisson surface reconstruction algorithm to create a triangle mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, width=30, scale=1.0001, linear_fit=True)[0:2]
    # Remove the vertices with low density
    vertices_to_remove = densities < np.quantile(densities, 0.0185) # 0.0185
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh = fill_hole_open3d(mesh)

    # Try to make mesh Manifold, not propperly working
    mesh = mesh.merge_close_vertices(1.0)
    mesh = repair_mesh(mesh, voxel_size/3.0)

    area_factor_cm = ((0.00324 * 2) ** 2) / 100.0
    print(f"Surface Area is {mesh.get_surface_area() * area_factor_cm:.3f} cm^2")    
    
    # Return the mesh
    return mesh

def scale_points(points, scale, axis_offset=-500):
    """
    Scale points
    """
    # Scale points
    points_ = points * scale
    # Offset points
    points_ = points_ + axis_offset # WTF SPELUFO? ;)
    # Return points
    return points_

def shuffling_points_axis(points, normals, axis_indices):
    """
    Rotate points by reshuffling axis
    """
    # Reshuffle axis in points and normals
    points = points[:, axis_indices]
    normals = normals[:, axis_indices]
    # Return points and normals
    return points, normals

def save_mesh(mesh, path):
    """
    Save mesh to path in binary format.
    """
    area_factor_cm = ((0.00324 * 2) ** 2) / 100.0
    print(f"Surface Area is {mesh.get_surface_area() * area_factor_cm:.3f} cm^2")
    # Save mesh to path in binary format
    o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)

def load_mesh(path):
    """
    Load mesh from path.
    """
    # Load mesh from path
    mesh = o3d.io.read_triangle_mesh(path)
    # Return mesh
    return mesh

def filter_points(points, normals, max_single_dist=20, min_cluster_size=8000):
    """
    Filter points based on the largest connected component.

    Parameters:
    points (np.ndarray): The point cloud data (Nx3).
    normals (np.ndarray): The normals associated with the points (Nx3).
    max_single_dist (float): The maximum single linkage distance to define connectivity.

    Returns:
    np.ndarray: Points and normals belonging to the largest connected component.
    """

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=max_single_dist)
    clusters = clusterer.fit_predict(points)

    # find all clusters of size 10000 or more
    cluster_labels, cluster_counts = np.unique(clusters, return_counts=True)
    large_clusters = cluster_labels[cluster_counts >= min_cluster_size]
    large_clusters = large_clusters[large_clusters != -1]

    # filter points and normals based on the largest clusters
    mask = np.isin(clusters, large_clusters)
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    print(f"Selected {len(filtered_points)} points from {len(points)} points when filtering pointcloud with max_single_dist {max_single_dist} in single linkage agglomerative clustering")

    return filtered_points, filtered_normals
    

def mesh_cut_computation(path, main_sheet_points, main_sheet_normals, scale, axis_indices, radii, umbilicus_func, idx, enable_filtering=True):
    if enable_filtering:
        print("Filtering mesh ...")
        filtered_points, filtered_normals = filter_points(main_sheet_points, main_sheet_normals)
        save_surface_ply(filtered_points, filtered_normals, path + f"_points_filtered_{idx}.ply")
    else:
        filtered_points = main_sheet_points
        filtered_normals = main_sheet_normals
    save_surface_ply(main_sheet_points, main_sheet_normals, path + f"_points_unfiltered_{idx}.ply")
    if filtered_points.shape[0] < 10000:
        print("Not enough points for meshing")
        return None
    # convert points to mesh
    print("Converting points to mesh")
    mesh = points_to_mesh(filtered_points, filtered_normals, radii=radii, umbilicus_func=umbilicus_func)
    return mesh

def final_mesh_alignment(mesh, umbilicus_points, side, axis_indices):
    mesh.compute_vertex_normals()
    mesh = orient_normals_towards_umbilicus(mesh, umbilicus_points, side, axis_indices)

    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    mesh.orient_triangles()

    return mesh

def fill_hole_open3d(mesh, hole_size=150):
    # Convert to tensor-based TriangleMesh
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tensor_mesh = tensor_mesh.fill_holes(hole_size)
    mesh = tensor_mesh.to_legacy()
    return mesh

def debug_pointcloud_sheet_colored(main_sheet, subvolume_size, path, axis_indices):
    main_sheet_volume_patches = build_main_sheet_patches_list([volume_id for volume_id in main_sheet], main_sheet)
    points, colors = build_main_sheet_volume_from_patches_list(main_sheet_volume_patches, subvolume_size, path, 0.1)
    points = scale_points(points, 200.0 / 50.0, axis_offset=-500)
    points, _ = shuffling_points_axis(points, points, axis_indices)
    filename = path.replace("point_cloud_colorized_test_subvolume_blocks", "debug_volume_colored.ply")
    normals = np.zeros_like(points) + np.array([0, 1, 0])
    save_surface_ply(points, normals, filename, colors=colors)

def triangle_area(v0, v1, v2):
    """Compute the area of a triangle given its vertices."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def longest_triangle_side(v0, v1, v2):
    """Compute the length of the longest side of a triangle given its vertices."""
    return max(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2))

def subdivide_triangle(vertices, tri, area_threshold):
    """Recursively subdivide a triangle if its area is above the threshold."""
    v0 = vertices[tri[0]]
    v1 = vertices[tri[1]]
    v2 = vertices[tri[2]]
    area = triangle_area(v0, v1, v2)

    if area > area_threshold:
        m0 = (v0 + v1 + v2) / 3
        vertices.extend([m0])
        
        children = [
            [tri[0], tri[1], len(vertices)-1],
            [tri[1], tri[2], len(vertices)-1],
            [tri[2], tri[0], len(vertices)-1]
        ]

        # Recursively subdivide child triangles
        result = []
        for child in children:
            result.extend(subdivide_triangle(vertices, child, area_threshold))
        return result
    else:
        return [tri]

def isotropic_remeshing(mesh, area_threshold=700.0):
    vertices = list(np.asarray(mesh.vertices))
    triangles = np.asarray(mesh.triangles)
    new_triangles = []

    for tri in tqdm(triangles):
        new_triangles.extend(subdivide_triangle(vertices, tri, area_threshold))

    print(f"Extended from {len(triangles)} to {len(new_triangles)} triangles")
    # Construct the new mesh with subdivided triangles
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    print(f"Number of vertices: {len(mesh.vertices)}, number of triangles: {len(mesh.triangles)}")
    mesh = mesh.remove_duplicated_vertices()
    print(f"Number of vertices after merging: {len(mesh.vertices)}, number of triangles: {len(mesh.triangles)}")
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    return mesh

def main():
    continue_meshing = False
    sheet_length = 14
    only_originals = True # whether to not also compute winding 2.5D surface to generate interpolated 3d pointcloud instead of the raw data
    sample_ratio = 0.1
    min_num_points = 50000 * sample_ratio
    overlap_y = 20 # overlap of cuts
    subvolume_size = 50.0
    scale = 200.0 / subvolume_size
    axis_indices = [2, 0, 1]
    x_range = (0, 8000)
    y_range = (0, 8000)
    z_range = (0, 14000)
    winding_direction = 1.0 # fixed, no matter the scroll's winding direction, here it IS ALLWAYS 1.0 for stitching based on the angle of the cut
    side = "verso"
    path_base = "/media/julian/SSD4TB/scroll3_surface_points/"
    pointcloud_folder_name = f"point_cloud_colorized_{side}_subvolume_blocks"
    path = path_base + pointcloud_folder_name
    path_ta = f"point_cloud_colorized_{side}_subvolume_main_sheet.ta"

    umbilicus_path = '/media/julian/HDD8TB/PHerc0332.volpkg/volumes/2dtifs_8um_grids/umbilicus.txt'

    import argparse
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Compute ThaumatoAnakalyptor Papyrus Sheets to Mesh')
    parser.add_argument('--side', type=str, help='Side of the scroll (recto or verso)', default=side)
    parser.add_argument('--path_base', type=str, help='Base path for instances (.tar) and main sheet (.ta)', default=path_base)
    parser.add_argument('--path_ta', type=str, help='Papyrus sheet under path_base (with custom .ta ending)', default=path_ta)
    parser.add_argument('--umbilicus_path', type=str, help='Path to umbilicus file', default=umbilicus_path)
    parser.add_argument('--include_boarder', action="store_true", help="Include boarder windings in final mesh generation")

    
    # Take arguments back over
    args = parser.parse_args()
    path_base = args.path_base
    path_ta = args.path_ta
    path_ta = path_base + path_ta
    side = args.side
    include_boarder = args.include_boarder
    pointcloud_folder_name = f"point_cloud_colorized_{side}_subvolume_blocks"
    path = path_base + pointcloud_folder_name
    path_load = os.path.dirname(os.path.dirname(path_base)) + "/" + pointcloud_folder_name
    print(f"Path load: {path_load}")
    umbilicus_path = args.umbilicus_path
    mesh_save_path = path_base + "half_windings/"
    if not os.path.exists(mesh_save_path):
        os.makedirs(mesh_save_path, exist_ok=True)

    print(f"Loading main sheet from {path_ta} from instances path {path}")
    
    winding_angle_threshold = 90
    # winding_angle_threshold_cut = 180
    winding_angle_threshold_cut = 90
    raster_distance = 3.0
    raster_distance_scaled = raster_distance * scale
    radii=[0.5 * raster_distance_scaled, 0.8 * raster_distance_scaled, 1.0 * raster_distance_scaled, 1.8 * raster_distance_scaled, 2.4 * raster_distance_scaled, 3.6 * raster_distance_scaled]

    # Load the umbilicus data
    umbilicus_data = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_data)
    # Red color for umbilicus
    colors = np.zeros_like(umbilicus_points)
    colors[:,0] = 1.0
    # Save umbilicus as a PLY file, for visualization (CloudCompare)
    save_surface_ply(umbilicus_points, np.zeros_like(umbilicus_points), path_load.replace(pointcloud_folder_name, "umbilicus.ply"), colors=colors)
    umbilicus_points = scale_points(umbilicus_points, 1.0, axis_offset=-500)
    umbilicus_points, _ = shuffling_points_axis(umbilicus_points, umbilicus_points, axis_indices)
    save_surface_ply(umbilicus_points, np.zeros_like(umbilicus_points), path_load.replace(pointcloud_folder_name, "umbilicus_scaled_rotated.ply"), colors=colors)
    # scale and swap axis
    umbilicus_data = scale_points(umbilicus_data, 1.0, axis_offset=-500)
    umbilicus_data, _ = shuffling_points_axis(umbilicus_data, umbilicus_data, axis_indices)
    # Define a wrapper function for umbilicus_xz_at_y
    umbilicus_func = lambda z: umbilicus_xy_at_z(umbilicus_data, z)
    # load main sheet info
    print("Loading main sheet info")
    if not continue_meshing:
        main_sheet_info, volume_blocks_scores = load_main_sheet(path=path_load, path_ta=path_ta, sample_ratio_score=None, sample_ratio=sample_ratio, add_display_points=False) # Only load the main sheet information. (patch angle, normal and score)

        # build main sheet pointcloud
        print("Building main sheet pointcloud")
        main_sheet_points_org, cut_results = build_main_sheet_points_interpolted(main_sheet_info, subvolume_size=subvolume_size, path=path_load, winding_angle_threshold=winding_angle_threshold, winding_angle_threshold_cut=winding_angle_threshold_cut, min_num_points=min_num_points, umbilicus_path=umbilicus_path, sample_ratio=sample_ratio, overlap_y=overlap_y, only_originals=only_originals)

        main_sheet_points_org = scale_points(main_sheet_points_org, scale, axis_offset=-500)

        main_sheet_points_org, _ = shuffling_points_axis(main_sheet_points_org, main_sheet_points_org, axis_indices)
        save_surface_ply(main_sheet_points_org, main_sheet_points_org, path + "_points_org.ply")

        meshes = []
        meshing_beginning_windings = True
        for cut_nr, (cut_result, cut_normal) in enumerate(cut_results):
            main_sheet_points, main_sheet_normals, main_sheet_winding_angles = cut_result
            save_surface_ply(main_sheet_points, main_sheet_normals, path + f"_cut_{cut_nr}.ply")
            # scale points back to original size
            print("Scaling points back to original size")
            main_sheet_points = scale_points(main_sheet_points, scale, axis_offset=-500)

            # rotate points back to original orientation
            print("Rotating points back to original orientation")
            main_sheet_points, main_sheet_normals = shuffling_points_axis(main_sheet_points, main_sheet_normals, axis_indices)
            
            mesh = mesh_cut_computation(path, main_sheet_points, main_sheet_normals, scale, axis_indices, radii, umbilicus_func, cut_nr)
            # Exit as soon as there is a half winding without a mesh
            if mesh is None: # no mesh was created
                if meshing_beginning_windings:
                    continue
                else:
                    break
            meshing_beginning_windings = False # found first valid mesh, if a later winding invalid mesh is found, stop meshing since mesh cuts not continuous

            # Trim to scroll size
            mesh_range = np.array([np.min(main_sheet_points_org, axis=0), np.max(main_sheet_points_org, axis=0)])
            additional_range = np.array([[-5, -5, 0], [5, 5, 0]])
            mesh_range += additional_range
            mesh_range = np.clip(mesh_range, 0, np.array([x_range[1], y_range[1], z_range[1]]))
            mesh = trim_to_scroll(mesh, mesh_range[:,0], mesh_range[:,1], mesh_range[:,2])

            save_mesh(mesh, path + f"_cut_{cut_nr}.obj")
            meshes.append((mesh, cut_normal))
    else:
        print("Continue meshing")
        meshes = [load_mesh(path + f"_cut_{i}.obj") for i in range(sheet_length)]
        start_piece_normal = np.array([1, 0, 0])
        for i in range(sheet_length):
            meshes[i] = (meshes[i], start_piece_normal * ((-1)**(i)))
    # Stitch the cut meshes together
    mesh = stitch_cuts(meshes, load_xyz_from_file(umbilicus_path), umbilicus_func, mesh_save_path=mesh_save_path, continue_meshing=continue_meshing, winding_direction=winding_direction, include_boarder=include_boarder)
    mesh = orient_normals_towards_curve_mesh(mesh, umbilicus_func)
    save_mesh(mesh, path + "_raw" + ".obj")

    mesh = subsample_mesh(mesh)
    mesh = find_degenerated_triangles_and_delete(mesh)
    mesh = find_degenerated_triangles_and_delete(mesh)
    mesh = isotropic_remeshing(mesh)
    mesh = orient_normals_towards_curve_mesh(mesh, umbilicus_func)
    # generate normals as vector between point and umbilicus
    mesh = generate_normals_toward_curve(mesh, umbilicus_func)
    # save mesh
    print("Saving mesh")
    save_mesh(mesh, path + ".obj")

if __name__ == "__main__":
    main()
