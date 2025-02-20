### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import os
import shutil
import open3d as o3d
#import tarfile
import py7zr
import tarfile
import h5py
import time
# import colorcet as cc
# surface points extraction
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.distributed as dist
from tqdm import tqdm
import atexit

# show cuda devices
print(torch.cuda.device_count())
print(torch.cuda.current_device())
# show name of current device
print(torch.cuda.get_device_name(torch.cuda.current_device()))

import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from .mask3d.inference import to_surfaces, init, preprocess_points, get_model

import json
import argparse

from .surface_fitting_utilities import get_vector_mean, rotation_matrix_to_align_z_with_v, optimize_sheet
from .grid_to_pointcloud import load_xyz_from_file, umbilicus, umbilicus_xz_at_y, fix_umbilicus_recompute

def get_optimized_rotation_matrix(angles):
    """
    Computes the optimized 3D rotation matrix using extrinsic Euler angles.

    Parameters:
    angles (tuple): Rotation angles (Rx, Ry, Rz) in degrees.

    Returns:
    numpy.ndarray: The 3x3 rotation matrix.
    """
    # Convert angles to radians
    theta_x, theta_y, theta_z = np.radians(angles)

    # Rotation matrix around X-axis
    R_x = np.array([
        [1,  0,               0],
        [0,  np.cos(theta_x), -np.sin(theta_x)],
        [0,  np.sin(theta_x), np.cos(theta_x)]
    ])

    # Rotation matrix around Y-axis
    R_y = np.array([
        [np.cos(theta_y),  0, np.sin(theta_y)],
        [0,               1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Rotation matrix around Z-axis
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z),  0],
        [0,               0,                1]
    ])

    # Compute the final extrinsic rotation matrix: R = Rz * Ry * Rx
    R_final = np.dot(R_z, np.dot(R_y, R_x))

    return R_final


def transform_to_new_system(R, points):
    """
    Transforms points from the original coordinate system to the new coordinate system.

    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix.
    points (numpy.ndarray): Nx3 array of points in the original coordinate system.

    Returns:
    numpy.ndarray: Transformed points in the new coordinate system.
    """
    return np.dot(points, R.T)  # Apply rotation


def transform_to_original_system(R, points):
    """
    Transforms points from the new coordinate system back to the original coordinate system.

    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix.
    points (numpy.ndarray): Nx3 array of points in the new coordinate system.

    Returns:
    numpy.ndarray: Transformed points back in the original coordinate system.
    """
    return np.dot(points, R)  # Apply inverse rotation (R.T is the inverse of a rotation matrix)

def load_ply(filename, main_drive="", alternative_drives=[]):
    """
    Load point cloud data from a .ply file.
    """
    # Check that the file exists
    i = 0
    filename_temp = filename
    while i < len(alternative_drives) and not os.path.isfile(filename_temp):
        filename_temp = filename.replace(main_drive, alternative_drives[i])
        i += 1
    filename = filename_temp
    assert os.path.isfile(filename), f"File {filename} not found."

    # Load the file and extract the points and normals
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # Extract additional features
    colors = np.asarray(pcd.colors)

    return points, normals, colors

def save_surface_ply(surface_points, normals, colors, score, distance, coeff, n, filename):
    # Create an Open3D point cloud object and populate it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float16))

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save as a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    
    # Save metadata as a JSON
    metadata = {
        'score': float(score),
        'distance': float(distance),
        'coeff': [float(item) for item in list(coeff)], # Convert numpy array to list for JSON serialization
        'n': int(n),
    }
    
    # Construct metadata filename
    base_dir, base_filename = os.path.split(filename)
    base_filename_without_extension = os.path.splitext(base_filename)[0]
    metadata_filename = os.path.join(base_dir, f"metadata_{base_filename_without_extension}.json")
    
    with open(metadata_filename, 'w') as metafile:
        json.dump(metadata, metafile)

def save_block_ply_args(args):
    # print("save_block_ply_args")
    save_block_ply(*args)

def save_block_ply(block_points, block_normals, block_colors, block_scores, block_name, score_threshold=0.5, distance_threshold=10.0, n=4, alpha=1000.0, slope_alpha=0.1, post_process=True, block_distances_precomputed=None, block_coeffs_precomputed=None, check_exist=True, use_7z=True):
    # Check if 7z file exists
    # print(f"Saving {block_name}")
    if check_exist and os.path.exists(block_name + '.7z'):
        return
    # Check if tar exists
    if check_exist and os.path.exists(block_name + '.tar'):
        return

    # Save to a temporary file first to ensure data integrity
    temp_block_name = block_name + "_temp"
    # print(f"Saving {temp_block_name}")

    # Check if folder exists
    if not os.path.exists(block_name):
        # post-process the surfaces
        if post_process:
            block_points, block_normals, block_colors, block_scores, block_distances, block_coeffs = post_process_surfaces(block_points, block_normals, block_colors, block_scores, score_threshold=score_threshold, distance_threshold=distance_threshold, n=n, alpha=alpha, slope_alpha=slope_alpha)
        else:
            assert (block_distances_precomputed is not None) and (block_coeffs_precomputed is not None), "block_distances_precomputed and block_coeffs_precomputed must be provided if post_process=False"
            block_distances = block_distances_precomputed
            block_coeffs = block_coeffs_precomputed

        if sum([len(block) for block in block_points]) < 10:
            # Save empty tar just to indicate that it was computed
            block_points = []
        
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(temp_block_name), exist_ok=True)

        # Clean folder (if it exists)
        if os.path.exists(block_name):
            for file in os.listdir(block_name):
                os.remove(os.path.join(block_name, file))

        if os.path.exists(temp_block_name):
            for file in os.listdir(temp_block_name):
                os.remove(os.path.join(temp_block_name, file))

        # Save each surface instance
        for i in range(len(block_points)):
            if len(block_points[i]) < 10:
                continue
            save_surface_ply(block_points[i], block_normals[i], block_colors[i], block_scores[i], block_distances[i], block_coeffs[i], n, os.path.join(temp_block_name, f"surface_{i}.ply"))

    # Delete the temporary 7z file if it exists
    if os.path.exists(temp_block_name + '.7z'):
        os.remove(temp_block_name + '.7z')
    # Delete the temporary tar file if it exists
    if os.path.exists(temp_block_name + '.tar'):
        os.remove(temp_block_name + '.tar')

    used_name_block = block_name if os.path.exists(block_name) else temp_block_name

    archive_name_type = temp_block_name + '.7z' if use_7z else temp_block_name + '.tar'
    block_name_type = block_name + '.7z' if use_7z else block_name + '.tar'
    if use_7z:
        # Create the 7z archive
        with py7zr.SevenZipFile(archive_name_type, 'w') as archive:
            try:
                archive.writeall(used_name_block, '')
            except Exception as e:
                print(e)
                print(f"Error writing {used_name_block} to {temp_block_name}.7z. e: {e}")
    else:
        # Tar the temp folder without including the 'temp' name inside the tar
        with tarfile.open(archive_name_type, 'w') as tar:
            for root, _, files in os.walk(used_name_block):
                for file in files:
                    full_file_path = os.path.join(root, file)
                    arcname = full_file_path[len(used_name_block) + 1:]
                    tar.add(full_file_path, arcname=arcname)

    # Remove the temp folder
    try:
        shutil.rmtree(used_name_block)
    except Exception as e:
        print(e)
        print(f"Error removing {used_name_block}")

    # Rename the 7z archive to the original filename (without '_temp')
    try:
        os.rename(archive_name_type, block_name_type)
        #print(f"Saved {block_name}.7z")
    except Exception as e:
        print(e)
        print(f"Error renaming {temp_block_name}.7z to {block_name}.7z")
        
def save_block_h5_args(args):
    # print("save_block_h5_args")
    save_block_h5(*args)

def save_block_h5(h5f, block_points, block_normals, block_colors, block_scores, block_name, 
                  score_threshold=0.5, distance_threshold=10.0, 
                  n=4, alpha=1000.0, slope_alpha=0.1, post_process=True, 
                  block_distances_precomputed=None, block_coeffs_precomputed=None, 
                  check_exist=True):
    """
    Saves point cloud data into an HDF5 file using the block_name as the group identifier.
    """
    group_name = os.path.basename(block_name)

    # Post-process the data if required
    if post_process:
        block_points, block_normals, block_colors, block_scores, block_distances, block_coeffs = post_process_surfaces(
            block_points, block_normals, block_colors, block_scores, 
            score_threshold=score_threshold, distance_threshold=distance_threshold, 
            n=n, alpha=alpha, slope_alpha=slope_alpha)
    else:
        assert (block_distances_precomputed is not None) and (block_coeffs_precomputed is not None), \
            "block_distances_precomputed and block_coeffs_precomputed must be provided if post_process=False"
        block_distances = block_distances_precomputed
        block_coeffs = block_coeffs_precomputed

    # Skip if too few points
    if sum(len(block) for block in block_points) < 10:
        print(f"Skipping block {block_name} due to insufficient points.")
        return

    # check if group exists
    if group_name in h5f:
        # overwrite
        print(f"Overwriting {group_name}")
        del h5f[group_name]
    # Create a group for the block
    grp = h5f.create_group(group_name)

    for nr_surface in range(len(block_points)):
        if len(block_points[nr_surface]) < 10:
            continue
        # new group in the block group
        surface_grp = grp.create_group(f"surface_{nr_surface}")
        surface_grp.create_dataset("points", data=block_points[nr_surface], compression="lzf") #, compression="gzip", compression_opts=8)
        surface_grp.create_dataset("normals", data=block_normals[nr_surface], compression="lzf") #, compression="gzip", compression_opts=8)
        surface_grp.create_dataset("colors", data=block_colors[nr_surface], compression="lzf") #, compression="gzip", compression_opts=8)
        surface_grp.create_dataset("coeffs", data=np.array(block_coeffs[nr_surface]), compression="lzf") #, compression="gzip", compression_opts=8)
        
        # Store metadata attributes
        surface_grp.attrs["score_threshold"] = score_threshold
        surface_grp.attrs["distance_threshold"] = distance_threshold
        surface_grp.attrs["n"] = n
        surface_grp.attrs["alpha"] = alpha
        surface_grp.attrs["slope_alpha"] = slope_alpha
        surface_grp.attrs["scores"] = block_scores[nr_surface]
        surface_grp.attrs["distances"] = block_distances[nr_surface]
            
def post_process_surfaces(surfaces, surfaces_normals, surfaces_colors, scores, score_threshold=0.5, distance_threshold=10.0, n=4, alpha = 1000.0, slope_alpha = 0.1):
    indices = [] # valid surfaces
    coeff_list = [] # coefficients of surfaces
    distances_list = [] # distances of surfaces
    for i in range(len(surfaces)): # loop over all surface of block
        if len(surfaces[i]) < 10: # too small surface
            continue
        if scores[i] < score_threshold: # low score
            continue

        # Calculate the normal vector of the surface
        v = get_vector_mean(surfaces_normals[i])
        R = rotation_matrix_to_align_z_with_v(v) # rotation matrix to align z-axis with normal vector
        coeff, _, points_mask, sheet_distance = optimize_sheet(surfaces[i], R, n, max_iters=2, alpha=alpha, slope_alpha=slope_alpha) # fit sheet to points
        if sheet_distance > distance_threshold: # sheet is too far away from lots of points
            continue
        indices.append(i) # take valid surfaces
        coeff_list.append(coeff) # save coefficients
        distances_list.append(sheet_distance) # save distances
        # mask sheet
        surfaces[i] = surfaces[i][points_mask]
        surfaces_normals[i] = surfaces_normals[i][points_mask]
        surfaces_colors[i] = surfaces_colors[i][points_mask]
    # Select valid surfaces
    surfaces = [surfaces[i] for i in indices]
    surfaces_normals = [surfaces_normals[i] for i in indices]
    surfaces_colors = [surfaces_colors[i] for i in indices]
    scores = [scores[i] for i in indices]
    
    #  # Get the Glasbey colormap
    # glasbey_cmap = cc.cm.glasbey
    # num_colors = len(indices)
    # for i in range(len(indices)):
    #     color = glasbey_cmap(i/num_colors)[:3]
    #     surfaces_colors[i][:] = color

    return surfaces, surfaces_normals, surfaces_colors, scores, distances_list, coeff_list

def normalize_volume_scale(points, grid_block_size=200):
    # Normalize volume to size 50x50x50
    points = 50.0 * points / grid_block_size
    return points

def load_single_ply(ply_file, grid_block_size, main_drive="", alternative_drives=[]):
    try:
        # Load volume
        points_, normals_, colors_ = load_ply(ply_file, main_drive, alternative_drives)
        # normalize size
        points_ = normalize_volume_scale(points_, grid_block_size=grid_block_size)
        return points_, normals_, colors_
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def load_plys(src_folder, main_drive, alternative_drives, start, size, grid_block_size=200, num_processes=3, load_multithreaded=True, executor=None):
    path_template = "cell_yxz_{:03}_{:03}_{:03}.ply"
    ply_files = []
    for x in range(start[0]-2, start[0]+size[0]+1):
        for y in range(start[1]-2, start[1]+size[1]+1):
            for z in range(start[2]-2, start[2]+size[2]+1):
                ply_files.append(os.path.join(src_folder, path_template.format(x,y,z)))

    if executor and load_multithreaded:
        # Prepare the tasks
        tasks = [(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files]
        # Schedule the tasks and collect the futures
        futures = [executor.submit(load_single_ply, *task) for task in tasks]
        # Wait for the futures to complete and collect the results
        results = [future.result() for future in futures]
    elif load_multithreaded:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(load_single_ply, [(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files])
    else:
        results = [load_single_ply(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files]

    # Filter out None results
    results = [res for res in results if res is not None]

    if len(results) == 0:
        return None

    # Unzip the results to get points, normals, and colors lists
    points, normals, colors = zip(*results)

    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    colors = np.concatenate(colors, axis=0)

    # Randomly shuffle the points (for data looking more like the training data during instance prediction with mask3d)
    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)
    points = points[indices]
    normals = normals[indices]
    colors = colors[indices]

    return points, normals, colors

def load_pc_start(src_folder, main_drive, alternative_drives, start, grid_block_size=200, load_multithreaded=True, executor=None):
    path_template = "cell_yxz_{:03}_{:03}_{:03}.ply"
    indices = start_to_cell_indices(start, grid_block_size)
    points = []
    normals = []
    colors = []

    ply_files = []
    for index in indices:
        ply_file = os.path.join(src_folder, path_template.format(index[0], index[1], index[2]))
        ply_files.append(ply_file)

    if executor and load_multithreaded:
        # Prepare the tasks
        tasks = [(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files]
        # Schedule the tasks and collect the futures
        futures = [executor.submit(load_single_ply, *task) for task in tasks]
        # Wait for the futures to complete and collect the results
        results = [future.result() for future in futures]
    elif load_multithreaded:
        with Pool(processes=3) as pool:
            results = pool.starmap(load_single_ply, [(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files])
    else:
        results = [load_single_ply(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files]

    # Filter out None results
    results = [res for res in results if res is not None]

    if len(results) == 0:
        return None
    
    # Unzip the results to get points, normals, and colors lists
    points, normals, colors = zip(*results)

    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    colors = np.concatenate(colors, axis=0)

    # Randomly shuffle the points (for data looking more like the training data during instance prediction with mask3d)
    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)
    points = points[indices]
    normals = normals[indices]
    colors = colors[indices]

    points, normals, colors = remove_duplicate_points_normals(points, normals, colors)

    return points, normals, colors
        
def extract_subvolume(points, normals, colors, angles, start, size=50):
    """
    Extract a subvolume of size "size" from "points" starting at "start".
    """
    # Size is int
    if isinstance(size, int):
        size = np.array([size, size, size])

    start_ = np.array(start).copy() + np.array(size).copy() // 2
    max_d = max(size) / (2.0 ** 0.5)  # maximum distance from the center of the subvolume
    # remove points that are too far away
    mask = np.all(np.abs(points.copy() - start_) <= max_d, axis=1)
    points = points[mask]
    normals = normals[mask]
    colors = colors[mask]
    angles = angles[mask]

    # random rotation
    random_angles = np.random.randint(0, 360, size=(3))
    R = get_optimized_rotation_matrix(tuple(random_angles))
    # translate to system 2
    points_2 = transform_to_new_system(R, points.copy() - start_) + start_
    # points_2 = points.copy()
    # Remove entries that have np.any(size[a] == 0)
    mask = size > 0
    start = start[mask]
    size = size[mask]

    # Check if entries exist after removing entries
    if start.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    # any dimension of size is 0
    if np.any(size <= 0):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    # Find all points in the subvolume
    mask = np.ones(points_2.shape[0], dtype=bool)
    for i in range(start.shape[0]):
        mask_i = np.logical_and(points_2[:,i] >= start[i], points_2[:,i] < start[i] + size[i])
        mask = np.logical_and(mask, mask_i)

    # No points in the subvolume
    if np.all(~mask):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    subvolume_points = points_2[mask]
    original_subvolume_points = points[mask]
    subvolume_normals = normals[mask]
    subvolume_colors = colors[mask]
    subvolume_angles = angles[mask]

    return subvolume_points, original_subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles

def get_rotated_original_bounds(start, size):
    # retrieve the original bounds of the rotated subvolume
    start_ = np.array(start).copy() + np.array(size).copy() // 2
    max_d = max(size) / (2.0 ** 0.5)  # maximum distance from the center of the subvolume
    # create a bounding box around the subvolume
    min_bound = start_ - max_d
    max_bound = start_ + max_d
    return min_bound, max_bound

def position_to_pc_cell(position, grid_block_size=200):
    return tuple((np.array(position) // grid_block_size).astype(int))

def start_to_cell_indices(start, grid_block_size=200):
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    # print(f"Start: {start}")
    min_bound, max_bound = get_rotated_original_bounds(start, grid_block_size)
    start_indx = position_to_pc_cell(min_bound, grid_block_size)
    stop_indx = position_to_pc_cell(max_bound, grid_block_size)
    indices = []
    for x in range(start_indx[0], stop_indx[0]+1):
        for y in range(start_indx[1], stop_indx[1]+1):
            for z in range(start_indx[2], stop_indx[2]+1):
                indices.append([x, y, z])
    # print(f"Indices: {indices}")
    return indices

def remove_duplicate_points(points):
    """
    Remove duplicate points from a point cloud.
    """
    unique_points = np.unique(points, axis=0)
    return unique_points

def remove_duplicate_points_normals(points, normals, colors=None, angles=None):
    """
    Remove duplicate points from a point cloud.
    """
    unique_points, indices = np.unique(points, axis=0, return_index=True)
    unique_normals = normals[indices]
    if colors is None and angles is None:
        return unique_points, unique_normals
    elif colors is None:
        unique_angles = angles[indices]
        return unique_points, unique_normals, unique_angles
    elif angles is None:
        unique_colors = colors[indices]
        return unique_points, unique_normals, unique_colors
    else:
        unique_colors = colors[indices]
        unique_angles = angles[indices]
        return unique_points, unique_normals, unique_colors, unique_angles

def filter_umilicus_distance(start_list, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold, grid_block_size=200):
    # Load umbilicus points
    umbilicus_raw_points = load_xyz_from_file(umbilicus_points_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)

    # loop over all start points
    start_list_filtered = []
    for start in tqdm(start_list, desc="Filtering start list"):
        #check if start block is existing
        empty_block = True
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    if os.path.exists(os.path.join(path, folder, f"cell_yxz_{start[0]+i:03}_{start[1]+j:03}_{start[2]+k:03}.ply")):
                        empty_block = False
                        break
                if not empty_block:
                    break
            if not empty_block:
                break
        if empty_block:
            continue
        # calculate umbilicus point
        block_point = np.array(start) * grid_block_size + grid_block_size//2
        umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
        umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
        umbilicus_point[2] = block_point[2] # if umbilicus is not in the same y plane as the block, set umbilicus y to block y (linear cast down along z axis)
        umbilicus_normal = block_point - umbilicus_point

        # distance umbilicus_normal
        distance = np.linalg.norm(umbilicus_normal)
        if (umbilicus_distance_threshold <= 0) or ((umbilicus_distance_threshold > 0) and (distance < umbilicus_distance_threshold)):
            start_list_filtered.append(start)
    
    return start_list_filtered

def update_progress_file(progress_file, indices, config):
    # Update the progress file
    with open(progress_file, 'w') as file:
        json.dump({'indices': indices, 'config': config}, file)

def build_start_list(start, stop, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold):
    start_list = []
    print(f"Start: {start}, Stop: {stop}, Size: {size}")
    for x in range(start[0], stop[0], size[0]):
        for y in range(start[1], stop[1], size[1]):
            for z in range(start[2], stop[2], size[2]):
                start_list.append([x, y, z])
    print(f"Length of start list: {len(start_list)}")
    start_list = filter_umilicus_distance(start_list, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)
    print(f"Length of start list after filtering: {len(start_list)}")
    return start_list
    
def to_surfaces_args(args):
    # print("to_surfaces_args")
    return to_surfaces(*args)

import os
import json
import h5py
import numpy as np
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist

##########################################
#  Modified Prediction Writer Class      #
##########################################
class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, 
                 path="/media/julian/FastSSD/scroll3_surface_points", 
                 folder="point_cloud_colorized", 
                 dest="/media/julian/HDD8TB/scroll3_surface_points", 
                 main_drive="", alternative_drives=[], 
                 fix_umbilicus=True, umbilicus_points_path="", 
                 start=[0, 0, 0], stop=[16, 17, 29], size=[3, 3, 3], 
                 umbilicus_distance_threshold=1500, score_threshold=0.5, 
                 batch_size=4, gpus=1, num_processes=4,
                 use_h5=False, use_7z=False,
                 overlap_denumerator=3):  # added overlap_denumerator parameter
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.num_threads = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(processes=self.num_threads)  # Initialize once

        self.path = path
        self.folder = folder
        self.dest = dest
        self.main_drive = main_drive
        self.alternative_drives = alternative_drives
        self.fix_umbilicus = fix_umbilicus
        self.umbilicus_points_path = umbilicus_points_path
        self.start = start
        self.stop = stop
        self.size = size
        self.umbilicus_distance_threshold = umbilicus_distance_threshold
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.gpus = gpus
        self.use_h5 = use_h5
        self.use_7z = use_7z
        self.overlap_denumerator = overlap_denumerator

        # Create a fixed thread pool of 8 threads.
        self.executor = ThreadPoolExecutor(max_workers=8)
        # Initialize a semaphore to limit pending tasks to 100.
        self.task_semaphore = threading.Semaphore(100)

        self.start_list = build_start_list(start, stop, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)
        # Now total number of tasks equals number of start_list entries * (overlap_denumerator^3)
        self.num_tasks = len(self.start_list) * (self.overlap_denumerator ** 3)
        self.to_compute_indices = list(range(self.num_tasks))
        self.computed_indices = []
        self.progress_file = os.path.join(dest, "progress.json")
        self.config = {"path": path, "folder": folder, "dest": dest, "main_drive": main_drive, 
                       "alternative_drives": alternative_drives, "fix_umbilicus": fix_umbilicus, 
                       "umbilicus_points_path": umbilicus_points_path, "start": start, "stop": stop, 
                       "size": size, "umbilicus_distance_threshold": umbilicus_distance_threshold, 
                       "score_threshold": score_threshold, "batch_size": batch_size, "gpus": gpus,
                       "overlap_denumerator": overlap_denumerator}
        self._load_progress()

        # For HDF5 saving: these will be lazily initialized on the first call.
        self.final_filename = None           # Final HDF5 filename (always os.path.dirname(names_batch[0]) + ".h5")
        self.intermediate_filenames = None   # List of 4 intermediate filenames (one per thread)
        self.intermediate_handles = None     # Open HDF5 file handles for each thread
        self.interm_locks = None             # One lock per intermediate file
       
    def _load_progress(self):
        nr_total_indices = len(self.to_compute_indices)
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as file:
                progress = json.load(file)
                if 'config' in progress:
                    progress_config = progress['config']
                    # Remove batch_size and gpus for comparison.
                    if 'batch_size' in progress_config:
                        del progress_config['batch_size']
                    if 'batch_size' in self.config:
                        del self.config['batch_size']
                    if 'gpus' in progress_config:
                        del progress_config['gpus']
                    if 'gpus' in self.config:
                        del self.config['gpus']
                    if progress_config != self.config:
                        print("Progress file found but with different config. Overwriting.")
                    else:
                        print("Progress file found with same config. Resuming computation.")
                        if 'indices' in progress:
                            self.computed_indices = progress['indices']
                            self.to_compute_indices = list(set(self.to_compute_indices) - set(self.computed_indices))
                            print(f"Resuming computation. {len(self.to_compute_indices)} blocks of {nr_total_indices} left to compute.")
                        else:
                            print("No progress file found.")

    def submit_task(self, fn, *args, **kwargs):
        # Block if there are already 100 pending tasks.
        self.task_semaphore.acquire()
        # Submit the wrapped function.
        future = self.executor.submit(self._wrapped_fn, fn, *args, **kwargs)
        return future
       
    def _wrapped_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            # Ensure that the semaphore is released even if the task fails.
            self.task_semaphore.release()

    def write_on_batch_end(self, trainer:"pl.Trainer", pl_module:"pl.LightningModule", 
                             prediction, batch_indices, batch, batch_idx:int, dataloader_idx:int) -> None:
        """
        In this multi-GPU setting each GPU writes its own batch.
        """
        # Unpack your batch info as defined in your dataset:
        items_pytorch, points_batch, normals_batch, colors_batch, names_batch, indxs = batch
        
        # Determine world size (if not distributed, this will be 1)
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Prepare lists to hold gathered objects from each GPU.
        # The order is by global rank: index 0 will be rank 0, index 1 will be rank 1, etc.
        all_indices = [None] * world_size
        # Gather predictions and batch objects from all processes.
        dist.all_gather_object(all_indices, indxs)
        
        # Only execute file writing on the master process.
        if trainer.is_global_zero:
            for indxs in all_indices:
                indxs = list(indxs)
                # Filter out negative indices.
                # indxs = [i for i in indxs if i >= 0]
                self.computed_indices = list(set(self.computed_indices) + set(indxs))
            update_progress_file(self.progress_file, self.computed_indices, self.config)
            
        if prediction and len(prediction) > 0:
            self.post_process(
                prediction,
                items_pytorch,
                points_batch,
                normals_batch,
                colors_batch,
                names_batch,
                use_multiprocessing=False,  # Change to True if preferred.
                use_h5=self.use_h5,
                use_7z=self.use_7z
            )
    
    def post_process(self, res, items_pytorch, points_batch, normals_batch, colors_batch, names_batch, 
                     use_multiprocessing=False, distance_threshold=10.0, n=4, alpha=1000.0, 
                     slope_alpha=0.1, use_h5=True, use_7z=False):
        if res is None:
            print("batch_inference result is None")
            res = [{"pred_classes": []}] * len(items_pytorch)
        else:
            res = list(res.values())

        if use_multiprocessing:
            res = self.pool.map(to_surfaces_args, [(points_batch[i], normals_batch[i], colors_batch[i], res[i])
                                                    for i in range(len(points_batch))])
            surfaces, surfaces_normals, surfaces_colors, scores = zip(*res)
        else:
            surfaces, surfaces_normals, surfaces_colors, scores = to_surfaces(points_batch, normals_batch, colors_batch, res)

        if use_h5:
            # Determine the final HDF5 filename (this is always the same for a given dataset).
            final_h5_filename = os.path.dirname(names_batch[0]) + ".h5"
            if self.final_filename is None:
                self.final_filename = final_h5_filename
                dir_name = os.path.dirname(final_h5_filename)
                basename = os.path.basename(final_h5_filename)
                base, ext = os.path.splitext(basename)

                gpu_rank = dist.get_rank() if dist.is_initialized() else 0
                # Create 4 intermediate filenames (one per thread)
                self.intermediate_filenames = [os.path.join(dir_name, "gpu_threads_h5s", f"{base}_gpu{gpu_rank}_thread{i}{ext}") for i in range(4)]
                try:
                    os.makedirs(os.path.dirname(self.intermediate_filenames[0]))
                except FileExistsError:
                    pass
                # Open each intermediate file in append mode.
                self.intermediate_handles = [h5py.File(fname, "a") for fname in self.intermediate_filenames]
                for intermediate_handle in self.intermediate_handles:
                    atexit.register(intermediate_handle.close)
                
                # Create one lock per intermediate file.
                self.interm_locks = [threading.Lock() for _ in range(4)]
            
            # Group the blocks (each block corresponds to one surface from the batch) by thread.
            # Round-robin assignment: block i is assigned to thread index = i % 4.
            random_offset = np.random.randint(4)
            tasks_by_thread = {i: [] for i in range(4)}
            for i in range(len(surfaces)):
                thread_idx = (i + random_offset) % 4
                tasks_by_thread[thread_idx].append((surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], names_batch[i]))
            
            # For each thread that has tasks, submit a single task that writes all its assigned blocks.
            for thread_idx, tasks in tasks_by_thread.items():
                if tasks:
                    self.submit_task(self.async_save_batch_h5, thread_idx, tasks, distance_threshold, n, alpha, slope_alpha)
        else:
            # single threaded version
            for i in range(len(surfaces)):
                save_block_ply(surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], names_batch[i], self.score_threshold, distance_threshold, n, alpha, slope_alpha, False, [0] * len(surfaces[i]), [[]] * len(surfaces[i]), True, use_7z)
            
            # # Group the blocks (each block corresponds to one surface from the batch) by thread.
            # # Round-robin assignment: block i is assigned to thread index = i % 4.
            # random_offset = np.random.randint(4)
            # tasks_by_thread = {i: [] for i in range(4)}
            # for i in range(len(surfaces)):
            #     thread_idx = (i + random_offset) % 4
            #     tasks_by_thread[thread_idx].append((surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], names_batch[i]))
            
            # # For each thread that has tasks, submit a single task that writes all its assigned blocks.
            # for thread_idx, tasks in tasks_by_thread.items():
            #     if tasks:
            #         self.submit_task(self.async_save_batch_tar, tasks, distance_threshold, n, alpha, slope_alpha, use_7z)
    
    def async_save_batch_h5(self, thread_idx, tasks, distance_threshold, n, alpha, slope_alpha):
        """
        This method runs in one of the fixed threads.
        It writes all blocks in 'tasks' to the intermediate HDF5 file corresponding to thread_idx.
        Each task in tasks is a tuple: (surf, norm, col, scr, nname).
        """
        with self.interm_locks[thread_idx]:
            for (surf, norm, col, scr, nname) in tasks:
                save_block_h5(self.intermediate_handles[thread_idx],
                              surf, norm, col, scr, nname,
                              self.score_threshold, distance_threshold, n, alpha, slope_alpha,
                              False, [0] * len(surf), [[]] * len(surf))

    def async_save_batch_tar(self, tasks, distance_threshold, n, alpha, slope_alpha, use_7z):
        """
        This method runs in one of the fixed threads.
        It writes all blocks in 'tasks' to the intermediate tar file.
        Each task in tasks is a tuple: (surf, norm, col, scr, nname).
        """
        for (surf, norm, col, scr, nname) in tasks:
            save_block_ply(surf, norm, col, scr, nname,
                           self.score_threshold, distance_threshold, n, alpha, slope_alpha, False,
                           [0] * len(surf), [[]] * len(surf), True, use_7z)
    
    def finalize(self):
        """
        At the end of the epoch, wait for all threads to finish.
        Then (only on global rank 0) merge the 4 intermediate HDF5 files into the final file.
        """
        # Wait for all asynchronous tasks to complete.
        self.executor.shutdown(wait=True)
        # Close all intermediate HDF5 file handles.
        if self.intermediate_handles is not None:
            for h5f in self.intermediate_handles:
                h5f.close()
        # Only merge on global rank 0.
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            # Find all intermediate files in the h5 directory.
            h5_dir = os.path.dirname(self.intermediate_filenames[0])
            self.intermediate_filenames = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5")]
            with h5py.File(self.final_filename, "w") as h5f_final:
                for fname in tqdm(self.intermediate_filenames, desc="Merging intermediate files"):
                    if os.path.exists(fname):
                        with h5py.File(fname, "r") as h5f_in:
                            for group in tqdm(h5f_in, desc=f"Merging {os.path.basename(fname)}"):
                                h5f_in.copy(h5f_in[group], h5f_final, name=group)
            print(f"Final merged HDF5 file: {self.final_filename}")

##########################################
#      Modified Dataset Class            #
##########################################
class PointCloudDataset(Dataset):
    def __init__(self, path="/media/julian/FastSSD/scroll3_surface_points", folder="point_cloud_colorized", dest="/media/julian/HDD8TB/scroll3_surface_points", main_drive="", alternative_drives=[], fix_umbilicus=True, umbilicus_points_path="", 
                 start=[0, 0, 0], stop=[16, 17, 29], size = [3, 3, 3], umbilicus_distance_threshold=1500, score_threshold=0.5, batch_size=4, gpus=1, num_processes=3, recompute=False, rotate=False, overlap_denumerator=3,
                 use_h5=False, use_7z=False, update_saved_index_coords=False):
        # Set rotation (if any)
        self.rotate = rotate
        self.R = np.eye(3) if not self.rotate else get_optimized_rotation_matrix((45, 45, 45))
        self.overlap_denumerator = overlap_denumerator
        self.recompute = recompute
        self.path = path
        self.folder = folder
        self.dest = dest
        self.main_drive = main_drive
        self.alternative_drives = alternative_drives
        self.fix_umbilicus = fix_umbilicus
        self.umbilicus_points_path = umbilicus_points_path
        self.start = start
        self.stop = stop
        self.size = size
        self.umbilicus_distance_threshold = umbilicus_distance_threshold
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.gpus = gpus
        # Thread pool for any parallel tasks
        self.executor = ThreadPoolExecutor(max_workers=num_processes)

        # Load umbilicus points (and old version if requested)
        umbilicus_raw_points = load_xyz_from_file(umbilicus_points_path)
        self.umbilicus_points = umbilicus(umbilicus_raw_points)
        if fix_umbilicus:
            umbilicus_path_old = umbilicus_points_path.replace("umbilicus", "umbilicus_old")
            umbilicus_raw_points_old = load_xyz_from_file(umbilicus_path_old)
            self.umbilicus_points_old = umbilicus(umbilicus_raw_points_old)
        else:
            self.umbilicus_points_old = None

        # Build the list of start positions for blocks
        self.start_list = build_start_list(start, stop, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)
        # Total number of dataset items equals number of start positions * (overlap_denumerator^3)
        total_items = len(self.start_list) * (self.overlap_denumerator ** 3)

        # Update configuration (including the overlap parameter)
        self.config = {"path": path, "folder": folder, "dest": dest, "main_drive": main_drive, 
                       "alternative_drives": alternative_drives, "fix_umbilicus": fix_umbilicus, "umbilicus_points_path": umbilicus_points_path, 
                       "start": start, "stop": stop, "size": size, "umbilicus_distance_threshold": umbilicus_distance_threshold, 
                       "score_threshold": score_threshold, "batch_size": batch_size, "gpus": gpus,
                       "overlap_denumerator": overlap_denumerator}

        progress_path = os.path.join(dest, "progress.json")
        # Load saved progress if available and if not forcing a recompute
        if os.path.exists(progress_path) and (not self.recompute):
            with open(progress_path, 'r') as file:
                progress = json.load(file)
                if 'config' in progress:
                    progress_config = progress['config']
                    # Remove keys that we don’t use for config comparison
                    if 'batch_size' in progress_config:
                        del progress_config['batch_size']
                    if 'batch_size' in self.config:
                        del self.config['batch_size']
                    if 'gpus' in progress_config:
                        del progress_config['gpus']
                    if 'gpus' in self.config:
                        del self.config['gpus']
                    # Compare with the current config (which we set later)
                    if progress_config != self.config:
                        print("Progress file found but with different config. Overwriting.")
                        self.computed_indices = []
                    else:
                        print("Progress file found with same config. Resuming computation.")
                        if 'indices' in progress:
                            saved = progress['indices']
                            print(f"Found {len(saved)} computed items out of {total_items}.")
                            self.computed_indices = saved
                        else:
                            self.computed_indices = []
                else:
                    self.computed_indices = []
        else:
            self.computed_indices = []

        # If the update flag is set, re-check each dataset item to see if it is truly computed.
        if update_saved_index_coords:
            print("Update progress flag set: checking saved computations using index coordinates.")
            dest_path = os.path.join(dest, folder)
            new_computed_indices = []
            # Iterate over all subvolume items
            for idx in tqdm(range(total_items), desc="Checking saved computations"):
                if self.is_saved_index_coords(idx, self.size, dest_path, self.main_drive, self.alternative_drives):
                    new_computed_indices.append(idx)
            if set(new_computed_indices) != set(self.computed_indices):
                print("Saved computations differ from progress file. Updating progress file.")
                self.computed_indices = new_computed_indices
                update_progress_file(progress_path, self.computed_indices, self.config if hasattr(self, 'config') else {})
        # Compute remaining indices (those items not yet computed)
        self.remaining_indices = sorted(list(set(range(total_items)) - set(self.computed_indices)))
        print(f"Resuming: {len(self.remaining_indices)} items left out of {total_items}.")

        # Pass overlap_denumerator to the writer as well.
        self.writer = MyPredictionWriter(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, umbilicus_distance_threshold, score_threshold, batch_size, gpus, num_processes, use_h5=use_h5, use_7z=use_7z, overlap_denumerator=overlap_denumerator)
    
    def get_writer(self):
        return self.writer
    
    def is_saved_index_coords(self, idx, size, path, main_drive, alternative_drives, subvolume_size=50):
        """
        Check if the subvolume corresponding to dataset index `idx` has been saved.
        """
        size = np.array(size).copy() + 1  # for the -1 in the range
        if isinstance(subvolume_size, int):
            subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])
        block_idx = idx // (self.overlap_denumerator ** 3)
        sub_idx = idx % (self.overlap_denumerator ** 3)
        x = self.start_list[block_idx]  # start coordinate for this block
        # Swap axes
        axis_swap = [0, 2, 1]
        start_index = np.array(x)[axis_swap]
        # Make blocks of size '50x50x50'
        indx_count = 0
        for xx in range(int((size[0] - 1) * self.overlap_denumerator)):
            x_coord = start_index[0] * subvolume_size[0] + (xx * subvolume_size[0] // self.overlap_denumerator)
            for y in range(int((size[1] - 1) * self.overlap_denumerator)):
                y_coord = start_index[1] * subvolume_size[1] + (y * subvolume_size[1] // self.overlap_denumerator)
                for z in range(int((size[2] - 1) * self.overlap_denumerator)):
                    indx_count += 1
                    if indx_count != sub_idx + 1:
                        continue
                    z_coord = start_index[2] * subvolume_size[2] + (z * subvolume_size[2] // self.overlap_denumerator)
                    start_coord = np.array([x_coord, y_coord, z_coord])
                    block_name = path + f"_subvolume_blocks/{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}" # nice ordering in the folder
                    block_name_tar = block_name + ".tar"
                    block_name_tar_alternatives = []
                    for alternative_drive in alternative_drives:
                        block_name_tar_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".tar")
                    block_name_zip = block_name + ".7z"
                    block_name_zip_alternatives = []
                    for alternative_drive in alternative_drives:
                        block_name_zip_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".7z")
                    # Computation exists
                    if os.path.exists(block_name_tar) or any([os.path.exists(name) for name in block_name_tar_alternatives]) or os.path.exists(block_name_zip) or any([os.path.exists(name) for name in block_name_zip_alternatives]):
                        return True
        # No computation exists
        return False

    def create_batches(self, idx__, path, src_folder, start, size, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive, alternative_drives, subvolume_size=50, load_multithreaded=True, executor=None):
        # Size is int
        if isinstance(subvolume_size, int):
            subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

        # Iterate over all subvolumes
        start = np.array(start)
        # Swap axes
        axis_swap = [0, 2, 1]
        axis_inverse_swap = [0, 2, 1]
        start_index = start[axis_swap]
        # size is size = original size + 1, we want original size * 2 subvolume blocks that overlap
        subvolumes_points = []
        original_subvolumes_points = []
        subvolumes_normals = []
        subvolumes_colors = []
        start_coords = []
        block_names = []
        
        # Make blocks of size '50x50x50'
        indx_count = 0
        for xx in range(int((size[0] - 1) * self.overlap_denumerator)):
            x_coord = start_index[0] * subvolume_size[0] + (xx * subvolume_size[0] // self.overlap_denumerator)
            for y in range(int((size[1] - 1) * self.overlap_denumerator)):
                y_coord = start_index[1] * subvolume_size[1] + (y * subvolume_size[1] // self.overlap_denumerator)
                for z in range(int((size[2] - 1) * self.overlap_denumerator)):
                    indx_count += 1
                    if indx_count != idx__ + 1:
                        continue
                    z_coord = start_index[2] * subvolume_size[2] + (z * subvolume_size[2] // self.overlap_denumerator)
                    start_coord = np.array([x_coord, y_coord, z_coord])
                    block_name = path + f"_subvolume_blocks/{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}" # nice ordering in the folder
                    block_name_tar = block_name + ".tar"
                    block_name_tar_alternatives = []
                    for alternative_drive in alternative_drives:
                        block_name_tar_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".tar")
                    block_name_zip = block_name + ".7z"
                    block_name_zip_alternatives = []
                    for alternative_drive in alternative_drives:
                        block_name_zip_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".7z")
                    
                    res_pc = load_pc_start(src_folder, main_drive, alternative_drives, start_coord[axis_inverse_swap] * 200.0 / 50.0, grid_block_size=200, load_multithreaded=load_multithreaded, executor=executor)
                    if res_pc is None:
                        continue
                    points, normals, colors = res_pc
                    subvolume_points, original_subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles = extract_subvolume(points, normals, colors, colors, start=start_coord, size=subvolume_size) 
                    if len(subvolume_points) < 10:
                        continue
                    
                    subvolumes_points.append(subvolume_points)
                    original_subvolumes_points.append(original_subvolume_points)
                    subvolumes_normals.append(subvolume_normals)
                    subvolumes_colors.append(subvolume_colors)
                    start_coords.append(start_coord)
                    block_names.append(block_name)

        assert indx_count == self.overlap_denumerator ** 3, f"Index count is {indx_count} and should be exactly {self.overlap_denumerator ** 3}"
        return subvolumes_points, original_subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names

    def precompute(self, idx__, index, start, size, path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points, umbilicus_points_old, use_multiprocessing, executor=None):
        src_path = os.path.join(path, folder)
        dest_path = os.path.join(dest, folder)
        size = np.array(size).copy() + 1 # +1 because we want to include the last subvolume for tiling operation
        
        subvolumes_points, original_subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names = self.create_batches(idx__, dest_path, src_path, start, size, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive, alternative_drives, load_multithreaded=use_multiprocessing, executor=executor)
        return subvolumes_points, original_subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names

    def __len__(self):
        # Return the number of remaining subvolume items (each subvolume is now one dataset item)
        return len(self.remaining_indices)

    def __getitem__(self, idx):
        # Get the actual dataset index (in the full range) from the remaining indices.
        actual_idx = self.remaining_indices[idx]
        block_idx = actual_idx // (self.overlap_denumerator ** 3)
        sub_idx = actual_idx % (self.overlap_denumerator ** 3)
        x = self.start_list[block_idx]

        res = self.precompute(sub_idx, block_idx, x, self.size, self.path, self.folder, self.dest, self.main_drive, self.alternative_drives, self.fix_umbilicus, self.umbilicus_points, self.umbilicus_points_old, False, None)
        points_batch, original_points_batch, normals_batch, colors_batch, start_coords, names_batch = res

        # Filter out subvolumes with too few points
        valid_indices = [i for i, points in enumerate(points_batch) if points.shape[0] > 100]
        points_batch = [points_batch[i] for i in valid_indices]
        original_points_batch = [original_points_batch[i] for i in valid_indices]
        normals_batch = [normals_batch[i] for i in valid_indices]
        colors_batch = [colors_batch[i] for i in valid_indices]
        names_batch = [names_batch[i] for i in valid_indices]
        # Deep copy the points
        coords_batch = [np.copy(points) for points in points_batch]
        # Translate the points so that the volume starts at (0,0,0)
        min_coord_batch = [np.min(coords, axis=0) for coords in coords_batch]
        coords_batch = [coords - min_coord_batch[i] for i, coords in enumerate(coords_batch)]

        items_pytorch = [preprocess_points(c) for c in coords_batch]
        # Return the batch along with the dataset index (actual_idx) so that progress can be updated for each item.
        return items_pytorch, original_points_batch, normals_batch, colors_batch, names_batch, actual_idx

# Custom collation function
def custom_collate_fn(batches):
    # Initialize containers for the aggregated items
    items_pytorch_agg = []
    points_batch_agg = []
    normals_batch_agg = []
    colors_batch_agg = []
    names_batch_agg = []
    indxs_agg = []

    # Loop through each batch and aggregate its items
    for batch in batches:
        items_pytorch, points_batch, normals_batch, colors_batch, names_batch, indxs = batch
        
        items_pytorch_agg.extend(items_pytorch)
        points_batch_agg.extend(points_batch)
        normals_batch_agg.extend(normals_batch)
        colors_batch_agg.extend(colors_batch)
        names_batch_agg.extend(names_batch)
        indxs_agg.append(indxs)
        
    # Return a single batch containing all aggregated items
    return items_pytorch_agg, points_batch_agg, normals_batch_agg, colors_batch_agg, names_batch_agg, indxs_agg

def pointcloud_inference(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, umbilicus_distance_threshold, score_threshold, batch_size, gpus, recompute, overlap_denumerator, use_h5, use_7z, update_saved_index_coords):
    init()
    model = get_model()
    # model = torch.nn.DataParallel(model)
    # model.to('cuda')  # Move model to GPU
    # compile the model for faster inference
    # model = torch.compile(model) # too low torch version

    dataset = PointCloudDataset(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, 
                                umbilicus_distance_threshold, score_threshold, batch_size, gpus, recompute=recompute, overlap_denumerator=overlap_denumerator, use_h5=use_h5, use_7z=use_7z,
                                update_saved_index_coords=update_saved_index_coords)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=24, prefetch_factor=5)  # Adjust num_workers as per your system

    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], gpus=gpus, strategy="ddp")

    print("Start prediction")
    # Run prediction
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Prediction done")
    if use_h5:
        writer.finalize()
        print("Finalize done")

def main():
    side = "_verso" # actually recto
    path = "/media/julian/SSD2/scroll3_surface_points"
    folder = f"point_cloud_colorized{side}"
    fix_umbilicus = False
    umbilicus_points_path = "/media/julian/SSD4TB/PHerc0332.volpkg/volumes/2dtifs_8um_grids/umbilicus.txt"
    dest = f"/media/julian/SSD4TB/scroll3_surface_points"
    main_drive = "SSD2"
    alternative_ply_drives = ["FastSSD", "HDD8TB"]
    # umbilicus_distance_threshold=-1 #2250 scroll 1
    # umbilicus_distance_threshold=2200 # scroll 3
    umbilicus_distance_threshold = -1
    score_threshold=0.10
    batch_size = 5
    gpus = -1
    pointcloud_size = 1
    overlap_denumerator = 3

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compute surface patches from pointcloud.")
    parser.add_argument("--path", type=str, help="Base path to the data", default=path)
    parser.add_argument("--folder", type=str, help="Folder containing the point cloud data", default=folder)
    parser.add_argument("--dest", type=str, help="Folder to save the output data", default=dest)
    parser.add_argument("--main_drive", type=str, help="Main drive that contains the input data", default=main_drive)
    parser.add_argument("--alternative_ply_drives", type=str, nargs='+', help="Alternative drives that may contain additional input data in the same path naming sceme. To split data over multiple drives if needed.", default=alternative_ply_drives)
    parser.add_argument("--umbilicus_path", type=str, help="Path to the umbilicus.txt", default=umbilicus_points_path)
    parser.add_argument("--max_umbilicus_dist", type=float, help="Maximum distance between the umbilicus and blocks that should be computed. -1.0 for no distance restriction", default=umbilicus_distance_threshold)
    parser.add_argument("--fix_umbilicus", action='store_true', help="Flag, recompute all close to the updated umbilicus (make sure to also save the old umbilicus.txt as umbilicus_old.txt)")
    parser.add_argument("--score_threshold", type=float, help="Minimum score for a surface to be saved", default=score_threshold)
    parser.add_argument("--pointcloud_size", type=int, help="Size of the pointcloud", default=pointcloud_size)
    parser.add_argument("--batch_size", type=int, help="Batch size for Mask3D", default=batch_size)
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use", default=gpus)
    parser.add_argument("--recompute", action='store_true', help="Flag, recompute all blocks, even if they already exist")
    parser.add_argument("--z_min", type=int, help="Minimum slice index for computation", default=-500)
    parser.add_argument("--z_max", type=int, help="Maximum slice index for computation", default=50000)
    parser.add_argument("--overlap_denumerator", type=int, help="Denominator for overlap of subvolumes", default=overlap_denumerator)
    parser.add_argument("--use_h5", action='store_true', help="Flag, use HDF5 as storage (Tar default).")
    parser.add_argument("--use_7z", action='store_true', help="Flag, use 7z as storage (Tar default).")
    parser.add_argument("--update_progress", action='store_true', help="Flag, update the progress file with the saved index coordinates.")

    # Parse the arguments
    args, unknown = parser.parse_known_args()
    path = args.path
    folder = args.folder
    dest = args.dest
    main_drive = args.main_drive
    alternative_ply_drives = args.alternative_ply_drives
    umbilicus_points_path = args.umbilicus_path
    umbilicus_distance_threshold = args.max_umbilicus_dist
    fix_umbilicus = args.fix_umbilicus
    score_threshold = args.score_threshold
    pointcloud_size = args.pointcloud_size
    batch_size = args.batch_size
    gpus = args.gpus
    recompute = args.recompute
    z_start = int((args.z_min +500)//200)
    z_end = int((args.z_max +500)//200)
    overlap_denumerator = args.overlap_denumerator
    use_h5 = args.use_h5
    use_7z = args.use_7z
    update_saved_index_coords = args.update_progress

    # Compute the surface patches
    pointcloud_inference(path, folder, dest, main_drive, alternative_ply_drives, fix_umbilicus, umbilicus_points_path, [0, 0, z_start], [100, 100, z_end], 
                         [pointcloud_size, pointcloud_size, pointcloud_size], umbilicus_distance_threshold, score_threshold, batch_size, gpus, recompute, overlap_denumerator,
                        use_h5, use_7z, update_saved_index_coords)
    
if __name__ == "__main__":
    main()

# iostat -dx 1
# python3 -m ThaumatoAnakalyptor.pointcloud_to_instances --path /scroll_pcs --dest /scroll_pcs --umbilicus_path /scroll_pcs/umbilicus.txt --main_drive "" --alternative_ply_drives "" "" --batch_size 4 --gpus 1 --z_min 3000 --z_max 4000
