### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import os
import open3d as o3d
#import tarfile
import py7zr
import time
# import colorcet as cc
# surface points extraction
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

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
        'coeff': [float(item) for item in coeff.tolist()], # Convert numpy array to list for JSON serialization
        'n': int(n),
    }
    
    # Construct metadata filename
    base_dir, base_filename = os.path.split(filename)
    base_filename_without_extension = os.path.splitext(base_filename)[0]
    metadata_filename = os.path.join(base_dir, f"metadata_{base_filename_without_extension}.json")
    
    with open(metadata_filename, 'w') as metafile:
        json.dump(metadata, metafile)

def save_block_ply_args(args):
    save_block_ply(*args)

def save_block_ply(block_points, block_normals, block_colors, block_scores, block_name, score_threshold=0.5, distance_threshold=10.0, n=4, alpha=1000.0, slope_alpha=0.1, post_process=True, block_distances_precomputed=None, block_coeffs_precomputed=None, check_exist=True):
    # Check if 7z file exists
    if check_exist and os.path.exists(block_name + '.7z'):
        return

    # Save to a temporary file first to ensure data integrity
    temp_block_name = block_name + "_temp"

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
            return
        
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

    used_name_block = block_name if os.path.exists(block_name) else temp_block_name

    # Create the 7z archive
    with py7zr.SevenZipFile(temp_block_name + '.7z', 'w') as archive:
        archive.writeall(used_name_block, '')

    # Remove the temp folder
    for root, dirs, files in os.walk(used_name_block, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    
    try:
        os.rmdir(used_name_block)
    except Exception as e:
        print(e)
        print(f"Error removing {used_name_block}")

    # Rename the 7z archive to the original filename (without '_temp')
    try:
        os.rename(temp_block_name + '.7z', block_name + '.7z')
        #print(f"Saved {block_name}.7z")
    except Exception as e:
        print(e)
        print(f"Error renaming {temp_block_name}.7z to {block_name}.7z")

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
    for x in range(start[0], start[0]+size[0]):
        for y in range(start[1], start[1]+size[1]):
            for z in range(start[2], start[2]+size[2]):
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

def extract_subvolume(points, normals, colors, angles, start, size=50):
    """
    Extract a subvolume of size "size" from "points" starting at "start".
    """
    # Size is int
    if isinstance(size, int):
        size = np.array([size, size, size])

    # Ensure start is in shape (a, 3)
    if start.shape == (3,):
        start = np.expand_dims(start, axis=0)

    # Ensure size is in shape (a, 3)
    if size.shape == (3,):
        size = np.expand_dims(size, axis=0)

    # Remove entries that have np.any(size[a] == 0)
    mask = np.all(size > 0, axis=1)
    start = start[mask]
    size = size[mask]

    # Check if entries exist after removing entries
    if start.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    # any dimension of size is 0
    if np.any(size <= 0):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    # Find all points in the subvolume
    mask = np.zeros(points.shape[0], dtype=bool)
    for i in range(start.shape[0]):
        mask_i = np.all(np.logical_and(points >= start[i], points < start[i] + size[i]), axis=1)
        mask = np.logical_or(mask, mask_i)


    # No points in the subvolume
    if np.all(~mask):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    subvolume_points = points[mask]
    subvolume_normals = normals[mask]
    subvolume_colors = colors[mask]
    subvolume_angles = angles[mask]

    return subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles

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
    for start in start_list:
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
    return to_surfaces(*args)


class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, path="/media/julian/FastSSD/scroll3_surface_points", folder="point_cloud_colorized", dest="/media/julian/HDD8TB/scroll3_surface_points", main_drive="", alternative_drives=[], fix_umbilicus=True, umbilicus_points_path="", start=[0, 0, 0], stop=[16, 17, 29], size = [3, 3, 3], umbilicus_distance_threshold=1500, score_threshold=0.5, batch_size=4, gpus=1, num_processes=3):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        num_threads = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(processes=num_threads)  # Initialize the pool once

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
        # Initialize the ThreadPoolExecutor with the desired number of threads
        self.executor = ThreadPoolExecutor(max_workers=num_processes)

        self.start_list = build_start_list(start, stop, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)

        self.num_tasks = len(self.start_list)
        self.to_compute_indices = range(self.num_tasks)
        self.computed_indices = []
        self.progress_file = os.path.join(dest, "progress.json")
        self.config = {"path": path, "folder": folder, "dest": dest, "main_drive": main_drive, "alternative_drives": alternative_drives, "fix_umbilicus": fix_umbilicus, "umbilicus_points_path": umbilicus_points_path, "start": start, "stop": stop, "size": size, "umbilicus_distance_threshold": umbilicus_distance_threshold, "score_threshold": score_threshold, "batch_size": batch_size, "gpus": gpus}
        nr_total_indices = len(self.to_compute_indices)
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as file:
                progress = json.load(file)
                if 'config' in progress:
                    if progress['config'] != self.config:
                        print("Progress file found but with different config. Overwriting.")
                    else:
                        print("Progress file found with same config. Resuming computation.")
                        if 'indices' in progress:
                            self.computed_indices = progress['indices']
                            self.to_compute_indices = list(set(self.to_compute_indices) - set(self.computed_indices))
                            print(f"Resuming computation. {len(self.to_compute_indices)} blocks of {nr_total_indices} left to compute.")
                        else:
                            print("No progress file found.")

    def write_on_predict(self, predictions: list, batch_indices: list, dataloader_idx: int, batch, batch_idx: int, dataloader_len: int):
        # Example: Just print the predictions
        print(predictions)
        print("On predict")
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if prediction is None:
            # print("Prediction is None")
            return
        # print(f"On batch end, len: {len(prediction)}")
        if len(prediction) == 0:
            # print("Prediction is empty")
            return

        items_pytorch, points_batch, normals_batch, colors_batch, names_batch, indxs = batch

        # Use a multiprocessing pool to handle post processing in parallel
        self.post_process(indxs, prediction, items_pytorch, points_batch, normals_batch, colors_batch, names_batch, use_multiprocessing=True)
        # print(f"On batch end, len: {len(prediction)} ... finished")

    def post_process(self, indxs, res, items_pytorch, points_batch, normals_batch, colors_batch, names_batch, use_multiprocessing=False, distance_threshold=10.0, n=4, alpha = 1000.0, slope_alpha = 0.1):
        # print GPU memory usage
        if res is None:
            print("batch_inference result is None")
            res = [{"pred_classes": []}]*len(items_pytorch)
        else:
            res = list(res.values())

        if use_multiprocessing:
            # Save the block
            res = self.pool.map(to_surfaces_args, [(points_batch[i], normals_batch[i], colors_batch[i], res[i]) for i in range(len(points_batch))])
            surfaces, surfaces_normals, surfaces_colors, scores = zip(*res)
        else:
            # Single threaded version
            surfaces, surfaces_normals, surfaces_colors, scores = to_surfaces(points_batch, normals_batch, colors_batch, res)
        # return surfaces, surfaces_normals, surfaces_colors, names_batch, scores

        # save each instance for each subvolume
        # Setting up multiprocessing, process count
        if use_multiprocessing:
            # Save the block
            self.pool.map(save_block_ply_args, [(surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], names_batch[i], self.score_threshold, distance_threshold, n, alpha, slope_alpha) for i in range(len(surfaces))])
        else:
            # single threaded version
            for i in range(len(surfaces)):
                save_block_ply(surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], names_batch[i], self.score_threshold, distance_threshold, n, alpha, slope_alpha)

        # Update the progress file
        indxs = list(set(indxs) - set(self.computed_indices))
        for indx in indxs:
            self.computed_indices.append(indx)
        update_progress_file(self.progress_file, self.computed_indices, self.config)


class PointCloudDataset(Dataset):
    def __init__(self, path="/media/julian/FastSSD/scroll3_surface_points", folder="point_cloud_colorized", dest="/media/julian/HDD8TB/scroll3_surface_points", main_drive="", alternative_drives=[], fix_umbilicus=True, umbilicus_points_path="", start=[0, 0, 0], stop=[16, 17, 29], size = [3, 3, 3], umbilicus_distance_threshold=1500, score_threshold=0.5, batch_size=4, gpus=1, num_processes=3, recompute=False):
        self.writer = MyPredictionWriter(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, umbilicus_distance_threshold, score_threshold, batch_size, gpus, num_processes)

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
        # Initialize the ThreadPoolExecutor with the desired number of threads
        self.executor = ThreadPoolExecutor(max_workers=num_processes)

        umbilicus_raw_points = load_xyz_from_file(umbilicus_points_path)
        self.umbilicus_points = umbilicus(umbilicus_raw_points)
        if fix_umbilicus:
            # Load old umbilicus
            umbilicus_path_old = umbilicus_points_path.replace("umbilicus", "umbilicus_old")
            # Usage
            umbilicus_raw_points_old = load_xyz_from_file(umbilicus_path_old)
            self.umbilicus_points_old = umbilicus(umbilicus_raw_points_old)
        else:
            self.umbilicus_points_old = None

        self.start_list = build_start_list(start, stop, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)

        self.num_tasks = len(self.start_list)
        self.to_compute_indices = range(self.num_tasks)
        self.computed_indices = []
        self.progress_file = os.path.join(dest, "progress.json")
        self.config = {"path": path, "folder": folder, "dest": dest, "main_drive": main_drive, "alternative_drives": alternative_drives, "fix_umbilicus": fix_umbilicus, "umbilicus_points_path": umbilicus_points_path, "start": start, "stop": stop, "size": size, "umbilicus_distance_threshold": umbilicus_distance_threshold, "score_threshold": score_threshold, "batch_size": batch_size, "gpus": gpus}
        nr_total_indices = len(self.to_compute_indices)
        if os.path.exists(self.progress_file) and (not recompute):
            with open(self.progress_file, 'r') as file:
                progress = json.load(file)
                if 'config' in progress:
                    if progress['config'] != self.config:
                        print("Progress file found but with different config. Overwriting.")
                    else:
                        print("Progress file found with same config. Resuming computation.")
                        if 'indices' in progress:
                            self.computed_indices = progress['indices']
                            self.to_compute_indices = list(set(self.to_compute_indices) - set(self.computed_indices))
                            print(f"Resuming computation. {len(self.to_compute_indices)} blocks of {nr_total_indices} left to compute.")
                        else:
                            print("No progress file found.")

    def get_writer(self):
        return self.writer

    def create_batches(self, path, points, normals, colors, start, size, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive, alternative_drives, subvolume_size=50):
        # Size is int
        if isinstance(subvolume_size, int):
            subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

        # Iterate over all subvolumes
        start = np.array(start)
        # Swap axes
        start = start[[0,2,1]]
        min_coord = start * subvolume_size
        # size is size = original size + 1, we want original size * 2 subvolume blocks that overlap
        ranges = (size - 1) * subvolume_size # -1 because we want to include the last subvolume for tiling operation from later calls starting at the last subvolume but not save one half filled block
        subvolumes_points = []
        subvolumes_normals = []
        subvolumes_colors = []
        start_coords = []
        block_names = []
        block_names_created = []
        
        # Find block coords that contain points
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        start_ = np.floor(min_coords / (subvolume_size//2)) * (subvolume_size//2)
        # Swap axes
        # Max between start and min_coord
        start = np.maximum(start_, min_coord)

        stop_coord = min_coord + ranges

        stop_max_coords = np.ceil(max_coords / (subvolume_size//2)) * (subvolume_size//2)
        # Min between stop and max_coord
        stop = np.minimum(stop_max_coords, stop_coord)
        # print(f"Start_: {start_}, min_coord {min_coord}, Stop_max_coords: {stop_max_coords}, Stop: {stop_coord}, Range: {ranges}, Stop: {stop}")
        # time.sleep(15)
        # Make blocks of size '50x50x50'
        for x in range(int(start[0]), int(stop[0]), subvolume_size[0] // 2):
            for y in range(int(start[1]), int(stop[1]), subvolume_size[1] // 2):
                for z in range(int(start[2]), int(stop[2]), subvolume_size[2] // 2):
                    x_prime = x
                    y_prime = y
                    z_prime = z
                    start_coord = np.array([x_prime,y_prime,z_prime])
                    block_name = path + f"_subvolume_blocks/{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}" # nice ordering in the folder
                    block_name_tar = block_name + ".7z"
                    block_name_tar_alternatives = []
                    for alternative_drive in alternative_drives:
                        block_name_tar_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".7z")

                    # Skip if the block tar already exists. Same for alternative path
                    if (not (fix_umbilicus and fix_umbilicus_recompute(start_coord * 200.0 / 50.0, 200, umbilicus_points, umbilicus_points_old))) and (os.path.exists(block_name_tar) or any([os.path.exists(block_name_tar_alternative) for block_name_tar_alternative in block_name_tar_alternatives])):
                        # print(f"Block {block_name} already exists.")
                        block_names_created.append(block_name)
                        continue

                    # Extract a subvolume
                    subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles = extract_subvolume(points, normals, colors, colors, start=start_coord, size=subvolume_size) # Second colors is just a placeholder since we dont yet have any angles which the function expects
                    if len(subvolume_points) < 10:
                        # print(f"Subvolume {start_coord} has no points.")
                        continue
                    


                    subvolumes_points.append(subvolume_points)
                    subvolumes_normals.append(subvolume_normals)
                    subvolumes_colors.append(subvolume_colors)
                    start_coords.append(start_coord)
                    block_names.append(block_name)

        return subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names

    def precompute(self, index, start, size, path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points, umbilicus_points_old, use_multiprocessing, executor=None):
        src_path = os.path.join(path, folder)
        dest_path = os.path.join(dest, folder)
        size = np.array(size) + 1 # +1 because we want to include the last subvolume for tiling operation from later calls starting at the last subvolume
        
        res = load_plys(src_path, main_drive, alternative_drives, start, size, grid_block_size=200, load_multithreaded=use_multiprocessing, executor=executor)
        points, normals, colors = res
        points, normals, colors = remove_duplicate_points_normals(points, normals, colors)
        subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names = self.create_batches(dest_path, points, normals, colors, start, size, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive, alternative_drives)
        return subvolumes_points, subvolumes_normals, subvolumes_colors, start_coords, block_names

    def __len__(self):
        return len(self.to_compute_indices)

    def __getitem__(self, idx):
        i = self.to_compute_indices[idx] # index of the start_list
        x = self.start_list[i]

        res = self.precompute(i, x, self.size, self.path, self.folder, self.dest, self.main_drive, self.alternative_drives, self.fix_umbilicus, self.umbilicus_points, self.umbilicus_points_old, False, None)
        points_batch, normals_batch, colors_batch, start_coords, names_batch = res

        # Detect the surfaces in the subvolume
        points_batch_indices = [i for i, points in enumerate(points_batch) if points.shape[0] > 100]
        points_batch = [points_batch[i] for i in points_batch_indices]
        normals_batch = [normals_batch[i] for i in points_batch_indices]
        colors_batch = [colors_batch[i] for i in points_batch_indices]
        names_batch = [names_batch[i] for i in points_batch_indices]
        # Deep copy the points
        coords_batch = [np.copy(points) for points in points_batch]
        # Translate the points so that the volume starts at (0,0,0)
        min_coord_batch = [np.min(coords, axis=0) for coords in coords_batch]
        coords_batch = [coords - min_coord_batch[i] for i, coords in enumerate(coords_batch)]

        items_pytorch = [preprocess_points(c) for c in coords_batch]

        return items_pytorch, points_batch, normals_batch, colors_batch, names_batch, i

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

def pointcloud_inference(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, umbilicus_distance_threshold, score_threshold, batch_size, gpus, recompute):
    init()
    model = get_model()
    # model = torch.nn.DataParallel(model)
    # model.to('cuda')  # Move model to GPU

    dataset = PointCloudDataset(path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points_path, start, stop, size, umbilicus_distance_threshold, score_threshold, batch_size, recompute=recompute)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=12, prefetch_factor=2)  # Adjust num_workers as per your system

    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], gpus=gpus, strategy="ddp")

    print("Start prediction")
    # Run prediction
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Prediction done")

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
    batch_size = 1
    gpus = -1
    pointcloud_size = 1

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

    # import sys
    # # Remove command-line arguments for later internal calls to Mask3D
    # sys.argv = [sys.argv[0]]

    # Compute the surface patches
    pointcloud_inference(path, folder, dest, main_drive, alternative_ply_drives, fix_umbilicus, umbilicus_points_path, [0, 0, 0], [100, 100, 100], [pointcloud_size, pointcloud_size, pointcloud_size], umbilicus_distance_threshold, score_threshold, batch_size, gpus, recompute)
    
if __name__ == "__main__":
    main()