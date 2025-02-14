### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import pickle
import torch

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from .surface_detection import surface_detection

import numpy as np
import tifffile
import zarr
import tqdm
from PIL import Image
import open3d as o3d
import time

import os
import numpy as np
# import torch
# torch.set_float32_matmul_precision('medium')
# from torch.utils.data import Dataset, DataLoader

from scipy.interpolate import interp1d
from .add_random_colors_to_pointcloud import add_random_colors
# import torch.multiprocessing as multiprocessing
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
import glob
import argparse

CFG = {'num_threads': 4, 'GPUs': 1}

# Signal handler function
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def cleanup_temp_files(directory):
    # Get a list of all .temp files in the specified directory
    temp_files = glob.glob(os.path.join(directory, '*_temp.ply'))

    # Report if any temp files are found
    if temp_files:
        print(f"Found {len(temp_files)} leftover .temp files. Cleaning up...")
        for file in temp_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    else:
        print("No leftover .temp files found.")

def load_grid(path_template, zarr_volume, cords, grid_block_size=500, cell_block_size=500, uint8=True):
    if path_template.endswith('.zarr'):
        return load_grid_zarr(zarr_volume, cords, grid_block_size=grid_block_size, cell_block_size=cell_block_size, uint8=uint8)
    else:
        return load_grid_tiff(path_template, cords, grid_block_size=grid_block_size, cell_block_size=cell_block_size, uint8=uint8)

def load_grid_zarr(zarr_volume, cords, grid_block_size=500, cell_block_size=500, uint8=True):
    # make grid_block_size an array with 3 elements
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    axis_switch = [2, 0, 1] # [2, 0, 1]
    grid_block_size = grid_block_size.copy()[axis_switch]
    grid_index_ = np.array(cords).copy()[axis_switch] # swap axis for zar indexing. z y x (grid cell input is y x z)
    grid_index = grid_index_ - cell_block_size # Spelufo Julia language offset in indexing starting from 1
    grid_index_end = grid_index + grid_block_size
    # clip to 0 - zarr volume size
    grid_index = np.clip(grid_index, 0, None)
    grid_index = np.minimum(grid_index, np.array(zarr_volume.shape))
    grid_index_end = np.clip(grid_index_end, 0, None)
    grid_index_end = np.minimum(grid_index_end, np.array(zarr_volume.shape))

    # to int
    grid_index = grid_index.astype(int)
    grid_index_end = grid_index_end.astype(int)
    grid_block = zarr_volume[grid_index[0]:grid_index[0]+grid_block_size[0], grid_index[1]:grid_index[1]+grid_block_size[1], grid_index[2]:grid_index[2]+grid_block_size[2]]

    # ensure the grid block is of the correct grid block size
    if np.any(grid_block.shape != grid_block_size):
        grid_block_ = np.zeros((grid_block_size[0], grid_block_size[1], grid_block_size[2]), dtype=grid_block.dtype)
        grid_block_[:grid_block.shape[0], :grid_block.shape[1], :grid_block.shape[2]] = grid_block
        grid_block = grid_block_

    if uint8 and grid_block.dtype == np.uint16:
        grid_block = np.uint8(grid_block//256)

    if uint8:
        grid_block = np.uint8(grid_block)
    else:
        grid_block = np.uint16(grid_block)
    return grid_block

def load_grid_tiff(path_template, cords, grid_block_size=500, cell_block_size=500, uint8=True):
    """
    path_template: Template for the path to load individual grid files
    cords: Tuple (x, y, z) representing the corner coordinates of the grid block
    grid_block_size: Size of the grid block
    cell_block_size: Size of the individual grid files
    """
    # make grid_block_size an array with 3 elements
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    
    # Convert corner coordinates to file indices and generate the file path
    # Starting indices
    file_x_start, file_y_start, file_z_start = cords[0]//cell_block_size, cords[1]//cell_block_size, cords[2]//cell_block_size
    # Ending indices
    file_x_end, file_y_end, file_z_end = (cords[0] + grid_block_size[0])//cell_block_size, (cords[1] + grid_block_size[1])//cell_block_size, (cords[2] + grid_block_size[2])//cell_block_size

    # Generate the grid block
    if uint8:
        grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint8)
    else:
        grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint16)

    # Load the grid block from the individual grid files and place it in the larger grid block
    for file_x in range(file_x_start, file_x_end + 1):
        for file_y in range(file_y_start, file_y_end + 1):
            for file_z in range(file_z_start, file_z_end + 1):
                path = path_template.format(file_x, file_y, file_z)

                # Check if the file exists
                if not os.path.exists(path):
                    # print(f"File {path} does not exist.")
                    continue

                try:
                    # Read the image
                    with tifffile.TiffFile(path) as tif:
                        images = tif.asarray()
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

                if uint8:
                    images = np.uint8(images//256)

                # grid block slice position for the current file
                x_start = max(file_x*cell_block_size, cords[0])
                x_end = min((file_x + 1) * cell_block_size, cords[0] + grid_block_size[0])
                y_start = max(file_y*cell_block_size, cords[1])
                y_end = min((file_y + 1) * cell_block_size, cords[1] + grid_block_size[1])
                z_start = max(file_z*cell_block_size, cords[2])
                z_end = min((file_z + 1) * cell_block_size, cords[2] + grid_block_size[2])

                # Place the current file in the grid block
                try:
                    grid_block[z_start - cords[2]:z_end - cords[2], x_start - cords[0]:x_end - cords[0], y_start - cords[1]:y_end - cords[1]] = images[z_start - file_z*cell_block_size: z_end - file_z*cell_block_size, x_start - file_x*cell_block_size: x_end - file_x*cell_block_size, y_start - file_y*cell_block_size: y_end - file_y*cell_block_size]
                except:
                    print(f"Error in grid block placement for grid block {cords} and file {file_x}, {file_y}, {file_z}")

    return grid_block
    
def grid_empty(path_template, cords, grid_block_size=500, cell_block_size=500):
    """
    Determines wheter a grid block is empty or not.
        path_template: Template for the path to load individual grid files
        cords: Tuple (x, y, z) representing the corner coordinates of the grid block
        grid_block_size: Size of the grid block
        cell_block_size: Size of the individual grid files
    """
    # make grid_block_size an array with 3 elements
    if isinstance(grid_block_size, int):
        grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
    
    # Convert corner coordinates to file indices and generate the file path
    # Starting indices
    file_x_start, file_y_start, file_z_start = cords[0]//cell_block_size, cords[1]//cell_block_size, cords[2]//cell_block_size
    # Ending indices
    file_x_end, file_y_end, file_z_end = (cords[0] + grid_block_size[0])//cell_block_size, (cords[1] + grid_block_size[1])//cell_block_size, (cords[2] + grid_block_size[2])//cell_block_size

    # Check wheter none of the files exist
    for file_x in range(file_x_start, file_x_end + 1):
        for file_y in range(file_y_start, file_y_end + 1):
            for file_z in range(file_z_start, file_z_end + 1):
                path = path_template.format(file_x, file_y, file_z)

                # Check if the file exists
                if os.path.exists(path):
                    return False
    
    return True

def save_surface_ply(surface_points, normals, filename, color=None):
    try:
        if (len(surface_points)  < 1):
            return
        # Create an Open3D point cloud object and populate it
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float16))

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to a temporary file first to ensure data integrity
        temp_filename = filename.replace(".ply", "_temp.ply")
        o3d.io.write_point_cloud(temp_filename, pcd)

        # Rename the temp file to the original filename
        os.rename(temp_filename, filename)
    except Exception as e:
        print(f"Error saving surface PLY: {e}")

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
    res = np.array([x_new, y_new, z_new])
    return res

# Picks block closest to the edge of the complete volume to more accurately calculate global reference vectors (less curvature outside the scroll)
def pick_block(blocks, xy_min=2000, xy_max=6000, z_min=500, z_max=5000):

    min_border_distance = float("inf")
    min_border_block = None
    for block in blocks:
        x, y, z = block
        if abs(x - xy_min) < min_border_distance:
            min_border_distance = x - xy_min
            min_border_block = block
        if abs(xy_max - x) < min_border_distance:
            min_border_distance = xy_max - x
            min_border_block = block
        if abs(y - xy_min) < min_border_distance:
            min_border_distance = y - xy_min
            min_border_block = block
        if abs(xy_max - y) < min_border_distance:
            min_border_distance = xy_max - y
            min_border_block = block

    print("Picked block:", min_border_block, "Distance to border:", min_border_distance)
    return min_border_block

# Picks block radius away from center of the scroll
def pick_block_away_from_center(blocks, umbilicus_points, radius=1000, grid_block_size=500):
    min_border_distance = float("inf")
    min_border_block = None
    for block in blocks:
        y, x, z = block
        umbilicus_point = umbilicus_xz_at_y(umbilicus_points, z-grid_block_size//2)
        umbilicus_point = umbilicus_point[[0, 2, 1]] - grid_block_size//2 # cell/corner coord
        cost = abs(abs((y-(umbilicus_point[0]))**2 + (x-(umbilicus_point[1]))**2) - radius**2)
        if cost < min_border_distance:
            min_border_distance = cost
            min_border_block = block

    print("Picked block:", min_border_block, "Distance to border:", min_border_distance)
    return min_border_block

def extract_size(points, normals, grid_block_position_min, grid_block_position_max):
    """
    Extract points and corresponding normals that lie within the given size range.

    Parameters:
        points (numpy.ndarray): The point coordinates, shape (n, 3).
        normals (numpy.ndarray): The point normals, shape (n, 3).
        grid_block_position_min (numpy.ndarray): The minimum block size, shape (3,).
        grid_block_position_max (numpy.ndarray): The maximum block size, shape (3,).

    Returns:
        filtered_points (numpy.ndarray): The filtered points, shape (m, 3).
        filtered_normals (numpy.ndarray): The corresponding filtered normals, shape (m, 3).
    """
    
    # Create a mask to filter points within the specified range
    mask_min = np.all(points >= grid_block_position_min, axis=-1)
    mask_max = np.all(points <= grid_block_position_max, axis=-1)
    
    # Combine the masks to get the final mask
    mask = np.logical_and(mask_min, mask_max)
    
    # Apply the mask to filter points and corresponding normals
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # Reposition the points to be relative to the grid block
    filtered_points -= grid_block_position_min
    
    return filtered_points, filtered_normals

def extract_size_tensor(points, normals, grid_block_position_min, grid_block_position_max):
    """
    Extract points and corresponding normals that lie within the given size range.

    Parameters:
        points (torch.Tensor): The point coordinates, shape (n, 3).
        normals (torch.Tensor): The point normals, shape (n, 3).
        grid_block_position_min (int): The minimum block size.
        grid_block_position_max (int): The maximum block size.

    Returns:
        filtered_points (torch.Tensor): The filtered points, shape (m, 3).
        filtered_normals (torch.Tensor): The corresponding filtered normals, shape (m, 3).
    """

    # Convert min and max to tensors for comparison
    min_tensor = torch.tensor([grid_block_position_min] * 3, dtype=points.dtype, device=points.device)
    max_tensor = torch.tensor([grid_block_position_max] * 3, dtype=points.dtype, device=points.device)

    # Create a mask to filter points within the specified range
    mask_min = torch.all(points >= min_tensor, dim=-1)
    mask_max = torch.all(points <= max_tensor, dim=-1)

    # Combine the masks to get the final mask
    mask = torch.logical_and(mask_min, mask_max)

    # Apply the mask to filter points and corresponding normals
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # Reposition the points to be relative to the grid block
    filtered_points -= min_tensor

    return filtered_points, filtered_normals

# fixing the pointcloud because of computation with too short umbilicus
def fix_umbilicus_recompute(corner_coords, grid_block_size, umbilicus_points, umbilicus_points_old, additional_distance=300):
    block_point = np.array(corner_coords) + grid_block_size//2
    umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
    umbilicus_point_old = umbilicus_xz_at_y(umbilicus_points_old, block_point[2])
    umbilicus_point_old = umbilicus_point_old[[0, 2, 1]] # ply to corner coords
    umbilicus_point_middle = (umbilicus_point + umbilicus_point_old) / 2
    umbilicus_point_dist = umbilicus_point_middle - umbilicus_point
    umbilicus_point_dist = np.linalg.norm(umbilicus_point_dist)
    # block dist to disaster
    umbilicus_dist_middle = block_point - umbilicus_point_middle
    umbilicus_dist_middle = np.linalg.norm(umbilicus_dist_middle)
    if umbilicus_dist_middle <= umbilicus_point_dist * 1.5 + additional_distance:
        return True
    return False

# fixing the pointcloud because of computation with too short umbilicus
def skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=2500):
    if maximum_distance <= 0:
        return False
    
    block_point = np.array(corner_coords) + grid_block_size//2
    umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords

    umbilicus_point_dist = umbilicus_point - block_point
    umbilicus_point_dist = np.linalg.norm(umbilicus_point_dist)
    return umbilicus_point_dist > maximum_distance

class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, computed_blocks, pointcloud_base, save_template_v, save_template_r, grid_block_size=200):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.computed_blocks = computed_blocks 
        self.pointcloud_base = pointcloud_base
        self.save_template_v = save_template_v
        self.save_template_r = save_template_r
        self.grid_block_size = grid_block_size
        # Use a thread pool for IO-bound tasks; threads avoid pickling overhead.
        all_threads = multiprocessing.cpu_count()
        print(f"Saving with {all_threads} threads.")
        self.executor = ThreadPoolExecutor(max_workers=all_threads)
        # A lock to protect shared resources (computed_blocks, file writing)
        self.lock = threading.Lock()

    def write_on_predict(self, predictions: list, batch_indices: list, dataloader_idx: int, batch, batch_idx: int, dataloader_len: int):
        print(predictions)
        print("On predict")
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
                            prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if prediction is None:
            return
        if len(prediction) == 0:
            return

        # Unpack prediction tuple
        (points_r_tensors, normals_r_tensors), (points_v_tensors, normals_v_tensors), corner_coordss = prediction

        # Schedule each save operation in parallel
        futures = []
        for i in range(len(points_r_tensors)):
            futures.append(self.executor.submit(
                self.parallel_save_surface,
                points_r_tensors[i],
                normals_r_tensors[i],
                points_v_tensors[i],
                normals_v_tensors[i],
                corner_coordss[i]
            ))
        # (Optionally, wait for all tasks to complete)
        for future in futures:
            future.result()

        # Save the computed blocks file (this is typically very fast)
        computed_blocks_filepath = os.path.join(self.pointcloud_base, "computed_blocks.txt")
        with open(computed_blocks_filepath, "w") as f:
            with self.lock:
                for block in self.computed_blocks:
                    f.write(str(block) + "\n")

    def parallel_save_surface(self, points_r_tensor, normals_r_tensor, points_v_tensor, normals_v_tensor, corner_coords):
        # Convert the tensors to numpy arrays
        points_r = points_r_tensor.cpu().numpy()
        normals_r = normals_r_tensor.cpu().numpy()
        points_v = points_v_tensor.cpu().numpy()
        normals_v = normals_v_tensor.cpu().numpy()

        # Compute file indices based on the grid block size
        file_x = corner_coords[0] // self.grid_block_size
        file_y = corner_coords[1] // self.grid_block_size
        file_z = corner_coords[2] // self.grid_block_size

        surface_ply_filename_r = self.save_template_r.format(file_x, file_y, file_z)
        surface_ply_filename_v = self.save_template_v.format(file_x, file_y, file_z)

        # Save the two surfaces (one for each prediction)
        save_surface_ply(points_r, normals_r, surface_ply_filename_r)
        save_surface_ply(points_v, normals_v, surface_ply_filename_v)

        # Update computed blocks in a thread-safe manner.
        # Convert corner_coords to tuple to ensure it is hashable for a set.
        with self.lock:
            self.computed_blocks.add(tuple(corner_coords))

class GridDataset(Dataset):
    def __init__(self, pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1, min_z=None, max_z=None):
        self.min_z = min_z
        self.max_z = max_z
        if not hasattr(self, 'zarr_volume'):
            self.zarr_volume = None
        self.grid_block_size = grid_block_size
        self.path_template = path_template
        self.umbilicus_points = umbilicus_points
        self.blocks_to_process, blocks_processed = self.init_blocks_to_process(pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance)
        self.blocks_to_process = self.filter_blocks_to_process(self.blocks_to_process)
        print(f"Found {len(self.blocks_to_process)} blocks to process. And {len(blocks_processed)} blocks already processed.")

        self.writer = MyPredictionWriter(blocks_processed, pointcloud_base, save_template_v, save_template_r, grid_block_size=grid_block_size)
        
    def init_blocks_to_process(self, pointcloud_base, start_block, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        # Load the set of computed blocks
        computed_blocks = self.load_computed_blocks(pointcloud_base)

        # Initialize the blocks that need computing
        blocks_to_process, blocks_processed = self.blocks_to_compute(
            start_block, computed_blocks, umbilicus_points, umbilicus_points_old, path_template,
            grid_block_size, recompute, fix_umbilicus, maximum_distance
        )
        
        blocks_to_process = sorted(list(blocks_to_process)) # Sort the blocks to process for deterministic behavior
        return blocks_to_process, blocks_processed
        
    def load_computed_blocks(self, pointcloud_base):
        computed_blocks = set()
        # Try to load the list of computed blocks
        try:
            with open(os.path.join("/", pointcloud_base, "computed_blocks.txt"), "r") as f:
                # load saved tuples with 3 elements
                computed_blocks = set([eval(line.strip()) for line in f])
        except FileNotFoundError:
            print("[INFO]: No computed blocks found.")
        except Exception as e:
            print(f"Error loading computed blocks: {e}")
        return computed_blocks

    def blocks_to_compute(self, start_coord, computed_blocks, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        padding = 50

        all_corner_coords = set() # Set containing all the corner coords that need to be placed into processing/processed set.
        all_corner_coords.add(start_coord) # Add the start coord to the set of all corner coords
        blocks_to_process = set() # Blocks that need to be processed
        blocks_processed = set() # Set to hold the blocks that do not need to be processed. Either have been processed or don't need to be processed.
        
        while len(all_corner_coords) > 0:
            corner_coords = all_corner_coords.pop()

            if fix_umbilicus:
                fix_umbilicus_indicator = fix_umbilicus_recompute(corner_coords, grid_block_size, umbilicus_points, umbilicus_points_old)
            else:
                fix_umbilicus_indicator = False
            recompute = recompute or fix_umbilicus_indicator

            # Load the grid block from corner_coords and grid size
            corner_coords_padded = np.array(corner_coords) - padding
            grid_block_size_padded = grid_block_size + 2 * padding

            previously_computed = corner_coords in computed_blocks
            if previously_computed and (not recompute): # Block was already computed and is valid
                blocks_processed.add(corner_coords)
            else: # Recompute if wasn't computed or recompute flag is set
                # Check if the block is empty
                if grid_empty(path_template, corner_coords_padded, grid_block_size=grid_block_size_padded):
                    blocks_processed.add(corner_coords)
                    # Outside of the scroll, don't add neighbors
                    continue
                # this next check comes after the empty check, else under certain umbilicus distances, there might be an infinite loop
                skip_computation_flag = skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=maximum_distance)
                if skip_computation_flag:
                    # Block not needed in processing
                    blocks_processed.add(corner_coords)
                else:
                    # Otherwise add corner coords to the blocks that need processing
                    blocks_to_process.add(corner_coords)
                
            # Compute neighboring blocks
            for dx in [-grid_block_size, 0, grid_block_size]:
                for dy in [-grid_block_size, 0, grid_block_size]:
                    for dz in [-grid_block_size, 0, grid_block_size]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        if abs(dx) + abs(dy) + abs(dz) > grid_block_size:
                            continue
                        neighbor_coords = (corner_coords[0] + dx, corner_coords[1] + dy, corner_coords[2] + dz)
                        
                        # Add the neighbor to the list of blocks to process if it hasn't been processed yet
                        if (neighbor_coords not in blocks_processed) and (neighbor_coords not in blocks_to_process) and (neighbor_coords not in all_corner_coords):
                            all_corner_coords.add(neighbor_coords)

        return blocks_to_process, blocks_processed
    
    def filter_blocks_to_process(self, blocks_to_process):
        i = 0
        while i < len(blocks_to_process):
            corner_coords = blocks_to_process[i]
            # check if block is in z range, else remove
            if self.min_z is not None and corner_coords[2] < self.min_z:
                # remove from processing index i
                blocks_to_process = blocks_to_process[:i] + blocks_to_process[i+1:]
            elif self.max_z is not None and corner_coords[2] > self.max_z:
                # remove from processing index i
                blocks_to_process = blocks_to_process[:i] + blocks_to_process[i+1:]
            else:
                i += 1
        return blocks_to_process
    
    def get_writer(self):
        return self.writer
    
    # def get_reference_vector(self, corner_coords):
    #     block_point = np.array(corner_coords) + self.grid_block_size//2 # y, x, z
    #     umbilicus_point = umbilicus_xz_at_y(self.umbilicus_points, block_point[2]) # x, z, y
    #     umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords # x, y, z
    #     umbilicus_normal = block_point - umbilicus_point 
    #     # umbilicus_normal = umbilicus_normal[[2, 0, 1]] # corner coords to tif 
    #     umbilicus_normal = umbilicus_normal[[2, 1, 0]] # corner coords to tif 
    #     unit_umbilicus_normal = umbilicus_normal / np.linalg.norm(umbilicus_normal)
    #     return unit_umbilicus_normal # y, z, x

    def get_reference_vector(self, corner_coords):
        block_point = np.array(corner_coords) + self.grid_block_size//2 # y, x, z
        block_point = block_point[[0, 2, 1]] # y, z, x
        umbilicus_point = umbilicus_xz_at_y(self.umbilicus_points, block_point[1]) # y, z, x
        umbilicus_normal = (block_point - umbilicus_point).astype(np.float32)
        unit_umbilicus_normal = umbilicus_normal / np.linalg.norm(umbilicus_normal)
        unit_umbilicus_normal = unit_umbilicus_normal[[2, 1, 0]]
        return unit_umbilicus_normal.astype(np.float32) # y, z, x
    
    def __len__(self):
        return len(self.blocks_to_process)

    def __getitem__(self, idx):
        corner_coords = self.blocks_to_process[idx]
        # load the grid block from corner_coords and grid size
        padding = 50
        corner_coords_padded = np.array(corner_coords) - padding
        grid_block_size_padded = self.grid_block_size + 2 * padding
        block = load_grid(self.path_template, self.zarr_volume, corner_coords_padded, grid_block_size=grid_block_size_padded)
        reference_vector = self.get_reference_vector(corner_coords)
        
        # Convert NumPy arrays to PyTorch tensors
        block_tensor = torch.from_numpy(block).float()  # Convert to float32 tensor
        reference_vector_tensor = torch.from_numpy(reference_vector).to(dtype=torch.float16)
        
        return block_tensor, reference_vector_tensor, corner_coords, self.grid_block_size, padding
    
class ZarrDataset(GridDataset):
    def __init__(self, pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1, min_z=None, max_z=None):
        self.zarr_volume = zarr.open(path_template, mode='r')
        self.zarr_volume = self.zarr_volume[0]
        super().__init__(pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=grid_block_size, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance, min_z=min_z, max_z=max_z)

    def blocks_to_compute(self, start_coord, computed_blocks, umbilicus_points, umbilicus_points_old, path_template, grid_block_size, recompute, fix_umbilicus, maximum_distance):
        print("Using zarr dataset")

        all_corner_coords = set() # Set containing all the corner coords that need to be placed into processing/processed set.
        all_corner_coords.add(start_coord) # Add the start coord to the set of all corner coords
        blocks_to_process = set() # Blocks that need to be processed
        blocks_processed = set() # Set to hold the blocks that do not need to be processed. Either have been processed or don't need to be processed.

        z_height = self.zarr_volume.shape[0]
        y_height = self.zarr_volume.shape[1]
        x_height = self.zarr_volume.shape[2]
        
        while len(all_corner_coords) > 0:
            corner_coords = all_corner_coords.pop()

            if fix_umbilicus:
                fix_umbilicus_indicator = fix_umbilicus_recompute(corner_coords, grid_block_size, umbilicus_points, umbilicus_points_old)
            else:
                fix_umbilicus_indicator = False
            recompute = recompute or fix_umbilicus_indicator

            previously_computed = corner_coords in computed_blocks
            if previously_computed and (not recompute): # Block was already computed and is valid
                blocks_processed.add(corner_coords)
            else: # Recompute if wasn't computed or recompute flag is set
                if corner_coords[0] < 500 or corner_coords[1] < 500 or corner_coords[2] < 500 or corner_coords[0] - 500 > y_height or corner_coords[1] - 500 > x_height or corner_coords[2] - 500 > z_height:
                    blocks_processed.add(corner_coords)
                    # Outside of the scroll, don't add neighbors
                    continue
                # this next check comes after the empty check, else under certain umbilicus distances, there might be an infinite loop
                skip_computation_flag = skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=maximum_distance)
                if skip_computation_flag:
                    # Block not needed in processing
                    blocks_processed.add(corner_coords)
                else:
                    # Otherwise add corner coords to the blocks that need processing
                    blocks_to_process.add(corner_coords)
                
            # Compute neighboring blocks
            for dx in [-grid_block_size, 0, grid_block_size]:
                for dy in [-grid_block_size, 0, grid_block_size]:
                    for dz in [-grid_block_size, 0, grid_block_size]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        if abs(dx) + abs(dy) + abs(dz) > grid_block_size:
                            continue
                        neighbor_coords = (corner_coords[0] + dx, corner_coords[1] + dy, corner_coords[2] + dz)
                        
                        # Add the neighbor to the list of blocks to process if it hasn't been processed yet
                        if (neighbor_coords not in blocks_processed) and (neighbor_coords not in blocks_to_process) and (neighbor_coords not in all_corner_coords):
                            all_corner_coords.add(neighbor_coords)

        return blocks_to_process, blocks_processed
    
# Custom collation function
def custom_collate_fn(batches):
    # Initialize containers for the aggregated items
    blocks = []
    reference_vectors = []
    corner_coordss = []
    grid_block_sizes = []
    paddings = []

    # Loop through each batch and aggregate its items
    for batch in batches:
        block, reference_vector, corner_coords, grid_block_size, padding = batch
        blocks.append(block)
        reference_vectors.append(reference_vector)
        corner_coordss.append(corner_coords)
        grid_block_sizes.append(grid_block_size)
        paddings.append(padding)
        
    # Return a single batch containing all aggregated items
    return blocks, reference_vectors, corner_coordss, grid_block_sizes, paddings

class PointCloudModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()

    def forward(self, x):
        # Extract input information
        grid_volumes, reference_vectors, corner_coordss, grid_block_sizes, paddings = x
        
        points_r_tensors, normals_r_tensors, points_v_tensors, normals_v_tensors = [], [], [], []
        for grid_volume, reference_vector, corner_coords, grid_block_size, padding in zip(grid_volumes, reference_vectors, corner_coordss, grid_block_sizes, paddings):
            # Perform surface detection for this block block
            recto_tensor_tuple, verso_tensor_tuple = surface_detection(grid_volume, reference_vector, blur_size=11, window_size=9, stride=1, threshold_der=0.075, threshold_der2=0.002, convert_to_numpy=False)
            points_r_tensor, normals_r_tensor = recto_tensor_tuple # z, y, x
            points_v_tensor, normals_v_tensor = verso_tensor_tuple
            # Extract actual volume size from the oversized input block
            points_r_tensor, normals_r_tensor = extract_size_tensor(points_r_tensor, normals_r_tensor, padding, grid_block_size+padding) # 0, 0, 0 is the minimum corner of the grid block
            points_v_tensor, normals_v_tensor = extract_size_tensor(points_v_tensor, normals_v_tensor, padding, grid_block_size+padding) # 0, 0, 0 is the minimum corner of the grid block

            ### Adjust the 3D coordinates of the points based on their position in the larger volume

            # permute the axes to match the original volume
            points_r_tensor = points_r_tensor[:, [1, 0, 2]] # y, z, x
            normals_r_tensor = normals_r_tensor[:, [1, 0, 2]]
            points_v_tensor = points_v_tensor[:, [1, 0, 2]]
            normals_v_tensor = normals_v_tensor[:, [1, 0, 2]]

            y_d, x_d, z_d = corner_coords
            points_r_tensor += torch.tensor([y_d, z_d, x_d], dtype=points_r_tensor.dtype, device=points_r_tensor.device)
            points_v_tensor += torch.tensor([y_d, z_d, x_d], dtype=points_v_tensor.dtype, device=points_v_tensor.device)
            
            points_r_tensors.append(points_r_tensor)
            normals_r_tensors.append(normals_r_tensor)
            points_v_tensors.append(points_v_tensor)
            normals_v_tensors.append(normals_v_tensor)

        return (points_r_tensors, normals_r_tensors), (points_v_tensors, normals_v_tensors), corner_coordss
    
def grid_inference(pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=200, recompute=False, fix_umbilicus=False, maximum_distance=-1, batch_size=1, min_z=None, max_z=None):
    if path_template.endswith(".zarr"):
        dataset = ZarrDataset(pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=grid_block_size, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance, min_z=min_z, max_z=max_z)
    else:
        dataset = GridDataset(pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_points, umbilicus_points_old, grid_block_size=grid_block_size, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance, min_z=min_z, max_z=max_z)
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(CFG['GPUs']))
    num_treads_for_gpus = 5
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=3)
    model = PointCloudModel()
    
    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], gpus=int(CFG['GPUs']), strategy="ddp")
    
    print("Start prediction")
    # Run prediction
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Prediction done")
    
    return

def compute(disk_load_save, base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, num_threads, gpus, skip_surface_blocks, min_z, max_z):
    # Initialize CUDA context
    # _ = torch.tensor([0.0]).cuda()
    # try:
    #     multiprocessing.set_start_method('spawn')
    # except Exception as e:
    #     print(f"Error setting multiprocessing start method: {e}")

    CFG['num_threads'] = num_threads
    CFG['GPUs'] = gpus

    pointcloud_base = os.path.dirname(pointcloud_subpath)
    pointcloud_subpath_recto = pointcloud_subpath + "_recto"
    pointcloud_subpath_verso = pointcloud_subpath + "_verso"
    pointcloud_subpath_colorized = pointcloud_subpath + "_colorized"
    src_dir = os.path.join(base_path, volume_subpath)
    dest_dir_r = src_dir.replace(volume_subpath, pointcloud_subpath_recto).replace(disk_load_save[0], disk_load_save[1])
    dest_dir_v = src_dir.replace(volume_subpath, pointcloud_subpath_verso).replace(disk_load_save[0], disk_load_save[1])

    # Cleanup temp files
    cleanup_temp_files(dest_dir_r)
    cleanup_temp_files(dest_dir_v)
    path_template = os.path.join(src_dir, "cell_yxz_{:03}_{:03}_{:03}.tif")
    save_template_r = path_template.replace(".tif", ".ply").replace(volume_subpath, pointcloud_subpath_recto).replace(disk_load_save[0], disk_load_save[1])
    save_template_v = path_template.replace(".tif", ".ply").replace(volume_subpath, pointcloud_subpath_verso).replace(disk_load_save[0], disk_load_save[1])

    if src_dir.endswith(".zarr"):
        path_template = src_dir

    umbilicus_path = os.path.join(os.path.dirname(src_dir), "umbilicus.txt")
    save_umbilicus_path = umbilicus_path.replace(".txt", ".ply").replace(disk_load_save[0], disk_load_save[1])
    save_umbilicus_path = save_umbilicus_path.replace(os.path.dirname(volume_subpath), pointcloud_subpath)

    # Copy umbilicus.txt to pointcloud_subpath dir
    umbilicus_copy_path = os.path.dirname(save_umbilicus_path)
    #directory of umbilicus_copy_path directory
    umbilicus_copy_path = os.path.dirname(umbilicus_copy_path) + "/umbilicus.txt"
    os.makedirs(os.path.dirname(umbilicus_copy_path), exist_ok=True)
    print(f"Copying umbilicus.txt from {umbilicus_path} to {umbilicus_copy_path}")
    os.system("cp " + umbilicus_path + " " + umbilicus_copy_path)

    # Usage
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)
    # Red color for umbilicus
    colors = np.zeros_like(umbilicus_points)
    colors[:,0] = 1.0
    # Save umbilicus as a PLY file, for visualization (CloudCompare)
    save_surface_ply(umbilicus_points, np.zeros_like(umbilicus_points), save_umbilicus_path, color=colors)

    if fix_umbilicus:
        # Load old umbilicus
        umbilicus_path_old = os.path.join(src_dir, "umbilicus_old.txt")
        # Usage
        umbilicus_raw_points_old = load_xyz_from_file(umbilicus_path_old)
        umbilicus_points_old = umbilicus(umbilicus_raw_points_old)
    else:
        umbilicus_points_old = None
        umbilicus_raw_points_old = None

    # Starting grid block at corner (3000, 4000, 2000) to match cell_yxz_006_008_004
    # (2600, 2200, 5000)
    if not skip_surface_blocks:
        # compute_surface_for_block_multiprocessing(start_block, pointcloud_base, path_template, save_template_v, save_template_r, umbilicus_points, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, umbilicus_points_old=umbilicus_points_old, maximum_distance=maximum_distance)
        grid_inference(pointcloud_base, start_block, path_template, save_template_v, save_template_r, umbilicus_raw_points, umbilicus_raw_points_old, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, maximum_distance=maximum_distance, batch_size=1*int(CFG['GPUs']), min_z=min_z, max_z=max_z)
    else:
        print("Skipping surface block computation.")

    # Sample usage:
    print("Adding random colors to pointclouds...")
    # src is folder of save_umbilicus_path
    dest_folder_v = dest_dir_v.replace(pointcloud_subpath, pointcloud_subpath_colorized).replace(disk_load_save[0], disk_load_save[1])
    add_random_colors(dest_dir_v, dest_folder_v)

    dest_folder_r = dest_dir_r.replace(pointcloud_subpath, pointcloud_subpath_colorized).replace(disk_load_save[0], disk_load_save[1])
    add_random_colors(dest_dir_r, dest_folder_r)

def main():
    # Parse arguments defaults
    maximum_distance = 2300 # scroll 3 all
    maximum_distance= -1 #1750 # maximum distance between blocks to compute and the umbilicus (speed up pointcloud generation if only interested in inner part of scrolls)
    recompute=False # whether to completely recompute all already processed blocks or continue (recompute=False). 
    fix_umbilicus = False
    disk_load_save = ["SSD4TB", "SSD2"] # Disk that contains input data, and disk that should be used to save output data
    base_path = "/media/julian/SSD4TB"
    volume_subpath = "PHerc0332.volpkg/volumes/2dtifs_8um_grids"
    pointcloud_subpath = "scroll3_surface_points/point_cloud"
    # start_block = (3000, 4000, 2000) # scroll1
    start_block = (600, 600, 600)

    parser = argparse.ArgumentParser(description="Extract papyrus sheet surface points from a 3D volume of a scroll CT scan")
    parser.add_argument("--base_path", type=str, help="Base path to the data", default=base_path)
    parser.add_argument("--disk_load_save", type=str, nargs=2, help="Disk that contains input data, and dist that should be used to save output data", default=disk_load_save)
    parser.add_argument("--volume_subpath", type=str, help="Subpath to the volume data", default=volume_subpath)
    parser.add_argument("--pointcloud_subpath", type=str, help="Subpath to the pointcloud data", default=pointcloud_subpath)
    parser.add_argument("--max_umbilicus_dist", type=float, help="Maximum distance between the umbilicus and blocks that should be computed. -1.0 for no distance restriction", default=-1.0)
    parser.add_argument("--recompute", action='store_true', help="Flag, recompute all blocks, even if they already exist")
    parser.add_argument("--fix_umbilicus", action='store_true', help="Flag, recompute all close to the updated umbilicus (make sure to also save the old umbilicus.txt as umbilicus_old.txt)")
    parser.add_argument("--start_block", type=int, nargs=3, help="Starting block to compute", default=start_block)
    parser.add_argument("--num_threads", type=int, help="Number of threads to use", default=CFG['num_threads'])
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use", default=CFG['GPUs'])
    parser.add_argument("--skip_surface_blocks", action='store_true', help="Flag, skip the surface block computation")
    parser.add_argument('--sheet_z_range', type=int, nargs=2,help='Z range of segmentation', default=[None, None])

    # only parse known args for PL multi GPU DDP setup to work
    args, unknown = parser.parse_known_args()

    # Print arguments
    print(f"Arguments: {args}")

    # update variables from arguments
    disk_load_save = args.disk_load_save
    base_path = args.base_path
    volume_subpath = args.volume_subpath
    pointcloud_subpath = args.pointcloud_subpath
    maximum_distance = args.max_umbilicus_dist
    recompute = args.recompute
    fix_umbilicus = args.fix_umbilicus
    start_block = tuple(args.start_block)
    assert start_block[0]%200==0 and start_block[1]%200==0 and start_block[2]%200==0, "Start block position should be cleanly divisable by 200"
    skip_surface_blocks = args.skip_surface_blocks
    
    # Compute the surface points
    compute(disk_load_save, base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, args.num_threads, args.gpus, skip_surface_blocks, args.sheet_z_range[0], args.sheet_z_range[1])

if __name__ == "__main__":
    main()