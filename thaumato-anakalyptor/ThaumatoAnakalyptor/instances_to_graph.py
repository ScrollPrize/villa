### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2024

import numpy as np
from numba import njit
from tqdm import tqdm
import pickle
import glob
import os
import tarfile
import py7zr
import h5py
import tempfile
import open3d as o3d
import json
from multiprocessing import Pool, cpu_count, Manager
import time
import argparse
import yaml
import ezdxf
from ezdxf.math import Vec3
import random
from copy import deepcopy
import struct
import multiprocessing
from scipy.spatial import cKDTree

from .instances_to_sheets import select_points, get_vector_mean, alpha_angles, adjust_angles_zero, adjust_angles_offset, add_overlapp_entries_to_patches_list, assign_points_to_tiles, compute_overlap_for_pair, compute_overlap_for_pair_old, overlapp_score, fit_sheet, winding_switch_sheet_score_raw_precomputed_surface, find_starting_patch, save_main_sheet, update_main_sheet
from .sheet_to_mesh import load_xyz_from_file, scale_points, umbilicus_xz_at_y, shuffling_points_axis
from .mesh_to_mask3d_labels import set_up_mesh
from .split_mesh import MeshSplitter
from .mesh_quality import calculate_winding_angle_pointcloud_instance, align_winding_angles, find_best_alignment
from .mesh_quality import load_mesh_vertices as load_mesh_vertices_quality
import sys
### C++ speed up Random Walks
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import sheet_generation

global centroid_method

@njit
def numba_mean_axis0(X):
    """
    Compute the mean of a 2D array along axis 0 manually for Numba compatibility.
    """
    n_rows, n_cols = X.shape
    mean_values = np.zeros(n_cols, dtype=X.dtype)
    
    for col in range(n_cols):
        col_sum = 0.0
        for row in range(n_rows):
            col_sum += X[row, col]
        mean_values[col] = col_sum / n_rows
    
    return mean_values

@njit
def numba_norm(arr, axis=1):
    """
    Custom implementation of np.linalg.norm for Numba, supporting axis argument.
    Computes the Euclidean norm along the specified axis.
    """
    if axis == 1:
        # Calculate the norm along rows (for 2D array)
        return np.sqrt(np.sum(arr ** 2, axis=1))
    elif axis == 0:
        # Calculate the norm along columns (for 2D array)
        return np.sqrt(np.sum(arr ** 2, axis=0))
    else:
        raise ValueError("Unsupported axis. Use 0 or 1.")
    
@njit
def geometric_median_(X, eps=1e-5):
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm,
    avoiding unsupported NumPy features in Numba.
    """
    # Manually compute mean along axis 0
    y = numba_mean_axis0(X)
    
    while True:
        # Compute distances (norms) from all points to the current median estimate
        D = numba_norm(X - y, axis=1)
        non_zeros = (D != 0)
        
        if not non_zeros.any():
            return y
        
        D_inv = 1 / D[non_zeros]
        T = np.sum(X[non_zeros] * D_inv[:, None], axis=0) / np.sum(D_inv)
        
        if numba_norm(y - T, axis=0) < eps:
            return T
        
        y = T

def closest_to_geometric_median(X):
    """
    Compute the closest point to geometric median in the original pointset
    """
    y = geometric_median_(X)
    D_T = numba_norm(X - y, axis=1)
    min_idx = np.argmin(D_T)
    closest_point = X[min_idx]
    return closest_point

def centroid_method_mean(X):
    """
    Compute the centroid of a set of points using the mean.
    """
    return np.mean(X, axis=0)

def centroid_method_geometric_mean(X):
    """
    Compute the centroid of a set of points using the geometric mean.
    """
    return np.exp(np.mean(np.log(C), axis=0))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2=np.array([1, 0])):
    """
    Returns the signed angle in radians between vectors 'v1' and 'v2'.

    Examples:
        >>> angle_between(np.array([1, 0]), np.array([0, 1]))
        1.5707963267948966
        >>> angle_between(np.array([1, 0]), np.array([1, 0]))
        0.0
        >>> angle_between(np.array([1, 0]), np.array([-1, 0]))
        3.141592653589793
        >>> angle_between(np.array([1, 0]), np.array([0, -1]))
        -1.5707963267948966
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
    assert angle is not None, "Angle is None."
    return angle

def build_kd_tree(blocks_files):
    """
    Builds a KD-tree from the available block files.

    Args:
        blocks_files (str): Block files.

    Returns:
        tuple: (cKDTree object, numpy array of block_ids)
    """
    # Extract block IDs
    block_ids = []
    for file_path in blocks_files:
        file_name = ".".join(file_path.split(".")[:-1])
        raw_id_str = file_name.split('/')[-1]  # e.g., "100_200_300"
        block_id = [int(i) for i in raw_id_str.split('_')]
        block_ids.append(block_id)

    block_ids = np.array(block_ids)  # Convert list to numpy array

    if len(block_ids) == 0:
        raise ValueError("No valid block IDs found in the specified directory.")

    # Build the KD-tree
    kd_tree = cKDTree(block_ids)
    return kd_tree, block_ids

def surrounding_volumes_kdtree(volume_id, overlapp_threshold, block_ids, kd_tree, volume_size=50):
    """
    Returns all volumes within `max_distance` of `volume_id` using a KD-tree.

    Args:
        volume_id   (array-like): The 3D integer ID (x,y,z) of the volume of interest.
        kd_tree     (cKDTree)   : Pre-built KD-tree of all block IDs.
        block_ids   (ndarray)   : Array of shape (N, 3) with all known block IDs.
        max_distance (float)    : The maximum Euclidean distance from volume_id.

    Returns:
        np.ndarray: Subset of block_ids that lie within max_distance of volume_id.
    """
    # Query KD-tree for blocks within max_distance
    idxs = kd_tree.query_ball_point(volume_id, volume_size / 2.0)
    surrounding_volumes = block_ids[idxs]
    # filter in sheet z range
    surrounding_volumes = [volume for volume in surrounding_volumes if volume[1] >= overlapp_threshold["sheet_z_range"][0] and volume[1] <= overlapp_threshold["sheet_z_range"][1]]
    return surrounding_volumes

def build_surrounding_volumes_dict(kd_tree, block_ids, overlapp_threshold, volume_size=50):
    """
    Precomputes and stores surrounding volumes for each volume_id.

    Args:
        kd_tree (cKDTree): Pre-built KD-tree of all block IDs.
        block_ids (ndarray): Array of shape (N, 3) with all known block IDs.
        overlapp_threshold (dict): If provided, filters based on `sheet_z_range`.
        volume_size (float, optional): Search distance (default: 50).

    Returns:
        dict: A dictionary {volume_id: surrounding_volumes}.
    """
    surrounding_dict = {}

    for volume_id in tqdm(block_ids, desc="Building surrounding volumes dict"):
        idxs = kd_tree.query_ball_point(volume_id, volume_size / 2.0 + 1)
        surrounding_volumes = block_ids[idxs]

        # filter out volume ids that have more than one dimension difference
        surrounding_volumes = [volume for volume in surrounding_volumes if np.sum(volume == volume_id) == 1]

        # filter sucht that only connections to larger ids are stored
        surrounding_volumes = [volume for volume in surrounding_volumes if not (volume[0] < volume_id[0] or volume[1] < volume_id[1] or volume[2] < volume_id[2])]

        # filter out volume_id itself
        surrounding_volumes = [volume for volume in surrounding_volumes if not np.array_equal(volume, volume_id)]

        # Apply `sheet_z_range` filtering
        z_min, z_max = overlapp_threshold["sheet_z_range"]
        surrounding_volumes = [volume for volume in surrounding_volumes if z_min <= volume[1] <= z_max]

        surrounding_dict[tuple(volume_id)] = surrounding_volumes  # Store as tuple key

    return surrounding_dict

def surrounding_volumes_kdtree_dict(volume_id):
    """
    Returns surrounding volumes using precomputed dictionary lookups.

    Args:
        volume_id (tuple or list): The 3D integer ID (x,y,z) of the volume of interest.

    Returns:
        list: Surrounding volume IDs.
    """
    return surrounding_dict.get(tuple(volume_id), [])  # Default to empty list if not found
    

def surrounding_volumes(volume_id, overlapp_threshold, volume_size=50):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*3
    surrounding_volumes[0] = (volume_id_x + vs, volume_id_y, volume_id_z)
    surrounding_volumes[1] = (volume_id_x, volume_id_y + vs, volume_id_z)
    surrounding_volumes[2] = (volume_id_x, volume_id_y, volume_id_z + vs)
    # filter in sheet z range
    surrounding_volumes = [volume for volume in surrounding_volumes if volume[1] >= overlapp_threshold["sheet_z_range"][0] and volume[1] <= overlapp_threshold["sheet_z_range"][1]]
    return surrounding_volumes

def volumes_of_point(point, volume_size=50):
    """
    Returns all the volumes containing the point
    """
    point = np.array(point)
    size_half = volume_size//2
    volume_quadrant = np.floor(point / size_half).astype(int) * size_half
    volumes = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                volumes.append(tuple(volume_quadrant + size_half * np.array([i, j, k])))
    return volumes


def load_ply(ply_file_path):
    """
    Load point cloud data from a .ply file.
    """
    # Check if the .ply file exists
    assert os.path.isfile(ply_file_path), f"File {ply_file_path} not found."

    # Load the .ply file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors)

    # Derive metadata file path from .ply file path
    base_filename_without_extension = os.path.splitext(os.path.basename(ply_file_path))[0]
    metadata_file_path = os.path.join(os.path.dirname(ply_file_path), f"metadata_{base_filename_without_extension}.json")

    # Initialize metadata-related variables
    coeff, n, score, distance = None, None, None, None

    if os.path.isfile(metadata_file_path):
        with open(metadata_file_path, 'r') as metafile:
            metadata = json.load(metafile)
            coeff = np.array(metadata['coeff']) if 'coeff' in metadata and metadata['coeff'] is not None else None
            n = int(metadata['n']) if 'n' in metadata and metadata['n'] is not None else None
            score = metadata.get('score')
            distance = metadata.get('distance')

    return points, normals, colors, score, distance, coeff, n

def build_patch_from_data(patch_data, main_sheet_patch, subvolume_size, sample_ratio=1.0, path=""):
    """
    Build the patch dictionary from already loaded pointcloud data.
    
    Parameters
    ----------
    patch_data : tuple
        A tuple containing (patch_points, patch_normals, patch_color, patch_score, patch_distance, patch_coeff, n).
    main_sheet_patch : tuple
        A tuple in the form ((x, y, z), main_sheet_surface_nr, offset_angle).
    subvolume_size : int or sequence of int
        The size of the subvolume; will be converted to a NumPy array.
    sample_ratio : float, optional
        The ratio of points to sample.
    file_path : str, optional
        If provided, used to extract the patch id from the file name.
    
    Returns
    -------
    additional_main_patch : dict
        The dictionary containing the patch information.
    offset_angle : float
        The offset angle (as provided in main_sheet_patch).
    """
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    ply_file_path = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    (patch_points, patch_normals, patch_color, patch_score, patch_distance, patch_coeff, n) = patch_data

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    # Determine the patch id.
    # If a file_path is provided (as when loading from a ply file), extract it from the path.
    # Otherwise, default to using the main_sheet_patch information.
    patch_id = tuple([*map(int, ply_file_path.split("/")[-2].split("_"))]+[int(ply_file_path.split("/")[-1].split(".")[-2].split("_")[-1])])

    x, y, z, id_ = patch_id
    anchor_normal = get_vector_mean(patch_normals)
    anchor_angle = alpha_angles(np.array([anchor_normal]))[0]
    centroid = centroid_method(patch_points)

    additional_main_patch = {
        "ids": [patch_id],
        "points": patch_points,
        "normals": patch_normals,
        "colors": patch_color,
        "centroid": centroid,
        "anchor_points": [patch_points[0]],
        "anchor_normals": [anchor_normal],
        "anchor_angles": [anchor_angle],
        "angles": adjust_angles_offset(
                      adjust_angles_zero(alpha_angles(patch_normals), -anchor_angle),
                      offset_angle
                  ),
        "subvolume": [(x, y, z)],
        "subvolume_size": [subvolume_size],
        "iteration": 0,
        "patch_prediction_scores": [patch_score],
        "patch_prediction_distances": [patch_distance],
        "patch_prediction_coeff": [patch_coeff],
        "n": [n],
    }
    
    return additional_main_patch, offset_angle

# def build_patch(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
#     subvolume_size = np.array(subvolume_size)
#     ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
#     file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
#     res = load_ply(path)
#     patch_points = res[0]
#     patch_normals = res[1]
#     patch_color = res[2]
#     patch_score = res[3]
#     patch_distance = res[4]
#     patch_coeff = res[5]
#     n = res[6]

#     # Sample points from picked patch
#     patch_points, patch_normals, patch_color, _ = select_points(
#         patch_points, patch_normals, patch_color, patch_color, sample_ratio
#     )

#     patch_id = tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])])

#     x, y, z, id_ = patch_id
#     anchor_normal = get_vector_mean(patch_normals)
#     anchor_angle = alpha_angles(np.array([anchor_normal]))[0]
#     centroid = centroid_method(patch_points)

#     additional_main_patch = {"ids": [patch_id],
#                     "points": patch_points,
#                     "normals": patch_normals,
#                     "colors": patch_color,
#                     "centroid": centroid,
#                     "anchor_points": [patch_points[0]], 
#                     "anchor_normals": [anchor_normal],
#                     "anchor_angles": [anchor_angle],
#                     "angles": adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_normals), - anchor_angle), offset_angle),
#                     "subvolume": [(x, y, z)],
#                     "subvolume_size": [subvolume_size],
#                     "iteration": 0,
#                     "patch_prediction_scores": [patch_score],
#                     "patch_prediction_distances": [patch_distance],
#                     "patch_prediction_coeff": [patch_coeff],
#                     "n": [n],
#                     }
    
#     return additional_main_patch, offset_angle

def build_patch(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    """
    Build a patch from a PLY file.
    
    Loads the ply file from the given path (constructed using the main_sheet_patch information),
    calls load_ply to obtain the pointcloud data, and then delegates to build_patch_from_data.
    
    Parameters
    ----------
    main_sheet_patch : tuple
        A tuple in the form ((x, y, z), main_sheet_surface_nr, offset_angle).
    subvolume_size : int or sequence of int
        The size of the subvolume.
    path : str
        The base directory where the patch is stored.
    sample_ratio : float, optional
        The ratio of points to sample.
    align_and_flip_normals : bool, optional
        (Currently not used; included for compatibility.)
    
    Returns
    -------
    additional_main_patch : dict
        The dictionary containing the patch information.
    offset_angle : float
        The offset angle.
    """
    patch_data = load_ply(path)
    return build_patch_from_data(patch_data, main_sheet_patch, subvolume_size, sample_ratio, path=path)

def subvolume_surface_patches_folder(file, subvolume_size=50, sample_ratio=1.0):
    """
    Load surface patches from overlapping subvolume instance predictions.
    
    This function first looks for an HDF5 file (named "<file>.h5") and, if found,
    iterates through its groups (each group corresponds to one saved patch, as in
    save_block_h5). If no HDF5 file is found, it falls back to extracting a TAR 
    or 7z archive (with extension .tar or .7z) that contains .ply files. Each PLY 
    file is then processed with build_patch.
    
    Parameters
    ----------
    file : str
        Base name (or path without extension) of the archive/HDF5 file.
    subvolume_size : int or sequence of int, optional
        The size of the subvolume (will be converted to a 3-tuple).
    sample_ratio : float, optional
        The ratio of points to sample from each patch.
        
    Returns
    -------
    patches_list : list of dict
        A list of dictionaries containing the patch data.
    """
    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    patches_list = []

    # Construct possible filenames.
    h5_filename = f"{os.path.dirname(file)}.h5"
    tar_filename = f"{file}.tar"
    zip_filename = f"{file}.7z"

    if os.path.isfile(h5_filename):
        group_name = os.path.basename(file)
        # print(f"Loading patches from {h5_filename} with group name: {group_name}")
        # Load from HDF5 file.
        with h5py.File(h5_filename, "r") as h5f:
            if group_name in h5f:
                grp = h5f[group_name]
                for surface in grp:
                    surface_nr = int(surface.split("_")[-1])
                    try:
                        # print(f"Loading patch {surface} from {h5_filename}")
                        # Extract the datasets saved in each group.
                        points = grp[surface]["points"][()]       # e.g. an (N,3) array.
                        normals = grp[surface]["normals"][()]
                        colors = grp[surface]["colors"][()]
                        coeffs = grp[surface]["coeffs"][()]
                        # Retrieve attributes; in particular we need 'n'.
                        attrs = dict(grp[surface].attrs)
                        n_val = attrs.get("n", 0)
                        scores = attrs.get("scores", None)
                        distances = attrs.get("distances", None)

                        # print(f"min max colors: {np.min(colors, axis=0)} {np.max(colors, axis=0)}")
                        # print(f"shape colors: {colors.shape}, shape points: {points.shape}, shape normals: {normals.shape}, shape coeffs: {coeffs.shape}")
                        # time.sleep(10)
                        assert points.shape[0] == normals.shape[0], f"Inconsistent normals shapes. Points: {points.shape[0]}, Normals: {normals.shape[0]}, Colors: {colors.shape[0]}"
                        assert points.shape[0] == colors.shape[0], f"2 Inconsistent colors shapes. Points: {points.shape[0]}, Normals: {normals.shape[0]}, Colors: {colors.shape[0]}, Coeffs: {coeffs.shape[0]}"

                        # Package the data as expected by build_patch_from_data.
                        patch_data = (points, normals, colors, scores, distances, coeffs, n_val)

                        # Attempt to parse main_sheet_patch from the group name.
                        # Expecting a format like "x_y_z_patchNr"
                        parts = group_name.split('_')
                        if len(parts) >= 3:
                            try:
                                x = int(parts[0])
                                y = int(parts[1])
                                z = int(parts[2])
                            except Exception as e:
                                x, y, z = 0, 0, 0
                        else:
                            x, y, z = 0, 0, 0
                        # Use 0.0 as a default offset_angle.
                        main_sheet_patch = ((x, y, z), surface_nr, float(0.0))
                        # print(f"Loading patch {main_sheet_patch}")
                        # time.sleep(10)

                        # Rebuild the patch dictionary using the loaded data.
                        patch_dict, _ = build_patch_from_data(patch_data, main_sheet_patch, subvolume_size, sample_ratio, path=os.path.dirname(file))
                        patches_list.append(patch_dict)
                    except Exception as e:
                        print(f"Error loading subvolume {group_name} patch {surface_nr} from {h5_filename}: {e}")
    elif os.path.isfile(tar_filename) or os.path.isfile(zip_filename):
        # Open the archive
        if os.path.isfile(tar_filename):
            archive = tarfile.open(tar_filename, 'r')
            archive_type = 'tar'
        else:
            archive = py7zr.SevenZipFile(zip_filename, 'r')
            archive_type = '7z'

        with archive, tempfile.TemporaryDirectory() as temp_dir:
            if archive_type == 'tar':
                # Extract all .ply files
                archive.extractall(path=temp_dir)
                # Get list of .ply files
                ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]
            else:
                # Extract all files
                archive.extractall(path=temp_dir)
                # Get list of .ply files
                ply_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file_name in files:
                        if file_name.endswith(".ply"):
                            ply_files.append(os.path.relpath(os.path.join(root, file_name), temp_dir))

            # Process each .ply file
            for ply_file in ply_files:
                ply_file_path = os.path.join(temp_dir, ply_file)
                volume_ids = extract_ids(file)
                patch_number = int(os.path.splitext(os.path.basename(ply_file))[0].split('_')[-1])
                ids = (*volume_ids, patch_number)
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                surface_dict, _ = build_patch(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
                patches_list.append(surface_dict)
    return patches_list

def extract_ids(file_name):
    """
    Extract IDs from file name.
    """
    base_name = os.path.basename(file_name)
    name_parts = base_name.split("_")
    ids = tuple(map(int, name_parts))
    return ids

def build_patch_tar(main_sheet_patch, subvolume_size, path, sample_ratio=1.0):
    """
    Load surface patch from overlapping subvolumes instances predictions.
    """

    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    xyz, patch_nr, _ = main_sheet_patch
    file = path + f"/{xyz[0]:06}_{xyz[1]:06}_{xyz[2]:06}"
    tar_filename = f"{file}.tar"
    zip_filename = f"{file}.7z"

    if os.path.isfile(tar_filename) or os.path.isfile(zip_filename):
        # Open the archive
        if os.path.isfile(tar_filename):
            archive = tarfile.open(tar_filename, 'r')
            archive_type = 'tar'
        else:
            archive = py7zr.SevenZipFile(zip_filename, 'r')
            archive_type = '7z'

        with archive, tempfile.TemporaryDirectory() as temp_dir:
            # Extract all files
            archive.extractall(path=temp_dir)

            ply_file = f"surface_{patch_nr}.ply"
            ply_file_path = os.path.join(temp_dir, ply_file)
            volume_ids = extract_ids(file)
            ids = (*volume_ids, patch_nr)
            res = build_patch(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
            return res

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

    def add_node(self, node, centroid, winding_angle=None, winding_angle_gt=None, sample_points=None):
        node = tuple(int(node[i]) for i in range(4))
        self.nodes[node] = {'centroid': centroid, "winding_angle": winding_angle}
        if not winding_angle_gt is None:
            self.nodes[node]['winding_angle_gt'] = winding_angle_gt
        if not sample_points is None:
            self.nodes[node]['sample_points'] = sample_points

    def compute_node_edges(self, verbose=True):
        """
        Compute the edges for each node and store them in the node dictionary.
        """
        if verbose:
            print("Computing node edges...")
        for node in tqdm(self.nodes, desc="Adding nodes") if verbose else self.nodes:
            self.nodes[node]['edges'] = []
        for edge in tqdm(self.edges, desc="Computing node edges") if verbose else self.edges:
            for k in self.edges[edge]:
                self.nodes[edge[0]]['edges'].append(edge)
                self.nodes[edge[1]]['edges'].append(edge)

    def remove_nodes_edges(self, nodes):
        """
        Remove nodes and their edges from the graph.
        """
        for node in tqdm(nodes, desc="Removing nodes"):
            node = tuple(int(node[i]) for i in range(4))
            # Delete Node Edges
            node_edges = list(self.nodes[node]['edges'])
            for edge in node_edges:
                node_ = edge[0] if edge[0] != node else edge[1]
                if node_ in self.nodes:
                    self.nodes[node_]['edges'].remove(edge)
                # Delete Edges
                if edge in self.edges:
                    del self.edges[edge]
            # Delete Node
            del self.nodes[node]
        # set length of nodes and edges
        self.nodes_length = len(self.nodes)
        self.edges_length = len(self.edges)

    def delete_edge(self, edge):
        """
        Delete an edge from the graph.
        """
        # Delete Edges
        try:
            del self.edges[edge]
        except:
            pass
        # Delete Node Edges
        node1 = edge[0]
        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = edge[1]
        node2 = tuple(int(node2[i]) for i in range(4))
        self.nodes[node1]['edges'].remove(edge)
        self.nodes[node2]['edges'].remove(edge)

    def edge_exists(self, node1, node2):
        """
        Check if an edge exists in the graph.
        """
        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))
        return (node1, node2) in self.edges or (node2, node1) in self.edges

    def add_edge(self, node1, node2, certainty, sheet_offset_k=0.0, same_block=False, bad_edge=False):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            sheet_offset_k = sheet_offset_k * (-1.0)
        sheet_offset_k = (float)(sheet_offset_k)
        if not (node1, node2) in self.edges:
            self.edges[(node1, node2)] = {}
        self.edges[(node1, node2)][sheet_offset_k] = {'certainty': certainty, 'sheet_offset_k': sheet_offset_k, 'same_block': same_block, 'bad_edge': bad_edge}

    def add_increment_edge(self, node1, node2, certainty, sheet_offset_k=0.0, same_block=False, bad_edge=False):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            sheet_offset_k = sheet_offset_k * (-1.0)
        sheet_offset_k = (float)(sheet_offset_k)
        if not (node1, node2) in self.edges:
            self.edges[(node1, node2)] = {}
        if not sheet_offset_k in self.edges[(node1, node2)]:
            self.edges[(node1, node2)][sheet_offset_k] = {'certainty': 0.0, 'sheet_offset_k': sheet_offset_k, 'same_block': same_block, 'bad_edge': bad_edge}

        assert bad_edge == self.edges[(node1, node2)][sheet_offset_k]['bad_edge'], "Bad edge must be the same."
        if same_block != self.edges[(node1, node2)][sheet_offset_k]['same_block']:
            print("Same block must be the same.")
        self.edges[(node1, node2)][sheet_offset_k]['same_block'] = self.edges[(node1, node2)][sheet_offset_k]['same_block'] and same_block
        # Increment certainty
        self.edges[(node1, node2)][sheet_offset_k]['certainty'] += certainty

    def get_certainty(self, node1, node2, k):
        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))
        if node2 < node1:
            node1, node2 = node2, node1
            k = k * (-1.0)
        edge_dict = self.edges.get((node1, node2))
        if edge_dict is not None:
            edge = edge_dict.get(k)
            if edge is not None:
                return edge['certainty']
        return None
    
    def get_edge(self, node1, node2):
        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))
        if node2 < node1:
            node1, node2 = node2, node1
        return (node1, node2)
        
    def get_edge_ks(self, node1, node2):
        node1 = tuple(int(node1[i]) for i in range(4))
        node2 = tuple(int(node2[i]) for i in range(4))           
        k_factor = 1.0
        # Maintain bidirectional invariant
        if node2 < node1:
            node1, node2 = node2, node1
            k_factor = - 1.0
        edge_dict = self.edges.get((node1, node2))
        if edge_dict is not None:
            ks = []
            for k in edge_dict.keys():
                ks.append(k * k_factor)
            return ks
        else:
            raise KeyError(f"No edge found from {node1} to {node2}")
    
    def remove_unused_nodes(self, used_nodes):
        used_nodes = set([tuple(int(node[i]) for i in range(4)) for node in used_nodes])
        unused_nodes = []
        # Remove unused nodes
        for node in list(self.nodes.keys()):
            if tuple(node) not in used_nodes:
                unused_nodes.append(node)
        self.remove_nodes_edges(unused_nodes)
        self.compute_node_edges()
        
    def update_winding_angles(self, nodes, ks, update_winding_angles=False):
        nodes = [tuple(int(node[i]) for i in range(4)) for node in nodes]
        nodes_ks_dict = {}
        ks_min = np.min(ks)
        ks = np.array(ks) - ks_min
        for i, node in enumerate(nodes):
            nodes_ks_dict[node] = ks[i]
        # Update winding angles
        for node in nodes_ks_dict:
            k = nodes_ks_dict[node]
            node = tuple(int(node[i]) for i in range(4))
            self.nodes[node]['assigned_k'] = k
            if update_winding_angles:
                self.nodes[node]['winding_angle'] = - k*360 + self.nodes[node]['winding_angle']

    def get_nodes_and_ks(self):
        nodes = []
        ks = []
        for node in self.nodes:
            nodes.append(node)
            if 'assigned_k' in self.nodes[node]:
                ks.append(self.nodes[node]['assigned_k'])
            else:
                raise KeyError(f"No assigned k for node {node}")
        return nodes, ks

    def save_graph(self, path):
        print(f"Saving graph to {path} ...")
        # Save graph class object to file
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Graph saved to {path}")

    def bfs(self, start_node):
        """
        Breadth-first search from start_node.
        """
        start_node = tuple(int(start_node[i]) for i in range(4))
        visited = set()
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                edges = self.nodes[node]['edges']
                for edge in edges:
                    queue.append(edge[0] if edge[0] != node else edge[1])
        return list(visited)
    
    def neighbours(self, node, bfs_depth=3):
        """
        Return the list of neighbours of a node. Using bfs
        """
        node = tuple(int(node[i]) for i in range(4))
        visited = set()
        queue = [(node, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth + 1 >= bfs_depth:
                continue
            if node not in visited:
                visited.add(node)
                edges = self.nodes[node]['edges']
                for edge in edges:
                    queue.append((edge[0] if edge[0] != node else edge[1], depth + 1))
        return visited
    
    def update_neighbours_count(self, new_nodes, nodes, bfs_depth=3):
        """
        Update the neighbours count of the new nodes.
        """
        to_update_set = set([tuple(int(n[i]) for i in range(4)) for n in new_nodes])
        # print(f"Updating neighbours count for {len(new_nodes)} new nodes...")
        for node in new_nodes:
            node = tuple(int(node[i]) for i in range(4))
            neighbours_set = self.neighbours(tuple(node), bfs_depth)
            to_update_set = to_update_set.union(neighbours_set)

            self.nodes[tuple(node)]['neighbours_count'] = len(neighbours_set)
        
        others_to_update = to_update_set.difference(set([tuple(int(n[i]) for i in range(4)) for n in new_nodes]))
        # print(f"Reupdating neighbours count for {len(others_to_update)} other nodes...")
        for node in others_to_update:
            node = tuple(int(node[i]) for i in range(4))
            neighbours_set = self.neighbours(tuple(node), bfs_depth)
            self.nodes[tuple(node)]['neighbours_count'] = len(neighbours_set)
            # if len(neighbours_set) > 1:
            #     print(f"Node {node} has {len(neighbours_set)} neighbours.")

        all_nodes_counts = []
        # print(f"Adding neighbours count to all {len(nodes)} nodes...")
        for node in nodes:
            node = tuple(int(node[i]) for i in range(4))
            all_nodes_counts.append(self.nodes[tuple(node)]['neighbours_count'])

        if len(all_nodes_counts) > 0:
            return np.array(all_nodes_counts)
        else:
            return np.zeros(0)

def score_same_block_patches(patch1, patch2, overlapp_threshold, umbilicus_distance, min_points_factor):
    """
    Calculate the score between two patches from the same block.
    """
    # Calculate winding switch sheet scores
    distance_raw, orthogonal_d, geo_d = umbilicus_distance(patch1, patch2)
    score_raw, k = np.abs(distance_raw), np.sign(distance_raw)
    k = k * overlapp_threshold["winding_direction"] # because of scroll's specific winding direction

    score_val = (overlapp_threshold["max_winding_switch_sheet_distance"] - score_raw) / (overlapp_threshold["max_winding_switch_sheet_distance"] - overlapp_threshold["min_winding_switch_sheet_distance"]) # calculate score
    
    score = score_val * overlapp_threshold["winding_switch_sheet_score_factor"]

    valid = True
    # Check for enough points
    if geo_d > overlapp_threshold["max_winding_switch_sheet_distance"]:
        valid = False
        score = -0.5
    if geo_d < overlapp_threshold["min_winding_switch_sheet_distance"]:
        valid = False
        score = -1.0
    if orthogonal_d > overlapp_threshold["max_winding_switch_sheet_distance"]: # too far away
        valid = False
        score = -0.5
    if patch1["points"].shape[0] < overlapp_threshold["min_points_winding_switch"] * overlapp_threshold["sample_ratio_score"] * min_points_factor:
        valid = False
        score = -0.5
    if patch2["points"].shape[0] < overlapp_threshold["min_points_winding_switch"] * overlapp_threshold["sample_ratio_score"] * min_points_factor:
        valid = False
        score = -0.5
    if score_val <= 0.0: # too far away
        valid = False
        score = -0.5
    if score_raw > overlapp_threshold["max_winding_switch_sheet_distance"]: # not good patch, but whole volume might still be good for winding switch
        valid = False
        score = -0.5
    if (patch1["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]) or (patch2["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]):
        valid = False
        score = -0.5
    if score_val >= 1.0: # too close, whole volume went bad for winding switch
        valid = False
        score = -1.0
    if score_raw < overlapp_threshold["min_winding_switch_sheet_distance"]: # whole volume went bad for winding switch
        valid = False
        score = -1.0

    return (score, k, valid), patch1["anchor_angles"][0], patch2["anchor_angles"][0]

def process_same_block(main_block_patches_list, overlapp_threshold, umbilicus_distance):
    def scores_cleaned(direction_scores):
        if len(direction_scores) == 0:
            return []
        
        score_min = min(direction_scores, key=lambda x: x[2]) # return if there were bad scores
        if score_min[2] == -1.0:
            return []
        
        score = max(direction_scores, key=lambda x: x[2])
        score_val = score[2]
        if score_val < 0.0:
            score_val = -1.0
        score = (score[0], score[1], score_val, score[3], score[4], score[5], score[6], score[7])

        return [score]

    # calculate switching scores of main patches
    score_switching_sheets = []
    score_bad_edges = []
    for i in range(len(main_block_patches_list)):
        min_points_factor = 1.0
        score_switching_sheets_ = []
        for j in range(len(main_block_patches_list)):
            if i == j:
                continue
            (score_, k_, valid_), anchor_angle1, anchor_angle2 = score_same_block_patches(main_block_patches_list[i], main_block_patches_list[j], overlapp_threshold, umbilicus_distance, min_points_factor)
            if score_ > 0.0:
                score_switching_sheets_.append((main_block_patches_list[i]['ids'][0], main_block_patches_list[j]['ids'][0], score_, k_, anchor_angle1, anchor_angle2, main_block_patches_list[i]["centroid"], main_block_patches_list[j]["centroid"], valid_))

            # Add bad edges
            if valid_:
                score_bad_edges.append((main_block_patches_list[i]['ids'][0], main_block_patches_list[j]['ids'][0], 1.0, 0.0, anchor_angle1, anchor_angle2, main_block_patches_list[i]["centroid"], main_block_patches_list[j]["centroid"]))

        # filter and only take the scores closest to the main patch (smallest scores) for each k in +1, -1
        direction1_scores = [score for score in score_switching_sheets_ if score[3] > 0.0]
        direction2_scores = [score for score in score_switching_sheets_ if score[3] < 0.0]
        
        score1 = scores_cleaned(direction1_scores)
        score2 = scores_cleaned(direction2_scores)
        score_switching_sheets += score1 + score2
    return score_switching_sheets, score_bad_edges

def score_other_block_patches(patches_list, i, j, overlapp_threshold):
    """
    Calculate the score between two patches from different blocks.
    """
    # find out if close to angle%90 == 0
    # TODO: actually calculate this from the sheet configuration instead of approximating from the sheet position around the umbilicus
    # angle = (patch_angle(patches_list[i]) + patch_angle(patches_list[j])) / 2.0 # approximation
    min_points_factor = 1.0
    # right_angle = angle / 90.0 + 0.5
    # right_angle = right_angle - np.floor(right_angle) - 0.5
    # if abs(right_angle) < 0.333: # not needed anymore after graph gap fix
    #     min_points_factor = 0.25
    
    patch1 = patches_list[i]
    patch2 = patches_list[j]
    patches_list = [patch1, patch2]
    # Single threaded
    results = []
    results.append(compute_overlap_for_pair((0, patches_list, overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])))

    assert len(results) == 1, "Only one result should be returned."
    assert len(results[0]) == 1, "Only one pair of patches should be returned."

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold, sample_ratio=overlapp_threshold["sample_ratio_score"], min_points_factor=min_points_factor)

            if overlap < min_points_factor * overlapp_threshold["nr_points_min"] * overlapp_threshold["sample_ratio_score"] or overlap <= 0:
                score = -1.0
            elif score <= 0.0:
                score = -1.0
            elif patches_list[j]["points"].shape[0] < min_points_factor * overlapp_threshold["min_patch_points"] * overlapp_threshold["sample_ratio_score"]:
                score = -1.0
            elif patches_list[i]["points"].shape[0] < min_points_factor * overlapp_threshold["min_patch_points"] * overlapp_threshold["sample_ratio_score"]:
                score = -1.0
            elif patches_list[i]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"] or patches_list[j]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]:
                score = -1.0
            elif overlapp_threshold["fit_sheet"]:
                cost_refined, cost_percentile, cost_sheet_distance, surface = fit_sheet(patches_list, i, j, overlapp_threshold["cost_percentile"], overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])
                if cost_refined >= overlapp_threshold["cost_threshold"]:
                    score = -1.0
                elif cost_percentile >= overlapp_threshold["cost_percentile_threshold"]:
                    score = -1.0
                elif cost_sheet_distance >= overlapp_threshold["cost_sheet_distance_threshold"]:
                    score = -1.0

    return score, patch1["anchor_angles"][0], patch2["anchor_angles"][0]

def score_other_block_patches_old(patches_list, i, j, overlapp_threshold, angle):
    """
    Calculate the score between two patches from different blocks.
    """
    # find out if close to angle%90 == 0
    # TODO: actually calculate this from the sheet configuration instead of approximating from the sheet position around the umbilicus
    # angle = (patch_angle(patches_list[i]) + patch_angle(patches_list[j])) / 2.0 # approximation
    right_angle = angle / 90.0 + 0.5
    right_angle = right_angle - np.floor(right_angle) - 0.5
    min_points_factor = 1.0
    if abs(right_angle) < 0.333:
        min_points_factor = 0.25
    
    patch1 = patches_list[i]
    patch2 = patches_list[j]
    patches_list = [patch1, patch2]
    # Single threaded
    results = []
    results.append(compute_overlap_for_pair_old((0, patches_list, overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])))

    assert len(results) == 1, "Only one result should be returned."
    assert len(results[0]) == 1, "Only one pair of patches should be returned."

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold, sample_ratio=overlapp_threshold["sample_ratio_score"], min_points_factor=min_points_factor)

            if overlap < min_points_factor * overlapp_threshold["nr_points_min"] * overlapp_threshold["sample_ratio_score"] or overlap <= 0:
                score = -1.0
            elif score <= 0.0:
                score = -1.0
            elif patches_list[j]["points"].shape[0] < min_points_factor * overlapp_threshold["min_patch_points"] * overlapp_threshold["sample_ratio_score"]:
                score = -1.0
            elif patches_list[i]["points"].shape[0] < min_points_factor * overlapp_threshold["min_patch_points"] * overlapp_threshold["sample_ratio_score"]:
                score = -1.0
            elif patches_list[i]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"] or patches_list[j]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]:
                score = -1.0
            elif overlapp_threshold["fit_sheet"]:
                cost_refined, cost_percentile, cost_sheet_distance, surface = fit_sheet(patches_list, i, j, overlapp_threshold["cost_percentile"], overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])
                if cost_refined >= overlapp_threshold["cost_threshold"]:
                    score = -1.0
                elif cost_percentile >= overlapp_threshold["cost_percentile_threshold"]:
                    score = -1.0
                elif cost_sheet_distance >= overlapp_threshold["cost_sheet_distance_threshold"]:
                    score = -1.0

    return score, patch1["anchor_angles"][0], patch2["anchor_angles"][0]

def process_block(args):
    """
    Worker function to process a single block.
    """
    try:
        file_path, path_instances, overlapp_threshold, umbilicus_data = args
        umbilicus_func = lambda z: umbilicus_xz_at_y(umbilicus_data, z)
        def umbilicus_distance(patch1, patch2):
            # Geometric mean
            geo1 = patch1["centroid"]
            geo2 = patch2["centroid"]
            assert geo1.shape[0] == 3, "Points must be 3D."
            def d_(patch_point):
                umbilicus_point = umbilicus_func(patch_point[1])
                patch_point_vec = patch_point - umbilicus_point
                return np.linalg.norm(patch_point_vec), patch_point_vec
            d1, pv1 = d_(geo1)
            d2, pv2 = d_(geo2)
            # Find ortoghonal distance between the two points
            xv = pv1 * np.abs(d1 - d2) / d1
            pv12 = pv2 - pv1
            pv12_x = xv + pv12
            ortogh_length = np.linalg.norm(pv12_x)
            geo_d = np.linalg.norm(geo1 - geo2)
            return d1 - d2, ortogh_length, geo_d

        file_name = ".".join(file_path.split(".")[:-1])
        main_block_patches_list = subvolume_surface_patches_folder(file_name, sample_ratio=overlapp_threshold["sample_ratio_score"])

        patches_centroids = {}
        patches_points = {}
        for patch in main_block_patches_list:
            patches_centroids[tuple(patch["ids"][0])] = patch["centroid"]
            # take 10 points (if it has that many) from each patch
            random_points = patch["points"][np.random.choice(patch["points"].shape[0], size=min(10, patch["points"].shape[0]), replace=False)]
            patches_points[tuple(patch["ids"][0])] = random_points

        # Extract block's integer ID
        block_id = [int(i) for i in file_path.split('/')[-1].split('.')[0].split("_")]
        block_id = np.array(block_id)
        surrounding_ids = surrounding_volumes_kdtree_dict(block_id)
        # surrounding_ids = surrounding_volumes(block_id, overlapp_threshold)
        surrounding_blocks_patches_list = []
        surrounding_blocks_patches_list_ = []
        for surrounding_id in surrounding_ids:
            volume_path = path_instances + f"{file_path.split('/')[0]}/{surrounding_id[0]:06}_{surrounding_id[1]:06}_{surrounding_id[2]:06}"
            patches_temp = subvolume_surface_patches_folder(volume_path, sample_ratio=overlapp_threshold["sample_ratio_score"])
            surrounding_blocks_patches_list.append(patches_temp)
            surrounding_blocks_patches_list_.extend(patches_temp)

        # Add the overlap base to the patches list that contains the points + normals + scores only before
        patches_list = main_block_patches_list + surrounding_blocks_patches_list_
        add_overlapp_entries_to_patches_list(patches_list)

        # calculate scores between each main block patch and surrounding blocks patches
        score_sheets = []
        for i, main_block_patch in enumerate(main_block_patches_list):
            for surrounding_blocks_patches in surrounding_blocks_patches_list:
                score_sheets_patch = []
                for j, surrounding_block_patch in enumerate(surrounding_blocks_patches):
                    patches_list_ = [main_block_patch, surrounding_block_patch]
                    score_ = score_other_block_patches(patches_list_, 0, 1, overlapp_threshold) # score, anchor_angle1, anchor_angle2
                    if score_[0] > overlapp_threshold["final_score_min"]:
                        score_sheets_patch.append((main_block_patch['ids'][0], surrounding_block_patch['ids'][0], score_[0], None, score_[1], score_[2], main_block_patch["centroid"], surrounding_block_patch["centroid"]))

                # Find the best score for each main block patch
                if len(score_sheets_patch) > 0:
                    # sort and take top_n
                    top_n = 1
                    score_sheets_patch = sorted(score_sheets_patch, key=lambda x: x[2], reverse=True)
                    max_ind = min(top_n, len(score_sheets_patch))
                    score_sheets_patch = score_sheets_patch[:max_ind]
                    # score_sheets_patch = max(score_sheets_patch, key=lambda x: x[2])
                    score_sheets.extend(score_sheets_patch)
        
        score_switching_sheets, score_bad_edges = process_same_block(main_block_patches_list, overlapp_threshold, umbilicus_distance)

        # Process and return results...
        return score_sheets, score_switching_sheets, score_bad_edges, patches_centroids, patches_points
    except Exception as e:
        print(f"function process_block (instances_to_graph.py) Error: {e}")
        return [], [], [], {}, {}

def init_worker_build_GT(mesh_file, pointcloud_dir):
    """
    Initialize worker for building the ground truth.
    """
    global valid_triangles, winding_angles, output_dir, gt_splitter, mesh_gt_stuff
    valid_triangles, winding_angles, output_dir, gt_splitter = set_up_mesh(mesh_file, pointcloud_dir, continue_from=3, use_tempfile=False) # already initialized, only load data
    mesh_gt_stuff = load_mesh_vertices_quality(mesh_file, use_tempfile=False) # each worker needs its own copy. # spelufo coordinate system
    print("Worker initialized for building the ground truth.")

def worker_build_GT(args):
    # print(f"Processing: {args}")
    filename, umbilicus_path = args
    nodes_winding_alignment = []

    def process_ply_files(ply_files):
        # Process each .ply file
        for i, ply_file_path in enumerate(ply_files):
            ply_file = os.path.basename(ply_file_path)
            ids = tuple([*map(int, filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
            ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))

            # 3D instance pointcloud
            vertices1 = o3d.io.read_point_cloud(ply_file_path)
            # to numpy
            vertices1 = np.asarray(vertices1.points)
            # adjust coordinate frame
            vertices1 = scale_points(vertices1, 4.0, axis_offset=-500)
            vertices1_spelufo = vertices1 + 500
            vertices1, _ = shuffling_points_axis(vertices1, vertices1, [2, 0, 1]) # Scan Coordinate System
            winding_angles1 = calculate_winding_angle_pointcloud_instance(vertices1, gt_splitter) # instance winding angles. # Takes Scan coordinate system

            # align winding angles
            winding_angle_difference, winding_angles2, scene2, mesh2, percentage_valid = align_winding_angles(vertices1_spelufo, winding_angles1, mesh_gt_stuff, umbilicus_path, 15, gt_splitter, debug=False) # aling to GT mesh per point. # takes spelufo coordinate system
            if percentage_valid > 0.5:
                best_alignment = find_best_alignment(winding_angle_difference) # best alignment to a wrap
                # Adjust winding angle of graph nodes
                nodes_winding_alignment.append((ids, best_alignment))
            elif i == 0:
                # Check if too far away from GT Mesh
                winding_angle_difference, winding_angles2, scene2, mesh2, percentage_valid = align_winding_angles(vertices1_spelufo, winding_angles1, mesh_gt_stuff, umbilicus_path, 210, gt_splitter) # aling to GT mesh per point. # takes spelufo coordinate system
                # Still 0 percent -> no patch in this subvolume is valid
                if percentage_valid == 0.0:
                    break
                
    if os.path.isfile(filename):
        # Handle .tar files
        if filename.endswith(".tar"):
            with tarfile.open(filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
                # Extract all .ply files at once
                archive.extractall(path=temp_dir, members=archive.getmembers())
                ply_files = [os.path.join(temp_dir, m.name) for m in archive.getmembers() if m.name.endswith(".ply")]
                process_ply_files(ply_files)
                        
        # Handle .7z files
        elif filename.endswith(".7z"):
            with py7zr.SevenZipFile(filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
                archive.extractall(path=temp_dir)
                # List all extracted .ply files
                ply_files = [
                    os.path.join(root, file)
                    for root, _, files in os.walk(temp_dir)
                    for file in files if file.endswith(".ply")
                ]
                process_ply_files(ply_files)

    return nodes_winding_alignment
    
def init_worker_build_graph(centroid_method_, surrounding_dict_):
    global centroid_method, surrounding_dict
    centroid_method = centroid_method_
    surrounding_dict = surrounding_dict_

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()
        self.set_overlapp_threshold(overlapp_threshold)
        self.umbilicus_path = umbilicus_path
        self.init_umbilicus(umbilicus_path)

    def init_umbilicus(self, umbilicus_path):
        # Load the umbilicus data
        umbilicus_data = load_xyz_from_file(umbilicus_path)
        # scale and swap axis
        self.umbilicus_data = scale_points(umbilicus_data, 50.0/200.0, axis_offset=0)

    def add_switch_edge(self, node_lower, node_upper, certainty, same_block):
        # Build a switch edge between two nodes
        self.add_edge(node_lower, node_upper, certainty, sheet_offset_k=1.0, same_block=same_block)

    def add_same_sheet_edge(self, node1, node2, certainty, same_block):
        # Build a same sheet edge between two nodes
        self.add_edge(node1, node2, certainty, sheet_offset_k=0.0, same_block=same_block)

    def build_other_block_edges(self, score_sheets, update_edges=False, nodes_set=None):
        if update_edges:
            # Delete all edges
            for edge in list(self.edges.keys()):
                for k in list(self.edges[edge].keys()):
                    if not self.edges[edge][k]['same_block']:
                        del self.edges[edge][k]
                if len(self.edges[edge]) == 0:
                    del self.edges[edge]

        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Build edges between patches from different blocks
        for score_ in score_sheets:
            id1, id2, score, _, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            if score < self.overlapp_threshold["final_score_min"]:
                continue
            umbilicus_point1 = umbilicus_func(centroid1[1])[[0, 2]]
            umbilicus_vector1 = umbilicus_point1 - centroid1[[0, 2]]
            angle1 = angle_between(umbilicus_vector1) * 180.0 / np.pi
            umbilicus_point2 = umbilicus_func(centroid2[1])[[0, 2]]
            umbilicus_vector2 = umbilicus_point2 - centroid2[[0, 2]]
            angle2 = angle_between(umbilicus_vector2) * 180.0 / np.pi
            angle_diff = angle2 - angle1
            if angle_diff > 180.0:
                angle_diff -= 360.0
            if angle_diff < -180.0:
                angle_diff += 360.0
            if update_edges:
                # Only add edges if both nodes are in the set
                if id1 in nodes_set and id2 in nodes_set:
                    self.add_edge(id1, id2, score, sheet_offset_k=angle_diff, same_block=False)
            else:
                # Add all edges
                self.add_edge(id1, id2, score, sheet_offset_k=angle_diff, same_block=False)

    def build_same_block_edges(self, score_switching_sheets, update_edges=False, nodes_set=None):
        if update_edges:
            # Delete all edges
            for edge in list(self.edges.keys()):
                for k in list(self.edges[edge].keys()):
                    if self.edges[edge][k]['same_block']:
                        del self.edges[edge][k]
                if len(self.edges[edge]) == 0:
                    del self.edges[edge]
        
        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Build edges between patches from the same block
        disregarding_count = 0
        total_count = 0
        grand_total = 0
        for score_ in score_switching_sheets:
            grand_total += 1
            id1, id2, score, k, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            if score <= 0.0:
                continue
            total_count += 1
            umbilicus_point1 = umbilicus_func(centroid1[1])[[0, 2]]
            umbilicus_vector1 = umbilicus_point1 - centroid1[[0, 2]]
            angle1 = angle_between(umbilicus_vector1) * 180.0 / np.pi
            umbilicus_point2 = umbilicus_func(centroid2[1])[[0, 2]]
            umbilicus_vector2 = umbilicus_point2 - centroid2[[0, 2]]
            angle2 = angle_between(umbilicus_vector2) * 180.0 / np.pi
            angle_diff = angle2 - angle1
            if angle_diff > 180.0:
                angle_diff -= 360.0
            if angle_diff < -180.0:
                angle_diff += 360.0
            angle_diff -= k * 360.0 # Next/Previous winding
            if update_edges:
                # Only add edges if both nodes are in the set
                if id1 in nodes_set and id2 in nodes_set:
                    self.add_edge(id1, id2, score, sheet_offset_k=angle_diff, same_block=True)
            else:
                # Add all edges
                self.add_edge(id1, id2, score, sheet_offset_k=angle_diff, same_block=True)

    def build_bad_edges(self, score_bad_edges):
        for score_ in score_bad_edges:
            node1, node2, score, k, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            # Build a same sheet edge between two nodes in same subvolume that is bad
            self.add_edge(node1, node2, 1.0, sheet_offset_k=0.0, same_block=True, bad_edge=True)

    def filter_blocks(self, blocks_tar_files, blocks_tar_files_int, start_block, distance):
        blocks_tar_files_int = np.array(blocks_tar_files_int)
        start_block = np.array(start_block)
        distances = np.abs(blocks_tar_files_int - start_block)
        distances = np.sum(distances, axis=1)
        filter_mask = np.sum(np.abs(blocks_tar_files_int - start_block), axis=1) < distance
        blocks_tar_files_int = blocks_tar_files_int[filter_mask]
        blocks_tar_files = [blocks_tar_files[i] for i in range(len(blocks_tar_files)) if filter_mask[i]]
        return blocks_tar_files, blocks_tar_files_int
    
    def filter_blocks_z(self, blocks_tar_files, z_min, z_max):
        # Filter blocks by z range
        blocks_tar_files_int = [[int(i) for i in x.split('/')[-1].split('.')[0].split("_")] for x in blocks_tar_files]
        blocks_tar_files_int = np.array(blocks_tar_files_int)
        filter_mask = np.logical_and(blocks_tar_files_int[:,1] >= z_min, blocks_tar_files_int[:,1] <= z_max)
        blocks_tar_files = [blocks_tar_files[i] for i in range(len(blocks_tar_files)) if filter_mask[i]]
        return blocks_tar_files
    
    def largest_connected_component(self, delete_nodes=True, min_size=None):
        print("Finding largest connected component...")        
        # walk the graph from the start node
        visited = set()
        # tqdm object showing process of visited nodes untill all nodes are visited
        tqdm_object = tqdm(total=len(self.nodes))
        components = []
        starting_index = 0
        def build_nodes(edges):
            nodes = set()
            for edge in edges:
                nodes.add(edge[0])
                nodes.add(edge[1])
            return list(nodes)
        nodes = build_nodes(self.edges)
        max_component_length = 0
        nodes_length = len(nodes)
        nodes_remaining = nodes_length
        print(f"Number of active nodes: {nodes_length}, number of total nodes: {len(self.nodes)}")
        while True:
            start_node = None
            # Pick first available unvisited node
            for node_idx in range(starting_index, nodes_length):
                node = nodes[node_idx]
                if node not in visited:
                    start_node = node
                    break
            if start_node is None:
                break    

            queue = [start_node]
            component = set()
            while queue:
                node = queue.pop(0)
                component.add(node)
                if node not in visited:
                    tqdm_object.update(1)
                    visited.add(node)
                    edges = self.nodes[node]['edges']
                    for edge in edges:
                        queue.append(edge[0] if edge[0] != node else edge[1])

            components.append(component)
            # update component length tracking
            component_length = len(component)
            max_component_length = max(max_component_length, component_length)
            nodes_remaining -= component_length
            if (min_size is None) and (nodes_remaining < max_component_length): # early stopping, already found the largest component
                print(f"breaking in early stopping")
                print()
                break

        nodes_total = len(self.nodes)
        edges_total = len(self.edges)
        print(f"number of nodes: {nodes_total}, number of edges: {edges_total}")

        # largest component
        largest_component = list(max(components, key=len))
        if min_size is not None:
            components = [component for component in components if len(component) >= min_size]
            for component in components:
                largest_component.extend(list(component))
        largest_component = set(largest_component)
        
        # Remove all other Nodes, Edges and Node Edges
        if delete_nodes:
            other_nodes = [node for node in self.nodes if node not in largest_component]
            self.remove_nodes_edges(other_nodes)
        
        print(f"Pruned {nodes_total - len(self.nodes)} nodes. Of {nodes_total} nodes.")
        print(f"Pruned {edges_total - len(self.edges)} edges. Of {edges_total} edges.")

        result = list(largest_component)
        print(f"Found largest connected component with {len(result)} nodes.")
        return result
    
    def update_graph_version(self):
        edges = self.edges
        self.edges = {}
        for edge in tqdm(edges):
            self.add_edge(edge[0], edge[1], edges[edge]['certainty'], edges[edge]['sheet_offset_k'], edges[edge]['same_block'])
        self.compute_node_edges()

    def flip_winding_direction(self):
        for edge in tqdm(self.edges):
            for k in self.edges[edge]:
                if self.edges[edge][k]['same_block']:
                    self.edges[edge][k]['sheet_offset_k'] = - self.edges[edge][k]['sheet_offset_k']

    def compute_bad_edges(self, iteration, k_factor_bad=1.0):
        print(f"Bad k factor is {k_factor_bad}.")
        # Compute bad edges
        node_cubes = {}
        for node in tqdm(self.nodes):
            if node[:3] not in node_cubes:
                node_cubes[node[:3]] = []
            node_cubes[node[:3]].append(node)

        # add bad edges between adjacent nodes
        # two indices equal, abs third is 25
        count_added_bad_edges = 0
        for node_cube in node_cubes:
            node_cube = np.array(node_cube)
            adjacent_cubes = [node_cube, node_cube + np.array([0, 0, 25]), node_cube + np.array([0, 25, 0]), node_cube + np.array([25, 0, 0]), node_cube + np.array([0, 0, -25]), node_cube + np.array([0, -25, 0]), node_cube + np.array([-25, 0, 0])]
            for adjacent_cube in adjacent_cubes:
                if tuple(adjacent_cube) in node_cubes:
                    node1s = node_cubes[tuple(node_cube)]
                    node2s = node_cubes[tuple(adjacent_cube)]
                    for node1 in node1s:
                        for node2 in node2s:
                            if node1 == node2:
                                continue
                            if not self.edge_exists(node1, node2):
                                self.add_edge(node1, node2, k_factor_bad*((iteration+1)**2), 0.0, same_block=True, bad_edge=True)
                                count_added_bad_edges += 1
                            else:
                                edge = self.get_edge(node1, node2)
                                for k in self.get_edge_ks(edge[0], edge[1]):
                                    if k == 0:
                                        continue
                                    same_block = self.edges[edge][k]['same_block']
                                    bad_edge = self.edges[edge][k]['bad_edge']
                                    if not bad_edge and same_block:
                                        if not k in self.edges[edge]:
                                            self.add_increment_edge(node1, node2, k_factor_bad*((iteration+1)**2), k, same_block=True, bad_edge=True)
                                            count_added_bad_edges += 1

            nodes = node_cubes[tuple(node_cube)]
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    self.add_edge(nodes[i], nodes[j], k_factor_bad*((iteration+1)**2), 0.0, same_block=True, bad_edge=True)

        print(f"Added {count_added_bad_edges} bad edges.")

    def adjust_edge_certainties(self):
        # Adjust same block and other block edge certainty values, such that both have the total quantity as even
        certainty_same_block = 0.0
        certainty_other_block = 0.0
        certainty_bad_block = 0.0
        for edge in tqdm(self.edges, desc="Calculating edge certainties"):
            for k in self.edges[edge]:
                if self.edges[edge][k]['bad_edge']:
                    certainty_bad_block += self.edges[edge][k]['certainty']
                elif self.edges[edge][k]['same_block']:
                    certainty_same_block += self.edges[edge][k]['certainty']
                else:
                    certainty_other_block += self.edges[edge][k]['certainty']

        factor_0 = 1.0
        factor_not_0 = 1.0*certainty_same_block / certainty_other_block
        factor_bad = certainty_same_block / certainty_bad_block
        # factor_bad = factor_bad**(2.0/3.0)
        # factor_bad = 2*factor_bad
        # factor_bad = 1.0
        print(f"Adjusted certainties: factor_0: {factor_0}, factor_not_0: {factor_not_0}, factor_bad: {factor_bad}")

        # adjust graph edge certainties
        for edge in tqdm(self.edges, desc="Adjusting edge certainties"):
            for k in self.edges[edge]:
                if self.edges[edge][k]['bad_edge']:
                    self.edges[edge][k]['certainty'] = factor_bad * self.edges[edge][k]['certainty']
                elif self.edges[edge][k]['same_block']:
                    self.edges[edge][k]['certainty'] = factor_0 * self.edges[edge][k]['certainty']
                else:
                    self.edges[edge][k]['certainty'] = factor_not_0 * self.edges[edge][k]['certainty']

        self.compute_node_edges()
        return (factor_0, factor_not_0, factor_bad)
    
    def add_ground_truth_from_mesh(self, path_instances, mesh_file, continue_from=0):
        umbilicus_path = os.path.join(os.path.dirname(path_instances), "umbilicus.txt")
        # Load GT data

        if continue_from <= 2:
            set_up_mesh(mesh_file, path_instances, continue_from=continue_from) # each worker needs its own copy, initialize single threaded, then load the result into the threads
        # Load the node instance, compare to scroll GT mesh, add winding angle information if instance within the GT mesh.
        # use multiprocessing pool with initializer for scene
        with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
            # copy mesh to tempfile
            temp_path = temp_file.name
            # os copy
            os.system(f"cp {mesh_file} {temp_path}")
            
            # with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2, initializer=init_worker_build_GT, initargs=(mesh_file, path_instances)) as pool:
            with multiprocessing.Pool(processes=min(48, multiprocessing.cpu_count()//2), initializer=init_worker_build_GT, initargs=(temp_path, path_instances)) as pool:
                # PC instance path from node name
                blocks_tar_files = glob.glob(path_instances + '/*.tar')
                ## multithread
                tasks = []
                for i, blocks_tar_file in enumerate(blocks_tar_files):
                    # Append tasks without passing the large constant data
                    tasks.append((blocks_tar_file, umbilicus_path))

                print(f"Processing {len(tasks)} blocks")
                
                # tasks = tasks[:min(len(tasks), 10)] # DEBUG ONLY, TODO: remove
                # print(f"Processing {len(tasks)} blocks")

                # Use pool.imap to execute the worker function
                results = list(tqdm(pool.imap(worker_build_GT, tasks), desc="Processing GT point clouds", total=len(tasks)))

        # # single threaded
        # init_worker_build_GT(mesh_file, path_instances)
        # # PC instance path from node name
        # blocks_tar_files = glob.glob(path_instances + '/*.tar')
        # ## multithread
        # tasks = []
        # for i, blocks_tar_file in enumerate(blocks_tar_files):
        #     # Append tasks without passing the large constant data
        #     tasks.append((blocks_tar_file, umbilicus_path))

        # print(f"Processing {len(tasks)} blocks")
        
        # # tasks = tasks[:min(len(tasks), 10)] # DEBUG ONLY, TODO: remove
        # # print(f"Processing {len(tasks)} blocks")

        # # Use pool.imap to execute the worker function
        # results = []
        # for task in tqdm(tasks, desc="Processing GT point clouds"):
        #     results.append(worker_build_GT(task))

        # for all edges between two nodes with GT data, decide if edge is in-/valid
        count_adjusted_nodes_windings = 0
        for result in tqdm(results, desc="Adding GT data to nodes"):
            for node_id, best_alignment in result:
                if node_id in self.nodes:
                    self.nodes[node_id]['winding_angle_gt'] = self.nodes[node_id]['winding_angle'] - best_alignment
                    count_adjusted_nodes_windings += 1
                else:
                    print(f"Node {node_id} not found in graph.")
        print(f"Adjusted winding angles for {count_adjusted_nodes_windings} nodes.")

    
    def build_graph(self, path_instances, start_point, num_processes=4, prune_unconnected=False, start_fresh=True, gt_mesh_file=None, continue_from=0, update_edges=False):
        #from original coordinates to instance coordinates
        start_block, patch_id = (0, 0, 0), 0
        self.start_block, self.patch_id, self.start_point = start_block, patch_id, start_point

        if start_fresh:
            blocks_tar_files = glob.glob(path_instances + '/*.tar')
            blocks_7z_files  = glob.glob(path_instances + '/*.7z')
            blocks_h5_filename = os.path.join(path_instances + ".h5")
            blocks_h5_files = []
            
            if os.path.exists(blocks_h5_filename):
                with h5py.File(blocks_h5_filename, "r") as h5f:
                    blocks_h5_files = [os.path.join(path_instances, key + ".ply") for key in h5f.keys()]
                    print(f"Found {len(blocks_h5_files)} blocks in HDF5 file.")
                    # group_path = "scroll_pcs/point_cloud_colorized_verso_subvolume_blocks"

                    # with h5py.File(blocks_h5_filename.replace(".h5", "_.h5"), "w") as new_h5:
                    #     for key in h5f[group_path]:  # Iterate over items in group
                    #         new_h5.copy(h5f[group_path][key], new_h5, name=key)
                    #         print(f"Copied '{key}' to root level.")

            blocks_files = blocks_tar_files + blocks_7z_files + blocks_h5_files
            # filter out blocks that contain '_temp' in their name
            blocks_files = [block_file for block_file in blocks_files if '_temp' not in block_file]
            # sort blocks files
            blocks_files = sorted(blocks_files) # gimmick
            # debug with first 100
            # blocks_files = blocks_files[:1000]
            print(f"Found {len(blocks_files)} blocks before filtering.")
            blocks_files = [block_file for block_file in blocks_files if int(block_file.split('/')[-1].split('.')[0].split("_")[1]) >= self.overlapp_threshold["sheet_z_range"][0] and int(block_file.split('/')[-1].split('.')[0].split("_")[1]) <= self.overlapp_threshold["sheet_z_range"][1]]
            print(f"Found {len(blocks_files)} blocks.")
            kd_tree, block_ids = build_kd_tree(blocks_files)
            # Create a shared dictionary using Manager
            manager = Manager()
            surrounding_dict_shared = manager.dict(build_surrounding_volumes_dict(kd_tree, block_ids, self.overlapp_threshold))
            print("Building graph...")
            # Create a pool of worker processes
            with Pool(num_processes, initializer=init_worker_build_graph, initargs=(centroid_method,surrounding_dict_shared,)) as pool: # num_processes
                print(f"Number of blocks: {len(blocks_files)}")
                # Map the process_block function to each file
                zipped_args = list(zip(blocks_files, [path_instances] * len(blocks_files), [self.overlapp_threshold] * len(blocks_files), [self.umbilicus_data] * len(blocks_files)))
                # for za in zipped_args:
                #     process_block(za)
                results = list(tqdm(pool.imap(process_block, zipped_args), total=len(zipped_args)))

            print(f"Number of results: {len(results)}")

            count_res = 0
            patches_centroids = {}
            patches_points = {}
            nodes_set = frozenset(self.nodes.keys()) # faster membership test
            # Process results from each worker
            for score_sheets, score_switching_sheets, score_bad_edges, volume_centroids, volume_points in tqdm(results, desc="Processing results"):
                count_res += len(score_sheets)
                # Calculate scores, add patches edges to graph, etc.
                self.build_other_block_edges(score_sheets, update_edges=update_edges, nodes_set=nodes_set)
                self.build_same_block_edges(score_switching_sheets, update_edges=update_edges, nodes_set=nodes_set)
                # self.build_bad_edges(score_bad_edges)
                patches_centroids.update(volume_centroids)
                patches_points.update(volume_points)
            print(f"Number of results: {count_res}")

            if not update_edges:
                # Define a wrapper function for umbilicus_xz_at_y
                umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)

                # Add patches as nodes to graph
                edges_keys = list(self.edges.keys())
                nodes_from_edges = set()
                for edge in tqdm(edges_keys, desc="Initializing nodes"):
                    nodes_from_edges.add(edge[0])
                    nodes_from_edges.add(edge[1])
                for node in tqdm(nodes_from_edges, desc="Adding nodes"):
                    try:
                        umbilicus_point = umbilicus_func(patches_centroids[node][1])[[0, 2]]
                        umbilicus_vector = umbilicus_point - patches_centroids[node][[0, 2]]
                        angle = angle_between(umbilicus_vector) * 180.0 / np.pi
                        node_points = patches_points[node]
                        self.add_node(node, patches_centroids[node], winding_angle=angle, sample_points=node_points)
                    except:
                        try:
                            del self.edges[edge]
                        except:
                            pass

            node_id = tuple((*start_block, patch_id))
            print(f"Start node: {node_id}, nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")
            self.compute_node_edges()
            if prune_unconnected:
                print("Prunning unconnected nodes...")
                self.largest_connected_component()

            print(f"Nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")

        # Add GT data to nodes
        if gt_mesh_file is not None:
            self.add_ground_truth_from_mesh(path_instances, gt_mesh_file, continue_from=continue_from) # TODO

        return start_block, patch_id
    
    def naked_graph(self):
        """
        Return a naked graph with only nodes.
        """
        # Remove all nodes and edges
        graph = ScrollGraph(self.overlapp_threshold, self.umbilicus_path)
        # add all nodes
        for node in self.nodes:
            graph.add_node(node, self.nodes[node]['centroid'], winding_angle=self.nodes[node]['winding_angle'], sample_points=self.nodes[node]['sample_points'] if 'sample_points' in self.nodes[node] else None)
        return graph
    
    def set_overlapp_threshold(self, overlapp_threshold):
        if hasattr(self, "overlapp_threshold"):
            # compare if the new threshold is different from the old one in any aspect or subaspect
            different = False
            if not different and set(overlapp_threshold.keys()) != set(self.overlapp_threshold.keys()):
                different = True

            # Check if all values for each key are the same
            if not different:
                for key in overlapp_threshold:
                    if overlapp_threshold[key] != self.overlapp_threshold[key]:
                        different = True
                        break

            if not different:
                print("Overlapping threshold is the same. Not updating.")
                return

        print("Setting overlapping threshold...")
        self.overlapp_threshold = overlapp_threshold
        print("Overlapping threshold set.")

    def create_dxf_with_colored_polyline(self, filename, color=1, min_z=None, max_z=None):
        # Create a new DXF document.
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for edge in tqdm(self.edges, desc="Creating DXF..."):
            ks = self.get_edge_ks(edge[0], edge[1])
            for k in ks:
                same_block = self.edges[edge][k]['same_block']
                c = color
                if same_block:
                    c = 2
                elif k != 0.0:
                    c = 5
                # Create polyline points 
                polyline_points = []
                polyline_points_raw = []
                to_add = True
                for pi, point in enumerate(edge):
                    centroid = self.nodes[point]['centroid']
                    if min_z is not None and centroid[1] < min_z:
                        to_add = False
                    if max_z is not None and centroid[1] > max_z:
                        to_add = False
                    if not to_add:
                        break
                    polyline_points.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))
                    polyline_points_raw.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))
                    
                    # Add an indicator which node of the edge has the smaller k value
                    if same_block:
                        if k > 0.0 and pi == 0:
                            polyline_points = [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))] + polyline_points
                        elif k < 0.0 and pi == 1:
                            polyline_points += [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))]

                if same_block and k == 0.0:
                    c = 4

                # Add an indicator which node of the edge has the smaller k value
                if k != 0.0:
                    # Add indicator for direction
                    start_point = Vec3(polyline_points_raw[0].x+1, polyline_points_raw[0].y+1, polyline_points_raw[0].z+1)
                    end_point = Vec3(polyline_points_raw[1].x+1, polyline_points_raw[1].y+1, polyline_points_raw[1].z+1)
                    if k < 0.0:
                        start_point, end_point = end_point, start_point
                    indicator_vector = (end_point - start_point) * 0.5
                    indicator_end = start_point + indicator_vector
                    if c == 2:
                        indicator_color = 7
                    else:
                        indicator_color = 6  # Different color for the indicator
                    
                    # Add the indicator line
                    msp.add_line(start_point, indicator_end, dxfattribs={'color': indicator_color})

                if to_add:
                    # Add the 3D polyline to the model space
                    msp.add_polyline3d(polyline_points, dxfattribs={'color': c})

        # Save the DXF document
        doc.saveas(filename)
    
    def compare_polylines_graph(self, other, filename, color=1, min_z=None, max_z=None):
        # Create a new DXF document.
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for edge in tqdm(self.edges):
            ks = self.get_edge_ks(edge[0], edge[1])
            for k in ks:
                same_block = self.edges[edge][k]['same_block']
                c = color
                if same_block:
                    c = 2
                elif k != 0.0:
                    c = 5
                if edge not in other.edges:
                    if c == 2:
                        c = 4
                    elif c == 5:
                        c = 6
                    else:
                        c = 3
                
                # Create polyline points 
                polyline_points = []
                to_add = True
                for pi, point in enumerate(edge):
                    centroid = self.nodes[point]['centroid']
                    if min_z is not None and centroid[1] < min_z:
                        to_add = False
                    if max_z is not None and centroid[1] > max_z:
                        to_add = False
                    if not to_add:
                        break
                    polyline_points.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))

                    # Add an indicator which node of the edge has the smaller k value
                    if same_block:
                        if k > 0.0 and pi == 0:
                            polyline_points = [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))] + polyline_points
                        elif k < 0.0 and pi == 1:
                            polyline_points += [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))]

                if to_add:
                    # Add the 3D polyline to the model space
                    msp.add_polyline3d(polyline_points, dxfattribs={'color': c})

        # Save the DXF document
        doc.saveas(filename)

    def extract_subgraph(self, min_z=None, max_z=None, umbilicus_max_distance=None, add_same_block_edges=False, tolerated_nodes=None, min_x=None, max_x=None, min_y=None, max_y=None):
        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Extract subgraph with nodes within z range
        subgraph = ScrollGraph(self.overlapp_threshold, self.umbilicus_path)
        # starting block info
        subgraph.start_block = self.start_block
        subgraph.patch_id = self.patch_id
        for node in tqdm(self.nodes, desc="Extracting subgraph..."):
            centroid = self.nodes[node]['centroid']
            winding_angle = self.nodes[node]['winding_angle']
            if (tolerated_nodes is not None) and (node in tolerated_nodes):
                subgraph.add_node(node, centroid, winding_angle=winding_angle, winding_angle_gt=self.nodes[node]['winding_angle_gt'] if 'winding_angle_gt' in self.nodes[node] else None, sample_points=self.nodes[node]['sample_points'] if 'sample_points' in self.nodes[node] else None)
                continue
            elif (min_z is not None) and (centroid[1] < min_z):
                continue
            elif (max_z is not None) and (centroid[1] > max_z):
                continue
            elif (min_x is not None) and (centroid[2] < min_x):
                continue
            elif (max_x is not None) and (centroid[2] > max_x):
                continue
            elif (min_y is not None) and (centroid[0] < min_y):
                continue
            elif (max_y is not None) and (centroid[0] > max_y):
                continue
            elif (umbilicus_max_distance is not None) and np.linalg.norm(umbilicus_func(centroid[1]) - centroid) > umbilicus_max_distance:
                continue
            else:
                subgraph.add_node(node, centroid, winding_angle=winding_angle, winding_angle_gt=self.nodes[node]['winding_angle_gt'] if 'winding_angle_gt' in self.nodes[node] else None, sample_points=self.nodes[node]['sample_points'] if 'sample_points' in self.nodes[node] else None)
        for edge in self.edges:
            node1, node2 = edge
            if (tolerated_nodes is not None) and (node1 in tolerated_nodes) and (node2 in tolerated_nodes): # dont add useless edges
                continue
            if node1 in subgraph.nodes and node2 in subgraph.nodes:
                for k in self.get_edge_ks(node1, node2):
                    if add_same_block_edges or (not self.edges[edge][k]['same_block']):
                        subgraph.add_edge(node1, node2, self.edges[edge][k]['certainty'], k, self.edges[edge][k]['same_block'], bad_edge=self.edges[edge][k]['bad_edge'])
        subgraph.compute_node_edges()
        return subgraph
    
    def graph_selected_nodes(self, nodes, ks, other_block_edges_only=False):
        print(f"Graphing {len(nodes)} nodes...")
        nodes = [tuple([int(n) for n in node]) for node in nodes]
        nodes_set = set([tuple([int(n) for n in node]) for node in nodes])
        # Extract the subgraph only containing the selected nodes and connections between them
        subgraph = ScrollGraph(self.overlapp_threshold, self.umbilicus_path)
        # Add nodes
        for node in nodes_set:
            subgraph.add_node(node, self.nodes[node]['centroid'], winding_angle=self.nodes[node]['winding_angle'], sample_points=self.nodes[node]['sample_points'] if 'sample_points' in self.nodes[node] else None)
        # add ks
        subgraph.update_winding_angles(nodes, ks, update_winding_angles=False)
        
        for edge in self.edges:
            node1, node2 = edge
            node1 = tuple([int(n) for n in node1])
            node2 = tuple([int(n) for n in node2])
            if node1 in nodes_set and node2 in nodes_set:
                if node1[:3] == node2[:3]:
                    continue
                # Check if there is a edge with the right k connecting the nodes
                k1 = subgraph.nodes[node1]['assigned_k']
                k2 = subgraph.nodes[node2]['assigned_k']
                k_search = k2 - k1
                ks_edge = self.get_edge_ks(node1, node2)
                # print(f"Edge: {node1} -> {node2}, k1: {k1}, k2: {k2}, searching k: {k_search} in ks: {ks_edge}")
                k_in_ks = False
                for k in ks_edge:
                    if k == k_search:
                        k_in_ks = True
                        break
                if not k_in_ks:
                    continue
                if self.edges[edge][k_search]['bad_edge']:
                    continue
                if other_block_edges_only and self.edges[edge][k_search]['same_block']:
                    continue
                subgraph.add_edge(node1, node2, self.edges[edge][k_search]['certainty'], k_search, self.edges[edge][k_search]['same_block'], bad_edge=self.edges[edge][k_search]['bad_edge'])
        
        subgraph.compute_node_edges()
        return subgraph
    
def write_graph_to_binary(file_name, graph):
    """
    Writes the graph to a binary file.
    
    Parameters:
    - file_name (str): The name of the file to write to.
    - graph (dict): A dictionary where the keys are node IDs and the values are lists of tuples.
                    Each tuple contains (target_node_id, w, k).
    """
    print(f"Writing graph to binary file: {file_name}")
    nodes = graph.nodes
    edges = graph.edges

    nodes_list = list(nodes.keys())
    node_index_dict = {nodes_list[i]: i for i in range(len(nodes_list))}

    nr_edges = 0
    # create adjacency list
    adj_list = {}
    for edge in tqdm(edges, desc="Building adjacency list"):
        node1, node2 = edge
        node1_index = node_index_dict[node1]
        node2_index = node_index_dict[node2]
        if node1_index not in adj_list:
            adj_list[node1_index] = []
        if node2_index not in adj_list:
            adj_list[node2_index] = []
        for k in edges[edge]:
            # no bad edges
            if edges[edge][k]['bad_edge']:
                continue
            adj_list[node1_index].append((node2_index, edges[edge][k]['certainty'], k, edges[edge][k]['same_block']))
            adj_list[node2_index].append((node1_index, edges[edge][k]['certainty'], -k, edges[edge][k]['same_block']))
            nr_edges += 1

    # save the nodes_list

    print(f"Number of nodes: {len(nodes_list)}, number of edges: {nr_edges}")

    # Write the graph to a binary file
    with open(file_name, 'wb') as f:
        # Write the number of nodes
        f.write(struct.pack('I', len(nodes)))
        for node in tqdm(nodes_list, desc="Writing nodes"):
            # write the node z positioin as f
            f.write(struct.pack('f', float(nodes[node]['centroid'][1])))
            # write the node winding angle as f
            f.write(struct.pack('f', float(nodes[node]['winding_angle'])))
            # write gt flag as bool
            f.write(struct.pack('?', bool('winding_angle_gt' in nodes[node])))
            # write gt winding angle as f
            if 'winding_angle_gt' in nodes[node]:
                f.write(struct.pack('f', float(nodes[node]['winding_angle_gt'])))
            else:
                f.write(struct.pack('f', float(0.0)))

        for node in tqdm(adj_list, desc="Writing edges"):
            # Write the node ID
            f.write(struct.pack('I', node))

            # Write the number of edges
            f.write(struct.pack('I', len(adj_list[node])))

            for edge in adj_list[node]:
                target_node, w, k, same_block = edge
                # Write the target node ID
                f.write(struct.pack('I', target_node))
                # Write the weight w (float)
                f.write(struct.pack('f', w))
                # Write the weight k (float)
                f.write(struct.pack('f', k))
                # Same block
                f.write(struct.pack('?', same_block))

    print(f"Graph written to binary file: {file_name}")

def load_graph_winding_angle_from_binary(filename, graph):
    nodes = {}

    with open(filename, 'rb') as f:
        # Read the number of nodes
        num_nodes = struct.unpack('I', f.read(4))[0]

        for i in range(num_nodes):
            # Read f_star (float) and deleted flag (bool)
            f_star = struct.unpack('f', f.read(4))[0]
            deleted = struct.unpack('?', f.read(1))[0]

            # Store in a dictionary with the index as the key
            nodes[i] = {'f_star': f_star, 'deleted': deleted}

    nodes_graph = graph.nodes
    nodes_list = list(nodes_graph.keys())
    # assign winding angles
    for i in range(num_nodes):
        if nodes[i]['deleted']:
            # try detelting assigned k
            if 'assigned_k' in nodes_graph[nodes_list[i]]:
                del nodes_graph[nodes_list[i]]['assigned_k']
            continue
        node = nodes_list[i]
        nodes_graph[node]['assigned_k'] = (nodes_graph[node]['winding_angle'] - nodes[i]['f_star']) // 360
        nodes_graph[node]['winding_angle'] = nodes[i]['f_star']
    # delete deleted nodes
    for i in range(num_nodes):
        node = nodes_list[i]
        if nodes[i]['deleted']:
            del nodes_graph[node]

    print(f"Number of nodes remaining: {len(nodes_graph)} from {num_nodes}. Number of nodes in graph: {len(graph.nodes)}")
    return graph

def compute(overlapp_threshold, start_point, path, recompute=False, stop_event=None, toy_problem=False, update_graph=False, flip_winding_direction=False, gt_mesh_file=None, continue_from=0, update_edges=False):

    umbilicus_path = os.path.dirname(path) + "/umbilicus.txt"
    start_block, patch_id = (0, 0, 0), 0

    save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Configs: {overlapp_threshold}")

    recompute_path = path.replace("blocks", "scroll_graph_angular") + ".pkl"
    recompute = recompute or not os.path.exists(recompute_path)
    
    # Build graph
    if recompute:
        load_graph_fresh = continue_from <= -2
        start_fresh = continue_from <= -1
        if load_graph_fresh:
            scroll_graph = ScrollGraph(overlapp_threshold, umbilicus_path)
        else:
            scroll_graph = load_graph(recompute_path)
        count_gt_added = 0
        for node in scroll_graph.nodes:
            if 'winding_angle_gt' in scroll_graph.nodes[node]:
                count_gt_added += 1
        print(f"Number of nodes with GT winding angle: {count_gt_added}")
        num_processes = max(1, cpu_count() - 2)
        start_block, patch_id = scroll_graph.build_graph(path, num_processes=num_processes, start_point=start_point, prune_unconnected=False, start_fresh=start_fresh, gt_mesh_file=gt_mesh_file, continue_from=continue_from, update_edges=update_edges)
        print("Saving built graph...")
        scroll_graph.save_graph(recompute_path)
    
    # Graph generation area. CREATE subgraph or LOAD graph
    if update_graph:
        scroll_graph = load_graph(recompute_path)
        scroll_graph.set_overlapp_threshold(overlapp_threshold)
        scroll_graph.start_block, scroll_graph.patch_id = start_block, patch_id
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 575, 775, 625, 825, 700, 900, None # 2x2x2 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 475, 875, 525, 925, 700, 900, None # 4x4x2 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 475, 875, 525, 925, 600, 1000, None # 4x4x4 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = None, None, None, None, 700, 900, None # all x all x 2 blocks with middle
        min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = None, None, None, None, 700, 1300, None # all x all x 6 blocks with middle
        subgraph = scroll_graph.extract_subgraph(min_z=min_z, max_z=max_z, umbilicus_max_distance=umbilicus_max_distance, add_same_block_edges=True, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        subgraph.save_graph(save_path.replace("blocks", "subgraph_angular") + ".pkl")
        if toy_problem:
            scroll_graph = subgraph
    else:
        if toy_problem:
            scroll_graph = load_graph(save_path.replace("blocks", "subgraph_angular") + ".pkl")
        else:
            scroll_graph = load_graph(recompute_path)

    if (not toy_problem) and flip_winding_direction:
        print("Flipping winding direction ...")
        scroll_graph.flip_winding_direction()
        scroll_graph.save_graph(recompute_path)
        print("Done flipping winding direction.")
    elif flip_winding_direction:
        raise ValueError("Cannot flip winding direction for toy problem.")

    scroll_graph.set_overlapp_threshold(overlapp_threshold)
    scroll_graph.start_block, scroll_graph.patch_id = start_block, patch_id

    # min max centroid[1]
    min_z = min([scroll_graph.nodes[node]["centroid"][1] for node in scroll_graph.nodes])
    max_z = max([scroll_graph.nodes[node]["centroid"][1] for node in scroll_graph.nodes])
    print(f"Min z: {min_z}, Max z: {max_z}")

    print(f"Number of nodes in the graph: {len(scroll_graph.nodes)}")

    # create binary graph file
    write_graph_to_binary(os.path.join(os.path.dirname(save_path), "graph.bin"), scroll_graph)

def random_walks():
    path = "/media/julian/SSD4TB/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks"
    # sample_ratio_score = 0.03 # 0.1
    start_point=[1650, 3300, 5000] # seg 1
    start_point=[1450, 3500, 5000] # seg 2
    start_point=[1350, 3600, 5000] # seg 3
    start_point=[1352, 3600, 5002] # unused / pyramid random walk indicator
    continue_segmentation = 0
    overlapp_threshold = {"sample_ratio_score": 0.1, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.0005, "score_threshold": 0.005, "fit_sheet": False, "cost_threshold": 17, "cost_percentile": 75, "cost_percentile_threshold": 14, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.005,
                          "cost_threshold_prediction": 2.5, "min_prediction_threshold": 0.15, "nr_points_min": 100.0, "nr_points_max": 2000.0, "min_patch_points": 140.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 1000, "min_winding_switch_sheet_distance": 4, "max_winding_switch_sheet_distance": 7, "winding_switch_sheet_score_factor": 1.5, "winding_direction": 1.0,
                          "enable_winding_switch": True, "max_same_block_jump_range": 3,
                          "pyramid_up_nr_average": 10000, "nr_walks_per_node":5000,
                          "enable_winding_switch_postprocessing": False,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1000, 1000), "volume_min_certainty_total_percentage": 0.0, "max_umbilicus_difference": 30,
                          "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1,
                          "bad_edge_threshold": 3
                          }
    # Scroll 1: "winding_direction": -1.0
    # Scroll 2: "winding_direction": 1.0
    # Scroll 3: "winding_direction": 1.0

    max_nr_walks = 10000
    max_steps = 101
    min_steps = 16
    min_end_steps = 16
    max_tries = 6
    max_unchanged_walks = 100 * max_nr_walks # 30 * max_nr_walks
    recompute = 1
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Cut out ThaumatoAnakalyptor Papyrus Sheet. TAKE CARE TO SET THE "winding_direction" CORRECTLY!')
    parser.add_argument('--path', type=str, help='Papyrus instance patch path (containing .tar)', default=path)
    parser.add_argument('--recompute', type=int,help='Recompute graph', default=recompute)
    parser.add_argument('--print_scores', type=bool,help='Print scores of patches for sheet', default=overlapp_threshold["print_scores"])
    parser.add_argument('--sample_ratio_score', type=float,help='Sample ratio to apply to the pointcloud patches', default=overlapp_threshold["sample_ratio_score"])
    parser.add_argument('--score_threshold', type=float,help='Score threshold to add patches to sheet', default=overlapp_threshold["score_threshold"])
    parser.add_argument('--min_prediction_threshold', type=float,help='Min prediction threshold to add patches to sheet', default=overlapp_threshold["min_prediction_threshold"])
    parser.add_argument('--final_score_min', type=float,help='Final score min threshold to add patches to sheet', default=overlapp_threshold["final_score_min"])
    parser.add_argument('--rounddown_best_score', type=float,help='Pick best score threshold to round down to zero from. Combats segmentation speed slowdown towards the end of segmentation.', default=overlapp_threshold["rounddown_best_score"])
    parser.add_argument('--winding_direction', type=int,help='Winding direction of sheet in scroll scan. Examples: SCroll 1: "-1", Scroll 3: "1"', default=overlapp_threshold["winding_direction"])
    parser.add_argument('--sheet_z_range', type=int, nargs=2,help='Z range of segmentation', default=[overlapp_threshold["sheet_z_range"][0], overlapp_threshold["sheet_z_range"][1]])
    parser.add_argument('--sheet_k_range', type=int, nargs=2,help='Angle range (as k 1k = 360 deg, k is int) of the sheet winding for segmentation', default=[overlapp_threshold["sheet_k_range"][0], overlapp_threshold["sheet_k_range"][1]])
    parser.add_argument('--starting_point', type=int, nargs=3,help='Starting point for a new segmentation', default=start_point)
    parser.add_argument('--continue_segmentation', type=int,help='Continue previous segmentation (point_cloud_colorized_subvolume_main_sheet.ta). 1 to continue, 0 to restart.', default=int(continue_segmentation))
    parser.add_argument('--enable_winding_switch', type=int,help='Enable switching of winding if two sheets lay on top of each eather. 1 enable, 0 disable.', default=int(overlapp_threshold["enable_winding_switch"]))
    parser.add_argument('--enable_winding_switch_postprocessing', type=int,help='Enable postprocessing switching of winding if two sheets lay on top of each eather. 1 enable, 0 disable.', default=int(overlapp_threshold["enable_winding_switch_postprocessing"]))
    parser.add_argument('--max_threads', type=int,help='Maximum number of thread to use during computation. Has a slight influence on the quality of segmentations for small segments.', default=int(overlapp_threshold["max_threads"]))
    parser.add_argument('--surrounding_patches_size', type=int,help=f'Number of surrounding half-overlapping patches in each dimension direction to consider when calculating patch pair similarity scores. Default is {overlapp_threshold["surrounding_patches_size"]}.', default=int(overlapp_threshold["surrounding_patches_size"]))
    parser.add_argument('--max_nr_walks', type=int,help=f'Maximum number of random walks to perform. Default is {max_nr_walks}.', default=int(max_nr_walks))
    parser.add_argument('--min_steps', type=int,help=f'Minimum number of steps for a random walk to be considered valid. Default is {min_steps}.', default=int(min_steps))
    parser.add_argument('--min_end_steps', type=int,help=f'Minimum number of steps for a random walk to be considered valid at the end of a random walk. Default is {min_end_steps}.', default=int(min_end_steps))
    parser.add_argument('--max_unchanged_walks', type=int,help=f'Maximum number of random walks to perform without updating the graph before finishing the segmentation. Default is {max_unchanged_walks}.', default=int(max_unchanged_walks))
    parser.add_argument('--min_certainty_p', type=float,help=f'Minimum percentage of certainty of a volume to be considered for random walks. Default is {overlapp_threshold["volume_min_certainty_total_percentage"]}.', default=overlapp_threshold["volume_min_certainty_total_percentage"])
    parser.add_argument('--max_umbilicus_dif', type=float,help=f'Maximum difference in umbilicus distance between two patches to be considered valid. Default is {overlapp_threshold["max_umbilicus_difference"]}.', default=overlapp_threshold["max_umbilicus_difference"])
    parser.add_argument('--walk_aggregation_threshold', type=int,help=f'Number of random walks to aggregate before updating the graph. Default is {overlapp_threshold["walk_aggregation_threshold"]}.', default=int(overlapp_threshold["walk_aggregation_threshold"]))
    parser.add_argument('--walk_aggregation_max_current', type=int,help=f'Maximum number of random walks to aggregate before updating the graph. Default is {overlapp_threshold["walk_aggregation_max_current"]}.', default=int(overlapp_threshold["walk_aggregation_max_current"]))
    parser.add_argument('--pyramid_up_nr_average', type=int,help=f'Number of random walks to aggregate per landmark before walking up the graph. Default is {overlapp_threshold["pyramid_up_nr_average"]}.', default=int(overlapp_threshold["pyramid_up_nr_average"]))
    parser.add_argument('--toy_problem', help='Create toy subgraph for development', action='store_true')
    parser.add_argument('--update_graph', help='Update graph', action='store_true')
    parser.add_argument('--create_graph', help='Create graph. Directly creates the binary .bin graph file from a previously constructed graph .pkl', action='store_true')
    parser.add_argument('--flip_winding_direction', help='Flip winding direction', action='store_true')
    parser.add_argument('--gt_mesh_file', type=str, help='Ground truth mesh file', default=None)
    parser.add_argument('--continue_from', type=int, help='Continue from a certain point in the graph', default=-2)
    parser.add_argument('--update_edges', help='Update edges', action='store_true')
    parser.add_argument('--centroid_method', type=str, help='Method to calculate patch centroid', default="mean") # tested, MUCH better than geometric_median

    # Take arguments back over
    args = parser.parse_args()
    print(f"Args: {args}")

    path = args.path
    recompute = bool(int(args.recompute))
    overlapp_threshold["print_scores"] = args.print_scores
    overlapp_threshold["sample_ratio_score"] = args.sample_ratio_score
    overlapp_threshold["score_threshold"] = args.score_threshold
    overlapp_threshold["min_prediction_threshold"] = args.min_prediction_threshold
    overlapp_threshold["final_score_min"] = args.final_score_min
    overlapp_threshold["rounddown_best_score"] = args.rounddown_best_score
    overlapp_threshold["winding_direction"] = args.winding_direction
    overlapp_threshold["sheet_z_range"] = [(z_range_ + 500) /(200.0 / 50.0) for z_range_ in args.sheet_z_range]
    overlapp_threshold["sheet_k_range"] = args.sheet_k_range
    start_point = args.starting_point
    continue_segmentation = bool(args.continue_segmentation)
    overlapp_threshold["enable_winding_switch"] = bool(args.enable_winding_switch)
    overlapp_threshold["enable_winding_switch_postprocessing"] = bool(args.enable_winding_switch_postprocessing)
    overlapp_threshold["max_threads"] = args.max_threads
    overlapp_threshold["surrounding_patches_size"] = args.surrounding_patches_size
    min_steps = args.min_steps
    max_nr_walks = args.max_nr_walks
    min_end_steps = args.min_end_steps
    overlapp_threshold["volume_min_certainty_total_percentage"] = args.min_certainty_p
    overlapp_threshold["max_umbilicus_difference"] = args.max_umbilicus_dif
    overlapp_threshold["walk_aggregation_threshold"] = args.walk_aggregation_threshold
    overlapp_threshold["walk_aggregation_max_current"] = args.walk_aggregation_max_current
    overlapp_threshold["max_nr_walks"] = max_nr_walks
    overlapp_threshold["max_unchanged_walks"] = max_unchanged_walks
    overlapp_threshold["continue_walks"] = continue_segmentation
    overlapp_threshold["max_steps"] = max_steps
    overlapp_threshold["max_tries"] = max_tries
    overlapp_threshold["min_steps"] = min_steps
    overlapp_threshold["min_end_steps"] = min_end_steps
    overlapp_threshold["pyramid_up_nr_average"] = args.pyramid_up_nr_average

    # set centroid method
    global centroid_method
    if args.centroid_method == "geo_mean":
        centroid_method = centroid_method_geometric_mean
    elif args.centroid_method == "mean":
        centroid_method = centroid_method_mean
    elif args.centroid_method == "geo_median":
        centroid_method = closest_to_geometric_median
    else:
        raise ValueError(f"Centroid method {args.centroid_method} not supported.")

    if args.create_graph:
        save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
        if args.toy_problem:
            scroll_graph = load_graph(save_path.replace("blocks", "subgraph_angular") + ".pkl")
        else:
            scroll_graph = load_graph(path.replace("blocks", "scroll_graph_angular") + ".pkl")
        
        scroll_graph_solved = load_graph_winding_angle_from_binary(os.path.join(os.path.dirname(save_path), "output_graph.bin"), scroll_graph)

        # save graph pickle
        scroll_graph_solved.save_graph(save_path.replace("blocks", "graph_BP_solved") + ".pkl")
    else:
        # Compute
        compute(overlapp_threshold=overlapp_threshold, start_point=start_point, path=path, recompute=recompute, toy_problem=args.toy_problem, update_graph=args.update_graph, flip_winding_direction=args.flip_winding_direction, gt_mesh_file=args.gt_mesh_file, continue_from=args.continue_from, update_edges=args.update_edges)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') # need sppawn because of open3d initialization deadlock in the init worker function
    random_walks()

# Example command: python3 -m ThaumatoAnakalyptor.instances_to_graph --path /scroll.volpkg/working/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks --recompute 1