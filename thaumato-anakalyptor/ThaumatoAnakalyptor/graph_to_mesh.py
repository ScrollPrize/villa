import numpy as np
import os
import shutil
import open3d as o3d
from tqdm import tqdm
import time
import threading
import multiprocessing
import pickle
from copy import deepcopy
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Custom imports
from .instances_to_graph import load_graph, ScrollGraph
from .slim_uv import Flatboi, print_array_to_file
from .split_mesh import MeshSplitter

# import colormap from matplotlib
import matplotlib

import sys
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import pointcloud_processing

from PIL import Image
# This disables the decompression bomb protection in Pillow
Image.MAX_IMAGE_PIXELS = None

angle_vector_indices_dp = {}

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def scale_points(points, scale=4.0, axis_offset=-500):
    """
    Scale points
    """
    # Scale points
    points_ = points * scale
    # Offset points
    points_ = points_ + axis_offset # WTF SPELUFO? ;)
    # Return points
    return points_

def shuffling_points_axis(points, normals, axis_indices=[2, 0, 1]):
    """
    Rotate points by reshuffling axis
    """
    # Reshuffle axis in points and normals
    points = points[:, axis_indices]
    normals = normals[:, axis_indices]
    # Return points and normals
    return points, normals

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2=np.array([1, 0])):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def flatten_args(args):
    save_path, mesh_path, downsample = args
    print(f"Flattening {mesh_path}")
    return flatten_subprocess(mesh_path, downsample)

def flatten(save_path, mesh_path):
    mesh_output_path = mesh_path.replace(".obj", "_flatboi.obj")
    flatboi = Flatboi(mesh_path, 5, output_obj=mesh_output_path)
    fresh_start = True
    harmonic_uvs_path = os.path.join(save_path, os.path.basename(mesh_path).replace(".obj", "harmonic_uvs.pkl"))
    if fresh_start:
        # harmonic_uvs, harmonic_energies = flatboi.slim(initial_condition='harmonic')
        harmonic_uvs, harmonic_energies = flatboi.slim(initial_condition='ordered')
        # Get the directory of the input file
        input_directory = os.path.dirname(mesh_output_path)
        # Filename for the energies file
        energies_file = os.path.join(input_directory, 'energies_flatboi.txt')
        print_array_to_file(harmonic_energies, energies_file)       

        # save harmonic_uvs as pkl
        with open(harmonic_uvs_path, 'wb') as f:
            pickle.dump(harmonic_uvs, f)
    else:
        with open(harmonic_uvs_path, 'rb') as f:
            harmonic_uvs = pickle.load(f)

    # Save Flattened mesh
    flatboi.save_img(harmonic_uvs)
    flatboi.save_obj(harmonic_uvs)
    flatboi.save_mtl()
    print(f"Saved flattened mesh to {mesh_output_path}")

def flatten_subprocess(mesh_path, downsample):
    # Call mesh_to_surface as a separate process
    command = [
                "python3", "-m", "ThaumatoAnakalyptor.slim_uv", 
                "--path", mesh_path, 
                "--iter", str(5), 
                "--ic", "ordered"
            ]
    if downsample:
        command += ["--downsample"]
    # Running the command
    flatteing = subprocess.Popen(command)
    flatteing.wait()
    return None

def compute_means_adjacent_args(args):
    return compute_means_adjacent(*args)

def compute_means_adjacent(adjacent_ts, adjacent_normals, winding_direction):
    def calculate_means(ts_lists):
        res = []
        for ts in ts_lists:
            res_ = [np.mean(t) if len(t) > 0 else None for t in ts]
            res.append(res_)
        return res
    # Copy to preserve original lists
    original_ts = deepcopy(adjacent_ts)

    # Create dictionaries to map each t to its corresponding normal
    t_normals_dict_list = []
    for ts, normals in zip(adjacent_ts, adjacent_normals):
        dict_list = []
        for t, normal in zip(ts, normals):
            dict_ts = {}
            for t_, normal_ in zip(t, normal):
                dict_ts[t_] = normal_
            dict_list.append(dict_ts)
        t_normals_dict_list.append(dict_list)
    
    def print_none_vs_means(t_means):
        count_fixed = 0
        count_total = 0
        count_original = 0
        for i in range(len(t_means)):
            for u in range(len(t_means[i])):
                count_total += 1
                if t_means[i][u] is not None:
                    count_fixed += 1
                if len(original_ts[i][u]) > 0:
                    count_original += 1

        print(f"T means: Fixed {count_fixed}/{count_original} original out of total {count_total} t values spaces.")

    # Calculate initial means of each list, handle empty lists by setting means to None
    t_means = calculate_means(adjacent_ts)
    # debug
    normals_means = []
    for i, ts in enumerate(adjacent_ts):
        normals_means.append([])
        for e, t in enumerate(ts):
            filtered_normals = [t_normals_dict_list[i][e][t_] for t_ in t]
            normals_means[i].append(np.mean(filtered_normals, axis=0) if len(filtered_normals) > 0 else None)

    # Function to refine means based on adjacent means
    def refine_means(t_means, fixed, fixed_adjacent_ts):
        for u in range(len(original_ts)) if random.random() < 0.5 else range(len(original_ts)-1, -1, -1): # randomly choose direction in which to iterate and refine
            for i in range(len(original_ts[u])):
                # Determine the valid next mean
                next_mean = next((t_means[j][i] for j in range(u-1, -1, -1) if t_means[j][i] is not None), None)
                # Determine the valid previous mean
                prev_mean = next((t_means[j][i] for j in range(u+1, len(t_means)) if t_means[j][i] is not None), None)
                # Filter t values based on previous and next means
                if not winding_direction:
                    prev_mean, next_mean = next_mean, prev_mean
                adjacent_ts[u][i] = [t for t in original_ts[u][i] if (prev_mean is None or t > prev_mean) and (next_mean is None or t < next_mean)] if ((prev_mean is not None) or (random.random() < 0.75)) and ((next_mean is not None) or (random.random() < 0.75)) else []
                if len(adjacent_ts[u][i]) == 0: # ensure that the t value is fixed if no valid t values are found
                    if fixed[u] == 1:
                        adjacent_ts[u][i] = fixed_adjacent_ts[u][i]

        # Recalculate means after filtering
        return calculate_means(adjacent_ts)

    def optimization_step(t_means, fixed, fixed_adjacent_ts):
        # Iteratively refine means until they are in the correct order
        max_iterations = 100  # Limit iterations to prevent infinite loop
        for _ in range(max_iterations):
            previous_means = t_means[:]
            t_means = refine_means(t_means, fixed, fixed_adjacent_ts)
            # Randomly set some t values to None #
            for u in range(len(t_means)):
                for i in range(len(t_means[u])):
                    if random.random() < 0.1:
                        t_means[u][i] = None

        # Iteratively refine means until they are in the correct order
        max_iterations = 10  # Limit iterations to prevent infinite loop
        for _ in range(max_iterations):
            previous_means = t_means[:]
            t_means = refine_means(t_means, fixed, fixed_adjacent_ts)
            if t_means == previous_means:  # Stop if no change
                break

        normals_means = []
        for i, ts in enumerate(adjacent_ts):
            normals_means.append([])
            for e, t in enumerate(ts):
                filtered_normals = [t_normals_dict_list[i][e][t_] for t_ in t]
                normals_means[i].append(np.mean(filtered_normals, axis=0) if len(filtered_normals) > 0 else None)
        return t_means, normals_means

    def fix_windings(t_means, fixed):
        column_length = len(t_means[0])
        nr_selected = []
        # Check for all good t values
        for i in range(len(t_means)):
            n_sel = sum([1 for u in range(column_length) if t_means[i][u] is not None])
            nr_selected.append(n_sel)
        
        # sort descending and get indices of top % of selected t values. 
        sorted_indices = np.argsort(nr_selected)[::-1]

        # Fix top % of nr of selected t values
        top_percentage = 0.35
        nr_fixing = int(len(t_means) * top_percentage)
        current_fix_count = 0
        for i in range(len(t_means)):
            sorted_index = sorted_indices[i]
            if fixed[sorted_index] == 0:
                fixed[sorted_index] = 1
                current_fix_count += 1
            if nr_fixing <= current_fix_count:
                break

        # # Alternative fix top % of selected t values
        # top_percentage = 0.25
        # for i in range(len(t_means)):
        #     n_sel = nr_selected[i]
        #     sel_p = n_sel / column_length
        #     if sel_p > top_percentage and n_sel > 0:
        #         fixed[i] = 1

        return fixed, deepcopy(adjacent_ts)

    # Do optimization step
    fixed = np.zeros(len(t_means))
    fixed_adjacent_ts = deepcopy(adjacent_ts)
    for step in range(3):
        t_means, normals_means = optimization_step(t_means, fixed, fixed_adjacent_ts)
        fixed, fixed_adjacent_ts = fix_windings(t_means, fixed)

    # Check for all good t values
    z_len = len(t_means[0])
    count_fixed = 0
    count_total = 0
    count_wrong = 0
    for u in range(z_len):
        last_t = None
        for i in range(len(t_means)):
            count_total += 1
            if t_means[i][u] is not None:
                if last_t is not None:
                    if winding_direction and t_means[i][u] >= last_t:
                        # print(f"Something is wrong with t values: {t_means[i][u]} < {last_t}")
                        count_wrong += 1
                        t_means[i][u] = None
                        continue
                    elif not winding_direction and t_means[i][u] <= last_t:
                        # print(f"Something is wrong with t values: {t_means[i][u]} > {last_t}")
                        count_wrong += 1
                        t_means[i][u] = None
                        continue
                count_fixed += 1
                last_t = t_means[i][u]

    return t_means, normals_means 

class WalkToSheet():
    def __init__(self, graph, path, start_point=[3164, 3476, 3472], scale_factor=1.0, split_width=50000):
        self.scale_factor = scale_factor
        self.split_width = split_width
        self.graph = graph
        self.path = path
        self.save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
        self.lock = threading.Lock()

    def find_minmax_winding_angle_z_range(self):
        min_winding_angle = float('inf')
        max_winding_angle = -float('inf')
        z_min = float('inf')
        z_max = -float('inf')
        for node in self.graph.nodes:
            if 'assigned_k' not in self.graph.nodes[node]:
                continue
            winding_angle = self.graph.nodes[node]['winding_angle']
            min_winding_angle = min(min_winding_angle, winding_angle)
            max_winding_angle = max(max_winding_angle, winding_angle)
            z = node[1]
            z_min = min(z_min, z)
            z_max = max(z_max, z)
        return min_winding_angle, max_winding_angle, z_min, z_max

    def build_points(self, z_range=None):
        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []

        sheet_infos = []
        for node in tqdm(self.graph.nodes, desc="Building points"):
            if 'assigned_k' not in self.graph.nodes[node]:
                continue
            winding_angle = self.graph.nodes[node]['winding_angle']
            block, patch_id = node[:3], node[3]
            if block[1] < z_range[0] - 75 or block[1] > z_range[1] + 75:
                continue
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            sheet_infos.append(patch_sheet_patch_info)
        time_start = time.time()
        # print(sheet_infos)
        points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path, z_range[0], z_range[1], True)
        print(f"Time to load pointclouds: {time.time() - time_start}")
        print(f"Shape of patch_points: {np.array(points).shape}")

        return points, normals, colors
    
    def build_points_reference(self, graph_path):
        graph = load_graph(graph_path)

        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []

        sheet_infos = []
        for node in tqdm(graph.nodes, desc="Building points"):
            winding_angle = 0.0
            block, patch_id = node[:3], node[3]
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            sheet_infos.append(patch_sheet_patch_info)

        time_start = time.time()
        points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path)
        print(f"Time to load pointclouds: {time.time() - time_start}")
        print(f"Shape of patch_points: {np.array(points).shape}")

        # print first 5 points
        for i in range(5):
            print(f"Point {i}: {points[i]}")
            print(f"Normal {i}: {normals[i]}")
            print(f"Color {i}: {colors[i]}")

        return points, normals, colors

    def points_at_winding_angle(self, points, winding_angle, max_angle_diff=30):
        """
        Assumes points are sorted by winding angle ascendingly
        """
        # extract all points indices with winding angle within max_angle_diff of winding_angle
        start_index = np.searchsorted(points[:, 3], winding_angle - max_angle_diff, side='left')
        end_index = np.searchsorted(points[:, 3], winding_angle + max_angle_diff, side='right')
        return start_index, end_index
    
    def extract_all_same_vector(self, angle_vector, vector):
        # Check if the angle_vector is already in the dictionary
        if (tuple(vector) in angle_vector_indices_dp):
            return angle_vector_indices_dp[tuple(vector)]
        
        indices = []
        i = 0
        first_found = None
        index_dif = None
        while i < len(angle_vector):
            if np.allclose(angle_vector[i], vector):
                if first_found is None:
                    first_found = i
                elif index_dif is None:
                    index_dif = i - first_found
                indices.append(i)
            else:
                assert index_dif is None, f"Index difference is not consistent!"
            if index_dif is None:
                i += 1
            else:
                i += index_dif                  

        # Add to dictionary
        angle_vector_indices_dp[tuple(vector)] = indices
        return indices
    
    # Calculate initial means of each list, handle empty lists by setting means to None
    def calculate_means(self, ts_lists):
        res = []
        for ts in ts_lists:
            res_ = [np.mean(t) if len(t) > 0 else None for t in ts]
            res.append(res_)
        return res    
    
    def find_inner_outermost_winding_direction(self, t_means, angle_vector):
        # Split t_means into outermost and innermost half
        t_means_outermost_half = t_means[:len(t_means)//2]
        t_means_innermost_half = t_means[len(t_means)//2:]

        # Compute mean ts value of furthest out t values
        mean_outermost_half = 0.0
        count_outermost = 0
        for i in range(len(t_means_outermost_half)):
            for u in range(len(t_means_outermost_half[i])):
                if t_means_outermost_half[i][u] is not None:
                    mean_outermost_half += t_means_outermost_half[i][u]
                    count_outermost += 1
        mean_outermost_half /= count_outermost

        # Compute mean ts value of innermost t values
        mean_innermost_half = 0.0
        count_innermost = 0
        for i in range(len(t_means_innermost_half)):
            for u in range(len(t_means_innermost_half[i])):
                if t_means_innermost_half[i][u] is not None:
                    mean_innermost_half += t_means_innermost_half[i][u]
                    count_innermost += 1
        mean_innermost_half /= count_innermost

        winding_direction = True
        if mean_innermost_half > mean_outermost_half:
            winding_direction = False
            print("Winding direction is normal.")
        else:
            print("Winding direction is reversed.")

        # compute mean ts value of furthest out t values
        count = 0
        mean_innermost_ts = 0.0
        mean_outermost_ts = 0.0

        computed_vectors_set = set()
        for i in tqdm(range(len(t_means))):
            curve_angle_vector = angle_vector[i]
            if not tuple(curve_angle_vector) in computed_vectors_set:
                computed_vectors_set.add(tuple(curve_angle_vector))
            else:
                continue
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)

            ts_angle = []
            for j in same_vector_indices:
                ts_angle += [t for t in t_means[j] if t is not None]
            if len(ts_angle) == 0:
                continue

            count += 1
            mean_innermost_ts += np.min(ts_angle)
            mean_outermost_ts += np.max(ts_angle)
        
        # get mean values
        if count == 0:
            print("[ERROR (find_inner_outermost_winding_direction)]: No t values found.")
            count += 1
            mean_outermost_ts = 1
        mean_innermost_ts /= count
        mean_outermost_ts /= count

        print(f"Mean innermost: {mean_innermost_ts}, mean outermost: {mean_outermost_ts}")
        return mean_innermost_ts, mean_outermost_ts, winding_direction

    def initial_full_pointset(self, t_means, normals_means, angle_vector, mean_innermost_ts, mean_outermost_ts, winding_direction):
        mean_dist = mean_innermost_ts - mean_outermost_ts
        
        def adjacent_means(position, t_mean):
            # Determine the valid next mean
            next_mean = None
            next_ind = None
            for i in range(position-1, -1, -1):
                if t_mean[i] is not None:
                    next_mean = t_mean[i]
                    next_ind = i
                    break
            # Determine the valid previous mean
            prev_mean = None
            prev_ind = None
            for i in range(position+1, len(t_mean)):
                if t_mean[i] is not None:
                    prev_mean = t_mean[i]
                    prev_ind = i
                    break

            # Reverse if winding direction is reversed
            if not winding_direction:
                prev_mean, next_mean = next_mean, prev_mean
                prev_ind, next_ind = next_ind, prev_ind

            length_ts = len(t_mean)
            # use outer_mean and inner_mean if prev and next are both none
            if prev_mean is None and next_mean is None:
                prev_mean = mean_innermost_ts
                prev_ind = length_ts - 1 if winding_direction else 0
                next_mean = mean_outermost_ts
                next_ind = 0 if winding_direction else length_ts - 1
            # Filter based on previous mean
            elif prev_mean is None:
                prev_ind = length_ts - 1 if winding_direction else 0
                prev_mean = next_mean + mean_dist * abs(next_ind - prev_ind) / length_ts
                if prev_mean > 0.0:
                    prev_mean = 0.0
            # Filter based on next mean
            elif next_mean is None:
                next_ind = 0 if winding_direction else length_ts - 1
                next_mean = prev_mean - mean_dist * abs(next_ind - prev_ind) / length_ts
                if next_mean > 0.0:
                    next_mean = 0.0

            if next_mean == prev_mean: # if next and prev mean are the same, space them apart
                next_mean = prev_mean + mean_dist
            
            return prev_mean, prev_ind, next_mean, next_ind

        computed_vectors_set = set()
        interpolated_ts = []
        interpolated_normals = []
        fixed_points = []
        for i in range(len(t_means)):
            interpolated_ts.append([None]*len(t_means[i]))
            interpolated_normals.append([None]*len(t_means[i]))
            fixed_points.append([False]*len(t_means[i]))
        
        for i in range(len(t_means)):
            curve_angle_vector = angle_vector[i]
            if not tuple(curve_angle_vector) in computed_vectors_set:
                computed_vectors_set.add(tuple(curve_angle_vector))
            else:
                continue
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            normal = unit_vector(curve_angle_vector)
            same_vector_ts = [t_means[j] for j in same_vector_indices]
            for e, j in enumerate(same_vector_indices):
                for z in range(len(t_means[j])):
                    if t_means[j][z] is not None:
                        interpolated_ts[j][z] = t_means[j][z]
                        interpolated_normals[j][z] = normals_means[j][z]
                        fixed_points[j][z] = True
                        continue
                    adjacent_vector_ts = [same_vector_ts[a][z] for a in range(len(same_vector_ts))]
                    prev_mean, prev_ind, next_mean, next_ind = adjacent_means(e, adjacent_vector_ts)
                    
                    # Interpolate between the previous and next mean
                    if (next_ind - prev_ind) != 0:
                        interpolated_t = (next_mean - prev_mean) / (next_ind - prev_ind) * (e - prev_ind) + prev_mean
                    else:
                        interpolated_t = next_mean
                    interpolated_t = min(0.0, interpolated_t)
                    interpolated_ts[j][z] = interpolated_t
                    interpolated_normals[j][z] = normal
                    fixed_points[j][z] = False

        # fix wrong and too close t values
        interpolated_ts_range_valid = [True] * len(interpolated_ts)
        interpolated_ts_range = [*range(len(interpolated_ts))]
        interpolated_ts_range_valid_check = [i_ts for i_ts in interpolated_ts_range if interpolated_ts_range_valid[i_ts]]
        while True:
            interpolated_ts_range_valid = [False] * len(interpolated_ts)
            anything_changed = False
            # Adjust for at least 0.001 difference between t values
            abs_thresh = 0.001
            abs_iterations = 0
            total_interpolations = 0
            while True:
                abs_interpolations = 0
                for i in interpolated_ts_range_valid_check:
                    curve_angle_vector = angle_vector[i]
                    # get all indices with the same angle vector
                    same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
                    i_pos_in_same_vector = same_vector_indices.index(i)
                    old_abs_interpolations = abs_interpolations
                    for j in range(len(interpolated_ts[i])):
                        if interpolated_ts[i][j] is None:
                            continue
                        if winding_direction:
                            if i_pos_in_same_vector > 0 and abs(interpolated_ts[i][j] - interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]) < abs_thresh:
                                abs_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[i][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] - 2 * abs_thresh
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and abs(interpolated_ts[i][j] - interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]) < abs_thresh:
                                abs_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] = interpolated_ts[i][j] - 2 * abs_thresh
                        else:
                            if i_pos_in_same_vector > 0 and abs(interpolated_ts[i][j] - interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]) < abs_thresh:
                                abs_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] = interpolated_ts[i][j] - 2 * abs_thresh
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and abs(interpolated_ts[i][j] - interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]) < abs_thresh:
                                abs_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[i][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] - 2 * abs_thresh
                    if old_abs_interpolations != old_abs_interpolations: # Update the valid indices
                        for pos in same_vector_indices:
                            interpolated_ts_range_valid[pos] = True
                total_interpolations += abs_interpolations
                if abs_interpolations == 0:
                    break
                else:
                    anything_changed = True
                    print(f"Fixed {abs_interpolations} with total {total_interpolations} interpolations at iteration {abs_iterations}")
                abs_iterations += 1

            # Check interpolated
            check_iterations = 0
            total_interpolations = 0
            while True:
                flipped_interpolations = 0
                for i in interpolated_ts_range_valid_check:
                    curve_angle_vector = angle_vector[i]
                    # get all indices with the same angle vector
                    same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
                    i_pos_in_same_vector = same_vector_indices.index(i)
                    old_flipped_interpolations = flipped_interpolations
                    for j in range(len(interpolated_ts[i])):
                        if interpolated_ts[i][j] is None:
                            print(f"Interpolated ts is None at {i}, {j}")
                        if winding_direction:
                            if i_pos_in_same_vector > 0 and interpolated_ts[i][j] > interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                                flipped_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector-1]][j] = False
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] < interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                                flipped_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector+1]][j] = False
                        else:
                            if i_pos_in_same_vector > 0 and interpolated_ts[i][j] < interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                                flipped_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector-1]][j] = False
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] > interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                                flipped_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector+1]][j] = False
                    if old_flipped_interpolations != flipped_interpolations:
                        for pos in same_vector_indices:
                            interpolated_ts_range_valid[pos] = True

                for i in interpolated_ts_range_valid_check[::-1]:
                    curve_angle_vector = angle_vector[i]
                    # get all indices with the same angle vector
                    same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
                    i_pos_in_same_vector = same_vector_indices.index(i)
                    old_flipped_interpolations = flipped_interpolations
                    for j in range(len(interpolated_ts[i])-1, -1, -1):
                        if interpolated_ts[i][j] is None:
                            print(f"Interpolated ts is None at {i}, {j}")
                        if winding_direction:
                            if i_pos_in_same_vector > 0 and interpolated_ts[i][j] > interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                                flipped_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector-1]][j] = False
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] < interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                                flipped_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector+1]][j] = False
                        else:
                            if i_pos_in_same_vector > 0 and interpolated_ts[i][j] < interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                                flipped_interpolations += 1
                                # print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector-1]][j] = False
                            if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] > interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                                flipped_interpolations += 1
                                # print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                                interpolated_ts[i][j], interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j] = interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j], interpolated_ts[i][j]
                                fixed_points[i][j] = False
                                fixed_points[same_vector_indices[i_pos_in_same_vector+1]][j] = False
                    if old_flipped_interpolations != flipped_interpolations:
                        for pos in same_vector_indices:
                            interpolated_ts_range_valid[pos] = True

                total_interpolations += flipped_interpolations
                if flipped_interpolations == 0:
                    break
                else:
                    anything_changed = True
                    print(f"Flipped {flipped_interpolations} with total {total_interpolations} interpolations at iteration {check_iterations}")
                check_iterations += 1
                # update the range that had changes
                interpolated_ts_range_valid_check = [i_ts for i_ts in interpolated_ts_range if interpolated_ts_range_valid[i_ts]]
            
            if not anything_changed:
                break

        return interpolated_ts, interpolated_normals, fixed_points
    
    def deduct_ordered_pointset_neighbours(self, ordered_pointset, angle_vector, winding_direction):
        # create a dictionary with the indices of the points in the ordered pointset as keys and a list of the indices of the neighbouring points as values
        neighbours_dict = {}
        angle_set = set()
        for i in range(len(ordered_pointset)):
            if tuple(angle_vector[i]) in angle_set:
                continue
            angle_set.add(tuple(angle_vector[i]))
            same_vector_indices = self.extract_all_same_vector(angle_vector, angle_vector[i])
            for e, j in enumerate(same_vector_indices):
                for k in range(len(ordered_pointset[j])):
                    assert ordered_pointset[j][k] is not None, f"Point at {j}, {k} is None"
                    dict_key = (j, k)
                    if dict_key not in neighbours_dict:
                        neighbours_dict[dict_key] = {"front": None, "back": None, "top": None, "bottom": None, "left": None, "right": None}
                    # append top and bottom neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if k + l >= 0 and k + l < len(ordered_pointset[j]):
                            assert ordered_pointset[j][k+l] is not None, f"Point at {j}, {k+l} is None"
                            if l == -1:
                                neighbours_dict[dict_key]["top"] = (j, k+l)
                            else:
                                neighbours_dict[dict_key]["bottom"] = (j, k+l)
                    # append left and right neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if e + l >= 0 and e + l < len(same_vector_indices):
                            assert ordered_pointset[same_vector_indices[e+l]][k] is not None, f"Point at {same_vector_indices[e+l]}, {k} is None"
                            if l == -1:
                                neighbours_dict[dict_key]["left"] = (same_vector_indices[e+l], k)
                            else:
                                neighbours_dict[dict_key]["right"] = (same_vector_indices[e+l], k)
                    # swap left and right if winding direction is reversed
                    if not winding_direction:
                        neighbours_dict[dict_key]["left"], neighbours_dict[dict_key]["right"] = neighbours_dict[dict_key]["right"], neighbours_dict[dict_key]["left"]
                    # append front and back neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if j + l >= 0 and j + l < len(ordered_pointset):
                            assert ordered_pointset[j+l][k] is not None, f"Point at {j+l}, {k} is None"
                            if l == -1:
                                neighbours_dict[dict_key]["front"] = (j+l, k)
                            else:
                                neighbours_dict[dict_key]["back"] = (j+l, k)
        return neighbours_dict
    
    def optimize_adjacent(self, interpolated_ts, neighbours_dict, fixed_points, learning_rate=0.1):
        # Optimize the full pointset for smooth surface with best guesses for interpolated t values
        error_val_d = 0.01
        iterations = 3
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            nr_vertices = len(interpolated_ts) * len(interpolated_ts[0])
            nr_fixed = np.sum(fixed_points)
            nr_floating = nr_vertices - nr_fixed
            last_error_val = None
            for i in tqdm(range(10000), desc="Optimizing full pointset"): # 10000
                interpolated_ts, error_val = self.compute_interpolated_adjacent(neighbours_dict, interpolated_ts, fixed_points, learning_rate=learning_rate)
                error_val = error_val / nr_floating
                print(f"Error value per floating vertex: {error_val:.5f}")
                if last_error_val is not None and ((abs(last_error_val - error_val) < error_val_d) or (last_error_val - error_val < 0)):
                    break
                last_error_val = error_val

            self.detect_and_unfix_wrong_fixed_adjacent(neighbours_dict, interpolated_ts, fixed_points)

        return interpolated_ts
    
    def optimize_adjacent_cpp(self, interpolated_ts, neighbours_dict, fixed_points, learning_rate=0.1, iterations=3, error_val_d=0.01, unfix_factor=3.0, verbose=True):
        # translate neighbours_dict to list[list[list[list[int]]]] (i, j, [front, back, top, bottom, left, right], [i_n, j_n])
        neighbours_list = []
        for i in range(len(interpolated_ts)):
            neighbours_list.append([])
            for j in range(len(interpolated_ts[i])):
                neighbours_list[i].append([])
                dict_key = (i, j)
                neighbours = neighbours_dict[dict_key]
                # append front and back neighbours
                if neighbours["front"] is not None:
                    neighbours_list[i][j].append([int(neighbours["front"][0]), int(neighbours["front"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
                if neighbours["back"] is not None:
                    neighbours_list[i][j].append([int(neighbours["back"][0]), int(neighbours["back"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
                # same for top and bottom neighbours, left and right neighbours
                if neighbours["top"] is not None:
                    neighbours_list[i][j].append([int(neighbours["top"][0]), int(neighbours["top"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
                if neighbours["bottom"] is not None:
                    neighbours_list[i][j].append([int(neighbours["bottom"][0]), int(neighbours["bottom"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
                if neighbours["left"] is not None:
                    neighbours_list[i][j].append([int(neighbours["left"][0]), int(neighbours["left"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
                if neighbours["right"] is not None:
                    neighbours_list[i][j].append([int(neighbours["right"][0]), int(neighbours["right"][1])])
                else:
                    neighbours_list[i][j].append([int(-1), int(-1)])
        
        # convert interpolated_ts to list[list[float]]
        interpolated_ts_list = []
        for i in range(len(interpolated_ts)):
            interpolated_ts_list.append([])
            for j in range(len(interpolated_ts[i])):
                interpolated_ts_list[i].append(float(interpolated_ts[i][j]))
        
        # convert fixed_points to list[list[bool]]
        fixed_points_list = []
        for i in range(len(fixed_points)):
            fixed_points_list.append([])
            for j in range(len(fixed_points[i])):
                fixed_points_list[i].append(bool(fixed_points[i][j]))
        
        # call C++ function
        interpolated_ts_list = pointcloud_processing.optimize_adjacent(
                                    interpolated_ts_list, 
                                    fixed_points_list, 
                                    neighbours_list, 
                                    learning_rate=float(learning_rate), 
                                    iterations=int(iterations),
                                    error_val_d=float(error_val_d),
                                    unfix_factor=float(unfix_factor),
                                    verbose=bool(verbose)
                                    )
        
        return interpolated_ts_list
                
    
    def ordered_pointset_to_3D(self, interpolated_ts, ordered_umbilicus_points, angle_vector):
        # go from interpolated t values to ordered pointset (3D points)
        interpolated_points = []
        print(f"Length of interpolated_ts: {len(interpolated_ts)}")
        print(f"Length of ordered_umbilicus_points (should be same as other two, but since every entry is same it is okay that it didnt get angle clipped): {len(ordered_umbilicus_points)}") # Todo: clip as well when clipping away angles
        print(f"Length of angle_vector: {len(angle_vector)}")
        for i in tqdm(range(len(interpolated_ts)), desc="Converting to Scroll Thaumato carthesian 3D coordinates"):
            interpolated_points.append([])
            # print(f"Length of interpolated_ts[i]: {len(interpolated_ts[i])}")
            # print(f"Length of ordered_umbilicus_points[i]: {len(ordered_umbilicus_points[i])}")
            # print(f"Shape ordered umbilicus points: {np.array(ordered_umbilicus_points[i]).shape}")
            for j in range(len(interpolated_ts[i])):
                if interpolated_ts[i][j] is not None:
                    # try:
                    #     print(f"Shape o u p: {np.array(ordered_umbilicus_points[i][j]).shape}")
                    #     print(f"Shape o a v: {np.array(angle_vector[i]).shape}")
                    # except:
                    #     pass
                    # print(f"Shapes of ordered_umbilicus_points: {np.array(ordered_umbilicus_points[i][j]).shape}, {interpolated_ts[i][j].shape}, {np.array(angle_vector[i]).shape}")
                    interpolated_points[i].append(np.array(ordered_umbilicus_points[i][j]) + interpolated_ts[i][j] * np.array(angle_vector[i]))
                else:
                    raise ValueError("Interpolated t is None")
        return interpolated_points
    
    def value_from_key(self, key, ordered_pointset):
        if key is None:
            return None
        else:
            return ordered_pointset[key[0]][key[1]]
                    
    def calculate_ms(self, ordered_pointset, neighbours_dict):
        m_r = [None]*len(ordered_pointset)
        m_l = [None]*len(ordered_pointset)
        m_ts = [None]*len(ordered_pointset)
        r = [None]*len(ordered_pointset)
        l = [None]*len(ordered_pointset)
        for i in range(len(ordered_pointset)):
            m_r[i] = [None]*len(ordered_pointset[i])
            m_l[i] = [None]*len(ordered_pointset[i])
            m_ts[i] = [None]*len(ordered_pointset[i])
            r[i] = [None]*len(ordered_pointset[i])
            l[i] = [None]*len(ordered_pointset[i])
        for i in range(len(ordered_pointset)):
            for k in range(len(ordered_pointset[i])):
                dict_key = (i, k)
                neighbours = neighbours_dict[dict_key]
                l[i][k] = neighbours["left"]
                r[i][k] = neighbours["right"]
                l[i][k] = self.value_from_key(l[i][k], ordered_pointset)
                assert l[i][k] is None or l[i][k] <= 0.0, f"l[i][k] is not less than 0: {l[i][k]}"
                r[i][k] = self.value_from_key(r[i][k], ordered_pointset)
                assert r[i][k] is None or r[i][k] <= 0.0, f"r[i][k] is not less than 0: {r[i][k]}"
                fb = [neighbours["front"], neighbours["back"]]
                tb = [neighbours["top"], neighbours["bottom"]]
                same_sheet_neighbours = fb + tb
                same_sheet_neighbours = [n for n in same_sheet_neighbours if n is not None] # filter out None values
                m_r_ = 0.0
                m_l_ = 0.0
                m_ts_ = 0.0
                count_r = 0
                count_l = 0
                for n in same_sheet_neighbours:
                    ts_n = ordered_pointset[n[0]][n[1]]
                    assert ts_n <= 0.0, f"ts_n is not less than 0: {ts_n}"
                    m_ts_ += ts_n

                    l_n = neighbours_dict[n]["left"]
                    r_n = neighbours_dict[n]["right"]

                    l_n = self.value_from_key(l_n, ordered_pointset)
                    r_n = self.value_from_key(r_n, ordered_pointset)

                    if l_n is not None:
                        m_l_ += (l_n - ts_n)
                        count_l += 1
                    if r_n is not None:
                        m_r_ += (r_n - ts_n)
                        count_r += 1

                if count_r == 0:
                    m_r_ = None
                else:
                    m_r_ /= count_r
                if count_l == 0:
                    m_l_ = None
                else:
                    m_l_ /= count_l
                m_ts_ /= len(same_sheet_neighbours)
                m_r[i][k] = m_r_
                m_l[i][k] = m_l_
                m_ts[i][k] = m_ts_

        return r, l, m_r, m_l, m_ts
    
    def solve_for_t_individual(self, r, l, m_r, m_l, m_ts, a=0.01):
        t_ts = m_ts
        t_total = t_ts
        count_total = a
        if (r is not None) and (m_r is not None):
            t_r = r - m_r
            t_total += t_r
            count_total += 1.0
        if (l is not None) and (m_l is not None):
            t_l = l - m_l
            t_total += t_l
            count_total += 1.0
        t_total /= count_total

        if t_total > 0.0:
            t_total = 0.0

        return t_total
    
    def respect_non_overlapping(self, i, j, neighbours_dict, interpolated_ts, new_interpolated_ts, new_ts_d):
        # respect the non-overlapping invariant of the ordered pointset
        def side_of(ts_, n):
            # calculate what side ts_ is wrt to n
            return ts_ > n, ts_ == n
        old_ts = interpolated_ts[i][j]
        l = neighbours_dict[(i, j)]["left"]
        r = neighbours_dict[(i, j)]["right"]
        ln = self.value_from_key(l, new_interpolated_ts)
        rn = self.value_from_key(r, new_interpolated_ts)
        l = self.value_from_key(l, interpolated_ts)
        r = self.value_from_key(r, interpolated_ts)
        if l is not None:
            side_old, invalid = side_of(old_ts, l)
            assert not invalid, f"old_ts: {old_ts}, l: {l}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, l)
            if side_old != side_new:
                new_ts_d = l - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        if r is not None:
            side_old, invalid = side_of(old_ts, r)
            assert not invalid, f"old_ts: {old_ts}, r: {r}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, r)
            if side_old != side_new:
                new_ts_d = r - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        if ln is not None:
            side_old, invalid = side_of(old_ts, ln)
            assert not invalid, f"old_ts: {old_ts}, ln: {ln}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, ln)
            if side_old != side_new:
                new_ts_d = ln - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        if rn is not None:
            side_old, invalid = side_of(old_ts, rn)
            assert not invalid, f"old_ts: {old_ts}, rn: {rn}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, rn)
            if side_old != side_new:
                new_ts_d = rn - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        return new_ts_d
    
    def compute_interpolated_adjacent(self, neighbours_dict, interpolated_ts, fixed_points, learning_rate=0.1):
        r, l, m_r, m_l, m_ts = self.calculate_ms(interpolated_ts, neighbours_dict)
        error_val = 0.0

        new_interpolated_ts = deepcopy(interpolated_ts)

        # Compute interpolated values for all points
        for i in range(len(interpolated_ts)):
            for j in range(len(interpolated_ts[i])):
                if not fixed_points[i][j]:
                    t = self.solve_for_t_individual(r[i][j], l[i][j], m_r[i][j], m_l[i][j], m_ts[i][j], a=1.0)
                    assert t <= 0.0, f"t is not less than 0: {t}"
                    d_t = t - interpolated_ts[i][j]
                    # the raw d_t might be breaking the non-overlapping invariant
                    d_t = self.respect_non_overlapping(i, j, neighbours_dict, interpolated_ts, new_interpolated_ts, d_t)
                    error_val += abs(d_t)
                    new_interpolated_ts[i][j] = interpolated_ts[i][j] + learning_rate * d_t
        
        return new_interpolated_ts, error_val
    
    def interpolated_adjacent_errors(self, neighbours_dict, interpolated_ts):
        # Calculate the errors for all points
        errors = [[None]*len(interpolated_ts[i]) for i in range(len(interpolated_ts))]
        r, l, m_r, m_l, m_ts = self.calculate_ms(interpolated_ts, neighbours_dict)

        # Compute interpolated values for all points
        for i in range(len(interpolated_ts)):
            for j in range(len(interpolated_ts[i])):
                t = self.solve_for_t_individual(r[i][j], l[i][j], m_r[i][j], m_l[i][j], m_ts[i][j], a=1.0)
                assert t <= 0.0, f"t is not less than 0: {t}"
                d_t = t - interpolated_ts[i][j]
                # the raw d_t might be breaking the non-overlapping invariant
                d_t = self.respect_non_overlapping(i, j, neighbours_dict, interpolated_ts, interpolated_ts, d_t)
                errors[i][j] = abs(d_t)
        return errors
    
    def detect_and_unfix_wrong_fixed_adjacent(self, neighbours_dict, interpolated_ts, fixed_points):
        # Detect wrong fixed points
        errors = self.interpolated_adjacent_errors(neighbours_dict, interpolated_ts)
        error_mean_fixed = np.mean([errors[i][j] for i in range(len(errors)) for j in range(len(errors[i])) if fixed_points[i][j]])
        error_threshold = 1.1 * error_mean_fixed
        # unfix wrongly fixed points
        for i in range(len(errors)):
            for j in range(len(errors[i])):
                if fixed_points[i][j] and errors[i][j] > error_threshold:
                    fixed_points[i][j] = False

    def interpolate_ordered_pointset(self, ordered_pointset, ordered_normals, angle_vector, winding_direction):
        computed_vectors_set = set()
        result_ts = []
        result_normals = []
        for i in range(len(ordered_pointset)):
            result_ts.append([None]*len(ordered_pointset[i]))
            result_normals.append([None]*len(ordered_pointset[i]))
            
        for i in tqdm(range(len(ordered_pointset))):
            curve_angle_vector = angle_vector[i]
            if tuple(curve_angle_vector) in computed_vectors_set:
                continue
            computed_vectors_set.add(tuple(curve_angle_vector))
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            same_vector_ts = [ordered_pointset[j] for j in same_vector_indices]
    
            same_vector_normals = [ordered_normals[j] for j in same_vector_indices]
            t_means, normals_means = compute_means_adjacent(same_vector_ts, same_vector_normals, winding_direction)

            # print(f"length of t_means: {len(t_means)}")
            for e, j in enumerate(same_vector_indices):
                result_ts[j] = t_means[e]
                result_normals[j] = normals_means[e]
                # # Only for debugging without subsequent initial pointset and optimization
                # for o in range(len(t_means[e])):
                #     if t_means[e][o] is None:
                #         result_ts[j][o] = 0.0
                #         result_normals[j][o] = np.array([1.0, 0.0, 0.0])

        return result_ts, result_normals
    
    def interpolate_ordered_pointset_multithreaded(self, ordered_pointset, ordered_normals, angle_vector, winding_direction):
        computed_vectors_set = set()
        result_ts = []
        result_normals = []
        for i in range(len(ordered_pointset)):
            result_ts.append([None]*len(ordered_pointset[i]))
            result_normals.append([None]*len(ordered_pointset[i]))
            
        args = []
        list_same_v_i = []
        for i in tqdm(range(len(ordered_pointset))):
            curve_angle_vector = angle_vector[i]
            if tuple(curve_angle_vector) in computed_vectors_set:
                continue
            computed_vectors_set.add(tuple(curve_angle_vector))
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            same_vector_ts = [ordered_pointset[j] for j in same_vector_indices]
            same_vector_normals = [ordered_normals[j] for j in same_vector_indices]
    
            args.append((same_vector_ts, same_vector_normals, winding_direction))
            list_same_v_i.append(same_vector_indices)
            
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(compute_means_adjacent_args, args), total=len(args), desc="Interpolating ordered pointset"))

        for i, (t_mean, normals_mean) in enumerate(results):
            for e, j in enumerate(list_same_v_i[i]):
                result_ts[j] = t_mean[e]
                result_normals[j] = normals_mean[e]

        return result_ts, result_normals
    
    def clip_valid_windings(self, result_ts, result_normals, angle_vector, angle_step, valid_p_winding=0.1, valid_p_z=0.1):
        # Clip away invalid windings at the beginning and end of the ordered pointset
        steps_per_winding = 360 / angle_step
        nr_windings = int(np.ceil(len(result_ts)/steps_per_winding))

        valid_start_indx = 0
        for i in range(nr_windings):
            start_index = int(i * steps_per_winding)
            end_index = int(min((i+1) * steps_per_winding, len(result_ts)))
            winding_ts = [result_ts[j] for j in range(start_index, end_index)]
            winding_normals = [result_normals[j] for j in range(start_index, end_index)]
            # check if winding is valid
            valid_winding = self.check_valid_winding(winding_ts, winding_normals, valid_p_winding, valid_p_z)
            if not valid_winding:
                valid_start_indx = i+1
            else:
                break
        print(f"Found valid start index: {valid_start_indx}")
        
        valid_end_indx = nr_windings
        for i in range(nr_windings-2, valid_start_indx, -1): # skip last winding
            start_index = int(i * steps_per_winding)
            end_index = int(min((i+1) * steps_per_winding, len(result_ts)))
            # skip last winding if it is too short
            if end_index - start_index < steps_per_winding / 2:
                continue
            winding_ts = [result_ts[j] for j in range(start_index, end_index)]
            winding_normals = [result_normals[j] for j in range(start_index, end_index)]
            # check if winding is valid
            valid_winding = self.check_valid_winding(winding_ts, winding_normals, valid_p_winding, valid_p_z)
            if not valid_winding:
                valid_end_indx = i
            else:
                break

        print(f"Clipped windings from {valid_start_indx} to {valid_end_indx}. Total length: {nr_windings}.")
        # Translate indices from winding to pointset indices
        valid_start_indx = int(valid_start_indx * steps_per_winding)
        valid_end_indx = min(int(valid_end_indx * steps_per_winding), len(result_ts))

        # Clip the valid windings
        if valid_start_indx < valid_end_indx:
            valid_ts = result_ts[valid_start_indx:valid_end_indx]
            valid_normals = result_normals[valid_start_indx:valid_end_indx]
            angle_vector = angle_vector[valid_start_indx:valid_end_indx]
            # reset same vector indices
            global angle_vector_indices_dp
            angle_vector_indices_dp = {}
        else:
            valid_ts = []
            valid_normals = []
            angle_vector = []

        return [valid_ts], [valid_normals], [angle_vector]
    
    def clip_valid_angles_fragment(self, result_ts, result_normals, angle_vector, angle_step, valid_p_winding=0.01, valid_p_z=0.01):
        # Clip away invalid windings at the beginning and end of the ordered pointset
        steps_per_winding = 360 / angle_step / 120
        nr_windings = int(np.ceil(len(result_ts)/steps_per_winding))

        start_indices = []
        end_indices = []

        valid_start_indx = 0
        valid_end_indx = 0
        while valid_start_indx < nr_windings:
            # Continue from end
            valid_start_indx = valid_end_indx + 1
            for i in range(valid_start_indx, nr_windings):
                start_index = int(i * steps_per_winding)
                end_index = int(min((i+1) * steps_per_winding, len(result_ts)))
                winding_ts = [result_ts[j] for j in range(start_index, end_index)]
                winding_normals = [result_normals[j] for j in range(start_index, end_index)]
                # check if winding is valid
                valid_winding = self.check_valid_winding(winding_ts, winding_normals, valid_p_winding, valid_p_z)
                if not valid_winding:
                    valid_start_indx = i+1
                else:
                    break
            print(f"Found valid start index: {valid_start_indx}")
            
            valid_end_indx = valid_start_indx
            for i in range(valid_end_indx, nr_windings-2): # skip last winding
                start_index = int(i * steps_per_winding)
                end_index = int(min((i+1) * steps_per_winding, len(result_ts)))
                winding_ts = [result_ts[j] for j in range(start_index, end_index)]
                winding_normals = [result_normals[j] for j in range(start_index, end_index)]
                # check if winding is valid
                valid_winding = self.check_valid_winding(winding_ts, winding_normals, valid_p_winding, valid_p_z)
                if valid_winding:
                    valid_end_indx = i
                else:
                    break

            # Add some extra windings
            if valid_start_indx < valid_end_indx:
                valid_start_indx = max(valid_start_indx - 3, 0)
                valid_end_indx = min(valid_end_indx + 3, nr_windings)
            # Translate indices from winding to pointset indices
            print(f"Clipped fragment windings from {valid_start_indx} to {valid_end_indx}. Total length: {nr_windings}.")
            valid_start_indx_ = int(valid_start_indx * steps_per_winding)
            valid_end_indx_ = min(int(valid_end_indx * steps_per_winding), len(result_ts))

            # Clip the valid windings
            if valid_start_indx_ < valid_end_indx_:
                start_indices.append(valid_start_indx_)
                end_indices.append(valid_end_indx_)

                # # reset same vector indices
                global angle_vector_indices_dp
                angle_vector_indices_dp = {}

        return start_indices, end_indices
    
    def valid_z_values_indices(self, result_ts, valid_p_z=0.1):
        # Clip away invalid z values
        assert len(result_ts) > 0, "No t values found."
        
        z_height = len(result_ts[0])

        valid_bottom_index = 0
        # Check for all good z values
        for u in range(z_height):
            valid_bottom_count = 0
            valid_bottom_index = u
            for i in range(len(result_ts)):
                if result_ts[i][u] is not None:
                    valid_bottom_count += 1
            if valid_bottom_count / len(result_ts) > valid_p_z:
                break

        valid_top_index = z_height
        # Check for all good z values
        for u in range(z_height-1, valid_bottom_index, -1):
            valid_top_count = 0
            valid_top_index = u
            for i in range(len(result_ts)):
                if result_ts[i][u] is not None:
                    valid_top_count += 1
            if valid_top_count / len(result_ts) > valid_p_z:
                break
            
        print(f"Clipped z values from {valid_bottom_index} to {valid_top_index}. Total length: {z_height}.")
        return valid_bottom_index, valid_top_index
    
    def check_valid_winding(self, winding_ts, winding_normals, valid_p_winding=0.25, valid_p_z=0.25):
        # if more than valid_p_z percent of the t values in a step are None, the winding step is invalid
        # if more than valid_p_winding percent of the winding steps are invalid, the winding is invalid
        valid_winding = False
        count_valid_steps = 0
        for i in range(len(winding_ts)):
            count_invalid_z = 0
            for j in range(len(winding_ts[i])):
                if winding_ts[i][j] is None:
                    count_invalid_z += 1
            p_valid_z = 1 - count_invalid_z / len(winding_ts[i])
            # print(f"Valid z values: {p_valid_z}")
            if p_valid_z > valid_p_z:
                count_valid_steps += 1
        p_valid_steps = count_valid_steps / len(winding_ts)
        # print(f"Valid winding steps: {p_valid_steps}")
        if p_valid_steps > valid_p_winding:
            valid_winding = True
        return valid_winding

    def ordered_pointset_to_pointcloud_save(self, ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector):
        pointset = []

        for i in tqdm(range(len(ordered_pointset))):
            ordered_curve = ordered_pointset[i]
            ordered_normals_curve = ordered_normals[i]
            ordered_umbilicus_curve = ordered_umbilicus_points[i]
            curve_angle_vector = angle_vector[i]
            for j in range(len(ordered_curve)):
                umbilicus_p = np.array(ordered_umbilicus_curve[j])
                np_angle_vec = np.array(curve_angle_vector)
                # print(f"shape of umbilicus_p: {umbilicus_p.shape}, shape of curve_angle_vector: {np_angle_vec.shape}")
                point = umbilicus_p + ordered_curve[j] * np_angle_vec
                point = np.array(point)
                normal = ordered_normals_curve[j]
                normal = np.array(normal)
                pointset.append(([point], [normal]))

        # remove test folder
        test_folder = os.path.join(self.save_path, "test_cpp")
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        os.makedirs(test_folder)
        test_pointset_ply_path = os.path.join(test_folder, f"ordered_pointset_test_cpp.ply")
        self.pointcloud_from_ordered_pointset(pointset, os.path.join(self.save_path, test_pointset_ply_path))
           
    def rolled_ordered_pointset(self, result, continue_from=0, fragment=False, angle_step=0.5, learning_rate=0.2, iterations=11, unfix_factor=2.5):
        print("length of result: ", len(result))
        ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector = zip(*result)
        print("length of ordered_pointset: ", len(ordered_pointset))

        # determine winding direction
        t_means = self.calculate_means(ordered_pointset)
        mean_innermost_ts, mean_outermost_ts, winding_direction = self.find_inner_outermost_winding_direction(t_means, angle_vector)

        # Set to false to load precomputed partial results during development
        fresh_start2 = continue_from <= 3
        if fresh_start2:
            result_ts, result_normals = self.interpolate_ordered_pointset_multithreaded(ordered_pointset, ordered_normals, angle_vector, winding_direction)
            interpolated_ts, interpolated_normals = result_ts, result_normals
            # save result as pkl
            result_pkl_path = os.path.join(self.save_path, "results_ts_normals.pkl")
            with open(result_pkl_path, 'wb') as f:
                pickle.dump((result_ts, result_normals), f)
        elif continue_from <= 4:
            result_pkl_path = os.path.join(self.save_path, "results_ts_normals.pkl")
            with open(result_pkl_path, 'rb') as f:
                (result_ts, result_normals) = pickle.load(f)
        
        ordered_pointsets_final_s = []
        fresh_start3 = continue_from <= 4
        if fresh_start3:
            valid_p = 0.4
            if not fragment:
                valid_ts_s, valid_normals_s, angle_vector_s = self.clip_valid_windings(result_ts, result_normals, angle_vector, angle_step, valid_p_winding=valid_p, valid_p_z=valid_p)
            else:
                valid_p = 0.1
                start_indices, end_indices = self.clip_valid_angles_fragment(result_ts, result_normals, angle_vector, angle_step, valid_p_winding=valid_p, valid_p_z=valid_p)

                # optimize before cutting the pointset apart
                start_index_0 = start_indices[0]
                end_index_last = end_indices[-1]
                valid_ts = result_ts[start_index_0:end_index_last]
                valid_normals = result_normals[start_index_0:end_index_last]
                angle_vector_ = angle_vector[start_index_0:end_index_last]

                # interpolate initial full pointset. After this step there exists an "ordered pointset" prototype without any None values
                interpolated_ts, interpolated_normals, fixed_points = self.initial_full_pointset(valid_ts, valid_normals, angle_vector_, mean_innermost_ts, mean_outermost_ts, winding_direction)

                # Calculate for each point in the ordered pointset its neighbouring indices (3d in a 2d list). on same sheet, top bottom, front back, adjacent sheets neighbours: left right
                neighbours_dict = self.deduct_ordered_pointset_neighbours(interpolated_ts, angle_vector_, winding_direction)

                # Optimize the full pointset for smooth surface with best guesses for interpolated t values
                interpolated_ts = self.optimize_adjacent_cpp(interpolated_ts, neighbours_dict, fixed_points, 
                                                            learning_rate=learning_rate, iterations=iterations, error_val_d=0.0001, unfix_factor=unfix_factor,
                                                            verbose=True)
                
                valid_ts_s = [interpolated_ts[start_indices[i]-start_index_0:end_indices[i]-start_index_0] for i in range(len(start_indices))]
                valid_normals_s = [interpolated_normals[start_indices[i]-start_index_0:end_indices[i]-start_index_0] for i in range(len(start_indices))]
                angle_vector_s = [angle_vector[start_indices[i]:end_indices[i]] for i in range(len(start_indices))]

            for valid_i_s in range(len(valid_ts_s)):
                valid_ts = valid_ts_s[valid_i_s]
                valid_normals = valid_normals_s[valid_i_s]
                angle_vector = angle_vector_s[valid_i_s]

                # clear angular precomputation
                global angle_vector_indices_dp
                angle_vector_indices_dp = {}

                valid_bottom_index, valid_top_index = self.valid_z_values_indices(valid_ts, valid_p_z=valid_p)

                if not fragment:
                    # interpolate initial full pointset. After this step there exists an "ordered pointset" prototype without any None values
                    interpolated_ts, interpolated_normals, fixed_points = self.initial_full_pointset(valid_ts, valid_normals, angle_vector, mean_innermost_ts, mean_outermost_ts, winding_direction)

                    # Calculate for each point in the ordered pointset its neighbouring indices (3d in a 2d list). on same sheet, top bottom, front back, adjacent sheets neighbours: left right
                    neighbours_dict = self.deduct_ordered_pointset_neighbours(interpolated_ts, angle_vector, winding_direction)

                    # Optimize the full pointset for smooth surface with best guesses for interpolated t values
                    # interpolated_ts = self.optimize_adjacent_cpp(interpolated_ts, neighbours_dict, fixed_points, 
                    #                                             learning_rate=0.1, iterations=7, error_val_d=0.0001, unfix_factor=5.0,
                    #                                             verbose=True)
                    interpolated_ts = self.optimize_adjacent_cpp(interpolated_ts, neighbours_dict, fixed_points, 
                                                                learning_rate=0.2, iterations=iterations, error_val_d=0.0001, unfix_factor=unfix_factor,
                                                                verbose=True)
                else:
                    interpolated_ts = valid_ts
                    interpolated_normals = valid_normals
                    print(f"length of interpolated_ts: {len(interpolated_ts)}, length of interpolated_normals: {len(interpolated_normals)}, length of angle_vector: {len(angle_vector)}")

                # Clip away invalid z values
                interpolated_ts = [interpolated_ts[i][valid_bottom_index:valid_top_index] for i in range(len(interpolated_ts))]
                interpolated_normals = [interpolated_normals[i][valid_bottom_index:valid_top_index] for i in range(len(interpolated_normals))]
                ordered_umbilicus_points_ = [ordered_umbilicus_points[i][valid_bottom_index:valid_top_index] for i in range(len(ordered_umbilicus_points))]

                # go from interpolated t values to ordered pointset (3D points)
                interpolated_points = self.ordered_pointset_to_3D(interpolated_ts, ordered_umbilicus_points_, angle_vector)

                interpolated_subsample_indices = self.subsample_interpolated_points(interpolated_points, mean_distance_threshold=3.333)
                # subsample everything
                interpolated_ts = [interpolated_ts[i] for i in interpolated_subsample_indices]
                interpolated_points = [interpolated_points[i] for i in interpolated_subsample_indices]
                interpolated_normals = [interpolated_normals[i] for i in interpolated_subsample_indices]
                ordered_umbilicus_points_ = [ordered_umbilicus_points_[i] for i in interpolated_subsample_indices]
                angle_vector = [angle_vector[i] for i in interpolated_subsample_indices]

                print("Finished Cpp rolled_ordered_pointset")

                # Creating ordered pointset output format
                ordered_pointsets_final = []
                for i in range(len(interpolated_points)):
                    ordered_pointsets_final.append((np.array(interpolated_points[i]), np.array(interpolated_normals[i])))

                # Debug output
                self.ordered_pointset_to_pointcloud_save(interpolated_ts, interpolated_normals, ordered_umbilicus_points, angle_vector)

                ordered_pointsets_final_s.append(ordered_pointsets_final)     
            # Save the ordered_pointsets_final
            ordered_pointsets_final_pkl_path = os.path.join(self.save_path, "ordered_pointsets_final_s.pkl")
            with open(ordered_pointsets_final_pkl_path, 'wb') as f:
                pickle.dump(ordered_pointsets_final_s, f)
        else:
            ordered_pointsets_final_pkl_path = os.path.join(self.save_path, "ordered_pointsets_final_s.pkl")
            with open(ordered_pointsets_final_pkl_path, 'rb') as f:
                ordered_pointsets_final_s = pickle.load(f)       

        return ordered_pointsets_final_s
    
    def subsample_interpolated_points(self, interpolated_points, mean_distance_threshold=5):
        # subsample ordered pointset columns to reduce number of columns. they are too dense in the ceter and inflate the mesh size. 
        # make it harder to work with, store and flatten. nearly no information lost by subsampling there.
        interpolated_points_ = np.array(interpolated_points)
        interpolated_subsample_indices = []
        last_taken_position = None # subsample the pointset to have approximately the same distance between two connecting verices in the xy plane
        for i in range(len(interpolated_points_)):
            if last_taken_position is not None:
                # compute mean distance between points in the xy plane
                distances = np.linalg.norm(interpolated_points_[i] - interpolated_points_[last_taken_position], axis=1)
                assert len(distances) == len(interpolated_points_[i]), f"Length of distances: {len(distances)}, length of interpolated_points_: {len(interpolated_points_)}"
                mean_distance = np.mean(distances)
                if mean_distance < mean_distance_threshold:
                    continue
            last_taken_position = i # store this position since it was added
            interpolated_subsample_indices.append(i)
        print(f"Taken {len(interpolated_subsample_indices)} ordered columns from {len(interpolated_points)} total.")
        return interpolated_subsample_indices
    
    def save_graph_pointcloud(self, graph_path):
        save_path = os.path.join(self.save_path, "graph_pointcloud.ply")
        print(save_path)
        points, normals, colors = self.build_points_reference(graph_path)

        # Scale and translate points to original coordinate system
        points = scale_points(np.array(points[:, :3]))
        # Rotate back to original axis
        points, normals = shuffling_points_axis(points, np.array(normals))
        # open3d
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(save_path, pcl)
        
    def pointcloud_from_ordered_pointset(self, ordered_pointset, filename, color=None):
        points, normals = [], []
        for point, normal in tqdm(ordered_pointset, desc="Building pointcloud"):
            if point is not None:
                points.append(point)
                normals.append(normal)
        points = np.concatenate(points, axis=0).astype(np.float64)
        normals = np.concatenate(normals, axis=0).astype(np.float64)

        # Scale and translate points to original coordinate system
        points[:, :3] = scale_points(points[:, :3])
        # Rotate back to original axis
        points, normals = shuffling_points_axis(points, normals)
        # invert normals (layer 0 in the papyrus, layer 64 on top and outside the papyrus, layer 32 surface layer (for 64 layer surface volume))
        normals = -normals

        print(f"Saving {points.shape} points to {filename}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        if not color is None:
            # color is shape n, 1, go to rgb with hsv
            color = (color - np.min(color)) / (np.max(color) - np.min(color))
            # use colormap cold
            cmap = matplotlib.cm.get_cmap('coolwarm')
            color = cmap(color)[:, :3]
            print(color.shape, color.dtype)
            pcd.colors = o3d.utility.Vector3dVector(color)
        # Create folder if it doesn't exist
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save as a PLY file
        o3d.io.write_point_cloud(filename, pcd)

    def mesh_from_ordered_pointset(self, ordered_pointsets):
        """
        Creates a mesh from an ordered list of point sets. Each point set is a list of entries, and each entry is a tuple of point and normal.
        Assumes each point set has the same number of entries and they are ordered consistently.

        Parameters:
        ordered_pointsets (list of lists of tuples): The ordered list of pointsets where each pointset contains tuples of (x, y, z) coordinates.

        Returns:
        o3d.geometry.TriangleMesh: The resulting mesh.
        """
        vertices = []
        normals = []
        triangles = []
        tringles_uv = []

        # First, convert all pointsets into a single list of vertices
        vertices, normals = [], []
        for point, normal in ordered_pointsets:
            if point is not None:
                vertices.append(point)
                normals.append(normal)
        # Convert to Open3D compatible format
        vertices = np.concatenate(vertices, axis=0)
        normals = np.concatenate(normals, axis=0)
        # Invert normals
        normals = -normals

        # Scale and translate vertices to original coordinate system
        vertices[:, :3] = scale_points(vertices[:, :3])
        # Rotate back to original axis
        vertices, normals = shuffling_points_axis(vertices, normals)
        # scale vertices back to CT scan coordinates
        vertices[:, :3] = self.scale_factor * vertices[:, :3]
        
        # Number of rows and columns in the pointset grid
        num_rows = len(ordered_pointsets)
        num_cols = len(ordered_pointsets[0][0]) if num_rows > 0 else 0
        ordered_pointsets = np.array(ordered_pointsets)

        print(f"Number of rows: {num_rows}, number of columns: {num_cols}")

        # Generate triangles
        for i in tqdm(range(num_rows - 1), desc="Generating triangles"):
            for j in range(num_cols - 1):
                # Current point and the points to its right and below it
                idx = i * num_cols + j
                triangles.append([idx, idx + 1, idx + num_cols])
                tringles_uv.append(((i / num_rows, j / num_cols), (i / num_rows, (j + 1) / num_cols), ((i+1) / num_rows, j / num_cols)))
                triangles.append([idx + 1, idx + num_cols + 1, idx + num_cols])
                tringles_uv.append(((i / num_rows, (j+1) / num_cols), ((i + 1) / num_rows, (j + 1) / num_cols), ((i+1) / num_rows, j / num_cols)))

        # Create the mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        triangle_uvs = np.array(tringles_uv).reshape((-1, 2))
        # assert triangle uvs are not nan or inf
        assert not np.any(np.isnan(triangle_uvs)), "Triangle UVs contain NaN values"
        assert not np.any(np.isinf(triangle_uvs)), "Triangle UVs contain Inf values"
        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
        # compute triangle normals and normals
        mesh = mesh.compute_triangle_normals()
        mesh = mesh.compute_vertex_normals()

        # Create a grayscale image with a specified size
        n, m = int(np.ceil(num_rows)), int(np.ceil(num_cols))  # replace with your dimensions
        uv_image = Image.new('L', (n, m), color=255)  # 255 for white background, 0 for black

        return mesh, uv_image
    
    def save_mesh(self, mesh, uv_image, filename):
        """
        Save a mesh to a file.

        Parameters:
        mesh (o3d.geometry.TriangleMesh): The mesh to save.
        filename (str): The path to save the mesh to.
        """
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save as a PLY file
        o3d.io.write_triangle_mesh(filename, mesh)

        # Save the UV image
        if uv_image is not None:
            uv_image.save(filename[:-4] + ".png")

        print(f"Saved mesh to {filename}")

    def split(self, mesh_path, split_width=50000, fresh_start=True, stamp=None):
        # Split Scroll mesh into smaller parts of width split_width
        umbilicus_path = os.path.join(os.path.dirname(self.path), "umbilicus.txt")
        splitter = MeshSplitter(mesh_path, umbilicus_path, self.scale_factor)
        split_mesh_paths, stamp = splitter.compute(split_width=split_width, fresh_start=fresh_start, stamp=stamp)
        # Return the paths to the split meshes
        return split_mesh_paths, stamp
    
    def load_pointcloud_to_raw_ordered_pointset(self, debug=False, continue_from=0, z_range=None, angle_step=1, z_spacing=10):
        # retrieve min max approximate winding angles
        approx_min_angle, approx_max_angle, approx_min_z, approx_max_z = self.find_minmax_winding_angle_z_range()
        print(f"Approximate min and max winding angles: {approx_min_angle}, {approx_max_angle} with z range: {approx_min_z}, {approx_max_z}")
        # adjust to desired z range
        if z_range is not None:
            approx_min_z = max(approx_min_z, z_range[0])
            approx_max_z = min(approx_max_z, z_range[1])
        approx_min_angle -= 360.0
        approx_max_angle += 360.0
        approx_min_z -= 50
        approx_max_z += 50
        print(f"Approximate min and max winding angles: {approx_min_angle}, {approx_max_angle} with z range: {approx_min_z}, {approx_max_z}")

        # loop over build + ordered pointset creation for each z range step (with overlap in loaded pointclouds)
        z_step = 50 * z_spacing
        results = None
        for step_index, z_start in enumerate(tqdm(range(approx_min_z, approx_max_z, z_step), desc="Z range steps")):
            result_pkl_path = os.path.join(self.save_path, f"ordered_pointset_{step_index}.pkl")
            z_end = z_start + z_step
            z_range = (z_start - 20, z_end + 20)
            print(f"z_range: {z_range}")

            # Set to false to load precomputed partial results during development
            start_fresh = continue_from <= 1
            path_points = os.path.join(self.save_path, f"points_{step_index}.npz")
            if start_fresh or not os.path.exists(path_points):
                # Set to false to load precomputed partial results during development
                start_fresh_build_points = continue_from <= 0 or not os.path.exists(path_points)
                if start_fresh_build_points:
                    # get points
                    points, normals, colors = self.build_points(z_range=z_range)

                    # Make directory if it doesn't exist
                    os.makedirs(self.save_path, exist_ok=True)
                    # Save as npz
                    with open(path_points, 'wb') as f:
                        np.savez(f, points=points, normals=normals, colors=colors)
                else:
                    # Open the npz file
                    with open(path_points, 'rb') as f:
                        npzfile = np.load(f)
                        points = npzfile['points']
                        normals = npzfile['normals']
                        colors = npzfile['colors']

                # Save as npz
                with open(os.path.join(self.save_path, f"points_selected_{step_index}.npz"), 'wb') as f:
                    np.savez(f, points=points, normals=normals)
            else:
                load_points = continue_from <= 2 or debug or not os.path.exists(result_pkl_path)
                if load_points:
                    # Open the npz file
                    with open(os.path.join(self.save_path, f"points_selected_{step_index}.npz"), 'rb') as f:
                        npzfile = np.load(f)
                        points = npzfile['points']
                        normals = npzfile['normals']
                    print(f"Shape of points_originals_selected: {points.shape}")
                else:
                    points = None
                    normals = None

            # some debugging visualization of seperate pointcloud windings
            if debug:
                # get winding angles
                winding_angles = points[:, 3]

                min_wind = np.min(winding_angles)
                max_wind = np.max(winding_angles)
                print(f"Min and max winding angles: {min_wind}, {max_wind}")

                # remove test folder
                test_folder = os.path.join(self.save_path, "test_winding_angles")
                if os.path.exists(test_folder):
                    shutil.rmtree(test_folder)
                os.makedirs(test_folder)
                # test
                for test_angle in range(int(min_wind), int(max_wind), 360):
                    if (test_angle // 360 % 20) > 2:
                        continue
                    start_index, end_index = self.points_at_winding_angle(points, test_angle+180, max_angle_diff=180)
                    points_test = points[start_index:end_index]
                    colors_test = points_test[:,3]
                    points_test = points_test[:,:3]
                    normals_test = normals[start_index:end_index]
                    print(f"extracted {len(points_test)} points at {test_angle} degrees")
                    # save as ply
                    ordered_pointsets_test = [(points_test, normals_test)]
                    test_pointset_ply_path = os.path.join(test_folder, f"ordered_pointset_test_{test_angle}_{step_index}.ply")
                    try:
                        self.pointcloud_from_ordered_pointset(ordered_pointsets_test, test_pointset_ply_path, color=colors_test)
                    except:
                        print("Error saving test pointset")

                # if debug:
                #     # save complete pointcloud
                #     complete_pointset_ply_path = os.path.join(test_folder, "complete_pointset.ply")
                #     try:
                #         self.pointcloud_from_ordered_pointset([(points, normals)], complete_pointset_ply_path, color=points[:,3])
                #     except:
                #         print("Error saving complete pointset")

            print("Using Cpp rolled_ordered_pointset")
            # Set to false to load precomputed partial results during development
            fresh_start = continue_from <= 2
            if fresh_start or not os.path.exists(result_pkl_path):
                result = pointcloud_processing.create_ordered_pointset(points, normals, self.graph.umbilicus_data, angleStep=float(angle_step), z_spacing=int(z_spacing), max_eucledian_distance=10, min_z=z_start, max_z=z_end, min_wind=approx_min_angle, max_wind=approx_max_angle) # named parameters for mesh detail level: float angleStep, int z_spacing, float max_eucledian_distance, bool verbose
                # save result as pkl
                with open(result_pkl_path, 'wb') as f:
                    pickle.dump(result, f)
            else:
                with open(result_pkl_path, 'rb') as f:
                    result = pickle.load(f)

            if results is None:
                results = result
            else:
                # results shape: total num angle steps, "ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector", z steps, "individual shape"
                for i in range(len(result)): # angle steps
                    for j in range(len(result[i])): # type of data
                        if j == 3: # angle vector
                            continue
                        results[i][j].extend(result[i][j]) # extend the z steps
            
            # Get the size of the object in bytes
            size_in_bytes = sys.getsizeof(results)
            size_in_gbs = size_in_bytes / 1024 / 1024 / 1024
            print(f"Size of results: {size_in_gbs} GBs")
        
        return results

    def unroll(self, fragment=False, debug=False, continue_from=0, z_range=None, angle_step=1, z_spacing=10, learning_rate=0.2, iterations=11, unfix_factor=2.5, downsample=False):
        # Load the graph
        result = self.load_pointcloud_to_raw_ordered_pointset(debug=debug, continue_from=continue_from, z_range=z_range, angle_step=angle_step, z_spacing=z_spacing)

        # get nodes
        ordered_pointsets_s = self.rolled_ordered_pointset(result, angle_step=angle_step, continue_from=continue_from, fragment=fragment, learning_rate=learning_rate, iterations=iterations, unfix_factor=unfix_factor)

        pointset_ply_path = os.path.join(self.save_path, "ordered_pointset.ply")
        self.pointcloud_from_ordered_pointset(ordered_pointsets_s[0], pointset_ply_path)
        stamp = None
        for i in range(len(ordered_pointsets_s)):
            print(f"Computing Mesh piece {i+1} of {len(ordered_pointsets_s)}.")
            mesh_path = os.path.join(self.save_path, f"mesh_{i}.obj")
            ordered_pointsets = ordered_pointsets_s[i]
            if continue_from <= 5:
                mesh, uv_image = self.mesh_from_ordered_pointset(ordered_pointsets)
                self.save_mesh(mesh, uv_image, mesh_path)

            split_mesh_paths, stamp = self.split(mesh_path, split_width=self.split_width, fresh_start=(continue_from <= 6), stamp=stamp)

            # Flatten mesh
            args = [(self.save_path, split_mesh_path, downsample) for split_mesh_path in split_mesh_paths]
            # num_threads = min(max(1, multiprocessing.cpu_count() // 2), 5)
            # with multiprocessing.Pool(num_threads) as pool:
            #     tqdm(pool.imap(flatten_args, args), total=len(args), desc="Flattening meshes")
            num_procs = 12
            # for i in range(0, len(args), num_procs):
            #     flat_ps = []
            #     stop = min(i+num_procs, len(args))
            #     for arg in args[i:stop]:
            #         f_p = flatten_args(arg)
            #         flat_ps.append(f_p)
            #     for f_p in flat_ps:
            #         f_p.wait()
            #         print(f"Finished flattening mesh piece {f_p}.")
            with ProcessPoolExecutor(max_workers=num_procs) as executor:
                futures = {executor.submit(flatten_args, arg): arg for arg in args}
                for future in as_completed(futures):
                    arg = futures[future]
                    print(f"Finished flattening mesh piece {arg}.")
                print(f"Finished mesh piece {i+1} of {len(ordered_pointsets_s)}.")

if __name__ == '__main__':
    start_point = [3164, 3476, 3472]
    import argparse
    parser = argparse.ArgumentParser(description='Unroll a graph to a sheet')
    parser.add_argument('--path', type=str, help='Path to the instances', required=True)
    parser.add_argument('--graph', type=str, help='Path to the graph file from --Path', required=True)
    parser.add_argument('--fragment', action='store_true', help='Meshing Fragment, each layer as separate mesh')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--start_point', type=int, nargs=3, default=start_point, help='Start point for the unrolling')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for the mesh')
    parser.add_argument('--split_width', type=int, default=50000, help='Width for the mesh splitting')
    parser.add_argument('--continue_from', type=int, default=0, help='Continue from a specific processing step. 0: beginning, 1: build points, 2: rolled ordered pointset, 3: interpolate ordered pointset, 4: optimize pointset, 5: meshing, 6: splitting mesh, 7: flattening')
    parser.add_argument('--z_range', type=int, nargs=2, default=[-2147483648, 2147483647], help='Range of z values to mesh in')
    parser.add_argument('--angle_step', type=float, default=0.5, help='Angle step for the unrolling')
    parser.add_argument('--z_spacing', type=int, default=10, help='Angle step for the unrolling')
    parser.add_argument('--learning_rate', type=float, default=0.2, help='Learning rate for the optimization')
    parser.add_argument('--iterations', type=int, default=11, help='Number of iterations for the optimization')
    parser.add_argument('--unfix_factor', type=float, default=2.5, help='Unfix factor for the optimization. Higher = less unfixed points')
    parser.add_argument('--downsample', action='store_true', help='Downsample the mesh')

    args = parser.parse_args()

    graph_path = os.path.join(os.path.dirname(args.path), args.graph)
    if True or args.continue_from <= 3:
        graph = load_graph(graph_path)
        min_z = min([graph.nodes[node]["centroid"][1] for node in graph.nodes])
        max_z = max([graph.nodes[node]["centroid"][1] for node in graph.nodes])
        print(f"Min z: {min_z}, Max z: {max_z}")
    else:
        graph = None
    reference_path = graph_path.replace("evolved_graph", "subgraph")
    start_point = args.start_point
    scale_factor = args.scale_factor
    z_range = args.z_range
    # scale, coordinate transformatioin and clip z range. get it from original volume coordinate to instance patches coordinates
    z_range[0] = int((int(z_range[0] / scale_factor) + 500) // (200.0 / 50.0))
    z_range[1] = int((int(z_range[1] / scale_factor) + 500) // (200.0 / 50.0))
    if z_range[0] < -2147483648:
        z_range[0] = -2147483648
    if z_range[1] > 2147483647:
        z_range[1] = 2147483647
    
    walk = WalkToSheet(graph, args.path, start_point, scale_factor, split_width=args.split_width)
    # walk.save_graph_pointcloud(reference_path)
    walk.unroll(fragment=args.fragment, debug=args.debug, continue_from=args.continue_from, z_range=z_range, angle_step=args.angle_step, z_spacing=args.z_spacing, learning_rate=args.learning_rate, iterations=args.iterations, unfix_factor=args.unfix_factor, downsample=args.downsample)

# Example command: python3 -m ThaumatoAnakalyptor.graph_to_mesh --path /scroll.volpkg/working/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks --graph /scroll.volpkg/working/scroll3_surface_points/1352_3600_5002/point_cloud_colorized_verso_subvolume_graph_BP_solved.pkl --start_point 1352 3600 5002 --debug
# python3 -m ThaumatoAnakalyptor.graph_to_mesh --path /scroll2v2_surface_points/point_cloud_colorized_verso_subvolume_blocks --graph /scroll2v2_surface_points/1352_3600_5002/point_cloud_colorized_verso_subvolume_graph_BP_solved.pkl --start_point 1352 3600 5002
# python3 -m ThaumatoAnakalyptor.graph_to_mesh --path /scroll_pcs/point_cloud_colorized_verso_subvolume_blocks --graph /scroll_pcs/1352_3600_5007/point_cloud_colorized_verso_subvolume_graph_BP_solved.pkl --start_point 1352 3600 5007 --downsample --angle_step 1.0 --z_spacing 5 --start_point 1352 3600 5007