### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import pickle
import os
import open3d as o3d
import json

# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import glob
import tarfile
import tempfile
from .surface_fitting_utilities import get_vector_mean, rotation_matrix_to_align_z_with_v, fit_surface_to_points_n_regularized, distance_from_surface, distance_from_surface_clipped

import warnings
warnings.filterwarnings("ignore")

def load_ply(filename):
    """
    Load point cloud data from a .ply file inside a tarball.
    """
    # Derive tar filename from the given filename
    base_dir = os.path.dirname(filename)
    parent_dir = os.path.dirname(base_dir)
    tar_filename = os.path.join(parent_dir, f"{os.path.basename(base_dir)}.tar")
    ply_filename_inside_tar = os.path.basename(filename)

    # Check that the tar file exists
    assert os.path.isfile(tar_filename), f"File {tar_filename} not found."

    # Extract the desired ply and metadata files from the tarball to a temporary location
    with tarfile.open(tar_filename, 'r') as tar:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct paths to the temporary extracted files
            temp_ply_path = os.path.join(temp_dir, ply_filename_inside_tar)
            base_filename_without_extension = os.path.splitext(ply_filename_inside_tar)[0]
            metadata_filename_inside_tar = f"metadata_{base_filename_without_extension}.json"
            temp_metadata_path = os.path.join(temp_dir, metadata_filename_inside_tar)
            
            tar.extract(ply_filename_inside_tar, path=temp_dir)
            tar.extract(metadata_filename_inside_tar, path=temp_dir)
            
            # Load the ply file
            pcd = o3d.io.read_point_cloud(temp_ply_path)
            
            # Load metadata
            with open(temp_metadata_path, 'r') as metafile:
                metadata = json.load(metafile)

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            # Extract additional features
            colors = np.asarray(pcd.colors)

    # Convert lists back to numpy arrays where needed
    if 'coeff' in metadata and metadata['coeff'] is not None:
        coeff = np.array(metadata['coeff'])
    else:
        coeff = None
    
    if 'n' in metadata and metadata['n'] is not None:
        n = int(metadata['n'])
    else:
        n = None

    score = metadata.get('score')
    distance = metadata.get('distance')

    return points, normals, colors, score, distance, coeff, n

def load_instance(path, sample_ratio=1.0):
    """
    Load instance from path.
    """
    points_list = []
    normals_list = []
    colors_list = []
    pred_score_list = []
    distance_list = []
    id_list = []

    # Deduce tar filename from given path
    tar_filename = f"{path}.tar"

    # Check that the tar file exists
    assert os.path.isfile(tar_filename), f"File {tar_filename} not found."

    # Retrieve a list of all .ply files within the tarball
    with tarfile.open(tar_filename, 'r') as archive:
        ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]
        # Iterate and load all instances prediction .ply files
        for ply_file in ply_files:
            file = os.path.join(path, ply_file)
            res = load_ply(file)
            points = res[0]
            normals = res[1]
            colors = res[2]
            pred_score = res[3]
            distance = res[4]
            # Sample points from picked patch
            points, normals, colors, _ = select_points(
                points, normals, colors, colors, sample_ratio
            )
            if points.shape[0] < 10:
                continue
            points_list.append(points)
            normals_list.append(normals)
            colors_list.append(colors)
            pred_score_list.append(pred_score)
            distance_list.append(distance)
            id_list.append(tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])]))
    return points_list, normals_list, colors_list, pred_score_list, distance_list, id_list

def angle_to_180(angle):
    """
    Convert angle to range [-180, 180].
    """
    found_adjustables = True
    while(found_adjustables):
        mask_low = angle < -180
        mask_high = angle > 180

        angle[mask_low] = angle[mask_low] + 360
        angle[mask_high] = angle[mask_high] - 360

        found_adjustables = np.any(mask_low) or np.any(mask_high)
    return angle

def alpha_angles(normals):
    '''
    Calculates the turn angle of the sheet with respect to the y-axis. this indicates the orientation of the sheet. Alpha is used to position the point in the scroll. Orientation alpha' and winding number k correspond to alpha = 360*k + alpha'
    '''
    # Calculate angles in radians
    theta_rad = np.arctan2(normals[:, 2], normals[:, 0]) # Y axis is up

    # Convert radians to degrees
    theta_deg = np.degrees(theta_rad)

    theta_deg = angle_to_180(theta_deg)
        
    return theta_deg

def adjust_angles_zero(angles, offset):
    '''
    When combinding two segments -> adjusting the angles of the second segment to the first segment.
    '''
    # Angles is not a numpy array
    added_dimension = False
    if not isinstance(angles, np.ndarray):
        angles = np.array([angles])
        added_dimension = True
    # Offset angles by alpha_p - alpha_p_old
    offset_k = int(offset / 360)
    offset_r = offset - offset_k * 360
    angles_k = (angles / 360).astype(int)
    angles_r = angles - angles_k * 360
    angles = angles_r + offset_r
    angles_mask_high = angles > 180
    angles_mask_low = angles < -180
    angles[angles_mask_high] = angles[angles_mask_high] - 360
    angles[angles_mask_low] = angles[angles_mask_low] + 360
    angles = angles + offset_k * 360 + angles_k * 360

    if added_dimension:
        angles = angles[0]
    
    return angles

def adjust_angles_offset(angles, offset):
    '''
    When combinding two segments -> adjusting the angles of the second segment to the first segment.
    '''
    # Angles is not a numpy array
    added_dimension = False
    if not isinstance(angles, np.ndarray):
        angles = np.array([angles])
        added_dimension = True

    angles = angles + offset

    if added_dimension:
        angles = angles[0]
    
    return angles

def compute_angle_range(angles, angle_tolerance):
    if len(angles) == 0:
        return [(0, 0)]
    
    # Compute lower and upper bounds of the ranges
    lower_bounds = angles - angle_tolerance
    upper_bounds = angles + angle_tolerance

    # Zip the bounds and sort them
    ranges = sorted(zip(lower_bounds, upper_bounds))

    # Merge overlapping or adjacent ranges
    merged_ranges = [ranges[0]]
    for current_start, current_end in ranges[1:]:
        last_end = merged_ranges[-1][1]
        if current_start <= last_end:
            merged_ranges[-1] = (merged_ranges[-1][0], max(last_end, current_end))
        else:
            merged_ranges.append((current_start, current_end))
    return merged_ranges

def largest_group_offset(offsets, epsilon=1e-5):
    assert len(offsets) > 0, "Offsets is empty."
    # Sort the offsets
    sorted_offsets = np.sort(offsets)

    # Compute differences between consecutive sorted offsets
    diffs = np.diff(sorted_offsets)

    # Identify where the difference exceeds the epsilon indicating a new group
    group_indices = np.where(diffs > epsilon)[0] + 1
    group_starts = np.insert(group_indices, 0, 0)
    group_ends = np.append(group_indices, len(sorted_offsets))

    # Calculate group sizes and identify the largest group
    group_sizes = group_ends - group_starts
    largest_group_index = np.argmax(group_sizes)
    largest_group = sorted_offsets[group_starts[largest_group_index]:group_ends[largest_group_index]]

    # Return the mean offset value of the largest group
    assert len(largest_group) > 0, "Largest group is empty."
    return np.mean(largest_group)

def select_tiles_points(patches_list, i, tiles):   
    patch1 = patches_list[i]

    all_points_tiles = None
    for tile in tiles:
        if tile in patch1["tiles_points"]: 
            points = patch1["tiles_points"][tile]
            # Construct all overlap points
            if all_points_tiles is None:
                all_points_tiles = points
            else:
                all_points_tiles = np.concatenate([all_points_tiles, points], axis=0)
    
    return all_points_tiles

def select_overlapping_angles(patches_list, i, j, tiles):   
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    all_angles_overlap1 = None
    all_angles_overlap2 = None
    for tile in tiles:

        if tile in patch1["tiles_points"]: 
            # Tile mask overlap
            mask1 = patch1["tiles_overlap_mask"][tile][j]
            # Angles of tile points
            angles_tile1 = patch1["tiles_angles"][tile]
            # Extract overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the overlapping points
            overlap_angles_tile1 = angles_tile1[mask1]
            # Construct all angles. overlap
            if all_angles_overlap1 is None:
                all_angles_overlap1 = overlap_angles_tile1
            else:
                all_angles_overlap1 = np.concatenate([all_angles_overlap1, overlap_angles_tile1], axis=0)

        if tile in patch2["tiles_points"]:
            # Tile mask overlap
            mask2 = patch2["tiles_overlap_mask"][tile][i]
            # Angles of tile points
            angles_tile2 = patch2["tiles_angles"][tile]
            # Extract overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the overlapping points
            overlap_angles_tile2 = angles_tile2[mask2]
            # Construct all angles. overlap
            if all_angles_overlap2 is None:
                all_angles_overlap2 = overlap_angles_tile2
            else:
                all_angles_overlap2 = np.concatenate([all_angles_overlap2, overlap_angles_tile2], axis=0)
    
    return all_angles_overlap1, all_angles_overlap2

def select_non_overlapping_angles(patches_list, i, j, tiles):
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    all_angles_non_overlap1 = None
    all_angles_non_overlap2 = None
    for tile in tiles:

        if tile in patch1["tiles_points"]:
            # Tile mask overlap
            mask1 = patch1["tiles_overlap_mask"][tile][j]
            # Non-overlapping mask
            mask1 = np.logical_not(mask1)
            # Angles of tile points
            angles_tile1 = patch1["tiles_angles"][tile]
            # Extract non-overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the non-overlapping points
            non_overlap_angles_tile1 = angles_tile1[mask1]
            # Construct all angles. non-overlap
            if all_angles_non_overlap1 is None:
                all_angles_non_overlap1 = non_overlap_angles_tile1
            else:
                all_angles_non_overlap1 = np.concatenate([all_angles_non_overlap1, non_overlap_angles_tile1], axis=0)

        if tile in patch2["tiles_points"]:
            # Tile mask overlap
            mask2 = patch2["tiles_overlap_mask"][tile][i]
            # Non-overlapping mask
            mask2 = np.logical_not(mask2)
            # Angles of tile points
            angles_tile2 = patch2["tiles_angles"][tile]
            # Extract non-overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the non-overlapping points
            non_overlap_angles_tile2 = angles_tile2[mask2]
            # Construct all angles. non-overlap
            if all_angles_non_overlap2 is None:
                all_angles_non_overlap2 = non_overlap_angles_tile2
            else:
                all_angles_non_overlap2 = np.concatenate([all_angles_non_overlap2, non_overlap_angles_tile2], axis=0)

    return all_angles_non_overlap1, all_angles_non_overlap2

def filter_by_angle(angles2, angle_ranges):
    # Check which angles in angles2 are within the merged ranges
    masks = [np.logical_and(angles2 >= start, angles2 <= end) for start, end in angle_ranges]
    mask = np.any(masks, axis=0)

    # Return filtered points
    return mask

def load_ply_folder(ply_file_path):
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

def build_patch_folder(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    res = load_ply_folder(path)
    patch_points = res[0]
    patch_normals = res[1]
    patch_color = res[2]
    patch_score = res[3]
    patch_distance = res[4]
    patch_coeff = res[5]
    n = res[6]

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    patch_id = tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])])

    x, y, z, id_ = patch_id
    anchor_normal = get_vector_mean(patch_normals)
    anchor_angle = alpha_angles(np.array([anchor_normal]))[0]

    additional_main_patch = {"ids": [patch_id],
                    "points": patch_points,
                    "normals": patch_normals,
                    "colors": patch_color,
                    "anchor_points": [patch_points[0]], 
                    "anchor_normals": [anchor_normal],
                    "anchor_angles": [anchor_angle],
                    "angles": adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_normals), - anchor_angle), offset_angle),
                    "subvolume": [(x, y, z)],
                    "subvolume_size": [subvolume_size],
                    "iteration": 0,
                    "patch_prediction_scores": [patch_score],
                    "patch_prediction_distances": [patch_distance],
                    "patch_prediction_coeff": [patch_coeff],
                    "n": [n],
                    }
    
    return additional_main_patch, offset_angle

def subvolume_surface_patches_folder(file, subvolume_size=50, sample_ratio=1.0):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    patches_list = []
    tar_filename = f"{file}.tar"

    if os.path.isfile(tar_filename):
        with tarfile.open(tar_filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
            # Extract all .ply files at once
            ply_files = [m for m in archive.getmembers() if m.name.endswith(".ply")]
            archive.extractall(path=temp_dir, members=archive.getmembers())

            # Process each .ply file
            for ply_member in ply_files:
                ply_file_path = os.path.join(temp_dir, ply_member.name)
                ply_file = ply_member.name
                ids = tuple([*map(int, tar_filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
                ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                surface_dict, _ = build_patch_folder(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
                patches_list.append(surface_dict)

    return patches_list

def subvolume_surface_patches(path, subvolume_size=50):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Size is int
    if isinstance(subvolume_size, int):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    patches_list = []
    # Iterate over all subvolumes. glob folders in path
    for file in tqdm(glob.glob(path + "/*/")):
        surfaces, surfaces_normals, surfaces_colors, ids = load_instance(file)
        x, y, z, id_ = ids[0]
        for i in range(len(surfaces)):
            surface_dict = {"ids": [ids[i]], 
                            "points": surfaces[i],
                            "normals": surfaces_normals[i],
                            "colors": surfaces_colors[i], 
                            "anchor_points": [surfaces[i][0]], 
                            "anchor_normals": [surfaces_normals[i][0]],
                            "anchor_angles": [alpha_angles(np.array([surfaces_normals[i][0]]))[0]],
                            "angles": alpha_angles(surfaces_normals[i]),
                            "subvolume": [np.array([x,y,z])], 
                            "subvolume_size": [subvolume_size],
                            "iteration": 0,}
            # Add the surface dict to the list
            patches_list.append(surface_dict)

    return patches_list

def save_main_sheet(main_sheet, volume_blocks_scores, path="main_sheet.ta"):
    """
    Save the main sheet to a file.
    """
    save_sheet = {}
    # remove points for display
    for volume_id in main_sheet:
        save_sheet[volume_id] = {}
        for patch_id in main_sheet[volume_id]:
            save_sheet[volume_id][patch_id] = {}
            save_sheet[volume_id][patch_id]["offset_angle"] = main_sheet[volume_id][patch_id]["offset_angle"]

    sheet_segmentation = {'main_sheet': save_sheet, 'volume_blocks_scores': volume_blocks_scores}

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(sheet_segmentation, f)

def load_main_sheet(subvolume_size=50, path="", path_ta="main_sheet.ta", sample_ratio_score=0.1, sample_ratio=0.1, add_display_points=True):
    """
    Load the main sheet from a file.
    """
    with open(path_ta, 'rb') as f:
        sheet_segmentation = pickle.load(f)

    main_sheet = sheet_segmentation['main_sheet']
    volume_blocks_scores = sheet_segmentation['volume_blocks_scores']

    # Add points and angles of main patches to main sheet
    if add_display_points:
        main_sheet_add_display_points(main_sheet, subvolume_size, path, sample_ratio)

    return main_sheet, volume_blocks_scores

def select_points(points, normals, colors, angles, original_ratio):
    # indices where colors[1] < original_ratio
    indices = np.where(colors[:, 1] < original_ratio)

    # at least one point:
    if len(indices[0]) == 0:
        indices = np.array([0])
    
    points_ = points[indices]
    normals_ = normals[indices]
    colors_ = colors[indices]
    angles_ = angles[indices]
    
    return points_, normals_, colors_, angles_

def add_overlapp_entries_to_patches_list(patches_list):
    for i in range(len(patches_list)):
        patches_list[i]["overlapp_percentage"] = np.zeros(len(patches_list))
        patches_list[i]["overlap"] = np.zeros(len(patches_list))
        patches_list[i]["non_overlap"] = np.zeros(len(patches_list))
        patches_list[i]["points_overlap"] = [None] * len(patches_list)
        patches_list[i]["scores"] = np.zeros(len(patches_list)) - 1

def overlapping_tiles(patch1, patch2):
    indices1 = patch1["tiles_points"].keys()
    indices2 = patch2["tiles_points"].keys()

    overlapping_indices = []
    for index1 in indices1:
        if patch1["tiles_points"][index1].shape[0] <= 10:
            continue
        if index1 in indices2:
            if patch2["tiles_points"][index1].shape[0] <= 10:
                continue
            overlapping_indices.append(index1)
    assert len(overlapping_indices) <= 4, f"More than 4 overlapping tiles: {len(overlapping_indices)}"

    if len(overlapping_indices) == 0:
        return overlapping_indices

    return overlapping_indices

def overlap_mask(patch1, patch2, index):
    # Concatenate points from both point clouds
    if index in patch1["tiles_points"] and index in patch2["tiles_points"]: # both tiles have points
        mask_shape = patch1["tiles_points"][index].shape[0]
        all_points = np.vstack([patch1["tiles_points"][index], patch2["tiles_points"][index]])
    elif index in patch1["tiles_points"]: # only patch1 has points
        mask_shape = patch1["tiles_points"][index].shape[0]
        all_points = patch1["tiles_points"][index]
    elif index in patch2["tiles_points"]: # only patch2 has points
        mask_shape = 0
        all_points = patch2["tiles_points"][index]
    else: # neither patch has points
        return np.zeros(0), np.zeros(0)
    # Find the unique points and their counts
    unique_points, indices, counts = np.unique(all_points, axis=0, return_counts=True, return_index=True)
    # Extract overlapping points from both point clouds
    mask = np.ones(all_points.shape[0], dtype=bool)
    single_count_indices = indices[counts == 1]
    mask[single_count_indices] = False

    # Split the mask into two masks for the two point clouds
    mask1 = mask[:mask_shape]
    mask2 = mask[mask_shape:]

    return mask1, mask2

def compute_overlap_for_pair(args):
    i, patches_list, epsilon, angle_tolerance = args
    patch_i = patches_list[i]
    points_i = patch_i["points"]
    result_overlapps = []
    for j in range(i+1, len(patches_list)):
        # extract the unique points and the nr of overlapping points
        patch_j = patches_list[j]
        points_j = patch_j["points"]
        points_ij = np.vstack([points_i, points_j])
        unique_points_size = np.unique(points_ij, axis=0).shape[0]
        total_points_size = points_ij.shape[0]
        overlapping_points_size = total_points_size - unique_points_size        

        # Compute the surface overlap between the two patches
        overlapp_percentage = 1.0 * overlapping_points_size / (unique_points_size + 0.00001)
        overlap = overlapping_points_size
        non_overlap = unique_points_size - overlapping_points_size
        points_overlap = None
        angles_offset = 0.0

        result_overlapps.append((i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset))
    return result_overlapps

def compute_overlap_for_pair_old(args):
    i, patches_list, epsilon, angle_tolerance = args
    result_overlapps = []
    for j in range(i+1, len(patches_list)):
        # Extract indexes of overlapping tiles
        overlap_tiles = overlapping_tiles(patches_list[i], patches_list[j])

        # Calculate overlap mask for each tile both for i and j perspective
        for tile in overlap_tiles:
            overlap_mask_i, overlap_mask_j = overlap_mask(patches_list[i], patches_list[j], tile)
            # print(f"length of tiles overlap mask of tile {tile} for patch {i} is {len(overlap_mask_i)}")
            patches_list[i]["tiles_overlap_mask"][tile][j] = overlap_mask_i
            patches_list[j]["tiles_overlap_mask"][tile][i] = overlap_mask_j        

        # Continue if there are no points in the overlapping tiles
        if len(overlap_tiles) == 0:
            overlapp_percentage = 0.0
            overlap = 0
            non_overlap = 0
            points_overlap = []
            angles_offset = None
        else:
            # Compute the surface overlap between the two patches
            overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset = surface_overlapp_tiles_complete(patches_list, i, j, overlap_tiles)

        result_overlapps.append((i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset))
    return result_overlapps

def filter_tiles(overlap_angles, non_overlap_agles, tiles, patch):
    overlapp_count = 0
    non_overlapp_count = 0
    for tile in tiles:
        if tile in patch["tiles_points"]: 
            # Angles of tile points
            angles_tile1 = patch["tiles_angles"][tile]
            # Check how many angles of the tile are in the overlap_angles
            angles_overlap1 = np.isin(overlap_angles, angles_tile1)
            # Check how many angles of the tile are in the non_overlap_agles
            angles_non_overlap1 = np.isin(non_overlap_agles, angles_tile1)
            if np.sum(angles_overlap1) > 0:
                overlapp_count += np.sum(angles_overlap1)
                non_overlapp_count += np.sum(angles_non_overlap1)
    
    return overlapp_count, non_overlapp_count

def surface_overlapp_tiles(patches_list, i, j, overlap_tiles, angle_tolerance=90.0, epsilon=1e-5):
    """
    Calculate the surface overlap between two point clouds.
    Points have to be exactly on the same position to be considered the same.
    """
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    # Find the angles of the overlapping points
    overlap1_angles, overlap2_angles = select_overlapping_angles(patches_list, i, j, overlap_tiles)
    # Find the non-overlapping angles
    angles1_non_overlap, angles2_non_overlap = select_non_overlapping_angles(patches_list, i, j, overlap_tiles)

    # Return early if there are no overlapping points
    if len(overlap1_angles) == 0 and len(overlap2_angles) == 0:
        return 0.0, 0, 0, None, None

    assert not ((overlap1_angles is None) or (overlap2_angles is None) or (len(overlap1_angles) == 0) or (len(overlap2_angles) == 0)), f"angles for tiles {overlap_tiles} are wack: {overlap1_angles} {overlap2_angles}"
    # Calculate the angle offsets between the overlapping points
    overlap1_angle_offsets = overlap1_angles - overlap2_angles
    overlap2_angle_offsets = overlap2_angles - overlap1_angles

    # Find the offset with the highest number of occurences in the angle offsets
    # Prefilter angles for winding number (different offsets), select max overlap offset
    angles_offset1 = largest_group_offset(overlap1_angle_offsets, epsilon=epsilon)
    
    # Find winding number k with the most overlapping points
    angle1_ranges, angle2_ranges, mean_angle2_adjusted = find_patch_angle_range(angles_offset1, patch2["normals"], angle_tolerance=angle_tolerance)

    # # Ranges of angles in the first point cloud
    # angle1_ranges = compute_angle_range(overlap1_angles[filter1], angle_tolerance)
    # angle2_ranges = compute_angle_range(overlap2_angles[filter2], angle_tolerance)

    # Filter overlapping points in the first point cloud by angles within range
    filtered_overlap1_mask = filter_by_angle(overlap1_angles, angle1_ranges)
    # Filter overlapping points in the second point cloud by angles within range
    filtered_overlap2_mask = filter_by_angle(overlap2_angles, angle2_ranges)

    # Filter non-overlapping points in the first point cloud by angles within range of the second point cloud angles (+- angle_tolerance)
    filtered_non_overlap1_mask = filter_by_angle(angles1_non_overlap, angle1_ranges)

    # Filter non-overlapping points in the second point cloud by angles within range of the first point cloud angles (+- angle_tolerance)
    filtered_non_overlap2_mask = filter_by_angle(angles2_non_overlap, angle2_ranges)

    overlap1, non_overlap1 = filter_tiles(overlap1_angles[filtered_overlap1_mask], angles1_non_overlap[filtered_non_overlap1_mask], overlap_tiles, patch1)
    overlap2, non_overlap2 = filter_tiles(overlap2_angles[filtered_overlap2_mask], angles2_non_overlap[filtered_non_overlap2_mask], overlap_tiles, patch2)
    
    # Calculate the overlap percentage
    overlapp_percentage1 = 1.0 * overlap1 / (overlap1 + non_overlap1 + 0.00001)
    overlapp_percentage2 = 1.0 * overlap2 / (overlap2 + non_overlap2 + 0.00001)

    # Take the larger overlap percentage
    if overlapp_percentage1 < overlapp_percentage2:
        overlap = overlap1
        overlapp_percentage = overlapp_percentage1
        non_overlap = non_overlap1
    else:
        overlap = overlap2
        overlapp_percentage = overlapp_percentage2
        non_overlap = non_overlap2
    
    unique_points = None # Not needed anymore

    return overlapp_percentage, overlap, non_overlap, unique_points, mean_angle2_adjusted

def surface_overlapp_tiles_complete(patches_list, i, j, overlap_tiles):
    """
    Calculate the surface overlap between two point clouds.
    Points have to be exactly on the same position to be considered the same.
    """
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    count_overlap1 = 0
    count_overlap2 = 0
    count_non_overlap1 = 0
    count_non_overlap2 = 0
    for tile in overlap_tiles:
        if tile in patch1["tiles_points"]: 
            # Tile mask overlap
            mask1 = patch1["tiles_overlap_mask"][tile][j]
            mask_non_overlap1 = np.logical_not(mask1)
            # Sum of overlapping points
            count_overlap1 += np.sum(mask1)
            count_non_overlap1 += np.sum(mask_non_overlap1)

        if tile in patch2["tiles_points"]:
            # Tile mask overlap
            mask2 = patch2["tiles_overlap_mask"][tile][i]
            mask_non_overlap2 = np.logical_not(mask2)
            # Sum of overlapping points
            count_overlap2 += np.sum(mask2)
            count_non_overlap2 += np.sum(mask_non_overlap2)

    assert count_overlap1 == count_overlap2, f"Overlap counts are not equal: {count_overlap1} {count_overlap2}"
    
    # # Calculate the overlap percentage
    overlapp_percentage1 = 1.0 * count_overlap1 / (count_overlap1 + count_non_overlap1 + 0.00001)
    overlapp_percentage2 = 1.0 * count_overlap2 / (count_overlap2 + count_non_overlap2 + 0.00001)

    # overlap = (count_overlap1+count_overlap2)/2
    # non_overlap = (count_non_overlap1+count_non_overlap2)/2
    # overlapp_percentage = (overlapp_percentage1+overlapp_percentage2)/2

    # Take the larger overlap percentage
    if overlapp_percentage1 < overlapp_percentage2:
        overlap = count_overlap1
        overlapp_percentage = overlapp_percentage1
        non_overlap = count_non_overlap1
    else:
        overlap = count_overlap2
        overlapp_percentage = overlapp_percentage2
        non_overlap = count_non_overlap2
    
    return overlapp_percentage, overlap, non_overlap, None, 0.0

def find_patch_angle_range(offset, patch_normals2, angle_tolerance=90):
    mean_angle2 = 0.0 # angles of patch2 already zeroed when building the patch
    filter_angle_range2 = (mean_angle2 - angle_tolerance, mean_angle2 + angle_tolerance)

    mean_angle2_adjusted = mean_angle2
    mean_angle2_adjusted += offset
    filter_angle_range1 = (mean_angle2_adjusted - angle_tolerance, mean_angle2_adjusted + angle_tolerance)
    
    return [filter_angle_range1], [filter_angle_range2], mean_angle2_adjusted

def overlapp_score(i, j, patches_list, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}, sample_ratio=0.1, min_points_factor=1.0):
    overlapp_percentage = patches_list[i]["overlapp_percentage"][j]
    overlap = patches_list[i]["overlap"][j] / (min_points_factor * sample_ratio) # Adjust for calculation based on total points

    # nrp = min(max(overlapp_threshold["nr_points_min"], overlap), overlapp_threshold["nr_points_max"]) / (overlapp_threshold["nr_points_min"])
    # nrp_factor = np.log(np.log(nrp) + 1.0) + 1.0

    nrp = (min(max(2.0 * overlapp_threshold["nr_points_min"], overlap), 2.0 * overlapp_threshold["nr_points_max"]) - 2.0 * overlapp_threshold["nr_points_min"]) / (2.0 * overlapp_threshold["nr_points_max"] - 2.0 * overlapp_threshold["nr_points_min"])
    nrp_factor = nrp + 1.0

    non_overlap_percentage = 1.0 - overlapp_percentage
    score = 1.0 - (min_points_factor**2.0) * non_overlap_percentage
    # score = overlapp_percentage

    if score >= overlapp_threshold["score_threshold"]:
        score = ((score - overlapp_threshold["score_threshold"])/ (1 - overlapp_threshold["score_threshold"])) ** 2
        return score * nrp_factor
    else:
        return -1.0

def assign_points_to_tiles(patches_list, subvolume, tiling=2):
    start = subvolume['start']
    end = subvolume['end']

    # Calculate tile size and position
    tiles_per_axis = 2 * tiling - 1
    tiles_per_axis = 1 + tiling
    tile_size = (end - start) / tiles_per_axis

    # Assign points to tiles
    for patch in patches_list:
        # Calculate tile indices
        tile_indices_raw = (patch["points"] - start) / tile_size
        tile_indices = np.floor(tile_indices_raw).astype(int)
        # assert all positive indices
        assert np.all(tile_indices >= 0), f"Tile indices should be positive, but is {tile_indices}"
        # find indices that are larger than tiles_per_axis
        mask_offending = np.any(tile_indices >= tiles_per_axis, axis=1)
        assert np.all(tile_indices_raw <= tiles_per_axis), f"Tile indices should be less than tiles_per_axis, but is {tile_indices_raw[mask_offending][0]}"
        # for each unique index, add the points to the corresponding tile
        unique_indices = np.unique(tile_indices, axis=0)
        assert unique_indices.shape[1] == 3, f"Unique indices should have 3 columns, but has {unique_indices.shape[1]}"
        patch["tiles"] = {}
        patch["tiles_overlap_mask"] = {}
        for tile_x in range(tiles_per_axis):
            for tile_y in range(tiles_per_axis):
                for tile_z in range(tiles_per_axis):
                    index = (tile_x, tile_y, tile_z)
                    patch["tiles_overlap_mask"][index] = np.zeros((len(patches_list), 0), dtype=bool)
        patch["tiles_points"] = {}
        patch["tiles_colors"] = {}
        patch["tiles_normals"] = {}
        patch["tiles_normals_mean"] = {}
        patch["tiles_angles"] = {}
        indices_full = []
        for index in unique_indices:
            index = tuple(index)
            indices_full.append(index)
            indices = np.all(tile_indices == index, axis=1)
            patch["tiles"][index] = True
            patch["tiles_overlap_mask"][index] = np.zeros((len(patches_list), patch["points"][indices].shape[0]), dtype=bool)

            patch["tiles_points"][index] = patch["points"][indices]
            patch["tiles_colors"][index] = patch["colors"][indices]
            patch["tiles_normals"][index] = patch["normals"][indices]
            patch["tiles_angles"][index] = patch["angles"][indices]

            # sort the points in the tile by their lexicographical order
            sort_indices1 = np.lexsort(patch["tiles_points"][index].T)
            patch["tiles_points"][index] = patch["tiles_points"][index][sort_indices1]
            patch["tiles_colors"][index] = patch["tiles_colors"][index][sort_indices1]
            patch["tiles_normals"][index] = patch["tiles_normals"][index][sort_indices1]
            patch["tiles_angles"][index] = patch["tiles_angles"][index][sort_indices1]
            patch["tiles_normals_mean"][index] = np.mean(patch["tiles_normals"][index], axis=0)

def surrounding_volumes_main_sheet(volume_id, volume_size=50, window_size=4):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*((2*window_size+1)**3)
    count = 0
    for x in range(-window_size, window_size+1):
        vs_x = vs * x
        vi_x = volume_id_x + vs_x
        for y in range(-window_size, window_size+1):
            vs_y = vs * y
            vi_y = volume_id_y + vs_y
            for z in range(-window_size, window_size+1):
                vs_z = vs * z
                vi_z = volume_id_z + vs_z
                surrounding_volumes[count] = (vi_x, vi_y, vi_z)
                count += 1
    return surrounding_volumes

def surrounding_volumes(volume_id, volume_size=50):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*(3**3)
    count = 0
    for x in range(-1, 2):
        vs_x = vs * x
        vi_x = volume_id_x + vs_x
        for y in range(-1, 2):
            vs_y = vs * y
            vi_y = volume_id_y + vs_y
            for z in range(-1, 2):
                vs_z = vs * z
                vi_z = volume_id_z + vs_z
                surrounding_volumes[count] = (vi_x, vi_y, vi_z)
                count += 1
    return surrounding_volumes

def fit_sheet(patches_list, i, j, percentile, epsilon, angle_tolerance):
    # fit sheet surface to main sheet + other patch and determine how well the other patch fits into the main sheet
    main_sheet = patches_list[i]
    other_patch = patches_list[j]

    tiles = overlapping_tiles(main_sheet, other_patch)
    if len(tiles) == 0:
        return float("inf"), None
    # Find the angles of the overlapping points
    angles_i, angles_j = select_overlapping_angles(patches_list, i, j, tiles)
    points_j = select_tiles_points(patches_list, j, tiles)
    # Find the angles offsets
    angles_offset = angles_i - angles_j
    offset = largest_group_offset(angles_offset, epsilon=epsilon)

    # mean_angle_j is already 0 from build_patch
    mean_angle_j = 0.0
    mean_angle_j += offset
    filter_angle_range = (mean_angle_j - angle_tolerance, mean_angle_j + angle_tolerance)
    main_angles = main_sheet["angles"]
    main_sheet_mask = filter_by_angle(main_angles, [filter_angle_range])

    points = np.concatenate([main_sheet["points"][main_sheet_mask], other_patch["points"]], axis=0)
    points = np.unique(points, axis=0)
    normals = np.concatenate([main_sheet["normals"][main_sheet_mask], other_patch["normals"]], axis=0)
    normal_vector = get_vector_mean(normals)
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Sheet fitting params
    n = 4
    alpha = 1000.0
    slope_alpha = 0.1

    # Fit other patch and main sheet together
    coeff_all = fit_surface_to_points_n_regularized(points, R, n, alpha=alpha, slope_alpha=slope_alpha)
    surface = (normal_vector, R, n, coeff_all)

    n_patch = 4 # TODO: make this a parameter loaded from patch
    distances_points_other = distance_from_surface(other_patch["points"], R, n, coeff_all)
    distance_percentile = np.percentile(distances_points_other, percentile)
    distance_other = np.mean(distances_points_other)

    distances_points_ = distance_from_surface(points_j, R, n_patch, coeff_all)
    if len(distances_points_) > 0:
        mean_distance_ = np.mean(distances_points_)
    else:
        mean_distance_ = 0.0

    # Fit main sheet if containing points
    if np.sum(main_sheet_mask) < 1:
        distance_main_sheet = 100.0
        distance_main_sheet_dif = 100.0
    else:
        coeff_main_sheet = fit_surface_to_points_n_regularized(main_sheet["points"][main_sheet_mask], R, n, alpha=alpha, slope_alpha=slope_alpha)
        distances_points_main_sheet = distance_from_surface(main_sheet["points"][main_sheet_mask], R, n, coeff_main_sheet)
        distance_main_sheet = np.sum(distances_points_main_sheet)

        distances_points_main_sheet_all = distance_from_surface(main_sheet["points"][main_sheet_mask], R, n, coeff_all)
        distance_main_sheet_all = np.sum(distances_points_main_sheet_all)

        distance_main_sheet_dif = (distance_main_sheet_all - distance_main_sheet) / main_sheet["points"][main_sheet_mask].shape[0]
        distance_main_sheet_dif = max(0.0, distance_main_sheet_dif)

    # Compute the distance between the two surfaces
    distance = distance_other + distance_main_sheet_dif

    return distance, distance_percentile, mean_distance_, surface

def main_sheet_add_display_points(main_sheet, subvolume_size, path, sample_ratio):
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            main_sheet_patch = (volume_id, patch_id, main_sheet[volume_id][patch_id]["offset_angle"])
            # Get the patch
            patch, offset_angle = build_patch(main_sheet_patch, subvolume_size, path)
            # Add display points

            # Sample points from picked patch
            subsampled_points_picked_patch = select_points(
                patch['points'], patch['normals'], patch['colors'], patch['angles'], sample_ratio
            )

            # Add display points
            main_sheet[volume_id][patch_id]["displaying_points"] = subsampled_points_picked_patch

def build_patch(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    res = load_ply(file)
    patch_points = res[0]
    patch_normals = res[1]
    patch_color = res[2]
    patch_score = res[3]
    patch_distance = res[4]
    patch_coeff = res[5]
    n = res[6]

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    patch_id = tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])])

    x, y, z, id_ = patch_id
    anchor_normal = get_vector_mean(patch_normals)
    anchor_angle = alpha_angles(np.array([anchor_normal]))[0]

    additional_main_patch = {"ids": [patch_id],
                    "points": patch_points,
                    "normals": patch_normals,
                    "colors": patch_color,
                    "anchor_points": [patch_points[0]], 
                    "anchor_normals": [anchor_normal],
                    "anchor_angles": [anchor_angle],
                    "angles": adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_normals), - anchor_angle), offset_angle),
                    "subvolume": [(x, y, z)],
                    "subvolume_size": [subvolume_size],
                    "iteration": 0,
                    "patch_prediction_scores": [patch_score],
                    "patch_prediction_distances": [patch_distance],
                    "patch_prediction_coeff": [patch_coeff],
                    "n": [n],
                    }
    
    return additional_main_patch, offset_angle

def make_unique(patch):
    # unique points
    unique_points, unique_indices = np.unique(patch["points"], axis=0, return_index=True)
    patch["points"] = patch["points"][unique_indices]
    patch["normals"] = patch["normals"][unique_indices]
    patch["colors"] = patch["colors"][unique_indices]
    patch["angles"] = patch["angles"][unique_indices]

    return patch

def add_patch(patch1, patch2, offset_angle):
    patch1["ids"] += patch2["ids"]
    patch1["points"] = np.concatenate([patch1["points"], patch2["points"]], axis=0)
    patch1["normals"] = np.concatenate([patch1["normals"], patch2["normals"]], axis=0)
    patch1["colors"] = np.concatenate([patch1["colors"], patch2["colors"]], axis=0)
    patch1["anchor_points"] += patch2["anchor_points"]
    patch1["anchor_normals"] += patch2["anchor_normals"]
    patch1["anchor_angles"] += patch2["anchor_angles"]
    patch1["angles"] =  np.concatenate([patch1["angles"], adjust_angles_offset(patch2["angles"], offset_angle)], axis=0)
    patch1["subvolume"] += patch2["subvolume"]
    patch1["subvolume_size"] += patch2["subvolume_size"]
    patch1["patch_prediction_scores"] += patch2["patch_prediction_scores"]
    patch1["patch_prediction_distances"] += patch2["patch_prediction_distances"]
    patch1["patch_prediction_coeff"] += patch2["patch_prediction_coeff"]
    patch1["n"] += patch2["n"]

    # unique points
    patch1 = make_unique(patch1)

    return patch1

def build_main_sheet_volume_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score):
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])


    main_sheet, _ = build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score)
    main_points = main_sheet["points"]
    # Coloring Red
    main_colors = np.zeros_like(main_points) + np.array([1, 0, 0])
    other_points = main_points.astype(np.float16)
    other_colors = main_colors
    other_points_list = []

    main_sheet_volumes = list(set([mid[:3] for mid in main_sheet["ids"]]))
    for volume_id in tqdm(main_sheet_volumes):
        path_surrounding = path + f"/{volume_id[0]:06}_{volume_id[1]:06}_{volume_id[2]:06}"
        patches_list = subvolume_surface_patches_folder(path_surrounding, subvolume_size)
        # Build main sheet
        for volume_sheet_patch in patches_list:
            if not volume_sheet_patch["ids"][0] in main_sheet["ids"]:
                other_point = volume_sheet_patch["points"]
                other_points_list.append(other_point)
    
    other_points = np.concatenate([other_points] + other_points_list, axis=0).astype(np.float16)
    other_colors = np.zeros_like(other_points) + 1
    unique_points, indices = np.unique(other_points, return_index=True, axis=0)
    unique_colors = other_colors[indices]
    return unique_points, unique_colors

def build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score, align_and_flip_normals=True):
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    main_sheet = None
    main_patches = []
    
    # Build main sheet
    for main_sheet_patch in main_sheet_patches_list:
        volume_start, surface_nr, offset_angle = main_sheet_patch
        volume_start = (int(volume_start[0]), int(volume_start[1]), int(volume_start[2]))

        main_sheet_patch = (volume_start, int(surface_nr), float(offset_angle))
        additional_main_patch, offset_angle = build_patch(main_sheet_patch, tuple(subvolume_size), path, float(sample_ratio_score), align_and_flip_normals=align_and_flip_normals)
        main_patches.append((additional_main_patch, offset_angle))

        if main_sheet is None:
            main_sheet = additional_main_patch
        else:            
            main_sheet = add_patch(main_sheet, additional_main_patch, 0.0)
    
    return main_sheet, main_patches

def build_main_sheet_patches_list(main_sheet_volume_ids, main_sheet):
    main_sheet_volume_patches = []
    for volume_id in main_sheet_volume_ids:
        if volume_id in main_sheet: # slow? -> no
            for patch_id in main_sheet[volume_id]:
                main_sheet_volume_patches.append((volume_id, patch_id, main_sheet[volume_id][patch_id]["offset_angle"]))

    return main_sheet_volume_patches

def winding_switch_sheet_score_raw_precomputed_surface(patch_start, patch, overlapp_threshold):
    # Calculate distance between the two patches
    n = patch_start["n"][0]
    normals_start = patch_start['anchor_normals'][0]
    coeff_start = patch_start["patch_prediction_coeff"][0]
    normal_vector = normals_start
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Fit other patch and main sheet together, 10 to have reasonable range around max clip distance
    min_ = - 10*overlapp_threshold["max_sheet_clip_distance"]
    max_ = 10*overlapp_threshold["max_sheet_clip_distance"]
    points = patch["points"]
    distances_to_start_patch, patch_patch_start_direction_vector = distance_from_surface_clipped(points, R, n, coeff_start, min_, max_, return_direction=True)

    # Dot product of the normals of normal_vector and patch_patch_start_direction_vector
    dot_product = np.dot(normal_vector, patch_patch_start_direction_vector)
    # calculate patch winding direction offset (+1, -1). That determines in what direction this patch lies wrt the main sheet
    # get the sign of the dot product
    patch_winding_direction = np.sign(dot_product)
    k = patch_winding_direction
    
    score_raw = np.mean(distances_to_start_patch)

    return score_raw, k

def update_main_sheet(main_sheet, patches, offset_angles, points):
    for i in range(len(patches)):
        patch = patches[i]
        volume_id, patch_id, offset_angle, switched_winding = patch

        if volume_id not in main_sheet:
            main_sheet[volume_id] = {}
        if patch_id not in main_sheet[volume_id]:
            main_sheet[volume_id][patch_id] = {}
        main_sheet[volume_id][patch_id]["offset_angle"] = offset_angles[i]
        main_sheet[volume_id][patch_id]["switched_winding"] = switched_winding
        main_sheet[volume_id][patch_id]["displaying_points"] = points[i]

def build_main_sheet_points(main_sheet):
    points = []
    normals = []
    angles = []
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            offset_angle = main_sheet[volume_id][patch_id]["offset_angle"]
            displaying_points = main_sheet[volume_id][patch_id]["displaying_points"]
            points_, normals_, _, angles_ = displaying_points
            points.append(points_)
            normals.append(normals_)
            angles.append(angles_)
    
    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    angles = np.concatenate(angles, axis=0)

    return points, normals, angles

def display_main_sheet(main_sheet, nr_points_show=10000, show_normals=True):
    points, normals, angles = build_main_sheet_points(main_sheet)

    # Visualize with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # random subsample points
    print(f"Number of points in patch {0}: {points.shape[0]}")
    subsample = min(nr_points_show, points.shape[0])
    indices = np.random.choice(points.shape[0], subsample, replace=True)
    # map to 0-1
    point_max, point_min = np.max(angles), np.min(angles)
    k = np.ceil(np.max((angles)  - np.min(angles)) / 360)
    angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))

    # Min and max angles over all volume id, patch id:
    min_angle, max_angle = None, None
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            if min_angle is None or min_angle > main_sheet[volume_id][patch_id]["offset_angle"]:
                min_angle = main_sheet[volume_id][patch_id]["offset_angle"]
            if max_angle is None or max_angle < main_sheet[volume_id][patch_id]["offset_angle"]:
                max_angle = main_sheet[volume_id][patch_id]["offset_angle"]
    print(f"Windings around umbilicus {k}, min angle: {min_angle}, max angle: {max_angle} and on point basis: {point_max} and {point_min}")
    
    # map to color with colormap
    color = plt.cm.cool(angles)
    # subsample
    color = color[indices]
    ax.scatter(points[indices][:,0], points[indices][:,1], points[indices][:,2], c=color, marker='o', s=3)

    # Subsample again for a few normals
    points_subsample = points[indices]
    normals_subsample = normals[indices]

    indices_normals = np.random.choice(points_subsample.shape[0], 100, replace=True)
    points_subsample = points_subsample[indices_normals]
    normals_subsample = normals_subsample[indices_normals]
    color_subsample = color[indices_normals]
    del points, normals, color

    if show_normals:
        ax.quiver(points_subsample[:, 0], points_subsample[:, 1], points_subsample[:, 2], 
                normals_subsample[:, 0], normals_subsample[:, 1], normals_subsample[:, 2], color='g', length=100, normalize=True, arrow_length_ratio=0.1, label='Vector v')
        ax.scatter(points_subsample[:,0], points_subsample[:,1], points_subsample[:,2], c=color_subsample, marker='o', s=5)

    
    # same ratio for all axes
    ax.set_aspect('equal')

    plt.show()

    del ax
    del fig

def find_starting_patch(sheet_points, path, subvolume_size=50, min_points=1000):
    subvolume_size_half = subvolume_size // 2 # is 25
    # From True Scroll to working frame. spelufo offset, axis flips, scaling
    # Spelufo offset
    sheet_points = np.array(sheet_points) + 500
    # Axis flips
    axis_indices = [1, 2, 0]
    sheet_points = sheet_points[:, axis_indices]
    # Scaling
    sheet_points = sheet_points / (200.0 / 50.0)
    # Check that all axis min max dist is less then 50 withing steps of 0-50, 25-75, 50-100, ...
    rounded_down_25_min = np.floor(np.min(sheet_points, axis=0) / subvolume_size_half) * subvolume_size_half
    rounded_up_25_max = np.ceil(np.max(sheet_points, axis=0) / subvolume_size_half) * subvolume_size_half
    min_max_dist = rounded_up_25_max - rounded_down_25_min
    assert np.all(min_max_dist <= subvolume_size), f"Min max dist is {min_max_dist} for sheet points {sheet_points}. you should pick Points closer together for starting patch selection. Continueing, but this might mean the wrong starting patch is selected."
    # centroid
    centroid = np.mean(sheet_points, axis=0)
    # calculate volume of interest
    volume_start = np.array(rounded_down_25_min).astype(int)
    # load all patches in the volume of interest
    path_surrounding = path + f"/{volume_start[0]:06}_{volume_start[1]:06}_{volume_start[2]:06}"
    patches_list = subvolume_surface_patches_folder(path_surrounding, subvolume_size)
    print(f"Length of patches list: {len(patches_list)}")
    # min dist to rounded down centroid for each patch
    dists_vectors = [patch["points"] - centroid for patch in patches_list]
    min_dists = [np.min(np.linalg.norm(dists_vector, axis=1)) for dists_vector in dists_vectors if dists_vector.shape[0] > min_points]
    filtered_indices = [i for i in range(len(dists_vectors)) if dists_vectors[i].shape[0] > min_points]
    # get argmin of min dist from interest points to min patch point dist
    min_dist_arg_filtered = np.argmin(min_dists)
    min_dist_arg = filtered_indices[min_dist_arg_filtered]
    print(f"Minimum distance to centroid for closest valid patch is {min_dists[min_dist_arg_filtered]} and has {dists_vectors[min_dist_arg].shape[0]} points.")
    # return volume id and patch id
    volume_id = np.array(patches_list[min_dist_arg]["ids"][0][:3]).astype(int)
    patch_id = int(patches_list[min_dist_arg]["ids"][0][3])
    return volume_id, patch_id