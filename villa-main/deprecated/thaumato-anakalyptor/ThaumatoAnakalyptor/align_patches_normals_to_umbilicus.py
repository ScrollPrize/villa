### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
from .pointcloud_to_instances import save_block_ply
from .instances_to_sheets import load_ply
from .sheet_to_mesh import load_xyz_from_file, umbilicus, umbilicus_xz_at_y, scale_points

from multiprocessing import Pool, cpu_count
import os
from glob import glob
import tarfile

from tqdm import tqdm
from copy import deepcopy

def load_tar_block(file):
    """
    Load  patches and metadata from tar in volume
    input file: tar name + path (without .tar)
    """
    # Deduce folder filename from given path
    tar_filename = f"{file}.tar"
    # Check that the tar file exists
    assert os.path.isfile(tar_filename), f"File {tar_filename} not found."

    # base path of tar file
    path = os.path.dirname(tar_filename)

    files_content = []
    # Check that the tar file exists
    if os.path.isfile(tar_filename):
        # Retrieve a list of all .ply files within the tarball
        with tarfile.open(tar_filename, 'r') as archive:
            ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]
            # Iterate and load all instances prediction .ply files
            for ply_file_ in ply_files:
                ply_file = os.path.join(file, ply_file_)

                ids = tuple([*map(int, ply_file.split("/")[-2].split("_"))]+[int(ply_file.split("/")[-1].split(".")[-2].split("_")[-1])])
                ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                # Load ply
                ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
                file_ = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
                res = load_ply(file_)

                files_content.append(res)

    assert len(files_content) > 0, f"No .ply files found in {tar_filename}. Something is fishy."
    return files_content

def realign_normals_to_umbilicus(block_points, block_normals, umbilicus_points, axis_indices):
    main_sheet_points_scaled = scale_points(deepcopy(block_points), 200.0 / 50.0, axis_offset=0.0)
    main_sheet_points_scaled = main_sheet_points_scaled[:, axis_indices]
    block_normals_umbilicus = deepcopy(block_normals)[:, axis_indices]
    umbilicus_points_main = umbilicus_xz_at_y(umbilicus_points, main_sheet_points_scaled[:, 1])

    # points-umbilicus vectors
    block_points_umbilicus = main_sheet_points_scaled - umbilicus_points_main

    # check alignment between block_normals and block_points_umbilicus
    # dot product between block_normals_umbilicus and block_points_umbilicus
    dot = np.sum(block_normals_umbilicus * block_points_umbilicus, axis=1)
    # switch sign of block_normals if dot product is negative
    block_normals[dot < 0] = -block_normals[dot < 0]

    return block_points, block_normals, np.sum(dot < 0)

def process_block(args):
    block_tar, umbilicus_points, axis_indices = args

    block_path = block_tar[:-4]
    try:
        block = load_tar_block(block_path)
    except:
        return 0
    
    # Realign normals towards umbilicus
    new_block = []
    aligned_count = 0
    for patch in block:
        patch_points, patch_normals = patch[0], patch[1]
        patch_points, patch_normals, nr_aligned = realign_normals_to_umbilicus(patch_points, patch_normals, umbilicus_points, axis_indices)
        aligned_count += nr_aligned
        patch = (patch_points, patch_normals, *patch[2:])
        new_block.append(patch)

    # block format is [(points, normals, color, score, distance, coeff, n), ...], get all the data separated
    block_points, block_normals, block_color, block_score, block_distance, block_coeff, block_n = zip(*new_block)

    # Save again
    n = block_n[0] # All the same
    save_block_ply(block_points, block_normals, block_color, block_score, block_distances_precomputed=block_distance, block_coeffs_precomputed=block_coeff, n=n, post_process=False, block_name=block_path, check_exist=False, score_threshold=None, distance_threshold=None, alpha=None, slope_alpha=None)

    return aligned_count

def main():
    # Axis swap mesh PC
    axis_indices = [1, 2, 0]
    path = "scroll3_surface_points/point_cloud_verso_colorized_subvolume_blocks"
    # Load the umbilicus data
    umbilicus_path = '../scroll3_grids/umbilicus.txt'
    umbilicus_data = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_data)

    # Find all block tars in the scroll
    block_tars = glob(os.path.join(path, "*.tar"))

    # Multi Threaded
    args = [(block_tar, umbilicus_points, axis_indices) for block_tar in block_tars]
    with Pool(processes=cpu_count()) as pool:
        res = list(tqdm(pool.imap(process_block, args), total=len(block_tars)))
    print(f"Aligned {np.sum(res)} normals towards umbilicus")


if __name__ == '__main__':
    main()
    print("Done!")