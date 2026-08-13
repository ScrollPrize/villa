### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

from .rendering_utils.interpolate_image_3d import extract_from_image_3d, insert_into_image_3d
from .rendering_utils.ppmparser import PPMParser
from .grid_to_pointcloud import load_grid
import argparse
from tqdm import tqdm
import struct

import os
import tifffile

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
# nr threads
import multiprocessing

def load_ppm_cubes(path, cube_size=500):
    with PPMParser(path).open() as ppm:
        im_shape = ppm.im_shape()
        cubes = ppm.classify_entries_to_cubes(cube_size=cube_size)
    return cubes, im_shape

def cube_coords(cube_key, padding, cube_size):
    x, y, z = cube_key
    start_coords = np.array([x*cube_size - padding, y*cube_size - padding, z*cube_size - padding]) + cube_size # spelufo offset, 1 indexing
    grid_block_size = cube_size + 2*padding + 1
    return start_coords, grid_block_size

def load_and_process_grid_volume(cubes, cube, args, path_template, axis_swap_trans, gpu_num=0):
    # construct volume indexing
    cube_ppm = cubes[cube]
    cube_xyz = np.zeros((len(cube_ppm), 3), dtype=np.float32)
    cube_normals = np.zeros((len(cube_ppm), 3), dtype=np.float32)
    cube_image_positions = np.zeros((len(cube_ppm), 2), dtype=np.int32)
    for i, (imx, imy, c_) in enumerate(cube_ppm):
        c = struct.unpack('<dddddd', c_)
        cube_xyz[i] = c[:3]
        cube_normals[i] = c[3:]
        cube_image_positions[i] = [imy, imx]

    xyz = torch.tensor(cube_xyz, dtype=torch.float32).cuda()
    normals = torch.tensor(cube_normals, dtype=torch.float32).cuda()
    # construct all coordinate in positive and negative r
    coords = torch.cat([xyz + r * normals for r in range(-args.r, args.r+1)], dim=0)

    # find min and max values in each dimension
    coords_cpu = coords.cpu().numpy()
    min_coords = np.min(coords_cpu, axis=0).astype(np.int32)
    max_coords = np.max(coords_cpu, axis=0).astype(np.int32)
    start_coords = np.array(min_coords).astype(np.int32)
    axis_swap = [1, 0, 2]
    start_coords = start_coords[axis_swap] + args.cube_size
    grid_block_size = np.array(max_coords - min_coords + 1).astype(np.int32)[axis_swap]

    grid_volume = load_grid(path_template, tuple(start_coords), grid_block_size, args.cube_size, uint8=False).astype(np.float32)
    grid_volume = np.transpose(grid_volume.copy(), axes=axis_swap_trans)
    grid_volume = torch.from_numpy(grid_volume).cuda()
    
    device = torch.device("cuda:" + str(gpu_num))
    # recalculate coords to zero on grid_volume
    coords = coords - torch.tensor(min_coords, dtype=torch.float32, device=device)
    
    # extract from grid volume
    samples = extract_from_image_3d(grid_volume, coords).cpu()
    del grid_volume
    del coords

    # construct layers coords
    xy = torch.tensor(cube_image_positions, dtype=torch.int32).cpu()  # x, y coordinates
    # construct z coordinates for each layer
    z_layers = torch.arange(0, 2*args.r+1, dtype=torch.int32).repeat(len(cube_ppm), 1).T.contiguous().view(-1).cpu()
    # repeat xy coordinates for each layer
    xy_repeated = xy.repeat(2*args.r+1, 1)
    # combine xy and z coordinates
    xyz_layers = torch.cat([z_layers[:, None], xy_repeated], dim=1).cpu()  # z, x, y order
    
    return samples, xyz_layers

def main(args):
    working_path = os.path.dirname(args.ppm_path)
    path_template = args.grid_volume_path + "/cell_yxz_{:03}_{:03}_{:03}.tif"

    # load ppm cubes
    cubes, im_shape = load_ppm_cubes(args.ppm_path, cube_size=args.rendering_size)
    print(f"Loaded {len(cubes)} cubes from {args.ppm_path}")

    # pytorch array uint16 on cpu of size 2*r, im_shape
    layers = torch.zeros((2*args.r + 1, im_shape[1], im_shape[0]), dtype=torch.float32, device='cpu')
    layers_path = working_path + "/layers/"

    print(f"All parameters: {args}, im_shape: {im_shape}, layers_path: {layers_path}, path_template: {path_template}")
    axis_swap_trans = [2, 1, 0]

    # Process cubes in batches
    cube_list = list(cubes.keys())
    total_cubes = len(cube_list)
    batch_size = args.max_workers  # Process cubes in batches equal to max_workers

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        with tqdm(total=total_cubes, desc="Processing Cubes") as progress:
            for i in range(0, total_cubes, batch_size):
                gpu_num = i % args.gpus
                # Submit a batch of tasks
                futures = [executor.submit(load_and_process_grid_volume, cubes, cube, args, path_template, axis_swap_trans, gpu_num) for cube in cube_list[i:i+batch_size]]

                # Process completed tasks before moving on to the next batch
                for future in as_completed(futures):
                    # Update the progress bar each time a future is completed
                    progress.update(1)

                    result = future.result()
                    samples, xyz_layers = result
                    # Insert into layers
                    insert_into_image_3d(samples, xyz_layers, layers)

    # save layers
    nr_zeros = len(str(2*args.r))
    for i in range(layers.shape[0]):
        layer = layers[i].cpu().numpy().astype(np.uint16)
        # save layer with leading 0's for 2*r layers
        layer_nr = str(i).zfill(nr_zeros)
        layer_path = layers_path + f"{layer_nr}.tif"
        tifffile.imwrite(layer_path, layer)

if __name__ == '__main__':
    # parse ppm path, grid volume path, r=32, cube_size=500 default, all cores
    parser = argparse.ArgumentParser()
    parser.add_argument('ppm_path', type=str)
    parser.add_argument('grid_volume_path', type=str)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--cube_size', type=int, default=500)
    parser.add_argument('--rendering_size', type=int, default=400)
    parser.add_argument('--max_workers', type=int, default=multiprocessing.cpu_count()//2)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    main(args)