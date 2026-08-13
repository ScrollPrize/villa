## Giorgio Angelotti - 2024
## Based on ppm_to_layers by Julian Schilliger

from .rendering_utils.interpolate_image_3d import extract_from_image_3d, insert_into_image_3d
from .grid_to_pointcloud import load_grid
import argparse
from tqdm import tqdm
import os
import tifffile
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
# nr threads
import multiprocessing

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

layers = None

def process_chunk(data_chunk, offset_x, offset_y, cube_size):
    cube_chunk = {}
    # Create a mask for rows where the first three elements are not all zeroes
    mask = np.any(data_chunk[:, :, :3] != 0, axis=2)
    for imx in range(data_chunk.shape[0]):
        for imy in range(data_chunk.shape[1]):
            if mask[imx, imy]:  # Only process if the mask at (imx, imy) is True
                key = tuple(np.floor(data_chunk[imx, imy, :3] / cube_size).astype(np.int32))
                if key not in cube_chunk:
                    cube_chunk[key] = [[imx + offset_x, imy + offset_y]]
                else:
                    cube_chunk[key].append([imx + offset_x, imy + offset_y])
    return cube_chunk



def merge_dictionaries(dicts):
    """Merge dictionaries, combining lists of coordinates for the same key."""
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key].extend(value)
            else:
                result[key] = value
    return result


def classify_entries_to_cubes(cube_size, start, data):
    chunk_size_x, chunk_size_y = 256, 256  # Example sizes
    chunks = []

    for start_x in range(0, data.shape[0], chunk_size_x):
        for start_y in range(0, data.shape[1], chunk_size_y):
            end_x = min(start_x + chunk_size_x, data.shape[0])
            end_y = min(start_y + chunk_size_y, data.shape[1])
            chunks.append((data[start_x:end_x, start_y:end_y, :], start_x+start[0], start_y+start[1]))

    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk[0], chunk[1], chunk[2], cube_size) for chunk in chunks]
        partial_results = [future.result() for future in futures]

    # Merge the partial results
    cube = merge_dictionaries(partial_results)
    return cube

def calculate_chunk_indices(coords, start_coords, chunk_sizes):
    # Adjust coords relative to the start of the grid
    adjusted_coords = coords - start_coords
    
    # Calculate chunk indices for each coordinate
    chunk_indices = adjusted_coords // chunk_sizes
    
    return chunk_indices

def global_to_chunk_coords(coords, start_coords, chunk_sizes):
    chunk_sizes = torch.tensor(chunk_sizes)
    # Adjust coords relative to the start of the grid
    adjusted_coords = coords - start_coords
    
    # Calculate chunk indices for each coordinate
    chunk_indices = adjusted_coords // chunk_sizes

    # Calculate the start of each coordinate's chunk
    chunk_starts = chunk_indices * chunk_sizes
    
    # Calculate local (chunk) coordinates
    chunk_coords = adjusted_coords - chunk_starts
    
    return chunk_coords, chunk_indices

def load_and_process_grid_volume(cubes, cube, data, args, path_template, axis_swap_trans, gpu_num):
    # construct volume indexing
    device = 'cpu'
    cube_ppm = cubes[cube]
    # Vectorize the extraction of xyz and normals directly without loop
    imx, imy = np.transpose(cube_ppm)  # Transpose to separate x and y
    cube_image_positions = np.vstack((imy, imx)).T.astype(np.int32)
    
    xyz = torch.from_numpy(data[imx, imy, :3])
    normals = torch.from_numpy(data[imx, imy, 3:])
    # construct all coordinate in positive and negative r
    coords = torch.cat([xyz + r * normals for r in range(-args.r, args.r+1)], dim=0)

    # find min and max values in each dimension
    coords_cpu = coords.cpu().numpy()
    min_coords = np.min(coords_cpu, axis=0).astype(np.int32)
    max_coords = np.max(coords_cpu, axis=0).astype(np.int32)
    start_coords = np.array(min_coords).astype(np.int32)
    axis_swap = [1, 0, 2]
    start_coords = start_coords[axis_swap] + args.cube_size
    #print(start_coords)
    grid_block_size = np.array(max_coords - min_coords + 1).astype(np.int32)[axis_swap]
    
    grid_volume = load_grid(path_template, tuple(start_coords), grid_block_size, args.cube_size, uint8=False).astype(np.float64)
    grid_volume = np.transpose(grid_volume.copy(), axes=axis_swap_trans)
    grid_volume = torch.from_numpy(grid_volume)
    
    #device = torch.device("cuda:" + str(gpu_num))
    # recalculate coords to zero on grid_volume
    coords = coords - torch.tensor(min_coords, dtype=torch.float64, device=device)
    
    # extract from grid volume
    samples = extract_from_image_3d(grid_volume, coords).cpu()
    del grid_volume
    del coords
    

    
    xy = torch.tensor(cube_image_positions, dtype=torch.int32).cpu()  # x, y coordinates
    # construct z coordinates for each layer
    z_layers = torch.arange(0, 2*args.r+1, dtype=torch.int32).repeat(len(cube_ppm), 1).T.contiguous().view(-1).cpu()
    # repeat xy coordinates for each layer
    xy_repeated = xy.repeat(2*args.r+1, 1)
    # combine xy and z coordinates
    xyz_layers = torch.cat([z_layers[:, None], xy_repeated], dim=1).cpu()  # z, x, y order

    global layers
    insert_into_image_3d(samples, xyz_layers, layers)

def main(args):
    working_path = os.path.dirname(args.ppm_path)
    path_template = args.grid_volume_path + "/cell_yxz_{:03}_{:03}_{:03}.tif"
    layers_path = args.output_dir

    # Create the output layers directory if it doesn't exist.
    if not os.path.isdir(layers_path):
        os.mkdir(layers_path)

    base_name = os.path.splitext(os.path.basename(args.ppm_path))[0]

    # Construct the potential paths for the .tif and .png files
    tif_path = os.path.join(working_path, f"{base_name}.tif")
    png_path = os.path.join(working_path, f"{base_name}.png")

    # Check if the .tif version exists
    if os.path.exists(tif_path):
        image_path = tif_path
        print(f"Found TIF image at: {image_path}", end="\n")
    # If not, check if the .png version exists
    elif os.path.exists(png_path):
        image_path = png_path
        print(f"Found PNG image at: {image_path}", end="\n")
    # If neither exists, handle the case (e.g., error message)
    else:
        image_path = None
        print("No corresponding TIF or PNG image found.", end="\n")
    
    # Open the image file
    with Image.open(image_path) as img:
        # Get dimensions
        im_shape = img.size

    # load ppm cubes
    print("Accessing memmap...", end="\n")
    data = np.memmap(args.ppm_path, mode='r', dtype=np.float64, shape=(im_shape[0], im_shape[1], 6))

    global layers
    layers = torch.zeros((2*args.r + 1, im_shape[1], im_shape[0]), dtype=torch.float64, device='cpu')

    print(f"All parameters: {args}, im_shape: {im_shape}, layers_path: {layers_path}, path_template: {path_template}")
    axis_swap_trans = [2, 1, 0]

    for y in tqdm(range(0, im_shape[0], args.block_render), desc='Spanning image'):
        for x in range(0, im_shape[1], args.block_render):
            # Process cubes in batches
            cubes = classify_entries_to_cubes(cube_size=args.cube_size, start=(y, x), data=data[y: min(y+args.block_render, im_shape[0]), x: min(x+args.block_render, im_shape[1])])
            cube_list = list(cubes.keys())
            #print(cube_list)
            total_cubes = len(cube_list)
            batch_size = args.max_workers  # Process cubes in batches equal to max_workers
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                with tqdm(total=total_cubes, desc=f"Processing Cubes for {y,x}") as progress:
                    for i in range(0, total_cubes, batch_size):
                        gpu_num = i % args.gpus
                        # Submit a batch of tasks
                        futures = [executor.submit(load_and_process_grid_volume, cubes, cube, data, args, path_template, axis_swap_trans, gpu_num) for cube in cube_list[i:i+batch_size]]

                        # Process completed tasks before moving on to the next batch
                        for future in as_completed(futures):
                            # Update the progress bar each time a future is completed
                            progress.update(1)
                            future.result()


    # save layers
    nr_zeros = len(str(2*args.r))
    for i in range(layers.shape[0]):
        layer = layers[i].cpu().numpy().astype(np.uint16)
        # save layer with leading 0's for 2*r layers
        layer_nr = str(i).zfill(nr_zeros)
        layer_path = layers_path + f"/{layer_nr}.tif"
        tifffile.imwrite(layer_path, layer)

if __name__ == '__main__':
    # parse memmap path, grid volume path, r=32, cube_size=500, default, all cores
    # u can go up with block_render according to your RAM
    # the algorithm generates the "image" in memory in blocks of pixels block_render x block_render
    # this chunk processing avoids the allocation of a too big grid in memory, allowing rendering on
    # a simple laptop
    parser = argparse.ArgumentParser()
    parser.add_argument('ppm_path', type=str)
    parser.add_argument('grid_volume_path', type=str)   
    parser.add_argument('--output_dir', type=str, default=None, help="The output \"layers\" directory path. By default it is created in the same directory as the ppm.")
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--cube_size', type=int, default=500)
    parser.add_argument('--rendering_size', type=int, default=400)
    parser.add_argument('--block_render', type=int, default=2048)
    parser.add_argument('--max_workers', type=int, default=multiprocessing.cpu_count()//2)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.ppm_path) + "/layers"

    main(args)
