### Giorgio Angelotti - 2024

from .rendering_utils.torch_ppm import points_in_triangles_batched
import open3d as o3d
import argparse
import gc
import os
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def main(args):
    working_path = os.path.dirname(args.obj_path)
    base_name = os.path.splitext(os.path.basename(args.obj_path))[0]
    ppm_path = os.path.join(working_path, f"{base_name}.memmap")

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
        y_size, x_size = img.size
    print(f"Y-size: {y_size}, X-size: {x_size}", end="\n")

    mesh = o3d.io.read_triangle_mesh(args.obj_path)
    print(f"Loaded mesh from {args.obj_path}", end="\n")

    V = torch.tensor(np.asarray(mesh.vertices))
    N = torch.tensor(np.asarray(mesh.vertex_normals))
    UV = torch.tensor(np.asarray(mesh.triangle_uvs))  # Ensure your mesh has UV coordinates
    F = torch.tensor(np.asarray(mesh.triangles))

    # Adjust UV coordinates as per the requirement
    UV_scaled = UV * torch.tensor([y_size, x_size])

    # Generate UV_TARGET
    x = torch.arange(x_size-1, -1, -1)
    y = torch.arange(0, y_size, 1)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    UV_TARGET = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).to(torch.float64)
    print(f"Created grid.", end="\n")

    del mesh, x, y, grid_x, grid_y, UV
    gc.collect()

    print(f"Computing coordinates...", end="\n")

    points_in_triangles_batched(ppm_path, (y_size, x_size), UV_TARGET, UV_scaled, F, V, N, pts_batch_size=args.pts_batch, tri_batch_size=args.tri_batch)

if __name__ == '__main__':
    # parse obj path, process batches of 4096 points, check against 64 nearest triangles (according to centroid)
    # change the batch sizes according to your resources. A too low tri_batch can affect the result, since KDTree might not find the
    # right triangle in the list... play with the parameters according to your RAM and mesh size
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_path', type=str)
    parser.add_argument('--pts_batch', type=int, default=4096)
    parser.add_argument('--tri_batch', type=int, default=64)
    args = parser.parse_args()

    main(args)
