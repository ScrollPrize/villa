### Julian Schilliger - ThaumatoAnakalyptor - 2024

from .finalize_mesh import main as finalize_mesh_main
from .mesh_to_surface import ppm_and_texture

import subprocess
from tqdm import tqdm
import os
from glob import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut a mesh into pieces and texture the surface")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument('--scroll', type=str, required=True, help='Path to the grid cells')
    parser.add_argument('--format', type=str, default='jpg')
    parser.add_argument("--cut_size", type=int, help="Size of each cut piece along the X axis", default=40000)
    parser.add_argument("--output_folder", type=str, help="Folder to save the cut meshes", default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--nr_workers', type=int, default=None)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)

    args = parser.parse_known_args()[0]
    print(f"Known args: {args}")
    if args.input_mesh.endswith('.obj'):
        obj_paths = finalize_mesh_main(args.output_folder, args.input_mesh, 1.0, args.cut_size, False)
    else:
        # Find all .obj files in the input directory
        input_objs = glob(os.path.join(args.input_mesh, '*_flatboi.obj'))
        # Copy input meshes to the output folder
        obj_paths = []
        for input_obj in input_objs:
            # copy input mesh to the output folder
            print(f"Copying {input_obj} to {args.output_folder}")
            obj_path_ = finalize_mesh_main(args.output_folder, input_obj, 1.0, args.cut_size, False)
            obj_paths.extend(obj_path_)
    # sort the obj_paths
    obj_paths.sort()
    obj_paths = obj_paths[::-1]
    if args.end is not None:
        obj_paths = obj_paths[args.start:args.end]
    else:
        start = min(args.start, len(obj_paths))
        obj_paths = obj_paths[start:]
    for obj_path in tqdm(obj_paths, desc='Texturing meshes'):
        print(f"Texturing {obj_path}")
        # ppm_and_texture(obj_path, gpus=args.gpus, grid_cell_path=args.grid_cell, output_path=None, r=args.r, format=args.format, display=args.display, nr_workers=args.nr_workers, prefetch_factor=args.prefetch_factor)

        # Call mesh_to_surface as a separate process
        command = [
                    "python3", "-m", "ThaumatoAnakalyptor.mesh_to_surface", 
                    obj_path, args.scroll, 
                    "--gpus", str(args.gpus), 
                    "--r", str(args.r),
                    "--format", args.format,
                    "--prefetch_factor", str(args.prefetch_factor)
                ]
        if args.display:
            command.append("--display")
        if args.nr_workers is not None:
            command.append("--nr_workers", str(args.nr_workers))
        # Running the command
        process_rendering = subprocess.Popen(command)
        process_rendering.wait()

# python3 -m ThaumatoAnakalyptor.large_mesh_to_surface --input_mesh /scroll_pcs/1352_3600_5007/point_cloud_colorized_verso_subvolume_blocks/windowed_mesh_20250121085101 --scroll /scroll.volpkg/volumes/20241024131838.zarr --cut_size 20000 --display --start 15