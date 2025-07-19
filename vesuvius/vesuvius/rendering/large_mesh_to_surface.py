from .finalize_mesh import main as finalize_mesh_main

import subprocess
from tqdm import tqdm
import os
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cut a mesh into pieces and texture the surface")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument('--scroll', type=str, required=True, help='Path to the grid cells')
    parser.add_argument('--format', type=str, default='jpg')
    parser.add_argument("--cut_size", type=int, help="Size of each cut piece along the X axis", default=40000)
    parser.add_argument("--output_folder", type=str, help="Folder to save the cut meshes", default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--nr_workers', type=int, default=None)
    parser.add_argument('--max_side_triangle', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)

    args = parser.parse_args()
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
    if args.end is not None:
        obj_paths = obj_paths[args.start:args.end]
    else:
        start = min(args.start, len(obj_paths))
        obj_paths = obj_paths[start:]
    for obj_path in tqdm(obj_paths, desc='Texturing meshes'):
        print(f"Texturing {obj_path}")
        # Call mesh_to_surface as a separate process
        command = [
            "mesh_to_surface",
            obj_path, args.scroll,
            "--gpus", str(args.gpus),
            "--r", str(args.r),
            "--format", args.format,
        ]
        if args.display:
            command.append("--display")
        if args.remote:
            command.append("--remote")
        if args.nr_workers is not None:
            command.extend(["--nr_workers", str(args.nr_workers)])
        if args.max_side_triangle is not None:
            command.extend(["--max_side_triangle", str(args.max_side_triangle)])
     
        # Running the command
        process_rendering = subprocess.Popen(command)
        process_rendering.wait()

if __name__ == "__main__":
    main()
