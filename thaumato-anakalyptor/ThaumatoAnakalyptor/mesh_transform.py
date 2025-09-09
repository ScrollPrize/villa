### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import json
import numpy as np
import open3d as o3d
from PIL import Image
# no max size for images
Image.MAX_IMAGE_PIXELS = None
import glob
import argparse
import os
# temp file
import tempfile

def load_transform(transform_path):
    with open(transform_path, 'r') as file:
        data = json.load(file)
    return np.array(data["params"])

def invert_transform(transform_matrix):
    inv_transform = np.linalg.inv(transform_matrix)
    transform_matrix = transform_matrix / transform_matrix[3, 3] # Homogeneous coordinates
    return inv_transform

def combine_transforms(transform_a, transform_b):
    return np.dot(transform_b, transform_a)

def apply_transform_to_mesh(mesh_path, transform_matrix):
    # copy mesh to tempfile
    with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
        # copy mesh to tempfile
        temp_path = temp_file.name
        # os copy
        os.system(f"cp {mesh_path} {temp_path}")
        # load mesh
        mesh = o3d.io.read_triangle_mesh(temp_path)
    
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ transform_matrix[:3, :3].T + transform_matrix[:3, 3])
    normals = np.asarray(mesh.vertex_normals) @ transform_matrix[:3, :3].T
    # Normalize Normals
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh

def apply_transform_to_image(image_path, transform_matrix):
    image = Image.open(image_path)
    image = image.convert('L')
    scale_x, scale_y = transform_matrix[0, 0], transform_matrix[1, 1]
    new_size = int(image.width * scale_x), int(image.height * scale_y)
    return image.resize(new_size, Image.ANTIALIAS)

def parse_mtl_for_texture_filenames(mtl_filepath):
    texture_filenames = []
    with open(mtl_filepath, 'r') as file:
        for line in file:
            if line.strip().startswith('map_Kd'):  # This line indicates a texture file in MTL
                parts = line.split()
                if len(parts) > 1:
                    texture_filenames.append(parts[1])  # The second part is the filename
    return texture_filenames

def compute(transform_path, original_volume_id, target_volume_id, mesh_path, scale_factor):
    # Load scale transformation
    scale_transform = np.eye(4)
    scale_transform[0, 0] = scale_factor
    scale_transform[1, 1] = scale_factor
    scale_transform[2, 2] = scale_factor

    # Load transform from original to canonical
    if not (original_volume_id is None):
        transform_to_canonical_path = glob.glob(f"{transform_path}/*-to-{original_volume_id}.json")
        if len(transform_to_canonical_path) == 0:
            inverted_transform_to_canonical = np.eye(4)
        else:
            transform_to_canonical_path = transform_to_canonical_path[0]
            transform_to_canonical = load_transform(transform_to_canonical_path)

            # Invert it since we need to go from original to canonical
            inverted_transform_to_canonical = invert_transform(transform_to_canonical)
            assert np.allclose(np.dot(transform_to_canonical, inverted_transform_to_canonical), np.eye(4))
    else:
        inverted_transform_to_canonical = np.eye(4)
    
    is_canonical = len(glob.glob(f"{transform_path}/{target_volume_id}-to-*.json")) > 0
    if not is_canonical:
        # Load transform from canonical to target
        transform_to_target_path = glob.glob(f"{transform_path}/*-to-{target_volume_id}.json")[0]
        transform_to_target = load_transform(transform_to_target_path)
    else:
        transform_to_target = np.eye(4)
    
    # Combine the transformations
    combined_transform = combine_transforms(scale_transform, inverted_transform_to_canonical)
    combined_transform = combine_transforms(combined_transform, transform_to_target)
    
    # Paths and application of transforms
    segment_name = str(os.path.basename(mesh_path)[:-4])
    base_path = str(os.path.dirname(mesh_path))
    mtl_path = f"{base_path}/{segment_name}.mtl"
    if not os.path.exists(mtl_path):
        mtl_path.replace('.mtl', '_0.mtl')

    texture_filenames = parse_mtl_for_texture_filenames(mtl_path)
    if len(texture_filenames) > 0:
        image_path = os.path.join(base_path, texture_filenames[0])
        print(f"Found material texture image at: {image_path}", end="\n")
    else:
        image_path = None
        print("No corresponding TIF, PNG, or MTL image found.", end="\n")
    print("Texture Image Name:", image_path)

    base_path_save = base_path + f"_{target_volume_id}"
    mesh_save_path = mesh_path.replace(base_path, base_path_save)
    mtl_save_path = mtl_path.replace(base_path, base_path_save)
    image_save_path = image_path.replace(base_path, base_path_save)

    mesh = apply_transform_to_mesh(mesh_path, combined_transform)
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)

    transformed_image = apply_transform_to_image(image_path, combined_transform)
    transformed_image.save(image_save_path)

    with open(mtl_path, 'r') as file:
        mtl_data = file.read()
    with open(mtl_save_path, 'w') as file:
        file.write(mtl_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform mesh .obj with VC volume-volume transformation .json')
    parser.add_argument("--transform_path", type=str, required=True, help="Folder containing the transform .json files")
    parser.add_argument("--original_volume_id", type=str, required=True, help="Volume ID of the original scroll volume for the input mesh")
    parser.add_argument("--target_volume_id", type=str, required=True, help="Volume ID of the target scroll volume to transform the input mesh to")
    parser.add_argument("--obj_path", type=str, required=True, help="Path to your .obj file")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor for the input mesh")

    args = parser.parse_args()

    print(f"Processing from {args.original_volume_id} to {args.target_volume_id}")

    # if the obj path does not end in .obj, find all "*_flatboi.obj" files in the directory
    if not args.obj_path.endswith('.obj'):
        obj_paths = glob.glob(os.path.join(args.obj_path, '*_flatboi.obj'))
        for obj_path in obj_paths:
            print(f"Transforming {obj_path}")
            compute(args.transform_path, args.original_volume_id, args.target_volume_id, obj_path, args.scale_factor)
    else:
        compute(args.transform_path, args.original_volume_id, args.target_volume_id, args.obj_path, args.scale_factor)

# Example command: python3 -m ThaumatoAnakalyptor.mesh_transform --transform_path /scroll.volpkg/transforms --original_volume_id 20231027191953 --target_volume_id 20231117143551 --obj_path /scroll.volpkg/working/scroll3_surface_points/1352_3600_5002/point_cloud_colorized_verso_subvolume_blocks/windowed_mesh_20240830103110 --scale_factor 2.0