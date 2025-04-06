import os
import argparse
import numpy as np
import open3d as o3d
import webknossos as wk
#from webknossos import webknossos_context
import networkx as nx
from tqdm import tqdm

def mesh_to_skeleton(mesh_path, dataset_name):
    """
    Convert a triangular mesh to a skeleton annotation.
    Each vertex of the mesh becomes a node, and for every triangle,
    edges are created connecting the triangle's vertices.
    Duplicate edges are removed by sorting vertex indices.
    """
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Create a WebKnossos annotation
    annotation = wk.Annotation(
        name=os.path.basename(mesh_path),
        dataset_name=dataset_name,
        voxel_size=(7.91, 7.91, 7.91)
    )
    skeleton = annotation.skeleton

    # Create a graph for the skeleton
    g = nx.Graph()
    # Add nodes with vertex positions as attributes
    for i, vertex in tqdm(enumerate(vertices), desc=f"Processing vertices in {mesh_path}", total=len(vertices)):
        # Convert coordinates to uint32
        rounded = vertex.astype(np.uint32)
        g.add_node(i, position=(rounded[0], rounded[1], rounded[2]))
    
    # Add edges for each triangle
    for tri in tqdm(triangles, desc=f"Processing triangles in {mesh_path}", total=len(triangles)):
        i, j, k = tri
        g.add_edge(i, j)
        g.add_edge(j, k)
        g.add_edge(k, i)

    # Add the graph to the skeleton and save the annotation as a .zip file
    skeleton.add_nx_graphs([g])
    output_zip_path = os.path.join(
        os.path.dirname(mesh_path), 
        f"{os.path.splitext(os.path.basename(mesh_path))[0]}_webknossos.zip"
    )
    annotation.save(output_zip_path)
        
    print(f"Converted {mesh_path} to {output_zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively convert all .obj meshes in a directory to skeleton annotations using WebKnossos."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to search for .obj mesh files",
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name",
    )
    
    args = parser.parse_args()
    
    print(f"Dataset name: {args.dataset_name}")
    # Verify the directory exists
    if not os.path.isdir(args.root_dir):
        raise NotADirectoryError(f"Directory not found: {args.root_dir}")
    
    # Traverse the directory tree and process every .obj file found
    for dirpath, _, filenames in os.walk(args.root_dir):
        for filename in filenames:
            if "points" not in filename.lower() and "intermediate" not in filename.lower():
                if filename.lower().endswith(".obj"):
                    mesh_path = os.path.join(dirpath, filename)
                    try:
                        mesh_to_skeleton(mesh_path, args.dataset_name)
                    except Exception as e:
                        print(f"Error processing {mesh_path}: {e}")
