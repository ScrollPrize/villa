import open3d as o3d
import argparse
import numpy as np

axis_indices = [2, 0, 1]

def ply_to_obj(ply_path, obj_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    # Empty triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    # Write vertices to mesh
    mesh.vertices = pcd.points
    vertices = np.asarray(mesh.vertices) - 500
    vertices = vertices[:, axis_indices]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Write triangles to mesh
    o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f"Converted {ply_path} to {obj_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_path', type=str)
    parser.add_argument('obj_path', type=str)
    args = parser.parse_args()
    ply_to_obj(args.ply_path, args.obj_path)