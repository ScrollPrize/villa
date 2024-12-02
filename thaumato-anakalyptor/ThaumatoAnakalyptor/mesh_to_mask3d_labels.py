### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2024

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import argparse
import os
from .split_mesh import MeshSplitter
import tempfile
from tqdm import tqdm
import pickle
import multiprocessing
import sys
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import meshing_utils

def load_point_cloud(ply_file):
    # Load the point cloud from a PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    return np.asarray(pcd.points)

def save_point_cloud(filename, points):
    # mkdir if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def load_mesh_vertices(obj_file, use_tempfile=True):
    # copy mesh to tempfile
    if use_tempfile:
        # print("using tempfile")
        with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
            # print("created temp file")
            # copy mesh to tempfile
            temp_path = temp_file.name
            # print(f"copying mesh from {obj_file} to {temp_path}")
            # os copy
            os.system(f"cp {obj_file} {temp_path}")
            # load mesh
            mesh = o3d.io.read_triangle_mesh(temp_path, print_progress=True)
    else:
        mesh = o3d.io.read_triangle_mesh(obj_file, print_progress=True)
    print("loaded mesh")
    # check if mesh is empty
    if mesh is None:
        raise ValueError("Mesh is None")
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh is empty")
    # Coordinate transform to pointcloud coordinate system
    vertices = np.asarray(mesh.vertices)
    # print(f"Shape: {vertices.shape}, dtype: {vertices.dtype}")
    vertices += 500
    vertices = vertices[:,[1, 2, 0]]
    # print("Did some math on the mesh")
    # print(f"Shape: {vertices.shape}, dtype: {vertices.dtype}")
    vertices_o3d = o3d.utility.Vector3dVector(vertices) # CAREFULL: numpy version + open3d version issue. open3d 0.18.0 only works with numpy version < 2.0.0. eg 1.26.4/1.22.4
    # print(f"converted to o3d type")
    mesh.vertices = vertices_o3d

    # print("transformed vertices")

    triangles = np.asarray(mesh.triangles)

    print(f"Number triangles in mesh: {len(triangles)}")

    scene = setup_closest_triangles(mesh)
    print(f"Set up scene with {len(triangles)} triangles")
    return triangles, scene

def save_mesh(mesh, triangles, filename):
    selected_mesh = o3d.geometry.TriangleMesh()
    selected_mesh = o3d.geometry.TriangleMesh()
    selected_mesh.vertices = mesh.vertices
    selected_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # Remove unused vertices
    selected_mesh = selected_mesh.remove_unreferenced_vertices()
    selected_mesh = selected_mesh.compute_vertex_normals()
    selected_mesh = selected_mesh.compute_triangle_normals()
    # save the non-selected mesh
    o3d.io.write_triangle_mesh(filename, selected_mesh)

def calculate_winding_angle(mesh_path, pointcloud_dir):
    umbilicus_path = os.path.join(os.path.dirname(pointcloud_dir), "umbilicus.txt")
    splitter =  MeshSplitter(mesh_path, umbilicus_path)
    splitter.compute_uv_with_bfs(0)
    winding_angles = splitter.vertices_np[:, 0]
    return winding_angles, splitter

def find_closest_vertices(points, tree):    
    # Find the closest vertex for each point
    distances, indices = tree.query(points)
    return indices, distances

def setup_closest_triangles(mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    return scene

def find_closest_triangles(points, scene):
    # Find the closest triangle and the distance for each point
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    # We compute the closest point on the surface for the point at position [0,0,0].
    ans = scene.compute_closest_points(query_points)
    points_on_triangle = ans['points'].numpy()
    distances = np.linalg.norm(points_on_triangle - points, axis=-1)
    triangles_id = ans['primitive_ids'].numpy()
    return triangles_id, distances

def cluster_points_by_winding_angle(winding_angles_points, threshold_angle=45):
    # Use linkage with 'single' method specifying the clustering criterion
    # Here 'euclidean' is used; ensure your angles are prepared for this metric (could use 'cityblock' if thinking in terms of cyclic angles)
    Z = linkage(winding_angles_points.reshape(-1, 1), method='single', metric='euclidean')

    # Form flat clusters with the specified maximum cophenetic distance (threshold)
    clusters = fcluster(Z, t=threshold_angle, criterion='distance')
    return clusters

def generate_sample(index, valid_triangles, winding_angles, scene, triangles, ply_path, output_dir, max_distance, max_distance_valid):
    # Load data
    points = load_point_cloud(ply_path)

    # Find closest vertices
    closest_triangle, closest_distances = find_closest_triangles(points, scene)
    # Get the index of the first vertex in the closest triangle
    valid_triangles_points = np.array(valid_triangles)[closest_triangle]
    
    # Only keep points that are close enough to the mesh
    mask_distance = closest_distances < max_distance
    mask = np.logical_and(mask_distance, valid_triangles_points)

    # Check that every point is close enough to the mesh
    valid_distance = np.all(closest_distances < max_distance_valid)
    # Check that at least 50% of the points are masked true
    valid_mask = np.sum(mask) > 0.5 * len(mask)
    valid = valid_distance and valid_mask
    # print("Valid point cloud:", valid, "Valid distance:", valid_distance, "Valid mask:", valid_mask)
    if not valid:
        return

    closest_indices = triangles[closest_triangle][:, 0]
    closest_indices = closest_indices[mask]
    clostest_points = points[mask]

    # Find the winding angle for each point
    winding_angles_points = winding_angles[closest_indices]

    # Check if there are any points
    if len(winding_angles_points) == 0:
        return
    
    if len(winding_angles_points) < 10000:
        print(f"Skipping cube {index} with {len(winding_angles_points)} points")
        return
    
    # Save GT PointCloud
    ply_name = os.path.basename(ply_path)[:-4]
    gt_file = os.path.join(output_dir, ply_name, f"{ply_name}.ply")
    save_point_cloud(gt_file, clostest_points)

    # Cluster points by winding angle
    clusters = cluster_points_by_winding_angle(winding_angles_points)
    unique_clusters = np.unique(clusters)

    # Calculate the mean winding angle per cluster
    mean_winding_angles = np.array([np.mean(winding_angles_points[clusters == cluster_id]) for cluster_id in unique_clusters])
    # Sort unique clusters by mean winding angle
    cluster_indices = np.argsort(mean_winding_angles)

    # Optionally, output clustered points
    for sorted_index, cluster_index in enumerate(cluster_indices):
        cluster_id = unique_clusters[cluster_index]
        mask = clusters == cluster_id
        cluster_points = clostest_points[mask]
        cluster_file = os.path.join(output_dir, ply_name, "surfaces", f"{sorted_index}.ply")
        # Save cluster to a PLY file
        save_point_cloud(cluster_file, cluster_points)

def check_triangle_intersection(tri1, tri2):
    """
    Check if two triangles intersect using the Separating Axis Theorem (SAT).
    tri1 and tri2 are lists of three points each, defining the triangles.
    Each point is an (x, y, z) coordinate tuple or a numpy array.
    """
    def edge_directions(triangle):
        return [triangle[1] - triangle[0], triangle[2] - triangle[1], triangle[0] - triangle[2]]
    
    def normal(triangle):
        # Compute the normal vector of the triangle
        return np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])

    def project(triangle, axis):
        # Project triangle vertices onto the axis and return the min and max
        projections = np.dot(triangle, axis)
        return np.min(projections), np.max(projections)

    def overlaps(min1, max1, min2, max2):
        return max1 >= min2 and max2 >= min1

    def separating_axis_test(tri1, tri2, axis):
        min1, max1 = project(tri1, axis)
        min2, max2 = project(tri2, axis)
        return overlaps(min1, max1, min2, max2)

    tri1 = np.array(tri1)
    tri2 = np.array(tri2)

    # Triangle normals
    axis_tests = [normal(tri1), normal(tri2)]

    # Edge cross products
    edges1 = edge_directions(tri1)
    edges2 = edge_directions(tri2)

    for edge1 in edges1:
        for edge2 in edges2:
            axis_tests.append(np.cross(edge1, edge2))

    # Perform the separating axis test for all potential separating axes
    for axis in axis_tests:
        if np.linalg.norm(axis) == 0:  # Parallel or degenerate triangles
            continue
        if not separating_axis_test(tri1, tri2, axis):
            return False

    # No separating axis found, triangles must intersect
    return True

# Helper function to check triangle intersections and winding angles in parallel
def check_intersection_and_winding(args):
    pair, tri1, tri2, w1, w2, angle_range = args
    i, j = pair
    def check_winding_angle():
        return abs(w1 - w2) < angle_range
    
    if check_triangle_intersection(tri1, tri2):
        if not check_winding_angle():
            return i, j, w1, w2
    return None

def compute_intersections_and_winding_angles(mesh, winding_angles, angle_range, path):
    """
    For a given Open3D mesh, this function finds intersecting triangles
    and checks if their winding angles fall outside a specified range.

    Uses multiprocessing to parallelize the intersection and winding angle check.

    Returns a boolean list where each element corresponds to a triangle
    and is True if it intersects with another triangle, with winding angles
    outside the given range.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_vertices = [[[float(x) for x in t] for t in triangle] for triangle in vertices[triangles]]
    triangle_winding_angles = [float(w) for w in winding_angles[triangles[:, 0]]]

    print(f"Type of triangle_vertices at 0: {type(triangle_vertices[0])}")

    result = meshing_utils.compute_intersections_and_winding_angles(triangle_vertices, triangle_winding_angles, float(angle_range), 70.0)
    return np.array(result, dtype=bool)

# Global variables to be initialized in each worker
def init_worker(mesh_file, valid_triangles_data, winding_angles_data):
    global triangles, scene, valid_triangles, winding_angles
    # Load the mesh data
    triangles, scene = load_mesh_vertices(mesh_file)
    # Copy the large constant data to each worker
    valid_triangles = valid_triangles_data
    winding_angles = winding_angles_data
    print("Worker initialized")

# Worker function that will perform the task for each point cloud file
def process_point_cloud(args):
    i, ply_path, output_dir, max_distance, max_distance_valid = args
    try:
        # Use global `triangles`, `scene`, `valid_triangles`, and `winding_angles` in each worker
        generate_sample(i, valid_triangles, winding_angles, scene, triangles, ply_path, output_dir, max_distance, max_distance_valid)
    except Exception as e:
        print(f"Error processing {ply_path}: {e}")

def set_up_mesh(mesh_file, pointcloud_dir, continue_from=0, save_meshes=False, use_tempfile=True):
    output_dir = pointcloud_dir + "_mask3d_labels"
    
    fresh_start = continue_from <= 0
    if fresh_start:
        print("Setting up mesh")
        # Load the mesh vertices
        triangles, scene = load_mesh_vertices(mesh_file, use_tempfile=use_tempfile)
        # print(f"Loaded {len(triangles)} triangles")
        # Calculate winding angles for each vertex
        winding_angles, splitter = calculate_winding_angle(mesh_file, pointcloud_dir)
        print(f"Calculated winding angles for {len(winding_angles)} vertices")
        # pickle winding angles
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "winding_angles_triangles.pkl"), "wb") as f:
            pickle.dump((winding_angles, triangles), f)
    else:
        # Load winding angles
        with open(os.path.join(output_dir, "winding_angles_triangles.pkl"), "rb") as f:
            winding_angles, triangles = pickle.load(f)
        umbilicus_path = os.path.join(os.path.dirname(pointcloud_dir), "umbilicus.txt")
        splitter =  MeshSplitter(mesh_file, umbilicus_path)
        splitter.vertices_np[:, 0] = winding_angles

    print(f"Loaded winding angles for {len(winding_angles)} vertices")
    
    # Calculate the intersecting out of range triangles mask
    angle_range = 180 # the intersection at the steep bends go and cross more than 2 windings, filtering out to and disregarding the first winding intersection is okay since in close proximity there will also be the second intersection. large speedup
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    fresh_start2 = continue_from <= 1
    if fresh_start2:
        intersecting_triangles = compute_intersections_and_winding_angles(mesh, winding_angles, angle_range, output_dir)
        # Save the intersecting triangles mask
        with open(os.path.join(output_dir, "intersecting_triangles.pkl"), "wb") as f:
            pickle.dump(intersecting_triangles, f)
    else:
        # Load intersecting triangles
        with open(os.path.join(output_dir, "intersecting_triangles.pkl"), "rb") as f:
            intersecting_triangles = pickle.load(f)

    print(f"Found {np.sum(intersecting_triangles)} intersecting triangles")

    # Create mesh with only the intersecting triangles
    if save_meshes:
        save_mesh(mesh, triangles[intersecting_triangles], os.path.join(output_dir, "intersecting_mesh.obj"))

    # Valid triangles
    fresh_start3 = continue_from <= 2
    if fresh_start3:
        valid_triangles = meshing_utils.cluster_triangles(triangles, intersecting_triangles)
        # Save the valid triangles mask
        with open(os.path.join(output_dir, "valid_triangles.pkl"), "wb") as f:
            pickle.dump(valid_triangles, f)
    else:
        # Load valid triangles
        with open(os.path.join(output_dir, "valid_triangles.pkl"), "rb") as f:
            valid_triangles = pickle.load(f)
    non_selected_triangles = np.logical_not(valid_triangles)

    # Save the non-selected triangles mesh
    if save_meshes:
        save_mesh(mesh, triangles[non_selected_triangles], os.path.join(output_dir, "non_selected_mesh.obj"))

    # Save selected triangles mesh
    if save_meshes:
        save_mesh(mesh, triangles[valid_triangles], os.path.join(output_dir, "selected_mesh.obj"))

    return valid_triangles, winding_angles, output_dir, splitter

def compute(mesh_file, pointcloud_dir, max_distance, max_distance_valid, continue_from=0):
    valid_triangles, winding_angles, output_dir, splitter = set_up_mesh(mesh_file, pointcloud_dir, continue_from=continue_from)
    ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    # shuffle
    np.random.shuffle(ply_files)

    ### Single Threaded ###
    # triangles, scene = load_mesh_vertices(mesh_file)
    # ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    # for i, ply_file in enumerate(tqdm(ply_files, desc="Processing point clouds")):
    #     ply_path = os.path.join(pointcloud_dir, ply_file)
    #     try:
    #         generate_sample(i, valid_triangles, winding_angles, scene, triangles, ply_path, output_dir, max_distance, max_distance_valid)
    #     except Exception as e:
    #         print(f"Error processing {ply_path}: {e}")
    #         continue

    ### Multi Threaded ###
    # Initialize the pool and pass the large constant data once to each worker
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2, initializer=init_worker, initargs=(mesh_file, valid_triangles, winding_angles)) as pool:
        # Find all .ply files in the pointcloud directory
        tasks = []

        for i, ply_file in enumerate(ply_files):
            ply_path = os.path.join(pointcloud_dir, ply_file)
            # Append tasks without passing the large constant data
            tasks.append((i, ply_path, output_dir, max_distance, max_distance_valid))

        print(f"Processing {len(tasks)} point clouds")
        # Use pool.imap to execute the worker function
        list(tqdm(pool.imap(process_point_cloud, tasks), desc="Processing point clouds", total=len(tasks)))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert a 3D mesh and a PointCloud to 3D PointCloud instance labels.")
    parser.add_argument("--mesh_file", type=str, help="Path to the 3D mesh file (e.g., .obj)")
    parser.add_argument("--pointcloud_dir", type=str, help="Path to the 3D point cloud directory (containing .ply)")
    parser.add_argument("--max_distance", type=float, help="Maximum distance for a point to be considered part of the mesh", default=15)
    parser.add_argument("--max_distance_valid", type=int, help="Maximum distance all points must be to be considered valid a valid pointcloud", default=100)
    parser.add_argument("--continue_from", type=int, help="Continue from a specific step", default=0)

    args = parser.parse_args()

    # Compute the 3D mask with labels
    compute(args.mesh_file, args.pointcloud_dir, args.max_distance, args.max_distance_valid, continue_from=args.continue_from)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # need sppawn because of open3d initialization deadlock in the init worker function
    main()

# Example command: python3 -m ThaumatoAnakalyptor.mesh_to_mask3d_labels --mesh_file /scroll.volpkg/merging_test_merged/20230929220926-20231005123336-20231007101619-20231210121321-20231012184424-20231022170901-20231221180251-20231106155351-20231031143852-20230702185753-20231016151002.obj --pointcloud_dir /scroll1_surface_points/point_cloud_colorized_verso