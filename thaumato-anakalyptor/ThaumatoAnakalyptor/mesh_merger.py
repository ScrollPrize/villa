### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2024

import open3d as o3d
import igl
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import argparse
import os
from .split_mesh import MeshSplitter
import tempfile
from tqdm import tqdm
import pickle
from copy import deepcopy

def load_stitching_order(txt_path):
    # loads from a txt file the mesh filenames. each line contains one mesh name
    with open(txt_path, "r") as f:
        mesh_files = f.readlines()
    mesh_files = [m.strip() for m in mesh_files]
    print(f"Loaded {mesh_files} mesh files")
    return mesh_files

def load_mesh_vertices(obj_file):
    # copy mesh to tempfile
    with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
        # copy mesh to tempfile
        temp_path = temp_file.name
        # os copy
        os.system(f"cp {obj_file} {temp_path}")
        # load mesh
        mesh = o3d.io.read_triangle_mesh(temp_path, print_progress=True)

    # Coordinate transform to pointcloud coordinate system
    vertices = np.asarray(mesh.vertices)
    vertices += 500
    vertices = vertices[:,[1, 2, 0]]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    normals = np.asarray(mesh.vertex_normals)
    normals = normals[:,[1, 2, 0]]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    triangles = np.asarray(mesh.triangles)

    print(f"Number triangles in mesh: {len(triangles)}")

    scene = setup_closest_triangles(mesh)
    return mesh, vertices, triangles, scene

def save_mesh(save_mesh_path, mesh):
    vertices = np.asarray(mesh.vertices)
    vertices -= 500
    vertices = vertices[:,[2, 0, 1]].astype(np.float64)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    normals = np.asarray(mesh.vertex_normals)
    normals = normals[:,[2, 0, 1]]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    triangles = np.asarray(mesh.triangles).astype(np.int64)
    o3d.io.write_triangle_mesh(save_mesh_path, mesh, write_ascii=False)
    # success = igl.write_obj(save_mesh_path, vertices, triangles)
    # print(f"Save mesh success: {success}")

def calculate_winding_angle(mesh_path, pointcloud_dir):
    umbilicus_path = os.path.join(os.path.dirname(pointcloud_dir), "umbilicus.txt")
    splitter = MeshSplitter(mesh_path, umbilicus_path, use_tempfile=True)
    # splitter.compute_uv_with_bfs(np.random.randint(0, splitter.vertices_np.shape[0]))
    splitter.compute_uv_with_bfs(0) # for production
    winding_angles = splitter.vertices_np[:, 0]
    return winding_angles

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
    # distances = np.sqrt((points_on_triangle - points ** 2).sum(-1))
    distances = np.linalg.norm(points_on_triangle - points, axis=-1)
    triangles_id = ans['primitive_ids'].numpy()
    return triangles_id, distances

def find_closest_triangles_same_winding(vertices1, winding_angles1, winding_angles2, triangles2, scene2, default_splitter):
    # optimizeable normal = umbilicus point - vertex
    umbilicus_xy1_x, umbilicus_xy1_y, umbilicus_xy1_z = default_splitter.interpolate_umbilicus(vertices1[:, 1] - 500)
    umbilicus_xy1 = np.stack([umbilicus_xy1_y, umbilicus_xy1_z, umbilicus_xy1_x], axis=-1) + 500
    optimizable_normals1 = vertices1 - umbilicus_xy1
    # normalize the normals
    optimizable_normals1 = optimizable_normals1 / np.linalg.norm(optimizable_normals1, axis=-1)[:, None]

    # list_intersections from open3d raycasting
    print(f"Total number of vertices: {len(vertices1)}")
    nr_sample = 1000
    directions1_0 = {}
    distances1_0 = {}
    directions1_1 = {}
    distances1_1 = {}
    for i in tqdm(range(0, len(vertices1), nr_sample), desc="list_intersections (vertex + normal all triangle intersections)"):
        # print(f"Processing vertices {i} to {i+nr_sample}")
        end_v = min(i+nr_sample, len(vertices1))
        vertices1_sample = vertices1[i:end_v]
        normals1_sample = optimizable_normals1[i:end_v]
        # Concatenate vertices and normals
        rays1_direction1 = np.concatenate([vertices1_sample, normals1_sample], axis=-1).astype(np.float32)
        rays1_direction2 = np.concatenate([vertices1_sample, -normals1_sample], axis=-1).astype(np.float32)

        # Compute the ray intersections.
        lx1_direction1 = scene2.list_intersections(rays1_direction1)
        lx1_direction1 = {k:v.numpy() for k,v in lx1_direction1.items()}

        lx1_direction2 = scene2.list_intersections(rays1_direction2)
        lx1_direction2 = {k:v.numpy() for k,v in lx1_direction2.items()}

        # Check if all the intersections are the same. first start by the number of hits
        hits1 = lx1_direction1["ray_ids"].shape[0]
        hits2 = lx1_direction2["ray_ids"].shape[0]
        # assert hits1 == hits2, f"Number of hits is different: {hits1} vs {hits2}"
        # print(f"Number of hits: {hits1} vs {hits2}")

        ray_splits = lx1_direction1["ray_splits"]
        ray_ids = lx1_direction1["ray_ids"]
        primitive_ids = lx1_direction1["primitive_ids"]
        t_hit = lx1_direction1["t_hit"]
        for ray_id, (start, end) in enumerate(zip(ray_splits[:-1], ray_splits[1:])):
            w1 = winding_angles1[i+ray_id]
            best_distance = float("inf")
            best_normal = None
            for j, triangle_id in enumerate(primitive_ids[start:end]):
                ray_id_check = ray_ids[start+j]
                assert ray_id == ray_id_check, f"Ray id {ray_id} is different from ray id check {ray_id_check}"
                t2 = triangles2[triangle_id]
                w2 = winding_angles2[t2[0]]
                assert t_hit[start+j] >= 0, f"t_hit is negative: {t_hit[start+j]}"
                if abs(w1 - w2) < 45 and abs(t_hit[start+j]) < best_distance:
                    best_distance = abs(t_hit[start+j])
                    best_normal = normals1_sample[ray_id]
            # Add the best normal and distance
            directions1_0[i+ray_id] = best_normal
            distances1_0[i+ray_id] = best_distance                
            
        ray_splits = lx1_direction2["ray_splits"]
        ray_ids = lx1_direction2["ray_ids"]
        primitive_ids = lx1_direction2["primitive_ids"]
        t_hit = lx1_direction2["t_hit"]
        for ray_id, (start, end) in enumerate(zip(ray_splits[:-1], ray_splits[1:])):
            w1 = winding_angles1[i+ray_id]
            best_distance = float("inf")
            best_normal = None
            for j, triangle_id in enumerate(primitive_ids[start:end]):
                ray_id_check = ray_ids[start+j]
                assert ray_id == ray_id_check, f"Ray id {ray_id} is different from ray id check {ray_id_check}"
                t2 = triangles2[triangle_id]
                w2 = winding_angles2[t2[0]]
                assert t_hit[start+j] >= 0, f"t_hit is negative: {t_hit[start+j]}"
                if abs(w1 - w2) < 45 and abs(t_hit[start+j]) < best_distance:
                    best_distance = abs(t_hit[start+j])
                    best_normal = -normals1_sample[ray_id]
            # Add the best normal and distance
            directions1_1[i+ray_id] = best_normal
            distances1_1[i+ray_id] = best_distance

    directions = np.zeros((len(vertices1), 3), dtype=np.float32)
    distances = np.zeros(len(vertices1), dtype=np.float32)
    for ray_id in directions1_0:
        if directions1_0[ray_id] is None or (directions1_1[ray_id] is not None and distances1_1[ray_id] < distances1_0[ray_id]):
            directions[ray_id] = directions1_1[ray_id]
            distances[ray_id] = distances1_1[ray_id]
        elif directions1_0[ray_id] is not None:
            assert ray_id in directions1_0, f"Ray id {ray_id} not in directions1_0"
            directions[ray_id] = directions1_0[ray_id]
            distances[ray_id] = distances1_0[ray_id]
        else:
            distances[ray_id] = float("inf")
    
    assert len(directions) == len(vertices1), f"Number of directions {len(directions)} is different from number of vertices {len(vertices1)}"
    print("end list_intersections")
    return directions, np.array(distances)


def winding_difference(vertex1, angle1, vertex2, angle2, default_splitter):
    # Calculate the angle between two vertices
    angle_dif = default_splitter.angle_between_vertices(vertex1[[2, 0, 1]] - 500, vertex2[[2, 0, 1]] - 500)
    angle_vertex1_on_vertex2 = angle1 + angle_dif
    winding_difference = angle2 - angle_vertex1_on_vertex2
    k = winding_difference / 360.0
    k = np.round(k)
    return k

def find_valid_relations(points2, scene1, triangles1, winding_angles2, max_distance):
    # Find closest vertices
    closest_triangle, closest_distances = find_closest_triangles(points2, scene1)
    # Get the index of the first vertex in the closest triangle
    closest_indices1 = triangles1[closest_triangle][:, 0]

    # Only keep points that are close enough to the mesh
    mask_distance = closest_distances < max_distance
    points_indices2 = np.arange(len(points2))[mask_distance]
    closest_indices1 = closest_indices1[mask_distance]
    clostest_points2 = points2[mask_distance]
    # Find the winding angle for each point
    winding_angles_points2 = winding_angles2[mask_distance]

    return closest_indices1, winding_angles_points2, clostest_points2, points_indices2

def align_meshes(mesh1_path, mesh2_path, umbilicus_path, max_distance, default_splitter):
    # load the meshes
    mesh1, vertices1, triangles1, scene1 = load_mesh_vertices(mesh1_path)
    mesh2, vertices2, triangles2, scene2 = load_mesh_vertices(mesh2_path)

    # calculate the winding angles
    winding_angles1 = calculate_winding_angle(mesh1_path, umbilicus_path)
    winding_angles2 = calculate_winding_angle(mesh2_path, umbilicus_path)
    assert len(winding_angles1) == len(vertices1)
    assert len(winding_angles2) == len(vertices2)

    # find the closest vertices
    closest_indices1, winding_angles_vertices2, clostest_vertices2, points_indices2 = find_valid_relations(vertices2, scene1, triangles1, winding_angles2, max_distance)
    closest_indices2, winding_angles_vertices1, clostest_vertices1, points_indices1 = find_valid_relations(vertices1, scene2, triangles2, winding_angles1, max_distance)

    # Make statistic about the winding differences
    winding_angle_difference = {}
    for i in range(len(closest_indices1)):
        vertex1 = vertices1[closest_indices1[i]]
        angle1 = winding_angles1[closest_indices1[i]]
        vertex2 = clostest_vertices2[i]
        angle2 = winding_angles_vertices2[i]
        w_a_diff = winding_difference(vertex1, angle1, vertex2, angle2, default_splitter)
        if not int(w_a_diff) in winding_angle_difference:
            winding_angle_difference[int(w_a_diff)] = 0
        winding_angle_difference[int(w_a_diff)] += 1

    print(winding_angle_difference)
    return winding_angle_difference, winding_angles1, winding_angles2, points_indices1, points_indices2, scene1, scene2, mesh1, mesh2

def find_best_alignment(winding_angle_difference):
    # find the max used k value
    max_k = max(winding_angle_difference, key=winding_angle_difference.get)
    return 360.0*max_k

def adjust_vertices_positions(mesh1_mask, mesh2_mask, mesh1, mesh2, vertices1, vertices2, normals1, normals2, winding_angles1, winding_angles2, triangles1, triangles2, scene1, scene2, default_splitter, distance_threshold=0.1):
    # Mask the vertices that are optimizable
    optimizable_vertices1 = vertices1[mesh1_mask]
    optimizable_vertices2 = vertices2[mesh2_mask]

    # Dont forget to mask the corresponding winding angles too
    optimizeable_winding_angles1 = winding_angles1[mesh1_mask]
    optimizeable_winding_angles2 = winding_angles2[mesh2_mask]

    optimizable_direction1, optimizable_distance1 = find_closest_triangles_same_winding(optimizable_vertices1, optimizeable_winding_angles1, winding_angles2, triangles2, scene2, default_splitter) # winding_angles2 is index from triangles -> complete vertices, no masking ther on the second winding angles of theis functions
    optimizable_direction2, optimizable_distance2 = find_closest_triangles_same_winding(optimizable_vertices2, optimizeable_winding_angles2, winding_angles1, triangles1, scene1, default_splitter) # same here

    optimizable_distance1_print = np.array([d for d in optimizable_distance1 if d < float("inf")]) # remove non-intersecting vertices
    optimizable_distance2_print = np.array([d for d in optimizable_distance2 if d < float("inf")]) # remove non-intersecting vertices
    print(f"Distances 1: {np.sum(optimizable_distance1_print, axis=0)}, Distances 2: {np.sum(optimizable_distance2_print, axis=0)}")

    # Start to update the masks
    optimizable_mask1 = optimizable_distance1 > distance_threshold
    optimizable_mask2 = optimizable_distance2 > distance_threshold

    count_optimizable = 0
    # move the vertices in the direction of the normal
    for i in range(len(optimizable_direction1)):
        # if not optimizable_mask1[i]:
        #     continue
        if optimizable_distance1[i] == float("inf"): # no intersection
            optimizable_mask1[i] = False
            continue
        # Skip randomly 
        if np.random.rand() > 0.5:
            continue
        assert optimizable_distance1[i] != float("inf"), f"Distance is infinite for vertex {i}"
        opt_d = min(10, optimizable_distance1[i] / 2.0)
        optimizable_vertices1[i] += optimizable_direction1[i] * opt_d
        count_optimizable += 1
    for i in range(len(optimizable_direction2)):
        # if not optimizable_mask2[i]:
        #     continue
        if optimizable_distance2[i] == float("inf"): # no intersection
            optimizable_mask2[i] = False
            continue
        # Skip randomly
        if np.random.rand() > 0.5:
            continue
        assert optimizable_distance2[i] != float("inf"), f"Distance is infinite for vertex {i}"
        opt_d = min(10, optimizable_distance2[i] / 2.0)
        optimizable_vertices2[i] += optimizable_direction2[i] * opt_d
        count_optimizable += 1

    print(f"Number of optimizable vertices: {count_optimizable}")

    # update the vertices
    vertices1[mesh1_mask] = optimizable_vertices1
    vertices2[mesh2_mask] = optimizable_vertices2
    # vertices1[:] = optimizable_vertices1[:]
    # vertices2[:] = optimizable_vertices2[:]

    # Finish updating the masks
    mesh1_mask[mesh1_mask] = optimizable_mask1
    mesh2_mask[mesh2_mask] = optimizable_mask2
    # mesh1_mask[:] = optimizable_mask1[:]
    # mesh2_mask[:] = optimizable_mask2[:]

    # Update the meshes and the scenes
    mesh1.vertices = o3d.utility.Vector3dVector(vertices1)
    mesh2.vertices = o3d.utility.Vector3dVector(vertices2)

    scene1 = setup_closest_triangles(mesh1)
    scene2 = setup_closest_triangles(mesh2)

    return scene1, scene2, vertices1, vertices2

def calculate_overlapping(secondary_vertices, winding_angles_secondary, winding_angles_primary, triangles_primary, scene_primary, default_splitter):
    optimizable_direction1, optimizable_distance1 = find_closest_triangles_same_winding(secondary_vertices, winding_angles_secondary, winding_angles_primary, triangles_primary, scene_primary, default_splitter)
    overlapping_vertices_secondary_mask = optimizable_distance1 < 10
    return overlapping_vertices_secondary_mask

def get_closest_boundary_vertex_index(tree, vertices1, winding_angles1, vertex2, winding_angle2, winding_angle_threshold=45):
    # Get closest and second closest vertices index
    distances, indices = tree.query(vertex2, k=20)

    # Filter for vertices with similar winding angles
    mask = np.abs(winding_angles1[indices] - winding_angle2) < winding_angle_threshold
    distances = distances[mask]
    indices = indices[mask]

    if len(indices) < 1:
        return None, None
    if len(indices) == 1:
        return indices[0], None
    else:
        distance_closest = np.linalg.norm(vertices1[indices[0]] - vertex2)
        distance_second_closest = np.linalg.norm(vertices1[indices[1]] - vertex2)
        # print(f"Distance closest: {int(distance_closest):6}, Distance second closest: {int(distance_second_closest):6}")
        return indices[0], indices[1]

def generate_stitching_triangles(primary_vertices, secondary_vertices, secondary_triangles, winding_angles1, winding_angles2, overlapping_vertices_secondary_mask, secondary_triangles_mask_partially_overlapping, secondary_vertices_index_offset):
    # Generate KD tree of primary vertices
    tree = KDTree(primary_vertices)

    print(f"Vertices index offset: {secondary_vertices_index_offset}")
    # For each partially overlapping secondary triangle, move the overlapping points to the closest boundary vertex of the primary mesh
    stitching_triangles = []
    secondary_triangles_masked = secondary_triangles[secondary_triangles_mask_partially_overlapping]
    for secondary_triangle in secondary_triangles_masked:
        # Find the closest two boundary vertices for each partially overlapping triangle
        triangle_vertices = secondary_vertices[secondary_triangle]
        triangle_vertices_overlap = overlapping_vertices_secondary_mask[secondary_triangle]
        # Find the closest boundary vertices
        triangle_winding_angles = winding_angles2[secondary_triangle]
        first_boundary_vertex = None
        second_boundary_vertex = None
        at_least_once_non_overlapping = False
        at_least_once_overlapping = False
        for j in range(3):
            if triangle_vertices_overlap[j]:
                at_least_once_overlapping = True
                closest_boundary_vertex_index, second_closest_boundary_vertex_index = get_closest_boundary_vertex_index(tree, primary_vertices, winding_angles1, triangle_vertices[j], triangle_winding_angles[j])
                if closest_boundary_vertex_index is None:
                    print(f"No closest boundary vertex found for vertex {j} in triangle {secondary_triangle}")
                    at_least_once_non_overlapping = True
                    at_least_once_overlapping = True
                    first_boundary_vertex = second_boundary_vertex = None
                    break
                triangle_vertices[j] = primary_vertices[closest_boundary_vertex_index]
                secondary_triangle[j] = closest_boundary_vertex_index
                if first_boundary_vertex is None:
                    first_boundary_vertex = closest_boundary_vertex_index
                else:
                    second_boundary_vertex = closest_boundary_vertex_index
            else:
                secondary_triangle[j] += secondary_vertices_index_offset
                overlapping_vertex_index = secondary_triangle[j]
                at_least_once_non_overlapping = True
        assert at_least_once_non_overlapping, f"No non-overlapping vertices in triangle {secondary_triangle}"
        assert at_least_once_overlapping, f"No overlapping vertices in triangle {secondary_triangle}"
        # Check if the triangle has zero area
        if first_boundary_vertex != second_boundary_vertex:
            # Update the triangle vertices
            stitching_triangles.append(deepcopy(secondary_triangle))
            at_least_once_greater_offset = False
            at_least_once_smaller_offset = False
            for j in range(3):
                if secondary_triangle[j] >= secondary_vertices_index_offset:
                    at_least_once_greater_offset = True
                else:
                    at_least_once_smaller_offset = True
            assert at_least_once_greater_offset and at_least_once_smaller_offset, f"No offset in triangle {secondary_triangle}"
        if second_boundary_vertex is None and first_boundary_vertex is not None and second_closest_boundary_vertex_index is not None:
            # add a second triangle with first boundary vertex and second_closest_boundary_vertex_index and overlapping_vertex
            stitching_triangles.append([first_boundary_vertex, second_closest_boundary_vertex_index, overlapping_vertex_index])
 
    return np.array(stitching_triangles, dtype=np.int32)

def merge_meshes(path, primary_mesh, secondary_mesh, points_indices1, points_indices2, winding_angles1, winding_angles2, scene1, scene2, default_splitter):
    fresh_start = True
    if fresh_start:
        primary_vertices = np.asarray(primary_mesh.vertices)
        primary_normals = np.asarray(primary_mesh.vertex_normals)
        primary_triangles = np.asarray(primary_mesh.triangles)
        secondary_vertices = np.asarray(secondary_mesh.vertices)
        secondary_normals = np.asarray(secondary_mesh.vertex_normals)
        secondary_triangles = np.asarray(secondary_mesh.triangles)
        # Iteratively adjust vertices positions of both meshes in the overlapping region
        mesh1_mask = np.ones(len(primary_vertices), dtype=bool)
        mesh2_mask = np.ones(len(secondary_vertices), dtype=bool)
        while np.any(mesh1_mask) or np.any(mesh2_mask):
            # Iteratively refine vertices positions of both meshes in the overlapping region
            scene1, scene2, primary_vertices, secondary_vertices = adjust_vertices_positions(mesh1_mask, mesh2_mask, primary_mesh, secondary_mesh, primary_vertices, secondary_vertices, primary_normals, secondary_normals, winding_angles1, winding_angles2, primary_triangles, secondary_triangles, scene1, scene2, default_splitter, distance_threshold=0.1)
            print(f"Number of vertices in mesh1: {np.sum(mesh1_mask)}, Number of vertices in mesh2: {np.sum(mesh2_mask)}")

        # Get for each vertex in the secondary mesh if it is overlapping with the primary mesh
        overlapping_vertices_secondary_mask = calculate_overlapping(secondary_vertices, winding_angles2, winding_angles1, primary_triangles, scene1, default_splitter)

        # Remove all triangles from the secondary mesh that are completely overlaping with the primary mesh
        secondary_triangles_mask_completely_overlapping = np.all(overlapping_vertices_secondary_mask[secondary_triangles], axis=-1)

        # Calculate the partially overlapping trianges of the secondary mesh
        secondary_triangles_mask_partially_overlapping = np.logical_and(np.any(overlapping_vertices_secondary_mask[secondary_triangles], axis=-1), ~secondary_triangles_mask_completely_overlapping)

        # pickle save
        with open(os.path.join(path, "development2.pkl"), "wb") as f:
            pickle.dump((primary_vertices, primary_normals, primary_triangles, secondary_vertices, secondary_normals, secondary_triangles, overlapping_vertices_secondary_mask, secondary_triangles_mask_partially_overlapping, secondary_triangles_mask_completely_overlapping), f)
    else:
        with open(os.path.join(path, "development2.pkl"), "rb") as f:
            primary_vertices, primary_normals, primary_triangles, secondary_vertices, secondary_normals, secondary_triangles, overlapping_vertices_secondary_mask, secondary_triangles_mask_partially_overlapping, secondary_triangles_mask_completely_overlapping = pickle.load(f)

    assert primary_vertices.shape == np.asarray(primary_mesh.vertices).shape, f"Shape of primary vertices {primary_vertices.shape} is different from shape of primary mesh vertices {np.asarray(primary_mesh.vertices).shape}"
    assert secondary_vertices.shape == np.asarray(secondary_mesh.vertices).shape, f"Shape of secondary vertices {secondary_vertices.shape} is different from shape of secondary mesh vertices {np.asarray(secondary_mesh.vertices).shape}"
    print(f"shape primary vertices: {primary_vertices.shape}")

    # For each partially overlapping triangle of the secondary mesh create new triangles between the closest two boundary vertices of the primary mesh
    secondary_vertices_index_offset = primary_vertices.shape[0]
    stitching_triangles = generate_stitching_triangles(primary_vertices, secondary_vertices, np.array(secondary_mesh.triangles), winding_angles1, winding_angles2, overlapping_vertices_secondary_mask, secondary_triangles_mask_partially_overlapping, secondary_vertices_index_offset)
    # stitching_triangles = np.zeros((0, 3), dtype=np.int32) # Debug
    print(f"Adding {stitching_triangles.shape} stitching triangles to the combined mesh")
    print(f"Shape triangles of primary mesh: {primary_triangles.shape}, Shape triangles of secondary mesh: {secondary_triangles.shape}, Shape stitching triangles: {stitching_triangles.shape}")
    print(f"Dtype triangles: {np.array(secondary_mesh.triangles).dtype}, Dtype stitching triangles: {stitching_triangles.dtype}")

    # Add all vertices and triangles into the same mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_vertices = np.concatenate([primary_vertices, secondary_vertices], axis=0)
    combined_normals = np.concatenate([np.array(primary_mesh.vertex_normals), np.array(secondary_mesh.vertex_normals)], axis=0)
    combined_triangles = np.concatenate([np.array(primary_mesh.triangles), secondary_vertices_index_offset + np.array(secondary_mesh.triangles), stitching_triangles], axis=0)
    print(f"Triangles shape: {combined_triangles.shape}, Vertices shape: {combined_vertices.shape}")
    combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    combined_mesh.vertex_normals = o3d.utility.Vector3dVector(combined_normals)
    combined_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)

    # Extend the secondary triangles masks to the combined mesh
    combined_triangles_mask_partially_overlapping = np.concatenate([np.zeros(len(primary_triangles), dtype=bool), secondary_triangles_mask_partially_overlapping, np.zeros(len(stitching_triangles), dtype=bool)], axis=0)
    combined_triangles_mask_completely_overlapping = np.concatenate([np.zeros(len(primary_triangles), dtype=bool), secondary_triangles_mask_completely_overlapping, np.zeros(len(stitching_triangles), dtype=bool)], axis=0)

    # Remove each partially and completely overlapping triangle of the secondary mesh
    combined_triangles_mask = np.logical_or(combined_triangles_mask_partially_overlapping, combined_triangles_mask_completely_overlapping)
    combined_mesh.remove_triangles_by_mask(combined_triangles_mask)

    # Remove all vertices that are not part of the triangles
    combined_mesh.remove_unreferenced_vertices()

    # Find non manifold vertices and faces
    print(f"Number of non manifold vertices: {len(combined_mesh.get_non_manifold_vertices())}")
    print(f"Number of non manifold edges: {len(combined_mesh.get_non_manifold_edges())}")
    # Remove non manifold vertices
    combined_mesh.remove_non_manifold_edges()
    print(f"Mesh is edge manifold: {combined_mesh.is_edge_manifold()}")
    print(f"Mesh is vertex manifold: {combined_mesh.is_vertex_manifold()}")

    # Print stats of the combined mesh
    print(f"Number of vertices in the combined mesh: {len(combined_mesh.vertices)}")
    print(f"Number of triangles in the combined mesh: {len(combined_mesh.triangles)}")
    # check
    index_c, count_c, area_c = combined_mesh.cluster_connected_triangles()
    print(f"Found {len(area_c)} clusters.")

    # Only take the largest cluster
    largest_cluster = np.argmax(count_c)
    mask = index_c == largest_cluster
    combined_mesh.remove_triangles_by_mask(~mask)
    combined_mesh.remove_unreferenced_vertices()

    # Return the combined mesh
    return combined_mesh

def merge(mesh_path1, mesh_path2, umbilicus_path, max_distance, output_dir, save_name):
    print(f"Merging {mesh_path1} and {mesh_path2}")
    default_splitter = MeshSplitter(mesh_path1, umbilicus_path)
    fresh_start = True
    if fresh_start:
        winding_angle_difference, winding_angles1, winding_angles2, points_indices1, points_indices2, scene1, scene2, mesh1, mesh2 = align_meshes(mesh_path1, mesh_path2, umbilicus_path, max_distance, default_splitter)
        best_alignment = find_best_alignment(winding_angle_difference)
        print(f"Best alignment: {best_alignment}")

        # pickle save 
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "development.pkl"), "wb") as f:
            pickle.dump((winding_angle_difference, winding_angles1, winding_angles2, points_indices1, points_indices2, best_alignment), f)
    else:
        with open(os.path.join(output_dir, "development.pkl"), "rb") as f:
            winding_angle_difference, winding_angles1, winding_angles2, points_indices1, points_indices2, best_alignment = pickle.load(f)

            mesh1, _, _, scene1 = load_mesh_vertices(mesh_path1)
            mesh2, _, _, scene2 = load_mesh_vertices(mesh_path2)

    # Adjust the winding angles of the second mesh to have the same winding angle base as the first mesh
    print(f"Best alignment: {best_alignment}")
    winding_angles2 -= best_alignment

    print("Mesh 1 min max winding angle: ", np.min(winding_angles1), np.max(winding_angles1))
    print("Mesh 2 min max winding angle: ", np.min(winding_angles2), np.max(winding_angles2))

    merged_mesh = merge_meshes(output_dir, mesh1, mesh2, points_indices1, points_indices2, winding_angles1, winding_angles2, scene1, scene2, default_splitter)

    # Save the merged mesh
    save_mesh_path = os.path.join(output_dir, f"{save_name}.obj")
    save_mesh_ = deepcopy(merged_mesh)
    save_mesh(save_mesh_path, save_mesh_)

    # Delete all vertices not in the upermost 1500 units
    vertices = np.asarray(merged_mesh.vertices)
    max_z = np.max(vertices[:, 1])
    mask = vertices[:, 1] < max_z - 1500
    top_part_mesh = deepcopy(merged_mesh)
    top_part_mesh.remove_vertices_by_mask(mask)

    # Save the merged mesh
    save_mesh_path = os.path.join(output_dir, f"{save_name}_slab.obj")
    save_mesh(save_mesh_path, top_part_mesh)

    # Bottom part mesh
    vertices = np.asarray(merged_mesh.vertices)
    min_z = np.min(vertices[:, 1])
    mask = vertices[:, 1] > min_z + 2500
    bottom_part_mesh = deepcopy(merged_mesh)
    bottom_part_mesh.remove_vertices_by_mask(mask)

    # Save the merged mesh
    save_mesh_path = os.path.join(output_dir, f"{save_name}_slab_bottom.obj")
    save_mesh(save_mesh_path, bottom_part_mesh)

def compute(mesh_dir, umbilicus_path, max_distance):
    output_dir = mesh_dir + "_merged"
    stitching_order_file = os.path.join(mesh_dir, "stitching_order.txt")
    stitching_order = load_stitching_order(stitching_order_file)

    # mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    # # Sort the mesh files
    # mesh_files.sort()
    mesh_files = [os.path.join(mesh_dir, s) for s in stitching_order]
    new_mesh_files = []
    skip_big_nr = 1

    while len(mesh_files) > 1:
        for i in range(len(mesh_files)):
            if i % 2 == 0:
                if i + 1 < len(mesh_files):
                    mesh_path1 =mesh_files[i]
                    mesh_path2 = mesh_files[i+1]
                    save_name = os.path.basename(mesh_files[i]).split(".")[0] + "-" + os.path.basename(mesh_files[i+1]).split(".")[0]
                    if skip_big_nr <= 0:
                        merge(mesh_path1, mesh_path2, umbilicus_path, max_distance, output_dir, save_name)
                    new_mesh_files.append(os.path.join(output_dir, f"{save_name}.obj"))
                else:
                    new_mesh_files.append(mesh_files[i])
        mesh_files = new_mesh_files
        new_mesh_files = []
        skip_big_nr -= 1

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert a 3D mesh and a PointCloud to 3D PointCloud instance labels.")
    parser.add_argument("--mesh_dir", type=str, help="Path to the 3D mesh directory (containing .obj)")
    parser.add_argument("--umbilicus_path", type=str, help="Path to the 3D point cloud directory (containing .ply)")
    parser.add_argument("--max_distance", type=float, help="Maximum distance for a point to be considered part of the mesh", default=1)

    args = parser.parse_args()

    # Compute the 3D mask with labels
    compute(args.mesh_dir, args.umbilicus_path, args.max_distance)


if __name__ == "__main__":
    main()

# Example: python3 -m ThaumatoAnakalyptor.mesh_merger --mesh_dir /scroll.volpkg/merging_test --umbilicus_path /scroll.volpkg/merging_test/umbilicus.txt