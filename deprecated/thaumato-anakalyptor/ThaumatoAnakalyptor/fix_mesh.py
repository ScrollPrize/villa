### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import open3d as o3d
import numpy as np

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return mesh

def save_mesh(mesh, path):
    o3d.io.write_triangle_mesh(path, mesh)

def smooth_mesh(mesh, iterations=1):
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()
    return mesh

def find_degenerated_triangles_and_delete(mesh, threshold=0.001):
    vertices = np.asarray(mesh.vertices)
    unique_vertices, indices_inv, count = np.unique(vertices, axis=0, return_inverse=True, return_counts=True)
    count_org = count[indices_inv]
    print(count_org.shape, vertices.shape)
    print(f"Found {len(unique_vertices)} unique vertices, max count is {np.max(count)}")
    triangles = np.asarray(mesh.triangles)
    indices_to_remove = []
    for indx, triangle in enumerate(triangles):
        idx1_pos = vertices[triangle[0]]
        idx2_pos = vertices[triangle[1]]
        idx3_pos = vertices[triangle[2]]

        idx1_bad = abs(np.linalg.norm(idx1_pos - idx2_pos)) < threshold
        idx2_bad = abs(np.linalg.norm(idx2_pos - idx3_pos)) < threshold
        idx3_bad = abs(np.linalg.norm(idx3_pos - idx1_pos)) < threshold
        if idx1_bad or idx2_bad or idx3_bad:
            print(f"Found degenerated triangle {indx} with vertices {triangle}")
            indices_to_remove.append(indx)

    # find all triangles containing vertices with count > 1
    mask_count_greater_one = count_org[triangles]
    print(mask_count_greater_one.shape)
    mask_bad_triangles_vertices = np.any(mask_count_greater_one > 1, axis=1)
    print(mask_bad_triangles_vertices.shape)
    trinagles_bad_indices = np.where(mask_bad_triangles_vertices)[0]
    triangles_bad = triangles[mask_bad_triangles_vertices]
    print(triangles_bad.shape)

    indices_to_remove += list(trinagles_bad_indices)
    indices_to_remove = list(set(indices_to_remove))
    print(f"Removing {len(indices_to_remove)} triangles")
    mesh.remove_triangles_by_index(indices_to_remove)
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()

    area_factor_cm = (0.008 ** 2) / 100.0
    print(f"Surface Area is {mesh.get_surface_area() * area_factor_cm:.3f} cm^2")
    area = mesh.get_surface_area() * area_factor_cm
    nr_points = int(area * 100 * 60)
    print(f"Target Nr points is {nr_points}")

    print(f"Number of triangles is {len(mesh.triangles)} and number of vertices is {len(mesh.vertices)}")
    mesh = mesh.remove_non_manifold_edges()
    non_manifold_vertices = np.asarray(mesh.get_non_manifold_vertices())
    mesh.remove_vertices_by_index(non_manifold_vertices)
    mesh = mesh.remove_degenerate_triangles()

    # cluster triangles and find largest surface are cluster
    index_c, count_c, area_c = mesh.cluster_connected_triangles()
    area_c = np.asarray(area_c)
    index_c = np.asarray(index_c)
    print(f"Found {len(area_c)} clusters.")
    # get ind of largest area cluster
    ind_largest_cluster = np.argmax(area_c)
    mask_triangles = index_c == ind_largest_cluster
    mesh.remove_triangles_by_mask(~mask_triangles)
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    mesh.orient_triangles()

    return mesh

def main(path):
    mesh = load_mesh(path)
    mesh = find_degenerated_triangles_and_delete(mesh)
    save_mesh(mesh, path)

if __name__ == '__main__':
    path = "/home/julian/scroll1_surface_points/point_cloud_colorized_subvolume_blocks.obj"
    main(path)    