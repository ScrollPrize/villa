# Much of this file is a port of code originally found in the Open3D library,
# but changed to fit this project's purposes: https://github.com/isl-org/Open3D

import logging
from collections import deque
from typing import List, Set

import numpy as np

from pgs_recon.utils import wavefront as wobj


class Mesh:
    """Custom mesh class"""
    vertices: np.ndarray = None
    faces: np.ndarray = None
    normals: np.ndarray = None
    uv_coords: np.ndarray = None
    mtl_ids: np.ndarray = None


def wavefront_to_mesh(obj: wobj.WavefrontOBJ) -> Mesh:
    """Convert a Wavefront object to a Mesh"""
    mesh = Mesh()
    mesh.vertices = np.array(obj.vertices)
    mesh.faces = np.array(obj.polygons)
    mesh.normals = np.array(obj.normals)
    mesh.uv_coords = np.array(obj.texcoords)
    mesh.mtl_ids = np.array(obj.mtlid)
    return mesh


def mesh_to_wavefront(mesh: Mesh, obj: wobj.WavefrontOBJ = None):
    """Convert a Mesh to a Wavefront object. If obj is provided, its geometry
    will be modified."""
    if obj is None:
        obj = wobj.WavefrontOBJ()
    obj.vertices = [v.tolist() for v in mesh.vertices]
    obj.polygons = [[v.tolist() for v in f] for f in mesh.faces]
    obj.normals = [n.tolist() for n in mesh.normals]
    obj.texcoords = [uv.tolist() for uv in mesh.uv_coords]
    obj.mtlid = mesh.mtl_ids.tolist()

    return obj


def segment_plane(mesh, dist_threshold=0.1, point_samples=3, iterations=1000,
                  prob=0.99999999):
    class RANSACResult:
        inliers: List[int]
        error: float
        fitness: float
        inlier_rmse: float

        def __init__(self):
            self.inliers = []
            self.error = 0.
            self.fitness = 0.
            self.inlier_rmse = 0.

    def get_plane_from_points(pts: np.ndarray):
        """Fit a plane to a set of points"""
        if point_samples == 3:
            e0 = pts[1] - pts[0]
            e1 = pts[2] - pts[0]
            abc = np.cross(e0, e1)
            norm = np.linalg.norm(abc)
            if np.isclose(norm, 0.):
                return None
            abc /= norm
            d = -np.dot(abc, pts[0])
        else:
            centroid = np.mean(pts, axis=0)
            cov = np.cov(pts - centroid, rowvar=False)
            val, vec = np.linalg.eig(cov)
            abc = vec[:, np.argmin(val, axis=0)]
            d = -np.dot(abc, centroid)
        x = np.concatenate([abc, (d,)])
        return x

    def eval_from_distance(pts: np.ndarray, plane: np.ndarray,
                           threshold: float):
        """Evaluate plane fit against a set of points"""
        res = RANSACResult()
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        dist = np.einsum('j,ij->i', plane, pts)
        dist = np.abs(dist)
        idx = dist < threshold
        res.error = np.sum(dist[idx])
        res.inliers = idx.nonzero()[0].tolist()

        if len(res.inliers) > 0:
            res.fitness = len(res.inliers) / pts.shape[0]
            res.inlier_rmse = res.error / np.sqrt(len(res.inliers))

        return res

    # Setup outputs
    best_result = RANSACResult()
    best_model = np.zeros((4,))

    # Iterate up to some max iterations
    rng = np.random.default_rng()
    break_it = iterations
    for i in range(iterations):
        # Break early based on fitness/rmse
        if i > break_it:
            break

        # Fit a plane to N random vertices
        samples = rng.choice(mesh.vertices, point_samples, replace=False)
        model = get_plane_from_points(samples)

        # Skip if the model calculation failed
        if model is None:
            continue

        # Evaluate the model against the entire pointset
        result = eval_from_distance(mesh.vertices, model, dist_threshold)

        # Update our best results if the fitness/rmse are better
        better_fitness = result.fitness > best_result.fitness
        better_rmse = result.fitness == best_result.fitness and result.inlier_rmse < best_result.inlier_rmse
        if better_fitness or better_rmse:
            best_result = result
            best_model = model
            if result.fitness < 1:
                break_it = min(iterations, np.log(1 - prob) / np.log(
                    1 - result.fitness ** point_samples))
            else:
                break

    # get final inlier set
    final_result = eval_from_distance(mesh.vertices, best_model, dist_threshold)
    # update the model using all inliers
    best_model = get_plane_from_points(mesh.vertices[final_result.inliers])

    return best_model, final_result.inliers


def index_to_boolean_mask(mask, shape):
    """Convert an index mask to a boolean mask"""
    bool_mask = np.zeros(shape, dtype=bool)
    bool_mask[mask] = True
    return bool_mask


def keep_vertices_by_mask(mesh: Mesh, mask):
    # get the vertices
    len_v = mesh.vertices.shape[0]
    mesh.vertices = mesh.vertices[mask, ...]

    # keep faces which don't reference the removed vertices
    tri_mask = mask[mesh.faces[..., 0].astype(int)]
    tri_mask = np.all(tri_mask, axis=-1)
    keep_triangles_by_mask(mesh, tri_mask)

    # LUT for (vid + None,) -> new_vid
    # Unlike other mesh properties, vid's should never be None
    lut = np.full((len_v,), -1.)
    new_idx = np.arange(mask.nonzero()[0].shape[0], dtype=int)
    lut[mask] = new_idx
    v_map = mesh.faces[..., 0].astype(int)
    lut.take(v_map, out=v_map)
    v_map = v_map.astype('O')
    mesh.faces[..., 0] = v_map


def remove_vertices_by_index(mesh: Mesh, index_mask):
    # convert the mask to a boolean mask
    mask = index_to_boolean_mask(index_mask, shape=mesh.vertices.shape[0:1])
    # keep vertices not in the mask
    keep_vertices_by_mask(mesh, np.invert(mask))


def remove_unreferenced_vertices(mesh: Mesh):
    # get a list of unique vertices attached to faces
    index_mask = list(set(mesh.faces[..., 0].flatten().tolist()))
    # convert to a boolean mask
    mask = index_to_boolean_mask(index_mask, mesh.vertices.shape[0:1])
    # keep these
    keep_vertices_by_mask(mesh, mask)


def keep_triangles_by_mask(mesh: Mesh, mask):
    # Remove triangles
    faces = mesh.faces[mask, ...]

    # Remove mtl ids
    if mesh.mtl_ids.shape[0] > 0:
        mesh.mtl_ids = mesh.mtl_ids[mask, ...]

    # Filter out unreferenced uvs/normals and update refs
    for a in (1, 2):
        # get the remaining set of unique uv/normal indices
        a_map = faces[..., a]
        a_mask = a_map[a_map != None]
        a_mask = list(set(a_mask.tolist()))
        # keep only the referenced indices
        if a == 1:
            len_a = mesh.uv_coords.shape[0]
            mesh.uv_coords = mesh.uv_coords[a_mask, ...]
        else:
            len_a = mesh.normals.shape[0]
            mesh.normals = mesh.normals[a_mask, ...]
        # Construct a LUT for (id + None,) -> new_id
        lut = np.full((len_a + 1,), -1, dtype=int)
        new_idx = np.arange(len(a_mask), dtype=int)
        lut[a_mask] = new_idx
        # Replace None with -1. Will map to -1 at end of LUT
        a_map[a_map == None] = -1
        # Apply the LUT
        a_map = a_map.astype(int)
        lut.take(a_map, out=a_map)
        # Convert back to None
        a_map = a_map.astype('O')
        a_map[a_map == -1] = None
        # Assign to mesh
        faces[..., a] = a_map

    # Update the mesh
    mesh.faces = faces


def order_edge(a, b):
    return min(a, b), max(a, b)


def generate_edge_map(mesh: Mesh):
    edge_map = {}

    def add_edge(a, b, idx):
        e = order_edge(a, b)
        if e in edge_map.keys():
            edge_map[e].append(idx)
        else:
            edge_map[e] = [idx]

    for i, f in enumerate(mesh.faces[..., 0]):
        add_edge(f[0], f[1], i)
        add_edge(f[1], f[2], i)
        add_edge(f[2], f[0], i)
    return edge_map


def get_face_area(f_id, mesh: Mesh):
    vid0, vid1, vid2 = mesh.faces[f_id, ..., 0]
    v0 = mesh.vertices[vid0]
    v1 = mesh.vertices[vid1]
    v2 = mesh.vertices[vid2]
    a = v0 - v1
    b = v0 - v2
    return 0.5 * np.linalg.norm(np.cross(a, b))


def cluster_connected_components(mesh: Mesh):
    logger = logging.getLogger(__name__)
    logger.info('Computing connected components...')
    # Compute adjacency
    logger.debug('Computing adjacency map...')
    adjacency: List[Set[int]] = []
    edge_map = generate_edge_map(mesh)
    for i, f in enumerate(mesh.faces[..., 0]):
        f_adj = set()
        ab = edge_map[order_edge(f[0], f[1])]
        bc = edge_map[order_edge(f[1], f[2])]
        ca = edge_map[order_edge(f[2], f[0])]
        adjacency.append(f_adj.union(ab, bc, ca))

    logger.debug('Clustering triangles...')
    face_cluster = [None, ] * mesh.faces.shape[0]
    cluster_metrics = []
    c_idx = 0
    for i, f in enumerate(mesh.faces[..., 0]):
        # skip this face if its already clustered
        if face_cluster[i] is not None:
            continue

        # setup outputs for new cluster
        cluster_faces = []
        area = 0.

        # iterate over adjacent faces
        queue = deque([i])
        face_cluster[i] = c_idx
        while queue:
            qf_i = queue.popleft()
            cluster_faces.append(qf_i)
            area += get_face_area(qf_i, mesh)

            for n in adjacency[qf_i]:
                if face_cluster[n] is None:
                    queue.append(n)
                    face_cluster[n] = c_idx

        cluster_metrics.append({'area': area, 'faces': cluster_faces})
        c_idx += 1

    return face_cluster, cluster_metrics


def keep_largest_connected_component(mesh: Mesh, filter_vertices=False):
    _, metrics = cluster_connected_components(mesh)
    metrics = sorted(metrics, key=lambda m: m['area'], reverse=True)
    keep_triangles_by_mask(mesh, metrics[0]['faces'])
    if filter_vertices:
        remove_unreferenced_vertices(mesh)


def remove_degenerate_faces(mesh: Mesh):
    # get list of good faces
    mask = []
    start = mesh.faces.shape[0]
    for idx, f in enumerate(mesh.faces):
        if f[0] != f[1] != f[2] != f[0]:
            mask.append(idx)
    keep_triangles_by_mask(mesh, mask)
    end = mesh.faces.shape[0]
    print(f'Removed {start - end} degenerate faces.')
