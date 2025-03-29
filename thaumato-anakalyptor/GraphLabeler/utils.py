import numpy as np
import open3d as o3d

def vectorized_point_to_polyline_distance(point, polyline):
    """
    Compute the minimum distance from a point (2,) to a polyline.
    Uses vectorized operations.
    """
    p1 = polyline[:-1]
    p2 = polyline[1:]
    v = p2 - p1
    w = point - p1
    dot_wv = np.einsum('ij,ij->i', w, v)
    dot_vv = np.einsum('ij,ij->i', v, v)
    t = np.divide(dot_wv, dot_vv, out=np.zeros_like(dot_wv), where=dot_vv != 0)
    t = np.clip(t, 0, 1)
    proj = p1 + (t[:, None] * v)
    dists = np.linalg.norm(point - proj, axis=1)
    return np.min(dists)

def build_temporary_group_edges(solver, labels, group_mask):
    # adds randomly for each node 3 edges in the group. then prunes each node down to max 6 added edges.
    n_edges = 6
    max_node_added_edges = 12

    len_group = np.sum(group_mask)
    group_indices = np.where(group_mask)[0]
    random_targets = np.random.choice(len_group, size=(len_group, n_edges), replace=True)
    # bookkeeping nr added edges per node, count with np.bincount
    unique, count_unique = np.unique(random_targets, return_counts=True)
    to_add_edges_per_node = np.full(len_group, n_edges)
    to_add_edges_per_node[unique] = max_node_added_edges - count_unique
    to_add_edges_per_node = np.clip(to_add_edges_per_node, 0, n_edges)
    sum_total_possible_edges = len_group * max_node_added_edges
    sum_edded_edges = np.sum(to_add_edges_per_node)
    print(f"Adding {sum_edded_edges} edges out of {sum_total_possible_edges} potentially possible edges.")
    # add edges
    source = []
    target = []
    winding_number_difference = []
    for i, node in enumerate(group_indices):
        source_ = node
        for j in range(to_add_edges_per_node[i]):
            target_ = group_indices[random_targets[i, j]]
            if target_ == source_:
                continue
            wnr_d_ = labels[target_] - labels[source_]
            source.append(source_)
            target.append(target_)
            winding_number_difference.append(int(wnr_d_))
    print(f"Length of source: {len(source)}, Length of target: {len(target)}, Length of winding_number_difference: {len(winding_number_difference)}")

    solver.add_temporary_fixed_edges(source=source, target=target, winding_number_difference=winding_number_difference)

    return source, target, winding_number_difference

def filter_point_normals(points: np.ndarray, filter_normal: np.ndarray, angle_threshold: float,
                         radius: float = 10.0, max_nn: int = 30):
    """
    Given an array of 3D points, estimate normals using Open3D, and filter out points
    whose normals deviate from a given filter_normal by more than angle_threshold (in degrees).

    Parameters:
    - points (np.ndarray): Input array of shape (N, 3) representing 3D points.
    - filter_normal (np.ndarray): A 3-element array representing the target normal. Should be non-zero.
    - angle_threshold (float): The maximum allowed angular difference in degrees.
    - radius (float): Radius for neighborhood search when estimating normals (default: 0.1).
    - max_nn (int): Maximum number of nearest neighbors to consider (default: 30).

    Returns:
    - filtered_points (np.ndarray): Subset of input points that pass the normal filter.
    - filtered_normals (np.ndarray): Corresponding normals of the filtered points.
    """
    # Create a point cloud from the input points.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals using a KDTree search with the specified parameters.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    # Convert the estimated normals to a NumPy array.
    normals = np.asarray(pcd.normals)
    
    # Normalize the filter_normal to ensure accurate angle calculations.
    filter_normal = filter_normal / np.linalg.norm(filter_normal)
    
    # Compute the dot product between each normal and the filter_normal,
    # taking the absolute value so that flipped normals are considered equivalent.
    dots = np.clip(np.abs(np.dot(normals, filter_normal)), 0.0, 1.0)
    angles = np.arccos(dots)  # Angles in radians
    angles_deg = np.degrees(angles)  # Convert to degrees
    
    # Create a mask for normals within the specified angular threshold.
    mask = angles_deg <= angle_threshold
    
    # Filter the original points and their corresponding normals.
    filtered_points = points[mask]
    print(f"Filtered points shape: {filtered_points.shape} vs original points shape: {points.shape}")
    
    return ~mask