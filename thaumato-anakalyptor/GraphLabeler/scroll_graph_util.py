import numpy as np
from scipy.interpolate import interp1d
import h5py
from tqdm import tqdm
import sys

########################################
# Utility Functions
########################################

def load_xyz_from_file(filename='umbilicus.txt'):
    """
    Load a file with comma-separated xyz coordinates into a 2D numpy array.
    """
    return np.loadtxt(filename, delimiter=',')

def umbilicus_xy_at_z(points_array, z_val):
    """
    Given umbilicus data in the order (y, z, x), interpolate using the z values
    to obtain the corresponding (x, y) coordinate for the current z.
    
    :param points_array: A 2D numpy array of shape (n, 3) with columns (y, z, x).
    :param z_val: The z value at which to interpolate.
    :return: A 1D numpy array [x, y] for the rotation center.
    """
    Y = points_array[:, 0]
    Z = points_array[:, 1]
    X = points_array[:, 2]
    fy = interp1d(Z, Y, kind='linear', fill_value="extrapolate")
    fx = interp1d(Z, X, kind='linear', fill_value="extrapolate")
    return np.array([fx(z_val), fy(z_val)])

def umbilicus_xy_at_z_vector(points_array, z_vals):
    """
    Given umbilicus data in the order (y, z, x), interpolate using the z values
    to obtain the corresponding (x, y) coordinates for each z in z_vals.

    Parameters:
        points_array (np.ndarray): A 2D numpy array of shape (n, 3) with columns (y, z, x).
        z_vals (array-like): A 1D array of z values at which to interpolate.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(z_vals), 2) where each row is [x, y]
                    corresponding to the interpolated coordinates for the given z value.
    """
    # Extract columns: Y, Z, X from the points_array.
    Y = points_array[:, 0]
    Z = points_array[:, 1]
    X = points_array[:, 2]
    
    # Build interpolation functions for X and Y.
    fx = interp1d(Z, X, kind='linear', fill_value="extrapolate", assume_sorted=True)
    fy = interp1d(Z, Y, kind='linear', fill_value="extrapolate", assume_sorted=True)
    
    # Evaluate the functions at each z value in z_vals.
    x_interp = fx(z_vals)
    y_interp = fy(z_vals)
    
    # Combine the interpolated values into a 2D array with columns [x, y].
    return np.column_stack([x_interp, y_interp])

def compute_mean_windings_from_partial(inverse_indices, winding, unlabeled):
    print(f"All shapes: inverse_indices: {inverse_indices.shape}, winding: {winding.shape}")

    mask_labeled = np.abs(winding / 360 - unlabeled) > 2

    # Compute the number of occurrences for each unique coordinate.
    count_winding = np.bincount(inverse_indices, weights=mask_labeled)

    # Compute the sum of winding angles for each unique coordinate.
    sum_winding = np.bincount(inverse_indices, weights=winding * mask_labeled)
    
    # Calculate the mean winding angle by dividing the sum by the count.
    mean_winding = sum_winding / count_winding
    mask_0 = mean_winding == 0
        
    # Compute the first occurrence for each unique group.
    # np.unique returns the sorted unique group labels and the corresponding first indices.
    groups, first_indices = np.unique(inverse_indices, return_index=True)
    first_winding = winding[first_indices]

    # Adjust the computed mean:
    # Use the base from the first occurrence (i.e. floor(first_winding/360)*360)
    # and add the remainder of the computed mean (i.e. mean_winding % 360).
    adjusted_mean_winding = np.floor(mean_winding / 360) * 360 + (first_winding % 360)
    # Set the entries without any labels to exactly 0 for later displaying them as white (0 is white indicator)
    adjusted_mean_winding[mask_0] = 0.0
    
    return adjusted_mean_winding

def compute_mean_windings_precomputed(inverse_indices, winding, winding_computed, unlabeled):
    mean_winding = compute_mean_windings_from_partial(inverse_indices, winding, unlabeled)
    mean_winding_computed = compute_mean_windings_from_partial(inverse_indices, winding_computed, 0)
    
    return mean_winding, mean_winding_computed

def compute_mean_winding(coords, winding, winding_computed, unlabeled):
    # Extract the 3D coordinates and the winding angle values.
    # coords = np.round(points[:, :3])
    # winding = points[:, 3]
    coords = np.round(coords)

    # Get the unique 3D points and also obtain an array that can map each row of
    # the original array to its unique group.
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        
    mean_winding = compute_mean_windings_from_partial(inverse_indices, winding, unlabeled)
    mean_winding_computed = compute_mean_windings_from_partial(inverse_indices, winding_computed, 0)
    
    print(f"Shape of unique_coords: {unique_coords.shape}, shape of mean_winding: {mean_winding.shape}")
    print(f"Total mean winding values: {np.mean(mean_winding) / 360}, total mean winding computed values: {np.mean(mean_winding_computed) / 360}")

    return unique_coords, mean_winding, mean_winding_computed, inverse_indices, winding, winding_computed

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()

    def load_nodes(self, h5_filename, close_nodes, winding_angles_nodes, winding_angles_nodes_computed):
        nodes_points = []
        # Build a dictionary mapping each unique group name to a list of (surface_nr, index) tuples.
        groups_dict = {}
        for idx, close_node in enumerate(close_nodes):
            start_coord = close_node[:3]
            group_name = f"{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}"
            surface_nr = close_node[3]
            groups_dict.setdefault(group_name, []).append((surface_nr, idx))
        
        # Calculate the total number of surfaces to process.
        total_surfaces = sum(len(entries) for entries in groups_dict.values())

        point_nodes_indices = []
        
        # Open the HDF5 file once and use a tqdm progress bar for the surfaces.
        with h5py.File(h5_filename, "r") as h5f, tqdm(total=total_surfaces, desc="Loading nodes") as pbar:
            # Iterate over unique group names.
            for group_name, entries in groups_dict.items():
                if group_name not in h5f:
                    print(f"Group {group_name} not found in {h5_filename}")
                    pbar.update(len(entries))
                    continue
                grp = h5f[group_name]
                # Process each requested surface within this group.
                for surface_nr, idx in entries:
                    surface_name = f"surface_{surface_nr}"
                    if surface_name not in grp:
                        print(f"Surface {surface_name} not found in {h5_filename}")
                        pbar.update(1)
                        continue
                    surface = grp[surface_name]
                    try:
                        points = surface["points"][()]
                        if points.shape[1] != 3:
                            raise ValueError(f"Invalid points shape: {points.shape}")
                        
                        # Append the corresponding winding angle as a new column.
                        winding_angle = winding_angles_nodes[idx]
                        winding_angle_computed = winding_angles_nodes_computed[idx]
                        winding_col = np.full((points.shape[0], 1), winding_angle, dtype=np.float32)
                        winding_col_computed = np.full((points.shape[0], 1), winding_angle_computed, dtype=np.float32)
                        points = np.concatenate([points, winding_col, winding_col_computed], axis=1)
                        nodes_points.append(points)
                        point_nodes_indices.extend([idx for _ in range(points.shape[0])])
                    except Exception as e:
                        print(f"Error loading subvolume {group_name} patch {surface_nr} from {h5_filename}: {e}")
                    pbar.update(1)

        if nodes_points:
            points_all = np.concatenate(nodes_points, axis=0)
        else:
            points_all = np.empty((0, 4), dtype=np.float32)
        return points_all, np.array(point_nodes_indices)
    
    def load_nodes_graph(self, close_nodes, winding_angles_nodes, winding_angles_nodes_computed):
        """
        Backup that loads a reduced pointcloud from the .pkl file itself
        """
        nodes_points = []
        # Build a dictionary mapping each unique group name to a list of (surface_nr, index) tuples.
        groups_dict = {}
        for idx, close_node in enumerate(close_nodes):
            start_coord = close_node[:3]
            group_name = f"{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}"
            surface_nr = close_node[3]
            groups_dict.setdefault(group_name, []).append((surface_nr, idx))
        
        # Calculate the total number of surfaces to process.
        total_surfaces = sum(len(entries) for entries in groups_dict.values())

        point_nodes_indices = []
        
        # Use a tqdm progress bar for the surfaces.
        with tqdm(total=total_surfaces, desc="Loading nodes") as pbar:
            # Iterate over unique group names.
            for group_name, entries in groups_dict.items():
                # Process each requested surface within this group.
                for surface_nr, idx in entries:
                    node_key = close_nodes[idx]
                    try:
                        points = self.nodes[node_key]['sample_points']
                        if points.shape[1] != 3:
                            raise ValueError(f"Invalid points shape: {points.shape}")
                        
                        # Append the corresponding winding angle as a new column.
                        winding_angle = winding_angles_nodes[idx]
                        winding_angle_computed = winding_angles_nodes_computed[idx]
                        winding_col = np.full((points.shape[0], 1), winding_angle, dtype=np.float32)
                        winding_col_computed = np.full((points.shape[0], 1), winding_angle_computed, dtype=np.float32)
                        points = np.concatenate([points, winding_col, winding_col_computed], axis=1)
                        nodes_points.append(points)
                        point_nodes_indices.extend([idx for _ in range(points.shape[0])])
                    except Exception as e:
                        print(f"Error loading node {node_key} from the graph: {e}")
                    pbar.update(1)

        if nodes_points:
            points_all = np.concatenate(nodes_points, axis=0)
        else:
            points_all = np.empty((0, 4), dtype=np.float32)
        return points_all, np.array(point_nodes_indices)
    


    def get_points_XY(self, z_index, h5_filename, labels, computed_labels, f_init, undeleted_nodes_indices, unlabeled, block_size=200):
        """
        Get the points for the XY view at the given z_index.
        """
        use_h5 = h5_filename.endswith(".h5")
        if not use_h5:
            # If using subsampled points from graph, use more nodes for fuller pointcloud display
            block_size *= 2
        
        all_node_keys = list(self.nodes.keys())
        node_keys = [all_node_keys[i] for i in undeleted_nodes_indices]
        winding_angles_nodes = np.asarray(labels) * 360.0 + np.asarray(f_init)
        winding_angles_nodes_computed = np.asarray(computed_labels) * 360.0 + np.asarray(f_init)
        assert len(winding_angles_nodes) == len(node_keys), f"len(winding_angles_nodes)={len(winding_angles_nodes)} != len(all_node_keys)={len(node_keys)}"

        # check if the node is touching the z_index
        close_mask = np.abs(np.asarray([self.nodes[node_key]['centroid'][1] * 4.0 - 500 - z_index for node_key in node_keys])) < block_size
        close_nodes = [node_keys[i] for i in range(len(node_keys)) if close_mask[i]]
        close_angles = [winding_angles_nodes[i] for i in range(len(node_keys)) if close_mask[i]]
        close_angles_computed = [winding_angles_nodes_computed[i] for i in range(len(node_keys)) if close_mask[i]]

        print(f"Found {len(close_nodes)} close nodes at z_index {z_index} of total {len(node_keys)} undeleted nodes and {len(all_node_keys)} all nodes")

        # sort close nodes by key
        print(f"first few close nodes: {close_nodes[:min(5, len(close_nodes))]}")

        # Load points from these nodes.
        plane_point_filter_distance = 3
        if not use_h5:
            # Use backup graph node points
            points, point_nodes_indices = self.load_nodes_graph(close_nodes, close_angles, close_angles_computed)
            plane_point_filter_distance = 10
        else:
            points, point_nodes_indices = self.load_nodes(h5_filename, close_nodes, close_angles, close_angles_computed)

        print(f"Loaded {points.shape} points for XY view at z_index {z_index}")

        # scale points to scroll coordinates
        points[:, :3] = points[:, :3] * 4.0 - 500
        # swap axis
        points = points[:, [1, 0, 2, 3, 4]]

        # filter points in z slice
        print(f"Filtering points for XY view with distance {plane_point_filter_distance}")
        mask = np.abs(points[:, 0] - z_index) < plane_point_filter_distance
        points = points[mask]
        point_nodes_indices = point_nodes_indices[mask]

        print(f"Filtered {points.shape} points for XY view at z_index {z_index}")

        # sort points by ZXY
        points_sort_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]))
        points = points[points_sort_indices]
        point_nodes_indices = point_nodes_indices[points_sort_indices]

        # compute unique points with mean winding angle
        points, windings, windings_computed, inverse_indices, winding, winding_computed = compute_mean_winding(points[:, :3], points[:, 3], points[:, 4], unlabeled)

        print(f"Computed mean winding for {points.shape} unique 3D points for XY view at z_index {z_index}")

        return points, point_nodes_indices, windings, windings_computed, inverse_indices, winding, winding_computed, close_mask

    def get_points_XZ(self, f_target, umbilicus_data, h5_filename, labels, computed_labels, 
                        f_init, undeleted_nodes_indices, unlabeled, block_size=200):
        """
        Get the points for the XZ view on the umbilicus plane defined by the target angle f_target.

        This function first filters nodes by using their centroids: for each node,
        the (x,y) scroll coordinates are compared against the corresponding umbilicus center 
        (obtained via umbilicus_xy_at_z) using the distance along the direction normal to the 
        umbilicus plane (f_target). Only nodes whose centroids lie within block_size of the plane 
        are kept. Then, after loading the points for these nodes, each point is filtered based on 
        its (x,y) distance to the corresponding umbilicus center (with a tighter tolerance of 3 units).
        
        Finally, the point's position is updated so that columns 0 and 1 contain the XZ umbilicus 
        plane view coordinate. Here the new x coordinate is computed as the projection of the 
        (x,y) difference (point's (x,y) minus the umbilicus center) onto the tangent direction 
        (cos(f_target), sin(f_target)) and the z coordinate is retained.

        Parameters:
            f_target (float): Angle (in degrees) defining the target umbilicus plane.
            umbilicus_data (array-like): Data used to compute umbilicus centers (e.g. loaded from file and shifted by -500).
            h5_filename (str): Path to the HDF5 file with node data.
            labels (array-like): Current labels for each node.
            computed_labels (array-like): Computed labels for each node.
            f_init (array-like): Initial winding offset per node.
            undeleted_nodes_indices (array-like): Indices of undeleted nodes.
            unlabeled (numeric): Value that indicates an unlabeled node.
            block_size (int): Tolerance (in scroll coordinates) for selecting nodes based on centroid distance (default=200).

        Returns:
            A tuple containing:
                - points: Unique 3D points (after scaling, coordinate update, and filtering) for the XZ view.
                        In each point, columns 0 and 1 store the new XZ umbilicus plane view coordinate.
                - point_nodes_indices: Indices mapping points back to nodes.
                - windings: Mean winding values for each unique point.
                - windings_computed: Mean computed winding values.
                - inverse_indices: Mapping indices used in computing the mean.
                - winding: Original per-point winding values.
                - winding_computed: Original per-point computed winding values.
                - centroid_close_mask: Boolean mask indicating which nodes passed the centroid filtering.
        """


        # Convert target angle to radians and compute its normal and tangent vectors.
        f_target_rad = np.deg2rad(f_target)
        normal = np.array([-np.sin(f_target_rad), np.cos(f_target_rad)])
        tangent = np.array([-np.cos(f_target_rad), -np.sin(f_target_rad)])

        print(f"Angle: {f_target} degrees, normal: {normal}, tangent: {tangent}")
        
        use_h5 = h5_filename.endswith(".h5")
        if not use_h5:
            # If using subsampled points from graph, use more nodes for fuller pointcloud display
            block_size *= 2
        
        # Get node keys corresponding to undeleted nodes.
        all_node_keys = list(self.nodes.keys())
        node_keys = [all_node_keys[i] for i in undeleted_nodes_indices]
        
        # Compute winding angles for each node.
        winding_angles_nodes = np.asarray(labels) * 360.0 + np.asarray(f_init)
        winding_angles_nodes_computed = np.asarray(computed_labels) * 360.0 + np.asarray(f_init)
        assert len(winding_angles_nodes) == len(node_keys), (
            f"Mismatch: len(winding_angles_nodes)={len(winding_angles_nodes)} vs len(node_keys)={len(node_keys)}"
        )
        
        # Filter nodes based on centroid distance from the umbilicus plane.
        # Get all centroids at once
        centroids = np.array([self.nodes[node_key]['centroid'] for node_key in node_keys])  # shape: (N, 3) [y, z, x]
        
        # Scale all centroids to scroll coordinates
        scaled_centroids = centroids * 4.0 - 500
        
        # Extract centroid_xy and z_val arrays
        centroid_xy = scaled_centroids[:, [2, 0]]  # shape: (N, 2) [x, y]
        z_vals = scaled_centroids[:, 1]  # shape: (N,)
        
        # Get umbilicus centers for all z slices (vectorized)
        centers = umbilicus_xy_at_z_vector(umbilicus_data, z_vals)  # shape: (N, 2)
        
        # Compute all distances at once
        distances = np.abs((centroid_xy - centers) @ normal)  # shape: (N,)
        
        # Apply the mask
        centroid_close_mask = distances < block_size
        
        # Select only the nodes that are close enough.
        close_nodes = [node_keys[i] for i in range(len(node_keys)) if centroid_close_mask[i]]
        close_angles = [winding_angles_nodes[i] for i in range(len(node_keys)) if centroid_close_mask[i]]
        close_angles_computed = [winding_angles_nodes_computed[i] for i in range(len(node_keys)) if centroid_close_mask[i]]
        
        print(f"Found {len(close_nodes)} close nodes based on umbilicus plane filtering out of {len(node_keys)} undeleted nodes.")
        
        plane_point_filter_distance = 3
        # Load points from these nodes.
        if not use_h5:
            # Use backup graph node points
            points, point_nodes_indices = self.load_nodes_graph(close_nodes, close_angles, close_angles_computed)
            plane_point_filter_distance = 10
        else:
            points, point_nodes_indices = self.load_nodes(h5_filename, close_nodes, close_angles, close_angles_computed)
        print(f"Loaded {points.shape} points for XZ view on umbilicus plane with f_target {f_target}")
        
        # Scale point coordinates.
        points[:, :3] = points[:, :3] * 4.0 - 500
        # swap axis
        points = points[:, [1, 2, 0, 3, 4]]
        
        # ---------------------------
        # Vectorized per-point filtering:
        # ---------------------------
        # 1. Get the z values (rounded) from points.
        z_vals = points[:, 0]
        # 2. Use the vectorized interpolation to get the umbilicus center for each z value.
        centers = umbilicus_xy_at_z_vector(umbilicus_data, z_vals)  # shape: (N, 2), columns: [x, y]
        # 3. Compute the difference between each point's (x,y) and its corresponding center.
        diff = points[:, 1:3] - centers
        # 4. Compute the distance from the umbilicus plane (by projecting diff onto the normal vector).
        distance = np.abs(np.sum(diff * normal, axis=1))
        # 5. Create the mask for points within a distance of 3.
        print(f"Filtering points for XZ view with distance {plane_point_filter_distance}")
        points_distance_mask = distance < plane_point_filter_distance
        
        # Filter points and associated node indices.
        points = points[points_distance_mask]
        point_nodes_indices = point_nodes_indices[points_distance_mask]
        # Also filter centers for the remaining points.
        centers = centers[points_distance_mask]
        print(f"Filtered {points.shape} points based on per-point distance to the umbilicus plane.")
        
        # ---------------------------
        # Update point coordinates to store the XZ umbilicus plane view coordinate.
        # For each point, compute the new x coordinate as the projection of (point_xy - center)
        # onto the tangent vector. The new coordinate will be (new_x, original_z).
        # ---------------------------
        # Compute new x coordinate.
        new_x = np.sum((points[:, [1,2]] - centers) * tangent, axis=1)
        # The new view coordinate: first column becomes new_x and second column is the original z.
        # Retain the other columns (here, we store original y in column 2 and winding info in columns 3 and 4).
        points[:, 1] = new_x
        # points = points[:, [1, 0, 2, 3, 4]]
        
        # Sort points lexicographically.
        points_sort_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]))
        points = points[points_sort_indices]
        point_nodes_indices = point_nodes_indices[points_sort_indices]
        
        # Compute unique points with mean winding values.
        points, windings, windings_computed, inverse_indices, winding, winding_computed = \
            compute_mean_winding(points[:, :3], points[:, 3], points[:, 4], unlabeled)
        print(f"Computed mean winding for {points.shape} unique 3D points for XZ view on umbilicus plane with f_target {f_target}")
        
        return points, point_nodes_indices, windings, windings_computed, inverse_indices, winding, winding_computed, centroid_close_mask

########################################
# Array-based versions of ScrollGraph methods
########################################

def load_nodes_from_arrays(centroids, node_keys, sample_points, close_node_indices, winding_angles_nodes, winding_angles_nodes_computed, h5_filename=None):
    """
    Load nodes from array-based data instead of graph object.
    
    Parameters:
        centroids: numpy 2D array (n_nodes, 3) as float16
        node_keys: numpy 2D array (n_nodes, 4) as int16 (4-tuple structure, can be negative)
        sample_points: list of 1D float16 arrays (or None for H5 mode)
        close_node_indices: indices of nodes to load
        winding_angles_nodes: winding angles for each node
        winding_angles_nodes_computed: computed winding angles for each node
        h5_filename: path to H5 file (if None, uses sample_points)
    
    Returns:
        points_all: concatenated points with winding info
        point_nodes_indices: mapping from points back to node indices
    """
    use_h5 = h5_filename is not None and h5_filename.endswith(".h5")
    
    if use_h5:
        # Use H5 file loading (similar to original load_nodes method)
        return load_nodes_h5_from_arrays(centroids, node_keys, close_node_indices, winding_angles_nodes, winding_angles_nodes_computed, h5_filename)
    else:
        # Use sample_points from arrays
        return load_nodes_graph_from_arrays(sample_points, close_node_indices, winding_angles_nodes, winding_angles_nodes_computed)

def load_nodes_h5_from_arrays(centroids, node_keys, close_node_indices, winding_angles_nodes, winding_angles_nodes_computed, h5_filename):
    """Load nodes from H5 file using array-based node information."""
    nodes_points = []
    point_nodes_indices = []
    
    # Build groups dictionary using array data - use node keys directly like original code
    groups_dict = {}
    
    # Debug: Print some sample node keys
    print(f"[H5Debug] Processing {len(close_node_indices)} close nodes", file=sys.stderr)
    if len(close_node_indices) > 0:
        sample_indices = close_node_indices[:min(5, len(close_node_indices))]
        print(f"[H5Debug] Sample close_node_indices: {sample_indices}", file=sys.stderr)
        print(f"[H5Debug] Sample node_keys shape: {node_keys.shape}", file=sys.stderr)
        for i, node_idx in enumerate(sample_indices):
            print(f"[H5Debug] Node {node_idx}: key = {node_keys[node_idx]}", file=sys.stderr)
    
    for i, node_idx in enumerate(close_node_indices):
        # Get the node key (4-tuple) for this node - this is the "close_node" in original code
        close_node = node_keys[node_idx]  # This is the 4-tuple node key
        
        # Use first 3 elements as coordinates for group name (like original code)
        start_coord = close_node[:3]
        group_name = f"{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}"
        
        # Use 4th element as surface number (like original code)
        surface_nr = close_node[3]
        groups_dict.setdefault(group_name, []).append((surface_nr, i))
        
        # Debug: Print first few group names
        if i < 5:
            print(f"[H5Debug] Node {i}: group_name = {group_name}, surface_nr = {surface_nr}", file=sys.stderr)
    
    total_surfaces = sum(len(entries) for entries in groups_dict.values())
    print(f"[H5Debug] Generated {len(groups_dict)} unique groups with {total_surfaces} total surfaces", file=sys.stderr)
    
    # Load from H5 file
    with h5py.File(h5_filename, "r") as h5f, tqdm(total=total_surfaces, desc="Loading nodes from arrays") as pbar:
        # Debug: Print some H5 group names
        h5_groups = list(h5f.keys())
        print(f"[H5Debug] H5 file has {len(h5_groups)} groups. Sample: {h5_groups[:5]}", file=sys.stderr)
        
        for group_name, entries in groups_dict.items():
            if group_name not in h5f:
                if len([g for g in groups_dict.keys() if g not in h5f]) < 10:  # Only print first 10 missing groups
                    print(f"[H5Debug] Group {group_name} not found in {h5_filename}")
                pbar.update(len(entries))
                continue
            grp = h5f[group_name]
            
            for surface_nr, idx in entries:
                surface_name = f"surface_{surface_nr}"
                if surface_name not in grp:
                    print(f"Surface {surface_name} not found in {h5_filename}")
                    pbar.update(1)
                    continue
                surface = grp[surface_name]
                try:
                    points = surface["points"][()]
                    if points.shape[1] != 3:
                        raise ValueError(f"Invalid points shape: {points.shape}")
                    
                    # Add winding angles
                    winding_angle = winding_angles_nodes[idx]
                    winding_angle_computed = winding_angles_nodes_computed[idx]
                    winding_col = np.full((points.shape[0], 1), winding_angle, dtype=np.float32)
                    winding_col_computed = np.full((points.shape[0], 1), winding_angle_computed, dtype=np.float32)
                    points = np.concatenate([points, winding_col, winding_col_computed], axis=1)
                    nodes_points.append(points)
                    point_nodes_indices.extend([idx for _ in range(points.shape[0])])
                except Exception as e:
                    print(f"Error loading subvolume {group_name} patch {surface_nr}: {e}")
                pbar.update(1)
    
    if nodes_points:
        points_all = np.concatenate(nodes_points, axis=0)
    else:
        points_all = np.empty((0, 5), dtype=np.float32)
    return points_all, np.array(point_nodes_indices)

def load_nodes_graph_from_arrays(sample_points, close_node_indices, winding_angles_nodes, winding_angles_nodes_computed):
    """Load nodes from sample_points arrays."""
    nodes_points = []
    point_nodes_indices = []
    
    with tqdm(total=len(close_node_indices), desc="Loading nodes from sample_points") as pbar:
        for i, node_idx in enumerate(close_node_indices):
            try:
                if sample_points is None or node_idx >= len(sample_points):
                    print(f"No sample points for node index {node_idx}")
                    pbar.update(1)
                    continue
                    
                points_1d = sample_points[node_idx]
                if len(points_1d) == 0 or len(points_1d) % 3 != 0:
                    print(f"Invalid sample points shape for node {node_idx}: {len(points_1d)}")
                    pbar.update(1)
                    continue
                
                # Reshape 1D array to Nx3
                points = points_1d.reshape(-1, 3)
                
                # Add winding angles
                winding_angle = winding_angles_nodes[i]
                winding_angle_computed = winding_angles_nodes_computed[i]
                winding_col = np.full((points.shape[0], 1), winding_angle, dtype=np.float32)
                winding_col_computed = np.full((points.shape[0], 1), winding_angle_computed, dtype=np.float32)
                points = np.concatenate([points, winding_col, winding_col_computed], axis=1)
                nodes_points.append(points)
                point_nodes_indices.extend([i for _ in range(points.shape[0])])
            except Exception as e:
                print(f"Error loading node {node_idx} from sample_points: {e}")
            pbar.update(1)
    
    if nodes_points:
        points_all = np.concatenate(nodes_points, axis=0)
    else:
        points_all = np.empty((0, 5), dtype=np.float32)
    return points_all, np.array(point_nodes_indices)

def get_points_XY_from_arrays(centroids, node_keys, sample_points, z_index, h5_filename, labels, computed_labels, f_init, undeleted_nodes_indices, unlabeled, block_size=200):
    """
    Array-based version of ScrollGraph.get_points_XY.
    
    Parameters:
        centroids: numpy 2D array (n_nodes, 3) as float16
        node_keys: numpy 2D array (n_nodes, 4) as int16 (4-tuple structure, can be negative)
        sample_points: list of 1D float16 arrays (or None for H5)
        z_index: Z slice index
        h5_filename: path to H5 file
        labels: current labels for each node
        computed_labels: computed labels for each node
        f_init: initial winding offset per node
        undeleted_nodes_indices: indices of undeleted nodes
        unlabeled: value indicating unlabeled node
        block_size: tolerance for node filtering
    
    Returns:
        Same as original get_points_XY method
    """
    use_h5 = h5_filename is not None and h5_filename.endswith(".h5")
    if not use_h5:
        block_size *= 2  # Use more nodes for fuller pointcloud display
    
    # Get subset of nodes corresponding to undeleted indices
    node_keys_subset = node_keys[undeleted_nodes_indices]
    centroids_subset = centroids[undeleted_nodes_indices]
    
    winding_angles_nodes = np.asarray(labels) * 360.0 + np.asarray(f_init)
    winding_angles_nodes_computed = np.asarray(computed_labels) * 360.0 + np.asarray(f_init)
    assert len(winding_angles_nodes) == len(node_keys_subset), f"Length mismatch: {len(winding_angles_nodes)} != {len(node_keys_subset)}"

    # Check which nodes are close to the z_index
    # centroids are in [y, z, x] order, scale to scroll coordinates
    scaled_centroids = centroids_subset * 4.0 - 500
    close_mask = np.abs(scaled_centroids[:, 1] - z_index) < block_size  # Check z coordinate
    
    close_node_indices = np.where(close_mask)[0]  # Indices in the subset
    close_angles = winding_angles_nodes[close_mask]
    close_angles_computed = winding_angles_nodes_computed[close_mask]

    print(f"Found {len(close_node_indices)} close nodes at z_index {z_index} of total {len(node_keys_subset)} undeleted nodes")

    # Load points from these nodes
    plane_point_filter_distance = 3
    if not use_h5:
        # Map subset indices back to full array indices for sample_points access
        full_indices = undeleted_nodes_indices[close_node_indices]
        points, point_nodes_indices = load_nodes_graph_from_arrays(sample_points, full_indices, close_angles, close_angles_computed)
        plane_point_filter_distance = 10
    else:
        # For H5, we can use the subset indices directly
        points, point_nodes_indices = load_nodes_h5_from_arrays(centroids_subset, node_keys_subset, close_node_indices, close_angles, close_angles_computed, h5_filename)

    print(f"Loaded {points.shape} points for XY view at z_index {z_index}")

    # Scale points to scroll coordinates
    points[:, :3] = points[:, :3] * 4.0 - 500
    # Swap axis: [y, z, x] -> [z, y, x]
    points = points[:, [1, 0, 2, 3, 4]]

    # Filter points in z slice
    print(f"Filtering points for XY view with distance {plane_point_filter_distance}")
    mask = np.abs(points[:, 0] - z_index) < plane_point_filter_distance
    points = points[mask]
    point_nodes_indices = point_nodes_indices[mask]

    print(f"Filtered {points.shape} points for XY view at z_index {z_index}")

    # Sort points by ZXY
    points_sort_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]))
    points = points[points_sort_indices]
    point_nodes_indices = point_nodes_indices[points_sort_indices]

    # Compute unique points with mean winding angle
    points, windings, windings_computed, inverse_indices, winding, winding_computed = compute_mean_winding(points[:, :3], points[:, 3], points[:, 4], unlabeled)

    print(f"Computed mean winding for {points.shape} unique 3D points for XY view at z_index {z_index}")

    return points, point_nodes_indices, windings, windings_computed, inverse_indices, winding, winding_computed, close_mask

def get_points_XZ_from_arrays(centroids, node_keys, sample_points, f_target, umbilicus_data, h5_filename, labels, computed_labels, f_init, undeleted_nodes_indices, unlabeled, block_size=200):
    """
    Array-based version of ScrollGraph.get_points_XZ.
    
    Parameters:
        centroids: numpy 2D array (n_nodes, 3) as float16
        node_keys: numpy 2D array (n_nodes, 4) as int16 (4-tuple structure, can be negative)
        sample_points: list of 1D float16 arrays (or None for H5)
        f_target: angle defining the target umbilicus plane
        umbilicus_data: umbilicus center data
        h5_filename: path to H5 file
        labels: current labels for each node
        computed_labels: computed labels for each node
        f_init: initial winding offset per node
        undeleted_nodes_indices: indices of undeleted nodes
        unlabeled: value indicating unlabeled node
        block_size: tolerance for node filtering
    
    Returns:
        Same as original get_points_XZ method
    """
    # Convert target angle to radians and compute vectors
    f_target_rad = np.deg2rad(f_target)
    normal = np.array([-np.sin(f_target_rad), np.cos(f_target_rad)])
    tangent = np.array([-np.cos(f_target_rad), -np.sin(f_target_rad)])

    print(f"Angle: {f_target} degrees, normal: {normal}, tangent: {tangent}")
    
    use_h5 = h5_filename is not None and h5_filename.endswith(".h5")
    if not use_h5:
        block_size *= 2
    
    # Get subset of nodes corresponding to undeleted indices
    node_keys_subset = node_keys[undeleted_nodes_indices]
    centroids_subset = centroids[undeleted_nodes_indices]
    
    winding_angles_nodes = np.asarray(labels) * 360.0 + np.asarray(f_init)
    winding_angles_nodes_computed = np.asarray(computed_labels) * 360.0 + np.asarray(f_init)
    assert len(winding_angles_nodes) == len(node_keys_subset), f"Length mismatch: {len(winding_angles_nodes)} != {len(node_keys_subset)}"
    
    # Filter nodes based on centroid distance from umbilicus plane
    # Scale centroids to scroll coordinates
    scaled_centroids = centroids_subset * 4.0 - 500
    
    # Extract centroid_xy and z_val arrays - centroids are [y, z, x], we want [x, y]
    centroid_xy = scaled_centroids[:, [2, 0]]  # [x, y]
    z_vals = scaled_centroids[:, 1]  # z values
    
    # Get umbilicus centers for all z slices
    centers = umbilicus_xy_at_z_vector(umbilicus_data, z_vals)
    
    # Compute distances to umbilicus plane
    distances = np.abs((centroid_xy - centers) @ normal)
    centroid_close_mask = distances < block_size
    
    close_node_indices = np.where(centroid_close_mask)[0]
    close_angles = winding_angles_nodes[centroid_close_mask]
    close_angles_computed = winding_angles_nodes_computed[centroid_close_mask]
    
    print(f"Found {len(close_node_indices)} close nodes based on umbilicus plane filtering out of {len(node_keys_subset)} undeleted nodes.")
    
    plane_point_filter_distance = 3
    # Load points from these nodes
    if not use_h5:
        # Map subset indices back to full array indices for sample_points access
        full_indices = undeleted_nodes_indices[close_node_indices]
        points, point_nodes_indices = load_nodes_graph_from_arrays(sample_points, full_indices, close_angles, close_angles_computed)
        plane_point_filter_distance = 10
    else:
        points, point_nodes_indices = load_nodes_h5_from_arrays(centroids_subset, node_keys_subset, close_node_indices, close_angles, close_angles_computed, h5_filename)
    
    print(f"Loaded {points.shape} points for XZ view on umbilicus plane with f_target {f_target}")
    
    # Scale point coordinates
    points[:, :3] = points[:, :3] * 4.0 - 500
    # Swap axis: [y, z, x] -> [z, x, y]
    points = points[:, [1, 2, 0, 3, 4]]
    
    # Vectorized per-point filtering
    z_vals = points[:, 0]
    centers = umbilicus_xy_at_z_vector(umbilicus_data, z_vals)
    diff = points[:, 1:3] - centers
    distance = np.abs(np.sum(diff * normal, axis=1))
    
    print(f"Filtering points for XZ view with distance {plane_point_filter_distance}")
    points_distance_mask = distance < plane_point_filter_distance
    
    points = points[points_distance_mask]
    point_nodes_indices = point_nodes_indices[points_distance_mask]
    centers = centers[points_distance_mask]
    print(f"Filtered {points.shape} points based on per-point distance to the umbilicus plane.")
    
    # Update point coordinates for XZ umbilicus plane view
    new_x = np.sum((points[:, [1,2]] - centers) * tangent, axis=1)
    points[:, 1] = new_x
    
    # Sort points lexicographically
    points_sort_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]))
    points = points[points_sort_indices]
    point_nodes_indices = point_nodes_indices[points_sort_indices]
    
    # Compute unique points with mean winding values
    points, windings, windings_computed, inverse_indices, winding, winding_computed = \
        compute_mean_winding(points[:, :3], points[:, 3], points[:, 4], unlabeled)
    print(f"Computed mean winding for {points.shape} unique 3D points for XZ view on umbilicus plane with f_target {f_target}")
    
    return points, point_nodes_indices, windings, windings_computed, inverse_indices, winding, winding_computed, centroid_close_mask