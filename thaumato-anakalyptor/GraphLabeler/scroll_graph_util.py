import numpy as np
import h5py
from tqdm import tqdm

########################################
# Utility Functions
########################################

def compute_mean_winding(coords, winding):
    # Extract the 3D coordinates and the winding angle values.
    # coords = np.round(points[:, :3])
    # winding = points[:, 3]
    coords = np.round(coords)

    # Get the unique 3D points and also obtain an array that can map each row of
    # the original array to its unique group.
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    
    # Compute the sum of winding angles for each unique coordinate.
    sum_winding = np.bincount(inverse_indices, weights=winding)
    
    # Compute the number of occurrences for each unique coordinate.
    count_winding = np.bincount(inverse_indices)
    
    # Calculate the mean winding angle by dividing the sum by the count.
    mean_winding = sum_winding / count_winding
    
    # Combine to unique 4D points.
    # points = np.concatenate([unique_coords, mean_winding[:, None]], axis=1)
    print(f"Shape of unique_coords: {unique_coords.shape}, shape of mean_winding: {mean_winding.shape}")

    return unique_coords, mean_winding

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()

    def load_nodes(self, h5_filename, close_nodes, winding_angles_nodes):
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
                        winding_col = np.full((points.shape[0], 1), winding_angle, dtype=np.float32)
                        points = np.concatenate([points, winding_col], axis=1)
                        nodes_points.append(points)
                    except Exception as e:
                        print(f"Error loading subvolume {group_name} patch {surface_nr} from {h5_filename}: {e}")
                    pbar.update(1)

        if nodes_points:
            points_all = np.concatenate(nodes_points, axis=0)
        else:
            points_all = np.empty((0, 4), dtype=np.float32)
        return points_all

    def get_points_XY(self, z_index, h5_filename, labels, f_init, undeleted_nodes_indices, block_size=200):
        """
        Get the points for the XY view at the given z_index.
        """
        
        all_node_keys = list(self.nodes.keys())
        node_keys = [all_node_keys[i] for i in undeleted_nodes_indices]
        winding_angles_nodes = np.asarray(labels) * 360.0 + np.asarray(f_init)
        assert len(winding_angles_nodes) == len(node_keys), f"len(winding_angles_nodes)={len(winding_angles_nodes)} != len(all_node_keys)={len(node_keys)}"

        # check if the node is touching the z_index
        close_nodes = []
        close_angles = []
        for i, node_key in enumerate(node_keys):
            if abs(self.nodes[node_key]['centroid'][1] * 4.0 - 500 - z_index) < block_size:
                close_nodes.append(node_key)
                close_angles.append(winding_angles_nodes[i])
        print(f"Found {len(close_nodes)} close nodes at z_index {z_index} of total {len(node_keys)} undeleted nodes and {len(all_node_keys)} all nodes")

        # sort close nodes by key
        print(f"first few close nodes: {close_nodes[:min(5, len(close_nodes))]}")

        # load all point of close blocks
        points = self.load_nodes(h5_filename, close_nodes, close_angles)

        print(f"Loaded {points.shape} points for XY view at z_index {z_index}")

        # scale points to scroll coordinates
        points[:, :3] = points[:, :3] * 4.0 - 500
        # swap axis
        points = points[:, [1, 0, 2, 3]]

        # filter points in z slice
        mask = np.abs(points[:, 0] - z_index) < 3
        points = points[mask]

        print(f"Filtered {points.shape} points for XY view at z_index {z_index}")

        # sort points by ZXY
        points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2], points[:, 3]))]

        # compute unique points with mean winding angle
        points, windings = compute_mean_winding(points[:, :3], points[:, 3])

        print(f"Computed mean winding for {points.shape} unique 3D points for XY view at z_index {z_index}")

        return points, windings