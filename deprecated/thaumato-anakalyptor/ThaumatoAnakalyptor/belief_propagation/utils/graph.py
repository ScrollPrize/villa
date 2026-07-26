# Giorgio Angelotti - 2024
# Loopy Belief Propagation Solver with Iterated MAP estimation for the Sheet Stitching Problem in PGMax + Laplacian Smoothing
# BP's Log-potential implementation by Francesco Mori
# Laplacian smoothing inspired by Julian Schilliger Graph Problem Solver

import numpy as np
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import List, Any, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import networkx as nx
import struct

@dataclass(slots=True)
class Graph:
    edges_nodes: NDArray[np.uint32] = field(default_factory=lambda: np.empty((0, 2), dtype=np.uint32))  # Nx2 array (N, (Source, Target))
    edges_feats: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))  # Nx2 array (N, (weight, k))
    nodes_f: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))  # M array (M, )
    nodes_z: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))  # M array (M, )
    same_blocks: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))  # N array (N, )
    nodes_d: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))  # N array (N, )

class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=np.int32)
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def maximum_spanning_tree(graph: Graph):
    # Extract edges and weights
    edges = graph.edges_nodes
    weights = graph.edges_feats[:, 0]  # Assuming the first column represents the weight
    ks = graph.edges_feats[:, 1]

    #sames = graph.same_blocks
    # Number of nodes
    num_nodes = graph.nodes_f.shape[0]

    #weights[sames] += 2

    sorted_indices = np.argsort(-weights)
    sorted_edges = edges[sorted_indices]
    sorted_ks = ks[sorted_indices]

    # Kruskal's algorithm using Union-Find
    uf = UnionFind(num_nodes)
    mst_edges = []
    
    for edge, k in tqdm(zip(sorted_edges, sorted_ks), total=graph.edges_nodes.shape[0]):
        u, v = edge
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_edges.append((u, v, k))
            # Stop when we have enough edges for the MST
            if len(mst_edges) == num_nodes - 1:
                break


    return mst_edges

def extract_largest_connected_component(graph: Graph) -> Graph:
    """
    Extracts the largest connected component from the input graph.

    Args:
    - graph (Graph): The original input graph.

    Returns:
    - Graph: The largest connected component as a new Graph object with adjusted node indices.
    """

    # Build a networkx graph from edges_nodes
    G = nx.Graph()
    G.add_edges_from(graph.edges_nodes)

    # Find the largest connected component
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    largest_cc_nodes = sorted(largest_cc_nodes)  # Sort nodes for consistent ordering

    # Create a mapping from old node indices to new node indices (0 to n-1)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_cc_nodes)}

    # Create a set for quick lookup
    largest_cc_set = set(largest_cc_nodes)

    # Filter edges where both nodes are in the largest connected component
    mask = np.array([(u in largest_cc_set) and (v in largest_cc_set) for u, v in graph.edges_nodes])

    # Extract edges and their features in the largest connected component
    edges_in_largest_cc = graph.edges_nodes[mask]
    edges_feats_in_largest_cc = graph.edges_feats[mask]

    # Adjust node indices in edges
    adjusted_edges_nodes = np.array(
        [[old_to_new[u], old_to_new[v]] for u, v in edges_in_largest_cc],
        dtype=np.uint32
    )

    # Extract node features for nodes in the largest connected component
    nodes_f_in_largest_cc = graph.nodes_f[largest_cc_nodes]

    # Create the new Graph object
    largest_component_graph = Graph(
        edges_nodes=adjusted_edges_nodes,
        edges_feats=edges_feats_in_largest_cc,
        nodes_f=nodes_f_in_largest_cc
    )

    return largest_component_graph, largest_cc_nodes



def sum_errors_per_node(graph, error_vector):
    # Initialize an array to store the sum of errors for each node
    node_error_sums = np.zeros(len(graph.nodes_f), dtype=np.float64)
    
    # Extract source and target nodes
    src_nodes = graph.edges_nodes[:, 0]
    tgt_nodes = graph.edges_nodes[:, 1]
    
    # Use np.add.at to accumulate the errors for both source and target nodes
    np.add.at(node_error_sums, src_nodes, error_vector)  # Add errors to source nodes
    np.add.at(node_error_sums, tgt_nodes, error_vector)  # Add errors to target nodes
    
    return node_error_sums

def filter_nodes(graph: Graph, error_per_node, quantile_threshold: float = 0.95) -> Graph:
    """
    Filters out nodes whose error is above the given quantile threshold.

    Args:
    - graph (Graph): The input graph.
    - error_per_node (NDArray[np.float64]): The error values for each node.
    - quantile_threshold (float): The quantile threshold to filter nodes by (default 0.95).

    Returns:
    - Graph: A new graph with nodes whose error is below the quantile threshold and adjusted node indices.
    """

    # Compute the threshold error value at the specified quantile
    threshold_value = np.quantile(error_per_node, quantile_threshold)

    # Identify nodes that should be kept (error_per_node <= threshold)
    nodes_to_keep = np.where(error_per_node <= threshold_value)[0]

    # Create a mapping from old node indices to new node indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}

    # Create a set for quick lookup of valid nodes
    nodes_to_keep_set = set(nodes_to_keep)

    # Filter edges: keep only edges where both nodes are in the nodes_to_keep_set
    mask = np.array([(u in nodes_to_keep_set) and (v in nodes_to_keep_set) for u, v in graph.edges_nodes])

    # Extract the filtered edges and their features
    edges_filtered = graph.edges_nodes[mask]
    edges_feats_filtered = graph.edges_feats[mask]

    # Adjust node indices in the edges to match the new indices (from old_to_new)
    adjusted_edges_nodes = np.array(
        [[old_to_new[u], old_to_new[v]] for u, v in edges_filtered],
        dtype=np.uint32
    )

    # Extract node features for the remaining nodes
    nodes_f_filtered = graph.nodes_f[nodes_to_keep]

    # Return a new graph with filtered and adjusted node indices
    filtered_graph = Graph(
        edges_nodes=adjusted_edges_nodes,
        edges_feats=edges_feats_filtered,
        nodes_f=nodes_f_filtered
    )

    return filtered_graph

def bfs(graph: Graph, tree:List[Any], start: int):
    visited = set()
    queue = deque([start])
    
    # Initialize the start node's properties
    graph.nodes_f[start] = 0  # f_init, f_star
    
    visited.add(start)

    # Create an adjacency list from the MST
    adj_list = defaultdict(list)
    for node1, node2, k in tree:
        adj_list[node1].append((node2, k))
        adj_list[node2].append((node1, -k))

    # Progress bar setup
    total_nodes = len(graph.nodes_f)  # Assuming all nodes in graph are in nodes_f
    with tqdm(total=total_nodes, desc="Processing nodes") as pbar:
        while queue:
            node = queue.popleft()
            
            # Update the progress bar for each node processed
            pbar.update(1)
            
            # Iterate over the neighbors of the current node in the MST
            for neighbor, k in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Update the neighbor's f_star and f_init values based on the edge weight
                    graph.nodes_f[neighbor] = graph.nodes_f[node] + k  # f_star

def f_ranges(graph: Graph) -> Tuple[float, float]:
    star_range = np.max(graph.nodes_f) - np.min(graph.nodes_f)
    return star_range

def validate_node_edge_mapping(original_graph: Graph, largest_cc_graph: Graph, old_to_new_node_index: dict) -> bool:
    """
    Validates that the node and edge mapping from the original graph to the largest connected component is correct.

    Args:
    - original_graph (Graph): The original input graph.
    - largest_cc_graph (Graph): The graph corresponding to the largest connected component.
    - old_to_new_node_index (dict): A dictionary mapping the old node indices to new ones in the largest connected component.

    Returns:
    - bool: True if the node and edge mapping is valid, otherwise False.
    """

    # Reverse mapping: new index -> old index
    new_to_old_node_index = {new_idx: old_idx for old_idx, new_idx in old_to_new_node_index.items()}

    # 1. Check that each edge in largest_cc_graph corresponds to an edge in original_graph
    # Create a set of tuples of original edges for fast lookup
    original_edges_set = set(map(tuple, map(sorted, original_graph.edges_nodes)))

    # Loop through each edge in the largest connected component and check if it maps back correctly
    for new_edge in tqdm(largest_cc_graph.edges_nodes):
        # Get the original indices of the nodes in this edge
        old_source = new_to_old_node_index[new_edge[0]]
        old_target = new_to_old_node_index[new_edge[1]]
        
        # Check if the corresponding edge (old_source, old_target) exists in the original graph
        sorted_edge = tuple(sorted([old_source, old_target]))  # Sort to handle undirected edges
        if sorted_edge not in original_edges_set:
            print(f"Invalid edge mapping: {old_source}-{old_target} not found in original edges")
            return False

    # 2. Ensure that the node features in largest_cc_graph match the corresponding nodes in original_graph
    for new_idx, node_features in tqdm(enumerate(largest_cc_graph.nodes_f)):
        # Find the original node index
        old_idx = new_to_old_node_index[new_idx]
        
        # Check that the node features match between the largest_cc_graph and original_graph
        if not np.array_equal(node_features, original_graph.nodes_f[old_idx]):
            print(f"Node feature mismatch for new node {new_idx} (mapped from original node {old_idx})")
            return False

    # If all checks pass
    print("The node and edge mapping is valid.")
    return True

def update_f_star(graph: Graph, bp_decoding, variables, L: int):
    decoded_states = np.asarray(bp_decoding[variables]).astype(np.int8)

    # Precompute the decoded shifts
    decoded_shifts = decoded_states - L  # Shift by L to map [0, 2L] -> [-L, L]

    graph.nodes_f += decoded_shifts.astype(np.float64) * 360
    
def save_graph(file_name, graph):
    """
    Saves the graph's node data (stored in graph.nodes_f) to a compressed NPZ file.

    Parameters:
    - file_name (str): The name of the file to save the data (without extension).
    - graph: An object containing a numpy array `nodes_f` with node data.
    """
    # Save the graph nodes_f as a compressed .npz file
    np.savez_compressed(file_name, nodes_f=graph.nodes_f, nodes_d=graph.nodes_d)
    print(f"Graph saved to {file_name}")

def load_graph_from_binary_results(file_name: str) -> Graph:
    graph = Graph()

    deleted = []
    with open(file_name, 'rb') as infile:
        # Read the number of nodes
        num_nodes = struct.unpack('I', infile.read(4))[0]  # 'I' is for unsigned int
        
        graph.nodes_f = np.empty(num_nodes, dtype=np.float64)
        graph.nodes_d = np.empty(num_nodes, dtype=np.bool_)
        # Read each node's f_star and deleted status
        for node_id in range(num_nodes):
            f_star = struct.unpack('f', infile.read(4))[0]  # 'f' is for float
            deleted = struct.unpack('?', infile.read(1))[0]  # '?' is for boolean
            graph.nodes_f[node_id] = f_star
            graph.nodes_d[node_id] = deleted
    return graph

def load_graph_from_binary(file_name: str) -> Graph:
    # Initialize empty graph
    graph = Graph()
    existing_edges = set()  # To track existing edges as frozensets

    with open(file_name, 'rb') as infile:
        # Read the number of nodes (unpack as unsigned int)
        num_nodes = struct.unpack('I', infile.read(4))[0]
        
        # Initialize the nodes_f array with size for num_nodes
        graph.nodes_f = np.empty(num_nodes, dtype=np.float64)
        
        # Read node initial features (f_init) and set f_star the same as f_init for each node
        for node_id in range(num_nodes):
            f_init = struct.unpack('f', infile.read(4))[0]  # Unpack 4-byte float for f_init
            graph.nodes_f[node_id] = f_init  # Store f_init for both f_init and f_star
        
        # Temporary lists to hold edge data before converting to NumPy arrays
        edge_nodes_list = []
        edge_feats_list = []

        # Read the adjacency list
        for _ in range(num_nodes):
            # Unpack node_id (unsigned int) and number of edges (unsigned int)
            node_id = struct.unpack('I', infile.read(4))[0]
            num_edges = struct.unpack('I', infile.read(4))[0]

            for _ in range(num_edges):
                # Unpack target node (unsigned int)
                target_node = struct.unpack('I', infile.read(4))[0]
                # Unpack edge features: certainty (float) and k (float)
                certainty = struct.unpack('f', infile.read(4))[0]
                k = struct.unpack('f', infile.read(4))[0]
                # Unpack the same_block boolean (1 byte)
                same_block = struct.unpack('?', infile.read(1))[0]
                
                # Create an edge pair and check for duplicates (treat A->B as B->A with frozenset)
                edge_pair = frozenset([node_id, target_node])
                
                if edge_pair not in existing_edges:
                    # Add the edge if it hasn't been added before
                    edge_nodes_list.append([node_id, target_node])
                    edge_feats_list.append([certainty, k])
                    existing_edges.add(edge_pair)  # Mark this edge as added

        # Convert the gathered edge data into NumPy arrays and store in the graph
        if edge_nodes_list:
            graph.edges_nodes = np.array(edge_nodes_list, dtype=np.uint32)  # Convert to uint32 array
            graph.edges_feats = np.array(edge_feats_list, dtype=np.float64)  # Convert to float64 array
    
    return graph


def load_graph_from_binary_new(file_name: str) -> Graph:
    """
    Loads a graph from a binary file, compatible with the write_graph_to_binary function provided.
    
    Parameters:
    - file_name (str): Path to the binary file.
    
    Returns:
    - Graph: An instance of the Graph class with loaded data.
    """
    graph = Graph()
    existing_edges = set()  # To track existing edges as frozensets

    with open(file_name, 'rb') as infile:
        # Read the number of nodes (unsigned int)
        num_nodes_data = infile.read(4)
        if len(num_nodes_data) != 4:
            raise ValueError("Error reading number of nodes.")
        num_nodes = struct.unpack('I', num_nodes_data)[0]
        print(f"Number of nodes: {num_nodes}")

        # Initialize the nodes_f array with size for num_nodes
        graph.nodes_f = np.empty(num_nodes, dtype=np.float64)
        graph.nodes_z = np.empty(num_nodes, dtype=np.float64)
        # Read node data
        for node_id in tqdm(range(num_nodes)):
            # Read z_position (float)
            z_data = infile.read(4)
            if len(z_data) != 4:
                raise ValueError(f"Error reading z_position of node {node_id}.")
            z_position = struct.unpack('f', z_data)[0]
            graph.nodes_z[node_id] = z_position
            # Read winding_angle (float)
            winding_angle_data = infile.read(4)
            if len(winding_angle_data) != 4:
                raise ValueError(f"Error reading winding_angle of node {node_id}.")
            winding_angle = struct.unpack('f', winding_angle_data)[0]

            # Read gt_flag (bool)
            gt_flag_data = infile.read(1)
            if len(gt_flag_data) != 1:
                raise ValueError(f"Error reading gt_flag of node {node_id}.")
            #gt_flag = struct.unpack('?', gt_flag_data)[0]

            # Read gt_winding_angle (float)
            gt_winding_angle_data = infile.read(4)
            if len(gt_winding_angle_data) != 4:
                raise ValueError(f"Error reading gt_winding_angle of node {node_id}.")
            #gt_winding_angle = struct.unpack('f', gt_winding_angle_data)[0]

            # For nodes_f, store the winding_angle
            graph.nodes_f[node_id] = winding_angle

        # Temporary lists to hold edge data before converting to NumPy arrays
        edge_nodes_list = []
        edge_feats_list = []
        same_block_list = []
        # Read the adjacency list
        for _ in tqdm(range(num_nodes)):
            # Read node_id (unsigned int)
            node_id_data = infile.read(4)
            if len(node_id_data) != 4:
                raise ValueError("Error reading node_id in adjacency list.")
            node_id = struct.unpack('I', node_id_data)[0]

            # Read number of edges (unsigned int)
            num_edges_data = infile.read(4)
            if len(num_edges_data) != 4:
                raise ValueError(f"Error reading number of edges for node {node_id}.")
            num_edges = struct.unpack('I', num_edges_data)[0]

            for _ in range(num_edges):
                # Read target_node_id (unsigned int)
                target_node_data = infile.read(4)
                if len(target_node_data) != 4:
                    raise ValueError("Error reading target_node_id.")
                target_node_id = struct.unpack('I', target_node_data)[0]

                # Read w (float)
                w_data = infile.read(4)
                if len(w_data) != 4:
                    raise ValueError("Error reading w.")
                w = struct.unpack('f', w_data)[0]

                # Read k (float)
                k_data = infile.read(4)
                if len(k_data) != 4:
                    raise ValueError("Error reading k.")
                k = struct.unpack('f', k_data)[0]

                # Read same_block (bool)
                same_block_data = infile.read(1)
                if len(same_block_data) != 1:
                    raise ValueError("Error reading same_block.")
                same_block = struct.unpack('?', same_block_data)[0]

                # Create an edge pair and check for duplicates (undirected edges)
                edge_pair = frozenset([node_id, target_node_id])

                if edge_pair not in existing_edges:
                    # Add the edge if it hasn't been added before
                    edge_nodes_list.append([node_id, target_node_id])
                    # Store edge features: w and k
                    edge_feats_list.append([w, k])
                    same_block_list.append(same_block)
                    existing_edges.add(edge_pair)  # Mark this edge as added

        # Convert the gathered edge data into NumPy arrays and store in the graph
        if edge_nodes_list:
            graph.edges_nodes = np.array(edge_nodes_list, dtype=np.uint32)
            graph.edges_feats = np.array(edge_feats_list, dtype=np.float64)
            graph.same_blocks = np.array(same_block_list)
        else:
            graph.edges_nodes = np.empty((0, 2), dtype=np.uint32)
            graph.edges_feats = np.empty((0, 2), dtype=np.float64)

    return graph