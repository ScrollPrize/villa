# Giorgio Angelotti - 2024
# Loopy Belief Propagation Solver with Iterated MAP estimation for the Sheet Stitching Problem in PGMax + Laplacian Smoothing
# BP's Log-potential implementation by Francesco Mori
# Laplacian smoothing inspired by Julian Schilliger Graph Problem Solver

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from .graph import Graph

@njit(parallel=True)
def module_angle(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply modulo operation to an array of angles and wrap them to the range [-180, 180].
    
    Arguments:
    - theta: An array of angle values.
    
    Returns:
    - An array of angles wrapped within the range [-180, 180].
    """
    theta = np.mod(theta + 180, 360) - 180  # Shift the range to [-180, 180]
    return np.where(np.abs(theta) < 1e-9, 0.0, theta)  # Treat near-zero values as 0

def compute_ell(node1_f_star: NDArray[np.float64], node2_f_star: NDArray[np.float64], k: NDArray[np.float64], tol: float = 1e-3, verbose: bool = True) -> NDArray[np.float64]:
    """
    Vectorized computation of the ell value between two nodes based on their f_star values and the edge's k value.
    
    Arguments:
    - node1_f_star: an array of f_star values for the first set of nodes
    - node2_f_star: an array of f_star values for the second set of nodes
    - k: an array of k values associated with the edges
    - tol: tolerance for contradiction checking
    
    Returns:
    - ell: an array of computed ell values
    """
    
    # Compute the difference and apply the module_angle (assuming it's a function that wraps an angle)
    delta_f_star = node1_f_star - node2_f_star + k
    module_diff = np.abs(np.round(module_angle(delta_f_star), decimals=9))

    
    if verbose and np.any(module_diff > tol):
        print("Edge with contradiction detected in the graph.")
       #raise ValueError(f"Edges with contradiction detected in the graph, max: {np.max(module_diff)}")

    ell = np.round(delta_f_star / 360 + 1e-9)

    
    return ell

def create_log_potential_matrix(certainty: NDArray[np.float64], L: np.uint8,
                                           node1_f_star: NDArray[np.float64], 
                                           node2_f_star: NDArray[np.float64], 
                                           k: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Vectorized creation of log potential matrices for multiple edges.
    
    Arguments:
    - certainty: an array of certainty values (shape: [num_edges])
    - L: a scalar (np.uint8) value, range of the states (states go from -L to L)
    - node1_f_star: an array of f_star values for the first set of nodes (shape: [num_edges])
    - node2_f_star: an array of f_star values for the second set of nodes (shape: [num_edges])
    - k: an array of k values (shape: [num_edges])

    Returns:
    - log_potential_matrices: an array of computed log potential matrices for all edges
                              (shape: [num_edges, 2L+1, 2L+1])
    """    
    # Step 1: Vectorized computation of ell for all edges
    ell = compute_ell(node1_f_star, node2_f_star, k)  # shape: [num_edges]
    
    # Step 2: Create shifts from -L to L
    shifts = np.arange(-L, L + 1, dtype=np.float64)  # shape: [matrix_size]
    
    # Step 3: Create grids of shift1 and shift2 values, broadcasted over all edges
    shift1_grid, shift2_grid = np.meshgrid(shifts, shifts, indexing='ij')  # shape: [matrix_size, matrix_size]
    
    # Step 4: Compute the interaction for each edge (vectorized over all edges)
    # We expand ell and certainty for broadcasting with the grid
    interaction = -certainty[:, None, None] * np.abs(ell[:, None, None] + shift1_grid - shift2_grid)

    return interaction

def compute_loss(graph, bp_decoding, variables, L, verbose=True):
    """
    Compute the average error for the graph based on the decoded MAP states obtained from the BP inference results,
    in a vectorized way while avoiding unnecessary data copying.
    
    Arguments:
    - graph: The Graph dataclass containing the nodes and edges
    - bp_decoding: The decoded MAP states from the BP inference (dictionary with NDVarArray as keys)
    - variables: The NDVarArray corresponding to the graph's variables
    - L: The range of the l values (shifts from -L to L)
    
    Returns:
    - The average error across all edges
    """
    # Get the decoded states (array of l values) from the bp_decoding dictionary
    decoded_states = np.asarray(bp_decoding[variables]).astype(np.float64)

    # Precompute the decoded shifts
    decoded_shifts = decoded_states - L  # Shift by L to map [0, 2L] -> [-L, L]



    # Get the f_star values directly from the graph (without copying)
    node_f_star = np.copy(graph.nodes_f)  # Use direct reference to the f_star array

    # Compute the effective f_star for each edge without creating new arrays (use float32 to avoid overflow)
    node_f_star += decoded_shifts.astype(np.float64) * 360

    # Get the certainty and k values from the edges (convert to float32 for stability)
    edge_certainties = graph.edges_feats[:, 0] # Certainty values
    edge_k_vals = graph.edges_feats[:, 1]      # k values

    # Vectorized computation of ell
    ell_values = compute_ell(node_f_star[graph.edges_nodes[:,0]], node_f_star[graph.edges_nodes[:,1]], edge_k_vals, verbose=verbose)  # Ensure ell_values are float32

    # Check for inf values in ell_values
    if verbose and np.any(np.isinf(ell_values)):
        print("Warning: Inf values found in ell_values!")

    # Compute the total error without copying arrays (use float32 for accumulation)
    total_error = np.sum(edge_certainties* np.abs(ell_values))

    # Calculate the average error
    count = len(graph.edges_nodes)
    
    return total_error / count if count > 0 else 0.0

@njit(parallel=True)
def weighted_laplacian_smoothing(edges_nodes: NDArray[np.uint32], 
                                 edges_feats: NDArray[np.float64], 
                                 nodes_f: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Perform weighted Laplacian smoothing on the graph node features.
    
    Arguments:
    - edges_nodes: Nx2 array of edge source and target nodes.
    - edges_feats: Nx2 array of edge features (weights and k values).
    - nodes_f: M array of node features.
    
    Returns:
    - smoothed_f: M array of smoothed node features.
    """
    # Initialize a copy of the nodes_f array for the smoothed values
    smoothed_f = np.copy(nodes_f)
    
    # Extract source and target nodes from edges
    source_nodes = edges_nodes[:, 0]  # Array of source nodes
    target_nodes = edges_nodes[:, 1]  # Array of target nodes
    
    # Extract edge weights and k values (from edges_feats)
    edge_weights = edges_feats[:, 0]  # Array of edge weights
    k_values = edges_feats[:, 1]      # Array of k values (second feature of edges_feats)

    # Step 1: Compute modified weighted sum of neighbors' features with the new expression
    weighted_sum = np.zeros_like(nodes_f)

    # Compute the modified feature transformation
    delta_f = nodes_f[target_nodes] - nodes_f[source_nodes]
    modified_diff = np.round((delta_f - k_values) / 360)

    # Add the contribution using the new formula for both source and target
    for i in prange(len(source_nodes)):
        weighted_sum[source_nodes[i]] += edge_weights[i] * modified_diff[i]
        weighted_sum[target_nodes[i]] += edge_weights[i] * (-modified_diff[i])  # Negative for opposite direction

    # Step 2: Compute sum of weights for each node
    sum_weights = np.zeros_like(nodes_f)
    for i in prange(len(source_nodes)):
        sum_weights[source_nodes[i]] += edge_weights[i]
        sum_weights[target_nodes[i]] += edge_weights[i]

    # Step 3: Normalize weighted sums to get smoothed features (avoid division by zero)
    for i in prange(len(nodes_f)):
        if sum_weights[i] > 0:
            smoothed_f[i] = np.round(weighted_sum[i] / sum_weights[i])

    return smoothed_f

def smooth_graph(graph: Graph) -> NDArray[np.float64]:
    """
    Wrapper function to call Numba-optimized smoothing with a Graph object.
    
    Arguments:
    - graph: Graph object containing edges, features, and node features.
    
    Returns:
    - smoothed_f: Smoothed node features.
    """
    return weighted_laplacian_smoothing(graph.edges_nodes, graph.edges_feats, graph.nodes_f)
