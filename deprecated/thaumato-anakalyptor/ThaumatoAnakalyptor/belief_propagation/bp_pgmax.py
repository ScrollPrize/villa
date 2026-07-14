# Giorgio Angelotti - 2024
# Loopy Belief Propagation Solver with Iterated MAP estimation for the Sheet Stitching Problem in PGMax + Laplacian Smoothing
# BP's Log-potential implementation by Francesco Mori
# Laplacian smoothing inspired by Julian Schilliger Graph Problem Solver

import argparse
import jax
import numpy as np
from pgmax import vgroup, fgroup, fgraph, infer
from utils.factors import compute_ell, create_log_potential_matrix, compute_loss, smooth_graph
from utils.graph import load_graph_from_binary, Graph, extract_largest_connected_component, maximum_spanning_tree, bfs, update_f_star, save_graph, load_graph_from_binary_new, load_graph_from_binary_results
from tqdm import tqdm
from typing import Optional
import gc

class BPSolver:
    def __init__(self, graph_path, L: int, bp_iterations: int = 1000, z_min: Optional[int] = None, z_max: Optional[int] = None):
        self.graph_path = graph_path
        self.L = L
        self.tollerance = 10
        self.bp_iterations = bp_iterations

        if graph_path.endswith(".npz"):
            data = np.load(graph_path)
            self.graph = Graph(edges_nodes=data['edges_nodes'], edges_feats=data['edges_feats'], nodes_f=data['nodes_f'])
        else:
            try:
                self.graph = load_graph_from_binary_new(graph_path)
            except:
                self.graph = load_graph_from_binary(graph_path) # old format
        print(f"Num nodes: {self.graph.nodes_f.shape[0]}, num edges: {self.graph.edges_feats.shape[0]}")

        if (z_min is not None) & (z_max is not None):
            assert(z_max > z_min)
            old_to_new = self.z_filter(z_min, z_max)
            # Step 1: Reverse the old-to-new dictionary to get new-to-old
            new_to_old = {v: k for k, v in old_to_new.items()}
            print(f"After z_cut, Num nodes: {self.graph.nodes_f.shape[0]}, num edges: {self.graph.edges_feats.shape[0]}")
        print("Extracting largest connected component...")
        self.graph, keep_indices = extract_largest_connected_component(self.graph)
        # Step 2: Map the new indices of the connected component back to the old ones before cutting
        if (z_min is not None) & (z_max is not None):
            self.old_keep_indices = [new_to_old.get(new, None) for new in keep_indices]
        else:
            self.old_keep_indices = keep_indices

        self.num_nodes = self.graph.nodes_f.shape[0]
        self.num_edges = self.graph.edges_feats.shape[0]
        print(f"Num nodes: {self.num_nodes}, num edges: {self.num_edges}")
        print("Assigning initial condition...")
        mst = maximum_spanning_tree(self.graph)
        bfs(self.graph, mst, start=0)
        del mst
        gc.collect()
        self.error_stats()
        print("Initial Laplacian smoothing (100 iterations)")
        for _ in tqdm(range(100)):
            self.graph.nodes_f += smooth_graph(self.graph)*360
        self.error_stats()

        print("Graph loaded and initialized")
        self.initialize_bp()
        gc.collect()

        print("BP initialized")




    def error_stats(self):
        abs_error = np.abs(compute_ell(self.graph.nodes_f[self.graph.edges_nodes[:, 0]], self.graph.nodes_f[self.graph.edges_nodes[:, 1]], self.graph.edges_feats[:, 1], tol=0.01))
        # Define the quantiles you want to compute (e.g., 0% to 100% with small intervals)
        quantiles = np.linspace(0, 100, num=21)  # 21 quantiles (from 0% to 100%)

        # Compute the quantiles
        quantile_values = np.percentile(abs_error, quantiles)

        print(f"Min absolute error: {np.min(abs_error):.6f}")

        # Print the quantile values along with the corresponding percentiles
        for q, value in zip(quantiles, quantile_values):
            print(f"{q:.1f}th percentile: {value:.6f}")

        self.max_error = np.max(abs_error)
        print(f"Max absolute error: {self.max_error}")
        print(f"Avg absolute error: {np.mean(abs_error):.6f}")

    

    def initialize_bp(self):
        print("Initializing BP with updated factors")
        # Defining variabls and initializing factor graph on variables
        self.variables = vgroup.NDVarArray(num_states=2*self.L+1, shape=(self.num_nodes,))
        self.fg = fgraph.FactorGraph(variable_groups=self.variables)

        # Creating log potential matrix
        log_potential_matrices = create_log_potential_matrix(self.graph.edges_feats[:, 0], self.L, self.graph.nodes_f[self.graph.edges_nodes[:, 0]], self.graph.nodes_f[self.graph.edges_nodes[:, 1]], self.graph.edges_feats[:, 1])

        skipped = 0
        # Now, loop over the edges to create pairwise factors and add them to the factor graph
        for i in tqdm(range(self.graph.edges_nodes.shape[0]), desc="Creating factors"):
            # Use precomputed values to create the pairwise factor
            pairwise_factor = fgroup.PairwiseFactorGroup(
                variables_for_factors=[[self.variables[self.graph.edges_nodes[i, 0]], self.variables[self.graph.edges_nodes[i, 1]]]],
                log_potential_matrix=log_potential_matrices[i]
            )
            
            # Add the factor to the factor graph
            try:
                self.fg.add_factors(pairwise_factor)
            except:
                skipped += 1
                continue
        print(f"Factors skipped: {skipped/self.graph.edges_nodes.shape[0]}%")
        gc.collect()

        print("Initializing the BP state, can take a while...")
        # Set random seed
        rng = jax.random.PRNGKey(0)
        # Step 1: Initialize the BP solver
        self.bp = infer.build_inferer(self.fg.bp_state, backend="bp")

        # Step 2: Initialize the BP state
        self.bp_arrays = self.bp.init()

    def run(self):
        iteration = 0
        counter = 0
        while True:
            current_max_error = self.max_error
            bp_decoding = infer.decode_map_states(self.bp.get_beliefs(self.bp_arrays))
            # Compute the error using the decoded MAP states
            loss = compute_loss(self.graph, bp_decoding, self.variables, self.L, verbose=False)
            print(f"Big iteration: {iteration}, Loss: {loss}")

            self.bp_arrays = self.bp.run(self.bp_arrays, num_iters=self.bp_iterations, temperature=0)
            bp_decoding = infer.decode_map_states(self.bp.get_beliefs(self.bp_arrays))
            # Compute the error using the decoded MAP states
            loss = compute_loss(self.graph, bp_decoding, self.variables, self.L, verbose=False)
            print(f"Loss after BP: {loss}")

            # Update graph
            update_f_star(self.graph, bp_decoding, self.variables, self.L)
            self.error_stats()
            if self.max_error >= current_max_error:
                if counter >= self.tollerance:
                    break
                else:
                    counter += 1
            jax.clear_backends()
            gc.collect()
            self.initialize_bp()
            iteration += 1
        
        print("Final smoothing")
        # Final Laplacian smoothing
        self.graph.nodes_f += smooth_graph(self.graph)*360
        self.error_stats()


    def z_filter(self, z_min: int, z_max: int):
        nodes_to_keep = np.where((self.graph.nodes_z >= z_min) & (self.graph.nodes_z <= z_max))[0]
        #nodes_to_keep = np.where(~g_results.nodes_d)[0]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}
        nodes_to_keep_set = set(nodes_to_keep)
        mask = np.array([(u in nodes_to_keep_set) and (v in nodes_to_keep_set) for u, v in self.graph.edges_nodes])
        edges_filtered = self.graph.edges_nodes[mask]
        edges_feats_filtered = self.graph.edges_feats[mask]
        # Adjust node indices in the edges to match the new indices (from old_to_new)
        adjusted_edges_nodes = np.array(
                [[old_to_new[u], old_to_new[v]] for u, v in edges_filtered],
                dtype=np.uint32
            )
        nodes_f_filtered = self.graph.nodes_f[nodes_to_keep]
        #nodes_z_filtered = g.nodes_z[nodes_to_keep]
        self.graph = Graph(
                edges_nodes=adjusted_edges_nodes,
                edges_feats=edges_feats_filtered,
                nodes_f=nodes_f_filtered,
                #nodes_z=nodes_z_filtered,
            )
        return old_to_new
        
    def save_result(self, output):
        print("Loading again the original graph")
        if self.graph_path.endswith(".npz"):
            data = np.load(self.graph_path)
            original_graph = Graph(edges_nodes=data['edges_nodes'], edges_feats=data['edges_feats'], nodes_f=data['nodes_f'])
        else:
            try:
                original_graph = load_graph_from_binary_new(self.graph_path)
            except:
                original_graph = load_graph_from_binary(self.graph_path) # old format
        
        original_graph.nodes_f[self.old_keep_indices] = self.graph.nodes_f

        # Initializing deleted nodes
        original_graph.nodes_d = np.ones(original_graph.nodes_f.shape[0], dtype=np.bool_)
        original_graph.nodes_d[self.old_keep_indices] = 0

        save_graph(output, self.graph)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run BP Solver on a graph.")
    
    # Define the arguments
    parser.add_argument("graph_path", type=str, help="Path to the graph file (binary or npz format).")
    parser.add_argument("L", type=int, help="The range L (maximum shift) for the BP solver.")
    parser.add_argument("--bp_iterations", type=int, default=500, help="Number of BP iterations to run (default: 500).")
    parser.add_argument("--z_min", type=int, default=None, help="Minimum z value for filtering (default: None).")
    parser.add_argument("--z_max", type=int, default=None, help="Maximum z value for filtering (default: None).")
    parser.add_argument("--output", type=str, default="output_graph.npz", help="Output file path for saving results (default: output_graph.npz).")
    
    # Parse the arguments
    args = parser.parse_args()

    # Create the BPSolver instance with parsed arguments
    solver = BPSolver(
        graph_path=args.graph_path, 
        L=args.L, 
        bp_iterations=args.bp_iterations, 
        z_min=args.z_min, 
        z_max=args.z_max
    )

    # Run the BP solver
    solver.run()

    # Save the result
    solver.save_result(args.output)

if __name__ == "__main__":
    main()

# Example usage:
# python bp_pgmax.py path/to/graph.npz 4 --bp_iterations 500 --output final_result.npz




    










