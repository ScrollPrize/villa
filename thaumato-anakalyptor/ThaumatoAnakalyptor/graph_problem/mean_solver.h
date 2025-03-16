// mean_solver.h
#ifndef MEAN_SOLVE_GPU_H
#define MEAN_SOLVE_GPU_H

#include <vector>
#include "node_structs.h"
#include <string>

std::vector<Node> run_solver_f_star_with_labels(std::vector<Node>& graph, int num_iterations, size_t seed_node, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, float spring_constant, float other_block_factor, float lr, float error_cutoff, bool visualize);

#endif // MEAN_SOLVE_GPU_H