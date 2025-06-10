// mean_solver.h
#ifndef MEAN_NORMALIZED_SOLVE_GPU_H
#define MEAN_NORMALIZED_SOLVE_GPU_H

#include <vector>
#include "node_structs.h"
#include <string>

std::vector<Node> run_solver_f_star_normalized(std::vector<Node>& graph,
    int num_iterations,
    std::vector<size_t>& valid_indices,
    Edge** h_all_edges,
    float** h_all_sides,
    float spring_constant,
    float other_block_factor,
    float lr,
    float error_cutoff,
    bool visualize);

// Plot nodes (scatter f_star vs f_init) via OpenCV
void plot_nodes(const std::vector<Node>& graph, const std::string& filename);

#endif // MEAN_SOLVE_GPU_H