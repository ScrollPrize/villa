// solve_tugging.h
#pragma once
#include <vector>
#include "node_structs.h"
// Runs the f_star solver with an additional "tugging" step:
// Each iteration, for each active node, counts neighbors with f_star above/below,
// computes diff = (#front - #behind), stores diff in happiness_old,
// splits diff by 'distribute' between self and neighbors,
// then applies the standard f_star update.
// graph: input graph (will be updated in place),
// num_iterations: number of solver iterations,
// valid_indices: list of active node indices,
// h_all_edges, h_all_sides: host pointers for edge/sides memory management,
// distribute: fraction [0,1] of diff distributed to neighbors,
// diff_step: scaling factor for tugging diff when added to f_star.
// Runs the tugging-augmented f_star solver with full f_star parameters.
std::vector<Node> run_solver_tugging(
    std::vector<Node>& graph,
    int num_iterations,
    std::vector<size_t>& valid_indices,
    Edge** h_all_edges,
    float** h_all_sides,
    int i_round,
    float o,
    float spring_constant,
    float step_sigma,
    float distribute,
    float diff_step,
    bool visualize
);