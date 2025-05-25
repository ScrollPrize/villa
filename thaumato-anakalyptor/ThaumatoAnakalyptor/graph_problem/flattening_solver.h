// flattening_solver.h
#ifndef FLATTENING_SOLVER_H
#define FLATTENING_SOLVER_H

#include <vector>
#include <utility>  // for std::pair
#include "node_structs.h"

/**
 * Flattening solver: adjusts both f_init (z) and f_star (winding) coordinates
 * so that the Euclidean distance between nodes matches edge.k.
 * Applies additional tugs based on initial z (wnr_side) and initial winding (wnr_side_old).
 * NEW: Also includes angle-based updates that preserve 3D angles between neighbor triplets
 * by adjusting 2D positions to match the angles formed by neighbor1-node-neighbor2 in 3D space.
 * @param graph           Input graph (vector of Node), will be modified in place and returned.
 * @param num_iterations  Number of iterations to run.
 * @param valid_indices   Indices of nodes that are not deleted (to update).
 * @param h_all_edges     Unused: placeholder for edge memory management.
 * @param h_all_sides     Unused: placeholder for side memory management.
 * @param z_tug_min       Lower threshold on initial z for additional tug.
 * @param z_tug_max       Upper threshold on initial z for additional tug.
 * @param angle_tug_min   Lower threshold on initial winding angle for additional tug.
 * @param angle_tug_max   Upper threshold on initial winding angle for additional tug.
 * @param tug_step        Magnitude of additional tug step.
 * @return Modified graph with updated f_init and f_star.
 */
std::vector<Node> run_solver_flattening(
    std::vector<Node>& graph,
    int num_iterations,
    const std::vector<size_t>& valid_indices,
    Edge** h_all_edges,
    float** h_all_sides,
    float z_tug_min,
    float z_tug_max,
    float angle_tug_min,
    float angle_tug_max,
    float tug_step,
    float init_z_tug,
    const std::vector<std::pair<float, float>>& zero_ranges,
    bool visualize = false
);

#endif // FLATTENING_SOLVER_H