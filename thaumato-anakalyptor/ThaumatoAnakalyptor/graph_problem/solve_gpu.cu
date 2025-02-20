/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/
#include "solve_gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <stack>
#include <cmath>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <random>
#include <queue>
#include <numeric>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>

__device__ float update_sides(Node node, Node target_node, float certainty, int side_transition, bool same_block, bool convergence_speedup = false) {
    float loss = 0.0f; // missalignment loss
    float target_node_sum = 0.0f;
    float node_sum = 0.0f;
    // calculate the sum of the sides for normalization
    for (int i = 0; i < target_node.sides_nr; ++i) {
        target_node_sum += fmaxf(0.0f, target_node.sides_old[i]);
    }
    for (int i = 0; i < node.sides_nr; ++i) {
        node_sum += fmaxf(0.0f, node.sides_old[i]);
    }
    // assign node_sides to the sides array
    for (int i = 0; i < node.sides_nr; ++i) {
        int index = (i - side_transition + 3 * node.sides_nr) % node.sides_nr; // - side_transition to move the side to the correct position in the inverse direction (target -> node)
        node.sides[index] = fmaxf(0.0f, node.sides[index] + certainty * target_node.sides_old[i]);
        // decrease the opposite sides value
        if (convergence_speedup) {
            int index_opposite = (index + node.sides_nr / 2) % node.sides_nr; // opposite side
            node.sides[index_opposite] = fmaxf(0.0f, node.sides[index_opposite] - 0.25 * certainty * target_node.sides_old[i]);
        }

        if (node_sum > 0.0f) {
            loss += fmaxf(0.0f, fmaxf(0.00001f, node.sides_old[index]) / node_sum) * fmaxf(0.00001f, target_node_sum - fmaxf(0.0f, target_node.sides_old[i]))/fmaxf(0.00001f, target_node_sum); // node index p * target not index p
        }
        else { // no sides with probabilities, no loss
            loss += 1.0f;
        }
    }
    // too little edges to even calculate the loss
    if (node.sides_nr <= 2) {
        loss = 1.0f;
    }
    loss = fmaxf(0.0f, fminf(1.0f, loss));
    return loss;
}

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel(Node* d_graph, size_t* d_valid_indices, float o, float spring_constant, int num_valid_nodes, float lowest_adjustment, float highest_adjustment, float z_min, float z_max, int estimated_windings, float small_pull, float edge_pull, int down_index, int up_index, float min_f_star, float max_f_star, bool apply_certainty_k_scaling, int i_round, int seed_node, float other_block_factor, bool adjusting_side, unsigned long seed, int iter, float sides_moving_eps, bool increase_same_block_weight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];

    if (node.deleted) return;
    // if (node.fixed) {
    //     return;
    // }

    // Initialize random number generator
    curandState state;
    curand_init(seed * idx, idx, 0, &state);

    // f star alignment discrepancy
    float f_star = node.f_star;
    bool is_compute = f_star + 4 * 360 <= lowest_adjustment || f_star - 4 * 360 >= highest_adjustment;
    // Check for z range
    is_compute = is_compute || (node.z < z_min || node.z > z_max);

    // if (!is_compute) {
    // Check if the f_star alignment to f_init is off by a lot for windings that are close to the adjustment/fix boarder
    float f_init = node.f_init;
    float f_star_discrepancy = roundf((f_star - f_init) / 360.0f) * 360.0f;
    // f star alignment
    float f_star_aligned = f_init + f_star_discrepancy;
    float f_star_offset = f_star_aligned - f_star;
    // Construct f stars with weights of the offset
    float f_star_offset_1 = f_star + 4 * f_star_offset;
    float f_star_offset_2 = f_star - 4 * f_star_offset;
    // Check if any value is outside the lowest or highest adjustment
    is_compute = is_compute || f_star_offset_1 <= lowest_adjustment || f_star_offset_1 >= highest_adjustment || f_star_offset_2 <= lowest_adjustment || f_star_offset_2 >= highest_adjustment;
    // }

    // Only adjust the node if the f_star winding angle value is outside the fixed angles
    float sum_w_f_tilde_k = 0.0f;
    float sum_w = 0.0f;
    float node_f_tilde = node.f_tilde;
    float happiness = 0.0f;
    float edge_happy_accuracy = 0.0f;
    float edge_happy_accuracy_sum = 0.0f;
    float happiness_sum = 0.0f;
    float happiness_keepfactor = 5.0f;
    float edge_contribution = 2.0f;
    float side_sum = 0.0f;
    float side_sum_count = 0.0f;
    float side_sum_count_array = 0.0f;
    float side_missalignment_loss = 0.0f;
    // float f_star_factor = (node.f_star - min_f_star) / (max_f_star - min_f_star);
    // f_star_factor = fminf(1.0, fmaxf(0.1f, f_star_factor));
    // float f_star_factor = 1.0f;

    int max_winding_number = 0;
    float max_winding_number_side = 0.0f;
    float total_winding_number_side = 0.0f;
    float total_winding_number_side_certainty = 0.0f;
    float smeared_confidence_sum = 0.0f;
    float smeared_confidence_sum_count = 0.0f;

    Edge* edges = node.edges;
    int num_active_edges = 0;

    // recalculate edges wnr based on the target wnr
    for (int j = 0; j < node.num_edges; ++j) {
        Node& target_node = d_graph[edges[j].target_node];
        int target_wnr = target_node.winding_nr_old;

        float target_angle = 360 * target_wnr + target_node.f_init;
        float recalculated_node_wnr = (target_angle - edges[j].k - node.f_init) / 360.0f;
        int node_wnr = static_cast<int>(std::round(recalculated_node_wnr));
        edges[j].wnr_from_target = node_wnr;
        if (edges[j].wnr_from_target != node_wnr) {
            printf("Error: recalculated wnr is not an integer\n");
        }
        // Check if edge is closeish in f star
        if (!target_node.deleted && fabsf(node.f_tilde + spring_constant * edges[j].k - target_node.f_tilde) < (3 * 360.0f)) {
            float certainty = fmaxf(0.001f, edges[j].certainty);
            if (certainty < 0.001f) {
                continue;
            }
            if (!edges[j].same_block) {
                certainty *= other_block_factor;
            }
            else {
                certainty *= 7.5f;
            }
            // wrong edge wheight, less confidence in side
            if (edges[j].wnr_from_target != node.winding_nr_old) {
                certainty *= 10.0f;
            }
            else {
                total_winding_number_side += fmaxf(0.0f, certainty * target_node.wnr_side_old);
            }
            total_winding_number_side_certainty += certainty;
        }
    }
    if (total_winding_number_side_certainty < 0.0001f) {
        total_winding_number_side_certainty = 0.0001f;
    }
    total_winding_number_side = fmaxf(0.0f, fminf(1.0f, total_winding_number_side / total_winding_number_side_certainty));
    node.total_wnr_side = total_winding_number_side; // save the total winding number side for the node. usedto calculate percentual winding number side accuracy

    // loop over all edges and update the nodes
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        if (d_graph[target_node].deleted) continue;
        float n2n1 = d_graph[target_node].f_tilde - node_f_tilde;
        float k_abs = fabsf(spring_constant * edge.k) - fabsf(n2n1);
        // bool k_abs1 = k_abs > 0.0f; // push out the nodes
        // bool k_abs2 = fabsf(k_abs) - 2 * 360 < 0.0f; // pull in the nodes if they are not very far away
        // bool k_abs3 = edge.k * n2n1 < 0; // push out the nodes if the edge points in the opposite way
        // bool overshot = n2n1 - edge.k * spring_constant > 0.0f;

        float certainty = edge.certainty;
        float k = edge.k;

        if (certainty < 0.001f) {
            continue;
        }

        // adjust certainty from connectivity
        if (node.connectivity >= 0.0f) {
            float d_con = fabsf(node.connectivity - d_graph[target_node].connectivity);
            if (!edge.same_block) { // other block edge, same sheet
                if (d_con < 0.5f) {
                    certainty *= 1000.0f;
                }
            }
            else { // adjacent sheet
                if (d_con > 0.5f && d_con < 1.5f)  {
                    certainty *= 1000.0f;
                }
            }
        }

        float k_diff = n2n1 - k * spring_constant;

        // Calculate side update
        {
            // Assign if edge inside same, or going up or down the side
            bool same = fabsf(node.f_init + edge.k - d_graph[target_node].f_init) < 1.0f; // same winding
            bool up = node.f_init + edge.k - d_graph[target_node].f_init > 1.0f; // up winding
            bool down = node.f_init + edge.k - d_graph[target_node].f_init < -1.0f; // down winding
            bool seam = fabsf(node.f_init - d_graph[target_node].f_init) >= 180.0f; // seam winding
            
            float side = d_graph[target_node].side;
            // space side a little bit
            // side -= 0.5f;
            // side *= 1.1f;
            // side += 0.5f;

            bool switched = fabsf(node.f_tilde - d_graph[target_node].f_tilde) > 30.0f;

            // Adjust the certainty based on how accurate the edge is
            float k_asb_d = fabsf(node_f_tilde + spring_constant * edge.k - d_graph[target_node].f_tilde) / fmaxf(fabsf(edge.k), 0.01f);
            float k_abs_certainty_factor = 1.0f / (k_asb_d * k_asb_d / 2.0f + 1.0f);
            k_abs_certainty_factor = fmaxf(0.2f, k_abs_certainty_factor);

            float certainty = edge.certainty;
            // certainty *= k_abs_certainty_factor;
            // scale by accuracy of edge
            // float dk = n2n1 - spring_constant * k;
            // float acc_scale = fmaxf(0.0f, 1.0f - fabsf(dk) / (4*360.0f));
            // certainty *= acc_scale;

            // if (fabsf(d_graph[target_node].z - node.z) > 15.0f) {
            //     certainty *= 1.10f;
            // }

            // Check if edge is closeish in f star
            if (fabsf(node.f_tilde + spring_constant * edge.k - d_graph[target_node].f_tilde) < (3 * 360.0f)) {
                // Only use good ones
                if (!edge.same_block) {
                    certainty *= other_block_factor;
                    // float certainty = 0.1f;
                    if (same) {
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                    else if (up && (up_index == node.side_number)) {
                        side = - side;
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                    else if (down && (down_index == node.side_number)) {
                        side = - side;
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                    else {
                        // inside the same group of layers
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                }
                else {
                    // if (edge.k < 0.0f) { // adjust to only have one main scroll direction flow
                    //     certainty *= 0.9f;
                    // }
                    certainty *= 10.5f;
                    // float certainty = 0.1f;
                    if (seam) {
                        if ((node.side_number + 1 == up_index) && (edge.k > 0.0f)) { // switching the side
                            side = - side;
                            side_sum += certainty * side;
                            side_sum_count += certainty;
                        }
                        else if ((node.side_number == up_index) && (edge.k > 0.0f)) {
                            // only one side number? if so it is a double switch
                            if (1 > up_index) { // double switching, = same side
                                side_sum += certainty * side;
                                side_sum_count += certainty;
                            }
                            else { // switching the side
                                side = - side;
                                side_sum += certainty * side;
                                side_sum_count += certainty;
                            }
                        }
                        else if ((node.side_number == down_index) && (edge.k < 0.0f)) {
                            // only one side number? if so it is a double switch
                            if (1 > up_index) { // double switching, = same side (down index allways 0, up index is length of grouped side numbers.)
                                side_sum += certainty * side;
                                side_sum_count += certainty;
                            }
                            else { // switching the side
                                side = - side;
                                side_sum += certainty * side;
                                side_sum_count += certainty;
                            }
                        }
                        else if ((node.side_number - 1 == down_index) && (edge.k < 0.0f)) { // switching the side
                            side = - side;
                            side_sum += certainty * side;
                            side_sum_count += certainty;
                        }
                        else {
                            // inside the same group of layers
                            side_sum += certainty * side;
                            side_sum_count += certainty;
                        }
                    }
                    else if ((up_index == node.side_number) && edge.k > 0.0f) {
                        side = - side;
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                    else if ((down_index == node.side_number) && edge.k < 0.0f) {
                        side = - side;
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                    else {
                        // inside the same group of layers
                        side_sum += certainty * side;
                        side_sum_count += certainty;
                    }
                }
            }
        }

        // Calculate side array update
        {
            // calculate winding nr (side) difference between node and target node
            float recalculated_node_wnr_d = (node.f_init + edge.k - d_graph[target_node].f_init) / 360.0f;
            int node_wnr_dif = __float2int_rn(recalculated_node_wnr_d);
            float k_tilde_dif = d_graph[target_node].f_tilde - node.f_tilde;

            float certainty = fmaxf(0.001f, edge.certainty);
            // calculate the max indices for the side to compare node to target edge wnr dif
            int index_max_sides_node = 0;
            int index_max_sides_target = 0;
            for (int i = 0; i < node.sides_nr; ++i) {
                if (node.sides_old[i] > node.sides_old[index_max_sides_node]) {
                    index_max_sides_node = i;
                }
                if (d_graph[target_node].sides_old[i] > d_graph[target_node].sides_old[index_max_sides_target]) {
                    index_max_sides_target = i;
                }
            }
            int num_same_block = 0;
            if (!edge.same_block) {
                // if (fabsf(node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde) < 1.0f) { // edge should most likely be good
                //     certainty *= 10.0f;
                // }
                if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.10f) { // edge should most likely be good
                    certainty *= 60.0f;
                }
                else if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.20f) { // edge should most likely be good
                    certainty *= 30.0f;
                }
                else if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.30f) { // edge should most likely be good
                    certainty *= 10.0f;
                }
                // higher certainty if dist is smaller than 2
                // int dist_max_sides = abs(((index_max_sides_node + node_wnr_dif - index_max_sides_target) + 3 * node.sides_nr) % node.sides_nr); // think of the sides as a circle
                // dist_max_sides = min(dist_max_sides, node.sides_nr - dist_max_sides);
                // if (dist_max_sides <= 1 && fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.30f) {
                //     certainty *= 3.0f;
                // }
                certainty *= other_block_factor;
            }
            else {
                if (k_tilde_dif * node_wnr_dif < 0.0f) {
                    // edge probably points in the wrong direction
                    certainty *= 0.01f;
                    node_wnr_dif = node_wnr_dif - (node_wnr_dif > 0 ? 1 : -1);
                }
                if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.10f) { // edge should most likely be good
                    certainty *= 90.0f; // 90
                    if (increase_same_block_weight) {
                        certainty *= 1.5f;
                    }
                }
                else if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.20f) { // edge should most likely be good
                    certainty *= 55.0f; // 55
                    if (increase_same_block_weight) {
                        certainty *= 1.25f;
                    }
                }
                else if (fabsf((node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde)/edge.k) < 0.30f) { // edge should most likely be good
                    certainty *= 35.0f;
                }
                // higher certainty if dist of max index in node vs target is larger eq than 2
                // int dist_max_sides = abs(((index_max_sides_node + node_wnr_dif - index_max_sides_target) + 3 * node.sides_nr) % node.sides_nr); // think of the sides as a circle
                // dist_max_sides = min(dist_max_sides, node.sides_nr - dist_max_sides);
                // if (dist_max_sides >= 2 && fabsf(node.f_tilde + spring_constant * edge.k - d_graph[edge.target_node].f_tilde) < 180.0f) {
                // if (dist_max_sides >= 2) {
                //     certainty *= 3.0f;
                // }
                // certainty *= 30.5f;
                if (increase_same_block_weight) {
                    certainty *= 3.0f;
                }
                num_same_block++;
            }
            // adjust edge based on the number of edges and the number of same block edges
            float nr_edges_factor = 1.0f / fmaxf(1.0f, 2.15f - node.num_edges);
            float same_block_factor = 1.0f / fmaxf(1.0f, 1.025f - num_same_block);
            float certainty_factor = nr_edges_factor * same_block_factor;
            certainty *= certainty_factor;
            if (edge.certainty > 0.0f || edge.certainty > 0.001f) {
                // Check if edge is closeish in f tilde
                if (fabsf(node.f_tilde + spring_constant * edge.k - d_graph[target_node].f_tilde) < (2.5f * 360.0f)) {
                    side_missalignment_loss += update_sides(node, d_graph[target_node], certainty, node_wnr_dif, edge.same_block);
                    side_sum_count_array += certainty;
                }
                // else {
                //     // bad connection
                //     side_missalignment_loss += update_sides(node, d_graph[target_node], -certainty, node_wnr_dif, edge.same_block);
                // }
            }
        }

        // Calculate winding number update
        {   
            // Find the max winding number wrt sum of certainty
            float sum_unnormalized_side_j = 0.0f;
            float sum_normalization_j = 0.0f;
            int winding_number_j = edges[j].wnr_from_target; // set the winding number we search for
            float certainty_sum = 0.0f;
            int num_same_block = 0;
            for (int l = 0; l < node.num_edges; ++l) {
                if (d_graph[edges[l].target_node].deleted) continue;
                float certainty = fmaxf(0.001f, edges[l].certainty);
                certainty_sum += certainty;
                // if (node.winding_nr_old * edges[l].k < 0.0f) { // prefer direction away from 0 winding
                //     certainty *= 0.9f;
                // }
                if (!edges[l].same_block) {
                    if (fabsf(node.f_tilde + spring_constant * edges[l].k - d_graph[edges[l].target_node].f_tilde) < (1.0f)) { // edge should most likely be good
                        certainty *= 10.0f;
                    }
                    certainty *= other_block_factor;
                }
                else {
                    certainty *= 30.5f;
                    num_same_block++;
                }
                // edge is certain enough
                if (edges[l].certainty > 0.001f) {
                    float wnr_old = d_graph[edges[l].target_node].wnr_side_old;
                    // Check if edge is closeish in f star
                    if (fabsf(node.f_tilde + spring_constant * edges[l].k - d_graph[edges[l].target_node].f_tilde) < (2.5f * 360.0f)) {
                        // Only use good ones for the winding number side value
                        if (edges[l].wnr_from_target == winding_number_j) {
                            sum_unnormalized_side_j += certainty * wnr_old;
                        }
                        // else {
                        //     // scale by how much off it is
                        //     float wnr_d_scale = fabsf(edges[l].wnr_from_target - winding_number_j);
                        //     certainty *= wnr_d_scale;
                        //     // if (!edges[l].same_block && fabsf(node.f_star + spring_constant * edges[l].k - d_graph[edges[l].target_node].f_star) < 3.0f) {
                        //     //     // edge should most likely be good but has the wrong winding number
                        //     //     certainty *= 10.0f;
                        //     // }
                        //     sum_unnormalized_side_j -= certainty * wnr_old / 2.0f; // make certainty lower if the winding number is not correctly matched
                        // }
                    }
                    else {
                        sum_unnormalized_side_j -= certainty * wnr_old; // make certainty lower if the winding number is not correctly matched
                    }
                }
                sum_normalization_j += certainty; // sum up the certainty over all edges of non-deleted nodes
            }
            // Update the max winding number
            float nr_edges_factor = 1.0f / fmaxf(1.0f, 2.15f - node.num_edges);
            float sum_normalization_j_factor = fminf(1.0f, certainty_sum / 0.05f);
            float same_block_factor = 1.0f / fmaxf(1.0f, 1.025f - num_same_block);
            float sum_side_j = nr_edges_factor * same_block_factor * fminf(1.0f, fmaxf(0.0f, sum_unnormalized_side_j / sum_normalization_j));
            if (sum_normalization_j <= 0.001f) {
                sum_side_j = 0.0f;
            }
            if (sum_side_j > max_winding_number_side) {
                max_winding_number_side = sum_side_j;
                max_winding_number = winding_number_j;
            }
        }

        // happyness calculation
        {
            int wnr_from_target = edge.wnr_from_target;
            int wnr_node = node.winding_nr_old;

            bool seam = fabsf(node.f_init - d_graph[target_node].f_init) >= 180.0f; // seam winding
            float certainty = 1.0f;
            float side_node_confidence = fmaxf(0.0f, fminf(1.0f, node.wnr_side_old / node.total_wnr_side));
            float side_target_confidence = fmaxf(0.0f, fminf(1.0f, d_graph[target_node].wnr_side_old / d_graph[target_node].total_wnr_side));
            float confidence = fmaxf(0.001f, fminf(1.0f, side_node_confidence * side_target_confidence));
            confidence = fmaxf(0.0f, sqrt(confidence)); // normalize the confidence
            // certainty *= * node.sides_old[max_sides_node] * d_graph[target_node].sides_old[max_sides_target];
            certainty *= confidence;
            // side confidence
            float side_confidence = fmaxf(0.001f, fminf(1.0f, fmaxf(0.0f, fminf(1.0f, node.wnr_side_old)) * fmaxf(0.0f, fminf(1.0f, d_graph[target_node].wnr_side_old))));
            side_confidence = fmaxf(0.0f, sqrt(side_confidence)); // normalize the confidence
            certainty *= side_confidence;
            if (node.wnr_side_old > 0.001f && edge.certainty > 0.0f && certainty > 0.0f) { // only calculate hapyness if the node actually is reasonable certain of its winding number
                if (fabsf(node.f_tilde + spring_constant * edge.k - d_graph[target_node].f_tilde) < (3 * 360.0f)) {
                    if (!edge.same_block) { // other block edge, same sheet
                        certainty *= other_block_factor;
                        if (!seam && wnr_from_target == wnr_node) { 
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else if (seam && edge.k > 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else if (seam && edge.k < 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else {
                            // bad connection, weight higher
                            certainty *= 10.0f;
                            happiness += 10.0f * d_graph[target_node].happiness_old;
                            happiness_sum += 10.0f;
                        }
                    }
                    else {
                        certainty *= 10.5f;
                        // float same_block_surrounding_nodes_factor = 6.0f;
                        // float same_block_surrounding_nodes_factor = 10.0f;
                        float same_block_surrounding_nodes_factor = 1.0f;
                        float same_block_bad_surrounding_nodes_factor = 0.5f;
                        if (edge.k * (d_graph[target_node].f_tilde - node.f_tilde) > 0.0f) {
                            // bad connection
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else if (seam) {
                            if (edge.k > 0 && wnr_from_target == wnr_node) { // It is a double switch
                                edge_happy_accuracy += certainty;
                                happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_surrounding_nodes_factor;
                            }
                            else if (edge.k < 0 && wnr_from_target == wnr_node) { // It is a double switch
                                edge_happy_accuracy += certainty;
                                happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_surrounding_nodes_factor;
                            }
                            else {
                                // bad connection, weight higher
                                certainty *= 2.0f;
                                happiness +=same_block_bad_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_bad_surrounding_nodes_factor;
                            }
                        }
                        else if (edge.k > 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else if (edge.k < 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else {
                            // bad connection, weight higher
                            certainty *= 10.0f;
                            happiness += same_block_bad_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_bad_surrounding_nodes_factor;
                        }
                    }
                }
                else {
                    //Extremely bad node
                    certainty *= 1000.0f;
                    // happiness += 10.0f * d_graph[target_node].happiness_old;
                    // happiness_sum += 10.0f;
                }
            }
            else {
                certainty = fabsf(certainty) + 1.0f;
                // happiness += 10.0f * d_graph[target_node].happiness_old;
                // happiness_sum += 10.0f;
                if (!edge.same_block) { // other block edge, same sheet
                    certainty *= other_block_factor;
                }
                else {
                    certainty *= 10.5f;
                }
            }
            edge_happy_accuracy_sum += certainty;
        }

        // smeared confidence calculation
        {
            float old_factor = 200.0f;
            float edge_type_factor = 1.0f;
            if (edge.same_block) { // too strong of an impact on the same block edges
                edge_type_factor = 0.1f;
            }
            smeared_confidence_sum += edge_type_factor * (d_graph[target_node].confidence_old + old_factor*d_graph[target_node].smeared_confidence_old);
            smeared_confidence_sum_count += edge_type_factor * (1.0f + old_factor);
        }

        // Node winding angle update calculation
        if (edge.same_block) {
            if (i_round > 4 && fabsf(k_diff) > 5*360) {
                continue;
            }
            // only use edge to push out, not contract. continue if edge would contract
            float divisor_k = fmaxf(fabsf(k), 0.01f) * ( edge.k > 0.0f ? 1.0f : -1.0f);
            float ratio = n2n1 / divisor_k;
            if (ratio > 1.0f && i_round > 0) {
                continue;
            }
            if (ratio < -0.50f) {
                continue;
            }

            if (ratio > 0.35 && ratio < 1.0f) {
                float additional_windings = k > 0.0f ? 360.0f : -360.0f;
                // add extra push out windings to the same block edges
                k += 3*additional_windings;
            }
        }
        else if (true || apply_certainty_k_scaling) {
            float dk = n2n1 - spring_constant * k;
            float di = d_graph[edge.target_node].f_init - node.f_init;
            if (di >= 180.0f) {
                di -= 360.0f;
            }
            else if (di <= -180.0f) {
                di += 360.0f;
            }
            else if (di == 0.0f) {
                di = 0.0001f;
            }
            if (fabsf(di) > 10.0f) {
                di = 10.0f * (di > 0.0f ? 1.0f : -1.0f);
            }
            
            di *= spring_constant;


            if (i_round <= 1) {
                float ratio = dk / di;
                // make surface flatter, not flat at this edge, increase certainty to make flatter
                if ((ratio > 1.1f && ratio < 5.0f) || (ratio < 0.9f && ratio > -4.0f)) {
                    certainty *= 10.0f;
                }
                else if ((ratio > 5.0f || ratio < -4.0f) && fabsf(dk) < 10*360) {
                    certainty *= 100.0f;
                }
                // scale by length of edge
                float edge_length = fabsf(edge.k);
                float scale_factor = fmaxf(1.0f, edge_length) / 10;
                certainty *= scale_factor;
            }
            else {
                float p_wa = d_graph[target_node].f_tilde - spring_constant * k;
                if (p_wa < node.f_tilde) {
                    certainty *= 1.001f;
                }
            }
        }

        float predicted_winding_angle = d_graph[target_node].f_tilde - spring_constant * k;
        float actual_angle = node.f_tilde;
        float abs_winding = fabsf(predicted_winding_angle - actual_angle) / 360.0f;
        float abs_winding_percent = abs_winding / edge_pull;
        if (edge.same_block) {
            abs_winding_percent /= 4.0f;
        }
        abs_winding_percent = abs_winding_percent * abs_winding_percent;
        float abs_factor = fminf(1.0f, fmaxf(0.0, 1.0f - abs_winding_percent));
        
        // sum_w_f_tilde_k += certainty * ((1.0f - edge_pull) * predicted_winding_angle + edge_pull * f_star_aligned_predicted);
        sum_w_f_tilde_k += certainty * predicted_winding_angle;
        sum_w += certainty;

        // Calculate node happiness: mean difference between k and target f_tilde - node f_tilde target + target node happiness weighted multiplied by the certainty
        num_active_edges++;
    }
    if (true || is_compute) {
        if (sum_w > 0)
        {
            node.f_star = (sum_w_f_tilde_k + o * node_f_tilde) / (sum_w + o);
        }
        // Pull node a little bit towards the f_init + k * 360 place
        if (small_pull > 0.0f) {
            node.f_star = small_pull * f_star_aligned + (1 - small_pull) * node.f_star;
        }
    }
    else {
        // Make sure to have the f_star aligned to f_init + k*360
        node.f_star = f_star_aligned;
    }
    // Clip f_star to the range [ - 2 * 360 * estimated_windings, 2 * 360 * estimated_windings]
    node.f_star = fmaxf(- 3 * 360 * estimated_windings, fminf(3 * 360  * estimated_windings, node.f_star));

    // Assign side
    if (side_sum_count > 0) {
        float side_o = 0.01f * o;
        float side_o_factor = (node.f_star - min_f_star) / (max_f_star - min_f_star);
        side_o_factor = fmaxf(0.1f, side_o_factor);
        // clip to 0.1 - 1.0
        side_o = side_o + side_o * side_o_factor;
        node.side = (side_sum + side_o * node.side) / (side_sum_count + side_o);
        // // space the side a little bit
        // node.side -= 0.5f;
        // node.side *= 1.08f;
        // node.side += 0.5f;
        // clip 0 - 1
        // node.side = fmaxf(0.0f, fminf(1.0f, node.side));
    }

    float max_value = 1000000.0f;
    if (idx == seed_node && !adjusting_side) {
        node.side = -max_value;
        // print in kernel
        // printf("Node %d: %f\n", i, node.side);
    }
    else if (seed_node < 0) {
        // space side a little bit
        node.side *= 1.01f;
        node.side = fmaxf(-max_value, fminf(max_value, node.side));
    }
    else if (node.fixed && !adjusting_side) {
        // fix the side value
        node.side = node.side > 0.0f ? max_value : -max_value;
    }

    // sides update normalization
    float norm_sum = 0.0f;
    int max_index_sides = 0;
    side_missalignment_loss = fmaxf(0.0f, fminf(1.0f, 1.0f - (side_missalignment_loss / node.num_edges)));
    if (side_sum_count_array <= 0.001) {
        side_missalignment_loss = 0.0f;
        node.deleted = true;
    }
    // float sides_diminishing = 0.999f;
    float noise_level = 0.00000001f;
    for (int i = 0; i < node.sides_nr; ++i) {
        if (side_sum_count_array > 0.0f) {
            node.sides[i] = node.sides[i] / side_sum_count_array;
            // node.sides[i] *= sides_diminishing;
            // add a little random noise
            float noise = curand_uniform(&state) * 2.0f * noise_level - noise_level; // Noise in range [-noise_level, noise_level]
            node.sides[i] += noise;
            node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i]));
        }
        else {
            node.sides[i] = 0.0f;
        }
        if (node.sides[i] > node.sides[max_index_sides]) {
            max_index_sides = i;
        }
        // node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i]));
        if (side_sum_count_array <= 0.0f) {
            side_sum_count_array = 1.0f;
        }
        norm_sum += node.sides[i];
    }
    float nr_edges_factor = 1.0f / fmaxf(1.0f, 5.0f - node.num_edges); // higher confidence in well connected nodes
    float confidence_node = cbrt(side_missalignment_loss * nr_edges_factor * fmaxf(0.0f, fminf(1.0f, (6.0f / 5.0f) * (node.sides[max_index_sides] / norm_sum - 1.0f / 6.0f))));
    // float same_block_factor = 1.0f / fmaxf(1.0f, 1.025f - num_same_block);
    node.confidence = fmaxf(0.0f, fminf(1.0f, confidence_node));
    node.smeared_confidence = fmaxf(0.0f, fminf(1.0f, smeared_confidence_sum / smeared_confidence_sum_count));
    node.closeness = fmaxf(0.0f, fminf(1.0f, norm_sum));
    float confidence_t = 0.5f;
    float smeared_confidence_t = 0.45f;
    float closeness_t = 0.3f;
    float t1 = node.confidence < confidence_t ? 0.0f : (node.confidence - confidence_t) / (1.0f - confidence_t);
    float t2 = node.smeared_confidence < smeared_confidence_t ? 0.0f : (node.smeared_confidence - smeared_confidence_t) / (1.0f - smeared_confidence_t);
    float t3 = node.closeness < closeness_t ? 0.0f : 1.0f; // indicator variable
    float simple_muliplied = sqrt(node.confidence * node.smeared_confidence * t3);
    float complex_multiplied = sqrt(t1 * t2 * t3);
    if (true) {
        node.happiness_v2 = simple_muliplied;
    }
    else {
        node.happiness_v2 = complex_multiplied;
    }
    // // Make nonmax indices smaller/0
    // for (int i = 0; i < node.sides_nr; ++i) {
    //     if (i != max_index_sides) {
    //         node.sides[i] = 0.0f;
    //     }
    // }

    // BUCKET TRICK: move believes from bucket to bucket to counteract two believes per wrap (bleeding)
    if (sides_moving_eps > 0.0f) {
        float sum_bucket = 0.0f;
        float sum_full_bucket = 0.0f;
        int random_direction = curand(&state) % 2;
        if (random_direction % 2 == 0) {
            float sides_0 = node.sides[0];
            sum_bucket = sides_0;
            for (int i = 0; i < node.sides_nr - 1; ++i) {
                sum_bucket += node.sides[i + 1];
                node.sides[i] += (sides_moving_eps * node.sides[i + 1]);
                sum_full_bucket += node.sides[i];
            }
            node.sides[node.sides_nr - 1] += (sides_moving_eps * sides_0);
            sum_full_bucket += node.sides[node.sides_nr - 1];
        }
        else {
            float sides_last = node.sides[node.sides_nr - 1];
            sum_bucket = sides_last;
            for (int i = node.sides_nr - 1; i > 0; --i) {
                sum_bucket += node.sides[i - 1];
                node.sides[i] += (sides_moving_eps * node.sides[i - 1]);
                sum_full_bucket += node.sides[i];
            }
            node.sides[0] += (sides_moving_eps * sides_last);
            sum_full_bucket += node.sides[0];
        }
        // normalize bucket sides
        for (int i = 0; i < node.sides_nr; ++i) {
            node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i] * sum_bucket / sum_full_bucket));
        }
    }

    // if (norm_sum > 0.0f) {
    //     // normalize and standardize sides
    //     float mean_edge = 1.0f / node.sides_nr;
    //     float sides_std = 0.0f;
    //     for (int i = 0; i < node.sides_nr; ++i) {
    //         node.sides[i] = node.sides[i] / norm_sum;
    //         sides_std += (node.sides[i] - mean_edge) * (node.sides[i] - mean_edge);
    //     }
    //     sides_std = sqrt(sides_std / node.sides_nr);
    //     float target_std = 0.1f;
    //     float std_factor = fmaxf(0.0f, fminf(1.0f, 1.0f - sides_std / target_std));
    //     for (int i = 0; i < node.sides_nr; ++i) {
    //         // node.sides[i] = fminf(1.0f, fmaxf(0.0f, mean_edge + 1.02f*((node.sides[i] / norm_sum) - mean_edge)));
    //         node.sides[i] = fminf(1.0f, fmaxf(0.0f, mean_edge + std_factor*((node.sides[i] / norm_sum) - mean_edge)));
    //     }
    // }
    // else {
    //     // set to sheet 0
    //     for (int i = 0; i < node.sides_nr; ++i) {
    //         if (i == 0) {
    //             node.sides[i] = 1.0f;
    //         }
    //         else {
    //             node.sides[i] = 0.0f;
    //         }
    //     }
    // }

    // winding numbers update
    if (max_winding_number_side > 0.0f) {
        node.winding_nr = max_winding_number;
        node.wnr_side = max_winding_number_side;
    }
    else {
        node.winding_nr = max_winding_number;
        node.wnr_side = 0.0f;
    }
    // sides update
    if (idx == seed_node && !adjusting_side) {
        for (int i = 0; i < node.sides_nr; ++i) {
            if (i == 0) {
                node.sides[i] = 1.0f;
            }
            else {
                node.sides[i] = 0.0f;
            }
        }
        // set wnr
        node.winding_nr = node.winding_nr_old;
        node.wnr_side = 1.0f;
        node.wnr_side_old = 1.0f;
    }
    else if (false && seed_node < 0) {
        for (int i = 0; i < node.sides_nr; ++i) {
            node.sides[i] *= 1.01f;
            node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i]));
        }
        node.wnr_side *= 1.1f;
        node.wnr_side = fmaxf(0.0f, fminf(1.0f, node.wnr_side));
    }
    else if (node.fixed && !adjusting_side) {
        // find max index
        int max_index = 0;
        for (int i = 0; i < node.sides_nr; ++i) {
            if (fabsf(node.sides[i]) > fabsf(node.sides[max_index])) {
                max_index = i;
            }
        }
        // fix the side value
        for (int i = 0; i < node.sides_nr; ++i) {
            if (i == max_index) {
                node.sides[i] = 1.0f;
            }
            else {
                node.sides[i] = 0.0f;
            }
        }
        // set wnr
        node.winding_nr = node.winding_nr_old; // reset to old value
        node.wnr_side = 1.0f;
        node.wnr_side_old = 1.0f;
    }
    // node.wnr_side = 0.999 * node.wnr_side;
    node.wnr_side = fmaxf(0.0f, fminf(1.0f, node.wnr_side));

    // Calculate the node happiness
    // // node.happiness = (1.0f - happiness_keepfactor) * happiness / happiness_sum + happiness_keepfactor * node.happiness_old;
    // node.happiness = (happiness / happiness_sum + happiness_keepfactor * node.happiness_old) / (1.0f + happiness_keepfactor);
    if (happiness_sum <= 0.00001f) {
        happiness_sum = 0.00001f;
    }
    if (edge_happy_accuracy_sum <= 0.00001f) {
        edge_happy_accuracy_sum = 0.00001f;
    }
    if (edge_happy_accuracy_sum <= 7 * 0.1f) {
        edge_happy_accuracy_sum = 7 * 0.1f;
    }
    float diminish_factor = 0.999f;
    float happyness_old_part = (happiness)/ (happiness_sum + 0.01f);
    happyness_old_part = diminish_factor * fmaxf(0.0f, fminf(1.0f, happyness_old_part));
    float happyness_edge_part = edge_happy_accuracy / (edge_happy_accuracy_sum + 0.01f);
    happyness_edge_part = diminish_factor * fmaxf(0.0f, fminf(1.0f, happyness_edge_part));
    float happyness_new_part = diminish_factor * fmaxf(0.0001f, fminf(0.9999f, node.wnr_side_old));
    float node_happiness_nr_edge_factor = 1.0f / fmaxf(8.0f - node.num_edges, 1.0f); // node needs to have at least 7 edges to be happy
    float node_happiness_part = happyness_new_part;
    node.happiness = diminish_factor * (happiness_keepfactor * happyness_old_part + edge_contribution * node_happiness_part) / (edge_contribution + happiness_keepfactor);
    // scale node happiness by f_star_factor
    // node.happiness *= f_star_factor;
    // node.happiness = 1.0f;
    // node.happiness *= 1.5f;
    // clip to 0 - 3 * 360
    node.happiness = fmaxf(0.000f, fminf(0.9999f, node.happiness));
    // node.happiness = fmaxf(0.0001f, fminf(0.9999f, node.wnr_side_old)); // debug to see what this value actually is over the complete scroll
}

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel_f_star(Node* d_graph, size_t* d_valid_indices, float o, float spring_constant, int num_valid_nodes, int estimated_windings, int i_round) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    // Calculate the f_star by mean
    float sum_w_f_tilde_k = 0.0f;
    float sum_w = 0.0f;
    float node_f_tilde = node.f_tilde;

    Edge* edges = node.edges;
    int num_active_edges = 0;
    // loop over all edges and update the nodes
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        if (d_graph[target_node].deleted) continue;
        float n2n1 = d_graph[target_node].f_tilde - node_f_tilde;
        float k_abs = fabsf(spring_constant * edge.k) - fabsf(n2n1);
        float k_dif = edge.k - n2n1 / spring_constant;

        float certainty = edge.certainty;
        certainty = 0.01f;
        float k = edge.k;

        if (certainty < 0.001f) {
            continue;
        }

        // Node winding angle update calculation
        if (edge.same_block) {
            // update closeness
            node.same_block_closeness += fabsf(k_dif);
            certainty *= fmaxf(1.0f, 0.25f * node.same_block_closeness_old / 360.0f);
            certainty *= fmaxf(1.0f, 0.25f * k_dif / 360.0f);
        }
        else {
            float dk = n2n1 - spring_constant * k;
            float fitting_factor = fmaxf(1.0f, 1.0f / (1.0f + 0.01f * fabsf(dk)));
            float same_block_factor2 = fmaxf(1.0f, 1.0f * node.num_same_block_edges);
            float edges_factor = sqrt(1.0f * (node.num_edges - node.num_same_block_edges) * (d_graph[target_node].num_edges - d_graph[target_node].num_same_block_edges));
            float certainty_factor = 0.05f * fitting_factor * same_block_factor2 * edges_factor;
            // float certainty_factor = 0.05f * same_block_factor2 * edges_factor;
            certainty *= certainty_factor;
            k *= 0.02f; // wrong other block edges make the adjacent windings be closer together, if k is "the perfect" angle step, then we would have too steep winding lines, since they wrap around that would then lead to places where the lines need to bend abruptly to compensate for the too steepness compared to the distance between the windings
        }
        // calculate f star update
        float predicted_winding_angle = d_graph[target_node].f_tilde - spring_constant * k;
        sum_w_f_tilde_k += certainty * predicted_winding_angle;
        sum_w += certainty;

        // Calculate node happiness: mean difference between k and target f_tilde - node f_tilde target + target node happiness weighted multiplied by the certainty
        num_active_edges++;
    }

    if (sum_w > 0)
    {
        node.f_star = (sum_w_f_tilde_k + o * node_f_tilde) / (sum_w + o);
    }
    // Clip f_star to the range [ - 2 * 360 * estimated_windings, 2 * 360 * estimated_windings]
    float winding_max =  4 * 360 * estimated_windings;
    if (fabsf(node.f_star) >= winding_max) {
        node.deleted = true;
    }
    node.f_star = fmaxf(- winding_max, fminf(winding_max, node.f_star));
}

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel_sides(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float other_block_factor, unsigned long seed, float sides_moving_eps, bool increase_same_block_weight, bool convergence_speedup, float min_f_star, float max_f_star, int iteration, bool wiggle, bool standard_winding_direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];

    if (node.deleted) return;
    
    // Initialize random number generator
    curandState state;
    curand_init(seed * idx, idx, 0, &state);

    // Only adjust the node if the f_star winding angle value is outside the fixed angles
    float side_sum_count_array = 0.0f;
    float side_missalignment_loss = 0.0f;

    Edge* edges = node.edges;
    // loop over all edges and update the nodes
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        if (d_graph[target_node].deleted) continue;

        // Calculate side array update
        {
            // calculate winding nr (side) difference between node and target node
            float recalculated_node_wnr_d = (node.f_init + edge.k - d_graph[target_node].f_init) / 360.0f;
            int node_wnr_dif = __float2int_rn(recalculated_node_wnr_d);

            float certainty = fmaxf(0.001f, edge.certainty);
            if (!edge.same_block) {
                certainty *= other_block_factor;
            }
            else {
                float range = 3.0f * 360.0f;
                float offset = (iteration / 1500) % 2 == 0 ? -range : range;
                if (!wiggle) {
                    offset = 0.0f;
                }
                float adjusted_scroll_f_star_position = node.f_star - ((max_f_star + min_f_star) / 2.0f) + offset;
                if (standard_winding_direction) {
                    adjusted_scroll_f_star_position *= -1.0f;
                }
                adjusted_scroll_f_star_position -= 7.0f * 360.0f;
                if (convergence_speedup && adjusted_scroll_f_star_position < 0.0f) {
                    // increase the same block edge weight a little to make the winding lines more connected
                    certainty *= 1.2f;
                }
                if (convergence_speedup && ((!standard_winding_direction && (edge.k * adjusted_scroll_f_star_position > 0.0f)) || (standard_winding_direction && (edge.k * adjusted_scroll_f_star_position < 0.0f)))) { // center of scroll is in positive direction, adjust everything to the center
                // if (convergence_speedup && edge.k > 0.0f) { // center of scroll is in positive direction, adjust everything to the center
                    float f_star_factor = 4.50f * fmaxf(0.0f, fminf(1.0f, fabsf(adjusted_scroll_f_star_position) / (max_f_star - min_f_star)));
                    f_star_factor = fminf(7.0f, fmaxf(0.0f, f_star_factor));
                    // float dif_factor = 0.125f;
                    float dif_factor = 0.125f;
                    if (adjusted_scroll_f_star_position > 0.0f) {
                        dif_factor = 0.15f;
                    }
                    else {
                        dif_factor = 0.070f;
                    }
                    float certainty_factor = fmaxf(0.1f, 1.0f - dif_factor * f_star_factor);
                    certainty *= certainty_factor; // adjust to only have one main scroll direction flow
                    // certainty *= 0.99f; // adjust to only have one main scroll direction flow
                }
                certainty *= 100.0f;
            }
            if (edge.certainty > 0.0f || edge.certainty > 0.001f) {
                side_missalignment_loss += update_sides(node, d_graph[target_node], certainty, node_wnr_dif, edge.same_block, convergence_speedup);
                side_sum_count_array += certainty;
            }
        }
    }
}

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel_winding_number_step1(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float other_block_factor, int seed_node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted && idx == seed_node) {
        printf("Error: seed node is deleted\n");
    }
    if (node.deleted) return;

    Edge* edges = node.edges;
    
    float total_winding_number_side = 0.0f;
    float total_winding_number_side_certainty = 0.0f;

    // recalculate edges wnr based on the target wnr
    for (int j = 0; j < node.num_edges; ++j) {
        Node& target_node = d_graph[edges[j].target_node];
        int target_wnr = target_node.winding_nr_old;

        float target_angle = 360 * target_wnr + target_node.f_init;
        float recalculated_node_wnr = (target_angle - edges[j].k - node.f_init) / 360.0f;
        int node_wnr = static_cast<int>(std::round(recalculated_node_wnr));
        edges[j].wnr_from_target = node_wnr;
        if (edges[j].wnr_from_target != node_wnr) {
            printf("Error: recalculated wnr is not an integer\n");
        }
        // Check if edge is closeish in f star
        if (!target_node.deleted && fabsf(node.f_tilde + edges[j].k - target_node.f_tilde) < (3 * 360.0f)) {
            float certainty = fmaxf(0.001f, edges[j].certainty);
            if (certainty < 0.001f) {
                continue;
            }
            if (!edges[j].same_block) {
                certainty *= other_block_factor;
            }
            else {
                certainty *= 7.5f;
            }
            // wrong edge wheight, less confidence in side
            if (edges[j].wnr_from_target != node.winding_nr_old) {
                certainty *= 10.0f;
            }
            else {
                total_winding_number_side += fmaxf(0.0f, certainty * target_node.wnr_side_old);
            }
            total_winding_number_side_certainty += certainty;
        }
    }
    if (total_winding_number_side_certainty < 0.0001f) {
        total_winding_number_side_certainty = 0.0001f;
    }
    total_winding_number_side = fmaxf(0.0f, fminf(1.0f, total_winding_number_side / total_winding_number_side_certainty));
    node.total_wnr_side = total_winding_number_side; // save the total winding number side for the node. usedto calculate percentual winding number side accuracy
}

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel_winding_number_step2(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float other_block_factor, int seed_node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted && idx == seed_node) {
        printf("Error: seed node is deleted\n");
    }
    if (node.deleted) return;

    Edge* edges = node.edges;

    float happiness = 0.0f;
    float happiness_sum = 0.0f;
    float happiness_keepfactor = 5.0f;
    float edge_contribution = 2.0f;

    int max_winding_number = 0;
    float max_winding_number_side = 0.0f;
    float edge_happy_accuracy = 0.0f;
    float edge_happy_accuracy_sum = 0.0f;

    // loop over all edges and update the nodes
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        if (d_graph[target_node].deleted) continue;
        if (edge.certainty < 0.001f) {
            continue;
        }

        // Calculate winding number update
        {   
            // Find the max winding number wrt sum of certainty
            float sum_unnormalized_side_j = 0.0f;
            float sum_normalization_j = 0.0f;
            int winding_number_j = edges[j].wnr_from_target; // set the winding number we search for
            float certainty_sum = 0.0f;
            int num_same_block = 0;
            for (int l = 0; l < node.num_edges; ++l) {
                if (d_graph[edges[l].target_node].deleted) continue;
                float certainty = fmaxf(0.001f, edges[l].certainty);
                certainty_sum += certainty;
                if (!edges[l].same_block) {
                    if (fabsf(node.f_tilde + edges[l].k - d_graph[edges[l].target_node].f_tilde) < (1.0f)) { // edge should most likely be good
                        certainty *= 10.0f;
                    }
                    certainty *= other_block_factor;
                }
                else {
                    certainty *= 30.5f;
                    num_same_block++;
                }
                // edge is certain enough
                if (edges[l].certainty > 0.001f) {
                    float wnr_old = d_graph[edges[l].target_node].wnr_side_old;
                    // Check if edge is closeish in f star
                    if (fabsf(node.f_tilde + edges[l].k - d_graph[edges[l].target_node].f_tilde) < (2.5f * 360.0f)) {
                        // Only use good ones for the winding number side value
                        if (edges[l].wnr_from_target == winding_number_j) {
                            sum_unnormalized_side_j += certainty * wnr_old;
                        }
                    }
                    else {
                        sum_unnormalized_side_j -= certainty * wnr_old; // make certainty lower if the winding number is not correctly matched
                    }
                }
                sum_normalization_j += certainty; // sum up the certainty over all edges of non-deleted nodes
            }
            // Update the max winding number
            float nr_edges_factor = 1.0f / fmaxf(1.0f, 2.15f - node.num_edges);
            float sum_normalization_j_factor = fminf(1.0f, certainty_sum / 0.05f);
            float same_block_factor = 1.0f / fmaxf(1.0f, 1.025f - num_same_block);
            float sum_side_j = nr_edges_factor * same_block_factor * fminf(1.0f, fmaxf(0.0f, sum_unnormalized_side_j / sum_normalization_j));
            if (sum_normalization_j <= 0.001f) {
                sum_side_j = 0.0f;
            }
            if (sum_side_j > max_winding_number_side) {
                max_winding_number_side = sum_side_j;
                max_winding_number = winding_number_j;
            }
        }

        // happyness calculation
        {
            int wnr_from_target = edge.wnr_from_target;
            int wnr_node = node.winding_nr_old;

            bool seam = fabsf(node.f_init - d_graph[target_node].f_init) >= 180.0f; // seam winding
            float certainty = 1.0f;
            float side_node_confidence = fmaxf(0.0f, fminf(1.0f, node.wnr_side_old / node.total_wnr_side));
            float side_target_confidence = fmaxf(0.0f, fminf(1.0f, d_graph[target_node].wnr_side_old / d_graph[target_node].total_wnr_side));
            float confidence = fmaxf(0.001f, fminf(1.0f, side_node_confidence * side_target_confidence));
            confidence = fmaxf(0.0f, sqrt(confidence)); // normalize the confidence
            certainty *= confidence;
            // side confidence
            float side_confidence = fmaxf(0.001f, fminf(1.0f, fmaxf(0.0f, fminf(1.0f, node.wnr_side_old)) * fmaxf(0.0f, fminf(1.0f, d_graph[target_node].wnr_side_old))));
            side_confidence = fmaxf(0.0f, sqrt(side_confidence)); // normalize the confidence
            certainty *= side_confidence;
            if (node.wnr_side_old > 0.001f && edge.certainty > 0.0f && certainty > 0.0f) { // only calculate hapyness if the node actually is reasonable certain of its winding number
                if (fabsf(node.f_tilde + edge.k - d_graph[target_node].f_tilde) < (3 * 360.0f)) {
                    if (!edge.same_block) { // other block edge, same sheet
                        certainty *= other_block_factor;
                        if (!seam && wnr_from_target == wnr_node) { 
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else if (seam && edge.k > 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else if (seam && edge.k < 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += d_graph[target_node].happiness_old;
                            happiness_sum += 1.0f;
                        }
                        else {
                            // bad connection, weight higher
                            certainty *= 10.0f;
                            happiness += 10.0f * d_graph[target_node].happiness_old;
                            happiness_sum += 10.0f;
                        }
                    }
                    else {
                        certainty *= 10.5f;
                        float same_block_surrounding_nodes_factor = 1.0f;
                        float same_block_bad_surrounding_nodes_factor = 0.5f;
                        if (edge.k * (d_graph[target_node].f_tilde - node.f_tilde) > 0.0f) {
                            // bad connection
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else if (seam) {
                            if (edge.k > 0 && wnr_from_target == wnr_node) { // It is a double switch
                                edge_happy_accuracy += certainty;
                                happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_surrounding_nodes_factor;
                            }
                            else if (edge.k < 0 && wnr_from_target == wnr_node) { // It is a double switch
                                edge_happy_accuracy += certainty;
                                happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_surrounding_nodes_factor;
                            }
                            else {
                                // bad connection, weight higher
                                certainty *= 2.0f;
                                happiness +=same_block_bad_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                                happiness_sum += same_block_bad_surrounding_nodes_factor;
                            }
                        }
                        else if (edge.k > 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else if (edge.k < 0 && wnr_from_target == wnr_node) { // same side with seam crossing
                            edge_happy_accuracy += certainty;
                            happiness += same_block_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_surrounding_nodes_factor;
                        }
                        else {
                            // bad connection, weight higher
                            certainty *= 10.0f;
                            happiness += same_block_bad_surrounding_nodes_factor * d_graph[target_node].happiness_old;
                            happiness_sum += same_block_bad_surrounding_nodes_factor;
                        }
                    }
                }
            }
            edge_happy_accuracy_sum += certainty;
        }
    }

    // winding numbers update
    if (max_winding_number_side > 0.0f) {
        node.winding_nr = max_winding_number;
        node.wnr_side = max_winding_number_side;
    }
    else {
        node.winding_nr = max_winding_number;
        node.wnr_side = 0.0f;
    }
    // sides update
    if (idx == seed_node) {
        // set wnr
        node.winding_nr = node.winding_nr_old;
        node.wnr_side = 1.0f;
        node.wnr_side_old = 1.0f;
    }
    else if (node.fixed) {
        // set wnr
        node.winding_nr = node.winding_nr_old; // reset to old value
        node.wnr_side = 1.0f;
        node.wnr_side_old = 1.0f;
    }
    node.wnr_side = fmaxf(0.0f, fminf(1.0f, node.wnr_side));

    // Calculate the node happiness
    if (happiness_sum <= 0.00001f) {
        happiness_sum = 0.00001f;
    }
    float diminish_factor = 0.999f;
    float happyness_old_part = (happiness)/ (happiness_sum + 0.01f);
    happyness_old_part = diminish_factor * fmaxf(0.0f, fminf(1.0f, happyness_old_part));
    float happyness_part = diminish_factor * fmaxf(0.0001f, fminf(0.9999f, node.wnr_side_old));
    node.happiness = diminish_factor * (happiness_keepfactor * happyness_old_part + edge_contribution * happyness_part) / (edge_contribution + happiness_keepfactor);
    node.happiness = fmaxf(0.000f, fminf(0.9999f, node.happiness));
}

// Kernel to update f_tilde with f_star on the GPU
__global__ void update_f_tilde_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    // Update f_tilde with the computed f_star
    node.f_tilde = node.f_star;
    // Update the happiness_old with the computed happiness
    node.happiness_old = node.happiness;
    // update sides
    for (int i = 0; i < node.sides_nr; ++i) {
        node.sides_old[i] = node.sides[i];
        node.sides[i] = 0.0f;
    }
    // update winding number
    node.winding_nr_old = node.winding_nr;
    node.wnr_side_old = node.wnr_side;
    node.confidence_old = node.confidence;
    node.smeared_confidence_old = node.smeared_confidence;
    node.closeness_old = node.closeness;
}

// Kernel to update f_star solver fields on the GPU
__global__ void update_f_star_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float median_f_star) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    if (median_f_star != 0.0f) {
        node.f_star -= median_f_star;
    }

    // Update f_tilde with the computed f_star
    node.f_tilde = node.f_star;

    // Update the closeness between nodes
    node.same_block_closeness /= fmaxf(1.0, 1.0f * node.num_same_block_edges);
    node.same_block_closeness_old = node.same_block_closeness;
    node.same_block_closeness = 0.0f; // reset
}

// Kernel to update sides solver fields on the GPU
__global__ void update_sides_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float* d_max_convergence, float* d_total_convergence) {
    __shared__ float shared_max[256];    // Shared memory for block maximum
    __shared__ float shared_sum[256];   // Shared memory for block sum

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_max[tid] = 0.0f;
    shared_sum[tid] = 0.0f;

    if (idx < num_valid_nodes) {
        size_t i = d_valid_indices[idx];
        Node& node = d_graph[i];
        if (!node.deleted) {
            float sum_abs_diff = 0.0f;

            // Update sides and calculate abs(sides - sides_old)
            for (int i = 0; i < node.sides_nr; ++i) {
                float diff = fabsf(node.sides[i] - node.sides_old[i]); // Absolute difference
                sum_abs_diff += diff;

                // Update sides_old and reset sides
                if (node.fixed && node.gt) { // If ground truth is available
                int winding_number = static_cast<int>(std::floor(node.gt_f_star / 360.0)) % node.sides_nr;
                assert (winding_number >= 0.0f);
                    if (winding_number == i) {
                        node.sides_old[i] = 1.0f;
                    }
                    else {
                        node.sides_old[i] = 0.0f;
                    }
                }
                else {
                    node.sides_old[i] = (4.0f * node.sides[i] + node.sides_old[i]) / 5.0f;
                }
                node.sides[i] = 0.0f;
            }

            // Store per-thread local values in shared memory
            shared_max[tid] = sum_abs_diff;  // For max reduction
            shared_sum[tid] = sum_abs_diff; // For sum reduction
        }
    }
    __syncthreads();

    // Perform block-level reduction for maximum and total sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Write block results to global memory
    if (tid == 0) {
        // Update the global maximum convergence using atomic operation
        atomicMax((int*)d_max_convergence, __float_as_int(shared_max[0]));

        // Update the global total convergence using atomic add
        atomicAdd(d_total_convergence, shared_sum[0]);
    }
}


// Kernel to update winding number solver fields on the GPU
__global__ void update_winding_number_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    // update winding number
    node.winding_nr_old = node.winding_nr;
    node.wnr_side_old = node.wnr_side;
    // Update the happiness_old with the computed happiness
    node.happiness_old = node.happiness;
}

// Kernel to update certainty with happyness on the GPU
__global__ void update_certainty_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float node_deactivation_threshold, float threshold_happyness, float certainty_good, float certainty_bad, float spring_constant) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    Edge* edges = node.edges;
    for (int j = 0; j < node.num_edges; ++j) {
        Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        if (d_graph[target_node].deleted) continue; // Edge not active
        float edge_fit = fabsf(node.f_star + spring_constant * edge.k - d_graph[target_node].f_star);
        float n2n1 = d_graph[target_node].f_star - node.f_star;
        if (node.happiness > node_deactivation_threshold || d_graph[target_node].happiness > node_deactivation_threshold) { // node of edge is "bad"
            edge.certainty = fminf(edge.certainty, certainty_bad);
        }
        else if (d_graph[target_node].happiness > threshold_happyness) { // Edge not yet fixed
            // edge.certainty = fminf(edge.certainty, 0.1f);
            continue;
        }
        else if (edge_fit < 10.0f) {
            edge.certainty = certainty_good;
        }
        else if (edge_fit > 400.0f) {
            edge.certainty = fminf(edge.certainty, certainty_bad);
        }
        else if (edge.same_block && (edge.k * n2n1 < 0 && false)) { // same block edge pointing in same direction as layers lay. if pointing too far, we dont care since the f star update take care of it
            edge.certainty = certainty_good;
        }
        else {
            // edge.certainty = 0.2f;
            // interpolate between good and bad
            edge.certainty = certainty_good + ((edge_fit - 10.0f) / (400.0f - 10.0f)) * (certainty_bad - certainty_good);
        }
        if (!edge.same_block) // edge on the same sheet. make it stronger
        {
            edge.certainty *= 10.0f; // divide would not work, change if instead
        }
    }
}

__global__ void normalize_sides(Node* d_graph, size_t *d_valid_indices, size_t num_valid_nodes, size_t num_nodes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    idx = d_valid_indices[idx];
    if (idx < num_nodes) {
        Node& node = d_graph[idx];
        if (node.deleted) return;
        
        // Compute sum of sides
        float sum = 0.0f;
        for (int i = 0; i < Node::sides_nr; i++) {
            sum += node.sides[i];
        }

        if (sum <= 0.0f) {
            // Set to winding nr 0
            for (int i = 0; i < Node::sides_nr; i++) {
                if (i == 0) {
                    node.sides[i] = 1.0f / Node::sides_nr + 0.00001f;
                }
                else {
                    node.sides[i] = 1.0f / Node::sides_nr - 0.00001f;
                }
            }
            sum = 0.0f;
            // Compute sum of sides
            for (int i = 0; i < Node::sides_nr; i++) {
                sum += node.sides[i];
            }
        }
        // Normalize the sides
        for (int i = 0; i < Node::sides_nr; i++) {
            node.sides[i] /= sum;
            node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i]));
        }
    }
}

__global__ void bucket_trick(Node* d_graph, size_t *d_valid_indices, size_t num_valid_nodes, size_t num_nodes, float sides_moving_eps, unsigned long seed, float std_cutoff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    idx = d_valid_indices[idx];
    if (idx < num_nodes) {
        Node& node = d_graph[idx];
        if (node.deleted) return;

        // Initialize random number generator
        curandState state;
        curand_init(seed * idx, idx, 0, &state);

        float noise_level = 0.00000001f;
        for (int i = 0; i < node.sides_nr; ++i) {
            // add a little random noise
            float noise = curand_uniform(&state) * 2.0f * noise_level - noise_level; // Noise in range [-noise_level, noise_level]
            node.sides[i] += noise;
            node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i]));
        }

        // BUCKET TRICK: move believes from bucket to bucket to counteract two believes per wrap (bleeding)
        if (sides_moving_eps > 0.0f) {
            float sum_bucket = 0.0f;
            float sum_full_bucket = 0.0f;
            int random_direction = curand(&state) % 2;
            if (random_direction % 2 == 0) {
                float sides_0 = node.sides[0];
                sum_bucket = sides_0;
                for (int i = 0; i < node.sides_nr - 1; ++i) {
                    sum_bucket += node.sides[i + 1];
                    node.sides[i] += (sides_moving_eps * node.sides[i + 1]);
                    sum_full_bucket += node.sides[i];
                }
                node.sides[node.sides_nr - 1] += (sides_moving_eps * sides_0);
                sum_full_bucket += node.sides[node.sides_nr - 1];
            }
            else {
                float sides_last = node.sides[node.sides_nr - 1];
                sum_bucket = sides_last;
                for (int i = node.sides_nr - 1; i > 0; --i) {
                    sum_bucket += node.sides[i - 1];
                    node.sides[i] += (sides_moving_eps * node.sides[i - 1]);
                    sum_full_bucket += node.sides[i];
                }
                node.sides[0] += (sides_moving_eps * sides_last);
                sum_full_bucket += node.sides[0];
            }
            // normalize bucket sides
            for (int i = 0; i < node.sides_nr; ++i) {
                node.sides[i] = fmaxf(0.0f, fminf(1.0f, node.sides[i] * sum_bucket / sum_full_bucket));
            }
        }
        
        // smooth the sides for a "nice" gaussian-like distribution
        float sides_temp[Node::sides_nr];
        for (int i = 0; i < node.sides_nr; ++i) {
            float sum = 0.0f;
            float weight = 0.0f;
            for (int j = -2; j <= 2; ++j) {
                int idx = (i + j + 4*node.sides_nr) % node.sides_nr;
                float w = 1.0f / (1.0f + j*j);
                // w = w * w; // square the weight
                sum += w * node.sides[idx];
                weight += w;
            }
            sides_temp[i] = sum / weight;
        }
        for (int i = 0; i < node.sides_nr; ++i) {
            node.sides[i] = sides_temp[i];
        }
    }
}

__global__ void compute_sum_squares(Node* d_graph, size_t *d_valid_indices, size_t num_valid_nodes, size_t num_nodes, float* sum_squares, int* sum_nodes, float mean) {
    __shared__ float local_sum_squares[256];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = d_valid_indices[idx];
    int tid = threadIdx.x;

    local_sum_squares[tid] = -1.0f;

    if (idx < num_nodes) {
        Node& node = d_graph[idx];
        if (!node.deleted && !node.fixed) {
            local_sum_squares[tid] = 0.0f;
            for (int i = 0; i < Node::sides_nr; i++) {
                float deviation = node.sides[i] - mean;
                local_sum_squares[tid] += deviation * deviation;
            }
        }

    }

    __syncthreads();

    // Reduce within block
    if (tid == 0) {
        float block_sum_squares = 0.0f;
        int sum_nodes_ = 0;

        for (int i = 0; i < blockDim.x; i++) {
            if (local_sum_squares[i] >= 0.0f) {
                block_sum_squares += local_sum_squares[i];
                sum_nodes_++;
            }
        }
        atomicAdd(sum_squares, block_sum_squares);
        atomicAdd(sum_nodes, sum_nodes_);
    }
}

__global__ void standardize_sides(Node* d_graph, size_t *d_valid_indices, size_t num_valid_nodes, size_t num_nodes, float mean, float std, float std_target, float std_cutoff, float min_f_star, float max_f_star, unsigned long seed, int iteration, bool wiggle, 
                                    bool standard_winding_direction, float scale_left, float scale_right, bool enable_delete_nodes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    idx = d_valid_indices[idx];
    float max_z_score = 0.0f;
    if (idx < num_nodes) {
        Node& node = d_graph[idx];
        if (node.deleted || node.fixed) return;

        // Initialize random number generator
        curandState state;
        curand_init(seed * idx, idx, 0, &state);

        int max_side = 0;
        // int secondmax_side = 0;
        // int thirdmax_side = 0;
        float max_side_value = 0.0f;
        float std_scale = std_target / std;
        float mean_std = 0.0f;
        for (int i = 0; i < Node::sides_nr; i++) {
            if (node.sides[i] > max_side_value) {
                max_side_value = node.sides[i];
                // thirdmax_side = secondmax_side;
                // secondmax_side = max_side;
                max_side = i;
            }
            node.sides[i] = fminf(1.0f, fmaxf(0.0f, mean + (node.sides[i] - mean) * std_scale));            
            if (std_cutoff > 0.0f) {
                if (((node.sides[i] - mean) / std_target) < -std_cutoff/2.0f) {
                    // set side to std cutoff / 2
                    node.sides[i] = fminf(1.0f, fmaxf(0.0f, mean + ((std_cutoff/2.0f) * std_target * ((node.sides[i] - mean) > 0 ? 1 : -1))));
                }
                else if (((node.sides[i] - mean) / std_target) > std_cutoff) {
                    // set side to std cutoff
                    node.sides[i] = fminf(1.0f, fmaxf(0.0f, mean + (std_cutoff * std_target * ((node.sides[i] - mean) > 0 ? 1 : -1))));
                }
            }
            mean_std += fabsf(node.sides[i] - mean);
            max_z_score = fmaxf(max_z_score, (node.sides[i] - mean) / std_target); // only interested in peak, not in what wrap it is not
        }
        mean_std /= Node::sides_nr;
        if (enable_delete_nodes && max_z_score > 8.0f && !node.fixed) { // delete nodes that are not well reachable and make the convergence slow
            node.deleted = true;
        }

        if (std_cutoff > 0.0f) {
            float range = 3.0f * 360.0f;
            float offset = (iteration / 1500) % 2 == 0 ? -range : range;
            if (!wiggle) {
                offset = 0.0f;
            }
            float adjusted_scroll_f_star_position = node.f_star - ((max_f_star + min_f_star) / 2.0f) + offset;
            float fixed_offset = 7.0f * 360.0f;
            if (standard_winding_direction) {
                adjusted_scroll_f_star_position *= -1.0f;
            }
            adjusted_scroll_f_star_position -=  fixed_offset;
            float divisor_left = 0.025f;
            float divisor_right = 0.025f;
            if (adjusted_scroll_f_star_position < 0.0f) {
                divisor_left = 0.0380f * scale_left;
            }
            else if (adjusted_scroll_f_star_position > 0.0f) {
                divisor_right = 0.0380f * scale_right;
            }
            int max_left = (max_side - 1 + Node::sides_nr) % Node::sides_nr;
            int max_left2 = (max_side - 2 + Node::sides_nr) % Node::sides_nr;
            int max_left3 = (max_side - 3 + Node::sides_nr) % Node::sides_nr;
            int max_right = (max_side + 1) % Node::sides_nr;
            int max_right2 = (max_side + 2) % Node::sides_nr;
            int max_right3 = (max_side + 3) % Node::sides_nr;
            node.sides[max_left] = fminf(1.0f, fmaxf(0.0f, node.sides[max_left] / ((1.0f + divisor_left))));
            node.sides[max_left2] = fminf(1.0f, fmaxf(0.0f, node.sides[max_left2] / (1.1f * (1.0f + divisor_left))));
            node.sides[max_left3] = fminf(1.0f, fmaxf(0.0f, node.sides[max_left3] / (1.2f * (1.0f + divisor_left))));
            node.sides[max_right] = fminf(1.0f, fmaxf(0.0f, node.sides[max_right] / ((1.0f + divisor_right)))); // from right lower value, to have a balance of left and right pushing forces
            node.sides[max_right2] = fminf(1.0f, fmaxf(0.0f, node.sides[max_right2] / (1.1f * (1.0f + divisor_right)))); // from right lower value, to have a balance of left and right pushing forces
            node.sides[max_right3] = fminf(1.0f, fmaxf(0.0f, node.sides[max_right3] / (1.2f * (1.0f + divisor_right)))); // from right lower value, to have a balance of left and right pushing forces
        }
    }
}

float standardize_graph(Node* d_graph, size_t *d_valid_indices, size_t num_valid_nodes, int num_nodes, float std_target = 0.013f, float std_cutoff = -1.0f, float sides_moving_eps = 0.0025f, unsigned long seed = 0, float min_f_star = 0.0f, float max_f_star = 0.0f, int iteration = 0, 
                        bool wiggle = true, bool standard_winding_direction = false, float scale_left = 1.0f, float scale_right = 1.0f, bool enable_delete_nodes = true) {
    // CUDA kernel configuration
    int threads_per_block = 256;
    int num_blocks = (num_valid_nodes + threads_per_block - 1) / threads_per_block;

    // Normalize sides for each node
    normalize_sides<<<num_blocks, threads_per_block>>>(d_graph, d_valid_indices, num_valid_nodes, num_nodes);

    cudaDeviceSynchronize(); // Check for errors during kernel execution

    // Apply the bucket trick
    bucket_trick<<<num_blocks, threads_per_block>>>(d_graph, d_valid_indices, num_valid_nodes, num_nodes, sides_moving_eps, seed, std_cutoff);

    cudaDeviceSynchronize(); // Check for errors during kernel execution

    // Precompute mean
    float mean = 1.0f / Node::sides_nr;

    // Allocate device memory for sum of squares
    float* d_sum_squares;
    cudaMalloc(&d_sum_squares, sizeof(float));
    cudaMemset(d_sum_squares, 0, sizeof(float));
    int* d_sum_nodes;
    cudaMalloc(&d_sum_nodes, sizeof(int));
    cudaMemset(d_sum_nodes, 0, sizeof(int));

    // Compute sum of squares
    compute_sum_squares<<<num_blocks, threads_per_block>>>(d_graph, d_valid_indices, num_valid_nodes, num_nodes, d_sum_squares, d_sum_nodes, mean);

    // Copy sum of squares back to host
    float h_sum_squares;
    cudaMemcpy(&h_sum_squares, d_sum_squares, sizeof(float), cudaMemcpyDeviceToHost);
    int h_sum_nodes;
    cudaMemcpy(&h_sum_nodes, d_sum_nodes, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize(); // Check for errors during kernel execution

    // Compute standard deviation
    float total_values = h_sum_nodes * Node::sides_nr;
    float std = sqrt(h_sum_squares / total_values);
    // std::cout << "Standard deviation scaling: " << std_target / std << std::endl;
    // Standardize sides
    standardize_sides<<<num_blocks, threads_per_block>>>(d_graph, d_valid_indices, num_valid_nodes, num_nodes, mean, std, std_target, std_cutoff, min_f_star, max_f_star, seed, iteration, wiggle, standard_winding_direction, scale_left, scale_right, enable_delete_nodes);

    cudaDeviceSynchronize(); // Check for errors during kernel execution

    // Normalize sides for each node
    normalize_sides<<<num_blocks, threads_per_block>>>(d_graph, d_valid_indices, num_valid_nodes, num_nodes);

    // Free device memory
    cudaFree(d_sum_squares);
    cudaDeviceSynchronize(); // Check for errors during kernel execution

    // return the std scaling
    return std_target / std;
}

// Copy edges from CPU to GPU with batched allocation
void copy_edges_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, bool copy_edges = true) {
    size_t total_edges = 0;

    // Step 1: Calculate the total number of edges
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].edges == nullptr) {
            std::cerr << "Error: Node " << i << " has a null edges pointer!" << std::endl;
            exit(EXIT_FAILURE);
        }
        total_edges += h_graph[i].num_edges;
    }

    size_t offset = 0;
    if (copy_edges) {
        // Step 2: Allocate memory for all edges at once on the GPU
        Edge* d_all_edges;
        cudaError_t err = cudaMalloc(&d_all_edges, total_edges * sizeof(Edge));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for edges: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        if (*d_all_edges_ptr != nullptr) {
            // Free existing edges on the GPU
            cudaFree(*d_all_edges_ptr);
            *d_all_edges_ptr = nullptr;
        }
        *d_all_edges_ptr = d_all_edges;  // Store the pointer for later use when freeing

        // Step 3: Copy all edges to a temporary host array
        Edge* h_all_edges = new Edge[total_edges];
        for (size_t i = 0; i < num_nodes; ++i) {
            if (h_graph[i].num_edges > 0) {
                memcpy(&h_all_edges[offset], h_graph[i].edges, h_graph[i].num_edges * sizeof(Edge));
                offset += h_graph[i].num_edges;
            }
        }

        // Copy the host edges array to GPU
        cudaMemcpy(*d_all_edges_ptr, h_all_edges, total_edges * sizeof(Edge), cudaMemcpyHostToDevice);
        delete[] h_all_edges;
    }

    // Step 4: Update the d_graph[i].edges pointers
    offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].num_edges > 0) {
            // Use a device pointer (d_all_edges + offset)
            Edge* d_edges_offset = *d_all_edges_ptr + offset;
            cudaError_t err = cudaMemcpyAsync(&d_graph[i].edges, &d_edges_offset, sizeof(Edge*), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy failed for node " << i << " edges pointer: " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
            offset += h_graph[i].num_edges;
        }
    }

    // Synchronize device to ensure memory transfer completion
    cudaDeviceSynchronize();
}

// Copy edges from GPU to CPU with batched allocation
Edge* copy_edges_from_gpu(Node* h_graph, size_t num_nodes, Edge** d_all_edges_ptr) {
    size_t total_edges = 0;

    // Step 1: Calculate the total number of edges
    for (size_t i = 0; i < num_nodes; ++i) {
        total_edges += h_graph[i].num_edges;
    }

    // Step 2: Allocate memory for all edges at once on the host
    Edge* h_all_edges = new Edge[total_edges];  // Temporary array to hold all edges

    // Step 3: Copy all edges in one go
    cudaMemcpy(h_all_edges, *d_all_edges_ptr, total_edges * sizeof(Edge), cudaMemcpyDeviceToHost);

    // Step 4: Update the h_graph[i].edges pointers
    size_t offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].num_edges > 0) {
            h_graph[i].edges = &h_all_edges[offset];
            offset += h_graph[i].num_edges;
        }
    }
    return h_all_edges;
}

// Function to free the memory
void free_edges_from_gpu(Edge* d_all_edges) {
    // Free existing edges on the GPU
    if (d_all_edges != nullptr) {
        cudaFree(d_all_edges);
        d_all_edges = nullptr;
    }
}

void copy_sides_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, float** d_all_sides_ptr, bool copy_sides = true) {
    size_t total_sides = 0;

    // Step 1: Calculate the total number of sides
    for (size_t i = 0; i < num_nodes; ++i) {
        // sides
        total_sides += h_graph[i].sides_nr;
        // old sides
        total_sides += h_graph[i].sides_nr;
    }

    size_t offset = 0;
    if (copy_sides) {
        // Step 2: Allocate memory for all sides at once on the GPU
        float* d_all_sides;
        cudaError_t err = cudaMalloc(&d_all_sides, total_sides * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for sides: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        if (*d_all_sides_ptr != nullptr) {
            // Free existing sides on the GPU
            cudaFree(*d_all_sides_ptr);
            *d_all_sides_ptr = nullptr;
        }
        *d_all_sides_ptr = d_all_sides;  // Store the pointer for later use when freeing
        // Step 3: Copy all sides to a temporary host array
        float* h_all_sides = new float[total_sides];
        for (size_t i = 0; i < num_nodes; ++i) {
            if (h_graph[i].sides_nr > 0) {
                memcpy(&h_all_sides[offset], h_graph[i].sides, h_graph[i].sides_nr * sizeof(float));
                offset += h_graph[i].sides_nr;
                memcpy(&h_all_sides[offset], h_graph[i].sides_old, h_graph[i].sides_nr * sizeof(float));
                offset += h_graph[i].sides_nr;
            }
        }

        // Copy the host sides array to GPU
        cudaMemcpy(*d_all_sides_ptr, h_all_sides, total_sides * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_all_sides;
    }

    // Step 4: Update the d_graph[i].sides pointers
    offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].sides_nr > 0) {
            // Use a device pointer (d_all_sides + offset)
            float* d_sides_offset = *d_all_sides_ptr + offset;
            cudaError_t err = cudaMemcpyAsync(&d_graph[i].sides, &d_sides_offset, sizeof(float*), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy failed for node " << i << " sides pointer: " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
            offset += h_graph[i].sides_nr;
            float* d_sides_old_offset = *d_all_sides_ptr + offset;
            err = cudaMemcpyAsync(&d_graph[i].sides_old, &d_sides_old_offset, sizeof(float*), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy failed for node " << i << " sides_old pointer: " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
            offset += h_graph[i].sides_nr;
        }
    }
}

// Copy sides from GPU to CPU with batched allocation
float* copy_sides_from_gpu(Node* h_graph, size_t num_nodes, float** d_all_sides_ptr) {
    size_t total_sides = 0;

    // Step 1: Calculate the total number of sides
    for (size_t i = 0; i < num_nodes; ++i) {
        // sides
        total_sides += h_graph[i].sides_nr;
        // old sides
        total_sides += h_graph[i].sides_nr;
    }

    // Step 2: Allocate memory for all sides at once on the host
    float* h_all_sides = new float[total_sides];  // Temporary array to hold all sides

    // Step 3: Copy all sides in one go
    cudaMemcpy(h_all_sides, *d_all_sides_ptr, total_sides * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 4: Update the h_graph[i].sides pointers
    size_t offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].sides_nr > 0) {
            h_graph[i].sides = &h_all_sides[offset];
            offset += h_graph[i].sides_nr;
            h_graph[i].sides_old = &h_all_sides[offset];
            offset += h_graph[i].sides_nr;}
    }
    return h_all_sides;
}

// Function to free the memory
void free_sides_from_gpu(float* d_all_sides) {
    // Free existing sides on the GPU
    if (d_all_sides != nullptr) {
        cudaFree(d_all_sides);
        d_all_sides = nullptr;
    }
}

// copy complete graph from cpu to gpu
void copy_graph_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, float** d_all_sides_ptr, bool copy_edges = true, bool copy_sides = true) {
    // Copy the graph to the GPU
    cudaError_t err = cudaMemcpy(d_graph, h_graph, num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for graph to gpu: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // copy edges to gpu
    copy_edges_to_gpu(h_graph, d_graph, num_nodes, d_all_edges_ptr, copy_edges);
    // copy sides to gpu
    copy_sides_to_gpu(h_graph, d_graph, num_nodes, d_all_sides_ptr, copy_sides);
}

// copy complete graph from gpu to cpu
std::pair<Edge*, float*> copy_graph_from_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, float** d_all_sides_ptr, bool copy_edges = true, bool copy_sides = true) {
    // Copy the graph from the GPU
    cudaError_t err = cudaMemcpy(h_graph, d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for graph from gpu: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    // copy edges from gpu
    Edge* h_all_edges = nullptr;
    if (copy_edges) {
        h_all_edges = copy_edges_from_gpu(h_graph, num_nodes, d_all_edges_ptr);
    }
    float* h_all_sides = nullptr; 
    if (copy_sides) {
        // copy sides from gpu
        h_all_sides = copy_sides_from_gpu(h_graph, num_nodes, d_all_sides_ptr);
    }
    return std::make_pair(h_all_edges, h_all_sides);
}

__global__ void update_fixed_kernel(Node* d_graph, const bool* d_fixed, int num_nodes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_nodes) {
        d_graph[idx].fixed = d_fixed[idx];
    }
}

void update_fixed_field(Node* d_graph, const Node* h_graph, size_t num_nodes) {
    // Step 0: create a fixed array on the host
    bool* h_fixed = new bool[num_nodes];
    for (int i = 0; i < num_nodes; ++i) {
        h_fixed[i] = h_graph[i].fixed;
    }
    // Step 1: Allocate and copy the fixed array to the device
    bool* d_fixed;
    cudaMalloc(&d_fixed, num_nodes * sizeof(bool));
    cudaMemcpy(d_fixed, h_fixed, num_nodes * sizeof(bool), cudaMemcpyHostToDevice);

    // Deallocate the host memory
    delete[] h_fixed;

    // Step 2: Launch the kernel to update the `fixed` field
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    update_fixed_kernel<<<numBlocks, blockSize>>>(d_graph, d_fixed, num_nodes);
    cudaDeviceSynchronize();

    // Step 3: Free temporary memory
    cudaFree(d_fixed);
}

__device__ float atomicMinFloat(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void sum_mean_happiness_kernel(Node* graph, float* max, float* sum_out, int* count_out, int num_nodes) {
    extern __shared__ float sdata[];
    __shared__ int count_non_deleted;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = 0.0f;
    if (tid == 0) count_non_deleted = 0;

    if (idx < num_nodes && !graph[idx].deleted) {
        sdata[tid] = graph[idx].happiness;
        atomicAdd(&count_non_deleted, 1);  // Count non-deleted nodes
    }

    // find max
    if (idx < num_nodes && !graph[idx].deleted) {
        atomicMaxFloat(max, sdata[tid]);
    }

    __syncthreads();

    // Perform parallel reduction to compute sum of happiness
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write sum and count result from each block to global memory
    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
        atomicAdd(count_out, count_non_deleted);
    }
}

__global__ void min_f_star_kernel(Node* graph, float* min_f_star_out, int num_nodes) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with max float value
    sdata[tid] = FLT_MAX;

    if (idx < num_nodes && !graph[idx].deleted) {
        sdata[tid] = graph[idx].f_star;
    }

    __syncthreads();

    // Perform parallel reduction to find the minimum f_star
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result from block to global memory
    if (tid == 0) {
        atomicMinFloat(min_f_star_out, sdata[0]);  // Replace atomicMin with atomicMinFloat
    }
}

__global__ void max_f_star_kernel(Node* graph, float* max_f_star_out, int num_nodes) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with min float value
    sdata[tid] = -FLT_MAX;

    if (idx < num_nodes && !graph[idx].deleted) {
        sdata[tid] = graph[idx].f_star;
    }

    __syncthreads();

    // Perform parallel reduction to find the maximum f_star
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result from block to global memory
    if (tid == 0) {
        atomicMaxFloat(max_f_star_out, sdata[0]);  // Replace atomicMax with atomicMaxFloat
    }
}

float min_f_star(const std::vector<Node>& graph, bool use_gt) {
    float min_f = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            if (node.gt_f_star < min_f) {
                min_f = node.gt_f_star;
            }
        } else {
            if (node.f_star < min_f) {
                min_f = node.f_star;
            }
        }
    }

    return min_f;
}

float max_f_star(const std::vector<Node>& graph, bool use_gt) {
    float max_f = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            if (node.gt_f_star > max_f) {
                max_f = node.gt_f_star;
            }
        } else {
            if (node.f_star > max_f) {
                max_f = node.f_star;
            }
        }
    }

    return max_f;
}

float min_happyness(const std::vector<Node>& graph) {
    float min_h = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.happiness < min_h) {
            min_h = node.happiness;
        }
    }

    return min_h;
}

float max_happyness(const std::vector<Node>& graph) {
    float max_h = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.happiness > max_h) {
            max_h = node.happiness;
        }
    }

    return max_h;
}

std::pair<float, float> min_max_percentile_f_star(const std::vector<Node>& graph, float percentile, bool use_gt = false) {
    std::vector<float> f_star_values;
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            f_star_values.push_back(node.gt_f_star);
        } else {
            f_star_values.push_back(node.f_star);
        }
    }

    std::sort(f_star_values.begin(), f_star_values.end());

    size_t num_values = f_star_values.size();
    size_t min_index = static_cast<size_t>(std::floor(percentile * num_values));
    size_t max_index = static_cast<size_t>(std::floor((1.0f - percentile) * num_values));
    return std::make_pair(f_star_values[min_index], f_star_values[max_index]);
}

float min_z_node(const std::vector<Node>& graph) {
    float min_z = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.z < min_z) {
            min_z = node.z;
        }
    }

    return min_z;
}

float max_z_node(const std::vector<Node>& graph) {
    float max_z = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.z > max_z) {
            max_z = node.z;
        }
    }

    return max_z;
}

void create_ply_pointcloud(const std::vector<Node>& graph, const std::string& filename) {
    std::ofstream ply_file(filename);
    if (!ply_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::ofstream ply_file_debug("colormap.ply");
    if (!ply_file_debug.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PLY header
    int valid_nodes = 0;
    float min_f_star = std::numeric_limits<float>::max();
    for (const auto& node : graph) {
        if (!node.deleted) {
            valid_nodes++;
        }

        min_f_star = std::min(min_f_star, - node.gt_f_star);
    }

    // round to nearest 360
    min_f_star = roundf(min_f_star / 360.0f - 1.0f) * 360.0f;

    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << 3*valid_nodes << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "end_header\n";

    ply_file_debug << "ply\n";
    ply_file_debug << "format ascii 1.0\n";
    ply_file_debug << "element vertex " << 4*360 << "\n";
    ply_file_debug << "property float x\n";
    ply_file_debug << "property float y\n";
    ply_file_debug << "property float z\n";
    ply_file_debug << "property uchar red\n";
    ply_file_debug << "property uchar green\n";
    ply_file_debug << "property uchar blue\n";
    ply_file_debug << "end_header\n";

    for (int i = 0; i < 4*360; i=i+1) {
        // Determine color based on gt_f_star
        float color_gradient = (- i + 4.0f*360.0f + 180.0f) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        // std::cout << "Color gradient: " << color_gradient << " Color index: " << color_index << std::endl;

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        ply_file_debug << -1.0f*i << " " << 0.0f  << " " << 0.0f << " "
                       << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    // Write node data
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        // Determine color based on gt_f_star
        float color_gradient = (- node.gt_f_star - min_f_star + 180.0f) / 360.0f;
        if (node.f_init <= -90.0f && node.f_init >= -145.0f) { // gt assignment bug most probably. post processing fix
            color_gradient += 1.0f;
        }
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 255; g = 255; b = 255; break;   // White
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 255; g_next = 255; b_next = 255; break;   // White
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        if (!node.gt) {
            // Gray color for nodes without ground truth
            r = 128;
            g = 128;
            b = 128;
        }

        ply_file << node.f_star/20 << " " << node.f_init  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init - 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init + 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    ply_file.close();
    ply_file_debug.close();
    std::cout << "PLY point cloud saved to " << filename << std::endl;
}

void create_ply_pointcloud_side(const std::vector<Node>& graph, const std::string& filename) {
    std::ofstream ply_file(filename);
    if (!ply_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::ofstream ply_file_side_gt("initial_side_gt.ply");
    if (!ply_file_side_gt.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PLY header
    int valid_nodes = 0;
    for (const auto& node : graph) {
        if (!node.deleted) {
            valid_nodes++;
        }
    }

    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << 3*valid_nodes << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "end_header\n";

    ply_file_side_gt << "ply\n";
    ply_file_side_gt << "format ascii 1.0\n";
    ply_file_side_gt << "element vertex " << 3*valid_nodes << "\n";
    ply_file_side_gt << "property float x\n";
    ply_file_side_gt << "property float y\n";
    ply_file_side_gt << "property float z\n";
    ply_file_side_gt << "property uchar red\n";
    ply_file_side_gt << "property uchar green\n";
    ply_file_side_gt << "property uchar blue\n";
    ply_file_side_gt << "end_header\n";

    // Write node data
    float min_gt = std::numeric_limits<float>::max();
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        min_gt = std::min(min_gt, - node.gt_f_star);

        // Determine color based on gt_f_star
        float color_gradient = 2.0f + node.side;
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        ply_file << node.f_star/20 << " " << node.f_init  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init - 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init + 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    // Write node data
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        // Determine color based on gt_f_star
        float color_gradient = 2.0f + int((- node.gt_f_star - min_gt + 180.0f) / 360) % 2;
        if (node.f_init <= -90.0f && node.f_init >= -145.0f) { // gt assignment bug most probably. post processing fix
            color_gradient += 1.0f;
        }
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        if (!node.gt) {
            // Gray color for nodes without ground truth
            r = 128;
            g = 128;
            b = 128;
        }

        ply_file_side_gt << node.f_star/20 << " " << node.f_init  << " " << node.z << " "
                         << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file_side_gt << node.f_star/20 << " " << node.f_init - 360.0f  << " " << node.z << " "
                            << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file_side_gt << node.f_star/20 << " " << node.f_init + 360.0f  << " " << node.z << " "
                            << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    ply_file.close();
    ply_file_side_gt.close();
    std::cout << "PLY point cloud saved to " << filename << std::endl;
}

void create_ply_pointcloud_sides(const std::vector<Node>& graph, const std::string& filename) {
    std::ofstream ply_file(filename);
    if (!ply_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PLY header
    int valid_nodes = 0;
    for (const auto& node : graph) {
        if (!node.deleted) {
            valid_nodes++;
        }
    }

    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << 3*valid_nodes << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "end_header\n";

    // Write node data
    float min_gt = std::numeric_limits<float>::max();
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        min_gt = std::min(min_gt, - node.gt_f_star);

        int max_sides_index = 0;
        for (int i = 0; i < node.sides_nr; i++) {
            if (node.sides_old[i] > node.sides_old[max_sides_index]) {
                max_sides_index = i;
            }
        }

        // Determine color based on gt_f_star
        float color_gradient = 2.0f + max_sides_index;
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 255; g = 255; b = 255; break;   // White
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 255; g_next = 255; b_next = 255; break;   // White
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        ply_file << node.f_star/20 << " " << node.f_init  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init - 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
        ply_file << node.f_star/20 << " " << node.f_init + 360.0f  << " " << node.z << " "
                 << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    ply_file.close();
    std::cout << "PLY point cloud saved to " << filename << std::endl;
}

void plot_nodes(const std::vector<Node>& graph, const std::string& filename) {
    // Find min and max values for f_star and f_init
    // float min_f_star = std::numeric_limits<float>::max();
    // float max_f_star = std::numeric_limits<float>::lowest();
    float min_f_init = std::numeric_limits<float>::max();
    float max_f_init = std::numeric_limits<float>::lowest();
    float min_gt = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        // min_f_star = std::min(min_f_star, node.f_star);
        // max_f_star = std::max(max_f_star, node.f_star);
        min_f_init = std::min(min_f_init, node.f_init);
        max_f_init = std::max(max_f_init, node.f_init);
        min_gt = std::min(min_gt, - node.gt_f_star);
    }
    // round to nearest 360
    min_gt = roundf(min_gt / 360.0f - 1.0f) * 360.0f;

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.001f);
    float dif_03 = 0.3f * (max_f_star - min_f_star);
    min_f_star -= dif_03;
    max_f_star += dif_03;

    // Define image dimensions and margins
    int img_width = 2000;
    int img_height = 1000;
    int margin = 50;

    cv::Mat scatter_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axes
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(img_width - margin, img_height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2);

    // Scale factors to map values to pixel coordinates
    float x_scale = (img_width - 2 * margin) / (max_f_star - min_f_star);
    float y_scale = (img_height - 2 * margin) / (max_f_init - min_f_init);

    // Plot each node as a point
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int x = static_cast<int>((node.f_star - min_f_star) * x_scale) + margin;
        int y = img_height - margin - static_cast<int>((node.f_init - min_f_init) * y_scale);

        // Determine color based on gt_f_star
        float color_gradient = (- node.gt_f_star - min_gt + 90.0f) / 360.0f;
        // To floored color
        color_gradient = roundf(color_gradient + 0.5f);
        if (node.f_init <= -90.0f && node.f_init >= -145.0f) { // gt assignment bug most probably. post processing fix
            color_gradient += 1.0f;
        }
        // float color_gradient = (node.gt_f_star - min_gt + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;
        // color_gradient = 1.0f - color_gradient; // buggy gt winding angle fix

        if (color_gradient < 0.0f) {
            std::cout << "Color gradient: " << color_gradient << " Color index: " << color_index << std::endl;
        }
        if (color_gradient > 1.0f) {
            std::cout << "Color index: " << color_index << std::endl;
        }

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0, std::min(255, int(r * (1.0f - color_gradient) + r_next * color_gradient))));
        g = static_cast<unsigned char>(std::max(0, std::min(255, int(g * (1.0f - color_gradient) + g_next * color_gradient))));
        b = static_cast<unsigned char>(std::max(0, std::min(255, int(b * (1.0f - color_gradient) + b_next * color_gradient))));

        if (!node.gt) {
            // Black color for nodes without ground truth
            r = 0;
            g = 0;
            b = 0;
        }

        if (node.fixed) {
            // Browner
            r = (165 + r) / 2;
            g = (42 + g) / 2;
            b = (42 + b) / 2;
        }

        cv::circle(scatter_image, cv::Point(x, y), 1, cv::Scalar(b, g, r), cv::FILLED);
    }

    // Add axis labels
    cv::putText(scatter_image, "f_star", cv::Point(img_width / 2, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(scatter_image, "f_init", cv::Point(10, img_height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Add indicators to the x-axis every 360
    for (float tick = min_f_star; tick <= max_f_star; tick += 360.0f) {
        int x = static_cast<int>((tick - min_f_star) * x_scale) + margin;
        cv::line(scatter_image, cv::Point(x, img_height - margin), cv::Point(x, img_height - margin + 10), cv::Scalar(0, 0, 0), 1);
    }

    // Save the scatter plot image to a file if filename is not empty
    if (!filename.empty()) {
        cv::imwrite(filename, scatter_image);
    }

    // Display the scatter plot
    cv::imshow("Scatter Plot of Nodes", scatter_image);
    cv::waitKey(1);
}

void plot_nodes_winding_numbers(const std::vector<Node>& graph, const std::string& filename) {
    // Find min and max values for f_star and f_init
    // float min_f_star = std::numeric_limits<float>::max();
    // float max_f_star = std::numeric_limits<float>::lowest();
    float min_f_init = std::numeric_limits<float>::max();
    float max_f_init = std::numeric_limits<float>::lowest();
    int min_wnr = std::numeric_limits<int>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        // min_f_star = std::min(min_f_star, node.f_star);
        // max_f_star = std::max(max_f_star, node.f_star);
        min_f_init = std::min(min_f_init, node.f_init);
        max_f_init = std::max(max_f_init, node.f_init);
        min_wnr = std::min(min_wnr, - node.winding_nr);
    }
    // round to save 360
    min_wnr = (min_wnr - 1) * 360;

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.001f);
    float dif_03 = 0.3f * (max_f_star - min_f_star);
    min_f_star -= dif_03;
    max_f_star += dif_03;

    // Define image dimensions and margins
    int img_width = 2000;
    int img_height = 1000;
    int margin = 50;

    cv::Mat scatter_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axes
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(img_width - margin, img_height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2);

    // Scale factors to map values to pixel coordinates
    float x_scale = (img_width - 2 * margin) / (max_f_star - min_f_star);
    float y_scale = (img_height - 2 * margin) / (max_f_init - min_f_init);

    // Plot each node as a point
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int x = static_cast<int>((node.f_star - min_f_star) * x_scale) + margin;
        int y = img_height - margin - static_cast<int>((node.f_init - min_f_init) * y_scale);

        // Determine color based on gt_f_star
        float color_gradient = - node.winding_nr - min_wnr;

        // float color_gradient = (node.gt_f_star - min_gt + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;
        // color_gradient = 1.0f - color_gradient; // buggy gt winding angle fix

        if (color_gradient < 0.0f) {
            std::cout << "Color gradient: " << color_gradient << " Color index: " << color_index << std::endl;
        }
        if (color_gradient > 1.0f) {
            std::cout << "Color index: " << color_index << std::endl;
        }

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0, std::min(255, int(r * (1.0f - color_gradient) + r_next * color_gradient))));
        g = static_cast<unsigned char>(std::max(0, std::min(255, int(g * (1.0f - color_gradient) + g_next * color_gradient))));
        b = static_cast<unsigned char>(std::max(0, std::min(255, int(b * (1.0f - color_gradient) + b_next * color_gradient))));

        if (node.fixed) {
            // Browner
            r = (165 + r) / 2;
            g = (42 + g) / 2;
            b = (42 + b) / 2;
        }

        cv::circle(scatter_image, cv::Point(x, y), 1, cv::Scalar(b, g, r), cv::FILLED);
    }

    // Add axis labels
    cv::putText(scatter_image, "f_star", cv::Point(img_width / 2, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(scatter_image, "f_init", cv::Point(10, img_height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Add indicators to the x-axis every 360
    for (float tick = min_f_star; tick <= max_f_star; tick += 360.0f) {
        int x = static_cast<int>((tick - min_f_star) * x_scale) + margin;
        cv::line(scatter_image, cv::Point(x, img_height - margin), cv::Point(x, img_height - margin + 10), cv::Scalar(0, 0, 0), 1);
    }

    // Save the scatter plot image to a file if filename is not empty
    if (!filename.empty()) {
        cv::imwrite(filename, scatter_image);
    }

    // Display the scatter plot
    cv::imshow("Scatter Plot of Nodes Winding Numbers", scatter_image);
    cv::waitKey(1);
}

void plot_nodes_side(const std::vector<Node>& graph, const std::string& filename) {
    // Find min and max values for f_star and f_init
    // float min_f_star = std::numeric_limits<float>::max();
    // float max_f_star = std::numeric_limits<float>::lowest();
    float min_f_init = std::numeric_limits<float>::max();
    float max_f_init = std::numeric_limits<float>::lowest();
    float min_gt = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        // min_f_star = std::min(min_f_star, node.f_star);
        // max_f_star = std::max(max_f_star, node.f_star);
        min_f_init = std::min(min_f_init, node.f_init);
        max_f_init = std::max(max_f_init, node.f_init);
        min_gt = std::min(min_gt, - node.gt_f_star);
    }
    // round to nearest 360
    min_gt = roundf(min_gt / 360.0f - 1.0f) * 360.0f;

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.001f);
    float dif_03 = 0.3f * (max_f_star - min_f_star);
    min_f_star -= dif_03;
    max_f_star += dif_03;

    // Define image dimensions and margins
    int img_width = 2000;
    int img_height = 1000;
    int margin = 50;

    cv::Mat scatter_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axes
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(img_width - margin, img_height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2);

    // Scale factors to map values to pixel coordinates
    float x_scale = (img_width - 2 * margin) / (max_f_star - min_f_star);
    float y_scale = (img_height - 2 * margin) / (max_f_init - min_f_init);

    // Plot each node as a point
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int x = static_cast<int>((node.f_star - min_f_star) * x_scale) + margin;
        int y = img_height - margin - static_cast<int>((node.f_init - min_f_init) * y_scale);

        // Determine color based on gt_f_star
        float side_sign = node.side > 0.0f ? 1.0f : 0.0f;
        float color_gradient = 2.0f + side_sign;
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        if (node.fixed) {
            b = 255;
        }

        cv::circle(scatter_image, cv::Point(x, y), 1, cv::Scalar(b, g, r), cv::FILLED);
    }

    // Add axis labels
    cv::putText(scatter_image, "f_star", cv::Point(img_width / 2, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(scatter_image, "f_init", cv::Point(10, img_height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Add indicators to the x-axis every 360
    for (float tick = min_f_star; tick <= max_f_star; tick += 360.0f) {
        int x = static_cast<int>((tick - min_f_star) * x_scale) + margin;
        cv::line(scatter_image, cv::Point(x, img_height - margin), cv::Point(x, img_height - margin + 10), cv::Scalar(0, 0, 0), 1);
    }

    // Save the scatter plot image to a file if filename is not empty
    if (!filename.empty()) {
        cv::imwrite(filename, scatter_image);
    }

    // Display the scatter plot
    cv::imshow("Scatter Plot Side of Nodes", scatter_image);
    cv::waitKey(1);
}

void plot_nodes_sides_array(const std::vector<Node>& graph, const std::string& filename) {
    // Find min and max values for f_star and f_init
    // float min_f_star = std::numeric_limits<float>::max();
    // float max_f_star = std::numeric_limits<float>::lowest();
    float min_f_init = std::numeric_limits<float>::max();
    float max_f_init = std::numeric_limits<float>::lowest();
    float min_gt = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        // min_f_star = std::min(min_f_star, node.f_star);
        // max_f_star = std::max(max_f_star, node.f_star);
        min_f_init = std::min(min_f_init, node.f_init);
        max_f_init = std::max(max_f_init, node.f_init);
        min_gt = std::min(min_gt, - node.gt_f_star);
    }
    // round to nearest 360
    min_gt = roundf(min_gt / 360.0f - 1.0f) * 360.0f;

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.001f);
    float dif_03 = 0.3f * (max_f_star - min_f_star);
    min_f_star -= dif_03;
    max_f_star += dif_03;

    // Define image dimensions and margins
    int img_width = 2000;
    int img_height = 1000;
    int margin = 50;

    cv::Mat scatter_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axes
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(img_width - margin, img_height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2);

    // Scale factors to map values to pixel coordinates
    float x_scale = (img_width - 2 * margin) / (max_f_star - min_f_star);
    float y_scale = (img_height - 2 * margin) / (max_f_init - min_f_init);

    // Plot each node as a point
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int x = static_cast<int>((node.f_star - min_f_star) * x_scale) + margin;
        int y = img_height - margin - static_cast<int>((node.f_init - min_f_init) * y_scale);

        // Determine color based on gt_f_star
        int max_side_index = 0;
        for (int i = 1; i < node.sides_nr; i++) {
            if (node.sides_old[i] > node.sides_old[max_side_index]) {
                max_side_index = i;
            }
        }

        float color_gradient = node.sides_nr - max_side_index;
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        if (node.fixed) {
            // Browner
            r = (165 + r) / 2;
            g = (42 + g) / 2;
            b = (42 + b) / 2;
        }

        cv::circle(scatter_image, cv::Point(x, y), 1, cv::Scalar(b, g, r), cv::FILLED);
    }

    // Add axis labels
    cv::putText(scatter_image, "f_star", cv::Point(img_width / 2, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(scatter_image, "f_init", cv::Point(10, img_height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Add indicators to the x-axis every 360
    for (float tick = min_f_star; tick <= max_f_star; tick += 360.0f) {
        int x = static_cast<int>((tick - min_f_star) * x_scale) + margin;
        cv::line(scatter_image, cv::Point(x, img_height - margin), cv::Point(x, img_height - margin + 10), cv::Scalar(0, 0, 0), 1);
    }

    // Save the scatter plot image to a file if filename is not empty
    if (!filename.empty()) {
        cv::imwrite(filename, scatter_image);
    }

    // Display the scatter plot
    cv::imshow("Scatter Plot Sides of Nodes", scatter_image);
    cv::waitKey(1);
}

void plot_nodes_happyness(const std::vector<Node>& graph, const std::string& filename, int happiness_selection=0) {
    float confidence_t = 0.6f;
    float smeared_confidence_t = 0.55f;
    float closeness_t = 0.3f;
    
    // Find min and max values for f_star and f_init
    // float min_f_star = std::numeric_limits<float>::max();
    // float max_f_star = std::numeric_limits<float>::lowest();
    float min_f_init = std::numeric_limits<float>::max();
    float max_f_init = std::numeric_limits<float>::lowest();
    float min_gt = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        // min_f_star = std::min(min_f_star, node.f_star);
        // max_f_star = std::max(max_f_star, node.f_star);
        min_f_init = std::min(min_f_init, node.f_init);
        max_f_init = std::max(max_f_init, node.f_init);
        min_gt = std::min(min_gt, - node.gt_f_star);
    }
    // round to nearest 360
    min_gt = roundf(min_gt / 360.0f - 1.0f) * 360.0f;

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.001f);
    float dif_03 = 0.3f * (max_f_star - min_f_star);
    min_f_star -= dif_03;
    max_f_star += dif_03;

    // Define image dimensions and margins
    int img_width = 2000;
    int img_height = 1000;
    int margin = 50;

    cv::Mat scatter_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axes
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(img_width - margin, img_height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(scatter_image, cv::Point(margin, img_height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2);

    // Scale factors to map values to pixel coordinates
    float x_scale = (img_width - 2 * margin) / (max_f_star - min_f_star);
    float y_scale = (img_height - 2 * margin) / (max_f_init - min_f_init);

    std::string happiness_label = "not-selected";
    // Plot each node as a point
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int x = static_cast<int>((node.f_star - min_f_star) * x_scale) + margin;
        int y = img_height - margin - static_cast<int>((node.f_init - min_f_init) * y_scale);

        // Determine color based on gt_f_star
        float happiness = 0.0f;
        switch (happiness_selection)
        {
        case 0:
            happiness = node.happiness;
            happiness_label = "Scatter Plot Happyness of Nodes";
            break;
        case 1:
            happiness = node.confidence;
            happiness_label = "Scatter Plot Confidence of Nodes";
            break;
        case 2:
            happiness = node.smeared_confidence;
            happiness_label = "Scatter Plot Smeared Confidence of Nodes";
            break;
        case 3:
            happiness = node.closeness;
            happiness_label = "Scatter Plot Closeness of Nodes";
            break;
        case 4:
            happiness = node.confidence > confidence_t ? 1.0f : 0.0f;
            happiness_label = "Scatter Plot Confidence > thresh of Nodes";
            break;
        case 5:
            happiness = node.smeared_confidence > smeared_confidence_t ? 1.0f : 0.0f;
            happiness_label = "Scatter Plot Smeared Confidence > thresh of Nodes";
            break;
        case 6:
            happiness = node.closeness > closeness_t ? 1.0f : 0.0f;
            happiness_label = "Scatter Plot Closeness > thresh of Nodes";
            break;
        case 7:
            // float t1 = node.confidence > confidence_t ? 1.0f : 0.0f;
            // float t2 = node.smeared_confidence > smeared_confidence_t ? 1.0f : 0.0f;
            // float t3 = node.closeness > closeness_t ? 1.0f : 0.0f;
            // happiness = t1 * t2 * t3;
            happiness = node.happiness_v2;
            happiness_label = "Scatter Plot Confidence > 0.5 and Smeared Confidence > 0.4 and Closeness > 0.1 of Nodes";
            break;
        }
        float color_gradient = 2.0f + happiness;
        // float color_gradient = (node.gt_f_star - min_f_star + 180) / 360.0f;
        int cg = static_cast<int>(color_gradient);
        int color_index = static_cast<int>(color_gradient) % 3;
        int color_index_next = (color_index + 1) % 3;
        color_gradient -= cg;

        unsigned char r = 0, g = 0, b = 0;
        unsigned char r_next = 0, g_next = 0, b_next = 0;

        switch (color_index) {
            case 0: r = 255; g = 0; b = 0; break;   // Red
            case 1: r = 0; g = 0; b = 255; break;   // Blue
            case 2: r = 0; g = 255; b = 0; break;   // Green
        }

        switch (color_index_next) {
            case 0: r_next = 255; g_next = 0; b_next = 0; break;   // Red
            case 1: r_next = 0; g_next = 0; b_next = 255; break;   // Blue
            case 2: r_next = 0; g_next = 255; b_next = 0; break;   // Green
        }

        r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r * (1.0f - color_gradient) + r_next * color_gradient)));
        g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g * (1.0f - color_gradient) + g_next * color_gradient)));
        b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b * (1.0f - color_gradient) + b_next * color_gradient)));

        cv::circle(scatter_image, cv::Point(x, y), 1, cv::Scalar(b, g, r), cv::FILLED);
    }

    // Add axis labels
    cv::putText(scatter_image, "f_star", cv::Point(img_width / 2, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(scatter_image, "f_init", cv::Point(10, img_height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Add indicators to the x-axis every 360
    for (float tick = min_f_star; tick <= max_f_star; tick += 360.0f) {
        int x = static_cast<int>((tick - min_f_star) * x_scale) + margin;
        cv::line(scatter_image, cv::Point(x, img_height - margin), cv::Point(x, img_height - margin + 10), cv::Scalar(0, 0, 0), 1);
    }

    // Save the scatter plot image to a file if filename is not empty
    if (!filename.empty()) {
        cv::imwrite(filename, scatter_image);
    }

    // Display the scatter plot
    cv::imshow(happiness_label, scatter_image);
    cv::waitKey(1);
}

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    // Find min and max f_star values
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);

    // Calculate bucket size
    float bucket_size = (max_f - min_f) / num_buckets;

    // Initialize the histogram with 0 counts
    std::vector<int> histogram(num_buckets, 0);

    // Fill the histogram
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        int bucket_index = static_cast<int>((node.f_star - min_f) / bucket_size);
        if (bucket_index >= 0 && bucket_index < num_buckets) {
            histogram[bucket_index]++;
        }
    }

    // Create a blank image for the histogram with padding on the left
    int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
    int hist_h = 800;  // height of the histogram image
    int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel
    int left_padding = 50;  // Add 50 pixels of padding on the left side

    cv::Mat hist_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels and padding

    // Normalize the histogram to fit in the image
    int max_value = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < num_buckets; ++i) {
        histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
    }

    // Draw the histogram with left padding
    for (int i = 0; i < num_buckets; ++i) {
        cv::rectangle(hist_image, 
                      cv::Point(left_padding + i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
                      cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
                      cv::Scalar(0, 0, 0), 
                      cv::FILLED);
    }

    // Add x-axis labels
    std::string min_label = "Min: " + std::to_string(min_f);
    std::string max_label = "Max: " + std::to_string(max_f);
    cv::putText(hist_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(hist_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Save the histogram image to a file if string not empty
    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    // Display the histogram
    cv::imshow("Histogram of f_star values", hist_image);
    cv::waitKey(1);
}

void calculate_happyness_histogram(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    // Find min and max f_star values
    float min_h = min_happyness(graph);
    float max_h = max_happyness(graph);

    // Calculate bucket size
    float bucket_size = (max_h - min_h) / num_buckets;

    // Initialize the histogram with 0 counts
    std::vector<int> histogram(num_buckets, 0);

    // Fill the histogram
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        int bucket_index = static_cast<int>((node.happiness - min_h) / bucket_size);
        if (bucket_index >= 0 && bucket_index < num_buckets) {
            histogram[bucket_index]++;
        }
    }

    // Create a blank image for the histogram with padding on the left
    int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
    int hist_h = 800;  // height of the histogram image
    int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel
    int left_padding = 50;  // Add 50 pixels of padding on the left side

    cv::Mat hist_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels and padding

    // Normalize the histogram to fit in the image
    int max_value = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < num_buckets; ++i) {
        histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
    }

    // Draw the histogram with left padding
    for (int i = 0; i < num_buckets; ++i) {
        cv::rectangle(hist_image, 
                      cv::Point(left_padding + i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
                      cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
                      cv::Scalar(0, 0, 0), 
                      cv::FILLED);
    }

    // Add x-axis labels
    std::string min_label = "Min: " + std::to_string(min_h);
    std::string max_label = "Max: " + std::to_string(max_h);
    cv::putText(hist_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(hist_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Save the histogram image to a file if string not empty
    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    // Display the histogram
    cv::imshow("Histogram of happyness values", hist_image);
    cv::waitKey(1);
}

void calculate_histogram_k(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    float min_f_init = -180.0f;
    float max_f_init = 180.0f;
    for (const auto& node : graph) {
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            min_f_init = std::min(min_f_init, edge.k);
            max_f_init = std::max(max_f_init, edge.k);
        }
    }
    float bucket_size = (max_f_init - min_f_init) / num_buckets;

    // Histograms for each category
    std::vector<int> hist_good_same(num_buckets, 0);
    std::vector<int> hist_good_diff(num_buckets, 0);
    std::vector<int> hist_bad_same(num_buckets, 0);
    std::vector<int> hist_bad_diff(num_buckets, 0);

    for (const auto& node : graph) {
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];

            // Bin source node's f_init value
            int source_bucket = static_cast<int>((edge.k - min_f_init) / bucket_size);
            source_bucket = std::max(0, std::min(num_buckets - 1, source_bucket)); // Ensure bucket is in range

            // Bin target node's f_init value

            // Increment counts in appropriate histogram based on edge type
            if (edge.same_block) {
                hist_good_same[source_bucket]++;
            } else if (!edge.same_block) {
                hist_good_diff[source_bucket]++;
            }
        }
    }

    int hist_w = num_buckets;  
    int hist_h = 400;  
    int bin_w = std::max(1, hist_w / num_buckets);
    int left_padding = 50;  

    cv::Mat hist_image(hist_h * 2, (hist_w + left_padding + 100) * 2, CV_8UC3, cv::Scalar(255, 255, 255));  

    std::vector<std::vector<int>> histograms = {hist_good_same, hist_good_diff, hist_bad_same, hist_bad_diff};
    std::vector<std::string> labels = {"Good, Same Block", "Good, Different Block", "Bad, Same Block", "Bad, Different Block"};

    for (int k = 0; k < 4; ++k) {
        int max_value = *std::max_element(histograms[k].begin(), histograms[k].end());
        max_value = max_value / 10000;
        max_value = std::max(max_value, 1);
        for (int i = 0; i < num_buckets; ++i) {
            histograms[k][i] = std::min((hist_h - 50), (histograms[k][i] * (hist_h - 50)) / max_value);
        }

        int x_offset = (k % 2) * (hist_w + left_padding + 100);
        int y_offset = (k / 2) * hist_h;

        for (int i = 0; i < num_buckets; ++i) {
            cv::rectangle(hist_image,
                          cv::Point(x_offset + left_padding + i * bin_w, y_offset + hist_h - histograms[k][i] - 50),
                          cv::Point(x_offset + left_padding + (i + 1) * bin_w, y_offset + hist_h - 50),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);
        }

        cv::putText(hist_image, labels[k], cv::Point(x_offset + 10, y_offset + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(hist_image, std::to_string(min_f_init), cv::Point(x_offset + 10, y_offset + hist_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(hist_image, std::to_string(max_f_init), cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    cv::imshow("Histogram of Edge k by Edge Type", hist_image);
    cv::waitKey(1);
}

void calculate_histogram_edges(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    float min_certainty = std::numeric_limits<float>::max();
    float max_certainty = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            min_certainty = std::min(min_certainty, edge.certainty);
            max_certainty = std::max(max_certainty, edge.certainty);
        }
    }

    float scale = 1.0f / (max_certainty - min_certainty + 1.0e-03f) - 1.0e-03f;
    float offset = -min_certainty + 1.0e-05f;
    float bucket_size = 1.0f / num_buckets;

    // Histograms for each category
    std::vector<int> hist_good_same(num_buckets, 0);
    std::vector<int> hist_good_diff(num_buckets, 0);
    std::vector<int> hist_bad_same(num_buckets, 0);
    std::vector<int> hist_bad_diff(num_buckets, 0);

    // Edge counts for display in the middle of each histogram
    int count_good_same = 0;
    int count_good_diff = 0;
    int count_bad_same = 0;
    int count_bad_diff = 0;

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (!edge.gt_edge) {
                continue;
            }
            float value = edge.certainty;
            value = (value + offset) * scale;

            if (value < 0.0f) {
                std::cout << "Value: " << value << std::endl;
                continue;
            }
            else if (value > 1.0f) { // normalization fucked up, fixing it
                value = 1.0f;
            }
            int bucket_index = static_cast<int>(value / bucket_size);

            if (bucket_index >= 0 && bucket_index < num_buckets) {
                if (edge.good_edge && edge.same_block) {
                    hist_good_same[bucket_index]++;
                    count_good_same++;
                } else if (edge.good_edge && !edge.same_block) {
                    hist_good_diff[bucket_index]++;
                    count_good_diff++;
                } else if (!edge.good_edge && edge.same_block) {
                    hist_bad_same[bucket_index]++;
                    count_bad_same++;
                } else {
                    hist_bad_diff[bucket_index]++;
                    count_bad_diff++;
                }
            }
            else {
                std::cout << "Bucket index out of range: " << bucket_index << std::endl;
            }
        }
    }

    int hist_w = num_buckets;  
    int hist_h = 400;  
    int bin_w = std::max(1, hist_w / num_buckets);
    int left_padding = 50;  

    cv::Mat hist_image(hist_h * 2, (hist_w + left_padding + 100) * 2, CV_8UC3, cv::Scalar(255, 255, 255));  

    std::vector<std::vector<int>> histograms = {hist_good_same, hist_good_diff, hist_bad_same, hist_bad_diff};
    std::vector<std::string> labels = {"Good, Same Block", "Good, Different Block", "Bad, Same Block", "Bad, Different Block"};
    std::vector<int> edge_counts = {count_good_same, count_good_diff, count_bad_same, count_bad_diff};

    // std::cout << "Histogram bins. Good, Same Block: " << hist_good_same.size() << ", Good, Different Block: " << hist_good_diff.size() << ", Bad, Same Block: " << hist_bad_same.size() << ", Bad, Different Block: " << hist_bad_diff.size() << std::endl;

    for (int k = 0; k < 4; ++k) {
        int max_value = *std::max_element(histograms[k].begin(), histograms[k].end());
        int threshold = 0.01 * edge_counts[k];
        threshold = edge_counts[k];
        max_value = std::min(max_value, threshold);
        max_value = std::max(max_value, 1);
        for (int i = 0; i < num_buckets; ++i) {
            histograms[k][i] = std::min((hist_h - 50), (histograms[k][i] * (hist_h - 50)) / max_value);
        }

        int x_offset = (k % 2) * (hist_w + left_padding + 100);
        int y_offset = (k / 2) * hist_h;

        for (int i = 0; i < num_buckets; ++i) {
            cv::rectangle(hist_image,
                          cv::Point(x_offset + left_padding + i * bin_w, y_offset + hist_h - histograms[k][i] - 50),
                          cv::Point(x_offset + left_padding + (i + 1) * bin_w, y_offset + hist_h - 50),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);
        }

        cv::putText(hist_image, labels[k], cv::Point(x_offset + 10, y_offset + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        if (k < 2) {
            cv::putText(hist_image, "0 (Bad)", cv::Point(x_offset + 10, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            cv::putText(hist_image, "1 (Good)", cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
        else {
            cv::putText(hist_image, "0 (Good)", cv::Point(x_offset + 10, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            cv::putText(hist_image, "1 (Bad)", cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }

        // Display the edge count in the middle of each histogram
        std::string count_text = "Edges: " + std::to_string(edge_counts[k]);
        // Adjusted position for edge count to be centered between "0" and "1" labels
        cv::putText(hist_image, count_text,
                    cv::Point(x_offset + left_padding + hist_w / 2 - 40, y_offset + hist_h - 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    }

    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    cv::imshow("Histogram of Edge Certainties by Category", hist_image);
    cv::waitKey(1);
}

void calculate_histogram_edges_k(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    float min_certainty_same = std::numeric_limits<float>::max();
    float max_certainty_same = std::numeric_limits<float>::min();
    float min_certainty_other = std::numeric_limits<float>::max();
    float max_certainty_other = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (!edge.gt_edge) {
                continue;
            }
            if (edge.same_block) {
                min_certainty_same = std::min(min_certainty_same, edge.k);
                max_certainty_same = std::max(max_certainty_same, edge.k);
            } else {
                min_certainty_other = std::min(min_certainty_other, edge.k);
                max_certainty_other = std::max(max_certainty_other, edge.k);
            }
        }
    }

    // Histograms for each category
    std::vector<int> hist_good_same(num_buckets, 0);
    std::vector<int> hist_good_diff(num_buckets, 0);
    std::vector<int> hist_bad_same(num_buckets, 0);
    std::vector<int> hist_bad_diff(num_buckets, 0);

    float mean_value_good_same = 0.0f;
    float mean_value_good_diff = 0.0f;
    float mean_value_bad_same = 0.0f;
    float mean_value_bad_diff = 0.0f;

    // Edge counts for display in the middle of each histogram
    int count_good_same = 0;
    int count_good_diff = 0;
    int count_bad_same = 0;
    int count_bad_diff = 0;

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (!edge.gt_edge) {
                continue;
            }
            float value = edge.k;

            float scale = 1.0f;
            float offset = - min_certainty_other;
            float bucket_size = (max_certainty_other - min_certainty_other) / num_buckets;
            if (edge.same_block) {
                offset = - min_certainty_same;
                bucket_size = (max_certainty_same - min_certainty_same) / num_buckets;
            }
            value = (value + offset) * scale;

            if (value < 0.0f) {
                std::cout << "Value: " << value << std::endl;
                continue;
            }

            int bucket_index = static_cast<int>(value / bucket_size);

            if (bucket_index >= 0 && bucket_index < num_buckets) {
                if (edge.good_edge && edge.same_block) {
                    hist_good_same[bucket_index]++;
                    mean_value_good_same += edge.k;
                    count_good_same++;
                } else if (edge.good_edge && !edge.same_block) {
                    hist_good_diff[bucket_index]++;
                    mean_value_good_diff += edge.k;
                    count_good_diff++;
                } else if (!edge.good_edge && edge.same_block) {
                    hist_bad_same[bucket_index]++;
                    mean_value_bad_same += edge.k;
                    count_bad_same++;
                } else {
                    hist_bad_diff[bucket_index]++;
                    mean_value_bad_diff += edge.k;
                    count_bad_diff++;
                }
            }
            else {
                std::cout << "Bucket index out of range: " << bucket_index << std::endl;
            }
        }
    }

    int hist_w = num_buckets;  
    int hist_h = 400;  
    int bin_w = std::max(1, hist_w / num_buckets);
    int left_padding = 50;  

    cv::Mat hist_image(hist_h * 2, (hist_w + left_padding + 100) * 2, CV_8UC3, cv::Scalar(255, 255, 255));  

    std::vector<std::vector<int>> histograms = {hist_good_same, hist_good_diff, hist_bad_same, hist_bad_diff};
    std::vector<std::string> labels = {"Good, Same Block", "Good, Different Block", "Bad, Same Block", "Bad, Different Block"};
    std::vector<int> edge_counts = {count_good_same, count_good_diff, count_bad_same, count_bad_diff};
    std::vector<float> mean_values = {mean_value_good_same, mean_value_good_diff, mean_value_bad_same, mean_value_bad_diff};
    // std::cout << "Histogram bins. Good, Same Block: " << hist_good_same.size() << ", Good, Different Block: " << hist_good_diff.size() << ", Bad, Same Block: " << hist_bad_same.size() << ", Bad, Different Block: " << hist_bad_diff.size() << std::endl;

    for (int k = 0; k < 4; ++k) {
        int max_value = *std::max_element(histograms[k].begin(), histograms[k].end());
        // mean value
        float mean_value = mean_values[k] / edge_counts[k];
        int threshold = 0.01 * edge_counts[k];
        threshold = edge_counts[k];
        max_value = std::min(max_value, threshold);
        max_value = std::max(max_value, 1);
        for (int i = 0; i < num_buckets; ++i) {
            histograms[k][i] = std::min((hist_h - 50), (histograms[k][i] * (hist_h - 50)) / max_value);
        }

        int x_offset = (k % 2) * (hist_w + left_padding + 100);
        int y_offset = (k / 2) * hist_h;

        for (int i = 0; i < num_buckets; ++i) {
            cv::rectangle(hist_image,
                          cv::Point(x_offset + left_padding + i * bin_w, y_offset + hist_h - histograms[k][i] - 50),
                          cv::Point(x_offset + left_padding + (i + 1) * bin_w, y_offset + hist_h - 50),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);
        }

        cv::putText(hist_image, labels[k], cv::Point(x_offset + 10, y_offset + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        if (k < 2) {
            cv::putText(hist_image, std::to_string(min_certainty_same), cv::Point(x_offset + 10, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            cv::putText(hist_image, std::to_string(max_certainty_same), cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
        else {
            cv::putText(hist_image, std::to_string(min_certainty_other), cv::Point(x_offset + 10, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            cv::putText(hist_image, std::to_string(max_certainty_other), cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }

        // Display the edge count in the middle of each histogram
        std::string count_text = "Edges: " + std::to_string(edge_counts[k]) + ", Mean: " + std::to_string(mean_value);
        // Adjusted position for edge count to be centered between "0" and "1" labels
        cv::putText(hist_image, count_text,
                    cv::Point(x_offset + left_padding + hist_w / 2 - 40, y_offset + hist_h - 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    }

    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    cv::imshow("Histogram of Edge Certainties by Category", hist_image);
    cv::waitKey(1);
}

void calculate_histogram_edges_f_init(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    float min_f_init = -180.0f;
    float max_f_init = 180.0f;
    float bucket_size = (max_f_init - min_f_init) / num_buckets;

    // Histograms for each category
    std::vector<int> hist_good_same(num_buckets, 0);
    std::vector<int> hist_good_diff(num_buckets, 0);
    std::vector<int> hist_bad_same(num_buckets, 0);
    std::vector<int> hist_bad_diff(num_buckets, 0);

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (!edge.gt_edge) {
                continue;
            }

            // Bin source node's f_init value
            int source_bucket = static_cast<int>((node.f_init - min_f_init) / bucket_size);
            source_bucket = std::max(0, std::min(num_buckets - 1, source_bucket)); // Ensure bucket is in range

            // Bin target node's f_init value
            int target_bucket = static_cast<int>((graph[edge.target_node].f_init - min_f_init) / bucket_size);
            target_bucket = std::max(0, std::min(num_buckets - 1, target_bucket)); // Ensure bucket is in range

            // Increment counts in appropriate histogram based on edge type
            if (edge.good_edge && edge.same_block) {
                hist_good_same[source_bucket]++;
                hist_good_same[target_bucket]++;
            } else if (edge.good_edge && !edge.same_block) {
                hist_good_diff[source_bucket]++;
                hist_good_diff[target_bucket]++;
            } else if (!edge.good_edge && edge.same_block) {
                hist_bad_same[source_bucket]++;
                hist_bad_same[target_bucket]++;
            } else {
                hist_bad_diff[source_bucket]++;
                hist_bad_diff[target_bucket]++;
            }
        }
    }

    int hist_w = num_buckets;  
    int hist_h = 400;  
    int bin_w = std::max(1, hist_w / num_buckets);
    int left_padding = 50;  

    cv::Mat hist_image(hist_h * 2, (hist_w + left_padding + 100) * 2, CV_8UC3, cv::Scalar(255, 255, 255));  

    std::vector<std::vector<int>> histograms = {hist_good_same, hist_good_diff, hist_bad_same, hist_bad_diff};
    std::vector<std::string> labels = {"Good, Same Block", "Good, Different Block", "Bad, Same Block", "Bad, Different Block"};

    for (int k = 0; k < 4; ++k) {
        int max_value = *std::max_element(histograms[k].begin(), histograms[k].end());
        max_value = std::max(max_value, 1);
        for (int i = 0; i < num_buckets; ++i) {
            histograms[k][i] = std::min((hist_h - 50), (histograms[k][i] * (hist_h - 50)) / max_value);
        }

        int x_offset = (k % 2) * (hist_w + left_padding + 100);
        int y_offset = (k / 2) * hist_h;

        for (int i = 0; i < num_buckets; ++i) {
            cv::rectangle(hist_image,
                          cv::Point(x_offset + left_padding + i * bin_w, y_offset + hist_h - histograms[k][i] - 50),
                          cv::Point(x_offset + left_padding + (i + 1) * bin_w, y_offset + hist_h - 50),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);
        }

        cv::putText(hist_image, labels[k], cv::Point(x_offset + 10, y_offset + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(hist_image, "-180", cv::Point(x_offset + 10, y_offset + hist_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(hist_image, "180", cv::Point(x_offset + hist_w - 100, y_offset + hist_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    cv::imshow("Histogram of Node f_init by Edge Type", hist_image);
    cv::waitKey(1);
}

void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps) {
    std::vector<cv::String> filenames;
    cv::glob(directory + "/*.png", filenames);

    if (filenames.empty()) {
        std::cerr << "No images found in directory: " << directory << std::endl;
        return;
    }

    // Function to extract numbers from filenames
    auto extract_numbers = [](const std::string& filename) {
        std::vector<int> numbers;
        std::regex re(R"(_(-?\d+))"); // Match digits (including negative) preceded by an underscore
        std::smatch match;
        std::string::const_iterator search_start(filename.cbegin());
        while (std::regex_search(search_start, filename.cend(), match, re)) {
            numbers.push_back(std::stoi(match[1]));
            search_start = match.suffix().first;
        }
        return numbers;
    };

    // Custom comparator to sort filenames based on extracted numbers
    auto filename_comparator = [&extract_numbers](const std::string& a, const std::string& b) {
        std::vector<int> numbers_a = extract_numbers(a);
        std::vector<int> numbers_b = extract_numbers(b);
        return numbers_a < numbers_b; // Compare lexicographically
    };

    // Sort filenames using the custom comparator
    std::sort(filenames.begin(), filenames.end(), filename_comparator);

    // Read the first image to get the frame size
    cv::Mat first_image = cv::imread(filenames[0]);
    if (first_image.empty()) {
        std::cerr << "Error reading image: " << filenames[0] << std::endl;
        return;
    }

    // Create a VideoWriter object
    cv::VideoWriter video(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, first_image.size());

    for (const auto& file : filenames) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "Error reading image: " << file << std::endl;
            continue;
        }
        video.write(img);
    }

    video.release();
    std::cout << "Video created successfully: " << output_file << std::endl;
}

void shear_graph(std::vector<Node>& graph, float shear_amount, float scale_factor) {
    // fix upper nodes
    float start_f_init = 160.0f;
    float end_f_init = 180.0f;

    float min_f_star_value = min_f_star(graph);

    for (auto& node : graph) {
        if (node.f_init >= start_f_init && node.f_init <= end_f_init) {
            node.f_star = (node.f_star - min_f_star_value) * scale_factor + shear_amount;
            node.f_tilde = node.f_star;
            node.fixed = true;
        }
        else {
            node.fixed = false;
        }
    }

    // fix lower nodes
    start_f_init = -180.0f;
    end_f_init = -160.0f;

    for (auto& node : graph) {
        if (node.f_init >= start_f_init && node.f_init <= end_f_init) {
            node.f_star = (node.f_star - min_f_star_value) * scale_factor + shear_amount;
            node.f_tilde = node.f_star;
            node.fixed = true;
        }
        else {
            node.fixed = false;
        }
    }

    for (auto& node : graph) {
        if (node.f_init > -160.0f && node.f_init < 160.0f) {
            node.f_star = (node.f_star - min_f_star_value) * scale_factor + node.f_init*shear_amount/180.0f;
            node.f_tilde = node.f_star;
        }
    }
}

void unshear_graph(std::vector<Node>& graph, float shear_amount, float scale_factor) {
    // fix upper nodes
    float start_f_init = 160.0f;
    float end_f_init = 180.0f;

    for (auto& node : graph) {
        if (node.f_init >= start_f_init && node.f_init <= end_f_init) {
            node.f_star = (node.f_star - shear_amount) / scale_factor;
            node.f_tilde = node.f_star;
            node.fixed = false;
        }
        else {
            node.fixed = false;
        }
    }

    // fix lower nodes
    start_f_init = -180.0f;
    start_f_init = -160.0f;

    for (auto& node : graph) {
        if (node.f_init >= start_f_init && node.f_init <= end_f_init) {
            node.f_star = (node.f_star - shear_amount) / scale_factor;
            node.f_tilde = node.f_star;
            node.fixed = false;
        }
        else {
            node.fixed = false;
        }
    }

    for (auto& node : graph) {
        if (node.f_init > -160.0f && node.f_init < 160.0f) {
            node.f_star = (node.f_star - shear_amount/node.f_init) / scale_factor;
            node.f_tilde = node.f_star;
        }
    }
}

bool check_graph_symmetric(const std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        for (int j = 0; j < graph[i].num_edges; ++j) {
            const Edge& edge = graph[i].edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            // find edge in target node
            bool found = false;
            for (int k = 0; k < graph[edge.target_node].num_edges; ++k) {
                if (graph[edge.target_node].edges[k].target_node == i && std::abs(graph[edge.target_node].edges[k].k + edge.k) < 1.0e-03) {
                    if (found) {
                        std::cout << "Edge check found twice" << std::endl;
                        return false;
                    }
                    found = true;
                }
            }
            if (!found) {
                std::cout << "Edge check not found" << std::endl;
                return false;
            }
        }
    }
    return true;
}

int find_edge(std::vector<Edge>& edges, size_t target_node, float k) {
    for (int i = 0; i < edges.size(); ++i) {
        if (edges[i].target_node == target_node && std::abs(edges[i].k - k) < 1.0e-02) {
            return i;
        }
    }
    return -1;
}

int find_edge_array(Edge* edges, size_t num_edges, size_t target_node, float k) {
    for (int i = 0; i < num_edges; ++i) {
        if (edges[i].target_node == target_node && std::abs(edges[i].k - k) < 1.0e-02) {
            return i;
        }
    }
    return -1;
}

void filter_graph_f_star(std::vector<Node>& graph, float f_star_threshold) {
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        std::vector<Edge> new_edges;
        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (std::abs(graph[i].f_star + edge.k - graph[edge.target_node].f_star) <= f_star_threshold) {
                new_edges.push_back(edge);
            }
        }
        graph[i].num_edges = new_edges.size();
        for (int j = 0; j < graph[i].num_edges; ++j) {
            graph[i].edges[j] = new_edges[j];
        }
    }
}

void filter_graph_sides(std::vector<Node>& graph) {
    // remove edges if the k value is not correct wrt the sides
    for (size_t i = 0; i < graph.size(); ++i) {
        Node& node = graph[i];
        if (node.deleted) {
            continue;
        }
        int max_side_node = 0;
        for (int j = 0; j < node.sides_nr; ++j) {
            if (node.sides_old[j] > node.sides_old[max_side_node]) {
                max_side_node = j;
            }
        }

        std::vector<Edge> new_edges;
        for (int j = 0; j < node.num_edges; ++j) {
            Edge edge = node.edges[j]; // Copy of edge to store in vector and later assign back to node
            size_t target_node = edge.target_node;
            if (graph[target_node].deleted) {
                continue;
            }
            // get max side of target node
            int max_side_target = 0;
            for (int l = 0; l < node.sides_nr; ++l) {
                if (graph[target_node].sides_old[l] > graph[target_node].sides_old[max_side_target]) {
                    max_side_target = l;
                }
            }
            // calculate sides side distance
            int sides_dif = (max_side_target - max_side_node + 2*node.sides_nr) % node.sides_nr; // is a ring
            // calculate k side distance
            int k_dif = static_cast<int>(std::round((node.f_init + edge.k - graph[target_node].f_init) / 360.0f)); // round and to int
            // cast into node.sides_nr ring
            k_dif = (k_dif + 2*node.sides_nr) % node.sides_nr;
            // compare 
            if (sides_dif - k_dif == 0) {
                new_edges.push_back(edge);
            }
        }

        // update edges
        node.num_edges = new_edges.size();
        // std::cout << "Num edges: " << node.num_edges << std::endl;
        for (int j = 0; j < node.num_edges; ++j) {
            node.edges[j] = new_edges[j];
        }

        if (node.num_edges == 0) {
            node.deleted = true;
        }
    }
}

bool fix_nodes_assignment(std::vector<Node>& graph, float fix_other_threshold) {
    // construct vector with edge accuracies vs f star
    std::vector<std::pair<std::tuple<size_t, size_t, float>, float>> edge_accuracies;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        if (graph[i].fixed) {
            continue;
        }
        for (int j = 0; j < graph[i].num_edges; ++j) {
            const Edge& edge = graph[i].edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (graph[edge.target_node].fixed) {
                continue;
            }
            // only undirected case, i < edge.target_node
            if (i >= edge.target_node) {
                continue;
            }
            // only other edges
            if (edge.same_block) {
                continue;
            }
            float accuracy = std::abs((graph[i].f_star + edge.k - graph[edge.target_node].f_star) / std::max(0.01f, std::abs(edge.k)));
            edge_accuracies.push_back({{i, edge.target_node, edge.k}, accuracy});
        }
    }
    // sort from lowest to highest accuracy
    std::sort(edge_accuracies.begin(), edge_accuracies.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // print first and last accuracy
    std::cout << "First accuracy: " << edge_accuracies[0].second << ", Last accuracy: " << edge_accuracies[edge_accuracies.size() - 1].second << std::endl;

    // fix lowest 1000 node edges
    int num_fixes = std::min(10000, static_cast<int>(edge_accuracies.size()));
    size_t fixed_count = 0;
    std::cout << "Worst accuracy fix: " << edge_accuracies[num_fixes - 1].second << std::endl;
    for (size_t i = 0; fixed_count < num_fixes && i < edge_accuracies.size(); ++i) {
        // if (i % 100 == 0) {
        //     if (!check_graph_symmetric(graph)) {
        //         std::cerr << "Graph is not symmetric " << i << std::endl;
        //     }
        // }
        // std::cout << "i position: " << i << ", Accuracy: " << edge_accuracies[i].second << std::endl;
        // break if accuracy is higher than threshold
        if (edge_accuracies[i].second > fix_other_threshold) {
            std::cout << i << " Accuracy higher than threshold" << std::endl;
            return true;
        }
        size_t source_node = std::get<0>(edge_accuracies[i].first);
        size_t target_node = std::get<1>(edge_accuracies[i].first);
        // continue if source or target got fixed
        if (graph[source_node].fixed || graph[target_node].fixed) {
            continue;
        }
        float k = std::get<2>(edge_accuracies[i].first);
        // find edge index
        int source_edge_index = -1;
        for (int j = 0; j < graph[source_node].num_edges; ++j) {
            if (graph[source_node].edges[j].target_node == target_node && std::abs(graph[source_node].edges[j].k - k) < 1.0e-02) {
                graph[source_node].edges[j].certainty = 10.0f;
                if (source_edge_index != -1) {
                    std::cout << i << " Error: Edge found twice source" << std::endl;
                }
                source_edge_index = j;
            }
        }
        int target_edge_index = -1;
        for (int j = 0; j < graph[target_node].num_edges; ++j) {
            if (graph[target_node].edges[j].target_node == source_node && std::abs(graph[target_node].edges[j].k + k) < 1.0e-02) {
                graph[target_node].edges[j].certainty = 10.0f;
                if (target_edge_index != -1) {
                    std::cout << i << " Error: Edge found twice target" << std::endl;
                }
                target_edge_index = j;
            }
        }
        if (source_edge_index == -1 || target_edge_index == -1) {
            std::cerr << i << " Error: Edge not found" << std::endl;
            continue;
        }
        // continue if target node got already fixed
        if (graph[target_node].fixed) {
            continue;
        }
        // Fix target node
        graph[target_node].fixed = true;
        // update edges
        // remove all edges from target node and add to source node
        std::vector<Edge> new_edges;
        for (int j = 0; j < graph[target_node].num_edges; ++j) {
            Edge edge = graph[target_node].edges[j];
            // continue if target node is source node
            if (edge.target_node == source_node) {
                continue;
            }
            // update source edge
            int source_i = find_edge_array(graph[source_node].edges, graph[source_node].num_edges, edge.target_node, k + edge.k);
            if (source_i == -1) {
                // no previous edge found, create new one
                Edge new_edge;
                new_edge.target_node = edge.target_node;
                new_edge.k = k + edge.k;
                new_edge.certainty = edge.certainty;
                new_edge.same_block = edge.same_block;
                new_edges.push_back(new_edge);
            }
            else {
                graph[source_node].edges[source_i].certainty += edge.certainty;
                graph[source_node].edges[source_i].same_block = graph[source_node].edges[source_i].same_block && edge.same_block;
            }

            // find edge in target node
            bool found_target = false;
            std::vector<Edge> new_target_edges; // aggregate edges in vector
            for (int l = 0; l < graph[edge.target_node].num_edges; ++l) {
                if (graph[edge.target_node].edges[l].target_node == source_node && std::abs(graph[edge.target_node].edges[l].k + (k + edge.k)) < 1.0e-02) {
                    found_target = true;
                    // update target edge
                    Edge new_edge;
                    new_edge.target_node = source_node;
                    new_edge.k = graph[edge.target_node].edges[l].k;
                    new_edge.certainty = graph[edge.target_node].edges[l].certainty + edge.certainty;
                    new_edge.same_block = graph[edge.target_node].edges[l].same_block && edge.same_block;
                    new_target_edges.push_back(new_edge);
                }
                else if (graph[edge.target_node].edges[l].target_node != target_node || std::abs(graph[edge.target_node].edges[l].k + edge.k) >= 1.0e-02) {
                    // add edge to target node vector
                    new_target_edges.push_back(graph[edge.target_node].edges[l]);
                }
                
            }
            if (!found_target) {
                // add edge to target node vector
                Edge new_edge;
                new_edge.target_node = source_node;
                new_edge.k = - (k + edge.k);
                new_edge.certainty = edge.certainty;
                new_edge.same_block = edge.same_block;
                new_target_edges.push_back(new_edge);
            }
            // update target node edges
            Edge* new_target_edges_array = new Edge[new_target_edges.size()];
            size_t nr_old_edges = 0;
            for (int l = 0; l < new_target_edges.size(); ++l) {
                new_target_edges_array[nr_old_edges].certainty = new_target_edges[l].certainty;
                new_target_edges_array[nr_old_edges].k = new_target_edges[l].k;
                new_target_edges_array[nr_old_edges].same_block = new_target_edges[l].same_block;
                new_target_edges_array[nr_old_edges].target_node = new_target_edges[l].target_node;
                nr_old_edges++;
            }
            // delete old edges
            // delete[] graph[edge.target_node].edges;
            // update target node edges
            graph[edge.target_node].edges = new_target_edges_array;
            graph[edge.target_node].num_edges = new_target_edges.size();
        }
        // new array for source node edges
        size_t nr_old_edges = 0;
        for (int j = 0; j < graph[source_node].num_edges; ++j) {
            if (graph[source_node].edges[j].target_node != target_node) {
                nr_old_edges++;
            }
        }
        Edge* new_source_edges = new Edge[nr_old_edges + 1 + new_edges.size()];
        new_source_edges[0].target_node = target_node;
        new_source_edges[0].k = k;
        new_source_edges[0].certainty = 0.10f;
        new_source_edges[0].same_block = graph[source_node].edges[source_edge_index].same_block;
        nr_old_edges = 1;
        // add old edges to source node
        for (int j = 0; j < graph[source_node].num_edges; ++j) {
            if (graph[source_node].edges[j].target_node != target_node) {
                new_source_edges[nr_old_edges].certainty = graph[source_node].edges[j].certainty;
                new_source_edges[nr_old_edges].k = graph[source_node].edges[j].k;
                new_source_edges[nr_old_edges].same_block = graph[source_node].edges[j].same_block;
                new_source_edges[nr_old_edges].target_node = graph[source_node].edges[j].target_node;
                nr_old_edges++;
            }
        }
        // add new edges to source and target nodes
        for (const auto& edge : new_edges) {
            new_source_edges[nr_old_edges].certainty = edge.certainty;
            new_source_edges[nr_old_edges].k = edge.k;
            new_source_edges[nr_old_edges].same_block = edge.same_block;
            new_source_edges[nr_old_edges].target_node = edge.target_node;
            nr_old_edges++;
        }
        // delete old edges
        // delete[] graph[source_node].edges;
        // update source node edges
        graph[source_node].edges = new_source_edges;
        graph[source_node].num_edges = nr_old_edges;
        // update target node edges
        graph[target_node].num_edges = 1;
        graph[target_node].edges = new Edge[1];
        graph[target_node].edges[0].target_node = source_node;
        graph[target_node].edges[0].k = -k;
        graph[target_node].edges[0].certainty = 1.0f;
        graph[target_node].edges[0].same_block = graph[source_node].edges[source_edge_index].same_block;

        // update edge count
        fixed_count++;
    }
    return false;
}

void fix_nodes(std::vector<Node>& graph, float p) {
    size_t fixed_count = graph.size() * p;
    // randomly fix fixed_count nodes
    std::vector<size_t> fixed_nodes;
    for (size_t i = 0; i < fixed_count; ++i) {
        size_t node_index = rand() % graph.size();
        while (graph[node_index].deleted || graph[node_index].fixed) {
            node_index = rand() % graph.size();
        }
        graph[node_index].fixed = true;
        fixed_nodes.push_back(node_index);
    }
}

void fix_percentile(std::vector<Node>& graph, float p) {
    auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph, p);
    for (auto& node : graph) {
        if (node.f_star >= max_percentile) {
            node.fixed = true;
        }
    }
}

void unfix_nodes(std::vector<Node>& graph) {
    for (auto& node : graph) {
        node.fixed = false;
    }
}

size_t fix_top_nodes(std::vector<Node>& graph, size_t nr_nodes, int round) {
    std::vector<std::pair<float, size_t>> score_values;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        if (graph[i].happiness_old < 0.0f) {
            continue;
        }
        score_values.push_back({graph[i].happiness_old, i});
    }

    // sort by side values in descending order
    std::sort(score_values.begin(), score_values.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // check highest score
    int nth = round * 100;
    float nth_score = score_values[std::min(nth, static_cast<int>(score_values.size()) - 1)].first;
    // float score_threshold = 0.5f; // 0.02f; for .happiness_old
    float score_threshold = 0.02f;
    // Remove the lowest scores if enough good nodes are available
    if (nth_score >= score_threshold) {
        // Take all nodes with score higher than threshold
        std::vector<std::pair<float, size_t>> new_score_values;
        for (int i = 0; i < score_values.size(); ++i) {
            if (score_values[i].first > score_threshold) {
                new_score_values.push_back(score_values[i]);
            }
        }
        score_values = new_score_values;
    }

    // update the list's scores based on proximity to winding 0
    for (size_t i = 0; i < score_values.size(); ++i) {
        float winding_nr = graph[score_values[i].second].winding_nr;
        float abs_winding_nr = std::abs(winding_nr);
        float winding_factor = std::pow(0.98f, abs_winding_nr);
        float score = score_values[i].first;
        score_values[i].first = score;
    }

    // sort again
    std::sort(score_values.begin(), score_values.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // if max score is below threshold, remove all except the highest n entries from the list
    if (nth_score < score_threshold) {
        std::cout << nth << "-th score below threshold: " << nth_score << std::endl;
        std::vector<std::pair<float, size_t>> new_score_values;
        for (int i = 0; i < nth && i < score_values.size(); ++i) {
            new_score_values.push_back(score_values[i]);
        }
        score_values = new_score_values;
    }

    // unfix all graph nodes
    for (int i = 0; i < graph.size(); ++i) {
        graph[i].fixed = false;
    }
    // fix top nr_nodes nodes
    size_t fixed_index = 0;
    int max_fixed_wnr = 0;
    int min_fixed_wnr = 0;
    for (; fixed_index < nr_nodes && fixed_index < score_values.size(); ++fixed_index) {
        graph[score_values[fixed_index].second].fixed = true;
        if (graph[score_values[fixed_index].second].winding_nr > max_fixed_wnr) {
            max_fixed_wnr = graph[score_values[fixed_index].second].winding_nr;
        }
        if (graph[score_values[fixed_index].second].winding_nr < min_fixed_wnr) {
            min_fixed_wnr = graph[score_values[fixed_index].second].winding_nr;
        }
    }
    std::cout << " Min fixed winding nr: " << min_fixed_wnr << " Max fixed winding nr: " << max_fixed_wnr << " Fixed nodes: " << fixed_index << std::endl;
    size_t fixed_nodes = fixed_index;
    // unfix the rest
    for (; fixed_index < score_values.size(); ++fixed_index) {
        graph[score_values[fixed_index].second].fixed = false;
        graph[score_values[fixed_index].second].side = 0.0f;

    }
    return fixed_nodes;
}

int fix_winding_nodes(std::vector<Node>& graph, int nr_nodes, int seed_node_old) {
    int wnr_seed_node_old = graph[seed_node_old].winding_nr;
    std::cout << "Old seed node winding number: " << wnr_seed_node_old << std::endl;
    // fixing nodes that are at most winding_offset away from the 0 winding number
    std::vector<std::pair<float, size_t>> score_values;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        // extract side accuracy of winding number
        float side_accuracy = graph[i].wnr_side;
        if (side_accuracy > 1.0f) {
            std::cout << "Error: Side accuracy is higher than 1" << std::endl;
        }
        if (side_accuracy < 0.5f) {
            continue;
        }
        float winding_nr = graph[i].winding_nr;
        // caclulate the score of each node
        float abs_winding_nr = std::abs(winding_nr);
        float score = -(abs_winding_nr / 2.0f) + side_accuracy;
        
        score_values.push_back({score, i});
    }
    // sort by side values in descending order
    std::sort(score_values.begin(), score_values.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    // unfix all graph nodes
    for (int i = 0; i < graph.size(); ++i) {
        graph[i].fixed = false;
    }
    // fix top nr_nodes nodes
    int fixed_index = 0;
    int max_fixed_wnr = 0;
    int min_fixed_wnr = 0;
    for (; fixed_index < nr_nodes && fixed_index < score_values.size(); ++fixed_index) {
        graph[score_values[fixed_index].second].fixed = true;
        if (graph[score_values[fixed_index].second].winding_nr > max_fixed_wnr) {
            max_fixed_wnr = graph[score_values[fixed_index].second].winding_nr;
        }
        if (graph[score_values[fixed_index].second].winding_nr < min_fixed_wnr) {
            min_fixed_wnr = graph[score_values[fixed_index].second].winding_nr;
        }
    }
    std::cout << "Min fixed winding nr: " << min_fixed_wnr << " Max fixed winding nr: " << max_fixed_wnr << " Fixed nodes: " << fixed_index << std::endl;
    // unfix the rest
    for (; fixed_index < score_values.size(); ++fixed_index) {
        graph[score_values[fixed_index].second].fixed = false;
        graph[score_values[fixed_index].second].side = 0.0f;
        for (int j = 0; j < graph[score_values[fixed_index].second].sides_nr; ++j) {
            graph[score_values[fixed_index].second].sides_old[j] = 0.0f;
        }
    }
    // new seed node
    if (score_values.size() == 0) {
        std::cout << "Error: No nodes found for winding number" << std::endl;
        return seed_node_old;
    }
    else {
        return static_cast<int>(score_values[0].second);
    }
}

std::vector<Node> run_solver_f_star(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float o, float spring_constant, bool visualize) {
    if (i_round < 0) {
        o = o * 0.25f;
    }
    float o_current = o;

    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // Allocate memory on the GPU
    size_t* d_valid_indices;
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));
    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    Node* d_graph;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges = nullptr;
    float* d_all_sides = nullptr;
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
    float median_f_star = 0.0f;

    std::cout << "Copied data to GPU" << std::endl;
    
    std::cout << "Solving on GPU... string constants: " << spring_constant << std::endl;
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    // Run the iterations
    for (int iter = 1; iter < num_iterations; iter++) {
        // Launch the kernel to update nodes
        update_nodes_kernel_f_star<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, o_current, spring_constant, num_valid_nodes, 200, i_round);

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        // Launch the kernel to update f_tilde with f_star
        update_f_star_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, median_f_star);
        median_f_star = 0.0f;

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        // Adjusting side logic
        int step_size = 120;
        if (iter % step_size == 0) {
            // Copy results back to the host
            auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
            // Generate filename with zero padding
            std::ostringstream filename_plot;
            filename_plot << "python_angles/winding_angles_python_" << i_round << "_" << iter << ".png";
            if (visualize) {
                plot_nodes(graph, filename_plot.str());
            }

            // median_f_star
            auto [median1, median2] = min_max_percentile_f_star(graph, 0.5f);
            median_f_star = median1;
            
            // free old host memory
            if (h_all_edges_ != nullptr) {
                delete[] h_all_edges_;
            }
            if (h_all_sides_ != nullptr) {
                delete[] h_all_sides_;
            }
            // Print
            std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line
        }
    }
    std::cout << std::endl;

    // Graph to host
    auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
    // free old h_all_edges
    if (*h_all_edges != nullptr) {
        delete[] *h_all_edges;
    }
    *h_all_edges = h_all_edges_;
    // free old h_all_sides
    if (*h_all_sides != nullptr) {
        delete[] *h_all_sides;
    }
    *h_all_sides = h_all_sides_;
    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_graph);

    return graph;
}

std::vector<Node> run_solver_ring(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float other_block_factor, float std_target, float std_target_step, bool increase_same_block_weight, bool convergence_speedup, float convergence_thresh, bool wiggle, bool standard_winding_direction, float scale_left, float scale_right, bool enable_delete_nodes) {
    float sides_moving_eps = 0.0025f;

    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // Allocate memory on the GPU
    size_t* d_valid_indices;
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));
    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    Node* d_graph;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges = nullptr;
    float* d_all_sides = nullptr;
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);

    float* d_max_convergence;
    cudaMalloc(&d_max_convergence, sizeof(float));
    cudaMemset(d_max_convergence, 0, sizeof(float));
    float* d_total_convergence;
    cudaMalloc(&d_total_convergence, sizeof(float));
    cudaMemset(d_total_convergence, 0, sizeof(float));

    std::cout << "Copied data to GPU" << std::endl;
    
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    // random seed generation unsigned long
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);

    auto [min_f_star, max_f_star] = min_max_percentile_f_star(graph, 0.1f);


    // Run the iterations
    for (int iter = 1; iter < num_iterations || num_iterations < 0; iter++) {
        // Launch the kernel to update nodes
        update_nodes_kernel_sides<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, other_block_factor, dis(gen), 
                                                                        sides_moving_eps, increase_same_block_weight, convergence_speedup, 
                                                                        min_f_star, max_f_star, iter, wiggle, standard_winding_direction);

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        float std_cutoff = increase_same_block_weight ? 5.0f : -1.0f;
        float std_scaling = standardize_graph(d_graph, d_valid_indices, num_valid_nodes, num_nodes, std_target, std_cutoff, sides_moving_eps, dis(gen), min_f_star, max_f_star, iter, wiggle, standard_winding_direction, scale_left, scale_right, enable_delete_nodes);

        // set the device convergence max and sum variable to 0
        cudaMemset(d_max_convergence, 0, sizeof(float));
        cudaMemset(d_total_convergence, 0, sizeof(float));
        // Launch the kernel to update f_tilde with f_star
        update_sides_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, d_max_convergence, d_total_convergence);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        float max_convergence, total_convergence;
        cudaMemcpy(&max_convergence, d_max_convergence, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&total_convergence, d_total_convergence, sizeof(float), cudaMemcpyDeviceToHost);
        float mean_convergence = total_convergence / num_valid_nodes;

        // Adjusting side logic
        if (std_target_step > 0.0f) {
            std_target += std_target_step;
        }
        int step_size = 120;
        if (iter % step_size == 0) {
            if (iter/step_size % 5 == 0) {
                // // Copy results back to the host
                auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, true);

                // Generate filename with zero padding
                std::ostringstream filename_plot_nodes_sides;
                filename_plot_nodes_sides << "python_sides/side_plot_nodes_python_" << i_round << "_" << iter << ".png";
                plot_nodes_sides_array(graph, filename_plot_nodes_sides.str());
                plot_nodes(graph, ""); // do not save the gt plot, saved before during f* solver run already

                // free old host memory
                if (h_all_edges_ != nullptr) {
                    delete[] h_all_edges_;
                }
                if (h_all_sides_ != nullptr) {
                    delete[] h_all_sides_;
                }
            }

            // Print
            std::cout << "\rIteration: " << iter << " std target: " << std::fixed << std::setprecision(6) << std_target << " std scaling: " << std_scaling << " max convergence: " << max_convergence << " mean convergence: " << mean_convergence << std::flush;  // Updates the same line
        }
        if (mean_convergence < convergence_thresh && convergence_thresh > 0.0f) {
            std::cout << "Convergence " << mean_convergence << " reached at iteration " << iter << std::endl;
            break;
        }
    }
    std::cout << std::endl;

    // Graph to host
    auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
    // free old h_all_edges
    if (*h_all_edges != nullptr) {
        delete[] *h_all_edges;
    }
    *h_all_edges = h_all_edges_;
    // free old h_all_sides
    if (*h_all_sides != nullptr) {
        delete[] *h_all_sides;
    }
    *h_all_sides = h_all_sides_;
    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_graph);

    return graph;
}

void spanning_tree_winding_number(std::vector<Node>& graph) {
    // builds a spanning tree and assigns the winding number from the sides
    std::vector<size_t> reached_nodes;
    std::vector<bool> visited_nodes(graph.size(), false);
    // pick start node
    size_t start_node = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            break;
        }
    }
    reached_nodes.push_back(start_node);
    visited_nodes[start_node] = true;
    // assign winding number to start node
    graph[start_node].winding_nr = 0;
    // assign winding number to all nodes connected to nodes in reached_nodes
    size_t count_assigned = 1;
    while (reached_nodes.size() > 0) {
        size_t node_index = reached_nodes.back();
        reached_nodes.pop_back();
        Node& node = graph[node_index];
        // find max sides of node
        int max_index_node = 0;
        for (int i = 0; i < node.sides_nr; ++i) {
            if (node.sides_old[i] > node.sides_old[max_index_node]) {
                max_index_node = i;
            }
        }
        for (int i = 0; i < node.num_edges; ++i) {
            Edge& edge = node.edges[i];
            size_t target_node = edge.target_node;
            if (!visited_nodes[target_node]) {
                Node& target = graph[target_node];
                if (target.deleted) {
                    continue;
                }
                // find max sides of target node
                int max_index_target = 0;
                for (int j = 0; j < target.sides_nr; ++j) {
                    if (target.sides_old[j] > target.sides_old[max_index_target]) {
                        max_index_target = j;
                    }
                }
                float recalculated_node_wnr_d = (node.f_init + edge.k - target.f_init) / 360.0f;
                // round float to int on host
                int node_wnr_dif = static_cast<int>(std::round(recalculated_node_wnr_d));
                // check if winding number is correct
                int target_edge_wnr = (max_index_node + node_wnr_dif + 3 * node.sides_nr) % node.sides_nr;
                if (target_edge_wnr != max_index_target) {
                    continue;
                }
                visited_nodes[target_node] = true;
                reached_nodes.push_back(target_node);
                // assign winding number
                target.winding_nr = node.winding_nr + node_wnr_dif;
                count_assigned++;
            }
        }
    }
    std::cout << "Assigned winding numbers to " << count_assigned << " nodes" << std::endl;
}

std::vector<Node> run_solver_winding_number(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float other_block_factor, int seed_node, int side_fix_nr, bool display) {
    size_t fix_count = 0;
    size_t fixed_deficit = 0;
    bool adjusting_side = false;

    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // Allocate memory on the GPU
    size_t* d_valid_indices;
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));
    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    Node* d_graph;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges = nullptr;
    float* d_all_sides = nullptr;
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);

    std::cout << "Copied data to GPU" << std::endl;
    
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    size_t fixed_nr_nodes = 0;
    // Run the iterations
    for (int iter = 1; side_fix_nr > 0 || iter < num_iterations; iter++) {
        // Launch the kernel to update nodes
        update_nodes_kernel_winding_number_step1<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, other_block_factor, seed_node);

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        // Launch the kernel to update nodes
        update_nodes_kernel_winding_number_step2<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, other_block_factor, seed_node);

        err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        // Launch the kernel to update f_tilde with f_star
        update_winding_number_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        // Adjusting side logic
        int step_size = 120;
        if (iter % step_size == 0) {
            // Copy results back to the host
            bool fix_nodes = side_fix_nr > 0 && !adjusting_side;
            // bool fix_nodes = side_fix_nr > 0;
            auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
            
            if (display) {
                // Generate filename with zero padding
                std::ostringstream filename_plot_happyness;
                filename_plot_happyness << "python_happyness/happyness_plot_python_" << i_round << "_" << iter << ".png";
                plot_nodes_happyness(graph, filename_plot_happyness.str()); // plotting "standard" happyness
                std::ostringstream filename_plot_nodes_winding_nrs;
                filename_plot_nodes_winding_nrs << "python_winding_numbers/plot_nodes_winding_nrs_python_" << i_round << "_" << iter << ".png";
                plot_nodes_winding_numbers(graph, filename_plot_nodes_winding_nrs.str());
            }

            // fix the top side_fix_nr nodes
            if (fix_nodes) {
                if (iter > 1000 || side_fix_nr < 500) {
                    fix_count = fix_count + side_fix_nr;
                    fixed_deficit = 6*(fix_count - fixed_deficit)/7; // 6/7 of deficit should not be fixed in next round to combat bad node assignment
                }
                else {
                    fix_count = fix_count + 500;
                }
                fixed_deficit = fix_top_nodes(graph, fix_count-fixed_deficit, iter/step_size);
                if (fixed_deficit == fixed_nr_nodes) {
                    std::cout << "All nodes are fixed" << std::endl;
                    break;
                }
                fixed_nr_nodes = fixed_deficit;
            }

            if (fix_nodes) {
                // Re-copy the updated graph to GPU
                num_nodes = graph.size();
                update_fixed_field(d_graph, graph.data(), num_nodes);
            }

            // free old host memory
            if (h_all_edges_ != nullptr) {
                delete[] h_all_edges_;
            }
            if (h_all_sides_ != nullptr) {
                delete[] h_all_sides_;
            }

            // Print
            std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line
        }
    }
    std::cout << std::endl;

    // Graph to host
    auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
    // free old h_all_edges
    if (*h_all_edges != nullptr) {
        delete[] *h_all_edges;
    }
    *h_all_edges = h_all_edges_;
    // free old h_all_sides
    if (*h_all_sides != nullptr) {
        delete[] *h_all_sides;
    }
    *h_all_sides = h_all_sides_;
    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_graph);

    return graph;
}

std::vector<Node> run_solver(std::vector<Node>& graph, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, int seed_node, float other_block_factor, int down_index, int up_index, int side_fix_nr, float std_target, float std_target_step, bool increase_same_block_weight) {
    if (i_round < 0) {
        o = o * 0.25f;
    }
    float o_current = o;
    size_t fix_count = 0;
    size_t fixed_deficit = 0;
    // float std_target = 0.013f;
    float sides_moving_eps = 0.0025f;

    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // Allocate memory on the GPU
    size_t* d_valid_indices;
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));
    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    Node* d_graph;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges = nullptr;
    float* d_all_sides = nullptr;
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);

    std::cout << "Copied data to GPU" << std::endl;
    
    std::cout << "Solving on GPU... string constants: " << spring_constant << std::endl;
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    // Float min value
    float max_winding_angle = max_f_star(graph) + 5 * 360.0f;
    float min_winding_angle = min_f_star(graph) - 5 * 360.0f;
    auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph, 0.1f);

    bool adjusting_side = false;

    // random seed generation unsigned long
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);

    // Run the iterations
    for (int iter = 1; iter < num_iterations; iter++) {
        // Launch the kernel to update nodes
        update_nodes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, o_current, spring_constant, num_valid_nodes, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), 200, 0.0f, 0.0f, down_index, up_index, min_percentile, max_percentile, false, i_round, seed_node, other_block_factor, adjusting_side, dis(gen), iter, sides_moving_eps, increase_same_block_weight);

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        standardize_graph(d_graph, d_valid_indices, num_valid_nodes, num_nodes, std_target, -1.0f, sides_moving_eps, dis(gen));

        // Launch the kernel to update f_tilde with f_star
        update_f_tilde_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        // Adjusting side logic
        int step_size = 120;
        if (side_fix_nr > 0 && iter % step_size == 0 && !adjusting_side) {
            // adjusting_side = true;
        }
        if (std_target_step > 0.0f) {
            std_target += std_target_step;
            // sides_moving_eps = fmaxf(0.0f, sides_moving_eps - 0.001f / 3000);
        }
        // else if (side_fix_nr > 0 && iter % step_size == 35 && !adjusting_side) {
        //     iter -= 2; // less subtraction forces closer 
        //     adjusting_side = true;
        // }
        // else if (side_fix_nr > 0 && iter % step_size == 35 && adjusting_side) {
        //     adjusting_side = false;
        //     for (size_t graph_index = 0; graph_index < graph.size(); ++graph_index) {
        //         if (graph[graph_index].fixed) {
        //             // fix winding nr side certainty
        //             graph[graph_index].wnr_side = 1.0f;
        //             graph[graph_index].wnr_side_old = 1.0f;
        //         }
        //     }
        // }

        if (iter % step_size == 0) {
            // Copy results back to the host
            bool fix_nodes = side_fix_nr > 0 && !adjusting_side;
            // bool fix_nodes = side_fix_nr > 0;
            auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, true);
            max_winding_angle = max_f_star(graph) + 5 * 360.0f;
            min_winding_angle = min_f_star(graph) - 5 * 360.0f;
            auto res = min_max_percentile_f_star(graph, 0.1f);
            min_percentile = res.first;
            max_percentile = res.second;
            // Generate filename with zero padding
            std::ostringstream filename_plot;
            filename_plot << "python_angles/winding_angles_python_" << i_round << "_" << iter << ".png";
            plot_nodes(graph, filename_plot.str());
            std::ostringstream filename_plot_side;
            filename_plot_side << "python_side/side_plot_side_python_" << i_round << "_" << iter << ".png";
            plot_nodes_side(graph, filename_plot_side.str());
            std::ostringstream filename_plot_happyness;
            filename_plot_happyness << "python_happyness/happyness_plot_python_" << i_round << "_" << iter << ".png";
            plot_nodes_happyness(graph, filename_plot_happyness.str(), 1);
            // plot_nodes_happyness(graph, filename_plot_happyness.str(), 2);
            plot_nodes_happyness(graph, filename_plot_happyness.str(), 3);
            // plot_nodes_happyness(graph, filename_plot_happyness.str(), 4);
            // plot_nodes_happyness(graph, filename_plot_happyness.str(), 5);
            // plot_nodes_happyness(graph, filename_plot_happyness.str(), 6);
            plot_nodes_happyness(graph, filename_plot_happyness.str(), 7);
            // plot_nodes_happyness(graph, filename_plot_happyness.str());
            std::ostringstream filename_plot_nodes_sides;
            filename_plot_nodes_sides << "python_sides/side_plot_nodes_python_" << i_round << "_" << iter << ".png";
            plot_nodes_sides_array(graph, filename_plot_nodes_sides.str());
            std::ostringstream filename_plot_nodes_winding_nrs;
            filename_plot_nodes_winding_nrs << "python_winding_numbers/plot_nodes_winding_nrs_python_" << i_round << "_" << iter << ".png";
            // plot_nodes_winding_numbers(graph, filename_plot_nodes_winding_nrs.str());

            // fix the top side_fix_nr nodes
            // int steps_inner = 8000;
            // int reset_steps = 2000;
            // towards_inside = true;
            // if (fix_nodes && (towards_inside || (iter >= (steps_inner + reset_steps)))) {
            if (fix_nodes) {
                if (iter > 1000 || side_fix_nr < 500) {
                    fix_count = fix_count + side_fix_nr;
                    fixed_deficit = 6*(fix_count - fixed_deficit)/7; // 6/7 of deficit should not be fixed in next round to combat bad node assignment
                }
                else {
                    fix_count = fix_count + 500;
                }
                fixed_deficit = fix_top_nodes(graph, fix_count-fixed_deficit, iter/step_size);
                // seed_node = fix_winding_nodes(graph, fix_count, seed_node);

                // if (towards_inside && iter >= steps_inner) {
                //     towards_inside = false;
                //     // unifx all nodes with winding number >= -1
                //     fix_count = 0;
                //     for (size_t i = 0; i < graph.size(); ++i) {
                //         if (graph[i].winding_nr <= 1) {
                //             graph[i].fixed = false;
                //         }
                //         if (graph[i].fixed) {
                //             fix_count++;
                //         }
                //     }
                //     fixed_deficit = fix_count;
                // }
            }

            if (fix_nodes) {
                // Re-copy the updated graph to GPU
                num_nodes = graph.size();
                update_fixed_field(d_graph, graph.data(), num_nodes);
            }

            // free old host memory
            if (h_all_edges_ != nullptr) {
                delete[] h_all_edges_;
            }
            if (h_all_sides_ != nullptr) {
                delete[] h_all_sides_;
            }

            // print side of seed node
            // std::cout << "Seed node: " << valid_indices[seed_node] << ", side: " << graph[valid_indices[seed_node]].side << std::endl;

            // Print
            std::cout << "\rIteration: " << iter << " std target: " << std_target << std::flush;  // Updates the same line
        }
    }
    std::cout << std::endl;

    // Graph to host
    auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
    // free old h_all_edges
    if (*h_all_edges != nullptr) {
        delete[] *h_all_edges;
    }
    *h_all_edges = h_all_edges_;
    // free old h_all_sides
    if (*h_all_sides != nullptr) {
        delete[] *h_all_sides;
    }
    *h_all_sides = h_all_sides_;
    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_graph);

    return graph;
}

// Main GPU solver function
void solve_gpu(std::vector<Node>& graph, int i, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, bool first_estimated_iteration, int estimated_windings, Node* d_graph, Edge* d_all_edges, float* d_all_sides, size_t* d_valid_indices, int num_valid_nodes, int num_nodes, bool adjust_pull, bool adjust_certainty, bool adjust_certainty_last, bool adjust_lowest, bool increase_edge_pull) {
    std::cout << "Solving on GPU... string constants: " << spring_constant << std::endl;
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    int up_index = 0;
    int down_index = 0;

    // Float min value
    float lowest_adjustment = adjust_lowest ? (min_f_star(graph) + max_f_star(graph)) / 2.0 : std::numeric_limits<float>::lowest();
    float highest_adjustment = lowest_adjustment;
    float angle_per_iteration = 360.0f / 2000.0 * 1.0f;
    float max_winding_angle = max_f_star(graph) + 5 * 360.0f;
    float min_winding_angle = min_f_star(graph) - 5 * 360.0f;
    float min_z = min_z_node(graph) - 100;
    float max_z = max_z_node(graph) + 100;
    float z_step = 3000 / 500000.0f;
    float z_adjustment_lowest = adjust_lowest ? (min_z + max_z) / 2.0 : std::numeric_limits<float>::lowest();
    float z_adjustment_highest = z_adjustment_lowest;
    float max_pull = 0.005f;
    float small_pull = (adjust_lowest || adjust_certainty_last) ? max_pull : 0.0f;
    if (adjust_lowest) {
        small_pull = max_pull;
    }
    // small_pull = 0.002f;
    small_pull = 0.0f;
    float happyness_threshold = 0.0f;
    float happyness_threshold_step = 0.0005f;
    float certainty_good = 0.5f;
    float certainty_bad = 0.001f;
    int certainty_adjustment_rounds = 1;
    int certainty_adjustment_rounds_counter = 0;
    float edge_pull = -1.0f; // TODO
    if (increase_edge_pull) {
        edge_pull = 40.0f; // activating edge pull
    }
    int edge_pull_iterations = num_iterations; // 40000;
    float edge_pull_step = 10.0f / edge_pull_iterations;

    if (adjust_lowest) {
        edge_pull = 0.5f;
    }

    // random seed generation unsigned long
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);

    // Run the iterations
    for (int iter = 0; (iter < num_iterations) || (increase_edge_pull && iter < 3*edge_pull_iterations) || (first_estimated_iteration && iter < 60000) || (adjust_lowest && ((lowest_adjustment >= min_winding_angle) || (highest_adjustment <= max_winding_angle) || (z_adjustment_lowest >= min_z) || (z_adjustment_highest <= max_z)) || adjust_certainty); ++iter) {
        // Launch the kernel to update nodes
        update_nodes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, o, spring_constant, num_valid_nodes, lowest_adjustment, highest_adjustment, z_adjustment_lowest, z_adjustment_highest, estimated_windings, small_pull, edge_pull, up_index, down_index, min_winding_angle, max_winding_angle, i >= 3, i, 100, 5.0f, false, dis(gen), iter, 0.0015f, false);

        std::cout << "Iteration: " << iter << std::endl;

        // update edge_pull
        if (increase_edge_pull) {
            float tmp_step = edge_pull_step;
            if (edge_pull < 5.0f) {
                tmp_step /= 10.0f;
            }
            edge_pull -= tmp_step;
            edge_pull = std::max(0.1f, edge_pull);
        }

        // Update the lowest adjustment value
        if (adjust_lowest) {
            lowest_adjustment -= angle_per_iteration;
            highest_adjustment += angle_per_iteration;
            z_adjustment_lowest -= z_step;
            z_adjustment_highest += z_step;
            if (iter % 100 == 0) {
                std::cout << "\rLowest adjustment: " << lowest_adjustment << ", Highest adjustment: " << highest_adjustment << ", Z lowest adjustment: " << z_adjustment_lowest << ", Z highest adjustment: " << z_adjustment_highest << std::flush;
            }
        }

        // if (adjust_pull) {
        //     small_pull = iter * max_pull / num_iterations;
        // }

        // if (adjust_certainty) {
        //     small_pull = 0.0f;
        // }

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        // Launch the kernel to update f_tilde with f_star
        update_f_tilde_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        if (adjust_certainty) {
            // Launch the kernel to get mean happiness
            float sum_happiness = 0.0f;
            float max_happiness = 0.0f;
            int count_non_deleted = 0;
            int* d_count_non_deleted;
            float* d_sum_happiness;
            float* d_max_happiness;
            cudaMalloc(&d_count_non_deleted, sizeof(int));
            cudaMalloc(&d_sum_happiness, sizeof(float));
            cudaMemset(d_count_non_deleted, 0, sizeof(int));
            cudaMemset(d_sum_happiness, 0, sizeof(float));
            cudaMalloc(&d_max_happiness, sizeof(float));

            sum_mean_happiness_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_graph, d_max_happiness, d_sum_happiness, d_count_non_deleted, num_valid_nodes);

            cudaMemcpy(&max_happiness, d_max_happiness, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sum_happiness, d_sum_happiness, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&count_non_deleted, d_count_non_deleted, sizeof(int), cudaMemcpyDeviceToHost);

            float mean_happiness = sum_happiness / count_non_deleted;
            float node_deactivation_threshold = 0.75 * ((i + 1.0f) / 5.0f) * mean_happiness;

            if (node_deactivation_threshold < happyness_threshold) {
                std::cout << "Finished adjusting certainty. Max happiness: " << max_happiness << ", Mean happiness: " << mean_happiness << std::endl;
                certainty_adjustment_rounds_counter++;
                if (certainty_adjustment_rounds_counter < certainty_adjustment_rounds) {
                    happyness_threshold = 0.0f;
                    certainty_good *= 2;
                    certainty_bad /= 2;
                    continue;
                }
                else {
                    break;
                }
            }

            happyness_threshold += happyness_threshold_step;

            // Free memory
            cudaFree(d_count_non_deleted);
            cudaFree(d_sum_happiness);

            // Synchronize to ensure all threads are finished
            cudaDeviceSynchronize();

            // Update the certainty with the computed happiness
            update_certainty_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, node_deactivation_threshold, happyness_threshold, certainty_good, certainty_bad, spring_constant);

            // Synchronize to ensure all threads are finished
            cudaDeviceSynchronize();
        }

        // std::cout << "Iteration: " << iter << std::endl;

        if (iter % 100 == 0) {
            // Copy results back to the host
            auto [h_all_edges, h_all_floats] = copy_graph_from_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
            // cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
            max_winding_angle = max_f_star(graph) + 5 * 360.0f;
            min_winding_angle = min_f_star(graph) - 5 * 360.0f;
            // Generate filename with zero padding
            std::ostringstream filename;
            filename << "histogram/histogram_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            std::ostringstream filename_happyness;
            filename_happyness << "histogram/histogram_happyness_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            std::ostringstream filename_edges;
            filename_edges << "histogram/histogram_edges_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            std::ostringstream filename_plot;
            filename_plot << "winding_angles/plot_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            std::ostringstream filename_plot_side;
            filename_plot_side << "side/plot_side_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            std::ostringstream filename_happyness_plot;
            filename_happyness_plot << "happyness_plot/happyness_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            plot_nodes(graph, filename_plot.str());
            plot_nodes_side(graph, filename_plot_side.str());
            plot_nodes_happyness(graph, filename_happyness_plot.str());

            // print side of seed node
            // std::cout << "Seed node: " << valid_indices[100] << ", side: " << graph[valid_indices[100]].side << std::endl;
            
            // Calculate and display the histogram of f_star values
            if (false && i == 3 && iter % 1000 == 0) {
                std::cout << "Copied edges to host..." << std::endl;
                bool finished = fix_nodes_assignment(graph, 100.0f);
                std::cout << "Fixed nodes assignment..." << std::endl;
                if (finished) {
                    std::cout << "Finished fixing nodes assignment" << std::endl;
                    break;
                }

                cudaDeviceSynchronize();
                // Re-copy the updated graph to GPU
                cudaFree(d_graph);
                num_nodes = graph.size();
                cudaMalloc(&d_graph, num_nodes * sizeof(Node));
                cudaError_t err = cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA memcpy failed for graph: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
                // d edges
                free_edges_from_gpu(d_all_edges);
                free_sides_from_gpu(d_all_sides);
                copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);
                cudaDeviceSynchronize();  // Synchronize at the end, after all async transfers
            }
            if (video_mode) {
                calculate_histogram(graph, filename.str());
                calculate_happyness_histogram(graph, filename_happyness.str());
                if (iter == 0 || adjust_certainty_last || (adjust_certainty && iter % 600 == 0)) {
                    // Graph edges to host
                    // std::cout << "Calculating histogram of edge certainties..." << std::endl;
                    calculate_histogram_edges(graph, filename_edges.str());
                }
            }

            if (h_all_edges != nullptr) {
                delete [] h_all_edges;
            }
            if (h_all_floats != nullptr) {
                delete [] h_all_floats;
            }

            // Print
            std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line

            // escape if estimated windings reached
            if (first_estimated_iteration) {
                // float min_f = min_f_star(graph);
                // float max_f = max_f_star(graph);
                auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph, 0.1f);
                // std::cout << " Min percentile: " << min_percentile << ", Max percentile: " << max_percentile << std::endl;
                if (max_percentile - min_percentile > 1.0f * 360.0f * estimated_windings) {
                    break;
                }
            }
        }
    }
}

void solve_gpu_session(std::vector<Node>& graph, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, int num_iterations, float o, float spring_factor, int steps, std::vector<float>& spring_constants, std::vector<size_t>& valid_indices, int iterations_factor, float o_factor, int estimated_windings, const std::string& histogram_dir, bool adjust_lowest_only) {
    calculate_histogram_edges(graph, "edges_hist.png");
    calculate_histogram_edges_f_init(graph, "edges_f_init.png", 1000);
    calculate_histogram_edges_k(graph, "edges_k.png", 1000);
    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // original edges pointers
    std::vector<Edge*> original_edges(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        original_edges[i] = graph[i].edges;
    }

    // Allocate memory on the GPU
    Node* d_graph;
    size_t* d_valid_indices;
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));

    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges;
    float* d_all_sides;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides);

    std::cout << "Copied data to GPU" << std::endl;
    
    bool continue_solver = false;
    int checkpoint_step = 1;
    // Solve for each spring constant
    steps = steps + 2;
    for (int i = -1; i < steps+1; ++i) {
        if (adjust_lowest_only) {
            // directly go to lowest adjustment
            i = steps;
        }
        int num_iterations_iteration = num_iterations;
        float o_iteration = o;
        float spring_constant_iteration = i == -1 ? spring_constants[0] : spring_constants[i];
        bool first_estimated_iteration = i == -1 && edges_deletion_round == 0 && estimated_windings > 0;
        if (i == -1 && edges_deletion_round == 0) {
            // Use a warmup iteration with 10x the spring constant
            num_iterations_iteration *= iterations_factor;
            o_iteration = o * o_factor;
            spring_constant_iteration = spring_factor;
        }
        else if (i == -1 &&edges_deletion_round < 2) {
            // Use a warmup iteration with 10x the spring constant
            num_iterations_iteration *= iterations_factor;
            spring_constant_iteration = spring_factor;
            first_estimated_iteration = true;
        }
        // // Skip the first iterations if the warmup is already done
        // else if (i == -1) {
        //     // Skip the warmup iteration for subsequent rounds
        //     continue;
        // }
        else if (i == steps && edges_deletion_round >= 1) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations_iteration *= 3.0f;
            // spring_constant_iteration = 1.0f;
            o_iteration = o * o_factor;
        }
        else if (i == steps) {
            break;
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations_iteration *= 1.5f;
            o_iteration = o * o_factor;
        }
        std::cout << "Spring Constant " << i << ": " << std::setprecision(10) << spring_constant_iteration << std::endl;
        bool adjust_pull = i == steps - 2;
        adjust_pull = false;
        // if (i == steps - 1) {
        //     // break out for edge direction solution trial
        //     break;
        // }
        // bool adjust_certainty = i == steps - 1;
        bool adjust_certainty = false;
        if (adjust_certainty) {
            num_iterations_iteration = num_iterations_iteration * 10;
        }
        // adjust_certainty = false;
        bool adjust_lowest = i == steps;

        bool increase_edge_pull = i == steps - 1;
        // if (increase_edge_pull) {
        //     spring_constant_iteration = 1.4f;
        // }
        
        // Checkpointing to continue runs during development
        {
            std::string filename = "checkpoint_graph_" + std::to_string(edges_deletion_round) + ".bin";
            if (continue_solver && i <= checkpoint_step) {
                continue;
            }
            if (continue_solver && i == checkpoint_step + 1) {
                // Load graph from binary file
                graph = loadGraph(filename);

                cudaMalloc(&d_graph, num_nodes * sizeof(Node));
                cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));

                // Copy graph and valid indices to the GPU
                cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
                cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

                // Allocate and copy edges to GPU
                copy_edges_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges);
            }
            // Save graph at checkpoint step
            else if (i == checkpoint_step + 1) {
                std::cout << "edge update step done" << std::endl;
                // Copy results back to the host
                cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
                Edge* h_all_edges = copy_edges_from_gpu(graph.data(), num_nodes, &d_all_edges); // A little slow because of the copying. should only be used when neccessary. TODO: adjust display when logic

                // Save complete graph computation to a binary file
                saveGraph(graph, filename);
                // Load graph from binary file
                std::vector<Node> graph_loaded = loadGraph(filename);
                bool is_same = graph == graph_loaded;
                std::cout << "Graph loaded is same as original: " << is_same << std::endl;

                // free edges
                delete[] h_all_edges;
            }
        }

        // if (i >= 1) {
        //     // Copy results back to the host
        //     cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
        //     Edge* h_all_edges = copy_edges_from_gpu(graph.data(), num_nodes, &d_all_edges); // A little slow because of the copying. should only be used when neccessary. TODO: adjust display when logic

        //     // unfix nodes
        //     unfix_nodes(graph);

        //     // Fix p percent nodes
        //     // float p = 0.001f;
        //     // fix_nodes(graph, p);
        //     fix_percentile(graph, 0.05);

        //     cudaDeviceSynchronize();
        //     // Re-copy the updated graph to GPU
        //     cudaFree(d_graph);
        //     num_nodes = graph.size();
        //     cudaMalloc(&d_graph, num_nodes * sizeof(Node));
        //     cudaError_t err = cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        //     if (err != cudaSuccess) {
        //         std::cerr << "CUDA memcpy failed for graph: " << cudaGetErrorString(err) << std::endl;
        //         return;
        //     }
        //     // d edges
        //     free_edges_from_gpu(d_all_edges);
        //     copy_edges_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges);
        //     cudaDeviceSynchronize();  // Synchronize at the end, after all async transfers

        //     delete [] h_all_edges;

        //     num_iterations_iteration *= 10;
        // }

        
        // Run GPU solver
        solve_gpu(graph, i, edges_deletion_round, video_mode, max_index_digits, max_iter_digits, o_iteration, spring_constant_iteration, num_iterations_iteration, valid_indices, first_estimated_iteration, estimated_windings, d_graph, d_all_edges, d_all_sides, d_valid_indices, num_valid_nodes, num_nodes, adjust_pull, adjust_certainty, adjust_certainty, adjust_lowest, increase_edge_pull);
        // if (i >= 0 && !adjust_certainty && !adjust_lowest) {
        //     solve_gpu(graph, i, edges_deletion_round, video_mode, max_index_digits, max_iter_digits, o_iteration, spring_constant_iteration, num_iterations_iteration, valid_indices, first_estimated_iteration, estimated_windings, d_graph, d_all_edges, d_valid_indices, num_valid_nodes, num_nodes, adjust_pull, true, adjust_certainty, adjust_lowest);
        // }

        // if (i == 1) {
        //     // Copy results back to the host
        //     cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
        //     Edge* h_all_edges = copy_edges_from_gpu(graph.data(), num_nodes, &d_all_edges); // A little slow because of the copying. should only be used when neccessary. TODO: adjust display when logic

        //     shear_graph(graph, 10 * 360.0f, 1.0f);

        //     cudaDeviceSynchronize();
        //     // Re-copy the updated graph to GPU
        //     cudaFree(d_graph);
        //     num_nodes = graph.size();
        //     cudaMalloc(&d_graph, num_nodes * sizeof(Node));
        //     cudaError_t err = cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        //     if (err != cudaSuccess) {
        //         std::cerr << "CUDA memcpy failed for graph: " << cudaGetErrorString(err) << std::endl;
        //         return;
        //     }
        //     // d edges
        //     free_edges_from_gpu(d_all_edges);
        //     copy_edges_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges);
        //     cudaDeviceSynchronize();  // Synchronize at the end, after all async transfers

        //     delete [] h_all_edges;
        // }

        // endline
        std::cout << std::endl;

        // After generating histograms, create a video from the images
        // if (video_mode) {
        //     create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
        // }
    }
    std::cout << "Finished solving on GPU" << std::endl;

    // Copy results back to the host
    cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    // Restore the original edges pointers (no need to copy edges back)
    for (int i = 0; i < num_nodes; ++i) {
        // Just set the edges pointer back to its original location on the CPU
        // for (int j = 0; j < graph[i].num_edges; ++j) {
        //     original_edges[i][j].certainty = graph[i].edges[j].certainty;
        // }
        graph[i].edges = original_edges[i]; // Assuming original_edges[i] points to the correct Edge array on the CPU
    }
    // // Copy edges from GPU to CPU. memory leak. oh well...
    // Edge* h_all_edges = copy_edges_from_gpu(graph.data(), num_nodes, &d_all_edges);

    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);

    cudaFree(d_graph);
    cudaFree(d_valid_indices);

    std::cout << "Freed GPU memory" << std::endl;
}
