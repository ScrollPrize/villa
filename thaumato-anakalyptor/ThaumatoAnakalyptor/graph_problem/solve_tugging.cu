// solve_tugging.cu
#include "solve_tugging.h"
#include "solve_gpu.h"
#include "node_structs.h"
#include <cuda_runtime.h>
#include <utility>
#include <sstream>
#include <iomanip>
#include <iostream>

// External functions from solve_gpu.cu
extern void copy_graph_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes,
    Edge** d_all_edges_ptr, float** d_all_sides_ptr,
    bool copy_edges, bool copy_sides);
extern std::pair<Edge*, float*> copy_graph_from_gpu(Node* h_graph, Node* d_graph,
    size_t num_nodes, Edge** d_all_edges_ptr, float** d_all_sides_ptr,
    bool copy_edges, bool copy_sides);
extern void free_edges_from_gpu(Edge* d_all_edges);
extern void free_sides_from_gpu(float* d_all_sides);

// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel_tugging_f_star(Node* d_graph, size_t* d_valid_indices, float o, float spring_constant, float step_sigma, int num_valid_nodes, int estimated_windings, int i_round) {
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
        // certainty = 0.01f;
        float k = edge.k;

        if (certainty < 0.001f) {
            continue;
        }

        float step_edge = 1.0f;
        // Node winding angle update calculation
        if (edge.same_block) {
            // update closeness
            node.same_block_closeness += fabsf(k_dif);
            // certainty *= fmaxf(1.0f, 0.25f * node.same_block_closeness_old / 360.0f);
            // certainty *= fmaxf(1.0f, 0.25f * k_dif / 360.0f);
        }
        else {
            step_edge *= 0.2f;
            float dk = n2n1 - spring_constant * k;
            float fitting_factor = fmaxf(1.0f, 1.0f / (1.0f + 0.01f * fabsf(dk)));
            float same_block_factor2 = fmaxf(1.0f, 1.0f * node.num_same_block_edges);
            float edges_factor = sqrt(1.0f * (node.num_edges - node.num_same_block_edges) * (d_graph[target_node].num_edges - d_graph[target_node].num_same_block_edges));
            float certainty_factor = 0.05f * fitting_factor * same_block_factor2 * edges_factor;
            // float certainty_factor = 0.05f * same_block_factor2 * edges_factor;
            // certainty *= certainty_factor;
            // k *= 0.02f; // wrong other block edges make the adjacent windings be closer together, if k is "the perfect" angle step, then we would have too steep winding lines, since they wrap around that would then lead to places where the lines need to bend abruptly to compensate for the too steepness compared to the distance between the windings
        }
        // calculate f star update
        float predicted_winding_angle = d_graph[target_node].f_tilde - spring_constant * k;
        float error_k = node_f_tilde - predicted_winding_angle;
        if (!edge.same_block && std::abs(error_k) > 0.250f) {
            step_edge *= 10.2f;
        }
        float k_diff = predicted_winding_angle - node.f_star;
        float step_loss = expf(-(k_diff * k_diff) / (2.0f * step_sigma * step_sigma));
        sum_w_f_tilde_k += step_edge * certainty * step_loss * (predicted_winding_angle - node.f_star);
        sum_w += step_edge * certainty * step_loss;

        // Calculate node happiness: mean difference between k and target f_tilde - node f_tilde target + target node happiness weighted multiplied by the certainty
        num_active_edges++;
    }

    if (sum_w > 0)
    {
        float step = sum_w_f_tilde_k / sum_w;
        node.f_star += step;
    }
    // Clip f_star to the range [ - 2 * 360 * estimated_windings, 2 * 360 * estimated_windings]
    float winding_max =  4 * 360 * estimated_windings;
    if (fabsf(node.f_star) >= winding_max) {
        node.f_star_momentum = 0.0f;
    }
    node.f_star = fmaxf(- winding_max, fminf(winding_max, node.f_star));
}
// f_star update kernels from solve_gpu
extern __global__ void update_nodes_kernel_f_star(
    Node* d_graph,
    size_t* d_valid_indices,
    float o,
    float spring_constant,
    float step_sigma,
    int num_valid_nodes,
    int estimated_windings,
    int i_round
);
extern __global__ void update_f_star_kernel(
    Node* d_graph,
    size_t* d_valid_indices,
    int num_valid_nodes,
    float median_f_star,
    int teflon_winding_nr,
    bool blow_away
);

// Kernel: compute diff (#front - #behind) and distribute tugging
__global__ void update_nodes_kernel_tugging(Node* d_graph, size_t* d_valid_indices,
    int num_valid_nodes, float distribute) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;
    // Count neighbors in front/behind
    int front = 0, behind = 0;
    float f_loc = node.f_star;
    int active_neighbors = 0;
    for (int j = 0; j < node.num_edges; ++j) {
        Node& nei = d_graph[node.edges[j].target_node];
        if (nei.deleted) continue;
        if (!node.edges[j].same_block) continue;
        ++active_neighbors;
        if (nei.f_star > f_loc) ++front;
        else if (nei.f_star < f_loc) ++behind;
    }
    int diff = front - behind;
    node.happiness_old = (float)diff * 0.5f + 0.5f * node.happiness_old * 0.999f;
    node.happiness = node.happiness_old / (active_neighbors + 1);
}

// Kernel: apply tugging diff to f_star before standard update
__global__ void apply_tugging_kernel(Node* d_graph, size_t* d_valid_indices,
    int num_valid_nodes, float diff_step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    // Distribute to neighbors
    float steps = node.happiness;
    for (int j = 0; j < node.num_edges; ++j) {
        Node& nei = d_graph[node.edges[j].target_node];
        if (nei.deleted) continue;
        if (!node.edges[j].same_block) continue;
        steps += nei.happiness;
    }
    node.happiness_old = steps;
    node.f_star -= diff_step * steps;
    node.f_star = fmaxf(fminf(node.f_star, 360.0f*200), -360.0f*200);
}
// Kernel: compute sum of f_star across valid nodes
__global__ void compute_f_star_sum_kernel(Node* d_graph, size_t* d_valid_indices,
    int num_valid_nodes, float* d_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    size_t i = d_valid_indices[idx];
    atomicAdd(d_sum, d_graph[i].f_star);
}

// Kernel: apply tug based on difference from mean f_star
__global__ void apply_proportional_tugging_kernel(Node* d_graph, size_t* d_valid_indices,
    int num_valid_nodes, float diff_step, float mean_f_star) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    float diff = mean_f_star - node.f_star;
    node.f_star -= diff_step * diff;
    // node.f_star = fmaxf(fminf(node.f_star, 360.0f*200), -360.0f*200);
}

// Host wrapper for tugging solver with visualization
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
) {
    std::cout << "Running tugging solver..." << std::endl;
    std::vector<Node> graph_copy = graph;
    auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph_copy, 0.1f);
    size_t num_nodes = graph.size();
    size_t num_valid = valid_indices.size();
    // GPU memory
    size_t* d_valid_indices = nullptr;
    cudaMalloc(&d_valid_indices, num_valid * sizeof(size_t));
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid * sizeof(size_t), cudaMemcpyHostToDevice);
    Node* d_graph = nullptr;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    // Copy edges and sides
    Edge* d_all_edges = nullptr;
    float* d_all_sides = nullptr;
    copy_graph_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, true, true);
    // Kernel launch parameters
    int threads = 256;
    int blocks = (int)((num_valid + threads - 1) / threads);
    // Main iteration loop
    for (int iter = 1; iter <= num_iterations; ++iter) {
        // 1) Standard f_star update step
        update_nodes_kernel_tugging_f_star<<<blocks, threads>>>(d_graph, d_valid_indices,
            o, spring_constant, step_sigma, (int)num_valid, 200, i_round);
        // update_nodes_kernel_f_star<<<blocks, threads>>>(d_graph, d_valid_indices,
        //     o, spring_constant, step_sigma, (int)num_valid, 200, i_round);
        cudaDeviceSynchronize();
        // 2) Proportional tugging based on difference from mean f_star
        // Compute sum of f_star across valid nodes
        float* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(float));
        cudaMemset(d_sum, 0, sizeof(float));
        compute_f_star_sum_kernel<<<blocks, threads>>>(d_graph, d_valid_indices, (int)num_valid, d_sum);
        cudaDeviceSynchronize();
        // Copy sum to host and compute mean
        float sum_f_star = 0.0f;
        cudaMemcpy(&sum_f_star, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_sum);
        float mean_f_star = sum_f_star / (float)num_valid;
        mean_f_star = (min_percentile + max_percentile) / 2.0f;
        // Apply proportional tugging
        apply_proportional_tugging_kernel<<<blocks, threads>>>(d_graph, d_valid_indices, (int)num_valid, diff_step, mean_f_star);
        cudaDeviceSynchronize();
        // 3) Finalize f_tilde from updated f_star
        update_f_star_kernel<<<blocks, threads>>>(d_graph, d_valid_indices, (int)num_valid, 0.0f, 0, false);
        cudaDeviceSynchronize();
        // Visualization
        int step_size = 120;
        if (iter % step_size == 0) {
            std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line
        }
        if (visualize && iter % step_size == 0) {
            // copy host state
            auto [h_edges_vis, h_sides_vis] = copy_graph_from_gpu(
                graph_copy.data(), d_graph, num_nodes,
                &d_all_edges, &d_all_sides, false, false);
            std::ostringstream filename;
            filename << "python_angles/tugging_" << i_round << "_"
                     << std::setw(5) << std::setfill('0') << iter << ".png";
            plot_nodes(graph_copy, filename.str());
            if (h_edges_vis) delete[] h_edges_vis;
            if (h_sides_vis) delete[] h_sides_vis;
            auto [min_percentile_, max_percentile_] = min_max_percentile_f_star(graph_copy, 0.1f);
            min_percentile = min_percentile_;
            max_percentile = max_percentile_;
        }
    }
    std::cout << std::endl;

    // Copy back to host
    auto pair_host = copy_graph_from_gpu(graph_copy.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
    Edge* h_edges_out = pair_host.first;
    float* h_sides_out = pair_host.second;
    // Update original graph
    for (size_t i = 0; i < num_nodes; ++i) {
        graph[i].f_star = graph_copy[i].f_star;
        graph[i].f_tilde = graph_copy[i].f_tilde;
        graph[i].happiness_old = graph_copy[i].happiness_old;
        graph[i].happiness = graph_copy[i].happiness;
    }
    // Cleanup GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_valid_indices);
    cudaFree(d_graph);
    // Cleanup host buffers
    if (h_edges_out) delete[] h_edges_out;
    if (h_sides_out) delete[] h_sides_out;
    return graph;
}