// flattening_solver.cu
#include "flattening_solver.h"
#include "mean_solver.h"    // for plot_nodes
#include "node_structs.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Copy edges from CPU to GPU (batched allocation)
inline void copy_edges_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr) {
    size_t total_edges = 0;
    for (size_t i = 0; i < num_nodes; ++i) total_edges += h_graph[i].num_edges;
    // Allocate contiguous device buffer for all edges
    Edge* d_all_edges;
    cudaMalloc(&d_all_edges, total_edges * sizeof(Edge));
    // Gather and copy all edges via a host buffer
    Edge* h_all_edges = new Edge[total_edges];
    size_t offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        int ne = h_graph[i].num_edges;
        if (ne > 0) {
            memcpy(h_all_edges + offset, h_graph[i].edges, ne * sizeof(Edge));
            offset += ne;
        }
    }
    cudaMemcpy(d_all_edges, h_all_edges, total_edges * sizeof(Edge), cudaMemcpyHostToDevice);
    delete[] h_all_edges;
    // Update device graph edge pointers
    offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        int ne = h_graph[i].num_edges;
        if (ne > 0) {
            Edge* ptr = d_all_edges + offset;
            cudaMemcpyAsync(&d_graph[i].edges, &ptr, sizeof(Edge*), cudaMemcpyHostToDevice);
            offset += ne;
        }
    }
    cudaDeviceSynchronize();
    *d_all_edges_ptr = d_all_edges;
}

// Kernel: apply temporaries to f_init (z) and f_star from f_tilde
__global__ void flattening_apply_kernel(
    Node* d_graph,
    size_t* d_valid,
    int num_valid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid) return;
    size_t i = d_valid[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;
    // Commit computed values
    node.f_init = node.z;
    node.f_star = node.f_tilde;
}

// Free device edge buffer
inline void free_edges_from_gpu(Edge* d_all_edges) {
    if (d_all_edges) cudaFree(d_all_edges);
}
// Kernel: compute sum and count of f_init per zero range group
__global__ void compute_zero_means_kernel(
    Node* d_graph,
    size_t* d_valid,
    int num_valid,
    const float2* ranges,
    int num_ranges,
    float* sums,
    int* counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid) return;
    size_t i = d_valid[idx];
    const Node& node = d_graph[i];
    float f = node.f_init;
    float z = node.wnr_side;
    float w = node.wnr_side_old;
    for (int j = 0; j < num_ranges; ++j) {
        float min_a = ranges[j].x;
        float max_a = ranges[j].y;
        if (w > min_a && w < max_a) {
            atomicAdd(&sums[j], f-z);
            atomicAdd(&counts[j], 1);
            break;
        }
    }
}
// Kernel: compute means from sums and counts
__global__ void compute_means_kernel(
    const float* sums,
    const int* counts,
    float* means,
    int num_ranges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ranges) return;
    int cnt = counts[idx];
    float denom = cnt > 0 ? (float)cnt : 1.0f;
    means[idx] = sums[idx] / denom;
}

// Kernel: count total edges and active edges
__global__ void count_edges_kernel(
    Node* d_graph,
    size_t* d_valid,
    int num_valid,
    int* d_total_edges,
    int* d_active_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid) return;
    
    size_t i = d_valid[idx];
    const Node& node = d_graph[i];
    
    // Count all edges from this node
    for (int e = 0; e < node.num_edges; ++e) {
        const Edge& edge = node.edges[e];
        const Node& target_node = d_graph[edge.target_node];
        
        // Count total edges (avoid double counting by only counting edges where i < target)
        if (i < edge.target_node) {
            atomicAdd(d_total_edges, 1);
        }
        
        // Count active edges: both nodes not deleted, not fixed, edge not temporary
        if (i < edge.target_node && // avoid double counting
            !node.deleted && !target_node.deleted && // both nodes not deleted
            !node.fixed && !target_node.fixed &&     // both nodes not fixed
            !edge.temporary) {                        // edge not temporary
            atomicAdd(d_active_edges, 1);
        }
    }
}

// Kernel: updates f_init (z) and f_star coordinates to match edge lengths
__global__ void flattening_update_kernel(
    Node* d_graph,
    size_t* d_valid,
    int num_valid,
    float z_min,
    float z_max,
    float a_min,
    float a_max,
    float tug_step,
    const float2* zero_ranges,
    int num_zero_ranges,
    const float* zero_means,
    bool enable_spring_push_multiplier
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid) return;
    size_t i = d_valid[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;
    if (node.fixed) return;
    float acc_z = 0.0f, acc_s = 0.0f, sum_w = 0.0f;
    float angle_acc_z = 0.0f, angle_acc_s = 0.0f, angle_sum_w = 0.0f;
    
    // Spring forces in 2D
    for (int e = 0; e < node.num_edges; ++e) {
        const Edge& edge = node.edges[e];
        if (edge.temporary) continue;
        const Node& nb = d_graph[edge.target_node];
        if (nb.deleted) continue;
        float dz = nb.f_init - node.f_init;
        float ds = nb.f_star - node.f_star;
        // Accumulate initial-z difference tug
        float dist = sqrtf(dz*dz + ds*ds);
        if (dist == 0.0f) {
            if (acc_z == 0.0f && acc_s == 0.0f) {
                // Random small offset to avoid stagnation
                float seed = float(i) * 12.9898f + float(idx) * 78.233f + edge.target_node * 3.5453f + 400.1f * nb.f_init * node.f_star;
                float rnd1 = sinf(seed) * 43758.5453123f;
                float rnd2 = sinf(seed + 1.0f) * 5.5453123f;
                rnd1 = rnd1 - floorf(rnd1);
                rnd2 = rnd2 - floorf(rnd2);
                dz = rnd1;
                ds = rnd2;
                dist = sqrtf(dz*dz + ds*ds);
            }
        }
        // Determine direction: if the current distance is greater than desired, move closer (+), else move apart (-)
        float sign = (dist > edge.k) ? 1.0f : -1.0f;
        dist = fmaxf(dist, 1e-3f);  // Avoid division by zero

        // acc_z += dz * sign;
        // acc_s += ds * sign;
        // sum_w += 1.0f;
        
        float error = fabsf(dist - edge.k);
        error = fminf(error, 0.5 * dist);  // Clamp error to avoid overflow
        error = fmaxf(error, 1.0f);  // Avoid slow final convergence
        
        // Add linear scaling when distance is less than 0.5 of desired distance (only if enabled)
        float force_multiplier = 1.0f;
        float activation_ratio = 0.75f;
        float activation_ratio_push_in = 2.0f;
        if (enable_spring_push_multiplier && dist < activation_ratio * edge.k) {
            // Linear scaling: 1x at 0.5*edge.k -> 5x at 0.0
            float ratio = dist / (activation_ratio * edge.k);  // ratio goes from 1.0 at 0.5*edge.k to 0.0 at dist=0
            force_multiplier = 1.0f + 10.0f * (1.0f - ratio);  // 1 + 4*(1-ratio) = 5 at ratio=0, 1 at ratio=1
        }

        // if (enable_spring_push_multiplier && dist > activation_ratio_push_in * edge.k) {
        //     // Linear scaling: 1x at 0.5*edge.k -> 5x at 0.0
        //     float ratio = dist / (activation_ratio_push_in * edge.k);  // ratio goes from 1.0 at activation_ratio_push_in*edge.k to 5.0 at dist=
        //     force_multiplier = fminf(2.0f, 1.0f + 0.1f*ratio);
        // }
        
        float ux = dz / dist;
        float us = ds / dist;
        acc_z += error * ux * sign * force_multiplier;
        acc_s += error * us * sign * force_multiplier;
        sum_w += 1.0f;
    }
    
    // Angle-based forces: sample two random neighbors and try to preserve angles
    if (node.num_edges >= 2) {
        // Use deterministic "random" selection based on node index and iteration
        unsigned int seed = i * 2654435761u + blockIdx.x * 16807u;
        
        // Select two different neighbors
        int n1_idx = seed % node.num_edges;
        int n2_idx = (seed / node.num_edges + 1) % node.num_edges;
        if (n1_idx == n2_idx && node.num_edges > 1) {
            n2_idx = (n2_idx + 1) % node.num_edges;
        }
        
        const Edge& edge1 = node.edges[n1_idx];
        const Edge& edge2 = node.edges[n2_idx];
        
        if (!edge1.temporary && !edge2.temporary && 
            !d_graph[edge1.target_node].deleted && !d_graph[edge2.target_node].deleted) {
            
            const Node& nb1 = d_graph[edge1.target_node];  // This will be the angle vertex
            const Node& nb2 = d_graph[edge2.target_node];
            
            // Calculate 3D angle at nb1: (node - nb1) and (nb2 - nb1)
            float dx1_3d = node.coord_x - nb1.coord_x;
            float dy1_3d = node.coord_y - nb1.coord_y;
            float dz1_3d = node.coord_z - nb1.coord_z;
            
            float dx2_3d = nb2.coord_x - nb1.coord_x;
            float dy2_3d = nb2.coord_y - nb1.coord_y;
            float dz2_3d = nb2.coord_z - nb1.coord_z;
            
            // Normalize 3D vectors
            float len1_3d = sqrtf(dx1_3d*dx1_3d + dy1_3d*dy1_3d + dz1_3d*dz1_3d);
            float len2_3d = sqrtf(dx2_3d*dx2_3d + dy2_3d*dy2_3d + dz2_3d*dz2_3d);
            
            if (len1_3d > 1e-6f && len2_3d > 1e-6f) {
                dx1_3d /= len1_3d; dy1_3d /= len1_3d; dz1_3d /= len1_3d;
                dx2_3d /= len2_3d; dy2_3d /= len2_3d; dz2_3d /= len2_3d;
                
                // 3D angle cosine
                float cos_3d = dx1_3d*dx2_3d + dy1_3d*dy2_3d + dz1_3d*dz2_3d;
                cos_3d = fmaxf(-1.0f, fminf(1.0f, cos_3d));  // Clamp to [-1, 1]
                float angle_3d = acosf(fabsf(cos_3d));  // Always positive angle
                
                // Calculate 2D angle at nb1: (node - nb1) and (nb2 - nb1) in UV space
                float du1_2d = node.f_star - nb1.f_star;
                float dv1_2d = node.f_init - nb1.f_init;
                float du2_2d = nb2.f_star - nb1.f_star;
                float dv2_2d = nb2.f_init - nb1.f_init;
                
                float len1_2d = sqrtf(du1_2d*du1_2d + dv1_2d*dv1_2d);
                float len2_2d = sqrtf(du2_2d*du2_2d + dv2_2d*dv2_2d);
                
                if (len1_2d > 1e-6f && len2_2d > 1e-6f) {
                    du1_2d /= len1_2d; dv1_2d /= len1_2d;
                    du2_2d /= len2_2d; dv2_2d /= len2_2d;
                    
                    // 2D angle cosine
                    float cos_2d = du1_2d*du2_2d + dv1_2d*dv2_2d;
                    cos_2d = fmaxf(-1.0f, fminf(1.0f, cos_2d));
                    float angle_2d = acosf(fabsf(cos_2d));  // Always positive angle
                    
                    // Angle difference (how much 2D angle deviates from 3D)
                    float angle_error = angle_3d - angle_2d;
                    
                    // Calculate gradient to move node in direction that reduces angle error
                    // We want to move the node to change angle_2d towards angle_3d
                    
                    // For small angle changes, the gradient direction is approximately
                    // perpendicular to the current vector from nb1 to node
                    float perp_u = -dv1_2d;  // Perpendicular to normalized vector (du1_2d, dv1_2d)
                    float perp_v = du1_2d;
                    
                    // Determine the sign: if angle_2d < angle_3d, we need to increase angle_2d
                    float sign = (angle_error > 0.0f) ? 1.0f : -1.0f;
                    
                    // Scale by angle error and add to accumulator
                    float angle_weight = 1.5f;  // Weight for angle preservation vs distance preservation
                    float error_magnitude = fabsf(angle_error);
                    
                    angle_acc_s += angle_weight * error_magnitude * sign * perp_u;
                    angle_acc_z += angle_weight * error_magnitude * sign * perp_v;
                    angle_sum_w += angle_weight;
                }
            }
        }
    }
    
    if (sum_w == 0.0f) {
        sum_w = 1.0f;
    }
    if (angle_sum_w == 0.0f) {
        angle_sum_w = 1.0f;
    }
        
    float step_z = acc_z / sum_w + angle_acc_z / angle_sum_w;
    float step_s = acc_s / sum_w + angle_acc_s / angle_sum_w;
    
    // Additional z tug (only if a non-zero interval is specified)
    if (z_max > z_min) {
        if (node.wnr_side > z_max)      step_z += tug_step;
        else if (node.wnr_side < z_min) step_z -= tug_step;
    }
    // Additional angle tug (only if a non-zero interval is specified)
    if (a_max > a_min) {
        if (node.wnr_side_old > a_max)      step_s += tug_step;
        else if (node.wnr_side_old < a_min) step_s -= tug_step;
    }
    // Additional zero-range-based tug to drive mean to zero
    if (num_zero_ranges > 0) {
        float w_old = node.wnr_side_old;
        for (int j = 0; j < num_zero_ranges; ++j) {
            float min_a = zero_ranges[j].x;
            float max_a = zero_ranges[j].y;
            if (w_old > min_a && w_old < max_a) {
                float mean = zero_means[j];
                if (tug_step > 0) {
                    step_z -= mean;
                }
                else {
                    if (mean > 0) step_z -= fabsf(tug_step);
                    else if (mean < 0) step_z += fabsf(tug_step);
                }
                break;
            }
        }
    }
    // Store updates in temporaries: use node.z for new f_init, node.f_tilde for new f_star
    float lr = 1.5f;
    node.z       = node.f_init + lr * step_z;
    node.f_tilde = node.f_star + lr * step_s;

    // Clamp to min/max
    if (node.z < -1e9f) node.z = -1e9f;
    if (node.z > 1e9f) node.z = 1e9f;
    if (node.f_tilde < -1e9f) node.f_tilde = -1e9f;
    if (node.f_tilde > 1e9f) node.f_tilde = 1e9f;
}

std::vector<Node> run_solver_flattening(
    std::vector<Node>& graph,
    int num_iterations,
    const std::vector<size_t>& valid_indices,
    Edge** /*h_all_edges*/,
    float** /*h_all_sides*/,
    float z_tug_min,
    float z_tug_max,
    float angle_tug_min,
    float angle_tug_max,
    float tug_step,
    float init_z_tug,
    const std::vector<std::pair<float, float>>& zero_ranges,
    bool visualize,
    bool enable_spring_push_multiplier
) {
    std::vector<Node> graph_copy = graph;
    size_t N = graph_copy.size();
    size_t M = valid_indices.size();
    // Allocate GPU memory
    Node* d_graph;
    cudaMalloc(&d_graph, N * sizeof(Node));
    cudaMemcpy(d_graph, graph_copy.data(), N * sizeof(Node), cudaMemcpyHostToDevice);
    size_t* d_valid;
    cudaMalloc(&d_valid, M * sizeof(size_t));
    cudaMemcpy(d_valid, valid_indices.data(), M * sizeof(size_t), cudaMemcpyHostToDevice);
    // Copy edges
    Edge* d_all_edges = nullptr;
    copy_edges_to_gpu(graph_copy.data(), d_graph, N, &d_all_edges);
    // Prepare initial-z array on device for optional initial-z tug
    float* d_init_z = nullptr;
    std::vector<float> host_init_z(N);
    for (size_t i = 0; i < N; ++i) host_init_z[i] = graph_copy[i].z;
    cudaMalloc(&d_init_z, N * sizeof(float));
    cudaMemcpy(d_init_z, host_init_z.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    // Prepare zero_ranges on device if provided
    int num_ranges = static_cast<int>(zero_ranges.size());
    float2* d_ranges = nullptr;
    float* d_sums = nullptr;
    int* d_counts = nullptr;
    float* d_means = nullptr;
    if (num_ranges > 0) {
        // Copy zero_ranges to device
        std::vector<float2> host_ranges(num_ranges);
        for (int j = 0; j < num_ranges; ++j) {
            host_ranges[j].x = zero_ranges[j].first;
            host_ranges[j].y = zero_ranges[j].second;
        }
        cudaMalloc(&d_ranges, num_ranges * sizeof(float2));
        cudaMemcpy(d_ranges, host_ranges.data(), num_ranges * sizeof(float2), cudaMemcpyHostToDevice);
        // Allocate buffers for sums, counts, and means
        cudaMalloc(&d_sums, num_ranges * sizeof(float));
        cudaMalloc(&d_counts, num_ranges * sizeof(int));
        cudaMalloc(&d_means, num_ranges * sizeof(float));
    }
    // Allocate device memory for edge counting
    int* d_total_edges;
    int* d_active_edges;
    cudaMalloc(&d_total_edges, sizeof(int));
    cudaMalloc(&d_active_edges, sizeof(int));
    
    // Initialize counters to zero
    cudaMemset(d_total_edges, 0, sizeof(int));
    cudaMemset(d_active_edges, 0, sizeof(int));
    
    // Launch parameters
    int TPB = 256;
    int blocks = (M + TPB - 1) / TPB;
    
    // Count edges before starting solver
    count_edges_kernel<<<blocks, TPB>>>(d_graph, d_valid, (int)M, d_total_edges, d_active_edges);
    cudaDeviceSynchronize();
    
    // Copy results back to host and print
    int h_total_edges = 0;
    int h_active_edges = 0;
    cudaMemcpy(&h_total_edges, d_total_edges, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_active_edges, d_active_edges, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Edge count: " << h_total_edges << " total edges, " << h_active_edges 
              << " active edges (both nodes unfixed, undeleted, edge not temporary)" << std::endl;
    
    // Launch loop
    const int step_size = 1000;
    for (int it = 1; it <= num_iterations; ++it) {
        // Reset and compute zero_ranges means if any
        if (num_ranges > 0) {
            cudaMemset(d_sums, 0, num_ranges * sizeof(float));
            cudaMemset(d_counts, 0, num_ranges * sizeof(int));
            // Compute sums and counts for each range
            compute_zero_means_kernel<<<blocks, TPB>>>(d_graph, d_valid, (int)M, d_ranges, num_ranges, d_sums, d_counts);
            cudaDeviceSynchronize();
            // Compute means per range
            int range_blocks = (num_ranges + TPB - 1) / TPB;
            compute_means_kernel<<<range_blocks, TPB>>>(d_sums, d_counts, d_means, num_ranges);
            cudaDeviceSynchronize();
        }
        // Compute new positions into temporaries
        flattening_update_kernel<<<blocks, TPB>>>(
            d_graph, d_valid, (int)M,
            z_tug_min, z_tug_max,
            angle_tug_min, angle_tug_max,
            tug_step,
            d_ranges, num_ranges, d_means,
            enable_spring_push_multiplier
        );
        cudaDeviceSynchronize();
        // Apply: commit temporaries to f_init and f_star
        flattening_apply_kernel<<<blocks, TPB>>>(d_graph, d_valid, (int)M);
        cudaDeviceSynchronize();
        if (visualize && it % step_size == 0) {
            // Print
            std::cout << "\rIteration: " << it << std::flush;  // Updates the same line
            // Copy back for visualization
            Node* h_buf = new Node[N];
            cudaMemcpy(h_buf, d_graph, N * sizeof(Node), cudaMemcpyDeviceToHost);
            std::vector<Node> graph_copy_(h_buf, h_buf + N);
            delete[] h_buf;
            // Scatter plot f_star vs f_init
            plot_nodes(graph_copy_, "");
        }
    }
    std::cout << std::endl;
    // Copy back results
    cudaMemcpy(graph_copy.data(), d_graph, N * sizeof(Node), cudaMemcpyDeviceToHost);
    // take over f star, f init to original graph
    for (size_t i = 0; i < N; ++i) {
        graph[i].f_star = graph_copy[i].f_star;
        graph[i].f_tilde = graph_copy[i].f_tilde;
        graph[i].z = graph_copy[i].z;
        graph[i].f_init = graph_copy[i].f_init;
    }
    // Cleanup zero_ranges device memory
    if (num_ranges > 0) {
        cudaFree(d_ranges);
        cudaFree(d_sums);
        cudaFree(d_counts);
        cudaFree(d_means);
    }
    // Cleanup initial-z device memory
    if (d_init_z) cudaFree(d_init_z);
    // Cleanup edges and graph memory
    free_edges_from_gpu(d_all_edges);
    cudaFree(d_graph);
    cudaFree(d_valid);
    return graph;
}