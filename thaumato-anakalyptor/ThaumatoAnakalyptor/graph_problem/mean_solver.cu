/*
Julian Schilliger 2025 ThaumatoAnakalyptor
*/
#include "mean_solver.h"
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

// Kernel to update nodes on the GPU with momentum
inline __global__ void update_nodes_kernel_f_star_step(Node* d_graph, size_t* d_valid_indices, size_t seed_node, float seed_tilde, float spring_constant, float other_block_factor, float lr, float error_cutoff, int num_valid_nodes, int estimated_windings, float* d_global_sum_compressed, float* d_global_sum_decompressed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;
    if (i == seed_node && seed_node != 0) {
        return;
    }

    float same_block_factor = 1.0f;
    if (other_block_factor > 1.0f) {
        same_block_factor = 1.0f / other_block_factor;
        other_block_factor = 1.0f;
    }

    // Calculate the f_star update using mean values
    float node_f_tilde = node.f_tilde;
    float step = 0.0f;
    float count_compressed = 0;
    float count_decompressed = 0;

    Edge* edges = node.edges;
    // Loop over all edges to compute the update step
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        size_t target_node = edge.target_node;
        float target_tilde = seed_tilde;
        float step_edge = lr;
        if (target_node != seed_node || seed_node == 0) {
            target_tilde = d_graph[target_node].f_tilde;
            if (d_graph[target_node].deleted) continue;
            if (node.fixed && d_graph[target_node].fixed && !edge.fixed) {
                continue;
            }
            // if (node.fixed && !d_graph[target_node].fixed) {
            //     step_edge *= 0.10f;
            // }
        }

        float k = spring_constant * edge.k;
        float predicted_winding_angle = target_tilde - k;
        if (predicted_winding_angle < node_f_tilde) {
            step_edge *= -1.0f;
        }

        // if (fabsf(predicted_winding_angle - node_f_tilde) < 0.1f) {
        //     continue;
        // }

        // Adjust step_edge based on edge properties
        if (edge.fixed) {
            step_edge *= 2.0f;
        }
        else if (edge.same_block) {
            step_edge *= same_block_factor;
        }
        else {
            step_edge *= other_block_factor;
        }

        float error_k = node_f_tilde - predicted_winding_angle;
        if (!edge.same_block && std::abs(error_k) > 0.750f) {
            step_edge *= 1.2f;
        }
        // float k_factor = 1.0f + 0.1f * (0.75f / fmaxf(0.1f, fminf(0.75f, fabsf(k))) - 1.0f);
        // step_edge *= k_factor;
        // // else 
        // if (edge.same_block && std::abs(error_k) > 180.0f) {
        //     step_edge *= 1.50f;
        // }
        if (k > 0.0f && error_k / k >= 1.0f) {
            count_decompressed += std::abs(error_k);
        }
        else {
            count_compressed += std::abs(error_k);
        }
        
        if (!node.fixed && !edge.fixed && error_cutoff >= 0.0f) {
            if (error_cutoff > 0.0f) {
                if (std::abs(error_k) > error_cutoff) {
                    continue;
                }
            }
            if ((target_tilde - node_f_tilde) * k < 0.0f && edge.same_block) { // wrong side
                if (std::abs(error_k) > 180.0f) {
                    continue;
                }
            }
        }
        step += step_edge;
    }

    // Add momentum: update the momentum field and then update f_star.
    const float momentum_coef = 0.999f; // momentum coefficient (can be tuned)
    node.f_star_momentum = momentum_coef * node.f_star_momentum + step;
    node.f_star += node.f_star_momentum;

    // Clip f_star to the allowed range [ - 4 * 360 * estimated_windings, 4 * 360 * estimated_windings ]
    float winding_max = 4 * 360 * estimated_windings;
    node.f_star = fmaxf(-winding_max, fminf(winding_max, node.f_star));

    // Update global compression/decompression sums
    if (count_compressed > 0.0f) {
        atomicAdd(d_global_sum_compressed, count_compressed);
    }
    if (count_decompressed > 0.0f) {
        atomicAdd(d_global_sum_decompressed, count_decompressed);
    }
}

// Kernel to update f_star solver fields on the GPU
inline __global__ void update_f_star_kernel_step(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, float scale_compression) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;
    
    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;
    
    if (node.fixed) {
        node.f_star = node.f_tilde;
    }
    else {
        // Update f_tilde with the computed f_star
        node.f_tilde = node.f_star * scale_compression;
    }
    // Update f_tilde with the computed f_star
    // node.f_tilde = node.f_star * scale_compression;
}

// Copy edges from CPU to GPU with batched allocation
inline void copy_edges_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, bool copy_edges = true) {
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
inline Edge* copy_edges_from_gpu(Node* h_graph, size_t num_nodes, Edge** d_all_edges_ptr) {
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
inline void free_edges_from_gpu(Edge* d_all_edges) {
    // Free existing edges on the GPU
    if (d_all_edges != nullptr) {
        cudaFree(d_all_edges);
        d_all_edges = nullptr;
    }
}

inline void copy_sides_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, float** d_all_sides_ptr, bool copy_sides = true) {
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
inline float* copy_sides_from_gpu(Node* h_graph, size_t num_nodes, float** d_all_sides_ptr) {
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
inline void free_sides_from_gpu(float* d_all_sides) {
    // Free existing sides on the GPU
    if (d_all_sides != nullptr) {
        cudaFree(d_all_sides);
        d_all_sides = nullptr;
    }
}

// copy complete graph from cpu to gpu
inline void copy_graph_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, float** d_all_sides_ptr, bool copy_edges = true, bool copy_sides = true) {
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
inline std::pair<Edge*, float*> copy_graph_from_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr, float** d_all_sides_ptr, bool copy_edges = true, bool copy_sides = true) {
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

inline std::pair<float, float> min_max_percentile_f_star(const std::vector<Node>& graph, float percentile, bool use_gt = false) {
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

inline void plot_nodes(const std::vector<Node>& graph, const std::string& filename) {
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

std::vector<Node> run_solver_f_star_with_labels(std::vector<Node>& graph, int num_iterations, size_t seed_node, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, float spring_constant, float other_block_factor, float lr, float error_cutoff, bool visualize) {
    std::vector<Node> graph_copy = graph;
    std::cout << "Visualize: " << visualize << std::endl;
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
    
    std::cout << "Solving on GPU... learning ratre: " << lr << " error cutoff: " << error_cutoff << std::endl;
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;
    float seed_tilde = graph[seed_node].f_tilde;

    // Run the iterations
    for (int iter = 1; iter < num_iterations; iter++) {        
        float* d_global_sum_compressed;
        float* d_global_sum_decompressed;
        cudaMalloc(&d_global_sum_compressed, sizeof(float));
        cudaMalloc(&d_global_sum_decompressed, sizeof(float));
        cudaMemset(d_global_sum_compressed, 0, sizeof(float));
        cudaMemset(d_global_sum_decompressed, 0, sizeof(float));

        // Launch the kernel to update nodes
        update_nodes_kernel_f_star_step<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, seed_node, seed_tilde, spring_constant, other_block_factor, lr, error_cutoff, num_valid_nodes, 200, d_global_sum_compressed, d_global_sum_decompressed);
        float h_global_sum_compressed = 0;
        float h_global_sum_decompressed = 0;
        cudaMemcpy(&h_global_sum_compressed, d_global_sum_compressed, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_global_sum_decompressed, d_global_sum_decompressed, sizeof(float), cudaMemcpyDeviceToHost);
        float scale_compression = 1.0f + 0.1f* (std::sqrt(h_global_sum_compressed / h_global_sum_decompressed) - 1.0f);
        scale_compression = 1.0f;
        // std::cout << "Compression scale: " << scale_compression << std::endl;
        
        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution
        
        // Launch the kernel to update f_tilde with f_star
        update_f_star_kernel_step<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes, scale_compression);
        
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
            auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph_copy.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
            if (visualize) {
                plot_nodes(graph_copy, "");
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

    auto [h_all_edges_, h_all_sides_] = copy_graph_from_gpu(graph_copy.data(), d_graph, num_nodes, &d_all_edges, &d_all_sides, false, false);
    // take over f star to original graph
    for (size_t i = 0; i < num_nodes; ++i) {
        graph[i].f_star = graph_copy[i].f_star;
        graph[i].f_tilde = graph_copy[i].f_tilde;
        // graph[i].f_star_momentum = graph_copy[i].f_star_momentum;
    }

    // Free GPU memory
    free_edges_from_gpu(d_all_edges);
    free_sides_from_gpu(d_all_sides);
    cudaFree(d_graph);

    return graph;
}
