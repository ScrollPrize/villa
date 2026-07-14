/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <stack>
#include <cmath>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <argparse.hpp>
#include <random>
#include <queue>
#include <numeric>

namespace fs = std::filesystem;

struct Edge {
    unsigned int target_node;
    float certainty;
    float certainty_factor = 1.0f;
    float certainty_factored;
    float k;
    bool same_block;
    bool fixed = false;
};

struct Node {
    float z;
    float f_init;
    float f_tilde;
    float f_star;
    bool gt;
    float gt_f_star;
    bool deleted = false;
    bool fixed = false;
    float fold = 0.3f;
    std::vector<Edge> edges;
};

std::pair<std::vector<Node>, float> load_graph_from_binary(const std::string &file_name, bool clip_z = false, float z_min = 0.0f, float z_max = 0.0f, float same_winding_factor = 1.0f, bool fix_same_block_edges = false) {
    std::vector<Node> graph;
    std::ifstream infile(file_name, std::ios::binary);

    if (!infile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return std::make_pair(graph, 0.0f);
    }

    // Read the number of nodes
    unsigned int num_nodes;
    infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(unsigned int));
    std::cout << "Number of nodes in graph: " << num_nodes << std::endl;

    // Prepare the graph with empty nodes
    graph.resize(num_nodes);

    // Read each node's winding angle and store it as int16_t
    for (unsigned int i = 0; i < num_nodes; ++i) {
        infile.read(reinterpret_cast<char*>(&graph[i].z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].gt), sizeof(bool));
        infile.read(reinterpret_cast<char*>(&graph[i].gt_f_star), sizeof(float));
        graph[i].f_tilde = graph[i].f_init;
        graph[i].f_star = graph[i].f_init;
    }
    std::cout << "Nodes loaded successfully." << std::endl;
    // Read the adjacency list
    int count_same_block_edges = 0;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        unsigned int node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(unsigned int));

        unsigned int num_edges;
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(unsigned int));

        for (unsigned int j = 0; j < num_edges; ++j) {
            Edge edge;
            infile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(unsigned int));
            infile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(float));
            // clip certainty
            // if (edge.certainty < 0.001f) {
            //     std::cout << "Certainty below 0.001: " << edge.certainty << std::endl;
            //     edge.certainty = 0.001f;
            // }
            // add certainty factored
            edge.certainty_factored = edge.certainty;
            infile.read(reinterpret_cast<char*>(&edge.k), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(bool));
            if (clip_z) {
                if (graph[edge.target_node].z < z_min || graph[edge.target_node].z > z_max) {
                    graph[edge.target_node].deleted = true;
                    continue;
                }
                if (graph[node_id].z < z_min || graph[node_id].z > z_max) {
                    graph[node_id].deleted = true;
                    continue;
                }
            }
            if (fix_same_block_edges) {
                if (std::abs(edge.k) > 180) {
                    edge.same_block = true;
                }
            }
            if (edge.same_block) { // no same subvolume edges
                edge.certainty *= same_winding_factor;
                edge.certainty_factored *= same_winding_factor;
                count_same_block_edges++;
                // continue;
                if (std::abs(edge.k) > 450) {
                    std::cout << "Edge with k > 450: " << edge.k << std::endl;
                }
                if (std::abs(edge.k) < 180) {
                    std::cout << "Edge with k < 180: " << edge.k << std::endl;
                }
            }
            graph[node_id].edges.push_back(edge);
        }
    }

    std::cout << "Same block edges: " << count_same_block_edges << std::endl;

    std::cout << "Graph loaded successfully." << std::endl;

    // Find largest certainty (scaled back to float for display)
    float max_certainty = 0;
    for (const auto& node : graph) {
        for (const auto& edge : node.edges) {
            if (edge.certainty > max_certainty) {
                max_certainty = edge.certainty;
            }
        }
    }

    std::cout << "Max Certainty: " << max_certainty << std::endl;

    // all edge certainties to max_certainty
    // for (auto& node : graph) {
    //     for (auto& edge : node.edges) {
    //         edge.certainty = max_certainty;
    //     }
    // }

    infile.close();
    //  return graph and max certainty
    return std::make_pair(graph, max_certainty);
}

void save_graph_to_binary(const std::string& file_name, const std::vector<Node>& graph) {
    std::ofstream outfile(file_name, std::ios::binary);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Write the number of nodes
    unsigned int num_nodes = graph.size();
    outfile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(unsigned int));

    // Write each node's f_star and deleted status
    for (const auto& node : graph) {
        outfile.write(reinterpret_cast<const char*>(&node.f_star), sizeof(float));
        outfile.write(reinterpret_cast<const char*>(&node.deleted), sizeof(bool));
    }

    outfile.close();
}

void set_z_range_graph(std::vector<Node>& graph, float z_min, float z_max) {
    for (auto& node : graph) {
        if (node.z < z_min || node.z > z_max) {
            node.deleted = true;
        }
    }
}

float closest_valid_winding_angle(float f_init, float f_target) {
    int x = static_cast<int>(std::round((f_target - f_init) / 360.0f));
    float result = f_init + x * 360.0f;
    if (std::abs(f_target - result) > 10.0f) {
        std::cout << "Difference between f_target and result: " << std::abs(f_target - result) << std::endl;
    }
    if (std::abs(x - (f_target - f_init) / 360.0f) > 1e-4) {
        std::cout << "Difference between x and (f_target - f_init) / 360.0f: " << std::abs(x - (f_target - f_init) / 360.0f) << std::endl;
        std::cout << "f_init: " << f_init << ", f_target: " << f_target << ", x: " << x << ", result: " << result << std::endl;
    }
    return result;
}

void dfs(size_t node_index, const std::vector<Node>& graph, std::vector<bool>& visited, std::vector<size_t>& component) {
    std::stack<size_t> stack;
    stack.push(node_index);
    visited[node_index] = true;

    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();
        component.push_back(current);

        for (const auto& edge : graph[current].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (!visited[edge.target_node]) {
                visited[edge.target_node] = true;
                stack.push(edge.target_node);
            }
        }
    }
}

void find_largest_connected_component(std::vector<Node>& graph) {
    size_t num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);
    std::vector<size_t> largest_component;

    size_t initial_non_deleted = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            initial_non_deleted++;
        }
        if (!visited[i] && !graph[i].deleted) {
            std::vector<size_t> current_component;
            dfs(i, graph, visited, current_component);

            if (current_component.size() > largest_component.size()) {
                largest_component = current_component;
            }
        }
    }

    // Flag nodes not in the largest connected component as deleted
    std::vector<bool> in_largest_component(num_nodes, false);
    for (size_t node_index : largest_component) {
        in_largest_component[node_index] = true;
    }

    size_t remaining_nodes = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!in_largest_component[i]) {
            graph[i].deleted = true;
        }
        if (!graph[i].deleted) {
            remaining_nodes++;
        }
    }
    std::cout << "Remaining nodes: " << remaining_nodes << " out of " << initial_non_deleted << " initial non deleted nodes, of total edge: " << num_nodes << std::endl;
}

void dfs_assign_f_star(int node_index, std::vector<Node>& graph, std::vector<bool>& visited) {
    std::stack<int> stack;
    stack.push(node_index);
    visited[node_index] = true;

    // Initialize f_star with f_init for the starting node
    graph[node_index].f_star = graph[node_index].f_init;

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        for (auto& edge : graph[current].edges) {
            if (!visited[edge.target_node] && !graph[edge.target_node].deleted) {
                visited[edge.target_node] = true;

                // Calculate the f_star for the target node
                graph[edge.target_node].f_star = closest_valid_winding_angle(graph[current].f_init , graph[current].f_tilde + edge.k);

                stack.push(edge.target_node);
            }
        }
    }
}

void assign_winding_angles_dfs(std::vector<Node>& graph) {
    size_t num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);

    // Find a non-deleted node in the largest connected component to start the DFS
    size_t start_node = 0;
    bool found_start_node = false;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            found_start_node = true;
            break;
        }
    }

    if (!found_start_node) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
        return;
    }

    // Perform DFS to assign f_star values
    dfs_assign_f_star(start_node, graph, visited);
}

using EdgeWithCertainty = std::pair<float, int>;  // {certainty, target_node}

void prim_mst_assign_f_star(size_t start_node, std::vector<Node>& graph, float scale) {
    size_t num_nodes = graph.size();
    std::vector<bool> in_mst(num_nodes, false);
    std::vector<float> min_k_delta(num_nodes, std::numeric_limits<float>::max());
    std::vector<size_t> parent(num_nodes, 0);
    std::vector<bool> valid(num_nodes, false);
    std::vector<float> k_values(num_nodes, 0.0);

    // Priority queue to pick the edge with the minimum k delta
    std::priority_queue<EdgeWithCertainty, std::vector<EdgeWithCertainty>, std::greater<EdgeWithCertainty>> pq;

    pq.push({0.0f, start_node});
    min_k_delta[start_node] = 0.0f;

    while (!pq.empty()) {
        size_t u = pq.top().second;
        pq.pop();

        if (in_mst[u]) continue;
        in_mst[u] = true;

        for (const auto& edge : graph[u].edges) {
            if (graph[u].deleted) {
                continue;
            }
            size_t v = edge.target_node;
            if (graph[v].deleted) {
                continue;
            }
            // float k_delta = std::abs((graph[v].f_tilde - graph[u].f_tilde) - edge.k) / std::abs(edge.k); // difference between BP solution and estimated k from the graph
            // float k_delta = std::abs((graph[v].f_tilde - graph[u].f_tilde) - edge.k); // difference between BP solution and estimated k from the graph
            float k_delta = std::abs(scale * (graph[v].f_tilde - graph[u].f_tilde) - edge.k); // difference between BP solution and estimated k from the graph
            if (edge.fixed && edge.certainty > 0.0f) {
                k_delta = -1.0f;
            }
            // float k_delta = std::abs(edge.certainty_factored); // adpated certainty of the edge

            if (!in_mst[v] && k_delta < min_k_delta[v]) {
                min_k_delta[v] = k_delta;
                pq.push({k_delta, v});
                parent[v] = u;
                k_values[v] = edge.k;
                valid[v] = true;
            }
        }
    }

    // Set f_star for the root node (start_node)
    graph[start_node].f_tilde = graph[start_node].f_init;
    graph[start_node].f_star = graph[start_node].f_init;

    // Find for each node the children
    std::vector<std::vector<size_t>> children(num_nodes);
    std::vector<std::vector<float>> children_k_values(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i]) {
            children[parent[i]].push_back(i);
            children_k_values[parent[i]].push_back(k_values[i]);
        }
    }

    // Traverse the MST in a DFS manner to assign f_star values
    std::stack<size_t> stack;
    stack.push(start_node);

    // Sanity Check
    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();

        if (graph[current].deleted) {
            // graph[current].f_star = 0;
            // graph[current].f_tilde = 0;
            continue;
        }

        for (size_t i = 0; i < children[current].size(); ++i) {
            size_t child = children[current][i];
            if (graph[child].deleted) {
                // graph[child].f_star = 0;
                // graph[child].f_tilde = 0;
                continue;
            }
            float k = children_k_values[current][i];
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + k);
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[child].f_star);
            
            graph[child].f_tilde = graph[child].f_star;
            stack.push(child);
        }
    }
}

void assign_winding_angles(std::vector<Node>& graph, float scale) {
    size_t num_nodes = graph.size();
    
    // Find a non-deleted node in the largest connected component to start the MST
    size_t start_node = 0;
    bool found_start_node = false;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            found_start_node = true;
            break;
        }
    }

    if (!found_start_node) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
        return;
    }

    // Perform MST to assign f_star values
    prim_mst_assign_f_star(start_node, graph, scale);

    // check the winding angles on f_star
    for (size_t i = 0; i < num_nodes; ++i) {
        if (graph[i].deleted) {
            continue;
        }
        closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }
}

float certainty_factor(float error) { // certainty factor based on the error. exp (-) as function. 1 for 0 error, 0.1 for 360 error (1 winding off)
    // x = - log (0.1) / 360
    float winding_off_factor = 0.4f;
    float x = - std::log(winding_off_factor) / 360.0f;
    float factor = std::exp(-x * error);
    // clip to range winding_off_factor to 1
    if (factor < winding_off_factor) {
        factor = winding_off_factor;
    }
    return factor;
}

void update_weights(std::vector<Node>& graph, float scale) {
    for (auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        std::vector<float> errors(node.edges.size());
        float weighted_mean_error = 0.0f;
        float mean_certainty = 0.0f;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            if (graph[node.edges[i].target_node].deleted) {
                continue;
            }
            // difference between BP solution and estimated k from the graph
            errors[i] = std::abs(scale * (graph[node.edges[i].target_node].f_tilde - node.f_tilde) - node.edges[i].k);
            weighted_mean_error += errors[i] * node.edges[i].certainty_factored;
            mean_certainty += node.edges[i].certainty_factored;
        }
        if (mean_certainty == 0.0f) {
            continue;
        }
        weighted_mean_error /= mean_certainty;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            // asolute distance from the mean error
            errors[i] = std::abs(errors[i] - weighted_mean_error);
        }
        float factors_mean = 0.0f;
        size_t factors_count = 0;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            if (graph[node.edges[i].target_node].deleted) {
                continue;
            }
            // certainty factor based on the error
            node.edges[i].certainty_factor = 0.8f * node.edges[i].certainty_factor + 0.2f * certainty_factor(errors[i]);
            factors_mean += node.edges[i].certainty_factor;
            factors_count++;
        }
        factors_mean /= factors_count;
        // Normalize the factors
        for (auto& edge : node.edges) {
            edge.certainty_factor = edge.certainty_factor / factors_mean;
            edge.certainty_factored = edge.certainty * edge.certainty_factor;
        }
    }
}

float min_f_star(const std::vector<Node>& graph, bool use_gt = false) {
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

float max_f_star(const std::vector<Node>& graph, bool use_gt = false) {
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

float calculate_scale(const std::vector<Node>& graph, int estimated_windings) {
    if (estimated_windings <= 0) {
        return 1.0f;
    }
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);
    return std::abs((360.0f * estimated_windings) / (max_f - min_f));
}

float exact_matching_score(std::vector<Node>& graph) {
    // Assing closest valid winding angle to f_star based on f_init
    std::vector<Node> graph_copy = graph;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        graph_copy[i].f_star = closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }

    float score = 0.0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        for (const auto& edge : graph_copy[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            float diff = graph_copy[edge.target_node].f_star - graph_copy[i].f_star;
            if (std::abs(diff - edge.k) < 1e-5) {
                score += edge.certainty;
            }
        }
    }

    return score;
}

float approximate_matching_loss(const std::vector<Node>& graph, float a = 1.0f) {
    float loss = 0.0;

    for (const auto& node : graph) {
        for (const auto& edge : node.edges) {
            float diff = graph[edge.target_node].f_star - node.f_star;
            float l = diff - edge.k;
            loss += edge.certainty * std::exp(-a * std::abs(l));
        }
    }

    return loss;
}

std::tuple<float, float, float, float, float, float> computeErrorStats(const std::vector<Node>& graph, const std::vector<size_t>& valid_gt_indices, int n_pairs = 10'000) {
    if (valid_gt_indices.size() < 2) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // Not enough valid ground truth nodes to form pairs
    }

    std::vector<float> errors; // To store individual errors
    
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine for random number generation
    std::uniform_int_distribution<> dis(0, valid_gt_indices.size() - 1);

    // Loop for a maximum of n_pairs random pairs
    for (int i = 0; i < n_pairs; ++i) {
        // Randomly pick two distinct nodes with valid ground truth
        size_t idx_i = valid_gt_indices[dis(gen)];
        size_t idx_j = valid_gt_indices[dis(gen)];
        
        // Ensure we don't compare a node with itself
        while (idx_i == idx_j) {
            idx_j = valid_gt_indices[dis(gen)];
        }

        const Node& node_i = graph[idx_i];
        const Node& node_j = graph[idx_j];

        // Compute the distance1 (ground truth distances)
        float dist1 = node_i.gt_f_star - node_j.gt_f_star;

        // Compute the distance2 (computed f_star distances)
        float dist2 = node_i.f_star - node_j.f_star;

        // Compute the absolute error
        float error = std::abs(dist1 - dist2);

        // Store the error
        errors.push_back(error);
    }

    // If no valid pairs are found, return all zeros to avoid division by zero
    if (errors.empty()) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Sort the error values to compute statistics
    std::sort(errors.begin(), errors.end());

    // Compute the mean
    float total_error = std::accumulate(errors.begin(), errors.end(), 0.0f);
    float mean_error = total_error / errors.size();

    // Min and Max
    float min_error = errors.front();
    float max_error = errors.back();

    // Quartiles
    float q1 = errors[errors.size() / 4];
    float median = errors[errors.size() / 2];
    float q3 = errors[(errors.size() * 3) / 4];

    // Return the tuple of statistics
    return std::make_tuple(mean_error, min_error, q1, median, q3, max_error);
}

// Perform a breadth-first search (BFS) to gather a patch of non-deleted nodes around a seed node, limited by breadth distance
std::vector<size_t> bfsExpand(const std::vector<Node>& graph, size_t seed_idx, size_t breath) {
    std::vector<size_t> patch;  // Stores the indices of nodes in the patch
    std::queue<std::pair<size_t, size_t>> node_queue;  // Pair of (node index, current distance)
    std::vector<bool> visited(graph.size(), false);

    // Start BFS from the seed node, with distance 0
    node_queue.push({seed_idx, 0});
    visited[seed_idx] = true;

    while (!node_queue.empty()) {
        auto [current_idx, current_breath] = node_queue.front();
        node_queue.pop();

        // Add the current node to the patch if it's not deleted and contains gt
        if (!graph[current_idx].deleted && graph[current_idx].gt) {
            patch.push_back(current_idx);
        }

        // Stop expanding further if we have reached the maximum breadth level
        if (current_breath >= breath) {
            continue;
        }

        // Explore neighbors (edges) of the current node
        for (const Edge& edge : graph[current_idx].edges) {
            // Check if edge is "active" in the graph
            float edge_fit = std::abs(graph[current_idx].f_star + edge.k - graph[edge.target_node].f_star);
            if (edge_fit > 180.0f) {
                continue;
            }
            if (!visited[edge.target_node] && !graph[edge.target_node].deleted && !edge.same_block) {
                visited[edge.target_node] = true;
                node_queue.push({edge.target_node, current_breath + 1});  // Push neighbor with incremented breath level
            }
        }
    }

    return patch;  // Return the indices of non-deleted nodes in the patch
}

// Function to compute errors between the seed node and nodes in its patch
std::tuple<float, float, float, float, float, float> computeLocalizedError(const std::vector<Node>& graph, const std::vector<size_t>& valid_gt_indices, int N = 100, int L = 10) {
    if (valid_gt_indices.size() < 2) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);  // Not enough valid ground truth nodes
    }

    std::vector<float> patch_errors;
    std::random_device rd;
    std::mt19937 gen(rd());  // Random number generator
    std::uniform_int_distribution<> dis(0, valid_gt_indices.size() - 1);

    // Loop for N patches
    for (int i = 0; i < N; ++i) {
        // Randomly pick a seed node
        size_t seed_idx = valid_gt_indices[dis(gen)];
        const Node& seed_node = graph[seed_idx];

        // Perform BFS to gather a patch around the seed node
        std::vector<size_t> patch = bfsExpand(graph, seed_idx, L);

        // Compute the error between the seed node and each node in the patch
        for (size_t patch_idx : patch) {
            if (patch_idx == seed_idx) continue;  // Skip the seed node itself

            const Node& patch_node = graph[patch_idx];

            // Compute the distance1 (ground truth distances)
            float dist1 = seed_node.gt_f_star - patch_node.gt_f_star;

            // Compute the distance2 (computed f_star distances)
            float dist2 = seed_node.f_star - patch_node.f_star;

            // Compute the absolute error
            float error = std::abs(dist1 - dist2);

            // Store the error
            patch_errors.push_back(error);
        }
    }

    // If no errors were calculated, return 0 values
    if (patch_errors.empty()) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Sort the error values to compute statistics
    std::sort(patch_errors.begin(), patch_errors.end());

    // Compute the mean
    float total_error = std::accumulate(patch_errors.begin(), patch_errors.end(), 0.0f);
    float mean_error = total_error / patch_errors.size();

    // Min and Max
    float min_error = patch_errors.front();
    float max_error = patch_errors.back();

    // Quartiles
    float q1 = patch_errors[patch_errors.size() / 4];
    float median = patch_errors[patch_errors.size() / 2];
    float q3 = patch_errors[(patch_errors.size() * 3) / 4];

    // Return the tuple of statistics
    return std::make_tuple(mean_error, min_error, q1, median, q3, max_error);
}

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000) {
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

void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps = 10) {
    std::vector<cv::String> filenames;
    cv::glob(directory + "/*.png", filenames);

    // Sort the filenames in ascending order
    std::sort(filenames.begin(), filenames.end());

    if (filenames.empty()) {
        std::cerr << "No images found in directory: " << directory << std::endl;
        return;
    }

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

void generate_edge_certainty_histograms(const std::vector<Node>& graph, float max_certainty, const std::string& filename_same = std::string(), const std::string& filename_non_same = std::string(), int num_buckets = 1000) {
    // Separate certainties for same_block and non_same_block edges
    std::vector<float> same_block_certainties, non_same_block_certainties;
    for (const auto& node : graph) {
        for (const auto& edge : node.edges) {
            if (edge.same_block) {
                same_block_certainties.push_back(edge.certainty);
            } else {
                non_same_block_certainties.push_back(edge.certainty);
            }
        }
    }

    // Define bucket size for histograms
    float bucket_size = max_certainty / num_buckets;

    // Initialize histograms with 0 counts
    std::vector<int> same_block_histogram(num_buckets, 0);
    std::vector<int> non_same_block_histogram(num_buckets, 0);

    // Fill the histograms for same_block and non_same_block edges
    for (const auto& certainty : same_block_certainties) {
        int bucket_index = static_cast<int>(certainty / bucket_size);
        if (bucket_index >= 0 && bucket_index < num_buckets) {
            same_block_histogram[bucket_index]++;
        }
    }
    for (const auto& certainty : non_same_block_certainties) {
        int bucket_index = static_cast<int>(certainty / bucket_size);
        if (bucket_index >= 0 && bucket_index < num_buckets) {
            non_same_block_histogram[bucket_index]++;
        }
    }

    // Create a blank image for both histograms
    int hist_w = num_buckets;
    int hist_h = 800;
    int bin_w = std::max(1, hist_w / num_buckets);
    int left_padding = 50;

    cv::Mat same_block_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat non_same_block_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));

    // Normalize the histograms to fit in the images
    int max_value_same = *std::max_element(same_block_histogram.begin(), same_block_histogram.end());
    int max_value_non_same = *std::max_element(non_same_block_histogram.begin(), non_same_block_histogram.end());
    std::cout << "Max value same block: " << max_value_same << std::endl;
    std::cout << "Max value non-same block: " << max_value_non_same << std::endl;
    for (int i = 0; i < num_buckets; ++i) {
        same_block_histogram[i] = (same_block_histogram[i] * (hist_h - 50)) / max_value_same;
        non_same_block_histogram[i] = (non_same_block_histogram[i] * (hist_h - 50)) / max_value_non_same;
    }

    // Draw the histograms with left padding
    for (int i = 0; i < num_buckets; ++i) {
        // Draw the same_block histogram
        cv::rectangle(same_block_image,
                      cv::Point(left_padding + i * bin_w, hist_h - same_block_histogram[i] - 50),
                      cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),
                      cv::Scalar(0, 0, 255),  // Red color for same_block edges
                      cv::FILLED);

        // Draw the non_same_block histogram
        cv::rectangle(non_same_block_image,
                      cv::Point(left_padding + i * bin_w, hist_h - non_same_block_histogram[i] - 50),
                      cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),
                      cv::Scalar(0, 255, 0),  // Green color for non_same_block edges
                      cv::FILLED);
    }

    // Add x-axis labels
    std::string min_label = "Min: 0";
    std::string max_label = "Max: " + std::to_string(max_certainty);
    cv::putText(same_block_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(same_block_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(non_same_block_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(non_same_block_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Save the histogram images to files if filenames are provided
    if (!filename_same.empty()) {
        cv::imwrite(filename_same, same_block_image);
        std::cout << "Same block histogram saved to " << filename_same << std::endl;
    }
    if (!filename_non_same.empty()) {
        cv::imwrite(filename_non_same, non_same_block_image);
        std::cout << "Non-same block histogram saved to " << filename_non_same << std::endl;
    }

    // Display the histograms
    cv::imshow("Same Block Edge Certainty Histogram", same_block_image);
    cv::imshow("Non-Same Block Edge Certainty Histogram", non_same_block_image);
    cv::waitKey(1);
}



bool is_edge_valid(const std::vector<Node>& graph, const Edge& edge, const Node& current_node, float threshold = 0.1) {
    float diff = graph[edge.target_node].f_star - current_node.f_star;
    bool valid =true;
    if (edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 * threshold;
    }
    else if (!edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 * threshold;
    }
    return valid;
}

bool remove_invalid_edges(std::vector<Node>& graph, float threshold = 0.1) {
    int erased_edges = 0;
    int remaining_edges = 0;
    
    for (auto& node : graph) {
        if (node.deleted) {
            continue;
        }

        int edges_before = node.edges.size();
        
        // Use the correct current_node reference
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [&](const Edge& edge) {
            return !edge.fixed && !is_edge_valid(graph, edge, node, threshold);
        }), node.edges.end());
        
        erased_edges += edges_before - node.edges.size();
        
        for (const auto& edge : node.edges) {
            if (!graph[edge.target_node].deleted) {
                remaining_edges++;
            }
        }
    }

    std::cout << "Erased edges: " << erased_edges << std::endl;
    std::cout << "Remaining edges: " << remaining_edges << std::endl;
    
    return erased_edges > 0;
}

void update_nodes(std::vector<Node>& graph, float o, float spring_constant, std::vector<size_t> valid_indices) {
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        float sum_w_f_tilde_k = 0.0f;
        float sum_w = 0.0f;

        for (const auto& edge : graph[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            size_t neighbor_node = edge.target_node;
            // sum_w_f_tilde_k += edge.certainty_factored * (graph[neighbor_node].f_tilde - spring_constant * edge.k);
            // sum_w += edge.certainty_factored;
            sum_w_f_tilde_k += edge.certainty * (graph[neighbor_node].f_tilde - spring_constant * edge.k);
            sum_w += edge.certainty;
        }

        // Calculate the new f_star for node i
        graph[i].f_star = (sum_w_f_tilde_k + o * graph[i].f_tilde) / (sum_w + o);
    }

    // Update f_tilde with the newly computed f_star values
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        graph[i].f_tilde = graph[i].f_star;
    }
}

void update_fold_nodes(std::vector<Node>& graph, float o, std::vector<size_t> valid_indices, float push_out_factor = 1.01f) {
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        float sum_w = 0.0f;
        float sum_folds = 0.0f;
        for (const auto& edge : graph[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            size_t neighbor_node = edge.target_node;
            float neighbour_fold = graph[neighbor_node].fold;
            // Switch fold if the edge is between two adjacent wraps
            if (edge.same_block) {
                neighbour_fold = 1.0f - neighbour_fold;
            }
            // sum_folds += edge.certainty * neighbour_fold;
            // sum_w += edge.certainty;
            sum_folds += neighbour_fold;
            sum_w += 1;
        }

        // Calculate the new f_star for node i
        if (sum_w != 0.0f) {
            sum_folds -= sum_w / 2;
            sum_folds *= push_out_factor;
            sum_folds += sum_w / 2;
            // clip to 0 to 1
            if (sum_folds < 0.0f) {
                sum_folds = 0.0f;
            }
            if (sum_folds > sum_w) {
                sum_folds = sum_w;
            }
        }
        graph[i].f_star = (sum_folds + o * graph[i].fold) / (sum_w + o);// f_star is used as fold during fold detection
    }

    // Update fold with the newly computed f_star values
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        graph[i].fold = graph[i].f_star;
    }
}

void solve_fold(std::vector<Node>& graph, argparse::ArgumentParser* program, std::vector<size_t> valid_indices, int num_iterations = 10000) {
    // Default values for parameters
    float o = 2.0f;
    bool video_mode = false;
    float push_out_factor = 1.01f;

    // Parse the arguments
    try {
        o = program->get<float>("--o");
        video_mode = program->get<bool>("--video");
        push_out_factor = program->get<float>("--push_out_factor");
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return;
    }

    // Calculate the number of digits needed for padding
    int max_iter_digits = static_cast<int>(std::log10(num_iterations - 1)) + 1;

    // BP
    for (int iter = 0; iter < num_iterations; ++iter) {
        update_fold_nodes(graph, o, valid_indices, push_out_factor);

        if (iter % 100 == 0) {
            std::cout << "Iteration: " << iter << std::endl;
            // Generate filename with zero padding
            std::ostringstream filename;
            filename << "histogram/fold_hist_" 
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            // Calculate and display the histogram of f_star values
            if (video_mode) {
                calculate_histogram(graph, filename.str());
            }
        }
    }
    // Extract fold

    // Get winding direction of middle (towards + or -), by counting nr of nodes between f star min and middle vs f star middle and f star max
    float f_min = min_f_star(graph);
    float f_max = max_f_star(graph);
    float middle = (f_min + f_max) / 2;
    int count_minus = 0;
    int count_plus = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        if (graph[i].f_star < middle) {
            count_minus++;
        }
        else {
            count_plus++;
        }
    }
    std::cout << "Count minus: " << count_minus << ", count plus: " << count_plus << std::endl;
    // decide which one is the middle of the scroll. Middle has less nodes per winding
    bool direction_minus = count_minus < count_plus;

    // Filter the folds close to 0 or 1 that are not in the middle. set them to 0.5
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        if (graph[i].f_tilde <= middle && !direction_minus) {
            graph[i].fold = 0.5f;
        }
        if (graph[i].f_tilde >= middle && direction_minus) {
            graph[i].fold = 0.5f;
        }
    }

    // Generate filename with zero padding
    std::ostringstream filename;
    filename << "histogram/fold_hist_" 
            << std::setw(max_iter_digits) << std::setfill('0') << max_iter_digits << ".png";
    // Calculate and display the histogram of f_star values
    if (video_mode) {
        calculate_histogram(graph, filename.str());
    }
}

std::vector<float> generate_spring_constants(float start_value, int steps) {
    std::vector<float> spring_constants;
    float current_value = start_value;
    float multiplier = std::pow(0.1f / (start_value - 1.0f), 1.0f / steps); // after steps should get value to 1.1
    std::cout << "Multiplier: " << multiplier << std::endl;

    for (int i = 0; i < steps; ++i) {
        spring_constants.push_back(current_value);

        // Alternate between above and below 1, gradually reducing the difference
        if (current_value > 1.0f) {
            // Reduce the multiplier slightly to get closer to 1
            current_value = 1.0f + multiplier * (current_value - 1.0f);
            current_value = 1.0f / current_value;
        }
        else {
            current_value = 1.0f / current_value;
            // Reduce the multiplier slightly to get closer to 1
            current_value = 1.0f + multiplier * (current_value - 1.0f);
        }
    }

    // Ensure the final value is exactly 1
    spring_constants.push_back(1.0f);

    return spring_constants;
}

std::vector<size_t> get_valid_indices(const std::vector<Node>& graph) {
    std::vector<size_t> valid_indices;
    size_t num_valid_nodes = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        // node is not deleted and not fixed, can be updated
        if (!graph[i].deleted) {
            valid_indices.push_back(i);
            num_valid_nodes++;
        }
    }
    std::cout << "Number of valid nodes: " << num_valid_nodes << std::endl;
    return valid_indices;
}

std::vector<size_t> get_valid_gt_indices(const std::vector<Node>& graph) {
    std::vector<size_t> valid_gt_indices;
    size_t num_valid_nodes = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!graph[i].deleted && graph[i].gt) {
            valid_gt_indices.push_back(i);
            num_valid_nodes++;
        }
    }
    std::cout << "Number of valid gt nodes: " << num_valid_nodes << std::endl;
    return valid_gt_indices;
}

size_t count_fixed_nodes(const std::vector<Node>& graph) {
    size_t num_fixed_nodes = 0;
    for (const auto& node : graph) {
        if (!node.deleted && node.fixed) {
            num_fixed_nodes++;
        }
    }
    return num_fixed_nodes;
}

// This is an example of a solve function that takes the graph and parameters as input
void solve(std::vector<Node>& graph, argparse::ArgumentParser* program, int num_iterations = 10000) {
    // Default values for parameters
    int estimated_windings = 0;
    float spring_constant = 2.0f;
    float o = 2.0f;
    float iterations_factor = 2.0f;
    float o_factor = 0.25f;
    float spring_factor = 6.0f;
    int steps = 5;
    bool auto_mode = false;
    bool video_mode = false;

    // Parse the arguments
    try {
        estimated_windings = program->get<int>("--estimated_windings");
        o = program->get<float>("--o");
        spring_constant = program->get<float>("--spring_constant");
        steps = program->get<int>("--steps");
        iterations_factor = program->get<float>("--iterations_factor");
        o_factor = program->get<float>("--o_factor");
        spring_factor = program->get<float>("--spring_factor");
        auto_mode = program->get<bool>("--auto");
        video_mode = program->get<bool>("--video");
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return;
    }

    // Print the parameters
    std::cout << "Estimated Windings: " << estimated_windings << std::endl;
    std::cout << "Number of Iterations: " << num_iterations << std::endl;
    std::cout << "O: " << o << std::endl;
    std::cout << "Spring Constant: " << spring_constant << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Iterations Factor: " << iterations_factor << std::endl;
    std::cout << "O Factor: " << o_factor << std::endl;
    std::cout << "Spring Factor: " << spring_factor << std::endl;
    std::cout << "Auto Mode: " << (auto_mode ? "Enabled" : "Disabled") << std::endl;

    // Path to the histogram directory
    std::string histogram_dir = "histogram";

    // Delete the existing histogram directory if it exists
    if (fs::exists(histogram_dir)) {
        fs::remove_all(histogram_dir);
    }

    // Create a new histogram directory
    fs::create_directory(histogram_dir);

    // Generate spring constants starting from 5.0 with 12 steps
    std::vector<float> spring_constants = generate_spring_constants(spring_constant, steps);

    // Calculate the number of digits needed for padding
    int max_index_digits = static_cast<int>(std::log10(spring_constants.size())) + 1;
    int max_iter_digits = static_cast<int>(std::log10(num_iterations - 1)) + 1;

    // store only the valid indices to speed up the loop
    std::vector<size_t> valid_indices;
    std::vector<size_t> valid_gt_indices;

    float invalid_edge_threshold = 3.5f;

    int edges_deletion_round = 0;
    while (true) {
        // store only the valid indices to speed up the loop
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph);
        // Do 2 rounds of edge deletion
        if (edges_deletion_round > 8 || invalid_edge_threshold <= 0.05) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations = num_iterations * 3;
            spring_constant = 1.0f;

            break;
        }
        // Solve for each spring constant
        for (int64_t i = -1; i < steps+1; ++i) {
            int num_iterations_iteration = num_iterations;
            float o_iteration = o;
            float spring_constant_iteration = i == -1 ? spring_constants[0] : spring_constants[i];
            if (i == -1 && edges_deletion_round == 0) {
                // Use a warmup iteration with 10x the spring constant
                num_iterations_iteration *= iterations_factor;
                o_iteration = o * o_factor;
                spring_constant_iteration = spring_factor;
            }
            else if (i == -1) {
                // Skip the warmup iteration for subsequent rounds
                continue;
            }
            else if (i == steps && edges_deletion_round >= 1) {
                // Do last of updates with 3x times iterations and spring constant 1.0
                num_iterations_iteration *= 3.0f;
                spring_constant_iteration = 1.0f;
            }
            else if (i == steps) {
                // Do last of updates with 3x times iterations and spring constant 1.0
                num_iterations_iteration *= 1.5f;
            }
            std::cout << "Spring Constant " << i << ": " << std::setprecision(10) << spring_constant_iteration << std::endl;
            bool first_estimated_iteration = i == -1 && edges_deletion_round == 0 && estimated_windings > 0;
            for (int iter = 0; first_estimated_iteration ? true : iter < num_iterations_iteration; ++iter) {
                update_nodes(graph, o_iteration, spring_constant_iteration, valid_indices);

                if (iter % 100 == 0) {
                    // Generate filename with zero padding
                    std::ostringstream filename;
                    filename << "histogram/histogram_" 
                            << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                            << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                            << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
                    // Calculate and display the histogram of f_star values
                    if (video_mode) {
                        calculate_histogram(graph, filename.str());
                    }

                    // // Calculate new weights factors
                    // float scale = calculate_scale(graph, estimated_windings);
                    // update_weights(graph, scale);

                    // Print
                    // auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
                    // auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
                    // std::cout << "\rIteration: " << iter << " Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::flush;  // Updates the same line
                    std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line

                    // escape if estimated windings reached
                    if (first_estimated_iteration) {
                        // float min_f = min_f_star(graph);
                        // float max_f = max_f_star(graph);
                        auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph, 0.02f);
                        // std::cout << " Min percentile: " << min_percentile << ", Max percentile: " << max_percentile << std::endl;
                        if (max_percentile - min_percentile > 1.0f * 360.0f * estimated_windings) {
                            break;
                        }
                    }
                }
            }
            // endline
            std::cout << std::endl;

            // After generating histograms, create a video from the images
            if (video_mode) {
                create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
            }
        }
        // After first edge deletion round remove the invalid edges
        if (edges_deletion_round >= 0) {
            // Remove edges with too much difference between f_star and k
            remove_invalid_edges(graph, invalid_edge_threshold);
        }
        find_largest_connected_component(graph);
        // Update the valid indices
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph);
        size_t num_fixed = count_fixed_nodes(graph);
        std::cout << "Number of fixed nodes remaining: " << num_fixed << std::endl;
        
        // Reduce the threshold by 20% each time
        invalid_edge_threshold *= 0.7f;
        invalid_edge_threshold -= 0.1f;
        if (invalid_edge_threshold < 0.30) {
            invalid_edge_threshold = 0.30;
        }
        std::cout << "Reducing invalid edges threshold to: " << invalid_edge_threshold << std::endl;
        // Assign winding angles again after removing invalid edges
        float scale = calculate_scale(graph, estimated_windings);
        std::cout << "Scale: " << scale << std::endl;
        
        // // Detect folds
        // if (edges_deletion_round == 0) {
        //     // Solve the fold detection
        //     solve_fold(graph, program, get_valid_indices(graph), 10000);
        // }

        // Assign winding angles again after removing invalid edges
        assign_winding_angles(graph, scale);
        
        // Print the error statistics
        auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
        auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
        std::cout << "After assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;

        // Save the graph back to a binary file
        save_graph_to_binary("temp_output_graph.bin", graph);

        edges_deletion_round++;
    }
    // Assign winding angles to the graph
    float scale = calculate_scale(graph, estimated_windings);
    assign_winding_angles(graph, scale);

    // Print the error statistics
    auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
    auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
    std::cout << "After final assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;

    if (video_mode) {
        // Calculate final histogram after all iterations
        calculate_histogram(graph, "final_histogram.png");
        // After generating all histograms, create a final video from the images
        create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
    }
}

void invert_winding_direction_graph(std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (auto& edge : graph[i].edges) {
            if (edge.same_block) {
                float turns = edge.k / 360;
                turns = std::round(turns);
                if (std::abs(turns) > 1 || std::abs(turns) == 0) {
                    std::cout << "Inverting winding direction failed " << turns << std::endl;
                }
                edge.k = edge.k - (2.0f * 360.0f * turns);
            }
        }
    }
}

void auto_winding_direction(std::vector<Node>& graph, argparse::ArgumentParser* program) {
    std::cout << "Auto Winding Direction" << std::endl;
    
    // Make a copy of the graph
    std::vector<Node> auto_graph = graph;
    // min max z values of graph
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::min();
    for (const auto& node : auto_graph) {
        if (node.deleted) {
            continue;
        }
        if (node.z < z_min) {
            z_min = node.z;
        }
        if (node.z > z_max) {
            z_max = node.z;
        }
    }
    float middle_z = (z_min + z_max) / 2.0f;
    // set z range: middle +- 250
    set_z_range_graph(auto_graph, middle_z - 250.0f, middle_z + 250.0f); // speedup the winding direction computation
    
    std::vector<Node> auto_graph_other_block = auto_graph;
    invert_winding_direction_graph(auto_graph_other_block); // build inverted other block graph

    // solve
    int auto_num_iterations = program->get<int>("--auto_num_iterations");
    if (auto_num_iterations == -1) {
        auto_num_iterations = program->get<int>("--num_iterations");
    }
    solve(auto_graph_other_block, program, auto_num_iterations);
    solve(auto_graph, program, auto_num_iterations);

    // Exact matching score
    float exact_score = exact_matching_score(auto_graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;
    float exact_score_other_block = exact_matching_score(auto_graph_other_block);
    std::cout << "Exact Matching Score Other Block: " << exact_score_other_block << std::endl;

    // delete all same_block edges
    for (auto& node : auto_graph_other_block) {
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [](const Edge& edge) {
            return edge.same_block;
        }), node.edges.end());
    }
    for (auto& node : auto_graph) {
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [](const Edge& edge) {
            return edge.same_block;
        }), node.edges.end());
    }

    // Calculate exact matching score for both graphs
    float exact_score2 = exact_matching_score(auto_graph);
    std::cout << "Exact Matching Score (no same block edges): " << exact_score2 << std::endl;
    float exact_score_other_block2 = exact_matching_score(auto_graph_other_block);
    std::cout << "Exact Matching Score Other Block (no same block edges): " << exact_score_other_block2 << std::endl;

    if (exact_score_other_block2 > exact_score2) {
        std::cout << "Inverting the winding direction" << std::endl;
        invert_winding_direction_graph(graph);
    }
    else {
        std::cout << "Standard winding direction has highest score. Not inverting the winding direction." << std::endl;
    }
}

void construct_ground_truth_graph(std::vector<Node>& graph) {
    std::cout << "Constructing Ground Truth Graph" << std::endl;

    // Delete node withouth ground truth
    for (size_t i = 0; i < graph.size(); ++i) {
        graph[i].deleted = !graph[i].gt;
        // assign gt to f_star
        graph[i].f_star = graph[i].gt_f_star;
    }
}

void fix_gt_parts(std::vector<Node>& graph, std::vector<float> fix_lines_z, int fix_windings = 0, float edge_good_certainty = 1.0f, bool fix_all = false) {
    size_t nr_fixed = 0;
    std::vector<size_t> fixed_nodes;
    // Update the lines coordinate system to mask3d system
    for (size_t i = 0; i < fix_lines_z.size(); ++i) {
        fix_lines_z[i] = (fix_lines_z[i] + 500) / 4.0f;
    }
    // fix lines. if the z value of the node is in the fix_lines_z +- 25 and gt is available, then set the gt to f_star and fix to true
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted || !graph[i].gt) {
            continue;
        }
        for (size_t j = 0; j < fix_lines_z.size(); ++j) {
            if (graph[i].z > fix_lines_z[j] - 25.0f && graph[i].z < fix_lines_z[j] + 25.0f) {
                graph[i].fixed = true;
            }
        }
    }
    float f_min = min_f_star(graph, true);
    float f_max = max_f_star(graph, true);
    // fix windings. if windings is negative go from the end of the graph this many windings and set the gt to f_star and fix to true
    float start_winding = f_min;
    float end_winding = f_max;
    if (fix_windings < 0) {
        start_winding = f_max - 360.0f * std::abs(fix_windings);
    }
    else if (fix_windings > 0) {
        end_winding = f_min + 360.0f * std::abs(fix_windings);
    }
    if (fix_windings != 0) {
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].deleted || !graph[i].gt) {
                continue;
            }
            if (graph[i].gt_f_star >= start_winding && graph[i].gt_f_star <= end_winding) {
                graph[i].fixed = true;
            }
        }
    }
    if (fix_all) {
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].deleted || !graph[i].gt) {
                continue;
            }
            graph[i].fixed = true;
        }
    }
    // now fix good edges and delete bad ones
    int good_edges = 0;
    int bad_edges = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted || !graph[i].fixed) {
            continue;
        }
        nr_fixed++;
        fixed_nodes.push_back(i);
        for (auto& edge : graph[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (graph[edge.target_node].fixed && graph[i].gt && graph[edge.target_node].gt) {
                // check if edge has right k
                float diff = graph[edge.target_node].gt_f_star - graph[i].gt_f_star;
                float k = edge.k;
                // closer than e-5?
                if (abs(k - diff) < 0.1) {
                    // good edge
                    edge.certainty = edge_good_certainty;
                    good_edges++;
                }
                else {
                    // delete edge
                    edge.certainty = 0.0f;
                    bad_edges++;
                }
                // fix edge 
                edge.fixed = true;
            }
        }
    }
    std::cout << "Fixed " << good_edges << " good edges and deleted " << bad_edges << " bad edges. Of " << nr_fixed << " fixed nodes." << std::endl;
    // add fixed edges between fixed nodes randomly with help of fixed_nodes
    for (size_t i_ = 0; i_ < fixed_nodes.size(); ++i_) {
        size_t i = fixed_nodes[i_];
        if (graph[i].deleted || !graph[i].fixed) {
            continue;
        }
        for (size_t j_ = 0; j_ < fixed_nodes.size(); ++j_) {
            size_t j = fixed_nodes[j_];
            if (graph[j].deleted || !graph[j].fixed) {
                continue;
            }
            if (i == j) {
                continue;
            }
            // check if edge already exists
            bool edge_exists = false;
            for (const auto& edge : graph[i].edges) {
                if (edge.target_node == j) {
                    edge_exists = true;
                    break;
                }
            }
            if (!edge_exists) {
                // randomly pick approximately 10 nodes to connect to
                float p = 10.0 / (float)nr_fixed;
                if (rand() % 100000 < p * 100000) {
                    // add edge
                    Edge edge;
                    edge.target_node = j;
                    edge.k = graph[j].gt_f_star - graph[i].gt_f_star;
                    edge.certainty = edge_good_certainty;
                    edge.fixed = true;
                    graph[i].edges.push_back(edge);
                    // add edge in other direction
                    Edge edge2;
                    edge2.target_node = i;
                    edge2.k = graph[i].gt_f_star - graph[j].gt_f_star;
                    edge2.certainty = edge_good_certainty;
                    edge2.fixed = true;
                    graph[j].edges.push_back(edge2);
                    // increase good edges
                    good_edges += 2;
                }
            }
        }
    }
    std::cout << "Added edges between fixed nodes. Fixed " << good_edges << " good edges and deleted " << bad_edges << " bad edges. Of " << nr_fixed << " fixed nodes." << std::endl;
}

int main(int argc, char** argv) {
    // Parse the input graph file from arguments using argparse
    argparse::ArgumentParser program("Graph Solver");

    // Default values for parameters
    int estimated_windings = 0;
    int num_iterations = 10000;
    int auto_num_iterations = -1;
    float spring_constant = 2.0f;
    float o = 2.0f;
    float iterations_factor = 2.0f;
    float o_factor = 0.25f;
    float spring_factor = 6.0f;
    int steps = 5;
    int z_min = -2147483648;
    int z_max = 2147483647;
    float same_winding_factor = 1.0f;

    // Add command-line arguments for graph input and output
    program.add_argument("--input_graph")
        .help("Input graph binary file")
        .default_value(std::string("graph.bin"));

    program.add_argument("--output_graph")
        .help("Output graph binary file")
        .default_value(std::string("output_graph.bin"));

    // Add command-line arguments
    program.add_argument("--estimated_windings")
        .help("Estimated windings (int)")
        .default_value(estimated_windings)
        .scan<'i', int>();

    program.add_argument("--num_iterations")
        .help("Number of iterations (int)")
        .default_value(num_iterations)
        .scan<'i', int>();

    program.add_argument("--o")
        .help("O parameter (float)")
        .default_value(o)
        .scan<'g', float>();

    program.add_argument("--spring_constant")
        .help("Spring constant (float)")
        .default_value(spring_constant)
        .scan<'g', float>();

    program.add_argument("--steps")
        .help("Steps (int)")
        .default_value(steps)
        .scan<'i', int>();

    program.add_argument("--iterations_factor")
        .help("Iterations factor (float)")
        .default_value(iterations_factor)
        .scan<'g', float>();

    program.add_argument("--o_factor")
        .help("O factor (float)")
        .default_value(o_factor)
        .scan<'g', float>();

    program.add_argument("--spring_factor")
        .help("Spring factor (float)")
        .default_value(spring_factor)
        .scan<'g', float>();

    program.add_argument("--same_winding_factor")
        .help("Same winding factor (float)")
        .default_value(same_winding_factor)
        .scan<'g', float>();

    program.add_argument("--z_min")
        .help("Z range (int)")
        .default_value(z_min)
        .scan<'i', int>();

    program.add_argument("--z_max")
        .help("Z range (int)")
        .default_value(z_max)
        .scan<'i', int>();

    // Add the boolean flag --auto
    program.add_argument("--auto")
        .help("Enable automatic mode")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Add the auto number of iterations
    program.add_argument("--auto_num_iterations")
        .help("Number of iterations for auto mode (int)")
        .default_value(auto_num_iterations)
        .scan<'i', int>();

    // Add the boolean flag --video
    program.add_argument("--video")
        .help("Enable video creation")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Add boolean flag --gt_graph
    program.add_argument("--gt_graph")
        .help("Enable ground truth graph construction")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Multithreading number threads
    program.add_argument("--threads")
        .help("Number of threads (int)")
        .default_value(-1)
        .scan<'i', int>();

    // Flag fix_same_block_edges
    program.add_argument("--fix_same_block_edges")
        .help("Fix same block edges")
        .default_value(false)
        .implicit_value(true);

    // Flag to invert graph
    program.add_argument("--invert")
        .help("Invert the winding direction of the graph")
        .default_value(false)
        .implicit_value(true);

    // Push out factor of the fold detection
    program.add_argument("--push_out_factor")
        .help("Push out factor of the fold detection (float)")
        .default_value(1.01f)
        .scan<'g', float>();

    // Flag fix all gt for solver
    program.add_argument("--fix_all_gt")
        .help("Fix all gt nodes")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
        same_winding_factor = program.get<float>("--same_winding_factor");
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Set number threads
    if (program.get<int>("--threads") > 0) {
        omp_set_num_threads(program.get<int>("--threads"));
    }


    // Load the graph from the input binary file
    std::string input_graph_file = program.get<std::string>("--input_graph");
    auto [graph, max_certainty] = load_graph_from_binary(input_graph_file, true, (static_cast<float>(program.get<int>("--z_min")) + 500) / 4, (static_cast<float>(program.get<int>("--z_max")) + 500) / 4, same_winding_factor, program.get<bool>("--fix_same_block_edges"));

    // invert graph
    if (program.get<bool>("--invert")) {
        invert_winding_direction_graph(graph);
    }

    // generate_edge_certainty_histograms(graph, 0.27, "edge_certainty_histograms_same.png", "edge_certainty_histograms_other.png");

    // Calculate the exact matching loss
    float exact_score = exact_matching_score(graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;

    // Calculate the approximate matching loss
    float approx_loss = approximate_matching_loss(graph, 1.0f);
    std::cout << "Approximate Matching Loss: " << approx_loss << std::endl;

    // Calculate and display the histogram of f_star values
    // calculate_histogram(graph);

    // Check if the ground truth graph construction is enabled
    if (program.get<bool>("--gt_graph")) {
        // Construct the ground truth graph
        construct_ground_truth_graph(graph);
    }
    else {
        // Automatically determing winding direction
        if (program.get<bool>("--auto")) {
            auto_winding_direction(graph, &program);
        }

        // Solve the problem using a solve function
        fix_gt_parts(graph, {6000}, -10, 10.0f * max_certainty, program.get<bool>("--fix_all_gt"));
        size_t num_fixed = count_fixed_nodes(graph);
        std::cout << "Number of fixed nodes: " << num_fixed << std::endl;
        num_iterations = program.get<int>("--num_iterations");
        solve(graph, &program, num_iterations);
    }

    // print the min and max f_star values
    std::cout << "Min f_star: " << min_f_star(graph) << std::endl;
    std::cout << "Max f_star: " << max_f_star(graph) << std::endl;

    // Save the graph back to a binary file
    std::string output_graph_file = program.get<std::string>("--output_graph");
    save_graph_to_binary(output_graph_file, graph);

    // Calculate the exact matching loss
    float exact_score2 = exact_matching_score(graph);
    std::cout << "Exact Matching Score: " << exact_score2 << std::endl;

    // Calculate the approximate matching loss
    float approx_loss2 = approximate_matching_loss(graph, 1.0f);
    std::cout << "Approximate Matching Loss: " << approx_loss2 << std::endl;

    return 0;
}

// Example command to run the program: ./build/graph_problem --input_graph graph.bin --output_graph output_graph.bin --auto --auto_num_iterations 2000 --video --z_min 5000 --z_max 7000 --num_iterations 5000 --estimated_windings 160 --steps 3 --spring_constant 1.2