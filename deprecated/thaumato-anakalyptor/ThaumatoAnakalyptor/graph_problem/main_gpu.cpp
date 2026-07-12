/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/

#include "solve_gpu.h"
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

void append_reverse_edges(std::vector<Node>& graph) {
    // Temporary list to store the reverse edges we need to add
    std::vector<std::pair<size_t, Edge>> reverse_edges;

    // Iterate over each node to collect reverse edges
    for (size_t node_idx = 0; node_idx < graph.size(); ++node_idx) {
        Node& node = graph[node_idx];

        // Skip deleted nodes
        if (node.deleted) continue;

        // Iterate over each edge of the current node
        for (size_t i = 0; i < node.num_edges; ++i) {
            Edge& edge = node.edges[i];
            size_t target_idx = edge.target_node;

            // Skip if the target node is deleted
            if (graph[target_idx].deleted) continue;

            // Create a reverse edge with -k as the edge weight and copy all other fields
            Edge reverse_edge = edge;
            reverse_edge.target_node = node_idx;  // Reverse the direction
            reverse_edge.k = -edge.k;             // Reverse the k value

            // Store the reverse edge in the list to add later
            reverse_edges.emplace_back(target_idx, reverse_edge);
        }
    }

    // Append all collected reverse edges to their respective nodes
    for (const auto& [target_idx, reverse_edge] : reverse_edges) {
        Node& target_node = graph[target_idx];

        // Expand the edges array for the target node
        Edge* new_edges = new Edge[target_node.num_edges + 1];
        std::copy(target_node.edges, target_node.edges + target_node.num_edges, new_edges);

        // Add the new reverse edge
        new_edges[target_node.num_edges] = reverse_edge;
        target_node.num_edges++;

        // Free the old edges array and assign the new one
        delete[] target_node.edges;
        target_node.edges = new_edges;
    }
}

std::vector<Node> cloneGraph(const std::vector<Node>& graph) {
    // Create a new graph with the same size as the original
    std::vector<Node> clonedGraph(graph.size());

    // Copy nodes
    for (size_t i = 0; i < graph.size(); ++i) {
        const Node& origNode = graph[i];
        Node& clonedNode = clonedGraph[i];

        // Copy node attributes
        clonedNode.z = origNode.z;
        clonedNode.f_init = origNode.f_init;
        clonedNode.f_tilde = origNode.f_tilde;
        clonedNode.f_star = origNode.f_star;
        clonedNode.gt = origNode.gt;
        clonedNode.gt_f_star = origNode.gt_f_star;
        clonedNode.deleted = origNode.deleted;
        clonedNode.fixed = origNode.fixed;
        clonedNode.fold = origNode.fold;

        // Allocate and copy edges if they exist
        if (origNode.num_edges > 0) {
            clonedNode.num_edges = origNode.num_edges;
            clonedNode.edges = new Edge[origNode.num_edges];  // Allocate raw array

            for (int j = 0; j < origNode.num_edges; ++j) {
                clonedNode.edges[j] = origNode.edges[j];  // Copy each edge
            }
        } else {
            clonedNode.num_edges = 0;
            clonedNode.edges = nullptr;
        }
    }

    return clonedGraph;
}

void free_graph(std::vector<Node>& graph) {
    for (auto& node : graph) {
        delete[] node.edges;  // Free the dynamic array of edges
        node.edges = nullptr; // Set pointer to nullptr to avoid dangling pointers
        node.num_edges = 0;   // Optionally reset edge count
    }
}

void calculate_and_normalize_edge_certainties(const std::vector<Node>& graph, int num_buckets=1000) {
    float min_f_init = -180.0f;
    float max_f_init = 180.0f;
    float bucket_size = (max_f_init - min_f_init) / num_buckets;

    // Single histogram for all edges
    std::vector<int> hist_all_edges(num_buckets, 0);

    // Step 1: Populate histogram based on f_init values of nodes connected by edges
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }

            // Bin source node's f_init value
            int source_bucket = static_cast<int>((node.f_init - min_f_init) / bucket_size);
            source_bucket = std::max(0, std::min(num_buckets - 1, source_bucket));

            // Bin target node's f_init value
            int target_bucket = static_cast<int>((graph[edge.target_node].f_init - min_f_init) / bucket_size);
            target_bucket = std::max(0, std::min(num_buckets - 1, target_bucket));

            // Increment counts in the histogram for both nodes
            hist_all_edges[source_bucket]++;
            hist_all_edges[target_bucket]++;
        }
    }

    // Step 2: Compute initial normalization factors (inverse of bin counts)
    std::vector<float> norm_factors(num_buckets, 1.0f);
    for (int i = 0; i < num_buckets; ++i) {
        if (hist_all_edges[i] > 0) {
            norm_factors[i] = 1.0f / hist_all_edges[i];
        }
    }

    // Step 3: Adjust normalization factors to ensure the mean is 1
    float mean_factor = std::accumulate(norm_factors.begin(), norm_factors.end(), 0.0f) / num_buckets;
    if (mean_factor != 0) {
        for (float& factor : norm_factors) {
            factor /= mean_factor;
        }
    }

    // Step 4: Save each edge's certainty normalization factor
    for (auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }

            // Bin source and target f_init values
            int source_bucket = static_cast<int>((node.f_init - min_f_init) / bucket_size);
            source_bucket = std::max(0, std::min(num_buckets - 1, source_bucket));

            int target_bucket = static_cast<int>((graph[edge.target_node].f_init - min_f_init) / bucket_size);
            target_bucket = std::max(0, std::min(num_buckets - 1, target_bucket));

            // Get the average normalization factor for the edge
            float norm_factor = (norm_factors[source_bucket] + norm_factors[target_bucket]) / 2.0f;

            // Save the factor to the edge's certainty_factor property
            edge.certainty_factor = norm_factor;
            edge.certainty = edge.certainty * norm_factor;
        }
    }
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

        // Traverse through the edges of the current node
        for (int i = 0; i < graph[current].num_edges; ++i) {
            const Edge& edge = graph[current].edges[i];

            // Skip the edge if the target node is deleted
            if (graph[edge.target_node].deleted) {
                continue;
            }

            // If the target node has not been visited, mark it and push it to the stack
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

    std::cout << "Size of largest connected component: " << largest_component.size() << std::endl;

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

void remove_low_degree_nodes(std::vector<Node>& graph, int deg) {
    int initial_nodes = 0;
    int deleted_nodes = 0;
    int remaining_nodes = 0;
    for (auto& node : graph) {
        if (!node.deleted) {
            initial_nodes++;
        }
        if (node.num_edges <= deg) {
            if(!node.deleted) {
                node.deleted = true;
                deleted_nodes++;
            }
        }
        else if (!node.deleted) {
            remaining_nodes++;
        }
    }
    std::cout << "Removing low degree nodes. Initial nodes: " << initial_nodes << ", deleted nodes: " << deleted_nodes << ", remaining nodes: " << remaining_nodes << std::endl;
}

using EdgeWithCertaintyV2 = std::tuple<float, size_t, size_t, float>;  // {k_delta, source_node, target_node, k}

// Union-Find (Disjoint Set Union) structure to track connected components
struct UnionFind {
    std::vector<size_t> parent, rank;

    UnionFind(size_t n) : parent(n), rank(n, 0) {
        for (size_t i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    size_t find(size_t u) {
        if (parent[u] != u) {
            parent[u] = find(parent[u]);
        }
        return parent[u];
    }

    void unite(size_t u, size_t v) {
        size_t root_u = find(u);
        size_t root_v = find(v);
        if (root_u != root_v) {
            if (rank[root_u] < rank[root_v]) {
                parent[root_u] = root_v;
            } else if (rank[root_u] > rank[root_v]) {
                parent[root_v] = root_u;
            } else {
                parent[root_v] = root_u;
                rank[root_u]++;
            }
        }
    }
};

void prim_mst_assign_f_star_v2(size_t start_node, std::vector<Node>& graph, float scale) {
    size_t num_nodes = graph.size();
    std::vector<size_t> parent(num_nodes, 0);
    std::vector<bool> valid(num_nodes, false);
    std::vector<float> k_values(num_nodes, 0.0);

    // Count all the edges
    size_t total_edges = 0;
    for (size_t node = 0; node < num_nodes; ++node) {
        if (graph[node].deleted) {
            continue;
        }
        for (size_t i = 0; i < graph[node].num_edges; ++i) {
            const Edge& edge = graph[node].edges[i];
            size_t v = edge.target_node;
            if (!graph[v].deleted) {
                total_edges++;
            }
        }
    }
    
    // Priority queue to collect edges and sort by k delta
    std::vector<EdgeWithCertaintyV2> edge_list(total_edges);
    std::cout << "Total edges: " << total_edges << std::endl;

    // Collect all the edges
    size_t edge_index = 0;
    for (size_t node = 0; node < num_nodes; ++node) {
        if (graph[node].deleted) {
            continue;
        }
        for (size_t i = 0; i < graph[node].num_edges; ++i) {
            const Edge& edge = graph[node].edges[i];
            size_t v = edge.target_node;
            if (!graph[v].deleted) {
                float k_delta = std::abs(scale * (graph[v].f_tilde - graph[node].f_tilde) - edge.k);
                if (edge.same_block) {
                    k_delta += 90.0f;
                }
                if (edge.fixed && edge.certainty > 0.0f) {
                    k_delta = -1.0f;
                }
                edge_list[edge_index++] = {k_delta, node, v, edge.k};
            }
        }
    }

    std::cout << "Edge list size: " << edge_list.size() << std::endl;

    // Sort edges by k_delta in ascending order
    std::sort(edge_list.begin(), edge_list.end());

    // List to store edges in the MST
    std::vector<EdgeWithCertaintyV2> mst_edges;
    UnionFind uf(num_nodes);

    // Process the edges to create an MST
    for (size_t i = 0; i < edge_list.size(); ++i) {
        auto [k_delta, u, v, k] = edge_list[i];

        // Use union-find to check if u and v are in the same component
        if (uf.find(u) != uf.find(v)) {
            // Add the edge to the MST list
            mst_edges.push_back({k_delta, u, v, k});

            // Unite the two nodes
            uf.unite(u, v);
        }
    }

    // Create children structures to store each node's children and their k_values
    std::vector<std::vector<size_t>> children(num_nodes);
    std::vector<std::vector<float>> children_k_values(num_nodes);
    for (const auto& [k_delta, u, v, k] : mst_edges) {
        children[u].push_back(v);
        children_k_values[u].push_back(k);
        children[v].push_back(u);
        children_k_values[v].push_back(-k);
    }

    // Traverse the MST to assign f_star values (DFS style traversal)
    std::stack<size_t> stack;
    std::vector<bool> visited_mst(num_nodes, false);
    stack.push(start_node);

    // Set f_star for the root node (start_node)
    graph[start_node].f_tilde = graph[start_node].f_init;
    graph[start_node].f_star = graph[start_node].f_init;

    int traversed_nodes = 0;
    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();

        if (graph[current].deleted) {
            std::cerr << "Error: Node " << current << " is marked as deleted." << std::endl;
            continue;
        }

        // Mark the current node as visited
        visited_mst[current] = true;
        traversed_nodes++;

        for (size_t i = 0; i < children[current].size(); ++i) {
            size_t child = children[current][i];
            if (graph[child].deleted) {
                std::cerr << "Error: Child node " << child << " is marked as deleted." << std::endl;
                continue;
            }
            if (visited_mst[child]) {
                // std::cout << "Cycle detected in MST traversal for node " << child << "." << std::endl;
                continue;
            }

            // Calculate the f_star for the child node
            float k = children_k_values[current][i];
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + k);
            // graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[child].f_star);

            graph[child].f_tilde = graph[child].f_star;

            // Print debug information
            // std::cout << "Node " << current << " -> Child " << child << " with k = " << k << ", f_star = " << graph[child].f_star << " current f_tilde = " << graph[current].f_tilde << std::endl;

            // Push child onto the stack for further processing
            stack.push(child);
        }
    }

    std::cout << "Traversed nodes: " << traversed_nodes << " out of " << num_nodes << std::endl;

    // Check if all nodes have been visited
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i] && !visited_mst[i]) {
            std::cerr << "Node " << i << " was not visited during MST traversal." << std::endl;
        }
    }
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

        for (int i = 0; i < graph[u].num_edges; ++i) {
            const Edge& edge = graph[u].edges[i];

            if (graph[u].deleted) {
                continue;
            }

            size_t v = edge.target_node;
            if (graph[v].deleted) {
                continue;
            }

            // Calculate k_delta (difference between BP solution and estimated k from the graph)
            float k_delta = std::abs(scale * (graph[v].f_tilde - graph[u].f_tilde) - edge.k);
            // Penalize same block edges
            if (edge.same_block) {
                k_delta += 90.0f;
            }
            // Select fixed edges with certainty > 0.0
            if (edge.fixed && edge.certainty > 0.0f) {
                k_delta = -1.0f;
            }

            // Check if this edge has a smaller k_delta and update
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

    // Create children structures to store each node's children and their k_values
    std::vector<std::vector<size_t>> children(num_nodes);
    std::vector<std::vector<float>> children_k_values(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i]) {
            children[parent[i]].push_back(i);
            children_k_values[parent[i]].push_back(k_values[i]);
        }
    }

    // Traverse the MST to assign f_star values (DFS style traversal)
    std::stack<size_t> stack;
    std::vector<bool> visited_mst(num_nodes, false);
    stack.push(start_node);

    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();

        if (graph[current].deleted) {
            continue;
        }

        // Mark the current node as visited
        visited_mst[current] = true;

        for (size_t i = 0; i < children[current].size(); ++i) {
            size_t child = children[current][i];
            if (graph[child].deleted) {
                continue;
            }
            if (visited_mst[child]) {
                std::cout << "Cycle detected in MST traversal." << std::endl;
                continue;
            }

            // Calculate the f_star for the child node
            float k = children_k_values[current][i];
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + k);
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[child].f_star);

            graph[child].f_tilde = graph[child].f_star;

            // Push child onto the stack for further processing
            stack.push(child);
        }
    }
    // Check if all nodes have been visited
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i] && !visited_mst[i]) {
            std::cerr << "Node " << i << " was not visited during MST traversal." << std::endl;
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
    prim_mst_assign_f_star_v2(start_node, graph, scale);

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

// float min_f_star(const std::vector<Node>& graph, bool use_gt = false) {
//     float min_f = std::numeric_limits<float>::max();

//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         if (use_gt) {
//             if (node.gt_f_star < min_f) {
//                 min_f = node.gt_f_star;
//             }
//         } else {
//             if (node.f_star < min_f) {
//                 min_f = node.f_star;
//             }
//         }
//     }

//     return min_f;
// }

// float max_f_star(const std::vector<Node>& graph, bool use_gt = false) {
//     float max_f = std::numeric_limits<float>::min();

//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         if (use_gt) {
//             if (node.gt_f_star > max_f) {
//                 max_f = node.gt_f_star;
//             }
//         } else {
//             if (node.f_star > max_f) {
//                 max_f = node.f_star;
//             }
//         }
//     }

//     return max_f;
// }

float calculate_scale(const std::vector<Node>& graph, int estimated_windings) {
    if (estimated_windings <= 0) {
        return 1.0f;
    }
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);
    return std::abs((360.0f * estimated_windings) / (max_f - min_f));
}

float exact_matching_score(std::vector<Node>& graph) {
    // Copy the graph and assign the closest valid winding angle to f_star based on f_init
    std::vector<Node> graph_copy = graph;

    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        graph_copy[i].f_star = closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }

    float score = 0.0f;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        for (int j = 0; j < graph_copy[i].num_edges; ++j) {
            const Edge& edge = graph_copy[i].edges[j];
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
    float loss = 0.0f;

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
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

        if (node_i.deleted || node_j.deleted) {
            continue; // Skip deleted nodes
        }

        if (node_i.f_init <= -90.0f && node_i.f_init >= -140.0f) { // gt assignment bug most probably. disregard
            continue;
        }
        if (node_j.f_init <= -90.0f && node_j.f_init >= -140.0f) { // gt assignment bug most probably. disregard
            continue;
        }

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
std::vector<size_t> bfsExpand(const std::vector<Node>& graph, size_t seed_idx, size_t breadth) {
    std::vector<size_t> patch;  // Stores the indices of nodes in the patch
    std::queue<std::pair<size_t, size_t>> node_queue;  // Pair of (node index, current distance)
    std::vector<bool> visited(graph.size(), false);

    // Start BFS from the seed node, with distance 0
    node_queue.push({seed_idx, 0});
    visited[seed_idx] = true;

    while (!node_queue.empty()) {
        auto [current_idx, current_breadth] = node_queue.front();
        node_queue.pop();

        // Add the current node to the patch if it's not deleted and contains gt
        if (!graph[current_idx].deleted && graph[current_idx].gt) {
            patch.push_back(current_idx);
        }

        // Stop expanding further if we have reached the maximum breadth level
        if (current_breadth >= breadth) {
            continue;
        }

        // Explore neighbors (edges) of the current node
        for (int j = 0; j < graph[current_idx].num_edges; ++j) {
            const Edge& edge = graph[current_idx].edges[j];
            // Check if edge is "active" in the graph
            float edge_fit = std::abs(graph[current_idx].f_star + edge.k - graph[edge.target_node].f_star);
            if (edge_fit > 180.0f) {
                continue;
            }
            if (!visited[edge.target_node] && !graph[edge.target_node].deleted && !edge.same_block) {
                visited[edge.target_node] = true;
                node_queue.push({edge.target_node, current_breadth + 1});  // Push neighbor with incremented breadth level
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

            if (seed_node.f_init <= -90.0f && seed_node.f_init >= -140.0f) { // gt assignment bug most probably. disregard
                continue;
            }
            if (patch_node.f_init <= -90.0f && patch_node.f_init >= -140.0f) { // gt assignment bug most probably. disregard
                continue;
            }

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

// void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000) {
//     // Find min and max f_star values
//     float min_f = min_f_star(graph);
//     float max_f = max_f_star(graph);

//     // Calculate bucket size
//     float bucket_size = (max_f - min_f) / num_buckets;

//     // Initialize the histogram with 0 counts
//     std::vector<int> histogram(num_buckets, 0);

//     // Fill the histogram
//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         int bucket_index = static_cast<int>((node.f_star - min_f) / bucket_size);
//         if (bucket_index >= 0 && bucket_index < num_buckets) {
//             histogram[bucket_index]++;
//         }
//     }

//     // Create a blank image for the histogram with padding on the left
//     int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
//     int hist_h = 800;  // height of the histogram image
//     int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel
//     int left_padding = 50;  // Add 50 pixels of padding on the left side

//     cv::Mat hist_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels and padding

//     // Normalize the histogram to fit in the image
//     int max_value = *std::max_element(histogram.begin(), histogram.end());
//     for (int i = 0; i < num_buckets; ++i) {
//         histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
//     }

//     // Draw the histogram with left padding
//     for (int i = 0; i < num_buckets; ++i) {
//         cv::rectangle(hist_image, 
//                       cv::Point(left_padding + i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
//                       cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
//                       cv::Scalar(0, 0, 0), 
//                       cv::FILLED);
//     }

//     // Add x-axis labels
//     std::string min_label = "Min: " + std::to_string(min_f);
//     std::string max_label = "Max: " + std::to_string(max_f);
//     cv::putText(hist_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
//     cv::putText(hist_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

//     // Save the histogram image to a file if string not empty
//     if (!filename.empty()) {
//         cv::imwrite(filename, hist_image);
//     }

//     // Display the histogram
//     cv::imshow("Histogram of f_star values", hist_image);
//     cv::waitKey(1);
// }

bool is_edge_valid(const std::vector<Node>& graph, const Edge& edge, const Node& current_node, float threshold = 0.1) {
    float diff = graph[edge.target_node].f_star - current_node.f_star;
    bool valid =true;
    if (edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 * threshold;
    }
    else if (!edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 / 2 * threshold;
    }
    return valid;
}

bool remove_invalid_edges(std::vector<Node>& graph, float threshold = 0.1) {
    int erased_edges = 0;
    int remaining_edges = 0;

    for (auto& node : graph) {
        if (node.deleted || node.num_edges == 0) {
            continue;
        }

        int edges_before = node.num_edges;
        int valid_edge_count = 0;

        // Create a temporary array to store valid edges
        Edge* valid_edges = new Edge[edges_before];

        // Copy valid edges to the temporary array
        for (int j = 0; j < edges_before; ++j) {
            const Edge& edge = node.edges[j];

            if (!edge.fixed && !is_edge_valid(graph, edge, node, threshold)) {
                // Edge is invalid and should be erased
                erased_edges++;
            } else {
                // Edge is valid and should be kept
                valid_edges[valid_edge_count] = edge;
                valid_edge_count++;
            }
        }

        // Free the old edges array
        delete[] node.edges;

        // If there are valid edges, update the node's edges and count
        if (valid_edge_count > 0) {
            node.edges = new Edge[valid_edge_count];
            std::copy(valid_edges, valid_edges + valid_edge_count, node.edges);
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, reset the edges
            node.edges = nullptr;
            node.num_edges = 0;
        }

        // Free the temporary valid_edges array
        delete[] valid_edges;

        // Count remaining valid edges
        remaining_edges += node.num_edges;
    }

    std::cout << "Erased edges: " << erased_edges << std::endl;
    std::cout << "Remaining edges: " << remaining_edges << std::endl;

    return erased_edges > 0;
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
    // spring_constants.push_back(1.0f);
    // spring_constants.push_back(1.0f);
    // last spring constant from list
    // spring_constants.push_back(spring_constants[spring_constants.size() - 1]);
    // spring_constants.push_back(spring_constants[spring_constants.size() - 1]);
    // spring_constants.push_back(spring_constants[spring_constants.size() - 1]);
    spring_constants.push_back(1.0f);
    spring_constants.push_back(1.0f);
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

std::vector<size_t> get_valid_gt_indices(const std::vector<Node>& graph, int fix_windings = 0) {
    // Get the min and max f_star using ground truth values
    float f_min = min_f_star(graph, true);
    float f_max = max_f_star(graph, true);

    // Fix windings based on the number of windings to fix
    float start_winding = f_max - 360.0f * std::abs(fix_windings);
    float end_winding = f_min + 360.0f * std::abs(fix_windings);

    std::vector<size_t> valid_gt_indices;
    size_t num_valid_nodes = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].gt_f_star > start_winding || graph[i].gt_f_star < end_winding) {
            continue;
        }
        if (!graph[i].deleted && graph[i].gt) {
            valid_gt_indices.push_back(i);
            num_valid_nodes++;
        }
    }
    std::cout << "Number of valid gt nodes: " << num_valid_nodes << std::endl;
    return valid_gt_indices;
}

std::pair<int, int> count_good_bad_edges(const Node& node, float test_f_star, const std::vector<Node>& graph) {
    int good_edges_count = 0;
    int bad_edges_count = 0;

    for (size_t i = 0; i < node.num_edges; ++i) {
        const Edge& edge = node.edges[i];
        float n2n1 = graph[edge.target_node].f_star - test_f_star;
        
        // Determine if the edge is in the right direction
        bool right_direction = (n2n1 > 0 && edge.k > 0) || (n2n1 < 0 && edge.k < 0) || (n2n1 == 0 && edge.k == 0);
        
        if (right_direction) {
            good_edges_count++;
        } else {
            bad_edges_count++;
        }
    }

    return {good_edges_count, bad_edges_count};
}

void optimize_f_star_for_good_edges(std::vector<Node>& graph) {
    while (true) {
        int count_updated_nodes = 0;
        for (Node& node : graph) {
            // Skip deleted nodes
            if (node.deleted || node.num_edges == 0) continue;

            auto [right_direction_count, wrong_direction_count] = count_good_bad_edges(node, node.f_star, graph);

            // Step 1: Count right and wrong direction edges
            for (size_t i = 0; i < node.num_edges; ++i) {
                Edge& edge = node.edges[i];
                float n2n1 = graph[edge.target_node].f_star - node.f_star;
                bool right_direction = (n2n1 > 0 && edge.k > 0) || (n2n1 < 0 && edge.k < 0);

                if (right_direction) {
                    right_direction_count++;
                } else {
                    wrong_direction_count++;
                }
            }
            
            // Step 2: Gather and sort target `f_star` values
            std::vector<float> target_f_stars;
            for (size_t i = 0; i < node.num_edges; ++i) {
                target_f_stars.push_back(graph[node.edges[i].target_node].f_star);
            }
            std::sort(target_f_stars.begin(), target_f_stars.end());

            // Step 3: Try placing `f_star` between each consecutive pair to maximize good edges
            float best_f_star = node.f_star;
            int max_good_edges = right_direction_count;

            for (size_t i = -1; i < target_f_stars.size(); ++i) {
                float test_f_star;
                if (i == -1) {
                    test_f_star = target_f_stars[0] - 10.0f;
                } else if (i == target_f_stars.size() - 1) {
                    test_f_star = target_f_stars[i] + 10.0f;
                } else {
                    test_f_star = (target_f_stars[i] + target_f_stars[i + 1]) / 2.0f;
                }

                auto [test_good_edges, test_bad_edges] = count_good_bad_edges(node, test_f_star, graph);

                // Check if this configuration improves the good edge count
                if (test_good_edges > max_good_edges) {
                    max_good_edges = test_good_edges;
                    best_f_star = test_f_star;
                }
            }

            // Step 4: Update `f_star` if an improvement was found
            if (max_good_edges > right_direction_count) {
                // std::cout << "Updating node f_star from " << node.f_star << " to " << best_f_star << " to increase good edge count from " << right_direction_count << " to " << max_good_edges << std::endl;
                node.f_star = best_f_star;
                count_updated_nodes++;
            }
        }

        // Break if no nodes were updated
        if (count_updated_nodes == 0) {
            break;
        }
        else {
            std::cout << "Updated " << count_updated_nodes << " nodes." << std::endl;
        }
    }
}

size_t find_valid_start_node(std::vector<Node>& graph) {
    size_t start_node = 0;
    bool found_start_node = false;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            found_start_node = true;
            break;
        }
    }

    if (!found_start_node) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
    }

    return start_node;
}

void deleted_wrong_direction_edges(std::vector<Node>& graph) {
    int deleted_edges = 0;
    for (auto& node : graph) {
        if (node.deleted || node.num_edges == 0) {
            continue;
        }

        int edges_before = node.num_edges;
        int valid_edge_count = 0;

        // Create a temporary array to store valid edges
        Edge* valid_edges = new Edge[edges_before];

        // Copy valid edges to the temporary array
        for (int j = 0; j < edges_before; ++j) {
            const Edge& edge = node.edges[j];

            // Skip deleted nodes
            if (graph[edge.target_node].deleted) {
                continue;
            }

            float n2n1 = graph[edge.target_node].f_star - node.f_star;
            float k = edge.k;
            bool right_direction = (n2n1 > 0.0f && k > 0.0f) || (n2n1 < 0.0f && k < 0.0f) || (n2n1 == 0.0f && k == 0.0f); // hacky way to check if edge is in the right direction

            if (right_direction) {
                // Edge is valid and should be kept
                valid_edges[valid_edge_count] = edge;
                valid_edge_count++;
            } else {
                // Edge is invalid and should be erased
                deleted_edges++;
            }
        }

        // Free the old edges array
        delete[] node.edges;

        // If there are valid edges, update the node's edges and count
        if (valid_edge_count > 0) {
            node.edges = new Edge[valid_edge_count];
            std::copy(valid_edges, valid_edges + valid_edge_count, node.edges);
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, reset the edges
            node.edges = nullptr;
            node.num_edges = 0;
        }

        // Free the temporary valid_edges array
        delete[] valid_edges;
    }

    std::cout << "Deleted edges with wrong direction: " << deleted_edges << std::endl;
}

void bfs_with_f_star_update(std::vector<Node>& graph, size_t start_node) {
    std::vector<bool> visited(graph.size(), false);
    std::vector<bool> fixed(graph.size(), false);
    std::vector<size_t> active_nodes;
    bool updates_occurred = true;
    int total_updates = 0;

    // Initialize start node
    visited[start_node] = true;
    graph[start_node].f_star = graph[start_node].f_init;
    active_nodes.push_back(start_node);

    float max_f_star = 0.0f;
    float min_f_star = 0.0f;
    float old_max_f_star = 0.0f;
    float old_min_f_star = 0.0f;

    int old_positive_updates = 0;
    int old_negative_updates = 0;

    // Loop until no updates occur in an entire cycle (positive + negative)
    while (updates_occurred) {
        std::cout << "New cycle." << std::endl;
        updates_occurred = false;

        // First iteration for positive `k` values
        int positive_updates = 0;
        size_t active_index = 0;
        while (active_index < active_nodes.size()) {
            size_t current = active_nodes[active_index];
            active_index++;
            if (graph[current].deleted) continue;

            for (size_t i = 0; i < graph[current].num_edges; ++i) {
                Edge& edge = graph[current].edges[i];

                // Not usable on the very outside, but might help inside the scroll?
                // if (edge.same_block) {
                //     continue;
                // }

                size_t target = edge.target_node;

                if (graph[target].deleted || edge.k < 0) continue;  // Skip negative `k` edges for this phase

                // Calculate the proposed f_star for the target node
                float proposed_f_star = graph[current].f_star + edge.k;
                float proposed_f_star_discrepancy = roundf((proposed_f_star - graph[target].f_init) / 360.0f) * 360.0f;
                proposed_f_star = graph[target].f_init + proposed_f_star_discrepancy;

                // Update conditions for positive `k`
                if (!fixed[target] && (!visited[target] || proposed_f_star > graph[target].f_star)) {
                    if (!visited[target]) {
                        visited[target] = true;
                        active_nodes.push_back(target);
                    }
                    graph[target].f_star = proposed_f_star;
                    updates_occurred = true;
                    positive_updates++;
                    if (proposed_f_star > max_f_star) {
                        max_f_star = proposed_f_star;
                    }
                    if (proposed_f_star < min_f_star) {
                        min_f_star = proposed_f_star;
                    }
                }
            }
        }

        for (size_t i = 0; i < graph.size(); ++i) {
            fixed[i] = visited[i];
        }

        std::cout << "Positive updates: " << positive_updates << std::endl;

        // Second iteration for negative `k` values
        int negative_updates = 0;
        active_index = 0;
        while (active_index < active_nodes.size()) {
            size_t current = active_nodes[active_index];
            active_index++;
            if (graph[current].deleted) continue;

            for (size_t i = 0; i < graph[current].num_edges; ++i) {
                Edge& edge = graph[current].edges[i];

                // Not usable on the very outside, but might help inside the scroll?
                // if (edge.same_block) {
                //     continue;
                // }

                size_t target = edge.target_node;

                if (graph[target].deleted || edge.k >= 0) continue;  // Skip positive `k` edges for this phase

                // Calculate the proposed f_star for the target node
                float proposed_f_star = graph[current].f_star + edge.k;
                float proposed_f_star_discrepancy = roundf((proposed_f_star - graph[target].f_init) / 360.0f) * 360.0f;
                proposed_f_star = graph[target].f_init + proposed_f_star_discrepancy;

                // Update conditions for negative `k`
                if (!fixed[target] && (!visited[target] || proposed_f_star < graph[target].f_star)) {
                    if (!visited[target]) {
                        visited[target] = true;
                        active_nodes.push_back(target);
                    }
                    graph[target].f_star = proposed_f_star;
                    updates_occurred = true;
                    negative_updates++;
                    if (proposed_f_star > max_f_star) {
                        max_f_star = proposed_f_star;
                    }
                    if (proposed_f_star < min_f_star) {
                        min_f_star = proposed_f_star;
                    }
                }
            }
        }

        for (size_t i = 0; i < graph.size(); ++i) {
            fixed[i] = visited[i];
        }

        std::cout << "Negative updates: " << negative_updates << std::endl;

        // Track total updates
        total_updates += positive_updates + negative_updates;

        // Print the number of updates for each direction
        std::cout << "Positive updates: " << positive_updates << ", Negative updates: " << negative_updates << std::endl;
        std::cout << "Max f_star: " << max_f_star << ", Min f_star: " << min_f_star << std::endl;

        // If no updates occurred in both phases, stop
        if (positive_updates == 0 && negative_updates == 0) {
            updates_occurred = false;
        }
        // Escape if we are stuck in a loop but mostly finished
        if (positive_updates == old_positive_updates && negative_updates == old_negative_updates && max_f_star == old_max_f_star && min_f_star == old_min_f_star) {
            std::cout << "Stuck in a loop. Exiting." << std::endl;
            break;
        }
        old_positive_updates = positive_updates;
        old_negative_updates = negative_updates;
        old_max_f_star = max_f_star;
        old_min_f_star = min_f_star;
    }

    // Delete non-visited nodes
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!visited[i]) {
            graph[i].deleted = true;
        }
    }

    // Print total updates made throughout the process
    std::cout << "Total updates: " << total_updates << std::endl;
}

float trilinear_interpolation(float x, float y, float z, float c000, float c100, float c010, float c110, float c001, float c101, float c011, float c111) {
    float x0 = 1.0f - x;
    float y0 = 1.0f - y;
    float z0 = 1.0f - z;
    
    float c00 = c000 * x0 + c100 * x;
    float c01 = c001 * x0 + c101 * x;
    float c10 = c010 * x0 + c110 * x;
    float c11 = c011 * x0 + c111 * x;

    float c0 = c00 * y0 + c10 * y;
    float c1 = c01 * y0 + c11 * y;

    return c0 * z0 + c1 * z;
}

float optimize_iron_scroll(std::vector<Node>& graph, int iterations) {
    // "iron" scroll to have "flat"/parallel sheets in the f_init, f_star, z 3d space
    float error_eps = 50.0f;
    float learning_rate = 1.0f;

    float total_error = 0.0f;
    int s_init = 30; // proper divisor of 360
    int s_star = 1080;
    int s_z = 100;
    float init_juggle = s_init / 2.0f;
    for (int iteration = 0; iteration < iterations; iteration++) {
        std::cout << "Iteration: " << iteration << std::endl;
        total_error = 0.0f;
        // Set up the meshgrid
        int min_f_init = std::numeric_limits<int>::max();
        int max_f_init = std::numeric_limits<int>::min();
        int min_f_star = std::numeric_limits<int>::max();
        int max_f_star = std::numeric_limits<int>::min();
        int min_z = std::numeric_limits<int>::max();
        int max_z = std::numeric_limits<int>::min();
        for (const auto& node : graph) {
            if (node.deleted) {
                continue;
            }
            min_f_init = std::min(min_f_init, static_cast<int>(node.f_init));
            max_f_init = std::max(max_f_init, static_cast<int>(node.f_init));
            min_f_star = std::min(min_f_star, static_cast<int>(node.f_star));
            max_f_star = std::max(max_f_star, static_cast<int>(node.f_star));
            min_z = std::min(min_z, static_cast<int>(node.z));
            max_z = std::max(max_z, static_cast<int>(node.z));
        }
        // Jiggle init value for smoother convergence
        if (iteration % 2 == 1) {
            min_f_init += init_juggle;
            min_f_star += s_star / 2;
            min_z += s_z / 2;
        }

        // Cast meshgrid over 3D space of size s
        int size_f_init = 360 / s_init - 1;
        int size_f_star = (max_f_star - min_f_star + 1) / s_star + 2;
        int size_z = (max_z - min_z + 1) / s_z + 2;

        std::cout << "Meshgrid size: " << size_f_init << " x " << size_f_star << " x " << size_z << std::endl;

        std::vector<std::vector<std::vector<std::vector<float>>>> meshgrid(size_f_init, std::vector<std::vector<std::vector<float>>>(size_f_star, std::vector<std::vector<float>>(size_z, std::vector<float>(0, 0.0f))));

        // For all nodes, calculate error a and store in the right mesh grid
        for (const auto& node : graph) {
            if (node.deleted) {
                continue;
            }
            // Iterate over all edges
            for (size_t i = 0; i < node.num_edges; ++i) {
                const Edge& edge = node.edges[i];
                // Only use other block edges
                // Error a = target f star - node f star - (target f init - node f init)
                float f_init_dist = graph[edge.target_node].f_init - node.f_init;
                // Adjust f init dist to take the circular natur of the scroll into consideration. 
                if (edge.k < 0 && f_init_dist > 0) {
                    f_init_dist -= 360;
                }
                if (edge.k > 0 && f_init_dist < 0) {
                    f_init_dist += 360;
                }
                if (edge.same_block) {
                    if (edge.k > 0) {
                        f_init_dist += 360;
                    }
                    if (edge.k < 0) {
                        f_init_dist -= 360;
                    }
                }
                // if (std::abs(f_init_dist + 360) < std::abs(f_init_dist)) {
                //     f_init_dist += 360;
                // }
                // if (std::abs(f_init_dist - 360) < std::abs(f_init_dist)) {
                //     f_init_dist -= 360;
                // }
                float error_a = graph[edge.target_node].f_star - node.f_star - f_init_dist;
                // take sign of error_a and set to 1/-1
                if (std::abs(error_a) > error_eps) {
                    error_a = error_eps * ((error_a > 0) - (error_a < 0));
                }
                // Store error a in the right mesh grid
                int init_index = (static_cast<int>(node.f_init) - min_f_init) / s_init;
                if (init_index > size_f_init - 1) {
                    init_index = 0;
                }
                if (init_index < 0) {
                    init_index = size_f_init - 1;
                }
                int star_index = (static_cast<int>(node.f_star) - min_f_star) / s_star;
                int z_index = (static_cast<int>(node.z) - min_z) / s_z;
                meshgrid[init_index][star_index][z_index].push_back(error_a);
            }
        }

        std::cout << "Meshgrid filled." << std::endl;
        
        // for each mesh grid calculate adjustment x
        std::vector<std::vector<std::vector<float>>> meshgrid_adjustments(size_f_init, std::vector<std::vector<float>>(size_f_star, std::vector<float>(size_z, 0.0f)));
        for (int i = 0; i < size_f_init; i++) {
            for (int j = 0; j < size_f_star; j++) {
                for (int k = 0; k < size_z; k++) {
                    if (meshgrid[i][j][k].empty()) {
                        continue;
                    }
                    // Calculate adjustment x
                    float sum_error_a = std::accumulate(meshgrid[i][j][k].begin(), meshgrid[i][j][k].end(), 0.0f);
                    float mean_error_a = sum_error_a / meshgrid[i][j][k].size();
                    total_error += std::abs(mean_error_a); // update total error
                    
                    float adjustment_x = mean_error_a;
                    // update each meshgrid point with learning_rate * x in f_star direction
                    meshgrid_adjustments[i][j][k] = learning_rate * adjustment_x;
                }
            }
        }
        // total_error /= (size_f_init * size_f_star * size_z); // Mean absolute error

        std::cout << "Total error: " << total_error << std::endl;

        // TODO: optional sanity checks for not inverting any part of the meshgrid

        // Elastic deformation of the meshgrid to make it flat with trilinear interpolation
        for (size_t i = 0; i < graph.size(); ++i) {
            Node& node = graph[i];
            if (node.deleted) {
                continue;
            }
            // Calculate the adjustment x for the node with trilinear interpolation
            float x = (node.f_init - min_f_init) / s_init - 0.5f;
            float y = (node.f_star - min_f_star) / s_star - 0.5f;
            float z = (node.z - min_z) / s_z - 0.5f;
            if (x < 0) {
                x += size_f_init;
            }
            if (y < 0) {
                y = 0;
            }
            if (z < 0) {
                z = 0;
            }
            int init_index = static_cast<int>(x);
            if (init_index > size_f_init - 1) {
                init_index = size_f_init - 1;
            }
            int init_index1 = (init_index + 1) % size_f_init;
            int star_index = static_cast<int>(y);
            int z_index = static_cast<int>(z);
            float adjustment_000 = meshgrid_adjustments[init_index][star_index][z_index];
            float adjustment_100 = meshgrid_adjustments[init_index1][star_index][z_index];
            float adjustment_010 = meshgrid_adjustments[init_index][star_index + 1][z_index];
            float adjustment_110 = meshgrid_adjustments[init_index1][star_index + 1][z_index];
            float adjustment_001 = meshgrid_adjustments[init_index][star_index][z_index + 1];
            float adjustment_101 = meshgrid_adjustments[init_index1][star_index][z_index + 1];
            float adjustment_011 = meshgrid_adjustments[init_index][star_index + 1][z_index + 1];
            float adjustment_111 = meshgrid_adjustments[init_index1][star_index + 1][z_index + 1];
            float adjustment_x = trilinear_interpolation(x-init_index, y-star_index, z-z_index, adjustment_000, adjustment_100, adjustment_010, adjustment_110, adjustment_001, adjustment_101, adjustment_011, adjustment_111);

            // Update the node f_star with the adjustment x
            node.f_star += adjustment_x;
        }

        // Print the total error
        std::cout << "Iteration: " << iteration << ", Total Error: " << total_error << std::endl;
    }

    // return mean abs x as error
    return total_error;
}

void solve_topological(std::vector<Node>& graph) {
    std::cout << "Solving topological." << std::endl;
    // optimize_f_star_for_good_edges(graph);
    std::cout << "Removing invalid edges." << std::endl;
    calculate_histogram_k(graph, "histogram_k_initial.png");
    calculate_histogram_edges(graph, "histogram_edges_intial.png");
    calculate_histogram(graph, "histogram_nodes_initial.png");
    // Remove wrong direction edges
    deleted_wrong_direction_edges(graph);
    std::cout << "Finding largest connected component." << std::endl;
    calculate_histogram_edges(graph, "histogram_edges_deleted.png");
    // Find largest connected component
    find_largest_connected_component(graph);
    std::cout << "Assigning winding angles." << std::endl;

    // append_reverse_edges(graph);

    // Assign winding angles
    bfs_with_f_star_update(graph, find_valid_start_node(graph));

    std::cout << "Done solving topological." << std::endl;

    // Validity checks
    // store only the valid indices to speed up the loop
    std::vector<size_t> valid_indices = get_valid_indices(graph);
    std::vector<size_t> valid_gt_indices = get_valid_gt_indices(graph, 10);

    // Print the error statistics
    auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
    auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
    std::cout << "After final assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;
}

// This is an example of a solve function that takes the graph and parameters as input
void solve(std::vector<Node>& graph, argparse::ArgumentParser* program, int num_iterations = 10000, bool adjust_lowest_only = false) {
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

    // Delete the existing winding angles directory if it exists
    if (fs::exists("winding_angles")) {
        fs::remove_all("winding_angles");
    }
    // Create a new histogram directory
    fs::create_directory("winding_angles");

    // Delete the existing winding angles directory if it exists
    if (fs::exists("side")) {
        fs::remove_all("side");
    }
    // Create a new histogram directory
    fs::create_directory("side");

    // Generate spring constants starting from 5.0 with 12 steps
    // std::vector<float> spring_constants = generate_spring_constants(spring_constant, steps);
    std::vector<float> spring_constants = {2.0f, 8.0f, 1.0f, 2.0f, 1.0f, 1.0f};

    // Calculate the number of digits needed for padding
    int max_index_digits = static_cast<int>(std::log10(spring_constants.size())) + 1;
    int max_iter_digits = static_cast<int>(std::log10(num_iterations - 1)) + 1;

    // store only the valid indices to speed up the loop
    std::vector<size_t> valid_indices = get_valid_indices(graph);
    std::vector<size_t> valid_gt_indices = get_valid_gt_indices(graph, 10);

    float invalid_edge_threshold = 1.5f;

    int edges_deletion_round = 0;
    while (true) {
        // Do 2 rounds of edge deletion
        if (edges_deletion_round > 0 || invalid_edge_threshold <= 0.05) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations = num_iterations * 3;
            spring_constant = 1.0f;

            break;
        }
        solve_gpu_session(graph, edges_deletion_round, video_mode, max_index_digits, max_iter_digits, num_iterations, o, spring_factor, steps, spring_constants, valid_indices, iterations_factor, o_factor, estimated_windings, histogram_dir, adjust_lowest_only);
        std::cout << "After solving with spring constant: " << spring_constant << ", Iterations: " << num_iterations << std::endl;
        // // After first edge deletion round remove the invalid edges
        // if (edges_deletion_round >= 0) {
        //     // Remove edges with too much difference between f_star and k
        //     remove_invalid_edges(graph, invalid_edge_threshold);
        // }
        std::cout << "After removing invalid edges with threshold: " << invalid_edge_threshold << std::endl;
        find_largest_connected_component(graph);
        std::cout << "After finding largest connected component." << std::endl;
        // Update the valid indices
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph);
        
        // Reduce the threshold by 20% each time
        invalid_edge_threshold *= 0.7f;
        invalid_edge_threshold -= 0.01f;
        if (invalid_edge_threshold < 0.30) {
            invalid_edge_threshold = 0.30;
        }
        std::cout << "Reducing invalid edges threshold to: " << invalid_edge_threshold << std::endl;
        // // Assign winding angles again after removing invalid edges
        // float scale = calculate_scale(graph, estimated_windings);
        
        // // Detect folds
        // if (edges_deletion_round == 0) {
        //     // Solve the fold detection
        //     solve_fold(graph, program, get_valid_indices(graph), 10000);
        // }

        // Assign winding angles again after removing invalid edges
        // assign_winding_angles(graph, scale);
        // check the winding angles on f_star
        // for (size_t i = 0; i < graph.size(); ++i) {
        //     if (graph[i].deleted) {
        //         continue;
        //     }
        //     closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
        // }

        // store only the valid indices to speed up the loop
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph, 10);
        
        // Print the error statistics
        auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
        auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
        std::cout << "After assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;

        // Save the graph back to a binary file
        save_graph_to_binary("temp_output_graph.bin", graph);

        edges_deletion_round++;
    }
    // // Assign winding angles to the graph
    // float scale = calculate_scale(graph, estimated_windings);
    // // assign_winding_angles(graph, scale);
    // // check the winding angles on f_star
    // for (size_t i = 0; i < graph.size(); ++i) {
    //     if (graph[i].deleted) {
    //         continue;
    //     }
    //     closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    // }

    // Print the error statistics
    auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
    auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
    std::cout << "After final assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;
}

void invert_winding_direction_graph(std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];
            if (edge.same_block) {
                float turns = edge.k / 360;
                turns = std::round(turns);
                if (std::abs(turns) > 1 || std::abs(turns) == 0) {
                    std::cout << "Inverting winding direction failed, turns: " << turns << std::endl;
                }
                edge.k = edge.k - (2.0f * 360.0f * turns);
            }
        }
    }
}

void auto_winding_direction(std::vector<Node>& graph, argparse::ArgumentParser* program) {
    std::cout << "Auto Winding Direction" << std::endl;
    
    // Make a copy of the graph
    std::vector<Node> auto_graph = cloneGraph(graph);
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
    
    std::vector<Node> auto_graph_other_block = cloneGraph(auto_graph);
    invert_winding_direction_graph(auto_graph_other_block); // build inverted other block graph

    // solve
    int auto_num_iterations = program->get<int>("--auto_num_iterations");
    if (auto_num_iterations == -1) {
        auto_num_iterations = program->get<int>("--num_iterations");
    }
    solve(auto_graph, program, auto_num_iterations);
    solve(auto_graph_other_block, program, auto_num_iterations);

    // Exact matching score
    float exact_score = exact_matching_score(auto_graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;
    float exact_score_other_block = exact_matching_score(auto_graph_other_block);
    std::cout << "Exact Matching Score Other Block: " << exact_score_other_block << std::endl;

    // Remove all same_block edges for further comparison
    for (auto& node : auto_graph_other_block) {
        int valid_edge_count = 0;
        
        // Count the number of valid edges (those not marked as same_block)
        for (int i = 0; i < node.num_edges; ++i) {
            if (!node.edges[i].same_block) {
                ++valid_edge_count;
            }
        }

        // If there are valid edges, allocate a new array for valid edges
        if (valid_edge_count > 0) {
            Edge* valid_edges = new Edge[valid_edge_count];
            int index = 0;

            // Copy valid edges to the new array
            for (int i = 0; i < node.num_edges; ++i) {
                if (!node.edges[i].same_block) {
                    valid_edges[index++] = node.edges[i];
                }
            }

            // Free the old edges array
            delete[] node.edges;

            // Assign the new array to the node
            node.edges = valid_edges;
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, free the old edges array and set edges to nullptr
            delete[] node.edges;
            node.edges = nullptr;
            node.num_edges = 0;
        }
    }

    for (auto& node : auto_graph) {
        int valid_edge_count = 0;
        
        // Count the number of valid edges (those not marked as same_block)
        for (int i = 0; i < node.num_edges; ++i) {
            if (!node.edges[i].same_block) {
                ++valid_edge_count;
            }
        }

        // If there are valid edges, allocate a new array for valid edges
        if (valid_edge_count > 0) {
            Edge* valid_edges = new Edge[valid_edge_count];
            int index = 0;

            // Copy valid edges to the new array
            for (int i = 0; i < node.num_edges; ++i) {
                if (!node.edges[i].same_block) {
                    valid_edges[index++] = node.edges[i];
                }
            }

            // Free the old edges array
            delete[] node.edges;

            // Assign the new array to the node
            node.edges = valid_edges;
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, free the old edges array and set edges to nullptr
            delete[] node.edges;
            node.edges = nullptr;
            node.num_edges = 0;
        }
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

    //Free auto graphs
    free_graph(auto_graph);
    free_graph(auto_graph_other_block);
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
    // Update the lines coordinate system to mask3d system
    for (size_t i = 0; i < fix_lines_z.size(); ++i) {
        fix_lines_z[i] = (fix_lines_z[i] + 500) / 4.0f;
    }

    // Fix lines: if the z value of the node is in the fix_lines_z +- 25 and ground truth (gt) is available, set the gt to f_star and fix it
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

    // Get the min and max f_star using ground truth values
    float f_min = min_f_star(graph, true);
    float f_max = max_f_star(graph, true);

    // Fix windings based on the number of windings to fix
    float start_winding = f_min;
    float end_winding = f_max;
    
    if (fix_windings < 0) {
        start_winding = f_max - 360.0f * std::abs(fix_windings);
    } else if (fix_windings > 0) {
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

    // Fix all ground truth nodes if the flag is set
    if (fix_all) {
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].deleted || !graph[i].gt) {
                continue;
            }
            graph[i].fixed = true;
        }
    }

    // Now fix good edges and delete bad ones
    int good_edges = 0;
    int bad_edges = 0;

    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }

        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];

            if (graph[edge.target_node].deleted) {
                continue;
            }

            // If both the current node and the target node are fixed and have ground truth, evaluate the edge
            if (graph[i].fixed && graph[edge.target_node].fixed && graph[i].gt && graph[edge.target_node].gt) {
                // Check if edge has the correct k value
                float diff = graph[edge.target_node].gt_f_star - graph[i].gt_f_star;
                float k = edge.k;

                // Check if the edge k value is close to the ground truth difference
                if (std::abs(k - diff) < 0.1) {
                    // Good edge: set high certainty
                    edge.certainty = edge_good_certainty;
                    good_edges++;
                } else {
                    // Bad edge: set certainty to zero
                    edge.certainty = 0.0f;
                    bad_edges++;
                }

                // Fix the edge
                edge.fixed = true;
            }
        }
    }

    std::cout << "Fixed " << good_edges << " good edges and deleted " << bad_edges << " bad edges." << std::endl;
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
    float spring_factor = 10.0f;
    int steps = 5;
    int z_min = -2147483648;
    int z_max = 2147483647;
    float same_winding_factor = 0.10f;

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

    // calculate_and_normalize_edge_certainties(graph, 720);

    calculate_histogram_edges(graph, "histogram_edges_start.png");
    std::cout << "Initial Graph Size: " << graph.size() << std::endl;

    // invert graph
    if (program.get<bool>("--invert")) {
        std::cout << "Inverting the winding direction of the graph" << std::endl;
        invert_winding_direction_graph(graph);
    }

    // // Calculate the exact matching loss
    // float exact_score = exact_matching_score(graph);
    // std::cout << "Exact Matching Score: " << exact_score << std::endl;

    // // Calculate the approximate matching loss
    // float approx_loss = approximate_matching_loss(graph, 1.0f);
    // std::cout << "Approximate Matching Loss: " << approx_loss << std::endl;

    // Calculate and display the histogram of f_star values
    // calculate_histogram(graph);

    // Check if the ground truth graph construction is enabled
    if (program.get<bool>("--gt_graph")) {
        // Construct the ground truth graph
        construct_ground_truth_graph(graph);
    }
    else {
        // Remove nodes with degree lower than deg
        // remove_low_degree_nodes(graph, 2);
        find_largest_connected_component(graph);

        // Automatically determing winding direction
        if (program.get<bool>("--auto")) {
            auto_winding_direction(graph, &program);
        }
        // Solve the problem using a solve function
        // fix_gt_parts(graph, {6000}, -10, 10.0f * max_certainty, program.get<bool>("--fix_all_gt"));
        num_iterations = program.get<int>("--num_iterations");
        if (true) {
            solve(graph, &program, num_iterations);

            // Save complete graph computation to a binary file
            saveGraph(graph, "graph_iterations_solution.bin");
        }
        else {
            // Load graph from binary file
            graph = loadGraph("graph_iterations_solution.bin");
        }

        // // Remove start and end 5% of scroll, run solver again
        // {
        //     float min_f_star_value = min_f_star(graph);
        //     float max_f_star_value = max_f_star(graph);
        //     // Delete lowest and highest 5% of f_star values
        //     float lowest_f_star = min_f_star_value + 0.05f * (max_f_star_value - min_f_star_value);
        //     float highest_f_star = max_f_star_value - 0.05f * (max_f_star_value - min_f_star_value);
        //     for (size_t i = 0; i < graph.size(); ++i) {
        //         if (graph[i].f_star < lowest_f_star || graph[i].f_star > highest_f_star) {
        //             graph[i].deleted = true;
        //         }
        //         graph[i].f_tilde = graph[i].f_init;
        //         graph[i].f_star = graph[i].f_init;
        //     }

        //     int count_good = 0;
        //     int count_bad = 0;
        //     // Increase certainty of top 3 edges per node
        //     for (size_t i = 0; i < graph.size(); ++i) {
        //         if (graph[i].deleted) {
        //             continue;
        //         }
        //         if (graph[i].num_edges < 6) {
        //             continue;
        //         }
                
        //         for (size_t j = 0; j < graph[i].num_edges; ++j) {
        //             if (graph[graph[i].edges[j].target_node].deleted) {
        //                 continue;
        //             }
        //             if (!graph[i].edges[j].same_block) {
        //                 continue;
        //             }
                    
        //             Edge& edge = graph[i].edges[j];
        //             float dir1 = graph[edge.target_node].f_star - graph[i].f_star;
        //             float threshold = 1.0f;
        //             if (dir1 * edge.k < 0.0f) {
        //                 if (std::abs(dir1) > threshold) {
        //                     edge.certainty = 0.0f;
        //                     if (edge.gt_edge) {
        //                         if (edge.good_edge) {
        //                             count_bad++;
        //                         }
        //                         else {
        //                             count_good++;
        //                         }
        //                     }
        //                 // Delete nodes
        //                 graph[i].deleted = true;
        //                 graph[edge.target_node].deleted = true;
        //                 }
        //             }
        //         }
                
        //     }
        //     std::cout << "Deleted edges of very close nodes Good: " << count_good << ", Bad: " << count_bad << std::endl;

        //     solve(graph, &program, num_iterations);
        // }

        create_ply_pointcloud(graph, "initial_pointcloud.ply");
        create_ply_pointcloud_side(graph, "initial_side_pointcloud.ply");
        // optimize_iron_scroll(graph, 150);
        // solve(graph, &program, num_iterations, true);
        // plot_nodes(graph, "final_nodes_plot.png");
        // create_ply_pointcloud(graph, "final_pointcloud.ply");
        // Finish the graph with topological sort
        // solve_topological(graph);
    }

    // print the min and max f_star values
    std::cout << "Min f_star: " << min_f_star(graph) << std::endl;
    std::cout << "Max f_star: " << max_f_star(graph) << std::endl;

    // Save the graph back to a binary file
    std::string output_graph_file = program.get<std::string>("--output_graph");
    save_graph_to_binary(output_graph_file, graph);

    // Calculate the exact matching loss
    // float exact_score2 = exact_matching_score(graph);
    // std::cout << "Exact Matching Score: " << exact_score2 << std::endl;

    // // Calculate the approximate matching loss
    // float approx_loss2 = approximate_matching_loss(graph, 1.0f);
    // std::cout << "Approximate Matching Loss: " << approx_loss2 << std::endl;
    create_video_from_histograms("winding_angles", "winding_angles_plot.avi", 20);
    create_video_from_histograms("side", "sides_plot.avi", 20);
    if (program.get<bool>("--video")) {
        // Calculate final histogram after all iterations
        calculate_histogram(graph, "final_histogram.png");
        // After generating all histograms, create a final video from the images
        create_video_from_histograms("histogram", "winding_angle_histogram.avi", 60);
        create_video_from_histograms("winding_angles", "winding_angle_histogram.avi", 60);
    }

    return 0;
}

// Example command to run the program: ./build/graph_problem --input_graph graph.bin --output_graph output_graph.bin --auto --auto_num_iterations 2000 --video --z_min 5000 --z_max 7000 --num_iterations 5000 --estimated_windings 160 --steps 3 --spring_constant 1.2