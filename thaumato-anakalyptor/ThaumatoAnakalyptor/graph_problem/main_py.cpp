/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
#include "mst_union.h"

namespace fs = std::filesystem;
namespace py = pybind11;

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

void flip_winding_direction_graph(std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];
            edge.k = -edge.k;
        }
    }
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

// solver class
class Solver {
    // o, graph path
    float o;
    std::vector<Node> graph;
    std::string graph_path;
    Edge* h_all_edges = nullptr; // edges memory managment
    float* h_all_sides = nullptr; // sides memory managment
    size_t nr_nodes = 0;
    public:
        // constructor
        Solver(std::string graph_path, float o=2.0, int z_min=-2147483648, int z_max=2147483647) : o(o), graph_path(graph_path) {
            // Load graph from binary file
            auto [graph_, max_certainty] = load_graph_from_binary(graph_path, true, (static_cast<float>(z_min) + 500) / 4, (static_cast<float>(z_max) + 500) / 4, 1.0f, false);
            graph = graph_;
            // calculate_histogram_edges(graph, "histogram_edges_start.png");
            largest_connected_component();
        }
        void invert_graph() {
            invert_winding_direction_graph(graph);
        }
        void flip_graph() {
            flip_winding_direction_graph(graph);
        }
        // solve function
        void solve(int num_iterations, float spring_constant, int i_round = 1, int seed_node = 100, float other_block_factor = 1.0f, int down_index = 0, int up_index = 0, int side_fix_nr = 0, float std_target = 0.013f, float std_target_step = 0.0f, bool increase_same_block_weight = true) {
            // store only the valid indices to speed up the loop
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            // solve the graph
            graph = run_solver(graph, o, spring_constant, num_iterations, valid_indices, &h_all_edges, &h_all_sides, i_round, seed_node, other_block_factor, down_index, up_index, side_fix_nr, std_target, std_target_step, increase_same_block_weight);
        }
        void solve_f_star(int num_iterations, float spring_constant, int i_round = 1, float o_ = 0.0f, bool visualize = false) {
            // use the f_star solver for the intermediate solution
            // store only the valid indices to speed up the loop
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            graph = run_solver_f_star(graph, num_iterations, valid_indices, &h_all_edges, &h_all_sides, i_round, o_, spring_constant, visualize);
        }
        void filter_f_star() {
            // Filters the graph edges based on the f star solution
            filter_graph_f_star(graph, 8 * 360.0f);
            // get largest connected component
            largest_connected_component();
        }
        void solve_union() {
            // use the union solver for the final solution
            solveGraphUnion(graph);
        }
        void solve_ring(int num_iterations, int i_round = 1, float other_block_factor = 1.0f, float std_target = 0.013f, float std_target_step = 0.0f, 
                        bool increase_same_block_weight = true, bool convergence_speedup = false, float convergence_thresh = 0.0f, bool wiggle = true, 
                        bool standard_winding_direction = false, float scale_left=1.0f, float scale_right=1.0f, bool enable_delete_nodes = true) {
            // use the ring solver to refine the graph edges with before running the winding number solver
            // store only the valid indices to speed up the loop
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            graph = run_solver_ring(graph, num_iterations, valid_indices, &h_all_edges, &h_all_sides, i_round, other_block_factor, std_target, std_target_step, increase_same_block_weight, 
            convergence_speedup, convergence_thresh, wiggle, standard_winding_direction, scale_left, scale_right, enable_delete_nodes);
        }
        void filter_graph_ring() {
            // Filters the graph edges based on the ring solver solution
            filter_graph_sides(graph);
            // get largest connected component
            largest_connected_component();
        }
        void set_labels(std::vector<float> gt_winding_nrs, std::vector<bool> gt) {
            // set labels
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                graph[index].winding_nr_old = gt_winding_nrs[i];
                graph[index].winding_nr = graph[index].winding_nr_old;                
                graph[index].fixed = gt[i];
            }
        }
        std::vector<float> get_labels() {
            // get labels
            std::vector<float> labels;
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                labels.push_back(graph[index].winding_nr_old);
            }
            return labels;
        }
        void solve_winding_number(int num_iterations, int i_round = 1, int seed_node = 100, float other_block_factor = 1.0f, int side_fix_nr = 0, bool display = true) {
            // use the winding number solver for the final solution
            // store only the valid indices to speed up the loop
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            graph = run_solver_winding_number(graph, num_iterations, valid_indices, &h_all_edges, &h_all_sides, i_round, other_block_factor, seed_node, side_fix_nr, display);
        }
        void assign_f_star(std::vector<Node>& graph) {
            // assign winding angles to f star
            for (size_t i = 0; i < graph.size(); ++i) {
                if (graph[i].deleted) {
                    continue;
                }
                graph[i].f_star = graph[i].winding_nr * 360.0f + graph[i].f_init;
            }
        }
        void save_solution(std::string graph_path) {
            // Extract the final winding angle assignment from the winding number solver and save it
            assign_f_star(graph);
            // save the graph f star solution
            save_graph_to_binary(graph_path, graph);
        }
        void load_graph(std::string graph_path) {
            graph = loadGraph(graph_path);
        }
        void save_graph(std::string graph_path) {
            saveGraph(graph, graph_path);
        }
        void set_connectivity(std::vector<float> connectivity) {
            // set connectivity
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                graph[index].connectivity = connectivity[i];
            }
        }
        std::vector<float> get_connectivity() {
            // get connectivity
            std::vector<float> connectivity;
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                connectivity.push_back(graph[index].connectivity);
            }
            return connectivity;
        }
        std::vector<float> get_sides() {
            // get connectivity
            std::vector<float> sides;
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                sides.push_back(graph[index].side);
            }
            return sides;
        }
        void set_side_numbers(std::vector<float> side_numbers) {
            // set connectivity
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                graph[index].side_number = side_numbers[i];
            }
        }
        std::vector<float> get_side_numbers() {
            // get connectivity
            std::vector<float> side_numbers;
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                side_numbers.push_back(graph[index].side_number);
            }
            return side_numbers;
        }
        void reset_sides() {
            for (size_t i = 0; i < graph.size(); ++i) {
                graph[i].side = 0;
            }
        }
        void reset_happiness() {
            for (size_t i = 0; i < graph.size(); ++i) {
                graph[i].happiness = 0.0f;
                graph[i].happiness_old = 0.0f;
            }
        }
        void set_z_range(float z_min, float z_max, bool undelete_inside = false) {
            // set z range
            for (size_t i = 0; i < graph.size(); ++i) {
                float z = graph[i].z;
                z = (4.0f * z - 500);
                if (z < z_min || z > z_max) {
                    graph[i].deleted = true;
                }
                else if (undelete_inside) {
                    graph[i].deleted = false;
                }
            }
        }
        std::vector<std::vector<float>> get_positions() {
            // get positions
            std::vector<std::vector<float>> positions;
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t index = valid_indices[i];
                std::vector<float> position = {graph[index].f_star / 20.0f, graph[index].f_init, graph[index].z};
                positions.push_back(position);
            }
            return positions;
        }
        void largest_connected_component() {
            find_largest_connected_component(graph);
            nr_nodes = get_valid_indices(graph).size();
        }
        void unfix() {
            // unfix all nodes
            for (size_t i = 0; i < graph.size(); ++i) {
                graph[i].fixed = false;
            }
        }
        void update_edges(int up_index) {
            // update edges
            std::vector<size_t> valid_indices = get_valid_indices(graph);
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                Node node = graph[valid_indices[i]];
                // check the edges with the side information
                for (size_t j = 0; j < node.num_edges; ++j) {
                    Edge edge = node.edges[j];
                    Node target_node = graph[edge.target_node];
                    if (target_node.deleted) {
                        continue;
                    }
                    bool seam = std::abs(node.f_init - target_node.f_init) >= 180.0f; // seam winding
                    if (seam) {
                        continue;
                    }
                    int side_number_d = node.side_number - target_node.side_number;
                    if (side_number_d > (up_index / 2 + 1)) {
                        side_number_d = up_index + 1 - side_number_d;
                    }
                    else if (side_number_d < -(up_index / 2 + 1)) {
                        side_number_d = - side_number_d - (up_index + 1);
                    }
                    if (edge.same_block) { // same block edges
                        if (edge.k > 0) {
                            if (side_number_d > 2) {
                                edge.certainty *= 0.7f;
                            }
                        }
                        else {
                            if (side_number_d < -2) {
                                edge.certainty *= 0.7f;
                            }
                        }
                        if (edge.k * (node.f_star - target_node.f_star) < 0 && std::abs(side_number_d) > 1) { // wrong direction of edge, decrease certainty
                            edge.certainty *= 0.9f;
                        }
                    }
                    else { // other block edges (on the same sheet)
                        if (std::abs(side_number_d) > 4) {
                            edge.certainty *= 0.9f;
                        }
                    }
                }
            }
        }
        int get_nr_nodes() {
            return nr_nodes;
        }
        void generate_video(std::string frame_dir, std::string video_path, int fps) {
            // create video from histograms
            create_video_from_histograms(frame_dir, video_path, fps);
        }
        void generate_ply(std::string ply_path) {
            // create ply point cloud
            create_ply_pointcloud_sides(graph, ply_path);
            std::string ply_path_gt;
            // path.ply -> path_gt.ply
            ply_path_gt = ply_path;
            ply_path_gt.replace(ply_path_gt.end() - 4, ply_path_gt.end(), "_gt.ply");
            create_ply_pointcloud(graph, ply_path_gt);
        }
        void solution_loss(bool use_sides = true) {
            std::vector<Node> graph_copy = graph;
            if (use_sides) {
                spanning_tree_winding_number(graph_copy);
            }
            assign_f_star(graph_copy);
            // calculate solution loss
            float loss = approximate_matching_loss(graph_copy);
            std::cout << "Solution loss: " << loss << std::endl;
            // Print the error statistics
            std::vector<size_t> valid_gt_indices = get_valid_gt_indices(graph_copy);
            auto [mean, min, q1, median, q3, max] = computeErrorStats(graph_copy, valid_gt_indices);
            auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph_copy, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
            std::cout << "After assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;
        }
};

PYBIND11_MODULE(graph_problem_gpu_py, m) {
    m.doc() = "pybind11 module for python solver class";

    // add Solver class
    py::class_<Solver>(m, "Solver")
        // solver with optional arguments and standard o=1.2 spring_constant=2.0
        .def(py::init<std::string, float, int, int>(),
            "Class method to solve the graph with manual input",
            py::arg("graph_path"),
            py::arg("o") = 2.0f,
            py::arg("z_min") = -2147483648,
            py::arg("z_max") = 2147483647)
        .def("invert_graph", &Solver::invert_graph,
            "Method to invert the winding direction of the graph")
        .def("flip_graph", &Solver::flip_graph,
            "Method to flip the winding direction of the graph")
        .def("solve", &Solver::solve,
            "Method to solve the graph with manual input",
            py::arg("num_iterations") = 10000,
            py::arg("spring_constant") = 1.2f,
            py::arg("i_round") = 1,
            py::arg("seed_node") = 100,
            py::arg("other_block_factor") = 1.0f,
            py::arg("down_index") = 0,
            py::arg("up_index") = 0,
            py::arg("side_fix_nr") = 0,
            py::arg("std_target") = 0.013f,
            py::arg("std_target_step") = 0.0f,
            py::arg("increase_same_block_weight") = false)
        .def("solve_f_star", &Solver::solve_f_star,
            "Method to intermediately solve the graph with a mean winding angle approach",
            py::arg("num_iterations") = 10000,
            py::arg("spring_constant") = 1.2f,
            py::arg("i_round") = -1,
            py::arg("o") = 0.0f,
            py::arg("visualize") = false)
        .def("filter_f_star", &Solver::filter_f_star,
            "Method to filter the graph edges after running the f star solver")
        .def("solve_union", &Solver::solve_union,
            "Method to solve the graph with the union solver")
        .def("solve_ring", &Solver::solve_ring,
            "Method to refine the graph edges with the ring solver",
            py::arg("num_iterations") = 10000,
            py::arg("i_round") = 1,
            py::arg("other_block_factor") = 1.0f,
            py::arg("std_target") = 0.013f,
            py::arg("std_target_step") = 0.0f,
            py::arg("increase_same_block_weight") = false,
            py::arg("convergence_speedup") = false,
            py::arg("convergence_thresh") = 0.0f,
            py::arg("wiggle") = true,
            py::arg("standard_winding_direction") = false,
            py::arg("scale_left") = 1.0f,
            py::arg("scale_right") = 1.0f,
            py::arg("enable_delete_nodes") = true)
        .def("filter_graph_ring", &Solver::filter_graph_ring,
            "Method to filter the graph edges after running the ring solver")
        .def("set_labels", &Solver::set_labels,
            "Method to set the labels of the graph",
            py::arg("gt_f_stars"),
            py::arg("gt"))
        .def("get_labels", &Solver::get_labels,
            "Method to get the labels of the graph")
        .def("solve_winding_number", &Solver::solve_winding_number,
            "Method to final solve the graph with the winding number solver",
            py::arg("num_iterations") = 10000,
            py::arg("i_round") = 1,
            py::arg("seed_node") = 100,
            py::arg("other_block_factor") = 1.0f,
            py::arg("side_fix_nr") = 0,
            py::arg("display") = true)
        .def("save_solution", &Solver::save_solution,
            "Method to save the final solution of the graph",
            py::arg("graph_path"))
        .def("set_connectivity", &Solver::set_connectivity)
        .def("get_connectivity", &Solver::get_connectivity)
        .def("get_sides", &Solver::get_sides)
        .def("set_side_numbers", &Solver::set_side_numbers)
        .def("get_side_numbers", &Solver::get_side_numbers)
        .def("reset_sides", &Solver::reset_sides)
        .def("reset_happiness", &Solver::reset_happiness)
        .def("set_z_range", &Solver::set_z_range,
            "Method to set the z range of the graph",
            py::arg("z_min"),
            py::arg("z_max"),
            py::arg("undelete_inside") = false)
        .def("get_positions", &Solver::get_positions)
        .def("load_graph", &Solver::load_graph,
            "Method to load the graph from a binary file",
            py::arg("graph_path"))
        .def("save_graph", &Solver::save_graph)
        .def("largest_connected_component", &Solver::largest_connected_component)
        .def("unfix", &Solver::unfix)
        .def("update_edges", &Solver::update_edges)
        .def("get_nr_nodes", &Solver::get_nr_nodes)
        .def("generate_video", &Solver::generate_video)
        .def("generate_ply", &Solver::generate_ply)
        .def("solution_loss", &Solver::solution_loss,
            "Method to calculate the solution loss",
            py::arg("use_sides") = true);
}

// Example command to run the program: ./build/graph_problem --input_graph graph.bin --output_graph output_graph.bin --auto --auto_num_iterations 2000 --video --z_min 5000 --z_max 7000 --num_iterations 5000 --estimated_windings 160 --steps 3 --spring_constant 1.2
