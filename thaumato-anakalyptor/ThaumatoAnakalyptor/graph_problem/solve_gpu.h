// solve_gpu.h
#ifndef SOLVE_GPU_H
#define SOLVE_GPU_H

#include <vector>
#include "node_structs.h"  // Assuming Node and Edge are declared here
#include <string>

// Declaration of the GPU solver function
void solve_gpu_session(std::vector<Node>& graph, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, int num_iterations, float o, float spring_factor, int steps, std::vector<float>& spring_constants, std::vector<size_t>& valid_indices, int iterations_factor, float o_factor, int estimated_windings, const std::string& histogram_dir, bool adjust_lowest_only);
// void solve_gpu(std::vector<Node>& graph, int i, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, bool first_estimated_iteration, int estimated_windings, Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, int num_nodes);
std::vector<Node> run_solver(std::vector<Node>& graph, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, int seed_node, float other_block_factor, int down_index, int up_index, int side_fix_nr, float std_target, float std_target_step, bool increase_same_block_weight);
std::vector<Node> run_solver_f_star(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float o, float spring_constant, bool visualize);
std::vector<Node> run_solver_ring(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float other_block_factor, float std_target, 
                                    float std_target_step, bool increase_same_block_weight, bool convergence_speedup, float convergence_thresh, bool wiggle, bool standard_winding_direction, float scale_left, float scale_right, bool enable_delete_nodes);
std::vector<Node> run_solver_winding_number(std::vector<Node>& graph, int num_iterations, std::vector<size_t>& valid_indices, Edge** h_all_edges, float** h_all_sides, int i_round, float other_block_factor, int seed_node, int side_fix_nr, bool display);

void filter_graph_sides(std::vector<Node>& graph);
void filter_graph_f_star(std::vector<Node>& graph, float f_star_threshold);

float min_f_star(const std::vector<Node>& graph, bool use_gt = false);
float max_f_star(const std::vector<Node>& graph, bool use_gt = false);

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000);
void calculate_happyness_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000);
void calculate_histogram_k(const std::vector<Node>& graph, const std::string& filename, int num_buckets = 1000);
void calculate_histogram_edges_f_init(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000);
void calculate_histogram_edges(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000);
void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps = 10);
void plot_nodes(const std::vector<Node>& graph, const std::string& filename);
void create_ply_pointcloud(const std::vector<Node>& graph, const std::string& filename);
void create_ply_pointcloud_side(const std::vector<Node>& graph, const std::string& filename);
void create_ply_pointcloud_sides(const std::vector<Node>& graph, const std::string& filename);


void spanning_tree_winding_number(std::vector<Node>& graph);

#endif // SOLVE_GPU_H
