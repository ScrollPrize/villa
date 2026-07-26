#include "node_structs.h"

#include <fstream>
#include <stdexcept>
#include <iostream>

bool operator==(const std::vector<Node>& graph1, const std::vector<Node>& graph2) {
    if (graph1.size() != graph2.size()) {
        return false;
    }
    for (size_t i = 0; i < graph1.size(); ++i) {
        if (!(graph1[i] == graph2[i])) {
            return false;
        }
    }
    return true;
}

// Function to save the graph to a binary file
void saveGraph(const std::vector<Node>& nodes, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Could not open file for writing");
    }

    size_t num_nodes = nodes.size();
    outFile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

    for (const auto& node : nodes) {
        outFile.write(reinterpret_cast<const char*>(&node.z), sizeof(node.z));
        outFile.write(reinterpret_cast<const char*>(&node.f_init), sizeof(node.f_init));
        outFile.write(reinterpret_cast<const char*>(&node.f_tilde), sizeof(node.f_tilde));
        outFile.write(reinterpret_cast<const char*>(&node.f_star), sizeof(node.f_star));
        outFile.write(reinterpret_cast<const char*>(&node.happiness), sizeof(node.happiness));
        outFile.write(reinterpret_cast<const char*>(&node.happiness_old), sizeof(node.happiness_old));
        outFile.write(reinterpret_cast<const char*>(&node.gt), sizeof(node.gt));
        outFile.write(reinterpret_cast<const char*>(&node.gt_f_star), sizeof(node.gt_f_star));
        outFile.write(reinterpret_cast<const char*>(&node.deleted), sizeof(node.deleted));
        outFile.write(reinterpret_cast<const char*>(&node.fixed), sizeof(node.fixed));
        outFile.write(reinterpret_cast<const char*>(&node.fold), sizeof(node.fold));
        outFile.write(reinterpret_cast<const char*>(&node.connectivity), sizeof(node.connectivity));
        outFile.write(reinterpret_cast<const char*>(&node.side), sizeof(node.side));
        outFile.write(reinterpret_cast<const char*>(&node.side_number), sizeof(node.side_number));
        outFile.write(reinterpret_cast<const char*>(&node.wnr_side), sizeof(node.wnr_side));
        outFile.write(reinterpret_cast<const char*>(&node.wnr_side_old), sizeof(node.wnr_side_old));
        outFile.write(reinterpret_cast<const char*>(&node.total_wnr_side), sizeof(node.total_wnr_side));
        outFile.write(reinterpret_cast<const char*>(&node.winding_nr), sizeof(node.winding_nr));
        outFile.write(reinterpret_cast<const char*>(&node.winding_nr_old), sizeof(node.winding_nr_old));
        for (int j = 0; j < Node::sides_nr; ++j) {
            outFile.write(reinterpret_cast<const char*>(&node.sides[j]), sizeof(node.sides[j]));
            outFile.write(reinterpret_cast<const char*>(&node.sides_old[j]), sizeof(node.sides_old[j]));
        }

        outFile.write(reinterpret_cast<const char*>(&node.num_same_block_edges), sizeof(node.num_same_block_edges));
        outFile.write(reinterpret_cast<const char*>(&node.num_edges), sizeof(node.num_edges));
        for (int i = 0; i < node.num_edges; ++i) {
            const Edge& edge = node.edges[i];
            outFile.write(reinterpret_cast<const char*>(&edge.target_node), sizeof(edge.target_node));
            outFile.write(reinterpret_cast<const char*>(&edge.certainty), sizeof(edge.certainty));
            outFile.write(reinterpret_cast<const char*>(&edge.certainty_factor), sizeof(edge.certainty_factor));
            outFile.write(reinterpret_cast<const char*>(&edge.certainty_factored), sizeof(edge.certainty_factored));
            outFile.write(reinterpret_cast<const char*>(&edge.k), sizeof(edge.k));
            outFile.write(reinterpret_cast<const char*>(&edge.same_block), sizeof(edge.same_block));
            outFile.write(reinterpret_cast<const char*>(&edge.fixed), sizeof(edge.fixed));
            outFile.write(reinterpret_cast<const char*>(&edge.gt_edge), sizeof(edge.gt_edge));
            outFile.write(reinterpret_cast<const char*>(&edge.good_edge), sizeof(edge.good_edge));
        }
    }

    outFile.close();
}

// Function to load the graph from a binary file
std::vector<Node> loadGraph(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Could not open file for reading");
    }

    size_t num_nodes;
    inFile.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    std::vector<Node> nodes(num_nodes);

    for (auto& node : nodes) {
        inFile.read(reinterpret_cast<char*>(&node.z), sizeof(node.z));
        inFile.read(reinterpret_cast<char*>(&node.f_init), sizeof(node.f_init));
        inFile.read(reinterpret_cast<char*>(&node.f_tilde), sizeof(node.f_tilde));
        inFile.read(reinterpret_cast<char*>(&node.f_star), sizeof(node.f_star));
        inFile.read(reinterpret_cast<char*>(&node.happiness), sizeof(node.happiness));
        inFile.read(reinterpret_cast<char*>(&node.happiness_old), sizeof(node.happiness_old));
        inFile.read(reinterpret_cast<char*>(&node.gt), sizeof(node.gt));
        inFile.read(reinterpret_cast<char*>(&node.gt_f_star), sizeof(node.gt_f_star));
        inFile.read(reinterpret_cast<char*>(&node.deleted), sizeof(node.deleted));
        inFile.read(reinterpret_cast<char*>(&node.fixed), sizeof(node.fixed));
        inFile.read(reinterpret_cast<char*>(&node.fold), sizeof(node.fold));
        inFile.read(reinterpret_cast<char*>(&node.connectivity), sizeof(node.connectivity));        
        inFile.read(reinterpret_cast<char*>(&node.side), sizeof(node.side));
        inFile.read(reinterpret_cast<char*>(&node.side_number), sizeof(node.side_number));
        inFile.read(reinterpret_cast<char*>(&node.wnr_side), sizeof(node.wnr_side));
        inFile.read(reinterpret_cast<char*>(&node.wnr_side_old), sizeof(node.wnr_side_old));
        inFile.read(reinterpret_cast<char*>(&node.total_wnr_side), sizeof(node.total_wnr_side));
        inFile.read(reinterpret_cast<char*>(&node.winding_nr), sizeof(node.winding_nr));
        inFile.read(reinterpret_cast<char*>(&node.winding_nr_old), sizeof(node.winding_nr_old));
        for (int j = 0; j < Node::sides_nr; ++j) {
            inFile.read(reinterpret_cast<char*>(&node.sides[j]), sizeof(node.sides[j]));
            inFile.read(reinterpret_cast<char*>(&node.sides_old[j]), sizeof(node.sides_old[j]));
        }

        inFile.read(reinterpret_cast<char*>(&node.num_same_block_edges), sizeof(node.num_same_block_edges));
        inFile.read(reinterpret_cast<char*>(&node.num_edges), sizeof(node.num_edges));
        node.edges = new Edge[node.num_edges];
        int num_same_block_edges = 0;
        for (int i = 0; i < node.num_edges; ++i) {
            Edge& edge = node.edges[i];
            inFile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(edge.target_node));
            inFile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(edge.certainty));
            inFile.read(reinterpret_cast<char*>(&edge.certainty_factor), sizeof(edge.certainty_factor));
            inFile.read(reinterpret_cast<char*>(&edge.certainty_factored), sizeof(edge.certainty_factored));
            inFile.read(reinterpret_cast<char*>(&edge.k), sizeof(edge.k));
            inFile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(edge.same_block));
            inFile.read(reinterpret_cast<char*>(&edge.fixed), sizeof(edge.fixed));
            inFile.read(reinterpret_cast<char*>(&edge.gt_edge), sizeof(edge.gt_edge));
            inFile.read(reinterpret_cast<char*>(&edge.good_edge), sizeof(edge.good_edge));
            if (edge.same_block) {
                num_same_block_edges++;
            }
        }
        node.num_same_block_edges = num_same_block_edges;
    }

    inFile.close();
    return nodes;
}

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

    // Read each node's winding angle and other attributes
    for (unsigned int i = 0; i < num_nodes; ++i) {
        infile.read(reinterpret_cast<char*>(&graph[i].z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].gt), sizeof(bool));
        infile.read(reinterpret_cast<char*>(&graph[i].gt_f_star), sizeof(float));
        float f_init = graph[i].f_init + 90;
        if (f_init > 180) {
            f_init -= 360;
        }
        graph[i].f_init = f_init;
        graph[i].f_tilde = 0.0f;
        graph[i].f_star = 0.0f;
        graph[i].deleted = false;
        graph[i].fixed = false;
    }
    std::cout << "Nodes loaded successfully." << std::endl;

    int count_same_block_edges = 0;
    int count_other_edges = 0;

    // Read the adjacency list and edges for each node
    size_t count_gt_nodes = 0;
    size_t count_non_gt_nodes = 0;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        if (graph[i].gt) {
            count_gt_nodes++;
        } else {
            count_non_gt_nodes++;
        }
        unsigned int node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(unsigned int));

        unsigned int num_edges;
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(unsigned int));

        // Allocate memory for edges in the node
        graph[node_id].edges = new Edge[num_edges];
        graph[node_id].num_edges = num_edges;

        int num_same_block_edges = 0;
        for (unsigned int j = 0; j < num_edges; ++j) {
            Edge& edge = graph[node_id].edges[j];
            infile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(unsigned int));
            infile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(float));

            // DEBUG
            // edge.certainty = 0.2f;
            // edge.certainty *= 10.0f;
            
            // Set the certainty factored value
            edge.certainty_factored = edge.certainty;

            infile.read(reinterpret_cast<char*>(&edge.k), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(bool));
            edge.fixed = false;  // Default initialization

            // Initialize good_edge to false by default
            edge.good_edge = false;

            // Check if the edge meets the "good_edge" criteria
            if (graph[node_id].gt && graph[edge.target_node].gt && (graph[node_id].gt_f_star != 0.0f && graph[edge.target_node].gt_f_star != 0.0f)) {
                edge.gt_edge = true;
                float edge_adjustment = 0.0f;
                // if (edge.same_block) {
                //     edge_adjustment = edge.k >= 0 ? -2*360.0f : 2*360.0f;
                // }
                if (std::abs(graph[node_id].gt_f_star + edge.k + edge_adjustment - graph[edge.target_node].gt_f_star) < 1.0e-3f) {
                    edge.good_edge = true;
                }
            }

            // Clip Z coordinates if required
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

            // Fix edges between nodes in the same block if needed
            if (fix_same_block_edges) {
                if (std::abs(edge.k) > 180) {
                    edge.same_block = true;
                }
            }

            // Apply same winding factor if necessary
            if (edge.same_block) {
                edge.certainty *= same_winding_factor;
                edge.certainty_factored *= same_winding_factor;
                count_same_block_edges++;
                if (std::abs(edge.k) > 450) {
                    std::cout << "Edge with k > 450: " << edge.k << std::endl;
                }
                if (std::abs(edge.k) < 180) {
                    std::cout << "Edge with k < 180: " << edge.k << std::endl;
                }
                num_same_block_edges++;
            }
            else {
                float node_i = graph[node_id].f_init;
                float node_j = graph[edge.target_node].f_init;
                float diff = node_j - node_i;
                if (diff > 180) {
                    diff -= 360;
                }
                if (diff < -180) {
                    diff += 360;
                }
                if (std::abs(diff - edge.k) > 1.0e-3f) {
                    std::cout << "Edge k value mismatch: " << diff << " vs " << edge.k << std::endl;
                }
                count_other_edges++;
            }
        }
        graph[node_id].num_same_block_edges = num_same_block_edges;
    }
    std::cout << "GT Nodes: " << count_gt_nodes << ", Non-GT Nodes: " << count_non_gt_nodes << std::endl;
    std::cout << "Same block edges: " << count_same_block_edges << std::endl;
    std::cout << "Other edges: " << count_other_edges << std::endl;
    std::cout << "Graph loaded successfully." << std::endl;

    // Find the largest certainty value for display
    float max_certainty = std::numeric_limits<float>::min();
    float min_certainty = std::numeric_limits<float>::max();
    for (const auto& node : graph) {
        if (node.deleted) continue;
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            if (edge.certainty > max_certainty) {
                max_certainty = edge.certainty;
            }
            if (edge.certainty < min_certainty) {
                min_certainty = edge.certainty;
            }
        }
    }

    // // Scale to 0 - 1 range
    // float offset = -min_certainty  + 1.0e-05f;
    // float scale = 1.0f / (max_certainty - min_certainty) - 1.0e-05f;
    // for (auto& node : graph) {
    //     if (node.deleted) continue;
    //     for (int j = 0; j < node.num_edges; ++j) {
    //         node.edges[j].certainty = (node.edges[j].certainty + offset) * scale;
    //     }
    // }

    std::cout << "Max Certainty: " << max_certainty << std::endl;

    infile.close();
    // Return the graph and the max certainty value
    return std::make_pair(graph, max_certainty);
}
