// node_structs.h
#pragma once

#include <cmath>  // For std::abs in float comparison
#include <vector> // For std::vector
#include <string>

struct Edge {
    unsigned int target_node;
    float certainty;
    float certainty_factor;
    float certainty_factored;
    float k;
    bool same_block;
    bool fixed;
    bool gt_edge;
    bool good_edge;
    bool wnr_added = false;      // Target winding number incorporated to the node
    int wnr_from_target = 0;          // Target winding number for the edge

    // Define the equality operator for Edge
    bool operator==(const Edge& other) const {
        return target_node == other.target_node &&
               std::abs(certainty - other.certainty) < 1e-6 &&
               std::abs(certainty_factor - other.certainty_factor) < 1e-6 &&
               std::abs(certainty_factored - other.certainty_factored) < 1e-6 &&
               std::abs(k - other.k) < 1e-6 &&
               same_block == other.same_block &&
               fixed == other.fixed &&
               gt_edge == other.gt_edge &&
               good_edge == other.good_edge;
    }
};

struct Node {
    float z;                     // Z-coordinate of the node
    float f_init;                // Initial value of f for this node
    float f_tilde;               // Current value of f_tilde for this node (used in BP updates)
    float f_star;                // The computed final value of f for this node
    float f_star_momentum = 0.0f;       // Momentum accumulator (initialize to 0)
    float happiness = 0.0f;      // Happiness value for this node
    float happiness_old = 0.0f;  // Old happiness value for this node
    bool gt;                     // Indicates if this node has ground truth available
    float gt_f_star;             // Ground truth f_star value, if available
    bool deleted;                // If the node is marked as deleted
    bool fixed;                  // Whether this node is fixed or not
    float fold;                  // Folding state for the node
    float connectivity = -1.0f;  // Connectivity information for the node
    float confidence = 0.0f;     // Confidence value for the node
    float confidence_old = 0.0f; // Old confidence value for the node
    float smeared_confidence = 0.0f; // Smeared confidence value for the node
    float smeared_confidence_old = 0.0f; // Old smeared confidence value for the node
    float confidence_mask = 0.0f; // Confidence mask value for the node
    float closeness = 0.0f;      // Closeness value for the node
    float closeness_old = 0.0f;  // Old closeness value for the node
    float happiness_v2 = 0.0f; // Updated happiness value for the node
    float same_block_closeness = 0.0f;
    float same_block_closeness_old = 0.0f;

    float side = 0.0f;           // Side value for the node wrt to nodes around itself
    static constexpr int sides_nr = 18; // Number of sides for the node
    float* sides = new float[sides_nr]; // Sides values for the node
    float* sides_old = new float[sides_nr]; // Sides values for the node
    int side_number = 0;         // Side of node
    float wnr_side = 0.0f;  // Certainty of the winding number
    float wnr_side_old = 0.0f;  // Certainty of the winding number
    float total_wnr_side = 0.0f;  // Total Certainty of the winding number over all valid edges
    int winding_nr = 0;          // Winding number for the node
    int winding_nr_old = 0;          // Winding number for the node

    Edge* edges;                 // Pointer to an array of edges (dynamic array)
    int num_edges;               // Number of edges connected to this node
    int num_same_block_edges;    // Number of edges connected to this node that are in the same block

    // Define the equality operator for Node
    bool operator==(const Node& other) const {
        if (!(std::abs(z - other.z) < 1e-6 &&
              std::abs(f_init - other.f_init) < 1e-6 &&
              std::abs(f_tilde - other.f_tilde) < 1e-6 &&
              std::abs(f_star - other.f_star) < 1e-6 &&
              std::abs(happiness - other.happiness) < 1e-6 &&
              gt == other.gt &&
              std::abs(gt_f_star - other.gt_f_star) < 1e-6 &&
              deleted == other.deleted &&
              fixed == other.fixed &&
              std::abs(fold - other.fold) < 1e-6 &&
              num_edges == other.num_edges)) {
            return false;
        }

        // Compare edges in the array
        for (int i = 0; i < num_edges; ++i) {
            if (!(edges[i] == other.edges[i])) {
                return false;
            }
        }

        return true;
    }
};

// Declaration of equality operator for std::vector<Node>
bool operator==(const std::vector<Node>& graph1, const std::vector<Node>& graph2);

// Function declarations for saving and loading the graph
void saveGraph(const std::vector<Node>& nodes, const std::string& filename);
std::vector<Node> loadGraph(const std::string& filename);

std::pair<std::vector<Node>, float> load_graph_from_binary(const std::string &file_name, bool clip_z, float z_min, float z_max, float same_winding_factor, bool fix_same_block_edges);