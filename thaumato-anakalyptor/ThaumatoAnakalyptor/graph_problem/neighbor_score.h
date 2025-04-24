// neighbor_score.h
#pragma once

#include <cstddef>
#include <vector>
#include <map>
#include "node_structs.h"

// Adds labeled undirected edges between nodes based on local f_star coherence.
// Returns a map from node index to the sampled neighbor indices (within delta_top) that were connected.
// graph: graph to process and modify in-place
// r: spatial radius for neighbor search on (f_init, z)
// delta_neg: max |f_star difference| for initial neighbor filter
// delta_top: threshold for neighbors to return and connect
// delta_perfect: threshold for 'perfect' coherence in scoring
// sample_fraction: fraction of neighbors_top to randomly sample for edge creation (e.g., 0.01 for 1%)
// Forward declaration of helper to add a directed labeled edge (defined in main_py.cpp)
void add_labeled_edge(std::vector<Node>& graph, size_t source_node, size_t target_node, float k);

// Find nodes with strong local f_star coherence, add sampled undirected edges,
// and return a map from node index to the sampled neighbor indices (within delta_top).
// graph: graph to process and modify in-place
// r: spatial radius for neighbor search on (f_init, z)
// delta_neg: max |f_star difference| for initial neighbor filter
// delta_top: threshold for neighbors to return and connect
// delta_perfect: threshold for 'perfect' coherence in scoring
// sample_fraction: fraction of neighbors_top to randomly sample for edge creation (e.g., 0.01 for 1%)
// min_neighbors: minimum number of neighbors_top required to consider a node; nodes with fewer are skipped
std::map<size_t, std::vector<size_t>> find_good_neighbors_and_add_edges(
    std::vector<Node>& graph,
    float r,
    float delta_neg,
    float delta_top,
    float delta_perfect,
    float sample_fraction = 0.01f,
    size_t min_neighbors = 30);