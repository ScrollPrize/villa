// solver.cpp

#include "mst_union.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// --- EdgeInfo structure ---
// Holds information about an edge for sorting and consensus.
// u: source node index, v: target node index.
struct EdgeInfo {
    int u;
    int v;
    float k;
    bool same_block;
    float error;       // computed error (with extra penalty if same_block==true)
    int candidate_d;   // computed using the original method
};

// --- Union-Find with Potential and Boundary Edges ---
// The boundaryEdges field stores, for each set (indexed by the set’s root),
// a vector of EdgeInfo for all edges incident to nodes in that set that lead outside.
struct UF {
    std::vector<int> parent;                     // parent[i] is the parent index; if parent[i]==i then i is a root.
    std::vector<int> rank;                       // for union-by-rank.
    std::vector<int> potential;                  // potential[i]: offset from node i to its parent's winding number.
    std::vector<std::vector<int>> nodesInSet;      // nodes in each set (maintained only at the root).
    std::vector<std::vector<EdgeInfo>> boundaryEdges; // boundary edges for each set.
};

// --- Standard union-find functions ---
int find(UF &uf, int x) {
    if (uf.parent[x] != x) {
        int orig = uf.parent[x];
        uf.parent[x] = find(uf, uf.parent[x]);
        uf.potential[x] += uf.potential[orig]; // update potential along the path
    }
    return uf.parent[x];
}

int getPotential(UF &uf, int x) {
    find(uf, x); // ensures path compression
    return uf.potential[x];
}

// Merges two sets: attach root_y to root_x so that the relationship becomes:
//   wnr_side[root_x] - wnr_side[root_y] == d,
// i.e. set uf.potential[root_y] = d.
void unionSets(UF &uf, int x, int y, int d) {
    // x and y are roots. Ensure x becomes the new root.
    if (uf.rank[x] < uf.rank[y]) {
        std::swap(x, y);
        d = -d;
    }
    uf.parent[y] = x;
    uf.potential[y] = d;
    if (uf.rank[x] == uf.rank[y]) {
        uf.rank[x]++;
    }
    // Merge nodesInSet: append all nodes from y into x.
    uf.nodesInSet[x].insert(uf.nodesInSet[x].end(),
                              uf.nodesInSet[y].begin(),
                              uf.nodesInSet[y].end());
    uf.nodesInSet[y].clear();
    // Merge boundary edges from both sets.
    // New boundaryEdges for x will be the union of boundaryEdges[x] and boundaryEdges[y],
    // then filter out edges that are now internal (both endpoints in the merged set).
    std::vector<EdgeInfo> merged;
    merged.reserve(uf.boundaryEdges[x].size() + uf.boundaryEdges[y].size());
    auto addIfExternal = [&](const EdgeInfo &ei) {
        int ru = find(uf, ei.u);
        int rv = find(uf, ei.v);
        if (ru != rv) {
            merged.push_back(ei);
        }
    };
    for (const auto &ei : uf.boundaryEdges[x])
        addIfExternal(ei);
    for (const auto &ei : uf.boundaryEdges[y])
        addIfExternal(ei);
    uf.boundaryEdges[x] = std::move(merged);
    uf.boundaryEdges[y].clear();
}

// --- Helper: Compute candidate difference using the original method ---
// candidate_d = round((f_init_v - f_init_u - k) / 360)
int computeCandidateD(const Node &node_u, const Node &node_v, float k) {
    return static_cast<int>(std::round((node_v.f_init - node_u.f_init - k) / 360.0f));
}

// --- Helper: Compute edge error using f_star values ---
// error = |(f_star_u + k) - f_star_v|, and if same_block is true, add 5.
float computeEdgeError(const Node &node_u, const Node &node_v, float k, bool same_block) {
    float err = std::fabs(((node_u.f_star + k) - node_v.f_star) / std::max(0.01f, std::min(1.0f, k)));
    if (same_block) {
        err += 5.0f;
    }
    return err;
}

// --- Main Solver ---
// This function uses the union-find with boundaryEdges to quickly retrieve cross-set edges.
void solveGraphUnion(std::vector<Node>& nodes) {
    int n = nodes.size();

    // Build a global edge list (for initial sorting).
    // Each edge from a non-deleted node is processed.
    std::vector<EdgeInfo> globalEdgeList;
    for (int u = 0; u < n; ++u) {
        if (nodes[u].deleted) continue;
        for (int i = 0; i < nodes[u].num_edges; ++i) {
            const Edge &e = nodes[u].edges[i];
            int v = static_cast<int>(e.target_node);
            if (v < 0 || v >= n) continue;
            if (nodes[v].deleted) continue;
            int cand = computeCandidateD(nodes[u], nodes[v], e.k);
            float err = computeEdgeError(nodes[u], nodes[v], e.k, e.same_block);
            EdgeInfo ei {u, v, e.k, e.same_block, err, cand};
            globalEdgeList.push_back(ei);
        }
    }
    std::sort(globalEdgeList.begin(), globalEdgeList.end(), [](const EdgeInfo &a, const EdgeInfo &b) {
        return a.error < b.error;
    });

    // Initialize union-find for non-deleted nodes.
    UF uf;
    uf.parent.resize(n);
    uf.rank.resize(n, 0);
    uf.potential.resize(n, 0);
    uf.nodesInSet.resize(n);
    uf.boundaryEdges.resize(n);
    for (int i = 0; i < n; ++i) {
        uf.parent[i] = i;
        uf.potential[i] = 0;
        if (!nodes[i].deleted) {
            uf.nodesInSet[i].push_back(i);
            // Initialize boundaryEdges for node i: add all its outgoing edges.
            for (int j = 0; j < nodes[i].num_edges; ++j) {
                const Edge &e = nodes[i].edges[j];
                int targ = static_cast<int>(e.target_node);
                if (targ < 0 || targ >= n) continue;
                if (nodes[targ].deleted) continue;
                int cand = computeCandidateD(nodes[i], nodes[targ], e.k);
                float err = computeEdgeError(nodes[i], nodes[targ], e.k, e.same_block);
                EdgeInfo ei { i, targ, e.k, e.same_block, err, cand };
                uf.boundaryEdges[i].push_back(ei);
            }
        }
    }

    // Process each edge from the sorted global list.
    int step_count = 0;
    for (const auto &edge : globalEdgeList) {
        std::cout << "Processing edge " << step_count++ << " of " << globalEdgeList.size() << std::endl;
        int u = edge.u, v = edge.v;
        if (nodes[u].deleted || nodes[v].deleted) continue;
        int root_u = find(uf, u), root_v = find(uf, v);
        if (root_u == root_v) continue; // Already in the same set.

        // Retrieve cross-set edges using the boundaryEdges of both sets.
        std::vector<std::pair<int, float>> candidates; // pair: (effective candidate d, error)
        auto processBoundary = [&](int rootFrom, int rootTo) {
            // For each edge in boundaryEdges of set rootFrom:
            for (const auto &ei : uf.boundaryEdges[rootFrom]) {
                // Check if the other endpoint lies in set rootTo.
                int other = (find(uf, ei.u) == rootFrom) ? ei.v : ei.u;
                if (find(uf, other) == rootTo) {
                    // Adjust candidate: effective candidate = candidate_d + (potential[source] - potential[target])
                    int eff = ei.candidate_d;
                    if (find(uf, ei.u) == rootFrom)
                        eff += (getPotential(uf, ei.u) - getPotential(uf, ei.v));
                    else
                        eff += (getPotential(uf, ei.v) - getPotential(uf, ei.u));
                    candidates.push_back({eff, ei.error});
                }
            }
        };

        processBoundary(root_u, root_v);
        processBoundary(root_v, root_u);

        // If no cross-set boundary edges are found, use the current edge.
        int consensus_d = edge.candidate_d + (getPotential(uf, u) - getPotential(uf, v));
        if (!candidates.empty()) {
            std::unordered_map<int, std::pair<float, int>> groups; // candidate -> (total error, count)
            for (auto &candPair : candidates) {
                int cand_val = candPair.first;
                float err = candPair.second;
                groups[cand_val].first += err;
                groups[cand_val].second += 1;
            }
            float bestEffectiveLoss = 1e9;
            for (auto &g : groups) {
                float avgError = g.second.first / g.second.second;
                // Apply additional loss based on the number of edges:
                // If there is 1 edge, multiplier = 10; for 10 or more edges, multiplier = 1.
                // For counts in between, multiplier scales as 10 / count.
                int thresh_size = 100;
                float multiplier = (g.second.second < thresh_size) ? (thresh_size * 1.0f / g.second.second) : 1.0f;
                float effectiveLoss = avgError * multiplier + 10.0f * g.second.second;
                if (effectiveLoss < bestEffectiveLoss) {
                    bestEffectiveLoss = effectiveLoss;
                    consensus_d = g.first;
                }
            }
        }

        // Re-check roots (they might have changed).
        root_u = find(uf, u);
        root_v = find(uf, v);
        if (root_u == root_v) continue;
        unionSets(uf, root_u, root_v, consensus_d);
    }

    // Assign final winding number for each node.
    for (int i = 0; i < n; ++i) {
        if (nodes[i].deleted) continue;
        find(uf, i); // ensure potential is updated
        nodes[i].winding_nr_old = static_cast<float>(uf.potential[i]);
    }
    std::cout << "Solver complete. Processed " << n << " nodes." << std::endl;
}
