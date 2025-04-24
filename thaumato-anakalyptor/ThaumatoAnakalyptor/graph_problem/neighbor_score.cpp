// neighbor_score.cpp
#include "neighbor_score.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include "node_structs.h"

// Compute circular difference a - b in degrees, wrapped to [-180, 180]
static inline float angle_diff(float a, float b) {
    float d = a - b;
    if (d < -180.0f) d += 360.0f;
    else if (d >  180.0f) d -= 360.0f;
    return d;
}

// Internal struct to accumulate neighbor statistics
struct StatEntry {
    size_t idx;
    size_t N_neg;
    size_t N_top;
    size_t N_perf;
    float score_top;
    float score_perf;
    std::vector<size_t> neighbors_top;
};

std::map<size_t, std::vector<size_t>> find_good_neighbors_and_add_edges(
    std::vector<Node>& graph,
    float r,
    float delta_neg,
    float delta_top,
    float delta_perfect,
    float sample_fraction,
    size_t min_neighbors)
    {
        std::cout << "find_good_neighbors_and_add_edges: graph size = " << graph.size() << std::endl;
        std::cout << "Collecting valid (non-deleted) node indices" << std::endl;
        // Collect indices of non-deleted nodes
    std::vector<size_t> valid_ids;
    valid_ids.reserve(graph.size());
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!graph[i].deleted) {
            valid_ids.push_back(i);
        }
    }
    size_t M = valid_ids.size();
    if (M == 0) {
        return {};
    }
    std::cout << "Valid nodes: " << M << std::endl;
    // Apply scaling to z dimension so that distance in z counts 10x
    const float z_scale = 0.1f;
    std::cout << "Scaling z-dimension by " << z_scale << " for radius search" << std::endl;

    // Build data matrix (rows = [f_init, z])
    cv::Mat points((int)M, 2, CV_32F);
    for (int qi = 0; qi < (int)M; ++qi) {
        const Node& node = graph[valid_ids[qi]];
        // f_init in column 0, scaled z in column 1
        points.at<float>(qi, 0) = node.f_init;
        points.at<float>(qi, 1) = node.z * z_scale;
    }
    std::cout << "Built points matrix for " << M << " nodes" << std::endl;

    // Build KD-tree index
    cv::flann::Index kdtree(points, cv::flann::KDTreeIndexParams(4));
    float radius2 = r * r;
    cv::flann::SearchParams searchParams(32);
    std::cout << "Built KD-tree index with radius^2 = " << radius2 << std::endl;
    std::cout << "Starting parallel neighbor search and stats computation" << std::endl;

    // Gather statistics for each node in parallel using OpenMP
    std::vector<StatEntry> stats(M);
    std::vector<char> used(M, 0);
    #pragma omp parallel
    {
        std::vector<int> indices;
        std::vector<float> dists;
        indices.reserve(64);
        dists.reserve(64);
        #pragma omp for schedule(dynamic, 10)
        for (int qi = 0; qi < (int)M; ++qi) {
            size_t i = valid_ids[qi];
            indices.clear(); dists.clear();
            kdtree.radiusSearch(points.row(qi), indices, dists, radius2, static_cast<int>(M), searchParams);
            // Filter by adjusted f_star (corrected for circular f_init)
            std::vector<size_t> neg_ids;
            neg_ids.reserve(indices.size());
            for (int kk = 0; kk < (int)indices.size(); ++kk) {
                int ni = indices[kk];
                if (ni == qi) continue;
                size_t j = valid_ids[ni];
                float dfi = angle_diff(graph[j].f_init, graph[i].f_init);
                float df_star = graph[j].f_star - graph[i].f_star;
                if (std::abs(df_star - dfi) <= delta_neg) neg_ids.push_back(j);
            }
            size_t N_neg = neg_ids.size();
            if (N_neg == 0) continue;
            // Compute top and perfect counts
            size_t N_top = 0, N_perf = 0;
            std::vector<size_t> top_ids;
            top_ids.reserve(neg_ids.size());
            for (size_t j : neg_ids) {
                float dfi = angle_diff(graph[j].f_init, graph[i].f_init);
                float df_star = graph[j].f_star - graph[i].f_star;
                float df_adj = std::abs(df_star - dfi);
                if (df_adj <= delta_top) { ++N_top; top_ids.push_back(j); }
                if (df_adj <= delta_perfect) ++N_perf;
            }
            if (top_ids.size() < min_neighbors) continue;
            float score_top = float(N_top) / float(N_neg);
            float score_perf = float(N_perf) / float(N_neg);
            stats[qi] = StatEntry{valid_ids[qi], N_neg, N_top, N_perf, score_top, score_perf, std::move(top_ids)};
            used[qi] = 1;
        }
    }
    std::cout << "Finished parallel neighbor stats; generating partitions" << std::endl;
    // Partition entries into good and rest based on perfect score
    std::vector<StatEntry*> good1;
    std::vector<StatEntry*> rest;
    good1.reserve(M);
    rest.reserve(M);
    for (int qi = 0; qi < (int)M; ++qi) {
        if (!used[qi]) continue;
        StatEntry* e = &stats[qi];
        if (e->score_perf > 0.75f) good1.push_back(e);
        else rest.push_back(e);
    }
    std::cout << "Partitioned entries: good1=" << good1.size() << ", rest=" << rest.size() << std::endl;

    // Sort rest by descending top/neg score
    std::sort(rest.begin(), rest.end(), [](const StatEntry* a, const StatEntry* b) {
        return a->score_top > b->score_top;
    });
    size_t top10cnt = size_t(std::ceil(rest.size() * 0.10f));
    if (top10cnt > rest.size()) top10cnt = rest.size();
    std::cout << "Selecting top " << top10cnt << " entries from rest based on top score" << std::endl;

    // Combine selected entries
    std::vector<StatEntry*> selected;
    selected.reserve(good1.size() + top10cnt);
    for (auto *e : good1) selected.push_back(e);
    for (size_t i = 0; i < top10cnt; ++i) selected.push_back(rest[i]);
    std::cout << "Total selected entries: " << selected.size() << std::endl;
    std::cout << "Sampling neighbors and adding edges, sample fraction = " << sample_fraction << std::endl;
    size_t sel_size = selected.size();
    size_t sel_count = 0;

    // Prepare result and random sampler
    std::map<size_t, std::vector<size_t>> result;
    std::random_device rd;
    std::mt19937 gen(rd());

    // For each selected node, sample neighbors and add undirected edges
    for (auto *e : selected) {
        ++sel_count;
        if (sel_count % 100 == 0 || sel_count == sel_size) {
            std::cout << "Sampling neighbors: processed " << sel_count << " / " << sel_size << std::endl;
        }
        size_t i = e->idx;
        auto &neigh = e->neighbors_top;
        size_t sample_cnt = size_t(std::ceil(neigh.size() * sample_fraction));
        if (sample_cnt < 1) sample_cnt = 1;
        if (sample_cnt > neigh.size()) sample_cnt = neigh.size();

        std::shuffle(neigh.begin(), neigh.end(), gen);
        std::vector<size_t> sampled(neigh.begin(), neigh.begin() + sample_cnt);
        // Add edges in both directions
        for (size_t j : sampled) {
            // Edge winding difference should be based on circular f_init delta
            float k_init = angle_diff(graph[j].f_init, graph[i].f_init);
            add_labeled_edge(graph, i, j, k_init);
            add_labeled_edge(graph, j, i, -k_init);
        }
        result[i] = std::move(sampled);
    }
    std::cout << "Finished adding edges for selected nodes" << std::endl;
    std::cout << "find_good_neighbors_and_add_edges: returning result of size " << result.size() << std::endl;
    return result;
}