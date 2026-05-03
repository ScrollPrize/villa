#pragma once

#include "GrowthConfig.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <vector>

struct OrderedGrowthCandidate {
    cv::Vec2i p;
    int valid_neighbor_count = 0;
    int support_depth = 0;
};

struct GrowthCandidateOrdering {
    std::vector<OrderedGrowthCandidate> candidates;
    std::vector<cv::Vec2i> points;
};

using GrowthNeighborCountFn = std::function<int(const cv::Vec2i&)>;
using GrowthSupportDepthFn = std::function<int(const cv::Vec2i&)>;

GrowthCandidateOrdering order_growth_candidates(const std::vector<cv::Vec2i>& candidates,
                                                const GrowthConfig& config,
                                                const GrowthNeighborCountFn& count_all_valid_neighbors,
                                                const GrowthSupportDepthFn& existing_support_depth);
