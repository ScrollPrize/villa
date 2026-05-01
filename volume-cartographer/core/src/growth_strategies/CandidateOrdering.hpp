#pragma once

#include "GrowthConfig.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <limits>
#include <vector>

struct OrderedGrowthCandidate {
    cv::Vec2i p;
    int valid_neighbor_count = 0;
    int radius_sq = 0;
    int support_depth = 0;
};

struct GrowthCandidateOrdering {
    std::vector<OrderedGrowthCandidate> candidates;
    std::vector<cv::Vec2i> points;
    int radial_min_candidate_radius_sq = std::numeric_limits<int>::max();
    int next_candidate_radius_sq = std::numeric_limits<int>::max();
};

using GrowthRadiusFn = std::function<int(const cv::Vec2i&)>;
using GrowthNeighborCountFn = std::function<int(const cv::Vec2i&)>;
using GrowthInwardNeighborFn = std::function<bool(const cv::Vec2i&, int)>;
using GrowthSupportDepthFn = std::function<int(const cv::Vec2i&)>;
using GrowthRejectCandidateFn = std::function<void(const cv::Vec2i&)>;

GrowthCandidateOrdering order_growth_candidates(const std::vector<cv::Vec2i>& candidates,
                                                const GrowthConfig& config,
                                                int& radial_max_radius_sq,
                                                const GrowthRadiusFn& radius_sq_from_seed,
                                                const GrowthNeighborCountFn& count_all_valid_neighbors,
                                                const GrowthInwardNeighborFn& has_inward_valid_neighbor,
                                                const GrowthSupportDepthFn& existing_support_depth,
                                                const GrowthRejectCandidateFn& reject_candidate);
