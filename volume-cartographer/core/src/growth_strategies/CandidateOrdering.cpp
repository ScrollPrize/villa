#include "CandidateOrdering.hpp"

#include <algorithm>

GrowthCandidateOrdering order_growth_candidates(const std::vector<cv::Vec2i>& candidates,
                                                const GrowthConfig& config,
                                                int& radial_max_radius_sq,
                                                const GrowthRadiusFn& radius_sq_from_seed,
                                                const GrowthNeighborCountFn& count_all_valid_neighbors,
                                                const GrowthInwardNeighborFn& has_inward_valid_neighbor,
                                                const GrowthSupportDepthFn& existing_support_depth,
                                                const GrowthRejectCandidateFn& reject_candidate)
{
    GrowthCandidateOrdering ordering;
    ordering.candidates.reserve(candidates.size());

    if (config.radial_growth) {
        for (const cv::Vec2i& p : candidates) {
            const int radius_sq = radius_sq_from_seed(p);
            ordering.radial_min_candidate_radius_sq =
                std::min(ordering.radial_min_candidate_radius_sq, radius_sq);
        }
    }
    if (config.radial_growth &&
        radial_max_radius_sq < 0 &&
        ordering.radial_min_candidate_radius_sq != std::numeric_limits<int>::max()) {
        radial_max_radius_sq = ordering.radial_min_candidate_radius_sq;
    }
    if (config.radial_growth && radial_max_radius_sq >= 0) {
        for (const cv::Vec2i& p : candidates) {
            const int radius_sq = radius_sq_from_seed(p);
            if (radius_sq > radial_max_radius_sq) {
                ordering.next_candidate_radius_sq =
                    std::min(ordering.next_candidate_radius_sq, radius_sq);
            }
        }
    }

    for (const cv::Vec2i& p : candidates) {
        const int radius_sq = radius_sq_from_seed(p);
        if (config.radial_growth &&
            ((radial_max_radius_sq >= 0 && radius_sq > radial_max_radius_sq) ||
             !has_inward_valid_neighbor(p, radius_sq))) {
            reject_candidate(p);
            continue;
        }
        ordering.candidates.push_back({
            p,
            count_all_valid_neighbors(p),
            radius_sq,
            existing_support_depth(p)
        });
    }

    std::sort(ordering.candidates.begin(), ordering.candidates.end(),
              [&config](const OrderedGrowthCandidate& a, const OrderedGrowthCandidate& b) {
                  if (config.radial_growth && a.radius_sq != b.radius_sq) {
                      return a.radius_sq < b.radius_sq;
                  }
                  if (config.candidate_priority_existing_depth && a.support_depth != b.support_depth) {
                      return a.support_depth > b.support_depth;
                  }
                  if (a.valid_neighbor_count != b.valid_neighbor_count) {
                      return a.valid_neighbor_count > b.valid_neighbor_count;
                  }
                  if (a.p[0] != b.p[0]) {
                      return a.p[0] < b.p[0];
                  }
                  return a.p[1] < b.p[1];
              });

    ordering.points.reserve(ordering.candidates.size());
    for (const OrderedGrowthCandidate& candidate : ordering.candidates) {
        ordering.points.push_back(candidate.p);
    }

    return ordering;
}
