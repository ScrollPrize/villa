#include "CandidateOrdering.hpp"

#include <algorithm>

GrowthCandidateOrdering order_growth_candidates(const std::vector<cv::Vec2i>& candidates,
                                                const GrowthConfig& config,
                                                const GrowthNeighborCountFn& count_all_valid_neighbors,
                                                const GrowthSupportDepthFn& existing_support_depth)
{
    GrowthCandidateOrdering ordering;
    ordering.candidates.reserve(candidates.size());

    for (const cv::Vec2i& p : candidates) {
        ordering.candidates.push_back({
            p,
            count_all_valid_neighbors(p),
            existing_support_depth(p)
        });
    }

    std::sort(ordering.candidates.begin(), ordering.candidates.end(),
              [&config](const OrderedGrowthCandidate& a, const OrderedGrowthCandidate& b) {
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
