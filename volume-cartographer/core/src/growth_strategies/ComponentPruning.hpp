#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <vector>

struct ComponentPruneResult {
    std::vector<cv::Vec2i> points_to_prune;
    std::size_t kept_component_size = 0;
    std::size_t component_count = 0;
};

using ComponentValidPointFn = std::function<bool(const cv::Vec2i&)>;
using ComponentPreservePointFn = std::function<bool(const cv::Vec2i&)>;

ComponentPruneResult find_largest_component_prune_points(const cv::Rect& active,
                                                         const cv::Vec2i& seed_loc,
                                                         const ComponentValidPointFn& is_valid_point,
                                                         const ComponentPreservePointFn& preserve_point);
