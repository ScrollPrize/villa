#include "ComponentPruning.hpp"

#include <algorithm>
#include <queue>

ComponentPruneResult find_largest_component_prune_points(const cv::Rect& active,
                                                         const cv::Vec2i& seed_loc,
                                                         const ComponentValidPointFn& is_valid_point,
                                                         const ComponentPreservePointFn& preserve_point)
{
    ComponentPruneResult result;
    if (active.area() <= 0) {
        return result;
    }

    cv::Mat_<uint8_t> visited(active.height, active.width, uint8_t{0});
    std::vector<std::vector<cv::Vec2i>> components;
    const std::vector<cv::Vec2i> component_neighs = {
        {-1,  0},
        {-1,  1},
        { 0,  1},
        { 1,  1},
        { 1,  0},
        { 1, -1},
        { 0, -1},
        {-1, -1},
    };

    for (int y = active.y; y < active.br().y; ++y) {
        for (int x = active.x; x < active.br().x; ++x) {
            const cv::Vec2i start(y, x);
            if (visited(y - active.y, x - active.x) || !is_valid_point(start)) {
                continue;
            }

            components.emplace_back();
            std::queue<cv::Vec2i> pending;
            pending.push(start);
            visited(y - active.y, x - active.x) = 1;

            while (!pending.empty()) {
                const cv::Vec2i p = pending.front();
                pending.pop();
                components.back().push_back(p);

                for (const cv::Vec2i& n : component_neighs) {
                    const cv::Vec2i pn = p + n;
                    if (!active.contains(cv::Point(pn[1], pn[0])) ||
                        visited(pn[0] - active.y, pn[1] - active.x) ||
                        !is_valid_point(pn)) {
                        continue;
                    }
                    visited(pn[0] - active.y, pn[1] - active.x) = 1;
                    pending.push(pn);
                }
            }
        }
    }

    result.component_count = components.size();
    if (components.size() <= 1) {
        return result;
    }

    std::size_t keep_index = 0;
    for (std::size_t i = 1; i < components.size(); ++i) {
        if (components[i].size() > components[keep_index].size()) {
            keep_index = i;
        } else if (components[i].size() == components[keep_index].size()) {
            const bool i_has_seed = std::find(components[i].begin(), components[i].end(), seed_loc) != components[i].end();
            const bool keep_has_seed =
                std::find(components[keep_index].begin(), components[keep_index].end(), seed_loc) != components[keep_index].end();
            if (i_has_seed && !keep_has_seed) {
                keep_index = i;
            }
        }
    }

    result.kept_component_size = components[keep_index].size();
    for (std::size_t i = 0; i < components.size(); ++i) {
        if (i == keep_index) {
            continue;
        }
        for (const cv::Vec2i& p : components[i]) {
            if (preserve_point(p)) {
                continue;
            }
            result.points_to_prune.push_back(p);
        }
    }

    return result;
}
