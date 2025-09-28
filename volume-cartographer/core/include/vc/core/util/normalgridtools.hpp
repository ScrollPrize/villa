#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <random>
#include <memory>

namespace vc {
namespace core {
namespace util {

class GridStore;

/**
 * @brief Aligns the normals of a set of QuadSurface segments and extracts an umbilicus point.
 *
 * This function implements a RANSAC-based approach to find a common umbilicus point
 * for a collection of surface segments. It then aligns the normals of these segments
 * to point outwards from this umbilicus.
 *
 * @param grid_store The GridStore containing the normal grid segments.
 * @return A cv::Vec2f representing the estimated umbilicus point in the 2D grid space.
 */
cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store);

void align_and_filter_segments(const GridStore& grid_store, GridStore& result, const cv::Vec2f& center_point = cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));

cv::Mat visualize_segment_directions(const GridStore& grid_store);

struct SegmentInfo;
class SegmentGrid;

cv::Mat visualize_assigned_segments(const SegmentGrid& assigned, const cv::Size& grid_size);

void convert_segment_grid_to_grid_store(const SegmentGrid& segment_grid, const GridStore& original_grid_store, GridStore& grid_store);

struct SegmentInfo {
    cv::Point2f middle_point;
    cv::Vec2f normal;
    size_t original_path_idx;
    size_t original_segment_idx;
    bool flipped = false;

    SegmentInfo(const cv::Point& p1, const cv::Point& p2, size_t path_idx, size_t seg_idx)
        : original_path_idx(path_idx), original_segment_idx(seg_idx) {
        middle_point = cv::Point2f((p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f);
        cv::Vec2f tangent(static_cast<float>(p2.x - p1.x), static_cast<float>(p2.y - p1.y));
        cv::normalize(tangent, tangent);
        normal = cv::Vec2f(-tangent[1], tangent[0]);
    }
};

class SegmentGrid {
public:
    SegmentGrid(const cv::Rect& rect, int grid_step);
    void add(const std::shared_ptr<SegmentInfo>& segment);
    void remove(const std::shared_ptr<SegmentInfo>& segment);
    std::vector<std::shared_ptr<SegmentInfo>> nearest_neighbors(const cv::Point2f& point, int n) const;
    std::shared_ptr<SegmentInfo> get_random_segment();
    size_t count() const;
    cv::Size size() const { return rect.size(); }

    const std::vector<std::shared_ptr<SegmentInfo>>& get_all_segments() const { return all_segments; }

private:
    cv::Rect rect;
    int grid_step;
    std::vector<std::shared_ptr<SegmentInfo>> all_segments;
    std::vector<std::vector<std::vector<std::weak_ptr<SegmentInfo>>>> grid;
    std::mt19937 gen;
};

} // namespace util
} // namespace core
} // namespace vc
