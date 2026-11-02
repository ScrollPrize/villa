#pragma once
#include <vc_object_store/geometry/BoundsIndex.hpp>
#include <opencv2/core.hpp>
#include <optional>
#include <limits>
#include <span>
namespace vc::geometry {
// Immutable resident vertices. Query IDs are original row-major indices, including
// gaps for invalid vertices; no application identity or database writes.
class VertexIndex {
public:
    struct Hit { std::uint64_t index; cv::Vec3f position; float distanceSq; };
    void rebuild(const cv::Mat_<cv::Vec3f>& points, std::span<const std::uint8_t> valid = {});
    bool empty() const { return points_.empty(); }
    std::optional<Hit> nearest(const cv::Vec3f& position,
                              float maxDistance = std::numeric_limits<float>::infinity()) const;
private:
    std::vector<cv::Vec3f> points_;
    std::vector<std::uint64_t> indices_;
    objects::detail::BoundsIndex bounds_;
};
}
