#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

// Immutable-after-build spatial index over connected polyline segments.
// Build a private generation, then publish it through shared_ptr for lock-free
// concurrent viewport/slab queries.
class PolylineIndex
{
public:
    struct Polyline {
        uint64_t objectId = 0;
        std::string category;
        std::vector<cv::Vec3f> points;
    };

    struct SegmentResult {
        uint64_t objectId = 0;
        std::string category;
        uint64_t segmentIndex = 0;
        cv::Vec3f first{0, 0, 0};
        cv::Vec3f second{0, 0, 0};
    };

    PolylineIndex();
    ~PolylineIndex();
    PolylineIndex(PolylineIndex&&) noexcept;
    PolylineIndex& operator=(PolylineIndex&&) noexcept;
    PolylineIndex(const PolylineIndex&) = delete;
    PolylineIndex& operator=(const PolylineIndex&) = delete;

    // Replaces the complete generation. Padding expands each segment AABB and
    // is useful for slice/slab tolerance without rebuilding per query.
    void build(std::vector<Polyline> polylines, float padding = 0.0f);
    void clear();
    bool empty() const;
    std::size_t polylineCount() const;
    std::size_t segmentCount() const;

    // The complete generation, in build() order. Valid until the next
    // build()/clear(); stable for shared_ptr-published immutable indices.
    const std::vector<Polyline>& polylines() const;

    std::vector<SegmentResult> query(
        const cv::Vec3f& minimum,
        const cv::Vec3f& maximum,
        const std::optional<std::string>& category = std::nullopt,
        std::size_t limit = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
