#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core/mat.hpp>

class PointIndex
{
public:
    struct QueryResult {
        uint64_t id = 0;
        uint64_t collectionId = 0;
        cv::Vec3f position{0, 0, 0};
        float distanceSq = 0.0f;
    };

    PointIndex();
    ~PointIndex();

    PointIndex(PointIndex&&) noexcept;
    PointIndex& operator=(PointIndex&&) noexcept;

    PointIndex(const PointIndex&) = delete;
    PointIndex& operator=(const PointIndex&) = delete;

    void clear();
    bool empty() const;

    void buildFromMat(const cv::Mat_<cv::Vec3f>& points, uint64_t collectionId = 0);

    std::optional<QueryResult> nearest(
        const cv::Vec3f& position,
        float maxDistance = std::numeric_limits<float>::max()) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
