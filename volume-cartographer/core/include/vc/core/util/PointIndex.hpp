#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

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
    size_t size() const;

    void insert(uint64_t id, uint64_t collectionId, const cv::Vec3f& position);
    void remove(uint64_t id);
    bool update(uint64_t id, const cv::Vec3f& newPosition);

    std::optional<QueryResult> nearestInCollection(
        const cv::Vec3f& position,
        uint64_t collectionId,
        float maxDistance = std::numeric_limits<float>::max()) const;

    std::vector<QueryResult> queryRadius(
        const cv::Vec3f& center,
        float radius) const;

    std::optional<QueryResult> nearest(
        const cv::Vec3f& position,
        float maxDistance = std::numeric_limits<float>::max()) const;

    std::vector<QueryResult> kNearest(
        const cv::Vec3f& position,
        size_t k,
        float maxDistance = std::numeric_limits<float>::max()) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
