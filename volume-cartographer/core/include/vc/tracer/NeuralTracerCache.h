#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/tracer/NeuralTracerConnection.h"


class NeuralTracerCache
{
public:
    using NextUvs = NeuralTracerConnection::NextUvs;
    using NextUvsWithJacobian = NeuralTracerConnection::NextUvsWithJacobian;

    NeuralTracerCache(std::unique_ptr<NeuralTracerConnection> connection, float radius)
        : connection_(std::move(connection)),
          radius_(radius)
    {
    }

    std::vector<NextUvs> get_next_points(
        std::vector<cv::Vec3f> const &center,
        std::vector<std::optional<cv::Vec3f>> const &prev_u,
        std::vector<std::optional<cv::Vec3f>> const &prev_v,
        std::vector<std::optional<cv::Vec3f>> const &prev_diag
    );

    std::vector<NextUvsWithJacobian> get_next_points_with_jacobian(
        std::vector<cv::Vec3f> const &center,
        std::vector<std::optional<cv::Vec3f>> const &prev_u,
        std::vector<std::optional<cv::Vec3f>> const &prev_v,
        std::vector<std::optional<cv::Vec3f>> const &prev_diag
    );

private:

    struct InputKey
    {
        cv::Vec3f center;
        std::optional<cv::Vec3f> prev_u;
        std::optional<cv::Vec3f> prev_v;
        std::optional<cv::Vec3f> prev_diag;
    };

    struct CacheEntry
    {
        InputKey input;
        std::optional<NextUvs> basic;
        std::optional<NextUvsWithJacobian> with_jacobian;
    };

    static bool within_radius(const cv::Vec3f &a, const cv::Vec3f &b, float radius);
    static bool matches(const CacheEntry &entry, const InputKey &key, float radius);
    static NextUvs basic_from_jacobian(const NextUvsWithJacobian &with_jacobian);
    void update_hit_rate(size_t batch_size, size_t miss_count);

    std::unique_ptr<NeuralTracerConnection> connection_;
    float radius_;
    std::vector<CacheEntry> cache_;
    static constexpr size_t kHitRateReportEvery = 10000;
    size_t total_requests_ = 0;
    size_t total_hits_ = 0;
    size_t next_report_at_ = kHitRateReportEvery;
};
