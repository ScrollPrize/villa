#include "vc/tracer/NeuralTracerCache.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
struct BatchInputs {
    std::vector<cv::Vec3f> center;
    std::vector<std::optional<cv::Vec3f>> prev_u;
    std::vector<std::optional<cv::Vec3f>> prev_v;
    std::vector<std::optional<cv::Vec3f>> prev_diag;
};

BatchInputs build_batch_inputs(
    std::vector<cv::Vec3f> const &center,
    std::vector<std::optional<cv::Vec3f>> const &prev_u,
    std::vector<std::optional<cv::Vec3f>> const &prev_v,
    std::vector<std::optional<cv::Vec3f>> const &prev_diag,
    std::vector<size_t> const &misses
) {
    BatchInputs batch;
    batch.center.reserve(misses.size());
    batch.prev_u.reserve(misses.size());
    batch.prev_v.reserve(misses.size());
    batch.prev_diag.reserve(misses.size());

    for (size_t idx : misses) {
        batch.center.push_back(center[idx]);
        batch.prev_u.push_back(prev_u[idx]);
        batch.prev_v.push_back(prev_v[idx]);
        batch.prev_diag.push_back(prev_diag[idx]);
    }

    return batch;
}
} // namespace

void NeuralTracerCache::update_hit_rate(size_t batch_size, size_t miss_count) {
    total_requests_ += batch_size;
    total_hits_ += batch_size - miss_count;

    if (total_requests_ < next_report_at_) {
        return;
    }

    double hit_rate = 0.0;
    if (total_requests_ > 0) {
        hit_rate = 100.0 * static_cast<double>(total_hits_) / static_cast<double>(total_requests_);
    }
    std::cout << "NeuralTracerCache hit rate: "
              << std::fixed << std::setprecision(2) << hit_rate
              << "% (hits=" << total_hits_
              << ", requests=" << total_requests_
              << ")\n";

    while (total_requests_ >= next_report_at_) {
        next_report_at_ += kHitRateReportEvery;
    }
}


bool NeuralTracerCache::within_radius(const cv::Vec3f &a, const cv::Vec3f &b, float radius) {
    cv::Vec3f diff = a - b;
    return diff.dot(diff) <= radius * radius;
}

bool NeuralTracerCache::matches(const CacheEntry &entry, const InputKey &key, float radius) {
    if (!within_radius(entry.input.center, key.center, radius)) {
        return false;
    }

    auto matches_optional = [&](const std::optional<cv::Vec3f> &cached, const std::optional<cv::Vec3f> &query) {
        if (!query) {
            return true;
        }
        if (!cached) {
            return false;
        }
        return within_radius(*cached, *query, radius);
    };

    return matches_optional(entry.input.prev_u, key.prev_u)
        && matches_optional(entry.input.prev_v, key.prev_v)
        && matches_optional(entry.input.prev_diag, key.prev_diag);
}

NeuralTracerCache::NextUvs NeuralTracerCache::basic_from_jacobian(
    const NextUvsWithJacobian &with_jacobian
) {
    NextUvs basic;
    basic.next_u_xyzs = with_jacobian.next_u_xyzs;
    basic.next_v_xyzs = with_jacobian.next_v_xyzs;
    return basic;
}

std::vector<NeuralTracerCache::NextUvs> NeuralTracerCache::get_next_points(
    std::vector<cv::Vec3f> const &center,
    std::vector<std::optional<cv::Vec3f>> const &prev_u,
    std::vector<std::optional<cv::Vec3f>> const &prev_v,
    std::vector<std::optional<cv::Vec3f>> const &prev_diag
) {
    if (center.size() != prev_u.size() || center.size() != prev_v.size() || center.size() != prev_diag.size() ) {
        throw std::runtime_error("NeuralTracerCache batch sizes do not match");
    }

    std::vector<NextUvs> results(center.size());
    std::vector<size_t> misses;
    misses.reserve(center.size());

    for (size_t i = 0; i < center.size(); ++i) {
        InputKey key{center[i], prev_u[i], prev_v[i], prev_diag[i]};
        bool found = false;
        for (const auto &entry : cache_) {
            if (!matches(entry, key, radius_)) {
                continue;
            }
            if (entry.basic) {
                results[i] = *entry.basic;
                found = true;
                break;
            }
            if (entry.with_jacobian) {
                results[i] = basic_from_jacobian(*entry.with_jacobian);
                found = true;
                break;
            }
        }
        if (!found) {
            misses.push_back(i);
        }
    }

    update_hit_rate(center.size(), misses.size());

    if (!misses.empty()) {
        BatchInputs batch = build_batch_inputs(center, prev_u, prev_v, prev_diag, misses);
        auto fetched = connection_->get_next_points(
            batch.center, batch.prev_u, batch.prev_v, batch.prev_diag);
        for (size_t j = 0; j < misses.size(); ++j) {
            size_t idx = misses[j];
            results[idx] = fetched[j];
            cache_.push_back({
                {center[idx], prev_u[idx], prev_v[idx], prev_diag[idx]},
                fetched[j]
            });
        }
    }

    return results;
}

std::vector<NeuralTracerCache::NextUvsWithJacobian> NeuralTracerCache::get_next_points_with_jacobian(
    std::vector<cv::Vec3f> const &center,
    std::vector<std::optional<cv::Vec3f>> const &prev_u,
    std::vector<std::optional<cv::Vec3f>> const &prev_v,
    std::vector<std::optional<cv::Vec3f>> const &prev_diag
) {
    if (center.size() != prev_u.size() || center.size() != prev_v.size() || center.size() != prev_diag.size() ) {
        throw std::runtime_error("NeuralTracerCache batch sizes do not match");
    }

    std::vector<NextUvsWithJacobian> results(center.size());
    std::vector<size_t> misses;
    misses.reserve(center.size());

    for (size_t i = 0; i < center.size(); ++i) {
        InputKey key{center[i], prev_u[i], prev_v[i], prev_diag[i]};
        bool found = false;
        for (const auto &entry : cache_) {
            if (!matches(entry, key, radius_)) {
                continue;
            }
            if (entry.with_jacobian) {
                results[i] = *entry.with_jacobian;
                found = true;
                break;
            }
        }
        if (!found) {
            misses.push_back(i);
        }
    }

    update_hit_rate(center.size(), misses.size());

    if (!misses.empty()) {
        BatchInputs batch = build_batch_inputs(center, prev_u, prev_v, prev_diag, misses);
        auto fetched = connection_->get_next_points_with_jacobian(
            batch.center, batch.prev_u, batch.prev_v, batch.prev_diag);
        for (size_t j = 0; j < misses.size(); ++j) {
            size_t idx = misses[j];
            results[idx] = fetched[j];
            CacheEntry entry;
            cache_.push_back({
                {center[idx], prev_u[idx], prev_v[idx], prev_diag[idx]},
                basic_from_jacobian(fetched[j]),
                fetched[j]
            });
        }
    }

    return results;
}
