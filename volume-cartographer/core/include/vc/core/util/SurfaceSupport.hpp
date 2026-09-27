#pragma once

#include <cmath>
#include <cstddef>

#include <opencv2/core.hpp>

namespace vc::surface
{
struct OnPredictionSupport {
    // Fraction of valid vertices landing on nonzero prediction voxels.
    double fraction = 1.0;
    std::size_t on = 0;
    std::size_t total = 0;
};

// A min_on_prediction_support threshold is valid only when it is a finite
// fraction in [0, 1]. Anything else is a configuration error: fail fast instead
// of silently disabling the check (negative or NaN can never trigger) or
// rejecting every surface (above 1). See #1675.
inline bool validSupportThreshold(double t)
{
    return std::isfinite(t) && t >= 0.0 && t <= 1.0;
}

// A vertex is valid when no component is the -1 invalid sentinel and all
// components are finite (std::lround on NaN/Inf is undefined behavior).
inline bool isValidVertex(const cv::Vec3f& p)
{
    return p[0] != -1.f && p[1] != -1.f && p[2] != -1.f && std::isfinite(p[0]) &&
           std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Axis-aligned bounds of the valid vertices in points (x, y, z order).
// Returns false when there are no valid vertices.
inline bool validVertexBounds(const cv::Mat_<cv::Vec3f>& points, cv::Vec3f& lo, cv::Vec3f& hi)
{
    bool any = false;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f p = points(r, c);
            if (!isValidVertex(p)) {
                continue;
            }
            if (!any) {
                lo = hi = p;
                any = true;
            } else {
                for (int i = 0; i < 3; ++i) {
                    lo[i] = std::min(lo[i], p[i]);
                    hi[i] = std::max(hi[i], p[i]);
                }
            }
        }
    }
    return any;
}

// True when the grown surface tracks the prediction no better than chance:
// its on-prediction fraction is at or below the background rate measured on
// uniform random points in the surface's neighborhood. Absolute thresholds
// cannot catch this on dense predictions, where even a random surface scores
// high.
inline bool noBetterThanChance(const OnPredictionSupport& support,
                               const OnPredictionSupport& background)
{
    return support.fraction <= background.fraction;
}

// Fraction of valid surface vertices that land on nonzero voxels of the
// prediction the surface was grown from, sampled nearest-neighbor in the
// prediction's native voxel frame.
//
// points holds (x, y, z) vertex positions; a vertex with any component == -1
// or non-finite is invalid and skipped. sample(z, y, x) returns the prediction
// voxel value at integer coordinates; the caller is responsible for bounds
// handling (the helper calls it for every rounded valid vertex).
//
// fraction is 1.0 when there are no valid vertices: an empty surface has
// nothing to judge. See #1675: grown surfaces that cut across windings instead
// of following a sheet land at ~6-10% (indistinguishable from random points in
// the volume), while true sheets land at ~100%, so a threshold around 0.5
// separates them with wide margin on real data.
template <typename Sampler>
OnPredictionSupport onPredictionSupport(const cv::Mat_<cv::Vec3f>& points, Sampler&& sample)
{
    OnPredictionSupport res;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f p = points(r, c);
            // -1 marks invalid vertices; non-finite coordinates must also be
            // skipped before std::lround (which is UB on NaN/Inf), matching
            // QuadSurface's isValidPointSample.
            if (!isValidVertex(p)) {
                continue;
            }
            ++res.total;
            const int z = static_cast<int>(std::lround(p[2]));
            const int y = static_cast<int>(std::lround(p[1]));
            const int x = static_cast<int>(std::lround(p[0]));
            if (sample(z, y, x) != 0) {
                ++res.on;
            }
        }
    }
    if (res.total != 0) {
        res.fraction = static_cast<double>(res.on) / static_cast<double>(res.total);
    }
    return res;
}
}  // namespace vc::surface
