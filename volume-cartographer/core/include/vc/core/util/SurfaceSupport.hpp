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

// Fraction of valid surface vertices that land on nonzero voxels of the
// prediction the surface was grown from, sampled nearest-neighbor in the
// prediction's native voxel frame.
//
// points holds (x, y, z) vertex positions; a vertex with any component == -1
// is invalid and skipped. sample(z, y, x) returns the prediction voxel value
// at integer coordinates; the caller is responsible for bounds handling (the
// helper calls it for every rounded valid vertex).
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
            if (p[0] == -1.f || p[1] == -1.f || p[2] == -1.f) {
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
