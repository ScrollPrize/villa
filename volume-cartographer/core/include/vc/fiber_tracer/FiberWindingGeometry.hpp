#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/core/types.hpp>

// Shared by the Fiber Map and the source-annotation cycle audit.
namespace vc::fiber_tracer::winding_geometry {
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2.0 * kPi;
// np.interp: linear interpolation with clamped ends.
inline double interpolate(double query, const std::vector<double>& xs,
                   const std::vector<double>& ys)
{
    if (xs.empty()) {
        return 0.0;
    }
    if (query <= xs.front()) {
        return ys.front();
    }
    if (query >= xs.back()) {
        return ys.back();
    }
    const auto upper = std::upper_bound(xs.begin(), xs.end(), query);
    const std::size_t hi = static_cast<std::size_t>(upper - xs.begin());
    const std::size_t lo = hi - 1;
    const double span = xs[hi] - xs[lo];
    if (span <= 0.0) {
        return ys[lo];
    }
    return ys[lo] + (query - xs[lo]) / span * (ys[hi] - ys[lo]);
}

// np.unwrap: shift each successive delta into (-pi, pi].
inline std::vector<double> unwrapAngles(const std::vector<double>& raw)
{
    std::vector<double> out(raw.size());
    if (raw.empty()) {
        return out;
    }
    out[0] = raw[0];
    for (std::size_t i = 1; i < raw.size(); ++i) {
        const double delta = raw[i] - raw[i - 1];
        double wrapped = std::fmod(delta + kPi, kTwoPi);
        if (wrapped < 0.0) {
            wrapped += kTwoPi;
        }
        wrapped -= kPi;
        if (wrapped == -kPi && delta > 0.0) {
            wrapped = kPi;
        }
        out[i] = out[i - 1] + wrapped;
    }
    return out;
}

// The umbilicus as three z-sorted interpolation tables.
struct UmbilicusInterp {
    std::vector<double> z;
    std::vector<double> x;
    std::vector<double> y;
};

template<class Point>
UmbilicusInterp interpolateUmbilicus(const std::vector<Point>& umbilicusCenters)
{
    std::vector<Point> centers = umbilicusCenters;
    std::stable_sort(centers.begin(), centers.end(),
                     [](const Point& a, const Point& b) { return a[2] < b[2]; });
    UmbilicusInterp interp;
    interp.z.resize(centers.size());
    interp.x.resize(centers.size());
    interp.y.resize(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i) {
        interp.z[i] = centers[i][2];
        interp.x[i] = centers[i][0];
        interp.y[i] = centers[i][1];
    }
    return interp;
}

struct PolarTrace {
    std::vector<double> theta, radius;
};
inline PolarTrace polarTrace(const std::vector<cv::Vec3d>& points, const UmbilicusInterp& umbilicus)
{
    PolarTrace result;
    std::vector<double> raw(points.size());
    result.radius.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        const double dx = point[0] - interpolate(point[2], umbilicus.z, umbilicus.x);
        const double dy = point[1] - interpolate(point[2], umbilicus.z, umbilicus.y);
        raw[i] = std::atan2(dy, dx);
        result.radius[i] = std::sqrt(dx * dx + dy * dy);
    }
    result.theta = unwrapAngles(raw);
    return result;
}
} // namespace vc::fiber_tracer::winding_geometry
