#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/core.hpp>

namespace vc::geometry {

struct CurveSample {
    cv::Vec3d value{0, 0, 0};
    cv::Vec3d derivative{0, 0, 0};
};

// Derivatives are with respect to chord arclength, not the segment index.
inline CurveSample hermite(const cv::Vec3d& a, const cv::Vec3d& b,
                           const cv::Vec3d& da, const cv::Vec3d& db,
                           double span, double t)
{
    if (span <= 1e-12) return {a, {0, 0, 0}};
    const double t2 = t * t, t3 = t2 * t;
    return {a * (2*t3 - 3*t2 + 1) + da * (span * (t3 - 2*t2 + t)) +
                b * (-2*t3 + 3*t2) + db * (span * (t3 - t2)),
            (a * (6*t2 - 6*t) + b * (-6*t2 + 6*t)) / span +
                da * (3*t2 - 4*t + 1) + db * (3*t2 - 2*t)};
}

template<class Points>
cv::Vec3d arcDerivative(const Points& points, size_t i)
{
    size_t lo = i, hi = i;
    while (lo > 0 && cv::norm(cv::Vec3d(points[i]) - cv::Vec3d(points[lo])) <= 1e-12) --lo;
    while (hi + 1 < points.size() && cv::norm(cv::Vec3d(points[hi]) - cv::Vec3d(points[i])) <= 1e-12) ++hi;
    const double span = cv::norm(cv::Vec3d(points[i]) - cv::Vec3d(points[lo])) +
                        cv::norm(cv::Vec3d(points[hi]) - cv::Vec3d(points[i]));
    return span > 1e-12 ? (cv::Vec3d(points[hi]) - cv::Vec3d(points[lo])) / span
                       : cv::Vec3d(0, 0, 0);
}

// Fractional index maps linearly to chord arclength, exactly as in Python's
// strip_geometry._cubic_hermite_line. Only neighbouring samples are needed.
template<class Points>
CurveSample sampleLine(const Points& points, double position)
{
    if (points.empty() || !std::isfinite(position)) return {{NAN, NAN, NAN}, {NAN, NAN, NAN}};
    if (points.size() == 1) return {cv::Vec3d(points.front()), {0, 0, 0}};
    position = std::clamp(position, 0.0, double(points.size() - 1));
    const size_t i = std::min(size_t(position), points.size() - 2);
    const double span = cv::norm(cv::Vec3d(points[i+1]) - cv::Vec3d(points[i]));
    if (span <= 1e-12) return {cv::Vec3d(points[i]), arcDerivative(points, i)};
    return hermite(cv::Vec3d(points[i]), cv::Vec3d(points[i+1]),
                   arcDerivative(points, i), arcDerivative(points, i+1),
                   span, position - i);
}

// Vector field over a fixed arclength domain (e.g. CP displacements).
// Flat knots are stationary boundaries, preventing motion leaking outwards.
inline cv::Vec3d sampleField(const std::vector<double>& arcs,
                             const std::vector<cv::Vec3d>& values,
                             double arc, const std::vector<bool>& flat)
{
    if (arcs.empty()) return {0, 0, 0};
    if (arc <= arcs.front()) return values.front();
    if (arc >= arcs.back()) return values.back();
    const size_t j = std::upper_bound(arcs.begin(), arcs.end(), arc) - arcs.begin();
    const size_t i = j - 1;
    auto derivative = [&](size_t k) {
        if (flat[k]) return cv::Vec3d(0, 0, 0);
        const size_t lo = k ? k-1 : k, hi = std::min(k+1, arcs.size()-1);
        const double span = arcs[hi] - arcs[lo];
        return span > 1e-12 ? (values[hi]-values[lo])/span : cv::Vec3d(0,0,0);
    };
    return hermite(values[i], values[j], derivative(i), derivative(j),
                   arcs[j]-arcs[i], (arc-arcs[i])/(arcs[j]-arcs[i])).value;
}

} // namespace vc::geometry
