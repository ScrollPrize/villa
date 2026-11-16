#pragma once
// Small statistics and angle helpers shared by the calibration's readers, checks and fits.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "vc_tomography/common.hpp"

namespace vc_tomography::calibration {

constexpr double kDegPerRad = 57.29577951308232;

// numpy.median: NaN for nothing or when a value is NaN; the mean of the two middle values for an even count
inline double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    for (double x : v)
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    if (v.size() % 2) return v[m];
    return (*std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m)) + v[m]) / 2.0;
}

// median(), an Error for nothing
inline double median_of(std::vector<double> v) {
    if (v.empty()) throw Error("calibration: the median of nothing");
    return median(std::move(v));
}

// 1.4826 x the median distance from the median: the standard deviation of a normal bulk, robust to outliers.
// NaN for nothing.
inline double mad_sd(std::vector<double> v) {
    const double centre = median(v);
    for (double& x : v) x = std::fabs(x - centre);
    return 1.4826 * median(std::move(v));
}

// an angle [deg] into [-180, 180]
inline double wrap180(double deg) { return deg - 360.0 * std::round(deg / 360.0); }

inline std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : std::string()) + v[i];
    return s;
}

}  // namespace vc_tomography::calibration
