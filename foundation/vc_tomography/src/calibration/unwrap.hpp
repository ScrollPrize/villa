#pragma once
// Angles unwrapped as numpy.unwrap does it, so that the values agree with the reference bit for bit.
#include <cmath>
#include <cstddef>
#include <vector>

namespace vc_tomography::calibration {

// numpy.remainder for doubles
inline double py_mod(double a, double b) {
    double mod = std::fmod(a, b);
    if (b == 0) return mod;
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

// numpy.unwrap(p, period=360)
inline std::vector<double> unwrap_360(const std::vector<double>& p) {
    std::vector<double> up(p);
    double sum = 0;
    for (std::size_t i = 1; i < p.size(); ++i) {
        const double dd = p[i] - p[i - 1];
        double ddmod = py_mod(dd - (-180.0), 360.0) + (-180.0);
        if (ddmod == -180.0 && dd > 0) ddmod = 180.0;
        double correct = ddmod - dd;
        if (std::fabs(dd) < 180.0) correct = 0;
        sum += correct;
        up[i] = p[i] + sum;
    }
    return up;
}

}  // namespace vc_tomography::calibration
