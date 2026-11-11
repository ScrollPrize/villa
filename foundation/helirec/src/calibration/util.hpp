#pragma once
// Small helpers of the calibration checks: text formatting, a straight-line fit, and the per-frame angle and
// height the record states.
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "helirec/calibration/record.hpp"

namespace helirec::calibration {

inline std::string fmt(const char* format, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* format, ...) {
    va_list a, b;
    va_start(a, format);
    va_copy(b, a);
    const int n = std::vsnprintf(nullptr, 0, format, a);
    va_end(a);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 0, '\0');
    if (n > 0) std::vsnprintf(s.data(), s.size() + 1, format, b);
    va_end(b);
    return s;
}

struct Line {
    double slope = 0, intercept = 0, rms = 0, max_abs = 0;
    std::size_t n = 0;
};
// least squares y = intercept + slope x over i in [i0, i1)
template <class X, class Y>
Line fit_line(const X& x, const Y& y, std::size_t i0, std::size_t i1) {
    Line l;
    l.n = i1 - i0;
    if (l.n < 2) return l;
    long double sx = 0, sy = 0;
    for (std::size_t i = i0; i < i1; ++i) sx += x[i], sy += y[i];
    const long double mx = sx / l.n, my = sy / l.n;
    long double sxx = 0, sxy = 0;
    for (std::size_t i = i0; i < i1; ++i) sxx += (x[i] - mx) * (x[i] - mx), sxy += (x[i] - mx) * (y[i] - my);
    l.slope = static_cast<double>(sxy / sxx);
    l.intercept = static_cast<double>(my - l.slope * mx);
    long double ss = 0;
    for (std::size_t i = i0; i < i1; ++i) {
        const double r = y[i] - (l.intercept + l.slope * x[i]);
        ss += static_cast<long double>(r) * r;
        l.max_abs = std::max(l.max_abs, std::fabs(r));
    }
    l.rms = std::sqrt(static_cast<double>(ss / l.n));
    return l;
}

struct Spread {
    double mean = 0, sd = 0, min = 0, max = 0;
    std::size_t n = 0;
    double sem() const { return n > 1 ? sd / std::sqrt(static_cast<double>(n)) : std::numeric_limits<double>::quiet_NaN(); }
};
template <class V>
Spread spread(const V& v, std::size_t i0, std::size_t i1) {
    Spread s;
    s.n = i1 > i0 ? i1 - i0 : 0;
    if (!s.n) return s;
    long double sum = 0;
    s.min = s.max = v[i0];
    for (std::size_t i = i0; i < i1; ++i) {
        sum += v[i];
        s.min = std::min<double>(s.min, v[i]);
        s.max = std::max<double>(s.max, v[i]);
    }
    s.mean = static_cast<double>(sum / s.n);
    long double ss = 0;
    for (std::size_t i = i0; i < i1; ++i) ss += (v[i] - s.mean) * (v[i] - s.mean);
    s.sd = s.n > 1 ? std::sqrt(static_cast<double>(ss / (s.n - 1))) : 0.0;
    return s;
}
template <class V>
Spread spread(const V& v) {
    return spread(v, 0, v.size());
}

}  // namespace helirec::calibration
