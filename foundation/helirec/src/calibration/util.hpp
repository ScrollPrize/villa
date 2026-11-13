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
#include "helirec/calibration/solve.hpp"

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

using Line = LineFit;
// Copy only the selected observations; fit_line centres and scales the predictor.
template <class X, class Y>
Line fit_line(const X& x, const Y& y, std::size_t i0, std::size_t i1) {
    if (i1 < i0 || i1 > x.size() || i1 > y.size()) throw Error("fit_line: invalid range");
    try {
        return fit_line(std::vector<double>(x.begin() + i0, x.begin() + i1),
                        std::vector<double>(y.begin() + i0, y.begin() + i1));
    } catch (const Error& e) {
        Line l;
        l.slope = l.intercept = l.rms = l.max_abs = std::numeric_limits<double>::quiet_NaN();
        l.termination = "INVALID_INPUT"; l.message = e.what();
        return l;
    }
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
