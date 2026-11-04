#pragma once
#include "../core.hpp"

namespace vcr::transforms::percentile_detail {
// NumPy's default ("linear") percentile, including its virtual-index and
// interpolation formulas, evaluated in float64 on sorted selected values.
VCR_HD inline double percentile_index(std::size_t n, double percent) {
    const double q = percent / 100;
    return double(n) * q + (1 + q * -1.) - 1;
}
VCR_HD inline double lerp(double a, double b, double t) {
    const double d = b - a;
    return t >= .5 ? b - d * (1 - t) : a + d * t;
}
// The index and lerp conventions are part of each transform's numerical contract.
enum class Rule { NumpyLinear, Linear };
template<Rule rule = Rule::NumpyLinear, class Get>
VCR_HD double sorted_percentile(Get&& get, std::size_t n, double percent) {
    const double v = rule == Rule::NumpyLinear ? percentile_index(n, percent) : percent / 100 * (n - 1);
    if (v >= double(n - 1)) return get(n - 1);
    if (v < 0) return get(0);
    const double below = std::floor(v);
    const auto j = std::size_t(below);
    const auto low = get(j), high = get(j + 1);
    if constexpr (rule == Rule::NumpyLinear) return lerp(low, high, v - below);
    else return low + (high - low) * (v - below);
}
// Order statistic k of values, which are reordered. Later calls only need
// the elements above the previous rank to stay above it.
template<class T> double rank(std::vector<T>& values, std::size_t k) {
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return double(values[k]);
}
template<Rule rule = Rule::NumpyLinear, class T> double percentile(std::vector<T>& values, double percent) {
    std::size_t last = std::size_t(-1); double previous = 0;
    return sorted_percentile<rule>([&](std::size_t k) {
        // Linear interpolation reads ranks j and j+1: after selecting j, rank
        // j+1 is the minimum of the upper partition.
        previous = last != std::size_t(-1) && k == last + 1
            ? double(*std::min_element(values.begin() + k, values.end())) : rank(values, k);
        last = k; return previous;
    }, values.size(), percent);
}
} // namespace vcr::transforms::percentile_detail
