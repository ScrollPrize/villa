#pragma once
#include "core.hpp"

namespace vcr::augment {
enum class Normalization { MinMax, ZScore };
struct Normalize { Normalization mode = Normalization::ZScore; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Normalize&, const Array&, Random&) { return {}; }

namespace normalization_detail {
// Retain the reader's fixed segment boundaries and ordered float64 sums.
// Augmentation pipelines have no internal CPU worker pool.
template<class F> void for_parts(std::size_t count, F&& fn) {
    for (std::size_t i = 0; i < count; ++i) fn(i);
}
// Fixed segments and ordered reductions: the answer does not depend on worker
// count or completion order. Statistics use float64, on the converted output.
template <class T>
void normalize_crop(T* values, std::size_t n, Normalization mode)
{
    if (!n) return;
    const bool mask_zero = mode == Normalization::ZScore;
    constexpr std::size_t grain = 65536;
    const std::size_t parts = (n + grain - 1) / grain;
    struct Stats {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        double sum = 0;
        std::size_t count = 0;
    };
    std::vector<Stats> stats(parts);
    for_parts(parts, [&](std::size_t part) {
        // Accumulate in registers, publishing once per segment. Updating the
        // shared vector in this loop causes false sharing between workers.
        Stats s;
        for (std::size_t i = part * grain; i < std::min(n, (part + 1) * grain); ++i) {
            if (mask_zero && values[i] == T{0}) continue;
            const double v = values[i];
            if (!std::isfinite(v)) throw std::invalid_argument("normalization requires finite crop values");
            s.lo = std::min(s.lo, v); s.hi = std::max(s.hi, v);
            s.sum += v; ++s.count;
        }
        stats[part] = s;
    });
    double lo = stats[0].lo, hi = stats[0].hi;
    std::size_t count = 0;
    for (const auto& s : stats) { lo = std::min(lo, s.lo); hi = std::max(hi, s.hi); count += s.count; }
    if (!count || lo == hi) {
        for_parts(parts, [&](std::size_t p) {
            std::fill(values + p * grain, values + std::min(n, (p + 1) * grain), T{0});
        });
        return;
    }
    double scale = 1, center = lo, divisor = hi - lo;
    if (mode == Normalization::MinMax) {
        if (!std::isfinite(divisor)) { scale = 2; center = lo / scale; divisor = hi / scale - center; }
    } else {
        const double magnitude = std::max(std::abs(lo), std::abs(hi));
        // Scale extreme finite doubles so sums of squared deviations cannot
        // overflow/underflow. Ordinary CT values take the unscaled path.
        if (magnitude > 1e150 || magnitude < 1e-150) {
            scale = magnitude;
            for_parts(parts, [&](std::size_t p) {
                double sum = 0;
                for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i)
                    if (!mask_zero || values[i] != T{0}) sum += static_cast<double>(values[i]) / scale;
                stats[p].sum = sum;
            });
        }
        center = 0;
        for (const auto& s : stats) center += s.sum / static_cast<double>(count);
        for_parts(parts, [&](std::size_t p) {
            double squared = 0;
            for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i) {
                if (mask_zero && values[i] == T{0}) continue;
                const double d = static_cast<double>(values[i]) / scale - center;
                squared += d * d;
            }
            stats[p].sum = squared;
        });
        double variance = 0;
        for (const auto& s : stats) variance += s.sum / static_cast<double>(count);
        divisor = std::sqrt(variance);  // population standard deviation, ddof=0
    }
    for_parts(parts, [&](std::size_t p) {
        for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i)
            values[i] = (mask_zero && values[i] == T{0}) || divisor == 0 ? T{0}
                : static_cast<T>((static_cast<double>(values[i]) / scale - center) / divisor);
    });
}

} // namespace normalization_detail
inline void apply(const Normalize& t, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array;
        const auto n = product(a.spatial());
        if (a.bytes == 4) normalization_detail::normalize_crop(static_cast<float*>(a.data())+c.channel*n,n,t.mode);
        else normalization_detail::normalize_crop(static_cast<double*>(a.data())+c.channel*n,n,t.mode);
    }
    restore_valid(s);
}
} // namespace vcr::augment
