#pragma once
#include "core.hpp"

#ifdef __CUDACC__
#define VCR_NORMALIZE_HD __host__ __device__
#else
#define VCR_NORMALIZE_HD
#endif

namespace vcr::augment {
enum class Normalization { MinMax, ZScore, Robust, Percentile, Fixed };
// Statistics use the selected voxels of each channel: those inside the image's
// validity mask when masked (all voxels if it has none, or no valid voxel), and
// for the default z-score only nonzero values. Every voxel is then transformed;
// padding is refilled from the validity mask afterwards.
struct Normalize {
    Normalization mode = Normalization::ZScore;
    bool nonzero = true;           // zscore: statistics exclude zeros, which stay zero
    bool masked = false;
    double eps = 0;                // zscore: lower bound of the standard deviation
    double lower = 1, upper = 99;  // robust/percentile: NumPy linear percentiles
    bool clip = true;              // robust: clip to the percentiles before median/MAD
    bool median = false;           // percentile: (x-median)/(span/2), else (x-lower)/span
    double center = 0, scale = 1;  // fixed: (clip(x, low, high)-center)/scale
    double low = -std::numeric_limits<double>::infinity(), high = std::numeric_limits<double>::infinity();
    ChannelOptions options;
};
inline ChannelPlan channel_parameters(const Normalize&, const Array&, Random&) { return {}; }

namespace normalization_detail {
// NumPy's default ("linear") percentile, including its virtual-index and
// interpolation formulas, evaluated in float64 on sorted selected values.
VCR_NORMALIZE_HD inline double percentile_index(std::size_t n, double percent) {
    const double q = percent / 100;
    return double(n) * q + (1 + q * -1.) - 1;
}
VCR_NORMALIZE_HD inline double lerp(double a, double b, double t) {
    const double d = b - a;
    return t >= .5 ? b - d * (1 - t) : a + d * t;
}
template<class Get> VCR_NORMALIZE_HD double sorted_percentile(Get&& get, std::size_t n, double percent) {
    const double v = percentile_index(n, percent);
    if (v >= double(n - 1)) return get(n - 1);
    if (v < 0) return get(0);
    const double below = std::floor(v);
    const auto j = std::size_t(below);
    return lerp(get(j), get(j + 1), v - below);
}
// The robust scale: 1.4826 MAD, else the first usable fallback.
VCR_NORMALIZE_HD inline double robust_scale(double mad, double deviation, double span) {
    double scale = 1.4826 * mad;
    if (std::isfinite(scale) && scale >= 1e-6) return scale;
    if (std::isfinite(deviation) && std::fabs(deviation) >= 1e-6) return std::fabs(deviation);
    if (std::isfinite(span) && std::fabs(span) / 2 >= 1e-6) return std::fabs(span) / 2;
    return 1;
}
// (x-lower)/span, or (x-median)/(span/2); zero span gives zeros or scale one.
VCR_NORMALIZE_HD inline double percentile_scale(double low, double high, bool median) {
    const double scale = median ? .5 * (high - low) : high - low;
    if (std::isfinite(scale) && scale >= 1e-6) return scale;
    return median ? 1 : 0;
}
template<class T> VCR_NORMALIZE_HD T clamp_to(T x, double low, double high) {
    const T lo = T(low), hi = T(high);
    return x < lo ? lo : x > hi ? hi : x;
}
template<class T> VCR_NORMALIZE_HD T finite_or_zero(double x) {
    const T value = T(x);
    return std::isfinite(value) ? value : T(0);
}

// Retain the reader's fixed segment boundaries and ordered float64 sums.
// Augmentation pipelines have no internal CPU worker pool.
template<class F> void for_parts(std::size_t count, F&& fn) {
    for (std::size_t i = 0; i < count; ++i) fn(i);
}
inline void require_finite(double v) {
    if (!std::isfinite(v)) throw std::invalid_argument("normalization requires finite crop values");
}
// Fixed segments and ordered reductions: the answer does not depend on worker
// count or completion order. Statistics use float64, on the converted output.
template <class T>
void normalize_crop(T* values, std::size_t n, const Normalize& t, const std::uint8_t* valid)
{
    if (!n) return;
    const bool nonzero = t.mode == Normalization::ZScore && t.nonzero;
    const auto skip = [&](std::size_t i) { return (valid && !valid[i]) || (nonzero && values[i] == T{0}); };
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
            if (skip(i)) continue;
            const double v = values[i];
            require_finite(v);
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
    if (t.mode == Normalization::MinMax) {
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
                    if (!skip(i)) sum += static_cast<double>(values[i]) / scale;
                stats[p].sum = sum;
            });
        }
        center = 0;
        for (const auto& s : stats) center += s.sum / static_cast<double>(count);
        for_parts(parts, [&](std::size_t p) {
            double squared = 0;
            for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i) {
                if (skip(i)) continue;
                const double d = static_cast<double>(values[i]) / scale - center;
                squared += d * d;
            }
            stats[p].sum = squared;
        });
        double variance = 0;
        for (const auto& s : stats) variance += s.sum / static_cast<double>(count);
        divisor = std::max(std::sqrt(variance), t.eps / scale);  // population standard deviation, ddof=0
    }
    // Unselected nonzero voxels (outside a validity mask) use the same formula.
    for_parts(parts, [&](std::size_t p) {
        for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i)
            values[i] = (nonzero && values[i] == T{0}) || divisor == 0 ? T{0}
                : static_cast<T>((static_cast<double>(values[i]) / scale - center) / divisor);
    });
}
// Order statistic k of values, which are reordered. Later calls only need
// the elements above the previous rank to stay above it.
template<class T> double rank(std::vector<T>& values, std::size_t k) {
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return double(values[k]);
}
template<class T> double percentile(std::vector<T>& values, double percent) {
    std::size_t last = std::size_t(-1); double previous = 0;
    return sorted_percentile([&](std::size_t k) {
        // Linear interpolation reads ranks j and j+1: after selecting j, rank
        // j+1 is the minimum of the upper partition.
        previous = k == last + 1 ? double(*std::min_element(values.begin() + k, values.end())) : rank(values, k);
        last = k; return previous;
    }, values.size(), percent);
}
template<class T>
void normalize_order(T* values, std::size_t n, const Normalize& t, const std::uint8_t* valid) {
    if (!n) return;
    std::vector<T> selected;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        require_finite(double(values[i]));
        if (!valid || valid[i]) { selected.push_back(values[i]); sum += double(values[i]); }
    }
    const double lo = double(*std::min_element(selected.begin(), selected.end()));
    const double hi = double(*std::max_element(selected.begin(), selected.end()));
    double deviation = 0;
    if (t.mode == Normalization::Robust) {
        // Population standard deviation before clipping, for the fallback scale.
        const double mean = sum / double(selected.size());
        for (auto v : selected) deviation += (double(v) - mean) * (double(v) - mean);
        deviation = std::sqrt(deviation / double(selected.size()));
    }
    const bool clip = t.mode == Normalization::Percentile || t.clip;
    double low = lo, high = hi;
    if (clip) {
        low = percentile(selected, t.lower); high = percentile(selected, t.upper);
        for (std::size_t i = 0; i < n; ++i) values[i] = clamp_to(values[i], low, high);
        for (auto& v : selected) v = clamp_to(v, low, high);
    }
    double center, scale;
    if (t.mode == Normalization::Percentile) {
        scale = percentile_scale(low, high, t.median);
        center = t.median ? percentile(selected, 50) : low;
        if (scale == 0) { std::fill(values, values + n, T{0}); return; }
    } else {
        center = percentile(selected, 50);
        std::vector<double> deviations(selected.size());
        for (std::size_t i = 0; i < selected.size(); ++i) deviations[i] = std::abs(double(selected[i]) - center);
        scale = robust_scale(percentile(deviations, 50), deviation, high - low);
    }
    for (std::size_t i = 0; i < n; ++i) values[i] = finite_or_zero<T>((double(values[i]) - center) / scale);
}
template<class T> void normalize_fixed(T* values, std::size_t n, const Normalize& t) {
    for (std::size_t i = 0; i < n; ++i) {
        require_finite(double(values[i]));
        values[i] = finite_or_zero<T>((double(clamp_to(values[i], t.low, t.high)) - t.center) / t.scale);
    }
}
template<class T> void normalize_channel(T* values, std::size_t n, const Normalize& t, const std::uint8_t* valid) {
    if (t.mode == Normalization::Fixed) normalize_fixed(values, n, t);
    else if (t.mode == Normalization::Robust || t.mode == Normalization::Percentile) normalize_order(values, n, t, valid);
    else normalize_crop(values, n, t, valid);
}
// Per-voxel selection for masked statistics; empty when none is valid.
inline std::vector<std::uint8_t> valid_voxels(const Sample& s, const std::string& key, std::size_t channel) {
    const auto& f = s.at(key);
    if (!s.count(f.target.valid)) return {};
    const auto& mask = s.at(f.target.valid).array;
    const auto n = product(mask.spatial()), offset = (mask.shape[0] == 1 ? 0 : channel) * n;
    std::vector<std::uint8_t> valid(n); bool any = false;
    for (std::size_t i = 0; i < n; ++i) any |= bool(valid[i] = mask.get(offset + i) != 0);
    if (!any) valid.clear();
    return valid;
}
} // namespace normalization_detail
inline void apply(const Normalize& t, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array;
        const auto n = product(a.spatial());
        const auto valid = t.masked ? normalization_detail::valid_voxels(s, c.key, c.channel) : std::vector<std::uint8_t>{};
        const auto* flags = valid.empty() ? nullptr : valid.data();
        if (a.bytes == 4) normalization_detail::normalize_channel(static_cast<float*>(a.data())+c.channel*n,n,t,flags);
        else normalization_detail::normalize_channel(static_cast<double*>(a.data())+c.channel*n,n,t,flags);
    }
    restore_valid(s);
}
} // namespace vcr::augment
