// Optional floating output conversion and deterministic per-crop normalization.
#pragma once
#include "chunk_io.hpp"
#include "element_type.hpp"
#include "thread_pool.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace vcr {

enum class Normalization { None, MinMax, ZScore };

// Rows of a strided box, with independent input/output element widths.
template <class Row>
void transform_box(const std::uint8_t* src, const Index& ss, std::uint8_t* dst,
                   const Index& ds, const Index& extent, Row&& row)
{
    const std::size_t nd = extent.size();
    Index pos(nd, 0);
    for (;;) {
        row(src, dst, static_cast<std::size_t>(extent.back()));
        std::size_t axis = nd - 1;
        for (;;) {
            if (axis-- == 0) return;
            if (++pos[axis] < extent[axis]) {
                src += ss[axis]; dst += ds[axis];
                break;
            }
            src -= (extent[axis] - 1) * ss[axis];
            dst -= (extent[axis] - 1) * ds[axis];
            pos[axis] = 0;
        }
    }
}

struct BoxOutput {
    std::size_t itemsize;
    std::vector<std::uint8_t> fill;
    std::function<void(const std::uint8_t*, const Index&, std::uint8_t*, const Index&, const Index&)> copy;

    template <class T>
    static BoxOutput make(const ElementType& source, const std::vector<std::uint8_t>& source_fill)
    {
        BoxOutput output;
        output.itemsize = sizeof(T);
        output.fill.resize(sizeof(T));
        source.visit_load([&](auto load) {
            const T value = static_cast<T>(load(source_fill.data()));
            std::memcpy(output.fill.data(), &value, sizeof(T));
            output.copy = [load, stride = source.itemsize](const std::uint8_t* s, const Index& ss,
                    std::uint8_t* d, const Index& ds, const Index& extent) {
                transform_box(s, ss, d, ds, extent, [&](const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
                    auto* out = reinterpret_cast<T*>(dst);
                    for (std::size_t i = 0; i < n; ++i)
                        out[i] = static_cast<T>(load(src + i * stride));
                });
            };
        });
        return output;
    }


};

// Fixed segments and ordered reductions: the answer does not depend on worker
// count or completion order. Statistics use float64, on the converted output.
template <class T>
void normalize_crop(T* values, std::size_t n, Normalization mode, ThreadPool& pool)
{
    if (mode == Normalization::None || !n) return;
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
    pool.parallel_for(parts, [&](std::size_t part) {
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
        pool.parallel_for(parts, [&](std::size_t p) {
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
            pool.parallel_for(parts, [&](std::size_t p) {
                double sum = 0;
                for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i)
                    if (!mask_zero || values[i] != T{0}) sum += static_cast<double>(values[i]) / scale;
                stats[p].sum = sum;
            });
        }
        center = 0;
        for (const auto& s : stats) center += s.sum / static_cast<double>(count);
        pool.parallel_for(parts, [&](std::size_t p) {
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
    pool.parallel_for(parts, [&](std::size_t p) {
        for (std::size_t i = p * grain; i < std::min(n, (p + 1) * grain); ++i)
            values[i] = (mask_zero && values[i] == T{0}) || divisor == 0 ? T{0}
                : static_cast<T>((static_cast<double>(values[i]) / scale - center) / divisor);
    });
}

}  // namespace vcr
