#pragma once
#include "core.hpp"
#include "normalization.hpp"
#include "local_field.hpp"
#include "median.hpp"
#include "oriented_blur.hpp"
#include <span>
#include <unordered_map>

namespace vcr::augment {
struct BrightnessMultiply { Scalar factor = Uniform{.75, 1.25}; ChannelOptions options; };
struct BrightnessAdd { Scalar shift = Normal{0, .1}; ChannelOptions options; };
struct Contrast { Scalar factor = Uniform{.75, 1.25}; bool preserve_range = true; ChannelOptions options; };
struct Gamma { Scalar gamma = Uniform{.7, 1.5}; double p_invert = 0, p_retain_stats = 0; ChannelOptions options; };
struct Invert { ChannelOptions options; };
struct GaussianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct RicianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct ClipOutliers { Scalar lower = .2, upper = 99.8; double p_retain_std = 1; ChannelOptions options; };
struct GaussianBlur { Scalar sigma = Uniform{.3, 1.5}; bool shared_axes = true; Boundary boundary = Boundary::Mirror; double truncate = 6; bool full_width = true; ChannelOptions options; };
struct MedianFilter { Scalar size = Uniform{3, 7}; bool quantize = false; ChannelOptions options; };
struct Sharpen { Scalar strength = Uniform{.1, .5}; double p_clamp = .5; ChannelOptions options; };
struct BrightnessGradient { Scalar strength = Uniform{-.5, .5}; bool mean_centered = true, clip = false; LocalOptions local; ChannelOptions options; };
struct LocalContrast { Scalar factor = Uniform{.5, 1.5}; LocalOptions local; ChannelOptions options; };
struct LocalGamma { Scalar gamma = Uniform{.5, 1.5}; LocalOptions local; ChannelOptions options; };
struct LocalSmoothing { Scalar strength = Uniform{.5, 1}, sigma = Uniform{1, 3}; LocalOptions local; ChannelOptions options; };
struct BlankRectangle {
    std::vector<Scalar> size{16.}; Scalar count = Uniform{1, 5}, value = 0.;
    std::string statistic = "mean"; bool square = false; ChannelOptions options;
};
struct SliceIllumination {
    char axis = 'z'; Scalar defects = Uniform{1, 5}, width = Uniform{1, 10}, reduction = Uniform{.1, .5};
    Scalar base_p = Uniform{.1, .3}, base_reduction = Uniform{.1, .5}; ChannelOptions options;
};
struct Smear { char axis = 'z'; std::vector<int> shift{10, 0}; double alpha = .5; int previous = 1; ChannelOptions options; };

inline ChannelPlan channel_parameters(const BrightnessMultiply& t, const Array&, Random& r) { return {"", 0, {draw(t.factor, r)}}; }
inline ChannelPlan channel_parameters(const BrightnessAdd& t, const Array&, Random& r) { return {"", 0, {draw(t.shift, r)}}; }
inline ChannelPlan channel_parameters(const Contrast& t, const Array&, Random& r) { return {"", 0, {draw(t.factor, r)}}; }
inline ChannelPlan channel_parameters(const Gamma& t, const Array&, Random& r) {
    auto g = draw(t.gamma, r); require(g > 0, "gamma must be positive");
    return {"", 0, {g, double(r.chance(t.p_invert)), double(r.chance(t.p_retain_stats))}};
}
inline ChannelPlan channel_parameters(const Invert&, const Array&, Random&) { return {}; }
inline ChannelPlan noise_parameters(const Scalar& std, Random& r) {
    auto sigma = draw(std, r); require(sigma >= 0, "noise std must be nonnegative");
    ChannelPlan p; p.values = {sigma}; p.seed = r.next(); return p;
}
inline ChannelPlan channel_parameters(const GaussianNoise& t, const Array&, Random& r) { return noise_parameters(t.std, r); }
inline ChannelPlan channel_parameters(const RicianNoise& t, const Array&, Random& r) { return noise_parameters(t.std, r); }
inline ChannelPlan channel_parameters(const ClipOutliers& t, const Array&, Random& r) {
    auto lo = draw(t.lower, r), hi = draw(t.upper, r);
    require(0 <= lo && lo <= hi && hi <= 100, "percentiles must satisfy 0 <= lower <= upper <= 100");
    return {"", 0, {lo, hi, double(r.chance(t.p_retain_std))}};
}
inline ChannelPlan channel_parameters(const GaussianBlur& t, const Array& a, Random& r) {
    ChannelPlan p; double shared = draw(t.sigma, r);
    for (std::size_t d = 1; d < a.shape.size(); ++d) p.values.push_back(t.shared_axes ? shared : draw(t.sigma, r));
    return p;
}
inline ChannelPlan channel_parameters(const MedianFilter& t, const Array&, Random& r) {
    auto n = draw_integer(t.size, r); require(n >= 1 && n <= 63, "median size must be in [1, 63]"); return {"", 0, {double(n)}};
}
inline ChannelPlan channel_parameters(const Sharpen& t, const Array&, Random& r) { return {"", 0, {draw(t.strength, r), double(r.chance(t.p_clamp))}}; }

inline ChannelPlan channel_parameters(const BrightnessGradient& t, const Array& a, Random& r) {
    return {"", 0, {draw(t.strength, r)}, local_coefficients(t.local, Grid(a.spatial()), r, t.mean_centered)};
}
inline ChannelPlan channel_parameters(const LocalContrast& t, const Array& a, Random& r) {
    return {"", 0, {draw(t.factor, r)}, local_coefficients(t.local, Grid(a.spatial()), r)};
}
inline ChannelPlan channel_parameters(const LocalGamma& t, const Array& a, Random& r) {
    auto gamma = draw(t.gamma, r); require(gamma > 0, "gamma must be positive");
    return {"", 0, {gamma}, local_coefficients(t.local, Grid(a.spatial()), r)};
}
inline ChannelPlan channel_parameters(const LocalSmoothing& t, const Array& a, Random& r) {
    auto strength = draw(t.strength, r); probability(strength);
    auto sigma = draw(t.sigma, r); require(sigma >= 0, "smoothing sigma must be nonnegative");
    return {"", 0, {strength, sigma}, local_coefficients(t.local, Grid(a.spatial()), r)};
}
inline ChannelPlan channel_parameters(const BlankRectangle& t, const Array& a, Random& r) {
    auto shape = a.spatial(); auto dim = shape.size();
    require(t.size.size() == 1 || t.size.size() == dim, "rectangle size needs one scalar or one scalar/range per axis");
    require(t.statistic.empty() || t.statistic == "mean" || t.statistic == "median", "rectangle statistic must be mean or median");
    ChannelPlan p;
    for (int n = draw_integer(t.count, r); n > 0; --n) {
        int shared = draw_integer(t.size[0], r);
        for (std::size_t d = 0; d < dim; ++d) {
            auto size = t.square ? shared : draw_integer(t.size[t.size.size() == 1 ? 0 : d], r);
            require(size > 0, "rectangle dimensions must be positive");
            auto start = r.index(shape[d] > std::size_t(size) ? shape[d] - size + 1 : 1);
            p.values.push_back(double(start)); p.values.push_back(double(std::min(shape[d], start + size)));
        }
        p.values.push_back(t.statistic.empty() ? draw(t.value, r) : 0);
    }
    return p;
}
inline ChannelPlan channel_parameters(const SliceIllumination& t, const Array& a, Random& r) {
    auto dim = a.shape.size() - 1, d = std::size_t(axis(t.axis, dim)), n = a.shape[d + 1];
    ChannelPlan p; p.values = {double(d)}; p.field.assign(n, 1);
    for (int count = int(std::round(draw(t.defects, r))); count > 0; --count) {
        double width = draw(t.width, r), reduction = draw(t.reduction, r), pos = double(r.index(n));
        require(width > 0, "illumination defect width must be positive");
        for (std::size_t i = 0; i < n; ++i) p.field[i] *= 1 - std::exp(-.5 * std::pow((i - pos) / width, 2)) * (1 - reduction);
    }
    for (auto& v : p.field) v = std::clamp(v, .1f, 1.f);
    double bp = draw(t.base_p, r); probability(bp);
    auto count = std::size_t(std::round(bp * n));
    std::vector<double> weights(n); for (std::size_t i = 0; i < n; ++i) weights[i] = 2 - p.field[i];
    for (std::size_t j = 0; j < count; ++j) {
        double threshold = r.uniform() * std::accumulate(weights.begin(), weights.end(), 0.);
        std::size_t i = 0; for (; i + 1 < n && threshold >= weights[i]; ++i) threshold -= weights[i];
        weights[i] = 0; p.field[i] *= draw(t.base_reduction, r);
    }
    for (auto& v : p.field) v = std::clamp(v, .1f, 2.f);
    return p;
}
inline ChannelPlan channel_parameters(const Smear& t, const Array& a, Random&) {
    probability(t.alpha); require(t.previous > 0 && t.shift.size() >= a.shape.size() - 2, "invalid smear previous or shift");
    return {"", 0, {double(axis(t.axis, a.shape.size() - 1))}};
}

// Normalized intensity powers need only image precision. Preserve the wider
// calculation when conversion would underflow, or a large exponent would
// amplify rounding a normalized base close to one.
template<class T> inline double intensity_power(double base, double exponent) {
    if constexpr (std::is_same_v<T, float>) {
        if ((base == 0 || base >= std::numeric_limits<float>::min()) &&
            exponent >= std::numeric_limits<float>::min() && exponent <= 16)
            return std::pow(float(base), float(exponent));
    }
    return std::pow(base, exponent);
}

// Share the transform with compact palette consumers. Their statistics and
// bounds must be weighted by the actual number of voxels at each palette entry.
template<class T, class Statistics, class Bounds>
inline void gamma_values(std::vector<T>& v, const ChannelPlan& p, Statistics&& statistics, Bounds&& bounds) {
    bool invert = p.values[1], retain = p.values[2];
    if (invert) for (auto& x : v) x = -x;
    const auto before = retain ? statistics(v) : std::pair{0., 0.};
    const auto [low, high] = bounds(v); double range = high - low;
    for (auto& x : v) {
        T value=std::pow((x-T(low))/std::max(T(range),T(1e-7)),T(p.values[0]))*T(range)+T(low);
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(value) || !std::isfinite(T(range)))
            value=T(std::pow((double(x)-low)/std::max(range,1e-7),p.values[0])*range+low);
        x=value;
    }
    if (retain) {
        const auto after = statistics(v); double scale = before.second / std::max(after.second, 1e-7);
        for (auto& x : v) x = (x - after.first) * scale + before.first;
    }
    if (invert) for (auto& x : v) x = -x;
}
template<class T>
inline void channel_apply(const Gamma&, std::vector<T>& v, const Grid&, const ChannelPlan& p) {
    gamma_values(v, p,
        [](const auto& values) { double m = mean(values); return std::pair{m, deviation(values, m)}; },
        [](const auto& values) {
            const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
            return std::pair<double, double>{*lo, *hi};
        });
}

template<class T,class Real>
inline Real pointwise_intensity(const T& t, Real y, const ChannelPlan& p, Real m, Real low, Real high) {
    if constexpr (std::is_same_v<T, BrightnessMultiply>) y *= Real(p.values[0]);
    if constexpr (std::is_same_v<T, BrightnessAdd>) y += Real(p.values[0]);
    if constexpr (std::is_same_v<T, Invert>) y = 2 * m - y;
    if constexpr (std::is_same_v<T, Contrast>) {
        y = (y - m) * Real(p.values[0]) + m;
        if (t.preserve_range) y = std::clamp(y, low, high);
    }
    return y;
}

template<class Values> void channel_apply(const GaussianNoise&, Values& v, const Grid&, const ChannelPlan& p) {
    using T=typename Values::value_type;
    const auto add=[&](T x,T noise) {
        T y=x+T(p.values[0])*noise;
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(y)) y=T(double(x)+p.values[0]*double(noise));
        return y;
    };
    Random r{p.seed};
    for (std::size_t i = 0; i < v.size(); i += 2) {
        const auto noise = r.normal_pair<typename Values::value_type>();
        v[i] = add(v[i],noise[0]);
        if (i + 1 < v.size()) v[i + 1] = add(v[i+1],noise[1]);
    }
}
template<class Values> void channel_apply(const RicianNoise&, Values& v, const Grid&, const ChannelPlan& p) {
    Random r{p.seed}; for (auto& x : v) {
        const auto noise = r.normal_pair<typename Values::value_type>();
        using T=typename Values::value_type;
        const T re=x+T(p.values[0])*noise[0], im=T(p.values[0])*noise[1];
        T y=std::hypot(re,im);
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(y))
            y=T(std::hypot(double(x)+p.values[0]*noise[0],p.values[0]*noise[1]));
        x=y;
    }
}
template<class T> inline void channel_apply(const ClipOutliers&, std::vector<T>& v, const Grid&, const ChannelPlan& p) {
    double std = p.values[2] ? deviation(v, mean(v)) : 0; auto sorted = v;
    auto percentile = [&](double q) {
        double pos = q / 100 * (v.size() - 1); auto lo = std::size_t(pos), hi = std::min(lo + 1, v.size() - 1);
        std::nth_element(sorted.begin(), sorted.begin() + lo, sorted.end());
        double low = sorted[lo], high = hi == lo ? low : *std::min_element(sorted.begin() + hi, sorted.end());
        return low + (high - low) * (pos - lo);
    };
    double low = percentile(p.values[0]), high = percentile(p.values[1]);
    for (auto& x : v) x = std::clamp(double(x), low, high);
    if (p.values[2]) { double m = mean(v), after = deviation(v, m); if (after > 1e-8) for (auto& x : v) x = (x - m) / after * std + m; }
}
template<class T>
inline void channel_apply(const GaussianBlur& t, std::vector<T>& v, const Grid& g, const ChannelPlan& p) { gaussian_blur(v, g, p.values, t.truncate, t.boundary, t.full_width); }
// Exact palette compression, not quantization. This benefits CT patches and
// pointwise intensity transforms of them; continuous-valued inputs use the
// small-window network or nth_element. The bounded palette stays in L1 cache.
// Fixed open addressing avoids allocating/hash-dividing for every float voxel.
// Compare bit patterns so encoding is lossless; mixed signed zeros retain the
// ordinary median selector's tie behavior via fallback.
template<class Values>
inline bool float_palette(const Values& v, std::vector<std::uint8_t>& codes,
                          std::vector<float>& values, bool sorted = true) {
    constexpr std::size_t slots = 1024;
    std::array<std::uint32_t, slots> keys{};
    std::array<std::uint16_t, slots> indices;
    indices.fill(256);
    values.clear();
    bool positive_zero = false, negative_zero = false;
    auto encode = [&](float value) -> int {
        if (!std::isfinite(value)) return -1;
        if (value == 0) { if (std::signbit(value)) negative_zero = true; else positive_zero = true; }
        if (positive_zero && negative_zero) return -1;
        const auto bits = std::bit_cast<std::uint32_t>(value);
        auto slot = (bits * UINT32_C(0x9e3779b9)) >> 22;
        while (indices[slot] != 256 && keys[slot] != bits) slot = (slot + 1) % slots;
        if (indices[slot] == 256) {
            if (values.size() == 256) return -1;
            keys[slot] = bits; indices[slot] = std::uint16_t(values.size()); values.push_back(value);
        }
        return indices[slot];
    };
    const auto probes = std::min<std::size_t>(4096, v.size());
    for (std::size_t i = 0; i < probes; ++i) if (encode(v[i * (v.size() / probes)]) < 0) { values.clear(); return false; }
    codes.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const auto code = encode(v[i]);
        if (code < 0) { values.clear(); return false; }
        codes[i] = std::uint8_t(code);
    }
    if (sorted) {
        auto previous = values;
        std::sort(values.begin(), values.end());
        std::array<std::uint8_t, 256> ranks{};
        for (std::size_t i = 0; i < previous.size(); ++i)
            ranks[i] = std::uint8_t(std::lower_bound(values.begin(), values.end(), previous[i]) - values.begin());
        for (auto& code : codes) code = ranks[code];
    }
    return true;
}
template<class Values, class T>
inline bool median_palette(const Values& v, std::vector<std::uint8_t>& codes,
                           std::vector<T>& values) {
    if constexpr (std::is_same_v<T, float>) return float_palette(v, codes, values);
    constexpr std::size_t limit = 256;
    std::unordered_map<T, std::uint16_t> palette;
    palette.reserve(limit + 1);
    bool positive_zero = false, negative_zero = false;
    auto encode = [&](T value) -> int {
        // Preserve the general selector's handling of signed-zero ties.
        if (value == 0) { if (std::signbit(value)) negative_zero = true; else positive_zero = true; }
        if (std::isnan(value) || (positive_zero && negative_zero)) return -1;
        const auto [it, inserted] = palette.try_emplace(value, static_cast<std::uint16_t>(palette.size()));
        return palette.size() <= limit ? int(it->second) : -1;
    };
    // Reject typical high-cardinality inputs before allocating an index volume.
    const auto probes = std::min<std::size_t>(4096, v.size());
    for (std::size_t i = 0; i < probes; ++i) if (encode(v[i * (v.size() / probes)]) < 0) return false;
    codes.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const auto code = encode(v[i]);
        if (code < 0) return false;
        codes[i] = static_cast<std::uint8_t>(code);
    }
    for (const auto& [value, code] : palette) values.push_back(value);
    std::sort(values.begin(), values.end());
    std::array<std::uint8_t, limit> ranks{};
    for (std::size_t rank = 0; rank < values.size(); ++rank) ranks[palette.at(values[rank])] = static_cast<std::uint8_t>(rank);
    for (auto& code : codes) code = ranks[code];
    return true;
}

template<class Values, class T>
inline bool encode_median_palette(const Values& v, std::vector<std::uint8_t>& codes,
                                  std::vector<T>& values, bool quantize) {
    constexpr std::size_t limit = 256;
    if (!median_palette(v, codes, values)) {
        if (!quantize || !std::all_of(v.begin(), v.end(), [](T x) { return std::isfinite(x); })) return false;
        const auto bounds = std::minmax_element(v.begin(), v.end());
        const double low = *bounds.first, high = *bounds.second, range = high - low;
        // Keep the exact fallback for unrepresentable ranges and signed-zero ties.
        if (!(range > 0) || !std::isfinite(range)) return false;
        values.resize(limit);
        for (std::size_t code = 0; code < limit; ++code)
            values[code] = T(std::lerp(low, high, double(code) / (limit - 1)));
        codes.resize(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            codes[i] = std::uint8_t(std::clamp(std::floor((double(v[i]) - low) / range * (limit - 1) + .5), 0., double(limit - 1)));
        // Monotone quantization commutes with rank selection: the result is
        // the quantized exact median, within range/510 plus output rounding.
    }
    return true;
}

// Exact rank selection shared by ordinary and compact images.
template<class Output, class Values>
inline void median_rank_filter(const std::vector<std::uint8_t>& codes, Output& v, const Values& values,
                               const Grid& g, std::size_t k, const std::vector<std::size_t>& iz,
                               const std::vector<std::size_t>& iy, const std::vector<std::size_t>& ix) {
    if (g.shape.size() == 3 && k == 3) {
        median_detail::median3_filter<64>(codes, v, g, iz, iy, ix,
                                        [&](std::uint8_t code) { return values[code]; });
        return;
    }
    constexpr std::size_t limit = 256;
    const auto dim = g.shape.size(), nz = dim == 3 ? g.shape[0] : 1;
    const auto ny = g.shape[dim - 2], nx = g.shape.back(), kz = dim == 3 ? k : 1;
    const auto middle = kz * k * k / 2;
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y) {
        std::array<std::uint32_t, limit> counts{};
        for (std::size_t dz = 0; dz < kz; ++dz) for (std::size_t dy = 0; dy < k; ++dy) {
            const auto* row = codes.data() + iz[z * kz + dz] + iy[y * k + dy];
            for (std::size_t dx = 0; dx < k; ++dx) ++counts[row[ix[dx]]];
        }
        // Adjacent windows tend to have nearby medians. Maintain the count
        // below the previous median as samples enter/leave, then adjust its
        // rank. This avoids a second histogram and restarting every search.
        std::size_t code = 0, below = 0;
        while (below + counts[code] <= middle) below += counts[code++];
        for (std::size_t x = 0; x < nx; ++x) {
            v[(z * ny + y) * nx + x] = values[code];
            if (x + 1 == nx) break;
            const auto leave = ix[x * k], enter = ix[(x + 1) * k + k - 1];
            for (std::size_t dz = 0; dz < kz; ++dz) for (std::size_t dy = 0; dy < k; ++dy) {
                const auto* row = codes.data() + iz[z * kz + dz] + iy[y * k + dy];
                auto from = row[leave], to = row[enter];
                if (from == to) continue;
                --counts[from]; ++counts[to];
                below -= from < code;
                below += to < code;
            }
            // Walk nearby ranks cheaply, but sum blocks when crossing a long
            // run of empty bins (e.g. sharp dark/bright stripes in a full
            // palette). This bounds branch-heavy searching on such inputs.
            if (below > middle) {
                for (int n = 0; n < 4 && below > middle; ++n) below -= counts[--code];
                while (below > middle && code >= 16) {
                    const auto mass = std::accumulate(counts.begin() + code - 16, counts.begin() + code, std::size_t(0));
                    if (below - mass <= middle) break;
                    below -= mass; code -= 16;
                }
                while (below > middle) below -= counts[--code];
            } else if (below + counts[code] <= middle) {
                for (int n = 0; n < 4 && below + counts[code] <= middle; ++n) below += counts[code++];
                while (below + counts[code] <= middle && code + 16 <= limit) {
                    const auto mass = std::accumulate(counts.begin() + code, counts.begin() + code + 16, std::size_t(0));
                    if (below + mass > middle) break;
                    below += mass; code += 16;
                }
                while (below + counts[code] <= middle) below += counts[code++];
            }
        }
    }
}

inline std::vector<std::size_t> median_indices(std::size_t n, std::size_t width, std::size_t stride) {
    std::vector<std::size_t> result(n * width);
    for (std::size_t x = 0; x < n; ++x) for (std::size_t j = 0; j < width; ++j)
        result[x * width + j] = reflect(std::int64_t(x + j) - std::int64_t(width / 2), n) * stride;
    return result;
}
template<class T>
inline bool median_histogram(std::vector<T>& v, const Grid& g, std::size_t k,
                             const std::vector<std::size_t>& iz, const std::vector<std::size_t>& iy,
                             const std::vector<std::size_t>& ix, bool quantize) {
    std::vector<std::uint8_t> codes;
    std::vector<T> values;
    if (!encode_median_palette(v, codes, values, quantize)) return false;
    if (values.size() > 1) median_rank_filter(codes, v, values, g, k, iz, iy, ix);
    return true;
}

template<class T>
inline void channel_apply(const MedianFilter& t, std::vector<T>& v, const Grid& g, const ChannelPlan& p) {
    auto k = std::size_t(p.values[0]);
    if (k == 1) return;
    // Precompute reflected coordinates once per axis, rather than performing
    // division and modulo for every neighbor of every voxel.
    const auto dim = g.shape.size();
    const auto nz = dim == 3 ? g.shape[0] : 1, ny = g.shape[dim - 2], nx = g.shape.back();
    const auto kz = dim == 3 ? k : 1;
    auto iz = median_indices(nz, kz, ny * nx), iy = median_indices(ny, k, nx), ix = median_indices(nx, k, 1);
    if (median_histogram(v, g, k, iz, iy, ix, t.quantize)) return;
    if (dim == 3 && k == 3 && median_detail::median3(v, g, iz, iy, ix)) return;
    auto src = v;
    std::vector<T> values(kz * k * k);
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; ++x) {
            std::size_t j = 0;
            for (std::size_t dz = 0; dz < kz; ++dz) for (std::size_t dy = 0; dy < k; ++dy) {
                const auto* row = src.data() + iz[z * kz + dz] + iy[y * k + dy];
                for (std::size_t dx = 0; dx < k; ++dx) values[j++] = row[ix[x * k + dx]];
            }
            std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
            v[(z * ny + y) * nx + x] = values[values.size() / 2];
        }
}
template<class T>
inline void channel_apply(const Sharpen&, std::vector<T>& v, const Grid& g, const ChannelPlan& p) {
    T low = 0, high = 0;
    if (p.values[1]) { const auto bounds = std::minmax_element(v.begin(), v.end()); low = *bounds.first; high = *bounds.second; }
    const T strength = T(p.values[0]);
    const auto dim = g.shape.size(), nz = dim == 3 ? g.shape[0] : 1;
    const auto ny = g.shape[dim - 2], nx = g.shape.back(), plane = ny * nx;
    std::vector<T> output(g.size);
    // Fuse the stencil and final update, retaining the original z/y/x
    // subtraction order. Neighbor rows are reused across contiguous outputs.
    for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y) {
        const auto base = z * plane + y * nx;
        const auto* row = v.data() + base;
        const auto* above = row - (y ? nx : 0);
        const auto* below = row + (y + 1 < ny ? nx : 0);
        const auto* front = row - (z ? plane : 0);
        const auto* back = row + (z + 1 < nz ? plane : 0);
        for (std::size_t x = 0; x < nx; ++x) {
            T laplace = T(2 * dim) * row[x];
            if (dim == 3) { laplace -= front[x]; laplace -= back[x]; }
            laplace -= above[x]; laplace -= below[x];
            laplace -= row[x ? x - 1 : x]; laplace -= row[x + 1 < nx ? x + 1 : x];
            T value = row[x] + strength * laplace;
            output[base + x] = p.values[1] ? std::clamp(value, low, high) : value;
        }
    }
    v.swap(output);
}
template<class Values> void channel_apply(const BrightnessGradient& t, Values& v, const Grid& g, const ChannelPlan& p) {
    using T=typename Values::value_type;
    T low=0,high=0;
    if(t.clip) { const auto [lo,hi]=std::minmax_element(v.begin(),v.end()); low=*lo; high=*hi; }
    LocalFieldView(p.field,g).each([&](std::size_t i,float w) {
        T value=v[i]+T(w)*T(p.values[0]);
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(value)) value=T(double(v[i])+w*p.values[0]);
        if(t.clip)value=std::clamp(value,low,high);v[i]=value;
    });
}
template<class Values> double local_mean(const Values& v, const LocalFieldView& field) {
    double sum = 0, weight = 1e-8;
    field.each([&](std::size_t i,float w) { sum += double(v[i])*w; weight += w; });
    return sum / weight;
}
template<class Values> void channel_apply(const LocalContrast&, Values& v, const Grid& g, const ChannelPlan& p) {
    using T=typename Values::value_type;
    const LocalFieldView field(p.field,g);
    const double wide_mean=local_mean(v,field);const T m=T(wide_mean),factor=T(p.values[0]);
    field.each([&](std::size_t i,float weight) {
        const T w=T(weight);T value=v[i]*(T(1)-w)+((v[i]-m)*factor+m)*w;
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(value))
            value=T(double(v[i])*(1-double(w))+((double(v[i])-wide_mean)*p.values[0]+wide_mean)*w);
        v[i]=value;
    });
}
template<class Values> void channel_apply(const LocalGamma&, Values& v, const Grid& g, const ChannelPlan& p) {
    using T = typename Values::value_type;
    auto [lo, hi] = std::minmax_element(v.begin(), v.end());T low=*lo,span=std::max(*hi-low,T(1e-8));const double wide_low=*lo,wide_high=*hi;
    LocalFieldView(p.field,g).each([&](std::size_t i,float weight) {
        const T norm=(v[i]-low)/span,w=T(weight);T value=(norm*(T(1)-w)+std::pow(norm,T(p.values[0]))*w)*span+low;
        if constexpr(std::is_same_v<T,float>) if(!std::isfinite(value)) {
            const double range=std::max(wide_high-wide_low,1e-8),n=(double(v[i])-wide_low)/range;
            value=T((n*(1-double(w))+std::pow(n,p.values[0])*w)*range+wide_low);
        }
        v[i]=value;
    });
}
template<class Values>
inline void channel_apply(const LocalSmoothing&, Values& v, const Grid& g, const ChannelPlan& p) {
    using T = typename Values::value_type;
    std::vector<T> smoothed(v.begin(), v.end());
    gaussian_blur(smoothed, g, std::vector<double>(g.shape.size(), p.values[1]));
    LocalFieldView(p.field,g).each([&](std::size_t i,float weight) { T w = T(weight) * T(p.values[0]); v[i] = v[i] * (T(1) - w) + smoothed[i] * w; });
}
// Shared region/statistic calculation; overlapping boxes observe earlier fills.
struct RectangleRegion {
    Shape extent;
    std::size_t origin = 0;
    RectangleRegion(const Grid& g, const ChannelPlan& p, std::size_t b) : extent(g.shape.size()) {
        for (std::size_t a = 0; a < extent.size(); ++a) {
            extent[a] = std::size_t(p.values[b + 2*a + 1] - p.values[b + 2*a]);
            origin += std::size_t(p.values[b + 2*a]) * g.stride[a];
        }
    }
};
template<class Values> double rectangle_value(const Values& v, const Grid& g, const ChannelPlan& p,
                                              std::size_t b, const std::string& statistic) {
    if (statistic.empty()) return p.values[b + 2*g.shape.size()];
    RectangleRegion region(g,p,b); Grid box(region.extent);
    std::vector<double> values; values.reserve(box.size);
    for (std::size_t j = 0; j < box.size; ++j) {
        auto i = region.origin;
        for (std::size_t a = 0; a < g.shape.size(); ++a) i += box.coord(j,a)*g.stride[a];
        values.push_back(v[i]);
    }
    if (statistic == "mean") return mean(values);
    const auto mid = (values.size()-1)/2;
    std::nth_element(values.begin(),values.begin()+mid,values.end()); return values[mid];
}
template<class Values> void channel_apply(const BlankRectangle& t, Values& v, const Grid& g, const ChannelPlan& p) {
    for (std::size_t b = 0; b < p.values.size(); b += 2*g.shape.size()+1) {
        RectangleRegion region(g,p,b); Grid box(region.extent);
        const double value = rectangle_value(v,g,p,b,t.statistic);
        const auto width = region.extent.back();
        for (std::size_t row = 0; row < box.size; row += width) {
            auto start = region.origin;
            for (std::size_t a = 0; a+1 < g.shape.size(); ++a) start += box.coord(row,a)*g.stride[a];
            std::fill_n(v.begin()+start,width,value);
        }
    }
}
template<class Values> void channel_apply(const SliceIllumination&, Values& v, const Grid& g, const ChannelPlan& p) {
    const auto a = std::size_t(p.values[0]);
    for (std::size_t base = 0; base < g.size; base += g.shape[a] * g.stride[a])
        for (std::size_t x = 0; x < g.shape[a]; ++x)
            for (std::size_t j = 0; j < g.stride[a]; ++j) {
                auto i = base + x * g.stride[a] + j;
                v[i] *= typename Values::value_type(p.field[x]);
            }
}
template<class T> inline void channel_apply(const Smear& t, std::vector<T>& v, const Grid& g, const ChannelPlan& p) {
    auto axis = std::size_t(p.values[0]);
    auto plane_shape = g.shape; plane_shape.erase(plane_shape.begin() + axis); Grid plane(plane_shape);
    // Cache in-plane addresses; the shifted slice is the same for each previous
    // depth. Slice order remains recursive, matching the original transform.
    std::vector<std::size_t> target(plane.size), source(plane.size);
    for (std::size_t pixel = 0; pixel < plane.size; ++pixel) {
        std::size_t pa = 0;
        for (std::size_t a = 0; a < g.shape.size(); ++a) if (a != axis) {
            auto x = plane.coord(pixel, pa); auto n = std::int64_t(g.shape[a]);
            auto shifted = (std::int64_t(x) - t.shift[pa++]) % n;
            target[pixel] += x * g.stride[a]; source[pixel] += std::size_t((shifted + n) % n) * g.stride[a];
        }
    }
    for (std::size_t slice = t.previous; slice < g.shape[axis]; ++slice)
        for (std::size_t pixel = 0; pixel < plane.size; ++pixel) {
            auto i = slice * g.stride[axis] + target[pixel];
            double sum = 0;
            for (int j = 1; j <= t.previous; ++j) sum += v[(slice - j) * g.stride[axis] + source[pixel]];
            v[i] = (1 - t.alpha) * v[i] + t.alpha * sum / t.previous;
        }
}

template<class T> requires requires(T t) { t.options; }
Parameters sample(const T& t, const Sample& s, Random& r) {
    auto result = sample_channels(s, t.options, r, [&](const Array& a, Random& rng) { return channel_parameters(t, a, rng); });
    if constexpr (std::is_same_v<T, GaussianNoise> || std::is_same_v<T, RicianNoise>) {
        auto base = r.next(); for (auto& c : result.channels) c.seed = t.shared_noise ? split_seed(base, c.key) : r.next();
    }
    return result;
}
template<class T> requires requires(T t) { t.options; }
void apply(const T& t, Sample& s, const Parameters& p) {
    if constexpr (std::is_same_v<T, Gamma> || std::is_same_v<T, GaussianBlur> || std::is_same_v<T, MedianFilter> ||
                  std::is_same_v<T, Sharpen> || std::is_same_v<T, LocalSmoothing>) {
        for (const auto& c : p.channels) {
            auto& a = s.at(c.key).array; Grid g(a.spatial());
            if (a.bytes == 4) {
                auto* data = static_cast<float*>(a.data()) + c.channel * g.size;
                // Float arithmetic is sufficient for ordinary images. Retain
                // the wider path near overflow (including large strengths),
                // where finite inputs can have unrepresentable intermediates.
                if constexpr (!std::is_same_v<T, MedianFilter> && !std::is_same_v<T, Gamma>) {
                    double limit = std::numeric_limits<float>::max() / 64.;
                    if constexpr (std::is_same_v<T, Sharpen>) limit /= std::max(1., std::abs(c.values[0]));
                    if (std::any_of(data, data + g.size, [&](float x) { return std::abs(double(x)) > limit; }) ||
                        (std::is_same_v<T, Sharpen> && std::abs(c.values[0]) > std::numeric_limits<float>::max())) {
                        auto values = a.channel(c.channel); channel_apply(t, values, g, c); a.channel(c.channel, values);
                        continue;
                    }
                }
                if constexpr (std::is_same_v<T, LocalSmoothing>) {
                    // The blur already owns a separate buffer. Blend directly
                    // into the writable image instead of copying out and back.
                    std::span<float> values(data, g.size);
                    channel_apply(t, values, g, c);
                } else {
                    std::vector<float> values(data, data + g.size);
                    channel_apply(t, values, g, c);
                    std::copy(values.begin(), values.end(), data);
                }
            } else {
                auto values = a.channel(c.channel); channel_apply(t, values, g, c); a.channel(c.channel, values);
            }
        }
        restore_valid(s);
    } else if constexpr (std::is_same_v<T, BrightnessMultiply> || std::is_same_v<T, BrightnessAdd> ||
                  std::is_same_v<T, Contrast> || std::is_same_v<T, Invert> ||
                  std::is_same_v<T, BlankRectangle> || std::is_same_v<T, SliceIllumination> ||
                  std::is_same_v<T, GaussianNoise> || std::is_same_v<T, RicianNoise> ||
                  std::is_same_v<T, BrightnessGradient> || std::is_same_v<T, LocalContrast> || std::is_same_v<T, LocalGamma>) {
        for (const auto& c : p.channels) {
            auto& a = s.at(c.key).array; Grid g(a.spatial());
            if constexpr (std::is_same_v<T, BlankRectangle>) {
                if (!t.statistic.empty()) { auto v = a.channel(c.channel); channel_apply(t, v, g, c); a.channel(c.channel, v); continue; }
            }
            auto run = [&](auto* data) {
                using V = std::remove_pointer_t<decltype(data)>;
                std::span<V> v(data + c.channel * g.size, g.size);
                if constexpr (std::is_same_v<T, BlankRectangle> || std::is_same_v<T, SliceIllumination> ||
                              std::is_same_v<T, GaussianNoise> || std::is_same_v<T, RicianNoise> ||
                              std::is_same_v<T, BrightnessGradient> || std::is_same_v<T, LocalContrast> || std::is_same_v<T, LocalGamma>)
                    channel_apply(t, v, g, c);
                else {
                    double m = 0, low = 0, high = 0;
                    if constexpr (std::is_same_v<T, Contrast> || std::is_same_v<T, Invert>) {
                        for (auto x : v) m += double(x);
                        m /= v.size();
                    }
                    if constexpr (std::is_same_v<T, Contrast>) {
                        auto bounds = std::minmax_element(v.begin(), v.end()); low = *bounds.first; high = *bounds.second;
                    }
                    for (auto& x : v) {
                        V y = pointwise_intensity(t, x, c, V(m), V(low), V(high));
                        if constexpr (std::is_same_v<V,float>)
                            if (!std::isfinite(y)) y = V(pointwise_intensity(t, double(x), c, m, low, high));
                        x = y;
                    }
                }
            };
            if (a.bytes == 4) run(static_cast<float*>(a.data()));
            else run(static_cast<double*>(a.data()));
        }
        restore_valid(s);
    } else apply_channels(s, p, [&](auto& v, const Grid& g, const ChannelPlan& c) { channel_apply(t, v, g, c); });
}
} // namespace vcr::augment
