#pragma once
#include "core.hpp"
#include <span>

namespace vcr::augment {
struct BrightnessMultiply { Scalar factor = Uniform{.75, 1.25}; ChannelOptions options; };
struct BrightnessAdd { Scalar shift = Normal{0, .1}; ChannelOptions options; };
struct Contrast { Scalar factor = Uniform{.75, 1.25}; bool preserve_range = true; ChannelOptions options; };
struct Gamma { Scalar gamma = Uniform{.7, 1.5}; double p_invert = 0, p_retain_stats = 0; ChannelOptions options; };
struct Invert { ChannelOptions options; };
struct GaussianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct RicianNoise { Scalar std = Uniform{0, .1}; bool shared_noise = false; ChannelOptions options; };
struct ClipOutliers { Scalar lower = .2, upper = 99.8; double p_retain_std = 1; ChannelOptions options; };
struct GaussianBlur { Scalar sigma = Uniform{.3, 1.5}; bool shared_axes = true, mirror_boundary = true; double truncate = 6; ChannelOptions options; };
struct MedianFilter { Scalar size = Uniform{3, 7}; ChannelOptions options; };
struct Sharpen { Scalar strength = Uniform{.1, .5}; double p_clamp = .5; ChannelOptions options; };
struct LocalOptions { Scalar scale = Uniform{10, 30}, loc = Uniform{-1, 2}; };
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

inline std::vector<double> local_kernel(const LocalOptions& t, const Grid& g, Random& r, bool centered = false) {
    std::vector<std::vector<double>> kernels(g.shape.size());
    for (std::size_t a = 0; a < g.shape.size(); ++a) {
        double center = draw(t.loc, r) * g.shape[a], sigma = draw(t.scale, r);
        require(sigma > 0, "local Gaussian scale must be positive");
        kernels[a].resize(g.shape[a]);
        for (std::size_t x = 0; x < g.shape[a]; ++x)
            kernels[a][x] = .5 * (std::erf((x + .5 - center) / (sigma * std::sqrt(2.))) - std::erf((x - .5 - center) / (sigma * std::sqrt(2.))));
    }
    std::vector<double> out(g.size, 1);
    for (std::size_t a = 0; a < g.shape.size(); ++a)
        for (std::size_t base = 0; base < g.size; base += g.shape[a] * g.stride[a])
            for (std::size_t x = 0; x < g.shape[a]; ++x)
                for (std::size_t j = 0; j < g.stride[a]; ++j)
                    out[base + x * g.stride[a] + j] *= kernels[a][x];
    auto [lo, hi] = std::minmax_element(out.begin(), out.end()); double low = *lo, range = *hi - low;
    for (auto& v : out) v = range ? (v - low) / range : 0;
    if (centered) {
        double m = mean(out), maxabs = 0;
        for (auto& v : out) maxabs = std::max(maxabs, std::abs(v -= m));
        if (maxabs > 1e-8) for (auto& v : out) v /= maxabs;
    }
    return out;
}
inline ChannelPlan channel_parameters(const BrightnessGradient& t, const Array& a, Random& r) {
    return {"", 0, {draw(t.strength, r)}, local_kernel(t.local, Grid(a.spatial()), r, t.mean_centered)};
}
inline ChannelPlan channel_parameters(const LocalContrast& t, const Array& a, Random& r) {
    return {"", 0, {draw(t.factor, r)}, local_kernel(t.local, Grid(a.spatial()), r)};
}
inline ChannelPlan channel_parameters(const LocalGamma& t, const Array& a, Random& r) {
    auto gamma = draw(t.gamma, r); require(gamma > 0, "gamma must be positive");
    return {"", 0, {gamma}, local_kernel(t.local, Grid(a.spatial()), r)};
}
inline ChannelPlan channel_parameters(const LocalSmoothing& t, const Array& a, Random& r) {
    auto strength = draw(t.strength, r); probability(strength);
    auto sigma = draw(t.sigma, r); require(sigma >= 0, "smoothing sigma must be nonnegative");
    return {"", 0, {strength, sigma}, local_kernel(t.local, Grid(a.spatial()), r)};
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
    for (auto& v : p.field) v = std::clamp(v, .1, 1.);
    double bp = draw(t.base_p, r); probability(bp);
    auto count = std::size_t(std::round(bp * n));
    std::vector<double> weights(n); for (std::size_t i = 0; i < n; ++i) weights[i] = 2 - p.field[i];
    for (std::size_t j = 0; j < count; ++j) {
        double threshold = r.uniform() * std::accumulate(weights.begin(), weights.end(), 0.);
        std::size_t i = 0; for (; i + 1 < n && threshold >= weights[i]; ++i) threshold -= weights[i];
        weights[i] = 0; p.field[i] *= draw(t.base_reduction, r);
    }
    for (auto& v : p.field) v = std::clamp(v, .1, 2.);
    return p;
}
inline ChannelPlan channel_parameters(const Smear& t, const Array& a, Random&) {
    probability(t.alpha); require(t.previous > 0 && t.shift.size() >= a.shape.size() - 2, "invalid smear previous or shift");
    return {"", 0, {double(axis(t.axis, a.shape.size() - 1))}};
}

inline void channel_apply(const Gamma&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    bool invert = p.values[1], retain = p.values[2];
    if (invert) for (auto& x : v) x = -x;
    double m = retain ? mean(v) : 0, std = retain ? deviation(v, m) : 0;
    auto [lo, hi] = std::minmax_element(v.begin(), v.end()); double low = *lo, range = *hi - low;
    for (auto& x : v) x = std::pow((x - low) / std::max(range, 1e-7), p.values[0]) * range + low;
    if (retain) {
        double after = mean(v), scale = std / std::max(deviation(v, after), 1e-7);
        for (auto& x : v) x = (x - after) * scale + m;
    }
    if (invert) for (auto& x : v) x = -x;
}

inline void channel_apply(const GaussianNoise&, std::vector<double>& v, const Grid&, const ChannelPlan& p) { Random r{p.seed}; for (auto& x : v) x += p.values[0] * r.normal(); }
inline void channel_apply(const RicianNoise&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    Random r{p.seed}; for (auto& x : v) { double re = x + p.values[0] * r.normal(), im = p.values[0] * r.normal(); x = std::sqrt(re * re + im * im); }
}
inline void channel_apply(const ClipOutliers&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    double std = p.values[2] ? deviation(v, mean(v)) : 0; auto sorted = v;
    auto percentile = [&](double q) {
        double pos = q / 100 * (v.size() - 1); auto lo = std::size_t(pos), hi = std::min(lo + 1, v.size() - 1);
        std::nth_element(sorted.begin(), sorted.begin() + lo, sorted.end());
        double low = sorted[lo], high = hi == lo ? low : *std::min_element(sorted.begin() + hi, sorted.end());
        return low + (high - low) * (pos - lo);
    };
    double low = percentile(p.values[0]), high = percentile(p.values[1]);
    for (auto& x : v) x = std::clamp(x, low, high);
    if (p.values[2]) { double m = mean(v), after = deviation(v, m); if (after > 1e-8) for (auto& x : v) x = (x - m) / after * std + m; }
}
inline void channel_apply(const GaussianBlur& t, std::vector<double>& v, const Grid& g, const ChannelPlan& p) { gaussian_blur(v, g, p.values, t.truncate, t.mirror_boundary, true); }
inline void channel_apply(const MedianFilter&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto k = std::size_t(p.values[0]);
    if (k == 1) return;
    // Precompute reflected coordinates once per axis, rather than performing
    // division and modulo for every neighbor of every voxel.
    const auto dim = g.shape.size();
    const auto nz = dim == 3 ? g.shape[0] : 1, ny = g.shape[dim - 2], nx = g.shape.back();
    const auto kz = dim == 3 ? k : 1;
    auto indices = [&](std::size_t n, std::size_t width, std::size_t stride) {
        std::vector<std::size_t> result(n * width);
        for (std::size_t x = 0; x < n; ++x) for (std::size_t j = 0; j < width; ++j)
            result[x * width + j] = reflect(std::int64_t(x + j) - std::int64_t(width / 2), n) * stride;
        return result;
    };
    auto iz = indices(nz, kz, ny * nx), iy = indices(ny, k, nx), ix = indices(nx, k, 1);
    auto src = v;
    std::vector<double> values(kz * k * k);
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
inline void channel_apply(const Sharpen&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto [lo, hi] = std::minmax_element(v.begin(), v.end());
    const double low = *lo, high = *hi;
    std::vector<double> laplace(g.size);
    for (std::size_t i = 0; i < g.size; ++i) laplace[i] = 2 * g.shape.size() * v[i];
    for (std::size_t a = 0; a < g.shape.size(); ++a) {
        const auto step = g.stride[a], extent = g.shape[a], block = step * extent;
        for (std::size_t base = 0; base < g.size; base += block)
            for (std::size_t x = 0; x < extent; ++x) {
                const auto at = base + x * step;
                const auto before = x ? at - step : at, after = x + 1 < extent ? at + step : at;
                for (std::size_t j = 0; j < step; ++j) {
                    laplace[at + j] -= v[before + j]; laplace[at + j] -= v[after + j];
                }
            }
    }
    for (std::size_t i = 0; i < g.size; ++i) {
        v[i] += p.values[0] * laplace[i]; if (p.values[1]) v[i] = std::clamp(v[i], low, high);
    }
}
inline void channel_apply(const BrightnessGradient& t, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    auto [lo, hi] = std::minmax_element(v.begin(), v.end()); double low = *lo, high = *hi;
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] += p.field[i] * p.values[0]; if (t.clip) v[i] = std::clamp(v[i], low, high); }
}
inline void channel_apply(const LocalContrast&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    double sum = 0, weight = 1e-8; for (std::size_t i = 0; i < v.size(); ++i) { sum += v[i] * p.field[i]; weight += p.field[i]; }
    double m = sum / weight;
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = v[i] * (1 - p.field[i]) + ((v[i] - m) * p.values[0] + m) * p.field[i];
}
inline void channel_apply(const LocalGamma&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    auto [lo, hi] = std::minmax_element(v.begin(), v.end()); double low = *lo, span = std::max(*hi - low, 1e-8);
    for (std::size_t i = 0; i < v.size(); ++i) { double norm = (v[i] - low) / span; v[i] = (norm * (1 - p.field[i]) + std::pow(norm, p.values[0]) * p.field[i]) * span + low; }
}
inline void channel_apply(const LocalSmoothing&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto smoothed = v; gaussian_blur(smoothed, g, std::vector<double>(g.shape.size(), p.values[1]));
    for (std::size_t i = 0; i < v.size(); ++i) { double w = p.field[i] * p.values[0]; v[i] = v[i] * (1 - w) + smoothed[i] * w; }
}
template<class Values> void channel_apply(const BlankRectangle& t, Values& v, const Grid& g, const ChannelPlan& p) {
    auto dim = g.shape.size(), stride = 2 * dim + 1;
    for (std::size_t b = 0; b < p.values.size(); b += stride) {
        std::vector<std::size_t> indices; std::vector<double> values;
        Shape extent(dim);
        std::size_t origin = 0;
        for (std::size_t a = 0; a < dim; ++a) {
            extent[a] = std::size_t(p.values[b + 2 * a + 1] - p.values[b + 2 * a]);
            origin += std::size_t(p.values[b + 2 * a]) * g.stride[a];
        }
        Grid box(extent);
        indices.reserve(box.size); values.reserve(box.size);
        for (std::size_t j = 0; j < box.size; ++j) {
            auto i = origin;
            for (std::size_t a = 0; a < dim; ++a) i += box.coord(j, a) * g.stride[a];
            indices.push_back(i); values.push_back(v[i]);
        }
        if (indices.empty()) continue;
        double value = p.values[b + 2 * dim];
        if (t.statistic == "mean") value = mean(values);
        else if (t.statistic == "median") { auto mid = (values.size() - 1) / 2; std::nth_element(values.begin(), values.begin() + mid, values.end()); value = values[mid]; }
        for (auto i : indices) v[i] = value;
    }
}
template<class Values> void channel_apply(const SliceIllumination&, Values& v, const Grid& g, const ChannelPlan& p) {
    const auto a = std::size_t(p.values[0]);
    for (std::size_t base = 0; base < g.size; base += g.shape[a] * g.stride[a])
        for (std::size_t x = 0; x < g.shape[a]; ++x)
            for (std::size_t j = 0; j < g.stride[a]; ++j) {
                auto i = base + x * g.stride[a] + j;
                v[i] = double(v[i]) * p.field[x];
            }
}
inline void channel_apply(const Smear& t, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
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
    if constexpr (std::is_same_v<T, BrightnessMultiply> || std::is_same_v<T, BrightnessAdd> ||
                  std::is_same_v<T, Contrast> || std::is_same_v<T, Invert> ||
                  std::is_same_v<T, BlankRectangle> || std::is_same_v<T, SliceIllumination>) {
        for (const auto& c : p.channels) {
            auto& a = s.at(c.key).array; Grid g(a.spatial());
            if constexpr (std::is_same_v<T, BlankRectangle>) {
                if (!t.statistic.empty()) { auto v = a.channel(c.channel); channel_apply(t, v, g, c); a.channel(c.channel, v); continue; }
            }
            auto run = [&](auto* data) {
                using V = std::remove_pointer_t<decltype(data)>;
                std::span<V> v(data + c.channel * g.size, g.size);
                if constexpr (std::is_same_v<T, BlankRectangle> || std::is_same_v<T, SliceIllumination>)
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
                        double y = x;
                        if constexpr (std::is_same_v<T, BrightnessMultiply>) y *= c.values[0];
                        if constexpr (std::is_same_v<T, BrightnessAdd>) y += c.values[0];
                        if constexpr (std::is_same_v<T, Invert>) y = 2 * m - y;
                        if constexpr (std::is_same_v<T, Contrast>) {
                            y = (y - m) * c.values[0] + m;
                            if (t.preserve_range) y = std::clamp(y, low, high);
                        }
                        x = static_cast<V>(y);
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
