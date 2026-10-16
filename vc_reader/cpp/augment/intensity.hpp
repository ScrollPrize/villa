#pragma once
#include "core.hpp"

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
    for (std::size_t i = 0; i < g.size; ++i)
        for (std::size_t a = 0; a < g.shape.size(); ++a) out[i] *= kernels[a][g.coord(i, a)];
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

inline void channel_apply(const BrightnessMultiply&, std::vector<double>& v, const Grid&, const ChannelPlan& p) { for (auto& x : v) x *= p.values[0]; }
inline void channel_apply(const BrightnessAdd&, std::vector<double>& v, const Grid&, const ChannelPlan& p) { for (auto& x : v) x += p.values[0]; }
inline void channel_apply(const Contrast& t, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    auto [lo, hi] = std::minmax_element(v.begin(), v.end()); double low = *lo, high = *hi, m = mean(v);
    for (auto& x : v) { x = (x - m) * p.values[0] + m; if (t.preserve_range) x = std::clamp(x, low, high); }
}
inline void channel_apply(const Gamma&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    bool invert = p.values[1], retain = p.values[2];
    if (invert) for (auto& x : v) x = -x;
    double m = mean(v), std = deviation(v, m);
    auto [lo, hi] = std::minmax_element(v.begin(), v.end()); double low = *lo, range = *hi - low;
    for (auto& x : v) x = std::pow((x - low) / std::max(range, 1e-7), p.values[0]) * range + low;
    if (retain) {
        double after = mean(v), scale = std / std::max(deviation(v, after), 1e-7);
        for (auto& x : v) x = (x - after) * scale + m;
    }
    if (invert) for (auto& x : v) x = -x;
}
inline void channel_apply(const Invert&, std::vector<double>& v, const Grid&, const ChannelPlan&) { double m = mean(v); for (auto& x : v) x = 2 * m - x; }
inline void channel_apply(const GaussianNoise&, std::vector<double>& v, const Grid&, const ChannelPlan& p) { Random r{p.seed}; for (auto& x : v) x += p.values[0] * r.normal(); }
inline void channel_apply(const RicianNoise&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    Random r{p.seed}; for (auto& x : v) { double re = x + p.values[0] * r.normal(), im = p.values[0] * r.normal(); x = std::sqrt(re * re + im * im); }
}
inline void channel_apply(const ClipOutliers&, std::vector<double>& v, const Grid&, const ChannelPlan& p) {
    double std = deviation(v, mean(v)); auto sorted = v; std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double q) { double pos = q / 100 * (v.size() - 1); auto lo = std::size_t(pos), hi = std::min(lo + 1, v.size() - 1); return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo); };
    double low = percentile(p.values[0]), high = percentile(p.values[1]);
    for (auto& x : v) x = std::clamp(x, low, high);
    if (p.values[2]) { double m = mean(v), after = deviation(v, m); if (after > 1e-8) for (auto& x : v) x = (x - m) / after * std + m; }
}
inline void channel_apply(const GaussianBlur& t, std::vector<double>& v, const Grid& g, const ChannelPlan& p) { gaussian_blur(v, g, p.values, t.truncate, t.mirror_boundary, true); }
inline void channel_apply(const MedianFilter&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto k = std::size_t(p.values[0]); Grid kernel(Shape(g.shape.size(), k)); auto src = v;
    std::vector<double> values(kernel.size);
    for (std::size_t i = 0; i < g.size; ++i) {
        for (std::size_t j = 0; j < kernel.size; ++j) {
            std::size_t at = 0;
            for (std::size_t a = 0; a < g.shape.size(); ++a) at += reflect(std::int64_t(g.coord(i, a) + kernel.coord(j, a)) - std::int64_t(k / 2), g.shape[a]) * g.stride[a];
            values[j] = src[at];
        }
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end()); v[i] = values[values.size() / 2];
    }
}
inline void channel_apply(const Sharpen&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto src = v; auto [lo, hi] = std::minmax_element(src.begin(), src.end());
    for (std::size_t i = 0; i < g.size; ++i) {
        double laplace = 2 * g.shape.size() * src[i];
        for (std::size_t a = 0; a < g.shape.size(); ++a) { auto x = g.coord(i, a); laplace -= src[x ? i - g.stride[a] : i]; laplace -= src[x + 1 < g.shape[a] ? i + g.stride[a] : i]; }
        v[i] += p.values[0] * laplace; if (p.values[1]) v[i] = std::clamp(v[i], *lo, *hi);
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
inline void channel_apply(const BlankRectangle& t, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto dim = g.shape.size(), stride = 2 * dim + 1;
    for (std::size_t b = 0; b < p.values.size(); b += stride) {
        std::vector<std::size_t> indices; std::vector<double> values;
        for (std::size_t i = 0; i < g.size; ++i) {
            bool inside = true; for (std::size_t a = 0; a < dim; ++a) { auto x = g.coord(i, a); inside &= x >= p.values[b + 2 * a] && x < p.values[b + 2 * a + 1]; }
            if (inside) { indices.push_back(i); values.push_back(v[i]); }
        }
        if (indices.empty()) continue;
        double value = p.values[b + 2 * dim];
        if (t.statistic == "mean") value = mean(values);
        else if (t.statistic == "median") { auto mid = (values.size() - 1) / 2; std::nth_element(values.begin(), values.begin() + mid, values.end()); value = values[mid]; }
        for (auto i : indices) v[i] = value;
    }
}
inline void channel_apply(const SliceIllumination&, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    for (std::size_t i = 0; i < g.size; ++i) v[i] *= p.field[g.coord(i, std::size_t(p.values[0]))];
}
inline void channel_apply(const Smear& t, std::vector<double>& v, const Grid& g, const ChannelPlan& p) {
    auto axis = std::size_t(p.values[0]);
    auto plane_shape = g.shape; plane_shape.erase(plane_shape.begin() + axis); Grid plane(plane_shape);
    // Deliberately recursive: previous slices have already been smeared, as in Vesuvius.
    for (std::size_t slice = t.previous; slice < g.shape[axis]; ++slice)
        for (std::size_t pixel = 0; pixel < plane.size; ++pixel) {
            std::size_t i = slice * g.stride[axis], pa = 0;
            for (std::size_t a = 0; a < g.shape.size(); ++a) if (a != axis) i += plane.coord(pixel, pa++) * g.stride[a];
            double sum = 0;
            for (int j = 1; j <= t.previous; ++j) {
                std::size_t at = (slice - j) * g.stride[axis], shift_axis = 0;
                for (std::size_t a = 0; a < g.shape.size(); ++a) if (a != axis) {
                    auto n = std::int64_t(g.shape[a]); auto x = (std::int64_t(g.coord(i, a)) - t.shift[shift_axis++]) % n;
                    at += std::size_t((x + n) % n) * g.stride[a];
                }
                sum += v[at];
            }
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
    apply_channels(s, p, [&](auto& v, const Grid& g, const ChannelPlan& c) { channel_apply(t, v, g, c); });
}
} // namespace vcr::augment
