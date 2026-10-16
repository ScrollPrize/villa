#pragma once
#include "labels.hpp"

namespace vcr::augment {
// Specialized tracing operations share the same native geometry/target machinery.
struct StripGeometry {
    Scalar shift_x = 0., shift_y = 0., degrees = 0., scale = 1., shear_x = 0., shear_y = 0.;
    Scalar smooth_offset = 0., stride = 16.; double p_flip_x = 0, p_flip_y = 0, p = 1; Shape shape;
};
inline double cubic(const std::vector<double>& values, double position) {
    auto i = std::int64_t(std::floor(position)); double t = position - i;
    auto get = [&](std::int64_t at) { return values[std::size_t(std::clamp<std::int64_t>(at, 0, std::int64_t(values.size()) - 1))]; };
    double v0 = get(i - 1), v1 = get(i), v2 = get(i + 1), v3 = get(i + 2);
    return .5 * (2 * v1 + (-v0 + v2) * t + (2 * v0 - 5 * v1 + 4 * v2 - v3) * t * t + (-v0 + 3 * v1 - 3 * v2 + v3) * t * t * t);
}
inline Parameters sample(const StripGeometry& t, const Sample& s, Random& r) {
    auto source = sample_shape(s); require(source.size() == 2, "StripGeometry requires a 2D grid");
    auto p = geometry(t.shape.empty() ? source : t.shape); require(p.shape.size() == 2 && product(p.shape), "invalid strip output shape");
    double scale = draw(t.scale, r), angle = draw(t.degrees, r) * 0.017453292519943295;
    double sx = draw(t.shear_x, r), sy = draw(t.shear_y, r), dx = draw(t.shift_x, r), dy = draw(t.shift_y, r);
    require(scale > 0 && std::abs(1 - sx * sy) > 1e-6, "strip geometry needs positive scale and invertible shear");
    Matrix flip_scale{(r.chance(t.p_flip_y) ? -1. : 1.) / scale, 0, 0, (r.chance(t.p_flip_x) ? -1. : 1.) / scale};
    Matrix shear{1, sy, sx, 1}, rotation{std::cos(angle), std::sin(angle), -std::sin(angle), std::cos(angle)};
    p.matrix = multiply(rotation, multiply(shear, flip_scale, 2), 2);
    Point center{double(p.shape[0] - 1) / 2 + dy, double(p.shape[1] - 1) / 2 + dx, 0}; center = matvec(p.matrix, center, 2);
    p.offset = {double(source[0] - 1) / 2 - center[0], double(source[1] - 1) / 2 - center[1]};
    double amplitude = draw(t.smooth_offset, r), stride = draw(t.stride, r); require(stride >= 1, "strip control stride must be at least one voxel");
    if (amplitude != 0) {
        auto count = std::max<std::size_t>(2, std::size_t(std::ceil(double(std::max<std::size_t>(1, source[1] - 1)) / stride)) + 3);
        std::vector<double> controls(count); double seed = double(r.next() & 0x7fffffff);
        for (std::size_t i = 0; i < count; ++i) {
            double h = std::fmod(double(i) * 1103515245. + seed * 12345. + 67890., 2147483647.);
            h = std::fmod(h * 1103515245. + 12345., 2147483647.); controls[i] = (h / 2147483647. * 2 - 1) * amplitude;
        }
        Grid g(p.shape); std::vector<double> offsets(g.size * 2);
        for (std::size_t i = 0; i < g.size; ++i) {
            auto q = backward(p, Point{double(g.coord(i, 0)), double(g.coord(i, 1)), 0});
            offsets[i * 2] = cubic(controls, q[1] / stride + 1);
        }
        p.field = std::move(offsets);
    }
    return p;
}
inline void apply(const StripGeometry&, Sample& s, const Parameters& p) { apply_geometry(s, p); }

struct SurfacePerturb {
    std::string on = "surface"; Scalar blobs = Uniform{1, 3}, sigma_fraction = Uniform{.04, .10}, amplitude = Uniform{.25, 1.25};
    std::size_t points_affected = 10; double radius_sigma = 2.5, max_displacement = 1.5, p = .35;
};
inline Parameters sample(const SurfacePerturb& t, const Sample& s, Random& r) {
    const auto& a = s.at(t.on).array;
    require(a.shape.size() == 3 && a.shape[0] == 3, "SurfacePerturb requires a (3, H, W) coordinate field");
    require(t.radius_sigma >= .5 && std::isfinite(t.radius_sigma) && t.max_displacement >= 0 && std::isfinite(t.max_displacement), "invalid surface perturbation limits");
    auto h = a.shape[1], w = a.shape[2], n = h * w; Parameters p; ChannelPlan plan; plan.key = t.on; plan.field.resize(3 * n);
    if (h < 2 || w < 2 || t.max_displacement == 0) { p.channels.push_back(std::move(plan)); return p; }
    std::vector<Point> normals(n); std::vector<std::size_t> valid;
    for (std::size_t y = 0; y < h; ++y) for (std::size_t x = 0; x < w; ++x) {
        auto i = y * w + x; Point row{}, col{};
        for (std::size_t c = 0; c < 3; ++c) {
            row[c] = a.get(c * n + std::min(y + 1, h - 1) * w + x) - a.get(c * n + (y ? y - 1 : y) * w + x);
            col[c] = a.get(c * n + y * w + std::min(x + 1, w - 1)) - a.get(c * n + y * w + (x ? x - 1 : x));
        }
        normals[i] = {col[1] * row[2] - col[2] * row[1], col[2] * row[0] - col[0] * row[2], col[0] * row[1] - col[1] * row[0]};
        double norm = 0; for (auto v : normals[i]) norm += v * v; norm = std::sqrt(norm);
        if (norm > 1e-6) { for (auto& v : normals[i]) v /= norm; valid.push_back(i); } else normals[i] = {};
    }
    if (valid.empty()) { p.channels.push_back(std::move(plan)); return p; }
    std::vector<double> displacement(n), distance(n);
    for (int blob = draw_integer(t.blobs, r); blob > 0; --blob) {
        auto center = valid[r.index(valid.size())]; double y = double(center / w), x = double(center % w);
        for (std::size_t i = 0; i < n; ++i) distance[i] = std::pow(double(i / w) - y, 2) + std::pow(double(i % w) - x, 2);
        double sigma, radius2;
        if (t.points_affected) { auto sorted = distance; auto k = std::min(t.points_affected, n) - 1; std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end()); radius2 = sorted[k]; sigma = std::max(.3, std::sqrt(std::max(radius2, 1e-6)) / t.radius_sigma); }
        else { sigma = std::max(.3, draw(t.sigma_fraction, r) * std::min(h, w)); radius2 = std::pow(t.radius_sigma * sigma, 2); }
        double amplitude = draw(t.amplitude, r) * (r.chance(.5) ? 1 : -1);
        for (std::size_t i = 0; i < n; ++i) if (distance[i] <= radius2) displacement[i] += amplitude * std::exp(-.5 * distance[i] / std::max(sigma * sigma, 1e-6));
    }
    for (std::size_t i = 0; i < n; ++i) for (std::size_t c = 0; c < 3; ++c) plan.field[c * n + i] = normals[i][c] * std::clamp(displacement[i], -t.max_displacement, t.max_displacement);
    p.channels.push_back(std::move(plan)); return p;
}
inline void apply(const SurfacePerturb&, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) { auto& a = s.at(c.key).array; require(c.field.size() == a.size(), "surface replay shape mismatch"); for (std::size_t i = 0; i < a.size(); ++i) a.set(i, a.get(i) + c.field[i]); }
}

// Crop planning happens before reading. This helper translates a volume-space
// box while retaining a requested fraction of already-filtered supervised points.
inline std::vector<std::int64_t> translate_crop_box(std::vector<std::int64_t> box, const std::vector<double>& points,
    std::uint64_t seed, double p = .3, int minimum = 10, int maximum = 40, double keep = 1. / 3, int attempts = 24) {
    require(box.size() == 6 && points.size() % 3 == 0, "expected ZYX start/stop box and (N, 3) points");
    require(minimum >= 0 && maximum >= minimum && maximum <= 100000 && attempts >= 0 && attempts <= 100000, "invalid crop translation limits"); probability(keep);
    for (int a = 0; a < 3; ++a) require(box[a] < box[a + 3], "box starts must precede stops");
    for (auto v : points) require(std::isfinite(v), "crop points must be finite");
    Random r{seed}; if (!r.chance(p) || points.empty()) return box;
    auto required = std::max<std::size_t>(1, std::size_t(std::ceil(points.size() / 3 * keep)));
    for (int attempt = 0; attempt < attempts; ++attempt) {
        auto proposed = box; int first = r.chance(.5) ? 1 : 2, count = r.chance(.5) ? 2 : 1;
        for (int j = 0; j < count; ++j) { int a = j == 0 ? first : 3 - first; auto shift = (minimum + std::int64_t(r.index(std::size_t(maximum - minimum + 1)))) * (r.chance(.5) ? 1 : -1); proposed[a] += shift; proposed[a + 3] += shift; }
        std::size_t within = 0;
        for (std::size_t i = 0; i < points.size(); i += 3) { bool inside = true; for (int a = 0; a < 3; ++a) inside &= points[i + a] >= proposed[a] && points[i + a] < proposed[a + 3]; within += inside; }
        if (within >= required) return proposed;
    }
    return box;
}
} // namespace vcr::augment
