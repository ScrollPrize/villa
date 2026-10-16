#pragma once
#include "intensity.hpp"

namespace vcr::augment {
using Matrix = std::vector<double>;
inline Matrix identity(std::size_t d) { Matrix m(d * d); for (std::size_t i = 0; i < d; ++i) m[i * d + i] = 1; return m; }
inline Matrix multiply(const Matrix& a, const Matrix& b, std::size_t d) {
    Matrix out(d * d); for (std::size_t i = 0; i < d; ++i) for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = 0; k < d; ++k) out[i * d + j] += a[i * d + k] * b[k * d + j];
    return out;
}
inline Matrix inverse(Matrix a, std::size_t d) {
    auto out = identity(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::size_t pivot = j; for (std::size_t i = j + 1; i < d; ++i) if (std::abs(a[i * d + j]) > std::abs(a[pivot * d + j])) pivot = i;
        require(std::isfinite(a[pivot * d + j]) && std::abs(a[pivot * d + j]) > 1e-12, "singular spatial transform");
        for (std::size_t k = 0; k < d; ++k) { std::swap(a[j * d + k], a[pivot * d + k]); std::swap(out[j * d + k], out[pivot * d + k]); }
        double scale = a[j * d + j]; for (std::size_t k = 0; k < d; ++k) { a[j * d + k] /= scale; out[j * d + k] /= scale; }
        for (std::size_t i = 0; i < d; ++i) if (i != j) {
            double factor = a[i * d + j]; for (std::size_t k = 0; k < d; ++k) { a[i * d + k] -= factor * a[j * d + k]; out[i * d + k] -= factor * out[j * d + k]; }
        }
    }
    return out;
}
using Point = std::array<double, 3>;
inline Point matvec(const Matrix& m, Point v, std::size_t d) {
    Point out{}; for (std::size_t i = 0; i < d; ++i) for (std::size_t j = 0; j < d; ++j) out[i] += m[i * d + j] * v[j]; return out;
}
template<class F> double linear_value(const Grid& g, Point q, bool border, double fill, F&& get) {
    std::int64_t lo[3]; double fraction[3]; auto d = g.shape.size();
    for (std::size_t a = 0; a < d; ++a) {
        require(std::isfinite(q[a]), "nonfinite sampling coordinate");
        if (border) q[a] = std::clamp(q[a], 0., double(g.shape[a] - 1));
        if (q[a] < -1 || q[a] > double(g.shape[a])) return fill;
        lo[a] = std::int64_t(std::floor(q[a])); fraction[a] = q[a] - lo[a];
    }
    double result = 0;
    for (std::size_t corner = 0; corner < (std::size_t(1) << d); ++corner) {
        bool valid = true; std::size_t at = 0; double weight = 1;
        for (std::size_t a = 0; a < d; ++a) {
            auto bit = (corner >> a) & 1; auto x = lo[a] + std::int64_t(bit);
            weight *= bit ? fraction[a] : 1 - fraction[a];
            if (x < 0 || x >= std::int64_t(g.shape[a])) valid = false;
            else at += std::size_t(x) * g.stride[a];
        }
        if (weight) result += weight * (valid ? get(at) : fill);
    }
    return result;
}
inline Point backward(const Parameters& p, Point out) {
    auto d = p.shape.size(); auto q = matvec(p.matrix, out, d);
    for (std::size_t a = 0; a < d; ++a) q[a] += p.offset[a];
    if (!p.field.empty()) {
        Grid g(p.shape);
        for (std::size_t a = 0; a < d; ++a) q[a] += linear_value(g, out, true, 0, [&](auto i) { return p.field[i * d + a]; });
    }
    return q;
}
inline Matrix jacobian(const Parameters& p, Point out) {
    auto d = p.shape.size(); auto j = p.matrix;
    if (!p.field.empty()) {
        for (std::size_t a = 0; a < d; ++a) {
            Point lo = out, hi = out; lo[a] -= .5; hi[a] += .5;
            auto v0 = backward(p, lo), v1 = backward(p, hi);
            for (std::size_t b = 0; b < d; ++b) j[b * d + a] = v1[b] - v0[b];
        }
    }
    return j;
}
inline Point forward(const Parameters& p, Point source) {
    auto d = p.shape.size(); Point offset = source;
    for (std::size_t a = 0; a < d; ++a) offset[a] -= p.offset[a];
    Point out = matvec(inverse(p.matrix, d), offset, d);
    if (p.field.empty()) return out;
    for (int iteration = 0; iteration < 50; ++iteration) {
        auto q = backward(p, out); double error = 0;
        for (std::size_t a = 0; a < d; ++a) { q[a] -= source[a]; error = std::max(error, std::abs(q[a])); }
        if (error < 1e-7) return out;
        auto step = matvec(inverse(jacobian(p, out), d), q, d);
        for (std::size_t a = 0; a < d; ++a) out[a] -= step[a];
    }
    throw std::invalid_argument("deformation inverse failed to converge for a point; reduce deformation strength");
}
inline Array resample(const Array& src, const Parameters& p, bool nearest, double fill, bool validity = false) {
    Grid input(src.spatial()), output(p.shape); auto d = p.shape.size();
    Shape shape{src.shape[0]}; shape.insert(shape.end(), p.shape.begin(), p.shape.end()); Array out(shape, src.kind, src.bytes);
    for (std::size_t i = 0; i < output.size; ++i) {
        Point pos{}; for (std::size_t a = 0; a < d; ++a) pos[a] = double(output.coord(i, a));
        auto q = backward(p, pos);
        if (nearest) {
            bool valid = true; std::size_t at = 0;
            for (std::size_t a = 0; a < d; ++a) {
                require(std::isfinite(q[a]), "nonfinite sampling coordinate");
                if (p.border) q[a] = std::clamp(q[a], 0., double(input.shape[a] - 1));
                if (q[a] < -1 || q[a] > double(input.shape[a])) { valid = false; continue; }
                auto lo = std::int64_t(std::floor(q[a])); auto fraction = q[a] - lo;
                auto x = p.nearest_mode == 2 ? lo : lo + (fraction > .5 || (fraction == .5 && (p.nearest_mode == 1 || lo % 2 != 0)));
                if (x < 0 || x >= std::int64_t(input.shape[a])) valid = false; else at += std::size_t(x) * input.stride[a];
            }
            for (std::size_t c = 0; c < src.shape[0]; ++c) {
                if (valid) out.copy_element(c * output.size + i, src, c * input.size + at);
                else out.set(c * output.size + i, fill);
            }
        } else {
            for (std::size_t c = 0; c < src.shape[0]; ++c) {
                double value = linear_value(input, q, validity ? false : p.border, fill, [&](auto at) { return src.get(c * input.size + at); });
                // Conservative support: every interpolation contributor must be valid.
                out.set(c * output.size + i, validity ? double(value >= 1.) : value);
            }
        }
    }
    return out;
}
inline void apply_geometry(Sample& sample, const Parameters& p) {
    auto d = p.shape.size(); auto canonical = axes(d); Grid output(p.shape);
    for (auto& [name, f] : sample) {
        auto& a = f.array; const auto& t = f.target; auto role = t.role;
        if (sparse(role)) {
            require(role != Role::Vectors || p.field.empty(), "nonlinear transforms of sparse vectors require locations; use a vector field");
            auto out = a; std::size_t kept = 0;
            for (std::size_t i = 0; i < a.shape[0]; ++i) {
                Point point{}; for (std::size_t j = 0; j < d; ++j) point[axis(t.components[j], d)] = a.get(i * d + j);
                point = role == Role::Points ? forward(p, point) : matvec(inverse(p.matrix, d), point, d);
                bool inside = true; for (std::size_t j = 0; j < d; ++j) inside &= point[j] >= 0 && point[j] <= double(p.shape[j] - 1);
                if (role == Role::Points && !inside) { require(t.outside != "error", name + ": point outside output"); if (t.outside == "drop") continue; }
                if (role == Role::Vectors && t.normalize) { double norm = 0; for (std::size_t j = 0; j < d; ++j) norm += point[j] * point[j]; if (norm > 0) for (auto& v : point) v /= std::sqrt(norm); }
                for (std::size_t j = 0; j < d; ++j) out.set(kept * d + j, point[axis(t.components[j], d)]);
                ++kept;
            }
            out.shape[0] = kept; a = std::move(out); continue;
        }
        auto out = resample(a, p, role == Role::Label || (p.exact && role != Role::Valid), t.fill, role == Role::Valid);
        if (directional(role) || (role == Role::CoordinateField && !t.external)) {
            for (std::size_t i = 0; i < output.size; ++i) {
                Point v{}, pos{}; for (std::size_t j = 0; j < d; ++j) { v[axis(t.components[j], d)] = out.get(j * output.size + i); pos[j] = double(output.coord(i, j)); }
                if (role == Role::CoordinateField) v = forward(p, v);
                else {
                    auto j = jacobian(p, pos);
                    if (role == Role::NormalField) { Matrix transpose(d * d); for (std::size_t x = 0; x < d; ++x) for (std::size_t y = 0; y < d; ++y) transpose[x * d + y] = j[y * d + x]; v = matvec(transpose, v, d); }
                    else v = matvec(inverse(j, d), v, d);
                    if (role == Role::NormalField || t.normalize) { double norm = 0; for (std::size_t j = 0; j < d; ++j) norm += v[j] * v[j]; if (norm > 0) for (auto& x : v) x /= std::sqrt(norm); }
                }
                for (std::size_t j = 0; j < d; ++j) out.set(j * output.size + i, v[axis(t.components[j], d)]);
            }
        }
        a = std::move(out);
    }
    restore_valid(sample);
}
inline Parameters geometry(Shape shape) { Parameters p; p.matrix = identity(shape.size()); p.offset.assign(shape.size(), 0); p.shape = std::move(shape); return p; }
struct Flip { std::string axes = "zyx"; double p_axis = .5, p = 1; };
struct Rotate90 { std::vector<std::string> planes; Choice turns{{1, 2, 3}}; bool resize = false; double p = 1; };
struct Transpose { std::string order; double p = 1; };
struct CropPad { Shape shape; bool random = false; double p = 1; };
struct Affine {
    Scalar degrees = 0., scale = 1., translate = 0., shear = 0.;
    std::string rotation_axes = "zyx"; bool shared_scale = true, border = false; Shape shape; double p = 1;
};
struct Elastic { Scalar sigma = Uniform{5, 10}, magnitude = Uniform{1, 5}; bool border = false; double p = 1; };
struct SheetCompression { Scalar strength = Uniform{.1, .3}, intensity_sigma = 1., smoothing = Uniform{3, 8}; std::string axes = "zyx", image = "image"; double threshold = -1, p = 1; };

inline Parameters sample(const Flip& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s)); auto d = p.shape.size();
    for (char c : t.axes) { auto a = std::size_t(axis(c, d)); if (r.chance(t.p_axis)) { p.matrix[a * d + a] *= -1; p.offset[a] = double(p.shape[a] - 1) - p.offset[a]; } }
    p.exact = true; return p;
}
inline Parameters sample(const Rotate90& t, const Sample& s, Random& r) {
    auto shape = sample_shape(s); auto d = shape.size(); auto names = axes(d); auto planes = t.planes;
    if (planes.empty()) for (std::size_t a = 0; a < d; ++a) for (std::size_t b = a + 1; b < d; ++b)
        if (t.resize || shape[a] == shape[b]) planes.push_back(std::string{names[a], names[b]});
    if (planes.empty()) return geometry(shape);
    auto plane = planes[r.index(planes.size())]; require(plane.size() == 2 && plane[0] != plane[1], "rotation plane needs two distinct axes");
    auto a = std::size_t(axis(plane[0], d)), b = std::size_t(axis(plane[1], d));
    auto turns = draw(Scalar(t.turns), r); require(std::floor(turns) == turns && std::abs(turns) < 100000, "turns must be integers");
    int k = (int(turns) % 4 + 4) % 4; require(t.resize || k % 2 == 0 || shape[a] == shape[b], "90 degree rotation would change shape; use resize=True");
    Matrix forward_m = identity(d); Point translation{};
    for (int n = 0; n < k; ++n) {
        Matrix rotation = identity(d); rotation[a * d + a] = rotation[b * d + b] = 0; rotation[a * d + b] = -1; rotation[b * d + a] = 1;
        translation = matvec(rotation, translation, d); translation[a] += double(shape[b] - 1);
        forward_m = multiply(rotation, forward_m, d); std::swap(shape[a], shape[b]);
    }
    auto p = geometry(shape); p.matrix = inverse(forward_m, d); auto offset = matvec(p.matrix, translation, d);
    for (std::size_t a = 0; a < d; ++a) p.offset[a] = -offset[a]; p.exact = true; return p;
}
inline Parameters sample(const Transpose& t, const Sample& s, Random& r) {
    auto shape = sample_shape(s); auto d = shape.size(); auto order = t.order;
    if (order.empty()) {
        order = axes(d); for (std::size_t i = d; i > 1; --i) { auto j = r.index(i); if (shape[i - 1] == shape[j]) std::swap(order[i - 1], order[j]); }
    }
    auto sorted = order; std::sort(sorted.begin(), sorted.end()); auto expected = axes(d); std::sort(expected.begin(), expected.end()); require(sorted == expected, "transpose order must be a permutation of spatial axes");
    auto p = geometry(shape); p.matrix.assign(d * d, 0);
    for (std::size_t a = 0; a < d; ++a) { auto b = std::size_t(axis(order[a], d)); p.matrix[b * d + a] = 1; p.shape[a] = shape[b]; }
    p.exact = true; return p;
}
inline Parameters sample(const CropPad& t, const Sample& s, Random& r) {
    auto input = sample_shape(s); require(t.shape.size() == input.size() && product(t.shape), "invalid crop shape"); auto p = geometry(t.shape);
    for (std::size_t a = 0; a < input.size(); ++a) {
        auto delta = std::int64_t(input[a]) - std::int64_t(t.shape[a]);
        p.offset[a] = t.random && delta > 0 ? double(r.index(std::size_t(delta + 1))) : std::floor(double(delta) / 2);
    }
    p.exact = true; return p;
}
inline Parameters sample(const Affine& t, const Sample& s, Random& r) {
    auto input = sample_shape(s); auto d = input.size(); auto p = geometry(t.shape.empty() ? input : t.shape);
    require(p.shape.size() == d && product(p.shape), "invalid affine output shape");
    Matrix forward_m = identity(d); double shared_scale = draw(t.scale, r);
    for (std::size_t a = 0; a < d; ++a) { double scale = t.shared_scale ? shared_scale : draw(t.scale, r); require(scale > 0, "scale must be positive"); forward_m[a * d + a] = scale; }
    for (char c : t.rotation_axes) {
        if (d == 2 && c != 'z') continue;
        auto fixed = d == 3 ? axis(c, d) : -1; std::vector<std::size_t> plane;
        for (std::size_t a = 0; a < d; ++a) if (int(a) != fixed) plane.push_back(a);
        auto a = plane[0], b = plane[1]; double angle = draw(t.degrees, r) * 0.017453292519943295;
        auto rot = identity(d); rot[a * d + a] = rot[b * d + b] = std::cos(angle); rot[a * d + b] = -std::sin(angle); rot[b * d + a] = std::sin(angle);
        forward_m = multiply(rot, forward_m, d);
    }
    auto shear = identity(d); for (std::size_t a = 0; a < d; ++a) for (std::size_t b = a + 1; b < d; ++b) shear[a * d + b] = draw(t.shear, r);
    forward_m = multiply(shear, forward_m, d); p.matrix = inverse(forward_m, d);
    Point center{}; for (std::size_t a = 0; a < d; ++a) center[a] = double(p.shape[a] - 1) / 2 + draw(t.translate, r);
    center = matvec(p.matrix, center, d);
    for (std::size_t a = 0; a < d; ++a) p.offset[a] = double(input[a] - 1) / 2 - center[a];
    p.border = t.border; return p;
}
inline Parameters sample(const Elastic& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s)); Grid g(p.shape); auto d = p.shape.size(); p.field.resize(g.size * d);
    for (std::size_t a = 0; a < d; ++a) {
        double sigma = draw(t.sigma, r), magnitude = draw(t.magnitude, r); require(magnitude >= 0, "elastic magnitude must be nonnegative");
        std::vector<double> noise(g.size); for (auto& v : noise) v = r.normal(); gaussian_blur(noise, g, std::vector<double>(d, sigma), 6);
        double maxabs = 0; for (auto v : noise) maxabs = std::max(maxabs, std::abs(v));
        double m = mean(noise); for (std::size_t i = 0; i < g.size; ++i) p.field[i * d + a] = maxabs ? (noise[i] - m) * magnitude / maxabs : 0;
    }
    p.border = t.border; return p;
}
inline Parameters sample(const SheetCompression& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s)); Grid g(p.shape); auto d = p.shape.size();
    require(s.count(t.image) && s.at(t.image).target.role == Role::Image, "sheet compression needs its reference image");
    auto chosen = std::size_t(axis(t.axes[r.index(t.axes.size())], d));
    double strength = draw(t.strength, r), sigma = draw(t.intensity_sigma, r), smooth = draw(t.smoothing, r);
    require(strength >= 0 && strength < 1, "compression strength must be in [0, 1)");
    auto values = s.at(t.image).array.channel(0); if (sigma > .5) gaussian_blur(values, g, std::vector<double>(d, sigma), 6);
    auto [lo, hi] = std::minmax_element(values.begin(), values.end()); double low = *lo, span = *hi - low;
    if (span <= 1e-8) return p;
    for (std::size_t i = 0; i < g.size; ++i) { double norm = (values[i] - low) / span; values[i] = (t.threshold >= 0 ? double(norm < t.threshold) : 1 - norm) * strength; if (g.coord(i, chosen)) values[i] += values[i - g.stride[chosen]]; }
    if (smooth > .5) { std::vector<double> sigmas(d, smooth); sigmas[chosen] = 0; gaussian_blur(values, g, sigmas, 6); }
    p.field.assign(g.size * d, 0);
    for (std::size_t i = 0; i < g.size; ++i) p.field[i * d + chosen] = -values[i] * double(g.shape[chosen] - 1) / g.shape[chosen];
    p.border = true; return p;
}
template<class T> concept Geometric = std::is_same_v<T, Flip> || std::is_same_v<T, Rotate90> || std::is_same_v<T, Transpose> || std::is_same_v<T, CropPad> || std::is_same_v<T, Affine> || std::is_same_v<T, Elastic> || std::is_same_v<T, SheetCompression>;
template<Geometric T> void apply(const T&, Sample& s, const Parameters& p) { apply_geometry(s, p); }

// Resolution simulation deliberately has its own image/target routing, distinct
// from a geometric deformation (point coordinates do not move).
struct LowResolution { Scalar scale = Uniform{.5, 1}; std::string axes = "zyx"; bool shared_axes = true; ChannelOptions options; };
struct ThickSlice { Scalar scale = Uniform{.25, .6}; std::string axes = "zyx"; bool labels = true; double p = 1; };
inline Array resize_array(const Array& a, const Shape& shape, bool nearest, bool legacy_nearest = false) {
    auto p = geometry(shape); auto d = shape.size(); p.border = true;
    for (std::size_t i = 0; i < d; ++i) {
        double ratio = double(a.shape[i + 1]) / shape[i]; p.matrix[i * d + i] = ratio;
        p.offset[i] = legacy_nearest ? 0 : .5 * ratio - .5;
    }
    p.nearest_mode = legacy_nearest ? 2 : 1;
    return resample(a, p, nearest, 0);
}
inline Parameters sample(const LowResolution& t, const Sample& s, Random& r) {
    return sample_channels(s, t.options, r, [&](const Array& a, Random& rng) {
        ChannelPlan p; auto d = a.shape.size() - 1; auto names = axes(d); double shared = draw(t.scale, rng);
        for (std::size_t j = 0; j < d; ++j) {
            double scale = t.shared_axes ? shared : draw(t.scale, rng); require(scale > 0 && scale <= 1, "resolution scale must be in (0, 1]");
            p.values.push_back(t.axes.find(names[j]) == std::string::npos ? 1 : scale);
        }
        return p;
    });
}
inline void apply(const LowResolution&, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; Shape shape = a.spatial(); auto n = product(shape); Shape one_shape{1}; one_shape.insert(one_shape.end(), shape.begin(), shape.end());
        Array one(one_shape, a.kind, a.bytes); for (std::size_t i = 0; i < n; ++i) one.copy_element(i, a, c.channel * n + i);
        for (std::size_t j = 0; j < shape.size(); ++j) shape[j] = std::max<std::size_t>(1, std::size_t(std::nearbyint(shape[j] * c.values[j])));
        auto restored = resize_array(resize_array(one, shape, true), a.spatial(), false);
        for (std::size_t i = 0; i < n; ++i) a.copy_element(c.channel * n + i, restored, i);
    }
    restore_valid(s);
}
inline Parameters sample(const ThickSlice& t, const Sample& s, Random& r) {
    auto shape = sample_shape(s); auto a = std::size_t(axis(t.axes[r.index(t.axes.size())], shape.size()));
    double scale = draw(t.scale, r); require(scale > 0 && scale <= 1, "thick-slice scale must be in (0, 1]");
    shape[a] = std::max<std::size_t>(1, std::size_t(std::nearbyint(shape[a] * scale))); Parameters p; p.shape = shape; return p;
}
inline void apply(const ThickSlice& t, Sample& s, const Parameters& p) {
    for (auto& [name, f] : s) if (f.target.role == Role::Image || (t.labels && (f.target.role == Role::Label || f.target.role == Role::Scalar))) {
        auto shape = f.array.spatial(); f.array = resize_array(resize_array(f.array, p.shape, true, true), shape, true, true);
    }
    restore_valid(s);
}
} // namespace vcr::augment
