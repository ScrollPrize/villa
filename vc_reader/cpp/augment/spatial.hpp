#pragma once
#include "intensity.hpp"
#include "elastic.hpp"

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
inline std::int64_t nearest_index(double q, int mode) {
    auto lo = std::int64_t(std::floor(q)); auto fraction = q - lo;
    return mode == 2 ? lo : lo + (fraction > .5 || (fraction == .5 && (mode == 1 || lo % 2 != 0)));
}
template<std::size_t D = 0, bool Validity = false, class Real, class F> Real linear_value(const Grid& g, std::array<Real,3> q, bool border, double fill, F&& get) {
    std::int64_t lo[3]; Real fraction[3]; const auto d = D ? D : g.shape.size();
    for (std::size_t a = 0; a < d; ++a) {
        require(std::isfinite(q[a]), "nonfinite sampling coordinate");
        if (border) q[a] = std::clamp(q[a], Real(0), Real(g.shape[a] - 1));
        if (q[a] < -1 || q[a] > Real(g.shape[a])) return fill;
        lo[a] = std::int64_t(std::floor(q[a])); fraction[a] = q[a] - lo[a];
    }
    // Interior stencils need no per-corner bounds checks. Keep the corner and
    // weight multiplication order identical to the boundary path below.
    bool interior = true;
    std::size_t base = 0;
    for (std::size_t a = 0; a < d; ++a) {
        interior &= lo[a] >= 0 && lo[a] + 1 < std::int64_t(g.shape[a]);
        if (lo[a] >= 0) base += std::size_t(lo[a]) * g.stride[a];
    }
    Real result = 0;
    if (interior) {
        for (std::size_t corner = 0; corner < (std::size_t(1) << d); ++corner) {
            std::size_t at = base; Real weight = 1;
            for (std::size_t a = 0; a < d; ++a) {
                auto bit = (corner >> a) & 1;
                weight *= bit ? fraction[a] : 1 - fraction[a];
                if (bit) at += g.stride[a];
            }
            if (weight) {
                if constexpr (Validity) { if (get(at) < 1.) return 0.; }
                else result += weight * get(at);
            }
        }
        return Validity ? 1. : result;
    }
    for (std::size_t corner = 0; corner < (std::size_t(1) << d); ++corner) {
        bool valid = true; std::size_t at = 0; Real weight = 1;
        for (std::size_t a = 0; a < d; ++a) {
            auto bit = (corner >> a) & 1; auto x = lo[a] + std::int64_t(bit);
            weight *= bit ? fraction[a] : 1 - fraction[a];
            if (x < 0 || x >= std::int64_t(g.shape[a])) valid = false;
            else at += std::size_t(x) * g.stride[a];
        }
        if (weight) {
            if constexpr (Validity) { if ((valid ? Real(get(at)) : Real(fill)) < 1.) return 0.; }
            else result += weight * (valid ? Real(get(at)) : Real(fill));
        }
    }
    return Validity ? 1. : result;
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
// Exact geometry is a signed axis permutation plus an integer translation.
// Compute row addresses once, then copy contiguous runs or gather by stride.
inline bool resample_exact(const Array& src, Array& out, const Parameters& p, double fill, bool validity) {
    if (!p.exact || !p.field.empty() || p.border) return false;
    Grid input(src.spatial()), output(p.shape);
    const auto d = p.shape.size();
    std::array<std::vector<std::ptrdiff_t>, 3> offsets;
    constexpr auto invalid = std::numeric_limits<std::ptrdiff_t>::min();
    std::ptrdiff_t xstep = 0;
    for (std::size_t b = 0; b < d; ++b) {
        std::size_t a = 0;
        while (a < d && p.matrix[a * d + b] == 0) ++a;
        if (a == d || std::abs(p.matrix[a * d + b]) != 1 || std::floor(p.offset[a]) != p.offset[a]) return false;
        for (std::size_t j = 0; j < d; ++j)
            if (j != b && p.matrix[a * d + j] != 0) return false;
        auto sign = std::ptrdiff_t(p.matrix[a * d + b]);
        if (b + 1 == d) xstep = sign * std::ptrdiff_t(input.stride[a]);
        offsets[b].resize(p.shape[b]);
        for (std::size_t x = 0; x < p.shape[b]; ++x) {
            auto at = std::ptrdiff_t(p.offset[a]) + sign * std::ptrdiff_t(x);
            offsets[b][x] = at >= 0 && at < std::ptrdiff_t(input.shape[a]) ? at * std::ptrdiff_t(input.stride[a]) : invalid;
        }
    }
    const auto nx = p.shape.back(), ny = p.shape[d - 2], nz = d == 3 ? p.shape[0] : 1;
    const auto& xs = offsets[d - 1];
    std::size_t begin = 0, end = nx;
    while (begin < nx && xs[begin] == invalid) ++begin;
    while (end > begin && xs[end - 1] == invalid) --end;
    // Validate the fill only when padding is actually required, as in resample.
    bool padding = false;
    for (std::size_t a = 0; a < d; ++a)
        padding |= std::find(offsets[a].begin(), offsets[a].end(), invalid) != offsets[a].end();
    if (padding) out.set(0, validity ? double(fill >= 1.) : fill);
    src.typed([&](const auto* source) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
        auto* dest = static_cast<T*>(out.data());
        T pad = padding ? dest[0] : T{};
        for (std::size_t c = 0; c < src.shape[0]; ++c)
            for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y) {
                auto* row = dest + c * output.size + (z * ny + y) * nx;
                auto zo = d == 3 ? offsets[0][z] : 0, yo = offsets[d - 2][y];
                if (zo == invalid || yo == invalid || begin == end) { std::fill(row, row + nx, pad); continue; }
                std::fill(row, row + begin, pad); std::fill(row + end, row + nx, pad);
                const auto* from = source + c * input.size + zo + yo + xs[begin];
                if (xstep == 1) std::copy_n(from, end - begin, row + begin);
                else if (xstep == -1) std::reverse_copy(from - (end - begin - 1), from + 1, row + begin);
                else for (std::size_t x = begin; x < end; ++x) row[x] = from[std::ptrdiff_t(x - begin) * xstep];
            }
    });
    return true;
}
// Infer this from the dense field so saved replay plans need no new metadata.
// General deformations normally fail the check within their first voxel.
inline std::size_t displacement_axis(const Parameters& p) {
    const auto d = p.shape.size();
    if (p.field.empty() || !p.border || p.matrix != identity(d) ||
        std::any_of(p.offset.begin(), p.offset.end(), [](double v) { return v != 0; })) return d;
    std::size_t chosen = d;
    for (std::size_t i = 0; i < p.field.size(); i += d)
        for (std::size_t a = 0; a < d; ++a) if (p.field[i + a] != 0) {
            if (chosen != d && chosen != a) return d;
            chosen = a;
        }
    return chosen == d ? 0 : chosen;
}
template<class Real> inline void resample_axis_displacement(const Array& src, Array& out, const Parameters& p,
                                       std::size_t axis, bool nearest, double fill) {
    Grid grid(p.shape), line({p.shape[axis]});
    const auto d = p.shape.size(), extent = p.shape[axis], step = grid.stride[axis];
    src.typed([&](const auto* source) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
        auto* destination = static_cast<T*>(out.data());
        for (std::size_t block = 0; block < grid.size; block += extent * step)
            for (std::size_t x = 0; x < extent; ++x)
                for (std::size_t j = 0; j < step; ++j) {
                    const auto i = block + x * step + j, base = block + j;
                    std::array<Real,3> q{Real(x) + p.field[i * d + axis], 0, 0};
                    if (nearest) {
                        require(std::isfinite(q[0]), "nonfinite sampling coordinate");
                        auto at = base + std::size_t(nearest_index(std::clamp(q[0], Real(0), Real(extent - 1)), p.nearest_mode)) * step;
                        for (std::size_t c = 0; c < src.shape[0]; ++c)
                            destination[c * grid.size + i] = source[c * grid.size + at];
                    } else {
                        for (std::size_t c = 0; c < src.shape[0]; ++c) {
                            // The omitted axes have fraction zero, so their
                            // weights are exactly one. Keep low/high corner
                            // accumulation (including the initial +0) unchanged.
                            Real value = linear_value<1>(line, q, true, fill, [&](auto at) {
                                return Real(source[c * grid.size + base + at * step]);
                            });
                            if constexpr (std::is_floating_point_v<T>) destination[c * grid.size + i] = static_cast<T>(value);
                            else out.set(c * grid.size + i, value);
                        }
                    }
                }
    });
}
// Float32 has enough resolution for ordinary crop-local resampling. Use a
// conservative absolute error budget; large coordinates/cancelling affine terms
// need more mantissa bits.
inline bool geometry_float32(const Parameters& p, const Shape& input) {
    constexpr double budget = 1e-3, roundoff = 8 * std::numeric_limits<float>::epsilon();
    for (std::size_t a=0;a<p.shape.size();++a) {
        double bound=std::abs(p.offset[a]);
        for (std::size_t b=0;b<p.shape.size();++b) bound += std::abs(p.matrix[a*p.shape.size()+b])*double(p.shape[b]-1);
        // For in-bounds samples, |displacement| <= |affine| + input extent.
        // Coordinates far outside the input require no subvoxel precision.
        bound = 2*bound + double(input[a]) + 1;
        if (bound*roundoff > budget) return false;
    }
    return true;
}
template<std::size_t D, class Real> Array resample_impl(const Array& src, const Parameters& p, bool nearest, double fill, bool validity = false, std::size_t field_axis = D) {
    Grid input(src.spatial()), output(p.shape); constexpr auto d = D;
    Shape shape{src.shape[0]}; shape.insert(shape.end(), p.shape.begin(), p.shape.end()); auto out = Array::uninitialized(shape, src.kind, src.bytes);
    if (resample_exact(src, out, p, fill, validity)) return out;
    if (field_axis < d && !validity && input.shape == p.shape) {
        resample_axis_displacement<Real>(src, out, p, field_axis, nearest, fill);
        return out;
    }
    src.typed([&](const auto* source) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
        auto* destination = static_cast<T*>(out.data());
        const auto nz = d == 3 ? p.shape[0] : 1, ny = p.shape[d - 2], nx = p.shape.back();
        const bool displaced_identity=!p.field.empty() && p.matrix==identity(d);
        for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
            for (std::size_t x = 0; x < nx; ++x) {
            const auto i = (z * ny + y) * nx + x;
            std::array<Real,3> pos{};
            if constexpr (d == 3) pos[0] = Real(z);
            pos[d - 2] = Real(y); pos[d - 1] = Real(x);
            auto q = pos;
            if (!displaced_identity) {
                q = {};
                for (std::size_t a=0;a<d;++a) for(std::size_t b=0;b<d;++b) q[a] += Real(p.matrix[a*d+b])*pos[b];
            }
            for (std::size_t a = 0; a < d; ++a) {
                q[a] += Real(p.offset[a]);
                // The displacement is already sampled on this output grid.
                if (!p.field.empty()) q[a] += p.field[i * d + a];
            }
            if (nearest) {
                bool valid = true; std::size_t at = 0;
                for (std::size_t a = 0; a < d; ++a) {
                    require(std::isfinite(q[a]), "nonfinite sampling coordinate");
                    if (p.border) q[a] = std::clamp(q[a], Real(0), Real(input.shape[a] - 1));
                    if (q[a] < -1 || q[a] > Real(input.shape[a])) { valid = false; continue; }
                    auto x = nearest_index(q[a], p.nearest_mode);
                    if (x < 0 || x >= std::int64_t(input.shape[a])) valid = false; else at += std::size_t(x) * input.stride[a];
                }
                for (std::size_t c = 0; c < src.shape[0]; ++c) {
                    if (valid) destination[c * output.size + i] = source[c * input.size + at];
                    else out.set(c * output.size + i, fill);
                }
            } else {
                for (std::size_t c = 0; c < src.shape[0]; ++c) {
                    auto get = [&](auto at) { return Real(source[c * input.size + at]); };
                    // Check support directly: a rounded sum of all-one samples
                    // can be below 1 even though every contributor is valid.
                    Real value = validity ? linear_value<D, true>(input,q,false,fill,get)
                                            : linear_value<D>(input,q,p.border,fill,get);
                    if constexpr (std::is_floating_point_v<T>) destination[c * output.size + i] = static_cast<T>(value);
                    else out.set(c * output.size + i, value);
                }
            }
        }
    });
    return out;
}
inline Array resample(const Array& src, const Parameters& p, bool nearest, double fill, bool validity = false, std::size_t field_axis = 3) {
    if (!(src.kind == 'f' && src.bytes == 8) && geometry_float32(p, src.spatial()))
        return p.shape.size() == 2 ? resample_impl<2,float>(src, p, nearest, fill, validity, field_axis)
                                   : resample_impl<3,float>(src, p, nearest, fill, validity, field_axis);
    return p.shape.size() == 2 ? resample_impl<2,double>(src, p, nearest, fill, validity, field_axis)
                               : resample_impl<3,double>(src, p, nearest, fill, validity, field_axis);
}
template<class Resample>
inline void apply_geometry_with(Sample& sample, const Parameters& p, Resample&& resample) {
    auto d = p.shape.size(); auto canonical = axes(d); Grid output(p.shape);
    const auto field_axis = displacement_axis(p);
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
        auto out = resample(a, p, role == Role::Label || (p.exact && role != Role::Valid), t.fill, role == Role::Valid, field_axis);
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
inline void apply_geometry(Sample& sample, const Parameters& p) { apply_geometry_with(sample, p, resample); }
inline Parameters geometry(Shape shape) { Parameters p; p.matrix = identity(shape.size()); p.offset.assign(shape.size(), 0); p.shape = std::move(shape); return p; }
struct Flip { std::string axes; double p_axis = .5, p = 1; };
struct Rotate90 { std::vector<std::string> planes; Choice turns{{1, 2, 3}}; bool resize = false; double p = 1; };
struct Transpose { std::string order; double p = 1; };
struct CropPad { Shape shape; bool random = false; double p = 1; };
struct Affine {
    Scalar degrees = 0., scale = 1., translate = 0., shear = 0.;
    std::string rotation_axes = "zyx"; bool shared_scale = true, border = false; Shape shape; double p = 1;
};
struct Elastic { Scalar sigma = Uniform{5, 10}, magnitude = Uniform{1, 5}; bool border = false; double p = 1; };
struct SheetCompression { Scalar strength = Uniform{.1, .3}, intensity_sigma = 1., smoothing = Uniform{3, 8}; std::string axes, image = "image"; double threshold = -1, p = 1; };

inline Parameters sample(const Flip& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s)); auto d = p.shape.size();
    for (char c : t.axes.empty() ? axes(d) : t.axes) { auto a = std::size_t(axis(c, d)); if (r.chance(t.p_axis)) { p.matrix[a * d + a] *= -1; p.offset[a] = double(p.shape[a] - 1) - p.offset[a]; } }
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
        order = axes(d);
        std::map<std::size_t, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < d; ++i) groups[shape[i]].push_back(i);
        for (const auto& [size, group] : groups) for (std::size_t i = group.size(); i > 1; --i) std::swap(order[group[i - 1]], order[group[r.index(i)]]);
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
inline std::array<double,2> elastic_settings(const Elastic& t, Random& r) {
    const double sigma=draw(t.sigma,r), magnitude=draw(t.magnitude,r);
    require(sigma>=0 && sigma<=std::numeric_limits<double>::max()/2,"elastic sigma must be nonnegative and finite");
    require(magnitude>=0,"elastic magnitude must be nonnegative");
    return {sigma,magnitude};
}
inline Parameters sample(const Elastic& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s));
    const auto [sigma,magnitude] = elastic_settings(t,r);
    // Historical argument names retained: sigma is half the knot spacing;
    // magnitude is now RMS vector displacement, limited to prevent folds.
    if (magnitude==0) { p.exact=true; p.border=t.border; return p; }
    p.field=elastic_detail::bspline_field(p.shape,std::max(1.,2*sigma),r);
    elastic_detail::normalize(p.field,p.shape,magnitude,.8);
    p.border = t.border; return p;
}
struct SheetSettings { Parameters parameters; std::size_t chosen; double strength,sigma,smooth; };
inline SheetSettings sheet_parameters(const SheetCompression& t, const Sample& s, Random& r) {
    auto p = geometry(sample_shape(s)); Grid g(p.shape); auto d = p.shape.size();
    require(s.count(t.image) && s.at(t.image).target.role == Role::Image, "sheet compression needs its reference image");
    auto allowed = t.axes.empty() ? axes(d) : t.axes;
    auto chosen = std::size_t(axis(allowed[r.index(allowed.size())], d));
    double strength = draw(t.strength, r), sigma = draw(t.intensity_sigma, r), smooth = draw(t.smoothing, r);
    require(strength >= 0 && strength < 1, "compression strength must be in [0, 1)");
    return {std::move(p),chosen,strength,sigma,smooth};
}
inline Parameters sample(const SheetCompression& t, const Sample& s, Random& r) {
    auto [p,chosen,strength,sigma,smooth]=sheet_parameters(t,s,r); Grid g(p.shape); const auto d=p.shape.size();
    const auto& image = s.at(t.image).array;
    const auto build = [&]<class Value>() {
    auto values = image.channel<Value>(0); if (sigma > .5) gaussian_blur(values, g, std::vector<double>(d, sigma), 6, Boundary::Mirror, true);
    auto [lo, hi] = std::minmax_element(values.begin(), values.end()); double low = *lo, span = *hi - low;
    if (span <= 1e-8) return;
    const auto step = g.stride[chosen], extent = g.shape[chosen];
    for (std::size_t block = 0; block < g.size; block += extent * step)
        for (std::size_t x = 0; x < extent; ++x)
            for (std::size_t j = 0; j < step; ++j) {
                const auto i = block + x * step + j;
                double norm = (values[i] - low) / span;
                values[i] = (t.threshold >= 0 ? double(norm < t.threshold) : 1 - norm) * strength;
                if (x) values[i] += values[i - step];
            }
    if (smooth > .5) { std::vector<double> sigmas(d, smooth); sigmas[chosen] = 0; gaussian_blur(values, g, sigmas, 6, Boundary::Mirror, true); }
    p.field = std::vector<float>(g.size * d);
    for (std::size_t i = 0; i < g.size; ++i) p.field[i * d + chosen] = -values[i] * double(g.shape[chosen] - 1) / g.shape[chosen];
    p.border = true;
    };
    if (image.bytes==4) build.template operator()<float>(); else build.template operator()<double>();
    return p;
}
template<class T> concept Geometric = std::is_same_v<T, Flip> || std::is_same_v<T, Rotate90> || std::is_same_v<T, Transpose> || std::is_same_v<T, CropPad> || std::is_same_v<T, Affine> || std::is_same_v<T, Elastic> || std::is_same_v<T, SheetCompression>;
template<Geometric T> void apply(const T&, Sample& s, const Parameters& p) { apply_geometry(s, p); }

// Resolution simulation deliberately has its own image/target routing, distinct
// from a geometric deformation (point coordinates do not move).
struct LowResolution { Scalar scale = Uniform{.5, 1}; std::string axes = "zyx"; bool shared_axes = true; ChannelOptions options; };
struct ThickSlice { Scalar scale = Uniform{.25, .6}; std::string axes; bool labels = true; double p = 1; };
template<std::size_t D,class Real> Array resize_array_impl(const Array& a, const Shape& shape, bool nearest, bool legacy_nearest) {
    Grid input(a.spatial()), output(shape);
    Shape out_shape{a.shape[0]}; out_shape.insert(out_shape.end(), shape.begin(), shape.end());
    auto out = Array::uninitialized(out_shape, a.kind, a.bytes);
    struct Axis { std::size_t lo, hi; Real fraction; };
    std::array<std::vector<Axis>, D> positions;
    for (std::size_t axis = 0; axis < D; ++axis) {
        const double ratio = double(input.shape[axis]) / shape[axis];
        const double offset = legacy_nearest ? 0 : .5 * ratio - .5;
        for (std::size_t x = 0; x < shape[axis]; ++x) {
            double q = std::clamp(ratio * double(x) + offset, 0., double(input.shape[axis] - 1));
            auto lo = std::size_t(std::floor(q)); Real fraction = q - lo;
            if (nearest) {
                auto at = lo + (!legacy_nearest && fraction >= .5);
                positions[axis].push_back({at * input.stride[axis], 0, 0});
            } else positions[axis].push_back({lo * input.stride[axis], std::min(lo + 1, input.shape[axis] - 1) * input.stride[axis], fraction});
        }
    }
    a.typed([&](const auto* src) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(src)>>;
        auto* dst = static_cast<T*>(out.data());
        const auto nz = D == 3 ? shape[0] : 1, ny = shape[D - 2], nx = shape.back();
        for (std::size_t c = 0; c < a.shape[0]; ++c)
            for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
                for (std::size_t x = 0; x < nx; ++x) {
                    std::array<Axis, D> q;
                    if constexpr (D == 3) q[0] = positions[0][z];
                    q[D - 2] = positions[D - 2][y]; q[D - 1] = positions[D - 1][x];
                    auto i = c * output.size + (z * ny + y) * nx + x;
                    if (nearest) {
                        std::size_t at = c * input.size;
                        for (const auto& axis : q) at += axis.lo;
                        dst[i] = src[at];
                    } else {
                        Real value = 0;
                        for (std::size_t corner = 0; corner < (1u << D); ++corner) {
                            Real weight = 1; std::size_t at = c * input.size;
                            for (std::size_t axis = 0; axis < D; ++axis) {
                                bool hi = (corner >> axis) & 1;
                                weight *= hi ? q[axis].fraction : 1 - q[axis].fraction;
                                at += hi ? q[axis].hi : q[axis].lo;
                            }
                            if (weight) value += weight * Real(src[at]);
                        }
                        if constexpr (std::is_floating_point_v<T>) dst[i] = static_cast<T>(value);
                        else out.set(i, value);
                    }
                }
    });
    return out;
}
inline Array resize_array(const Array& a, const Shape& shape, bool nearest, bool legacy_nearest = false) {
    if (a.kind != 'f' || a.bytes == 4)
        return shape.size() == 2 ? resize_array_impl<2,float>(a, shape, nearest, legacy_nearest)
                                 : resize_array_impl<3,float>(a, shape, nearest, legacy_nearest);
    return shape.size() == 2 ? resize_array_impl<2,double>(a, shape, nearest, legacy_nearest)
                             : resize_array_impl<3,double>(a, shape, nearest, legacy_nearest);
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
template<class Resize>
inline void apply_resolution(const LowResolution&, Sample& s, const Parameters& p, Resize&& resize) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; Shape shape = a.spatial(); auto n = product(shape); Shape one_shape{1}; one_shape.insert(one_shape.end(), shape.begin(), shape.end());
        Array one(one_shape, a.kind, a.bytes, static_cast<const char*>(std::as_const(a).data()) + c.channel * n * a.bytes);
        for (std::size_t j = 0; j < shape.size(); ++j) shape[j] = std::max<std::size_t>(1, std::size_t(round_even(shape[j] * c.values[j])));
        auto restored = resize(resize(one, shape, true, false), a.spatial(), false, false);
        std::memcpy(static_cast<char*>(a.data()) + c.channel * n * a.bytes, restored.data(), n * a.bytes);
    }
    restore_valid(s);
}
inline Parameters sample(const ThickSlice& t, const Sample& s, Random& r) {
    auto shape = sample_shape(s); auto allowed = t.axes.empty() ? axes(shape.size()) : t.axes;
    auto a = std::size_t(axis(allowed[r.index(allowed.size())], shape.size()));
    double scale = draw(t.scale, r); require(scale > 0 && scale <= 1, "thick-slice scale must be in (0, 1]");
    shape[a] = std::max<std::size_t>(1, std::size_t(round_even(shape[a] * scale))); Parameters p; p.shape = shape; return p;
}
template<class Resize>
inline void apply_resolution(const ThickSlice& t, Sample& s, const Parameters& p, Resize&& resize) {
    for (auto& [name, f] : s) if (f.target.role == Role::Image || (t.labels && (f.target.role == Role::Label || f.target.role == Role::Scalar))) {
        auto shape = f.array.spatial(); f.array = resize(resize(f.array, p.shape, true, true), shape, true, true);
    }
    restore_valid(s);
}
inline void apply(const LowResolution& t, Sample& s, const Parameters& p) { apply_resolution(t, s, p, resize_array); }
inline void apply(const ThickSlice& t, Sample& s, const Parameters& p) { apply_resolution(t, s, p, resize_array); }
} // namespace vcr::augment
