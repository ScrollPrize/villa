#pragma once
#include "labels.hpp"

// Derived geometric targets: exact Euclidean distance transforms with optional
// Scharr normals, and structure tensors (components, eigenvector fields or a
// point-evaluated tensor). Each creates new fields; place them after geometry.
// The 1D kernels, distance envelopes and eigen solver are shared with CUDA.
namespace vcr::augment {
enum class DistanceMode { Signed, Inside, Outside };
// vesuvius aux targets: inside = EDT(foreground), outside = EDT(background),
// signed = outside - inside; foreground is value > threshold of one channel.
struct DistanceTransform {
    std::vector<std::string> on; std::size_t channel = 0; double threshold = 0;
    DistanceMode mode = DistanceMode::Signed; std::string suffix = "_sdt";
    std::string normals; std::string components = "zyx";  // normals: field suffix, empty for none
    double p = 1;
};
enum class Derivative { Pavel, Gaussian };
enum class StructureField { Components, Normal, Tangent };
// pavel: vesuvius StructureTensorComputer (Gaussian presmoothing radius
// int(3 sigma), Holoborodko derivative, component smoothing int(3 rho)).
// gaussian: SciPy Gaussian derivative (radius int(truncate*sigma + .5)) and
// Gaussian integration, as tracers' ct_structure_tensor. at: evaluate one
// tensor with Gaussian weights normalized over the whole grid.
struct StructureTensor {
    std::string on = "image"; std::size_t channel = 0; std::string output;
    double sigma = 1, rho = 1.5, truncate = 3, scale = 1;
    Derivative derivative = Derivative::Pavel; Boundary boundary = Boundary::Zero;
    StructureField field = StructureField::Components; std::string components = "zyx", confidence;
    std::vector<double> at;
    double p = 1;
};

namespace structure_detail {
struct Taps { std::vector<double> weights; int radius = 0; };
// Correlation taps: out[i] = sum_k weights[k] * in[i + k - radius].
inline Taps torch_gaussian(double sigma) {
    const int radius = int(3 * sigma); Taps t{std::vector<double>(2 * radius + 1), radius}; double sum = 0;
    for (int k = -radius; k <= radius; ++k) sum += t.weights[k + radius] = std::exp(-double(k * k) / (2 * sigma * sigma));
    for (auto& w : t.weights) w /= sum;
    return t;
}
// SciPy _gaussian_kernel1d (order 0/1), reversed for correlation.
inline Taps scipy_gaussian(double sigma, int order, double truncate) {
    const int radius = int(truncate * sigma + .5); Taps t{std::vector<double>(2 * radius + 1), radius}; double sum = 0;
    for (int k = -radius; k <= radius; ++k) sum += t.weights[k + radius] = std::exp(-.5 / (sigma * sigma) * k * k);
    for (int k = -radius; k <= radius; ++k) {
        t.weights[k + radius] /= sum;
        if (order) t.weights[k + radius] *= double(k) / (sigma * sigma);  // phi(x) * (-x/sigma^2), reversed
    }
    return t;
}
// Holoborodko derivative as Torch conv taps (cross-correlation).
inline Taps pavel_derivative() { return {{2 / 96., 1 / 96., -16 / 96., -27 / 96., 0., 27 / 96., 16 / 96., -1 / 96., -2 / 96.}, 4}; }
inline Taps pavel_smooth() { return {{1 / 16., 4 / 16., 6 / 16., 4 / 16., 1 / 16.}, 2}; }
// Scharr (skimage, ndi.convolve): x[i+1] - x[i-1]; [3, 10, 3]/16 smoothing.
inline Taps scharr_derivative() { return {{-1., 0., 1.}, 1}; }
inline Taps scharr_smooth() { return {{3 / 16., 10 / 16., 3 / 16.}, 1}; }
inline Taps identity() { return {{1.}, 0}; }

// Index of a neighbour, or -1 for a zero-padded sample.
VCR_HD inline long long neighbour(long long x, long long n, Boundary edge) {
    if (x >= 0 && x < n) return x;
    if (edge == Boundary::Zero) return -1;
    if (edge == Boundary::Nearest) return x < 0 ? 0 : n - 1;
    if (n == 1) return 0;
    const bool mirror = edge == Boundary::Mirror;
    const long long period = mirror ? 2 * (n - 1) : 2 * n;
    x = (x % period + period) % period;
    return x < n ? x : period - x - (mirror ? 0 : 1);
}
template<class T> void correlate(const T* in, T* out, const Grid& g, std::size_t axis, const Taps& t, Boundary edge) {
    const auto n = (long long)g.shape[axis]; const auto stride = g.stride[axis];
    for (std::size_t i = 0; i < g.size; ++i) {
        const auto x = (long long)g.coord(i, axis); const auto base = i - std::size_t(x) * stride;
        double sum = 0;
        for (int k = -t.radius; k <= t.radius; ++k) {
            const auto j = neighbour(x + k, n, edge);
            if (j >= 0) sum += t.weights[k + t.radius] * double(in[base + std::size_t(j) * stride]);
        }
        out[i] = T(sum);
    }
}
// Applies taps[a] along each axis a (identity taps are skipped).
template<class T> void separable(std::vector<T>& data, const Grid& g, const std::vector<Taps>& taps, Boundary edge) {
    std::vector<T> tmp(data.size());
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        correlate(data.data(), tmp.data(), g, a, taps[a], edge); data.swap(tmp);
    }
}
// Taps per axis for each gradient component, plus optional presmoothing.
inline std::vector<std::vector<Taps>> gradient_taps(const StructureTensor& t, std::size_t d) {
    std::vector<std::vector<Taps>> result(d, std::vector<Taps>(d));
    for (std::size_t a = 0; a < d; ++a) for (std::size_t b = 0; b < d; ++b) {
        if (t.derivative == Derivative::Pavel) result[a][b] = a == b ? pavel_derivative() : pavel_smooth();
        else result[a][b] = scipy_gaussian(t.sigma, a == b, t.truncate);
    }
    return result;
}
inline std::vector<Taps> integration_taps(const StructureTensor& t, std::size_t d) {
    if (t.rho <= 0) return std::vector<Taps>(d, identity());
    return std::vector<Taps>(d, t.derivative == Derivative::Pavel ? torch_gaussian(t.rho) : scipy_gaussian(t.rho, 0, t.truncate));
}
// Component order: zz, zy, zx, yy, yx, xx (3D) or yy, yx, xx (2D).
VCR_HD inline int component_count(int d) { return d * (d + 1) / 2; }
VCR_HD inline void component_axes(int c, int d, int& a, int& b) {
    for (a = 0; a < d; ++a) { const int row = d - a; if (c < row) { b = a + c; return; } c -= row; }
}

// Cyclic Jacobi eigen-decomposition of a symmetric d x d matrix (d <= 3).
// Ascending values; vectors are columns, sign-normalized so the largest
// magnitude component is positive.
VCR_HD inline void symmetric_eigen(double* a, int d, double* values, double* vectors) {
    for (int i = 0; i < d * d; ++i) vectors[i] = i % (d + 1) == 0;
    for (int sweep = 0; sweep < 32; ++sweep) {
        double off = 0, total = 0;
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) { total += a[i * d + j] * a[i * d + j]; if (i != j) off += a[i * d + j] * a[i * d + j]; }
        if (off <= 1e-30 * total || off == 0) break;
        for (int p = 0; p < d - 1; ++p) for (int q = p + 1; q < d; ++q) {
            const double apq = a[p * d + q];
            if (apq == 0) continue;
            const double theta = (a[q * d + q] - a[p * d + p]) / (2 * apq);
            const double tt = (theta >= 0 ? 1. : -1.) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
            const double c = 1 / std::sqrt(tt * tt + 1), s = tt * c;
            for (int k = 0; k < d; ++k) {
                const double akp = a[k * d + p], akq = a[k * d + q];
                a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq;
            }
            for (int k = 0; k < d; ++k) {
                const double apk = a[p * d + k], aqk = a[q * d + k];
                a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < d; ++k) {
                const double vkp = vectors[k * d + p], vkq = vectors[k * d + q];
                vectors[k * d + p] = c * vkp - s * vkq; vectors[k * d + q] = s * vkp + c * vkq;
            }
        }
    }
    for (int i = 0; i < d; ++i) values[i] = a[i * d + i];
    for (int i = 0; i < d; ++i) for (int j = i + 1; j < d; ++j) if (values[j] < values[i]) {
        const double v = values[i]; values[i] = values[j]; values[j] = v;
        for (int k = 0; k < d; ++k) { const double x = vectors[k * d + i]; vectors[k * d + i] = vectors[k * d + j]; vectors[k * d + j] = x; }
    }
    for (int j = 0; j < d; ++j) {
        int largest = 0; for (int k = 1; k < d; ++k) if (std::fabs(vectors[k * d + j]) > std::fabs(vectors[largest * d + j])) largest = k;
        if (vectors[largest * d + j] < 0) for (int k = 0; k < d; ++k) vectors[k * d + j] = -vectors[k * d + j];
    }
}
// Eigenvector (array-axis order) and relative eigengap of one tensor: the
// normal uses the largest eigenvalue, gap (l1-l2)/l1; the tangent the smallest,
// gap (l2-l3)/l1 in 3D or (l1-l2)/l1 in 2D. A vanishing tensor gives zeros.
VCR_HD inline double principal(const double* components, int d, bool normal, double* vector) {
    double m[9];
    for (int c = 0; c < component_count(d); ++c) { int a, b; component_axes(c, d, a, b); m[a * d + b] = m[b * d + a] = components[c]; }
    double values[3], vectors[9]; symmetric_eigen(m, d, values, vectors);
    const double largest = values[d - 1];
    if (!(largest > 1e-12)) { for (int k = 0; k < d; ++k) vector[k] = 0; return 0; }
    const int column = normal ? d - 1 : 0;
    for (int k = 0; k < d; ++k) vector[k] = vectors[k * d + column];
    return normal ? (values[d - 1] - values[d - 2]) / largest : (values[1] - values[0]) / largest;
}

// Felzenszwalb-Huttenlocher squared distance of one line in place (stride
// apart). v/z are scratch of n and n+1 entries. Infinite lines stay infinite.
template<class T, class V, class Z> VCR_HD void envelope(T* f, std::size_t n, std::size_t stride, V* v, Z* z, T* scratch) {
    const double inf = 1e300;
    long long k = -1;
    for (long long q = 0; q < (long long)n; ++q) {
        const double fq = double(f[q * stride]);
        if (!(fq < inf)) continue;
        while (k >= 0) {
            const double fv = double(f[(long long)v[k] * stride]);
            const double s = ((fq + double(q) * q) - (fv + double(v[k]) * v[k])) / (2. * q - 2. * v[k]);
            if (s <= z[k]) --k; else { ++k; v[k] = q; z[k] = s; z[k + 1] = inf; break; }
        }
        if (k < 0) { k = 0; v[0] = q; z[0] = -inf; z[1] = inf; }
    }
    for (std::size_t q = 0; q < n; ++q) scratch[q] = f[q * stride];
    if (k < 0) return;
    long long j = 0;
    for (long long q = 0; q < (long long)n; ++q) {
        while (z[j + 1] < double(q)) ++j;
        const double dq = double(q) - double(v[j]);
        f[q * stride] = T(dq * dq + double(scratch[v[j]]));
    }
}
// Squared EDT to the nearest voxel where `feature` holds, all axes.
inline void squared_edt(std::vector<double>& f, const Grid& g) {
    const auto n = *std::max_element(g.shape.begin(), g.shape.end());
    std::vector<long long> v(n); std::vector<double> z(n + 1), scratch(n);
    for (std::size_t a = 0; a < g.shape.size(); ++a) {
        const auto stride = g.stride[a], extent = g.shape[a];
        for (std::size_t i = 0; i < g.size; ++i) {
            if (g.coord(i, a) != 0) continue;
            envelope(f.data() + i, extent, stride, v.data(), z.data(), scratch.data());
        }
    }
}
// Distances to the nearest voxel of the other class. A class absent from the
// grid contributes zero (SciPy's result is undefined there).
inline std::vector<double> distance(const std::vector<std::uint8_t>& foreground, const Grid& g, bool inside) {
    std::vector<double> f(g.size); bool any = false;
    for (std::size_t i = 0; i < g.size; ++i) { const bool feature = bool(foreground[i]) != inside; f[i] = feature ? 0 : 1e300; any |= feature; }
    if (!any) return std::vector<double>(g.size, 0.);
    squared_edt(f, g);
    for (auto& x : f) x = std::sqrt(x);
    return f;
}
inline Target derived(Role role, const std::string& components = "zyx") { Target t; t.role = role; t.components = components; return t; }
} // namespace structure_detail

inline Parameters sample(const DistanceTransform&, const Sample&, Random&) { return {}; }
inline void apply(const DistanceTransform& t, Sample& s, const Parameters&) {
    using namespace structure_detail;
    std::vector<std::pair<std::string, Field>> outputs;
    for (const auto& [name, f] : s) if (selected_label(name, f, t.on)) {
        const auto& a = f.array; Grid g(a.spatial()); const auto d = g.shape.size();
        require(t.channel < a.shape[0], "DistanceTransform channel out of bounds");
        const auto values = a.channel(t.channel);
        std::vector<std::uint8_t> foreground(g.size);
        for (std::size_t i = 0; i < g.size; ++i) foreground[i] = values[i] > t.threshold;
        std::vector<double> result(g.size, 0.);
        if (t.mode != DistanceMode::Inside) result = distance(foreground, g, false);
        if (t.mode != DistanceMode::Outside) { auto inside = distance(foreground, g, true); for (std::size_t i = 0; i < g.size; ++i) result[i] -= inside[i]; }
        if (t.mode == DistanceMode::Inside) for (auto& x : result) x = -x;
        Shape shape{1}; shape.insert(shape.end(), g.shape.begin(), g.shape.end());
        Field out{Array(shape), derived(Role::Scalar), name}; out.array.channel(0, result);
        if (!t.normals.empty()) {
            Shape vshape{d}; vshape.insert(vshape.end(), g.shape.begin(), g.shape.end());
            Field normals{Array(vshape), derived(Role::NormalField, t.components), name};
            std::vector<std::vector<double>> gradient(d, result);
            for (std::size_t a = 0; a < d; ++a) {
                std::vector<Taps> taps(d, scharr_smooth()); taps[a] = scharr_derivative();
                separable(gradient[a], g, taps, Boundary::Reflect);
            }
            for (std::size_t i = 0; i < g.size; ++i) {
                double norm = 0; for (std::size_t a = 0; a < d; ++a) norm += gradient[a][i] * gradient[a][i];
                norm = std::max(std::sqrt(norm), 1e-6);
                for (std::size_t a = 0; a < d; ++a) gradient[a][i] /= norm;
            }
            for (std::size_t j = 0; j < d; ++j) normals.array.channel(j, gradient[std::size_t(axis(t.components[j], d))]);
            outputs.emplace_back(name + t.normals, std::move(normals));
        }
        outputs.emplace_back(name + t.suffix, std::move(out));
    }
    for (auto& [name, f] : outputs) { require(!s.count(name), "derived field already exists: " + name); s.emplace(name, std::move(f)); }
}

inline Parameters sample(const StructureTensor&, const Sample&, Random&) { return {}; }
// Gradient components (d channels, array-axis order) of the scaled input.
template<class T> std::vector<std::vector<T>> structure_gradients(const StructureTensor& t, const std::vector<T>& input, const Grid& g) {
    using namespace structure_detail;
    const auto d = g.shape.size();
    auto base = input; for (auto& x : base) x = T(double(x) * t.scale);
    if (t.derivative == Derivative::Pavel && t.sigma > 0) separable(base, g, std::vector<Taps>(d, torch_gaussian(t.sigma)), t.boundary);
    std::vector<std::vector<T>> gradient(d, base);
    const auto taps = gradient_taps(t, d);
    for (std::size_t a = 0; a < d; ++a) separable(gradient[a], g, taps[a], t.boundary);
    return gradient;
}
inline void apply(const StructureTensor& t, Sample& s, const Parameters&) {
    using namespace structure_detail;
    const auto& f = s.at(t.on); const auto& a = f.array; Grid g(a.spatial()); const int d = int(g.shape.size());
    require(t.channel < a.shape[0], "StructureTensor channel out of bounds");
    const auto name = t.output.empty() ? t.on + "_st" : t.output;
    std::vector<std::pair<std::string, Field>> outputs;
    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const auto source = a.channel(t.channel); std::vector<T> input(source.begin(), source.end());
        auto gradient = structure_gradients(t, input, g);
        const int count = component_count(d);
        std::vector<std::vector<T>> components(count, std::vector<T>(g.size));
        for (int c = 0; c < count; ++c) { int p, q; component_axes(c, d, p, q); for (std::size_t i = 0; i < g.size; ++i) components[c][i] = gradient[p][i] * gradient[q][i]; }
        if (!t.at.empty()) {
            // One tensor: Gaussian weights normalized over the grid.
            std::vector<double> weight(g.size, 1.); double total = 0;
            for (std::size_t i = 0; i < g.size; ++i) {
                for (int axis_ = 0; axis_ < d; ++axis_) { const double u = (double(g.coord(i, axis_)) - t.at[axis_]) / t.rho; weight[i] *= std::exp(-.5 * u * u); }
                total += weight[i];
            }
            Array tensor(Shape{std::size_t(d), std::size_t(d)}, 'f', sizeof(T));
            for (int c = 0; c < count; ++c) {
                double sum = 0; for (std::size_t i = 0; i < g.size; ++i) sum += weight[i] / total * double(components[c][i]);
                int p, q; component_axes(c, d, p, q); tensor.set(p * d + q, sum); tensor.set(q * d + p, sum);
            }
            outputs.emplace_back(name, Field{std::move(tensor), derived(Role::Vectors), t.on});
            return;
        }
        for (auto& c : components) separable(c, g, integration_taps(t, d), t.boundary);
        Shape shape{std::size_t(t.field == StructureField::Components ? count : d)}; shape.insert(shape.end(), g.shape.begin(), g.shape.end());
        Field out{Array(shape, 'f', sizeof(T)), derived(t.field == StructureField::Components ? Role::Scalar : Role::NormalField, t.components), t.on};
        Array confidence(Shape{1}); if (!t.confidence.empty()) { Shape c{1}; c.insert(c.end(), g.shape.begin(), g.shape.end()); confidence = Array(c, 'f', sizeof(T)); }
        if (t.field == StructureField::Components) {
            for (int c = 0; c < count; ++c) out.array.channel(std::size_t(c), std::vector<double>(components[c].begin(), components[c].end()));
        } else {
            std::vector<std::vector<double>> vectors(d, std::vector<double>(g.size)); std::vector<double> gap(g.size);
            for (std::size_t i = 0; i < g.size; ++i) {
                double tensor[6], vector[3]; for (int c = 0; c < count; ++c) tensor[c] = double(components[c][i]);
                gap[i] = principal(tensor, d, t.field == StructureField::Normal, vector);
                for (int k = 0; k < d; ++k) vectors[k][i] = vector[k];
            }
            for (int j = 0; j < d; ++j) out.array.channel(std::size_t(j), vectors[std::size_t(axis(t.components[j], d))]);
            if (!t.confidence.empty()) confidence.channel(0, gap);
        }
        outputs.emplace_back(name, std::move(out));
        if (!t.confidence.empty()) outputs.emplace_back(t.confidence, Field{std::move(confidence), derived(Role::Scalar), t.on});
    };
    if (a.kind == 'f' && a.bytes == 4) run(float{}); else run(double{});
    for (auto& [key, field] : outputs) { require(!s.count(key), "derived field already exists: " + key); s.emplace(key, std::move(field)); }
}
} // namespace vcr::augment
