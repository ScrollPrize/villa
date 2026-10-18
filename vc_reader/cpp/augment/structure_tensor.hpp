#pragma once
#include "separable_filter.hpp"
#include "derived_field.hpp"

namespace vcr::augment {
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

namespace structure_tensor_detail {
using namespace filter_detail;
using namespace derived_field_detail;
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
inline Taps identity() { return {{1.}, 0}; }

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

} // namespace structure_tensor_detail

inline Parameters sample(const StructureTensor&, const Sample&, Random&) { return {}; }
// Gradient components (d channels, array-axis order) of the scaled input.
template<class T> std::vector<std::vector<T>> structure_gradients(const StructureTensor& t, const std::vector<T>& input, const Grid& g) {
    using namespace structure_tensor_detail;
    const auto d = g.shape.size();
    auto base = input; for (auto& x : base) x = T(double(x) * t.scale);
    if (t.derivative == Derivative::Pavel && t.sigma > 0) separable(base, g, std::vector<Taps>(d, torch_gaussian(t.sigma)), t.boundary);
    std::vector<std::vector<T>> gradient(d, base);
    const auto taps = gradient_taps(t, d);
    for (std::size_t a = 0; a < d; ++a) separable(gradient[a], g, taps[a], t.boundary);
    return gradient;
}
inline void apply(const StructureTensor& t, Sample& s, const Parameters&) {
    using namespace structure_tensor_detail;
    const auto& f = s.at(t.on); const auto& a = f.array; Grid g(a.spatial()); const int d = int(g.shape.size());
    require(t.channel < a.shape[0], "StructureTensor channel out of bounds");
    const auto name = t.output.empty() ? t.on + "_st" : t.output;
    std::vector<std::pair<std::string, Field>> outputs;
    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const auto input = a.channel<T>(t.channel);
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
            for (int c = 0; c < count; ++c) out.array.channel(std::size_t(c), components[c]);
        } else {
            std::vector<std::vector<T>> vectors(d, std::vector<T>(g.size)); std::vector<T> gap(g.size);
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
