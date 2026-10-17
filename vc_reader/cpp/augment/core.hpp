#pragma once
// Independent of Python, Torch, the reader, and process-global RNG/thread state.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace vcr::augment {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument(message);
}
inline void probability(double p) { require(std::isfinite(p) && p >= 0 && p <= 1, "probability must be in [0, 1]"); }
using Shape = std::vector<std::size_t>;
inline std::size_t product(const Shape& shape) {
    std::size_t n = 1;
    for (auto s : shape) {
        require(s == 0 || n <= std::numeric_limits<std::size_t>::max() / s, "shape overflow");
        n *= s;
    }
    return n;
}
struct Grid {
    Shape shape, stride;
    std::size_t size;
    explicit Grid(Shape s) : shape(std::move(s)), stride(shape.size()), size(product(shape)) {
        require(!shape.empty() && shape.size() <= 3 && size, "expected nonempty 1D, 2D or 3D spatial grid");
        std::size_t step = 1;
        for (auto a = shape.size(); a-- > 0;) { stride[a] = step; step *= shape[a]; }
    }
    std::size_t coord(std::size_t i, std::size_t a) const { return i / stride[a] % shape[a]; }
};
inline std::string axes(std::size_t dim) { require(dim == 2 || dim == 3, "expected 2D or 3D spatial data"); return dim == 3 ? "zyx" : "yx"; }
inline int axis(char c, std::size_t dim) {
    auto pos = axes(dim).find(c); require(pos != std::string::npos, "axis is not present in the sample"); return int(pos);
}

struct Random {
    std::uint64_t state;
    std::uint64_t next() {
        auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    double uniform() { return (double(next() >> 11) + 0.5) * 0x1.0p-53; }
    bool chance(double p) { probability(p); return uniform() < p; }
    std::size_t index(std::size_t n) {
        require(n > 0, "cannot sample an empty choice");
        auto threshold = (std::uint64_t(0) - n) % n;
        std::uint64_t value; do { value = next(); } while (value < threshold);
        return value % n;
    }
    double normal() { return std::sqrt(-2 * std::log(uniform())) * std::cos(6.2831853071795864769 * uniform()); }
    // Noise kernels consume both independent outputs. Keep normal() unchanged:
    // parameter sampling and elastic control lattices retain their RNG stream.
    template<class T = double> std::array<T, 2> normal_pair() {
        // Match image precision for the expensive transcendental operations.
        // uniform() is strictly positive, including after conversion to float.
        const T radius = std::sqrt(T(-2) * std::log(T(uniform())));
        const T angle = T(6.2831853071795864769 * uniform());
        return {radius * std::cos(angle), radius * std::sin(angle)};
    }
};
inline std::uint64_t split_seed(std::uint64_t base, const std::string& name) {
    // Stable across processes, standard libraries, and platforms.
    for (unsigned char c : name) { base ^= c; base *= UINT64_C(1099511628211); }
    Random r{base}; return r.next();
}
struct Uniform { double low, high; };
struct Normal { double mean = 0, std = .1; };
struct Choice { std::vector<double> values; };
struct BalancedContrast { double low = .75, high = 1.25; };
using Scalar = std::variant<double, Uniform, Normal, Choice, BalancedContrast>;
inline double draw(const Scalar& value, Random& r) {
    double result = std::visit([&](const auto& v) -> double {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, double>) return v;
        else if constexpr (std::is_same_v<T, Uniform>) {
            require(std::isfinite(v.low) && std::isfinite(v.high) && v.low <= v.high, "invalid uniform range");
            return v.low + (v.high - v.low) * r.uniform();
        } else if constexpr (std::is_same_v<T, Normal>) {
            require(std::isfinite(v.mean) && std::isfinite(v.std) && v.std >= 0, "invalid normal distribution");
            return v.mean + v.std * r.normal();
        } else if constexpr (std::is_same_v<T, Choice>) return v.values[r.index(v.values.size())];
        else {
            require(v.low > 0 && v.low <= 1 && v.high >= 1 && std::isfinite(v.high), "balanced contrast must bracket one");
            return r.chance(.5) ? v.low + (1 - v.low) * r.uniform() : 1 + (v.high - 1) * r.uniform();
        }
    }, value);
    require(std::isfinite(result), "sampled parameter must be finite"); return result;
}
inline int draw_integer(const Scalar& value, Random& r) {
    if (const auto* u = std::get_if<Uniform>(&value)) {
        require(u->low >= 0 && u->high <= 100000 && u->low <= u->high &&
                std::floor(u->low) == u->low && std::floor(u->high) == u->high, "invalid integer range");
        return int(u->low) + int(r.index(std::size_t(u->high - u->low + 1)));
    }
    auto v = draw(value, r);
    require(v >= 0 && v <= 100000 && std::floor(v) == v, "expected nonnegative integer parameter"); return int(v);
}

// Aligned storage with copy-on-write input views. Array copies always own their
// data; the binding retains borrowed owners until export. Nearest/permutation
// operations copy bytes, including
// int64/uint64 labels that cannot be represented exactly as floating point.
struct Array {
    Shape shape;
    char kind = 'f';
    std::size_t bytes = 4;
    std::unique_ptr<std::uint64_t[]> storage;
    const void* borrowed = nullptr;
    std::string owner;
    bool writable = false;
    Array() = default;
    void allocate(bool clear) {
        const auto words = std::max<std::size_t>(1, (size() * bytes + 7) / 8);
        storage = std::make_unique_for_overwrite<std::uint64_t[]>(words);
        if (clear) std::fill_n(storage.get(), words, 0);
    }
    Array(const Array& other) : shape(other.shape), kind(other.kind), bytes(other.bytes) {
        if (!other.data()) return;
        allocate(false);
        if (size()) std::memcpy(storage.get(), other.data(), size() * bytes);
    }
    Array& operator=(const Array& other) {
        if (this != &other) { Array copy(other); *this = std::move(copy); }
        return *this;
    }
    Array(Array&&) noexcept = default;
    Array& operator=(Array&&) noexcept = default;
    Array(Shape s, char k = 'f', std::size_t b = 4, const void* input = nullptr,
          std::string source = {}, bool inplace = false, bool zero = true)
        : shape(std::move(s)), kind(k), bytes(b), borrowed(input), owner(std::move(source)), writable(inplace) {
        require((k == 'f' && (b == 4 || b == 8)) || ((k == 'i' || k == 'u') && (b == 1 || b == 2 || b == 4 || b == 8)) || (k == 'b' && b == 1), "unsupported array dtype");
        require(size() <= (std::numeric_limits<std::size_t>::max() - 7) / b, "array size overflow");
        if (!borrowed) allocate(zero);
    }
    // For kernels that write every output element before exposing the array.
    static Array uninitialized(Shape shape, char kind, std::size_t bytes) {
        return Array(std::move(shape), kind, bytes, nullptr, {}, false, false);
    }
    std::size_t size() const { return product(shape); }
    void* data() {
        if (borrowed && !writable) {
            allocate(false);
            if (size()) std::memcpy(storage.get(), borrowed, size() * bytes);
            borrowed = nullptr; owner.clear();
        }
        return borrowed ? const_cast<void*>(borrowed) : storage.get();
    }
    const void* data() const { return borrowed ? borrowed : storage.get(); }
    Shape spatial() const { require(shape.size() >= 2, "expected a channel-first spatial array"); return Shape(shape.begin() + 1, shape.end()); }
    template<class F> decltype(auto) typed(F&& f) const {
        if (kind == 'f') { if (bytes == 4) return f(static_cast<const float*>(data())); return f(static_cast<const double*>(data())); }
        if (kind == 'b' || kind == 'u') {
            if (bytes == 1) return f(static_cast<const std::uint8_t*>(data()));
            if (bytes == 2) return f(static_cast<const std::uint16_t*>(data()));
            if (bytes == 4) return f(static_cast<const std::uint32_t*>(data()));
            return f(static_cast<const std::uint64_t*>(data()));
        }
        if (bytes == 1) return f(static_cast<const std::int8_t*>(data()));
        if (bytes == 2) return f(static_cast<const std::int16_t*>(data()));
        if (bytes == 4) return f(static_cast<const std::int32_t*>(data()));
        return f(static_cast<const std::int64_t*>(data()));
    }
    double get(std::size_t i) const { return typed([&](const auto* p) { return double(p[i]); }); }
    void require_exact_values() const {
        if ((kind != 'i' && kind != 'u') || bytes != 8) return;
        typed([&](const auto* p) {
            for (std::size_t i = 0; i < size(); ++i)
                require(p[i] >= -9007199254740991. && p[i] <= 9007199254740991.,
                        "value-based label transforms require integer IDs within +/- (2**53 - 1)");
        });
    }
    template<class T> static T checked_cast(double v) {
        if constexpr (std::is_integral_v<T>) {
            // An exclusive power-of-two upper bound also works on platforms
            // where long double cannot represent UINT64_MAX exactly.
            double upper = std::ldexp(1., std::numeric_limits<T>::digits);
            double lower = std::is_signed_v<T> ? -upper : 0;
            require(std::isfinite(v) && v >= lower && v < upper && std::trunc(v) == v,
                    "value is not representable in target dtype");
        }
        return static_cast<T>(v);
    }
    void set(std::size_t i, double v) {
        data();
        typed([&](const auto* p) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            const_cast<T*>(p)[i] = checked_cast<T>(v);
        });
    }
    void copy_element(std::size_t to, const Array& src, std::size_t from) {
        require(kind == src.kind && bytes == src.bytes, "copy dtype mismatch");
        std::memcpy(static_cast<char*>(data()) + to * bytes, static_cast<const char*>(src.data()) + from * bytes, bytes);
    }
    std::vector<double> channel(std::size_t c) const {
        require_exact_values();
        auto n = product(spatial()); require(c < shape[0], "channel out of bounds");
        std::vector<double> result(n);
        typed([&](const auto* p) { for (std::size_t i = 0; i < n; ++i) result[i] = double(p[c * n + i]); });
        return result;
    }
    void channel(std::size_t c, const std::vector<double>& values) {
        auto n = product(spatial()); require(c < shape[0] && values.size() == n, "channel shape mismatch");
        if (kind == 'f') {
            if (bytes == 4) { auto* p = static_cast<float*>(data()) + c * n; for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<float>(values[i]); }
            else std::copy(values.begin(), values.end(), static_cast<double*>(data()) + c * n);
        } else {
            data();
            typed([&](const auto* p) {
                using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
                auto* dest = const_cast<T*>(p) + c * n;
                for (std::size_t i = 0; i < n; ++i) dest[i] = checked_cast<T>(values[i]);
            });
        }
    }
};

enum class Role { Image, Label, Scalar, Valid, Points, Vectors, VectorField, NormalField, CoordinateField };
struct Target {
    Role role = Role::Image;
    double fill = 0;
    std::string valid = "padding_mask", components = "zyx", outside = "keep";
    bool normalize = false, external = true;
    bool operator==(const Target&) const = default;
};
struct Field { Array array; Target target; std::string source; std::vector<Array> pyramid; };
using Sample = std::map<std::string, Field>;
inline bool sparse(Role r) { return r == Role::Points || r == Role::Vectors; }
inline bool directional(Role r) { return r == Role::Vectors || r == Role::VectorField || r == Role::NormalField; }
inline Shape sample_shape(const Sample& s) {
    for (const auto& [name, f] : s) if (!sparse(f.target.role)) return f.array.spatial();
    throw std::invalid_argument("sample needs a spatial array");
}
inline void validate(const Sample& s) {
    auto shape = sample_shape(s); auto names = axes(shape.size());
    require(std::all_of(shape.begin(), shape.end(), [](auto extent) { return extent > 0; }), "spatial dimensions must be nonempty");
    for (const auto& [name, f] : s) {
        const auto& a = f.array; const auto& t = f.target;
        if (sparse(t.role)) require(a.shape.size() == 2 && a.shape[1] == shape.size(), name + ": expected (N, D)");
        else require(a.shape.size() >= 3 && a.shape[0] && a.spatial() == shape, name + ": expected channel-first matching spatial shape");
        if (t.role != Role::Label && t.role != Role::Valid) require(a.kind == 'f', name + ": continuous targets require float32 or float64");
        if (sparse(t.role) || directional(t.role) || t.role == Role::CoordinateField) {
            auto component_dim = t.role == Role::CoordinateField && t.external ? a.shape[0] : shape.size();
            auto order = t.components; std::sort(order.begin(), order.end()); auto expected = axes(component_dim); std::sort(expected.begin(), expected.end());
            require(order == expected, name + ": invalid component order");
            if (!sparse(t.role)) require(a.shape[0] == component_dim, name + ": one channel per component required");
        }
        // Dispatch once. Integer targets are inherently finite; construct error
        // messages only on failure, never in the voxel loop.
        const auto n = a.size();
        if (a.kind == 'f') a.typed([&](const auto* values) {
            using V = std::remove_cv_t<std::remove_pointer_t<decltype(values)>>;
            if constexpr (std::is_floating_point_v<V>) {
                // Integer OR reduction vectorizes without relaxed floating-point
                // flags. All-ones exponents identify both infinities and NaNs.
                using Bits = std::conditional_t<sizeof(V) == 4, std::uint32_t, std::uint64_t>;
                constexpr Bits exponent = sizeof(V) == 4 ? Bits(0x7f800000u) : Bits(UINT64_C(0x7ff0000000000000));
                Bits invalid = 0;
                for (std::size_t i = 0; i < n; ++i) invalid |= (std::bit_cast<Bits>(values[i]) & exponent) == exponent;
                if (invalid) throw std::invalid_argument(name + ": nonfinite input; use a validity mask for missing data");
            }
        });
        if (t.role == Role::Valid) a.typed([&](const auto* values) {
            for (std::size_t i = 0; i < n; ++i)
                if (values[i] != 0 && values[i] != 1)
                    throw std::invalid_argument(name + ": validity masks must contain only zero or one");
        });
        if (t.role == Role::Image && s.count(t.valid)) {
            const auto& mask = s.at(t.valid);
            require(mask.target.role == Role::Valid && (mask.array.shape[0] == 1 || mask.array.shape[0] == a.shape[0]), name + ": invalid validity-mask association");
        }
        require(t.outside == "keep" || t.outside == "drop" || t.outside == "error", "outside must be keep, drop or error");
    }
}
inline void restore_valid(Sample& s) {
    for (auto& [name, f] : s) {
        if (f.target.role != Role::Image || !s.count(f.target.valid)) continue;
        const auto& mask = s.at(f.target.valid).array;
        const bool pseudo = f.source.find('\x1f') != std::string::npos;
        require(mask.spatial() == f.array.spatial() &&
                (mask.shape[0] == 1 || mask.shape[0] == f.array.shape[0] ||
                 (pseudo && f.array.shape[0] % mask.shape[0] == 0)), "validity mask no longer matches image channels");
        auto n = product(f.array.spatial());
        for (std::size_t i = 0; i < f.array.size(); ++i)
            if (!mask.get(mask.shape[0] == 1 ? i % n : i % mask.size())) f.array.set(i, f.target.fill);
    }
}
template<class T> inline double mean(const std::vector<T>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
template<class T> inline double deviation(const std::vector<T>& v, double m) {
    double sum = 0; for (auto x : v) sum += (x - m) * (x - m);
    return v.size() > 1 ? std::sqrt(sum / (v.size() - 1)) : 0;
}
inline std::size_t reflect(std::int64_t i, std::size_t n) {
    auto p = static_cast<std::int64_t>(2 * n); i = (i % p + p) % p;
    return static_cast<std::size_t>(i < static_cast<std::int64_t>(n) ? i : p - 1 - i);
}
inline double round_even(double v) {
    double lo = std::floor(v), fraction = v - lo;
    return lo + (fraction > .5 || (fraction == .5 && std::fmod(lo, 2.) != 0));
}
inline std::size_t mirror(std::int64_t i, std::size_t n) {
    if (n == 1) return 0;
    auto p = std::int64_t(2 * (n - 1)); i = (i % p + p) % p;
    return std::size_t(i < std::int64_t(n) ? i : p - i);
}
// Accumulate a few adjacent outputs in registers, then write them once. Tap
// order is unchanged, including the initial addition to zero.
template<class T>
inline void convolve_run(const T* source, T* output, std::size_t count,
                         const std::vector<T>& kernel, const std::size_t* offsets) {
    constexpr std::size_t lanes = 16;
    std::size_t i = 0;
    for (; i + lanes <= count; i += lanes) {
        std::array<T, lanes> sum{};
        for (std::size_t tap = 0; tap < kernel.size(); ++tap) {
            const auto* input = source + offsets[tap] + i;
            const auto weight = kernel[tap];
            for (std::size_t j = 0; j < lanes; ++j) sum[j] += weight * input[j];
        }
        std::copy(sum.begin(), sum.end(), output + i);
    }
    for (; i < count; ++i) {
        T sum = 0;
        for (std::size_t tap = 0; tap < kernel.size(); ++tap) sum += kernel[tap] * source[offsets[tap] + i];
        output[i] = sum;
    }
}
// Coefficient construction is shared by CPU and CUDA convolution.
template<class T>
inline std::vector<T> gaussian_kernel(double sigma, double truncate, bool full_width) {
    require(sigma > 0 && std::isfinite(sigma) && truncate > 0 && std::isfinite(truncate) && sigma * truncate < 100000, "invalid Gaussian sigma");
    int radius = int(sigma * truncate + .5);
    if (full_width) {
        double extent = sigma * truncate + .5; int size = int(round_even(extent));
        if (size % 2 == 0) size += extent - size >= 0 ? 1 : -1;
        radius = std::max(0, (size - 1) / 2);
    }
    // Normalize the small coefficient table in double; full-volume buffers
    // and convolution arithmetic follow the image dtype.
    std::vector<double> weights(2 * radius + 1); double sum = 0;
    for (int d = -radius; d <= radius; ++d) sum += weights[d + radius] = std::exp(-.5 * (d / sigma) * (d / sigma));
    std::vector<T> kernel(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) kernel[i] = T(weights[i] / sum);
    return kernel;
}
template<class T>
inline void gaussian_blur(std::vector<T>& data, const Grid& g, const std::vector<double>& sigmas, double truncate = 4, bool mirror_boundary = false, bool full_width = false) {
    require(sigmas.size() == g.shape.size() && truncate > 0 && std::isfinite(truncate), "invalid Gaussian parameters");
    std::vector<T> tmp(g.size);
    for (std::size_t a = 0; a < sigmas.size(); ++a) {
        auto sigma = sigmas[a]; require(sigma >= 0 && std::isfinite(sigma) && sigma * truncate < 100000, "invalid Gaussian sigma");
        if (!sigma) continue;
        const auto kernel = gaussian_kernel<T>(sigma, truncate, full_width);
        const auto radius = int(kernel.size() / 2);
        const auto extent = g.shape[a], step = g.stride[a], block = extent * step;
        auto boundary = [&](std::int64_t x) { return mirror_boundary ? mirror(x, extent) : reflect(x, extent); };
        if (step == 1) {
            // Extend each contiguous row once; even wide kernels and singleton
            // dimensions then use the same branch-free convolution loop.
            std::vector<std::size_t> indices(extent + 2 * radius), offsets(kernel.size());
            for (std::size_t x = 0; x < indices.size(); ++x) indices[x] = boundary(std::int64_t(x) - radius);
            std::iota(offsets.begin(), offsets.end(), 0);
            std::vector<T> row(indices.size());
            for (std::size_t base = 0; base < g.size; base += extent) {
                for (std::size_t x = 0; x < row.size(); ++x) row[x] = data[base + indices[x]];
                convolve_run(row.data(), tmp.data() + base, extent, kernel, offsets.data());
            }
        } else {
            std::vector<std::size_t> offsets(extent * kernel.size());
            for (std::size_t x = 0; x < extent; ++x)
                for (std::size_t tap = 0; tap < kernel.size(); ++tap)
                    offsets[x * kernel.size() + tap] = boundary(std::int64_t(x + tap) - radius) * step;
            for (std::size_t base = 0; base < g.size; base += block)
                for (std::size_t x = 0; x < extent; ++x)
                    convolve_run(data.data() + base, tmp.data() + base + x * step, step,
                                 kernel, offsets.data() + x * kernel.size());
        }
        data.swap(tmp);
    }
}
struct ChannelOptions {
    double p = 1, p_channel = 1;
    bool shared = false;
    std::vector<std::size_t> channels;
    std::vector<std::string> on;
};
// A compact sampled record, owned by the native replay plan. Parameters are
// interpreted by their concrete transform's overload, never a string dispatcher.
struct ChannelPlan {
    std::string key;
    std::size_t channel = 0;
    std::vector<double> values, field;
    std::uint64_t seed = 0;
    // Transient backend-owned field; materialized into field for replay.
    std::shared_ptr<void> device_field;
};
struct Parameters {
    std::vector<ChannelPlan> channels;
    Shape shape;
    std::vector<double> matrix, offset, field;
    bool border = false, exact = false;
    std::shared_ptr<void> device_field;
    int nearest_mode = 0; // 0: ties-to-even, 1: half-up resize, 2: floor resize
};
template<class F>
Parameters sample_channels(const Sample& sample, const ChannelOptions& options, Random& r, F&& sample_one) {
    Parameters result; auto base = r.next();
    for (const auto& [name, f] : sample) {
        if (f.target.role != Role::Image || (!options.on.empty() && std::find(options.on.begin(), options.on.end(), name) == options.on.end())) continue;
        Random local{split_seed(base, name)};
        auto channels = options.channels;
        if (channels.empty()) { channels.resize(f.array.shape[0]); std::iota(channels.begin(), channels.end(), 0); }
        ChannelPlan shared;
        if (options.shared) shared = sample_one(f.array, local);
        for (auto c : channels) {
            require(c < f.array.shape[0], "channel out of bounds");
            if (!local.chance(options.p_channel)) continue;
            auto plan = options.shared ? shared : sample_one(f.array, local);
            plan.key = name; plan.channel = c; result.channels.push_back(std::move(plan));
        }
    }
    return result;
}
template<class F> void apply_channels(Sample& s, const Parameters& p, F&& apply_one) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; auto values = a.channel(c.channel);
        apply_one(values, Grid(a.spatial()), c); a.channel(c.channel, values);
    }
    restore_valid(s);
}
} // namespace vcr::augment
