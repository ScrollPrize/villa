#pragma once
#include "intensity.hpp"
#include "spatial.hpp"
#include <set>

namespace vcr::transforms {
struct Equalize; struct HistogramMatching; struct CLAHE; struct UnsharpMask;
// Transforms that read uint8 images themselves (writing float32), rather than widening them first.
template<class T> constexpr bool reads_uint8 =
    std::is_same_v<T, MedianFilter> || std::is_same_v<T, GaussianBlur> || std::is_same_v<T, Equalize> ||
    std::is_same_v<T, HistogramMatching> || std::is_same_v<T, CLAHE> || std::is_same_v<T, UnsharpMask>;
template<class Values, class Counts>
inline std::pair<double, double> palette_bounds(const Values& values, const Counts& counts) {
    double low = INFINITY, high = -INFINITY;
    for (std::size_t i = 0; i < values.size(); ++i) if (counts[i]) {
        low = std::min(low, double(values[i])); high = std::max(high, double(values[i]));
    }
    return {low, high};
}
template<class Values, class Counts>
inline std::pair<double, double> palette_statistics(const Values& values, const Counts& counts, std::size_t size) {
    double m = 0, variance = 0;
    for (std::size_t i = 0; i < values.size(); ++i) if (counts[i]) m += double(values[i]) * counts[i];
    m /= size;
    for (std::size_t i = 0; i < values.size(); ++i) if (counts[i]) {
        double delta = values[i] - m; variance += delta * delta * counts[i];
    }
    return {m, size > 1 ? std::sqrt(variance / (size - 1)) : 0};
}

struct PaletteChannel {
    std::vector<float> values;
    std::array<std::size_t, 256> counts{};
    bool counted = false;
    void histogram(const std::uint8_t* codes, std::size_t n) {
        if (counted) return;
        counts.fill(0);
        for (std::size_t i = 0; i < n; ++i) ++counts[codes[i]];
        counted = true;
    }
    template<class T> void intensity(const T& t, const ChannelPlan& p, const std::uint8_t* codes, std::size_t n) {
        if constexpr (std::is_same_v<T, Gamma> || std::is_same_v<T, Contrast> || std::is_same_v<T, Invert>) histogram(codes, n);
        if constexpr (std::is_same_v<T, Gamma>) {
            // Unoccupied entries must not affect bounds or produce irrelevant NaNs.
            const auto b = palette_bounds(values, counts);
            for (std::size_t i = 0; i < values.size(); ++i) if (!counts[i]) values[i] = float(b.first);
            gamma_values(values, p,
                [&](const auto& v) { return palette_statistics(v, counts, n); },
                [&](const auto& v) { return palette_bounds(v, counts); }, t);
        } else {
            double m = 0, low = 0, high = 0;
            if constexpr (std::is_same_v<T, Contrast> || std::is_same_v<T, Invert>) {
                m = palette_statistics(values, counts, n).first;
                std::tie(low, high) = palette_bounds(values, counts);
            }
            for (auto& x : values) x = float(pointwise_intensity(t, x, p, float(m), float(low), float(high)));
        }
    }
    void median(std::uint8_t* data, const Grid& g, std::size_t k) {
        if (k == 1 || values.size() == 1) return;
        // Invert and negative brightness can reverse the palette's order.
        // Rank selection must use intensity order, never the old code order.
        std::vector<std::uint8_t> order(values.size()), ranks(values.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return values[a] < values[b]; });
        auto previous = values;
        for (std::size_t i = 0; i < order.size(); ++i) { values[i] = previous[order[i]]; ranks[order[i]] = std::uint8_t(i); }
        std::vector<std::uint8_t> codes(g.size);
        for (std::size_t i = 0; i < g.size; ++i) codes[i] = ranks[data[i]];
        const auto dim = g.shape.size(), nz = dim == 3 ? g.shape[0] : 1;
        const auto ny = g.shape[dim - 2], nx = g.shape.back(), kz = dim == 3 ? k : 1;
        auto iz = median_indices(nz, kz, ny * nx), iy = median_indices(ny, k, nx), ix = median_indices(nx, k, 1);
        std::iota(order.begin(), order.end(), 0);
        std::span<std::uint8_t> output(data, g.size);
        median_rank_filter(codes.data(), output, order, g, k, iz, iy, ix);
        counted = false;
    }
};

// Local to one run/replay; the public sample always enters and leaves in its
// original floating dtype. Validity-masked images use the ordinary kernels so
// fill restoration continues to occur after every individual transform.
class CompactState {
    std::map<std::string, std::vector<PaletteChannel>> fields_;
    std::set<std::string> rejected_;
    bool enabled_;
    // A program of one transform: nothing later reuses an encoding.
    bool single_ = false;
    bool encode(Sample& s, const std::string& key, bool quantize, std::size_t quantized_channel = 0) {
        if (fields_.contains(key)) return true;
        if (rejected_.contains(key)) return false;
        auto& field = s.at(key); const auto& source = std::as_const(field.array);
        if (!encodable(s, key)) return false;
        const auto n = product(source.spatial());
        std::vector<PaletteChannel> channels(source.shape[0]);
        auto bytes = Array::uninitialized(source.shape, 'u', 1);
        bytes.widened = source.widened;  // the decoded image's destination
        if (narrow_image(field)) {
            for (std::size_t c = 0; c < channels.size(); ++c)
                narrow_palette(static_cast<const std::uint8_t*>(source.data()) + c * n, n,
                               static_cast<std::uint8_t*>(bytes.data()) + c * n, channels[c].values);
            field.array = std::move(bytes); fields_.emplace(key, std::move(channels)); return true;
        }
        for (std::size_t c = 0; c < channels.size(); ++c) {
            std::span<const float> input(static_cast<const float*>(source.data()) + c * n, n);
            std::vector<std::uint8_t> codes;
            const bool exact = float_palette(input, codes, channels[c].values, false);
            const bool encoded = exact || (quantize && c == quantized_channel &&
                encode_median_palette(input, codes, channels[c].values, true));
            if (!encoded ||
                !std::all_of(channels[c].values.begin(), channels[c].values.end(), [](float x) { return std::isfinite(x); })) {
                rejected_.insert(key); return false;
            }
            std::memcpy(static_cast<std::uint8_t*>(bytes.data()) + c * n, codes.data(), n);
        }
        field.array = std::move(bytes); fields_.emplace(key, std::move(channels)); return true;
    }
    bool encodable(const Sample& s, const std::string& key) const {
        const auto& field = s.at(key); const auto& a = field.array;
        return enabled_ && !rejected_.contains(key) && !s.contains(field.target.valid) &&
               ((a.kind == 'f' && a.bytes == 4) || narrow_image(field));
    }
    // float_palette(sorted = false) of the uint8 values' float32 form: entries in the order it meets
    // them (sampled probes first, then every voxel), so palette statistics sum in the same order.
    static void narrow_palette(const std::uint8_t* v, std::size_t n, std::uint8_t* codes, std::vector<float>& values) {
        std::array<std::int16_t, 256> code; code.fill(-1);
        values.clear();
        const auto meet = [&](std::uint8_t x) { if (code[x] < 0) { code[x] = std::int16_t(values.size()); values.push_back(float(x)); } };
        const auto probes = std::min<std::size_t>(4096, n);
        for (std::size_t i = 0; i < probes; ++i) meet(v[i * (n / probes)]);
        for (std::size_t i = 0; i < n && values.size() < 256; ++i) meet(v[i]);
        std::array<std::uint8_t, 256> table{};
        for (int x = 0; x < 256; ++x) table[x] = std::uint8_t(code[x] < 0 ? 0 : code[x]);
        for (std::size_t i = 0; i < n; ++i) codes[i] = table[v[i]];
    }
public:
    explicit CompactState(bool enabled, bool single = false) : enabled_(enabled), single_(single) {}
    void flush(Sample& s) {
        for (auto& [key, channels] : fields_) {
            auto& a = s.at(key).array; const auto n = product(a.spatial());
            auto output = widened_storage(a);
            const auto* codes = static_cast<const std::uint8_t*>(std::as_const(a).data());
            auto* values = static_cast<float*>(output.data());
            for (std::size_t c = 0; c < channels.size(); ++c)
                for (std::size_t i = 0; i < n; ++i) values[c * n + i] = channels[c].values[codes[c * n + i]];
            a = std::move(output);
        }
        fields_.clear();
        rejected_.clear();
    }
    template<class T> static constexpr bool supported =
        std::is_same_v<T, Gamma> || std::is_same_v<T, Contrast> || std::is_same_v<T, Invert> ||
        std::is_same_v<T, BrightnessAdd> || std::is_same_v<T, BrightnessMultiply> ||
        std::is_same_v<T, MedianFilter> || std::is_same_v<T, Flip> ||
        std::is_same_v<T, Rotate90> || std::is_same_v<T, Transpose>;
    template<class T> void before(const T&, Sample& s) {
        if constexpr (!supported<T>) flush(s);
        else {
            bool stop = false;
            for (const auto& [key, channels] : fields_) for (const auto& channel : channels) {
                bool positive_zero = false, negative_zero = false, finite = true;
                for (auto x : channel.values) {
                    finite &= std::isfinite(x);
                    if (x == 0) { if (std::signbit(x)) negative_zero = true; else positive_zero = true; }
                }
                stop |= !finite || (positive_zero && negative_zero);
            }
            if (stop) flush(s);
        }
        // uint8 images the step cannot read directly (or encode) become float32 first.
        for (auto& [key, f] : s) {
            if (!narrow_image(f) || fields_.contains(key) || reads_uint8<T>) continue;
            if (std::is_same_v<T, Gamma> && encodable(s, key)) continue;
            widen(f.array);
        }
    }
    template<class T> void apply_step(const T& t, Sample& s, const Parameters& p) {
        if constexpr (supported<T> && requires { t.options; }) {
            Parameters ordinary;
            for (const auto& c : p.channels) {
                if constexpr (std::is_same_v<T, Gamma>) encode(s, c.key, false);
                if constexpr (std::is_same_v<T, MedianFilter>) {
                    // A lone exact 2D size-3 median selects directly on the floats: same median,
                    // without encoding and decoding a palette that nothing else would use.
                    const bool direct = single_ && c.values[0] == 3 && !t.quantize && s.at(c.key).array.shape.size() == 3;
                    if (c.values[0] != 1 && !direct) encode(s, c.key, t.quantize, c.channel);
                }
                auto found = fields_.find(c.key);
                if (found == fields_.end()) { ordinary.channels.push_back(c); continue; }
                auto& a = s.at(c.key).array; Grid g(a.spatial());
                auto* codes = static_cast<std::uint8_t*>(a.data()) + c.channel * g.size;
                auto& channel = found->second[c.channel];
                if constexpr (std::is_same_v<T, MedianFilter>) channel.median(codes, g, std::size_t(c.values[0]));
                else channel.intensity(t, c, codes, g.size);
            }
            apply(t, s, ordinary);
            if constexpr (std::is_same_v<T, MedianFilter>)
                for (const auto& c : ordinary.channels) rejected_.erase(c.key);
        } else apply(t, s, p);
    }
};
} // namespace vcr::transforms
