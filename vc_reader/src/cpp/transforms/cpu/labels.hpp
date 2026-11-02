#pragma once
#include "../ops/labels.hpp"
#include <vc_reader/thinning.hpp>
#include <vc_reader/components.hpp>
#include "spatial.hpp"
#include "../shared/component_traversal.hpp"

namespace vcr::transforms {
using Footprint = std::vector<std::array<int, 3>>;
// Offsets within radius of the centre: Euclidean (ball), L1 (cross) or Chebyshev (cube).
inline Footprint norm_footprint(std::size_t dim, double radius, MorphFootprint shape) {
    require(std::isfinite(radius) && radius >= 0 && radius <= 63, "morphology radius must be in [0, 63]");
    int r = int(std::ceil(radius)); Grid g(Shape(dim, 2 * r + 1)); Footprint result;
    for (std::size_t i = 0; i < g.size; ++i) {
        std::array<int, 3> q{}; double dist = 0;
        for (std::size_t a = 0; a < dim; ++a) {
            q[a] = int(g.coord(i, a)) - r; const double m = std::abs(q[a]);
            dist = shape == MorphFootprint::Ball ? dist + m * m : shape == MorphFootprint::Cross ? dist + m : std::max(dist, m);
        }
        if (dist <= (shape == MorphFootprint::Ball ? radius * radius : radius)) result.push_back(q);
    }
    return result;
}
inline Footprint ball(std::size_t dim, double radius) { return norm_footprint(dim, radius, MorphFootprint::Ball); }
// Dilates (or erodes) a 0/1 mask by each footprint of `passes` in turn, in place.
// One padded copy is reused; its border keeps `outside` throughout.
inline void morphology_mask(std::vector<std::uint8_t>& values, const Grid& g, const std::vector<Footprint>& passes, bool dilate, bool outside = false) {
    const auto d = g.shape.size();
    Shape padding(d, 0), padded_shape = g.shape;
    for (const auto& footprint : passes) for (const auto& offset : footprint) for (std::size_t a = 0; a < d; ++a)
        padding[a] = std::max(padding[a], std::size_t(std::abs(offset[a])));
    for (std::size_t a = 0; a < d; ++a) padded_shape[a] += 2 * padding[a];
    Grid padded(padded_shape);
    std::vector<std::uint8_t> mask(padded.size, outside);
    const auto nz = d == 3 ? g.shape[0] : 1, ny = g.shape[d - 2], nx = g.shape.back();
    std::size_t origin = 0;
    for (std::size_t a = 0; a < d; ++a) origin += padding[a] * padded.stride[a];
    auto row_start = [&](std::size_t z, std::size_t y) { return origin + (d == 3 ? z * padded.stride[0] : 0) + y * padded.stride[d - 2]; };
    std::vector<std::ptrdiff_t> offsets;
    for (const auto& footprint : passes) {
        offsets.clear();
        for (const auto& offset : footprint) {
            std::ptrdiff_t step = 0;
            for (std::size_t a = 0; a < d; ++a) step += (dilate ? -offset[a] : offset[a]) * std::ptrdiff_t(padded.stride[a]);
            offsets.push_back(step);
        }
        for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y)
            std::copy_n(values.data() + (z * ny + y) * nx, nx, mask.data() + row_start(z, y));
        for (std::size_t z = 0; z < nz; ++z) for (std::size_t y = 0; y < ny; ++y) {
            const auto* row = mask.data() + row_start(z, y);
            auto* result_row = values.data() + (z * ny + y) * nx;
            // Small footprints benefit from contiguous, vectorizable boolean
            // operations. Large footprints retain early termination per voxel.
            if (offsets.size() <= 32) {
                std::fill_n(result_row, nx, std::uint8_t(!dilate));
                for (auto offset : offsets) {
                    const auto* neighbor = row + offset;
                    if (dilate) for (std::size_t x = 0; x < nx; ++x) result_row[x] |= neighbor[x];
                    else for (std::size_t x = 0; x < nx; ++x) result_row[x] &= neighbor[x];
                }
                continue;
            }
            for (std::size_t x = 0; x < nx; ++x) {
                bool result = !dilate;
                for (auto offset : offsets) {
                    bool value = row[std::ptrdiff_t(x) + offset];
                    if (dilate ? value : !value) { result = dilate; break; }
                }
                result_row[x] = result;
            }
        }
    }
}
inline std::vector<double> morphology(const std::vector<double>& input, const Grid& g, const Footprint& footprint, bool dilate, bool outside = false) {
    std::vector<std::uint8_t> mask(input.size());
    for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = input[i] != 0;
    morphology_mask(mask, g, {footprint}, dilate, outside);
    return std::vector<double>(mask.begin(), mask.end());
}
// Each stage applies `passes` in sequence: dilation (or erosion) by their Minkowski sum.
inline std::vector<double> morphology(const std::vector<double>& v, const Grid& g, const std::vector<Footprint>& passes, Morph op, bool neutral_border = false) {
    std::vector<std::uint8_t> mask(v.size());
    for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = v[i] != 0;
    if (op == Morph::Dilate || op == Morph::Close) morphology_mask(mask, g, passes, true);
    if (op != Morph::Dilate) morphology_mask(mask, g, passes, false, neutral_border);
    if (op == Morph::Open) morphology_mask(mask, g, passes, true);
    return std::vector<double>(mask.begin(), mask.end());
}
inline std::vector<double> morphology(std::vector<double> v, const Grid& g, const Footprint& footprint, Morph op, bool neutral_border = false) {
    return morphology(std::move(v), g, std::vector<Footprint>{footprint}, op, neutral_border);
}
// Repeated 3D cross used by the existing closing transform. Each cell of the
// 3x3x3 six-connected footprint expands to a size^3 block.
inline Footprint repeated_cross(std::size_t dim, double size) {
    require(dim == 3 && size >= 1 && size <= 21 && std::floor(size) == size,
            "repeated_cross requires 3D and an integer radius in [1, 21]");
    int n = int(size), width = 3 * n; Footprint result;
    for (int z = 0; z < width; ++z) for (int y = 0; y < width; ++y) for (int x = 0; x < width; ++x)
        if (std::abs(z / n - 1) + std::abs(y / n - 1) + std::abs(x / n - 1) <= 1)
            result.push_back({z - width / 2, y - width / 2, x - width / 2});
    return result;
}

inline Footprint morphology_footprint(std::size_t dim, double radius, MorphFootprint shape) {
    return shape == MorphFootprint::RepeatedCross ? repeated_cross(dim, radius) : norm_footprint(dim, radius, shape);
}
// Footprints whose sequential application equals morphology_footprint: the unit
// cross floor(radius) times, or a cube as one segment per axis. Between two
// voxels of the array there is a chain of unit steps inside its box, so
// clipping to the array between passes is exact for either border.
inline std::vector<Footprint> morphology_passes(std::size_t dim, double radius, MorphFootprint shape) {
    if (shape == MorphFootprint::Ball || shape == MorphFootprint::RepeatedCross) return {morphology_footprint(dim, radius, shape)};
    require(std::isfinite(radius) && radius >= 0 && radius <= 63, "morphology radius must be in [0, 63]");
    const int r = int(std::floor(radius));
    if (r == 0) return {Footprint{{0, 0, 0}}};
    if (shape == MorphFootprint::Cross) return std::vector<Footprint>(std::size_t(r), norm_footprint(dim, 1, shape));
    std::vector<Footprint> passes(dim);
    for (std::size_t a = 0; a < dim; ++a) for (int k = -r; k <= r; ++k) { std::array<int, 3> q{}; q[a] = k; passes[a].push_back(q); }
    return passes;
}

// Dilate, erode, open or close a 0/1 mask in place by a footprint shape.
inline void morphology_mask(std::vector<std::uint8_t>& mask, const Grid& g, MorphFootprint shape, double radius, Morph op, bool neutral_border) {
    const auto passes = morphology_passes(g.shape.size(), radius, shape);
    if (op == Morph::Dilate || op == Morph::Close) morphology_mask(mask, g, passes, true);
    if (op != Morph::Dilate) morphology_mask(mask, g, passes, false, neutral_border);
    if (op == Morph::Open) morphology_mask(mask, g, passes, true);
}

inline std::vector<std::size_t> selected_channels(std::vector<std::size_t> channels, std::size_t count, Random* r = nullptr) {
    if (channels.empty()) { channels.resize(count); std::iota(channels.begin(), channels.end(), 0); }
    for (auto c : channels) require(c < count, "channel out of bounds");
    if (r) for (std::size_t i = channels.size(); i > 1; --i) std::swap(channels[i - 1], channels[r->index(i)]);
    return channels;
}
inline bool selected_label(const std::string& key, const Field& f, const std::vector<std::string>& on) {
    return on.empty() ? f.target.role == Role::Label : std::find(on.begin(), on.end(), key) != on.end();
}
inline Parameters sample(const Morphology& t, const Sample& s, Random& r) {
    Parameters p;
    for (const auto& [key, f] : s) if (selected_label(key, f, t.on))
        for (std::size_t c = 0; c < f.array.shape[0]; ++c) p.channels.push_back({key, c, {draw(t.radius, r)}});
    return p;
}
inline void apply(const Morphology& t, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; const Grid g(a.spatial());
        a.data();
        a.typed([&](const auto* data) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(data)>>;
            auto* values = const_cast<T*>(data) + c.channel * g.size;
            // The channel maximum as std::max_element finds it on doubles.
            double maximum = double(values[0]);
            std::vector<std::uint8_t> mask(g.size);
            for (std::size_t i = 0; i < g.size; ++i) {
                const double v = double(values[i]);
                if (maximum < v) maximum = v;
                mask[i] = v > 0;
            }
            morphology_mask(mask, g, t.footprint, c.values[0], t.operation, t.neutral_border);
            // Binary results can only restore zero or the original channel maximum,
            // both already representable in this dtype; keep the double product
            // (including -0).
            const T set = static_cast<T>(1. * maximum), unset = static_cast<T>(0. * maximum);
            for (std::size_t i = 0; i < g.size; ++i) values[i] = mask[i] ? set : unset;
        });
    }
}
inline Parameters sample(const RandomBinaryOperator& t, const Sample& s, Random& r) {
    const auto& a = s.at(t.on).array; Parameters p;
    for (auto c : selected_channels(t.channels, a.shape[0], &r)) if (r.chance(t.p_channel))
        p.channels.push_back({t.on, c, {draw(t.radius, r), double(static_cast<int>(t.operations[r.index(t.operations.size())]))}});
    return p;
}
template<class MorphologyKernel>
inline void apply_binary(const RandomBinaryOperator& t, Sample& s, const Parameters& p, MorphologyKernel&& kernel) {
    auto& a = s.at(t.on).array; auto n = product(a.spatial()); auto channels = selected_channels(t.channels, a.shape[0]);
    for (const auto& c : p.channels) {
        auto original = a.channel(c.channel); auto result = kernel(original, Grid(a.spatial()), ball(a.shape.size() - 1, c.values[0]), static_cast<Morph>(int(c.values[1])));
        for (std::size_t i = 0; i < n; ++i) if (result[i] && !original[i]) for (auto other : channels) if (other != c.channel) a.set(other * n + i, 0);
        a.channel(c.channel, result);
    }
}
inline void apply(const RandomBinaryOperator& t, Sample& s, const Parameters& p) {
    apply_binary(t,s,p,[](const auto& v, const auto& g, const auto& footprint, Morph op) {
        return morphology(v,g,footprint,op);
    });
}
inline std::vector<std::vector<std::size_t>> connected_components(const std::vector<double>& input, const Grid& g, bool full) {
    std::vector<std::uint8_t> visited(g.size);
    std::vector<std::size_t> queue(g.size),offsets(g.size+1);
    const auto count=component_detail::components(input.data(),component_detail::GridView(g),full,visited.data(),queue.data(),offsets.data());
    std::vector<std::vector<std::size_t>> result; result.reserve(count);
    for(std::size_t c=0;c<count;++c)result.emplace_back(queue.begin()+offsets[c],queue.begin()+offsets[c+1]);
    return result;
}

inline void remove_component(Array& a, const ChannelPlan& p) {
    auto n = product(a.spatial());
    for (auto i : p.indices) { a.set(p.channel * n + i, 0); if (p.values[0] >= 0) a.set(std::size_t(p.values[0]) * n + i, 1); }
}
inline Parameters sample(const RemoveComponents& t, const Sample& s, Random& r) {
    probability(t.max_fraction); auto a = s.at(t.on).array; Grid g(a.spatial()); Parameters p;
    auto channels = selected_channels(t.channels, a.shape[0], &r);
    for (auto c : channels) if (r.chance(t.p_channel)) {
        auto components = connected_components(a.channel(c), g, t.full_connectivity);
        components.erase(std::remove_if(components.begin(), components.end(), [&](const auto& v) { return v.size() >= g.size * t.max_fraction; }), components.end());
        if (components.empty()) continue;
        ChannelPlan plan{t.on, c, {-1}}; plan.indices = components[r.index(components.size())];
        if (r.chance(t.p_fill) && channels.size() > 1) { auto others = channels; others.erase(std::remove(others.begin(), others.end(), c), others.end()); plan.values[0] = double(others[r.index(others.size())]); }
        remove_component(a, plan); p.channels.push_back(std::move(plan));
    }
    return p;
}
inline void apply(const RemoveComponents& t, Sample& s, const Parameters& p) { for (const auto& c : p.channels) remove_component(s.at(t.on).array, c); }
inline Parameters sample(const ReplaceLabel&, const Sample&, Random&) { return {}; }
inline void apply(const ReplaceLabel& t, Sample& s, const Parameters&) {
    auto& a = s.at(t.on).array; auto n = product(a.spatial());
    a.require_exact_values();
    require(std::isfinite(t.value) && std::abs(t.value) <= 9007199254740991. &&
            std::isfinite(t.replacement) && std::abs(t.replacement) <= 9007199254740991., "label value out of exact range");
    for (auto c : selected_channels(t.channels, a.shape[0])) for (std::size_t i = 0; i < n; ++i) if (a.get(c * n + i) == t.value) a.set(c * n + i, t.replacement);
}
inline Array regions(const Array& input, std::size_t channel, const std::vector<std::vector<double>>& labels) {
    input.require_exact_values();
    for (const auto& group : labels) for (auto value : group)
        require(std::isfinite(value) && std::abs(value) <= 9007199254740991., "label value out of exact range");
    require(channel < input.shape[0] && !labels.empty(), "invalid region channel or empty label groups");
    Shape shape = input.shape; shape[0] = labels.size(); auto out = Array::uninitialized(shape, 'b', 1); auto n = product(input.spatial());
    auto* dest = static_cast<std::uint8_t*>(out.data());
    input.typed([&](const auto* src) {
        for (std::size_t c = 0; c < labels.size(); ++c) {
            if (labels[c].size() == 1) {
                const auto label = labels[c][0];
                for (std::size_t i = 0; i < n; ++i) dest[c * n + i] = double(src[channel * n + i]) == label;
            } else for (std::size_t i = 0; i < n; ++i)
                dest[c * n + i] = std::find(labels[c].begin(), labels[c].end(), double(src[channel * n + i])) != labels[c].end();
        }
    });
    return out;
}
inline Parameters sample(const Regions&, const Sample&, Random&) { return {}; }
inline void apply(const Regions& t, Sample& s, const Parameters&) {
    const auto& source = s.at(t.on);
    Field out{regions(source.array, t.channel, t.labels), source.target, t.on, source.pyramid}; out.target.role = Role::Label;
    s[t.output.empty() ? t.on : t.output] = std::move(out);
}
inline Parameters sample(const OneHot&, const Sample&, Random&) { return {}; }
inline void apply(const OneHot& t, Sample& s, const Parameters&) {
    const auto& source = s.at(t.on); std::vector<std::vector<double>> labels; for (auto v : t.labels) labels.push_back({v});
    auto encoded = regions(source.array, t.channel, labels); auto n = product(encoded.spatial());
    if (!t.append_to.empty()) {
        auto& image = s.at(t.append_to).array; require(image.spatial() == encoded.spatial(), "one-hot image grid mismatch");
        auto shape = image.shape; shape[0] += encoded.shape[0]; auto combined = Array::uninitialized(shape, image.kind, image.bytes);
        const auto count = image.size(), encoded_count = encoded.size();
        std::memcpy(combined.data(), std::as_const(image).data(), count * image.bytes);
        const auto* bits = static_cast<const std::uint8_t*>(std::as_const(encoded).data());
        combined.typed([&](const auto* data) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(data)>>;
            auto* dest = const_cast<T*>(data) + count;
            for (std::size_t i = 0; i < encoded_count; ++i) dest[i] = static_cast<T>(bits[i]);
        });
        image = std::move(combined);
    } else { Field out{std::move(encoded), source.target, t.on, source.pyramid}; out.target.role = Role::Label; s[t.output] = std::move(out); }
    if (t.remove_source) {
        auto& a = s.at(t.on).array;
        if (a.shape[0] == 1) s.erase(t.on);
        else { auto shape = a.shape; --shape[0]; Array remaining(shape, a.kind, a.bytes); std::size_t c_out = 0;
            for (std::size_t c = 0; c < a.shape[0]; ++c) if (c != t.channel) { for (std::size_t i = 0; i < n; ++i) remaining.copy_element(c_out * n + i, a, c * n + i); ++c_out; } a = std::move(remaining); }
    }
}
inline Parameters sample(const MaskImage&, const Sample&, Random&) { return {}; }
inline void apply(const MaskImage& t, Sample& s, const Parameters&) {
    auto& image = s.at(t.on).array; const auto& mask = s.at(t.mask).array; auto n = product(image.spatial());
    require(image.spatial() == mask.spatial() && t.mask_channel < mask.shape[0], "invalid image mask");
    for (auto c : selected_channels(t.channels, image.shape[0])) for (std::size_t i = 0; i < n; ++i) if (mask.get(t.mask_channel * n + i) < 0) image.set(c * n + i, t.fill);
}
inline Parameters sample(const DeepSupervision&, const Sample&, Random&) { return {}; }
inline void apply(const DeepSupervision& t, Sample& s, const Parameters&) {
    auto out = s.at(t.on); require(!t.scales.empty(), "deep supervision requires scales");
    for (auto scale : t.scales) {
        auto shape = out.array.spatial(); if (scale.size() == 1) scale.resize(shape.size(), scale[0]); require(scale.size() == shape.size(), "invalid deep-supervision scale dimensions");
        for (std::size_t a = 0; a < shape.size(); ++a) { require(std::isfinite(scale[a]) && scale[a] > 0 && scale[a] <= 1, "deep-supervision scales must be in (0, 1]"); shape[a] = std::max<std::size_t>(1, std::size_t(round_half_even(shape[a] * scale[a]))); }
        out.pyramid.push_back(resize_array(out.array, shape, true));
    }
    out.source = t.on; s[t.output.empty() ? t.on : t.output] = std::move(out);
}
inline Parameters sample(const ToPseudo2D&, const Sample& s, Random&) {
    require(sample_shape(s).size() == 3, "ToPseudo2D requires 3D input"); Parameters p;
    for (const auto& [name, f] : s) { require(!sparse(f.target.role) && !directional(f.target.role) && f.target.role != Role::CoordinateField, "pseudo-2D supports scalar arrays only"); p.channels.push_back({name, f.array.shape[0], {double(f.array.shape[1])}}); }
    return p;
}
inline void apply(const ToPseudo2D&, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) { auto& f = s.at(c.key); auto& shape = f.array.shape; shape[0] *= shape[1]; shape.erase(shape.begin() + 1); f.source = c.key + "\x1f" + std::to_string(c.channel); }
}
inline Parameters sample(const FromPseudo2D&, const Sample& s, Random&) {
    Parameters p;
    for (const auto& [name, f] : s) {
        auto pos = f.source.find('\x1f'); require(pos != std::string::npos, "FromPseudo2D must follow ToPseudo2D in the same pipeline");
        auto channels = std::stoull(f.source.substr(pos + 1)); require(channels && f.array.shape[0] % channels == 0, "pseudo-2D channel count changed"); p.channels.push_back({name, channels});
    }
    return p;
}
inline void apply(const FromPseudo2D&, Sample& s, const Parameters& p) {
    for (const auto& c : p.channels) { auto& f = s.at(c.key); auto& shape = f.array.shape; auto depth = shape[0] / c.channel; shape[0] = c.channel; shape.insert(shape.begin() + 1, depth); f.source.clear(); }
}
inline Parameters sample(const ConnectedComponents&, const Sample&, Random&) { return {}; }
inline int component_connectivity(const ConnectedComponents& t, std::size_t dim) { return t.connectivity ? t.connectivity : dim == 3 ? 26 : 8; }
inline Parameters sample(const RidgeTrace&, const Sample&, Random&) { return {}; }
template<class T> inline void apply_label_analysis(const T& t, Sample& s) {
    std::vector<std::pair<std::string,Field>> outputs;
    for(const auto& [name,f]:s)if(selected_label(name,f,t.on)) {
        const auto& a=f.array;const auto shape=a.spatial();const auto n=product(shape);
        constexpr bool components=std::is_same_v<T,ConnectedComponents>;
        Field out{Array::uninitialized(a.shape,components?'i':'u',components?4:1),Target{Role::Label},name};
        std::vector<std::uint8_t> mask(n);
        for(std::size_t c=0;c<a.shape[0];++c) {
            a.typed([&](const auto* values){for(std::size_t i=0;i<n;++i)mask[i]=values[c*n+i]!=0;});
            if constexpr(components) vcr::components::label(mask.data(),shape,component_connectivity(t,shape.size()),static_cast<std::int32_t*>(out.array.data())+c*n,1);
            else {
                const auto h=shape[shape.size()-2],w=shape.back();thinning::Scratch scratch;
                for(std::size_t z=0;z<n;z+=h*w) thinning::trace(mask.data()+z,h,w,static_cast<std::uint8_t*>(out.array.data())+c*n+z,nullptr,scratch);
            }
        }
        outputs.emplace_back(name+t.suffix,std::move(out));
    }
    for(auto& [name,f]:outputs){require(!s.count(name),"derived label field already exists");s.emplace(name,std::move(f));}
}
inline void apply(const ConnectedComponents& t,Sample& s,const Parameters&) {apply_label_analysis(t,s);}
inline void apply(const RidgeTrace& t,Sample& s,const Parameters&) {apply_label_analysis(t,s);}
inline Parameters sample(const MedialSurface&, const Sample&, Random&) { return {}; }
inline void apply(const MedialSurface& t, Sample& s, const Parameters&) {
    std::vector<std::pair<std::string, Field>> outputs;
    for (const auto& [name, f] : s) if (selected_label(name, f, t.on)) {
        const auto& a = f.array; auto shape = a.spatial(); auto d = shape.size(); Grid grid(shape); auto footprint = ball(d, 1);
        Field out{Array(a.shape), Target{Role::Scalar}, name}; auto h = shape[d - 2], w = shape[d - 1];
        for (std::size_t c = 0; c < a.shape[0]; ++c) {
            auto values = a.channel(c); for (auto& v : values) if (t.ignore.count(name) && v == t.ignore.at(name)) v = 0;
            std::vector<std::uint8_t> binary(values.size());
            for (std::size_t i=0;i<values.size();++i) binary[i]=values[i]>0;
            thinning::skeleton(binary.data(),binary.data(),h,w,grid.size/(h*w),t.method,t.preserve_border);
            std::vector<double> skeleton(binary.size());
            for (std::size_t i=0;i<binary.size();++i) skeleton[i]=binary[i]!=0;
            if (t.tube) { skeleton = morphology(skeleton, grid, footprint, true); skeleton = morphology(skeleton, grid, footprint, true); }
            if (t.open) skeleton = morphology(skeleton, grid, footprint, Morph::Open, true);
            if (t.close) skeleton = morphology(skeleton, grid, footprint, Morph::Close, true);
            for (std::size_t i = 0; i < grid.size; ++i) skeleton[i] *= values[i]; out.array.channel(c, skeleton);
        }
        outputs.emplace_back(name + t.suffix, std::move(out));
    }
    for (auto& [name, f] : outputs) { require(!s.count(name), "derived skeleton field already exists"); s.emplace(name, std::move(f)); }
}
} // namespace vcr::transforms
