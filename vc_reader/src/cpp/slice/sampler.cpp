// CPU sampling of points, planes, bands and composites from dense blocks.
#include "vc_reader/slice.hpp"
#include "plan.hpp"

#include <cmath>
#include <limits>

namespace vcr::slice {
namespace {
using namespace detail;

CompositeConfig config(const Composite& c)
{
    CompositeConfig out;
    out.method = int(c.method);
    out.isoCutoff = c.isoCutoff;
    out.alphaMin = c.alphaMin;
    out.alphaMax = c.alphaMax;
    out.alphaOpacity = c.alphaOpacity;
    out.alphaCutoff = c.alphaCutoff;
    out.extinction = c.extinction;
    out.emission = c.emission;
    out.ambient = c.ambient;
    return out;
}

// Runs every wave of the request; unit(tile, data, p, sample) handles pixel p, where
// sample(k) is the value of its k-th sample.
template <class Unit>
void run(Array& array, const ElementType& type, const Source& src, Order order, Unit&& unit)
{
    const int o = int(order);
    const auto& shape = array.spec().shape;
    const float fill = float(type.load(array.spec().fill.data()));
    const std::int64_t W = src.width > 0 ? src.width : std::max<std::int64_t>(src.n, 1);
    std::vector<float> buffer;
    plan(src, shape, o, [&](const Wave& wave) {
        readBlocks(array, type, wave, buffer);
        array.parallel_for(wave.tiles.size(), [&](std::size_t i) {
            const Tile& t = wave.tiles[i];
            for (std::int64_t yy = t.y0; yy < t.y0 + t.h; ++yy)
                for (std::int64_t xx = t.x0; xx < t.x0 + t.w; ++xx) {
                    const std::int64_t p = yy * W + xx;
                    if (p >= src.n)
                        continue;
                    unit(p, [&](int k) {
                        float x, y, z;
                        if (!src.point(p, k, x, y, z))
                            return std::numeric_limits<float>::quiet_NaN();
                        if (!t.hasBlock || outside(x, y, z, shape[0], shape[1], shape[2], o))
                            return fill;
                        return sample(buffer.data(), t.block, x, y, z, o);
                    });
                }
        });
    });
}

void check(Order order)
{
    if (order != Order::Nearest && order != Order::Linear && order != Order::Cubic)
        throw std::invalid_argument("interpolation order must be nearest, linear or cubic");
}

void store(Array& array, const ElementType& type, const Source& src, Order order, float* out)
{
    check(order);
    run(array, type, src, order, [&](std::int64_t p, auto&& sample) {
        for (int k = 0; k < src.K; ++k)
            out[std::size_t(k) * std::size_t(src.n) + std::size_t(p)] = sample(k);
    });
}
}  // namespace

void points(Array& array, const ElementType& type, const float* xyz, std::size_t n, Order order, float* out)
{
    Source s;
    s.kind = Source::Points;
    s.xyz = xyz;
    s.n = std::int64_t(n);
    store(array, type, s, order, out);
}

void plane(Array& array, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
           std::array<float, 3> v, std::size_t width, std::size_t height, Order order, float* out)
{
    Source s;
    s.kind = Source::Plane;
    for (int a = 0; a < 3; ++a) {
        s.origin[a] = origin[a];
        s.u[a] = u[a];
        s.v[a] = v[a];
    }
    s.n = std::int64_t(width * height);
    s.width = std::int64_t(width);
    if (width == 0)
        return;
    store(array, type, s, order, out);
}

void band(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
          std::size_t width, std::span<const float> offsets, Order order, float* out)
{
    store(array, type, checkedBand(base, dirs, n, width, offsets.data(), offsets.size()), order, out);
}

void composite(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
               std::size_t width, std::span<const float> offsets, Order order, const Composite& method, float* out)
{
    check(order);
    const auto src = checkedBand(base, dirs, n, width, offsets.data(), offsets.size());
    if (method.method == Composite::Method::Median && offsets.size() > slice::kMaxMedianLayers)
        throw std::invalid_argument("median composites take at most 256 layers");
    const auto c = config(method);
    run(array, type, src, order, [&](std::int64_t p, auto&& sample) {
        Reducer r(c);
        float x, y, z;
        if (!src.point(p, 0, x, y, z)) {  // no surface at this pixel
            out[p] = std::numeric_limits<float>::quiet_NaN();
            return;
        }
        for (int k = 0; k < src.K && !r.done; ++k) {
            const float v = sample(k);
            if (!std::isnan(v))
                r.add(v);
        }
        out[p] = r.result();
    });
}

#ifdef VC_READER_CUDA
namespace cuda {
namespace {
using namespace detail;

device::Storage upload(const void* data, std::size_t bytes, int dev)
{
    auto out = device::allocate(bytes, dev);
    if (bytes)
        device::check(cudaMemcpyAsync(out->ptr, data, bytes, cudaMemcpyHostToDevice, device::stream()));
    return out;
}

// Plans on the host, reads each wave's blocks through the array's caches, and
// samples them on the device. `deviceSource` holds device copies of the inputs.
device::Storage run(Array& array, const ElementType& type, const Source& hostSource, const Source& deviceSource,
                    Order order, bool composite, const CompositeConfig& config, std::size_t outputs)
{
    check(order);
    int dev = 0;
    device::check(cudaGetDevice(&dev));
    auto out = device::allocate(outputs * sizeof(float), dev);
    const auto& shape = array.spec().shape;
    const std::int64_t extent[3] = {shape.size() > 0 ? shape[0] : 0, shape.size() > 1 ? shape[1] : 0,
                                    shape.size() > 2 ? shape[2] : 0};
    const float fill = float(type.load(array.spec().fill.data()));
    const std::int64_t W = hostSource.width > 0 ? hostSource.width : std::max<std::int64_t>(hostSource.n, 1);
    std::vector<float> buffer;
    device::Storage blocks, tiles;
    plan(hostSource, shape, int(order), [&](const Wave& wave) {
        readBlocks(array, type, wave, buffer);
        // The previous wave's copies finish before its buffers are reused.
        device::complete();
        if (!blocks || blocks->bytes < buffer.size() * sizeof(float))
            blocks = device::allocate(buffer.size() * sizeof(float), dev);
        if (!tiles || tiles->bytes < wave.tiles.size() * sizeof(Tile))
            tiles = device::allocate(wave.tiles.size() * sizeof(Tile), dev);
        if (!buffer.empty())
            device::check(cudaMemcpyAsync(blocks->ptr, buffer.data(), buffer.size() * sizeof(float),
                                          cudaMemcpyHostToDevice, device::stream()));
        device::check(cudaMemcpyAsync(tiles->ptr, wave.tiles.data(), wave.tiles.size() * sizeof(Tile),
                                      cudaMemcpyHostToDevice, device::stream()));
        launch(deviceSource, static_cast<const Tile*>(tiles->ptr), wave.tiles.size(),
               static_cast<const float*>(blocks ? blocks->ptr : nullptr), W, extent, int(order), fill, composite, config,
               static_cast<float*>(out->ptr), device::stream());
    });
    device::complete();
    return out;
}

Source bandSources(const float* base, const float* dirs, std::size_t n, std::size_t width,
                   std::span<const float> offsets, Source& deviceSource, std::vector<device::Storage>& keep)
{
    const auto host = checkedBand(base, dirs, n, width, offsets.data(), offsets.size());
    int dev = 0;
    device::check(cudaGetDevice(&dev));
    keep.push_back(upload(base, n * 3 * sizeof(float), dev));
    keep.push_back(upload(dirs, n * 3 * sizeof(float), dev));
    keep.push_back(upload(offsets.data(), offsets.size() * sizeof(float), dev));
    deviceSource = host;
    deviceSource.base = static_cast<const float*>(keep[0]->ptr);
    deviceSource.dirs = static_cast<const float*>(keep[1]->ptr);
    deviceSource.offsets = static_cast<const float*>(keep[2]->ptr);
    return host;
}
}  // namespace

device::Storage points(Array& array, const ElementType& type, const float* xyz, std::size_t n, Order order)
{
    Source host;
    host.kind = Source::Points;
    host.xyz = xyz;
    host.n = std::int64_t(n);
    int dev = 0;
    device::check(cudaGetDevice(&dev));
    const auto coordinates = upload(xyz, n * 3 * sizeof(float), dev);
    Source deviceSource = host;
    deviceSource.xyz = static_cast<const float*>(coordinates->ptr);
    return run(array, type, host, deviceSource, order, false, {}, n);
}

device::Storage plane(Array& array, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
                      std::array<float, 3> v, std::size_t width, std::size_t height, Order order)
{
    Source host;
    host.kind = Source::Plane;
    for (int a = 0; a < 3; ++a) {
        host.origin[a] = origin[a];
        host.u[a] = u[a];
        host.v[a] = v[a];
    }
    host.n = std::int64_t(width * height);
    host.width = std::int64_t(width);
    if (width == 0) {
        int dev = 0;
        device::check(cudaGetDevice(&dev));
        return device::allocate(0, dev);
    }
    return run(array, type, host, host, order, false, {}, width * height);
}

device::Storage band(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                     std::size_t width, std::span<const float> offsets, Order order)
{
    Source deviceSource;
    std::vector<device::Storage> keep;
    const auto host = bandSources(base, dirs, n, width, offsets, deviceSource, keep);
    return run(array, type, host, deviceSource, order, false, {}, n * offsets.size());
}

device::Storage composite(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                          std::size_t width, std::span<const float> offsets, Order order, const Composite& method)
{
    if (method.method == Composite::Method::Median && offsets.size() > slice::kMaxMedianLayers)
        throw std::invalid_argument("median composites take at most 256 layers");
    Source deviceSource;
    std::vector<device::Storage> keep;
    const auto host = bandSources(base, dirs, n, width, offsets, deviceSource, keep);
    return run(array, type, host, deviceSource, order, true, config(method), n);
}
}  // namespace cuda
#endif

}  // namespace vcr::slice
