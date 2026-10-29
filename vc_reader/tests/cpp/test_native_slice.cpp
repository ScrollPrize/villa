// Point, plane, band and composite sampling: tiled block reads agree with a direct
// evaluation over the whole array, interpolation reproduces linear fields, edges
// read the fill value, nearest half ties round up, and composites reduce as
// documented. With CUDA, the device backend matches the CPU.
#include "vc_reader/slice.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/native/slice/math.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
namespace slice = vcr::slice;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

constexpr int Z = 37, Y = 41, X = 45;
constexpr float kFill = 3.0f;

float field(int z, int y, int x) { return float((z * 7 + y * 13 + x * 3) % 251); }

// The whole array padded by 4 voxels of fill: a block every in-range sample fits.
struct Reference {
    std::vector<float> data;
    vcr::slice::detail::Block block{0, -4, -4, -4, Z + 8, Y + 8, X + 8};
    Reference()
    {
        data.assign(std::size_t(Z + 8) * (Y + 8) * (X + 8), kFill);
        for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    data[(std::size_t(z + 4) * (Y + 8) + (y + 4)) * (X + 8) + (x + 4)] = field(z, y, x);
    }
    float operator()(float x, float y, float z, int order) const
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            return NAN;
        if (vcr::slice::detail::outside(x, y, z, Z, Y, X, order))
            return kFill;
        return vcr::slice::detail::sample(data.data(), block, x, y, z, order);
    }
};

bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || std::abs(a - b) <= 1e-3f * (1 + std::abs(b)); }

void check_close(const std::vector<float>& got, const std::vector<float>& want, const std::string& what)
{
    require(got.size() == want.size(), what + ": size");
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!same(got[i], want[i]))
            throw std::runtime_error(what + ": element " + std::to_string(i) + " is " + std::to_string(got[i]) +
                                     ", expected " + std::to_string(want[i]));
}

struct Case {
    vcr::Array* array;
    vcr::ElementType type;
};

void points(Case c, const Reference& ref)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-6.0f, 52.0f);
    std::vector<float> xyz;
    for (int i = 0; i < 20000; ++i)
        for (int a = 0; a < 3; ++a)
            xyz.push_back(d(rng));
    xyz.insert(xyz.end(), {NAN, 1, 1, 1e30f, 2, 2, -0.5f, 0.5f, 1.5f, 44.5f, 40.49f, 36.5f});
    const std::size_t n = xyz.size() / 3;
    for (const auto order : {slice::Order::Nearest, slice::Order::Linear, slice::Order::Cubic}) {
        std::vector<float> got(n), want(n);
        slice::points(*c.array, c.type, xyz.data(), n, order, got.data());
        for (std::size_t p = 0; p < n; ++p)
            want[p] = ref(xyz[3 * p], xyz[3 * p + 1], xyz[3 * p + 2], int(order));
        check_close(got, want, "scattered points, order " + std::to_string(int(order)));
    }
    // Linear and cubic reproduce a linear field away from the edges.
    const float x = 4.75f, y = 3.5f, z = 2.25f;  // the field does not wrap around this neighbourhood
    for (const auto order : {slice::Order::Linear, slice::Order::Cubic}) {
        float v;
        const float p[3] = {x, y, z};
        slice::points(*c.array, c.type, p, 1, order, &v);
        require(std::abs(v - (z * 7 + y * 13 + x * 3)) < 1e-3f, "interpolation reproduces a linear field");
    }
}

void plane(Case c, const Reference& ref)
{
    const std::array<float, 3> origin{-3.5f, 2.25f, 30.0f}, u{0.37f, 0.11f, -0.2f}, v{0.05f, 0.29f, 0.07f};
    const std::size_t w = 150, h = 133;
    for (const auto order : {slice::Order::Nearest, slice::Order::Linear, slice::Order::Cubic}) {
        std::vector<float> got(w * h), want(w * h);
        slice::plane(*c.array, c.type, origin, u, v, w, h, order, got.data());
        // Coordinates come from the sampler's own Source so that nearest ties agree.
        vcr::slice::detail::Source src;
        src.kind = vcr::slice::detail::Source::Plane;
        for (int a = 0; a < 3; ++a) {
            src.origin[a] = origin[a];
            src.u[a] = u[a];
            src.v[a] = v[a];
        }
        src.n = std::int64_t(w * h);
        src.width = std::int64_t(w);
        for (std::size_t p = 0; p < w * h; ++p) {
            float x, y, z;
            src.point(std::int64_t(p), 0, x, y, z);
            want[p] = ref(x, y, z, int(order));
        }
        check_close(got, want, "oblique plane, order " + std::to_string(int(order)));
    }
}

// Exact half-voxel ties round up, checked against fixed indices rather than the
// shared formula. A plane at half-voxel steps gives every voxel two pixels.
struct Ties {
    std::vector<float> xyz{0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, -0.5f, 0.0f, 0.0f,
                           X - 0.5f, 0.0f, 0.0f, 10.49f, 10.5f, 10.51f};
    std::vector<float> points{field(3, 2, 1), field(6, 5, 4), field(0, 0, 0), kFill, field(11, 11, 10)};
    static constexpr std::size_t w = 2 * X - 2, h = 20;  // the last column is x = X - 1.5
    std::array<float, 3> origin{0, 0, 7}, u{0.5f, 0, 0}, v{0, 0.5f, 0};
    std::vector<float> plane;
    Ties()
    {
        for (std::size_t j = 0; j < h; ++j)
            for (std::size_t i = 0; i < w; ++i)
                plane.push_back(field(7, int(j + 1) / 2, int(i + 1) / 2));
    }
};

void nearest_ties(Case c)
{
    const Ties t;
    std::vector<float> got(t.points.size());
    slice::points(*c.array, c.type, t.xyz.data(), got.size(), slice::Order::Nearest, got.data());
    check_close(got, t.points, "nearest half ties");
    got.resize(t.w * t.h);
    slice::plane(*c.array, c.type, t.origin, t.u, t.v, t.w, t.h, slice::Order::Nearest, got.data());
    check_close(got, t.plane, "nearest 2x plane");
}

struct Band {
    std::vector<float> base, dirs, offsets;
    std::size_t n, width;
};

Band band_inputs()
{
    Band b;
    b.width = 61;
    b.n = b.width * 47;
    for (std::size_t p = 0; p < b.n; ++p) {
        const float col = float(p % b.width), row = float(p / b.width);
        const bool hole = p % 97 == 0;  // pixels without a surface
        b.base.insert(b.base.end(), {hole ? NAN : -2.0f + col * 0.8f, -1.0f + row * 0.9f, 18.0f + 0.1f * col});
        const float len = std::sqrt(0.3f * 0.3f + 0.2f * 0.2f + 1.0f);
        b.dirs.insert(b.dirs.end(), {0.3f / len, -0.2f / len, 1.0f / len});
    }
    for (int k = -6; k <= 9; ++k)
        b.offsets.push_back(float(k) * 1.5f);
    return b;
}

void band(Case c, const Reference& ref)
{
    const auto b = band_inputs();
    const std::size_t K = b.offsets.size();
    for (const auto order : {slice::Order::Nearest, slice::Order::Linear, slice::Order::Cubic}) {
        std::vector<float> got(K * b.n), want(K * b.n);
        slice::band(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets, order, got.data());
        vcr::slice::detail::Source src;
        src.kind = vcr::slice::detail::Source::Band;
        src.base = b.base.data();
        src.dirs = b.dirs.data();
        src.offsets = b.offsets.data();
        src.K = int(K);
        src.n = std::int64_t(b.n);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t p = 0; p < b.n; ++p) {
                float x, y, z;
                src.point(std::int64_t(p), int(k), x, y, z);
                want[k * b.n + p] = ref(x, y, z, int(order));
            }
        check_close(got, want, "band, order " + std::to_string(int(order)));
    }
}

void composite(Case c)
{
    const auto b = band_inputs();
    const std::size_t K = b.offsets.size();
    std::vector<float> layers(K * b.n);
    slice::band(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets, slice::Order::Linear,
                layers.data());
    using M = slice::Composite::Method;
    for (const auto method : {M::Mean, M::Max, M::Min, M::Median, M::MinAbs, M::Alpha, M::BeerLambert}) {
        slice::Composite cfg;
        cfg.method = method;
        cfg.isoCutoff = 40.0f;
        cfg.alphaMin = 0.2f;
        cfg.alphaMax = 0.9f;
        cfg.alphaOpacity = 0.6f;
        std::vector<float> got(b.n), want(b.n);
        slice::composite(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets,
                         slice::Order::Linear, cfg, got.data());
        vcr::slice::detail::CompositeConfig dc;
        dc.method = int(method);
        dc.isoCutoff = cfg.isoCutoff;
        dc.alphaMin = cfg.alphaMin;
        dc.alphaMax = cfg.alphaMax;
        dc.alphaOpacity = cfg.alphaOpacity;
        for (std::size_t p = 0; p < b.n; ++p) {
            if (std::isnan(b.base[3 * p])) {
                want[p] = NAN;
                continue;
            }
            vcr::slice::detail::Reducer r(dc);
            for (std::size_t k = 0; k < K && !r.done; ++k)
                r.add(layers[k * b.n + p]);
            want[p] = r.result();
        }
        check_close(got, want, "composite " + std::to_string(int(method)));
    }
    // Reduction semantics on a known stack.
    vcr::slice::detail::CompositeConfig dc;
    dc.isoCutoff = 10;
    const float stack[] = {5, 50, 20, 200, 90};
    auto reduce = [&](int method) {
        dc.method = method;
        vcr::slice::detail::Reducer r(dc);
        for (const float v : stack)
            r.add(v);
        return r.result();
    };
    using namespace vcr::slice::detail;
    require(reduce(Mean) == 90.0f && reduce(Max) == 200.0f && reduce(Min) == 20.0f && reduce(Median) == 90.0f &&
            reduce(MinAbs) == 90.0f, "known stack reductions (the value below the cutoff is dropped)");
    dc.isoCutoff = 1000;
    require(std::isnan(reduce(Mean)), "no surviving layer is NaN");
}

#ifdef VC_READER_CUDA
std::vector<float> download(const vcr::device::Storage& s)
{
    std::vector<float> out(s->bytes / sizeof(float));
    vcr::device::check(cudaMemcpy(out.data(), s->ptr, s->bytes, cudaMemcpyDeviceToHost));
    return out;
}

void cuda_matches(Case c)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device: CUDA comparison skipped\n";
        return;
    }
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-6.0f, 52.0f);
    std::vector<float> xyz;
    for (int i = 0; i < 5000; ++i)
        for (int a = 0; a < 3; ++a)
            xyz.push_back(d(rng));
    const auto b = band_inputs();
    for (const auto order : {slice::Order::Nearest, slice::Order::Linear, slice::Order::Cubic}) {
        std::vector<float> cpu(xyz.size() / 3);
        slice::points(*c.array, c.type, xyz.data(), cpu.size(), order, cpu.data());
        check_close(download(slice::cuda::points(*c.array, c.type, xyz.data(), cpu.size(), order)), cpu,
                    "CUDA points, order " + std::to_string(int(order)));
        std::vector<float> plane(90 * 70);
        slice::plane(*c.array, c.type, {1, 2, 3}, {0.4f, 0.1f, 0.2f}, {-0.1f, 0.5f, 0.05f}, 90, 70, order, plane.data());
        check_close(download(slice::cuda::plane(*c.array, c.type, {1, 2, 3}, {0.4f, 0.1f, 0.2f}, {-0.1f, 0.5f, 0.05f},
                                                90, 70, order)),
                    plane, "CUDA plane");
        std::vector<float> layers(b.offsets.size() * b.n);
        slice::band(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets, order, layers.data());
        check_close(download(slice::cuda::band(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets,
                                               order)),
                    layers, "CUDA band");
    }
    const Ties t;
    check_close(download(slice::cuda::points(*c.array, c.type, t.xyz.data(), t.points.size(), slice::Order::Nearest)),
                t.points, "CUDA nearest half ties");
    check_close(download(slice::cuda::plane(*c.array, c.type, t.origin, t.u, t.v, t.w, t.h, slice::Order::Nearest)),
                t.plane, "CUDA nearest 2x plane");
    using M = slice::Composite::Method;
    for (const auto method : {M::Mean, M::Max, M::Min, M::Median, M::MinAbs, M::Alpha, M::BeerLambert}) {
        slice::Composite cfg;
        cfg.method = method;
        cfg.isoCutoff = 30.0f;
        std::vector<float> cpu(b.n);
        slice::composite(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width, b.offsets, slice::Order::Cubic,
                         cfg, cpu.data());
        check_close(download(slice::cuda::composite(*c.array, c.type, b.base.data(), b.dirs.data(), b.n, b.width,
                                                    b.offsets, slice::Order::Cubic, cfg)),
                    cpu, "CUDA composite " + std::to_string(int(method)));
    }
    std::cout << "CUDA matches CPU\n";
}
#endif
}  // namespace

int main()
{
    const auto root = fs::temp_directory_path() /
                      ("vc-reader-slice-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};
    try {
        // uint16 in 8^3 chunks with one absent chunk region (fill) inside the array.
        vcr::zarr::CreateOptions create;
        create.shape = {Z, Y, X};
        create.chunks = {8, 8, 8};
        create.dtype = "<u2";
        create.compressor = "zstd";
        create.fillValue = kFill;
        auto writer = vcr::zarr::Writer::create(root / "v.zarr" / "0", create);
        std::vector<uint16_t> all(std::size_t(Z) * Y * X);
        for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    all[(std::size_t(z) * Y + y) * X + x] = uint16_t(field(z, y, x));
        writer.writeRegion({0, 0, 0}, {Z, Y, X}, reinterpret_cast<const uint8_t*>(all.data()));
        vcr::zarr::OpenOptions options;
        options.memoryCache = std::make_shared<vcr::MemoryCache>(64 << 20);
        auto opened = vcr::zarr::openArray((root / "v.zarr" / "0").string(), options);
        const Reference ref;
        const Case c{opened.array.get(), opened.meta.type};
        points(c, ref);
        plane(c, ref);
        nearest_ties(c);
        band(c, ref);
        composite(c);
#ifdef VC_READER_CUDA
        cuda_matches(c);
#endif
        std::cout << "points, planes, bands and composites passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
