// The CUDA sampler behind vc_render_tifxyz --gpu (apps/src/RenderCuda.cpp) against the CPU
// samplers it transcribes (core/src/Slicing.cpp), on a synthetic chunked array: the samples must
// agree byte for byte, including outside the volume, in missing chunks, on non-finite geometry,
// for uint16 rounding, for every scalar composite reducer, and when the band's chunks do not fit
// the device pool. Every case is a soft skip on a machine without a CUDA device or NVRTC.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "RenderCuda.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using vc::render::ChunkDtype;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkedArray;
using vc::render::cuda::GpuSampler;

// One level of pseudo-random chunks generated on request; chunks can be declared all fill,
// missing or failing.
class SyntheticArray final : public IChunkedArray {
public:
    SyntheticArray(std::array<int, 3> shape, std::array<int, 3> chunk, ChunkDtype dtype)
        : shape_(shape), chunk_(chunk), dtype_(dtype)
    {
    }

    void setStatus(int iz, int iy, int ix, ChunkStatus status) { special_[{iz, iy, ix}] = status; }

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return shape_; }
    std::array<int, 3> chunkShape(int) const override { return chunk_; }
    ChunkDtype dtype() const override { return dtype_; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override { return getChunkBlocking(level, iz, iy, ix); }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        ChunkResult r;
        r.dtype = dtype_;
        r.shape = chunk_;
        const int nz = (shape_[0] + chunk_[0] - 1) / chunk_[0];
        const int ny = (shape_[1] + chunk_[1] - 1) / chunk_[1];
        const int nx = (shape_[2] + chunk_[2] - 1) / chunk_[2];
        if (level != 0 || iz < 0 || iy < 0 || ix < 0 || iz >= nz || iy >= ny || ix >= nx) {
            r.status = ChunkStatus::Missing;
            return r;
        }
        fetches++;
        if (auto it = special_.find({iz, iy, ix}); it != special_.end()) {
            r.status = it->second;
            if (r.status == ChunkStatus::Error) r.error = "synthetic fetch failure";
            return r;
        }
        r.status = ChunkStatus::Data;
        r.bytes = bytesOf(iz, iy, ix);
        return r;
    }

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    std::size_t fetches = 0;

private:
    std::shared_ptr<const std::vector<std::byte>> bytesOf(int iz, int iy, int ix) const
    {
        const std::size_t n = std::size_t(chunk_[0]) * std::size_t(chunk_[1]) * std::size_t(chunk_[2])
            * (dtype_ == ChunkDtype::UInt16 ? 2 : 1);
        auto v = std::make_shared<std::vector<std::byte>>(n);
        std::uint32_t h = 2166136261u ^ std::uint32_t((iz * 73 + iy) * 91 + ix);
        for (std::size_t j = 0; j < n; j++) {
            h ^= std::uint32_t(j);
            h *= 16777619u;
            (*v)[j] = std::byte(h >> 13);
        }
        return v;
    }

    std::array<int, 3> shape_;
    std::array<int, 3> chunk_;
    ChunkDtype dtype_;
    std::map<std::tuple<int, int, int>, ChunkStatus> special_;
};

struct Geometry {
    cv::Mat_<cv::Vec3f> base;
    cv::Mat_<cv::Vec3f> dirs;
};

// Random base points over the volume and a margin around it (samples outside read 0), random
// unit directions, a few non-finite bases and directions.
Geometry randomGeometry(int h, int w, const std::array<int, 3>& shape, unsigned seed, float margin = 6.f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(-margin, float(shape[2]) + margin);
    std::uniform_real_distribution<float> uy(-margin, float(shape[1]) + margin);
    std::uniform_real_distribution<float> uz(-margin, float(shape[0]) + margin);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> coin(0.f, 1.f);
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    Geometry g{cv::Mat_<cv::Vec3f>(h, w), cv::Mat_<cv::Vec3f>(h, w)};
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            cv::Vec3f d(nd(rng), nd(rng), nd(rng));
            const float len = std::sqrt(d.dot(d));
            d = len > 1e-3f ? d * (1.0f / len) : cv::Vec3f(0.f, 0.f, 1.f);
            g.base(r, c) = cv::Vec3f(ux(rng), uy(rng), uz(rng));
            g.dirs(r, c) = d;
            const float p = coin(rng);
            if (p < 0.03f) g.base(r, c) = cv::Vec3f(qnan, qnan, qnan);
            else if (p < 0.05f) g.dirs(r, c) = cv::Vec3f(qnan, qnan, qnan);
        }
    }
    return g;
}

std::vector<float> layerOffsets(int n, double step)
{
    std::vector<float> out;
    const double center = 0.5 * (n - 1.0);
    for (int i = 0; i < n; i++) out.push_back(float((i - center) * step));
    return out;
}

template <typename T>
std::size_t differing(const std::vector<cv::Mat_<T>>& a, const std::vector<cv::Mat_<T>>& b, std::string& first)
{
    REQUIRE(a.size() == b.size());
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
        REQUIRE(a[i].size() == b[i].size());
        for (int r = 0; r < a[i].rows; r++)
            for (int c = 0; c < a[i].cols; c++)
                if (a[i](r, c) != b[i](r, c)) {
                    if (n == 0)
                        first = "slice " + std::to_string(i) + " (" + std::to_string(r) + ", " + std::to_string(c)
                            + "): cpu " + std::to_string(int(a[i](r, c))) + " gpu " + std::to_string(int(b[i](r, c)));
                    n++;
                }
    }
    return n;
}

template <typename T>
std::size_t nonzero(const std::vector<cv::Mat_<T>>& a)
{
    std::size_t n = 0;
    for (const auto& m : a)
        for (int r = 0; r < m.rows; r++)
            for (int c = 0; c < m.cols; c++) n += m(r, c) != 0;
    return n;
}

std::unique_ptr<GpuSampler> openOrSkip(IChunkedArray& array, std::size_t poolBytes = 0)
{
    std::string why;
    auto gpu = GpuSampler::open(array, 0, poolBytes, nullptr, why);
    if (!gpu) MESSAGE("Skipping: " << why);
    return gpu;
}

template <typename T>
void checkLayers(SyntheticArray& array, const Geometry& g, const std::vector<float>& offsets, GpuSampler& gpu)
{
    std::vector<cv::Mat_<T>> cpu, dev;
    readMultiSlice(cpu, &array, 0, g.base, g.dirs, offsets);
    gpu.sampleSlices(g.base, g.dirs, offsets, dev);
    std::string first;
    const std::size_t diff = differing(cpu, dev, first);
    CHECK_MESSAGE(diff == 0, diff << " samples differ, first " << first);
    CHECK(nonzero(cpu) > cpu.size() * std::size_t(g.base.rows) * std::size_t(g.base.cols) / 4);
}

}  // namespace

TEST_CASE("GPU layer samples equal readMultiSlice (uint8)")
{
    SyntheticArray array({90, 70, 100}, {32, 24, 40}, ChunkDtype::UInt8);
    array.setStatus(1, 1, 1, ChunkStatus::AllFill);
    array.setStatus(2, 0, 2, ChunkStatus::Missing);
    auto gpu = openOrSkip(array);
    if (!gpu) return;
    MESSAGE("device: " << gpu->device());
    const Geometry g = randomGeometry(37, 53, array.shape(0), 1);
    checkLayers<std::uint8_t>(array, g, layerOffsets(21, 1.0), *gpu);
    checkLayers<std::uint8_t>(array, g, layerOffsets(11, 0.7), *gpu);
    // the same band again on the same pool: every chunk is resident, nothing is fetched
    const std::size_t fetchesBefore = array.fetches;
    std::vector<cv::Mat_<std::uint8_t>> again;
    gpu->sampleSlices(g.base, g.dirs, layerOffsets(21, 1.0), again);
    CHECK(array.fetches == fetchesBefore);
    CHECK(gpu->stats().bands == 3);
    CHECK(gpu->stats().evictions == 0);
}

TEST_CASE("GPU layer samples equal readMultiSlice (uint16: rounded, clamped at 65535)")
{
    SyntheticArray array({64, 48, 80}, {16, 24, 20}, ChunkDtype::UInt16);
    array.setStatus(0, 1, 2, ChunkStatus::AllFill);
    auto gpu = openOrSkip(array);
    if (!gpu) return;
    const Geometry g = randomGeometry(29, 61, array.shape(0), 2);
    checkLayers<std::uint16_t>(array, g, layerOffsets(9, 1.0), *gpu);
    checkLayers<std::uint16_t>(array, g, layerOffsets(6, 1.3), *gpu);
}

TEST_CASE("GPU composites equal readCompositeFast for max, min, mean and median")
{
    SyntheticArray array({90, 70, 100}, {32, 24, 40}, ChunkDtype::UInt8);
    array.setStatus(1, 1, 1, ChunkStatus::AllFill);
    auto gpu = openOrSkip(array);
    if (!gpu) return;
    const Geometry g = randomGeometry(41, 47, array.shape(0), 3);
    for (const char* method : {"max", "min", "mean", "median"}) {
        for (const int cutoff : {0, 37}) {
            for (const float zStep : {1.0f, 0.75f}) {
                CAPTURE(method);
                CAPTURE(cutoff);
                CAPTURE(zStep);
                CompositeParams params;
                params.method = method;
                params.isoCutoff = std::uint8_t(cutoff);
                // a sentinel shows which pixels each path leaves alone
                cv::Mat_<std::uint8_t> cpu(g.base.rows, g.base.cols, std::uint8_t{77});
                cv::Mat_<std::uint8_t> dev = cpu.clone();
                readCompositeFast(cpu, &array, 0, g.base, g.dirs, zStep, -6, 6, params, vc::Sampling::Nearest);
                gpu->composite(g.base, g.dirs, zStep, -6, 6, params, dev);
                std::string first;
                const std::size_t diff = differing(std::vector<cv::Mat_<std::uint8_t>>{cpu},
                                                   std::vector<cv::Mat_<std::uint8_t>>{dev}, first);
                CHECK_MESSAGE(diff == 0, diff << " pixels differ, first " << first);
                std::size_t touched = 0;
                for (int r = 0; r < cpu.rows; r++)
                    for (int c = 0; c < cpu.cols; c++) touched += cpu(r, c) != 77;
                CHECK(touched > std::size_t(cpu.rows) * std::size_t(cpu.cols) / 2);
            }
        }
    }
}

TEST_CASE("a band whose chunks exceed the pool is split and the pool recycled")
{
    // 4096 chunks of 4 KB; the smallest pool the sampler accepts holds 64 of them
    SyntheticArray array({256, 256, 256}, {16, 16, 16}, ChunkDtype::UInt8);
    auto gpu = openOrSkip(array, 64 * 4096);
    if (!gpu) return;
    CHECK(gpu->poolChunks() == 64);
    const Geometry g = randomGeometry(1, 3000, array.shape(0), 4, 2.f);
    checkLayers<std::uint8_t>(array, g, layerOffsets(21, 1.0), *gpu);
    CHECK(gpu->stats().passes > 1);
    const Geometry g2 = randomGeometry(2, 1500, array.shape(0), 5, 2.f);
    checkLayers<std::uint8_t>(array, g2, layerOffsets(21, 1.0), *gpu);
    CHECK(gpu->stats().evictions > 0);
    MESSAGE(gpu->summary());
}

TEST_CASE("composite reducers the GPU does not take are reported")
{
    CompositeParams params;
    std::string why;
    for (const char* method : {"max", "min", "mean", "median"}) {
        params.method = method;
        CHECK(GpuSampler::compositeSupported(params, 13, &why));
    }
    for (const char* method : {"alpha", "beerLambert", "minabs"}) {
        params.method = method;
        CHECK_FALSE(GpuSampler::compositeSupported(params, 13, &why));
        CHECK(why.find(method) != std::string::npos);
    }
    params.method = "median";
    CHECK_FALSE(GpuSampler::compositeSupported(params, 1000, &why));
}

TEST_CASE("a failed chunk fetch surfaces as an exception, as on the CPU")
{
    SyntheticArray array({64, 64, 64}, {32, 32, 32}, ChunkDtype::UInt8);
    array.setStatus(0, 0, 0, ChunkStatus::Error);
    auto gpu = openOrSkip(array);
    if (!gpu) return;
    cv::Mat_<cv::Vec3f> base(3, 3, cv::Vec3f(5.f, 6.f, 7.f)), dirs(3, 3, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<cv::Mat_<std::uint8_t>> out;
    CHECK_THROWS_WITH_AS(gpu->sampleSlices(base, dirs, layerOffsets(3, 1.0), out),
                         doctest::Contains("synthetic fetch failure"), std::runtime_error);
}
