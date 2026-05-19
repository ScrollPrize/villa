// Perf guard for the ChunkedPlaneSampler hot path. Loose assertion;
// the printed Mvoxel/s numbers are the signal when comparing branches.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using vc::render::ChunkDtype;
using vc::render::ChunkedPlaneSampler;
using vc::render::ChunkKey;
using vc::render::ChunkKeyHash;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkedArray;

// Fake array that synthesizes deterministic chunk data on first access and
// retains it for its lifetime -- the same residency contract the real
// ChunkCache provides across frames, minus the decode cost.
class DataChunkedArray final : public IChunkedArray {
public:
    DataChunkedArray(std::array<int, 3> shape, std::array<int, 3> chunkShape)
        : shape_(shape)
        , chunkShape_(chunkShape)
    {
    }

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return shape_; }
    std::array<int, 3> chunkShape(int) const override { return chunkShape_; }
    ChunkDtype dtype() const override { return ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        const ChunkKey key{level, iz, iy, ix};
        auto it = chunks_.find(key);
        if (it == chunks_.end()) {
            const std::size_t n = std::size_t(chunkShape_[0]) *
                                  std::size_t(chunkShape_[1]) *
                                  std::size_t(chunkShape_[2]);
            auto bytes = std::make_shared<std::vector<std::byte>>(n);
            // Cheap deterministic fill; the exact pattern is irrelevant, we
            // only need real memory the interpolator can read.
            uint8_t seed = uint8_t((iz * 73 + iy * 19 + ix * 7) & 0xFF);
            for (std::size_t i = 0; i < n; ++i)
                (*bytes)[i] = std::byte(uint8_t(seed + uint8_t(i)));
            ChunkResult r;
            r.status = ChunkStatus::Data;
            r.dtype = ChunkDtype::UInt8;
            r.shape = chunkShape_;
            r.bytes = std::move(bytes);
            it = chunks_.emplace(key, std::move(r)).first;
        }
        return it->second;
    }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

private:
    std::array<int, 3> shape_;
    std::array<int, 3> chunkShape_;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks_;
};

constexpr int kSize = 1024;
constexpr int kFrames = 30;
constexpr int kReps = 9;  // best-of-N; min is the least-interfered run

double timeIt(auto&& fn)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

struct ScenarioResult {
    double minSeconds = 0.0;
    double medianSeconds = 0.0;
    long long covered = 0;
};

// Drives `kFrames` independent sampler calls (each rebuilds its own
// LocalChunkCache, like a real repaint). originStep/scaleGrowth let a
// scenario model panning vs. zooming.
ScenarioResult runScenario(DataChunkedArray& array,
                            vc::Sampling sampling,
                            float originStep,
                            float scaleGrowth)
{
    cv::Mat_<uint8_t> out(kSize, kSize);
    cv::Mat_<uint8_t> coverage(kSize, kSize);
    ChunkedPlaneSampler::Options opts(sampling, 32);

    auto oneRep = [&]() {
        long long covered = 0;
        const double s = timeIt([&]() {
            for (int f = 0; f < kFrames; ++f) {
                out.setTo(0);
                coverage.setTo(0);
                const float base = 64.0f + originStep * float(f);
                const float scale = 1.0f + scaleGrowth * float(f);
                const cv::Vec3f origin(base, base, 128.0f);
                const cv::Vec3f vxStep(scale, 0.0f, 0.0f);
                const cv::Vec3f vyStep(0.0f, scale, 0.0f);
                const auto stats = ChunkedPlaneSampler::samplePlaneLevel(
                    array, 0, origin, vxStep, vyStep, out, coverage, opts);
                covered += stats.coveredPixels;
            }
        });
        return std::pair<double, long long>{s, covered};
    };

    std::vector<double> times;
    times.reserve(kReps);
    long long covered = 0;
    for (int r = 0; r < kReps; ++r) {
        auto [s, c] = oneRep();
        times.push_back(s);
        covered = c;  // deterministic across reps
    }
    std::sort(times.begin(), times.end());

    ScenarioResult res;
    res.minSeconds = times.front();
    res.medianSeconds = times[times.size() / 2];
    res.covered = covered;
    return res;
}

void report(const char* name, const ScenarioResult& r)
{
    const double pixels = double(kSize) * double(kSize) * double(kFrames);
    const double mvoxMin = pixels / r.minSeconds / 1e6;
    const double spread = (r.medianSeconds - r.minSeconds) / r.minSeconds * 100.0;
    std::printf("  %-22s min %7.3f s  %8.1f Mvoxel/s  (median +%.1f%%, %lld covered)\n",
                name, r.minSeconds, mvoxMin, spread, r.covered);
}

}  // namespace

TEST_CASE("ChunkedPlaneSampler bench: data-chunk hot path")
{
    // 1024^3 logical volume, 128^3 chunks -> adjacent pixels share chunks,
    // which is exactly the LocalChunkCache reuse pattern the optimization
    // targets.
    DataChunkedArray array({1024, 1024, 1024}, {128, 128, 128});

    // Warm-up: materialize the chunks the scenarios will touch and let the
    // heap settle so the timed runs measure steady state.
    {
        cv::Mat_<uint8_t> out(kSize, kSize), cov(kSize, kSize);
        out.setTo(0);
        cov.setTo(0);
        ChunkedPlaneSampler::samplePlaneLevel(
            array, 0, {64, 64, 128}, {1, 0, 0}, {0, 1, 0}, out, cov,
            ChunkedPlaneSampler::Options(vc::Sampling::Trilinear, 32));
    }

    const ScenarioResult triPan =
        runScenario(array, vc::Sampling::Trilinear, /*originStep=*/3.0f, 0.0f);
    const ScenarioResult triZoom =
        runScenario(array, vc::Sampling::Trilinear, /*originStep=*/1.0f, 0.02f);
    const ScenarioResult nnPan =
        runScenario(array, vc::Sampling::Nearest, /*originStep=*/3.0f, 0.0f);

    std::printf("\nChunkedPlaneSampler bench (%dx%d, %d frames, Data chunks)\n",
                kSize, kSize, kFrames);
    report("trilinear pan", triPan);
    report("trilinear zoom", triZoom);
    report("nearest pan", nnPan);
    std::printf("\n");

    // Sanity: the scenarios must actually sample (a no-op regression that
    // covers nothing would otherwise post a meaningless huge Mvoxel/s).
    CHECK(triPan.covered > 0);
    CHECK(triZoom.covered > 0);
    CHECK(nnPan.covered > 0);

    // Loose floor: gross-regression trip wire, not a tight perf assert.
    const double pixels = double(kSize) * double(kSize) * double(kFrames);
    CHECK(pixels / triPan.minSeconds / 1e6 > 10.0);
    CHECK(pixels / nnPan.minSeconds / 1e6 > 10.0);
}
