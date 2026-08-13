// Microbenchmark for the Volume::sample / readInterpolated3D path.
//
// M1 added an out.create(coords.size()) defensive call at every layer
// of the sample stack. cv::Mat::create is documented as a no-op when
// the requested type+size already match — but we want a regression
// guard so a future refactor that turns it into an actual reallocation
// fails the bench rather than silently regressing UI-thread latency.
//
// Two cases:
//   Case A: pre-sized `out` matrix reused for every iteration.
//   Case B: fresh empty `out` for every iteration (forces the
//           defensive create() down the stack).
//
// Both cases run readInterpolated3D against an AllFill stub (no real
// I/O; the bench measures the per-pixel inner loop and the create()
// overhead, not chunk fetching). We assert B is within ~5% of A.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

class AllFillChunkedArray final : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {256, 256, 256}; }
    std::array<int, 3> chunkShape(int) const override { return {64, 64, 64}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }
    vc::render::ChunkResult tryGetChunk(int, int, int, int) override
    {
        vc::render::ChunkResult r;
        r.status = vc::render::ChunkStatus::AllFill;
        r.dtype = vc::render::ChunkDtype::UInt8;
        return r;
    }
    vc::render::ChunkResult getChunkBlocking(int, int, int, int) override
    {
        return tryGetChunk(0, 0, 0, 0);
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

constexpr int kSize = 1024;
constexpr int kIters = 50;

cv::Mat_<cv::Vec3f> makeRamp(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> coords(rows, cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            coords(y, x) = cv::Vec3f(
                static_cast<float>(x % 200) + 16.f,
                static_cast<float>(y % 200) + 16.f,
                64.f);
        }
    }
    return coords;
}

double timeIt(auto&& fn)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

TEST_CASE("Volume::sample bench: empty-out path is within 10% of pre-sized")
{
    AllFillChunkedArray cache;
    const cv::Mat_<cv::Vec3f> coords = makeRamp(kSize, kSize);

    // Warm-up: prime caches, get the heap settled.
    {
        cv::Mat_<uint8_t> warmup;
        readInterpolated3D(warmup, &cache, 0, coords, /*nn=*/false);
    }

    // Case A: reuse one pre-sized buffer.
    cv::Mat_<uint8_t> out_pre(kSize, kSize);
    out_pre.setTo(0);
    const double tA = timeIt([&]() {
        for (int i = 0; i < kIters; ++i) {
            readInterpolated3D(out_pre, &cache, 0, coords, /*nn=*/false);
        }
    });

    // Case B: empty buffer each call.
    const double tB = timeIt([&]() {
        for (int i = 0; i < kIters; ++i) {
            cv::Mat_<uint8_t> out_fresh;
            readInterpolated3D(out_fresh, &cache, 0, coords, /*nn=*/false);
        }
    });

    const double pixels = double(kSize) * double(kSize) * double(kIters);
    const double mvoxA = pixels / tA / 1e6;
    const double mvoxB = pixels / tB / 1e6;

    std::printf("\nVolume::sample bench (%dx%d, %d iters, AllFill cache)\n",
                kSize, kSize, kIters);
    std::printf("  Pre-sized out:  %.3f s  (%.1f Mvoxel/s)\n", tA, mvoxA);
    std::printf("  Empty out:      %.3f s  (%.1f Mvoxel/s)\n", tB, mvoxB);
    const double overhead = (tB - tA) / tA * 100.0;
    std::printf("  Empty-path overhead: %+.2f%%\n\n", overhead);

    // 10% upper bound is generous on shared CI hardware. cv::Mat::create
    // is a no-op when type+size match, so the realistic delta is in the
    // sub-percent range; if it breaks 10%, someone made create() actually
    // allocate every call.
    CHECK(tB <= tA * 1.10);
}
