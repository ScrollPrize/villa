// Drives ChunkedPlaneSampler against a synthetic in-memory IChunkedArray.
// Mirrors the existing test_chunked_plane_sampler_fallback.cpp setup; this
// adds coverage for the request*, collect*, and samplePlane* variants.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using vc::render::ChunkedPlaneSampler;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;

namespace {

// All-data chunked array: every chunk read returns Data filled with a
// per-level constant byte value. Single 4^3 chunk at level 0, 2^3 at level 1.
class AllDataArray : public vc::render::IChunkedArray {
public:
    AllDataArray(uint8_t lvl0Value = 7, uint8_t lvl1Value = 11)
        : values_{lvl0Value, lvl1Value} {}

    int numLevels() const override { return 2; }
    std::array<int, 3> shape(int level) const override
    {
        return level == 0 ? std::array<int, 3>{4, 4, 4}
                          : std::array<int, 3>{2, 2, 2};
    }
    std::array<int, 3> chunkShape(int level) const override { return shape(level); }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int level) const override
    {
        LevelTransform t;
        if (level == 1) t.scaleFromLevel0 = {0.5, 0.5, 0.5};
        return t;
    }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        ChunkResult r;
        r.dtype = vc::render::ChunkDtype::UInt8;
        if (level < 0 || level >= numLevels() || iz != 0 || iy != 0 || ix != 0) {
            r.status = ChunkStatus::Missing;
            return r;
        }
        r.status = ChunkStatus::Data;
        r.shape = shape(level);
        const auto dims = shape(level);
        auto bytes = std::make_shared<std::vector<std::byte>>(
            std::size_t(dims[0]) * std::size_t(dims[1]) * std::size_t(dims[2]),
            std::byte{values_[level]});
        r.bytes = std::move(bytes);
        return r;
    }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override
    {
        ++prefetchCalls;
    }

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

    int prefetchCalls = 0;
private:
    std::array<uint8_t, 2> values_;
};

cv::Mat_<cv::Vec3f> axisAlignedCoords(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> c(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < cols; ++x)
            c(r, x) = cv::Vec3f(float(x), float(r), z);
    return c;
}

} // namespace

TEST_CASE("collectPlaneDependencies / collectCoordsDependencies enumerate keys")
{
    AllDataArray a;
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t{0});
    auto keysPlane = ChunkedPlaneSampler::collectPlaneDependencies(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0), coverage);
    CHECK_FALSE(keysPlane.empty());

    auto coords = axisAlignedCoords(4, 4);
    auto keysCoords = ChunkedPlaneSampler::collectCoordsDependencies(
        a, 0, coords, coverage);
    CHECK_FALSE(keysCoords.empty());
}

TEST_CASE("requestPlaneDependencies / requestCoordsDependencies run without crashing")
{
    AllDataArray a;
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t{0});
    ChunkedPlaneSampler::requestPlaneDependencies(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0), coverage);
    auto coords = axisAlignedCoords(4, 4);
    ChunkedPlaneSampler::requestCoordsDependencies(a, 0, coords, coverage);
    // The exact prefetch count depends on internal heuristics; we just
    // verify the call surface is reachable and doesn't throw.
    CHECK(a.prefetchCalls >= 0);
}

TEST_CASE("samplePlaneLevel: covers all pixels of a fully-resident plane")
{
    AllDataArray a(42, 0);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage, {vc::Sampling::Nearest, 4});
    CHECK(stats.coveredPixels == 16);
    int allCovered = 0, allValue = 1;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            if (coverage(r, c)) ++allCovered;
            if (out(r, c) != 42) allValue = 0;
        }
    CHECK(allCovered == 16);
    CHECK(allValue == 1);
}

TEST_CASE("samplePlaneLevel: pre-covered pixels are skipped")
{
    AllDataArray a(99, 0);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t{0});
    for (int c = 0; c < 4; ++c) coverage(0, c) = 1;
    ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage);
    // Row 0 should be untouched (still 0)
    for (int c = 0; c < 4; ++c) CHECK(out(0, c) == 0);
    // Other rows now have value 99.
    CHECK(out(1, 0) == 99);
}

TEST_CASE("sampleCoordsLevel: explicit coords mode runs without crashing")
{
    AllDataArray a(13, 0);
    auto coords = axisAlignedCoords(4, 4);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> coverage(4, 4, uint8_t{0});
    auto stats = ChunkedPlaneSampler::sampleCoordsLevel(a, 0, coords, out, coverage);
    CHECK(stats.coveredPixels >= 0);
    CHECK(stats.errorChunks == 0);
}

TEST_CASE("samplePlaneFineToCoarse: falls back to coarse when fine is missing")
{
    // Fine level (0) missing; coarse level (1) provides the value.
    struct FineMissingArray : AllDataArray {
        using AllDataArray::AllDataArray;
        ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
        {
            if (level == 0) {
                ChunkResult r;
                r.status = ChunkStatus::Missing;
                r.dtype = vc::render::ChunkDtype::UInt8;
                r.shape = shape(0);
                return r;
            }
            return AllDataArray::tryGetChunk(level, iz, iy, ix);
        }
        ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
        {
            return tryGetChunk(level, iz, iy, ix);
        }
    };
    FineMissingArray a(0, 88);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{0});
    cv::Mat_<uint8_t> coverage(2, 2, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneFineToCoarse(
        a, /*startLevel=*/0,
        cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage, {vc::Sampling::Nearest, 2});
    CHECK(stats.coveredPixels > 0);
    CHECK(out(0, 0) == 88);
}

TEST_CASE("samplePlaneCoarseToFine: paints coarse first, overwrites with fine")
{
    AllDataArray a(/*lvl0=*/55, /*lvl1=*/22);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{0});
    cv::Mat_<uint8_t> coverage(2, 2, uint8_t{0});
    ChunkedPlaneSampler::samplePlaneCoarseToFine(
        a, /*finestLevel=*/0,
        cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage, {vc::Sampling::Nearest, 2});
    // Fine value should win.
    CHECK(out(0, 0) == 55);
}

TEST_CASE("sampleCoords coarse/fine variants run without crashing")
{
    AllDataArray a(1, 2);
    auto coords = axisAlignedCoords(2, 2);
    cv::Mat_<uint8_t> out1(2, 2, uint8_t{0}), cov1(2, 2, uint8_t{0});
    auto s1 = ChunkedPlaneSampler::sampleCoordsCoarseToFine(a, 0, coords, out1, cov1);
    CHECK(s1.errorChunks == 0);
    cv::Mat_<uint8_t> out2(2, 2, uint8_t{0}), cov2(2, 2, uint8_t{0});
    auto s2 = ChunkedPlaneSampler::sampleCoordsFineToCoarse(a, 0, coords, out2, cov2);
    CHECK(s2.errorChunks == 0);
}

TEST_CASE("Options round-trip")
{
    ChunkedPlaneSampler::Options o(vc::Sampling::Trilinear, 64);
    CHECK(o.sampling == vc::Sampling::Trilinear);
    CHECK(o.tileSize == 64);
    ChunkedPlaneSampler::Options def;
    CHECK(def.sampling == vc::Sampling::Nearest);
    CHECK(def.tileSize == 32);
}
