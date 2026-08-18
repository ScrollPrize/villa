// Drives ChunkedPlaneSampler against a synthetic in-memory IChunkedArray.
// Mirrors the existing test_chunked_plane_sampler_fallback.cpp setup; this
// adds coverage for the request*, collect*, and samplePlane* variants.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

class UniformStatusArray : public vc::render::IChunkedArray {
public:
    explicit UniformStatusArray(ChunkStatus status)
        : status_(status) {}

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {4, 4, 4}; }
    std::array<int, 3> chunkShape(int) const override { return {4, 4, 4}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int, int, int, int) override
    {
        ChunkResult r;
        r.dtype = vc::render::ChunkDtype::UInt8;
        r.shape = shape(0);
        r.status = status_;
        return r;
    }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

private:
    ChunkStatus status_;
};

class PyramidArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 8; }
    std::array<int, 3> shape(int level) const override
    {
        const int extent = std::max(1, 4096 >> level);
        return {extent, extent, extent};
    }
    std::array<int, 3> chunkShape(int) const override { return {32, 32, 32}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int level) const override
    {
        LevelTransform transform;
        const double scale = std::ldexp(1.0, -level);
        transform.scaleFromLevel0 = {scale, scale, scale};
        return transform;
    }
    ChunkResult tryGetChunk(int level, int, int, int) override
    {
        ChunkResult result;
        result.status = ChunkStatus::MissQueued;
        result.dtype = dtype();
        result.shape = chunkShape(level);
        return result;
    }
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
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

TEST_CASE("collectViewportDependencies is resident-only and keeps distant occurrences")
{
    class UnresolvedArray final : public UniformStatusArray {
    public:
        UnresolvedArray() : UniformStatusArray(ChunkStatus::MissQueued) {}

        ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
        {
            ++queuedReads;
            return UniformStatusArray::tryGetChunk(level, iz, iy, ix);
        }

        ChunkResult getChunkIfCached(int, int, int, int) override
        {
            ChunkResult result;
            result.status = ChunkStatus::MissQueued;
            result.dtype = vc::render::ChunkDtype::UInt8;
            result.shape = shape(0);
            return result;
        }

        int queuedReads = 0;
    } array;

    const std::vector<cv::Vec3f> coords{
        {1.0f, 1.0f, 1.0f}, {1.1f, 1.0f, 1.0f}, {1.2f, 1.0f, 1.0f}};
    const std::vector<std::array<float, 2>> viewport{
        {4.0f, 4.0f}, {8.0f, 4.0f}, {40.0f, 4.0f}};
    ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 8);
    options.queuedFallbackLevels = 0;
    const auto samples = ChunkedPlaneSampler::collectViewportDependencies(
        array, 0, coords, viewport, 1.0f, options);

    CHECK(array.queuedReads == 0);
    REQUIRE(samples.size() == 2);
    CHECK(samples[0].key == samples[1].key);
}

TEST_CASE("viewport dependency dedup uses projected chunk footprint, not sample spacing")
{
    class LargeChunkArray final : public UniformStatusArray {
    public:
        LargeChunkArray() : UniformStatusArray(ChunkStatus::MissQueued) {}
        std::array<int, 3> shape(int) const override { return {64, 64, 64}; }
        std::array<int, 3> chunkShape(int) const override { return {32, 32, 32}; }
    } array;

    const std::vector<cv::Vec3f> coords{
        {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};
    const std::vector<std::array<float, 2>> viewport{
        {0.0f, 0.0f}, {16.0f, 0.0f}, {40.0f, 0.0f}};
    ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 8);
    options.queuedFallbackLevels = 0;
    const auto samples = ChunkedPlaneSampler::collectViewportDependencies(
        array, 0, coords, viewport, 1.0f, options);

    REQUIRE(samples.size() == 2);
    CHECK(samples[0].key == samples[1].key);
    CHECK_THROWS_AS(
        ChunkedPlaneSampler::collectViewportDependencies(
            array, 0, coords, viewport, 0.0f, options),
        std::invalid_argument);
}

TEST_CASE("representative chunk extent honors anisotropic declared transforms")
{
    class AnisotropicArray final : public UniformStatusArray {
    public:
        AnisotropicArray() : UniformStatusArray(ChunkStatus::MissQueued) {}
        std::array<int, 3> shape(int) const override { return {64, 64, 64}; }
        std::array<int, 3> chunkShape(int) const override { return {8, 16, 32}; }
        LevelTransform levelTransform(int) const override
        {
            LevelTransform transform;
            transform.scaleFromLevel0 = {0.5, 0.25, 0.125};
            return transform;
        }
    } array;

    CHECK(ChunkedPlaneSampler::representativeChunkExtentBaseVoxels(array, 0) ==
          doctest::Approx(112.0));
}

TEST_CASE("viewport fallback range stops at coverage or five levels")
{
    PyramidArray array;
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 0, 300, 100, 1.0f) == 4);
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 0, 100, 300, 1.0f) == 4);
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 0, 4096, 4096, 1.0f) == 5);
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 0, 20, 20, 1.0f) == 0);
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 6, 4096, 4096, 1.0f) == 1);
}

TEST_CASE("parameterized viewport fallback uses the bounded full range")
{
    PyramidArray array;
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 0, 300, 100, std::nullopt) == 5);
    CHECK(ChunkedPlaneSampler::fallbackLevelCountForViewport(
              array, 6, 300, 100, std::nullopt) == 1);
}

TEST_CASE("viewport dependencies publish coarse levels first")
{
    PyramidArray array;
    ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 8);
    options.queuedFallbackLevels = 3;
    const auto samples = ChunkedPlaneSampler::collectViewportDependencies(
        array, 0, {{64.0f, 64.0f, 64.0f}}, {{4.0f, 4.0f}}, 1.0f,
        options);
    REQUIRE(samples.size() == 4);
    CHECK(samples[0].key.level == 3);
    CHECK(samples[0].relativeLevel == 3);
    CHECK(samples[1].key.level == 2);
    CHECK(samples[1].relativeLevel == 2);
    CHECK(samples[2].key.level == 1);
    CHECK(samples[2].relativeLevel == 1);
    CHECK(samples[3].key.level == 0);
    CHECK(samples[3].relativeLevel == 0);
}

TEST_CASE("compact chunk pixel lookup preserves levels transforms and source")
{
    PyramidArray array;
    cv::Mat_<cv::Vec3f> coords(2, 3);
    coords(0, 0) = {1.0f, 1.0f, 1.0f};
    coords(0, 1) = {33.0f, 1.0f, 1.0f};
    coords(0, 2) = {34.0f, 1.0f, 1.0f};
    coords(1, 0) = {-1.0f, -1.0f, -1.0f};
    coords(1, 1) = {0.0f, 0.0f, 0.0f};
    coords(1, 2) = {4097.0f, 1.0f, 1.0f};
    const vc::render::VolumeSourceId source{73};

    const auto lookup = ChunkedPlaneSampler::buildChunkPixelLookup(
        array, source, 0, 1, coords, vc::Sampling::Nearest);

    REQUIRE(lookup.size() == 2);
    CHECK(lookup[0].level == 0);
    REQUIRE(lookup[0].chunks.size() == 2);
    CHECK(lookup[0].pixelIds(0, 0) == 1);
    CHECK(lookup[0].pixelIds(0, 1) == 2);
    CHECK(lookup[0].pixelIds(0, 2) == 2);
    CHECK(lookup[0].pixelIds(1, 0) == 0);
    CHECK(lookup[0].pixelIds(1, 1) == 0);
    CHECK(lookup[0].pixelIds(1, 2) == 0);
    CHECK(lookup[0].chunks[0] == ChunkKey{0, 0, 0, 0, source});
    CHECK(lookup[0].chunks[1] == ChunkKey{0, 0, 0, 1, source});

    // Level 1 is half-resolution, so both x=33 and x=34 remain in chunk 0.
    CHECK(lookup[1].level == 1);
    REQUIRE(lookup[1].chunks.size() == 1);
    CHECK(lookup[1].pixelIds(0, 0) == 1);
    CHECK(lookup[1].pixelIds(0, 1) == 1);
    CHECK(lookup[1].pixelIds(0, 2) == 1);
    CHECK(lookup[1].chunks[0] == ChunkKey{1, 0, 0, 0, source});
    CHECK_FALSE(lookup[0].overflowed);
    CHECK_FALSE(lookup[1].overflowed);

    const auto planeLookup = ChunkedPlaneSampler::buildChunkPixelLookup(
        array, source, 0, 0, coords, vc::Sampling::Nearest,
        /*zeroIsSentinel=*/false);
    REQUIRE(planeLookup.size() == 1);
    CHECK(planeLookup[0].pixelIds(1, 1) != 0);
    CHECK(planeLookup[0].chunks[
        static_cast<std::size_t>(planeLookup[0].pixelIds(1, 1) - 1)] ==
        ChunkKey{0, 0, 0, 0, source});
}

TEST_CASE("compact chunk pixel lookup reports uint16 ID overflow")
{
    class ManyChunksArray final : public PyramidArray {
    public:
        int numLevels() const override { return 1; }
        std::array<int, 3> shape(int) const override { return {1, 1, 70000}; }
        std::array<int, 3> chunkShape(int) const override { return {1, 1, 1}; }
        LevelTransform levelTransform(int) const override { return {}; }
    } array;

    constexpr int distinctChunks = 65536;
    cv::Mat_<cv::Vec3f> coords(1, distinctChunks);
    for (int x = 0; x < distinctChunks; ++x)
        coords(0, x) = {static_cast<float>(x + 1), 0.0f, 0.0f};

    const auto lookup = ChunkedPlaneSampler::buildChunkPixelLookup(
        array, vc::render::VolumeSourceId{91}, 0, 0, coords,
        vc::Sampling::Nearest, /*zeroIsSentinel=*/false);

    REQUIRE(lookup.size() == 1);
    CHECK(lookup[0].overflowed);
    CHECK(lookup[0].chunks.size() == 65535);
    CHECK(lookup[0].pixelIds(0, distinctChunks - 2) == 65535);
    CHECK(lookup[0].pixelIds(0, distinctChunks - 1) == 0);
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

TEST_CASE("samplePlaneFineToCoarse: known-missing chunks become covered black")
{
    UniformStatusArray a(ChunkStatus::Missing);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{123});
    cv::Mat_<uint8_t> coverage(2, 2, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneFineToCoarse(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage, {vc::Sampling::Nearest, 2});
    CHECK(stats.coveredPixels == 4);
    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 0);
}

TEST_CASE("samplePlaneFineToCoarse: queued chunks remain uncovered")
{
    UniformStatusArray a(ChunkStatus::MissQueued);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{123});
    cv::Mat_<uint8_t> coverage(2, 2, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneFineToCoarse(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, coverage, {vc::Sampling::Nearest, 2});
    CHECK(stats.coveredPixels == 0);
    CHECK(coverage(0, 0) == 0);
    CHECK(out(0, 0) == 123);
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
