#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class PyramidChunkedArray final : public vc::render::IChunkedArray {
public:
    PyramidChunkedArray(vc::render::ChunkStatus level0Status,
                        uint8_t level0Value,
                        vc::render::ChunkStatus level1Status,
                        uint8_t level1Value,
                        std::array<int, 3> level0Shape = {4, 4, 4},
                        std::array<int, 3> level1Shape = {2, 2, 2},
                        std::array<int, 3> level0ChunkShape = {0, 0, 0},
                        std::array<int, 3> level1ChunkShape = {0, 0, 0})
        : statuses_{level0Status, level1Status}
        , values_{level0Value, level1Value}
        , shapes_{level0Shape, level1Shape}
        , chunkShapes_{
              level0ChunkShape[0] > 0 ? level0ChunkShape : level0Shape,
              level1ChunkShape[0] > 0 ? level1ChunkShape : level1Shape}
    {
    }

    int numLevels() const override { return 2; }

    std::array<int, 3> shape(int level) const override
    {
        return shapes_[level];
    }

    std::array<int, 3> chunkShape(int level) const override
    {
        return chunkShapes_[level];
    }

    vc::render::ChunkDtype dtype() const override
    {
        return vc::render::ChunkDtype::UInt8;
    }

    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int level) const override
    {
        LevelTransform transform;
        if (level == 1)
            transform.scaleFromLevel0 = {0.5, 0.5, 0.5};
        return transform;
    }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        if (level >= 0 && level < numLevels())
            ++queuedLookups[level];
        return chunkResult(level, iz, iy, ix);
    }

    vc::render::ChunkResult getChunkIfCached(
        int level, int iz, int iy, int ix) override
    {
        if (level >= 0 && level < numLevels())
            ++cachedLookups[level];
        return chunkResult(level, iz, iy, ix);
    }

    std::array<int, 2> queuedLookups{};
    std::array<int, 2> cachedLookups{};

private:
    vc::render::ChunkResult chunkResult(int level, int iz, int iy, int ix)
    {
        vc::render::ChunkResult result;
        result.dtype = vc::render::ChunkDtype::UInt8;
        if (level < 0 || level >= numLevels() || iz != 0 || iy != 0 || ix != 0) {
            result.status = vc::render::ChunkStatus::MissQueued;
            return result;
        }

        result.status = statuses_[level];
        result.shape = shape(level);
        if (result.status == vc::render::ChunkStatus::Data) {
            const auto dims = chunkShape(level);
            auto bytes = std::make_shared<std::vector<std::byte>>(
                std::size_t(dims[0]) * std::size_t(dims[1]) * std::size_t(dims[2]),
                std::byte{values_[level]});
            result.bytes = std::move(bytes);
        }
        return result;
    }

public:
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return chunkResult(level, iz, iy, ix);
    }

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override
    {
        return 0;
    }

    void removeChunkReadyListener(ChunkReadyCallbackId) override {}

private:
    std::array<vc::render::ChunkStatus, 2> statuses_;
    std::array<uint8_t, 2> values_;
    std::array<std::array<int, 3>, 2> shapes_;
    std::array<std::array<int, 3>, 2> chunkShapes_;
};

cv::Mat_<cv::Vec3f> singleCoord(const cv::Vec3f& coord)
{
    cv::Mat_<cv::Vec3f> coords(1, 1);
    coords(0, 0) = coord;
    return coords;
}

} // namespace

TEST_CASE("ChunkedPlaneSampler fine-to-coarse fills missing high-res from coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Missing, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse fills queued high-res from coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse keeps high-res value when present")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
}

TEST_CASE("ChunkedPlaneSampler fine-to-coarse skips empty high-res scale")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::AllFill, 0,
                              vc::render::ChunkStatus::Data, 42,
                              {0, 0, 0}, {2, 2, 2});
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler coarse-to-fine lets ready high-res overwrite coarse preview")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
}

TEST_CASE("ChunkedPlaneSampler coarse-to-fine keeps coarse preview when high-res scale is empty")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::AllFill, 0,
                              vc::render::ChunkStatus::Data, 42,
                              {0, 0, 0}, {2, 2, 2});
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsCoarseToFine(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("ChunkedPlaneSampler fallback leaves sentinel surface coords uncovered")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({0.0f, 0.0f, 0.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 0);
    CHECK(out(0, 0) == 0);
}

TEST_CASE("ChunkedPlaneSampler base and overlay buffers fall back independently")
{
    PyramidChunkedArray baseArray(vc::render::ChunkStatus::Missing, 0,
                                  vc::render::ChunkStatus::Data, 11);
    PyramidChunkedArray overlayArray(vc::render::ChunkStatus::Data, 99,
                                     vc::render::ChunkStatus::Data, 22);
    cv::Mat_<uint8_t> baseOut(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> baseCoverage(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> overlayOut(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> overlayCoverage(1, 1, uint8_t(0));
    const cv::Mat_<cv::Vec3f> coords = singleCoord({1.0f, 1.0f, 1.0f});

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        baseArray, 0, coords, baseOut, baseCoverage, {vc::Sampling::Nearest, 1});
    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        overlayArray, 0, coords, overlayOut, overlayCoverage, {vc::Sampling::Nearest, 1});

    CHECK(baseCoverage(0, 0) == 1);
    CHECK(baseOut(0, 0) == 11);
    CHECK(overlayCoverage(0, 0) == 1);
    CHECK(overlayOut(0, 0) == 99);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level does not fall back to coarse level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Missing, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(123));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 0);
    CHECK(stats.requestedLevelOnly);
    CHECK(stats.fallbackLevels == 0);
    CHECK(stats.requestedChunks == 1);
    CHECK(stats.missingChunks == 1);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level keeps requested-level data")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::Data, 7,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
        {vc::Sampling::Nearest, 1});

    CHECK(coverage(0, 0) == 1);
    CHECK(out(0, 0) == 7);
    CHECK(stats.requestedLevelOnly);
    CHECK(stats.fallbackLevels == 0);
    CHECK(stats.missingChunks == 0);
}

TEST_CASE("ChunkedPlaneSampler blocking requested-level rejects unresolved chunks")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::Data, 42);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));

    CHECK_THROWS_AS(
        vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage,
            {vc::Sampling::Nearest, 1}),
        std::runtime_error);
}

TEST_CASE("ChunkedPlaneSampler grouped corner batch shares geometry across arrays")
{
    PyramidChunkedArray first(vc::render::ChunkStatus::Data, 17,
                              vc::render::ChunkStatus::Data, 0);
    PyramidChunkedArray second(vc::render::ChunkStatus::Data, 93,
                               vc::render::ChunkStatus::Data, 0);
    std::vector<vc::render::IChunkedArray*> arrays{&first, &second};
    std::vector<std::vector<std::array<uint8_t, 8>>> values;
    std::vector<cv::Vec3f> fractions;
    std::vector<uint8_t> valid;

    const auto stats =
        vc::render::ChunkedPlaneSampler::sampleTrilinearCornersLevelBlockingRequestedLevel(
            arrays, 0, {{1.25f, 1.5f, 1.75f}}, values, fractions, valid);

    REQUIRE(values.size() == 2);
    REQUIRE(values[0].size() == 1);
    CHECK(std::all_of(values[0][0].begin(), values[0][0].end(),
                      [](uint8_t value) { return value == 17; }));
    CHECK(std::all_of(values[1][0].begin(), values[1][0].end(),
                      [](uint8_t value) { return value == 93; }));
    REQUIRE(fractions.size() == 1);
    CHECK(fractions[0][0] == doctest::Approx(0.25f));
    CHECK(fractions[0][1] == doctest::Approx(0.5f));
    CHECK(fractions[0][2] == doctest::Approx(0.75f));
    CHECK(valid == std::vector<uint8_t>{1});
    CHECK(stats.requestedChunks == 2);
}

TEST_CASE("ChunkedPlaneSampler grouped corner batch supports mixed chunk grids")
{
    PyramidChunkedArray coarseChunks(
        vc::render::ChunkStatus::Data, 17,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {4, 4, 4});
    PyramidChunkedArray fineChunks(
        vc::render::ChunkStatus::Data, 93,
        vc::render::ChunkStatus::Data, 0,
        {4, 4, 4}, {2, 2, 2}, {2, 2, 2});
    std::vector<vc::render::IChunkedArray*> arrays{&coarseChunks, &fineChunks};
    std::vector<std::vector<std::array<uint8_t, 8>>> values;
    std::vector<cv::Vec3f> fractions;
    std::vector<uint8_t> valid;

    const auto stats =
        vc::render::ChunkedPlaneSampler::sampleTrilinearCornersLevelBlockingRequestedLevel(
            arrays, 0, {{0.25f, 0.5f, 0.75f}}, values, fractions, valid);

    CHECK(valid == std::vector<uint8_t>{1});
    CHECK(std::all_of(values[0][0].begin(), values[0][0].end(),
                      [](uint8_t value) { return value == 17; }));
    CHECK(std::all_of(values[1][0].begin(), values[1][0].end(),
                      [](uint8_t value) { return value == 93; }));
    CHECK(stats.requestedChunks == 2);
}

TEST_CASE("ChunkedPlaneSampler can use fallback without queueing or promoting it")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::MissQueued, 0);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));
    vc::render::ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 1);
    options.queuedFallbackLevels = 0;

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage, options);

    CHECK(array.queuedLookups[0] > 0);
    CHECK(array.queuedLookups[1] == 0);
    CHECK(array.cachedLookups[1] > 0);
}

TEST_CASE("ChunkedPlaneSampler can bound transition work to one fallback level")
{
    PyramidChunkedArray array(vc::render::ChunkStatus::MissQueued, 0,
                              vc::render::ChunkStatus::MissQueued, 0);
    cv::Mat_<uint8_t> out(1, 1, uint8_t(0));
    cv::Mat_<uint8_t> coverage(1, 1, uint8_t(0));
    vc::render::ChunkedPlaneSampler::Options options(vc::Sampling::Nearest, 1);
    options.queuedFallbackLevels = 1;

    vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        array, 0, singleCoord({1.0f, 1.0f, 1.0f}), out, coverage, options);

    CHECK(array.queuedLookups[0] > 0);
    CHECK(array.queuedLookups[1] > 0);
    CHECK(array.cachedLookups[1] == 0);
}
