#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

class PyramidChunkedArray final : public vc::render::IChunkedArray {
public:
    PyramidChunkedArray(vc::render::ChunkStatus level0Status,
                        uint8_t level0Value,
                        vc::render::ChunkStatus level1Status,
                        uint8_t level1Value)
        : statuses_{level0Status, level1Status}
        , values_{level0Value, level1Value}
    {
    }

    int numLevels() const override { return 2; }

    std::array<int, 3> shape(int level) const override
    {
        return level == 0 ? std::array<int, 3>{4, 4, 4}
                          : std::array<int, 3>{2, 2, 2};
    }

    std::array<int, 3> chunkShape(int level) const override
    {
        return shape(level);
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
        vc::render::ChunkResult result;
        result.dtype = vc::render::ChunkDtype::UInt8;
        if (level < 0 || level >= numLevels() || iz != 0 || iy != 0 || ix != 0) {
            result.status = vc::render::ChunkStatus::MissQueued;
            return result;
        }

        result.status = statuses_[level];
        result.shape = shape(level);
        if (result.status == vc::render::ChunkStatus::Data) {
            const auto dims = shape(level);
            auto bytes = std::make_shared<std::vector<std::byte>>(
                std::size_t(dims[0]) * std::size_t(dims[1]) * std::size_t(dims[2]),
                std::byte{values_[level]});
            result.bytes = std::move(bytes);
        }
        return result;
    }

    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
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
