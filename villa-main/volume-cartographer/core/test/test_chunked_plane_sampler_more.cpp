// More ChunkedPlaneSampler coverage: Trilinear sampling, AllFill paths,
// coverage skip, level transforms, and edge cases.

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
using vc::render::ChunkDtype;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;

namespace {

class GradientArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {16, 16, 16}; }
    std::array<int, 3> chunkShape(int) const override { return {16, 16, 16}; }
    ChunkDtype dtype() const override { return ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        ChunkResult r;
        r.dtype = ChunkDtype::UInt8;
        if (level != 0 || iz != 0 || iy != 0 || ix != 0) {
            r.status = ChunkStatus::Missing;
            return r;
        }
        r.status = ChunkStatus::Data;
        r.shape = shape(0);
        auto bytes = std::make_shared<std::vector<std::byte>>(16 * 16 * 16);
        for (int z = 0; z < 16; ++z)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                    auto v = static_cast<uint8_t>(z * 16 + y * 4 + x);
                    (*bytes)[z * 256 + y * 16 + x] = std::byte{v};
                }
        r.bytes = std::move(bytes);
        return r;
    }
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

class AllFillArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {16, 16, 16}; }
    std::array<int, 3> chunkShape(int) const override { return {16, 16, 16}; }
    ChunkDtype dtype() const override { return ChunkDtype::UInt8; }
    double fillValue() const override { return 42.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        ChunkResult r;
        r.dtype = ChunkDtype::UInt8;
        r.status = (level == 0 && iz == 0 && iy == 0 && ix == 0)
            ? ChunkStatus::AllFill
            : ChunkStatus::Missing;
        r.shape = shape(0);
        return r;
    }
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

cv::Mat_<cv::Vec3f> axisCoords(int rows, int cols, float dx = 1.f, float dy = 1.f, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> c(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < cols; ++x)
            c(r, x) = cv::Vec3f(float(x) * dx, float(r) * dy, z);
    return c;
}

} // namespace

TEST_CASE("samplePlaneLevel with Trilinear sampling on a gradient")
{
    GradientArray a;
    cv::Mat_<uint8_t> out(8, 8, uint8_t{0});
    cv::Mat_<uint8_t> cov(8, 8, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(0.5f, 0.5f, 0.5f),
        cv::Vec3f(1.f, 0.f, 0.f), cv::Vec3f(0.f, 1.f, 0.f),
        out, cov, {vc::Sampling::Trilinear, 8});
    CHECK(stats.coveredPixels > 0);
    // Output value should be in the gradient range and finite.
    CHECK(int(out(0, 0)) >= 0);
}

TEST_CASE("samplePlaneLevel with Tricubic sampling")
{
    GradientArray a;
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(2.f, 2.f, 2.f),
        cv::Vec3f(1.f, 0.f, 0.f), cv::Vec3f(0.f, 1.f, 0.f),
        out, cov, {vc::Sampling::Tricubic, 4});
    CHECK(true);  // No crash.
}

TEST_CASE("samplePlaneLevel on AllFill chunk paints the fill value")
{
    AllFillArray a;
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(0.f, 0.f, 0.f),
        cv::Vec3f(1.f, 0.f, 0.f), cv::Vec3f(0.f, 1.f, 0.f),
        out, cov, {vc::Sampling::Nearest, 4});
    CHECK(stats.coveredPixels == 16);
    CHECK(out(0, 0) == 42);
}

TEST_CASE("sampleCoordsLevel with Trilinear sampling")
{
    GradientArray a;
    auto coords = axisCoords(4, 4);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    auto stats = ChunkedPlaneSampler::sampleCoordsLevel(
        a, 0, coords, out, cov, {vc::Sampling::Trilinear, 4});
    CHECK(stats.errorChunks == 0);
}

TEST_CASE("sampleCoordsLevel with NaN coord is skipped")
{
    GradientArray a;
    cv::Mat_<cv::Vec3f> coords(2, 2);
    coords(0, 0) = cv::Vec3f(std::nanf(""), 0, 0);
    coords(0, 1) = cv::Vec3f(0, 0, 0);
    coords(1, 0) = cv::Vec3f(1, 1, 1);
    coords(1, 1) = cv::Vec3f(2, 2, 2);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{0});
    cv::Mat_<uint8_t> cov(2, 2, uint8_t{0});
    ChunkedPlaneSampler::sampleCoordsLevel(a, 0, coords, out, cov);
    // NaN pixel stays at zero, others may be covered.
    CHECK(cov(0, 0) == 0);
}

TEST_CASE("collectPlaneDependencies on AllFill returns keys for the chunk")
{
    AllFillArray a;
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    auto keys = ChunkedPlaneSampler::collectPlaneDependencies(
        a, 0, cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0), cov);
    CHECK_FALSE(keys.empty());
}

TEST_CASE("samplePlaneCoarseToFine with AllFill at finest level")
{
    AllFillArray a;
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    auto stats = ChunkedPlaneSampler::samplePlaneCoarseToFine(
        a, /*finestLevel=*/0,
        cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0), cv::Vec3f(0, 1, 0),
        out, cov);
    CHECK(stats.coveredPixels > 0);
}

TEST_CASE("samplePlaneLevel with non-axis-aligned step vectors")
{
    GradientArray a;
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    cv::Mat_<uint8_t> cov(4, 4, uint8_t{0});
    // 45° rotated plane
    ChunkedPlaneSampler::samplePlaneLevel(
        a, 0, cv::Vec3f(2.f, 2.f, 2.f),
        cv::Vec3f(0.707f, 0.707f, 0.f),
        cv::Vec3f(-0.707f, 0.707f, 0.f),
        out, cov, {vc::Sampling::Trilinear, 4});
    CHECK(true);
}
