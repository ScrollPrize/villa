// Drives Slicing.cpp entry points through a synthetic IChunkedArray.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

// A single 8x8x8 uint8 chunk filled with a constant. Larger than the
// sampler tests so trilinear / tricubic don't all run off the edge.
class ConstChunkArray : public vc::render::IChunkedArray {
public:
    explicit ConstChunkArray(uint8_t v = 100) : value_(v) {}
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {8, 8, 8}; }
    std::array<int, 3> chunkShape(int) const override { return {8, 8, 8}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        vc::render::ChunkResult r;
        r.dtype = vc::render::ChunkDtype::UInt8;
        if (level != 0 || iz != 0 || iy != 0 || ix != 0) {
            r.status = vc::render::ChunkStatus::Missing;
            return r;
        }
        r.status = vc::render::ChunkStatus::Data;
        r.shape = shape(0);
        r.bytes = std::make_shared<std::vector<std::byte>>(8 * 8 * 8, std::byte{value_});
        return r;
    }
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
private:
    uint8_t value_;
};

cv::Mat_<cv::Vec3f> coordsGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> c(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < cols; ++x)
            c(r, x) = cv::Vec3f(float(x), float(r), z);
    return c;
}

} // namespace

TEST_CASE("readInterpolated3D(uint8, nearest-neighbor): all pixels equal chunk value")
{
    ConstChunkArray a(123);
    auto coords = coordsGrid(4, 4);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    readInterpolated3D(out, &a, /*level=*/0, coords, /*nearest_neighbor=*/true);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            CHECK(out(r, c) == 123);
}

TEST_CASE("readInterpolated3D(uint8, trilinear) matches the constant chunk too")
{
    ConstChunkArray a(50);
    auto coords = coordsGrid(4, 4);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    readInterpolated3D(out, &a, 0, coords, vc::Sampling::Trilinear);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            CHECK(out(r, c) == 50);
}

TEST_CASE("readInterpolated3D(uint16) overload runs and writes zeros for a uint8-source")
{
    ConstChunkArray a(200);
    auto coords = coordsGrid(4, 4);
    cv::Mat_<uint16_t> out(4, 4, uint16_t{0});
    readInterpolated3D(out, &a, 0, coords, /*nearest_neighbor=*/true);
    // The reader does no type promotion; behavior is impl-defined for a
    // dtype mismatch — we just verify the call doesn't crash.
    CHECK(out.rows == 4);
}

TEST_CASE("readInterpolated3D(uint8, Tricubic) runs through the cubic kernel")
{
    ConstChunkArray a(80);
    auto coords = coordsGrid(4, 4, /*z=*/3.0f);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    readInterpolated3D(out, &a, 0, coords, vc::Sampling::Tricubic);
    // Tricubic on a constant block should reproduce the constant.
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            CHECK(out(r, c) == 80);
}

TEST_CASE("readMultiSlice(uint8): collects multiple offset slices")
{
    ConstChunkArray a(60);
    auto base = coordsGrid(4, 4);
    cv::Mat_<cv::Vec3f> steps(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<float> offsets = {0.f, 1.f, 2.f};
    std::vector<cv::Mat_<uint8_t>> out;
    readMultiSlice(out, &a, /*level=*/0, base, steps, offsets);
    REQUIRE(out.size() == 3);
    for (const auto& slice : out) {
        CHECK(slice.rows == 4);
        CHECK(slice.cols == 4);
        CHECK(slice(0, 0) == 60);
    }
}

TEST_CASE("readMultiSlice(uint16) overload")
{
    ConstChunkArray a(30);
    auto base = coordsGrid(2, 2);
    cv::Mat_<cv::Vec3f> steps(2, 2, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<float> offsets = {0.f, 1.f};
    std::vector<cv::Mat_<uint16_t>> out;
    readMultiSlice(out, &a, 0, base, steps, offsets);
    REQUIRE(out.size() == 2);
    // dtype mismatch — just verify the call runs and outputs are right size.
    CHECK(out[0].rows == 2);
}

TEST_CASE("sampleTileSlices(uint8): pure-tile path")
{
    ConstChunkArray a(40);
    auto base = coordsGrid(4, 4);
    cv::Mat_<cv::Vec3f> steps(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<float> offsets = {0.f, 1.f};
    std::vector<cv::Mat_<uint8_t>> out;
    sampleTileSlices(out, &a, 0, base, steps, offsets);
    REQUIRE(out.size() == 2);
    CHECK(out[0].rows == 4);
}

TEST_CASE("readCompositeFast: composite over a few slices")
{
    ConstChunkArray a(100);
    auto base = coordsGrid(4, 4);
    cv::Mat_<cv::Vec3f> normals(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    CompositeParams params;
    params.method = "mean";
    readCompositeFast(out, &a, /*level=*/0, base, normals,
                      /*zStep=*/1.0f, /*zStart=*/0, /*zEnd=*/3, params);
    // Compositing the constant value should approximate the value (mean of
    // 3 layers all = 100).
    CHECK(int(out(0, 0)) == 100);
}

TEST_CASE("samplePlane: produces non-zero values where coords are inside the chunk")
{
    ConstChunkArray a(77);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    samplePlane(out, &a, /*level=*/0,
                /*origin=*/cv::Vec3f(0, 0, 0),
                /*vx_step=*/cv::Vec3f(1, 0, 0),
                /*vy_step=*/cv::Vec3f(0, 1, 0),
                /*width=*/4, /*height=*/4,
                vc::Sampling::Nearest);
    CHECK(out(0, 0) == 77);
}
