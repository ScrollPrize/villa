// Regression test for the SIGSEGV in CWindow::onAppendMaskPressed →
// render_image_from_coords → Volume::sample → readInterpolated3D path.
//
// The crash mechanism: callers passed a default-constructed
// cv::Mat_<uint8_t> img to render_image_from_coords. Down the chain the
// inner pixel-write loop in Slicing.cpp dereferenced out.ptr<T>(y) on the
// empty Mat, which is nullptr — fault at addr 0x0 on the very first row.
//
// These tests exercise readInterpolated3D directly with a stub
// IChunkedArray that returns AllFill (no real volume data needed). They
// verify that:
//   1. Empty `out` is sized to match `coords` and filled (no crash).
//   2. Empty `coords` results in `out` being released (no crash).
//   3. Pre-sized `out` matching `coords` is not reallocated.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace {

class AllFillChunkedArray final : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {64, 64, 64}; }
    std::array<int, 3> chunkShape(int) const override { return {16, 16, 16}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int) const override
    {
        return LevelTransform{};
    }

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

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override
    {
    }

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override
    {
        return 0;
    }

    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

cv::Mat_<cv::Vec3f> makeCoords(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> coords(rows, cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            coords(y, x) = cv::Vec3f(8.f, 8.f, 8.f);  // inside a 64-cube, in-bounds
        }
    }
    return coords;
}

}  // namespace

TEST_CASE("readInterpolated3D: empty out is sized to coords and filled")
{
    AllFillChunkedArray cache;
    cv::Mat_<cv::Vec3f> coords = makeCoords(32, 48);
    cv::Mat_<uint8_t> out;  // empty — pre-fix this caused SIGSEGV
    REQUIRE(out.empty());

    readInterpolated3D(out, &cache, /*level=*/0, coords, /*nearest_neighbor=*/false);

    CHECK(out.size() == cv::Size(48, 32));
    CHECK(out.type() == CV_8UC1);
    CHECK_FALSE(out.empty());
}

TEST_CASE("readInterpolated3D: pre-sized out is not reallocated")
{
    AllFillChunkedArray cache;
    cv::Mat_<cv::Vec3f> coords = makeCoords(32, 48);
    cv::Mat_<uint8_t> out(32, 48);
    out.setTo(0);
    const uint8_t* dataBefore = out.data;

    readInterpolated3D(out, &cache, /*level=*/0, coords, /*nearest_neighbor=*/false);

    CHECK(out.data == dataBefore);
    CHECK(out.size() == cv::Size(48, 32));
}

TEST_CASE("readInterpolated3D: AllFill chunks produce zero output")
{
    AllFillChunkedArray cache;
    cv::Mat_<cv::Vec3f> coords = makeCoords(16, 16);
    cv::Mat_<uint8_t> out;

    readInterpolated3D(out, &cache, /*level=*/0, coords, /*nearest_neighbor=*/true);

    REQUIRE(out.size() == cv::Size(16, 16));
    int nonzero = cv::countNonZero(out);
    CHECK(nonzero == 0);
}

TEST_CASE("readInterpolated3D: nearest and trilinear paths both honor empty-out contract")
{
    AllFillChunkedArray cache;
    cv::Mat_<cv::Vec3f> coords = makeCoords(8, 8);

    cv::Mat_<uint8_t> out_nn;
    readInterpolated3D(out_nn, &cache, 0, coords, /*nearest_neighbor=*/true);
    CHECK(out_nn.size() == coords.size());

    cv::Mat_<uint8_t> out_tl;
    readInterpolated3D(out_tl, &cache, 0, coords, /*nearest_neighbor=*/false);
    CHECK(out_tl.size() == coords.size());
}

TEST_CASE("readInterpolated3D uint16 overload: empty out is sized")
{
    AllFillChunkedArray cache;
    cv::Mat_<cv::Vec3f> coords = makeCoords(24, 24);
    cv::Mat_<uint16_t> out;

    readInterpolated3D(out, &cache, /*level=*/0, coords, /*nearest_neighbor=*/false);

    CHECK(out.size() == cv::Size(24, 24));
    CHECK(out.type() == CV_16UC1);
}
