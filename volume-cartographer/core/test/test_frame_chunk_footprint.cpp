// The private chunk pools raise their floor from these estimates, so an estimate
// that grows with zoom turns a bounded pool into an unbounded one. That shipped
// once: measuring the frame in level-0 voxels (pixels / camera scale) demanded a
// ~281 GB pool for a slice pane at the zoom the panes *open* at, so the LRU never
// evicted and RSS climbed to ~94 GB with no user interaction.
//
// These tests pin the property that matters: the estimate depends on viewport
// pixels and chunk geometry, never on zoom.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "volume_viewers/FrameChunkFootprint.hpp"

#include <array>
#include <cstddef>

using vc3d::chunkFootprintBytes;
using vc3d::PlaneFrameGeometry;
using vc3d::planeFrameChunkFootprintBytes;
using vc3d::surfaceFrameChunkFootprintBytes;
using vc3d::surfaceTileFillChunkFootprintBytes;

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * kMiB;

// Shapes are zarr order [z, y, x].
constexpr std::array<int, 3> kChunk128{128, 128, 128};
constexpr std::array<int, 3> kChunk256{256, 256, 256};
constexpr std::array<int, 3> kChunk64{64, 64, 64};

PlaneFrameGeometry axisAligned(int fbW, int fbH, double thickness = 1.0)
{
    PlaneFrameGeometry frame;
    frame.fbW = fbW;
    frame.fbH = fbH;
    frame.basisX = {1.0, 0.0, 0.0};
    frame.basisY = {0.0, 1.0, 0.0};
    frame.normal = {0.0, 0.0, 1.0};
    frame.layerThicknessVoxels = thickness;
    return frame;
}

} // namespace

TEST_CASE("a plane frame's footprint does not depend on zoom")
{
    // The signature takes no camera scale, so this is structural -- but assert
    // the resulting magnitude is sane for the zoom the panes open at, which is
    // where the regression was.
    const auto frame = axisAligned(1500, 1000);

    SUBCASE("128^3 chunks")
    {
        const std::size_t bytes = planeFrameChunkFootprintBytes(kChunk128, 1, frame);
        // 13 x 9 x 2 chunks of 2 MiB.
        CHECK(bytes == std::size_t(13) * 9 * 2 * 2 * kMiB);
        CHECK(bytes < 512 * kMiB);
    }

    SUBCASE("256^3 chunks -- the case a fixed 256 MB default would thrash on")
    {
        const std::size_t bytes = planeFrameChunkFootprintBytes(kChunk256, 1, frame);
        // 7 x 5 x 2 chunks of 16 MiB.
        CHECK(bytes == std::size_t(7) * 5 * 2 * 16 * kMiB);
        CHECK(bytes < 2 * kGiB);
    }

    SUBCASE("uint16 doubles it")
    {
        CHECK(planeFrameChunkFootprintBytes(kChunk128, 2, frame) ==
              2 * planeFrameChunkFootprintBytes(kChunk128, 1, frame));
    }
}

TEST_CASE("footprint stays bounded for every plane orientation")
{
    // seg xz / seg yz carry rotation and tilt, so the frame projects onto all
    // three axes. That may cost more than an axis-aligned frame, but it must stay
    // the same order of magnitude -- not grow without bound.
    const std::size_t aligned = planeFrameChunkFootprintBytes(kChunk128, 1, axisAligned(1500, 1000));

    PlaneFrameGeometry tilted;
    tilted.fbW = 1500;
    tilted.fbH = 1000;
    // 45 degrees about z, then the view's up axis along volume z.
    const double s = 0.70710678;
    tilted.basisX = {s, s, 0.0};
    tilted.basisY = {0.0, 0.0, 1.0};
    tilted.normal = {s, -s, 0.0};
    const std::size_t tiltedBytes = planeFrameChunkFootprintBytes(kChunk128, 1, tilted);

    CHECK(tiltedBytes > 0);
    CHECK(tiltedBytes < 8 * aligned);
    CHECK(tiltedBytes < 2 * kGiB);

    SUBCASE("a fully diagonal plane is still bounded")
    {
        PlaneFrameGeometry diagonal;
        diagonal.fbW = 1500;
        diagonal.fbH = 1000;
        const double t = 0.57735027;  // 1/sqrt(3)
        diagonal.basisX = {t, t, t};
        diagonal.basisY = {t, -t, t};
        diagonal.normal = {t, t, -t};
        const std::size_t bytes = planeFrameChunkFootprintBytes(kChunk128, 1, diagonal);
        CHECK(bytes > 0);
        CHECK(bytes < 2 * kGiB);
    }
}

TEST_CASE("composite thickness adds a chunk plane, not a multiple")
{
    // A layer stack steps one voxel per layer, so even the spinbox maximum of 64
    // fits inside one or two chunks along the normal.
    const std::size_t plain = planeFrameChunkFootprintBytes(kChunk128, 1, axisAligned(1500, 1000, 1.0));
    const std::size_t deep = planeFrameChunkFootprintBytes(kChunk128, 1, axisAligned(1500, 1000, 129.0));
    CHECK(deep > plain);
    // 129 voxels crosses at most two chunk boundaries, so 3 planes vs 2.
    CHECK(deep == plain / 2 * 3);
    CHECK(deep < 1 * kGiB);
}

TEST_CASE("a 4K viewport on 256^3 uint16 chunks stays inside the pool ceiling")
{
    // The worst realistic combination: 320 chunks of 33.5 MB is ~10.7 GiB for a
    // single frame. Large but correct -- and it has to fit under the derived
    // ceiling (8x the 2 GiB floor, capped at 16 GiB) or that pane thrashes.
    const std::size_t bytes =
        planeFrameChunkFootprintBytes(kChunk256, 2, axisAligned(3840, 2160, 65.0));
    CHECK(bytes > 8 * kGiB);
    CHECK(bytes < 16 * kGiB);
    // Doubled by noteChunkFootprint, then clamped -- the point is that it is a
    // finite number in the tens of GiB, not the hundreds.
    CHECK(2 * bytes < 32 * kGiB);
}

TEST_CASE("flattened-surface frames use the same pixel-extent rule")
{
    const std::size_t bytes = surfaceFrameChunkFootprintBytes(kChunk128, 1, 1500, 1000, 1.0);
    CHECK(bytes == planeFrameChunkFootprintBytes(kChunk128, 1, axisAligned(1500, 1000)));
    CHECK(bytes < 512 * kMiB);
}

TEST_CASE("one round of surface-tile fills fits the filler pool's floor")
{
    // The filler pool's internal floor is 2 GiB, and its derived requirement is
    // doubled before use, so one concurrent round must sit well inside that.
    for (const auto& chunk : {kChunk64, kChunk128, kChunk256}) {
        CAPTURE(chunk[0]);
        const std::size_t bytes =
            surfaceTileFillChunkFootprintBytes(chunk, 1, 128, 32, 8);
        CHECK(bytes > 0);
        CHECK(2 * bytes <= 2 * kGiB);
    }

    SUBCASE("independent of level, because a tile is a fixed sample count")
    {
        // The band shrinks in the sampled level's voxels, so a coarse level can
        // only ever need fewer chunks than the level-0 figure asserted above.
        const std::size_t atLevel0 = surfaceTileFillChunkFootprintBytes(kChunk128, 1, 128, 32, 8);
        const std::size_t atLevel3 = surfaceTileFillChunkFootprintBytes(kChunk128, 1, 128, 4, 8);
        CHECK(atLevel3 <= atLevel0);
    }
}

TEST_CASE("degenerate inputs yield zero rather than a huge number")
{
    CHECK(chunkFootprintBytes({0, 0, 0}, 1, {100.0, 100.0, 1.0}) == 0);
    CHECK(chunkFootprintBytes(kChunk128, 0, {100.0, 100.0, 1.0}) == 0);
    CHECK(chunkFootprintBytes(kChunk128, 1, {std::nan(""), 100.0, 1.0}) == 0);
    // Negative spans clamp to zero rather than wrapping: one chunk per axis.
    CHECK(chunkFootprintBytes(kChunk128, 1, {-5000.0, -5000.0, -5000.0}) == 2 * kMiB);
}
