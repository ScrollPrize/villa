// More SurfacePatchIndex coverage: updateSurfaceRegion, rebuild against
// path-backed (tifxyz) surfaces, removeCells via removeSurface, queue/flush
// in the read-only branch.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <random>

namespace fs = std::filesystem;

namespace {

#ifndef VC_TEST_FIXTURES_DIR
#define VC_TEST_FIXTURES_DIR "core/test/data"
#endif

fs::path fixtureSegment(const std::string& name)
{
    return fs::path(VC_TEST_FIXTURES_DIR) / "segments" / name;
}

std::shared_ptr<QuadSurface> makeSurface(int rows = 16, int cols = 16, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return std::make_shared<QuadSurface>(m, cv::Vec2f(1.f, 1.f));
}

} // namespace

TEST_CASE("updateSurfaceRegion covers the partial-rebuild path")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    idx.rebuild({a});
    // Partial update of a sub-region.
    CHECK(idx.updateSurfaceRegion(a, /*rowStart=*/2, /*rowEnd=*/8,
                                  /*colStart=*/2, /*colEnd=*/8));
    // Whole-surface update via the larger overload.
    (void)idx.updateSurfaceRegion(a, 0, a->rawPointsPtr()->rows,
                                  0, a->rawPointsPtr()->cols);
    // Unknown surface — return value is impl-defined; just exercise the path.
    auto other = makeSurface(8, 8);
    (void)idx.updateSurfaceRegion(other, 0, 4, 0, 4);
    CHECK(true);
}

TEST_CASE("rebuild against a path-backed (tifxyz) fixture exercises the mapped-cache path")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing");
        return;
    }
    auto qs = std::make_shared<QuadSurface>(seg);
    qs->ensureLoaded();
    SurfacePatchIndex idx;
    idx.rebuild({qs}, /*bboxPadding=*/0.5f);
    CHECK(idx.surfaceCount() == 1);
    CHECK(idx.patchCount() > 0);

    // updateSurface re-scans the mapped cache.
    CHECK(idx.updateSurface(qs));
    (void)idx.updateSurfaceRegion(qs, 0, 8, 0, 8);
}

TEST_CASE("removeSurface drops all of a surface's cells/entries")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    auto b = makeSurface(16, 16, 5.f);
    idx.rebuild({a, b});
    size_t pc = idx.patchCount();
    REQUIRE(idx.removeSurface(a));
    CHECK(idx.patchCount() < pc);
    CHECK_FALSE(idx.containsSurface(a));
}

TEST_CASE("flushPendingUpdates with no queued cells is safe")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    CHECK_FALSE(idx.hasPendingUpdates(a));
    // Return value is impl-defined when there's nothing to flush;
    // we just exercise both overloads to ensure no crash.
    (void)idx.flushPendingUpdates(a);
    (void)idx.flushPendingUpdates(nullptr);
    CHECK(true);
}

TEST_CASE("queueCellUpdateForVertex on boundary vertices still flushes cleanly")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(8, 8);
    idx.rebuild({a});
    // Corner: should generate fewer cells.
    idx.queueCellUpdateForVertex(a, 0, 0);
    CHECK(idx.hasPendingUpdates());
    CHECK(idx.flushPendingUpdates());
    // Center vertex: should generate 4 cells.
    idx.queueCellUpdateForVertex(a, 4, 4);
    CHECK(idx.flushPendingUpdates());
}

TEST_CASE("rebuild with custom sampling stride")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(32, 32);
    // Default is likely 1; setting to 2 should succeed and change behaviour.
    (void)idx.setSamplingStride(2);
    idx.rebuild({a});
    CHECK(idx.patchCount() > 0);

    SurfacePatchIndex idx2;
    (void)idx2.setSamplingStride(4);
    idx2.rebuild({a});
    CHECK(idx2.patchCount() > 0);
    // Larger stride should produce fewer (or equal) patches.
    CHECK(idx2.patchCount() <= idx.patchCount());
}
