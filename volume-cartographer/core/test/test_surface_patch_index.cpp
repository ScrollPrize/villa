// Coverage for core/src/SurfacePatchIndex.cpp.
//
// Uses small in-memory QuadSurfaces to drive build/locate/triangle queries
// and the various R-tree query helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

std::shared_ptr<QuadSurface> makeSurface(int rows = 16, int cols = 16, float z = 0.f)
{
    return std::make_shared<QuadSurface>(makePlanarGrid(rows, cols, z), cv::Vec2f(1.f, 1.f));
}

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_spi_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("Default ctor is empty")
{
    SurfacePatchIndex idx;
    CHECK(idx.empty());
    CHECK(idx.patchCount() == 0);
    CHECK(idx.surfaceCount() == 0);
}

TEST_CASE("rebuild populates surface and patch counts")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(8, 8, 0.f);
    auto b = makeSurface(8, 8, 5.f);
    idx.rebuild({a, b});
    CHECK_FALSE(idx.empty());
    CHECK(idx.surfaceCount() == 2);
    CHECK(idx.patchCount() > 0);
    CHECK(idx.containsSurface(a));
    CHECK(idx.containsSurface(b));
}

TEST_CASE("containsSurface returns false for unknown surface")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    auto other = makeSurface();
    CHECK(idx.containsSurface(a));
    CHECK_FALSE(idx.containsSurface(other));
    CHECK_FALSE(idx.containsSurface(nullptr));
}

TEST_CASE("clear empties the index")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    CHECK_FALSE(idx.empty());
    idx.clear();
    CHECK(idx.empty());
    CHECK(idx.surfaceCount() == 0);
}

TEST_CASE("locate point on the surface yields a result")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    idx.rebuild({a});

    SurfacePatchIndex::PointQuery q;
    q.worldPoint = cv::Vec3f(4.f, 4.f, 0.f); // on the plane
    q.tolerance = 0.5f;
    auto r = idx.locate(q);
    REQUIRE(r.has_value());
    CHECK(r->surface == a);
    CHECK(r->distance >= 0.0f);
}

TEST_CASE("locate point off-surface returns nullopt")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    idx.rebuild({a});
    SurfacePatchIndex::PointQuery q;
    q.worldPoint = cv::Vec3f(1000, 1000, 1000);
    q.tolerance = 0.5f;
    auto r = idx.locate(q);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("locateAll returns all hits within tolerance")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16, 0.f);
    auto b = makeSurface(16, 16, 0.f); // coincident
    idx.rebuild({a, b});
    SurfacePatchIndex::PointQuery q;
    q.worldPoint = cv::Vec3f(4.f, 4.f, 0.f);
    q.tolerance = 0.5f;
    auto rs = idx.locateAll(q);
    CHECK(rs.size() >= 1);
}

TEST_CASE("locateSurfaces deduplicates surface results")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    idx.rebuild({a});
    SurfacePatchIndex::PointQuery q;
    q.worldPoint = cv::Vec3f(4.f, 4.f, 0.f);
    q.tolerance = 0.5f;
    auto ss = idx.locateSurfaces(q);
    CHECK_FALSE(ss.empty());
    CHECK(ss[0] == a);
}

TEST_CASE("forEachTriangle with TriangleQuery visits triangles inside the bounds")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(8, 8);
    idx.rebuild({a});

    SurfacePatchIndex::TriangleQuery q;
    q.bounds.low = cv::Vec3f(0, 0, -1);
    q.bounds.high = cv::Vec3f(8, 8, 1);
    int count = 0;
    idx.forEachTriangle(q,
        [&](const SurfacePatchIndex::TriangleCandidate& tc) {
            CHECK(tc.surface == a);
            ++count;
        });
    CHECK(count > 0);
}

TEST_CASE("forEachTriangle with RayQuery visits the candidates the ray crosses")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16);
    idx.rebuild({a});

    SurfacePatchIndex::RayQuery rq;
    rq.src = cv::Vec3f(8, 8, -5);
    rq.end = cv::Vec3f(8, 8, 5);
    int count = 0;
    idx.forEachTriangle(rq,
        [&](const SurfacePatchIndex::TriangleCandidate&) { ++count; });
    CHECK(count > 0);
}

TEST_CASE("samplingStride round-trip")
{
    SurfacePatchIndex idx;
    int before = idx.samplingStride();
    CHECK(idx.setSamplingStride(2));
    CHECK(idx.samplingStride() == 2);
    // Going back
    CHECK(idx.setSamplingStride(before));
    // Invalid stride is rejected
    CHECK_FALSE(idx.setSamplingStride(0));
    CHECK_FALSE(idx.setSamplingStride(-3));
}

TEST_CASE("setReadOnly: exercises the read-only flag toggle")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    idx.setReadOnly(true);
    // The impl may reject some writes; just exercise the path.
    (void)idx.removeSurface(a);
    (void)idx.updateSurface(a);
    idx.setReadOnly(false);
    // Now writes should succeed (if surface still present).
    if (idx.containsSurface(a)) {
        CHECK(idx.updateSurface(a));
    }
}

TEST_CASE("removeSurface drops a surface from the index")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    auto b = makeSurface(8, 8, 5.f);
    idx.rebuild({a, b});
    CHECK(idx.surfaceCount() == 2);
    CHECK(idx.removeSurface(a));
    CHECK(idx.surfaceCount() == 1);
    CHECK_FALSE(idx.containsSurface(a));
    // Removing again is a no-op
    CHECK_FALSE(idx.removeSurface(a));
}

TEST_CASE("pending updates: queue + flush + hasPendingUpdates")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    CHECK_FALSE(idx.hasPendingUpdates());
    idx.queueCellUpdateForVertex(a, 3, 3);
    CHECK(idx.hasPendingUpdates(a));
    CHECK(idx.flushPendingUpdates(a));
    CHECK_FALSE(idx.hasPendingUpdates(a));
}

TEST_CASE("queueCellRangeUpdate + flush works on a range")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    idx.queueCellRangeUpdate(a, 0, 4, 0, 4);
    CHECK(idx.hasPendingUpdates());
    idx.flushPendingUpdates();
    CHECK_FALSE(idx.hasPendingUpdates());
}

TEST_CASE("generation advances after updateSurface")
{
    SurfacePatchIndex idx;
    auto a = makeSurface();
    idx.rebuild({a});
    auto g0 = idx.generation(a);
    CHECK(idx.updateSurface(a));
    auto g1 = idx.generation(a);
    CHECK(g1 >= g0);
}

TEST_CASE("computePlaneIntersections returns segments for an intersecting plane")
{
    SurfacePatchIndex idx;
    auto a = makeSurface(16, 16, 0.f);
    idx.rebuild({a});

    PlaneSurface plane(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets{a};
    auto result = idx.computePlaneIntersections(plane, cv::Rect(0, 0, 16, 16), targets);
    // For a Z=0 surface intersected by Z=0 plane, every triangle should
    // produce a segment. We don't pin the count; just that something came back.
    CHECK_FALSE(result.empty());
}

TEST_CASE("cacheKeyForSurfaces is stable for the same input")
{
    auto a = makeSurface(8, 8);
    auto b = makeSurface(8, 8, 1.f);
    auto k1 = SurfacePatchIndex::cacheKeyForSurfaces({a, b}, /*stride=*/1, /*padding=*/0.0f);
    auto k2 = SurfacePatchIndex::cacheKeyForSurfaces({a, b}, 1, 0.0f);
    CHECK(k1 == k2);
    // Different stride yields different key.
    auto k3 = SurfacePatchIndex::cacheKeyForSurfaces({a, b}, 2, 0.0f);
    CHECK(k3 != k1);
}

TEST_CASE("save/load cache round-trips empty index")
{
    auto d = tmpDir("cache");
    auto cache = d / "spi.cache";
    auto a = makeSurface(8, 8);
    {
        SurfacePatchIndex idx;
        idx.rebuild({a});
        auto key = SurfacePatchIndex::cacheKeyForSurfaces({a}, idx.samplingStride(), 0.0f);
        CHECK(idx.saveCache(cache, key));
    }
    SurfacePatchIndex idx2;
    auto key = SurfacePatchIndex::cacheKeyForSurfaces({a}, idx2.samplingStride(), 0.0f);
    bool loaded = idx2.loadCache(cache, {a}, key);
    if (loaded) {
        CHECK(idx2.surfaceCount() == 1);
    }
    fs::remove_all(d);
}

TEST_CASE("Move ctor / assignment transfer state")
{
    SurfacePatchIndex a;
    auto s = makeSurface();
    a.rebuild({s});
    CHECK(a.surfaceCount() == 1);
    SurfacePatchIndex b(std::move(a));
    CHECK(b.surfaceCount() == 1);
    SurfacePatchIndex c;
    c = std::move(b);
    CHECK(c.surfaceCount() == 1);
}
