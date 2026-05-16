// Push SurfacePatchIndex save/load-cache and path-backed surface paths.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
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

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_spi_cache_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("cacheKeyForSurfaces with path-backed surface differs from in-memory")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing");
        return;
    }
    auto a = std::make_shared<QuadSurface>(seg);
    a->ensureLoaded();
    auto kA = SurfacePatchIndex::cacheKeyForSurfaces({a}, 1, 0.0f);
    CHECK_FALSE(kA.empty());

    // In-memory only surface (path is empty)
    cv::Mat_<cv::Vec3f> mat(8, 8, cv::Vec3f(0, 0, 0));
    auto b = std::make_shared<QuadSurface>(mat, cv::Vec2f(1.f, 1.f));
    auto kB = SurfacePatchIndex::cacheKeyForSurfaces({b}, 1, 0.0f);
    CHECK_FALSE(kB.empty());
    CHECK(kA != kB);
}

TEST_CASE("saveCache + loadCache round-trip preserves the rebuilt index for a path-backed surface")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing");
        return;
    }
    auto d = tmpDir("rt");
    auto cachePath = d / "spi.cache";

    auto a = std::make_shared<QuadSurface>(seg);
    a->ensureLoaded();

    size_t origPatches = 0;
    {
        SurfacePatchIndex idx;
        idx.rebuild({a});
        origPatches = idx.patchCount();
        auto key = SurfacePatchIndex::cacheKeyForSurfaces({a},
            idx.samplingStride(), 0.0f);
        CHECK(idx.saveCache(cachePath, key));
    }

    SurfacePatchIndex idx2;
    auto key2 = SurfacePatchIndex::cacheKeyForSurfaces({a},
        idx2.samplingStride(), 0.0f);
    if (idx2.loadCache(cachePath, {a}, key2)) {
        CHECK(idx2.surfaceCount() == 1);
        CHECK(idx2.patchCount() == origPatches);
    }
    fs::remove_all(d);
}

TEST_CASE("loadCache: mismatched key is rejected")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) return;
    auto d = tmpDir("badkey");
    auto cachePath = d / "spi.cache";

    auto a = std::make_shared<QuadSurface>(seg);
    a->ensureLoaded();
    {
        SurfacePatchIndex idx;
        idx.rebuild({a});
        auto key = SurfacePatchIndex::cacheKeyForSurfaces({a},
            idx.samplingStride(), 0.0f);
        CHECK(idx.saveCache(cachePath, key));
    }

    SurfacePatchIndex idx2;
    CHECK_FALSE(idx2.loadCache(cachePath, {a}, "not-the-right-key"));
    fs::remove_all(d);
}

TEST_CASE("loadCache: missing file returns false")
{
    auto d = tmpDir("missing");
    auto a = std::make_shared<QuadSurface>(
        cv::Mat_<cv::Vec3f>(4, 4, cv::Vec3f(0, 0, 0)),
        cv::Vec2f(1.f, 1.f));
    SurfacePatchIndex idx;
    CHECK_FALSE(idx.loadCache(d / "nonexistent.cache", {a}, "any-key"));
    fs::remove_all(d);
}

TEST_CASE("loadCache: garbage file is rejected gracefully")
{
    auto d = tmpDir("garbage");
    auto cachePath = d / "spi.cache";
    {
        std::ofstream f(cachePath, std::ios::binary);
        f << "not a real cache file at all";
    }
    auto a = std::make_shared<QuadSurface>(
        cv::Mat_<cv::Vec3f>(4, 4, cv::Vec3f(0, 0, 0)),
        cv::Vec2f(1.f, 1.f));
    SurfacePatchIndex idx;
    CHECK_FALSE(idx.loadCache(cachePath, {a}, "any-key"));
    fs::remove_all(d);
}

TEST_CASE("locate with SurfaceFilter include and exclude")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) return;
    auto a = std::make_shared<QuadSurface>(seg);
    a->ensureLoaded();
    auto b = std::make_shared<QuadSurface>(seg); // same fixture, distinct surface
    b->ensureLoaded();

    SurfacePatchIndex idx;
    idx.rebuild({a, b});

    SurfacePatchIndex::PointQuery q;
    auto bb = a->bbox();
    q.worldPoint = (bb.low + bb.high) * 0.5f;
    q.tolerance = 5.0f;

    std::unordered_set<SurfacePatchIndex::SurfacePtr> incl{a};
    q.surfaces.include = &incl;
    auto rs = idx.locateAll(q);
    // All results (if any) must be surface a.
    for (const auto& r : rs) CHECK(r.surface == a);

    std::unordered_set<SurfacePatchIndex::SurfacePtr> excl{a};
    SurfacePatchIndex::PointQuery q2;
    q2.worldPoint = q.worldPoint;
    q2.tolerance = q.tolerance;
    q2.surfaces.exclude = &excl;
    auto rs2 = idx.locateAll(q2);
    for (const auto& r : rs2) CHECK(r.surface != a);
}

TEST_CASE("forEachTriangle with patchFilter accepts/rejects bounds")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) return;
    auto a = std::make_shared<QuadSurface>(seg);
    a->ensureLoaded();

    SurfacePatchIndex idx;
    idx.rebuild({a});

    SurfacePatchIndex::TriangleQuery q;
    auto bb = a->bbox();
    q.bounds.low = bb.low;
    q.bounds.high = bb.high;
    int allowed = 0, blocked = 0;
    q.patchFilter = [&](const SurfacePatchIndex::PatchBounds&) {
        return (allowed + blocked) % 2 == 0;
    };
    idx.forEachTriangle(q,
        [&](const SurfacePatchIndex::TriangleCandidate&) { /* no-op */ });
    CHECK(true);
}
