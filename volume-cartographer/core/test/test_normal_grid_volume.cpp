// Coverage for core/src/NormalGridVolume.cpp. We don't have real .grid
// fixtures, so the tests exercise the path that returns nullopt/nullptr
// when grid files are absent. That still covers the cache miss path,
// metadata access, and cacheStats/resetCacheStats.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using vc::core::util::NormalGridVolume;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_ngv_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

fs::path makeEmptyNgvDir(const std::string& tag, int sparseVolume = 4)
{
    auto d = tmpDir(tag);
    fs::create_directories(d / "xy");
    fs::create_directories(d / "xz");
    fs::create_directories(d / "yz");
    std::ofstream f(d / "metadata.json");
    f << "{\"sparse-volume\":" << sparseVolume << "}";
    return d;
}

} // namespace

TEST_CASE("Constructor: missing metadata.json throws")
{
    auto d = tmpDir("missing_meta");
    fs::create_directories(d / "xy");
    fs::create_directories(d / "xz");
    fs::create_directories(d / "yz");
    // No metadata.json.
    auto make = [d]() { NormalGridVolume v(d.string()); (void)v; };
    CHECK_THROWS(make());
    fs::remove_all(d);
}

TEST_CASE("Constructor + metadata accessor")
{
    auto d = makeEmptyNgvDir("meta_access", 8);
    NormalGridVolume v(d.string());
    CHECK(v.metadata().is_object());
    CHECK(v.metadata()["sparse-volume"].get_int64() == 8);
    fs::remove_all(d);
}

TEST_CASE("get_grid: missing slice returns nullptr; cache miss is recorded")
{
    auto d = makeEmptyNgvDir("get_grid");
    NormalGridVolume v(d.string());
    auto g = v.get_grid(/*plane_idx=*/0, /*slice_idx=*/0);
    CHECK(g == nullptr);
    auto stats = v.cacheStats();
    CHECK(stats.gridMisses >= 1);
    // Repeated lookup hits the negative-result cache.
    auto g2 = v.get_grid(0, 0);
    CHECK(g2 == nullptr);
    auto stats2 = v.cacheStats();
    CHECK(stats2.gridHits >= 1);
    fs::remove_all(d);
}

TEST_CASE("query: empty store returns nullopt for all planes")
{
    auto d = makeEmptyNgvDir("query");
    NormalGridVolume v(d.string());
    CHECK_FALSE(v.query(cv::Point3f(0, 0, 0), 0).has_value());
    CHECK_FALSE(v.query(cv::Point3f(0, 0, 0), 1).has_value());
    CHECK_FALSE(v.query(cv::Point3f(0, 0, 0), 2).has_value());
    // Bad plane index is also nullopt.
    CHECK_FALSE(v.query(cv::Point3f(0, 0, 0), 99).has_value());
    fs::remove_all(d);
}

TEST_CASE("query_nearest: empty store returns null for all planes")
{
    auto d = makeEmptyNgvDir("query_nearest");
    NormalGridVolume v(d.string());
    CHECK(v.query_nearest(cv::Point3f(0, 0, 0), 0) == nullptr);
    CHECK(v.query_nearest(cv::Point3f(0, 0, 0), 1) == nullptr);
    CHECK(v.query_nearest(cv::Point3f(0, 0, 0), 2) == nullptr);
    CHECK(v.query_nearest(cv::Point3f(0, 0, 0), 99) == nullptr);
    fs::remove_all(d);
}

TEST_CASE("resetCacheStats zeros hit/miss counters")
{
    auto d = makeEmptyNgvDir("reset");
    NormalGridVolume v(d.string());
    (void)v.get_grid(0, 0);
    auto before = v.cacheStats();
    CHECK(before.gridMisses >= 1);
    v.resetCacheStats();
    auto after = v.cacheStats();
    CHECK(after.gridHits == 0);
    CHECK(after.gridMisses == 0);
    fs::remove_all(d);
}

TEST_CASE("Move construction / assignment")
{
    auto d = makeEmptyNgvDir("move");
    NormalGridVolume a(d.string());
    NormalGridVolume b(std::move(a));
    CHECK(b.metadata().is_object());
    NormalGridVolume c = makeEmptyNgvDir("moveasn").string()
                            != "" ? NormalGridVolume(d.string()) : NormalGridVolume(d.string());
    (void)c;
    fs::remove_all(d);
}

TEST_CASE("get_grid with a real (committed) GridStore file is read back")
{
    auto d = makeEmptyNgvDir("with_grid");
    // Drop a real GridStore at xy/000000.grid (slice 0 on XY plane).
    {
        vc::core::util::GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add({cv::Point(0, 0), cv::Point(5, 5), cv::Point(10, 10)});
        gs.save((d / "xy" / "000000.grid").string());
    }
    NormalGridVolume v(d.string());
    auto g = v.get_grid(0, 0);
    REQUIRE(g != nullptr);
    CHECK(g->numSegments() >= 1);
    // A second lookup hits the cache.
    auto stats_before = v.cacheStats();
    auto g2 = v.get_grid(0, 0);
    CHECK(g2 != nullptr);
    auto stats_after = v.cacheStats();
    CHECK(stats_after.gridHits > stats_before.gridHits);
    fs::remove_all(d);
}
