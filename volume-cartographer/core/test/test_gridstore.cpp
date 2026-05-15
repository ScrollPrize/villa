#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using vc::core::util::GridStore;

namespace {

std::vector<cv::Point> diagonal(int n, int dx = 1, int dy = 1, cv::Point start = {0, 0})
{
    std::vector<cv::Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        pts.push_back(cv::Point(start.x + i * dx, start.y + i * dy));
    }
    return pts;
}

std::string tmpPath(const std::string& name)
{
    auto p = std::filesystem::temp_directory_path() / ("vc_gridstore_" + name);
    return p.string();
}

} // namespace

TEST_CASE("construct empty GridStore reports zero state")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    CHECK(gs.size() == cv::Size(100, 100));
    CHECK(gs.numSegments() == 0);
    CHECK(gs.numNonEmptyBuckets() == 0);
    CHECK(gs.get_all().empty());
    CHECK(gs.get_memory_usage() >= 0);
}

TEST_CASE("add single segment populates buckets and segments")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(5)); // 5 points: (0,0)..(4,4) — within one 10x10 bucket
    CHECK(gs.numSegments() == 4); // n-1 line segments
    CHECK(gs.numNonEmptyBuckets() >= 1);
    auto all = gs.get_all();
    REQUIRE(all.size() == 1);
    CHECK(all[0]->size() == 5);
}

TEST_CASE("add with fewer than 2 points is ignored")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add({}); // empty
    gs.add({cv::Point(0, 0)}); // single
    CHECK(gs.numSegments() == 0);
    CHECK(gs.get_all().empty());
}

TEST_CASE("add multiple non-overlapping segments")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3, 1, 1, {0, 0}));
    gs.add(diagonal(3, 1, 1, {50, 50}));
    CHECK(gs.get_all().size() == 2);
}

TEST_CASE("get(query_rect) returns segments overlapping rect")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3, 1, 1, {0, 0}));
    gs.add(diagonal(3, 1, 1, {80, 80}));
    auto in_lo = gs.get(cv::Rect(0, 0, 20, 20));
    CHECK(in_lo.size() == 1);
    auto in_hi = gs.get(cv::Rect(75, 75, 20, 20));
    CHECK(in_hi.size() == 1);
    auto in_mid = gs.get(cv::Rect(40, 40, 5, 5));
    CHECK(in_mid.empty());
}

TEST_CASE("get(query_rect) clamped fully outside returns empty")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3));
    auto outside = gs.get(cv::Rect(500, 500, 10, 10));
    CHECK(outside.empty());
}

TEST_CASE("get(center, radius) overload")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(5, 1, 1, {0, 0}));
    auto hits = gs.get(cv::Point2f(2.f, 2.f), 5.f);
    CHECK(hits.size() == 1);
    auto miss = gs.get(cv::Point2f(80.f, 80.f), 1.f);
    CHECK(miss.empty());
}

TEST_CASE("forEach visits each segment once")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3, 1, 1, {0, 0}));
    gs.add(diagonal(3, 1, 1, {5, 5}));
    GridStore::QueryScratch scratch;
    int count = 0;
    gs.forEach(cv::Rect(0, 0, 30, 30), scratch,
               [&](const std::shared_ptr<std::vector<cv::Point>>& path) {
                   CHECK(path);
                   ++count;
               });
    CHECK(count == 2);
}

TEST_CASE("forEach on empty rect does nothing")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3));
    GridStore::QueryScratch scratch;
    int count = 0;
    gs.forEach(cv::Rect(500, 500, 5, 5), scratch,
               [&](const std::shared_ptr<std::vector<cv::Point>>&) { ++count; });
    CHECK(count == 0);
}

TEST_CASE("cacheStats / resetCacheStats on in-memory store")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3));
    auto s = gs.cacheStats();
    // in-memory store doesn't populate decoded-path cache; just verify call works
    CHECK(s.decodedPathHits == 0);
    CHECK(s.decodedPathMisses == 0);
    gs.resetCacheStats();
    auto s2 = gs.cacheStats();
    CHECK(s2.decodedPathHits == 0);
}

TEST_CASE("meta member is a usable JSON object")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.meta = utils::Json::object();
    gs.meta["scroll"] = "1";
    CHECK(gs.meta.is_object());
}

TEST_CASE("save / load round-trips data")
{
    const auto path = tmpPath("save_load.bin");
    std::filesystem::remove(path);

    {
        GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add(diagonal(5, 1, 1, {0, 0}));
        gs.add(diagonal(5, 1, 1, {50, 50}));
        GridStore::SaveOptions opts;
        opts.verify_reload = true;
        gs.save(path, opts);
    }

    GridStore loaded(path);
    CHECK(loaded.size() == cv::Size(100, 100));
    auto all = loaded.get_all();
    CHECK(all.size() == 2);
    auto in_lo = loaded.get(cv::Rect(0, 0, 10, 10));
    CHECK(in_lo.size() == 1);

    std::filesystem::remove(path);
}

TEST_CASE("save with default options compiles & roundtrips")
{
    const auto path = tmpPath("save_default.bin");
    std::filesystem::remove(path);
    {
        GridStore gs(cv::Rect(0, 0, 50, 50), 10);
        gs.add(diagonal(4, 1, 1, {1, 1}));
        gs.save(path);
    }
    GridStore loaded(path);
    CHECK(loaded.get_all().size() == 1);
    std::filesystem::remove(path);
}

TEST_CASE("save throws on a read-only (mmap-loaded) store")
{
    const auto path = tmpPath("save_readonly.bin");
    std::filesystem::remove(path);
    {
        GridStore gs(cv::Rect(0, 0, 50, 50), 10);
        gs.add(diagonal(3));
        gs.save(path);
    }
    GridStore loaded(path);
    CHECK_THROWS_AS(loaded.save(tmpPath("save_readonly_2.bin")), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("add throws on a read-only (mmap-loaded) store")
{
    const auto path = tmpPath("add_readonly.bin");
    std::filesystem::remove(path);
    {
        GridStore gs(cv::Rect(0, 0, 50, 50), 10);
        gs.add(diagonal(3));
        gs.save(path);
    }
    GridStore loaded(path);
    CHECK_THROWS_AS(loaded.add(diagonal(3)), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("constructor from missing file throws")
{
    CHECK_THROWS(GridStore("/nonexistent/__no__/gridstore.bin"));
}
