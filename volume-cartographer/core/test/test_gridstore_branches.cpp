// Coverage gap-filler for GridStore: corrupt-file branches, empty-file load,
// repeated queries to exercise the seglist cache, queryBucketRange edge cases.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::GridStore;

namespace {

std::vector<cv::Point> diagonal(int n, cv::Point start = {0, 0})
{
    std::vector<cv::Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        pts.push_back(cv::Point(start.x + i, start.y + i));
    }
    return pts;
}

fs::path tmpFile(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    return fs::temp_directory_path() /
           ("vc_gridstore_branch_" + tag + "_" + std::to_string(rng()) + ".bin");
}

} // namespace

TEST_CASE("constructor from empty file: bounds zeroed, no crash on queries")
{
    auto p = tmpFile("empty");
    {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
    }
    GridStore gs(p.string());
    CHECK(gs.size() == cv::Size(0, 0));
    CHECK(gs.get_all().empty());
    // Query into an empty grid: shouldn't crash, just return empty.
    CHECK(gs.get(cv::Rect(0, 0, 10, 10)).empty());
    fs::remove(p);
}

TEST_CASE("constructor: file too small for header throws")
{
    auto p = tmpFile("tiny");
    {
        std::ofstream f(p, std::ios::binary);
        f << "ab"; // 2 bytes — far below the 11x uint32 header
    }
    CHECK_THROWS_AS(GridStore(p.string()), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("constructor: bad magic throws")
{
    auto p = tmpFile("badmagic");
    {
        std::ofstream f(p, std::ios::binary);
        // Write 11 uint32s of zeros (zero magic ≠ GRIDSTORE_MAGIC).
        uint32_t zeros[13] = {};
        f.write(reinterpret_cast<const char*>(zeros), sizeof(zeros));
    }
    CHECK_THROWS_AS(GridStore(p.string()), std::runtime_error);
    fs::remove(p);
}

TEST_CASE("repeated queries hit the seglist cache (read-only store)")
{
    auto p = tmpFile("cache");
    {
        GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        for (int i = 0; i < 5; ++i) {
            gs.add(diagonal(4, cv::Point(i * 10, i * 10)));
        }
        gs.save(p.string());
    }

    GridStore loaded(p.string());
    auto stats0 = loaded.cacheStats();

    // First query: pure cache misses
    auto r1 = loaded.get(cv::Rect(0, 0, 50, 50));
    CHECK_FALSE(r1.empty());

    auto stats1 = loaded.cacheStats();
    CHECK(stats1.decodedPathMisses >= stats0.decodedPathMisses);

    // Second identical query: should produce cache hits
    auto r2 = loaded.get(cv::Rect(0, 0, 50, 50));
    auto stats2 = loaded.cacheStats();
    CHECK(stats2.decodedPathHits > stats1.decodedPathHits);

    loaded.resetCacheStats();
    auto stats3 = loaded.cacheStats();
    CHECK(stats3.decodedPathHits == 0);
    CHECK(stats3.decodedPathMisses == 0);

    fs::remove(p);
}

TEST_CASE("queryBucketRange: query rect entirely outside bounds returns empty")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(5));
    // Query with negative-extent rect (degenerate after clamp).
    CHECK(gs.get(cv::Rect(50, 50, 0, 0)).empty());
    // Query way outside.
    CHECK(gs.get(cv::Rect(1000, 1000, 5, 5)).empty());
}

TEST_CASE("queryBucketRange on zero-size grid is safe")
{
    GridStore gs(cv::Rect(0, 0, 0, 0), 10);
    CHECK(gs.get(cv::Rect(0, 0, 100, 100)).empty());
    CHECK(gs.numNonEmptyBuckets() == 0);
}

TEST_CASE("forEach with read-only (mmap-loaded) store")
{
    auto p = tmpFile("readonly_foreach");
    {
        GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add(diagonal(4, cv::Point(0, 0)));
        gs.add(diagonal(4, cv::Point(50, 50)));
        gs.save(p.string());
    }
    GridStore loaded(p.string());
    GridStore::QueryScratch scratch;
    int count = 0;
    loaded.forEach(cv::Rect(0, 0, 100, 100), scratch,
                   [&](const std::shared_ptr<std::vector<cv::Point>>& path) {
                       CHECK(path);
                       ++count;
                   });
    CHECK(count == 2);
    fs::remove(p);
}

TEST_CASE("get(center, radius) with read-only store")
{
    auto p = tmpFile("readonly_radius");
    {
        GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add(diagonal(5));
        gs.save(p.string());
    }
    GridStore loaded(p.string());
    auto hits = loaded.get(cv::Point2f(2.f, 2.f), 5.f);
    CHECK(hits.size() == 1);
    fs::remove(p);
}

TEST_CASE("numSegments matches sum across all segments")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    gs.add(diagonal(3)); // 2 segments
    gs.add(diagonal(5)); // 4 segments
    gs.add(diagonal(2)); // 1 segment
    CHECK(gs.numSegments() == 2 + 4 + 1);
}
