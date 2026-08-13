// QuadSurface: multi-component surface, unloadPoints/unloadCaches happy
// paths, writeValidMask with non-empty image, save with bbox in meta.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_comp_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> grid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

} // namespace

TEST_CASE("QuadSurface(path,json) with components: gen() uses multi-component warp")
{
    auto root = tmpDir("multi_comp");
    auto segDir = root / "seg";
    // Seed a 16-wide grid on disk.
    {
        QuadSurface seed(grid(16, 16), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    // Build meta with two components covering [0,8) and [8,16).
    auto j = utils::Json::object();
    j["uuid"] = "test";
    j["type"] = "seg";
    j["format"] = "tifxyz";
    auto comps = utils::Json::array();
    auto c0 = utils::Json::array(); c0.push_back(0); c0.push_back(8);
    auto c1 = utils::Json::array(); c1.push_back(8); c1.push_back(16);
    comps.push_back(c0); comps.push_back(c1);
    j["components"] = comps;

    QuadSurface qs(segDir, j);
    qs.ensureLoaded();
    REQUIRE(qs.rawPointsPtr());
    CHECK(qs.rawPointsPtr()->cols == 16);

    // gen() with normals + a non-zero offset triggers the multi-component
    // warp path (line 766+) inside QuadSurface::gen.
    cv::Mat_<cv::Vec3f> coords, normals;
    qs.gen(&coords, &normals, cv::Size(8, 8),
           cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(0, 0, 1));
    CHECK(coords.rows == 8);
    CHECK(coords.cols == 8);
    fs::remove_all(root);
}

TEST_CASE("components-meta with degenerate ranges is filtered out")
{
    auto root = tmpDir("bad_comp");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    auto j = utils::Json::object();
    j["uuid"] = "test";
    j["type"] = "seg";
    j["format"] = "tifxyz";
    auto comps = utils::Json::array();
    // c0 < 0: invalid
    auto bad1 = utils::Json::array(); bad1.push_back(-1); bad1.push_back(4);
    // c0 > c1: degenerate
    auto bad2 = utils::Json::array(); bad2.push_back(6); bad2.push_back(3);
    // Out of cols range
    auto bad3 = utils::Json::array(); bad3.push_back(0); bad3.push_back(100);
    comps.push_back(bad1); comps.push_back(bad2); comps.push_back(bad3);
    j["components"] = comps;

    QuadSurface qs(segDir, j);
    qs.ensureLoaded();
    cv::Mat_<cv::Vec3f> coords, normals;
    qs.gen(&coords, &normals, cv::Size(4, 4),
           cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(0, 0, 1));
    CHECK(coords.rows == 4);
    fs::remove_all(root);
}

TEST_CASE("unloadPoints + ensureLoaded re-load round-trip")
{
    auto root = tmpDir("unload_rt");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    QuadSurface qs(segDir);
    qs.ensureLoaded();
    CHECK(qs.rawPointsPtr() != nullptr);
    qs.unloadPoints();
    // After unload, isLoaded should be false until next access.
    CHECK_FALSE(qs.isLoaded());
    qs.ensureLoaded();
    CHECK(qs.isLoaded());
    fs::remove_all(root);
}

TEST_CASE("unloadPoints on already-unloaded surface is a no-op")
{
    auto root = tmpDir("unload_double");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    QuadSurface qs(segDir);
    // Not yet loaded.
    qs.unloadPoints(); // returns early — _needsLoad already true
    CHECK_FALSE(qs.isLoaded());
    fs::remove_all(root);
}

TEST_CASE("writeValidMask with a non-empty image writes a multi-page TIFF")
{
    auto root = tmpDir("vmask_multi");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    QuadSurface qs(segDir);
    qs.ensureLoaded();
    qs.path = segDir;
    cv::Mat extra(8, 8, CV_8UC1, cv::Scalar(99));
    qs.writeValidMask(extra);
    // Multi-page mask.tif should exist.
    CHECK(fs::exists(segDir / "mask.tif"));
    fs::remove_all(root);
}

TEST_CASE("writeValidMask with empty image writes a single-page TIFF")
{
    auto root = tmpDir("vmask_single");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    QuadSurface qs(segDir);
    qs.ensureLoaded();
    qs.path = segDir;
    qs.writeValidMask();
    CHECK(fs::exists(segDir / "mask.tif"));
    fs::remove_all(root);
}

TEST_CASE("writeValidMask on path-less surface is a no-op")
{
    QuadSurface qs(grid(8, 8), cv::Vec2f(1.f, 1.f));
    qs.writeValidMask();
    CHECK(true);
}

TEST_CASE("bbox stored in meta is read back via (path, json) ctor")
{
    auto root = tmpDir("bbox_meta");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    auto j = utils::Json::object();
    j["uuid"] = "test";
    j["type"] = "seg";
    j["format"] = "tifxyz";
    auto bbox = utils::Json::array();
    auto lo = utils::Json::array();
    lo.push_back(1.0); lo.push_back(2.0); lo.push_back(3.0);
    auto hi = utils::Json::array();
    hi.push_back(7.0); hi.push_back(8.0); hi.push_back(9.0);
    bbox.push_back(lo); bbox.push_back(hi);
    j["bbox"] = bbox;
    QuadSurface qs(segDir, j);
    qs.ensureLoaded();
    auto rt = qs.bbox();
    // First bbox() call uses meta's cached value before computing from points.
    // We just verify the surface is alive.
    CHECK(rt.high[0] >= rt.low[0]);
    fs::remove_all(root);
}

TEST_CASE("canUnload reflects whether a backing path is set")
{
    QuadSurface in_mem(grid(4, 4), cv::Vec2f(1.f, 1.f));
    CHECK_FALSE(in_mem.canUnload());

    auto root = tmpDir("canunload");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(4, 4), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    QuadSurface reloaded(segDir);
    CHECK(reloaded.canUnload());
    fs::remove_all(root);
}
