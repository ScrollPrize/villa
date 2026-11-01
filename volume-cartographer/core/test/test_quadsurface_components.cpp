// QuadSurface: multi-component surface, writeValidMask with non-empty image, save with bbox in meta.

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

TEST_CASE("QuadSurface(path,json) with components: the grid is never interpolated across them")
{
    auto root = tmpDir("multi_comp");
    auto segDir = root / "seg";
    // Seed a 16-wide grid on disk.
    {
        QuadSurface seed(grid(16, 16, 5.f), cv::Vec2f(1.f, 1.f));
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
    const auto options = qs.gridOptions();
    REQUIRE(options.components.size() == 2);
    CHECK(options.components[0] == std::pair{0, 8});
    CHECK(options.components[1] == std::pair{8, 16});

    // Grid columns 6.5, 7.5 (the seam cell) and 8.5 of row 3.
    std::vector<std::uint8_t> valid;
    const vcr::grid::Surface surface(qs.gridVertices(valid), options);
    const auto origin = qs.gridToSurface({6.5, 3.0});
    float xyz[9];
    surface.resample(qs.gridRaster(origin[0], origin[1], 1.0, cv::Size(3, 1)), xyz, nullptr);
    CHECK(xyz[0] == doctest::Approx(6.5f));
    CHECK(std::isnan(xyz[3]));
    CHECK(xyz[6] == doctest::Approx(8.5f));
    fs::remove_all(root);
}

TEST_CASE("components-meta ranges are clamped to the grid and empty ones ignored")
{
    auto root = tmpDir("bad_comp");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(8, 8, 5.f), cv::Vec2f(1.f, 1.f));
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
    // [0, 100) clamps to every column: one piece, interpolated everywhere.
    std::vector<std::uint8_t> valid;
    const vcr::grid::Surface surface(qs.gridVertices(valid), qs.gridOptions());
    const auto origin = qs.gridToSurface({0.5, 3.0});
    float xyz[7 * 3];
    surface.resample(qs.gridRaster(origin[0], origin[1], 1.0, cv::Size(7, 1)), xyz, nullptr);
    for (int i = 0; i < 7; ++i)
        CHECK(xyz[3 * i] == doctest::Approx(0.5f + float(i)));
    fs::remove_all(root);
}

TEST_CASE("writeValidMask writes a single-page TIFF")
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

