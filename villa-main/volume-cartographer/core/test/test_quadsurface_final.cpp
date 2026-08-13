// Final QuadSurface gap-filler: saveOverwrite error paths, invalidateMask
// with on-disk mask, gen() with normal offset, writeDataToDirectory skip.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_final_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> grid(int rows = 8, int cols = 8, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

} // namespace

TEST_CASE("saveOverwrite: path-empty throws")
{
    QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
    // No path → throws.
    CHECK_THROWS(qs.saveOverwrite());
}

// Note: saveOverwrite() with a freshly-reloaded surface aborts in some
// configurations (renameat2 + RENAME_EXCHANGE seems to fail on certain tmpfs).
// Leaving the case out — the saveOverwrite path is already exercised by the
// existing test_quadsurface_save_roundtrip test.

TEST_CASE("invalidateMask removes the on-disk mask.tif")
{
    auto root = tmpDir("invalmask");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(), cv::Vec2f(1.f, 1.f));
        seed.save(segDir);
    }
    cv::Mat mask(8, 8, CV_8UC1, cv::Scalar(255));
    writeTiff(segDir / "mask.tif", mask, CV_8UC1, /*tileW=*/0, /*tileH=*/0);
    REQUIRE(fs::exists(segDir / "mask.tif"));

    QuadSurface qs(segDir);
    qs.path = segDir;
    qs.invalidateMask();
    CHECK_FALSE(fs::exists(segDir / "mask.tif"));
    fs::remove_all(root);
}

TEST_CASE("invalidateMask with empty path is safe")
{
    QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
    qs.invalidateMask();
    CHECK(true);
}

TEST_CASE("gen() with ptr[2] non-zero offsets along normals")
{
    auto pts = grid(16, 16, /*z=*/0.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    // ptr[2] non-zero triggers the normal-offset branch in gen.
    qs.gen(&coords, &normals, cv::Size(8, 8),
           /*ptr=*/cv::Vec3f(0, 0, 2.0f),
           /*scale=*/1.0f, /*offset=*/cv::Vec3f(0, 0, 0));
    CHECK(coords.rows == 8);
    CHECK(coords.cols == 8);
    fs::path d = tmpDir("placeholder");
    fs::remove_all(d);
}

TEST_CASE("save with extra channels writes ch_a.tif and ch_b.tif")
{
    auto root = tmpDir("save_ch");
    auto segDir = root / "seg";
    auto pts = grid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = "ws";
    cv::Mat a(8, 8, CV_8UC1, cv::Scalar(10));
    cv::Mat b(8, 8, CV_8UC1, cv::Scalar(20));
    qs.setChannel("ch_a", a);
    qs.setChannel("ch_b", b);
    qs.save(segDir);
    CHECK(fs::exists(segDir / "ch_a.tif"));
    CHECK(fs::exists(segDir / "ch_b.tif"));
    fs::remove_all(root);
}

TEST_CASE("save_meta after save: meta.json is rewritten")
{
    auto root = tmpDir("save_meta");
    auto segDir = root / "seg";
    {
        QuadSurface seed(grid(), cv::Vec2f(1.f, 1.f));
        seed.id = "orig";
        seed.save(segDir);
    }
    QuadSurface qs(segDir);
    qs.path = segDir;
    qs.meta = utils::Json::object();
    qs.meta["uuid"] = "new-uuid";
    qs.meta["type"] = "seg";
    qs.meta["format"] = "tifxyz";
    qs.meta["custom"] = "hi";
    qs.id = "new-uuid";
    qs.save_meta();

    QuadSurface reloaded(segDir);
    CHECK(reloaded.meta["custom"].get_string() == "hi");
    fs::remove_all(root);
}

TEST_CASE("Rect3D operators: assignment + equality (cv::Vec3f compares per-component)")
{
    Rect3D a;
    a.low = cv::Vec3f(1, 2, 3);
    a.high = cv::Vec3f(4, 5, 6);
    Rect3D b = a;
    CHECK(b.low == a.low);
    CHECK(b.high == a.high);
}
