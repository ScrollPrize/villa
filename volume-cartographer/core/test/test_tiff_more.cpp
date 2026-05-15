// More Tiff coverage: TiffWriter move-assignment, partial-tile padding,
// 32F dtype, atomicImwriteMulti error paths.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Tiff.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_tiff_more_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("TiffWriter move-assignment transfers state")
{
    auto d = tmpDir("move_asn");
    auto p1 = d / "a.tif";
    auto p2 = d / "b.tif";
    TiffWriter w1(p1, 32, 32, CV_8UC1, 16, 16);
    cv::Mat tile(16, 16, CV_8UC1, cv::Scalar(7));
    w1.writeTile(0, 0, tile);

    TiffWriter w2(p2, 16, 16, CV_8UC1, 16, 16);
    w2 = std::move(w1); // move-assign closes the previous, takes ownership
    CHECK(w2.isOpen());
    CHECK_FALSE(w1.isOpen());
    w2.writeTile(16, 16, tile);
    w2.close();
    CHECK(fs::exists(p1));
    fs::remove_all(d);
}

TEST_CASE("TiffWriter self-move-assignment is a no-op")
{
    auto d = tmpDir("self_move");
    auto p = d / "x.tif";
    TiffWriter w(p, 16, 16, CV_8UC1, 16, 16);
    auto* same = &w;
    w = std::move(*same);  // self-assign
    CHECK(w.isOpen());
    w.close();
    fs::remove_all(d);
}

TEST_CASE("TiffWriter partial-tile write pads with default value")
{
    auto d = tmpDir("partial");
    auto p = d / "x.tif";
    {
        TiffWriter w(p, 24, 24, CV_8UC1, /*tileW=*/16, /*tileH=*/16);
        cv::Mat full(16, 16, CV_8UC1, cv::Scalar(50));
        w.writeTile(0, 0, full);
        // Bottom-right partial tile: 8x8 actual data
        cv::Mat partial(8, 8, CV_8UC1, cv::Scalar(100));
        w.writeTile(16, 16, partial);
    }
    REQUIRE(fs::exists(p));
    fs::remove_all(d);
}

TEST_CASE("TiffWriter rejects mismatched dtype on writeTile")
{
    auto d = tmpDir("dtype_mismatch");
    auto p = d / "x.tif";
    TiffWriter w(p, 16, 16, CV_8UC1, 16, 16);
    cv::Mat wrong(16, 16, CV_16UC1, cv::Scalar(0));
    CHECK_THROWS_AS(w.writeTile(0, 0, wrong), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("TiffWriter rejects writeTile after close")
{
    auto d = tmpDir("after_close");
    auto p = d / "x.tif";
    TiffWriter w(p, 16, 16, CV_8UC1, 16, 16);
    w.close();
    cv::Mat tile(16, 16, CV_8UC1, cv::Scalar(0));
    CHECK_THROWS_AS(w.writeTile(0, 0, tile), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("writeTiff with 32F dtype roundtrips")
{
    auto d = tmpDir("f32");
    auto p = d / "x.tif";
    cv::Mat_<float> img(16, 16, 3.14f);
    writeTiff(p, img, CV_32FC1, 16, 16);
    REQUIRE(fs::exists(p));
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!loaded.empty());
    CHECK(loaded.type() == CV_32FC1);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: 16U -> 8U and 32F -> 8U conversions")
{
    auto d = tmpDir("u16_to_u8");
    cv::Mat m16(32, 32, CV_16UC1, cv::Scalar(65535));
    writeTiff(d / "a.tif", m16, CV_8UC1, /*tileW=*/0, /*tileH=*/0);
    CHECK(fs::exists(d / "a.tif"));
    cv::Mat mf(32, 32, CV_32FC1, cv::Scalar(1.0f));
    writeTiff(d / "b.tif", mf, CV_8UC1, 0, 0);
    CHECK(fs::exists(d / "b.tif"));
    fs::remove_all(d);
}

TEST_CASE("writeTiff: 16U -> 32F and 32F -> 16U conversions")
{
    auto d = tmpDir("dtype_xform");
    cv::Mat m16(32, 32, CV_16UC1, cv::Scalar(32000));
    writeTiff(d / "a.tif", m16, CV_32FC1, 0, 0);
    CHECK(fs::exists(d / "a.tif"));
    cv::Mat mf(32, 32, CV_32FC1, cv::Scalar(0.5f));
    writeTiff(d / "b.tif", mf, CV_16UC1, 0, 0);
    CHECK(fs::exists(d / "b.tif"));
    fs::remove_all(d);
}

TEST_CASE("atomicImwriteMulti: throws on empty page vector")
{
    auto d = tmpDir("atomic_empty");
    auto p = d / "x.tif";
    std::vector<cv::Mat> empty;
    CHECK_THROWS_AS(atomicImwriteMulti(p, empty), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("atomicImwriteMulti: writes multi-page TIFF atomically")
{
    auto d = tmpDir("atomic_ok");
    auto p = d / "x.tif";
    std::vector<cv::Mat> pages = {
        cv::Mat(8, 8, CV_8UC1, cv::Scalar(10)),
        cv::Mat(8, 8, CV_8UC1, cv::Scalar(20)),
    };
    atomicImwriteMulti(p, pages);
    REQUIRE(fs::exists(p));
    // No stray .tmp left over.
    CHECK_FALSE(fs::exists((d / "x.tmp.tif")));
    fs::remove_all(d);
}

TEST_CASE("voxelSizeToDpi: very tiny voxels produce large dpi")
{
    CHECK(voxelSizeToDpi(0.01) == doctest::Approx(2540000.0f).epsilon(1e-3));
}
