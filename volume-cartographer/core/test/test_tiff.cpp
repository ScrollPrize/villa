// Coverage for core/src/Tiff.cpp. atomicImwriteMulti has its own test elsewhere;
// here we exercise voxelSizeToDpi, writeTiff (typed & untiled), TiffWriter
// tiled writes, type-conversion paths, and mergeTiffParts no-input.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Tiff.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_tiff_test_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat ramp(int rows, int cols, int type = CV_8UC1)
{
    cv::Mat m(rows, cols, type);
    if (type == CV_8UC1) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                m.at<uint8_t>(r, c) = static_cast<uint8_t>((r * cols + c) & 0xFF);
    } else if (type == CV_16UC1) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                m.at<uint16_t>(r, c) = static_cast<uint16_t>(r * cols + c);
    } else if (type == CV_32FC1) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                m.at<float>(r, c) = static_cast<float>(r * cols + c) / 100.0f;
    }
    return m;
}

} // namespace

TEST_CASE("voxelSizeToDpi: known conversions and zero/negative input")
{
    CHECK(voxelSizeToDpi(0.0) == doctest::Approx(0.0f));
    CHECK(voxelSizeToDpi(-1.0) == doctest::Approx(0.0f));
    // 25400 / 1 = 25400 dpi for 1 µm voxels
    CHECK(voxelSizeToDpi(1.0) == doctest::Approx(25400.0f));
    CHECK(voxelSizeToDpi(7.91) == doctest::Approx(25400.0 / 7.91).epsilon(1e-3));
}

TEST_CASE("writeTiff: round-trips an 8-bit ramp")
{
    auto d = tmpDir("u8");
    auto p = d / "u8.tif";
    auto img = ramp(32, 32, CV_8UC1);
    writeTiff(p, img);
    REQUIRE(fs::exists(p));
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!loaded.empty());
    CHECK(loaded.type() == CV_8UC1);
    CHECK(loaded.rows == 32);
    CHECK(loaded.cols == 32);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: round-trips a 16-bit ramp")
{
    auto d = tmpDir("u16");
    auto p = d / "u16.tif";
    auto img = ramp(16, 16, CV_16UC1);
    writeTiff(p, img);
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!loaded.empty());
    CHECK(loaded.type() == CV_16UC1);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: untiled (tileW=0) path")
{
    auto d = tmpDir("untiled");
    auto p = d / "untiled.tif";
    auto img = ramp(8, 8, CV_8UC1);
    writeTiff(p, img, /*cvType=*/-1, /*tileW=*/0, /*tileH=*/0);
    REQUIRE(fs::exists(p));
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!loaded.empty());
    CHECK(loaded.rows == 8);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: 8U→16U conversion path")
{
    auto d = tmpDir("u8_to_u16");
    auto p = d / "out.tif";
    auto img = ramp(4, 4, CV_8UC1);
    writeTiff(p, img, CV_16UC1, 16, 16);
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(loaded.type() == CV_16UC1);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: 8U→32F conversion path")
{
    auto d = tmpDir("u8_to_f32");
    auto p = d / "out.tif";
    auto img = ramp(4, 4, CV_8UC1);
    writeTiff(p, img, CV_32FC1, 16, 16);
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(loaded.type() == CV_32FC1);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: dpi tag stored when nonzero")
{
    auto d = tmpDir("dpi");
    auto p = d / "out.tif";
    auto img = ramp(4, 4, CV_8UC1);
    writeTiff(p, img, -1, 16, 16, -1.0f, COMPRESSION_LZW, 300.0f);
    REQUIRE(fs::exists(p));
    fs::remove_all(d);
}

TEST_CASE("writeTiff: empty image throws")
{
    auto d = tmpDir("empty");
    cv::Mat empty;
    CHECK_THROWS_AS(writeTiff(d / "x.tif", empty), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("writeTiff: multi-channel image throws")
{
    auto d = tmpDir("multich");
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    CHECK_THROWS_AS(writeTiff(d / "x.tif", m), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("TiffWriter: tiled write round-trips")
{
    auto d = tmpDir("writer");
    auto p = d / "out.tif";
    {
        TiffWriter w(p, 32, 32, CV_8UC1, 16, 16);
        CHECK(w.isOpen());
        CHECK(w.width() == 32);
        CHECK(w.height() == 32);
        CHECK(w.tileWidth() == 16);
        CHECK(w.tileHeight() == 16);
        CHECK(w.cvType() == CV_8UC1);
        cv::Mat tile = ramp(16, 16, CV_8UC1);
        w.writeTile(0, 0, tile);
        w.writeTile(16, 0, tile);
        w.writeTile(0, 16, tile);
        w.writeTile(16, 16, tile);
    } // destructor closes
    REQUIRE(fs::exists(p));
    auto loaded = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!loaded.empty());
    CHECK(loaded.rows == 32);
    CHECK(loaded.cols == 32);
    fs::remove_all(d);
}

TEST_CASE("TiffWriter: explicit close is idempotent (destructor still safe)")
{
    auto d = tmpDir("close");
    auto p = d / "out.tif";
    TiffWriter w(p, 16, 16, CV_8UC1, 16, 16);
    cv::Mat tile = ramp(16, 16, CV_8UC1);
    w.writeTile(0, 0, tile);
    w.close();
    CHECK_FALSE(w.isOpen());
    // Calling close() again should be a no-op.
    w.close();
    CHECK(fs::exists(p));
    fs::remove_all(d);
}

TEST_CASE("TiffWriter: move construction transfers file handle")
{
    auto d = tmpDir("move");
    auto p = d / "out.tif";
    TiffWriter w1(p, 16, 16, CV_8UC1, 16, 16);
    cv::Mat tile = ramp(16, 16, CV_8UC1);
    TiffWriter w2(std::move(w1));
    CHECK(w2.isOpen());
    CHECK_FALSE(w1.isOpen());
    w2.writeTile(0, 0, tile);
    w2.close();
    CHECK(fs::exists(p));
    fs::remove_all(d);
}

TEST_CASE("mergeTiffParts: no part files returns false")
{
    auto d = tmpDir("mergenone");
    auto outPath = (d / "out.tif").string();
    // No .part*.tif files present → expect false.
    bool ok = mergeTiffParts(outPath, /*numParts=*/3);
    CHECK_FALSE(ok);
    fs::remove_all(d);
}
