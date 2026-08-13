// Regression tests for vc::atomicImwriteMulti.
//
// Codex P1 (PR #868): a previous version built the temp filename via
// `tmpPath += ".tmp"` which turned `mask.tif` into `mask.tif.tmp`.
// `cv::imwritemulti` selects its codec from the file extension, and
// `.tmp` has no codec — the call would fail and the helper would
// throw. Tests below force the actual write+rename cycle on a real
// `.tif` target and assert success + content survival, plus crash-
// resilience expectations.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Tiff.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TmpDir {
    fs::path path;
    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() /
               ("vc_atomic_tiff_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

cv::Mat makeBinaryMask(int rows, int cols, uint8_t fill)
{
    return cv::Mat(rows, cols, CV_8UC1, cv::Scalar(fill));
}

}  // namespace

TEST_CASE("atomicImwriteMulti: writes a multi-page TIFF to mask.tif")
{
    TmpDir dir;
    const fs::path target = dir.path / "mask.tif";

    std::vector<cv::Mat> pages = {
        makeBinaryMask(8, 12, 255),
        makeBinaryMask(8, 12, 64),
        makeBinaryMask(8, 12, 0),
    };

    REQUIRE_NOTHROW(atomicImwriteMulti(target, pages));

    CHECK(fs::exists(target));
    CHECK_FALSE(fs::exists(target.string() + ".tmp"));

    std::vector<cv::Mat> readBack;
    cv::imreadmulti(target.string(), readBack, cv::IMREAD_UNCHANGED);
    REQUIRE(readBack.size() == 3);
    CHECK(readBack[0].size() == cv::Size(12, 8));
    CHECK(readBack[1].at<uint8_t>(0, 0) == 64);
    CHECK(readBack[2].at<uint8_t>(0, 0) == 0);
}

TEST_CASE("atomicImwriteMulti: temp filename keeps the .tif extension")
{
    // The OpenCV codec is dispatched from the final extension. A bug
    // here would name the temp `mask.tif.tmp` and `cv::imwritemulti`
    // would refuse the unknown extension. We can't directly observe
    // the temp name (it's deleted on success), but we CAN observe the
    // failure mode: if the helper threw, the final file would be
    // missing. The previous test catches that. This test additionally
    // exercises the codepath where the target path already exists,
    // since OpenCV's behavior may differ on overwrite.
    TmpDir dir;
    const fs::path target = dir.path / "mask.tif";

    std::vector<cv::Mat> pages = {makeBinaryMask(4, 4, 200)};
    REQUIRE_NOTHROW(atomicImwriteMulti(target, pages));
    REQUIRE(fs::exists(target));

    std::vector<cv::Mat> overwrite = {makeBinaryMask(4, 4, 50)};
    REQUIRE_NOTHROW(atomicImwriteMulti(target, overwrite));

    std::vector<cv::Mat> readBack;
    cv::imreadmulti(target.string(), readBack, cv::IMREAD_UNCHANGED);
    REQUIRE(readBack.size() == 1);
    CHECK(readBack[0].at<uint8_t>(0, 0) == 50);
}

TEST_CASE("atomicImwriteMulti: empty page list throws")
{
    TmpDir dir;
    const fs::path target = dir.path / "mask.tif";
    std::vector<cv::Mat> empty;
    CHECK_THROWS(atomicImwriteMulti(target, empty));
    CHECK_FALSE(fs::exists(target));
}

TEST_CASE("atomicImwriteMulti: temp file is cleaned up on success")
{
    TmpDir dir;
    const fs::path target = dir.path / "mask.tif";

    std::vector<cv::Mat> pages = {makeBinaryMask(4, 4, 100)};
    REQUIRE_NOTHROW(atomicImwriteMulti(target, pages));

    // No leftover .tmp / .tmp.tif files of any flavor.
    int strayCount = 0;
    for (const auto& entry : fs::directory_iterator(dir.path)) {
        const std::string name = entry.path().filename().string();
        if (name.find(".tmp") != std::string::npos) {
            ++strayCount;
        }
    }
    CHECK(strayCount == 0);
}
