// Cover mergeTiffParts happy path: write two tiled .partN.tif files, merge,
// and verify the resulting file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Tiff.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tiffio.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_tiff_merge_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat tilesPart(int rows, int cols, uint8_t v)
{
    cv::Mat m(rows, cols, CV_8UC1, cv::Scalar(v));
    return m;
}

} // namespace

TEST_CASE("mergeTiffParts: two-part merge produces a final TIFF")
{
    auto d = tmpDir("ok");
    // Create two .part files, each a tiled 32x32 image.
    auto p0 = d / "img.part0.tif";
    auto p1 = d / "img.part1.tif";
    writeTiff(p0, tilesPart(32, 32, 50), CV_8UC1, /*tileW=*/16, /*tileH=*/16);
    writeTiff(p1, tilesPart(32, 32, 60), CV_8UC1, 16, 16);

    auto finalPath = (d / "img.tif").string();
    bool ok = mergeTiffParts(finalPath, /*numParts=*/2);
    CHECK(ok);
    CHECK(fs::exists(finalPath));
    // Part files removed.
    CHECK_FALSE(fs::exists(p0));
    CHECK_FALSE(fs::exists(p1));
    // File exists and is non-empty.
    CHECK(fs::file_size(finalPath) > 0);
    fs::remove_all(d);
}

TEST_CASE("mergeTiffParts: path is a file, parent dir is scanned")
{
    auto d = tmpDir("by_file");
    auto p0 = d / "x.part0.tif";
    writeTiff(p0, tilesPart(16, 16, 99), CV_8UC1, 16, 16);
    // Pass the final-path file (which doesn't exist yet) — merge should
    // scan the parent dir.
    auto finalFile = (d / "x.tif").string();
    bool ok = mergeTiffParts(finalFile, 1);
    CHECK(ok);
    CHECK(fs::exists(finalFile));
    fs::remove_all(d);
}

namespace {
// Write a STRIPPED (non-tiled) TIFF, i.e. one with no TILEWIDTH/TILELENGTH
// tags — what a corrupt or truncated render part might look like.
void writeStrippedTiff(const fs::path& path, int rows, int cols, uint8_t v)
{
    TIFF* t = TIFFOpen(path.string().c_str(), "w");
    REQUIRE(t != nullptr);
    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, uint32_t(cols));
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, uint32_t(rows));
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, uint16_t(8));
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, uint16_t(1));
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    // No TILEWIDTH/TILELENGTH -> libtiff writes scanline strips.
    std::vector<uint8_t> row(size_t(cols), v);
    for (int r = 0; r < rows; ++r)
        TIFFWriteScanline(t, row.data(), uint32_t(r), 0);
    TIFFClose(t);
}
} // namespace

TEST_CASE("mergeTiffParts: a part missing tile-geometry tags is skipped, not crashed")
{
    auto d = tmpDir("stripped");
    // A stripped part TIFF has no TILEWIDTH/TILELENGTH. On the unfixed merge,
    // tw/th stay uninitialized and (w + tw - 1) / tw divides by zero / reads
    // garbage. The hardened merge must skip the group and report failure
    // (returns false) without crashing.
    writeStrippedTiff(d / "bad.part0.tif", 32, 32, 7);

    auto finalPath = (d / "bad.tif").string();
    bool ok = mergeTiffParts(finalPath, /*numParts=*/1);
    CHECK_FALSE(ok);                       // the only group failed
    CHECK_FALSE(fs::exists(finalPath));    // nothing written
    fs::remove_all(d);
}
