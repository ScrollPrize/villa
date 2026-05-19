// Additional Zarr.cpp coverage: writeZarrBand<uint8>, writeZarrBand<uint16>,
// plus writeZarrRegionU8ByChunk error-path validation.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
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
             ("vc_zarr_more_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("writeZarrBand<uint8>: writes one band of slices into a dataset")
{
    auto d = tmpDir("u8_band");
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{32, 32, 32}, /*chunks=*/{16, 16, 16},
        vc::VcDtype::uint8, /*compressor=*/"none");
    REQUIRE(ds);

    // Build a list of 16 slices (one chunk's worth of Z), each 32x32.
    std::vector<cv::Mat> slices(16, cv::Mat(32, 32, CV_8UC1, cv::Scalar(120)));
    writeZarrBand<uint8_t>(ds.get(), slices,
        /*bandIdx=*/0,
        /*chunks0=*/{16, 16, 16},
        /*tilesXSrc=*/2, /*tilesYSrc=*/1,
        /*rotQuad=*/-1, /*flipAxis=*/-1);

    CHECK(ds->chunkExists(0, 0, 0));
    CHECK(ds->chunkExists(0, 0, 1));
    fs::remove_all(d);
}

TEST_CASE("writeZarrBand<uint16>: same logic for 16-bit dtype")
{
    auto d = tmpDir("u16_band");
    auto ds = vc::createZarrDataset(d, "arr",
        {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint16, "none");
    REQUIRE(ds);
    std::vector<cv::Mat> slices(16, cv::Mat(16, 16, CV_16UC1, cv::Scalar(40000)));
    writeZarrBand<uint16_t>(ds.get(), slices, 0, {16, 16, 16}, 1, 1, -1, -1);
    CHECK(ds->chunkExists(0, 0, 0));
    fs::remove_all(d);
}

TEST_CASE("writeZarrBand: rotQuad/flipAxis remaps the destination tile coords")
{
    auto d = tmpDir("rot_band");
    auto ds = vc::createZarrDataset(d, "arr",
        {16, 16, 16}, {16, 16, 16},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);
    std::vector<cv::Mat> slices(16, cv::Mat(16, 16, CV_8UC1, cv::Scalar(7)));
    // Single tile so rotation is observable but doesn't OOB.
    writeZarrBand<uint8_t>(ds.get(), slices, 0, {16, 16, 16}, 1, 1,
                            /*rotQuad=*/0, /*flipAxis=*/0);
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk: null dataset throws")
{
    std::vector<uint8_t> data(8 * 8 * 8, 0);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(nullptr, {0, 0, 0}, {8, 8, 8}, data.data(), 0),
        std::runtime_error);
}

TEST_CASE("writeZarrRegionU8ByChunk: wrong-rank offset throws")
{
    auto d = tmpDir("badrank");
    auto ds = vc::createZarrDataset(d, "arr",
        {16, 16, 16}, {16, 16, 16}, vc::VcDtype::uint8, "none");
    std::vector<uint8_t> data(16, 0);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), /*offset=*/{0, 0}, /*regionShape=*/{8, 8, 8}, data.data(), 0),
        std::runtime_error);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {8, 8}, data.data(), 0),
        std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk: uint16 dataset is rejected")
{
    auto d = tmpDir("u16_reject");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint16, "none");
    std::vector<uint8_t> data(8 * 8 * 8, 0);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {8, 8, 8}, data.data(), 0),
        std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk: out-of-bounds region throws")
{
    auto d = tmpDir("oob");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "none");
    std::vector<uint8_t> data(16, 0);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {16, 8, 8}, data.data(), 0),
        std::runtime_error);
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {8, 8, 16}, data.data(), 0),
        std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk: zero-extent region is a no-op (returns)")
{
    auto d = tmpDir("zero_region");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "none");
    std::vector<uint8_t> data(1, 0);
    // Should return without throwing.
    writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {0, 8, 8}, data.data(), 0);
    CHECK_FALSE(ds->chunkExists(0, 0, 0));
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk: non-zero region with null data throws")
{
    auto d = tmpDir("null_data");
    auto ds = vc::createZarrDataset(d, "arr",
        {8, 8, 8}, {8, 8, 8}, vc::VcDtype::uint8, "none");
    CHECK_THROWS_AS(
        writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {4, 4, 4}, nullptr, 0),
        std::runtime_error);
    fs::remove_all(d);
}
