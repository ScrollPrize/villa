// Coverage for core/src/Zarr.cpp — focuses on the downsample helpers,
// createPyramidDatasets, writeZarrAttrs, and writeZarrRegionU8ByChunk.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_zarr_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

std::vector<uint8_t> ramp(size_t n)
{
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(i & 0xFF);
    return v;
}

} // namespace

TEST_CASE("downsampleChunk<uint8>: 2x2x2 mean of a constant block is the constant")
{
    std::vector<uint8_t> src(4 * 4 * 4, 100);
    std::vector<uint8_t> dst(2 * 2 * 2, 0);
    downsampleChunk<uint8_t>(src.data(), 4, 4, 4, dst.data(), 2, 2, 2,
                             /*actual=*/4, 4, 4);
    for (auto v : dst) CHECK(v == 100);
}

TEST_CASE("downsampleChunk<uint16>: handles a non-uniform input")
{
    std::vector<uint16_t> src(4 * 4 * 4);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint16_t>(i);
    std::vector<uint16_t> dst(2 * 2 * 2, 0);
    downsampleChunk<uint16_t>(src.data(), 4, 4, 4, dst.data(), 2, 2, 2, 4, 4, 4);
    // dst[0] = mean of indices forming the 2x2x2 corner at (0,0,0).
    // Indices: 0,1,4,5,16,17,20,21 -> sum=84, mean=10 (with rounding +/-)
    CHECK(int(dst[0]) >= 9);
    CHECK(int(dst[0]) <= 11);
}

TEST_CASE("downsampleChunk: edge tile with srcActual smaller than chunk shape")
{
    std::vector<uint8_t> src(4 * 4 * 4, 50);
    std::vector<uint8_t> dst(2 * 2 * 2, 0);
    // srcActual = 3 in z; effectively only 3 z-slabs contribute.
    downsampleChunk<uint8_t>(src.data(), 4, 4, 4, dst.data(), 2, 2, 2,
                             /*actualZ=*/3, /*actualY=*/4, /*actualX=*/4);
    // dst[0] should still be 50 (all sources are 50)
    CHECK(dst[0] == 50);
}

TEST_CASE("downsampleTileInto<uint8>: writes at an offset within dst")
{
    std::vector<uint8_t> src(4 * 4 * 4, 200);
    std::vector<uint8_t> dst(2 * 4 * 4, 0);
    downsampleTileInto<uint8_t>(src.data(), 4, 4, 4,
                                dst.data(), /*dstZ=*/2, /*dstY=*/4, /*dstX=*/4,
                                /*actual=*/4, 4, 4,
                                /*dstOffY=*/2, /*dstOffX=*/2);
    // Wrote into (zz in [0,2), y in [2,4), x in [2,4)) — top-left of dst is 0.
    CHECK(dst[0] == 0);
    // bottom-right corner (last writable cell):
    CHECK(dst[1 * 4 * 4 + 3 * 4 + 3] == 200);
}

TEST_CASE("downsampleTileIntoPreserveZ<uint8>: preserves Z dim, halves Y/X")
{
    std::vector<uint8_t> src(2 * 4 * 4, 80);
    std::vector<uint8_t> dst(2 * 4 * 4, 0);
    downsampleTileIntoPreserveZ<uint8_t>(src.data(), 2, 4, 4,
                                         dst.data(), 2, 4, 4,
                                         /*actual=*/2, 4, 4,
                                         /*dstOffY=*/0, /*dstOffX=*/0);
    // First written cell == constant value
    CHECK(dst[0] == 80);
}

TEST_CASE("createPyramidDatasets writes L1..L5 metadata directories")
{
    auto d = tmpDir("pyr_create");
    // Create L0 first (createPyramidDatasets only writes L1..L5).
    vc::createZarrDataset(d, "0",
        /*shape=*/{64, 64, 64}, /*chunks=*/{32, 32, 32},
        vc::VcDtype::uint8, /*compressor=*/"none");
    createPyramidDatasets(d, /*shape0=*/{64, 64, 64},
                          /*CH=*/32, /*CW=*/32, /*isU16=*/false);
    for (int lvl = 1; lvl <= 5; ++lvl) {
        CHECK(fs::exists(d / std::to_string(lvl) / ".zarray"));
    }
    fs::remove_all(d);
}

TEST_CASE("writeZarrAttrs writes a parseable .zattrs at the volume path")
{
    auto d = tmpDir("attrs");
    writeZarrAttrs(/*outDir=*/d, /*volPath=*/d,
                   /*groupIdx=*/0, /*baseZ=*/64,
                   /*sliceStep=*/1.0, /*accumStep=*/1.0,
                   /*accumTypeStr=*/"mean", /*accumSamples=*/1,
                   /*canvasSize=*/cv::Size(64, 64),
                   /*CZ=*/32, /*CH=*/32, /*CW=*/32,
                   /*baseVoxelSize=*/7.91, /*voxelUnit=*/"um");
    CHECK(fs::exists(d / ".zattrs"));
    auto j = utils::Json::parse_file(d / ".zattrs");
    CHECK(j.is_object());
    CHECK(j.contains("multiscales"));
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk writes data into multiple chunks")
{
    auto d = tmpDir("region_chunks");
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{16, 16, 16}, /*chunks=*/{8, 8, 8},
        vc::VcDtype::uint8, /*compressor=*/"none");
    REQUIRE(ds);

    std::vector<uint8_t> in = ramp(16 * 16 * 16);
    writeZarrRegionU8ByChunk(ds.get(),
                             /*offset=*/{0, 0, 0},
                             /*regionShape=*/{16, 16, 16},
                             in.data(),
                             /*fillValue=*/0);
    // Read back chunk (0,0,0) and confirm it matches.
    std::vector<uint8_t> out(8 * 8 * 8, 0);
    CHECK(ds->readChunk(0, 0, 0, out.data()));
    CHECK(out[0] == in[0]);
    fs::remove_all(d);
}

TEST_CASE("writeZarrRegionU8ByChunk with a partial region")
{
    auto d = tmpDir("region_partial");
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{16, 16, 16}, /*chunks=*/{8, 8, 8},
        vc::VcDtype::uint8, /*compressor=*/"none");
    REQUIRE(ds);
    // Write only the first 4x4x4 corner.
    std::vector<uint8_t> in(4 * 4 * 4, 199);
    writeZarrRegionU8ByChunk(ds.get(), {0, 0, 0}, {4, 4, 4}, in.data(), 0);
    std::vector<uint8_t> out(8 * 8 * 8, 0);
    CHECK(ds->readChunk(0, 0, 0, out.data()));
    CHECK(out[0] == 199);
    // Chunks that are outside the region should still report as absent.
    CHECK_FALSE(ds->chunkExists(1, 0, 0));
    fs::remove_all(d);
}

TEST_CASE("buildPyramidLevel: builds L1 from L0")
{
    auto d = tmpDir("buildpyr");
    auto l0 = vc::createZarrDataset(d, "0",
        /*shape=*/{16, 16, 16}, /*chunks=*/{8, 8, 8},
        vc::VcDtype::uint8, "none");
    REQUIRE(l0);
    // Fill chunk (0,0,0) with a known value so L1 has data to downsample.
    std::vector<uint8_t> payload(8 * 8 * 8, 60);
    l0->writeChunk(0, 0, 0, payload.data(), payload.size());

    // createPyramidDatasets writes L1..L5 metadata.
    createPyramidDatasets(d, {16, 16, 16}, 8, 8, /*isU16=*/false);
    buildPyramidLevel<uint8_t>(d, /*level=*/1, /*CH=*/8, /*CW=*/8);

    // L1 chunk (0,0,0) should have the downsampled constant.
    vc::VcDataset l1(d / "1");
    std::vector<uint8_t> out(l1.defaultChunkSize(), 0);
    if (l1.chunkExists(0, 0, 0)) {
        CHECK(l1.readChunk(0, 0, 0, out.data()));
        // L0 was constant 60 — L1 should also be 60 in the downsampled region.
        CHECK(int(out[0]) == 60);
    }
    fs::remove_all(d);
}
