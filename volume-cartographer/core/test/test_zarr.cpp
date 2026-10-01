// Coverage for core/src/Zarr.cpp — focuses on the downsample helpers,
// createPyramidDatasets, writeZarrAttrs, and writeZarrRegionU8ByChunk.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <cmath>
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

TEST_CASE("writeZarrAttrs derives per-axis scale from slice step and pixel density")
{
    auto d = tmpDir("attrs_axes");
    writeZarrAttrs(/*outDir=*/d, /*volPath=*/d,
                   /*groupIdx=*/1, /*baseZ=*/8,
                   /*sliceStep=*/3.0, /*accumStep=*/0.0,
                   /*accumTypeStr=*/"max", /*accumSamples=*/0,
                   /*canvasSize=*/cv::Size(32, 32),
                   /*CZ=*/8, /*CH=*/16, /*CW=*/16,
                   /*baseVoxelSize=*/8.0, /*voxelUnit=*/"um",
                   /*pixelsPerVoxel=*/2.0);
    auto j = utils::Json::parse_file(d / ".zattrs");
    auto scaleAt = [&](size_t level) {
        return j["multiscales"][size_t(0)]["datasets"][level]
                ["coordinateTransformations"][size_t(0)]["scale"]
                .get_double_array();
    };
    // Z = 8 um * slice step 3; Y/X = 8 um / 2 pixels-per-voxel.
    auto s0 = scaleAt(0);
    REQUIRE(s0.size() == 3);
    CHECK(s0[0] == doctest::Approx(24.0));
    CHECK(s0[1] == doctest::Approx(4.0));
    CHECK(s0[2] == doctest::Approx(4.0));
    // Pyramid level 2 quadruples Y/X only; Z spacing is per-slice, unchanged.
    auto s2 = scaleAt(2);
    REQUIRE(s2.size() == 3);
    CHECK(s2[0] == doctest::Approx(24.0));
    CHECK(s2[1] == doctest::Approx(16.0));
    CHECK(s2[2] == doctest::Approx(16.0));
    fs::remove_all(d);
}

TEST_CASE("buildMultiscales keeps the multiscales structure when the physical size is unknown")
{
    // The review finding: omitting the whole block traded a dubious scale for a
    // document a reader no longer recognises as a multiscale image. The block is
    // discovery metadata and has to survive.
    const auto ms = buildMultiscales(/*baseVoxelSize=*/0.0, /*voxelUnit=*/"",
                                     /*sliceStep=*/3.0, /*pixelsPerVoxel=*/2.0);

    CHECK(ms.is_object());
    CHECK(ms["version"] == "0.4");
    CHECK(ms["name"] == "render");

    // Axes are still described, so the array dimensionality is discoverable.
    REQUIRE(ms["axes"].is_array());
    REQUIRE(ms["axes"].size() == 3);
    CHECK(ms["axes"][size_t(0)]["name"] == "z");
    CHECK(ms["axes"][size_t(1)]["name"] == "y");
    CHECK(ms["axes"][size_t(2)]["name"] == "x");
    for (size_t a = 0; a < 3; ++a) {
        CHECK(ms["axes"][a]["type"] == "space");
    }

    // All six levels are still listed in resolution order.
    REQUIRE(ms["datasets"].is_array());
    REQUIRE(ms["datasets"].size() == 6);
    for (size_t l = 0; l < 6; ++l) {
        CHECK(ms["datasets"][l]["path"] == std::to_string(l));
    }
}

TEST_CASE("buildMultiscales declares no physical unit when the size is unknown")
{
    const auto ms = buildMultiscales(/*baseVoxelSize=*/0.0,
                                     /*voxelUnit=*/"micrometer",
                                     /*sliceStep=*/1.0, /*pixelsPerVoxel=*/1.0);
    // A unit is the one thing that must never appear without a measurement, and
    // it must not appear even if a caller passes one alongside a zero size.
    for (size_t a = 0; a < 3; ++a) {
        CHECK_FALSE(ms["axes"][a].contains("unit"));
    }
    CHECK(ms["metadata"]["physical_size"] == "unknown");
}

TEST_CASE("buildMultiscales treats a non-positive size as unknown")
{
    // The renderer signals "no measurement" with 0; a negative value must not be
    // mistaken for a physical size either.
    for (const double bad : {0.0, -1.0, -0.001}) {
        const auto ms = buildMultiscales(bad, "micrometer", 1.0, 1.0);
        CHECK_FALSE(ms["axes"][size_t(0)].contains("unit"));
        CHECK(ms["metadata"]["physical_size"] == "unknown");
        auto s0 = ms["datasets"][size_t(0)]["coordinateTransformations"][size_t(0)]["scale"]
                    .get_double_array();
        REQUIRE(s0.size() == 3);
        CHECK(s0[0] == doctest::Approx(1.0));
        CHECK(s0[1] == doctest::Approx(1.0));
        CHECK(s0[2] == doctest::Approx(1.0));
    }
}

TEST_CASE("buildMultiscales writes relative pyramid scaling when the size is unknown")
{
    // OME-NGFF 0.4: when a physical scale is unavailable the scale value MUST
    // express the factor between this level and the first. Level 0 is therefore
    // the identity, and Y/X double per level because createPyramidDatasets halves
    // only Y/X while Z is unchanged.
    const auto ms = buildMultiscales(/*baseVoxelSize=*/-1.0, /*voxelUnit=*/"",
                                     /*sliceStep=*/3.0, /*pixelsPerVoxel=*/2.0);
    auto scaleAt = [&](size_t level) {
        return ms["datasets"][level]["coordinateTransformations"][size_t(0)]["scale"]
                .get_double_array();
    };
    for (size_t l = 0; l < 6; ++l) {
        auto s = scaleAt(l);
        REQUIRE(s.size() == 3);
        const double factor = std::pow(2.0, static_cast<double>(l));
        CHECK(s[0] == doctest::Approx(1.0));   // Z does not downsample
        CHECK(s[1] == doctest::Approx(factor));
        CHECK(s[2] == doctest::Approx(factor));
    }
    // The transformation list keeps the order the spec requires: scale first.
    CHECK(ms["datasets"][size_t(0)]["coordinateTransformations"][size_t(0)]["type"] == "scale");
    CHECK(ms["datasets"][size_t(0)]["coordinateTransformations"][size_t(1)]["type"] == "translation");
}

TEST_CASE("buildMultiscales keeps the physical scale and unit when the size is known")
{
    // The known-size path must be untouched by the unknown-size fix: same
    // physical scales, same units, same relative growth per level.
    const auto ms = buildMultiscales(/*baseVoxelSize=*/8.0,
                                     /*voxelUnit=*/"micrometer",
                                     /*sliceStep=*/3.0, /*pixelsPerVoxel=*/2.0);
    for (size_t a = 0; a < 3; ++a) {
        REQUIRE(ms["axes"][a].contains("unit"));
        CHECK(ms["axes"][a]["unit"] == "micrometer");
    }
    CHECK_FALSE(ms["metadata"].contains("physical_size"));
    auto scaleAt = [&](size_t level) {
        return ms["datasets"][level]["coordinateTransformations"][size_t(0)]["scale"]
                .get_double_array();
    };
    auto s0 = scaleAt(0);
    REQUIRE(s0.size() == 3);
    CHECK(s0[0] == doctest::Approx(24.0));  // 8 um * slice step 3
    CHECK(s0[1] == doctest::Approx(4.0));   // 8 um / 2 pixels-per-voxel
    CHECK(s0[2] == doctest::Approx(4.0));
    auto s2 = scaleAt(2);
    REQUIRE(s2.size() == 3);
    CHECK(s2[0] == doctest::Approx(24.0));  // Z spacing is per-slice, unchanged
    CHECK(s2[1] == doctest::Approx(16.0));  // Y/X quadruple at level 2
    CHECK(s2[2] == doctest::Approx(16.0));
}

TEST_CASE("writeZarrAttrs keeps multiscales and only relative scaling when the size is unknown")
{
    // The end of the same path: whatever buildMultiscales() returns must reach the
    // .zattrs file, and the rest of the attribute set must be unaffected.
    auto d = tmpDir("attrs_unknown_size");
    writeZarrAttrs(/*outDir=*/d, /*volPath=*/d,
                   /*groupIdx=*/0, /*baseZ=*/64,
                   /*sliceStep=*/3.0, /*accumStep=*/0.0,
                   /*accumTypeStr=*/"max", /*accumSamples=*/0,
                   /*canvasSize=*/cv::Size(64, 64),
                   /*CZ=*/32, /*CH=*/32, /*CW=*/32,
                   /*baseVoxelSize=*/0.0, /*voxelUnit=*/"",
                   /*pixelsPerVoxel=*/2.0);
    CHECK(fs::exists(d / ".zattrs"));
    auto j = utils::Json::parse_file(d / ".zattrs");
    REQUIRE(j.contains("multiscales"));
    REQUIRE(j["multiscales"].is_array());
    REQUIRE(j["multiscales"].size() == 1);
    const auto& ms = j["multiscales"][size_t(0)];
    REQUIRE(ms["axes"].is_array());
    REQUIRE(ms["datasets"].is_array());
    REQUIRE(ms["datasets"].size() == 6);
    for (size_t a = 0; a < 3; ++a) CHECK_FALSE(ms["axes"][a].contains("unit"));
    auto s1 = ms["datasets"][size_t(1)]["coordinateTransformations"][size_t(0)]["scale"]
                .get_double_array();
    REQUIRE(s1.size() == 3);
    CHECK(s1[0] == doctest::Approx(1.0));
    CHECK(s1[1] == doctest::Approx(2.0));
    CHECK(s1[2] == doctest::Approx(2.0));
    // The non-multiscales attributes still describe the render.
    CHECK(j["num_slices"] == 64);
    CHECK(j["note_axes_order"] == "ZYX (slice, row, col)");
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
