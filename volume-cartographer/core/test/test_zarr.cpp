// Coverage for core/src/Zarr.cpp — focuses on the downsample helpers,
// createPyramidDatasets and writeZarrAttrs.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Zarr.hpp"
#include <vc_reader/zarr.hpp>

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
    vcr::zarr::Writer::create(d / "0", {
        .shape = {64, 64, 64}, .chunks = {32, 32, 32}, .compressor = "none"});
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

template<class T>
void checkPyramidRoundTrip()
{
    auto d = tmpDir("buildpyr");
    const bool u16 = sizeof(T) == 2;
    auto l0 = vcr::zarr::Writer::create(d / "0", {
        .shape = {3, 19, 21}, .chunks = {3, 8, 8},
        .dtype = u16 ? "<u2" : "|u1", .compressor = "zstd", .separator = "/"});
    std::vector<T> input(3 * 19 * 21, 0);
    for (size_t z = 0; z < 3; ++z)
        for (size_t y = 0; y < 19; ++y)
            for (size_t x = 0; x < 21; ++x)
                if (y >= 8 || x >= 8)
                    input[(z * 19 + y) * 21 + x] = T((u16 ? 1000 : 10) + z * 23 + y + x);
    l0.writeRegion({0, 0, 0}, {3, 19, 21}, reinterpret_cast<const uint8_t*>(input.data()));
    // Remove the zero chunk so the build also has to handle absent data.
    l0.removeChunk({0, 0, 0});
    createPyramidDatasets(d, {3, 19, 21}, 8, 8, u16, "zstd", -1, "/");
    buildPyramidLevel<T>(d, 1);
    auto l1 = vcr::zarr::openArray((d / "1").string(), {});
    REQUIRE(l1.meta.shape == vcr::Index{3, 10, 11});
    std::vector<T> output(3 * 10 * 11);
    l1.array->read_many({{0, 0, 0}}, {{3, 10, 11}}, {reinterpret_cast<uint8_t*>(output.data())});
    for (size_t z = 0; z < 3; ++z)
        for (size_t y = 0; y < 10; ++y)
            for (size_t x = 0; x < 11; ++x) {
                uint32_t sum = 0, count = 0;
                for (size_t sy = y * 2; sy < std::min(y * 2 + 2, size_t(19)); ++sy)
                    for (size_t sx = x * 2; sx < std::min(x * 2 + 2, size_t(21)); ++sx) {
                        sum += input[(z * 19 + sy) * 21 + sx];
                        ++count;
                    }
                CHECK(output[(z * 10 + y) * 11 + x] == T((sum + count / 2) / count));
            }
    fs::remove_all(d);
}

TEST_CASE("buildPyramidLevel preserves uint8 slices, sparse chunks and odd edges")
{
    checkPyramidRoundTrip<uint8_t>();
}

TEST_CASE("buildPyramidLevel preserves uint16 slices, sparse chunks and odd edges")
{
    checkPyramidRoundTrip<uint16_t>();
}
