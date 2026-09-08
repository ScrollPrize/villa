// Coverage for voxel-size resolution from volume-store metadata. Every schema
// case below is a real document shape from the Vesuvius open-data catalog,
// reduced to the fields that matter. The failure this guards is #1603: a
// volume whose voxel size cannot be resolved reports 0, and every physical
// measurement derived from it silently becomes 0 too.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/VoxelSizeMetadata.hpp"

#include <string>

using utils::Json;
using vc::metadata::voxelSizeFromStoreMetadata;

namespace
{

Json parse(const std::string& text) { return Json::parse(text); }

// A scan record without the `scan` wrapper. samplePixelSize is in mm.
std::string scanRecord(double samplePixelSizeMm)
{
    return R"({"tomo":{"acquisition":{"detector":{"samplePixelSize":)" +
           std::to_string(samplePixelSizeMm) + R"(}}}})";
}

} // namespace

TEST_CASE("store metadata: source volume with a scan-wrapped acquisition record")
{
    const auto resolved = voxelSizeFromStoreMetadata(
        parse(R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":0.00791}}}}})"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(7.91));
}

TEST_CASE("store metadata: fused mosaic export with the record at the document root")
{
    // The 1.129 um mosaics publish the same record without the `scan` wrapper,
    // and alongside two numbers a search for anything resolution-shaped would
    // find first: a mask downsample factor and a z crop range.
    const auto resolved = voxelSizeFromStoreMetadata(parse(R"({
        "masking": {"mask_scale": 8, "dilation_iterations": "1"},
        "zarr_export": {"z_crop_start": 16, "z_crop_end": 42208},
        "mosaic": {"tile_count": 7},
        "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.001129}}}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(1.129));
}

TEST_CASE("store metadata: prediction run on level 0 of its source volume")
{
    const auto resolved = voxelSizeFromStoreMetadata(parse(R"({
        "kind": "surface-inference",
        "source": {"resolution": "0", "metadata": )" + scanRecord(0.00864) + R"(}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(8.64));
}

TEST_CASE("store metadata: prediction run on level 2 is four times its source voxel size")
{
    // Ignoring the level reports 2.4 um instead of 9.6, which makes every
    // derived area 16x too small -- small enough to fall under a min_area_cm
    // gate and be discarded as noise.
    const auto resolved = voxelSizeFromStoreMetadata(parse(R"({
        "kind": "surface-inference",
        "source": {
            "resolution": "2",
            "metadata": {"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.0024}}}}}
        }
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(9.6));
}

TEST_CASE("store metadata: an unreadable source level resolves nothing")
{
    // Falling back to level 0 would report a plausible number that is wrong by
    // an unknown power of two, which is worse than reporting none.
    for (const char* level : {R"("high")", "-1", "2.5", "null", "true", "99"}) {
        const auto doc = parse(std::string(R"({"source": {"resolution": )") + level +
                               R"(, "metadata": )" + scanRecord(0.0024) + "}}");
        CHECK_FALSE(voxelSizeFromStoreMetadata(doc).has_value());
    }
}

TEST_CASE("store metadata: a derived store must state its source level")
{
    CHECK_FALSE(voxelSizeFromStoreMetadata(parse(R"({
        "kind": "surface-inference",
        "source": {"metadata": {"scan": {"tomo": {"acquisition": {"detector":
                   {"samplePixelSize": 0.0024}}}}}}
    })")).has_value());
}

TEST_CASE("store metadata: an explicit voxelsize wins over an acquisition record")
{
    const auto resolved = voxelSizeFromStoreMetadata(parse(R"({
        "voxelsize": 3.24,
        "scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00791}}}}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(3.24));
}

TEST_CASE("store metadata: the alternative spellings of an explicit voxel size")
{
    for (const char* key : {"voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um",
                            "pixelSizeUm", "resolution_um"}) {
        const auto resolved = voxelSizeFromStoreMetadata(
            parse(std::string(R"({")") + key + R"(": 2.4})"));
        REQUIRE(resolved.has_value());
        CHECK(*resolved == doctest::Approx(2.4));
    }
}

TEST_CASE("store metadata: a nested pixel size is not read as this volume's")
{
    // On a derived store `metadata` holds the source volume's record. A pixel
    // size found there is the source's, at whatever pyramid level the run
    // consumed. Only `source.metadata` plus `source.resolution` together carry
    // enough to answer, and that path is tested above.
    CHECK_FALSE(voxelSizeFromStoreMetadata(parse(R"({
        "metadata": {"pixel_size_um": 2.4},
        "properties": {"voxel_size_um": 2.4},
        "volume": {"resolution_um": 2.4}
    })")).has_value());
}

TEST_CASE("store metadata: a document with no acquisition record resolves nothing")
{
    // One published volume really is in this state: its metadata.json holds
    // only export and masking parameters.
    CHECK_FALSE(voxelSizeFromStoreMetadata(parse(R"({
        "zarr_export": {"z_crop_start": 0, "z_crop_end": 20819, "window_u16_max": 65535},
        "masking": {"method": "SAM2", "mask_scale": 8}
    })")).has_value());

    CHECK_FALSE(voxelSizeFromStoreMetadata(parse("{}")).has_value());
    CHECK_FALSE(voxelSizeFromStoreMetadata(parse("[]")).has_value());
    CHECK_FALSE(voxelSizeFromStoreMetadata(parse("null")).has_value());
}

TEST_CASE("store metadata: numbers written as strings are accepted")
{
    // source.resolution is a string in every published prediction, so string
    // numbers have to be read rather than rejected.
    const auto resolved = voxelSizeFromStoreMetadata(
        parse(R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":"0.00791"}}}}})"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == doctest::Approx(7.91));
}

TEST_CASE("store metadata: a non-positive or unparseable pixel size resolves nothing")
{
    for (const char* value : {"0", "-0.00791", R"("")", R"("7.91 um")", "true", "null", "{}"}) {
        const auto doc = parse(std::string(
            R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":)") + value + "}}}}}");
        CHECK_FALSE(voxelSizeFromStoreMetadata(doc).has_value());
    }
}
