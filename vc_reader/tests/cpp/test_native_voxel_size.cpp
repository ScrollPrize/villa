// Coverage for voxel-size resolution from volume-store metadata. Every schema
// case below is a real document shape from the Vesuvius open-data catalog,
// reduced to the fields that matter. The failure this guards is #1603: a
// volume whose voxel size cannot be resolved reports 0, and every physical
// measurement derived from it silently becomes 0 too.

#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using vcr::zarr::Json;
using vcr::zarr::voxelSizeUm;

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<std::pair<std::string, std::function<void()>>>& cases()
{
    static std::vector<std::pair<std::string, std::function<void()>>> all;
    return all;
}

struct Register {
    Register(const char* name, void (*run)()) { cases().emplace_back(name, run); }
};

struct Approx {
    double value;
    friend bool operator==(double actual, Approx expected)
    {
        return std::abs(actual - expected.value) <= 1e-9 * std::max(1.0, std::abs(expected.value));
    }
};

} // namespace

#define JOIN2(a, b) a##b
#define JOIN(a, b) JOIN2(a, b)
#define TEST_CASE(name) \
    static void JOIN(run, __LINE__)(); \
    static const Register JOIN(reg, __LINE__)(name, JOIN(run, __LINE__)); \
    static void JOIN(run, __LINE__)()
// Subcases of one test share its directory `d`: each starts it empty.
#define SUBCASE(name) if (std::filesystem::remove_all(d), std::filesystem::create_directories(d), true)
#define CHECK(...) require((__VA_ARGS__), #__VA_ARGS__)
#define REQUIRE(...) require((__VA_ARGS__), #__VA_ARGS__)
#define CHECK_FALSE(...) require(!(__VA_ARGS__), "not " #__VA_ARGS__)

namespace
{

Json parse(const std::string& text) { return Json::parse(text); }

std::filesystem::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = std::filesystem::temp_directory_path() /
             ("vc_voxel_size_" + tag + "_" + std::to_string(rng()));
    std::filesystem::create_directories(p);
    return p;
}

void writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << text;
}

// A scan record without the `scan` wrapper. samplePixelSize is in mm.
std::string scanRecord(double samplePixelSizeMm)
{
    return R"({"tomo":{"acquisition":{"detector":{"samplePixelSize":)" +
           std::to_string(samplePixelSizeMm) + R"(}}}})";
}

} // namespace

TEST_CASE("store metadata: source volume with a scan-wrapped acquisition record")
{
    const auto resolved = voxelSizeUm(
        parse(R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":0.00791}}}}})"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{7.91});
}

TEST_CASE("store metadata: fused mosaic export with the record at the document root")
{
    // The 1.129 um mosaics publish the same record without the `scan` wrapper,
    // and alongside two numbers a search for anything resolution-shaped would
    // find first: a mask downsample factor and a z crop range.
    const auto resolved = voxelSizeUm(parse(R"({
        "masking": {"mask_scale": 8, "dilation_iterations": "1"},
        "zarr_export": {"z_crop_start": 16, "z_crop_end": 42208},
        "mosaic": {"tile_count": 7},
        "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.001129}}}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{1.129});
}

TEST_CASE("store metadata: prediction run on level 0 of its source volume")
{
    const auto resolved = voxelSizeUm(parse(R"({
        "kind": "surface-inference",
        "source": {"resolution": "0", "metadata": )" + scanRecord(0.00864) + R"(}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{8.64});
}

TEST_CASE("store metadata: prediction run on level 2 is four times its source voxel size")
{
    // Ignoring the level reports 2.4 um instead of 9.6, which makes every
    // derived area 16x too small -- small enough to fall under a min_area_cm
    // gate and be discarded as noise.
    const auto resolved = voxelSizeUm(parse(R"({
        "kind": "surface-inference",
        "source": {
            "resolution": "2",
            "metadata": {"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.0024}}}}}
        }
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{9.6});
}

TEST_CASE("store metadata: an unreadable source level resolves nothing")
{
    // Falling back to level 0 would report a plausible number that is wrong by
    // an unknown power of two, which is worse than reporting none.
    for (const char* level : {R"("high")", "-1", "2.5", "null", "true", "99"}) {
        const auto doc = parse(std::string(R"({"source": {"resolution": )") + level +
                               R"(, "metadata": )" + scanRecord(0.0024) + "}}");
        CHECK_FALSE(voxelSizeUm(doc).has_value());
    }
}

TEST_CASE("store metadata: a derived store must state its source level")
{
    CHECK_FALSE(voxelSizeUm(parse(R"({
        "kind": "surface-inference",
        "source": {"metadata": {"scan": {"tomo": {"acquisition": {"detector":
                   {"samplePixelSize": 0.0024}}}}}}
    })")).has_value());
}

TEST_CASE("store metadata: an explicit voxelsize wins over an acquisition record")
{
    const auto resolved = voxelSizeUm(parse(R"({
        "voxelsize": 3.24,
        "scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00791}}}}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{3.24});
}

TEST_CASE("store metadata: the alternative spellings of an explicit voxel size")
{
    for (const char* key : {"voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um",
                            "pixelSizeUm", "resolution_um"}) {
        const auto resolved = voxelSizeUm(
            parse(std::string(R"({")") + key + R"(": 2.4})"));
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{2.4});
    }
}

TEST_CASE("store metadata: a nested pixel size is not read as this volume's")
{
    // On a derived store `metadata` holds the source volume's record. A pixel
    // size found there is the source's, at whatever pyramid level the run
    // consumed. Only `source.metadata` plus `source.resolution` together carry
    // enough to answer, and that path is tested above.
    CHECK_FALSE(voxelSizeUm(parse(R"({
        "metadata": {"pixel_size_um": 2.4},
        "properties": {"voxel_size_um": 2.4},
        "volume": {"resolution_um": 2.4}
    })")).has_value());
}

TEST_CASE("store metadata: a document with no acquisition record resolves nothing")
{
    // One published volume really is in this state: its metadata.json holds
    // only export and masking parameters.
    CHECK_FALSE(voxelSizeUm(parse(R"({
        "zarr_export": {"z_crop_start": 0, "z_crop_end": 20819, "window_u16_max": 65535},
        "masking": {"method": "SAM2", "mask_scale": 8}
    })")).has_value());

    CHECK_FALSE(voxelSizeUm(parse("{}")).has_value());
    CHECK_FALSE(voxelSizeUm(parse("[]")).has_value());
    CHECK_FALSE(voxelSizeUm(parse("null")).has_value());
}

TEST_CASE("store metadata: numbers written as strings are accepted")
{
    // source.resolution is a string in every published prediction, so string
    // numbers have to be read rather than rejected.
    const auto resolved = voxelSizeUm(
        parse(R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":"0.00791"}}}}})"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{7.91});
}

TEST_CASE("store metadata: a non-positive or unparseable pixel size resolves nothing")
{
    for (const char* value : {"0", "-0.00791", R"("")", R"("7.91 um")", "true", "null", "{}"}) {
        const auto doc = parse(std::string(
            R"({"scan":{"tomo":{"acquisition":{"detector":{"samplePixelSize":)") + value + "}}}}}");
        CHECK_FALSE(voxelSizeUm(doc).has_value());
    }
}

TEST_CASE("store metadata: the legacy scan-wrapped voxelsize")
{
    // The shape the renderer's own pre-patch reader handled as its second
    // candidate: metadata.json -> scan -> voxelsize. Micrometers, like the
    // top-level field it mirrors -- the same reader produced both and applied no
    // unit conversion to either.
    const auto resolved = voxelSizeUm(
        parse(R"({"scan": {"voxelsize": 8.64}, "zarr_export": {"z_crop_start": 0}})"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{8.64});
}

TEST_CASE("store metadata: the legacy scan voxelsize beats an acquisition record in the same scan")
{
    // The historical precedence, which this must not reinterpret. The pre-patch
    // reader was:
    //
    //     tryFile(meta.json,             nullptr)   // meta.json root voxelsize
    //     tryFile(metadata.json,         "scan")    // <- root["scan"]["voxelsize"]
    //     tryFile(metadata.json,         nullptr)   // metadata.json root voxelsize
    //
    // It had no acquisition-record branch at all, so when a document carried both
    // `scan.voxelsize` and `scan.tomo...samplePixelSize` it returned
    // `scan.voxelsize`. Preserving previously supported local metadata means
    // preserving that choice, not replacing it with the modern field.
    //
    // Deliberately distinct numbers: 7.91 from the legacy field, 8.64 from the
    // acquisition record, so the assertion cannot pass by coincidence.
    const auto resolved = voxelSizeUm(parse(R"({
        "scan": {
            "voxelsize": 7.91,
            "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00864}}}
        }
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{7.91});
}

TEST_CASE("store metadata: a top-level voxelsize still beats the scan-wrapped one")
{
    // ... while the *first* historical candidate keeps its own priority. A
    // document stating both must resolve to the top-level value.
    const auto resolved = voxelSizeUm(parse(R"({
        "voxelsize": 2.4,
        "scan": {"voxelsize": 7.91}
    })"));
    REQUIRE(resolved.has_value());
    CHECK(*resolved == Approx{2.4});
}

TEST_CASE("store metadata: an unrelated nested voxelsize is not read")
{
    // Exact-path matching, not a search. Each of these looks resolution-shaped
    // and none of them is this volume's own scan-wrapped voxelsize.
    for (const char* doc : {
             R"({"metadata": {"scan": {"voxelsize": 8.64}}})",
             R"({"source": {"scan": {"voxelsize": 8.64}}})",
             R"({"properties": {"voxelsize": 8.64}})",
             R"({"scan": [{"voxelsize": 8.64}]})",
             R"({"scan": {"properties": {"voxelsize": 8.64}}})",
         }) {
        CHECK_FALSE(voxelSizeUm(parse(doc)).has_value());
    }
}

TEST_CASE("store metadata: a non-positive legacy scan voxelsize resolves nothing")
{
    for (const char* value : {"0", "-8.64", R"("")", R"("8.64 um")", "true", "null", "{}"}) {
        const auto doc = parse(std::string(R"({"scan": {"voxelsize": )") + value + "}}");
        CHECK_FALSE(voxelSizeUm(doc).has_value());
    }
}

TEST_CASE("storeVoxelSizeUm: a local store")
{
    const auto d = tmpDir("local_store");

    SUBCASE("nothing on disk resolves nothing, without throwing")
    {
        CHECK_FALSE(vcr::zarr::storeVoxelSizeUm(d.string()).has_value());
        CHECK_FALSE(vcr::zarr::storeVoxelSizeUm((d / "does_not_exist").string()).has_value());
    }

    SUBCASE("meta.json is authoritative over metadata.json")
    {
        writeTextFile(d / "meta.json", R"({"type": "vol", "voxelsize": 7.91})");
        writeTextFile(d / "metadata.json",
                      R"({"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.0024}}}}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{7.91});
    }

    SUBCASE("a published metadata.json is read when there is no meta.json")
    {
        writeTextFile(d / "metadata.json",
                      R"({"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.0024}}}}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{2.4});
    }

    SUBCASE("a meta.json with no usable size falls through to metadata.json")
    {
        // The reviewer finding: many published meta.json files state dimensions
        // and nothing else. Stopping at the first existing file lost the
        // resolution the store does publish.
        writeTextFile(d / "meta.json",
                      R"({"height": 9100, "width": 6700, "slices": 21000, "type": "vol"})");
        writeTextFile(d / "metadata.json",
                      R"({"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00864}}}}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{8.64});
    }

    SUBCASE("a meta.json whose size is present but unusable falls through too")
    {
        // Present but non-positive is "no usable size", not "size zero".
        writeTextFile(d / "meta.json", R"({"type": "vol", "voxelsize": 0})");
        writeTextFile(d / "metadata.json", R"({"scan": {"voxelsize": 8.64}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{8.64});
    }

    SUBCASE("a malformed meta.json falls through to metadata.json")
    {
        writeTextFile(d / "meta.json", "{not json");
        writeTextFile(d / "metadata.json",
                      R"({"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": 0.0024}}}}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{2.4});
    }

    SUBCASE("a malformed metadata.json alongside an unusable meta.json resolves nothing")
    {
        // Falling through must not invent an answer when the last candidate is
        // also unusable. One corrupt file no longer hides a good one; two do.
        writeTextFile(d / "meta.json", "{not json");
        writeTextFile(d / "metadata.json", "also not json");
        CHECK_FALSE(vcr::zarr::storeVoxelSizeUm(d.string()).has_value());
    }

    SUBCASE("a malformed document with no sibling counts as absent")
    {
        writeTextFile(d / "meta.json", "{not json");
        CHECK_FALSE(vcr::zarr::storeVoxelSizeUm(d.string()).has_value());
    }

    SUBCASE("the legacy scan-wrapped voxelsize resolves through a local store")
    {
        writeTextFile(d / "metadata.json",
                      R"({"scan": {"voxelsize": 7.91}, "zarr_export": {"z_crop_end": 42208}})");
        const auto resolved = vcr::zarr::storeVoxelSizeUm(d.string());
        REQUIRE(resolved.has_value());
        CHECK(*resolved == Approx{7.91});
    }

    std::filesystem::remove_all(d);
}

int main()
{
    for (const auto& [name, run] : cases()) {
        try {
            run();
        } catch (const std::exception& error) {
            std::cerr << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    // A numbered level finds its store's documents.
    const auto store = tmpDir("level");
    std::filesystem::create_directories(store / "3");
    writeTextFile(store / "meta.json", R"({"voxelsize": 7.91})");
    require(vcr::zarr::storeVoxelSizeUm((store / "3").string()) == 7.91, "a level reads its store's documents");
    std::filesystem::remove_all(store);
    std::cout << cases().size() << " voxel size cases passed\n";
}
