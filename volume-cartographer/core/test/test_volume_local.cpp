// Local Volume metadata (meta.json). Zarr reading is vc_reader's and tested there.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/util/LoadJson.hpp"


#include <array>
#include <cstddef>
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
             ("vc_vol_local_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

vc::test::ZarrVolumeSpec makeOpts(std::array<size_t, 3> shape = {64, 64, 64},
                                   std::array<size_t, 3> chunks = {32, 32, 32},
                                   size_t numLevels = 2)
{
    vc::test::ZarrVolumeSpec opts;
    opts.shapeZYX = shape;
    opts.chunkShapeZYX = chunks;
    opts.numLevels = numLevels;
    opts.fillValue = 0.0;
    opts.voxelSize = 1.0;
    opts.uuid = "test-vol-uuid";
    opts.name = "test-vol";
    return opts;
}

void writeTextFile(const fs::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << text;
}

} // namespace

TEST_CASE("Volume: an unusable voxelsize field reports 0 instead of throwing")
{
    // voxelSize() promises 0 for "unknown". A document may state the key as
    // null, as text, or not at all, and each of those used to make the
    // accessor throw from deep inside whatever asked for a physical area.
    auto d = tmpDir("unusable_voxelsize");
    auto opts = makeOpts({32, 32, 32}, {16, 16, 16}, 1);
    vc::test::createZarrVolume(d, opts);
    const auto meta = vc::json::load_json_file(d / "meta.json");
    const auto metaWithVoxelSize = [&](const std::string& voxelsizeJson) {
        std::string text = R"({"type": "vol", "uuid": ")" + meta["uuid"].get_string() +
                           R"(", "width": 32, "height": 32, "slices": 32)";
        if (!voxelsizeJson.empty()) {
            text += R"(, "voxelsize": )" + voxelsizeJson;
        }
        return text + "}";
    };

    SUBCASE("meta.json without a voxelsize key")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize(""));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with a null voxelsize")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize("null"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with a non-positive voxelsize")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize("-1"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with voxelsize as unknown text")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize(R"("unknown")"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    fs::remove_all(d);
}
