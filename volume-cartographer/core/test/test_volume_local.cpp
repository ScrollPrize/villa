// Local Volume metadata (meta.json). Zarr reading is vc_reader's and tested there.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>
#include "vc/core/util/LoadJson.hpp"


#include <array>
#include <cstddef>
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
    const std::filesystem::path volumeRoot = d;
    const vcr::zarr::CreateOptions levelZero{.shape = {32, 32, 32}, .chunks = {16, 16, 16}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot / "0", levelZero);
    std::ofstream(volumeRoot / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "test-vol-uuid"}, {"name", "test-vol"}, {"format", "zarr"}, {"width", 32}, {"height", 32}, {"slices", 32}, {"voxelsize", 1.0}}.dump();
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

TEST_CASE("Volume: base dimensions and level shapes follow the pyramid's placement, not powers of two")
{
    // Level 1 is placed by an OME scale of 3. The base dimensions come from level 0 as
    // placed, and level 1's shape is checked against ceil(base / 3), so the volume opens.
    auto d = tmpDir("placement");
    vcr::zarr::MultiscalesOptions multiscales;
    multiscales.levels = 1;
    multiscales.factors = {3, 3, 3};
    vcr::zarr::createGroup(d, vcr::zarr::Json{{"multiscales", vcr::zarr::Json::array({vcr::zarr::omeMultiscales(multiscales)})}});
    vcr::zarr::CreateOptions level0;
    level0.shape = {9, 12, 15};
    level0.chunks = {3, 3, 3};
    (void)vcr::zarr::Writer::create(d / "0", level0);
    vcr::zarr::PyramidOptions pyramid;
    pyramid.levels = 1;
    pyramid.factors = {3, 3, 3};
    vcr::zarr::createPyramidLevels(d, level0, pyramid);
    REQUIRE(vcr::zarr::openArray((d / "1").string(), {}).meta.shape == vcr::Index{3, 4, 5});

    const auto v = Volume::New(d);
    CHECK(v->shape() == std::array<int, 3>{9, 12, 15});
    fs::remove_all(d);
}

TEST_CASE("Volume: metadata.json's scan is flattened, and a shape that disagrees with it throws")
{
    auto d = tmpDir("sidecar");
    vcr::zarr::Writer::create(d / "0", {.shape = {4, 6, 8}, .chunks = {4, 6, 8}, .compressor = ""});
    writeTextFile(d / "metadata.json",
                  R"({"scan": {"type": "vol", "uuid": "from-scan", "width": 8, "height": 6, "slices": 4}})");
    const auto v = Volume::New(d);
    CHECK(v->id() == "from-scan");
    CHECK(v->shape() == std::array<int, 3>{4, 6, 8});

    fs::remove(d / "metadata.json");
    writeTextFile(d / "meta.json", R"({"type": "vol", "uuid": "wrong", "width": 800, "height": 6, "slices": 4})");
    CHECK_THROWS_WITH_AS(Volume::New(d), doctest::Contains("does not match"), std::runtime_error);
    fs::remove_all(d);
}
