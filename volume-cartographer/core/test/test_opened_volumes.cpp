// VC3D tools share one opened array per volume level: repeated requests return the
// same array, levels sample the stored voxels, and a coarser level's indices follow
// its placement (scale-only levels are block-reduced, half a voxel off-centre).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../../apps/VC3D/OpenedVolumes.hpp"
#include "support/LoopbackHttpServer.hpp"
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>

#include <vc_reader/array.hpp>

#include <chrono>
#include <vector>

namespace fs = std::filesystem;

TEST_CASE("OpenedVolumes shares opened levels and maps coordinates through placements")
{
    const auto dir = fs::temp_directory_path() /
                     ("vc-opened-volumes-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::filesystem::path volumeRoot1 = dir;
    const vcr::zarr::CreateOptions levelZero1{.shape = {16, 16, 16}, .chunks = {8, 8, 8}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot1 / "0", levelZero1);
    vcr::zarr::createPyramidLevels(volumeRoot1, levelZero1, {.levels = 1});
    vcr::zarr::createGroup(volumeRoot1, nlohmann::json{{"multiscales", nlohmann::json::array({vcr::zarr::omeMultiscales({.levels = 1})})}});
    auto volume = Volume::New(volumeRoot1);
    REQUIRE(volume);
    std::vector<std::uint8_t> level0(16 * 16 * 16), level1(8 * 8 * 8);
    for (std::size_t i = 0; i < level0.size(); ++i)
        level0[i] = std::uint8_t(i % 251);
    for (std::size_t i = 0; i < level1.size(); ++i)
        level1[i] = std::uint8_t(7 * i % 251);
    vcr::zarr::Writer::open(dir / "0").writeRegion({0, 0, 0}, {16, 16, 16}, level0.data());
    vcr::zarr::Writer::open(dir / "1").writeRegion({0, 0, 0}, {8, 8, 8}, level1.data());

    OpenedVolumes opened(64 << 20);
    const auto& pyramid = opened.pyramid(*volume);
    CHECK(pyramid.levels == std::vector<int>{0, 1});
    CHECK(pyramid.count() == 2);
    CHECK(pyramid.has(1));
    CHECK_FALSE(pyramid.has(2));
    CHECK(pyramid.dtype == "|u1");
    const auto a = opened.level(*volume, 0);
    const auto b = opened.level(*volume, 0);
    CHECK(a.array == b.array);

    const auto coarse = opened.level(*volume, 1);
    CHECK(coarse.array != a.array);
    CHECK(coarse.working.scale[0] == doctest::Approx(1.0));
    // Level-0 voxel 4.5 is the centre of level-1 voxel 2.
    CHECK(coarse.array->placement.fromLevel0(0, coarse.working.toLevel0(0, 4.5)) == doctest::Approx(2.0));
    fs::remove_all(dir);
}

TEST_CASE("OpenedVolumes describes each working base of one source, sharing the physical arrays")
{
    // Physical levels 0, 1, 2 served over loopback HTTP; working base 0 sees three working
    // levels, base 1 two, whichever is asked first, and both bases open the same arrays.
    const auto dir = fs::temp_directory_path() /
                     ("vc-opened-volumes-bases-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::filesystem::path volumeRoot2 = dir;
    const vcr::zarr::CreateOptions levelZero2{.shape = {16, 16, 16}, .chunks = {8, 8, 8}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot2 / "0", levelZero2);
    vcr::zarr::createPyramidLevels(volumeRoot2, levelZero2, {.levels = 2});
    vcr::zarr::createGroup(volumeRoot2, nlohmann::json{{"multiscales", nlohmann::json::array({vcr::zarr::omeMultiscales({.levels = 2})})}});
    std::ofstream(volumeRoot2 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "bases"}, {"name", "bases"}, {"format", "zarr"}, {"width", 16}, {"height", 16}, {"slices", 16}, {"voxelsize", 1.0}}.dump();
    REQUIRE(Volume::New(volumeRoot2));
    vc::test::LoopbackHttpServer server;
    server.serveDirectory(dir, "/vol.zarr");
    const std::string url = server.url() + "/vol.zarr";

    const auto describe = [&](OpenedVolumes& opened, const std::vector<int>& bases) {
        std::vector<std::vector<int>> levels;
        std::vector<const vcr::Array*> physical1;
        for (const int base : bases) {
            const auto volume = Volume::NewFromUrl(base ? url + "#vc-base-scale=" + std::to_string(base) : url);
            REQUIRE(volume->baseScaleLevel() == base);
            levels.push_back(opened.pyramid(*volume).levels);
            // Working level 1 - base of each volume is physical level 1.
            physical1.push_back(opened.level(*volume, 1 - base).array->array.get());
            CHECK(opened.pyramid(*volume).levels == levels.back());  // a repeated hit
        }
        CHECK(physical1[0] == physical1[1]);
        return levels;
    };
    {
        OpenedVolumes opened(64 << 20);
        const auto levels = describe(opened, {0, 1});
        CHECK(levels[0] == std::vector<int>{0, 1, 2});
        CHECK(levels[1] == std::vector<int>{0, 1});
    }
    {
        OpenedVolumes opened(64 << 20);
        const auto levels = describe(opened, {1, 0});
        CHECK(levels[0] == std::vector<int>{0, 1});
        CHECK(levels[1] == std::vector<int>{0, 1, 2});
    }
    fs::remove_all(dir);
}

TEST_CASE("OpenedVolumes gives only uint8 CT a GPU reader, and a listed level that is gone throws")
{
    const auto dir = fs::temp_directory_path() /
                     ("vc-opened-volumes-levels-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const vcr::zarr::CreateOptions wide{.shape = {16, 16, 16}, .chunks = {8, 8, 8}, .dtype = "<u2", .compressor = ""};
    vcr::zarr::Writer::create(dir / "0", wide);
    vcr::zarr::createPyramidLevels(dir, wide, {.levels = 1});
    vcr::zarr::createGroup(dir, nlohmann::json{{"multiscales", nlohmann::json::array({vcr::zarr::omeMultiscales({.levels = 1})})}});
    const auto volume = Volume::New(dir);
    {
        OpenedVolumes opened(64 << 20, 64 << 20);
        CHECK(opened.pyramid(*volume).dtype == "<u2");
        CHECK_FALSE(opened.level(*volume, 0).cuda);
    }
    fs::remove(dir / "1" / ".zarray");
    OpenedVolumes opened(64 << 20);
    CHECK_THROWS(opened.pyramid(*volume));
    fs::remove_all(dir);
}
