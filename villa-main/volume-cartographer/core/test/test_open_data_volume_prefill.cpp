#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "OpenDataVolumePrefill.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto path = fs::temp_directory_path() /
                ("vc_open_data_prefill_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

vc3d::opendata::OpenDataVolumePrefillMarkerInfo markerInfo()
{
    vc3d::opendata::OpenDataVolumePrefillMarkerInfo info;
    info.remoteUrl = "https://example.invalid/sample/volume.zarr";
    info.volumeId = "volume-123";
    info.level = 5;
    info.shape = {64, 32, 16};
    info.chunkShape = {16, 16, 16};
    info.chunkGridShape = {4, 2, 1};
    info.totalChunks = 8;
    return info;
}

} // namespace

TEST_CASE("open-data prefill marker path uses the target level")
{
    const auto root = fs::path("/tmp/cache-root");
    CHECK(vc3d::opendata::openDataVolumePrefillMarkerPath(root, 5) ==
          root / ".vc_prefill_level_5.json");
    CHECK(vc3d::opendata::openDataVolumePrefillMarkerPath(root, 3) ==
          root / ".vc_prefill_level_3.json");
}

TEST_CASE("open-data prefill marker write and match round-trip")
{
    const auto root = tmpDir("round_trip");
    std::string error;
    const auto info = markerInfo();

    CHECK(vc3d::opendata::writeOpenDataVolumePrefillMarker(root, info, &error));
    CHECK(error.empty());
    CHECK(vc3d::opendata::openDataVolumePrefillMarkerMatches(root, info));
    CHECK(fs::is_regular_file(vc3d::opendata::openDataVolumePrefillMarkerPath(root, info.level)));

    fs::remove_all(root);
}

TEST_CASE("open-data prefill marker rejects stale metadata")
{
    const auto root = tmpDir("stale");
    auto info = markerInfo();
    REQUIRE(vc3d::opendata::writeOpenDataVolumePrefillMarker(root, info));

    auto changedUrl = info;
    changedUrl.remoteUrl = "https://example.invalid/other.zarr";
    CHECK_FALSE(vc3d::opendata::openDataVolumePrefillMarkerMatches(root, changedUrl));

    auto changedShape = info;
    changedShape.shape = {65, 32, 16};
    CHECK_FALSE(vc3d::opendata::openDataVolumePrefillMarkerMatches(root, changedShape));

    auto changedChunks = info;
    changedChunks.totalChunks = 9;
    CHECK_FALSE(vc3d::opendata::openDataVolumePrefillMarkerMatches(root, changedChunks));

    fs::remove_all(root);
}

TEST_CASE("open-data prefill marker ignores corrupt json")
{
    const auto root = tmpDir("corrupt");
    const auto info = markerInfo();
    fs::create_directories(root);
    {
        std::ofstream out(vc3d::opendata::openDataVolumePrefillMarkerPath(root, info.level));
        out << "{ not json";
    }

    CHECK_FALSE(vc3d::opendata::openDataVolumePrefillMarkerMatches(root, info));
    fs::remove_all(root);
}
