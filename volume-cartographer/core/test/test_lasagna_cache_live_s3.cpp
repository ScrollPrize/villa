// Live S3: a remote Lasagna array cached whole through its read-through
// store reads as a local directory Zarr. Set VC_TEST_REQUIRE_NETWORK=1 to
// make failures hard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

// A coarse level of a published PHerc1447 normal channel (~14k objects).
constexpr const char* kArtifactUrl =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1447/"
    "representations/predictions/lasagna/20250521151220-lasagna-20260419180421";
constexpr const char* kArrayKey = "PHerc1447_nx.ome.zarr/4";

bool requireNetwork()
{
    const char* env = std::getenv("VC_TEST_REQUIRE_NETWORK");
    return env && env[0] && env[0] != '0';
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        std::random_device rd;
        path = fs::temp_directory_path() / ("vc_lasagna_cache_" + std::to_string(rd()));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

} // namespace

TEST_CASE("a remote Lasagna array is cached whole as a directory Zarr")
{
    TempDir cache;
    vc::lasagna::LasagnaChannelGroup group;
    group.name = "nx";
    group.relativeZarrKey = kArrayKey;
    group.remoteZarrBaseUrl = kArtifactUrl;
    group.remoteZarrKey = kArrayKey;
    group.remoteCacheRoot = cache.path;
    group.discoverAwsCredentials = false;

    const auto directory = vc::lasagna::lasagnaChannelArrayCacheDirectory(group);
    REQUIRE(directory);
    CHECK(*directory == (cache.path / kArrayKey).lexically_normal());

    std::size_t reported = 0;
    try {
        REQUIRE(vc::lasagna::cacheLasagnaChannelArray(
            group, [&](const vc::lasagna::LasagnaArrayCacheProgress& progress) { reported = progress.total; }));
    } catch (const std::exception& error) {
        if (requireNetwork()) FAIL(error.what());
        MESSAGE("skipping, S3 unreachable: " << error.what());
        return;
    }
    CHECK(reported > 0);

    // The metadata sits beside the chunks, and the marker lists every object.
    REQUIRE(fs::is_regular_file(*directory / ".zarray"));
    const auto meta = nlohmann::json::parse(std::ifstream(*directory / ".zarray"));
    CHECK(meta.at("shape").size() == 3);
    const auto marker = nlohmann::json::parse(std::ifstream(*directory / ".vc-cached-array.json"));
    const auto& objects = marker.at("objects");
    CHECK(objects.size() == reported);
    for (const auto& object : objects)
        REQUIRE(fs::file_size(*directory / object.at(0).get<std::string>()) == object.at(1).get<std::uint64_t>());

    // A complete cache fetches nothing; an evicted chunk is fetched again.
    reported = 0;
    REQUIRE(vc::lasagna::cacheLasagnaChannelArray(
        group, [&](const vc::lasagna::LasagnaArrayCacheProgress& progress) { reported = progress.total; }));
    CHECK(reported == 0);
    std::string chunk;
    for (const auto& object : objects) {
        const auto name = object.at(0).get<std::string>();
        if (name.front() != '.') {
            chunk = name;
            break;
        }
    }
    REQUIRE(!chunk.empty());
    fs::remove(*directory / chunk);
    REQUIRE(vc::lasagna::cacheLasagnaChannelArray(
        group, [&](const vc::lasagna::LasagnaArrayCacheProgress& progress) { reported = progress.total; }));
    CHECK(reported == 1);
    CHECK(fs::is_regular_file(*directory / chunk));
}

TEST_CASE("caching stops when cancelled")
{
    TempDir cache;
    vc::lasagna::LasagnaChannelGroup group;
    group.name = "nx";
    group.remoteZarrBaseUrl = kArtifactUrl;
    group.remoteZarrKey = kArrayKey;
    group.remoteCacheRoot = cache.path;
    group.discoverAwsCredentials = false;
    CHECK_FALSE(vc::lasagna::cacheLasagnaChannelArray(group, {}, []() { return true; }));
    CHECK_FALSE(fs::exists(cache.path / kArrayKey / ".vc-cached-array.json"));
}
