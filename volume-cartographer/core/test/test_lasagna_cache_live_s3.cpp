// Live S3: a remote Lasagna array cached whole, byte for byte, reads as a
// local directory Zarr. Set VC_TEST_REQUIRE_NETWORK=1 to
// make failures hard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"

#include <vc_reader/zarr.hpp>

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

vc::lasagna::LasagnaChannelGroup remoteGroup(const fs::path& cacheRoot)
{
    vc::lasagna::LasagnaChannelGroup group;
    group.name = "nx";
    group.relativeZarrKey = kArrayKey;
    group.sourceLocation = std::string(kArtifactUrl) + "/" + kArrayKey;
    group.remoteCacheRoot = cacheRoot;
    group.discoverAwsCredentials = false;
    return group;
}

} // namespace

TEST_CASE("a remote Lasagna array is cached whole as a directory Zarr")
{
    TempDir cache;
    const auto group = remoteGroup(cache.path);

    const auto directory = vc::lasagna::lasagnaChannelArrayCacheDirectory(group);
    REQUIRE(directory);
    // Beside, not inside, vc_reader's cache of the same array.
    CHECK(*directory == vcr::zarr::decodedCachePath(cache.path / "lasagna_arrays", group.sourceLocation));

    std::size_t reported = 0;
    const auto report = [&](const vc::lasagna::LasagnaArrayCacheProgress& progress) { reported = progress.total; };
    try {
        REQUIRE(vc::lasagna::cacheLasagnaChannelArray(group, report));
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

    // A complete cache fetches nothing; an evicted
    // chunk is fetched again.
    reported = 0;
    REQUIRE(vc::lasagna::cacheLasagnaChannelArray(group, report));
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
    REQUIRE(vc::lasagna::cacheLasagnaChannelArray(group, report));
    CHECK(reported == 1);
    CHECK(fs::is_regular_file(*directory / chunk));
}

TEST_CASE("caching stops when cancelled")
{
    TempDir cache;
    const auto group = remoteGroup(cache.path);
    CHECK_FALSE(vc::lasagna::cacheLasagnaChannelArray(group, {}, []() { return true; }));
    CHECK_FALSE(fs::exists(*vc::lasagna::lasagnaChannelArrayCacheDirectory(group) / ".vc-cached-array.json"));
}
