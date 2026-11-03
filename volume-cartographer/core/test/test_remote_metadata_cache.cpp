#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/RemoteMetadataCache.hpp"

#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::RemoteMetadataCache;

namespace
{

constexpr auto kVolume =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercX/volumes/a.zarr";

fs::path temporaryDirectory()
{
    std::mt19937_64 rng(std::random_device{}());
    const auto path = fs::temp_directory_path() / ("vc_remote_metadata_cache_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

RemoteMetadataCache::Answer bytes(std::string_view text)
{
    std::vector<std::byte> out(text.size());
    std::memcpy(out.data(), text.data(), text.size());
    return out;
}

}  // namespace

TEST_CASE("only Open Data public-channel metadata URLs are cacheable")
{
    CHECK(RemoteMetadataCache::cacheable(std::string(kVolume) + "/0/.zarray"));
    CHECK(RemoteMetadataCache::cacheable("s3://vesuvius-challenge-open-data/PHercX/volumes/a.zarr/.zattrs"));
    CHECK(RemoteMetadataCache::cacheable("https://data.aws.ash2txt.org/samples/PHercX/a.zarr/.zgroup"));
    // Other origins, local test servers and queries (S3 listings) are not.
    CHECK_FALSE(RemoteMetadataCache::cacheable("https://example.com/a.zarr/.zarray"));
    CHECK_FALSE(RemoteMetadataCache::cacheable("http://127.0.0.1:8080/a.zarr/.zarray"));
    CHECK_FALSE(RemoteMetadataCache::cacheable("s3://private-bucket/a.zarr/.zarray"));
    CHECK_FALSE(RemoteMetadataCache::cacheable(
        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/?list-type=2&prefix=PHercX/"));
}

TEST_CASE("a later session answers present and absent objects from disk")
{
    const auto root = temporaryDirectory();
    const std::string zarray = std::string(kVolume) + "/0/.zarray";
    const std::string zarrJson = std::string(kVolume) + "/0/zarr.json";
    {
        RemoteMetadataCache cache(root);
        CHECK_FALSE(cache.lookup(zarray).has_value());
        cache.record(zarray, bytes(R"({"zarr_format":2})"));
        cache.record(zarrJson, std::nullopt);
    }

    RemoteMetadataCache nextSession(root);
    const auto present = nextSession.lookup(zarray);
    REQUIRE(present.has_value());
    REQUIRE(present->has_value());
    CHECK(**present == *bytes(R"({"zarr_format":2})"));
    const auto absent = nextSession.lookup(zarrJson);
    REQUIRE(absent.has_value());
    CHECK_FALSE(absent->has_value());

    // An object that later appears replaces its absence.
    nextSession.record(zarrJson, bytes("{}"));
    RemoteMetadataCache third(root);
    REQUIRE(third.lookup(zarrJson).has_value());
    CHECK(third.lookup(zarrJson)->has_value());
    fs::remove_all(root);
}

TEST_CASE("invalidating an artifact forgets only its own objects")
{
    const auto root = temporaryDirectory();
    const std::string inside = std::string(kVolume) + "/0/.zarray";
    const std::string sibling = std::string(kVolume) + "2/0/.zarray";
    RemoteMetadataCache cache(root);
    cache.record(inside, bytes("a"));
    cache.record(sibling, bytes("b"));

    cache.invalidate(std::string(kVolume) + "/");

    CHECK_FALSE(cache.lookup(inside).has_value());
    CHECK_FALSE(RemoteMetadataCache(root).lookup(inside).has_value());
    REQUIRE(cache.lookup(sibling).has_value());
    CHECK(RemoteMetadataCache(root).lookup(sibling).has_value());
    fs::remove_all(root);
}
