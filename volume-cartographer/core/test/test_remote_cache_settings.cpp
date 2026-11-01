// VC's remote cache settings as vc_reader open options: the root, its byte limit (0 means
// unlimited and lifts a stored one), the free-space floor and the download parallelism.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/MemMap.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

// First in the file: remoteCachePath resolves once per process.
TEST_CASE("remoteCachePath takes cache.json before the INI and creates the root")
{
    const auto config = fs::temp_directory_path() / ("vc_remote_cache_path_" + std::to_string(vc::memmap::pid()));
    fs::remove_all(config);
    fs::create_directories(config);
    setenv("VC3D_CONFIG_DIR", config.c_str(), 1);
    const fs::path root = config / "configured" / "cache";
    std::ofstream(config / "VC3D.ini") << "[viewer]\nremote_cache_dir=" << (config / "ini-cache").string() << "\n";
    std::ofstream(config / "cache.json") << "{\"version\":1,\"root\":\"" << root.string() << "\"}";
    CHECK(vc::settings::remoteCachePath() == root.lexically_normal());
    CHECK(fs::is_directory(root));
    CHECK_FALSE(fs::exists(config / "ini-cache"));
}

TEST_CASE("remoteOpenOptions maps the INI to vc_reader options")
{
    const auto config = fs::temp_directory_path() / ("vc_remote_cache_settings_" + std::to_string(vc::memmap::pid()));
    fs::remove_all(config);
    fs::create_directories(config);
    setenv("VC3D_CONFIG_DIR", config.c_str(), 1);
    constexpr std::int64_t gib = std::int64_t{1} << 30;
    const fs::path root = config / "cache";

    SUBCASE("defaults: unlimited, 20 GiB floor, parallelism left to vc_reader")
    {
        std::ofstream(config / "VC3D.ini") << "[perf]\n";
        const auto options = vc::settings::remoteOpenOptions(root);
        CHECK(options.cacheDirs == std::vector<std::string>{root.string()});
        CHECK(options.cacheMaxBytes == std::numeric_limits<std::int64_t>::max());
        CHECK(options.cacheMinFreeBytes == 20 * gib);
        CHECK(options.connections == 16);
        CHECK(options.threads == 0);
        CHECK(options.access.kind == vcr::zarr::Access::Kind::Anonymous);
    }
    SUBCASE("explicit limit, floor and parallelism")
    {
        std::ofstream(config / "VC3D.ini") << "[perf]\nremote_cache_max_gib=3\nremote_cache_min_free_gib=2\n"
                                              "remote_download_automatic=false\nremote_download_parallelism=8\n";
        const auto options = vc::settings::remoteOpenOptions(root);
        CHECK(options.cacheMaxBytes == 3 * gib);
        CHECK(options.cacheMinFreeBytes == 2 * gib);
        CHECK(options.connections == 8);
        CHECK(options.threads == 8);
        CHECK(vc::settings::remoteCacheMaxBytes() == 3 * gib);
    }
    SUBCASE("0 means unlimited, and a limit over 64 workers is clamped")
    {
        std::ofstream(config / "VC3D.ini") << "[perf]\nremote_cache_max_gib=0\nremote_cache_min_free_gib=0\n"
                                              "remote_download_automatic=0\nremote_download_parallelism=500\n";
        const auto options = vc::settings::remoteOpenOptions(root);
        CHECK(options.cacheMaxBytes == std::numeric_limits<std::int64_t>::max());
        CHECK_FALSE(vc::settings::remoteCacheMaxBytes().has_value());
        CHECK(options.cacheMinFreeBytes == 0);
        CHECK(options.connections == 64);
    }
    SUBCASE("an empty root sets the parallelism only")
    {
        std::ofstream(config / "VC3D.ini") << "[perf]\nremote_download_automatic=false\nremote_download_parallelism=4\n";
        const auto options = vc::settings::remoteOpenOptions({});
        CHECK(options.cacheDirs.empty());
        CHECK_FALSE(options.cacheMaxBytes.has_value());
        CHECK_FALSE(options.cacheMinFreeBytes.has_value());
        CHECK(options.connections == 4);
    }
    unsetenv("VC3D_CONFIG_DIR");
    fs::remove_all(config);
}
