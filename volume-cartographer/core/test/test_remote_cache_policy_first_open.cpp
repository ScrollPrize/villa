// VC's cache settings reach the first read of a process: a CT volume opened before anything
// else has set the process-wide budget. Lasagna's equivalent is in test_lasagna_manifest.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "support/LoopbackHttpServer.hpp"

#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

TEST_CASE("a first CT open under a free-space floor reads on from the network and caches nothing")
{
    const auto root = fs::temp_directory_path() /
                      ("vc-first-open-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto store = root / "store" / "ct.zarr";
    vcr::zarr::Writer::create(store / "0", {.shape = {8, 8, 8}, .chunks = {8, 8, 8}, .compressor = ""});
    vcr::zarr::createGroup(store, nlohmann::json{{"multiscales", nlohmann::json::array({vcr::zarr::omeMultiscales({.levels = 0})})}});
    std::ofstream(store / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "first-open"}, {"format", "zarr"}, {"width", 8}, {"height", 8}, {"slices", 8}, {"voxelsize", 1.0}}.dump();

    vc::test::LoopbackHttpServer server;
    server.serveDirectory(root / "store");
    const auto config = root / "config";
    const auto cache = root / "cache";
    fs::create_directories(config);
    setenv("VC3D_CONFIG_DIR", config.c_str(), 1);
    std::ofstream(config / "cache.json") << nlohmann::json{{"version", 1}, {"root", cache.string()}}.dump();
    std::ofstream(config / "VC3D.ini") << "[perf]\nremote_cache_min_free_gib=1000000000\n";

    const std::string url = server.url() + "/ct.zarr";
    CHECK(Volume::NewFromUrl(url)->id() == "first-open");
    const auto cachedFiles = [&] {
        std::size_t count = 0;
        if (fs::exists(cache))
            for (const auto& entry : fs::recursive_directory_iterator(cache))
                if (entry.is_regular_file() && entry.path().filename().string().rfind(".z", 0) == 0) ++count;
        return count;
    };
    CHECK(cachedFiles() == 0);

    std::ofstream(config / "VC3D.ini", std::ios::trunc) << "[perf]\nremote_cache_min_free_gib=0\n";
    const auto volume = Volume::NewFromUrl(url);
    CHECK(volume->id() == "first-open");
    CHECK(cachedFiles() > 0);
    unsetenv("VC3D_CONFIG_DIR");
    fs::remove_all(root);
}
