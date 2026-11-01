// Streaming normal grids: a marker directory at the URL's place under a cache root,
// files fetched through vc_reader's file cache. Boundary checks against a loopback
// server: one download per file, concurrent misses overlap, what the store lacks is
// remembered in memory, low disk space stops caching and is reported, and the
// directory layout is enforced.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/MemMap.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

#include <vc_reader/zarr.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::NormalGridVolume;
using vc::test::LoopbackHttpServer;

namespace {

std::string gridName(int slice)
{
    char name[32];
    std::snprintf(name, sizeof name, "%06d.grid", slice);
    return name;
}

void writeIni(const fs::path& configDir, unsigned long long minFreeGib)
{
    std::ofstream(configDir / "VC3D.ini") << "[perf]\nremote_cache_min_free_gib=" << minFreeGib << "\n";
}

// A served normal-grid store, a cache root, and the marker directory at the store URL's place.
struct Streamed {
    fs::path scratch = fs::temp_directory_path() / ("vc_ngv_stream_" + std::to_string(vc::memmap::pid()));
    fs::path store = scratch / "store", config = scratch / "config", root = scratch / "cache";
    LoopbackHttpServer server;
    std::string url = server.url() + "/ng";
    fs::path dir = vcr::zarr::decodedCachePath(root, url);

    Streamed()
    {
        fs::remove_all(scratch);
        fs::create_directories(store / "xy");
        fs::create_directories(store / "xz");
        fs::create_directories(config);
        std::ofstream(store / "metadata.json") << R"({"sparse-volume": 1, "spiral-step": 20.0})";
        for (int slice = 0; slice < 8; ++slice) grid(store / "xy" / gridName(slice), slice);
        grid(store / "xz" / gridName(0), 0);
        server.serveDirectory(store, "/ng");
        fs::create_directories(dir);
        REQUIRE(vc::core::util::writeNormalGridsRemoteMarker(dir, url));
        // Settings come from this directory; the default 20 GiB floor must not decide these tests.
        setenv("VC3D_CONFIG_DIR", config.c_str(), 1);
        writeIni(config, 0);
    }
    ~Streamed()
    {
        std::error_code ec;
        fs::remove_all(scratch, ec);
    }
    static void grid(const fs::path& path, int slice)
    {
        vc::core::util::GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add({cv::Point(0, 0), cv::Point(slice % 50, 5), cv::Point(10, 10)});
        gs.save(path.string());
    }
    std::size_t fetches(const std::string& rel) const { return server.requestCount("/ng/" + rel); }
    std::vector<fs::path> temporaries() const
    {
        std::vector<fs::path> found;
        for (const auto& entry : fs::recursive_directory_iterator(root))
            if (entry.is_regular_file() && entry.path().filename().string().find(".tmp.") != std::string::npos)
                found.push_back(entry.path());
        return found;
    }
};

}  // namespace

TEST_CASE("Streaming store: each file downloads once, absent files are remembered, old markers are ignored")
{
    Streamed s;
    NormalGridVolume volume(s.dir.string());
    CHECK(s.fetches("metadata.json") == 1);
    CHECK(volume.outputSpiralStep() == 20.0);

    REQUIRE(volume.get_grid(0, 0) != nullptr);
    CHECK(fs::is_regular_file(s.dir / "xy" / "000000.grid"));
    CHECK(s.fetches("xy/000000.grid") == 1);
    REQUIRE(volume.get_grid(0, 0) != nullptr);
    CHECK(s.fetches("xy/000000.grid") == 1);

    // A slice the store does not have: one request, then remembered in memory, no marker file.
    CHECK(volume.get_grid(0, 42) == nullptr);
    CHECK(volume.get_grid(0, 42) == nullptr);
    CHECK(s.fetches("xy/000042.grid") == 1);
    CHECK_FALSE(fs::exists(s.dir / "xy" / "000042.grid.missing"));

    // A ".missing" marker left by an older VC3D does not stop the fetch.
    std::ofstream(s.dir / "xy" / "000003.grid.missing").flush();
    REQUIRE(volume.get_grid(0, 3) != nullptr);
    CHECK(s.fetches("xy/000003.grid") == 1);

    const auto between = volume.query(cv::Point3f(0.f, 0.f, 0.5f), 0);
    REQUIRE(between.has_value());
    CHECK(between->weight == doctest::Approx(0.5));
    CHECK(s.fetches("xy/000001.grid") == 1);
    CHECK(volume.cacheStats().fetchFailures == 0);

    // Another volume over the same directory reads the cached copies.
    NormalGridVolume again(s.dir.string());
    REQUIRE(again.get_grid(0, 0) != nullptr);
    CHECK(s.fetches("metadata.json") == 1);
    CHECK(s.fetches("xy/000000.grid") == 1);
    CHECK(s.temporaries().empty());
}

TEST_CASE("Streaming store: concurrent misses download in parallel, one request per file")
{
    Streamed s;
    // Four distinct files are requested by eight threads; the server releases them only
    // once all four downloads are in flight together, so serialized fetches time out.
    std::mutex mutex;
    std::condition_variable arrived;
    int inFlight = 0;
    s.server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (request.path.find("/ng/xy/00000") != 0) return std::nullopt;
        std::unique_lock lock(mutex);
        ++inFlight;
        arrived.notify_all();
        if (!arrived.wait_for(lock, std::chrono::seconds(10), [&] { return inFlight >= 4; }))
            return LoopbackHttpServer::Response{503, "downloads were serialized", {}, false};
        return std::nullopt;
    });
    NormalGridVolume volume(s.dir.string());
    std::vector<std::future<bool>> reads;
    for (int i = 0; i < 8; ++i)
        reads.push_back(std::async(std::launch::async, [&, i] { return volume.get_grid(0, 4 + i % 4) != nullptr; }));
    for (auto& read : reads) CHECK(read.get());
    for (int slice = 4; slice < 8; ++slice) CHECK(s.fetches("xy/" + gridName(slice)) == 1);
    CHECK(volume.cacheStats().fetchFailures == 0);
    CHECK(s.temporaries().empty());
}

TEST_CASE("Streaming store: low disk space stops caching, is reported, and deletes nothing")
{
    Streamed s;
    NormalGridVolume warm(s.dir.string());
    REQUIRE(warm.get_grid(0, 0) != nullptr);

    writeIni(s.config, 1000000000ULL);  // more free space than any disk has
    NormalGridVolume starved(s.dir.string());
    CHECK(starved.get_grid(1, 0) == nullptr);
    const auto stats = starved.cacheStats();
    CHECK(stats.fetchFailures == 1);
    CHECK(stats.lastFetchError.find("free") != std::string::npos);
    CHECK(stats.lastFetchError.find("nothing cached was removed") != std::string::npos);
    CHECK_FALSE(fs::exists(s.dir / "xz" / "000000.grid"));
    CHECK(fs::is_regular_file(s.dir / "xy" / "000000.grid"));
    CHECK(s.temporaries().empty());

    writeIni(s.config, 0);
    NormalGridVolume relieved(s.dir.string());
    REQUIRE(relieved.get_grid(1, 0) != nullptr);
    CHECK(fs::is_regular_file(s.dir / "xz" / "000000.grid"));
    CHECK(relieved.cacheStats().fetchFailures == 0);
}

TEST_CASE("Streaming store: a marker directory outside the cache layout is refused")
{
    Streamed s;
    const auto elsewhere = s.scratch / "elsewhere";
    fs::create_directories(elsewhere);
    REQUIRE(vc::core::util::writeNormalGridsRemoteMarker(elsewhere, s.url));
    CHECK_THROWS_WITH_AS(NormalGridVolume(elsewhere.string()), doctest::Contains("must be cached at <root>/"), std::runtime_error);
    CHECK(s.fetches("metadata.json") == 0);
}
