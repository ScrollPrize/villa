// Streaming normal grids: a marker directory at the URL's place under a cache root,
// files fetched through vc_reader's file cache. Boundary checks against a loopback
// server: one download per file, concurrent misses overlap, what the store lacks is
// remembered in memory, low disk space stops caching and is reported, and the
// directory layout is enforced.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/AnonymousPages.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/RemoteGridFile.hpp"

#include <vc_reader/zarr.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>

// Thread starts made by the thread that set threadStartsToFail fail with EAGAIN
// while it is > 0 (this program's pthread_create is found before libc's, so
// std::thread uses it too). Other threads -- the loopback server's -- start as usual.
static thread_local int threadStartsToFail = 0;
extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg)
{
    using Create = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static const auto real = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    if (threadStartsToFail > 0) {
        --threadStartsToFail;
        return EAGAIN;
    }
    return real(thread, attr, start, arg);
}

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
    fs::path scratch = fs::temp_directory_path() / ("vc_ngv_stream_" + std::to_string(vc::pages::pid()));
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
    {
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
    }  // a volume's destructor waits for its read-ahead opens
    CHECK(s.temporaries().empty());
    // Read-ahead fetched the neighbours of the slices asked for, once each.
    CHECK(s.fetches("xy/000002.grid") == 1);
    CHECK(s.fetches("xy/000004.grid") == 1);
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
    {
    NormalGridVolume volume(s.dir.string());
    std::vector<std::future<bool>> reads;
    for (int i = 0; i < 8; ++i)
        reads.push_back(std::async(std::launch::async, [&, i] { return volume.get_grid(0, 4 + i % 4) != nullptr; }));
    for (auto& read : reads) CHECK(read.get());
    for (int slice = 4; slice < 8; ++slice) CHECK(s.fetches("xy/" + gridName(slice)) == 1);
    CHECK(volume.cacheStats().fetchFailures == 0);
    }
    CHECK(s.temporaries().empty());
}

TEST_CASE("Streaming store: low disk space stops caching, is reported, and deletes nothing")
{
    Streamed s;
    {
    NormalGridVolume warm(s.dir.string());
    REQUIRE(warm.get_grid(0, 0) != nullptr);
    }  // the destructors wait for read-ahead, whose downloads hold temporaries

    writeIni(s.config, 1000000000ULL);  // more free space than any disk has
    {
    NormalGridVolume starved(s.dir.string());
    CHECK(starved.get_grid(1, 0) == nullptr);
    const auto stats = starved.cacheStats();
    CHECK(stats.fetchFailures == 1);
    CHECK(stats.lastFetchError.find("free") != std::string::npos);
    CHECK(stats.lastFetchError.find("nothing cached was removed") != std::string::npos);
    }
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

namespace {
// VC_NORMAL_GRID_RANGE_READS=1 for one scope.
struct RangeReads {
    RangeReads() { setenv("VC_NORMAL_GRID_RANGE_READS", "1", 1); }
    ~RangeReads() { unsetenv("VC_NORMAL_GRID_RANGE_READS"); }
};

std::vector<std::vector<cv::Point>> points(const std::vector<std::shared_ptr<std::vector<cv::Point>>>& paths)
{
    std::vector<std::vector<cv::Point>> out;
    for (const auto& p : paths) out.push_back(*p);
    return out;
}
}  // namespace

TEST_CASE("Range reads: grids are read in parts, never downloaded whole, and answer as whole files do")
{
    Streamed s;
    RangeReads on;
    std::atomic<int> wholeGets{0};
    s.server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (request.path.ends_with(".grid") && request.header("range").empty()) ++wholeGets;
        return std::nullopt;
    });
    {
    NormalGridVolume volume(s.dir.string());
    const auto grid = volume.get_grid(0, 0);
    REQUIRE(grid != nullptr);
    CHECK_FALSE(fs::exists(s.dir / "xy" / "000000.grid"));
    CHECK(fs::is_regular_file(s.dir / "xy" / "000000.grid.ranges"));
    vc::core::util::GridStore whole((s.store / "xy" / "000000.grid").string());
    for (const auto& rect : {cv::Rect(0, 0, 100, 100), cv::Rect(0, 0, 5, 5), cv::Rect(8, 8, 3, 3), cv::Rect(50, 50, 40, 40)})
        CHECK(points(grid->get(rect)) == points(whole.get(rect)));

    // A slice the store does not have: one request, then remembered.
    CHECK(volume.get_grid(0, 42) == nullptr);
    CHECK(volume.get_grid(0, 42) == nullptr);
    CHECK(s.fetches("xy/000042.grid") == 1);

    // query_nearest drives the read-ahead of the neighbouring slices' same region.
    REQUIRE(volume.query_nearest(cv::Point3f(5.f, 5.f, 2.f), 0) != nullptr);
    CHECK(volume.cacheStats().fetchFailures == 0);
    }  // the destructor waits for read-ahead
    CHECK(wholeGets == 0);
    CHECK(fs::is_regular_file(s.dir / "xy" / "000003.grid.ranges"));
    CHECK_FALSE(fs::exists(s.dir / "xy" / "000003.grid"));
    CHECK(s.temporaries().empty());

    // A later volume replays the pack: nothing fetched again.
    const auto before = s.fetches("xy/000000.grid");
    NormalGridVolume again(s.dir.string());
    REQUIRE(again.get_grid(0, 0) != nullptr);
    (void)again.get_grid(0, 0)->get(cv::Rect(0, 0, 100, 100));
    CHECK(s.fetches("xy/000000.grid") == before);
}

TEST_CASE("Range reads: under a tiny memory cap read-ahead files are dropped and replayed, answers unchanged")
{
    Streamed s;
    RangeReads on;
    setenv("VC_NORMAL_GRID_RANGE_MEMORY_BYTES", "1", 1);
    {
        NormalGridVolume volume(s.dir.string());
        for (int slice = 0; slice < 8; ++slice) {
            REQUIRE(volume.query_nearest(cv::Point3f(5.f, 5.f, float(slice)), 0) != nullptr);
            const auto grid = volume.get_grid(0, slice);
            REQUIRE(grid != nullptr);
            vc::core::util::GridStore whole((s.store / "xy" / gridName(slice)).string());
            CHECK(points(grid->get(cv::Rect(0, 0, 100, 100))) == points(whole.get(cv::Rect(0, 0, 100, 100))));
        }
        CHECK(volume.cacheStats().fetchFailures == 0);
    }
    unsetenv("VC_NORMAL_GRID_RANGE_MEMORY_BYTES");
    // What was dropped from memory is still in the packs: nothing is fetched again.
    std::vector<std::size_t> before;
    for (int slice = 0; slice < 8; ++slice) before.push_back(s.fetches("xy/" + gridName(slice)));
    NormalGridVolume again(s.dir.string());
    for (int slice = 0; slice < 8; ++slice) {
        const auto grid = again.get_grid(0, slice);
        REQUIRE(grid != nullptr);
        (void)grid->get(cv::Rect(0, 0, 100, 100));
        CHECK(s.fetches("xy/" + gridName(slice)) == before[slice]);
    }
}

TEST_CASE("Range reads: a store that ignores Range is downloaded whole, as without the option")
{
    Streamed s;
    RangeReads on;
    s.server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (!request.path.ends_with(".grid") || !fs::exists(s.store / request.path.substr(4))) return std::nullopt;
        std::ifstream in(s.store / request.path.substr(4), std::ios::binary);
        return LoopbackHttpServer::Response{200, std::string(std::istreambuf_iterator<char>(in), {}), {}, false};
    });
    NormalGridVolume volume(s.dir.string());
    const auto grid = volume.get_grid(0, 0);
    REQUIRE(grid != nullptr);
    CHECK(fs::is_regular_file(s.dir / "xy" / "000000.grid"));
    vc::core::util::GridStore whole((s.store / "xy" / "000000.grid").string());
    CHECK(points(grid->get(cv::Rect(0, 0, 100, 100))) == points(whole.get(cv::Rect(0, 0, 100, 100))));
}

TEST_CASE("Range reads: under the free-space floor grids are read into memory, not left out")
{
    Streamed s;
    RangeReads on;
    { NormalGridVolume warm(s.dir.string()); }  // the store's metadata is cached whole, as before
    writeIni(s.config, 1000000000ULL);  // more free space than any disk has
    const auto inMemory = vc::core::util::RemoteGridFile::globalStats().inMemory;
    {
    NormalGridVolume starved(s.dir.string());
    const auto grid = starved.get_grid(1, 0);
    REQUIRE(grid != nullptr);
    vc::core::util::GridStore whole((s.store / "xz" / "000000.grid").string());
    CHECK(points(grid->get(cv::Rect(0, 0, 100, 100))) == points(whole.get(cv::Rect(0, 0, 100, 100))));
    CHECK(starved.cacheStats().fetchFailures == 0);
    }
    CHECK(vc::core::util::RemoteGridFile::globalStats().inMemory > inMemory);
    CHECK_FALSE(fs::exists(s.dir / "xz" / "000000.grid"));
    CHECK_FALSE(fs::exists(s.dir / "xz" / "000000.grid.ranges"));
}

TEST_CASE("Range reads: when a grid's regions stop arriving, the grid is read whole, never as empty")
{
    Streamed s;
    RangeReads on;
    // A slice big enough that later regions need requests of their own.
    {
        vc::core::util::GridStore big(cv::Rect(0, 0, 4000, 3000), 32);
        for (int y = 5; y < 3000; y += 37)
            for (int x = 5; x < 4000; x += 41) big.add({cv::Point(x, y), cv::Point(x + 6, y + 3), cv::Point(x + 12, y + 1)});
        big.save((s.store / "xy" / gridName(5)).string());
    }
    std::atomic<bool> failRanges{false};
    std::atomic<int> failed{0}, wholeGets{0};
    s.server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (request.path != "/ng/xy/" + gridName(5)) return std::nullopt;
        if (request.header("range").empty()) ++wholeGets;
        else if (failRanges) {
            ++failed;
            return LoopbackHttpServer::Response{500, "range reads are down", {}, false};
        }
        return std::nullopt;
    });
    vc::core::util::GridStore whole((s.store / "xy" / gridName(5)).string());
    NormalGridVolume volume(s.dir.string());
    const auto grid = volume.get_grid(0, 5);  // opened by range reads
    REQUIRE(grid != nullptr);
    CHECK(wholeGets == 0);
    failRanges = true;
    for (const auto& rect : {cv::Rect(3000, 2000, 300, 300), cv::Rect(100, 2500, 200, 200), cv::Rect(0, 0, 4000, 3000)}) {
        const auto got = points(grid->get(rect));
        CHECK_FALSE(got.empty());
        CHECK(got == points(whole.get(rect)));
    }
    CHECK(failed > 0);
    CHECK(wholeGets == 1);  // read whole once, then every query answered from it
    CHECK(fs::is_regular_file(s.dir / "xy" / gridName(5)));
}

TEST_CASE("Range reads: in an outage a region that cannot be read is counted, not taken as empty, and reads after it")
{
    // As main's whole-file path serves a slice read before an outage, while a
    // region never fetched cannot be: the volume counts that (GrowPatch then
    // does not save), the grid's generation changes (GrowPatch's caches read
    // it again), and once the store answers again the region is read.
    Streamed s;
    RangeReads on;
    {
        vc::core::util::GridStore big(cv::Rect(0, 0, 4000, 3000), 32);
        for (int y = 5; y < 3000; y += 37)
            for (int x = 5; x < 4000; x += 41) big.add({cv::Point(x, y), cv::Point(x + 6, y + 3), cv::Point(x + 12, y + 1)});
        big.save((s.store / "xy" / gridName(5)).string());
    }
    std::atomic<bool> outage{false};
    s.server.setHandler([&](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (request.path != "/ng/xy/" + gridName(5) || !outage) return std::nullopt;
        return LoopbackHttpServer::Response{503, "down", {}, false};
    });
    vc::core::util::GridStore whole((s.store / "xy" / gridName(5)).string());
    NormalGridVolume volume(s.dir.string());
    const auto grid = volume.get_grid(0, 5);
    REQUIRE(grid != nullptr);
    const cv::Rect first(100, 100, 200, 200), second(3000, 2000, 300, 300);
    CHECK(points(grid->get(first)) == points(whole.get(first)));
    CHECK(volume.readFailures() == 0);
    outage = true;
    const auto generation = grid->generation();
    CHECK(grid->get(second).empty());
    CHECK(volume.readFailures() == 1);
    CHECK(grid->generation() != generation);
    outage = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    CHECK(points(grid->get(second)) == points(whole.get(second)));
    CHECK(volume.readFailures() == 1);  // what failed stays counted: a patch grown meanwhile is not saved
}

// Runs `work` on its own thread; nullopt when it has not finished within `limit`
// (the thread is left behind, so a hang fails the test instead of stalling it).
template <class Work>
std::optional<std::vector<std::vector<cv::Point>>> within(std::chrono::seconds limit, Work work)
{
    auto done = std::make_shared<std::promise<std::vector<std::vector<cv::Point>>>>();
    auto result = done->get_future();
    std::thread([done, work] { done->set_value(work()); }).detach();
    if (result.wait_for(limit) != std::future_status::ready) return std::nullopt;
    return result.get();
}

TEST_CASE("Range reads: when threads cannot start, queries still answer and the volume closes")
{
    Streamed s;
    RangeReads on;
    auto query = [&](int fail, float z) {
        return [&s, fail, z] {
            NormalGridVolume volume(s.dir.string());
            threadStartsToFail = fail;
            const auto grid = volume.query_nearest(cv::Point3f(5.f, 5.f, z), 0);
            auto got = grid ? points(grid->get(cv::Rect(0, 0, 100, 100))) : std::vector<std::vector<cv::Point>>{};
            threadStartsToFail = 0;
            return got;  // the volume closes here, waiting for its read-ahead
        };
    };
    auto expected = [&](int slice) {
        vc::core::util::GridStore whole((s.store / "xy" / gridName(slice)).string());
        return points(whole.get(cv::Rect(0, 0, 100, 100)));
    };
    // Every slice already read whole: read-ahead has nothing to do.
    fs::create_directories(s.dir / "xy");
    for (int slice = 0; slice < 8; ++slice) fs::copy_file(s.store / "xy" / gridName(slice), s.dir / "xy" / gridName(slice));
    for (const int fail : {1, 1 << 20}) {
        const auto got = within(std::chrono::seconds(10), query(fail, 3.f));
        REQUIRE(got.has_value());
        CHECK(*got == expected(3));
    }
    // Nothing read yet, and no thread can start: the query reads what it needs itself.
    for (int slice = 0; slice < 8; ++slice) fs::remove(s.dir / "xy" / gridName(slice));
    const auto got = within(std::chrono::seconds(30), query(1 << 20, 6.f));
    REQUIRE(got.has_value());
    CHECK(*got == expected(6));
}

TEST_CASE("Range reads: files that grow after they are opened stay within the memory cap")
{
    Streamed s;
    RangeReads on;
    constexpr int first = 20, count = 8;
    for (int slice = first; slice < first + count; ++slice) {
        vc::core::util::GridStore big(cv::Rect(0, 0, 4000, 3000), 32);
        for (int y = 5 + slice; y < 3000; y += 37)
            for (int x = 5; x < 4000; x += 41) big.add({cv::Point(x, y), cv::Point(x + 6, y + 3), cv::Point(x + 12, y + 1)});
        big.save((s.store / "xy" / gridName(slice)).string());
    }
    const uint64_t slice_bytes = fs::file_size(s.store / "xy" / gridName(first));
    const uint64_t cap = 2 * slice_bytes;
    setenv("VC_NORMAL_GRID_RANGE_MEMORY_BYTES", std::to_string(cap).c_str(), 1);
    using vc::core::util::RemoteGridFile;
    const uint64_t baseline = RemoteGridFile::residentTotal();
    {
        NormalGridVolume volume(s.dir.string());
        // A whole-slice reader (as vc_gen_umbilicus reads every slice): each
        // grid opens small, then grows to the whole file, and is let go.
        for (int slice = first; slice < first + count; ++slice) {
            auto grid = volume.get_grid(0, slice);
            REQUIRE(grid != nullptr);
            vc::core::util::GridStore whole((s.store / "xy" / gridName(slice)).string());
            CHECK(points(grid->get_all()).size() == points(whole.get_all()).size());
            grid.reset();
            // At most the cap, its trim step, and the slice just read.
            CHECK(RemoteGridFile::residentTotal() - baseline <= cap + cap / 16 + slice_bytes);
        }
        // A grid a query still holds is kept, and still answers.
        auto held = volume.get_grid(0, first);
        for (int slice = first + 1; slice < first + count; ++slice) (void)volume.get_grid(0, slice)->get_all();
        vc::core::util::GridStore whole((s.store / "xy" / gridName(first)).string());
        CHECK(points(held->get(cv::Rect(0, 0, 4000, 3000))) == points(whole.get(cv::Rect(0, 0, 4000, 3000))));
        CHECK(RemoteGridFile::residentTotal() - baseline <= cap + cap / 16 + 2 * slice_bytes);
    }
    unsetenv("VC_NORMAL_GRID_RANGE_MEMORY_BYTES");
}
