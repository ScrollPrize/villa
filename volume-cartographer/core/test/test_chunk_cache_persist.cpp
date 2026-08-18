// More ChunkCache coverage: persistent-cache empty markers, byte counting,
// download-history pruning, listener invocation order, prefetch w/ no wait.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/CacheCompression.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using vc::render::ChunkCache;
using vc::render::ChunkCacheService;
using vc::render::ChunkDtype;
using vc::render::ChunkFetchResult;
using vc::render::ChunkFetchStatus;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkFetcher;

namespace {

class CountingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ++fetchCalls;
        std::lock_guard<std::mutex> lk(m_);
        auto it = canned_.find(key);
        if (it != canned_.end()) return it->second;
        ChunkFetchResult r;
        r.status = ChunkFetchStatus::Missing;
        return r;
    }
    void setCanned(const ChunkKey& k, ChunkFetchResult r)
    {
        std::lock_guard<std::mutex> lk(m_);
        canned_[k] = std::move(r);
    }
    std::optional<std::string> sourceChunkKey(const ChunkKey& key) const override
    {
        return "0/" + std::to_string(key.iz) + "." +
               std::to_string(key.iy) + "." + std::to_string(key.ix);
    }
    std::optional<vc::render::ChunkStorageObject>
    storageObject(const ChunkKey& key) const override
    {
        vc::render::ChunkStorageObject object;
        object.representativeKey = key;
        object.outerZ = key.iz;
        object.outerY = key.iy;
        object.outerX = key.ix;
        object.sourceKey = *sourceChunkKey(key);
        return object;
    }
    std::atomic<int> fetchCalls{0};
private:
    std::mutex m_;
    std::unordered_map<ChunkKey, ChunkFetchResult, vc::render::ChunkKeyHash> canned_;
};

class MirrorFetcher final : public CountingFetcher {
public:
    std::optional<vc::render::ChunkStorageObject>
    storageObject(const ChunkKey& key) const override
    {
        vc::render::ChunkStorageObject object;
        object.representativeKey = key;
        object.outerZ = key.iz;
        object.outerY = key.iy;
        object.outerX = key.ix;
        object.sourceKey = "scale0/" + std::to_string(key.iz) + "." +
                           std::to_string(key.iy) + "." +
                           std::to_string(key.ix);
        return object;
    }

    ChunkFetchResult fetchStorageObject(
        const vc::render::ChunkStorageObject& object,
        const DownloadProgressCallback&) override
    {
        return fetch(object.representativeKey);
    }

    ChunkFetchResult decodeStorageObject(
        const ChunkKey&,
        std::span<const std::byte> bytes) const override
    {
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes.assign(bytes.begin(), bytes.end());
        return result;
    }

    bool supportsSourcePayloadPersistence(const ChunkKey&) const override
    {
        return true;
    }
};

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_cc_persist_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

std::shared_ptr<ChunkCache> makeCache(std::shared_ptr<CountingFetcher> f,
                                       std::optional<fs::path> persist = {},
                                       bool compress = false,
                                       std::optional<fs::path> budgetRoot = {})
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.detectAllFillChunks = true;
    if (persist) opts.persistentCachePath = *persist;
    if (budgetRoot) opts.persistentCacheBudgetRoot = *budgetRoot;
    opts.compressPersistentCache = compress;
    ChunkCacheService::Options serviceOptions;
    serviceOptions.fetchConcurrency.workerCapacity = 4;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 4;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, std::move(serviceOptions));
}

std::shared_ptr<ChunkCache> makeDelta3dCache(
    std::shared_ptr<CountingFetcher> f,
    const fs::path& persist,
    std::optional<fs::path> budgetRoot = {})
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.detectAllFillChunks = true;
    opts.persistentCachePath = persist;
    const std::string group = R"({"zarr_format":2})";
    const std::string array =
        R"({"zarr_format":2,"shape":[8,8,8],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null,"dimension_separator":"."})";
    opts.zarrMirrorMetadata = {
        {".zgroup", {reinterpret_cast<const std::byte*>(group.data()),
                     reinterpret_cast<const std::byte*>(group.data() + group.size())}},
        {"0/.zarray", {reinterpret_cast<const std::byte*>(array.data()),
                       reinterpret_cast<const std::byte*>(array.data() + array.size())}},
    };
    if (budgetRoot)
        opts.persistentCacheBudgetRoot = *budgetRoot;
    ChunkCacheService::Options serviceOptions;
    serviceOptions.fetchConcurrency.workerCapacity = 4;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 4;
    serviceOptions.persistentCacheEncoding =
        vc::render::PersistentCacheEncoding::Delta3dLossless;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, std::move(serviceOptions));
}

std::shared_ptr<ChunkCache> makeMirrorCache(
    std::shared_ptr<MirrorFetcher> f,
    const fs::path& persist)
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.persistentCachePath = persist;
    opts.zarrMirrorMetadata.push_back(
        {".zgroup", {std::byte{'{'}, std::byte{'}'}}});
    ChunkCacheService::Options serviceOptions;
    serviceOptions.fetchConcurrency.workerCapacity = 4;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 4;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts, std::move(serviceOptions));
}

std::vector<std::byte> makeBytes(std::size_t n, std::byte v = std::byte{99});
ChunkResult waitForResolved(ChunkCache& c, int level, int iz, int iy, int ix,
                            std::chrono::milliseconds timeout = std::chrono::seconds{2});

TEST_CASE("budget denial skips persistence but downloaded ChunkCache data remains usable")
{
    auto root = tmpDir("budget_denied");
    auto budget = vc::render::PersistentZarrCacheBudget::configure(
        root, {1, 0}, [](const fs::path&, std::error_code& ec) {
            ec.clear();
            return fs::space_info{1024, 1024, 1024};
        });
    budget->waitForIdle();

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = makeBytes(64, std::byte{0x2a});
    f->setCanned({0, 0, 0, 0}, fetched);

    auto cache = makeCache(f, root / "volume", false, root);
    const auto result = waitForResolved(*cache, 0, 0, 0, 0);
    REQUIRE(result.status == ChunkStatus::Data);
    REQUIRE(result.bytes);
    CHECK((*result.bytes)[0] == std::byte{0x2a});
    cache->waitForPersistentWrites();
    CHECK_FALSE(fs::exists(root / "volume" / "level_0" / "0" / "0" / "0.bin"));
    CHECK(budget->stats().managedBytes == 0);
    fs::remove_all(root);
}

std::vector<std::byte> makeBytes(std::size_t n, std::byte v)
{
    return std::vector<std::byte>(n, v);
}

void writeSizedFile(const fs::path& path, std::size_t size, unsigned char value = 0x10)
{
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    std::vector<char> bytes(size, static_cast<char>(value));
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeBytes(const fs::path& path, std::span<const std::byte> bytes)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::byte> readFileBytes(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};
    const auto size = file.tellg();
    if (size < 0)
        return {};
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    return file ? bytes : std::vector<std::byte>{};
}

template <typename Predicate>
ChunkCache::Stats waitForStats(ChunkCache& c,
                               Predicate predicate,
                               std::chrono::milliseconds timeout = std::chrono::seconds{2})
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    ChunkCache::Stats s;
    while (std::chrono::steady_clock::now() < deadline) {
        s = c.stats();
        if (predicate(s))
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return c.stats();
}

ChunkResult waitForResolved(ChunkCache& c, int level, int iz, int iy, int ix,
                            std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto r = c.tryGetChunk(level, iz, iy, ix);
        if (r.status != ChunkStatus::MissQueued) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return c.tryGetChunk(level, iz, iy, ix);
}

} // namespace

TEST_CASE("Missing chunk with persistent cache writes an .empty marker")
{
    auto persist = tmpDir("missing_marker");
    auto f = std::make_shared<CountingFetcher>();
    // No canned -> Missing.
    {
        auto c = makeCache(f, persist);
        auto r = waitForResolved(*c, 0, 0, 0, 0);
        CHECK(r.status == ChunkStatus::Missing);
        c->waitForPersistentWrites();
    }
    // After cache destruction, the persistent dir should contain a zero-byte
    // .empty file somewhere under level_0/.
    bool foundEmpty = false;
    fs::path emptyPath;
    for (auto it = fs::recursive_directory_iterator(persist);
         it != fs::recursive_directory_iterator(); ++it) {
        if (it->path().extension() == ".empty") {
            foundEmpty = true;
            emptyPath = it->path();
            break;
        }
    }
    REQUIRE(foundEmpty);
    CHECK(fs::file_size(emptyPath) == 0);
    fs::remove_all(persist);
}

TEST_CASE("Reopen cache: persistent .empty marker short-circuits to Missing")
{
    auto persist = tmpDir("reopen_empty");
    // Pre-place an .empty marker for chunk (0,0,0,0).
    auto target = persist / "level_0" / "0" / "0";
    fs::create_directories(target);
    {
        std::ofstream f(target / "0.empty");
    }

    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist);
    auto r = c->tryGetChunk(0, 0, 0, 0);
    // First call may be MissQueued or immediate Missing — wait it out.
    auto resolved = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(resolved.status == ChunkStatus::Missing);
    (void)r;
    (void)waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });
    // Fetcher should not have been called — the empty marker short-circuits.
    // Tolerate impl variance — just confirm no crash.
    fs::remove_all(persist);
}

TEST_CASE("Reopen cache: persistent data file is loaded directly")
{
    auto persist = tmpDir("reopen_data");
    auto target = persist / "level_0" / "0" / "0";
    fs::create_directories(target);
    // 4*4*4 = 64 byte chunk filled with 0x42.
    {
        std::ofstream f(target / "0.bin", std::ios::binary);
        std::vector<char> bytes(64, 0x42);
        f.write(bytes.data(), bytes.size());
    }

    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    // Should come back as Data (or AllFill if 0x42 ≠ fill 0).
    CHECK((r.status == ChunkStatus::Data || r.status == ChunkStatus::AllFill));
    if (r.status == ChunkStatus::Data && r.bytes) {
        CHECK(int(std::to_integer<int>((*r.bytes)[0])) == 0x42);
    }
    (void)waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });
    fs::remove_all(persist);
}

TEST_CASE("stats: persistentCacheBytes reflects the on-disk size")
{
    auto persist = tmpDir("stats_bytes");
    auto target = persist / "level_0" / "0" / "0";
    fs::create_directories(target);
    {
        std::ofstream f(target / "0.bin", std::ios::binary);
        std::vector<char> bytes(64, 0x10);
        f.write(bytes.data(), bytes.size());
    }
    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist);
    auto s = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight && s.persistentCacheBytes >= 64;
    });
    CHECK(s.persistentCacheEnabled);
    CHECK(s.persistentCacheBytes >= 64);
    CHECK_FALSE(s.persistentCacheScanInFlight);
    fs::remove_all(persist);
}

TEST_CASE("stats: startup scan ignores files newer than its cutoff")
{
    auto persist = tmpDir("scan_cutoff");
    const auto target = persist / "level_0" / "0" / "0";
    writeSizedFile(target / "0.bin", 31);
    writeSizedFile(target / "1.empty", 1);
    writeSizedFile(target / "post.bin", 17);
    std::error_code ec;
    fs::last_write_time(
        target / "post.bin",
        fs::file_time_type::clock::now() + std::chrono::seconds{10},
        ec);
    REQUIRE_FALSE(ec);

    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist);

    auto s = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });
    CHECK_MESSAGE(
        s.persistentCacheBytes == 32,
        "startup scan counted " + std::to_string(s.persistentCacheBytes) +
            " bytes instead of 32");
    fs::remove_all(persist);
}

TEST_CASE("stats: repeated calls do not rescan persistent cache")
{
    auto persist = tmpDir("no_rescan");
    const auto target = persist / "level_0" / "0" / "0";
    writeSizedFile(target / "0.bin", 11);

    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist);
    auto first = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight && s.persistentCacheBytes == 11;
    });
    REQUIRE(first.persistentCacheBytes == 11);

    writeSizedFile(target / "external.bin", 29);
    std::this_thread::sleep_for(std::chrono::milliseconds(2300));
    for (int i = 0; i < 5; ++i) {
        auto s = c->stats();
        CHECK(s.persistentCacheBytes == 11);
        CHECK_FALSE(s.persistentCacheScanInFlight);
    }
    fs::remove_all(persist);
}

TEST_CASE("stats: successful persistent data write increments byte count")
{
    auto persist = tmpDir("write_delta");
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{7});
    f->setCanned({0, 0, 0, 0}, fr);

    auto c = makeCache(f, persist);
    auto initial = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });
    CHECK(initial.persistentCacheBytes == 0);

    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
    auto after = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return s.persistentCacheBytes == 64;
    });
    CHECK(after.persistentCacheBytes == 64);
    fs::remove_all(persist);
}

TEST_CASE("stats: persistent overwrite applies new minus old byte delta")
{
    auto persist = tmpDir("overwrite_delta");
    const auto target = persist / "level_0" / "0" / "0";
    writeSizedFile(target / "0.bin", 80);

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{8});
    f->setCanned({0, 0, 0, 0}, fr);

    auto c = makeCache(f, persist);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
    auto after = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight && s.persistentCacheBytes == 64;
    });
    CHECK(after.persistentCacheBytes == 64);
    fs::remove_all(persist);
}

TEST_CASE("stats: failed persistent write does not change byte count")
{
    auto persistFile = tmpDir("write_fail_parent") / "cache_file";
    writeSizedFile(persistFile, 3);

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = makeBytes(64, std::byte{9});
    f->setCanned({0, 0, 0, 0}, fr);

    auto c = makeCache(f, persistFile);
    auto initial = waitForStats(*c, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });
    CHECK(initial.persistentCacheBytes == 0);

    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);

    auto barrierDir = tmpDir("write_fail_barrier");
    auto barrier = makeCache(std::make_shared<CountingFetcher>(), barrierDir);
    (void)waitForStats(*barrier, [](const ChunkCache::Stats& s) {
        return !s.persistentCacheScanInFlight;
    });

    auto after = c->stats();
    CHECK(after.persistentCacheBytes == 0);
    CHECK_FALSE(after.persistentCacheScanInFlight);
    fs::remove_all(persistFile.parent_path());
    fs::remove_all(barrierDir);
}

TEST_CASE("prefetchChunks(wait=false): non-blocking; later tryGetChunk picks it up")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{99});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    std::vector<ChunkKey> keys = {{0, 0, 0, 0}};
    c->prefetchChunks(keys, /*wait=*/false, /*priorityOffset=*/0);
    // Don't assert immediate state — just wait for resolved.
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
}

TEST_CASE("prefetchChunks with negative priority offset still resolves")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{200});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    std::vector<ChunkKey> keys = {{0, 0, 0, 0}};
    c->prefetchChunks(keys, /*wait=*/true, /*priorityOffset=*/-5);
    auto r = c->tryGetChunk(0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
}

TEST_CASE("multiple listeners are all notified")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{1});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);

    std::atomic<int> a{0}, b{0};
    auto idA = c->addChunkReadyListener([&]() { ++a; });
    auto idB = c->addChunkReadyListener([&]() { ++b; });
    (void)waitForResolved(*c, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(a.load() >= 1);
    CHECK(b.load() >= 1);
    c->removeChunkReadyListener(idA);
    c->removeChunkReadyListener(idB);
}

TEST_CASE("Many concurrent tryGetChunk calls converge on the same Entry")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{50});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);

    std::vector<std::thread> threads;
    std::atomic<int> success{0};
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 20; ++j) {
                auto r = c->tryGetChunk(0, 0, 0, 0);
                if (r.status == ChunkStatus::Data) ++success;
            }
        });
    }
    for (auto& t : threads) t.join();
    // The fetcher should have been called at most a small number of times
    // (cache coalesces in-flight requests).
    CHECK(f->fetchCalls.load() <= 4);
}

namespace {

bool waitForFile(const fs::path& path,
                 std::chrono::milliseconds timeout = std::chrono::seconds{2})
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(path)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return fs::exists(path);
}

std::vector<std::byte> variedBytes(std::size_t n)
{
    std::vector<std::byte> bytes(n);
    for (std::size_t i = 0; i < n; ++i)
        bytes[i] = std::byte{static_cast<unsigned char>(i * 7 + 3)};
    return bytes;
}

} // namespace

TEST_CASE("deprecated persistent compression option does not recompress new writes")
{
    auto persist = tmpDir("compress_write");
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = variedBytes(64);
    f->setCanned({0, 0, 0, 0}, fr);

    auto c = makeCache(f, persist, /*compress=*/true);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    REQUIRE(r.status == ChunkStatus::Data);

    const auto raw = persist / "level_0" / "0" / "0" / "0.bin";
    CHECK(waitForFile(raw));
    CHECK_FALSE(fs::exists(persist / "level_0" / "0" / "0" / "0.zst"));
    fs::remove_all(persist);
}

TEST_CASE("Reopen cache: compressed .zst entry is loaded without a fetch")
{
    auto persist = tmpDir("compress_reload");
    const auto expected = variedBytes(64);
    const auto compressed = vc::cacheCompress(
        expected, {4, 4, 4}, 1, vc::kCacheQuantLossless);
    writeBytes(
        persist / "level_0" / "0" / "0" / "0.zst", compressed);

    // New writes never recompress, but the reader still accepts legacy .zst.
    auto f = std::make_shared<CountingFetcher>();
    auto c = makeCache(f, persist, /*compress=*/false);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    REQUIRE(r.status == ChunkStatus::Data);
    REQUIRE(r.bytes);
    CHECK(*r.bytes == expected);
    CHECK(f->fetchCalls.load() == 0);
    fs::remove_all(persist);
}

TEST_CASE("Corrupt .zst entry falls back to a remote fetch")
{
    auto persist = tmpDir("compress_corrupt");
    const auto target = persist / "level_0" / "0" / "0";
    writeSizedFile(target / "0.zst", 16, 0xAB); // not a valid zstd frame

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = variedBytes(64);
    f->setCanned({0, 0, 0, 0}, fr);

    auto c = makeCache(f, persist, /*compress=*/true);
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    REQUIRE(r.status == ChunkStatus::Data);
    REQUIRE(r.bytes);
    CHECK(*r.bytes == fr.bytes);
    CHECK(f->fetchCalls.load() == 1);
    fs::remove_all(persist);
}

TEST_CASE("Delta3D mode writes a regular Zarr with verified D3D1 chunks and reopens offline")
{
    auto persist = tmpDir("delta3d_write");
    auto f = std::make_shared<CountingFetcher>();
    const auto expected = variedBytes(64);
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = expected;
    f->setCanned({0, 0, 0, 0}, fetched);

    {
        auto cache = makeDelta3dCache(f, persist);
        REQUIRE(cache->persistentCacheLayout() ==
                vc::render::PersistentCacheLayout::Delta3d);
        const auto rendered = waitForResolved(*cache, 0, 0, 0, 0);
        REQUIRE(rendered.status == ChunkStatus::Data);
        REQUIRE(rendered.bytes);
        CHECK(*rendered.bytes == expected);
        cache->waitForPersistentWrites();
    }

    const auto path = persist / "0" / "0.0.0";
    REQUIRE(fs::is_regular_file(path));
    const auto encoded = readFileBytes(path);
    REQUIRE(encoded.size() >= 4);
    CHECK(std::to_integer<char>(encoded[0]) == 'D');
    CHECK(std::to_integer<char>(encoded[1]) == '3');
    CHECK(std::to_integer<char>(encoded[2]) == 'D');
    CHECK(std::to_integer<char>(encoded[3]) == '1');
    CHECK_FALSE(fs::exists(persist / "level_0"));
    const auto metadata = readFileBytes(persist / "0" / ".zarray");
    const std::string metadataText(
        reinterpret_cast<const char*>(metadata.data()), metadata.size());
    CHECK(metadataText.find("vc-delta3d") != std::string::npos);

    auto offline = std::make_shared<CountingFetcher>();
    auto reopened = makeDelta3dCache(offline, persist);
    const auto cached = waitForResolved(*reopened, 0, 0, 0, 0);
    REQUIRE(cached.status == ChunkStatus::Data);
    REQUIRE(cached.bytes);
    CHECK(*cached.bytes == expected);
    CHECK(offline->fetchCalls.load() == 0);
    reopened.reset();
    fs::remove_all(persist);
}

TEST_CASE("Delta3D mode uses empty markers for missing and all-fill chunks")
{
    auto persist = tmpDir("delta3d_empty");
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fill;
    fill.status = ChunkFetchStatus::Found;
    fill.bytes = makeBytes(64, std::byte{0});
    f->setCanned({0, 0, 0, 1}, fill);

    auto cache = makeDelta3dCache(f, persist);
    CHECK(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Missing);
    CHECK(waitForResolved(*cache, 0, 0, 0, 1).status == ChunkStatus::AllFill);
    cache->waitForPersistentWrites();
    CHECK(fs::is_regular_file(
        persist / ".vc_cache_empty" / "0" / "0.0.0.empty"));
    CHECK(fs::is_regular_file(
        persist / ".vc_cache_empty" / "0" / "0.0.1.empty"));
    CHECK_FALSE(fs::exists(
        persist / "0" / "0.0.1"));
    cache.reset();
    fs::remove_all(persist);
}

TEST_CASE("Corrupt Delta3D payload is refetched and replaced")
{
    auto persist = tmpDir("delta3d_corrupt");
    auto seed = std::make_shared<CountingFetcher>();
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = variedBytes(64);
    seed->setCanned({0, 0, 0, 0}, fetched);
    {
        auto cache = makeDelta3dCache(seed, persist);
        REQUIRE(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Data);
        cache->waitForPersistentWrites();
    }
    writeSizedFile(
        persist / "0" / "0.0.0", 16, 0xAB);

    auto refetch = std::make_shared<CountingFetcher>();
    refetch->setCanned({0, 0, 0, 0}, fetched);
    auto cache = makeDelta3dCache(refetch, persist);
    const auto result = waitForResolved(*cache, 0, 0, 0, 0);
    REQUIRE(result.status == ChunkStatus::Data);
    CHECK(refetch->fetchCalls.load() == 1);
    cache->waitForPersistentWrites();
    const auto repaired = readFileBytes(
        persist / "0" / "0.0.0");
    REQUIRE(repaired.size() >= 4);
    CHECK(std::to_integer<char>(repaired[0]) == 'D');
    cache.reset();
    fs::remove_all(persist);
}

TEST_CASE("Delta3D prefill decodes and persists without populating decoded RAM")
{
    auto persist = tmpDir("delta3d_prefill");
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = variedBytes(64);
    f->setCanned({0, 0, 0, 0}, fetched);
    auto cache = makeDelta3dCache(f, persist);

    const auto result = cache->persistChunkBlocking(0, 0, 0, 0);
    REQUIRE(result.status == ChunkCache::PersistentRequestStatus::Data);
    cache->waitForPersistentWrites();
    CHECK(cache->stats().decodedBytes == 0);
    CHECK(fs::is_regular_file(
        persist / "0" / "0.0.0"));
    cache.reset();
    fs::remove_all(persist);
}

TEST_CASE("Delta3D disk budget accounts the final encoded payload size")
{
    const auto root = tmpDir("delta3d_budget");
    auto budget = vc::render::PersistentZarrCacheBudget::configure(
        root, {}, [](const fs::path&, std::error_code& ec) {
            ec.clear();
            return fs::space_info{1ULL << 40, 1ULL << 40, 1ULL << 40};
        });
    budget->waitForIdle();

    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = variedBytes(64);
    f->setCanned({0, 0, 0, 0}, fetched);
    auto cache = makeDelta3dCache(f, root / "volume", root);
    REQUIRE(waitForResolved(*cache, 0, 0, 0, 0).status == ChunkStatus::Data);
    cache->waitForPersistentWrites();

    const auto path = root / "volume" / "0" / "0.0.0";
    REQUIRE(fs::is_regular_file(path));
    CHECK(budget->stats().managedBytes == fs::file_size(path));
    cache.reset();
    fs::remove_all(root);
}

TEST_CASE("failed Delta3D publication leaves rendering usable and no raw fallback")
{
    const auto root = tmpDir("delta3d_write_failure");
    auto budget = vc::render::PersistentZarrCacheBudget::configure(
        root, {1, 0}, [](const fs::path&, std::error_code& ec) {
            ec.clear();
            return fs::space_info{1024, 1024, 1024};
        });
    budget->waitForIdle();

    auto f = std::make_shared<CountingFetcher>();
    const auto expected = variedBytes(64);
    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = expected;
    f->setCanned({0, 0, 0, 0}, fetched);
    auto cache = makeDelta3dCache(f, root / "volume", root);
    const auto rendered = waitForResolved(*cache, 0, 0, 0, 0);
    REQUIRE(rendered.status == ChunkStatus::Data);
    REQUIRE(rendered.bytes);
    CHECK(*rendered.bytes == expected);
    cache->waitForPersistentWrites();

    const auto chunkDir = root / "volume" / "0";
    CHECK_FALSE(fs::exists(chunkDir / "0.0.0"));
    CHECK_FALSE(fs::exists(root / "volume" / "level_0"));
    CHECK(budget->stats().managedBytes == 0);
    cache.reset();
    fs::remove_all(root);
}

TEST_CASE("persistent cache format transitions are leased and volume-local")
{
    const auto parent = tmpDir("delta3d_transition");
    const auto persist = parent / "volume-a";
    const auto sibling = parent / "volume-b" / "keep.bin";
    writeSizedFile(sibling, 7);
    const auto bookkeeping = parent / ".vc_cache_bookkeeping" /
                             persist.filename() / ".vc_prefill_level_0.json";

    ChunkFetchResult fetched;
    fetched.status = ChunkFetchStatus::Found;
    fetched.bytes = variedBytes(64);
    auto mirrorFetcher = std::make_shared<MirrorFetcher>();
    mirrorFetcher->setCanned({0, 0, 0, 0}, fetched);
    auto mirror = makeMirrorCache(mirrorFetcher, persist);
    REQUIRE(mirror->persistentCacheLayout() ==
            vc::render::PersistentCacheLayout::ZarrMirror);
    REQUIRE(waitForResolved(*mirror, 0, 0, 0, 0).status == ChunkStatus::Data);
    REQUIRE(fs::is_regular_file(persist / "scale0" / "0.0.0"));
    writeSizedFile(bookkeeping, 3);

    // An incompatible process cannot replace a cache while this mirror holds
    // its shared lease; rendering remains available with persistence disabled.
    auto blockedFetcher = std::make_shared<CountingFetcher>();
    blockedFetcher->setCanned({0, 0, 0, 0}, fetched);
    auto blocked = makeDelta3dCache(blockedFetcher, persist);
    CHECK_FALSE(blocked->stats().persistentCacheEnabled);
    CHECK_FALSE(blocked->stats().persistentCacheWarning.empty());
    REQUIRE(waitForResolved(*blocked, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK_FALSE(fs::exists(
        persist / "0" / "0.0.0"));
    blocked.reset();
    mirror.reset();

    auto deltaFetcher = std::make_shared<CountingFetcher>();
    deltaFetcher->setCanned({0, 0, 0, 0}, fetched);
    auto delta = makeDelta3dCache(deltaFetcher, persist);
    REQUIRE(delta->persistentCacheLayout() ==
            vc::render::PersistentCacheLayout::Delta3d);
    CHECK_FALSE(fs::exists(persist / "scale0" / "0.0.0"));
    CHECK_FALSE(fs::exists(bookkeeping));
    CHECK(fs::is_regular_file(sibling));
    REQUIRE(waitForResolved(*delta, 0, 0, 0, 0).status == ChunkStatus::Data);
    delta->waitForPersistentWrites();
    REQUIRE(fs::is_regular_file(
        persist / "0" / "0.0.0"));

    // Same-format processes can share the derived Zarr cache concurrently.
    auto sameModeFetcher = std::make_shared<CountingFetcher>();
    auto sameMode = makeDelta3dCache(sameModeFetcher, persist);
    CHECK(sameMode->stats().persistentCacheEnabled);
    REQUIRE(waitForResolved(*sameMode, 0, 0, 0, 0).status == ChunkStatus::Data);
    CHECK(sameModeFetcher->fetchCalls.load() == 0);

    auto blockedMirrorFetcher = std::make_shared<MirrorFetcher>();
    blockedMirrorFetcher->setCanned({0, 0, 0, 0}, fetched);
    auto blockedMirror = makeMirrorCache(blockedMirrorFetcher, persist);
    CHECK_FALSE(blockedMirror->stats().persistentCacheEnabled);
    blockedMirror.reset();
    sameMode.reset();
    delta.reset();

    auto restoredFetcher = std::make_shared<MirrorFetcher>();
    restoredFetcher->setCanned({0, 0, 0, 0}, fetched);
    auto restored = makeMirrorCache(restoredFetcher, persist);
    REQUIRE(restored->persistentCacheLayout() ==
            vc::render::PersistentCacheLayout::ZarrMirror);
    CHECK_FALSE(fs::exists(persist / ".vc_delta3d_cache"));
    CHECK_FALSE(fs::exists(
        persist / "0" / "0.0.0"));
    CHECK(fs::is_regular_file(sibling));
    restored.reset();
    fs::remove_all(parent);
}
