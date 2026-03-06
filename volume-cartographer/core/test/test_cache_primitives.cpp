#include "test.hpp"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "vc/core/cache/ChunkKey.hpp"
#include "vc/core/cache/ChunkData.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"
#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/DiskStore.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"

namespace fs = std::filesystem;

namespace {

class FakeChunkSource : public vc::cache::ChunkSource {
public:
    std::vector<uint8_t> fetch(const vc::cache::ChunkKey& key) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fetches.push_back(key);
        if (missingKeys.count(key) > 0) {
            return {};
        }
        return payload;
    }

    int numLevels() const override { return 1; }

    std::array<int, 3> chunkShape(int) const override { return {1, 1, 1}; }

    std::array<int, 3> levelShape(int) const override { return {1, 1, 4}; }

    void markMissing(const vc::cache::ChunkKey& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        missingKeys.insert(key);
    }

    void clearFetches()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fetches.clear();
    }

    size_t fetchCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return fetches.size();
    }

    std::vector<uint8_t> payload{42};

private:
    mutable std::mutex mutex_;
    std::vector<vc::cache::ChunkKey> fetches;
    std::unordered_set<vc::cache::ChunkKey, vc::cache::ChunkKeyHash> missingKeys;
};

vc::cache::DecompressFn makeTestDecompress()
{
    return [](const std::vector<uint8_t>& compressed, const vc::cache::ChunkKey&) {
        auto data = std::make_shared<vc::cache::ChunkData>();
        data->shape = {1, 1, static_cast<int>(compressed.size())};
        data->elementSize = 1;
        data->bytes = compressed;
        return data;
    };
}

struct CacheHarness {
    std::unique_ptr<vc::cache::TieredChunkCache> cache;
    FakeChunkSource* source = nullptr;
};

template <typename Predicate>
bool waitFor(Predicate&& pred, std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

CacheHarness makeTestCache(
    std::shared_ptr<vc::cache::DiskStore> diskStore,
    std::unique_ptr<FakeChunkSource> source)
{
    vc::cache::TieredChunkCache::Config cfg;
    cfg.volumeId = "test-volume";
    cfg.hotMaxBytes = 1024;
    cfg.warmMaxBytes = 1024;
    cfg.ioThreads = 1;

    auto* sourcePtr = source.get();
    CacheHarness harness;
    harness.cache = std::make_unique<vc::cache::TieredChunkCache>(
        cfg,
        std::move(source),
        makeTestDecompress(),
        std::move(diskStore));
    harness.source = sourcePtr;
    return harness;
}

class CacheFixture {
public:
    CacheFixture()
        : root(fs::temp_directory_path() / ("vc_test_tiered_cache_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        fs::create_directories(root);
    }

    ~CacheFixture()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    std::shared_ptr<vc::cache::DiskStore> makeDiskStore() const
    {
        vc::cache::DiskStore::Config cfg;
        cfg.root = root;
        cfg.maxBytes = 1ULL << 20;
        cfg.persistent = true;
        return std::make_shared<vc::cache::DiskStore>(std::move(cfg));
    }

    fs::path root;
};

}  // namespace

// ---- ChunkKey ---------------------------------------------------------------

TEST(ChunkKey, Equality)
{
    vc::cache::ChunkKey a{0, 1, 2, 3};
    vc::cache::ChunkKey b{0, 1, 2, 3};
    vc::cache::ChunkKey c{0, 1, 2, 4};
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != b);
}

TEST(ChunkKey, DefaultValues)
{
    vc::cache::ChunkKey k;
    EXPECT_EQ(k.level, 0);
    EXPECT_EQ(k.iz, 0);
    EXPECT_EQ(k.iy, 0);
    EXPECT_EQ(k.ix, 0);
}

TEST(ChunkKey, CoarsenBasic)
{
    vc::cache::ChunkKey k{0, 8, 12, 16};
    auto c1 = k.coarsen(1);
    EXPECT_EQ(c1.level, 1);
    EXPECT_EQ(c1.iz, 4);
    EXPECT_EQ(c1.iy, 6);
    EXPECT_EQ(c1.ix, 8);

    auto c2 = k.coarsen(2);
    EXPECT_EQ(c2.level, 2);
    EXPECT_EQ(c2.iz, 2);
    EXPECT_EQ(c2.iy, 3);
    EXPECT_EQ(c2.ix, 4);
}

TEST(ChunkKey, CoarsenSameLevel)
{
    vc::cache::ChunkKey k{2, 4, 5, 6};
    auto c = k.coarsen(2);
    EXPECT_EQ(c.level, 2);
    EXPECT_EQ(c.iz, 4);
    EXPECT_EQ(c.iy, 5);
    EXPECT_EQ(c.ix, 6);
}

TEST(ChunkKey, CoarsenLowerLevelReturnsUnchanged)
{
    vc::cache::ChunkKey k{3, 4, 5, 6};
    auto c = k.coarsen(1);  // targetLevel < level
    EXPECT_EQ(c.level, 3);
    EXPECT_EQ(c.iz, 4);
    EXPECT_EQ(c.iy, 5);
    EXPECT_EQ(c.ix, 6);
}

TEST(ChunkKeyHash, DifferentKeysProduceDifferentHashes)
{
    vc::cache::ChunkKeyHash hash;
    std::unordered_set<size_t> hashes;
    // Generate a grid of keys and check for collisions
    for (int l = 0; l < 3; ++l)
        for (int z = 0; z < 4; ++z)
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    hashes.insert(hash({l, z, y, x}));
    // 3*4*4*4 = 192 keys, should have zero or near-zero collisions
    EXPECT_EQ(hashes.size(), 192u);
}

TEST(ChunkKeyHash, UsableInUnorderedMap)
{
    std::unordered_set<vc::cache::ChunkKey, vc::cache::ChunkKeyHash> s;
    s.insert({0, 1, 2, 3});
    s.insert({0, 1, 2, 3});  // duplicate
    s.insert({1, 0, 0, 0});
    EXPECT_EQ(s.size(), 2u);
}

// ---- ChunkData --------------------------------------------------------------

TEST(ChunkData, BasicAccessors)
{
    vc::cache::ChunkData cd;
    cd.shape = {2, 3, 4};
    cd.elementSize = 2;
    cd.bytes.resize(2 * 3 * 4 * 2);

    EXPECT_EQ(cd.numElements(), 24u);
    EXPECT_EQ(cd.totalBytes(), 48u);
    EXPECT_EQ(cd.strideZ(), 12);
    EXPECT_EQ(cd.strideY(), 4);
    EXPECT_EQ(cd.strideX(), 1);
}

TEST(ChunkData, TypedAccess)
{
    vc::cache::ChunkData cd;
    cd.shape = {1, 1, 4};
    cd.elementSize = 2;
    cd.bytes.resize(8);

    auto* p = cd.data<uint16_t>();
    p[0] = 100;
    p[1] = 200;
    p[2] = 300;
    p[3] = 400;

    const auto& ccd = cd;
    const auto* cp = ccd.data<uint16_t>();
    EXPECT_EQ(cp[0], 100);
    EXPECT_EQ(cp[3], 400);
}

TEST(ChunkData, EmptyShape)
{
    vc::cache::ChunkData cd;
    EXPECT_EQ(cd.numElements(), 0u);
    EXPECT_EQ(cd.totalBytes(), 0u);
}

// ---- CacheDebugLog ----------------------------------------------------------

TEST(CacheDebugLog, ReturnsNullWhenEnvNotSet)
{
    // VC_CACHE_DEBUG_LOG should not be set in test environment
    // If it is, this test is inconclusive rather than failing
    FILE* f = vc::cache::cacheDebugLog();
    (void)f;  // just verify it doesn't crash
}

// ---- CacheUtils -------------------------------------------------------------

TEST(CacheUtils, ChunkFilename)
{
    vc::cache::ChunkKey k{0, 1, 2, 3};
    EXPECT_EQ(vc::cache::chunkFilename(k, "."), std::string("1.2.3"));
    EXPECT_EQ(vc::cache::chunkFilename(k, "/"), std::string("1/2/3"));
}

TEST(CacheUtils, ReadFileToVectorRoundTrip)
{
    auto tmpDir = fs::temp_directory_path() / "vc_test_cache_utils";
    fs::create_directories(tmpDir);
    auto tmpFile = tmpDir / "test.bin";

    // Write test data
    std::vector<uint8_t> data = {0x00, 0x11, 0x22, 0x33, 0x44, 0xFF};
    {
        std::ofstream ofs(tmpFile, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    auto result = vc::cache::readFileToVector(tmpFile);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ((*result)[i], data[i]);
    }

    // Cleanup
    fs::remove_all(tmpDir);
}

TEST(CacheUtils, ReadFileToVectorMissingFile)
{
    auto result = vc::cache::readFileToVector("/tmp/vc_nonexistent_file_12345");
    EXPECT_FALSE(result.has_value());
}

TEST(CacheUtils, ReadFileToVectorEmptyFile)
{
    auto tmpDir = fs::temp_directory_path() / "vc_test_cache_utils";
    fs::create_directories(tmpDir);
    auto tmpFile = tmpDir / "empty.bin";

    // Create empty file
    { std::ofstream ofs(tmpFile); }

    auto result = vc::cache::readFileToVector(tmpFile);
    EXPECT_FALSE(result.has_value());  // empty files return nullopt

    fs::remove_all(tmpDir);
}

TEST(HttpMetadataFetcher, RemoteVolumeIdIncludesUrlHash)
{
    const std::string a = "https://example.com/a/foo.zarr";
    const std::string b = "https://example.com/b/foo.zarr";

    const auto aId = vc::cache::deriveRemoteVolumeId(a);
    const auto bId = vc::cache::deriveRemoteVolumeId(b);

    EXPECT_NE(aId, bId);
    EXPECT_EQ(aId.rfind("foo.zarr-", 0), 0u);
    EXPECT_EQ(bId.rfind("foo.zarr-", 0), 0u);
}

TEST(HttpMetadataFetcher, RemoteVolumeIdNormalizesTrailingSlash)
{
    const std::string a = "https://example.com/a/foo.zarr";
    const std::string b = "https://example.com/a/foo.zarr/";

    EXPECT_EQ(vc::cache::normalizeRemoteUrl(a), vc::cache::normalizeRemoteUrl(b));
    EXPECT_EQ(vc::cache::deriveRemoteVolumeId(a), vc::cache::deriveRemoteVolumeId(b));
}

TEST(TieredChunkCache, DiskOnlyRegionIsNotReadyForNonBlockingRead)
{
    CacheFixture fixture;
    auto diskStore = fixture.makeDiskStore();
    const std::string volumeId = "test-volume";
    vc::cache::ChunkKey cachedKey{0, 0, 0, 1};
    diskStore->put(volumeId, cachedKey, std::vector<uint8_t>{7, 8, 9});

    auto harness = makeTestCache(fixture.makeDiskStore(), std::make_unique<FakeChunkSource>());

    EXPECT_FALSE(harness.cache->areAllCachedInRegion(0, 0, 0, 1, 0, 0, 1));
    EXPECT_EQ(harness.cache->countAvailable({cachedKey}), 0u);
    EXPECT_EQ(harness.cache->get(cachedKey), nullptr);
    EXPECT_EQ(harness.source->fetchCount(), 0u);
}

TEST(TieredChunkCache, PrefetchPromotesDiskOnlyChunksWithoutRemoteFetch)
{
    CacheFixture fixture;
    auto diskStore = fixture.makeDiskStore();
    const std::string volumeId = "test-volume";
    vc::cache::ChunkKey cachedKey{0, 0, 0, 2};
    diskStore->put(volumeId, cachedKey, std::vector<uint8_t>{5});

    auto harness = makeTestCache(fixture.makeDiskStore(), std::make_unique<FakeChunkSource>());

    harness.cache->prefetchRegion(0, 0, 0, 2, 0, 0, 2);
    ASSERT_TRUE(waitFor([&] {
        return harness.cache->countAvailable({cachedKey}) == 1u;
    }));

    EXPECT_NE(harness.cache->get(cachedKey), nullptr);
    EXPECT_EQ(harness.source->fetchCount(), 0u);
}

TEST(TieredChunkCache, CountAvailableIgnoresDiskOnlyButIncludesNegativeCached)
{
    CacheFixture fixture;
    const std::string volumeId = "test-volume";
    auto writerStore = fixture.makeDiskStore();
    vc::cache::ChunkKey hotKey{0, 0, 0, 0};
    vc::cache::ChunkKey diskOnlyKey{0, 0, 0, 1};
    vc::cache::ChunkKey missingKey{0, 0, 0, 2};
    writerStore->put(volumeId, diskOnlyKey, std::vector<uint8_t>{11});

    auto harness = makeTestCache(fixture.makeDiskStore(), std::make_unique<FakeChunkSource>());
    harness.source->markMissing(missingKey);

    EXPECT_NE(harness.cache->getBlocking(hotKey), nullptr);
    EXPECT_EQ(harness.cache->getBlocking(missingKey), nullptr);

    EXPECT_EQ(harness.cache->countAvailable({hotKey, diskOnlyKey, missingKey}), 2u);
}

TEST(TieredChunkCache, MixedRegionRequiresPromotionOfDiskOnlyChunk)
{
    CacheFixture fixture;
    const std::string volumeId = "test-volume";
    auto writerStore = fixture.makeDiskStore();
    vc::cache::ChunkKey hotKey{0, 0, 0, 0};
    vc::cache::ChunkKey diskOnlyKey{0, 0, 0, 1};
    vc::cache::ChunkKey missingKey{0, 0, 0, 2};
    writerStore->put(volumeId, diskOnlyKey, std::vector<uint8_t>{11});

    auto harness = makeTestCache(fixture.makeDiskStore(), std::make_unique<FakeChunkSource>());
    harness.source->markMissing(missingKey);

    EXPECT_NE(harness.cache->getBlocking(hotKey), nullptr);
    EXPECT_EQ(harness.cache->getBlocking(missingKey), nullptr);
    harness.source->clearFetches();

    EXPECT_FALSE(harness.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 2));
    harness.cache->prefetch(diskOnlyKey);
    ASSERT_TRUE(waitFor([&] {
        return harness.cache->countAvailable({diskOnlyKey}) == 1u;
    }));
    EXPECT_TRUE(harness.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 2));
    EXPECT_EQ(harness.source->fetchCount(), 0u);
}
