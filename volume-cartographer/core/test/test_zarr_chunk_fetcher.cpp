// Cover openLocalZarrPyramid + createChunkCache and the local-fetcher path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrDownloadBenchmark.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <utils/zarr.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using vc::render::openLocalZarrPyramid;
using vc::render::OpenedChunkedZarr;
using vc::render::createChunkCache;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_zcf_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

fs::path makeLocalVolume(const fs::path& dir, size_t numLevels = 2,
                        vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8)
{
    Volume::ZarrCreateOptions opts;
    opts.shapeZYX = {64, 64, 64};
    opts.chunkShapeZYX = {32, 32, 32};
    opts.numLevels = numLevels;
    opts.compressor = "none";
    opts.overwriteExisting = true;
    opts.dtype = dtype;
    auto v = Volume::New(dir, opts);
    REQUIRE(v);
    return dir;
}

std::vector<std::byte> readBytes(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    REQUIRE(file);
    const auto size = file.tellg();
    REQUIRE(size >= 0);
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), size);
    REQUIRE(file);
    return result;
}

std::shared_ptr<utils::ZarrArray> makeShardedArray(
    const fs::path& root, bool missingSecondChunk = false)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {4, 4, 4};
    meta.chunks = {4, 4, 4};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";
    utils::ShardConfig shard;
    shard.sub_chunks = {2, 2, 2};
    meta.shard_config = std::move(shard);

    fs::create_directories(root);
    {
        std::ofstream group(root / "zarr.json");
        group << R"({"zarr_format":3,"node_type":"group"})";
    }
    auto array = utils::ZarrArray::create(root / "0", meta);
    std::vector<std::optional<std::vector<std::byte>>> chunks(8);
    for (std::size_t i = 0; i < chunks.size(); ++i)
        chunks[i] = std::vector<std::byte>(8, static_cast<std::byte>(i + 1));
    if (missingSecondChunk)
        chunks[1] = std::nullopt;
    array.write_shard(std::array<std::size_t, 3>{0, 0, 0}, chunks);

    auto store = std::make_shared<utils::FileSystemStore>(root);
    return std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::open(store, "0"));
}

class BlockingCountingStore final : public utils::Store {
public:
    BlockingCountingStore(fs::path root, std::string target)
        : store_(std::move(root)), target_(std::move(target))
    {
    }

    bool exists(const std::string& key) const override
    {
        return store_.exists(key);
    }

    std::vector<std::byte> get(const std::string& key) const override
    {
        beforeFullRead(key);
        return store_.get(key);
    }

    std::optional<std::vector<std::byte>>
    get_if_exists(const std::string& key) const override
    {
        beforeFullRead(key);
        return store_.get_if_exists(key);
    }

    std::optional<std::vector<std::byte>> get_partial(
        const std::string& key,
        std::size_t offset,
        std::size_t length) const override
    {
        if (key == target_)
            ++partialReads;
        return store_.get_partial(key, offset, length);
    }

    void set(const std::string& key, std::span<const std::byte> value) override
    {
        store_.set(key, value);
    }

    void erase(const std::string& key) override
    {
        store_.erase(key);
    }

    void waitForTargetRead() const
    {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return fullReads.load() != 0; });
    }

    void releaseTargetRead()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    mutable std::atomic<int> fullReads{0};
    mutable std::atomic<int> partialReads{0};

private:
    void beforeFullRead(const std::string& key) const
    {
        if (key != target_)
            return;
        ++fullReads;
        cv_.notify_all();
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return released_; });
    }

    utils::FileSystemStore store_;
    std::string target_;
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable bool released_ = false;
};

} // namespace

TEST_CASE("openLocalZarrPyramid: opens a multi-level local zarr")
{
    auto d = tmpDir("multi");
    makeLocalVolume(d, /*numLevels=*/3);
    auto opened = openLocalZarrPyramid(d);
    CHECK(opened.fetchers.size() >= 1);
    CHECK_FALSE(opened.fetchers[0]->measuresRemoteTransfer());
    CHECK_FALSE(opened.shapes.empty());
    CHECK(opened.shapes[0][0] == 64);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: single-level (no subdir) is also accepted")
{
    auto d = tmpDir("single");
    // Single-level: just a .zarray at the root, no /0/, /1/ subdirs.
    // Create via VcDataset directly to control this.
    auto ds = vc::createZarrDataset(d, "arr",
        /*shape=*/{32, 32, 32}, /*chunks=*/{32, 32, 32},
        vc::VcDtype::uint8, "none");
    REQUIRE(ds);
    // openLocalZarrPyramid looks for numeric subdirs first, then falls back
    // to opening `root` directly. Point at the array dir.
    auto opened = openLocalZarrPyramid(d / "arr");
    CHECK(opened.fetchers.size() >= 1);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: uint16 volume")
{
    auto d = tmpDir("u16");
    makeLocalVolume(d, /*numLevels=*/2, vc::render::ChunkDtype::UInt16);
    auto opened = openLocalZarrPyramid(d);
    CHECK(opened.fetchers.size() >= 1);
    CHECK(opened.dtype == vc::render::ChunkDtype::UInt16);
    fs::remove_all(d);
}

TEST_CASE("openLocalZarrPyramid: missing dir throws")
{
    CHECK_THROWS(openLocalZarrPyramid("/__no__/__where__"));
}

TEST_CASE("createChunkCache wraps openLocalZarrPyramid result")
{
    auto d = tmpDir("cc_wrap");
    makeLocalVolume(d, 2);
    auto opened = openLocalZarrPyramid(d);
    auto cache = createChunkCache(std::move(opened),
        /*decodedByteCapacity=*/1ULL << 20);
    REQUIRE(cache);
    CHECK(cache->numLevels() >= 1);
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs binds numeric dataset paths by value")
{
    // A scaledown export (lasagna/tiled_predict3d.py with first_level=2)
    // advertises only its coarse datasets; positional binding would register
    // the level-2 array as full resolution.
    auto d = tmpDir("zattrs_numeric");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"2"},{"path":"3"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0] == std::pair<int, std::string>{2, "2"});
    CHECK(keys[1] == std::pair<int, std::string>{3, "3"});
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs keeps positional binding for named datasets")
{
    auto d = tmpDir("zattrs_named");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"s0"},{"path":"s1"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0] == std::pair<int, std::string>{0, "s0"});
    CHECK(keys[1] == std::pair<int, std::string>{1, "s1"});
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs rejects traversing dataset paths")
{
    auto d = tmpDir("zattrs_traversal");
    {
        std::ofstream file(d / ".zattrs");
        file << R"({"multiscales":[{"datasets":[{"path":"../outside"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    CHECK_THROWS_WITH_AS(
        vc::render::remoteLevelKeysFromZattrs(store, 0),
        doctest::Contains("unsafe dataset path"), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("remoteLevelKeysFromZattrs is unchanged for standard zero-based pyramids")
{
    auto d = tmpDir("zattrs_standard");
    {
        std::ofstream f(d / ".zattrs");
        f << R"({"multiscales":[{"datasets":[{"path":"0"},{"path":"1"},{"path":"2"}]}]})";
    }
    auto store = std::make_shared<utils::FileSystemStore>(d);
    const auto keys = vc::render::remoteLevelKeysFromZattrs(store, 0);
    REQUIRE(keys.size() == 3);
    for (int level = 0; level < 3; ++level) {
        CHECK(keys[static_cast<std::size_t>(level)] ==
              std::pair<int, std::string>{level, std::to_string(level)});
    }
    fs::remove_all(d);
}

TEST_CASE("validateAndRebaseVcPyramid maps physical level two to logical zero")
{
    auto d = tmpDir("rebase_two");
    makeLocalVolume(d, /*numLevels=*/6);
    auto physical = openLocalZarrPyramid(d);
    REQUIRE(physical.fetchers.size() == 6);
    const auto physicalLevelTwoFetcher = physical.fetchers[2];

    auto rebased = vc::render::validateAndRebaseVcPyramid(std::move(physical), 2);
    CHECK(rebased.fetchers.size() == 4);
    CHECK(rebased.shapes[0] == std::array<int, 3>{16, 16, 16});
    CHECK(rebased.chunkShapes.size() == 4);
    CHECK(rebased.storageChunkShapes.size() == 4);
    CHECK(rebased.levelNumbers == std::vector<int>{0, 1, 2, 3});
    CHECK(rebased.transforms[0].scaleFromLevel0 == std::array<double, 3>{1.0, 1.0, 1.0});
    CHECK(rebased.transforms[1].scaleFromLevel0 == std::array<double, 3>{0.5, 0.5, 0.5});
    CHECK(rebased.fetchers[0] == physicalLevelTwoFetcher);
    fs::remove_all(d);
}

TEST_CASE("validateAndRebaseVcPyramid accepts variable contiguous pyramid lengths")
{
    for (const size_t levels : {size_t{5}, size_t{7}}) {
        auto d = tmpDir("rebase_length");
        makeLocalVolume(d, levels);
        auto rebased = vc::render::validateAndRebaseVcPyramid(
            openLocalZarrPyramid(d), 2);
        CHECK(rebased.fetchers.size() == levels - 2);
        fs::remove_all(d);
    }
}

TEST_CASE("validateAndRebaseVcPyramid rejects gaps and invalid retained fill values")
{
    auto d = tmpDir("rebase_invalid");
    makeLocalVolume(d, /*numLevels=*/4);

    auto gap = openLocalZarrPyramid(d);
    gap.fetchers[1].reset();
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(gap), 2),
        doctest::Contains("gap at /1"), std::runtime_error);

    auto fills = openLocalZarrPyramid(d);
    fills.fillValues[3] = 1.0;
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(fills), 2),
        doctest::Contains("consistent fill value"), std::runtime_error);

    auto shape = openLocalZarrPyramid(d);
    shape.shapes[1][0] += shape.storageChunkShapes[1][0];
    CHECK_THROWS_WITH_AS(
        vc::render::validateAndRebaseVcPyramid(std::move(shape), 2),
        doctest::Contains("dyadic downscale"), std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("ZarrChunkFetcher fetches a present chunk from local")
{
    auto d = tmpDir("fetch_present");
    auto v = Volume::New(d, []() {
        Volume::ZarrCreateOptions o;
        o.shapeZYX = {32, 32, 32};
        o.chunkShapeZYX = {32, 32, 32};
        o.numLevels = 1;
        o.compressor = "none";
        o.overwriteExisting = true;
        return o;
    }());
    REQUIRE(v);
    Array3D<uint8_t> in({32, 32, 32}, /*fill=*/200);
    v->writeZYX(in, {0, 0, 0}, 0);

    auto opened = openLocalZarrPyramid(d);
    REQUIRE_FALSE(opened.fetchers.empty());
    vc::render::ChunkKey key{0, 0, 0, 0};
    auto r = opened.fetchers[0]->fetch(key);
    CHECK(r.status == vc::render::ChunkFetchStatus::Found);
    CHECK_FALSE(r.bytes.empty());
    fs::remove_all(d);
}

TEST_CASE("native mirror stores a complete shard and serves sibling inner chunks")
{
    auto source = tmpDir("mirror_sharded_source");
    auto mirror = tmpDir("mirror_sharded_cache");
    auto array = makeShardedArray(source);

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = mirror;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    const auto identity = "test:mirror-sharded:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, std::move(options));
    REQUIRE(cache);

    const auto first = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(first.status == vc::render::ChunkStatus::Data);
    REQUIRE(first.bytes);
    CHECK(first.bytes->front() == std::byte{1});

    const auto object = array->storage_object_location(
        std::array<std::size_t, 3>{0, 0, 0});
    const auto sourceShard = source / object.key;
    const auto mirrorShard = mirror / object.key;
    REQUIRE(fs::is_regular_file(sourceShard));
    REQUIRE(fs::is_regular_file(mirrorShard));
    CHECK(readBytes(mirrorShard) == readBytes(sourceShard));
    CHECK(readBytes(mirror / "zarr.json") == readBytes(source / "zarr.json"));
    CHECK(readBytes(mirror / "0" / "zarr.json") ==
          readBytes(source / "0" / "zarr.json"));
    CHECK(cache->storageObjectRepresentatives(0).size() == 1);

    fs::remove(sourceShard);
    const auto sibling = cache->getChunkBlocking(0, 0, 0, 1);
    REQUIRE(sibling.status == vc::render::ChunkStatus::Data);
    REQUIRE(sibling.bytes);
    CHECK(sibling.bytes->front() == std::byte{2});
    CHECK_FALSE(fs::exists(mirrorShard.string() + ".empty"));

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    fs::remove_all(source);
    fs::remove_all(mirror);
}

TEST_CASE("native mirror coalesces inner requests into one full shard transfer")
{
    auto source = tmpDir("mirror_shard_dedup_source");
    auto mirror = tmpDir("mirror_shard_dedup_cache");
    auto fixture = makeShardedArray(source);
    const auto object = fixture->storage_object_location(
        std::array<std::size_t, 3>{0, 0, 0});
    auto store = std::make_shared<BlockingCountingStore>(source, object.key);
    auto array = std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::open(store, "0"));

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = mirror;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    const auto identity = "test:mirror-shard-dedup:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, std::move(options));
    REQUIRE(cache);
    std::mutex activityMutex;
    std::vector<std::pair<vc::render::ChunkKey, bool>> activity;
    cache->addRemoteFetchActivityListener(
        [&](const vc::render::ChunkKey& key, bool active) {
            std::lock_guard lock(activityMutex);
            activity.emplace_back(key, active);
        });

    auto first = std::async(std::launch::async, [&] {
        return cache->getChunkBlocking(0, 0, 0, 0);
    });
    store->waitForTargetRead();
    std::promise<void> secondStarted;
    auto second = std::async(std::launch::async, [&] {
        secondStarted.set_value();
        return cache->getChunkBlocking(0, 0, 0, 1);
    });
    secondStarted.get_future().wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    store->releaseTargetRead();

    const auto firstResult = first.get();
    const auto secondResult = second.get();
    REQUIRE(firstResult.status == vc::render::ChunkStatus::Data);
    REQUIRE(secondResult.status == vc::render::ChunkStatus::Data);
    CHECK(firstResult.bytes->front() == std::byte{1});
    CHECK(secondResult.bytes->front() == std::byte{2});
    CHECK(store->fullReads.load() == 1);
    CHECK(store->partialReads.load() == 0);
    CHECK(readBytes(mirror / object.key) == readBytes(source / object.key));
    {
        std::lock_guard lock(activityMutex);
        const auto countEvent = [&](int ix, bool active) {
            return std::ranges::count_if(activity, [&](const auto& event) {
                return event.first.ix == ix && event.second == active;
            });
        };
        CHECK(countEvent(0, true) == 1);
        CHECK(countEvent(0, false) == 1);
        CHECK(countEvent(1, true) == 1);
        CHECK(countEvent(1, false) == 1);
    }
    CHECK(cache->activeRemoteFetches().empty());
    CHECK(cache->stats().remoteFetchesInFlight == 0);

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    fs::remove_all(source);
    fs::remove_all(mirror);
}

TEST_CASE("missing inner shard chunks do not mark the whole shard empty")
{
    auto source = tmpDir("mirror_shard_inner_missing_source");
    auto mirror = tmpDir("mirror_shard_inner_missing_cache");
    auto array = makeShardedArray(source, true);
    const auto object = array->storage_object_location(
        std::array<std::size_t, 3>{0, 0, 0});

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = mirror;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    const auto identity = "test:mirror-shard-inner-missing:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, std::move(options));
    REQUIRE(cache);

    CHECK(cache->getChunkBlocking(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);
    CHECK(cache->getChunkBlocking(0, 0, 0, 1).status ==
          vc::render::ChunkStatus::Missing);
    CHECK(fs::is_regular_file(mirror / object.key));
    CHECK_FALSE(fs::exists((mirror / object.key).string() + ".empty"));

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    fs::remove_all(source);
    fs::remove_all(mirror);
}

TEST_CASE("missing whole shard creates one physical empty marker")
{
    auto source = tmpDir("mirror_missing_shard_source");
    auto mirror = tmpDir("mirror_missing_shard_cache");
    auto array = makeShardedArray(source);
    const auto object = array->storage_object_location(
        std::array<std::size_t, 3>{0, 0, 0});
    fs::remove(source / object.key);

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = mirror;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    const auto identity = "test:mirror-missing-shard:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, std::move(options));
    REQUIRE(cache);

    CHECK(cache->getChunkBlocking(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Missing);
    CHECK(cache->getChunkBlocking(0, 0, 0, 1).status ==
          vc::render::ChunkStatus::Missing);
    CHECK_FALSE(fs::exists(mirror / object.key));
    CHECK(fs::is_regular_file((mirror / object.key).string() + ".empty"));

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    fs::remove_all(source);
    fs::remove_all(mirror);
}

TEST_CASE("existing legacy cache footprint remains legacy")
{
    auto source = tmpDir("legacy_layout_source");
    auto cacheRoot = tmpDir("legacy_layout_cache");
    auto array = makeShardedArray(source);
    fs::create_directories(cacheRoot / "level_0" / "0" / "0");
    {
        std::ofstream legacy(cacheRoot / "level_0" / "0" / "0" / "0.bin",
                             std::ios::binary);
        legacy.put('\0');
    }

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = cacheRoot;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    const auto identity = "test:legacy-layout:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, std::move(options));
    REQUIRE(cache);
    CHECK(cache->persistentCacheLayout() ==
          vc::render::PersistentCacheLayout::Legacy);

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    fs::remove_all(source);
    fs::remove_all(cacheRoot);
}

TEST_CASE("incidental legacy-looking paths do not select legacy layout")
{
    auto source = tmpDir("incidental_layout_source");
    auto cacheRoot = tmpDir("incidental_layout_cache");
    auto array = makeShardedArray(source);
    fs::create_directories(cacheRoot / "level_0");
    {
        std::ofstream incidental(cacheRoot / "level_0" / "note.empty");
    }

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = cacheRoot;
    options.zarrMirrorMetadata = {
        {"zarr.json", readBytes(source / "zarr.json")},
        {"0/zarr.json", readBytes(source / "0" / "zarr.json")},
    };
    CHECK_THROWS_WITH_AS(
        vc::render::acquireProcessChunkCache(
            "test:incidental-layout:" + source.string(),
            array, std::move(options)),
        doctest::Contains("neither a legacy cache nor a Zarr mirror"),
        std::runtime_error);

    fs::remove_all(source);
    fs::remove_all(cacheRoot);
}

TEST_CASE("native mirror stores exact unsharded bytes and reopens without source data")
{
    auto source = tmpDir("mirror_unsharded_source");
    auto mirror = tmpDir("mirror_unsharded_cache");
    auto volume = Volume::New(source, [] {
        Volume::ZarrCreateOptions options;
        options.shapeZYX = {32, 32, 32};
        options.chunkShapeZYX = {32, 32, 32};
        options.numLevels = 1;
        options.compressor = "none";
        options.overwriteExisting = true;
        return options;
    }());
    REQUIRE(volume);
    Array3D<std::uint8_t> values({32, 32, 32}, 173);
    volume->writeZYX(values, {0, 0, 0}, 0);
    volume.reset();

    auto store = std::make_shared<utils::FileSystemStore>(source);
    auto array = std::make_shared<utils::ZarrArray>(
        utils::ZarrArray::open(store, "0"));
    const auto object = array->storage_object_location(
        std::array<std::size_t, 3>{0, 0, 0});
    const auto sourceObject = source / object.key;
    REQUIRE(fs::is_regular_file(sourceObject));

    vc::render::ChunkCache::Options options;
    options.persistentCachePath = mirror;
    for (const auto& key : {".zgroup", ".zattrs", "0/.zarray", "0/.zattrs"}) {
        if (fs::is_regular_file(source / key))
            options.zarrMirrorMetadata.push_back({key, readBytes(source / key)});
    }
    const auto identity = "test:mirror-unsharded:" + source.string();
    auto cache = vc::render::acquireProcessChunkCache(
        identity, array, options);
    REQUIRE(cache);
    const auto first = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(first.status == vc::render::ChunkStatus::Data);
    CHECK(readBytes(mirror / object.key) == readBytes(sourceObject));

    fs::remove(sourceObject);
    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(identity);
    const auto reopenIdentity = identity + ":reopen";
    cache = vc::render::acquireProcessChunkCache(
        reopenIdentity, array, options);
    REQUIRE(cache);
    const auto reopened = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(reopened.status == vc::render::ChunkStatus::Data);
    REQUIRE(reopened.bytes);
    CHECK(reopened.bytes->front() == std::byte{173});

    cache.reset();
    vc::render::processChunkCacheService()->invalidateSource(reopenIdentity);
    fs::remove_all(source);
    fs::remove_all(mirror);
}

TEST_CASE("ZarrChunkFetcher fetches a Missing chunk")
{
    auto d = tmpDir("fetch_missing");
    auto v = Volume::New(d, []() {
        Volume::ZarrCreateOptions o;
        o.shapeZYX = {32, 32, 32};
        o.chunkShapeZYX = {32, 32, 32};
        o.numLevels = 1;
        o.compressor = "none";
        o.overwriteExisting = true;
        return o;
    }());
    REQUIRE(v);

    auto opened = openLocalZarrPyramid(d);
    REQUIRE_FALSE(opened.fetchers.empty());
    // No chunks written → first chunk is Missing.
    auto r = opened.fetchers[0]->fetch({0, 0, 0, 0});
    CHECK(r.status == vc::render::ChunkFetchStatus::Missing);
    fs::remove_all(d);
}

TEST_CASE("Zarr download benchmark selects unique distributed logical chunks")
{
    const auto first = vc::render::selectZarrDownloadBenchmarkChunks(
        {17, 9, 5}, {4, 4, 4}, 2, 100, 41);
    const auto second = vc::render::selectZarrDownloadBenchmarkChunks(
        {17, 9, 5}, {4, 4, 4}, 2, 100, 41);
    REQUIRE(first.size() == 30);
    CHECK(first == second);

    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> unique;
    for (const auto& key : first) {
        CHECK(key.level == 2);
        CHECK(key.iz >= 0);
        CHECK(key.iz < 5);
        CHECK(key.iy >= 0);
        CHECK(key.iy < 3);
        CHECK(key.ix >= 0);
        CHECK(key.ix < 2);
        unique.insert(key);
    }
    CHECK(unique.size() == first.size());
}

TEST_CASE("Zarr download benchmark uses encoded fetches and fixed admission")
{
    class EncodedFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            ++decodedCalls;
            return {};
        }

        vc::render::ChunkFetchResult fetchEncoded(
            const vc::render::ChunkKey&) override
        {
            ++calls;
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes.resize(1024);
            return result;
        }

        std::atomic_size_t calls{0};
        std::atomic_size_t decodedCalls{0};
    };

    auto fetcher = std::make_shared<EncodedFetcher>();
    OpenedChunkedZarr opened;
    opened.shapes = {{64, 64, 64}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {fetcher};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 12;
    options.workers = 1;
    options.schedule = vc::render::ZarrDownloadSchedule::Fixed;
    std::vector<vc::render::ZarrDownloadProgress> progress;
    options.progressInterval = std::chrono::hours(1);
    options.progressCallback = [&](const auto& update) {
        progress.push_back(update);
    };
    const auto result = vc::render::runZarrDownloadBenchmark(opened, options);

    CHECK(fetcher->calls.load() == 12);
    CHECK(fetcher->decodedCalls.load() == 0);
    CHECK(result.requestedChunks == 12);
    CHECK(result.foundChunks == 12);
    CHECK(result.encodedBytes == 12 * 1024);
    CHECK(result.httpErrors == 0);
    CHECK(result.ioErrors == 0);
    CHECK(result.decodeErrors == 0);
    CHECK(result.sinkErrors == 0);
    CHECK(result.peakActive == 1);
    CHECK(result.finalTransferStats.admissionLimit == 1);
    CHECK_FALSE(result.finalTransferStats.adaptive);
    CHECK(result.finalTransferStats.bytesPerSecond == 0.0);
    REQUIRE(progress.size() == 1);
    CHECK(progress[0].queuedChunks == 0);
    CHECK(progress[0].downloadingChunks == 0);
    CHECK(progress[0].completedChunks == 12);
    CHECK(progress[0].encodedBytes == 12 * 1024);
}

TEST_CASE("Zarr download benchmark replenishes work until its duration")
{
    class EncodedFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            return {};
        }

        vc::render::ChunkFetchResult fetchEncoded(
            const vc::render::ChunkKey&) override
        {
            ++calls;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes.resize(1024);
            return result;
        }

        std::atomic_size_t calls{0};
    };

    auto fetcher = std::make_shared<EncodedFetcher>();
    OpenedChunkedZarr opened;
    opened.shapes = {{64, 64, 64}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {fetcher};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 2;
    options.runDuration = std::chrono::milliseconds(15);
    options.workers = 2;
    options.schedule = vc::render::ZarrDownloadSchedule::Fixed;
    const auto result = vc::render::runZarrDownloadBenchmark(opened, options);

    CHECK(result.requestedChunks == fetcher->calls.load());
    CHECK(result.requestedChunks > options.chunkCount);
    CHECK(result.foundChunks == result.requestedChunks);
    CHECK(result.wallSeconds >= 0.015);
}

TEST_CASE("Zarr download benchmark rejects adaptive mode without remote measurement")
{
    class LocalFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            return {};
        }
    };

    OpenedChunkedZarr opened;
    opened.shapes = {{32, 32, 32}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {std::make_shared<LocalFetcher>()};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 1;
    options.workers = 2;
    options.schedule = vc::render::ZarrDownloadSchedule::Adaptive;
    CHECK_THROWS_AS(
        vc::render::runZarrDownloadBenchmark(opened, options),
        std::invalid_argument);
}

TEST_CASE("Zarr download benchmark can write encoded payloads to a temporary sink")
{
    class EncodedFetcher final : public vc::render::IChunkFetcher {
    public:
        vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
        {
            return {};
        }

        vc::render::ChunkFetchResult fetchEncoded(
            const vc::render::ChunkKey&) override
        {
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes.resize(17, std::byte{0x2a});
            return result;
        }
    };

    const auto output = tmpDir("download_benchmark_sink");
    OpenedChunkedZarr opened;
    opened.shapes = {{32, 32, 32}};
    opened.chunkShapes = {{16, 16, 16}};
    opened.fetchers = {std::make_shared<EncodedFetcher>()};

    vc::render::ZarrDownloadBenchmarkOptions options;
    options.chunkCount = 3;
    options.workers = 1;
    options.schedule = vc::render::ZarrDownloadSchedule::Fixed;
    options.outputDirectory = output;
    const auto result = vc::render::runZarrDownloadBenchmark(opened, options);

    CHECK(result.foundChunks == 3);
    CHECK(result.encodedBytes == 51);
    CHECK(result.sinkErrors == 0);
    std::size_t files = 0;
    for (const auto& entry : fs::directory_iterator(output)) {
        CHECK(entry.is_regular_file());
        CHECK(entry.file_size() == 17);
        ++files;
    }
    CHECK(files == 3);
    fs::remove_all(output);
}
