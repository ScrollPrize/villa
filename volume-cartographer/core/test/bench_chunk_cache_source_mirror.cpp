#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using vc::render::ChunkCache;
using vc::render::ChunkCacheService;
using vc::render::ChunkDtype;
using vc::render::ChunkFetchResult;
using vc::render::ChunkFetchStatus;
using vc::render::ChunkKey;
using vc::render::ChunkStorageObject;
using vc::render::IChunkFetcher;

namespace {

constexpr int kChunksPerAxis = 10;
constexpr int kChunkEdge = 32;
constexpr int kRepetitions = 11;

class SourceMirrorFetcher final : public IChunkFetcher {
public:
    explicit SourceMirrorFetcher(bool sourceAvailable)
        : sourceAvailable_(sourceAvailable)
    {
    }

    ChunkFetchResult fetch(const ChunkKey&) override
    {
        throw std::logic_error("source-mirror benchmark must fetch storage objects");
    }

    std::optional<ChunkStorageObject>
    storageObject(const ChunkKey& key) const override
    {
        ChunkStorageObject object;
        object.representativeKey = key;
        object.outerZ = key.iz;
        object.outerY = key.iy;
        object.outerX = key.ix;
        object.sourceKey = "0/" + std::to_string(key.iz) + "." +
                           std::to_string(key.iy) + "." +
                           std::to_string(key.ix);
        return object;
    }

    ChunkFetchResult fetchStorageObject(
        const ChunkStorageObject& object,
        const DownloadProgressCallback&) override
    {
        ++sourceFetches;
        if (!sourceAvailable_) {
            ChunkFetchResult result;
            result.status = ChunkFetchStatus::IoError;
            result.message = "benchmark remote source was unexpectedly used";
            return result;
        }

        constexpr std::size_t bytes =
            std::size_t{kChunkEdge} * kChunkEdge * kChunkEdge;
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes.resize(bytes);
        const auto seed = static_cast<unsigned char>(
            1 + object.outerZ * 31 + object.outerY * 7 + object.outerX);
        for (std::size_t i = 0; i < bytes; ++i) {
            result.bytes[i] = std::byte{
                static_cast<unsigned char>(seed + i * 17)};
        }
        return result;
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

    std::atomic<std::size_t> sourceFetches{0};

private:
    bool sourceAvailable_;
};

std::shared_ptr<ChunkCacheService> makeService()
{
    ChunkCacheService::Options options;
    options.decodedByteCapacity = 32ULL * 1024ULL * 1024ULL;
    options.fetchConcurrency.workerCapacity = 8;
    options.fetchConcurrency.maxConcurrentReads = 8;
    return std::make_shared<ChunkCacheService>(std::move(options));
}

std::shared_ptr<ChunkCache> makeCache(
    const std::shared_ptr<ChunkCacheService>& service,
    std::string sourceIdentity,
    const fs::path& persistentPath,
    const std::shared_ptr<SourceMirrorFetcher>& fetcher)
{
    ChunkCache::LevelInfo level;
    level.shape = {kChunksPerAxis * kChunkEdge,
                   kChunksPerAxis * kChunkEdge,
                   kChunksPerAxis * kChunkEdge};
    level.chunkShape = {kChunkEdge, kChunkEdge, kChunkEdge};

    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    options.persistentCachePath = persistentPath;
    options.zarrMirrorMetadata.push_back(
        {".zgroup", {std::byte{'{'}, std::byte{'}'}}});

    return service->acquireSource(
        std::move(sourceIdentity),
        std::vector<ChunkCache::LevelInfo>{level},
        std::vector<std::shared_ptr<IChunkFetcher>>{fetcher},
        0.0, ChunkDtype::UInt8, std::move(options));
}

std::vector<ChunkKey> allKeys()
{
    std::vector<ChunkKey> keys;
    keys.reserve(kChunksPerAxis * kChunksPerAxis * kChunksPerAxis);
    for (int iz = 0; iz < kChunksPerAxis; ++iz) {
        for (int iy = 0; iy < kChunksPerAxis; ++iy) {
            for (int ix = 0; ix < kChunksPerAxis; ++ix)
                keys.push_back({0, iz, iy, ix});
        }
    }
    return keys;
}

double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

struct Summary {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

Summary summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    Summary result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    result.median = values[values.size() / 2];
    result.p95 = values[static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size())) - 1.0)];
    result.minimum = values.front();
    result.maximum = values.back();
    return result;
}

void printSummary(const char* label, const std::vector<double>& values)
{
    const auto summary = summarize(values);
    std::printf(
        "%s: mean %.3f ms, p50 %.3f ms, p95 %.3f ms, min %.3f ms, max %.3f ms\n",
        label, summary.mean, summary.median, summary.p95,
        summary.minimum, summary.maximum);
}

} // namespace

int main()
{
    const auto keys = allKeys();
    std::vector<double> populateTimes;
    std::vector<double> reopenTimes;
    populateTimes.reserve(kRepetitions);
    reopenTimes.reserve(kRepetitions);

    std::mt19937_64 rng(std::random_device{}());
    const auto root = fs::temp_directory_path() /
        ("vc_source_mirror_bench_" + std::to_string(rng()));
    const auto populateService = makeService();
    const auto reopenService = makeService();

    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
        const auto persistentPath = root / std::to_string(repetition);
        auto remote = std::make_shared<SourceMirrorFetcher>(true);
        auto cache = makeCache(
            populateService, "populate-" + std::to_string(repetition),
            persistentPath, remote);
        const auto populateStart = Clock::now();
        cache->prefetchChunks(keys, true);
        cache->waitForPersistentWrites();
        populateTimes.push_back(elapsedMs(populateStart));
        if (remote->sourceFetches.load() != keys.size()) {
            throw std::runtime_error("cold population did not fetch every object");
        }
        if (cache->persistentCacheLayout() !=
                vc::render::PersistentCacheLayout::ZarrMirror ||
            !fs::is_regular_file(persistentPath / "0" / "0.0.0") ||
            fs::exists(persistentPath / ".vc_delta3d_cache")) {
            throw std::runtime_error(
                "benchmark did not exercise the uncompressed SourceMirror layout");
        }
        cache.reset();

        auto unavailableRemote =
            std::make_shared<SourceMirrorFetcher>(false);
        auto reopened = makeCache(
            reopenService, "reopen-" + std::to_string(repetition),
            persistentPath, unavailableRemote);
        const auto reopenStart = Clock::now();
        reopened->prefetchChunks(keys, true);
        reopenTimes.push_back(elapsedMs(reopenStart));
        if (unavailableRemote->sourceFetches.load() != 0) {
            throw std::runtime_error("reopen performed remote source work");
        }
        reopened.reset();
    }

    fs::remove_all(root);
    std::printf(
        "SourceMirror cache benchmark: %zu chunks, %d KiB/chunk, %d repetitions, Release\n",
        keys.size(), (kChunkEdge * kChunkEdge * kChunkEdge) / 1024,
        kRepetitions);
    printSummary("cold uncompressed populate", populateTimes);
    printSummary("disk-cache reopen", reopenTimes);
    return 0;
}
