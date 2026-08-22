#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <opencv2/core.hpp>

#if defined(__linux__) && defined(__x86_64__) && __has_include(<valgrind/callgrind.h>)
#include <valgrind/callgrind.h>
#include <valgrind/valgrind.h>
#define VC_HAS_CALLGRIND_CLIENT 1
#else
#define VC_HAS_CALLGRIND_CLIENT 0
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using vc::render::ChunkCache;
using vc::render::ChunkCacheService;
using vc::render::ChunkDtype;
using vc::render::ChunkedPlaneSampler;
using vc::render::ChunkFetchResult;
using vc::render::ChunkFetchStatus;
using vc::render::ChunkKey;
using vc::render::ChunkKeyHash;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkedArray;
using vc::render::IChunkFetcher;

constexpr int kLevels = 4;
constexpr int kTileSize = 32;
constexpr int kSerialSize = 96;
constexpr int kParallelSize = 256;
constexpr int kValueBand = 48;
constexpr std::array<float, 4> kRegionBaseX{256.0f, 1024.0f, 1792.0f, 2560.0f};
constexpr std::array<int, 3> kLevel0Shape{1024, 2048, 4096};
constexpr std::array<int, 3> kChunkShape{32, 32, 32};

enum class Scenario {
    FullRes,
    Fallback1,
    Fallback3,
    MixedCorrelated,
    MixedShuffled,
    FullResShuffled,
    Fallback3Shuffled,
    FullResCacheStress,
    FullResCacheStressShuffled,
};

struct FixtureSize {
    const char* name;
    int width;
    int height;
    int measuredRepetitions;
    bool expectsParallel;
};

constexpr FixtureSize kSerial{"serial", kSerialSize, kSerialSize, 4, false};
constexpr FixtureSize kParallel{"parallel", kParallelSize, kParallelSize, 2, true};

const char* scenarioName(Scenario scenario)
{
    switch (scenario) {
        case Scenario::FullRes:
            return "full_res";
        case Scenario::Fallback1:
            return "fallback_1";
        case Scenario::Fallback3:
            return "fallback_3";
        case Scenario::MixedCorrelated:
            return "mixed_correlated";
        case Scenario::MixedShuffled:
            return "mixed_shuffled";
        case Scenario::FullResShuffled:
            return "full_res_shuffled";
        case Scenario::Fallback3Shuffled:
            return "fallback_3_shuffled";
        case Scenario::FullResCacheStress:
            return "full_res_cache_stress";
        case Scenario::FullResCacheStressShuffled:
            return "full_res_cache_stress_shuffled";
    }
    throw std::runtime_error("unknown scenario");
}

Scenario parseScenario(std::string_view value)
{
    if (value == "full_res")
        return Scenario::FullRes;
    if (value == "fallback_1")
        return Scenario::Fallback1;
    if (value == "fallback_3")
        return Scenario::Fallback3;
    if (value == "mixed_correlated")
        return Scenario::MixedCorrelated;
    if (value == "mixed_shuffled")
        return Scenario::MixedShuffled;
    if (value == "full_res_shuffled")
        return Scenario::FullResShuffled;
    if (value == "fallback_3_shuffled")
        return Scenario::Fallback3Shuffled;
    if (value == "full_res_cache_stress")
        return Scenario::FullResCacheStress;
    if (value == "full_res_cache_stress_shuffled")
        return Scenario::FullResCacheStressShuffled;
    throw std::runtime_error("unknown scenario: " + std::string(value));
}

const FixtureSize& parseSize(std::string_view value)
{
    if (value == kSerial.name)
        return kSerial;
    if (value == kParallel.name)
        return kParallel;
    throw std::runtime_error("unknown fixture size: " + std::string(value));
}

std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t keySeed(const ChunkKey& key)
{
    std::uint64_t value = static_cast<std::uint32_t>(key.level);
    value = splitmix64(value ^ static_cast<std::uint32_t>(key.iz));
    value = splitmix64(value ^ (std::uint64_t(static_cast<std::uint32_t>(key.iy)) << 16U));
    return splitmix64(value ^ (std::uint64_t(static_cast<std::uint32_t>(key.ix)) << 32U));
}

class SyntheticChunkFetcher final : public IChunkFetcher
{
public:
    SyntheticChunkFetcher(const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<uint8_t>& expectedLevels)
    {
        if (coords.size() != expectedLevels.size())
            throw std::runtime_error("coordinate and expected-level shapes differ");
        buildEntries(coords, expectedLevels);
    }

    ChunkFetchResult fetch(const ChunkKey& key) override { return decodeFetched(key, fetchEncoded(key)); }

    ChunkFetchResult fetchEncoded(const ChunkKey& key) override
    {
        fetchCalls_.fetch_add(1, std::memory_order_relaxed);
        const auto found = entries_.find(normalized(key));
        if (found == entries_.end())
            return {ChunkFetchStatus::IoError, {}, {}, false, false, 0, "synthetic fixture accessed an undeclared chunk"};
        ChunkFetchResult result;
        result.status = found->second == ChunkStatus::Data ? ChunkFetchStatus::Found : ChunkFetchStatus::Missing;
        return result;
    }

    ChunkFetchResult decodeFetched(const ChunkKey& key, ChunkFetchResult fetched) const override
    {
        if (fetched.status != ChunkFetchStatus::Found)
            return fetched;
        const std::size_t chunkBytes = std::size_t(kChunkShape[0]) * std::size_t(kChunkShape[1]) * std::size_t(kChunkShape[2]);
        fetched.bytes.resize(chunkBytes);
        std::uint64_t state = keySeed(normalized(key));
        const int base = key.level * kValueBand + 8;
        for (std::size_t i = 0; i < fetched.bytes.size(); ++i) {
            state = splitmix64(state + i);
            fetched.bytes[i] = std::byte(base + int(state % 24U));
        }
        return fetched;
    }

    const std::unordered_map<ChunkKey, ChunkStatus, ChunkKeyHash>& entries() const { return entries_; }

    std::size_t fetchCalls() const { return fetchCalls_.load(std::memory_order_relaxed); }

private:
    static ChunkKey normalized(ChunkKey key)
    {
        key.sourceId = {};
        return key;
    }

    static std::array<ChunkKey, 8> dependencies(const cv::Vec3f& coord, int level)
    {
        const float scale = 1.0f / float(1 << level);
        const int ix = int(std::floor(coord[0] * scale));
        const int iy = int(std::floor(coord[1] * scale));
        const int iz = int(std::floor(coord[2] * scale));
        std::array<ChunkKey, 8> result{};
        std::size_t index = 0;
        for (int dz = 0; dz <= 1; ++dz)
            for (int dy = 0; dy <= 1; ++dy)
                for (int dx = 0; dx <= 1; ++dx) {
                    result[index++] = {level, (iz + dz) / kChunkShape[0], (iy + dy) / kChunkShape[1], (ix + dx) / kChunkShape[2]};
                }
        return result;
    }

    void declareEntry(const ChunkKey& key, ChunkStatus status)
    {
        auto [it, inserted] = entries_.try_emplace(key, status);
        if (!inserted && it->second != status) {
            throw std::runtime_error("synthetic fallback regions share a chunk with conflicting residency");
        }
    }

    void buildEntries(const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<uint8_t>& expectedLevels)
    {
        for (int y = 0; y < coords.rows; ++y) {
            const auto* coordRow = coords.ptr<cv::Vec3f>(y);
            const auto* levelRow = expectedLevels.ptr<uint8_t>(y);
            for (int x = 0; x < coords.cols; ++x) {
                const int expected = levelRow[x];
                for (int level = 0; level < kLevels; ++level) {
                    const ChunkStatus status = level < expected ? ChunkStatus::Missing : ChunkStatus::Data;
                    for (const ChunkKey& key : dependencies(coordRow[x], level))
                        declareEntry(key, status);
                }
            }
        }
    }

    std::unordered_map<ChunkKey, ChunkStatus, ChunkKeyHash> entries_;
    std::atomic_size_t fetchCalls_{0};
};

class ThreadTrackingArray final : public IChunkedArray
{
public:
    explicit ThreadTrackingArray(ChunkCache& cache) : cache_(cache) {}

    int numLevels() const override { return cache_.numLevels(); }
    std::array<int, 3> shape(int level) const override { return cache_.shape(level); }
    std::array<int, 3> chunkShape(int level) const override { return cache_.chunkShape(level); }
    ChunkDtype dtype() const override { return cache_.dtype(); }
    double fillValue() const override { return cache_.fillValue(); }
    LevelTransform levelTransform(int level) const override { return cache_.levelTransform(level); }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        recordThread();
        return cache_.tryGetChunk(level, iz, iy, ix);
    }

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix, const vc::render::ChunkRequestContext& request) override
    {
        recordThread();
        return cache_.tryGetChunk(level, iz, iy, ix, request);
    }

    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override
    {
        recordThread();
        return cache_.getChunkIfCached(level, iz, iy, ix);
    }

    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override { return cache_.getChunkBlocking(level, iz, iy, ix); }

    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset) override
    {
        cache_.prefetchChunks(keys, wait, priorityOffset);
    }

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback callback) override
    {
        return cache_.addChunkReadyListener(std::move(callback));
    }

    void removeChunkReadyListener(ChunkReadyCallbackId id) override { cache_.removeChunkReadyListener(id); }

    std::size_t observedThreads() const
    {
        std::lock_guard lock(threadMutex_);
        return threadIds_.size();
    }

private:
    void recordThread()
    {
        std::lock_guard lock(threadMutex_);
        threadIds_.insert(std::this_thread::get_id());
    }

    ChunkCache& cache_;
    mutable std::mutex threadMutex_;
    std::set<std::thread::id> threadIds_;
};

std::unique_ptr<ChunkCache> makeSyntheticCache(const std::shared_ptr<SyntheticChunkFetcher>& fetcher)
{
    std::vector<ChunkCache::LevelInfo> levels;
    levels.reserve(kLevels);
    for (int level = 0; level < kLevels; ++level) {
        const int scale = 1 << level;
        levels.push_back({
            {kLevel0Shape[0] / scale, kLevel0Shape[1] / scale, kLevel0Shape[2] / scale},
            kChunkShape,
            {{1.0 / double(scale), 1.0 / double(scale), 1.0 / double(scale)}, {0.0, 0.0, 0.0}},
        });
    }
    ChunkCache::Options options;
    options.detectAllFillChunks = false;
    ChunkCacheService::Options serviceOptions;
    serviceOptions.decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
    serviceOptions.fetchConcurrency.workerCapacity = 16;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 16;
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers(kLevels, fetcher);
    return std::make_unique<ChunkCache>(std::move(levels), std::move(fetchers), 0.0, ChunkDtype::UInt8, std::move(options), std::move(serviceOptions));
}

void preloadSyntheticCache(ChunkCache& cache, const SyntheticChunkFetcher& fetcher)
{
    for (const auto& [key, expected] : fetcher.entries()) {
        const ChunkResult result = cache.getChunkBlocking(key.level, key.iz, key.iy, key.ix);
        if (result.status != expected) {
            throw std::runtime_error("synthetic cache preload produced an unexpected chunk state");
        }
    }
}

struct Fixture {
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> expectedLevels;
};

int desiredLevel(Scenario scenario, int x, int width)
{
    if (scenario == Scenario::FullRes || scenario == Scenario::FullResShuffled || scenario == Scenario::FullResCacheStress ||
        scenario == Scenario::FullResCacheStressShuffled)
        return 0;
    if (scenario == Scenario::Fallback1)
        return 1;
    if (scenario == Scenario::Fallback3 || scenario == Scenario::Fallback3Shuffled)
        return 3;
    return std::min(3, x * 4 / width);
}

void deterministicShuffle(Fixture& fixture)
{
    const int count = fixture.coords.rows * fixture.coords.cols;
    auto* coords = fixture.coords.ptr<cv::Vec3f>();
    auto* levels = fixture.expectedLevels.ptr<uint8_t>();
    std::uint64_t state = 0x8d12e5a7c39b640fULL;
    for (int i = count - 1; i > 0; --i) {
        state = splitmix64(state);
        const int j = int(state % std::uint64_t(i + 1));
        std::swap(coords[i], coords[j]);
        std::swap(levels[i], levels[j]);
    }
}

Fixture makeFixture(Scenario scenario, const FixtureSize& size)
{
    Fixture fixture{cv::Mat_<cv::Vec3f>(size.height, size.width), cv::Mat_<uint8_t>(size.height, size.width)};
    for (int y = 0; y < size.height; ++y) {
        auto* coordRow = fixture.coords.ptr<cv::Vec3f>(y);
        auto* levelRow = fixture.expectedLevels.ptr<uint8_t>(y);
        for (int x = 0; x < size.width; ++x) {
            const int level = desiredLevel(scenario, x, size.width);
            const bool cacheStress = scenario == Scenario::FullResCacheStress || scenario == Scenario::FullResCacheStressShuffled;
            if (cacheStress) {
                const int chunk = (y * size.width + x) % 512;
                levelRow[x] = 0;
                coordRow[x] = {float((chunk % 128) * kChunkShape[2]) + 8.25f, float(((chunk / 128) % 4) * kChunkShape[1]) + 8.25f, 256.25f};
                continue;
            }
            const bool uniformLevel = scenario == Scenario::FullRes || scenario == Scenario::Fallback1 || scenario == Scenario::Fallback3 ||
                                      scenario == Scenario::FullResShuffled || scenario == Scenario::Fallback3Shuffled;
            const int regionWidth = uniformLevel ? size.width : size.width / 4;
            const int localX = x % regionWidth;
            levelRow[x] = uint8_t(level);
            coordRow[x] = {kRegionBaseX[level] + 8.25f + float(localX) * 0.5f, 256.25f + float(y) * 0.5f, 256.25f + float(x) * 0.0625f + float(y) * 0.03125f};
        }
    }
    if (scenario == Scenario::MixedShuffled || scenario == Scenario::FullResShuffled || scenario == Scenario::Fallback3Shuffled ||
        scenario == Scenario::FullResCacheStressShuffled)
        deterministicShuffle(fixture);
    return fixture;
}

std::uint64_t checksum(const cv::Mat_<uint8_t>& image)
{
    std::uint64_t result = 1469598103934665603ULL;
    for (int y = 0; y < image.rows; ++y) {
        const auto* row = image.ptr<uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            result ^= row[x];
            result *= 1099511628211ULL;
        }
    }
    return result;
}

struct RunResult {
    std::uint64_t checksum = 0;
    int coveredPixels = 0;
};

RunResult renderOnce(IChunkedArray& array, const Fixture& fixture, cv::Mat_<uint8_t>& output, cv::Mat_<uint8_t>& coverage, const ChunkedPlaneSampler::Options& options)
{
    const auto stats = ChunkedPlaneSampler::sampleCoordsFineToCoarse(array, 0, fixture.coords, output, coverage, options);
    return {0, stats.coveredPixels};
}

void validateResult(const Fixture& fixture, const cv::Mat_<uint8_t>& output, const cv::Mat_<uint8_t>& coverage, const RunResult& result)
{
    const int pixels = fixture.coords.rows * fixture.coords.cols;
    if (result.coveredPixels != pixels || cv::countNonZero(coverage) != pixels)
        throw std::runtime_error("render did not cover every output pixel");
    for (int y = 0; y < output.rows; ++y) {
        const auto* outputRow = output.ptr<uint8_t>(y);
        const auto* expectedRow = fixture.expectedLevels.ptr<uint8_t>(y);
        for (int x = 0; x < output.cols; ++x) {
            const int observedLevel = int(outputRow[x]) / kValueBand;
            if (observedLevel != expectedRow[x]) {
                throw std::runtime_error(
                    "pixel-level fallback oracle failed at x=" + std::to_string(x) + " y=" + std::to_string(y) +
                    " expected=" + std::to_string(expectedRow[x]) + " observed=" + std::to_string(observedLevel));
            }
        }
    }
}

struct CaseResult {
    std::uint64_t checksum = 0;
    std::size_t observedThreads = 1;
    int repetitions = 0;
    std::vector<double> trialSeconds;
};

CaseResult runCase(Scenario scenario, const FixtureSize& size, int repetitions, int nativeTrials, bool callgrind, bool requireParallelExecution = false)
{
    Fixture fixture = makeFixture(scenario, size);
    auto fetcher = std::make_shared<SyntheticChunkFetcher>(fixture.coords, fixture.expectedLevels);
    auto cache = makeSyntheticCache(fetcher);
    preloadSyntheticCache(*cache, *fetcher);
    const std::size_t preloadFetchCalls = fetcher->fetchCalls();
    ChunkedPlaneSampler::Options options(vc::Sampling::Trilinear, kTileSize);
    options.queueMisses = true;
    options.queuedFallbackLevels = 0;

    cv::Mat_<uint8_t> warmOutput(size.height, size.width, uint8_t(0));
    cv::Mat_<uint8_t> warmCoverage(size.height, size.width, uint8_t(0));
    RunResult warm = renderOnce(*cache, fixture, warmOutput, warmCoverage, options);
    warm.checksum = checksum(warmOutput);
    validateResult(fixture, warmOutput, warmCoverage, warm);

    cv::Mat_<uint8_t> threadOutput(size.height, size.width, uint8_t(0));
    cv::Mat_<uint8_t> threadCoverage(size.height, size.width, uint8_t(0));
    const char* workerOverride = std::getenv("VC_RENDER_SAMPLER_THREADS");
    const bool singleWorkerOverride = workerOverride && std::string_view(workerOverride) == "1";
    const bool requireMultipleThreads = requireParallelExecution && size.expectsParallel && !singleWorkerOverride && !callgrind;
    std::size_t observedThreads = 0;
    for (int attempt = 0; attempt < (requireMultipleThreads ? 4 : 1); ++attempt) {
        threadOutput.setTo(uint8_t(0));
        threadCoverage.setTo(uint8_t(0));
        ThreadTrackingArray trackingArray(*cache);
        RunResult threadRun = renderOnce(trackingArray, fixture, threadOutput, threadCoverage, options);
        observedThreads = std::max(observedThreads, trackingArray.observedThreads());
        threadRun.checksum = checksum(threadOutput);
        validateResult(fixture, threadOutput, threadCoverage, threadRun);
        if (!requireMultipleThreads || observedThreads >= 2)
            break;
    }
    if (requireMultipleThreads && observedThreads < 2)
        throw std::runtime_error("parallel fixture did not execute on multiple sampler threads");
    if (!size.expectsParallel && observedThreads != 1)
        throw std::runtime_error("serial fixture unexpectedly used multiple sampler threads");

    CaseResult result;
    result.checksum = warm.checksum;
    result.observedThreads = observedThreads;
    result.repetitions = repetitions;
    const int trials = callgrind ? 1 : nativeTrials;
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<cv::Mat_<uint8_t>> outputs;
        std::vector<cv::Mat_<uint8_t>> coverages;
        outputs.reserve(repetitions);
        coverages.reserve(repetitions);
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            outputs.emplace_back(size.height, size.width, uint8_t(0));
            coverages.emplace_back(size.height, size.width, uint8_t(0));
        }

        std::vector<RunResult> runs(static_cast<std::size_t>(repetitions));
        double trialSeconds = 0.0;
#if VC_HAS_CALLGRIND_CLIENT
        if (callgrind)
            CALLGRIND_ZERO_STATS;
#else
        if (callgrind)
            throw std::runtime_error("Callgrind client requests are unavailable on this platform");
#endif
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
#if VC_HAS_CALLGRIND_CLIENT
            if (callgrind)
                CALLGRIND_START_INSTRUMENTATION;
#endif
            runs[repetition] = renderOnce(*cache, fixture, outputs[repetition], coverages[repetition], options);
#if VC_HAS_CALLGRIND_CLIENT
            if (callgrind)
                CALLGRIND_STOP_INSTRUMENTATION;
#endif
            const auto finished = std::chrono::steady_clock::now();
            trialSeconds += std::chrono::duration<double>(finished - started).count();
        }
        result.trialSeconds.push_back(trialSeconds);

        for (int repetition = 0; repetition < repetitions; ++repetition) {
            runs[repetition].checksum = checksum(outputs[repetition]);
            validateResult(fixture, outputs[repetition], coverages[repetition], runs[repetition]);
            if (runs[repetition].checksum != result.checksum)
                throw std::runtime_error("render checksum changed between repetitions");
        }
    }
    if (fetcher->fetchCalls() != preloadFetchCalls) {
        throw std::runtime_error("timed render unexpectedly fetched a synthetic source chunk");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(values.size() - 1, static_cast<std::size_t>(std::ceil(fraction * double(values.size()))) - 1);
    return values[index];
}

std::string resultJson(Scenario scenario, const FixtureSize& size, const CaseResult& result, bool callgrind = false)
{
    const double mean = std::accumulate(result.trialSeconds.begin(), result.trialSeconds.end(), 0.0) / double(result.trialSeconds.size());
    const double median = percentile(result.trialSeconds, 0.5);
    const double p95 = percentile(result.trialSeconds, 0.95);
    const double measuredPixels = double(size.width) * double(size.height) * double(result.repetitions);
    const char* workerOverride = std::getenv("VC_RENDER_SAMPLER_THREADS");
    std::ostringstream out;
    out << std::setprecision(12) << "{\"scenario\":\"" << scenarioName(scenario) << "\",\"fixture\":\"" << size.name
        << "\",\"width\":" << size.width << ",\"height\":" << size.height << ",\"tile_size\":" << kTileSize
        << ",\"repetitions\":" << result.repetitions << ",\"measured_pixels\":" << std::uint64_t(measuredPixels)
        << ",\"checksum\":" << result.checksum << ",\"observed_threads\":" << result.observedThreads
        << ",\"worker_override\":" << (workerOverride && workerOverride[0] != '\0' ? workerOverride : "0");
    if (callgrind)
        out << ",\"instrumented_wall_seconds\":" << median;
    else
        out << ",\"native_mean_seconds\":" << mean << ",\"native_median_seconds\":" << median << ",\"native_p95_seconds\":" << p95
            << ",\"native_median_pixels_per_second\":" << measuredPixels / median;
    out << ",\"compiler_id\":\"" << VC_RENDER_BENCH_COMPILER_ID << "\",\"compiler_version\":\"" << VC_RENDER_BENCH_COMPILER_VERSION
        << "\",\"build_type\":\"" << VC_RENDER_BENCH_BUILD_TYPE << "\",\"architecture_target\":\"" << VC_RENDER_BENCH_ARCHITECTURE << "\"}";
    return out.str();
}

void writeText(const std::string& path, const std::string& value)
{
    if (path.empty())
        return;
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot open metadata output: " + path);
    output << value << '\n';
}

void verifyAll()
{
    constexpr std::array scenarios{Scenario::FullRes, Scenario::Fallback3, Scenario::MixedCorrelated, Scenario::MixedShuffled};
    for (const FixtureSize* size : {&kSerial, &kParallel}) {
        for (const Scenario scenario : scenarios) {
            const CaseResult result = runCase(scenario, *size, 1, 1, false, true);
            std::cout << resultJson(scenario, *size, result) << '\n';
        }
    }
}

struct Args {
    bool verifyAll = false;
    bool callgrind = false;
    bool requireParallelExecution = false;
    Scenario scenario = Scenario::FullRes;
    const FixtureSize* size = &kSerial;
    int repetitions = 0;
    int nativeTrials = 7;
    std::string metadataPath;
};

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto requireValue = [&](const char* option) -> std::string_view {
            if (++i >= argc)
                throw std::runtime_error(std::string(option) + " requires a value");
            return argv[i];
        };
        if (arg == "--verify-all")
            args.verifyAll = true;
        else if (arg == "--callgrind")
            args.callgrind = true;
        else if (arg == "--require-parallel-execution")
            args.requireParallelExecution = true;
        else if (arg == "--scenario")
            args.scenario = parseScenario(requireValue("--scenario"));
        else if (arg == "--fixture")
            args.size = &parseSize(requireValue("--fixture"));
        else if (arg == "--repetitions")
            args.repetitions = std::stoi(std::string(requireValue("--repetitions")));
        else if (arg == "--native-trials")
            args.nativeTrials = std::stoi(std::string(requireValue("--native-trials")));
        else if (arg == "--metadata")
            args.metadataPath = requireValue("--metadata");
        else
            throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (args.repetitions <= 0)
        args.repetitions = args.size->measuredRepetitions;
    if (args.repetitions <= 0 || args.nativeTrials <= 0)
        throw std::runtime_error("repetition and trial counts must be positive");
    return args;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parseArgs(argc, argv);
        if (args.verifyAll) {
            verifyAll();
            return 0;
        }
#if VC_HAS_CALLGRIND_CLIENT
        if (args.callgrind && !RUNNING_ON_VALGRIND)
            throw std::runtime_error("--callgrind must run under Valgrind Callgrind");
#endif
        const CaseResult result = runCase(args.scenario, *args.size, args.repetitions, args.nativeTrials, args.callgrind, args.requireParallelExecution);
        const std::string json = resultJson(args.scenario, *args.size, result, args.callgrind);
        std::cout << json << '\n';
        writeText(args.metadataPath, json);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_render_synthetic: " << error.what() << '\n';
        return 1;
    }
}
