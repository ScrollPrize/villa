#include "vc/core/render/ZarrDownloadBenchmark.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace vc::render {

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t checkedProduct(const std::array<std::uint64_t, 3>& values)
{
    std::uint64_t result = 1;
    for (const auto value : values) {
        if (value == 0 || result > std::numeric_limits<std::uint64_t>::max() / value)
            throw std::overflow_error("zarr chunk grid is empty or too large");
        result *= value;
    }
    return result;
}

double percentile(const std::vector<double>& sorted, double fraction)
{
    if (sorted.empty())
        return 0.0;
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0);
    return sorted[std::min(index, sorted.size() - 1)];
}

std::filesystem::path chunkOutputPath(
    const std::filesystem::path& root, const ChunkKey& key)
{
    return root / (std::to_string(key.level) + "_" +
                   std::to_string(key.iz) + "_" +
                   std::to_string(key.iy) + "_" +
                   std::to_string(key.ix) + ".chunk");
}

} // namespace

std::vector<ChunkKey> selectZarrDownloadBenchmarkChunks(
    const std::array<int, 3>& shape,
    const std::array<int, 3>& chunkShape,
    int level,
    std::size_t count,
    std::uint64_t seed)
{
    if (level < 0)
        throw std::invalid_argument("zarr benchmark level must be non-negative");

    std::array<std::uint64_t, 3> grid{};
    for (std::size_t axis = 0; axis < grid.size(); ++axis) {
        if (shape[axis] <= 0 || chunkShape[axis] <= 0)
            throw std::invalid_argument("zarr benchmark shape and chunk shape must be positive");
        grid[axis] = (static_cast<std::uint64_t>(shape[axis]) +
                      static_cast<std::uint64_t>(chunkShape[axis]) - 1) /
                     static_cast<std::uint64_t>(chunkShape[axis]);
    }

    const auto total = checkedProduct(grid);
    const auto selected = static_cast<std::size_t>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(count), total));
    std::vector<ChunkKey> keys;
    keys.reserve(selected);
    if (selected == 0)
        return keys;

    const auto start = splitmix64(seed) % total;
    std::uint64_t step = total == 1 ? 1 : splitmix64(seed + 1) % total;
    if (step == 0)
        step = 1;
    while (std::gcd(step, total) != 1) {
        ++step;
        if (step == total)
            step = 1;
    }

    auto linear = start;
    for (std::size_t index = 0; index < selected; ++index) {
        const auto x = linear % grid[2];
        const auto yz = linear / grid[2];
        const auto y = yz % grid[1];
        const auto z = yz / grid[1];
        keys.push_back({level,
                        static_cast<int>(z),
                        static_cast<int>(y),
                        static_cast<int>(x)});
        linear = (linear + step) % total;
    }
    return keys;
}

ZarrDownloadBenchmarkResult runZarrDownloadBenchmark(
    const OpenedChunkedZarr& opened,
    const ZarrDownloadBenchmarkOptions& options)
{
    if (options.level < 0 ||
        static_cast<std::size_t>(options.level) >= opened.fetchers.size() ||
        !opened.fetchers[static_cast<std::size_t>(options.level)]) {
        throw std::out_of_range("requested zarr benchmark level is not available");
    }
    if (options.workers == 0)
        throw std::invalid_argument("zarr benchmark worker count must be positive");
    if (options.outputDirectory)
        std::filesystem::create_directories(*options.outputDirectory);

    const auto keys = selectZarrDownloadBenchmarkChunks(
        opened.shapes.at(static_cast<std::size_t>(options.level)),
        opened.chunkShapes.at(static_cast<std::size_t>(options.level)),
        options.level,
        options.chunkCount,
        options.seed);

    std::optional<ChunkRequestScheduler::AdaptiveConcurrency> adaptive;
    if (options.schedule == ZarrDownloadSchedule::Adaptive) {
        adaptive = options.adaptive;
        adaptive->maximum = std::min(adaptive->maximum, options.workers);
    }
    ChunkRequestScheduler scheduler(options.workers, 7, {}, adaptive);
    ZarrDownloadBenchmarkResult result;
    result.requestedChunks = keys.size();
    result.concurrencySamples.push_back(
        {0, 0, scheduler.transferStats().admissionLimit, 0.0});

    std::atomic_size_t found{0};
    std::atomic_size_t missing{0};
    std::atomic_size_t httpErrors{0};
    std::atomic_size_t ioErrors{0};
    std::atomic_size_t decodeErrors{0};
    std::atomic_size_t sinkErrors{0};
    std::atomic_size_t encodedBytes{0};
    std::atomic_size_t completedChunks{0};
    std::atomic_size_t active{0};
    std::atomic_size_t peakActive{0};
    std::mutex samplesMutex;
    std::size_t accountedCompletedChunks = 0;
    std::size_t accountedEncodedBytes = 0;
    std::vector<double> latencyMilliseconds;
    std::optional<Clock::time_point> earliestTransferStart;
    std::optional<Clock::time_point> latestTransferCompletion;
    std::string firstError;

    const auto fetcher = opened.fetchers.at(static_cast<std::size_t>(options.level));
    const auto wallStarted = Clock::now();
    std::mutex progressMutex;
    std::condition_variable progressCv;
    std::jthread progressThread;
    if (options.progressCallback && options.progressInterval.count() > 0) {
        progressThread = std::jthread([&](std::stop_token stop) {
            std::unique_lock lock(progressMutex);
            while (!progressCv.wait_for(lock, options.progressInterval, [&] {
                return stop.stop_requested();
            })) {
                const ZarrDownloadProgress progress{
                    scheduler.pending(),
                    scheduler.active(),
                    completedChunks.load(std::memory_order_relaxed),
                    encodedBytes.load(std::memory_order_relaxed),
                    std::chrono::duration<double>(Clock::now() - wallStarted).count(),
                    scheduler.transferStats()};
                lock.unlock();
                options.progressCallback(progress);
                lock.lock();
            }
        });
    }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const auto key = keys[index];
        scheduler.submit(index + 1, {}, 0, 0, [&, key] {
            const auto activeNow = active.fetch_add(1, std::memory_order_relaxed) + 1;
            auto observedPeak = peakActive.load(std::memory_order_relaxed);
            while (activeNow > observedPeak &&
                   !peakActive.compare_exchange_weak(
                       observedPeak, activeNow, std::memory_order_relaxed)) {
            }

            const auto started = Clock::now();
            ChunkFetchResult fetch;
            try {
                fetch = fetcher->fetchEncoded(key);
            } catch (const std::exception& e) {
                fetch.status = ChunkFetchStatus::IoError;
                fetch.message = e.what();
            } catch (...) {
                fetch.status = ChunkFetchStatus::IoError;
                fetch.message = "unknown chunk fetch exception";
            }
            const auto completed = Clock::now();
            active.fetch_sub(1, std::memory_order_relaxed);

            const double latency = std::chrono::duration<double, std::milli>(
                completed - started).count();
            if (fetch.status == ChunkFetchStatus::Found) {
                ++found;
                const auto bytes = fetch.bytes.size();
                encodedBytes.fetch_add(bytes, std::memory_order_relaxed);

                {
                    // Preserve the scheduler's transfer-completion order in
                    // the reported admission history. Concurrent fetches can
                    // otherwise observe and publish stale intermediate stats.
                    std::lock_guard lock(samplesMutex);
                    if (bytes != 0)
                        scheduler.recordSuccessfulTransfer(bytes, started, completed);
                    ++accountedCompletedChunks;
                    accountedEncodedBytes += bytes;
                    const auto stats = scheduler.transferStats();
                    earliestTransferStart = earliestTransferStart
                        ? std::min(*earliestTransferStart, started)
                        : started;
                    latestTransferCompletion = latestTransferCompletion
                        ? std::max(*latestTransferCompletion, completed)
                        : completed;
                    if (result.concurrencySamples.back().admissionLimit !=
                        stats.admissionLimit) {
                        result.concurrencySamples.push_back(
                            {accountedCompletedChunks,
                             accountedEncodedBytes,
                             stats.admissionLimit,
                             stats.bytesPerSecond});
                    }
                }

                if (options.outputDirectory) {
                    try {
                        std::ofstream output(
                            chunkOutputPath(*options.outputDirectory, key),
                            std::ios::binary | std::ios::trunc);
                        output.write(reinterpret_cast<const char*>(fetch.bytes.data()),
                                     static_cast<std::streamsize>(fetch.bytes.size()));
                        if (!output)
                            throw std::runtime_error("failed to write temporary chunk payload");
                    } catch (const std::exception& e) {
                        ++sinkErrors;
                        std::lock_guard lock(samplesMutex);
                        if (firstError.empty())
                            firstError = e.what();
                    }
                }

            } else {
                if (fetch.status == ChunkFetchStatus::Missing)
                    ++missing;
                else if (fetch.status == ChunkFetchStatus::HttpError)
                    ++httpErrors;
                else if (fetch.status == ChunkFetchStatus::IoError)
                    ++ioErrors;
                else
                    ++decodeErrors;
                std::lock_guard lock(samplesMutex);
                if (firstError.empty() && !fetch.message.empty())
                    firstError = fetch.message;
            }
            {
                std::lock_guard lock(samplesMutex);
                latencyMilliseconds.push_back(latency);
            }
            completedChunks.fetch_add(1, std::memory_order_relaxed);
        });
    }
    scheduler.waitIdle();
    const auto wallCompleted = Clock::now();
    if (progressThread.joinable()) {
        progressThread.request_stop();
        progressCv.notify_all();
        progressThread.join();
    }
    if (options.progressCallback) {
        options.progressCallback({
            scheduler.pending(),
            scheduler.active(),
            completedChunks.load(std::memory_order_relaxed),
            encodedBytes.load(std::memory_order_relaxed),
            std::chrono::duration<double>(wallCompleted - wallStarted).count(),
            scheduler.transferStats()});
    }

    result.foundChunks = found.load();
    result.missingChunks = missing.load();
    result.httpErrors = httpErrors.load();
    result.ioErrors = ioErrors.load();
    result.decodeErrors = decodeErrors.load();
    result.sinkErrors = sinkErrors.load();
    result.encodedBytes = encodedBytes.load();
    result.peakActive = peakActive.load();
    result.wallSeconds = std::chrono::duration<double>(
        wallCompleted - wallStarted).count();
    if (earliestTransferStart && latestTransferCompletion) {
        result.transferWindowSeconds = std::chrono::duration<double>(
            *latestTransferCompletion - *earliestTransferStart).count();
    }
    result.finalTransferStats = scheduler.transferStats();
    result.firstError = std::move(firstError);

    std::sort(latencyMilliseconds.begin(), latencyMilliseconds.end());
    if (!latencyMilliseconds.empty()) {
        result.latencyMinimumMilliseconds = latencyMilliseconds.front();
        result.latencyMaximumMilliseconds = latencyMilliseconds.back();
        result.latencyMeanMilliseconds = std::accumulate(
            latencyMilliseconds.begin(), latencyMilliseconds.end(), 0.0) /
            static_cast<double>(latencyMilliseconds.size());
        result.latencyP50Milliseconds = percentile(latencyMilliseconds, 0.50);
        result.latencyP95Milliseconds = percentile(latencyMilliseconds, 0.95);
    }
    return result;
}

} // namespace vc::render
