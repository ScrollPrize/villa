#pragma once

#include "vc/core/render/ChunkRequestScheduler.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc::render {

enum class ZarrDownloadSchedule {
    Adaptive,
    Fixed,
};

struct ZarrDownloadProgress {
    std::size_t queuedChunks = 0;
    std::size_t downloadingChunks = 0;
    std::size_t completedChunks = 0;
    std::size_t encodedBytes = 0;
    double elapsedSeconds = 0.0;
    ChunkRequestScheduler::TransferStats transferStats;
};

struct ZarrDownloadBenchmarkOptions {
    int level = 0;
    // Number of unique candidate keys and maximum queued request slots.
    std::size_t chunkCount = 256;
    // Positive durations continuously replenish the candidate slots until the
    // deadline. Zero retains the finite one-request-per-candidate mode.
    std::chrono::duration<double> runDuration{0.0};
    std::uint64_t seed = 0;
    std::size_t workers = 64;
    ZarrDownloadSchedule schedule = ZarrDownloadSchedule::Adaptive;
    ChunkRequestScheduler::AdaptiveConcurrency adaptive;
    // Empty means discard the encoded payload after accounting it.
    std::optional<std::filesystem::path> outputDirectory;
    std::chrono::milliseconds progressInterval{0};
    std::function<void(const ZarrDownloadProgress&)> progressCallback;
};

struct ZarrDownloadConcurrencySample {
    std::size_t completedChunks = 0;
    std::size_t encodedBytes = 0;
    std::size_t admissionLimit = 0;
    double estimatedBytesPerSecond = 0.0;
};

struct ZarrDownloadBenchmarkResult {
    std::size_t requestedChunks = 0;
    std::size_t foundChunks = 0;
    std::size_t missingChunks = 0;
    std::size_t httpErrors = 0;
    std::size_t ioErrors = 0;
    std::size_t decodeErrors = 0;
    std::size_t sinkErrors = 0;
    std::size_t encodedBytes = 0;
    std::size_t peakActive = 0;
    double wallSeconds = 0.0;
    double transferWindowSeconds = 0.0;
    double latencyMeanMilliseconds = 0.0;
    double latencyP50Milliseconds = 0.0;
    double latencyP95Milliseconds = 0.0;
    double latencyMinimumMilliseconds = 0.0;
    double latencyMaximumMilliseconds = 0.0;
    ChunkRequestScheduler::TransferStats finalTransferStats;
    std::vector<ZarrDownloadConcurrencySample> concurrencySamples;
    std::string firstError;
};

// Select unique chunks spread deterministically across the complete logical
// chunk grid. `chunkShape` is the decoded/sub-chunk shape used by rendering,
// including for sharded Zarr arrays.
std::vector<ChunkKey> selectZarrDownloadBenchmarkChunks(
    const std::array<int, 3>& shape,
    const std::array<int, 3>& chunkShape,
    int level,
    std::size_t count,
    std::uint64_t seed);

// Benchmark encoded source transfers using the same fetcher and adaptive
// scheduler used by remote rendering. Decoding and the persistent cache are
// deliberately excluded so the result isolates source download behavior.
ZarrDownloadBenchmarkResult runZarrDownloadBenchmark(
    const OpenedChunkedZarr& opened,
    const ZarrDownloadBenchmarkOptions& options);

} // namespace vc::render
