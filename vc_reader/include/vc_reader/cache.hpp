#pragma once

#include <vc_reader/zarr.hpp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vcr::cache {

struct PrefillResult {
    enum class Status { Completed, Cancelled, Failed };
    Status status = Status::Completed;
    std::size_t totalChunks = 0, resolvedChunks = 0;
    std::size_t dataChunks = 0, emptyChunks = 0, errorChunks = 0, fetchedChunks = 0;
    std::string message;
};
using PrefillProgress = std::function<void(std::size_t resolved, std::size_t total)>;

// Read every chunk through the array's existing caches, then flush queued writes.
// Open with persistent caching and without a memory cache to prefill disk only.
// Counts remote fetches during this call; don't share the array with other reads
// if exact fetch counts are needed. Progress is serialized, every 32 chunks and
// at completion. Cancellation leaves completed chunks cached for a later run.
[[nodiscard]] PrefillResult prefill(Array& array,
                                    const std::atomic<bool>* cancel = nullptr,
                                    const PrefillProgress& progress = {});
// The same for the listed chunk-grid indices only (e.g. an offline_fill array's
// offline_misses), in order; totalChunks is their count.
[[nodiscard]] PrefillResult prefill(Array& array, const std::vector<Index>& chunks,
                                    const std::atomic<bool>* cancel = nullptr,
                                    const PrefillProgress& progress = {});

struct DownloadProgress {
    int totalFiles = 0, completedFiles = 0, failedFiles = 0;
    std::uint64_t totalBytes = 0, completedBytes = 0;
    std::string fileName;
    std::string status;  // "listing" or "downloading"
};
using DownloadCallback = std::function<void(const DownloadProgress&)>;
struct DownloadResult {
    DownloadProgress progress;
    bool cancelled = false;
    std::string error;
};

// List a remote prefix and open its files through EncodedFiles, using the supplied
// access and disk-cache options. Existing files are reused. No format-specific
// validation or completion marker is imposed. Progress callbacks are serialized;
// a throwing callback stops scheduling work and is rethrown after workers join.
// Persistent downloads require writable cacheDirs. workers must be positive.
[[nodiscard]] DownloadResult downloadPrefix(const std::string& location,
                                             const zarr::OpenOptions& options,
                                             int workers = 16,
                                             const DownloadCallback& progress = {},
                                             const std::atomic<bool>* cancel = nullptr);

} // namespace vcr::cache
