#include <vc_reader/cache.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/source_reader.hpp>
#include <vc_reader/s3_listing.hpp>
#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace vcr::cache {

namespace {

// Reads chunks 0..total-1 (chunkAt maps one to its grid index) through the array's caches.
template <typename ChunkAt>
PrefillResult prefillChunks(Array& array, std::size_t total, ChunkAt chunkAt, const std::atomic<bool>* cancel,
                            const PrefillProgress& progress)
{
    PrefillResult result;
    const auto cancelled = [&] { return cancel && cancel->load(std::memory_order_acquire); };
    const auto fetchedBefore = array.stats().remote_fetches.load();
    const auto writeErrorsBefore = array.stats().write_errors.load();
    try {
        result.totalChunks = total;
        std::mutex mutex;
        array.parallel_for(result.totalChunks, [&](std::size_t i) {
            if (cancelled()) return;
            const Index idx = chunkAt(i);
            bool stored = false;
            std::string error;
            try { stored = array.chunk(idx) != nullptr; }
            catch (const std::exception& e) { error = e.what(); }
            std::lock_guard lock(mutex);
            ++result.resolvedChunks;
            if (!error.empty()) {
                ++result.errorChunks;
                if (result.message.empty()) result.message = error;
            } else if (stored) ++result.dataChunks;
            else ++result.emptyChunks;
            if (progress && (result.resolvedChunks == result.totalChunks || result.resolvedChunks % 32 == 0))
                progress(result.resolvedChunks, result.totalChunks);
        });
        array.flush();
        if (cancelled() && result.resolvedChunks < result.totalChunks) {
            result.status = PrefillResult::Status::Cancelled;
            result.message = "cancelled";
        } else if (result.errorChunks) {
            result.status = PrefillResult::Status::Failed;
            result.message = std::to_string(result.errorChunks) + " chunk(s) failed: " + result.message;
        } else if (array.stats().write_errors.load() > writeErrorsBefore) {
            result.status = PrefillResult::Status::Failed;
            result.message = "cache write failed: " + array.last_write_error();
        } else result.message = "completed";
    } catch (const std::exception& e) {
        result.status = PrefillResult::Status::Failed;
        result.message = e.what();
    }
    result.fetchedChunks = array.stats().remote_fetches.load() - fetchedBefore;
    return result;
}

} // namespace

PrefillResult prefill(Array& array, const std::atomic<bool>* cancel, const PrefillProgress& progress)
{
    const auto grid = array.chunk_grid();
    std::size_t total = 1;
    for (const auto n : grid) {
        if (n < 0 || (n && total > std::numeric_limits<std::size_t>::max() / n)) {
            PrefillResult result;
            result.status = PrefillResult::Status::Failed;
            result.message = "chunk grid size overflow";
            return result;
        }
        total *= static_cast<std::size_t>(n);
    }
    return prefillChunks(array, total, [&](std::size_t i) {
        Index idx(grid.size());
        for (std::size_t d = grid.size(); d-- > 0;) {
            idx[d] = static_cast<std::int64_t>(i % static_cast<std::size_t>(grid[d]));
            i /= static_cast<std::size_t>(grid[d]);
        }
        return idx;
    }, cancel, progress);
}

PrefillResult prefill(Array& array, const std::vector<Index>& chunks, const std::atomic<bool>* cancel,
                      const PrefillProgress& progress)
{
    return prefillChunks(array, chunks.size(), [&](std::size_t i) { return chunks[i]; }, cancel, progress);
}

DownloadResult downloadPrefix(const std::string& location, const zarr::OpenOptions& options,
                              int workers, const DownloadCallback& progress, const std::atomic<bool>* cancel)
{
    if (workers < 1) throw std::invalid_argument("download workers must be positive");
    if (options.cacheDirs.empty() || !options.cacheWrites)
        throw std::invalid_argument("prefix download requires a writable persistent cache");
    DownloadResult result;
    auto& state = result.progress;
    const auto cancelled = [&] { return cancel && cancel->load(std::memory_order_acquire); };
    state.status = "listing";
    if (progress) progress(state);
    std::vector<s3::Object> objects;
    try { objects = zarr::listObjects(options.access, location); }
    catch (const std::exception& e) {
        result.error = std::string("listing failed: ") + e.what();
        return result;
    }
    std::erase_if(objects, [](const auto& object) { return object.key.empty() || object.key.back() == '/'; });
    if (objects.empty()) {
        result.error = "no objects found under " + location;
        return result;
    }
    if (objects.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("too many files in prefix");
    if (cancelled()) {
        result.cancelled = true;
        result.error = "cancelled";
        return result;
    }
    state.status = "downloading";
    state.totalFiles = static_cast<int>(objects.size());
    for (const auto& object : objects) state.totalBytes += object.size;
    if (progress) progress(state);
    sources::EncodedFiles files([options](const std::string&) { return options; });
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stopped{false};
    std::mutex mutex;
    std::exception_ptr callbackError;
    std::string firstError;
    const auto prefix = location.ends_with('/') ? location : location + '/';
    auto worker = [&] {
        while (!cancelled() && !stopped.load()) {
            const auto index = next.fetch_add(1);
            if (index >= objects.size()) return;
            const auto& object = objects[index];
            std::string error;
            try { (void)files.open(prefix + object.key); }
            catch (const std::exception& e) { error = e.what(); }
            std::lock_guard lock(mutex);
            if (error.empty()) {
                ++state.completedFiles;
                state.completedBytes += object.size;
            } else {
                ++state.failedFiles;
                if (firstError.empty()) firstError = error;
            }
            state.fileName = object.key;
            if (progress && !callbackError) {
                try { progress(state); }
                catch (...) { callbackError = std::current_exception(); stopped = true; }
            }
        }
    };
    {
        std::vector<std::jthread> threads;
        for (int i = 0; i < std::min(workers, state.totalFiles); ++i) threads.emplace_back(worker);
    }
    if (callbackError) std::rethrow_exception(callbackError);
    result.cancelled = cancelled();
    if (result.cancelled) result.error = "cancelled";
    else if (state.failedFiles)
        result.error = "failed to download " + std::to_string(state.failedFiles) + " file(s): " + firstError;
    return result;
}

} // namespace vcr::cache
