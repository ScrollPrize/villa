#pragma once

#include "vc/core/render/IChunkedArray.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc::render {

class ChunkCache final : public IChunkedArray {
public:
    struct LevelInfo {
        std::array<int, 3> shape{};
        std::array<int, 3> chunkShape{};
        LevelTransform transform{};
    };

    struct Options {
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        // Bound resolved non-data entries (all-fill/missing/error). These
        // entries are small individually, but sparse remote volumes can touch
        // unbounded empty chunk grids during exploration.
        std::size_t metadataEntryCapacity = 1ULL << 20;
        // Number of process-wide chunk I/O workers used by this cache. The
        // pool is shared by caches with the same worker count and is not
        // destroyed when a viewer is closed.
        std::size_t maxConcurrentReads = 16;
        bool detectAllFillChunks = true;
        std::optional<std::filesystem::path> persistentCachePath;
        // Store raw (".bin") persistent-cache chunks zstd-compressed
        // (".zst"). Reading handles both formats regardless of this flag;
        // it only selects the write format. Combined (OR) with the
        // process-wide default below at construction. Chunks whose source
        // encoding is already compact (".c3d") are never recompressed.
        bool compressPersistentCache = false;
        // Near-lossless quantization bin width for compressed persistent
        // writes (1 = lossless, 3 = max error +-1, 5 = +-2; see
        // CacheCompression.hpp). Combined (max) with the process-wide
        // default below at construction.
        int cacheQuantBinWidth = 1;
        // Entries read within this window are skipped by capacity eviction
        // (until the hard ceiling of twice the capacity): they belong to a
        // view that is still being rendered, so dropping them frees little
        // memory (render workers pin the bytes for the frame anyway) and
        // guarantees a refetch plus a visible blank on the next re-render.
        // Zero restores strict-capacity LRU.
        std::chrono::milliseconds evictionProtectionWindow{3000};
    };

    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t decodedByteCapacity = 0;
        std::size_t persistentCacheBytes = 0;
        bool persistentCacheEnabled = false;
        bool persistentCacheScanInFlight = false;
        std::size_t remoteFetchesInFlight = 0;
        double remoteDownloadBytesPerSecond = 0.0;
        // Monotonic count of capacity enforcements that stayed above the
        // hard ceiling because the excess belonged to still-active view
        // requests. A consumer that sees this advance across one of its own
        // renders is sharing a cache whose combined live working set does
        // not fit, and should render coarser instead of refetching.
        std::size_t viewProtectionStalls = 0;
    };

    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype);
    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype,
               Options options);
    ~ChunkCache() override;

    int numLevels() const override;
    std::array<int, 3> shape(int level) const override;
    std::array<int, 3> chunkShape(int level) const override;
    ChunkDtype dtype() const override;
    double fillValue() const override;
    LevelTransform levelTransform(int level) const override;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    Stats stats() const;
    void invalidate();
    // Marks the start of a view render. Entries touched while any view
    // request is active are protected from capacity eviction (up to a
    // last-resort OOM ceiling), so concurrent renders sharing this cache
    // cannot evict each other's working set and force an endless
    // refetch/re-render loop. The returned token must be passed to
    // endViewRequest when the render's sampling is done.
    std::int64_t beginViewRequest();
    void endViewRequest(std::int64_t token);
    void waitForPersistentWrites() const;

    // Process-wide default for Options::compressPersistentCache, OR-ed into
    // every cache built afterwards. Lets an application apply a user setting
    // without threading it through each construction site.
    static void setPersistentCompressionDefault(bool enabled);
    static bool persistentCompressionDefault();

    // Process-wide default for Options::cacheQuantBinWidth (same pattern as
    // the compression default above; the larger of the two values wins).
    static void setPersistentQuantizationDefault(int binWidth);
    static int persistentQuantizationDefault();

private:
    enum class EntryStatus {
        InFlight,
        Missing,
        AllFill,
        Data,
        Error
    };

    struct Entry {
        EntryStatus status = EntryStatus::InFlight;
        std::shared_ptr<const std::vector<std::byte>> bytes;
        std::string error;
        std::size_t decodedBytes = 0;
        bool persisted = false;
        bool persistentWriteQueued = false;
        bool inLru = false;
        int basePriority = 0;
        std::int64_t priority = 0;
        std::uint64_t fetchSerial = 0;
        std::chrono::steady_clock::time_point lastTouch{};
        std::int64_t lastEpoch = 0;
        std::list<ChunkKey>::iterator lruIt;
    };

    struct State {
        State(std::vector<LevelInfo> levels,
              std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
              double fillValue,
              ChunkDtype dtype,
              Options options)
            : levels_(std::move(levels))
            , fetchers_(std::move(fetchers))
            , fillValue_(fillValue)
            , dtype_(dtype)
            , options_(std::move(options))
        {}

        std::vector<LevelInfo> levels_;
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers_;
        double fillValue_ = 0.0;
        ChunkDtype dtype_ = ChunkDtype::UInt8;
        Options options_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
        std::list<ChunkKey> lru_;
        std::size_t decodedBytes_ = 0;
        std::uint64_t generation_ = 0;
        std::int64_t viewEpoch_ = 1;
        // Epochs of view requests whose renders are still sampling (epoch →
        // active request count). Entries last touched at or after the oldest
        // active epoch are protected from eviction.
        std::map<std::int64_t, int> activeViewRequests_;
        std::size_t viewProtectionStalls_ = 0;
        std::uint64_t nextFetchSerial_ = 1;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
        std::size_t remoteFetchesInFlight_ = 0;
        std::deque<std::pair<std::chrono::steady_clock::time_point, std::size_t>> remoteDownloadHistory_;
        std::atomic<std::int64_t> persistentCacheBytes_{0};
        std::atomic_bool persistentCacheScanInFlight_{false};
        std::atomic_size_t persistentWritesInFlight_{0};
    };

    static ChunkResult resultFromEntryLocked(State& state, const ChunkKey& key, Entry& entry);
    static int fetchBasePriority(const State& state, const ChunkKey& key, int priorityOffset);
    static void queueFetchLocked(const std::shared_ptr<State>& state,
                                 const ChunkKey& key,
                                 std::uint64_t generation,
                                 int priorityOffset);
    static void fetchAndStore(const std::shared_ptr<State>& state,
                              ChunkKey key,
                              std::uint64_t generation,
                              std::uint64_t fetchSerial);
    static void probePersistentAndStore(const std::shared_ptr<State>& state,
                                        ChunkKey key,
                                        std::uint64_t generation,
                                        std::uint64_t fetchSerial,
                                        std::int64_t priority);
    static void storeFetchResultLocked(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       ChunkFetchResult fetch,
                                       bool loadedFromPersistentCache);
    static std::optional<std::vector<std::byte>> readPersistent(const State& state, const ChunkKey& key);
    static bool readPersistentEmpty(const State& state, const ChunkKey& key);
    static bool queuePersistentWrite(const std::shared_ptr<State>& state,
                                     const ChunkKey& key,
                                     std::shared_ptr<const std::vector<std::byte>> bytes);
    static bool queuePersistentEmptyWrite(const std::shared_ptr<State>& state,
                                          const ChunkKey& key);
    static bool writePersistent(State& state, const ChunkKey& key, const std::vector<std::byte>& bytes);
    static bool writePersistentEmpty(State& state, const ChunkKey& key);
    static std::filesystem::path persistentPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentCompressedPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentEmptyPath(const State& state, const ChunkKey& key);
    static bool persistentEntryIsRaw(const State& state, const ChunkKey& key);
    static void startPersistentCacheSizeScan(const std::shared_ptr<State>& state);
    static std::size_t persistentCacheBytes(
        const std::filesystem::path& path,
        std::filesystem::file_time_type cutoff);
    static std::optional<std::size_t> regularFileSize(const std::filesystem::path& path);
    static void addPersistentCacheBytesDelta(State& state, std::int64_t delta);
    static void pruneDownloadHistoryLocked(State& state, std::chrono::steady_clock::time_point now);
    static void touchLocked(State& state, const ChunkKey& key, Entry& entry);
    static void enforceCapacityLocked(const std::shared_ptr<State>& state);
    static bool isValidKey(const State& state, const ChunkKey& key);
    static bool isAllFill(const State& state, const std::vector<std::byte>& bytes);
    static std::size_t dtypeSize(ChunkDtype dtype);
    static std::size_t expectedChunkBytes(const State& state, const ChunkKey& key);
    static void notifyListeners(const std::shared_ptr<State>& state);
    static void waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key);

    std::shared_ptr<State> state_;
};

} // namespace vc::render
