#pragma once

#include "vc/core/render/DecodedChunkCacheBudget.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc::render {

class PersistentZarrCacheBudget;

class ChunkCache final : public IChunkedArray {
public:
    struct LevelInfo {
        std::array<int, 3> shape{};
        std::array<int, 3> chunkShape{};
        LevelTransform transform{};
    };

    struct Options {
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        // Optional aggregate byte budget shared with other ChunkCache
        // instances. The local capacity above remains a per-cache safety
        // ceiling; the shared budget additionally constrains their sum.
        std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget;
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
        // When set to a root registered with PersistentZarrCacheBudget, disk
        // accounting and eviction are shared with every remote Zarr cache
        // beneath that root. Unregistered/core-only caches stay unlimited.
        std::optional<std::filesystem::path> persistentCacheBudgetRoot;
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
    };

    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t decodedByteCapacity = 0;
        std::size_t persistentCacheBytes = 0;
        bool persistentCacheEnabled = false;
        bool persistentCacheScanInFlight = false;
        bool persistentCacheTrimInFlight = false;
        bool persistentCacheLowSpace = false;
        std::size_t persistentCacheFreeBytes = 0;
        std::size_t persistentCacheMinimumFreeBytes = 0;
        std::optional<std::size_t> persistentCacheMaximumBytes;
        std::size_t remoteFetchesInFlight = 0;
        double remoteDownloadBytesPerSecond = 0.0;
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
    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    Stats stats() const;
    void invalidate();
    // Advances fetch priority so newer view renders supersede stale requests.
    void beginViewRequest();
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

    // Optional process-wide default aggregate budget. Applications install
    // this once; explicitly supplied budgets (for example the overlay pool)
    // take precedence.
    static void setDecodedByteBudgetDefault(
        const std::shared_ptr<DecodedChunkCacheBudget>& budget);
    static std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudgetDefault();

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
        std::uint64_t budgetTouch = 0;
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
        std::uint64_t decodedBudgetRegistration_ = 0;
        std::uint64_t generation_ = 0;
        std::int64_t viewEpoch_ = 1;
        std::uint64_t nextFetchSerial_ = 1;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
        std::size_t remoteFetchesInFlight_ = 0;
        std::deque<std::pair<std::chrono::steady_clock::time_point, std::size_t>> remoteDownloadHistory_;
        std::atomic<std::int64_t> persistentCacheBytes_{0};
        std::atomic_bool persistentCacheScanInFlight_{false};
        std::atomic_size_t persistentWritesInFlight_{0};
        std::shared_ptr<PersistentZarrCacheBudget> persistentBudget_;
    };

    static ChunkResult resultFromEntryLocked(
        State& state, const ChunkKey& key, Entry& entry, bool promote = true);
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
    static std::optional<std::uint64_t> oldestDecodedTouch(
        const std::shared_ptr<State>& state);
    static std::size_t evictOldestDecoded(const std::shared_ptr<State>& state);
    static std::size_t evictOldestDecodedLocked(const std::shared_ptr<State>& state);
    static void addDecodedBytesLocked(State& state, std::size_t bytes);
    static void removeDecodedBytesLocked(State& state, std::size_t bytes);
    static void enforceSharedBudget(const std::shared_ptr<State>& state);
    static bool isValidKey(const State& state, const ChunkKey& key);
    static bool isAllFill(const State& state, const std::vector<std::byte>& bytes);
    static std::size_t dtypeSize(ChunkDtype dtype);
    static std::size_t expectedChunkBytes(const State& state, const ChunkKey& key);
    static void notifyListeners(const std::shared_ptr<State>& state);
    static void waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key);

    std::shared_ptr<State> state_;
};

} // namespace vc::render
