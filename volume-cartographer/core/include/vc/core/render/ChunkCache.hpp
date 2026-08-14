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
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc::render {

class PersistentZarrCacheBudget;
class ChunkCache;
class ChunkRequestScheduler;
class ChunkRequestSelectionGate;
struct ChunkWorkPriority;

struct ChunkCacheLevelInfo {
    std::array<int, 3> shape{};
    std::array<int, 3> chunkShape{};
    IChunkedArray::LevelTransform transform{};
};

struct ChunkCacheOptions {
    // Bound resolved non-data entries for sparse volumes.
    std::size_t metadataEntryCapacity = 1ULL << 20;
    bool detectAllFillChunks = true;
    std::optional<std::filesystem::path> persistentCachePath;
    // Optional root registered with PersistentZarrCacheBudget.
    std::optional<std::filesystem::path> persistentCacheBudgetRoot;
    // Persistent writes only. Readers accept compressed and raw entries.
    bool compressPersistentCache = false;
    // Near-lossless persistent-cache quantization width; one is lossless.
    int cacheQuantBinWidth = 1;
};

// Application-owned decoded chunk-cache service. Source identity strings are
// interned when a source handle is acquired; all hot lookups use the numeric
// VolumeSourceId stored on that handle. Fetch policy belongs to the service and
// source acquisition cannot change it.
class ChunkCacheService final : public std::enable_shared_from_this<ChunkCacheService> {
public:
    struct AdaptiveDownloadState {
        std::size_t settledAdmissionLimit = 0;
        double longTermBytesPerSecond = 0.0;
        std::size_t maximumSaturatedParallelism = 0;
        double saturatedBytesPerSecondPerWorker = 0.0;
    };

    struct FetchConcurrency {
        // Number of physical source-read workers owned by this service.
        std::size_t workerCapacity = 16;
        // Fixed admission limit, or the upper adaptive bound.
        std::size_t maxConcurrentReads = 16;
        bool adaptive = false;
    };

    struct Options {
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget;
        FetchConcurrency fetchConcurrency;
        std::optional<AdaptiveDownloadState> initialAdaptiveDownloadState;
    };

    ChunkCacheService();
    explicit ChunkCacheService(Options options);
    ~ChunkCacheService();

    ChunkCacheService(const ChunkCacheService&) = delete;
    ChunkCacheService& operator=(const ChunkCacheService&) = delete;

    std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget() const;
    // Returns the reusable adaptive download model. Runtime probe phases and
    // stability windows are deliberately excluded.
    std::optional<AdaptiveDownloadState> adaptiveDownloadState() const;
    std::shared_ptr<ChunkCache> acquireSource(
        std::string sourceIdentity,
        std::vector<ChunkCacheLevelInfo> levels,
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
        double fillValue,
        ChunkDtype dtype,
        ChunkCacheOptions options = {});
    // Changes service-wide source-read admission in place. Running and queued
    // work is neither cancelled nor restarted.
    void configureFetchConcurrency(std::size_t maxConcurrentReads,
                                   bool adaptive);
    FetchConcurrency fetchConcurrency() const;
    std::size_t sourceCount() const;
    bool invalidateSource(std::string_view sourceIdentity);

private:
    friend class ChunkCache;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class ChunkCache final : public IChunkedArray {
public:
    using RemoteFetchActivityCallbackId = std::uint64_t;
    using RemoteFetchActivityCallback =
        std::function<void(const ChunkKey& key, bool active)>;

    using LevelInfo = ChunkCacheLevelInfo;
    using Options = ChunkCacheOptions;

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
        std::size_t pendingDecodeTasks = 0;
        // Unresolved chunk requests (queued or executing), indexed by pyramid
        // level. Persistent-cache probes count while the requested chunk is
        // still unavailable to rendering.
        std::vector<std::size_t> unresolvedFetchesByLevel;

    };

    struct PersistentChunkDependency {
        ChunkKey key;
        bool valid = false;
        std::optional<std::string> sourceChunkKey;
        std::filesystem::path persistentPath;
        std::filesystem::path persistentEmptyPath;
        std::string persistentExtension;
        bool sourcePayloadMatchesPersistentCache = false;
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
    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype,
               Options options,
               ChunkCacheService::Options serviceOptions);
    ~ChunkCache() override;

    VolumeSourceId sourceId() const noexcept;

    int numLevels() const override;
    std::array<int, 3> shape(int level) const override;
    std::array<int, 3> chunkShape(int level) const override;
    ChunkDtype dtype() const override;
    double fillValue() const override;
    LevelTransform levelTransform(int level) const override;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override;
    ChunkResult tryGetChunk(int level, int iz, int iy, int ix,
                            const ChunkRequestContext& request) override;
    ChunkResult getChunkIfCached(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys,
                        bool wait,
                        int priorityOffset,
                        const ChunkRequestContext& request) override;
    void replaceViewDemand(const ChunkRequestContext& request,
                           const std::array<float, 2>& focus,
                           std::vector<ChunkViewportSample> samples) override;
    void markViewActive(std::uint64_t viewId) override;
    void clearSourceViewDemand(std::uint64_t viewId,
                               std::uint64_t viewVersion = 0) override;
    void clearViewDemand(std::uint64_t viewId,
                         std::uint64_t viewVersion = 0) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    // Reports actual source fetch execution, excluding persistent-cache probes
    // and decoded-cache hits. Callbacks run on fetch workers.
    RemoteFetchActivityCallbackId addRemoteFetchActivityListener(
        RemoteFetchActivityCallback cb);
    void removeRemoteFetchActivityListener(RemoteFetchActivityCallbackId id);
    std::vector<ChunkKey> activeRemoteFetches() const;

    PersistentChunkDependency persistentChunkDependency(int level, int iz, int iy, int ix) const;

    Stats stats() const;
    void invalidate();
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
    friend class ChunkCacheService;
    struct State;
    ChunkCache(std::shared_ptr<ChunkCacheService> service,
               std::shared_ptr<State> state);
    enum class EntryStatus {
        InFlight,
        Missing,
        AllFill,
        Data,
        Error
    };

    struct ViewDemandSlot {
        std::uint64_t version = 0;
        int relativeLevel = 0;
        std::optional<float> distanceSquared;
    };

    struct PersistentProbeResult {
        bool compressedData = false;
        bool primaryData = false;
        bool empty = false;

        bool hasData() const noexcept
        {
            return compressedData || primaryData;
        }
    };

    struct Entry {
        EntryStatus status = EntryStatus::InFlight;
        std::shared_ptr<const std::vector<std::byte>> bytes;
        std::string error;
        std::size_t decodedBytes = 0;
        bool persisted = false;
        bool persistentWriteQueued = false;
        bool unresolvedCounted = false;
        bool inLru = false;
        int basePriority = 0;
        std::uint64_t fetchSerial = 0;
        std::uint64_t probeTaskId = 0;
        std::uint64_t fetchTaskId = 0;
        std::uint64_t decodeTaskId = 0;
        std::uint64_t budgetTouch = 0;
        bool backgroundDemand = false;
        std::unordered_map<std::uint64_t, ViewDemandSlot> viewDemands;
        std::list<ChunkKey>::iterator lruIt;
    };

    struct State {
        State(std::vector<LevelInfo> levels,
              std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
              double fillValue,
              ChunkDtype dtype,
              Options options,
              std::size_t decodedByteCapacity,
              std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget,
              VolumeSourceId sourceId,
              std::string sourceIdentity)
            : levels_(std::move(levels))
            , fetchers_(std::move(fetchers))
            , fillValue_(fillValue)
            , dtype_(dtype)
            , options_(std::move(options))
            , decodedByteCapacity_(decodedByteCapacity)
            , decodedByteBudget_(std::move(decodedByteBudget))
            , sourceId_(sourceId)
            , sourceIdentity_(std::move(sourceIdentity))
            , schedulerGroup_(ChunkCache::nextSchedulerGroup())
        {
            unresolvedFetchesByLevel_.resize(levels_.size(), 0);
            persistentExtensions_.resize(fetchers_.size());
            for (std::size_t level = 0; level < fetchers_.size(); ++level) {
                if (fetchers_[level]) {
                    persistentExtensions_[level] =
                        fetchers_[level]->persistentCacheExtension(
                            ChunkKey{static_cast<int>(level), 0, 0, 0});
                }
            }
        }

        std::vector<LevelInfo> levels_;
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers_;
        std::vector<std::string> persistentExtensions_;
        double fillValue_ = 0.0;
        ChunkDtype dtype_ = ChunkDtype::UInt8;
        Options options_;
        std::size_t decodedByteCapacity_ = 0;
        std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget_;
        VolumeSourceId sourceId_{};
        std::string sourceIdentity_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
        std::list<ChunkKey> lru_;
        std::vector<std::size_t> unresolvedFetchesByLevel_;
        std::size_t decodedBytes_ = 0;
        std::uint64_t decodedBudgetRegistration_ = 0;
        std::uint64_t generation_ = 0;
        std::uint64_t fetcherGeneration_ = 0;
        // Shared executor tasks carry this cache-specific group/epoch so
        // invalidation can cancel only this source's stale pending tasks.
        const std::uint64_t schedulerGroup_;
        std::uint64_t schedulerEpoch_ = 0;
        std::uint64_t nextFetchSerial_ = 1;
        std::weak_ptr<ChunkRequestScheduler> probeScheduler_;
        std::weak_ptr<ChunkRequestScheduler> fetchScheduler_;
        std::weak_ptr<ChunkRequestScheduler> decodeScheduler_;
        std::shared_ptr<ChunkRequestSelectionGate> schedulerSelectionGate_;
        std::shared_ptr<std::atomic<std::uint64_t>> activeViewId_;
        std::shared_ptr<std::atomic<std::uint64_t>> nextTaskId_;
        struct ViewSnapshot {
            std::uint64_t version = 0;
            bool closed = false;
            std::unordered_map<int, int> relativeLevels;
        };
        std::unordered_map<std::uint64_t, ViewSnapshot> viewSnapshots_;
        std::unordered_map<std::uint64_t,
                           std::unordered_set<ChunkKey, ChunkKeyHash>> viewDemandKeys_;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
        std::unordered_map<RemoteFetchActivityCallbackId,
                           RemoteFetchActivityCallback> remoteFetchCallbacks_;
        std::unordered_map<ChunkKey,
                           std::unordered_set<std::uint64_t>,
                           ChunkKeyHash> activeRemoteFetches_;
        std::size_t remoteFetchesInFlight_ = 0;
        std::atomic<std::int64_t> persistentCacheBytes_{0};
        std::atomic_bool persistentCacheScanInFlight_{false};
        std::atomic_size_t persistentWritesInFlight_{0};
        std::shared_ptr<PersistentZarrCacheBudget> persistentBudget_;

    };

    struct FetchContext {
        std::uint64_t generation = 0;
        std::uint64_t fetcherGeneration = 0;
        std::uint64_t fetchSerial = 0;
        std::uint64_t schedulerEpoch = 0;
        std::shared_ptr<IChunkFetcher> fetcher;
        std::shared_ptr<ChunkRequestScheduler> fetchScheduler;
    };

    static ChunkResult resultFromEntryLocked(
        State& state, const ChunkKey& key, Entry& entry, bool promote = true);
    static int fetchBasePriority(const State& state, const ChunkKey& key, int priorityOffset);
    static ChunkWorkPriority workPriorityLocked(const State& state,
                                                const ChunkKey& key,
                                                const Entry& entry);
    static void reprioritizeEntryLocked(const State& state,
                                        const ChunkKey& key,
                                        Entry& entry);
    static bool addRequestDemandLocked(State& state,
                                       const ChunkKey& key,
                                       Entry& entry,
                                       const ChunkRequestContext& request);
    static bool hasDemandLocked(const Entry& entry);
    static bool cancelUndemandedEntryLocked(State& state,
                                            const ChunkKey& key,
                                            Entry& entry);
    static void eraseUnresolvedEntryLocked(State& state,
                                           const ChunkKey& key);
    static void queueFetchLocked(const std::shared_ptr<State>& state,
                                 const ChunkKey& key,
                                 std::uint64_t generation,
                                 int priorityOffset);
    static void probePersistentAndDispatch(const std::shared_ptr<State>& state,
                                           ChunkKey key,
                                           FetchContext context);
    static void fetchRemoteAndDispatch(const std::shared_ptr<State>& state,
                                       ChunkKey key,
                                       FetchContext context);
    static void decodePersistentAndStore(const std::shared_ptr<State>& state,
                                         ChunkKey key,
                                         FetchContext context,
                                         PersistentProbeResult probe);
    static void decodeFetchedAndStore(
        const std::shared_ptr<State>& state,
        ChunkKey key,
        FetchContext context,
        std::shared_ptr<ChunkFetchResult> fetched);
    static void queueRemoteFetch(const std::shared_ptr<State>& state,
                                 const ChunkKey& key,
                                 FetchContext context);
    static void queuePersistentDecode(const std::shared_ptr<State>& state,
                                      const ChunkKey& key,
                                      FetchContext context,
                                      PersistentProbeResult probe);
    static void queueFetchedDecode(const std::shared_ptr<State>& state,
                                   const ChunkKey& key,
                                   FetchContext context,
                                   ChunkFetchResult fetched);
    static void finishAndStore(const std::shared_ptr<State>& state,
                               const ChunkKey& key,
                               FetchContext context,
                               ChunkFetchResult fetch,
                               bool loadedFromPersistentCache);
    static PersistentProbeResult probePersistent(const State& state,
                                                  const ChunkKey& key);
    static std::optional<std::vector<std::byte>> readPersistent(
        const State& state,
        const ChunkKey& key,
        const PersistentProbeResult& probe);
    static void storeFetchResultLocked(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       ChunkFetchResult fetch,
                                       bool loadedFromPersistentCache);
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
    static void notifyRemoteFetchListeners(const std::shared_ptr<State>& state,
                                           const ChunkKey& key,
                                           bool active);
    static void waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key);
    static std::uint64_t nextSchedulerGroup();
    static void invalidateState(const std::shared_ptr<State>& state);
    static void registerStateBudget(const std::shared_ptr<State>& state);
    static void unregisterStateBudget(State& state);

    static ChunkKey sourceKey(const State& state, ChunkKey key) noexcept;
    static ChunkKey fetcherKey(ChunkKey key) noexcept;
    static bool metadataCompatible(const State& state,
                                   const std::vector<LevelInfo>& levels,
                                   double fillValue,
                                   ChunkDtype dtype,
                                   const Options& options);
    static void validateSourceDefinition(
        const std::vector<LevelInfo>& levels,
        const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers);
    static void validateRefreshedFetchers(
        const State& state,
        const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers);
    static void refreshFetchers(
        const std::shared_ptr<State>& state,
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers);
    static void restartUnresolvedLocked(const std::shared_ptr<State>& state);
    static void clearSourceViewDemandLocked(State& state,
                                            std::uint64_t viewId,
                                            std::uint64_t viewVersion,
                                            bool closeView);

    std::shared_ptr<ChunkCacheService> service_;
    std::shared_ptr<State> state_;
    mutable std::mutex handleMutex_;
    std::unordered_set<ChunkReadyCallbackId> listenerIds_;
    std::unordered_set<RemoteFetchActivityCallbackId> remoteFetchListenerIds_;
};

} // namespace vc::render
