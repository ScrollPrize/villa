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

enum class PersistentCacheLayout {
    Auto,
    Legacy,
    ZarrMirror,
    // A regular, unsharded Zarr store with VC-Delta3D as its chunk codec.
    Delta3d,
};

enum class PersistentCacheEncoding {
    SourceMirror,
    Delta3dLossless,
};

struct PersistentCacheMetadataObject {
    std::string key;
    std::vector<std::byte> bytes;
};

struct ChunkCacheOptions {
    // Bound resolved non-data entries for sparse volumes.
    std::size_t metadataEntryCapacity = 1ULL << 20;
    bool detectAllFillChunks = true;
    std::optional<std::filesystem::path> persistentCachePath;
    // Optional root registered with PersistentZarrCacheBudget.
    std::optional<std::filesystem::path> persistentCacheBudgetRoot;
    // Auto selects an existing legacy footprint, otherwise an exact Zarr
    // mirror when metadata and physical-object fetchers are available.
    // Delta3d is selected only by ChunkCacheService and preserves the source
    // Zarr hierarchy and logical chunk keys while replacing its codec.
    PersistentCacheLayout persistentCacheLayout = PersistentCacheLayout::Auto;
    std::vector<PersistentCacheMetadataObject> zarrMirrorMetadata;
    // Deprecated compatibility fields. Readers accept legacy compressed
    // entries, but production writes never create recompressed cache data.
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
        // Initial capacity used only when decodedByteBudget is not supplied.
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget;
        FetchConcurrency fetchConcurrency;
        std::optional<AdaptiveDownloadState> initialAdaptiveDownloadState;
        // Startup-only selection for remote-volume persistent caches. Sources
        // acquired from this service cannot change representation in place.
        PersistentCacheEncoding persistentCacheEncoding =
            PersistentCacheEncoding::SourceMirror;
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
    // Changes the global decoded RAM ceiling in place. Sources and queued or
    // running work are preserved; reductions evict only decoded LRU entries.
    void configureDecodedByteCapacity(std::size_t decodedByteCapacity);
    FetchConcurrency fetchConcurrency() const;
    PersistentCacheEncoding persistentCacheEncoding() const noexcept;
    std::size_t sourceCount() const;
    bool invalidateSource(std::string_view sourceIdentity);

private:
    friend class ChunkCache;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Process-lifetime service used by every regular Volume cache. The first
// configure call may supply startup-only state such as the persisted adaptive
// download model. Later calls update mutable policy in place and preserve all
// sources and queued/running work.
std::shared_ptr<ChunkCacheService> processChunkCacheService();
std::shared_ptr<ChunkCacheService> configureProcessChunkCacheService(
    ChunkCacheService::Options options);

class ChunkCache final : public IChunkedArray {
public:
    using RemoteFetchActivityCallbackId = std::uint64_t;
    using RemoteFetchActivityCallback =
        std::function<void(const ChunkKey& key, bool active)>;

    using LevelInfo = ChunkCacheLevelInfo;
    using Options = ChunkCacheOptions;

    enum class PersistentRequestMode {
        Ensure,
        Refresh,
    };

    enum class PersistentRequestStatus {
        Data,
        Missing,
        Error,
    };

    struct PersistentRequestResult {
        PersistentRequestStatus status = PersistentRequestStatus::Error;
        std::string error;
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
        // Non-empty when persistence was disabled for this volume, for example
        // because another process is using an incompatible cache format.
        std::string persistentCacheWarning;
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
        PersistentCacheLayout layout = PersistentCacheLayout::Legacy;
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
    PersistentCacheLayout persistentCacheLayout() const noexcept;
    std::vector<ChunkKey> persistedStorageObjectRepresentatives() const;
    std::vector<ChunkKey> storageObjectRepresentatives(int level) const;

    Stats stats() const;
    void invalidate();
    void waitForPersistentWrites() const;
    // Uses the service's shared source scheduler and keyed transfer registry.
    // This request never decodes or populates decoded RAM by itself.
    PersistentRequestResult persistChunkBlocking(
        int level,
        int iz,
        int iy,
        int ix,
        PersistentRequestMode mode = PersistentRequestMode::Ensure);

    // Deprecated no-op compatibility API. Legacy compressed entries remain
    // readable; new writes are never recompressed.
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
        bool sourceData = false;
        bool compressedData = false;
        bool primaryData = false;
        bool empty = false;

        bool hasData() const noexcept
        {
            return sourceData || compressedData || primaryData;
        }
    };

    struct PersistentReadResult {
        std::vector<std::byte> bytes;
        bool sourcePayload = false;
        bool decodedPayload = false;
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

    struct PersistenceOperation {
        mutable std::mutex mutex;
        std::condition_variable cv;
        PersistentRequestResult result;
        bool completed = false;
        bool refresh = false;
        std::atomic_bool writeQueued{false};
        std::uint64_t probeTaskId = 0;
        std::uint64_t sourceTaskId = 0;
    };

    struct SourceTransfer {
        std::uint64_t serial = 0;
        std::uint64_t taskId = 0;
        std::uint64_t generation = 0;
        std::uint64_t fetcherGeneration = 0;
        std::uint64_t schedulerEpoch = 0;
        std::shared_ptr<IChunkFetcher> fetcher;
        bool decodeRequested = false;
        std::weak_ptr<PersistenceOperation> persistence;
    };

    struct StorageObjectKey {
        int level = 0;
        int iz = 0;
        int iy = 0;
        int ix = 0;

        friend bool operator==(const StorageObjectKey&, const StorageObjectKey&) = default;
    };

    struct StorageObjectKeyHash {
        std::size_t operator()(const StorageObjectKey& key) const noexcept
        {
            std::size_t seed = std::hash<int>{}(key.level);
            auto combine = [&seed](int value) {
                seed ^= std::hash<int>{}(value) + 0x9e3779b9 +
                        (seed << 6) + (seed >> 2);
            };
            combine(key.iz);
            combine(key.iy);
            combine(key.ix);
            return seed;
        }
    };

    struct StorageConsumer {
        std::uint64_t generation = 0;
        std::uint64_t fetcherGeneration = 0;
        std::uint64_t fetchSerial = 0;
        std::uint64_t schedulerEpoch = 0;
        std::shared_ptr<IChunkFetcher> fetcher;
    };

    struct StorageObjectTransfer {
        enum class Stage { Probe, PersistentRead, Source };
        std::uint64_t serial = 0;
        std::uint64_t taskId = 0;
        std::uint64_t schedulerEpoch = 0;
        Stage stage = Stage::Probe;
        ChunkStorageObject object;
        std::shared_ptr<IChunkFetcher> fetcher;
        std::unordered_map<ChunkKey, StorageConsumer, ChunkKeyHash> consumers;
        std::unordered_map<ChunkKey,
                           std::weak_ptr<PersistenceOperation>,
                           ChunkKeyHash> persistence;
        std::unordered_set<ChunkKey, ChunkKeyHash> notifiedConsumers;
        bool refreshRequested = false;
        bool sourceStarted = false;
    };

    struct State {
        State(std::vector<LevelInfo> levels,
              std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
              double fillValue,
              ChunkDtype dtype,
              Options options,
              std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget,
              VolumeSourceId sourceId,
              std::string sourceIdentity)
            : levels_(std::move(levels))
            , fetchers_(std::move(fetchers))
            , fillValue_(fillValue)
            , dtype_(dtype)
            , options_(std::move(options))
            , decodedByteBudget_(std::move(decodedByteBudget))
            , sourceId_(sourceId)
            , sourceIdentity_(std::move(sourceIdentity))
            , schedulerGroup_(ChunkCache::nextSchedulerGroup())
        {
            persistentLayout_ = options_.persistentCacheLayout;
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
        PersistentCacheLayout persistentLayout_ = PersistentCacheLayout::Legacy;
        std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget_;
        VolumeSourceId sourceId_{};
        std::string sourceIdentity_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
        std::unordered_map<ChunkKey,
                           std::shared_ptr<PersistenceOperation>,
                           ChunkKeyHash> persistenceOperations_;
        std::unordered_map<ChunkKey, SourceTransfer, ChunkKeyHash> sourceTransfers_;
        std::unordered_map<StorageObjectKey,
                           StorageObjectTransfer,
                           StorageObjectKeyHash> storageObjectTransfers_;
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
        std::uint64_t nextSourceTransferSerial_ = 1;
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
        std::shared_ptr<void> persistentLease_;
        std::string persistentCacheWarning_;

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
    static ChunkWorkPriority storageObjectPriorityLocked(
        const State& state, const StorageObjectTransfer& transfer);
    static void reprioritizeStorageObjectTransferLocked(
        const State& state, const StorageObjectTransfer& transfer);
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
    [[nodiscard]] static bool queueFetchLocked(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        std::uint64_t generation,
        int priorityOffset);
    static void probePersistentAndDispatch(const std::shared_ptr<State>& state,
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
    static void joinSourceTransferLocked(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        FetchContext context,
        bool decodeRequested,
        const std::shared_ptr<PersistenceOperation>& persistence);
    static void runSourceTransfer(const std::shared_ptr<State>& state,
                                  ChunkKey key,
                                  std::uint64_t transferSerial);
    [[nodiscard]] static bool joinStorageObjectTransferLocked(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        FetchContext context,
        bool decodeRequested,
        const std::shared_ptr<PersistenceOperation>& persistence);
    static void runStorageObjectProbe(
        const std::shared_ptr<State>& state,
        StorageObjectKey objectKey,
        std::uint64_t transferSerial);
    static void runStorageObjectRead(
        const std::shared_ptr<State>& state,
        StorageObjectKey objectKey,
        std::uint64_t transferSerial);
    static void queueStorageObjectSourceLocked(
        const std::shared_ptr<State>& state,
        StorageObjectKey objectKey,
        StorageObjectTransfer& transfer);
    static void runStorageObjectFetch(
        const std::shared_ptr<State>& state,
        StorageObjectKey objectKey,
        std::uint64_t transferSerial);
    static void dispatchStorageObjectResult(
        const std::shared_ptr<State>& state,
        StorageObjectKey objectKey,
        std::uint64_t transferSerial,
        ChunkFetchResult fetch,
        bool loadedFromPersistentCache);
    static void queueStorageObjectDecode(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        StorageConsumer consumer,
        std::shared_ptr<const std::vector<std::byte>> objectBytes,
        bool loadedFromPersistentCache);
    static void probePersistenceAndDispatch(
        const std::shared_ptr<State>& state,
        ChunkKey key,
        std::shared_ptr<PersistenceOperation> operation);
    static void completePersistenceOperation(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        const std::shared_ptr<PersistenceOperation>& operation,
        PersistentRequestResult result);
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
    static std::optional<PersistentReadResult> readPersistent(
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
    static bool queueDelta3dWrite(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        std::shared_ptr<const std::vector<std::byte>> bytes,
        std::shared_ptr<PersistenceOperation> operation = {});
    static void queueDelta3dMaintenanceDecode(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        FetchContext context,
        ChunkFetchResult fetched,
        std::shared_ptr<PersistenceOperation> operation);
    static bool queuePersistentEmptyWrite(const std::shared_ptr<State>& state,
                                          const ChunkKey& key);
    static bool queuePersistentSourceWrite(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        std::shared_ptr<const std::vector<std::byte>> bytes,
        std::shared_ptr<PersistenceOperation> operation);
    static bool queuePersistentSourceEmptyWrite(
        const std::shared_ptr<State>& state,
        const ChunkKey& key,
        std::shared_ptr<PersistenceOperation> operation);
    static bool writePersistentSource(
        State& state,
        const ChunkKey& key,
        const std::vector<std::byte>& bytes);
    static bool writePersistent(State& state, const ChunkKey& key, const std::vector<std::byte>& bytes);
    static bool writePersistentEmpty(State& state, const ChunkKey& key);
    static std::filesystem::path persistentPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentCompressedPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentDelta3dPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentEmptyPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentSourcePath(const State& state, const ChunkKey& key);
    static std::filesystem::path mirrorObjectPath(
        const State& state, const ChunkStorageObject& object);
    static std::filesystem::path mirrorEmptyPath(
        const State& state, const ChunkStorageObject& object);
    static bool writeMirrorObject(
        State& state,
        const ChunkStorageObject& object,
        std::span<const std::byte> bytes);
    static bool writeMirrorEmpty(
        State& state, const ChunkStorageObject& object);
    static PersistentCacheLayout resolvePersistentCacheLayout(
        const Options& options,
        const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers);
    static void publishMirrorMetadata(const std::shared_ptr<State>& state);
    static void publishDelta3dMetadata(const std::shared_ptr<State>& state);
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
