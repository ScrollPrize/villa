#include "ChunkCache.hpp"

#include <utils/thread_pool.hpp>

#include "vc/core/util/CacheCompression.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/render/ChunkRequestScheduler.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace vc::render {

namespace {

constexpr std::size_t kPersistentWriteBacklogBytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::size_t kPersistentProbeWorkers = 32;
constexpr std::size_t kDecodeWorkers = 8;
constexpr int kTerminalLevelPriorityBonus = 100;
std::atomic_size_t g_persistentWriteBacklogBytes{0};

bool reservePersistentWriteBytes(std::size_t bytes)
{
    std::size_t current = g_persistentWriteBacklogBytes.load(std::memory_order_relaxed);
    for (;;) {
        // Admit one oversized chunk only into an otherwise-empty queue. This
        // keeps the bound useful for ordinary chunks without making a chunk
        // larger than the bound permanently uncacheable.
        if (current != 0 &&
            (bytes > kPersistentWriteBacklogBytes ||
             current > kPersistentWriteBacklogBytes - bytes)) {
            return false;
        }
        if (g_persistentWriteBacklogBytes.compare_exchange_weak(
                current, current + bytes, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
    }
}

void releasePersistentWriteBytes(std::size_t bytes)
{
    g_persistentWriteBacklogBytes.fetch_sub(bytes, std::memory_order_acq_rel);
}

std::string uniqueTmpSuffix()
{
    // Several caches (viewers, core blocking readers, prefill — possibly in
    // different processes) share one persistent cache directory. A fixed
    // ".tmp" name lets concurrent writers of the same chunk interleave into
    // one file and rename a corrupt result into place.
    static const auto processTag = static_cast<std::uint64_t>(std::random_device{}());
    static std::atomic<std::uint64_t> counter{0};
    return ".tmp." + std::to_string(processTag) + "." +
           std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

std::size_t normalizedWorkerCount(std::size_t requested)
{
    return std::max<std::size_t>(1, requested);
}

utils::ThreadPool& persistentCacheWriterPool()
{
    // Keep disk-cache writes off the chunk read/fetch pool. A single writer
    // avoids same-path tmp/rename races while preventing writeback from
    // occupying workers needed by the current view.
#if defined(_WIN32)
    static auto* pool = new utils::ThreadPool(1);
    return *pool;
#else
    static utils::ThreadPool pool(1);
    return pool;
#endif
}

std::string fetchErrorMessage(const ChunkFetchResult& fetch)
{
    if (!fetch.message.empty())
        return fetch.message;
    switch (fetch.status) {
    case ChunkFetchStatus::HttpError:
        return fetch.httpStatus > 0 ? "HTTP error " + std::to_string(fetch.httpStatus) : "HTTP error";
    case ChunkFetchStatus::IoError:
        return "I/O error";
    case ChunkFetchStatus::DecodeError:
        return "decode error";
    case ChunkFetchStatus::Found:
    case ChunkFetchStatus::Missing:
        return {};
    }
    return "chunk fetch error";
}

} // namespace

struct ChunkCacheService::Impl {
    Impl(std::size_t capacity,
         std::shared_ptr<DecodedChunkCacheBudget> budget,
         bool createBudget,
         std::optional<AdaptiveDownloadState> initialAdaptiveState)
        : decodedByteCapacity(capacity)
        , decodedByteBudget(std::move(budget))
        , initialAdaptiveDownloadState(std::move(initialAdaptiveState))
    {
        if (!decodedByteBudget && createBudget)
            decodedByteBudget = std::make_shared<DecodedChunkCacheBudget>(capacity);
    }

    std::size_t decodedByteCapacity = 0;
    std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget;
    std::optional<AdaptiveDownloadState> initialAdaptiveDownloadState;
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<void>> sources;
    std::uint64_t nextSourceId = 1;
    std::shared_ptr<std::atomic<std::uint64_t>> activeViewId =
        std::make_shared<std::atomic<std::uint64_t>>(0);
    std::shared_ptr<std::atomic<std::uint64_t>> nextTaskId =
        std::make_shared<std::atomic<std::uint64_t>>(1);
    std::shared_ptr<ChunkRequestSelectionGate> schedulerSelectionGate =
        std::make_shared<ChunkRequestSelectionGate>();
    std::shared_ptr<ChunkRequestScheduler> probeScheduler =
        std::make_shared<ChunkRequestScheduler>(
            kPersistentProbeWorkers, 7, schedulerSelectionGate);
    std::shared_ptr<ChunkRequestScheduler> decodeScheduler =
        std::make_shared<ChunkRequestScheduler>(
            kDecodeWorkers, 7, schedulerSelectionGate);
    using FetchConfiguration = std::pair<std::size_t, bool>;
    std::map<FetchConfiguration, std::shared_ptr<ChunkRequestScheduler>> fetchSchedulers;
    std::shared_ptr<ChunkRequestScheduler> activeFetchScheduler;
    std::shared_ptr<ChunkRequestScheduler> lastAdaptiveFetchScheduler;
    std::size_t activeFetchWorkers = 0;
    bool activeFetchAdaptive = false;
    std::uint64_t fetchConfigurationGeneration = 0;

    std::shared_ptr<ChunkRequestScheduler> schedulerFor(
        std::size_t workers, bool adaptive)
    {
        workers = normalizedWorkerCount(workers);
        auto& scheduler = fetchSchedulers[{workers, adaptive}];
        if (!scheduler) {
            std::optional<ChunkRequestScheduler::AdaptiveConcurrency> adaptiveOptions;
            std::optional<ChunkRequestScheduler::AdaptiveState> initialState;
            if (adaptive) {
                adaptiveOptions.emplace();
                adaptiveOptions->maximum = workers;
                if (lastAdaptiveFetchScheduler) {
                    initialState = lastAdaptiveFetchScheduler->adaptiveState();
                } else if (initialAdaptiveDownloadState) {
                    initialState = ChunkRequestScheduler::AdaptiveState{
                        initialAdaptiveDownloadState->settledAdmissionLimit,
                        initialAdaptiveDownloadState->longTermBytesPerSecond,
                        initialAdaptiveDownloadState->maximumSaturatedParallelism,
                        initialAdaptiveDownloadState
                            ->saturatedBytesPerSecondPerWorker};
                }
            }
            scheduler = std::make_shared<ChunkRequestScheduler>(
                workers, 7, schedulerSelectionGate, adaptiveOptions, initialState);
        }
        if (adaptive)
            lastAdaptiveFetchScheduler = scheduler;
        return scheduler;
    }
};

ChunkCacheService::ChunkCacheService(
    std::size_t decodedByteCapacity,
    std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget,
    std::optional<AdaptiveDownloadState> initialAdaptiveDownloadState)
    : impl_(std::make_shared<Impl>(decodedByteCapacity,
                                  std::move(decodedByteBudget), true,
                                  std::move(initialAdaptiveDownloadState)))
{
}

ChunkCacheService::ChunkCacheService(
    std::size_t decodedByteCapacity,
    std::shared_ptr<DecodedChunkCacheBudget> decodedByteBudget,
    bool createDecodedByteBudget,
    std::optional<AdaptiveDownloadState> initialAdaptiveDownloadState)
    : impl_(std::make_shared<Impl>(decodedByteCapacity,
                                  std::move(decodedByteBudget),
                                  createDecodedByteBudget,
                                  std::move(initialAdaptiveDownloadState)))
{
}

ChunkCacheService::~ChunkCacheService()
{
    std::vector<std::shared_ptr<ChunkCache::State>> states;
    {
        std::lock_guard lock(impl_->mutex);
        states.reserve(impl_->sources.size());
        for (auto& [identity, state] : impl_->sources) {
            (void)identity;
            states.push_back(std::static_pointer_cast<ChunkCache::State>(state));
        }
        impl_->sources.clear();
    }
    for (const auto& state : states) {
        ChunkCache::invalidateState(state);
        ChunkCache::unregisterStateBudget(*state);
    }

    // Invalidation cancels pending work, but a running stage may still be
    // returning from its callback. Drain every stage while Impl still owns the
    // schedulers so their worker threads are joined by this thread.
    impl_->probeScheduler->waitIdle();
    for (const auto& [configuration, scheduler] : impl_->fetchSchedulers) {
        (void)configuration;
        scheduler->waitIdle();
    }
    impl_->decodeScheduler->waitIdle();
}

std::shared_ptr<DecodedChunkCacheBudget>
ChunkCacheService::decodedByteBudget() const
{
    return impl_->decodedByteBudget;
}

std::optional<ChunkCacheService::AdaptiveDownloadState>
ChunkCacheService::adaptiveDownloadState() const
{
    std::shared_ptr<ChunkRequestScheduler> scheduler;
    std::optional<AdaptiveDownloadState> initialState;
    {
        std::lock_guard lock(impl_->mutex);
        scheduler = impl_->lastAdaptiveFetchScheduler;
        initialState = impl_->initialAdaptiveDownloadState;
    }
    if (!scheduler)
        return initialState;
    const auto state = scheduler->adaptiveState();
    if (!state)
        return std::nullopt;
    return AdaptiveDownloadState{
        state->settledAdmissionLimit,
        state->longTermBytesPerSecond,
        state->maximumSaturatedParallelism,
        state->saturatedBytesPerSecondPerWorker};
}

std::shared_ptr<ChunkCache> ChunkCacheService::openSource(
    std::string sourceIdentity,
    std::vector<ChunkCacheLevelInfo> levels,
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
    double fillValue,
    ChunkDtype dtype,
    ChunkCacheOptions options)
{
    if (sourceIdentity.empty()) {
        throw std::invalid_argument(
            "ChunkCache requires a non-empty source identity");
    }
    options.decodedByteCapacity = impl_->decodedByteCapacity;
    options.decodedByteBudget = impl_->decodedByteBudget;
    options.compressPersistentCache =
        options.compressPersistentCache || ChunkCache::persistentCompressionDefault();
    options.cacheQuantBinWidth = std::max(
        options.cacheQuantBinWidth,
        ChunkCache::persistentQuantizationDefault());
    ChunkCache::validateSourceDefinition(levels, fetchers);

    // Invalid attempts to reopen a source must not change the service-wide
    // scheduler. Perform the immutable and fetcher checks before applying the
    // caller's last-writer-wins concurrency configuration.
    {
        std::lock_guard serviceLock(impl_->mutex);
        const auto existing = impl_->sources.find(sourceIdentity);
        if (existing != impl_->sources.end()) {
            const auto state =
                std::static_pointer_cast<ChunkCache::State>(existing->second);
            if (!ChunkCache::metadataCompatible(
                    *state, levels, fillValue, dtype, options)) {
                throw std::invalid_argument(
                    "ChunkCache source was registered with incompatible metadata: " +
                    sourceIdentity);
            }
            ChunkCache::validateRefreshedFetchers(*state, fetchers);
        }
    }

    auto service = shared_from_this();
    ChunkCache::configureServiceFetchScheduler(
        service, options.maxConcurrentReads, options.adaptiveConcurrentReads);

    std::unique_lock serviceLock(impl_->mutex);
    auto existing = impl_->sources.find(sourceIdentity);
    if (existing != impl_->sources.end()) {
        auto state =
            std::static_pointer_cast<ChunkCache::State>(existing->second);
        if (!ChunkCache::metadataCompatible(
                *state, levels, fillValue, dtype, options)) {
            throw std::invalid_argument(
                "ChunkCache source was registered with incompatible metadata: " +
                sourceIdentity);
        }
        ChunkCache::validateRefreshedFetchers(*state, fetchers);
        serviceLock.unlock();
        ChunkCache::refreshFetchers(state, std::move(fetchers));
        return std::shared_ptr<ChunkCache>(
            new ChunkCache(std::move(service), std::move(state)));
    }

    const VolumeSourceId sourceId{impl_->nextSourceId++};
    auto state = std::make_shared<ChunkCache::State>(
        std::move(levels), std::move(fetchers), fillValue, dtype,
        std::move(options), sourceId, sourceIdentity);
    state->probeScheduler_ = impl_->probeScheduler;
    state->fetchScheduler_ = impl_->activeFetchScheduler;
    state->fetchConfigurationGeneration_ = impl_->fetchConfigurationGeneration;
    state->decodeScheduler_ = impl_->decodeScheduler;
    state->schedulerSelectionGate_ = impl_->schedulerSelectionGate;
    state->activeViewId_ = impl_->activeViewId;
    state->nextTaskId_ = impl_->nextTaskId;
    if (!state->options_.decodedByteBudget) {
        state->options_.decodedByteBudget =
            ChunkCache::decodedByteBudgetDefault();
    }
    if (state->options_.persistentCacheBudgetRoot &&
        state->options_.persistentCachePath) {
        state->persistentBudget_ = PersistentZarrCacheBudget::findForPath(
            *state->options_.persistentCachePath);
    }
    if (state->options_.persistentCachePath && !state->persistentBudget_)
        ChunkCache::startPersistentCacheSizeScan(state);
    ChunkCache::registerStateBudget(state);
    impl_->sources.emplace(state->sourceIdentity_, state);
    return std::shared_ptr<ChunkCache>(
        new ChunkCache(std::move(service), std::move(state)));
}

void ChunkCacheService::configureFetchConcurrency(
    std::size_t maxConcurrentReads, bool adaptive)
{
    ChunkCache::configureServiceFetchScheduler(
        shared_from_this(), maxConcurrentReads, adaptive);
}

ChunkCacheService::FetchConcurrency ChunkCacheService::fetchConcurrency() const
{
    std::lock_guard lock(impl_->mutex);
    return {impl_->activeFetchWorkers, impl_->activeFetchAdaptive};
}

std::size_t ChunkCacheService::sourceCount() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->sources.size();
}

bool ChunkCacheService::invalidateSource(std::string_view sourceIdentity)
{
    std::shared_ptr<ChunkCache::State> state;
    {
        std::lock_guard lock(impl_->mutex);
        const auto it = impl_->sources.find(std::string(sourceIdentity));
        if (it == impl_->sources.end())
            return false;
        state = std::static_pointer_cast<ChunkCache::State>(it->second);
    }
    ChunkCache::invalidateState(state);
    return true;
}

std::uint64_t ChunkCache::nextSchedulerGroup()
{
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

ChunkCache::ChunkCache(std::vector<LevelInfo> levels,
                       std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
                       double fillValue,
                       ChunkDtype dtype)
    : ChunkCache(std::move(levels), std::move(fetchers), fillValue, dtype, Options{})
{
}

ChunkCache::ChunkCache(std::vector<LevelInfo> levels,
                       std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
                       double fillValue,
                       ChunkDtype dtype,
                       Options options)
    : service_(std::shared_ptr<ChunkCacheService>(new ChunkCacheService(
          options.decodedByteCapacity, options.decodedByteBudget, false,
          std::nullopt)))
{
    auto handle = service_->openSource(
        "private:" + std::to_string(nextSchedulerGroup()), std::move(levels),
        std::move(fetchers), fillValue, dtype, std::move(options));
    state_ = std::move(handle->state_);
}

ChunkCache::ChunkCache(std::shared_ptr<ChunkCacheService> service,
                       std::shared_ptr<State> state)
    : service_(std::move(service))
    , state_(std::move(state))
{
    if (!service_)
        throw std::invalid_argument("ChunkCache requires a cache service");
    if (!state_)
        throw std::invalid_argument("ChunkCache requires registered source state");
}

void ChunkCache::validateSourceDefinition(
    const std::vector<LevelInfo>& levels,
    const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers)
{
    if (levels.empty())
        throw std::invalid_argument("ChunkCache requires at least one level");
    if (levels.size() != fetchers.size())
        throw std::invalid_argument("ChunkCache level/fetcher count mismatch");
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const bool missingLevel =
            levels[level].shape[0] == 0 &&
            levels[level].shape[1] == 0 &&
            levels[level].shape[2] == 0;
        if (!fetchers[level] && !missingLevel) {
            throw std::invalid_argument(
                "ChunkCache fetcher must not be null for present level");
        }
        for (int dim : levels[level].shape) {
            if (dim < 0) {
                throw std::invalid_argument(
                    "ChunkCache level shape must be non-negative");
            }
        }
        for (int dim : levels[level].chunkShape) {
            if (dim <= 0) {
                throw std::invalid_argument(
                    "ChunkCache chunk shape must be positive");
            }
        }
    }
}

void ChunkCache::registerStateBudget(const std::shared_ptr<State>& state)
{
    if (!state || !state->options_.decodedByteBudget)
        return;
    std::weak_ptr<State> weakState = state;
    state->decodedBudgetRegistration_ =
        state->options_.decodedByteBudget->registerCache({
            [weakState]() -> std::optional<std::uint64_t> {
                auto locked = weakState.lock();
                return locked ? oldestDecodedTouch(locked) : std::nullopt;
            },
            [weakState]() -> std::size_t {
                auto locked = weakState.lock();
                return locked ? evictOldestDecoded(locked) : 0;
            },
        });
}

void ChunkCache::validateRefreshedFetchers(
    const State& state,
    const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers)
{
    if (fetchers.size() != state.levels_.size()) {
        throw std::invalid_argument(
            "ChunkCache fetcher refresh level count mismatch");
    }
    for (std::size_t level = 0; level < fetchers.size(); ++level) {
        const bool missingLevel =
            state.levels_[level].shape[0] == 0 &&
            state.levels_[level].shape[1] == 0 &&
            state.levels_[level].shape[2] == 0;
        if (!fetchers[level] && !missingLevel) {
            throw std::invalid_argument(
                "ChunkCache refreshed fetcher is null for present level");
        }
        if (fetchers[level] &&
            fetchers[level]->persistentCacheExtension(
                ChunkKey{static_cast<int>(level), 0, 0, 0}) !=
                state.persistentExtensions_[level]) {
            throw std::invalid_argument(
                "ChunkCache refreshed fetcher changed persistent encoding");
        }
    }
}

void ChunkCache::configureServiceFetchScheduler(
    const std::shared_ptr<ChunkCacheService>& service,
    std::size_t maxConcurrentReads,
    bool adaptive)
{
    maxConcurrentReads = normalizedWorkerCount(maxConcurrentReads);
    std::shared_ptr<ChunkRequestScheduler> scheduler;
    std::vector<std::shared_ptr<State>> states;
    std::uint64_t configurationGeneration = 0;
    {
        std::lock_guard serviceLock(service->impl_->mutex);
        if (service->impl_->activeFetchScheduler &&
            service->impl_->activeFetchWorkers == maxConcurrentReads &&
            service->impl_->activeFetchAdaptive == adaptive) {
            return;
        }
        scheduler = service->impl_->schedulerFor(maxConcurrentReads, adaptive);
        service->impl_->activeFetchScheduler = scheduler;
        service->impl_->activeFetchWorkers = maxConcurrentReads;
        service->impl_->activeFetchAdaptive = adaptive;
        configurationGeneration =
            ++service->impl_->fetchConfigurationGeneration;
        states.reserve(service->impl_->sources.size());
        for (const auto& [identity, source] : service->impl_->sources) {
            (void)identity;
            states.push_back(std::static_pointer_cast<State>(source));
        }
    }

    service->impl_->schedulerSelectionGate->publish([&] {
        {
            std::lock_guard serviceLock(service->impl_->mutex);
            if (service->impl_->fetchConfigurationGeneration !=
                configurationGeneration) {
                return;
            }
        }
        for (const auto& state : states)
            migrateFetchScheduler(
                state, scheduler, maxConcurrentReads, adaptive,
                configurationGeneration);
    });
}

void ChunkCache::migrateFetchScheduler(
    const std::shared_ptr<State>& state,
    const std::shared_ptr<ChunkRequestScheduler>& scheduler,
    std::size_t maxConcurrentReads,
    bool adaptive,
    std::uint64_t configurationGeneration)
{
    {
        std::lock_guard lock(state->mutex_);
        if (state->fetchConfigurationGeneration_ >= configurationGeneration)
            return;
        const auto oldScheduler = state->fetchScheduler_.lock();
        const std::uint64_t schedulerEpoch = ++state->schedulerEpoch_;
        if (oldScheduler)
            oldScheduler->cancelGroupBefore(
                state->schedulerGroup_, schedulerEpoch);
        if (auto probeScheduler = state->probeScheduler_.lock())
            probeScheduler->cancelGroupBefore(
                state->schedulerGroup_, schedulerEpoch);
        if (auto decodeScheduler = state->decodeScheduler_.lock())
            decodeScheduler->cancelGroupBefore(
                state->schedulerGroup_, schedulerEpoch);
        state->fetchScheduler_ = scheduler;
        state->options_.maxConcurrentReads = maxConcurrentReads;
        state->options_.adaptiveConcurrentReads = adaptive;
        state->fetchConfigurationGeneration_ = configurationGeneration;
        restartUnresolvedLocked(state);
    }
    state->cv_.notify_all();
}

void ChunkCache::refreshFetchers(
    const std::shared_ptr<State>& state,
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers)
{
    if (!state)
        throw std::invalid_argument("ChunkCache fetcher refresh requires state");
    validateRefreshedFetchers(*state, fetchers);

    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        if (std::equal(fetchers.begin(), fetchers.end(),
                       state->fetchers_.begin(), state->fetchers_.end())) {
            return;
        }
        state->fetchers_ = std::move(fetchers);
        ++state->fetcherGeneration_;
        const std::uint64_t schedulerEpoch = ++state->schedulerEpoch_;
        if (auto scheduler = state->fetchScheduler_.lock())
            scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
        if (auto scheduler = state->probeScheduler_.lock())
            scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
        if (auto scheduler = state->decodeScheduler_.lock())
            scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
        restartUnresolvedLocked(state);
    });
    state->cv_.notify_all();
}

void ChunkCache::restartUnresolvedLocked(const std::shared_ptr<State>& state)
{
    std::vector<std::pair<std::uint64_t, ChunkKey>> retry;
    retry.reserve(state->entries_.size());
    for (auto it = state->entries_.begin(); it != state->entries_.end();) {
        Entry& entry = it->second;
        if (entry.status == EntryStatus::InFlight && !hasDemandLocked(entry)) {
            if (entry.unresolvedCounted && it->first.level >= 0 &&
                it->first.level < static_cast<int>(
                    state->unresolvedFetchesByLevel_.size())) {
                auto& unresolved = state->unresolvedFetchesByLevel_[
                    static_cast<std::size_t>(it->first.level)];
                if (unresolved > 0)
                    --unresolved;
            }
            it = state->entries_.erase(it);
            continue;
        }
        if ((entry.status != EntryStatus::InFlight &&
             entry.status != EntryStatus::Error) ||
            !hasDemandLocked(entry)) {
            ++it;
            continue;
        }
        if (entry.inLru) {
            state->lru_.erase(entry.lruIt);
            entry.inLru = false;
        }
        entry.error.clear();
        entry.probeTaskId = 0;
        entry.fetchTaskId = 0;
        entry.decodeTaskId = 0;
        retry.emplace_back(entry.fetchSerial, it->first);
        ++it;
    }
    std::sort(retry.begin(), retry.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    for (const auto& [serial, key] : retry) {
        (void)serial;
        queueFetchLocked(state, key, state->generation_, 0);
    }
}

namespace {
std::atomic_bool g_persistentCompressionDefault{false};
std::atomic_int g_persistentQuantizationDefault{1};
std::mutex g_decodedBudgetDefaultMutex;
std::weak_ptr<DecodedChunkCacheBudget> g_decodedBudgetDefault;
}

void ChunkCache::setPersistentCompressionDefault(bool enabled)
{
    g_persistentCompressionDefault.store(enabled, std::memory_order_relaxed);
}

bool ChunkCache::persistentCompressionDefault()
{
    return g_persistentCompressionDefault.load(std::memory_order_relaxed);
}

void ChunkCache::setPersistentQuantizationDefault(int binWidth)
{
    g_persistentQuantizationDefault.store(std::clamp(binWidth, 1, 255),
                                          std::memory_order_relaxed);
}

int ChunkCache::persistentQuantizationDefault()
{
    return g_persistentQuantizationDefault.load(std::memory_order_relaxed);
}

void ChunkCache::setDecodedByteBudgetDefault(
    const std::shared_ptr<DecodedChunkCacheBudget>& budget)
{
    std::lock_guard lock(g_decodedBudgetDefaultMutex);
    g_decodedBudgetDefault = budget;
}

std::shared_ptr<DecodedChunkCacheBudget> ChunkCache::decodedByteBudgetDefault()
{
    std::lock_guard lock(g_decodedBudgetDefaultMutex);
    return g_decodedBudgetDefault.lock();
}

ChunkCache::~ChunkCache()
{
    std::vector<ChunkReadyCallbackId> listenerIds;
    std::vector<RemoteFetchActivityCallbackId> remoteFetchListenerIds;
    {
        std::lock_guard handleLock(handleMutex_);
        listenerIds.assign(listenerIds_.begin(), listenerIds_.end());
        listenerIds_.clear();
        remoteFetchListenerIds.assign(remoteFetchListenerIds_.begin(),
                                      remoteFetchListenerIds_.end());
        remoteFetchListenerIds_.clear();
    }
    if (!listenerIds.empty() || !remoteFetchListenerIds.empty()) {
        std::lock_guard stateLock(state_->mutex_);
        for (const auto id : listenerIds)
            state_->callbacks_.erase(id);
        for (const auto id : remoteFetchListenerIds)
            state_->remoteFetchCallbacks_.erase(id);
    }
}

VolumeSourceId ChunkCache::sourceId() const noexcept
{
    return state_->sourceId_;
}

ChunkKey ChunkCache::sourceKey(const State& state, ChunkKey key) noexcept
{
    key.sourceId = state.sourceId_;
    return key;
}

ChunkKey ChunkCache::fetcherKey(ChunkKey key) noexcept
{
    key.sourceId = {};
    return key;
}

bool ChunkCache::metadataCompatible(const State& state,
                                    const std::vector<LevelInfo>& levels,
                                    double fillValue,
                                    ChunkDtype dtype,
                                    const Options& options)
{
    if (state.levels_.size() != levels.size() ||
        state.dtype_ != dtype || state.fillValue_ != fillValue ||
        state.options_.persistentCachePath != options.persistentCachePath ||
        state.options_.persistentCacheBudgetRoot != options.persistentCacheBudgetRoot ||
        state.options_.compressPersistentCache != options.compressPersistentCache ||
        state.options_.cacheQuantBinWidth != options.cacheQuantBinWidth ||
        state.options_.detectAllFillChunks != options.detectAllFillChunks) {
        return false;
    }
    for (std::size_t i = 0; i < levels.size(); ++i) {
        if (state.levels_[i].shape != levels[i].shape ||
            state.levels_[i].chunkShape != levels[i].chunkShape ||
            state.levels_[i].transform.scaleFromLevel0 != levels[i].transform.scaleFromLevel0 ||
            state.levels_[i].transform.offsetFromLevel0 != levels[i].transform.offsetFromLevel0) {
            return false;
        }
    }
    return true;
}

int ChunkCache::numLevels() const
{
    return static_cast<int>(state_->levels_.size());
}

std::array<int, 3> ChunkCache::shape(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).shape;
}

std::array<int, 3> ChunkCache::chunkShape(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).chunkShape;
}

ChunkDtype ChunkCache::dtype() const
{
    return state_->dtype_;
}

double ChunkCache::fillValue() const
{
    return state_->fillValue_;
}

IChunkedArray::LevelTransform ChunkCache::levelTransform(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).transform;
}

ChunkResult ChunkCache::tryGetChunk(int level, int iz, int iy, int ix)
{
    return tryGetChunk(level, iz, iy, ix, {});
}

ChunkResult ChunkCache::tryGetChunk(int level, int iz, int iy, int ix,
                                    const ChunkRequestContext& request)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix, state->sourceId_};
    std::unique_lock lock(state->mutex_);
    if (level >= 0 && level < static_cast<int>(state->fetchers_.size()) &&
        !state->fetchers_[static_cast<std::size_t>(level)]) {
        return ChunkResult{
            ChunkStatus::Missing,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    if (!isValidKey(*state, key))
        return ChunkResult{ChunkStatus::AllFill, state->dtype_, {}, {}, {}};

    auto it = state->entries_.find(key);
    if (it != state->entries_.end()) {
        if (it->second.status == EntryStatus::InFlight) {
            if (addRequestDemandLocked(*state, key, it->second, request))
                reprioritizeEntryLocked(*state, key, it->second);
            return ChunkResult{
                ChunkStatus::MissQueued, state->dtype_,
                state->levels_[level].chunkShape, {}, {}};
        }
        return resultFromEntryLocked(*state, key, it->second);
    }

    auto [insertedIt, inserted] = state->entries_.emplace(key, Entry{});
    (void)inserted;
    if (!addRequestDemandLocked(*state, key, insertedIt->second, request)) {
        state->entries_.erase(insertedIt);
        return ChunkResult{
            ChunkStatus::MissQueued, state->dtype_,
            state->levels_[level].chunkShape, {}, {}};
    }
    queueFetchLocked(state, key, state->generation_, 0);
    return ChunkResult{
        ChunkStatus::MissQueued, state->dtype_,
        state->levels_[level].chunkShape, {}, {}};
}

ChunkResult ChunkCache::getChunkIfCached(int level, int iz, int iy, int ix)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix, state->sourceId_};
    std::lock_guard lock(state->mutex_);
    if (level >= 0 && level < static_cast<int>(state->fetchers_.size()) &&
        !state->fetchers_[static_cast<std::size_t>(level)]) {
        return ChunkResult{
            ChunkStatus::Missing,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    if (!isValidKey(*state, key))
        return ChunkResult{ChunkStatus::AllFill, state->dtype_, {}, {}, {}};

    auto it = state->entries_.find(key);
    if (it == state->entries_.end() || it->second.status == EntryStatus::InFlight) {
        return ChunkResult{
            ChunkStatus::MissQueued,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    return resultFromEntryLocked(*state, key, it->second, false);
}

ChunkResult ChunkCache::getChunkBlocking(int level, int iz, int iy, int ix)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix, state->sourceId_};
    std::unique_lock lock(state->mutex_);
    if (level >= 0 && level < static_cast<int>(state->fetchers_.size()) &&
        !state->fetchers_[static_cast<std::size_t>(level)]) {
        return ChunkResult{
            ChunkStatus::Missing,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    if (!isValidKey(*state, key))
        return ChunkResult{ChunkStatus::AllFill, state->dtype_, {}, {}, {}};

    auto [it, inserted] = state->entries_.emplace(key, Entry{});
    it->second.backgroundDemand = true;
    if (inserted)
        queueFetchLocked(state, key, state->generation_, 0);
    waitForResolvedLocked(*state, lock, key);
    it = state->entries_.find(key);
    if (it == state->entries_.end())
        return ChunkResult{
            ChunkStatus::Error, state->dtype_,
            state->levels_[level].chunkShape, {}, "chunk invalidated"};
    return resultFromEntryLocked(*state, key, it->second);
}

void ChunkCache::prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset)
{
    prefetchChunks(keys, wait, priorityOffset, {});
}

void ChunkCache::prefetchChunks(const std::vector<ChunkKey>& keys,
                                bool wait,
                                int priorityOffset,
                                const ChunkRequestContext& request)
{
    auto state = state_;
    std::unique_lock lock(state->mutex_);
    for (auto key : keys) {
        key = sourceKey(*state, key);
        if (!isValidKey(*state, key))
            continue;
        auto [it, inserted] = state->entries_.emplace(key, Entry{});
        if (inserted) {
            if (addRequestDemandLocked(*state, key, it->second, request)) {
                queueFetchLocked(state, key, state->generation_, priorityOffset);
            } else {
                state->entries_.erase(it);
            }
        } else if (it->second.status == EntryStatus::InFlight) {
            if (addRequestDemandLocked(*state, key, it->second, request)) {
                it->second.basePriority = std::min(
                    it->second.basePriority,
                    fetchBasePriority(*state, key, priorityOffset));
                reprioritizeEntryLocked(*state, key, it->second);
            }
        }
    }
    if (!wait)
        return;

    state->cv_.wait(lock, [&] {
        for (auto key : keys) {
            key = sourceKey(*state, key);
            if (!isValidKey(*state, key))
                continue;
            auto it = state->entries_.find(key);
            if (it != state->entries_.end() && it->second.status == EntryStatus::InFlight)
                return false;
        }
        return true;
    });
}

IChunkedArray::ChunkReadyCallbackId ChunkCache::addChunkReadyListener(ChunkReadyCallback cb)
{
    auto state = state_;
    std::scoped_lock lock(handleMutex_, state->mutex_);
    const auto id = state->nextCallbackId_++;
    state->callbacks_.emplace(id, std::move(cb));
    listenerIds_.insert(id);
    return id;
}

void ChunkCache::removeChunkReadyListener(ChunkReadyCallbackId id)
{
    auto state = state_;
    std::scoped_lock lock(handleMutex_, state->mutex_);
    state->callbacks_.erase(id);
    listenerIds_.erase(id);
}

ChunkCache::RemoteFetchActivityCallbackId
ChunkCache::addRemoteFetchActivityListener(RemoteFetchActivityCallback cb)
{
    auto state = state_;
    std::scoped_lock lock(handleMutex_, state->mutex_);
    const auto id = state->nextCallbackId_++;
    state->remoteFetchCallbacks_.emplace(id, std::move(cb));
    remoteFetchListenerIds_.insert(id);
    return id;
}

void ChunkCache::removeRemoteFetchActivityListener(
    RemoteFetchActivityCallbackId id)
{
    auto state = state_;
    std::scoped_lock lock(handleMutex_, state->mutex_);
    state->remoteFetchCallbacks_.erase(id);
    remoteFetchListenerIds_.erase(id);
}

std::vector<ChunkKey> ChunkCache::activeRemoteFetches() const
{
    auto state = state_;
    std::lock_guard lock(state->mutex_);
    std::vector<ChunkKey> keys;
    keys.reserve(state->activeRemoteFetches_.size());
    for (const auto& [key, count] : state->activeRemoteFetches_) {
        (void)count;
        keys.push_back(key);
    }
    return keys;
}

ChunkCache::PersistentChunkDependency ChunkCache::persistentChunkDependency(
    int level,
    int iz,
    int iy,
    int ix) const
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix};
    PersistentChunkDependency result;
    result.key = key;
    std::shared_ptr<IChunkFetcher> fetcher;
    {
        std::lock_guard lock(state->mutex_);
        if (!state->options_.persistentCachePath || !isValidKey(*state, key))
            return result;
        fetcher = state->fetchers_.at(static_cast<std::size_t>(key.level));
    }
    result.valid = static_cast<bool>(fetcher);
    if (!result.valid)
        return result;
    const ChunkKey externalKey = fetcherKey(key);
    result.sourceChunkKey = fetcher->sourceChunkKey(externalKey);
    result.persistentPath = persistentPath(*state, key);
    result.persistentEmptyPath = persistentEmptyPath(*state, key);
    result.persistentExtension = fetcher->persistentCacheExtension(externalKey);
    result.sourcePayloadMatchesPersistentCache = fetcher->sourcePayloadMatchesPersistentCache(externalKey);
    return result;
}

ChunkCache::Stats ChunkCache::stats() const
{
    auto state = state_;
    Stats result;
    {
        std::lock_guard lock(state->mutex_);
        if (state->options_.decodedByteBudget) {
            const auto budget = state->options_.decodedByteBudget->stats();
            result.decodedBytes = budget.decodedBytes;
            result.decodedByteCapacity = budget.maximumBytes;
        } else {
            result.decodedBytes = state->decodedBytes_;
            result.decodedByteCapacity = state->options_.decodedByteCapacity;
        }
        result.remoteFetchesInFlight = state->remoteFetchesInFlight_;
        if (auto scheduler = state->fetchScheduler_.lock()) {
            result.remoteDownloadBytesPerSecond =
                scheduler->transferStats().bytesPerSecond;
        }
        if (auto scheduler = state->decodeScheduler_.lock())
            result.pendingDecodeTasks = scheduler->pending();
        result.unresolvedFetchesByLevel = state->unresolvedFetchesByLevel_;
        result.persistentCacheEnabled = state->options_.persistentCachePath.has_value();
    }
    if (state->persistentBudget_) {
        const auto budget = state->persistentBudget_->stats();
        result.persistentCacheBytes = static_cast<std::size_t>(budget.managedBytes);
        result.persistentCacheScanInFlight = budget.scanInFlight;
        result.persistentCacheTrimInFlight = budget.trimInFlight;
        result.persistentCacheLowSpace = budget.lowSpace;
        result.persistentCacheFreeBytes = static_cast<std::size_t>(budget.freeBytes);
        result.persistentCacheMinimumFreeBytes =
            static_cast<std::size_t>(budget.minimumFreeBytes);
        if (budget.maximumBytes)
            result.persistentCacheMaximumBytes =
                static_cast<std::size_t>(*budget.maximumBytes);
    } else {
        const auto persistentBytes = state->persistentCacheBytes_.load(std::memory_order_acquire);
        result.persistentCacheBytes = persistentBytes > 0 ? static_cast<std::size_t>(persistentBytes) : 0;
        result.persistentCacheScanInFlight =
            state->persistentCacheScanInFlight_.load(std::memory_order_acquire);
    }
    return result;
}

void ChunkCache::invalidate()
{
    invalidateState(state_);
}

void ChunkCache::replaceViewDemand(const ChunkRequestContext& request,
                                   const std::array<float, 2>& focus,
                                   std::vector<ChunkViewportSample> samples)
{
    if (!request.interactive())
        return;

    struct PublishedDemand {
        int relativeLevel = 0;
        float distanceSquared = std::numeric_limits<float>::infinity();
    };
    std::unordered_map<ChunkKey, PublishedDemand, ChunkKeyHash> published;
    std::vector<ChunkKey> publishedKeys;
    std::unordered_map<int, int> relativeLevels;
    published.reserve(samples.size());
    publishedKeys.reserve(samples.size());
    relativeLevels.reserve(samples.size());
    for (auto& sample : samples) {
        sample.key = sourceKey(*state_, sample.key);
        const float dx = sample.viewportPosition[0] - focus[0];
        const float dy = sample.viewportPosition[1] - focus[1];
        const float distanceSquared = dx * dx + dy * dy;
        auto [found, inserted] = published.emplace(
            sample.key, PublishedDemand{sample.relativeLevel, distanceSquared});
        if (inserted) {
            publishedKeys.push_back(sample.key);
        } else {
            found->second.relativeLevel = std::max(
                found->second.relativeLevel, sample.relativeLevel);
            found->second.distanceSquared = std::min(
                found->second.distanceSquared, distanceSquared);
        }
        auto [relativeLevel, levelInserted] = relativeLevels.emplace(
            sample.key.level, sample.relativeLevel);
        if (!levelInserted)
            relativeLevel->second = std::max(
                relativeLevel->second, sample.relativeLevel);
    }

    auto state = state_;
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        const auto previousSnapshot = state->viewSnapshots_.find(request.viewId);
        if (previousSnapshot != state->viewSnapshots_.end() &&
            (previousSnapshot->second.version > request.viewVersion ||
             (previousSnapshot->second.closed &&
              previousSnapshot->second.version >= request.viewVersion))) {
            return;
        }

        std::vector<ChunkKey> previousKeys;
        if (auto old = state->viewDemandKeys_.find(request.viewId);
            old != state->viewDemandKeys_.end()) {
            previousKeys.reserve(old->second.size());
            for (const ChunkKey& key : old->second) {
                previousKeys.push_back(key);
                const auto entry = state->entries_.find(key);
                if (entry == state->entries_.end())
                    continue;
                entry->second.viewDemands.erase(request.viewId);
            }
            old->second.clear();
        }

        State::ViewSnapshot snapshot;
        snapshot.version = request.viewVersion;
        snapshot.closed = false;
        snapshot.relativeLevels = std::move(relativeLevels);
        state->viewSnapshots_[request.viewId] = std::move(snapshot);

        auto& demanded = state->viewDemandKeys_[request.viewId];
        demanded.reserve(published.size());
        for (const ChunkKey& key : publishedKeys) {
            const auto& demand = published.at(key);
            if (!isValidKey(*state, key))
                continue;
            auto [entryIt, inserted] = state->entries_.emplace(key, Entry{});
            if (!inserted && entryIt->second.status != EntryStatus::InFlight)
                continue;
            demanded.insert(key);
            auto& slot = entryIt->second.viewDemands[request.viewId];
            slot.version = request.viewVersion;
            slot.relativeLevel = demand.relativeLevel;
            slot.distanceSquared = demand.distanceSquared;
            if (inserted)
                queueFetchLocked(state, key, state->generation_, 0);
            else
                reprioritizeEntryLocked(*state, key, entryIt->second);
        }

        for (const ChunkKey& key : previousKeys) {
            auto entry = state->entries_.find(key);
            if (entry == state->entries_.end())
                continue;
            if (cancelUndemandedEntryLocked(*state, key, entry->second))
                eraseUnresolvedEntryLocked(*state, key);
            else
                reprioritizeEntryLocked(*state, key, entry->second);
        }

        // Active-view changes are intentionally not propagated from the mouse
        // path. The next accepted render atomically installs its demand and
        // re-sorts all pending work against the current active view here.
        for (auto& [key, entry] : state->entries_)
            reprioritizeEntryLocked(*state, key, entry);
    });
    state->cv_.notify_all();
}

void ChunkCache::markViewActive(std::uint64_t viewId)
{
    if (viewId == 0)
        return;
    service_->impl_->activeViewId->store(viewId, std::memory_order_release);
}

void ChunkCache::clearSourceViewDemandLocked(State& state,
                                             std::uint64_t viewId,
                                             std::uint64_t viewVersion,
                                             bool closeView)
{
    if (auto demanded = state.viewDemandKeys_.find(viewId);
        demanded != state.viewDemandKeys_.end()) {
        for (const ChunkKey& key : demanded->second) {
            auto entry = state.entries_.find(key);
            if (entry == state.entries_.end())
                continue;
            entry->second.viewDemands.erase(viewId);
            if (cancelUndemandedEntryLocked(state, key, entry->second))
                eraseUnresolvedEntryLocked(state, key);
            else
                reprioritizeEntryLocked(state, key, entry->second);
        }
        state.viewDemandKeys_.erase(demanded);
    }
    auto& snapshot = state.viewSnapshots_[viewId];
    snapshot.version = std::max(snapshot.version, viewVersion);
    snapshot.closed = closeView;
    snapshot.relativeLevels.clear();
}

void ChunkCache::clearSourceViewDemand(std::uint64_t viewId,
                                       std::uint64_t viewVersion)
{
    if (viewId == 0)
        return;
    auto state = state_;
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        // Close this view version only in the handle's bound source so an
        // already-running render cannot re-add the demand after overlay disable.
        // A newer render version reopens the source normally.
        clearSourceViewDemandLocked(*state, viewId, viewVersion, true);
    });
    state->cv_.notify_all();
}

void ChunkCache::clearViewDemand(std::uint64_t viewId,
                                 std::uint64_t viewVersion)
{
    if (viewId == 0)
        return;
    std::vector<std::shared_ptr<State>> states;
    service_->impl_->schedulerSelectionGate->publish([&] {
        std::uint64_t expected = viewId;
        const bool activeChanged = service_->impl_->activeViewId->compare_exchange_strong(
            expected, 0, std::memory_order_acq_rel);

        {
            std::lock_guard serviceLock(service_->impl_->mutex);
            states.reserve(service_->impl_->sources.size());
            for (const auto& [identity, source] : service_->impl_->sources) {
                (void)identity;
                states.push_back(std::static_pointer_cast<State>(source));
            }
        }
        for (const auto& state : states) {
            std::lock_guard lock(state->mutex_);
            clearSourceViewDemandLocked(*state, viewId, viewVersion, true);
            if (activeChanged) {
                for (auto& [key, entry] : state->entries_)
                    reprioritizeEntryLocked(*state, key, entry);
            }
        }
    });
    for (const auto& state : states)
        state->cv_.notify_all();
}

void ChunkCache::invalidateState(const std::shared_ptr<State>& state)
{
    std::uint64_t schedulerEpoch = 0;
    std::vector<ChunkKey> activeRemoteFetches;
    {
        std::lock_guard lock(state->mutex_);
        ++state->generation_;
        schedulerEpoch = ++state->schedulerEpoch_;
        state->entries_.clear();
        state->lru_.clear();
        state->viewSnapshots_.clear();
        state->viewDemandKeys_.clear();
        removeDecodedBytesLocked(*state, state->decodedBytes_);
        state->decodedBytes_ = 0;
        activeRemoteFetches.reserve(state->activeRemoteFetches_.size());
        for (const auto& [key, count] : state->activeRemoteFetches_) {
            (void)count;
            activeRemoteFetches.push_back(key);
        }
        state->activeRemoteFetches_.clear();
        state->remoteFetchesInFlight_ = 0;
        std::fill(state->unresolvedFetchesByLevel_.begin(),
                  state->unresolvedFetchesByLevel_.end(), 0);
    }
    if (auto scheduler = state->fetchScheduler_.lock())
        scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
    if (auto scheduler = state->probeScheduler_.lock())
        scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
    if (auto scheduler = state->decodeScheduler_.lock())
        scheduler->cancelGroupBefore(state->schedulerGroup_, schedulerEpoch);
    for (const auto& key : activeRemoteFetches)
        notifyRemoteFetchListeners(state, key, false);
    state->cv_.notify_all();
}

void ChunkCache::unregisterStateBudget(State& state)
{
    if (state.options_.decodedByteBudget &&
        state.decodedBudgetRegistration_ != 0) {
        state.options_.decodedByteBudget->unregisterCache(
            state.decodedBudgetRegistration_);
        state.decodedBudgetRegistration_ = 0;
    }
}

void ChunkCache::waitForPersistentWrites() const
{
    auto state = state_;
    std::unique_lock lock(state->mutex_);
    state->cv_.wait(lock, [&] {
        return state->persistentWritesInFlight_.load(std::memory_order_acquire) == 0;
    });
}

ChunkResult ChunkCache::resultFromEntryLocked(
    State& state, const ChunkKey& key, Entry& entry, bool promote)
{
    ChunkResult result;
    result.dtype = state.dtype_;
    result.shape = state.levels_[static_cast<std::size_t>(key.level)].chunkShape;

    switch (entry.status) {
    case EntryStatus::InFlight:
        result.status = ChunkStatus::MissQueued;
        break;
    case EntryStatus::Missing:
        result.status = ChunkStatus::Missing;
        if (promote)
            touchLocked(state, key, entry);
        break;
    case EntryStatus::AllFill:
        result.status = ChunkStatus::AllFill;
        if (promote)
            touchLocked(state, key, entry);
        break;
    case EntryStatus::Data:
        result.status = ChunkStatus::Data;
        result.bytes = entry.bytes;
        if (promote)
            touchLocked(state, key, entry);
        break;
    case EntryStatus::Error:
        result.status = ChunkStatus::Error;
        result.error = entry.error;
        if (promote)
            touchLocked(state, key, entry);
        break;
    }
    return result;
}

int ChunkCache::fetchBasePriority(const State& state, const ChunkKey& key, int priorityOffset)
{
    // Background requests retain absolute coarse-to-fine ordering.
    const int numLevels = static_cast<int>(state.levels_.size());
    return (numLevels - 1 - key.level) + priorityOffset;
}

ChunkWorkPriority ChunkCache::workPriorityLocked(const State& state,
                                                 const ChunkKey& key,
                                                 const Entry& entry)
{
    ChunkWorkPriority priority;
    priority.levelPriority = fetchBasePriority(state, key, 0);
    priority.backgroundPriority = entry.basePriority;
    if (entry.viewDemands.empty())
        return priority;

    priority.interactive = true;
    priority.levelPriority = std::numeric_limits<int>::min();
    priority.distanceSquared = std::numeric_limits<float>::infinity();
    const std::uint64_t active = state.activeViewId_
        ? state.activeViewId_->load(std::memory_order_acquire)
        : 0;
    for (const auto& [viewId, demand] : entry.viewDemands) {
        const bool demandIsActive = active != 0 && viewId == active;
        const float distance = demand.distanceSquared.value_or(
            std::numeric_limits<float>::infinity());
        // The source's terminal level is the best possible whole-view fallback,
        // even when a view starts near the end of a shallow pyramid and its
        // ordinary relative offset is small.
        const int levelPriority = demand.relativeLevel +
            (key.level + 1 == static_cast<int>(state.levels_.size())
                 ? kTerminalLevelPriorityBonus
                 : 0);
        if (levelPriority > priority.levelPriority ||
            (levelPriority == priority.levelPriority &&
             demandIsActive && !priority.activeView) ||
            (levelPriority == priority.levelPriority &&
             demandIsActive == priority.activeView &&
             distance < priority.distanceSquared)) {
            priority.levelPriority = levelPriority;
            priority.activeView = demandIsActive;
            priority.distanceSquared = distance;
        }
    }
    return priority;
}

void ChunkCache::reprioritizeEntryLocked(const State& state,
                                         const ChunkKey& key,
                                         Entry& entry)
{
    if (entry.status != EntryStatus::InFlight)
        return;
    const auto priority = workPriorityLocked(state, key, entry);
    if (entry.probeTaskId != 0) {
        if (auto scheduler = state.probeScheduler_.lock())
            scheduler->reprioritize(entry.probeTaskId, priority);
    }
    if (entry.fetchTaskId != 0) {
        if (auto scheduler = state.fetchScheduler_.lock())
            scheduler->reprioritize(entry.fetchTaskId, priority);
    }
    if (entry.decodeTaskId != 0) {
        if (auto scheduler = state.decodeScheduler_.lock())
            scheduler->reprioritize(entry.decodeTaskId, priority);
    }
}

bool ChunkCache::addRequestDemandLocked(State& state,
                                        const ChunkKey& key,
                                        Entry& entry,
                                        const ChunkRequestContext& request)
{
    if (!request.interactive()) {
        entry.backgroundDemand = true;
        return true;
    }
    const auto snapshot = state.viewSnapshots_.find(request.viewId);
    if (snapshot != state.viewSnapshots_.end()) {
        if (snapshot->second.closed ||
            request.viewVersion < snapshot->second.version) {
            return false;
        }
    }
    auto& slot = entry.viewDemands[request.viewId];
    if (request.viewVersion > slot.version) {
        slot.version = request.viewVersion;
        slot.distanceSquared.reset();
    }
    if (snapshot != state.viewSnapshots_.end()) {
        const auto relativeLevel = snapshot->second.relativeLevels.find(key.level);
        if (relativeLevel != snapshot->second.relativeLevels.end())
            slot.relativeLevel = relativeLevel->second;
    }
    state.viewDemandKeys_[request.viewId].insert(key);
    return true;
}

bool ChunkCache::hasDemandLocked(const Entry& entry)
{
    return entry.backgroundDemand || !entry.viewDemands.empty();
}

bool ChunkCache::cancelUndemandedEntryLocked(State& state,
                                             const ChunkKey&,
                                             Entry& entry)
{
    if (entry.status != EntryStatus::InFlight || hasDemandLocked(entry))
        return false;

    bool hadPendingTask = false;
    bool taskAlreadyRunning = false;
    auto cancel = [&](std::uint64_t& taskId,
                      const std::weak_ptr<ChunkRequestScheduler>& weakScheduler) {
        if (taskId == 0)
            return;
        hadPendingTask = true;
        auto scheduler = weakScheduler.lock();
        if (!scheduler || !scheduler->cancel(taskId)) {
            taskAlreadyRunning = true;
            return;
        }
        taskId = 0;
    };
    cancel(entry.probeTaskId, state.probeScheduler_);
    cancel(entry.fetchTaskId, state.fetchScheduler_);
    cancel(entry.decodeTaskId, state.decodeScheduler_);
    return hadPendingTask && !taskAlreadyRunning;
}

void ChunkCache::eraseUnresolvedEntryLocked(State& state,
                                            const ChunkKey& key)
{
    auto entry = state.entries_.find(key);
    if (entry == state.entries_.end())
        return;
    if (entry->second.unresolvedCounted && key.level >= 0 &&
        key.level < static_cast<int>(state.unresolvedFetchesByLevel_.size())) {
        auto& unresolved =
            state.unresolvedFetchesByLevel_[static_cast<std::size_t>(key.level)];
        if (unresolved > 0)
            --unresolved;
    }
    state.entries_.erase(entry);
}

void ChunkCache::queueFetchLocked(const std::shared_ptr<State>& state,
                                  const ChunkKey& key,
                                  std::uint64_t generation,
                                  int priorityOffset)
{
    auto it = state->entries_.find(key);
    if (it == state->entries_.end())
        return;
    Entry& entry = it->second;
    if (!entry.unresolvedCounted &&
        key.level >= 0 &&
        key.level < static_cast<int>(state->unresolvedFetchesByLevel_.size())) {
        ++state->unresolvedFetchesByLevel_[static_cast<std::size_t>(key.level)];
        entry.unresolvedCounted = true;
    }
    entry.status = EntryStatus::InFlight;
    entry.basePriority = fetchBasePriority(*state, key, priorityOffset);
    const std::uint64_t fetchSerial = state->nextFetchSerial_++;
    entry.fetchSerial = fetchSerial;
    const auto priority = workPriorityLocked(*state, key, entry);
    const auto schedulerEpoch = state->schedulerEpoch_;
    FetchContext context{
        generation,
        state->fetcherGeneration_,
        fetchSerial,
        schedulerEpoch,
        state->fetchers_.at(static_cast<std::size_t>(key.level)),
        {}};
    if (!context.fetcher)
        return;
    std::weak_ptr<State> weakState = state;
    entry.probeTaskId = 0;
    entry.fetchTaskId = 0;
    entry.decodeTaskId = 0;
    if (state->options_.persistentCachePath) {
        const auto taskId = state->nextTaskId_->fetch_add(1, std::memory_order_relaxed);
        entry.probeTaskId = taskId;
        auto scheduler = state->probeScheduler_.lock();
        if (!scheduler)
            return;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, schedulerEpoch,
            [weakState, key, context] {
                if (auto state = weakState.lock()) {
                    probePersistentAndDispatch(state, key, context);
                }
            });
    } else {
        const auto taskId = state->nextTaskId_->fetch_add(1, std::memory_order_relaxed);
        entry.fetchTaskId = taskId;
        auto scheduler = state->fetchScheduler_.lock();
        if (!scheduler)
            return;
        context.fetchScheduler = scheduler;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, schedulerEpoch,
            [weakState, key, context] {
                if (auto state = weakState.lock()) {
                    fetchRemoteAndDispatch(state, key, context);
                }
            });
    }
}

void ChunkCache::probePersistentAndDispatch(const std::shared_ptr<State>& state,
                                            ChunkKey key,
                                            FetchContext context)
{
    {
        std::lock_guard lock(state->mutex_);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_) {
            return;
        }
        auto it = state->entries_.find(key);
        if (it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        it->second.probeTaskId = 0;
    }

    PersistentProbeResult probe;
    try {
        probe = probePersistent(*state, key);
    } catch (...) {
        probe = {};
    }

    if (probe.hasData()) {
        queuePersistentDecode(state, key, context, probe);
        return;
    }

    if (probe.empty) {
        ChunkFetchResult fetch;
        fetch.status = ChunkFetchStatus::Missing;
        finishAndStore(state, key, context, std::move(fetch), true);
        return;
    }

    queueRemoteFetch(state, key, context);
}

void ChunkCache::queueRemoteFetch(const std::shared_ptr<State>& state,
                                  const ChunkKey& key,
                                  FetchContext context)
{
    bool pruned = false;
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        if (!hasDemandLocked(it->second)) {
            eraseUnresolvedEntryLocked(*state, key);
            pruned = true;
            return;
        }
        const auto taskId = state->nextTaskId_->fetch_add(
            1, std::memory_order_relaxed);
        it->second.fetchTaskId = taskId;
        auto scheduler = state->fetchScheduler_.lock();
        if (!scheduler)
            return;
        context.fetchScheduler = scheduler;
        const auto priority = workPriorityLocked(*state, key, it->second);
        std::weak_ptr<State> weakState = state;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, context.schedulerEpoch,
            [weakState, key, context] {
                if (auto state = weakState.lock()) {
                    fetchRemoteAndDispatch(state, key, context);
                }
            });
    });
    if (pruned)
        state->cv_.notify_all();
}

void ChunkCache::queuePersistentDecode(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       FetchContext context,
                                       PersistentProbeResult probe)
{
    bool pruned = false;
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        if (!hasDemandLocked(it->second)) {
            eraseUnresolvedEntryLocked(*state, key);
            pruned = true;
            return;
        }
        const auto taskId = state->nextTaskId_->fetch_add(
            1, std::memory_order_relaxed);
        it->second.decodeTaskId = taskId;
        auto scheduler = state->decodeScheduler_.lock();
        if (!scheduler)
            return;
        const auto priority = workPriorityLocked(*state, key, it->second);
        std::weak_ptr<State> weakState = state;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, context.schedulerEpoch,
            [weakState, key, context, probe] {
                if (auto state = weakState.lock()) {
                    decodePersistentAndStore(state, key, context, probe);
                }
            });
    });
    if (pruned)
        state->cv_.notify_all();
}

void ChunkCache::queueFetchedDecode(const std::shared_ptr<State>& state,
                                    const ChunkKey& key,
                                    FetchContext context,
                                    ChunkFetchResult fetched)
{
    bool pruned = false;
    auto payload = std::make_shared<ChunkFetchResult>(std::move(fetched));
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        if (!hasDemandLocked(it->second)) {
            eraseUnresolvedEntryLocked(*state, key);
            pruned = true;
            return;
        }
        const auto taskId = state->nextTaskId_->fetch_add(
            1, std::memory_order_relaxed);
        it->second.decodeTaskId = taskId;
        auto scheduler = state->decodeScheduler_.lock();
        if (!scheduler)
            return;
        const auto priority = workPriorityLocked(*state, key, it->second);
        std::weak_ptr<State> weakState = state;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, context.schedulerEpoch,
            [weakState, key, context, payload] {
                if (auto state = weakState.lock()) {
                    decodeFetchedAndStore(state, key, context, payload);
                }
            });
    });
    if (pruned)
        state->cv_.notify_all();
}

void ChunkCache::fetchRemoteAndDispatch(const std::shared_ptr<State>& state,
                                        ChunkKey key,
                                        FetchContext context)
{
    {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        it->second.fetchTaskId = 0;
    }

    ChunkFetchResult fetch;
    bool trackedRemoteFetch = false;
    const auto fetchStarted = std::chrono::steady_clock::now();
    try {
        if (state->options_.persistentCachePath) {
            trackedRemoteFetch = true;
            {
                std::lock_guard lock(state->mutex_);
                ++state->remoteFetchesInFlight_;
                state->activeRemoteFetches_[key].insert(context.fetchSerial);
            }
            notifyRemoteFetchListeners(state, key, true);
        }
        fetch = context.fetcher->fetchEncoded(fetcherKey(key));
    } catch (const std::exception& e) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = e.what();
        Logger()->error(
            "ChunkCache caught chunk fetch exception for {}/{}/{}/{}: {}",
            key.level,
            key.iz,
            key.iy,
            key.ix,
            fetch.message);
    } catch (...) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = "unknown chunk fetch exception";
        Logger()->error(
            "ChunkCache caught unknown chunk fetch exception for {}/{}/{}/{}",
            key.level,
            key.iz,
            key.iy,
            key.ix);
    }

    bool remoteActivityEnded = false;
    if (trackedRemoteFetch) {
        std::lock_guard lock(state->mutex_);
        auto active = state->activeRemoteFetches_.find(key);
        if (active != state->activeRemoteFetches_.end() &&
            active->second.erase(context.fetchSerial) != 0) {
            if (state->remoteFetchesInFlight_ > 0)
                --state->remoteFetchesInFlight_;
            if (active->second.empty()) {
                state->activeRemoteFetches_.erase(active);
                remoteActivityEnded = true;
            }
        }
    }
    if (fetch.status == ChunkFetchStatus::Found && !fetch.bytes.empty()) {
        if (context.fetchScheduler) {
            context.fetchScheduler->recordSuccessfulTransfer(
                fetch.bytes.size(), fetchStarted,
                std::chrono::steady_clock::now());
        }
    }
    if (remoteActivityEnded)
        notifyRemoteFetchListeners(state, key, false);

    if (fetch.status == ChunkFetchStatus::Found) {
        queueFetchedDecode(state, key, context, std::move(fetch));
        return;
    }

    finishAndStore(state, key, context, std::move(fetch), false);
}

void ChunkCache::decodePersistentAndStore(
    const std::shared_ptr<State>& state,
    ChunkKey key,
    FetchContext context,
    PersistentProbeResult probe)
{
    {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        it->second.decodeTaskId = 0;
    }

    ChunkFetchResult decoded;
    bool resolved = false;
    try {
        if (auto cached = readPersistent(*state, key, probe)) {
            decoded = context.fetcher->decodePersistentBytes(
                fetcherKey(key), std::move(*cached));
            resolved = decoded.status == ChunkFetchStatus::Found &&
                       decoded.bytes.size() == expectedChunkBytes(*state, key);
        }
    } catch (...) {
        resolved = false;
    }

    if (!resolved) {
        queueRemoteFetch(state, key, context);
        return;
    }

    finishAndStore(state, key, context, std::move(decoded), true);
}

void ChunkCache::decodeFetchedAndStore(
    const std::shared_ptr<State>& state,
    ChunkKey key,
    FetchContext context,
    std::shared_ptr<ChunkFetchResult> fetched)
{
    {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        it->second.decodeTaskId = 0;
    }

    ChunkFetchResult decoded;
    try {
        decoded = context.fetcher->decodeFetched(
            fetcherKey(key), std::move(*fetched));
    } catch (const std::exception& e) {
        decoded.status = ChunkFetchStatus::DecodeError;
        decoded.message = e.what();
    } catch (...) {
        decoded.status = ChunkFetchStatus::DecodeError;
        decoded.message = "unknown chunk decode exception";
    }

    finishAndStore(state, key, context, std::move(decoded), false);
}

void ChunkCache::finishAndStore(const std::shared_ptr<State>& state,
                                const ChunkKey& key,
                                FetchContext context,
                                ChunkFetchResult fetch,
                                bool loadedFromPersistentCache)
{
    {
        std::lock_guard lock(state->mutex_);
        auto it = state->entries_.find(key);
        if (context.generation != state->generation_ ||
            context.fetcherGeneration != state->fetcherGeneration_ ||
            it == state->entries_.end() ||
            it->second.fetchSerial != context.fetchSerial) {
            return;
        }
        it->second.probeTaskId = 0;
        it->second.fetchTaskId = 0;
        it->second.decodeTaskId = 0;
        storeFetchResultLocked(
            state, key, std::move(fetch), loadedFromPersistentCache);
    }
    enforceSharedBudget(state);
    state->cv_.notify_all();
    notifyListeners(state);
}

void ChunkCache::storeFetchResultLocked(const std::shared_ptr<State>& state,
                                        const ChunkKey& key,
                                        ChunkFetchResult fetch,
                                        bool loadedFromPersistentCache)
{
    auto it = state->entries_.find(key);
    if (it == state->entries_.end())
        return;

    Entry& entry = it->second;
    if (entry.unresolvedCounted &&
        key.level >= 0 &&
        key.level < static_cast<int>(state->unresolvedFetchesByLevel_.size())) {
        auto& unresolved =
            state->unresolvedFetchesByLevel_[static_cast<std::size_t>(key.level)];
        if (unresolved > 0)
            --unresolved;
        entry.unresolvedCounted = false;
    }
    if (entry.inLru) {
        state->lru_.erase(entry.lruIt);
        entry.inLru = false;
    }
    if (entry.status == EntryStatus::Data) {
        state->decodedBytes_ -= entry.decodedBytes;
        removeDecodedBytesLocked(*state, entry.decodedBytes);
    }

    entry.bytes.reset();
    entry.error.clear();
    entry.decodedBytes = 0;
    entry.persisted = false;
    entry.persistentWriteQueued = false;

    switch (fetch.status) {
    case ChunkFetchStatus::Found: {
        if (fetch.bytes.size() != expectedChunkBytes(*state, key)) {
            entry.status = EntryStatus::Error;
            entry.error = "decoded chunk byte size does not match full chunk shape";
            break;
        }
        if (state->options_.detectAllFillChunks && isAllFill(*state, fetch.bytes)) {
            entry.status = EntryStatus::AllFill;
            // `persisted` is set by the writer's completion callback once
            // the bytes are actually on disk (same for the cases below).
            entry.persisted = loadedFromPersistentCache;
            if (!loadedFromPersistentCache)
                entry.persistentWriteQueued = queuePersistentEmptyWrite(state, key);
            break;
        }
        entry.status = EntryStatus::Data;
        entry.decodedBytes = fetch.bytes.size();
        entry.bytes = std::make_shared<const std::vector<std::byte>>(std::move(fetch.bytes));
        state->decodedBytes_ += entry.decodedBytes;
        addDecodedBytesLocked(*state, entry.decodedBytes);
        std::shared_ptr<const std::vector<std::byte>> persistentBytes = entry.bytes;
        if (fetch.hasPersistentBytes) {
            persistentBytes = std::make_shared<const std::vector<std::byte>>(
                std::move(fetch.persistentBytes));
        }
        entry.persisted = loadedFromPersistentCache;
        if (!loadedFromPersistentCache)
            entry.persistentWriteQueued =
                queuePersistentWrite(state, key, std::move(persistentBytes));
        break;
    }
    case ChunkFetchStatus::Missing:
        entry.status = EntryStatus::Missing;
        entry.persisted = loadedFromPersistentCache;
        if (!loadedFromPersistentCache)
            entry.persistentWriteQueued = queuePersistentEmptyWrite(state, key);
        break;
    case ChunkFetchStatus::HttpError:
    case ChunkFetchStatus::IoError:
    case ChunkFetchStatus::DecodeError:
        entry.status = EntryStatus::Error;
        entry.error = fetchErrorMessage(fetch);
        break;
    }

    touchLocked(*state, key, entry);
    enforceCapacityLocked(state);
}

namespace {

std::optional<std::vector<std::byte>> readFileBytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return std::nullopt;

    const auto size = file.tellg();
    if (size < 0)
        return std::nullopt;

    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file)
        return std::nullopt;
    return bytes;
}

} // namespace

ChunkCache::PersistentProbeResult ChunkCache::probePersistent(
    const State& state,
    const ChunkKey& key)
{
    PersistentProbeResult result;
    if (!state.options_.persistentCachePath)
        return result;

    auto exists = [](const std::filesystem::path& path) {
        std::error_code ec;
        return std::filesystem::exists(path, ec) && !ec;
    };

    if (persistentEntryIsRaw(state, key))
        result.compressedData = exists(persistentCompressedPath(state, key));
    result.primaryData = exists(persistentPath(state, key));
    if (!result.hasData()) {
        const auto path = persistentEmptyPath(state, key);
        auto pin = state.persistentBudget_
            ? state.persistentBudget_->pinRead(path)
            : PersistentZarrCacheBudget::ReadPin{};
        result.empty = exists(path);
        pin.complete(result.empty);
    }
    return result;
}

std::optional<std::vector<std::byte>> ChunkCache::readPersistent(
    const State& state,
    const ChunkKey& key,
    const PersistentProbeResult& probe)
{
    if (!state.options_.persistentCachePath || !probe.hasData())
        return std::nullopt;

    const bool rawEntry = persistentEntryIsRaw(state, key);
    auto readManaged = [&](const std::filesystem::path& path) {
        auto pin = state.persistentBudget_
            ? state.persistentBudget_->pinRead(path)
            : PersistentZarrCacheBudget::ReadPin{};
        auto bytes = readFileBytes(path);
        pin.complete(bytes.has_value());
        return bytes;
    };
    if (rawEntry && probe.compressedData) {
        // Compressed variant wins when both formats exist: compaction and
        // compressed writes leave ".zst" as the authoritative copy.
        if (auto compressed = readManaged(persistentCompressedPath(state, key))) {
            auto decompressed = vc::cacheDecompress(
                std::span<const std::byte>(compressed->data(), compressed->size()),
                expectedChunkBytes(state, key));
            if (decompressed)
                return decompressed;
            // Corrupt compressed entry — fall through to ".bin"/refetch.
            Logger()->warn(
                "ChunkCache corrupt compressed cache entry for {}/{}/{}/{} ({} bytes); "
                "falling back to raw copy or refetch",
                key.level, key.iz, key.iy, key.ix, compressed->size());
        }
    }

    if (!probe.primaryData)
        return std::nullopt;
    auto bytes = readManaged(persistentPath(state, key));
    if (!bytes)
        return std::nullopt;
    if (rawEntry && bytes->size() != expectedChunkBytes(state, key))
        return std::nullopt;
    return bytes;
}

bool ChunkCache::queuePersistentWrite(const std::shared_ptr<State>& state,
                                      const ChunkKey& key,
                                      std::shared_ptr<const std::vector<std::byte>> bytes)
{
    if (!state || !state->options_.persistentCachePath || !bytes)
        return false;
    if (persistentEntryIsRaw(*state, key) &&
        bytes->size() != expectedChunkBytes(*state, key))
        return false;

    const std::size_t retainedBytes = bytes->size();
    if (!reservePersistentWriteBytes(retainedBytes)) {
        return false;
    }
    state->persistentWritesInFlight_.fetch_add(1, std::memory_order_acq_rel);
    try {
        persistentCacheWriterPool().enqueue(
            [state, key, retainedBytes, bytes = std::move(bytes)] {
                bool written = false;
                try {
                    written = writePersistent(*state, key, *bytes);
                } catch (...) {
                }
                {
                    std::lock_guard lock(state->mutex_);
                    // Flag persistence only once the bytes are actually on
                    // disk. Decrement under the mutex so
                    // waitForPersistentWrites cannot miss the wakeup.
                    auto it = state->entries_.find(key);
                    if (it != state->entries_.end() &&
                        it->second.status == EntryStatus::Data) {
                        it->second.persistentWriteQueued = false;
                        if (written)
                            it->second.persisted = true;
                    }
                    state->persistentWritesInFlight_.fetch_sub(
                        1, std::memory_order_acq_rel);
                }
                releasePersistentWriteBytes(retainedBytes);
                state->cv_.notify_all();
            });
    } catch (...) {
        state->persistentWritesInFlight_.fetch_sub(1, std::memory_order_acq_rel);
        releasePersistentWriteBytes(retainedBytes);
        return false;
    }
    return true;
}

bool ChunkCache::queuePersistentEmptyWrite(const std::shared_ptr<State>& state,
                                           const ChunkKey& key)
{
    if (!state || !state->options_.persistentCachePath)
        return false;

    state->persistentWritesInFlight_.fetch_add(1, std::memory_order_acq_rel);
    persistentCacheWriterPool().enqueue([state, key] {
        bool written = false;
        try {
            written = writePersistentEmpty(*state, key);
        } catch (...) {
        }
        {
            std::lock_guard lock(state->mutex_);
            if (written) {
                auto it = state->entries_.find(key);
                if (it != state->entries_.end() &&
                    (it->second.status == EntryStatus::Missing ||
                     it->second.status == EntryStatus::AllFill)) {
                    it->second.persistentWriteQueued = false;
                    it->second.persisted = true;
                }
            } else {
                auto it = state->entries_.find(key);
                if (it != state->entries_.end() &&
                    (it->second.status == EntryStatus::Missing ||
                     it->second.status == EntryStatus::AllFill))
                    it->second.persistentWriteQueued = false;
            }
            state->persistentWritesInFlight_.fetch_sub(1, std::memory_order_acq_rel);
        }
        state->cv_.notify_all();
    });
    return true;
}

bool ChunkCache::writePersistent(State& state, const ChunkKey& key, const std::vector<std::byte>& bytes)
{
    if (!state.options_.persistentCachePath)
        return false;
    const bool rawEntry = persistentEntryIsRaw(state, key);
    if (rawEntry && bytes.size() != expectedChunkBytes(state, key))
        return false;

    bool compress = rawEntry && state.options_.compressPersistentCache;
    const std::vector<std::byte>* payload = &bytes;
    std::vector<std::byte> compressed;
    if (compress) {
        try {
            compressed = vc::cacheCompress(
                std::span<const std::byte>(bytes.data(), bytes.size()),
                state.levels_[static_cast<std::size_t>(key.level)].chunkShape,
                dtypeSize(state.dtype_),
                vc::kCacheCompressionLevel,
                state.options_.cacheQuantBinWidth);
            // A frame the decoder cannot read back is worse than no entry:
            // readPersistent falls through to a remote refetch and the raw
            // counterpart gets deleted below. Verify decodability before
            // committing the compressed copy.
            if (!vc::cacheDecompress(
                    std::span<const std::byte>(compressed.data(), compressed.size()),
                    bytes.size())) {
                Logger()->warn(
                    "ChunkCache compressed self-check failed for {}/{}/{}/{}; storing raw",
                    key.level, key.iz, key.iy, key.ix);
                compress = false;
            }
        } catch (const std::exception& e) {
            Logger()->warn("ChunkCache persistent-cache compression failed: {}; storing raw",
                           e.what());
            compress = false;
        }
        if (compress)
            payload = &compressed;
    }

    const auto path = compress ? persistentCompressedPath(state, key)
                               : persistentPath(state, key);
    const auto counterpart = rawEntry
        ? (compress ? persistentPath(state, key) : persistentCompressedPath(state, key))
        : std::filesystem::path{};
    auto reservation = state.persistentBudget_
        ? state.persistentBudget_->reserveWrite(
              path, payload->size(), counterpart.empty()
                  ? std::vector<std::filesystem::path>{}
                  : std::vector<std::filesystem::path>{counterpart})
        : PersistentZarrCacheBudget::WriteReservation{};
    if (state.persistentBudget_ && !reservation)
        return false;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
        return false;
    const auto oldSize = regularFileSize(path).value_or(0);
    const auto tmp = path.string() + uniqueTmpSuffix();
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        file.write(reinterpret_cast<const char*>(payload->data()),
                   static_cast<std::streamsize>(payload->size()));
        if (!file) {
            file.close();
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        std::filesystem::remove(tmp, ec);
        const auto finalSize = regularFileSize(path).value_or(0);
        addPersistentCacheBytesDelta(
            state,
            static_cast<std::int64_t>(finalSize) - static_cast<std::int64_t>(oldSize));
        // The overwrite fallback may have removed the tracked destination even
        // though publishing failed. Refresh all reserved paths from disk.
        reservation.commit();
        return false;
    }
    std::int64_t removedCounterpart = 0;
    if (rawEntry) {
        // Drop the other-format copy so the freshly written file is
        // authoritative (reads prefer ".zst" over ".bin").
        if (const auto size = regularFileSize(counterpart)) {
            std::error_code removeEc;
            if (std::filesystem::remove(counterpart, removeEc) && !removeEc)
                removedCounterpart = static_cast<std::int64_t>(*size);
        }
    }
    const auto newSize = regularFileSize(path).value_or(payload->size());
    addPersistentCacheBytesDelta(
        state,
        static_cast<std::int64_t>(newSize) - static_cast<std::int64_t>(oldSize) -
            removedCounterpart);
    reservation.commit();
    return true;
}

bool ChunkCache::writePersistentEmpty(State& state, const ChunkKey& key)
{
    if (!state.options_.persistentCachePath)
        return false;

    const auto path = persistentEmptyPath(state, key);
    auto reservation = state.persistentBudget_
        ? state.persistentBudget_->reserveWrite(path, 0)
        : PersistentZarrCacheBudget::WriteReservation{};
    if (state.persistentBudget_ && !reservation)
        return false;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
        return false;
    const auto oldSize = regularFileSize(path).value_or(0);
    const auto tmp = path.string() + uniqueTmpSuffix();
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        if (!file) {
            file.close();
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        std::filesystem::remove(tmp, ec);
        const auto finalSize = regularFileSize(path).value_or(0);
        addPersistentCacheBytesDelta(
            state,
            static_cast<std::int64_t>(finalSize) - static_cast<std::int64_t>(oldSize));
        // The overwrite fallback may have removed the tracked destination even
        // though publishing failed. Refresh the reservation from disk.
        reservation.commit();
        return false;
    }
    const auto newSize = regularFileSize(path).value_or(std::size_t{0});
    addPersistentCacheBytesDelta(
        state,
        static_cast<std::int64_t>(newSize) - static_cast<std::int64_t>(oldSize));
    reservation.commit();
    return true;
}

std::filesystem::path ChunkCache::persistentPath(const State& state, const ChunkKey& key)
{
    return *state.options_.persistentCachePath /
           ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) /
           std::to_string(key.iy) /
           (std::to_string(key.ix) +
            state.persistentExtensions_.at(static_cast<std::size_t>(key.level)));
}

std::filesystem::path ChunkCache::persistentCompressedPath(const State& state, const ChunkKey& key)
{
    return *state.options_.persistentCachePath /
           ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) /
           std::to_string(key.iy) /
           (std::to_string(key.ix) + vc::kCompressedCacheExtension);
}

bool ChunkCache::persistentEntryIsRaw(const State& state, const ChunkKey& key)
{
    return state.persistentExtensions_.at(static_cast<std::size_t>(key.level)) == ".bin";
}

std::filesystem::path ChunkCache::persistentEmptyPath(const State& state, const ChunkKey& key)
{
    return *state.options_.persistentCachePath /
           ("level_" + std::to_string(key.level)) /
           std::to_string(key.iz) /
           std::to_string(key.iy) /
           (std::to_string(key.ix) + ".empty");
}

void ChunkCache::startPersistentCacheSizeScan(const std::shared_ptr<State>& state)
{
    if (!state || !state->options_.persistentCachePath)
        return;

    const auto path = *state->options_.persistentCachePath;
    const auto cutoff = std::filesystem::file_time_type::clock::now();
    state->persistentCacheScanInFlight_.store(true, std::memory_order_release);
    persistentCacheWriterPool().enqueue([state, path, cutoff] {
        const auto bytes = persistentCacheBytes(path, cutoff);
        addPersistentCacheBytesDelta(*state, static_cast<std::int64_t>(bytes));
        state->persistentCacheScanInFlight_.store(false, std::memory_order_release);
    });
}

std::size_t ChunkCache::persistentCacheBytes(
    const std::filesystem::path& path,
    std::filesystem::file_time_type cutoff)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec) || ec)
        return 0;

    std::size_t bytes = 0;
    std::filesystem::recursive_directory_iterator it(
        path,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    const std::filesystem::recursive_directory_iterator end;
    while (!ec && it != end) {
        if (it->is_regular_file(ec)) {
            const auto modified = it->last_write_time(ec);
            if (!ec && modified <= cutoff) {
                const auto size = it->file_size(ec);
                if (!ec)
                    bytes += static_cast<std::size_t>(size);
            }
            if (ec)
                ec.clear();
        } else {
            ec.clear();
        }
        it.increment(ec);
    }
    return bytes;
}

std::optional<std::size_t> ChunkCache::regularFileSize(const std::filesystem::path& path)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec)
        return std::nullopt;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec)
        return std::nullopt;
    return static_cast<std::size_t>(size);
}

void ChunkCache::addPersistentCacheBytesDelta(State& state, std::int64_t delta)
{
    if (delta == 0)
        return;
    auto current = state.persistentCacheBytes_.load(std::memory_order_acquire);
    while (true) {
        const auto next = std::max<std::int64_t>(0, current + delta);
        if (state.persistentCacheBytes_.compare_exchange_weak(
                current,
                next,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return;
        }
    }
}

void ChunkCache::touchLocked(State& state, const ChunkKey& key, Entry& entry)
{
    if (entry.status == EntryStatus::InFlight)
        return;
    if (entry.inLru)
        state.lru_.erase(entry.lruIt);
    state.lru_.push_front(key);
    entry.lruIt = state.lru_.begin();
    entry.inLru = true;
    if (state.options_.decodedByteBudget)
        entry.budgetTouch = state.options_.decodedByteBudget->nextTouch();
}

void ChunkCache::enforceCapacityLocked(const std::shared_ptr<State>& state)
{
    auto overBudget = [&] {
        return state->decodedBytes_ > state->options_.decodedByteCapacity ||
               state->entries_.size() > state->options_.metadataEntryCapacity;
    };
    if (!overBudget())
        return;

    while (overBudget() && !state->lru_.empty()) {
        auto victimIt = std::prev(state->lru_.end());
        auto entryIt = state->entries_.find(*victimIt);
        if (entryIt == state->entries_.end()) {
            state->lru_.erase(victimIt);
            continue;
        }
        Entry& entry = entryIt->second;
        const ChunkKey victim = *victimIt;
        state->lru_.erase(victimIt);
        entry.inLru = false;
        if (entry.status == EntryStatus::Data) {
            if (entry.bytes && !entry.persisted && !entry.persistentWriteQueued)
                entry.persistentWriteQueued = queuePersistentWrite(state, victim, entry.bytes);
            state->decodedBytes_ -= entry.decodedBytes;
            removeDecodedBytesLocked(*state, entry.decodedBytes);
        }
        state->entries_.erase(entryIt);
    }
}

std::optional<std::uint64_t> ChunkCache::oldestDecodedTouch(
    const std::shared_ptr<State>& state)
{
    std::lock_guard lock(state->mutex_);
    for (auto it = state->lru_.rbegin(); it != state->lru_.rend(); ++it) {
        auto entry = state->entries_.find(*it);
        if (entry != state->entries_.end() &&
            entry->second.status == EntryStatus::Data) {
            return entry->second.budgetTouch;
        }
    }
    return std::nullopt;
}

std::size_t ChunkCache::evictOldestDecoded(const std::shared_ptr<State>& state)
{
    std::lock_guard lock(state->mutex_);
    return evictOldestDecodedLocked(state);
}

std::size_t ChunkCache::evictOldestDecodedLocked(const std::shared_ptr<State>& state)
{
    for (auto it = state->lru_.end(); it != state->lru_.begin();) {
        --it;
        auto entryIt = state->entries_.find(*it);
        if (entryIt == state->entries_.end()) {
            it = state->lru_.erase(it);
            continue;
        }
        Entry& entry = entryIt->second;
        if (entry.status != EntryStatus::Data)
            continue;

        const ChunkKey victim = *it;
        const std::size_t bytes = entry.decodedBytes;
        if (entry.bytes && !entry.persisted && !entry.persistentWriteQueued)
            entry.persistentWriteQueued = queuePersistentWrite(state, victim, entry.bytes);
        state->lru_.erase(it);
        entry.inLru = false;
        state->decodedBytes_ -= bytes;
        removeDecodedBytesLocked(*state, bytes);
        state->entries_.erase(entryIt);
        return bytes;
    }
    return 0;
}

void ChunkCache::addDecodedBytesLocked(State& state, std::size_t bytes)
{
    if (bytes > 0 && state.options_.decodedByteBudget)
        state.options_.decodedByteBudget->addBytes(bytes);
}

void ChunkCache::removeDecodedBytesLocked(State& state, std::size_t bytes)
{
    if (bytes > 0 && state.options_.decodedByteBudget)
        state.options_.decodedByteBudget->removeBytes(bytes);
}

void ChunkCache::enforceSharedBudget(const std::shared_ptr<State>& state)
{
    if (state->options_.decodedByteBudget)
        state->options_.decodedByteBudget->enforce();
}

bool ChunkCache::isValidKey(const State& state, const ChunkKey& key)
{
    if (key.level < 0 || key.level >= static_cast<int>(state.levels_.size()))
        return false;
    if (!state.fetchers_[static_cast<std::size_t>(key.level)])
        return false;
    const auto& level = state.levels_[static_cast<std::size_t>(key.level)];
    const std::array<int, 3> coords{key.iz, key.iy, key.ix};
    for (int axis = 0; axis < 3; ++axis) {
        if (coords[axis] < 0)
            return false;
        const int chunks = (level.shape[axis] + level.chunkShape[axis] - 1) / level.chunkShape[axis];
        if (coords[axis] >= chunks)
            return false;
    }
    return true;
}

bool ChunkCache::isAllFill(const State& state, const std::vector<std::byte>& bytes)
{
    if (state.dtype_ == ChunkDtype::UInt8) {
        const auto fill = static_cast<unsigned char>(std::clamp(
            state.fillValue_, 0.0, static_cast<double>(std::numeric_limits<unsigned char>::max())));
        return std::all_of(bytes.begin(), bytes.end(), [fill](std::byte value) {
            return static_cast<unsigned char>(value) == fill;
        });
    }

    const auto fill = static_cast<std::uint16_t>(std::clamp(
        state.fillValue_, 0.0, static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
    if (bytes.size() % sizeof(std::uint16_t) != 0)
        return false;
    const auto* ptr = reinterpret_cast<const std::uint16_t*>(bytes.data());
    const std::size_t count = bytes.size() / sizeof(std::uint16_t);
    return std::all_of(ptr, ptr + count, [fill](std::uint16_t value) {
        return value == fill;
    });
}

std::size_t ChunkCache::dtypeSize(ChunkDtype dtype)
{
    switch (dtype) {
    case ChunkDtype::UInt8:
        return 1;
    case ChunkDtype::UInt16:
        return 2;
    }
    return 1;
}

std::size_t ChunkCache::expectedChunkBytes(const State& state, const ChunkKey& key)
{
    const auto& chunk = state.levels_[static_cast<std::size_t>(key.level)].chunkShape;
    return static_cast<std::size_t>(chunk[0]) *
           static_cast<std::size_t>(chunk[1]) *
           static_cast<std::size_t>(chunk[2]) *
           dtypeSize(state.dtype_);
}

void ChunkCache::notifyListeners(const std::shared_ptr<State>& state)
{
    std::vector<ChunkReadyCallback> callbacks;
    {
        std::lock_guard lock(state->mutex_);
        callbacks.reserve(state->callbacks_.size());
        for (const auto& [id, cb] : state->callbacks_) {
            (void)id;
            callbacks.push_back(cb);
        }
    }
    for (auto& cb : callbacks) {
        if (cb)
            cb();
    }
}

void ChunkCache::notifyRemoteFetchListeners(const std::shared_ptr<State>& state,
                                            const ChunkKey& key,
                                            bool active)
{
    std::vector<RemoteFetchActivityCallback> callbacks;
    {
        std::lock_guard lock(state->mutex_);
        callbacks.reserve(state->remoteFetchCallbacks_.size());
        for (const auto& [id, cb] : state->remoteFetchCallbacks_) {
            (void)id;
            callbacks.push_back(cb);
        }
    }
    for (auto& cb : callbacks) {
        if (cb)
            cb(key, active);
    }
}

void ChunkCache::waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key)
{
    state.cv_.wait(lock, [&] {
        auto it = state.entries_.find(key);
        return it == state.entries_.end() || it->second.status != EntryStatus::InFlight;
    });
}

} // namespace vc::render
