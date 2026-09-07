#include "ChunkCache.hpp"

#include <utils/thread_pool.hpp>
#include "vc/core/util/Logging.hpp"
#include "vc/core/render/ChunkRequestScheduler.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif

namespace vc::render {

namespace {

constexpr std::size_t kPersistentProbeWorkers = 32;
constexpr std::size_t kDecodeWorkers = 8;
constexpr int kTerminalLevelPriorityBonus = 100;

bool atomicWriteBytes(const std::filesystem::path& path,
                      std::span<const std::byte> bytes);
std::optional<std::vector<std::byte>>
readFileBytes(const std::filesystem::path& path);

class PersistentCacheLease final {
public:
    static std::shared_ptr<PersistentCacheLease> tryAcquire(
        const std::filesystem::path& cachePath)
    {
        std::error_code ec;
        std::filesystem::create_directories(cachePath.parent_path(), ec);
        if (ec)
            return {};
        auto lease = std::shared_ptr<PersistentCacheLease>(
            new PersistentCacheLease(leasePath(cachePath)));
        if (!lease->lock())
            return {};
        return lease;
    }

    ~PersistentCacheLease()
    {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED overlapped{};
            UnlockFileEx(handle_, 0, 1, 0, &overlapped);
            CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
#endif
    }

private:
    explicit PersistentCacheLease(std::filesystem::path path)
        : path_(std::move(path))
    {
    }

    static std::filesystem::path leasePath(
        const std::filesystem::path& cachePath)
    {
        return cachePath.parent_path() /
               ("." + cachePath.filename().string() + ".vc_cache.lock");
    }

    bool lock()
    {
#if defined(_WIN32)
        handle_ = CreateFileW(
            path_.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        return handle_ != INVALID_HANDLE_VALUE && lockWindows();
#else
        fd_ = open(path_.c_str(), O_RDWR | O_CREAT, 0666);
        return fd_ >= 0 && flock(fd_, LOCK_SH | LOCK_NB) == 0;
#endif
    }

#if defined(_WIN32)
    bool lockWindows()
    {
        OVERLAPPED overlapped{};
        return LockFileEx(
                   handle_, LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped) != 0;
    }
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    std::filesystem::path path_;
};

struct PersistentCachePreparation {
    std::shared_ptr<PersistentCacheLease> lease;
    std::string warning;
};

bool isExactCacheDirectoryTarget(const std::filesystem::path& path)
{
    if (path.empty() || !path.has_filename() || path == path.root_path())
        return false;
    if (path.parent_path().empty() || path.parent_path() == path.root_path())
        return false;
    const auto filename = path.filename();
    if (filename.empty() || filename == "." || filename == "..")
        return false;
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory)
        ec.clear();
    return !ec && !std::filesystem::is_symlink(status);
}

bool isStrictlyWithin(const std::filesystem::path& path,
                      const std::filesystem::path& root)
{
    std::error_code ec;
    const auto absolutePath = std::filesystem::absolute(path, ec).lexically_normal();
    if (ec)
        return false;
    const auto absoluteRoot = std::filesystem::absolute(root, ec).lexically_normal();
    if (ec || absolutePath == absoluteRoot)
        return false;
    auto pathIt = absolutePath.begin();
    for (auto rootIt = absoluteRoot.begin(); rootIt != absoluteRoot.end();
         ++rootIt, ++pathIt) {
        if (pathIt == absolutePath.end() || *pathIt != *rootIt)
            return false;
    }
    return true;
}

PersistentCachePreparation preparePersistentCache(
    const std::filesystem::path& root,
    const std::optional<std::filesystem::path>& budgetRoot)
{
    PersistentCachePreparation result;
    if (!isExactCacheDirectoryTarget(root)) {
        result.warning = "persistent cache path is not a safe per-volume directory: " +
                         root.string();
        return result;
    }
    if (budgetRoot && !isStrictlyWithin(root, *budgetRoot)) {
        result.warning = "persistent cache path is not an exact volume directory beneath its cache root: " +
                         root.string();
        return result;
    }

    result.lease = PersistentCacheLease::tryAcquire(root);
    if (!result.lease) {
        result.warning = "persistent cache lease is unavailable";
    }
    return result;
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

bool atomicWriteBytes(const std::filesystem::path& path,
                      std::span<const std::byte> bytes)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
        return false;
    const auto tmp = path.string() + uniqueTmpSuffix();
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file)
            return false;
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            file.close();
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (!ec)
        return true;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (!ec)
        return true;
    std::filesystem::remove(tmp, ec);
    return false;
}

std::optional<std::vector<std::byte>>
readFileBytes(const std::filesystem::path& path);

} // namespace

struct ChunkCacheService::Impl {
    explicit Impl(Options options)
        : decodedByteBudget(std::move(options.decodedByteBudget))
        , initialAdaptiveDownloadState(
              std::move(options.initialAdaptiveDownloadState))
        , activeFetchWorkers(options.fetchConcurrency.maxConcurrentReads)
        , activeFetchAdaptive(options.fetchConcurrency.adaptive)
    {
        const std::size_t workerCapacity =
            options.fetchConcurrency.workerCapacity;
        if (workerCapacity == 0 || activeFetchWorkers == 0 ||
            activeFetchWorkers > workerCapacity) {
            throw std::invalid_argument(
                "ChunkCacheService fetch concurrency must be within worker capacity");
        }
        if (!decodedByteBudget) {
            decodedByteBudget = std::make_shared<DecodedChunkCacheBudget>(
                options.decodedByteCapacity);
        }

        std::optional<ChunkRequestScheduler::AdaptiveConcurrency> adaptiveOptions;
        std::optional<ChunkRequestScheduler::AdaptiveState> initialState;
        if (activeFetchAdaptive) {
            adaptiveOptions.emplace();
            adaptiveOptions->minimum = std::min<std::size_t>(
                2, activeFetchWorkers);
            adaptiveOptions->maximum = activeFetchWorkers;
            if (initialAdaptiveDownloadState) {
                initialState = ChunkRequestScheduler::AdaptiveState{
                    initialAdaptiveDownloadState->settledAdmissionLimit,
                    initialAdaptiveDownloadState->longTermBytesPerSecond,
                    initialAdaptiveDownloadState->maximumSaturatedParallelism,
                    initialAdaptiveDownloadState
                        ->saturatedBytesPerSecondPerWorker};
            }
        }
        fetchScheduler = std::make_shared<ChunkRequestScheduler>(
            workerCapacity, 7, schedulerSelectionGate,
            adaptiveOptions, initialState);
        if (!activeFetchAdaptive) {
            fetchScheduler->configureConcurrency(activeFetchWorkers);
        }
    }

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
    std::shared_ptr<ChunkRequestScheduler> fetchScheduler;
    std::size_t activeFetchWorkers = 0;
    bool activeFetchAdaptive = false;
};

namespace {

struct ProcessChunkCacheServiceRegistry {
    std::mutex mutex;
    std::shared_ptr<ChunkCacheService> service;
};

ProcessChunkCacheServiceRegistry& processChunkCacheServiceRegistry()
{
    static ProcessChunkCacheServiceRegistry registry;
    return registry;
}

ChunkCacheService::Options defaultProcessChunkCacheServiceOptions()
{
    ChunkCacheService::Options options;
    options.decodedByteCapacity = 8ULL << 30;
    options.fetchConcurrency.workerCapacity = 64;
    options.fetchConcurrency.maxConcurrentReads = 64;
    options.fetchConcurrency.adaptive = true;
    return options;
}

} // namespace

ChunkCacheService::ChunkCacheService()
    : ChunkCacheService(Options{})
{
}

ChunkCacheService::ChunkCacheService(Options options)
    : impl_(std::make_shared<Impl>(std::move(options)))
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
    impl_->fetchScheduler->waitIdle();
    impl_->decodeScheduler->waitIdle();
}

std::shared_ptr<ChunkCacheService> processChunkCacheService()
{
    auto& registry = processChunkCacheServiceRegistry();
    std::lock_guard lock(registry.mutex);
    if (!registry.service) {
        registry.service = std::make_shared<ChunkCacheService>(
            defaultProcessChunkCacheServiceOptions());
    }
    return registry.service;
}

std::shared_ptr<ChunkCacheService> configureProcessChunkCacheService(
    ChunkCacheService::Options options)
{
    auto& registry = processChunkCacheServiceRegistry();
    std::lock_guard lock(registry.mutex);
    if (!registry.service) {
        registry.service = std::make_shared<ChunkCacheService>(
            std::move(options));
        return registry.service;
    }

    const auto current = registry.service->fetchConcurrency();
    if (options.fetchConcurrency.workerCapacity != current.workerCapacity) {
        throw std::invalid_argument(
            "process ChunkCacheService worker capacity is fixed after creation");
    }
    if (options.initialAdaptiveDownloadState) {
        throw std::invalid_argument(
            "process ChunkCacheService adaptive state must be supplied before first use");
    }
    registry.service->configureDecodedByteCapacity(
        options.decodedByteCapacity);
    registry.service->configureFetchConcurrency(
        options.fetchConcurrency.maxConcurrentReads,
        options.fetchConcurrency.adaptive);
    return registry.service;
}

std::shared_ptr<DecodedChunkCacheBudget>
ChunkCacheService::decodedByteBudget() const
{
    return impl_->decodedByteBudget;
}

std::optional<ChunkCacheService::AdaptiveDownloadState>
ChunkCacheService::adaptiveDownloadState() const
{
    const auto state = impl_->fetchScheduler->adaptiveState();
    if (!state)
        return impl_->initialAdaptiveDownloadState;
    return AdaptiveDownloadState{
        state->settledAdmissionLimit,
        state->longTermBytesPerSecond,
        state->maximumSaturatedParallelism,
        state->saturatedBytesPerSecondPerWorker};
}

std::shared_ptr<ChunkCache> ChunkCacheService::acquireSource(
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
    ChunkCache::validateSourceDefinition(levels, fetchers);
    PersistentCachePreparation cachePreparation;
    if (options.persistentCachePath) {
        ChunkCache::validatePersistentMirror(options, fetchers);
        cachePreparation = preparePersistentCache(
            *options.persistentCachePath,
            options.persistentCacheBudgetRoot);
        if (!cachePreparation.lease) {
            Logger()->warn(
                "ChunkCache disabled persistent caching for {}: {}",
                sourceIdentity, cachePreparation.warning);
            options.persistentCachePath.reset();
            options.persistentCacheBudgetRoot.reset();
        }
    }
    auto service = shared_from_this();
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
        if (options.decodedEvictionPreferSelf !=
            state->options_.decodedEvictionPreferSelf) {
            // Eviction preference is a per-source policy captured by the
            // FIRST acquisition's budget registration; it is deliberately not
            // part of metadataCompatible (it describes no data), but a
            // differing later request must not be ignored silently.
            Logger()->warn(
                "ChunkCache source {} already registered with "
                "decodedEvictionPreferSelf={}; ignoring differing request",
                sourceIdentity,
                state->options_.decodedEvictionPreferSelf);
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
        std::move(options), impl_->decodedByteBudget, sourceId, sourceIdentity);
    state->persistentLease_ = std::move(cachePreparation.lease);
    state->persistentCacheWarning_ = std::move(cachePreparation.warning);
    state->probeScheduler_ = impl_->probeScheduler;
    state->fetchScheduler_ = impl_->fetchScheduler;
    state->decodeScheduler_ = impl_->decodeScheduler;
    state->schedulerSelectionGate_ = impl_->schedulerSelectionGate;
    state->activeViewId_ = impl_->activeViewId;
    state->nextTaskId_ = impl_->nextTaskId;
    if (state->options_.persistentCacheBudgetRoot &&
        state->options_.persistentCachePath) {
        state->persistentBudget_ = PersistentZarrCacheBudget::findForPath(
            *state->options_.persistentCachePath);
    }
    if (state->options_.persistentCachePath && !state->persistentBudget_)
        ChunkCache::startPersistentCacheSizeScan(state);
    if (state->options_.persistentCachePath)
        ChunkCache::publishMirrorMetadata(state);
    ChunkCache::registerStateBudget(state);
    impl_->sources.emplace(state->sourceIdentity_, state);
    return std::shared_ptr<ChunkCache>(
        new ChunkCache(std::move(service), std::move(state)));
}

void ChunkCacheService::configureFetchConcurrency(
    std::size_t maxConcurrentReads, bool adaptive)
{
    std::optional<ChunkRequestScheduler::AdaptiveConcurrency> adaptiveOptions;
    if (adaptive) {
        adaptiveOptions.emplace();
        adaptiveOptions->minimum = std::min<std::size_t>(
            2, maxConcurrentReads);
        adaptiveOptions->maximum = maxConcurrentReads;
    }
    std::lock_guard lock(impl_->mutex);
    impl_->fetchScheduler->configureConcurrency(
        maxConcurrentReads, adaptiveOptions);
    impl_->activeFetchWorkers = maxConcurrentReads;
    impl_->activeFetchAdaptive = adaptive;
}

void ChunkCacheService::configureDecodedByteCapacity(
    std::size_t decodedByteCapacity)
{
    impl_->decodedByteBudget->setMaximumBytes(decodedByteCapacity);
}

ChunkCacheService::FetchConcurrency ChunkCacheService::fetchConcurrency() const
{
    std::lock_guard lock(impl_->mutex);
    return {impl_->fetchScheduler->workerCapacity(),
            impl_->activeFetchWorkers,
            impl_->activeFetchAdaptive};
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
    : ChunkCache(std::move(levels), std::move(fetchers), fillValue, dtype,
                 std::move(options), ChunkCacheService::Options{})
{
}

ChunkCache::ChunkCache(std::vector<LevelInfo> levels,
                       std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
                       double fillValue,
                       ChunkDtype dtype,
                       Options options,
                       ChunkCacheService::Options serviceOptions)
{
    if (!serviceOptions.decodedByteBudget)
        serviceOptions.decodedByteBudget = decodedByteBudgetDefault();
    service_ = std::make_shared<ChunkCacheService>(std::move(serviceOptions));
    auto handle = service_->acquireSource(
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
    if (!state || !state->decodedByteBudget_)
        return;
    std::weak_ptr<State> weakState = state;
    state->decodedBudgetRegistration_ =
        state->decodedByteBudget_->registerCache({
            [weakState]() -> std::optional<std::uint64_t> {
                auto locked = weakState.lock();
                return locked ? oldestDecodedTouch(locked) : std::nullopt;
            },
            [weakState]() -> std::size_t {
                auto locked = weakState.lock();
                return locked ? evictOldestDecoded(locked) : 0;
            },
            state->options_.decodedEvictionPreferSelf,
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
        if (fetchers[level] && state.options_.persistentCachePath) {
            const auto object = fetchers[level]->storageObject(
                ChunkKey{static_cast<int>(level), 0, 0, 0});
            if (!object || !isSafeZarrStoreKey(object->sourceKey)) {
                throw std::invalid_argument(
                    "ChunkCache refreshed fetcher has no safe physical Zarr object");
            }
        }
    }
}

void ChunkCache::refreshFetchers(
    const std::shared_ptr<State>& state,
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers)
{
    if (!state)
        throw std::invalid_argument("ChunkCache fetcher refresh requires state");
    validateRefreshedFetchers(*state, fetchers);

    std::vector<ChunkKey> stoppedRemoteActivity;
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
        for (auto& [key, transfer] : state->sourceTransfers_) {
            if (auto entry = state->entries_.find(key);
                entry != state->entries_.end() &&
                entry->second.fetchTaskId == transfer.taskId) {
                entry->second.fetchTaskId = 0;
            }
        }
        state->sourceTransfers_.clear();
        for (const auto& [objectKey, transfer] :
             state->storageObjectTransfers_) {
            (void)objectKey;
            if (transfer.sourceStarted && state->remoteFetchesInFlight_ > 0)
                --state->remoteFetchesInFlight_;
            for (const auto& key : transfer.notifiedConsumers) {
                auto active = state->activeRemoteFetches_.find(key);
                if (active == state->activeRemoteFetches_.end() ||
                    active->second.erase(transfer.serial) == 0) {
                    continue;
                }
                if (active->second.empty()) {
                    state->activeRemoteFetches_.erase(active);
                    stoppedRemoteActivity.push_back(key);
                }
            }
        }
        state->storageObjectTransfers_.clear();
        restartUnresolvedLocked(state);
        auto requeueOperation =
            [&](const ChunkKey& key,
                const std::shared_ptr<PersistenceOperation>& operation) {
            operation->probeTaskId = 0;
            operation->sourceTaskId = 0;
            FetchContext context{
                state->generation_,
                state->fetcherGeneration_,
                0,
                state->schedulerEpoch_,
                state->fetchers_.at(static_cast<std::size_t>(key.level)),
                {},
            };
            (void)joinStorageObjectTransferLocked(
                state, key, std::move(context), false, operation);
        };
        for (auto it = state->persistenceOperations_.begin();
             it != state->persistenceOperations_.end();) {
            if (it->second->writeQueued.load(std::memory_order_acquire)) {
                ++it;
                continue;
            }
            try {
                requeueOperation(it->first, it->second);
                ++it;
            } catch (...) {
                // A fetcher that throws during the requeue must not escape
                // before the cv notify below wakes parked blocking readers,
                // and this operation's own waiters must fail rather than
                // hang on a request nothing will run.
                auto operation = it->second;
                {
                    std::lock_guard operationLock(operation->mutex);
                    if (!operation->completed) {
                        operation->result = {
                            PersistentRequestStatus::Error,
                            "persistent request restart failed"};
                        operation->completed = true;
                    }
                }
                operation->cv.notify_all();
                Logger()->warn(
                    "[ChunkCache] persistent request restart failed for "
                    "({}, {}, {}, {}); request failed",
                    it->first.level, it->first.iz, it->first.iy,
                    it->first.ix);
                it = state->persistenceOperations_.erase(it);
            }
        }
    });
    for (const auto& key : stoppedRemoteActivity)
        notifyRemoteFetchListeners(state, key, false,
                                   /*isolateCallbackExceptions=*/true);
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
        try {
            (void)queueFetchLocked(state, key, state->generation_, 0);
        } catch (...) {
            // A fetcher that throws during queue setup must not orphan an
            // InFlight entry (parked blocking readers would wait forever;
            // the caller's cv notify wakes them to retry) or starve the
            // remaining keys' restarts.
            eraseUnresolvedEntryLocked(*state, key);
            Logger()->warn(
                "[ChunkCache] fetch restart failed for ({}, {}, {}, {}); "
                "entry dropped for refetch on demand",
                key.level, key.iz, key.iy, key.ix);
        }
    }
}

namespace {
std::mutex g_decodedBudgetDefaultMutex;
std::weak_ptr<DecodedChunkCacheBudget> g_decodedBudgetDefault;
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

void ChunkCache::validatePersistentMirror(
    const Options& options,
    const std::vector<std::shared_ptr<IChunkFetcher>>& fetchers)
{
    if (!options.persistentCachePath)
        return;

    const auto& root = *options.persistentCachePath;
    for (std::size_t level = 0; level < fetchers.size(); ++level) {
        if (!fetchers[level])
            continue;
        const ChunkKey key{static_cast<int>(level), 0, 0, 0};
        const auto object = fetchers[level]->storageObject(key);
        if (!object) {
            throw std::runtime_error(
                "persistent cache requires physical Zarr storage-object support");
        }
        if (!isSafeZarrStoreKey(object->sourceKey)) {
            throw std::runtime_error(
                "unsafe Zarr storage-object key: " + object->sourceKey);
        }
    }
    if (options.zarrMirrorMetadata.empty()) {
        throw std::runtime_error(
            "persistent cache requires authoritative source Zarr metadata");
    }
}

void ChunkCache::publishMirrorMetadata(const std::shared_ptr<State>& state)
{
    if (!state || !state->options_.persistentCachePath) {
        return;
    }
    for (const auto& object : state->options_.zarrMirrorMetadata) {
        if (!isSafeZarrStoreKey(object.key)) {
            throw std::runtime_error(
                "unsafe Zarr metadata cache key: " + object.key);
        }
        const auto path = *state->options_.persistentCachePath /
                          std::filesystem::path(object.key);
        if (auto existing = readFileBytes(path); existing && *existing == object.bytes)
            continue;
        auto reservation = state->persistentBudget_
            ? state->persistentBudget_->reserveProtectedWrite(
                  path, object.bytes.size())
            : PersistentZarrCacheBudget::WriteReservation{};
        if (state->persistentBudget_ && !reservation) {
            throw std::runtime_error(
                "insufficient disk budget for Zarr mirror metadata: " +
                path.string());
        }
        if (!atomicWriteBytes(path, object.bytes)) {
            reservation.commit();
            throw std::runtime_error(
                "failed to publish Zarr mirror metadata: " + path.string());
        }
        reservation.commit();
    }
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
    const bool notifyRemoteStart = queueFetchLocked(
        state, key, state->generation_, 0);
    lock.unlock();
    if (notifyRemoteStart)
        notifyRemoteFetchListeners(state, key, true);
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

    // A caller of this method is parked until the chunk resolves, so its
    // request must order ahead of the speculative prefetch backlog in the
    // background queue (smaller backgroundPriority runs first; prefetches
    // sit at >= 0). Without this, a blocking read issued while a solve
    // streams chunks waits FIFO behind every chunk the solve enqueued -
    // on remote sources that parked the caller (GUI thread included) for
    // the whole backlog's round trips.
    constexpr int kBlockingFetchPriorityOffset = -1024;
    constexpr int kLostEntryRetryLimit = 5;
    // Register this reader on the key before any entry work. The count -
    // not the entry - carries the eviction protection: both evictors and
    // the budget's victim probe skip keys with parked blocking readers
    // (hasParkedBlockingReaderLocked), so a resolved chunk cannot be erased
    // before its reader wakes, and a successor entry created after an
    // invalidation is protected from the moment it exists no matter which
    // thread created it. Being reader-owned, the registration survives any
    // entry lifecycle event and is released exactly once on every exit.
    // Only deliberate invalidation (and demand withdrawal of a successor
    // this reader never observed) can still remove the entry; the chunk
    // stays perfectly refetchable then, so the loop retries instead of
    // failing.
    ++state->blockingReadersByKey_[key];
    bool readerReleased = false;
    auto releaseReaderLocked = [&] {
        if (readerReleased)
            return;
        readerReleased = true;
        auto parked = state->blockingReadersByKey_.find(key);
        if (parked != state->blockingReadersByKey_.end() &&
            --parked->second == 0)
            state->blockingReadersByKey_.erase(parked);
    };
    // Exits release the reader registration, then re-run both limits: while
    // the registration protected the entry, enforcement may have given up
    // over budget. The entry-count capacity runs under the lock, the shared
    // byte budget outside it (it calls back into evictOldestDecoded, which
    // takes the lock).
    auto finishLocked = [&](ChunkResult result) {
        releaseReaderLocked();
        enforceCapacityLocked(state);
        lock.unlock();
        enforceSharedBudget(state);
        return result;
    };
    try {
        for (int attempt = 0;; ++attempt) {
            auto [it, inserted] = state->entries_.emplace(key, Entry{});
            it->second.backgroundDemand = true;
            bool notifyRemoteStart = false;
            if (inserted) {
                try {
                    notifyRemoteStart = queueFetchLocked(
                        state, key, state->generation_,
                        kBlockingFetchPriorityOffset);
                } catch (...) {
                    // Queue setup failed before any task was installed:
                    // erase the entry this reader created, or later readers
                    // would wait forever on an InFlight entry that nothing
                    // resolves.
                    eraseUnresolvedEntryLocked(*state, key);
                    throw;
                }
            } else if (it->second.status == EntryStatus::InFlight) {
                const int boosted =
                    fetchBasePriority(*state, key, kBlockingFetchPriorityOffset);
                if (boosted < it->second.basePriority) {
                    it->second.basePriority = boosted;
                    reprioritizeEntryLocked(*state, key, it->second);
                }
            }
            if (notifyRemoteStart) {
                lock.unlock();
                notifyRemoteFetchListeners(state, key, true);
                lock.lock();
            }
            // Wait keyed by chunk key. Marking demand on whatever entry sits
            // at the key (a mutex-protected side effect of the predicate)
            // keeps demand pruning from erasing a successor this reader has
            // observed; one it never observed can only vanish while the
            // reader's own wake is still pending, which lands in the retry
            // below.
            state->cv_.wait(lock, [&] {
                auto entry = state->entries_.find(key);
                if (entry == state->entries_.end())
                    return true;
                entry->second.backgroundDemand = true;
                return entry->second.status != EntryStatus::InFlight;
            });
            it = state->entries_.find(key);
            if (it != state->entries_.end())
                return finishLocked(
                    resultFromEntryLocked(*state, key, it->second));
            if (attempt >= kLostEntryRetryLimit)
                return finishLocked(ChunkResult{
                    ChunkStatus::Error, state->dtype_,
                    state->levels_[level].chunkShape, {},
                    "chunk entry removed before blocking reader woke "
                    "(retries exhausted)"});
            Logger()->warn(
                "[ChunkCache] blocking read lost chunk ({}, {}, {}, {}) "
                "mid-wait (invalidation, refresh, or demand withdrawal); "
                "refetching (attempt {}/{})",
                level, iz, iy, ix, attempt + 1, kLostEntryRetryLimit);
        }
    } catch (...) {
        if (!lock.owns_lock())
            lock.lock();
        releaseReaderLocked();
        try {
            enforceCapacityLocked(state);
            lock.unlock();
            enforceSharedBudget(state);
        } catch (...) {
            // Best-effort cleanup: keep the original exception.
        }
        throw;
    }
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
    std::vector<ChunkKey> remoteStarts;
    for (auto key : keys) {
        key = sourceKey(*state, key);
        if (!isValidKey(*state, key))
            continue;
        auto [it, inserted] = state->entries_.emplace(key, Entry{});
        if (inserted) {
            if (addRequestDemandLocked(*state, key, it->second, request)) {
                if (queueFetchLocked(
                        state, key, state->generation_, priorityOffset)) {
                    remoteStarts.push_back(key);
                }
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
    if (!remoteStarts.empty()) {
        lock.unlock();
        for (const auto& key : remoteStarts)
            notifyRemoteFetchListeners(state, key, true);
        if (!wait)
            return;
        lock.lock();
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
    const auto object = fetcher->storageObject(externalKey);
    if (!object) {
        result.valid = false;
        return result;
    }
    result.sourceChunkKey = object->sourceKey;
    result.persistentPath = mirrorObjectPath(*state, *object);
    result.persistentEmptyPath = mirrorEmptyPath(*state, *object);
    return result;
}

std::vector<ChunkKey> ChunkCache::persistedStorageObjectRepresentatives() const
{
    auto state = state_;
    std::vector<ChunkKey> result;
    if (!state->options_.persistentCachePath) {
        return result;
    }

    std::unordered_set<ChunkKey, ChunkKeyHash> seen;
    std::error_code ec;
    const auto& root = *state->options_.persistentCachePath;
    for (std::filesystem::recursive_directory_iterator it(
             root, std::filesystem::directory_options::skip_permission_denied,
             ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) {
            ec.clear();
            continue;
        }
        auto relative = std::filesystem::relative(it->path(), root, ec);
        if (ec) {
            ec.clear();
            continue;
        }
        std::string key = relative.generic_string();
        if (key.size() > 6 && key.ends_with(".empty"))
            key.resize(key.size() - 6);
        for (std::size_t level = 0; level < state->fetchers_.size(); ++level) {
            const auto& fetcher = state->fetchers_[level];
            if (!fetcher)
                continue;
            auto logical = fetcher->logicalRepresentativeForStorageKey(
                static_cast<int>(level), key);
            if (!logical)
                continue;
            *logical = sourceKey(*state, *logical);
            if (isValidKey(*state, *logical) && seen.insert(*logical).second)
                result.push_back(*logical);
            break;
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.level, lhs.iz, lhs.iy, lhs.ix) <
               std::tie(rhs.level, rhs.iz, rhs.iy, rhs.ix);
    });
    return result;
}

std::vector<ChunkKey> ChunkCache::storageObjectRepresentatives(int level) const
{
    auto state = state_;
    if (level < 0 || level >= static_cast<int>(state->levels_.size()))
        throw std::out_of_range("storage-object level out of range");
    const auto& fetcher = state->fetchers_.at(static_cast<std::size_t>(level));
    if (!fetcher)
        return {};
    const auto first = fetcher->storageObject(ChunkKey{level, 0, 0, 0});
    if (!first)
        return {};

    const auto& info = state->levels_.at(static_cast<std::size_t>(level));
    std::array<int, 3> logicalGrid{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        logicalGrid[axis] =
            (info.shape[axis] + info.chunkShape[axis] - 1) /
            info.chunkShape[axis];
    }
    const auto& factor = first->innerChunksPerObject;
    const std::array<int, 3> objectGrid{
        (logicalGrid[0] + factor[0] - 1) / factor[0],
        (logicalGrid[1] + factor[1] - 1) / factor[1],
        (logicalGrid[2] + factor[2] - 1) / factor[2]};
    std::vector<ChunkKey> result;
    result.reserve(static_cast<std::size_t>(objectGrid[0]) *
                   static_cast<std::size_t>(objectGrid[1]) *
                   static_cast<std::size_t>(objectGrid[2]));
    for (int oz = 0; oz < objectGrid[0]; ++oz) {
        for (int oy = 0; oy < objectGrid[1]; ++oy) {
            for (int ox = 0; ox < objectGrid[2]; ++ox) {
                result.push_back(sourceKey(
                    *state, ChunkKey{level, oz * factor[0],
                                     oy * factor[1], ox * factor[2]}));
            }
        }
    }
    return result;
}

ChunkCache::Stats ChunkCache::stats() const
{
    auto state = state_;
    Stats result;
    {
        std::lock_guard lock(state->mutex_);
        const auto budget = state->decodedByteBudget_->stats();
        result.decodedBytes = budget.decodedBytes;
        result.decodedByteCapacity = budget.maximumBytes;
        result.remoteFetchesInFlight = state->remoteFetchesInFlight_;
        if (auto scheduler = state->fetchScheduler_.lock()) {
            result.remoteDownloadBytesPerSecond =
                scheduler->transferStats().bytesPerSecond;
        }
        if (auto scheduler = state->decodeScheduler_.lock())
            result.pendingDecodeTasks = scheduler->pending();
        result.unresolvedFetchesByLevel = state->unresolvedFetchesByLevel_;
        for (const auto& [key, parked] : state->blockingReadersByKey_) {
            (void)key;
            result.blockingReaders += parked;
        }
        result.persistentCacheEnabled = state->options_.persistentCachePath.has_value();
        result.persistentCacheWarning = state->persistentCacheWarning_;
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
    std::vector<ChunkKey> remoteStarts;
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
            if (inserted) {
                if (queueFetchLocked(state, key, state->generation_, 0))
                    remoteStarts.push_back(key);
            } else {
                reprioritizeEntryLocked(*state, key, entryIt->second);
            }
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
    for (const auto& key : remoteStarts)
        notifyRemoteFetchListeners(state, key, true);
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
    std::vector<std::shared_ptr<PersistenceOperation>> cancelledPersistence;
    {
        std::lock_guard lock(state->mutex_);
        ++state->generation_;
        schedulerEpoch = ++state->schedulerEpoch_;
        state->entries_.clear();
        state->lru_.clear();
        state->viewSnapshots_.clear();
        state->viewDemandKeys_.clear();
        state->sourceTransfers_.clear();
        state->storageObjectTransfers_.clear();
        for (const auto& [key, operation] : state->persistenceOperations_) {
            (void)key;
            if (!operation->writeQueued.load(std::memory_order_acquire))
                cancelledPersistence.push_back(operation);
        }
        state->persistenceOperations_.clear();
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
    // Per-listener exception isolation: a throwing listener must not skip
    // other listeners' stop events, the remaining keys, or the cv notify
    // below - parked blocking readers whose entries were just cleared
    // depend on that wake.
    for (const auto& key : activeRemoteFetches)
        notifyRemoteFetchListeners(state, key, false,
                                   /*isolateCallbackExceptions=*/true);
    for (const auto& operation : cancelledPersistence) {
        {
            std::lock_guard lock(operation->mutex);
            if (operation->completed)
                continue;
            operation->result = {
                PersistentRequestStatus::Error,
                "persistent request was invalidated"};
            operation->completed = true;
        }
        operation->cv.notify_all();
    }
    state->cv_.notify_all();
}

void ChunkCache::unregisterStateBudget(State& state)
{
    if (state.decodedByteBudget_ &&
        state.decodedBudgetRegistration_ != 0) {
        state.decodedByteBudget_->unregisterCache(
            state.decodedBudgetRegistration_);
        state.decodedBudgetRegistration_ = 0;
    }
}

ChunkCache::PersistentRequestResult ChunkCache::persistChunkBlocking(
    int level,
    int iz,
    int iy,
    int ix,
    PersistentRequestMode mode)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix, state->sourceId_};
    std::shared_ptr<PersistenceOperation> operation;
    bool created = false;
    {
        std::lock_guard lock(state->mutex_);
        if (!state->options_.persistentCachePath) {
            return {PersistentRequestStatus::Error,
                    "persistent cache is not configured"};
        }
        if (!isValidKey(*state, key))
            return {PersistentRequestStatus::Missing, {}};
        const auto& fetcher = state->fetchers_.at(
            static_cast<std::size_t>(level));
        if (!fetcher || !fetcher->storageObject(fetcherKey(key))) {
            return {PersistentRequestStatus::Error,
                    "source fetcher has no physical Zarr storage object"};
        }

        auto existing = state->persistenceOperations_.find(key);
        if (existing != state->persistenceOperations_.end()) {
            operation = existing->second;
            if (mode == PersistentRequestMode::Refresh) {
                operation->refresh = true;
                if (const auto object = fetcher->storageObject(
                        fetcherKey(key))) {
                    const StorageObjectKey objectKey{
                        key.level, object->outerZ,
                        object->outerY, object->outerX};
                    if (auto transfer =
                            state->storageObjectTransfers_.find(objectKey);
                        transfer != state->storageObjectTransfers_.end()) {
                        transfer->second.refreshRequested = true;
                        reprioritizeStorageObjectTransferLocked(
                            *state, transfer->second);
                    }
                }
            }
        } else {
            operation = std::make_shared<PersistenceOperation>();
            operation->refresh = mode == PersistentRequestMode::Refresh;
            state->persistenceOperations_.emplace(key, operation);
            created = true;
        }
    }

    if (created) {
        state->schedulerSelectionGate_->publish([&] {
            std::lock_guard lock(state->mutex_);
            const auto current = state->persistenceOperations_.find(key);
            if (current == state->persistenceOperations_.end() ||
                current->second != operation) {
                return;
            }

            FetchContext context{
                state->generation_,
                state->fetcherGeneration_,
                0,
                state->schedulerEpoch_,
                state->fetchers_.at(static_cast<std::size_t>(level)),
                {},
            };
            (void)joinStorageObjectTransferLocked(
                state, key, std::move(context), false, operation);
        });
    }

    std::unique_lock operationLock(operation->mutex);
    operation->cv.wait(operationLock, [&] { return operation->completed; });
    return operation->result;
}

void ChunkCache::completePersistenceOperation(
    const std::shared_ptr<State>& state,
    const ChunkKey& key,
    const std::shared_ptr<PersistenceOperation>& operation,
    PersistentRequestResult result)
{
    {
        std::lock_guard lock(state->mutex_);
        {
            std::lock_guard operationLock(operation->mutex);
            if (operation->completed)
                return;
            operation->result = std::move(result);
            operation->completed = true;
        }
        const auto current = state->persistenceOperations_.find(key);
        if (current != state->persistenceOperations_.end() &&
            current->second == operation) {
            state->persistenceOperations_.erase(current);
        }
    }
    operation->cv.notify_all();
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
    bool sharedObjectTask = false;
    if (state.options_.persistentCachePath &&
        (entry.probeTaskId != 0 || entry.fetchTaskId != 0 ||
         entry.decodeTaskId != 0)) {
        const auto& fetcher = state.fetchers_.at(
            static_cast<std::size_t>(key.level));
        if (fetcher) {
            if (const auto object = fetcher->storageObject(fetcherKey(key))) {
                const StorageObjectKey objectKey{
                    key.level, object->outerZ, object->outerY, object->outerX};
                if (auto transfer = state.storageObjectTransfers_.find(objectKey);
                    transfer != state.storageObjectTransfers_.end()) {
                    sharedObjectTask =
                        transfer->second.taskId == entry.probeTaskId ||
                        transfer->second.taskId == entry.fetchTaskId ||
                        transfer->second.taskId == entry.decodeTaskId;
                    if (sharedObjectTask) {
                        reprioritizeStorageObjectTransferLocked(
                            state, transfer->second);
                    }
                }
            }
        }
    }
    if (!sharedObjectTask && entry.probeTaskId != 0) {
        if (auto scheduler = state.probeScheduler_.lock())
            scheduler->reprioritize(entry.probeTaskId, priority);
    }
    if (!sharedObjectTask && entry.fetchTaskId != 0) {
        if (auto scheduler = state.fetchScheduler_.lock())
            scheduler->reprioritize(entry.fetchTaskId, priority);
    }
    if (!sharedObjectTask && entry.decodeTaskId != 0) {
        if (auto scheduler = state.decodeScheduler_.lock())
            scheduler->reprioritize(entry.decodeTaskId, priority);
    }
}

ChunkWorkPriority ChunkCache::storageObjectPriorityLocked(
    const State& state,
    const StorageObjectTransfer& transfer)
{
    ChunkWorkPriority best;
    best.maintenance = transfer.consumers.empty();
    bool have = false;
    for (const auto& [consumerKey, consumer] : transfer.consumers) {
        (void)consumer;
        const auto entry = state.entries_.find(consumerKey);
        if (entry == state.entries_.end() || !hasDemandLocked(entry->second))
            continue;
        const auto candidate = workPriorityLocked(
            state, consumerKey, entry->second);
        const auto candidateDistance = std::isfinite(candidate.distanceSquared)
            ? candidate.distanceSquared
            : std::numeric_limits<float>::infinity();
        const auto bestDistance = std::isfinite(best.distanceSquared)
            ? best.distanceSquared
            : std::numeric_limits<float>::infinity();
        const bool better = !have ||
            (candidate.interactive && !best.interactive) ||
            (candidate.interactive == best.interactive &&
             candidate.levelPriority > best.levelPriority) ||
            (candidate.interactive == best.interactive &&
             candidate.levelPriority == best.levelPriority &&
             candidate.activeView && !best.activeView) ||
            (candidate.interactive == best.interactive &&
             candidate.levelPriority == best.levelPriority &&
             candidate.activeView == best.activeView &&
             candidateDistance < bestDistance) ||
            (!candidate.interactive && !best.interactive &&
             candidate.backgroundPriority < best.backgroundPriority);
        if (better) {
            best = candidate;
            have = true;
        }
    }
    best.maintenance = !have;
    return best;
}

void ChunkCache::reprioritizeStorageObjectTransferLocked(
    const State& state,
    const StorageObjectTransfer& transfer)
{
    std::shared_ptr<ChunkRequestScheduler> scheduler;
    switch (transfer.stage) {
    case StorageObjectTransfer::Stage::Probe:
        scheduler = state.probeScheduler_.lock();
        break;
    case StorageObjectTransfer::Stage::PersistentRead:
        scheduler = state.decodeScheduler_.lock();
        break;
    case StorageObjectTransfer::Stage::Source:
        scheduler = state.fetchScheduler_.lock();
        break;
    }
    if (scheduler)
        scheduler->reprioritize(
            transfer.taskId, storageObjectPriorityLocked(state, transfer));
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

bool ChunkCache::hasParkedBlockingReaderLocked(const State& state,
                                               const ChunkKey& key)
{
    if (state.blockingReadersByKey_.empty())
        return false;
    return state.blockingReadersByKey_.find(key) !=
           state.blockingReadersByKey_.end();
}

bool ChunkCache::cancelUndemandedEntryLocked(State& state,
                                             const ChunkKey& key,
                                             Entry& entry)
{
    if (entry.status != EntryStatus::InFlight || hasDemandLocked(entry))
        return false;

    if (state.options_.persistentCachePath) {
        const auto& fetcher = state.fetchers_.at(
            static_cast<std::size_t>(key.level));
        if (fetcher) {
            if (const auto object = fetcher->storageObject(fetcherKey(key))) {
                const StorageObjectKey objectKey{
                    key.level, object->outerZ, object->outerY, object->outerX};
                auto transfer = state.storageObjectTransfers_.find(objectKey);
                if (transfer != state.storageObjectTransfers_.end()) {
                    if (!transfer->second.notifiedConsumers.contains(key))
                        transfer->second.consumers.erase(key);
                    entry.probeTaskId = 0;
                    entry.fetchTaskId = 0;
                    entry.decodeTaskId = 0;
                    bool hasPersistence = false;
                    for (auto it = transfer->second.persistence.begin();
                         it != transfer->second.persistence.end();) {
                        if (it->second.expired())
                            it = transfer->second.persistence.erase(it);
                        else {
                            hasPersistence = true;
                            ++it;
                        }
                    }
                    if (transfer->second.consumers.empty() && !hasPersistence) {
                        std::shared_ptr<ChunkRequestScheduler> scheduler;
                        switch (transfer->second.stage) {
                        case StorageObjectTransfer::Stage::Probe:
                            scheduler = state.probeScheduler_.lock();
                            break;
                        case StorageObjectTransfer::Stage::PersistentRead:
                            scheduler = state.decodeScheduler_.lock();
                            break;
                        case StorageObjectTransfer::Stage::Source:
                            scheduler = state.fetchScheduler_.lock();
                            break;
                        }
                        if (scheduler && scheduler->cancel(transfer->second.taskId))
                            state.storageObjectTransfers_.erase(transfer);
                    } else {
                        reprioritizeStorageObjectTransferLocked(
                            state, transfer->second);
                    }
                    return true;
                }
            }
        }
    }

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
    const auto sourceTaskId = entry.fetchTaskId;
    cancel(entry.fetchTaskId, state.fetchScheduler_);
    if (sourceTaskId != 0 && entry.fetchTaskId == 0) {
        auto transfer = state.sourceTransfers_.find(key);
        if (transfer != state.sourceTransfers_.end() &&
            transfer->second.taskId == sourceTaskId) {
            state.sourceTransfers_.erase(transfer);
        }
    }
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

bool ChunkCache::queueFetchLocked(const std::shared_ptr<State>& state,
                                  const ChunkKey& key,
                                  std::uint64_t generation,
                                  int priorityOffset)
{
    auto it = state->entries_.find(key);
    if (it == state->entries_.end())
        return false;
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
        return false;
    std::weak_ptr<State> weakState = state;
    entry.probeTaskId = 0;
    entry.fetchTaskId = 0;
    entry.decodeTaskId = 0;
    if (state->options_.persistentCachePath) {
        return joinStorageObjectTransferLocked(
            state, key, context, true, {});
    } else {
        joinSourceTransferLocked(state, key, context, true);
    }
    return false;
}

void ChunkCache::queueRemoteFetch(const std::shared_ptr<State>& state,
                                  const ChunkKey& key,
                                  FetchContext context)
{
    bool pruned = false;
    bool remoteStart = false;
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
        if (state->options_.persistentCachePath) {
            remoteStart = joinStorageObjectTransferLocked(
                state, key, context, true, {});
        } else {
            joinSourceTransferLocked(state, key, context, true);
        }
    });
    if (remoteStart)
        notifyRemoteFetchListeners(state, key, true,
                                   /*isolateCallbackExceptions=*/true);
    if (pruned)
        state->cv_.notify_all();
}

void ChunkCache::joinSourceTransferLocked(
    const std::shared_ptr<State>& state,
    const ChunkKey& key,
    FetchContext context,
    bool decodeRequested)
{
    auto scheduler = state->fetchScheduler_.lock();
    if (!scheduler)
        return;

    auto found = state->sourceTransfers_.find(key);
    if (found != state->sourceTransfers_.end()) {
        SourceTransfer& transfer = found->second;
        if (decodeRequested) {
            transfer.decodeRequested = true;
            if (auto entry = state->entries_.find(key);
                entry != state->entries_.end()) {
                entry->second.fetchSerial = transfer.serial;
                entry->second.fetchTaskId = transfer.taskId;
            }
        }
        ChunkWorkPriority priority;
        if (transfer.decodeRequested) {
            const auto entry = state->entries_.find(key);
            if (entry != state->entries_.end())
                priority = workPriorityLocked(*state, key, entry->second);
        }
        scheduler->reprioritize(transfer.taskId, priority);
        return;
    }

    SourceTransfer transfer;
    transfer.serial = state->nextSourceTransferSerial_++;
    transfer.taskId = state->nextTaskId_->fetch_add(
        1, std::memory_order_relaxed);
    transfer.generation = context.generation;
    transfer.fetcherGeneration = context.fetcherGeneration;
    transfer.schedulerEpoch = context.schedulerEpoch;
    transfer.fetcher = std::move(context.fetcher);
    transfer.decodeRequested = decodeRequested;

    ChunkWorkPriority priority;
    if (decodeRequested) {
        if (auto entry = state->entries_.find(key);
            entry != state->entries_.end()) {
            entry->second.fetchSerial = transfer.serial;
            entry->second.fetchTaskId = transfer.taskId;
            priority = workPriorityLocked(*state, key, entry->second);
        }
    }

    const auto serial = transfer.serial;
    const auto taskId = transfer.taskId;
    state->sourceTransfers_.emplace(key, std::move(transfer));
    std::weak_ptr<State> weakState = state;
    scheduler->submit(
        taskId, priority, state->schedulerGroup_, context.schedulerEpoch,
        [weakState, key, serial] {
            if (auto state = weakState.lock())
                runSourceTransfer(state, key, serial);
        });
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

void ChunkCache::runSourceTransfer(const std::shared_ptr<State>& state,
                                   ChunkKey key,
                                   std::uint64_t transferSerial)
{
    SourceTransfer sourceTransfer;
    {
        std::lock_guard lock(state->mutex_);
        auto transfer = state->sourceTransfers_.find(key);
        if (transfer == state->sourceTransfers_.end() ||
            transfer->second.serial != transferSerial ||
            transfer->second.generation != state->generation_ ||
            transfer->second.fetcherGeneration != state->fetcherGeneration_) {
            return;
        }
        sourceTransfer = transfer->second;
    }

    ChunkFetchResult fetch;
    bool trackedRemoteFetch = false;
    const auto fetchStarted = std::chrono::steady_clock::now();
    std::optional<ChunkRequestScheduler::TransferMeasurement> transfer;
    auto scheduler = state->fetchScheduler_.lock();
    if (scheduler && sourceTransfer.fetcher->measuresRemoteTransfer())
        transfer.emplace(scheduler->beginTransfer(fetchStarted));
    auto observeProgress = [&](std::size_t bytes) {
        if (transfer)
            transfer->recordBytes(bytes);
    };
    try {
        if (sourceTransfer.fetcher->measuresRemoteTransfer()) {
            trackedRemoteFetch = true;
            {
                std::lock_guard lock(state->mutex_);
                ++state->remoteFetchesInFlight_;
                state->activeRemoteFetches_[key].insert(transferSerial);
            }
            notifyRemoteFetchListeners(state, key, true,
                                       /*isolateCallbackExceptions=*/true);
        }
        fetch = sourceTransfer.fetcher->fetchEncoded(
            fetcherKey(key), observeProgress);
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

    const auto fetchCompleted = std::chrono::steady_clock::now();
    if (transfer) {
        transfer->finish(
            fetch.status == ChunkFetchStatus::Found && !fetch.bytes.empty(),
            fetch.bytes.size(),
            fetchCompleted);
    }

    bool remoteActivityEnded = false;
    if (trackedRemoteFetch) {
        std::lock_guard lock(state->mutex_);
        auto active = state->activeRemoteFetches_.find(key);
        if (active != state->activeRemoteFetches_.end() &&
            active->second.erase(transferSerial) != 0) {
            if (state->remoteFetchesInFlight_ > 0)
                --state->remoteFetchesInFlight_;
            if (active->second.empty()) {
                state->activeRemoteFetches_.erase(active);
                remoteActivityEnded = true;
            }
        }
    }
    if (remoteActivityEnded)
        notifyRemoteFetchListeners(state, key, false,
                                   /*isolateCallbackExceptions=*/true);

    bool decodeRequested = false;
    {
        std::lock_guard lock(state->mutex_);
        auto current = state->sourceTransfers_.find(key);
        if (current == state->sourceTransfers_.end() ||
            current->second.serial != transferSerial) {
            return;
        }
        decodeRequested = current->second.decodeRequested;
        state->sourceTransfers_.erase(current);
        if (auto entry = state->entries_.find(key);
            entry != state->entries_.end() &&
            entry->second.fetchTaskId == sourceTransfer.taskId) {
            entry->second.fetchTaskId = 0;
        }
    }

    FetchContext context{
        sourceTransfer.generation,
        sourceTransfer.fetcherGeneration,
        transferSerial,
        sourceTransfer.schedulerEpoch,
        sourceTransfer.fetcher,
        scheduler,
    };

    if (fetch.status == ChunkFetchStatus::Found) {
        if (decodeRequested)
            queueFetchedDecode(state, key, context, std::move(fetch));
        return;
    }

    if (decodeRequested)
        finishAndStore(state, key, context, std::move(fetch));
}

bool ChunkCache::joinStorageObjectTransferLocked(
    const std::shared_ptr<State>& state,
    const ChunkKey& key,
    FetchContext context,
    bool decodeRequested,
    const std::shared_ptr<PersistenceOperation>& persistence)
{
    if (!context.fetcher)
        return false;
    const auto object = context.fetcher->storageObject(fetcherKey(key));
    if (!object || !isSafeZarrStoreKey(object->sourceKey)) {
        if (persistence) {
            state->persistenceOperations_.erase(key);
            {
                std::lock_guard operationLock(persistence->mutex);
                persistence->result = {
                    PersistentRequestStatus::Error,
                    "source fetcher has no valid physical Zarr object"};
                persistence->completed = true;
            }
            persistence->cv.notify_all();
        }
        return false;
    }
    const StorageObjectKey objectKey{
        key.level, object->outerZ, object->outerY, object->outerX};

    auto found = state->storageObjectTransfers_.find(objectKey);
    if (found != state->storageObjectTransfers_.end()) {
        auto& transfer = found->second;
        bool notifyRemoteStart = false;
        if (persistence && persistence->refresh)
            transfer.refreshRequested = true;
        if (decodeRequested) {
            transfer.consumers[key] = StorageConsumer{
                context.generation, context.fetcherGeneration,
                context.fetchSerial, context.schedulerEpoch, context.fetcher};
            if (auto entry = state->entries_.find(key);
                entry != state->entries_.end()) {
                if (transfer.stage == StorageObjectTransfer::Stage::Probe)
                    entry->second.probeTaskId = transfer.taskId;
                else if (transfer.stage == StorageObjectTransfer::Stage::PersistentRead)
                    entry->second.decodeTaskId = transfer.taskId;
                else
                    entry->second.fetchTaskId = transfer.taskId;
            }
            if (transfer.sourceStarted &&
                transfer.notifiedConsumers.insert(key).second) {
                state->activeRemoteFetches_[key].insert(transfer.serial);
                notifyRemoteStart = true;
            }
        }
        if (persistence) {
            transfer.persistence[key] = persistence;
            if (transfer.stage == StorageObjectTransfer::Stage::Source) {
                persistence->probeTaskId = 0;
                persistence->sourceTaskId = transfer.taskId;
            } else {
                persistence->probeTaskId = transfer.taskId;
                persistence->sourceTaskId = 0;
            }
        }
        reprioritizeStorageObjectTransferLocked(*state, transfer);
        return notifyRemoteStart;
    }

    StorageObjectTransfer transfer;
    transfer.serial = state->nextSourceTransferSerial_++;
    transfer.taskId = state->nextTaskId_->fetch_add(
        1, std::memory_order_relaxed);
    transfer.schedulerEpoch = context.schedulerEpoch;
    transfer.object = *object;
    transfer.fetcher = context.fetcher;
    transfer.refreshRequested = persistence && persistence->refresh;
    transfer.stage = persistence && persistence->refresh
        ? StorageObjectTransfer::Stage::Source
        : StorageObjectTransfer::Stage::Probe;
    if (decodeRequested) {
        transfer.consumers.emplace(
            key, StorageConsumer{context.generation, context.fetcherGeneration,
                                 context.fetchSerial, context.schedulerEpoch,
                                 context.fetcher});
        if (auto entry = state->entries_.find(key);
            entry != state->entries_.end()) {
            if (transfer.stage == StorageObjectTransfer::Stage::Probe)
                entry->second.probeTaskId = transfer.taskId;
            else if (transfer.stage == StorageObjectTransfer::Stage::PersistentRead)
                entry->second.decodeTaskId = transfer.taskId;
            else
                entry->second.fetchTaskId = transfer.taskId;
        }
    }
    if (persistence) {
        transfer.persistence.emplace(key, persistence);
        if (transfer.stage == StorageObjectTransfer::Stage::Source)
            persistence->sourceTaskId = transfer.taskId;
        else
            persistence->probeTaskId = transfer.taskId;
    }

    const auto serial = transfer.serial;
    const auto taskId = transfer.taskId;
    const auto stage = transfer.stage;
    const auto priority = storageObjectPriorityLocked(*state, transfer);
    state->storageObjectTransfers_.emplace(objectKey, std::move(transfer));
    auto scheduler = stage == StorageObjectTransfer::Stage::Probe
        ? state->probeScheduler_.lock()
        : state->fetchScheduler_.lock();
    if (!scheduler)
        return false;
    std::weak_ptr<State> weakState = state;
    scheduler->submit(
        taskId, priority, state->schedulerGroup_, context.schedulerEpoch,
        [weakState, objectKey, serial, stage] {
            if (auto state = weakState.lock()) {
                if (stage == StorageObjectTransfer::Stage::Probe)
                    runStorageObjectProbe(state, objectKey, serial);
                else
                    runStorageObjectFetch(state, objectKey, serial);
            }
        });
    return false;
}

void ChunkCache::runStorageObjectProbe(
    const std::shared_ptr<State>& state,
    StorageObjectKey objectKey,
    std::uint64_t transferSerial)
{
    ChunkStorageObject object;
    {
        std::lock_guard lock(state->mutex_);
        const auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial ||
            found->second.stage != StorageObjectTransfer::Stage::Probe) {
            return;
        }
        object = found->second.object;
    }

    const auto dataPath = mirrorObjectPath(*state, object);
    const auto emptyPath = mirrorEmptyPath(*state, object);
    std::error_code ec;
    const bool haveData = std::filesystem::is_regular_file(dataPath, ec) && !ec;
    ec.clear();
    const bool haveEmpty =
        std::filesystem::is_regular_file(emptyPath, ec) && !ec;

    bool dispatchMissing = false;

    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial ||
            found->second.stage != StorageObjectTransfer::Stage::Probe) {
            return;
        }
        auto& transfer = found->second;
        if (transfer.refreshRequested || (!haveData && !haveEmpty)) {
            queueStorageObjectSourceLocked(state, objectKey, transfer);
            return;
        }
        if (haveEmpty) {
            dispatchMissing = true;
            return;
        }

        transfer.stage = StorageObjectTransfer::Stage::PersistentRead;
        transfer.taskId = state->nextTaskId_->fetch_add(
            1, std::memory_order_relaxed);
        for (const auto& [key, consumer] : transfer.consumers) {
            (void)consumer;
            if (auto entry = state->entries_.find(key);
                entry != state->entries_.end()) {
                entry->second.probeTaskId = 0;
                entry->second.decodeTaskId = transfer.taskId;
            }
        }
        for (auto& [key, weakOperation] : transfer.persistence) {
            (void)key;
            if (auto operation = weakOperation.lock()) {
                operation->probeTaskId = transfer.taskId;
                operation->sourceTaskId = 0;
            }
        }
        auto scheduler = state->decodeScheduler_.lock();
        if (!scheduler)
            return;
        const auto taskId = transfer.taskId;
        const auto epoch = transfer.schedulerEpoch;
        const auto priority = storageObjectPriorityLocked(*state, transfer);
        std::weak_ptr<State> weakState = state;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, epoch,
            [weakState, objectKey, transferSerial] {
                if (auto state = weakState.lock())
                    runStorageObjectRead(state, objectKey, transferSerial);
            });
    });

    if (dispatchMissing) {
        ChunkFetchResult cached;
        cached.status = ChunkFetchStatus::Missing;
        dispatchStorageObjectResult(
            state, objectKey, transferSerial, std::move(cached), true);
    }
}

void ChunkCache::runStorageObjectRead(
    const std::shared_ptr<State>& state,
    StorageObjectKey objectKey,
    std::uint64_t transferSerial)
{
    ChunkStorageObject object;
    {
        std::lock_guard lock(state->mutex_);
        const auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial ||
            found->second.stage != StorageObjectTransfer::Stage::PersistentRead) {
            return;
        }
        object = found->second.object;
    }

    const auto dataPath = mirrorObjectPath(*state, object);
    auto pin = state->persistentBudget_
        ? state->persistentBudget_->pinRead(dataPath)
        : PersistentZarrCacheBudget::ReadPin{};
    auto bytes = readFileBytes(dataPath);
    pin.complete(bytes.has_value());

    bool useCached = false;
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial ||
            found->second.stage != StorageObjectTransfer::Stage::PersistentRead) {
            return;
        }
        if (!bytes || found->second.refreshRequested) {
            queueStorageObjectSourceLocked(state, objectKey, found->second);
            return;
        }
        useCached = true;
    });

    if (useCached) {
        ChunkFetchResult cached;
        cached.status = ChunkFetchStatus::Found;
        cached.bytes = std::move(*bytes);
        dispatchStorageObjectResult(
            state, objectKey, transferSerial, std::move(cached), true);
    }
}

void ChunkCache::queueStorageObjectSourceLocked(
    const std::shared_ptr<State>& state,
    StorageObjectKey objectKey,
    StorageObjectTransfer& transfer)
{
    transfer.stage = StorageObjectTransfer::Stage::Source;
    transfer.taskId = state->nextTaskId_->fetch_add(
        1, std::memory_order_relaxed);
    for (const auto& [key, consumer] : transfer.consumers) {
        (void)consumer;
        if (auto entry = state->entries_.find(key);
            entry != state->entries_.end()) {
            entry->second.probeTaskId = 0;
            entry->second.decodeTaskId = 0;
            entry->second.fetchTaskId = transfer.taskId;
        }
    }
    for (auto& [key, weakOperation] : transfer.persistence) {
        (void)key;
        if (auto operation = weakOperation.lock()) {
            operation->probeTaskId = 0;
            operation->sourceTaskId = transfer.taskId;
        }
    }
    auto scheduler = state->fetchScheduler_.lock();
    if (!scheduler)
        return;
    const auto taskId = transfer.taskId;
    const auto epoch = transfer.schedulerEpoch;
    const auto serial = transfer.serial;
    const auto priority = storageObjectPriorityLocked(*state, transfer);
    std::weak_ptr<State> weakState = state;
    scheduler->submit(
        taskId, priority, state->schedulerGroup_, epoch,
        [weakState, objectKey, serial] {
            if (auto state = weakState.lock())
                runStorageObjectFetch(state, objectKey, serial);
        });
}

void ChunkCache::runStorageObjectFetch(
    const std::shared_ptr<State>& state,
    StorageObjectKey objectKey,
    std::uint64_t transferSerial)
{
    StorageObjectTransfer snapshot;
    std::vector<ChunkKey> activityKeys;
    {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial ||
            found->second.stage != StorageObjectTransfer::Stage::Source) {
            return;
        }
        snapshot = found->second;
        found->second.sourceStarted = true;
        ++state->remoteFetchesInFlight_;
        for (const auto& [key, consumer] : found->second.consumers) {
            (void)consumer;
            found->second.notifiedConsumers.insert(key);
            state->activeRemoteFetches_[key].insert(transferSerial);
            activityKeys.push_back(key);
        }
    }
    for (const auto& key : activityKeys)
        notifyRemoteFetchListeners(state, key, true,
                                   /*isolateCallbackExceptions=*/true);

    ChunkFetchResult fetch;
    const auto started = std::chrono::steady_clock::now();
    auto scheduler = state->fetchScheduler_.lock();
    std::optional<ChunkRequestScheduler::TransferMeasurement> measurement;
    if (scheduler && snapshot.fetcher->measuresRemoteTransfer())
        measurement.emplace(scheduler->beginTransfer(started));
    auto progress = [&](std::size_t bytes) {
        if (measurement)
            measurement->recordBytes(bytes);
    };
    try {
        fetch = snapshot.fetcher->fetchStorageObject(snapshot.object, progress);
    } catch (const std::exception& error) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = error.what();
    } catch (...) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = "unknown storage-object fetch exception";
    }
    if (measurement) {
        measurement->finish(
            fetch.status == ChunkFetchStatus::Found && !fetch.bytes.empty(),
            fetch.bytes.size(), std::chrono::steady_clock::now());
    }

    std::vector<ChunkKey> stopped;
    {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found != state->storageObjectTransfers_.end() &&
            found->second.serial == transferSerial &&
            found->second.sourceStarted) {
            for (const auto& key : found->second.notifiedConsumers) {
                auto active = state->activeRemoteFetches_.find(key);
                if (active != state->activeRemoteFetches_.end() &&
                    active->second.erase(transferSerial) != 0) {
                    if (active->second.empty()) {
                        state->activeRemoteFetches_.erase(active);
                        stopped.push_back(key);
                    }
                }
            }
            found->second.sourceStarted = false;
            if (state->remoteFetchesInFlight_ > 0)
                --state->remoteFetchesInFlight_;
        }
    }
    for (const auto& key : stopped)
        notifyRemoteFetchListeners(state, key, false,
                                   /*isolateCallbackExceptions=*/true);

    dispatchStorageObjectResult(
        state, objectKey, transferSerial, std::move(fetch), false);
}

void ChunkCache::dispatchStorageObjectResult(
    const std::shared_ptr<State>& state,
    StorageObjectKey objectKey,
    std::uint64_t transferSerial,
    ChunkFetchResult fetch,
    bool loadedFromPersistentCache)
{
    ChunkStorageObject object;
    bool persistenceSucceeded = loadedFromPersistentCache;
    {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial) {
            return;
        }
        if (loadedFromPersistentCache && found->second.refreshRequested) {
            queueStorageObjectSourceLocked(state, objectKey, found->second);
            return;
        }
        object = found->second.object;
    }

    if (!loadedFromPersistentCache) {
        if (fetch.status == ChunkFetchStatus::Found) {
            persistenceSucceeded = writeMirrorObject(
                *state, object, fetch.bytes);
        } else if (fetch.status == ChunkFetchStatus::Missing) {
            persistenceSucceeded = writeMirrorEmpty(*state, object);
        }
    }

    StorageObjectTransfer transfer;
    {
        std::lock_guard lock(state->mutex_);
        auto found = state->storageObjectTransfers_.find(objectKey);
        if (found == state->storageObjectTransfers_.end() ||
            found->second.serial != transferSerial) {
            return;
        }
        transfer = std::move(found->second);
        state->storageObjectTransfers_.erase(found);
    }

    for (const auto& [key, weakOperation] : transfer.persistence) {
        if (auto operation = weakOperation.lock()) {
            PersistentRequestResult result;
            if ((fetch.status == ChunkFetchStatus::Found ||
                 fetch.status == ChunkFetchStatus::Missing) &&
                !persistenceSucceeded) {
                result = {PersistentRequestStatus::Error,
                          "could not persist exact Zarr storage object"};
            } else if (fetch.status == ChunkFetchStatus::Found) {
                result = {PersistentRequestStatus::Data, {}};
            } else if (fetch.status == ChunkFetchStatus::Missing) {
                result = {PersistentRequestStatus::Missing, {}};
            } else {
                result = {PersistentRequestStatus::Error,
                          fetchErrorMessage(fetch)};
            }
            completePersistenceOperation(state, key, operation, std::move(result));
        }
    }

    std::shared_ptr<const std::vector<std::byte>> payload;
    if (fetch.status == ChunkFetchStatus::Found) {
        payload = std::make_shared<const std::vector<std::byte>>(
            std::move(fetch.bytes));
    }
    for (const auto& [key, consumer] : transfer.consumers) {
        {
            std::lock_guard lock(state->mutex_);
            if (auto entry = state->entries_.find(key);
                entry != state->entries_.end()) {
                if (entry->second.probeTaskId == transfer.taskId)
                    entry->second.probeTaskId = 0;
                if (entry->second.fetchTaskId == transfer.taskId)
                    entry->second.fetchTaskId = 0;
            }
        }
        if (payload) {
            queueStorageObjectDecode(state, key, consumer, payload);
        } else {
            FetchContext context{
                consumer.generation, consumer.fetcherGeneration,
                consumer.fetchSerial, consumer.schedulerEpoch,
                consumer.fetcher, {}};
            finishAndStore(state, key, std::move(context), fetch);
        }
    }
}

void ChunkCache::queueStorageObjectDecode(
    const std::shared_ptr<State>& state,
    const ChunkKey& key,
    StorageConsumer consumer,
    std::shared_ptr<const std::vector<std::byte>> objectBytes)
{
    state->schedulerSelectionGate_->publish([&] {
        std::lock_guard lock(state->mutex_);
        auto entry = state->entries_.find(key);
        if (entry == state->entries_.end() ||
            entry->second.fetchSerial != consumer.fetchSerial ||
            consumer.generation != state->generation_ ||
            consumer.fetcherGeneration != state->fetcherGeneration_ ||
            !hasDemandLocked(entry->second)) {
            return;
        }
        const auto taskId = state->nextTaskId_->fetch_add(
            1, std::memory_order_relaxed);
        entry->second.decodeTaskId = taskId;
        auto scheduler = state->decodeScheduler_.lock();
        if (!scheduler)
            return;
        const auto priority = workPriorityLocked(*state, key, entry->second);
        std::weak_ptr<State> weakState = state;
        scheduler->submit(
            taskId, priority, state->schedulerGroup_, consumer.schedulerEpoch,
            [weakState, key, consumer, objectBytes] {
                auto state = weakState.lock();
                if (!state)
                    return;
                ChunkFetchResult decoded;
                try {
                    decoded = consumer.fetcher->decodeStorageObject(
                        fetcherKey(key), std::span<const std::byte>(
                            objectBytes->data(), objectBytes->size()));
                } catch (const std::exception& error) {
                    decoded.status = ChunkFetchStatus::DecodeError;
                    decoded.message = error.what();
                }
                FetchContext context{
                    consumer.generation, consumer.fetcherGeneration,
                    consumer.fetchSerial, consumer.schedulerEpoch,
                    consumer.fetcher, {}};
                finishAndStore(
                    state, key, std::move(context), std::move(decoded));
            });
    });
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
    finishAndStore(state, key, context, std::move(decoded));
}

void ChunkCache::finishAndStore(const std::shared_ptr<State>& state,
                                const ChunkKey& key,
                                FetchContext context,
                                ChunkFetchResult fetch)
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
        storeFetchResultLocked(state, key, std::move(fetch));
    }
    enforceSharedBudget(state);
    state->cv_.notify_all();
    notifyListeners(state);
}

void ChunkCache::storeFetchResultLocked(
    const std::shared_ptr<State>& state,
    const ChunkKey& key,
    ChunkFetchResult fetch)
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

    switch (fetch.status) {
    case ChunkFetchStatus::Found: {
        if (fetch.bytes.size() != expectedChunkBytes(*state, key)) {
            entry.status = EntryStatus::Error;
            entry.error = "decoded chunk byte size does not match full chunk shape";
            break;
        }
        if (state->options_.detectAllFillChunks && isAllFill(*state, fetch.bytes)) {
            entry.status = EntryStatus::AllFill;
            break;
        }
        entry.status = EntryStatus::Data;
        entry.decodedBytes = fetch.bytes.size();
        entry.bytes = std::make_shared<const std::vector<std::byte>>(std::move(fetch.bytes));
        state->decodedBytes_ += entry.decodedBytes;
        addDecodedBytesLocked(*state, entry.decodedBytes);
        break;
    }
    case ChunkFetchStatus::Missing:
        entry.status = EntryStatus::Missing;
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

std::filesystem::path ChunkCache::mirrorObjectPath(
    const State& state,
    const ChunkStorageObject& object)
{
    if (!state.options_.persistentCachePath ||
        !isSafeZarrStoreKey(object.sourceKey)) {
        throw std::runtime_error("invalid Zarr mirror storage-object key");
    }
    return *state.options_.persistentCachePath /
           std::filesystem::path(object.sourceKey);
}

std::filesystem::path ChunkCache::mirrorEmptyPath(
    const State& state,
    const ChunkStorageObject& object)
{
    auto path = mirrorObjectPath(state, object);
    path += ".empty";
    return path;
}

bool ChunkCache::writeMirrorObject(
    State& state,
    const ChunkStorageObject& object,
    std::span<const std::byte> bytes)
{
    const auto path = mirrorObjectPath(state, object);
    const auto emptyPath = mirrorEmptyPath(state, object);
    auto reservation = state.persistentBudget_
        ? state.persistentBudget_->reserveWrite(path, bytes.size(), {emptyPath})
        : PersistentZarrCacheBudget::WriteReservation{};
    if (state.persistentBudget_ && !reservation)
        return false;
    const auto oldSize = regularFileSize(path).value_or(0);
    const auto oldEmptySize = regularFileSize(emptyPath).value_or(0);
    if (!atomicWriteBytes(path, bytes)) {
        reservation.commit();
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(emptyPath, ec);
    const auto newSize = regularFileSize(path).value_or(bytes.size());
    addPersistentCacheBytesDelta(
        state, static_cast<std::int64_t>(newSize) -
                   static_cast<std::int64_t>(oldSize) -
                   static_cast<std::int64_t>(oldEmptySize));
    reservation.commit();
    return true;
}

bool ChunkCache::writeMirrorEmpty(
    State& state,
    const ChunkStorageObject& object)
{
    const auto path = mirrorEmptyPath(state, object);
    const auto dataPath = mirrorObjectPath(state, object);
    auto reservation = state.persistentBudget_
        ? state.persistentBudget_->reserveWrite(path, 0, {dataPath})
        : PersistentZarrCacheBudget::WriteReservation{};
    if (state.persistentBudget_ && !reservation)
        return false;
    const auto oldSize = regularFileSize(path).value_or(0);
    const auto oldDataSize = regularFileSize(dataPath).value_or(0);
    if (!atomicWriteBytes(path, {})) {
        reservation.commit();
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(dataPath, ec);
    const auto newSize = regularFileSize(path).value_or(0);
    addPersistentCacheBytesDelta(
        state, static_cast<std::int64_t>(newSize) -
                   static_cast<std::int64_t>(oldSize) -
                   static_cast<std::int64_t>(oldDataSize));
    reservation.commit();
    return true;
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
    if (state.decodedByteBudget_)
        entry.budgetTouch = state.decodedByteBudget_->nextTouch();
}

void ChunkCache::enforceCapacityLocked(const std::shared_ptr<State>& state)
{
    auto overBudget = [&] {
        return state->entries_.size() > state->options_.metadataEntryCapacity;
    };
    if (!overBudget())
        return;

    for (auto it = state->lru_.end();
         overBudget() && it != state->lru_.begin();) {
        --it;
        auto entryIt = state->entries_.find(*it);
        if (entryIt == state->entries_.end()) {
            it = state->lru_.erase(it);
            continue;
        }
        Entry& entry = entryIt->second;
        // Entries whose key has a parked blocking reader must survive until
        // that reader wakes; capacity overshoots by at most that count.
        if (hasParkedBlockingReaderLocked(*state, *it))
            continue;
        const ChunkKey victim = *it;
        it = state->lru_.erase(it);
        entry.inLru = false;
        if (entry.status == EntryStatus::Data) {
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
            entry->second.status == EntryStatus::Data &&
            !hasParkedBlockingReaderLocked(*state, *it)) {
            // Must mirror evictOldestDecodedLocked's reader skip: reporting
            // a protected entry's touch would make the budget's enforce()
            // loop keep selecting a victim whose eviction then refuses.
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
        if (hasParkedBlockingReaderLocked(*state, *it))
            continue;

        const ChunkKey victim = *it;
        const std::size_t bytes = entry.decodedBytes;
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
    if (bytes > 0 && state.decodedByteBudget_)
        state.decodedByteBudget_->addBytes(bytes);
}

void ChunkCache::removeDecodedBytesLocked(State& state, std::size_t bytes)
{
    if (bytes > 0 && state.decodedByteBudget_)
        state.decodedByteBudget_->removeBytes(bytes);
}

void ChunkCache::enforceSharedBudget(const std::shared_ptr<State>& state)
{
    if (state->decodedByteBudget_)
        state->decodedByteBudget_->enforce();
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
    const auto fillBytes =
        std::bit_cast<std::array<std::byte, sizeof(fill)>>(fill);
    for (std::size_t offset = 0; offset < bytes.size();
         offset += sizeof(fill)) {
        if (bytes[offset] != fillBytes[0] ||
            bytes[offset + 1] != fillBytes[1]) {
            return false;
        }
    }
    return true;
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
                                            bool active,
                                            bool isolateCallbackExceptions)
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
        if (!cb)
            continue;
        if (!isolateCallbackExceptions) {
            cb(key, active);
            continue;
        }
        try {
            cb(key, active);
        } catch (...) {
            // Isolation is requested by scheduler worker threads (an escape
            // would std::terminate the jthread) and by invalidation/refresh
            // paths (stop events must reach every listener, and the cv
            // notify that parked readers depend on must not be skipped).
            Logger()->warn(
                "[ChunkCache] remote-fetch listener threw; continuing");
        }
    }
}

} // namespace vc::render
