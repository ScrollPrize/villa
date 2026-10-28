#include "hub/HubWorker.hpp"
#include "hub/HubRules.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects {
LeaseHold::LeaseHold(LeaseHold&& other) noexcept
    : counts_(std::move(other.counts_)), objects_(std::move(other.objects_))
{
    other.objects_.clear();
}

LeaseHold& LeaseHold::operator=(LeaseHold&& other) noexcept
{
    if (this != &other) {
        reset();
        counts_ = std::move(other.counts_);
        objects_ = std::move(other.objects_);
        other.objects_.clear();
    }
    return *this;
}

PushHold::PushHold(PushHold&& other) noexcept : holds_(std::move(other.holds_)) {}

PushHold& PushHold::operator=(PushHold&& other) noexcept
{
    if (this != &other) {
        reset();
        holds_ = std::move(other.holds_);
    }
    return *this;
}

void PushHold::reset()
{
    if (const auto holds = holds_.lock()) {
        std::lock_guard lock(holds->mutex);
        if (--holds->count == 0 && holds->worker) holds->worker->requestSync(false);
    }
    holds_.reset();
}

void LeaseHold::reset()
{
    if (const auto counts = counts_.lock()) {
        std::lock_guard lock(counts->mutex);
        for (const auto& object : objects_)
            if (const auto it = counts->counts.find(object); it != counts->counts.end() && --it->second == 0)
                counts->counts.erase(it);
    }
    counts_.reset();
    objects_.clear();
}
}

namespace vc::objects::detail {
namespace {
// Renew a held lease once less than this much of it remains.
constexpr int64_t kRenewMarginUs = 5 * 60 * 1000000LL;
// A lease about to lapse is not trusted for a new edit.
constexpr int64_t kHoldMarginUs = 60 * 1000000LL;

std::string firstLine(const std::string& text) { return text.substr(0, text.find('\n')); }
}

std::set<std::string> HubLeaseCounts::held()
{
    std::lock_guard lock(mutex);
    std::set<std::string> result;
    for (const auto& [object, count] : counts) result.insert(object);
    return result;
}

HubWorker::HubWorker(ObjectStore& replica, const HubSessionOptions& options, std::function<void()> statusChanged)
    : replica_(replica), options_(options), statusChanged_(std::move(statusChanged))
{
    status_ = localStatus();
    due_ = std::chrono::steady_clock::now();
    pushHolds_->worker = this;
    thread_ = std::thread([this] { run(); });
}

HubWorker::~HubWorker()
{
    {
        std::lock_guard lock(pushHolds_->mutex);
        pushHolds_->worker = nullptr;
    }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

void HubWorker::requestSync(bool afterAction)
{
    {
        std::lock_guard lock(mutex_);
        due_ = std::min(due_, std::chrono::steady_clock::now() + (afterAction ? options_.pushDelay : std::chrono::milliseconds(0)));
    }
    wake_.notify_all();
}

HubSessionStatus HubWorker::syncAndWait(std::chrono::milliseconds timeout)
{
    std::unique_lock lock(mutex_);
    const auto target = started_ + 1;
    due_ = std::chrono::steady_clock::now();
    pushRequested_ = true;
    wake_.notify_all();
    if (!synced_.wait_for(lock, timeout, [&] { return finished_ >= target; }))
        throw std::runtime_error("hub: no sync finished within " + std::to_string(timeout.count()) + " ms" +
                                 (status_.error.empty() ? std::string() : ": " + status_.error));
    return status_;
}

void HubWorker::requestResolve(const std::string& keep)
{
    if (keep != "hub" && keep != "local") throw std::invalid_argument("hub: keep must be \"hub\" or \"local\"");
    {
        std::lock_guard lock(mutex_);
        resolve_ = keep;
    }
    requestSync(false);
}

HubSessionStatus HubWorker::status() const
{
    std::lock_guard lock(mutex_);
    auto result = status_;
    result.pushesHeld = pushHolds_->count > 0;
    return result;
}

PushHold HubWorker::holdPushes()
{
    PushHold result;
    ++pushHolds_->count;
    result.holds_ = pushHolds_;
    return result;
}

LeaseHold HubWorker::hold(const std::vector<std::string>& objects)
{
    std::vector<std::string> ids;
    for (const auto& object : objects) ids.push_back(formatUuid(parseUuid(object)));
    {
        std::lock_guard hub(hub_);
        const auto status = hubStatus(replica_);
        std::vector<std::string> missing;
        for (const auto& id : ids)
            if (std::none_of(status->leases.begin(), status->leases.end(), [&](const auto& lease) {
                    return lease.objectUuid == id && lease.holder == status->replica && lease.expiresUs - nowMicros() > kHoldMarginUs;
                }))
                missing.push_back(id);
        if (!missing.empty()) {
            try {
                acquireLeases(replica_, missing);
            } catch (const HubLeaseRefused& refused) {
                if (!refused.held.empty() || refused.stale.empty()) throw;
                // Editing starts from the hub's record: bring it in, then retry.
                hubPull(replica_);
                acquireLeases(replica_, missing);
            }
        }
    }
    LeaseHold result;
    {
        std::lock_guard lock(leases_->mutex);
        for (const auto& id : ids) ++leases_->counts[id];
    }
    result.counts_ = leases_;
    result.objects_ = std::move(ids);
    return result;
}

HubSessionStatus HubWorker::localStatus()
{
    HubSessionStatus result;
    if (const auto status = hubStatus(replica_)) {
        result.hub = status->hub.string();
        result.hubReplica = status->hubReplica;
        result.pending = status->pending;
        result.conflicts = status->conflicts;
        result.leases = status->leases;
    }
    return result;
}

HubSessionStatus HubWorker::syncOnce(const std::optional<std::string>& resolve, bool push)
{
    std::lock_guard hub(hub_);
    if (resolve)
        for (const auto& conflict : hubStatus(replica_)->conflicts) {
            const bool object = nlohmann::json::parse(conflict.localRecord).value("record_type", "") == "objects";
            if (*resolve == "local" && object) renewLeases(replica_, {conflict.uuid});
            resolveHubConflict(replica_, conflict.uuid, *resolve);
        }
    const auto pulled = hubPull(replica_);
    auto status = *hubStatus(replica_);
    if (!status.conflicts.empty()) {
        auto result = localStatus();
        result.position = pulled.pulledPosition;
        return result;
    }
    const auto held = leases_->held();
    const std::set<std::string> pending(status.pending.begin(), status.pending.end());
    // Edits made while offline outlive their leases; pushing needs them again.
    std::vector<std::string> renew, release;
    for (const auto& lease : status.leases)
        if (lease.holder == status.replica && (held.contains(lease.objectUuid) || pending.contains(lease.objectUuid)) &&
            lease.expiresUs - nowMicros() < kRenewMarginUs)
            renew.push_back(lease.objectUuid);
    if (!renew.empty()) renewLeases(replica_, renew);
    int64_t position = pulled.pulledPosition;
    if (push) {
        if (const auto pushed = hubPush(replica_); pushed.publishedPosition) position = pushed.publishedPosition;
        // Release what nobody holds once pushed, including objects this push created.
        status = *hubStatus(replica_);
        for (const auto& lease : status.leases)
            if (lease.holder == status.replica && !held.contains(lease.objectUuid) &&
                std::find(status.pending.begin(), status.pending.end(), lease.objectUuid) == status.pending.end())
                release.push_back(lease.objectUuid);
        if (!release.empty()) releaseLeases(replica_, release);
    }
    auto result = localStatus();
    result.syncedAtUs = nowMicros();
    result.position = position;
    return result;
}

void HubWorker::run()
{
    std::unique_lock lock(mutex_);
    while (true) {
        wake_.wait_until(lock, due_, [&] { return stopping_ || std::chrono::steady_clock::now() >= due_; });
        if (stopping_) return;
        due_ = std::chrono::steady_clock::now() + options_.syncInterval;
        ++started_;
        const auto resolve = std::exchange(resolve_, std::nullopt);
        const bool push = std::exchange(pushRequested_, false) || pushHolds_->count == 0;
        const auto previous = status_;
        lock.unlock();
        HubSessionStatus next;
        try {
            next = syncOnce(resolve, push);
            if (!next.syncedAtUs) next.syncedAtUs = previous.syncedAtUs;
        } catch (const std::exception& error) {
            try { next = localStatus(); } catch (...) { next = previous; }
            next.syncedAtUs = previous.syncedAtUs;
            next.position = previous.position;
            next.error = firstLine(error.what());
        }
        lock.lock();
        status_ = next;
        ++finished_;
        synced_.notify_all();
        lock.unlock();
        statusChanged_();
        lock.lock();
    }
}
}
