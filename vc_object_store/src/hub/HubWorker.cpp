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

// Begin/end pair on HubSessionOptions::trace. A throwing callback is
// contained: tracing can neither abort a hub operation nor replace the
// exception it is unwinding with. The callback runs outside the worker's
// ownership exemption: a direct lease call made from it is refused like any
// other direct call on a managed replica.
struct TraceScope {
    const HubSessionOptions& options;
    std::string_view name;
    TraceScope(const HubSessionOptions& o, std::string_view n) : options(o), name(n)
    {
        if (options.trace) try { OutsideWorkerScope outside; options.trace(name, true); } catch (...) {}
    }
    ~TraceScope()
    {
        if (options.trace) try { OutsideWorkerScope outside; options.trace(name, false); } catch (...) {}
    }
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
};

bool leasedBeyondMargin(const HubStatus& status, const std::string& id, int64_t now)
{
    return std::any_of(status.leases.begin(), status.leases.end(), [&](const auto& lease) {
        return lease.objectUuid == id && lease.holder == status.replica && lease.expiresUs - now > kHoldMarginUs;
    });
}
}

std::set<std::string> HubLeaseCounts::held()
{
    std::lock_guard lock(mutex);
    return heldLocked();
}

std::set<std::string> HubLeaseCounts::heldLocked() const
{
    std::set<std::string> result;
    for (const auto& [object, count] : counts) if (count > 0) result.insert(object);
    return result;
}

namespace {
// Traced before the ownership lock is taken: "start" marks the moment a
// replacement worker is about to claim the replica.
ReplicaWorkerLock takeOwnership(const HubSessionOptions& options, const std::filesystem::path& replica)
{
    { TraceScope start(options, "start"); }
    return ReplicaWorkerLock(replica);
}

ObjectStore openMirror(ObjectStore& replica)
{
    OpenOptions options;
    options.author = replica.author();
    options.readerPoolSize = 1;
    return ObjectStore::open(replica.path(), options);
}
}

HubWorker::HubWorker(ObjectStore& replica, const HubSessionOptions& options, std::function<void()> statusChanged)
    : replica_(replica), ownership_(takeOwnership(options, replica.path())), mirror_(openMirror(replica)), options_(options),
      statusChanged_(std::move(statusChanged))
{
    status_ = localStatus();
    due_ = std::chrono::steady_clock::now();
    thread_ = std::thread([this] { run(); });
}

HubWorker::~HubWorker()
{
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
    releaseOnStop();
}

void HubWorker::releaseOnStop() noexcept
{
    // The thread has stopped: nothing else touches the hub through this
    // worker. Surviving LeaseHolds are inert from here on (the session that
    // owned them is being replaced or destroyed), so their leases would only
    // block other replicas until expiry. Pending objects keep theirs: the
    // push they still need requires the lease.
    try {
        TraceScope trace(options_, "stop.release");
        InsideWorkerScope worker;
        const auto status = hubStatus(replica_);
        if (!status) return;
        const std::set<std::string> pending(status->pending.begin(), status->pending.end());
        std::vector<std::string> release;
        for (const auto& lease : status->leases)
            if (lease.holder == status->replica && !pending.contains(lease.objectUuid)) release.push_back(lease.objectUuid);
        if (!release.empty()) releaseLeases(replica_, release);
    } catch (...) {
        // Best effort: the leases expire on their own.
    }
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
    return status_;
}

void HubWorker::addLocked(LeaseHold& hold, HubLeaseCounts& counts, const std::string& id)
{
    // Record first: an allocation failure on either step leaves no count
    // without a record (reset() could not undo it) and no zero entry.
    hold.objects_.push_back(id);
    try {
        ++counts.counts.try_emplace(id, 0).first->second;
    } catch (...) {
        hold.objects_.pop_back();
        throw;
    }
}

void HubWorker::afterPull()
{
    // The pull replaced the whole lease mirror from one hub snapshot.
    std::lock_guard lock(leases_->mutex);
    leases_->untrusted.clear();
    leases_->mirrorVerified = true;
}

LeaseHold HubWorker::hold(const std::vector<std::string>& objects, bool createdHere)
{
    std::vector<std::string> ids;
    for (const auto& object : objects) {
        auto id = formatUuid(parseUuid(object));
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(std::move(id));
    }
    TraceScope trace(options_, "hold");
    InsideWorkerScope worker;
    // Owning from the start: whatever is registered below is released again
    // if a later step throws.
    LeaseHold result;
    result.counts_ = leases_;
    std::set<std::string> registered;
    {
        // Fast path: another live hold keeps the object leased (release
        // selection skips counted objects) and the replica's lease mirror,
        // which every lease operation on this replica updates (and only this
        // worker may perform, see ReplicaWorkerLock), shows a lease valid
        // beyond the margin. No hub_ and no pooled reader of replica_, so no
        // wait for a running sync; a local read of the mirror only.
        const auto status = hubStatus(mirror_);
        const auto now = nowMicros();
        std::lock_guard lock(leases_->mutex);
        for (const auto& id : ids) {
            const auto count = leases_->counts.find(id);
            if (leases_->mirrorVerified && count != leases_->counts.end() && count->second > 0 &&
                !leases_->untrusted.contains(id) && leasedBeyondMargin(*status, id, now)) {
                addLocked(result, *leases_, id);
                registered.insert(id);
            }
        }
    }
    if (registered.size() == ids.size()) return result;
    std::unique_lock hub(hub_, std::defer_lock);
    { TraceScope wait(options_, "hold.wait"); hub.lock(); }
    // An inherited mirror is verified against the hub before anything in it
    // is relied on (a lease the previous owner released may still show).
    bool verified;
    {
        std::lock_guard lock(leases_->mutex);
        verified = leases_->mirrorVerified;
    }
    if (!verified) {
        { TraceScope pull(options_, "hold.pull"); hubPull(replica_); }
        afterPull();
    }
    // The wait, and the acquisitions below, may take long: every requested
    // object is validated against the mirror again before each step and once
    // more before returning, the ones registered above included.
    const auto missingNow = [&] {
        std::set<std::string> untrusted;
        {
            std::lock_guard lock(leases_->mutex);
            untrusted = leases_->untrusted;
        }
        const auto status = hubStatus(replica_);
        const auto now = nowMicros();
        std::vector<std::string> missing;
        for (const auto& id : ids)
            if (untrusted.contains(id) || !leasedBeyondMargin(*status, id, now)) missing.push_back(id);
        return missing;
    };
    for (int attempt = 0;; ++attempt) {
        const auto missing = missingNow();
        if (missing.empty()) break;
        if (attempt == 3) throw std::runtime_error("hub: leases for " + missing.front() + " kept expiring while they were acquired");
        // Objects the caller vouches for as created here (a fresh identity)
        // and never pushed need no lease: the push grants their creator
        // lease. "Never pushed" means no hub version, no push in flight whose
        // outcome is unknown, and no recorded conflict. Everything else is
        // leased on the hub, a version-0 object included: a supplied identity
        // (an import) may already exist there under another replica's lease.
        std::vector<std::string> acquire;
        if (!createdHere) acquire = missing;
        else {
            auto txn = replica_.read();
            auto* db = ObjectStoreImplAccess::conn(txn);
            const auto inflight = uuidColumn(db, "SELECT record_uuid FROM hub_inflight");
            const auto conflicts = uuidColumn(db, "SELECT record_uuid FROM hub_conflicts");
            for (const auto& id : missing)
                if (knownHubVersion(db, id) != 0 || inflight.contains(id) || conflicts.contains(id)) acquire.push_back(id);
        }
        if (acquire.empty()) break;
        try {
            TraceScope acquiring(options_, "hold.acquire");
            acquireLeases(replica_, acquire);
        } catch (const HubLeaseRefused& refused) {
            if (!refused.held.empty() || refused.stale.empty()) throw;
            // Editing starts from the hub's record: bring it in, then retry.
            { TraceScope pull(options_, "hold.pull"); hubPull(replica_); }
            afterPull();
            TraceScope acquiring(options_, "hold.acquire");
            acquireLeases(replica_, acquire);
        }
        // A successful acquisition rewrote the mirror rows it concerns.
        std::lock_guard lock(leases_->mutex);
        for (const auto& id : acquire) leases_->untrusted.erase(id);
    }
    std::lock_guard lock(leases_->mutex);
    for (const auto& id : ids) if (!registered.contains(id)) addLocked(result, *leases_, id);
    return result;
}

HubSessionStatus HubWorker::localStatus()
{
    HubSessionStatus result;
    if (const auto status = hubStatus(replica_)) {
        result.pending = status->pending;
        result.conflicts = status->conflicts;
        result.leases = status->leases;
    }
    return result;
}

HubSessionStatus HubWorker::syncOnce(const std::optional<std::string>& resolve)
{
    TraceScope trace(options_, "sync");
    InsideWorkerScope worker;
    std::unique_lock hub(hub_, std::defer_lock);
    { TraceScope wait(options_, "sync.wait"); hub.lock(); }
    if (resolve)
        for (const auto& conflict : hubStatus(replica_)->conflicts) {
            const bool object = nlohmann::json::parse(conflict.localRecord).value("record_type", "") == "objects";
            if (*resolve == "local" && object) renewLeases(replica_, {conflict.uuid});
            resolveHubConflict(replica_, conflict.uuid, *resolve);
        }
    { TraceScope pull(options_, "sync.pull"); hubPull(replica_); }
    afterPull();
    auto status = *hubStatus(replica_);
    if (!status.conflicts.empty()) return localStatus();
    const auto held = leases_->held();
    const std::set<std::string> pending(status.pending.begin(), status.pending.end());
    // Edits made while offline outlive their leases; pushing needs them again.
    std::vector<std::string> renew;
    for (const auto& lease : status.leases)
        if (lease.holder == status.replica && (held.contains(lease.objectUuid) || pending.contains(lease.objectUuid)) &&
            lease.expiresUs - nowMicros() < kRenewMarginUs)
            renew.push_back(lease.objectUuid);
    if (!renew.empty()) { TraceScope renewal(options_, "sync.renew"); renewLeases(replica_, renew); }
    { TraceScope push(options_, "sync.push"); hubPush(replica_); }
    // Release what nobody holds once pushed, including objects this push
    // created. The selection reads the live counts: a hold registered before
    // it keeps its lease, and one that needs a lease the sync released waits
    // for hub_ and acquires it again.
    status = *hubStatus(replica_);
    std::vector<std::string> release;
    {
        std::lock_guard lock(leases_->mutex);
        const auto heldNow = leases_->heldLocked();
        for (const auto& lease : status.leases)
            if (lease.holder == status.replica && !heldNow.contains(lease.objectUuid) &&
                std::find(status.pending.begin(), status.pending.end(), lease.objectUuid) == status.pending.end())
                release.push_back(lease.objectUuid);
    }
    if (!release.empty()) {
        TraceScope released(options_, "sync.release");
        // Distrusted before the hub is asked: a release that reaches the hub
        // but not the mirror (or fails to record its own uncertainty) leaves
        // the mirror untrusted for these until a pull replaces it or an
        // acquisition rewrites them. Trusted again only after full success.
        {
            std::lock_guard lock(leases_->mutex);
            leases_->untrusted.insert(release.begin(), release.end());
        }
        releaseLeases(replica_, release);
        std::lock_guard lock(leases_->mutex);
        for (const auto& id : release) leases_->untrusted.erase(id);
    }
    auto result = localStatus();
    result.syncedAtUs = nowMicros();
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
        const auto previous = status_;
        lock.unlock();
        HubSessionStatus next;
        try {
            next = syncOnce(resolve);
            if (!next.syncedAtUs) next.syncedAtUs = previous.syncedAtUs;
        } catch (const std::exception& error) {
            try { next = localStatus(); } catch (...) { next = previous; }
            next.syncedAtUs = previous.syncedAtUs;
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
