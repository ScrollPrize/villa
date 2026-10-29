#pragma once
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "hub/HubRules.hpp"
#include <condition_variable>
#include <map>
#include <set>
#include <thread>

namespace vc::objects::detail {
// Objects held by live LeaseHolds, counted per object, with what the worker
// knows about this replica's leases. Everything is guarded by `mutex`.
struct HubLeaseCounts {
    std::mutex mutex;
    std::map<std::string, int> counts;         // live LeaseHolds per object; no zero entries
    std::set<std::string> untrusted;           // objects whose mirror may disagree with the hub
    // The mirror this worker inherited may be stale (a previous owner's
    // release may have reached the hub but not the mirror): nothing in it is
    // trusted before this worker's first successful pull.
    bool mirrorVerified = false;
    std::set<std::string> held();
    std::set<std::string> heldLocked() const;  // caller holds mutex
};

// Background sync of one session's hosted replica: periodic pull/push, a push
// shortly after local actions, renewal of held leases and release of the rest
// once pushed. Hub operations are serialized with lease requests, except that
// a hold on an object another live hold already keeps leased is granted from
// the replica's lease mirror and the live counts alone, without waiting for a
// running sync. Stopping the worker releases the leases of the pushed objects
// it still held, best effort: a lease outliving its holder would block other
// replicas until it expired.
class HubWorker {
public:
    HubWorker(ObjectStore& replica, const HubSessionOptions&, std::function<void()> statusChanged);
    ~HubWorker();
    void requestSync(bool afterAction);
    // Blocks until a sync started after this call has finished; returns its status.
    HubSessionStatus syncAndWait(std::chrono::milliseconds timeout);
    void requestResolve(const std::string& keep);
    // See ObjectStoreSession::holdLeases for `createdHere`.
    LeaseHold hold(const std::vector<std::string>& objects, bool createdHere);
    HubSessionStatus status() const;

private:
    void run();
    HubSessionStatus localStatus();
    HubSessionStatus syncOnce(const std::optional<std::string>& resolve);
    // Registers `id` on `hold` and counts it; strong exception guarantee.
    // Caller holds leases_->mutex.
    static void addLocked(LeaseHold& hold, HubLeaseCounts& counts, const std::string& id);
    // After a successful pull the mirror equals one hub snapshot.
    void afterPull();
    // Releases this replica's leases on pushed objects; never throws.
    void releaseOnStop() noexcept;

    ObjectStore& replica_;
    // The worker owns the replica's lease management while it lives (one
    // worker per replica; direct lease calls are refused meanwhile).
    ReplicaWorkerLock ownership_;
    // Fast-path reads of the lease mirror go through a connection of their
    // own: the replica's reader pool can be held by a sync across remote work.
    ObjectStore mirror_;
    HubSessionOptions options_;
    std::function<void()> statusChanged_;
    std::shared_ptr<HubLeaseCounts> leases_ = std::make_shared<HubLeaseCounts>();
    std::mutex hub_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, synced_;
    uint64_t started_ = 0, finished_ = 0;
    bool stopping_ = false;
    std::chrono::steady_clock::time_point due_;
    std::optional<std::string> resolve_;
    HubSessionStatus status_;
    std::thread thread_;
};
}
