#pragma once
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <condition_variable>
#include <map>
#include <set>
#include <thread>

namespace vc::objects::detail {
// Objects held by live LeaseHolds, counted per object.
struct HubLeaseCounts {
    std::mutex mutex;
    std::map<std::string, int> counts;
    std::set<std::string> held();
    bool retain(const std::vector<std::string>& objects, const HubStatus& status, int64_t validThroughUs);
};

// Background sync of one session's hosted replica: periodic pull/push, a push
// shortly after local actions, renewal of held leases and release of the rest
// once pushed. Hub operations are serialized with lease requests.
class HubWorker {
public:
    HubWorker(ObjectStore& replica, const HubSessionOptions&, std::function<void()> statusChanged);
    ~HubWorker();
    void requestSync(bool afterAction);
    // Blocks until a sync started after this call has finished; returns its status.
    HubSessionStatus syncAndWait(std::chrono::milliseconds timeout);
    void requestResolve(const std::string& keep);
    LeaseHold hold(const std::vector<std::string>& objects);
    HubSessionStatus status() const;

private:
    void run();
    HubSessionStatus localStatus();
    HubSessionStatus syncOnce(const std::optional<std::string>& resolve);

    ObjectStore& replica_;
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
