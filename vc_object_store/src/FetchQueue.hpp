#pragma once
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

namespace vc::objects::detail {

// Fetches the geometry of one attachment's objects on a worker thread. Requests
// from reads are deduplicated; a queued request not repeated within the
// request lifetime is dropped (its reader moved on); a failed fetch is retried
// by reads only after the retry delay, by waiters at once.
class FetchQueue {
public:
    using Notify = std::function<void(const std::string& uuid, GeometryState, const std::string& error)>;
    explicit FetchQueue(Notify notify);
    ~FetchQueue();
    FetchQueue(const FetchQueue&) = delete;
    FetchQueue& operator=(const FetchQueue&) = delete;

    // Run `fetch(uuid)` (which writes the object's geometry or throws) on the worker.
    void start(std::function<void(const std::string&)> fetch);
    void request(const std::vector<std::string>& uuids);
    // Fetch again (a changed published geometry), whether or not anyone reads it.
    void refetch(const std::vector<std::string>& uuids);
    // Fetch now and block until done; false when it failed or `cancel` said so.
    bool fetchNow(const std::string& uuid, const std::function<bool()>& cancel, std::string* error);
    std::optional<GeometryState> state(const std::string& uuid) const;
    void cancelQueued();

    std::chrono::milliseconds requestLifetime{10000};
    std::chrono::milliseconds retryDelay{30000};

private:
    struct Entry {
        GeometryState state = GeometryState::Remote;
        std::chrono::steady_clock::time_point requested, failedAt;
        std::string error;
        int waiters = 0;
        bool kept = false;  // refetch: not dropped for want of readers
    };
    void run();
    Notify notify_;
    std::function<void(const std::string&)> fetch_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::deque<std::string> queue_;
    std::map<std::string, Entry> entries_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace vc::objects::detail
