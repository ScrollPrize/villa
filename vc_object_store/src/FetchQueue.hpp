#pragma once
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

namespace vc::objects::detail {

// Where an object's geometry is: in the store, published but not fetched yet,
// being fetched, or the last fetch failed (it is retried on a later read).
enum class GeometryState { Local, Remote, Fetching, Failed };

// Fetches the geometry of one attachment's objects on a worker thread. Requests
// from reads are deduplicated; a queued request not repeated within the
// request lifetime is dropped (its reader moved on); a failed fetch is retried
// by reads only after the retry delay, by waiters at once.
class FetchQueue {
public:
    FetchQueue();
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
    std::function<void(const std::string&)> fetch_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::deque<std::string> queue_;
    std::map<std::string, Entry> entries_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace vc::objects::detail
