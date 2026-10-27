#include "FetchQueue.hpp"
#include <algorithm>

namespace vc::objects::detail {

FetchQueue::FetchQueue(Notify notify) : notify_(std::move(notify)) {}

FetchQueue::~FetchQueue()
{
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    done_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void FetchQueue::start(std::function<void(const std::string&)> fetch)
{
    std::lock_guard lock(mutex_);
    fetch_ = std::move(fetch);
    if (!worker_.joinable()) worker_ = std::thread([this] { run(); });
}

void FetchQueue::request(const std::vector<std::string>& uuids)
{
    const auto now = std::chrono::steady_clock::now();
    bool queued = false;
    {
        std::lock_guard lock(mutex_);
        for (const auto& uuid : uuids) {
            auto& entry = entries_[uuid];
            entry.requested = now;
            if (entry.state == GeometryState::Fetching || entry.state == GeometryState::Local) continue;
            if (entry.state == GeometryState::Failed && now - entry.failedAt < retryDelay) continue;
            if (std::find(queue_.begin(), queue_.end(), uuid) != queue_.end()) continue;
            queue_.push_back(uuid);
            queued = true;
        }
    }
    if (queued) wake_.notify_one();
}

void FetchQueue::refetch(const std::vector<std::string>& uuids)
{
    {
        std::lock_guard lock(mutex_);
        for (const auto& uuid : uuids) {
            auto& entry = entries_[uuid];
            entry.kept = true;
            if (entry.state == GeometryState::Fetching) continue;
            entry.state = GeometryState::Remote;
            if (std::find(queue_.begin(), queue_.end(), uuid) == queue_.end()) queue_.push_back(uuid);
        }
    }
    wake_.notify_one();
}

bool FetchQueue::fetchNow(const std::string& uuid, const std::function<bool()>& cancel, std::string* error)
{
    std::unique_lock lock(mutex_);
    auto& entry = entries_[uuid];
    entry.requested = std::chrono::steady_clock::now();
    if (entry.state == GeometryState::Failed || entry.state == GeometryState::Local) entry.state = GeometryState::Remote;
    if (entry.state != GeometryState::Fetching) {
        std::erase(queue_, uuid);
        queue_.push_front(uuid);
        wake_.notify_one();
    }
    ++entry.waiters;
    for (;;) {
        done_.wait_for(lock, std::chrono::milliseconds(100));
        auto& current = entries_[uuid];
        if (stopping_ || (cancel && cancel())) { --current.waiters; return false; }
        if (current.state == GeometryState::Local) { --current.waiters; return true; }
        if (current.state == GeometryState::Failed) {
            --current.waiters;
            if (error) *error = current.error;
            return false;
        }
    }
}

std::optional<GeometryState> FetchQueue::state(const std::string& uuid) const
{
    std::lock_guard lock(mutex_);
    const auto found = entries_.find(uuid);
    if (found == entries_.end()) return std::nullopt;
    return found->second.state;
}

void FetchQueue::cancelQueued()
{
    std::lock_guard lock(mutex_);
    for (const auto& uuid : queue_)
        if (entries_[uuid].waiters == 0) entries_[uuid].state = GeometryState::Remote;
    std::erase_if(queue_, [&](const std::string& uuid) { return entries_[uuid].waiters == 0; });
}

void FetchQueue::run()
{
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (stopping_) return;
        const auto uuid = queue_.front();
        queue_.pop_front();
        auto& entry = entries_[uuid];
        if (entry.waiters == 0 && !entry.kept && std::chrono::steady_clock::now() - entry.requested > requestLifetime) {
            entry.state = GeometryState::Remote;  // its reader moved on
            continue;
        }
        entry.state = GeometryState::Fetching;
        lock.unlock();
        if (notify_) notify_(uuid, GeometryState::Fetching, {});
        std::string error;
        try {
            fetch_(uuid);
        } catch (const std::exception& e) {
            error = e.what();
        }
        lock.lock();
        auto& finished = entries_[uuid];
        finished.state = error.empty() ? GeometryState::Local : GeometryState::Failed;
        finished.kept = false;
        finished.error = error;
        if (!error.empty()) finished.failedAt = std::chrono::steady_clock::now();
        const auto state = finished.state;
        lock.unlock();
        if (notify_) notify_(uuid, state, error);
        done_.notify_all();
        lock.lock();
    }
}

} // namespace vc::objects::detail
