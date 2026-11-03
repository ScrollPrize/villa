// Background disk-cache writes: reads return as soon as a fetched chunk is
// decoded, and its cache file is written (and encoded) afterwards. Chunks with
// a pending write stay visible to reads, so they are never fetched twice.
#pragma once

#include "memory_cache.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vcr {

class WriteBehind {
public:
    WriteBehind(std::size_t threads, std::size_t max_pending_bytes) : max_bytes_(max_pending_bytes)
    {
        threads = threads ? threads : 1;
        for (std::size_t i = 0; i < threads; ++i)
            workers_.emplace_back([this] { loop(); });
    }

    // Runs every queued write, then stops the writers.
    ~WriteBehind()
    {
        {
            std::unique_lock lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    WriteBehind(const WriteBehind&) = delete;
    WriteBehind& operator=(const WriteBehind&) = delete;

    // Queues `write` for `key`; until it has run, pending(key) returns `visible`
    // (nullptr: the chunk is known to be missing). Blocks while more than the
    // byte budget is waiting to be written.
    void submit(const std::string& key, ChunkPtr visible, std::size_t bytes, std::function<void()> write)
    {
        std::unique_lock lock(mutex_);
        room_.wait(lock, [&] { return queued_bytes_ == 0 || queued_bytes_ + bytes <= max_bytes_; });
        const std::uint64_t gen = ++generation_;
        visible_[key] = {std::move(visible), gen};
        queue_.push_back(Item{key, gen, bytes, std::move(write)});
        queued_bytes_ += bytes;
        ++outstanding_;
        cv_.notify_one();
    }

    // (true, chunk) if `key` has a write queued or running.
    std::pair<bool, ChunkPtr> pending(const std::string& key)
    {
        std::lock_guard lock(mutex_);
        auto it = visible_.find(key);
        if (it == visible_.end())
            return {false, nullptr};
        return {true, it->second.first};
    }

    // Waits until every write submitted so far has run.
    void flush()
    {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return outstanding_ == 0; });
    }

    std::size_t outstanding()
    {
        std::lock_guard lock(mutex_);
        return outstanding_;
    }

    void wait(const std::string& key)
    {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return !visible_.contains(key); });
    }

private:
    struct Item {
        std::string key;
        std::uint64_t generation;
        std::size_t bytes;
        std::function<void()> write;
    };

    void loop()
    {
        for (;;) {
            Item item;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (queue_.empty())
                    return;  // stopping, and nothing left to write
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                item.write();  // reports its own errors; never throws by contract
            } catch (...) {
            }
            {
                std::lock_guard lock(mutex_);
                auto it = visible_.find(item.key);
                if (it != visible_.end() && it->second.second == item.generation)
                    visible_.erase(it);
                queued_bytes_ -= item.bytes;
                --outstanding_;
                idle_.notify_all();
            }
            room_.notify_all();
        }
    }

    std::size_t max_bytes_;
    std::size_t queued_bytes_ = 0;
    std::size_t outstanding_ = 0;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
    std::deque<Item> queue_;
    std::unordered_map<std::string, std::pair<ChunkPtr, std::uint64_t>> visible_;
    std::mutex mutex_;
    std::condition_variable cv_, room_, idle_;
    std::vector<std::thread> workers_;
};

}  // namespace vcr
