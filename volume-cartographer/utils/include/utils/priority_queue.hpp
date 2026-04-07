#pragma once
#include <deque>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace utils {

// Thread-safe deque with deduplication.
// submit() pushes to back. boost() pushes to front (or moves to front if already queued).
template <typename T,
          typename Hash     = std::hash<T>,
          typename KeyEqual = std::equal_to<T>,
          typename Compare  = std::greater<T>>
class DeduplicatingPriorityQueue final {
public:
    DeduplicatingPriorityQueue() = default;
    ~DeduplicatingPriorityQueue() { shutdown(); }

    DeduplicatingPriorityQueue(const DeduplicatingPriorityQueue&) = delete;
    DeduplicatingPriorityQueue& operator=(const DeduplicatingPriorityQueue&) = delete;

    bool submit(const T& item) {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return false;
            if (set_.contains(item)) return false;
            set_.insert(item);
            deque_.push_back(item);
        }
        cv_.notify_one();
        return true;
    }

    template <typename Iter>
    std::size_t submit_batch(Iter begin, Iter end) {
        std::size_t added = 0;
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return 0;
            for (auto it = begin; it != end; ++it) {
                if (!set_.contains(*it)) {
                    set_.insert(*it);
                    deque_.push_back(*it);
                    ++added;
                }
            }
        }
        if (added > 0) cv_.notify_all();
        return added;
    }

    // Put at front. Stale duplicates in deque are skipped by pop().
    bool boost(const T& item) {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return false;
            set_.insert(item);
            deque_.push_front(item);
        }
        cv_.notify_one();
        return true;
    }

    [[nodiscard]] T pop() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !set_.empty() || shutdown_flag_; });
        if (set_.empty())
            throw std::runtime_error("DeduplicatingPriorityQueue::pop(): shutdown");
        // Skip stale duplicates from boost()
        while (!deque_.empty()) {
            T item = deque_.front();
            deque_.pop_front();
            if (set_.erase(item))
                return item;
        }
        throw std::runtime_error("DeduplicatingPriorityQueue::pop(): empty");
    }

    [[nodiscard]] std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        while (!deque_.empty()) {
            T item = deque_.front();
            deque_.pop_front();
            if (set_.erase(item)) return item;
        }
        return std::nullopt;
    }

    template <typename Rep, typename Period>
    [[nodiscard]] std::optional<T> pop_for(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !set_.empty() || shutdown_flag_; }))
            return std::nullopt;
        while (!deque_.empty()) {
            T item = deque_.front();
            deque_.pop_front();
            if (set_.erase(item)) return item;
        }
        return std::nullopt;
    }

    void cancel_pending() {
        std::lock_guard lock(mutex_);
        set_.clear();
        deque_.clear();
    }

    [[nodiscard]] bool is_queued(const T& item) const {
        std::lock_guard lock(mutex_);
        return set_.contains(item);
    }

    [[nodiscard]] std::size_t queued_count() const noexcept {
        std::lock_guard lock(mutex_);
        return set_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        std::lock_guard lock(mutex_);
        return set_.empty();
    }

    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return;
            shutdown_flag_ = true;
        }
        cv_.notify_all();
    }

    // resubmit = boost (compat alias)
    bool resubmit(const T& item) { return boost(item); }

private:
    using Set = std::unordered_set<T, Hash, KeyEqual>;

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::deque<T>           deque_;
    Set                     set_;
    bool                    shutdown_flag_ = false;
};

template <typename T, typename Hash, typename KeyEqual, typename Compare, typename F>
void consume_loop(DeduplicatingPriorityQueue<T, Hash, KeyEqual, Compare>& queue, F&& handler) {
    for (;;) {
        T item;
        try {
            item = queue.pop();
        } catch (const std::runtime_error&) {
            return;
        }
        handler(item);
    }
}

template <typename T, typename F>
void consume_loop(DeduplicatingPriorityQueue<T>& queue, F&& handler) {
    consume_loop<T, std::hash<T>, std::equal_to<T>, std::greater<T>, F>(
        queue, std::forward<F>(handler));
}

} // namespace utils
