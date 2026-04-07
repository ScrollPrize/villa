#pragma once
#include <queue>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstdint>
#include <concepts>
#include <functional>
#include <optional>
#include <atomic>
#include <utility>
#include <chrono>

namespace utils {

// ---------------------------------------------------------------------------
// DeduplicatingPriorityQueue
//
// Thread-safe priority queue with dual-set deduplication.  Items that are
// either already queued or currently in-flight are rejected on submit(),
// giving O(1) membership checks via unordered_set.
//
// The Compare parameter controls heap ordering (default std::greater<T>
// yields a min-queue, matching std::priority_queue's convention when the
// comparator returns true for "lower priority").  Users can supply an
// epoch-aware comparator to implement epoch-based reordering.
// ---------------------------------------------------------------------------
template <typename T,
          typename Hash     = std::hash<T>,
          typename KeyEqual = std::equal_to<T>,
          typename Compare  = std::greater<T>>
class DeduplicatingPriorityQueue final {
public:
    DeduplicatingPriorityQueue() = default;

    DeduplicatingPriorityQueue(const DeduplicatingPriorityQueue&)            = delete;
    DeduplicatingPriorityQueue& operator=(const DeduplicatingPriorityQueue&) = delete;
    DeduplicatingPriorityQueue(DeduplicatingPriorityQueue&&)                 = delete;
    DeduplicatingPriorityQueue& operator=(DeduplicatingPriorityQueue&&)      = delete;

    ~DeduplicatingPriorityQueue() { shutdown(); }

    // -- submission ---------------------------------------------------------

    /// Submit a single item.  Returns false if already queued.
    bool submit(const T& item) {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return false;
            if (queued_set_.contains(item))
                return false;
            queued_set_.insert(item);
            heap_.push(item);
        }
        cv_.notify_all();
        return true;
    }

    /// If the item is already queued, boost it to the front of the queue.
    /// If not queued, submit it at the front. Returns true if boosted/submitted.
    bool boost(const T& item) {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return false;
            boost_set_.insert(item);
            if (!queued_set_.contains(item)) {
                queued_set_.insert(item);
                // Don't need to push to heap — pop_locked checks boost_set first
            }
        }
        cv_.notify_all();
        return true;
    }

    /// Batch submit from an iterator range.  Returns the number of newly
    /// queued items (those not already queued).
    template <typename Iter>
    std::size_t submit_batch(Iter begin, Iter end) {
        std::size_t added = 0;
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return 0;
            for (auto it = begin; it != end; ++it) {
                if (!queued_set_.contains(*it)) {
                    queued_set_.insert(*it);
                    heap_.push(*it);
                    ++added;
                }
            }
        }
        if (added > 0) cv_.notify_all();
        return added;
    }

    // -- consumption --------------------------------------------------------

    /// Block until an item is available (or shutdown is signalled).
    /// Moves the item from queued to in-flight atomically.
    /// Throws std::runtime_error on shutdown with an empty queue.
    [[nodiscard]] T pop() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !queued_set_.empty() || shutdown_flag_; });
        if (queued_set_.empty())
            throw std::runtime_error("DeduplicatingPriorityQueue::pop(): shutdown");
        return pop_locked();
    }

    /// Non-blocking pop.
    [[nodiscard]] std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        if (queued_set_.empty()) return std::nullopt;
        return pop_locked();
    }

    /// Pop with a timeout.
    template <typename Rep, typename Period>
    [[nodiscard]] std::optional<T> pop_for(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queued_set_.empty() || shutdown_flag_; }))
            return std::nullopt;
        if (queued_set_.empty()) return std::nullopt;
        return pop_locked();
    }

    // -- cancellation --------------------------------------------------------

    /// Cancel every pending (queued) item.
    void cancel_pending() {
        std::lock_guard lock(mutex_);
        queued_set_.clear();
        boost_set_.clear();
        Heap empty_heap;
        std::swap(heap_, empty_heap);
    }

    // -- membership queries -------------------------------------------------

    [[nodiscard]] bool is_queued(const T& item) const {
        std::lock_guard lock(mutex_);
        return queued_set_.contains(item);
    }

    // -- size / state -------------------------------------------------------

    [[nodiscard]] std::size_t queued_count() const noexcept {
        std::lock_guard lock(mutex_);
        return queued_set_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        std::lock_guard lock(mutex_);
        return queued_set_.empty();
    }

    // -- lifecycle ----------------------------------------------------------

    /// Signal shutdown.  All blocked pop() calls will unblock.
    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            if (shutdown_flag_) return;
            shutdown_flag_ = true;
        }
        cv_.notify_all();
    }

private:
    using Set  = std::unordered_set<T, Hash, KeyEqual>;
    using Heap = std::priority_queue<T, std::vector<T>, Compare>;

    /// Must be called while holding mutex_ and with a non-empty queue.
    T pop_locked() {
        // Boosted items get popped first
        if (!boost_set_.empty()) {
            auto it = boost_set_.begin();
            T item = *it;
            boost_set_.erase(it);
            queued_set_.erase(item);
            return item;
        }
        // Skip heap items that were already popped via boost
        while (!heap_.empty()) {
            T item = heap_.top();
            heap_.pop();
            if (queued_set_.contains(item)) {
                queued_set_.erase(item);
                return item;
            }
            // Item was already consumed via boost — skip it
        }
        // Should not reach here if callers check empty()
        throw std::runtime_error("DeduplicatingPriorityQueue::pop_locked(): empty");
    }

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    Heap                    heap_;
    Set                     queued_set_;
    Set                     boost_set_;  // items to pop before the heap
    bool                    shutdown_flag_ = false;
};

// ---------------------------------------------------------------------------
// consume_loop  --  helper for worker / consumer threads
//
// Repeatedly pops items from the queue and passes them to `handler`.
// The loop exits when pop() throws due to shutdown.
// ---------------------------------------------------------------------------
template <typename T, typename Hash, typename KeyEqual, typename Compare, typename F>
void consume_loop(DeduplicatingPriorityQueue<T, Hash, KeyEqual, Compare>& queue, F&& handler) {
    for (;;) {
        T item;
        try {
            item = queue.pop();
        } catch (const std::runtime_error&) {
            // shutdown signalled
            return;
        }
        handler(item);
    }
}

/// Convenience overload that deduces default template parameters.
template <typename T, typename F>
void consume_loop(DeduplicatingPriorityQueue<T>& queue, F&& handler) {
    consume_loop<T, std::hash<T>, std::equal_to<T>, std::greater<T>, F>(
        queue, std::forward<F>(handler));
}

} // namespace utils
