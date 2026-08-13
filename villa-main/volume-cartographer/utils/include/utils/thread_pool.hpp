#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <latch>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <utility>
#include <algorithm>
#include <unordered_map>

namespace utils {

// ---------------------------------------------------------------------------
// ThreadPool
//
// Basic thread pool using std::jthread with cooperative shutdown via
// stop_token.  Extracted from VC3D's IOPool / RenderPool patterns.
// Replaces Qt QThreadPool for fire-and-forget and future-based submission.
// ---------------------------------------------------------------------------
class ThreadPool final {
public:
    explicit ThreadPool(std::size_t num_workers = 0)
        : active_{0}
    {
        if (num_workers == 0)
            num_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

        workers_.reserve(num_workers);
        for (std::size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this](std::stop_token stop) {
                worker_loop(stop);
            });
        }
    }

    ~ThreadPool() {
        // Request stop on all jthreads (auto-joined on destruction).
        for (auto& w : workers_)
            w.request_stop();
        cv_.notify_all();
        // jthread destructors join here.
    }

    // Submit a callable and its arguments, returning a future for the result.
    template <typename F, typename... Args>
    [[nodiscard]] auto submit(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind_front(std::forward<F>(func), std::forward<Args>(args)...)
        );
        auto fut = task->get_future();

        {
            std::lock_guard lk(mu_);
            queue_.emplace([t = std::move(task)]() mutable { (*t)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // Fire-and-forget submission (no future overhead).
    template <typename F>
    void enqueue(F&& func) {
        {
            std::lock_guard lk(mu_);
            queue_.emplace(std::forward<F>(func));
        }
        cv_.notify_one();
    }

    // Enqueue a fixed indexed batch with one queue lock and one completion wait.
    // The caller must not be one of this pool's workers.
    template <typename F>
    void run_indexed_batch(std::size_t task_count, F&& func) {
        if (task_count == 0)
            return;
        if (task_count > static_cast<std::size_t>(
                std::numeric_limits<std::ptrdiff_t>::max())) {
            throw std::overflow_error("thread-pool batch is too large");
        }

        using Function = std::decay_t<F>;
        struct BatchState {
            BatchState(std::size_t count, F&& input)
                : done(static_cast<std::ptrdiff_t>(count))
                , function(std::forward<F>(input))
            {
            }

            void run(std::size_t index) noexcept {
                try {
                    function(index);
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (!error)
                        error = std::current_exception();
                }
                done.count_down();
            }

            std::latch done;
            Function function;
            std::mutex error_mutex;
            std::exception_ptr error;
        };

        auto state = std::make_shared<BatchState>(
            task_count, std::forward<F>(func));
        std::size_t queued = 0;
        std::exception_ptr enqueue_error;
        {
            std::lock_guard lock(mu_);
            try {
                for (; queued < task_count; ++queued) {
                    queue_.emplace([state, index = queued] {
                        state->run(index);
                    });
                }
            } catch (...) {
                enqueue_error = std::current_exception();
            }
        }
        if (queued != task_count) {
            state->done.count_down(static_cast<std::ptrdiff_t>(task_count - queued));
        }
        cv_.notify_all();
        state->done.wait();
        if (enqueue_error)
            std::rethrow_exception(enqueue_error);
        if (state->error)
            std::rethrow_exception(state->error);
    }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return workers_.size();
    }

    [[nodiscard]] std::size_t pending() const noexcept {
        std::lock_guard lk(mu_);
        return queue_.size();
    }

    [[nodiscard]] std::size_t active() const noexcept {
        return active_.load(std::memory_order_relaxed);
    }

    // Block until queue is drained and no workers are active.
    void wait_idle() {
        std::unique_lock lk(mu_);
        idle_cv_.wait(lk, [this] {
            return queue_.empty() && active_.load(std::memory_order_acquire) == 0;
        });
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    void worker_loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] {
                    return stop.stop_requested() || !queue_.empty();
                });
                if (stop.stop_requested() && queue_.empty())
                    return;
                if (queue_.empty())
                    continue;
                task = std::move(queue_.front());
                queue_.pop();
                active_.fetch_add(1, std::memory_order_acq_rel);
            }
            task();
            active_.fetch_sub(1, std::memory_order_release);
            idle_cv_.notify_all();
        }
    }

    std::queue<std::function<void()>>    queue_;
    mutable std::mutex                   mu_;
    std::condition_variable              cv_;
    std::condition_variable              idle_cv_;
    std::atomic<std::size_t>             active_;
    std::vector<std::jthread>            workers_;  // Must be last: destroyed first to join threads
};

// ---------------------------------------------------------------------------
// PriorityThreadPool
//
// Thread pool with per-task priority and epoch-based staleness filtering.
// Extracted from VC3D's edt ThreadPool pattern where view-change epochs
// allow discarding obsolete work before it executes.
//
// Priority: lower numeric value = higher priority (dequeued first).
// Epoch:    tasks submitted with an epoch older than the current epoch
//           are silently discarded when they reach the front of the queue.
// ---------------------------------------------------------------------------
class PriorityThreadPool final {
public:
    using Priority = std::int64_t;
    using TaskGroup = std::uint64_t;

    explicit PriorityThreadPool(std::size_t num_workers = 0)
        : epoch_{0}, seq_{0}, active_{0}
    {
        if (num_workers == 0)
            num_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

        workers_.reserve(num_workers);
        for (std::size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this](std::stop_token stop) {
                worker_loop(stop);
            });
        }
    }

    ~PriorityThreadPool() {
        for (auto& w : workers_)
            w.request_stop();
        cv_.notify_all();
    }

    // Submit with priority (no epoch check — always valid).
    template <typename F>
    void submit(Priority priority, F&& func) {
        {
            std::lock_guard lk(mu_);
            auto seq = seq_++;
            queue_.push(Entry{
                priority, seq, std::uint64_t(-1), 0, 0,
                std::function<void()>(std::forward<F>(func))
            });
        }
        cv_.notify_one();
    }

    // Submit with priority and epoch (skipped if stale when dequeued).
    template <typename F>
    void submit(Priority priority, std::uint64_t epoch, F&& func) {
        {
            std::lock_guard lk(mu_);
            auto seq = seq_++;
            queue_.push(Entry{
                priority, seq, epoch, 0, 0,
                std::function<void()>(std::forward<F>(func))
            });
        }
        cv_.notify_one();
    }

    // Submit work owned by one independently-cancellable producer. The group
    // epoch is checked both here and when a worker dequeues the task. That
    // closes the race where an active first-stage task hands stale work to a
    // second pool just after its owner cancels a view.
    template <typename F>
    void submit(Priority priority,
                TaskGroup group,
                std::uint64_t task_epoch,
                F&& func) {
        {
            std::lock_guard lk(mu_);
            const auto minimum = group_epochs_.find(group);
            if (minimum != group_epochs_.end() &&
                task_epoch < minimum->second) {
                return;
            }
            auto seq = seq_++;
            queue_.push(Entry{
                priority, seq, std::uint64_t(-1), group, task_epoch,
                std::function<void()>(std::forward<F>(func))
            });
            ++group_pending_[group];
        }
        cv_.notify_one();
    }

    void set_epoch(std::uint64_t epoch) noexcept {
        epoch_.store(epoch, std::memory_order_release);
    }

    [[nodiscard]] std::uint64_t epoch() const noexcept {
        return epoch_.load(std::memory_order_acquire);
    }

    // Discard all pending tasks; in-flight tasks continue to completion.
    void cancel_pending() {
        std::lock_guard lk(mu_);
        queue_ = decltype(queue_){};
        group_pending_.clear();
        idle_cv_.notify_all();
    }

    // Drop only stale work belonging to `group`; other users of this shared
    // executor are untouched. Compaction is immediate, so cancelled tasks
    // release their captures now instead of occupying RAM until the queue
    // eventually drains.
    void cancel_group_before(TaskGroup group, std::uint64_t minimum_epoch) {
        std::lock_guard lk(mu_);
        auto& accepted = group_epochs_[group];
        accepted = std::max(accepted, minimum_epoch);

        PQ kept;
        group_pending_.clear();
        while (!queue_.empty()) {
            Entry entry = queue_.top();
            queue_.pop();
            if (entry.group == group && entry.task_epoch < accepted)
                continue;
            if (entry.group != 0)
                ++group_pending_[entry.group];
            kept.push(std::move(entry));
        }
        queue_ = std::move(kept);
        if (queue_.empty() && active_.load(std::memory_order_acquire) == 0)
            idle_cv_.notify_all();
    }

    [[nodiscard]] std::size_t pending() const noexcept {
        std::lock_guard lk(mu_);
        return queue_.size();
    }

    [[nodiscard]] std::size_t pending(TaskGroup group) const noexcept {
        std::lock_guard lk(mu_);
        const auto found = group_pending_.find(group);
        return found == group_pending_.end() ? 0 : found->second;
    }

    [[nodiscard]] std::size_t active() const noexcept {
        return active_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return workers_.size();
    }

    void wait_idle() {
        std::unique_lock lk(mu_);
        idle_cv_.wait(lk, [this] {
            return queue_.empty() && active_.load(std::memory_order_acquire) == 0;
        });
    }

    PriorityThreadPool(const PriorityThreadPool&) = delete;
    PriorityThreadPool& operator=(const PriorityThreadPool&) = delete;

private:
    struct Entry {
        Priority                priority;
        std::uint64_t           seq;     // FIFO tiebreaker within same priority
        std::uint64_t           epoch;   // uint64_t(-1) means "always valid"
        TaskGroup               group;
        std::uint64_t           task_epoch;
        std::function<void()>   func;

        // Min-heap: lower priority value wins; ties broken by lower seq.
        bool operator>(const Entry& rhs) const noexcept {
            if (priority != rhs.priority) return priority > rhs.priority;
            return seq > rhs.seq;
        }
    };

    void worker_loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] {
                    return stop.stop_requested() || !queue_.empty();
                });
                if (stop.stop_requested() && queue_.empty())
                    return;

                // Skip stale entries.
                auto cur = epoch_.load(std::memory_order_acquire);
                while (!queue_.empty()) {
                    auto& top = queue_.top();
                    const auto group_epoch = group_epochs_.find(top.group);
                    const bool stale_group =
                        top.group != 0 &&
                        group_epoch != group_epochs_.end() &&
                        top.task_epoch < group_epoch->second;
                    const bool stale_global =
                        top.epoch != std::uint64_t(-1) && top.epoch < cur;
                    if (stale_group || stale_global) {
                        if (top.group != 0) {
                            auto pending = group_pending_.find(top.group);
                            if (pending != group_pending_.end() &&
                                --pending->second == 0) {
                                group_pending_.erase(pending);
                            }
                        }
                        queue_.pop();
                        continue;
                    }
                    break;
                }
                if (queue_.empty()) {
                    if (active_.load(std::memory_order_acquire) == 0)
                        idle_cv_.notify_all();
                    continue;
                }

                // const_cast needed: priority_queue::top() returns const ref,
                // but we need to move the function out before popping.
                Entry& next = const_cast<Entry&>(queue_.top());
                if (next.group != 0) {
                    auto pending = group_pending_.find(next.group);
                    if (pending != group_pending_.end() &&
                        --pending->second == 0) {
                        group_pending_.erase(pending);
                    }
                }
                task = std::move(next.func);
                queue_.pop();
                active_.fetch_add(1, std::memory_order_acq_rel);
            }
            task();
            active_.fetch_sub(1, std::memory_order_release);
            idle_cv_.notify_all();
        }
    }

    using PQ = std::priority_queue<Entry, std::vector<Entry>, std::greater<>>;

    PQ                          queue_;
    std::unordered_map<TaskGroup, std::uint64_t> group_epochs_;
    std::unordered_map<TaskGroup, std::size_t> group_pending_;
    mutable std::mutex          mu_;
    std::condition_variable     cv_;
    std::condition_variable     idle_cv_;
    std::atomic<std::uint64_t>  epoch_;
    std::uint64_t               seq_;
    std::atomic<std::size_t>    active_;
    std::vector<std::jthread>   workers_;  // Must be last: destroyed first to join threads
};

} // namespace utils
