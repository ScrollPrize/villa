#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <concepts>
#include <type_traits>
#include <utility>
#include <optional>
#include <ranges>
#include <algorithm>
#include <numeric>

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
    using Priority = std::int32_t;

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
                priority, seq, std::uint64_t(-1),
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
                priority, seq, epoch,
                std::function<void()>(std::forward<F>(func))
            });
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
    }

    [[nodiscard]] std::size_t pending() const noexcept {
        std::lock_guard lk(mu_);
        return queue_.size();
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
                    if (top.epoch != std::uint64_t(-1) && top.epoch < cur) {
                        queue_.pop();
                        continue;
                    }
                    break;
                }
                if (queue_.empty())
                    continue;

                // const_cast needed: priority_queue::top() returns const ref,
                // but we need to move the function out before popping.
                task = std::move(const_cast<Entry&>(queue_.top()).func);
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
    mutable std::mutex          mu_;
    std::condition_variable     cv_;
    std::condition_variable     idle_cv_;
    std::atomic<std::uint64_t>  epoch_;
    std::uint64_t               seq_;
    std::atomic<std::size_t>    active_;
    std::vector<std::jthread>   workers_;  // Must be last: destroyed first to join threads
};

// ---------------------------------------------------------------------------
// Parallel algorithms — thin wrappers over ThreadPool for bulk work.
// ---------------------------------------------------------------------------

namespace detail {

inline std::size_t default_chunk_size(std::size_t count, std::size_t workers) noexcept {
    if (workers == 0) workers = 1;
    auto cs = count / (4 * workers);
    return cs > 0 ? cs : 1;
}

} // namespace detail

// Parallel for over [begin, end).
// func is shared across worker lambdas to stay valid if an exception
// causes early return (abandoned futures would otherwise hold dangling refs).
template <typename F>
void parallel_for(ThreadPool& pool, std::size_t begin, std::size_t end,
                  F&& func, std::size_t chunk_size = 0)
{
    if (begin >= end) return;
    const auto count = end - begin;
    if (chunk_size == 0)
        chunk_size = detail::default_chunk_size(count, pool.worker_count());

    auto fn = std::make_shared<std::decay_t<F>>(std::forward<F>(func));

    std::vector<std::future<void>> futures;
    futures.reserve((count + chunk_size - 1) / chunk_size);

    for (std::size_t lo = begin; lo < end; lo += chunk_size) {
        auto hi = std::min(lo + chunk_size, end);
        futures.push_back(pool.submit([fn, lo, hi] {
            for (std::size_t i = lo; i < hi; ++i)
                (*fn)(i);
        }));
    }
    for (auto& f : futures)
        f.get();
}

// Parallel for_each over a random-access range.
template <std::ranges::random_access_range R, typename F>
void parallel_for_each(ThreadPool& pool, R&& range, F&& func,
                       std::size_t chunk_size = 0)
{
    const auto sz = static_cast<std::size_t>(std::ranges::size(range));
    if (sz == 0) return;
    if (chunk_size == 0)
        chunk_size = detail::default_chunk_size(sz, pool.worker_count());

    auto fn = std::make_shared<std::decay_t<F>>(std::forward<F>(func));
    auto it = std::ranges::begin(range);

    std::vector<std::future<void>> futures;
    futures.reserve((sz + chunk_size - 1) / chunk_size);

    for (std::size_t lo = 0; lo < sz; lo += chunk_size) {
        auto hi = std::min(lo + chunk_size, sz);
        futures.push_back(pool.submit([fn, it, lo, hi] {
            for (std::size_t i = lo; i < hi; ++i)
                (*fn)(*(it + static_cast<std::ptrdiff_t>(i)));
        }));
    }
    for (auto& f : futures)
        f.get();
}

// Parallel transform-reduce over a random-access range.
template <std::ranges::random_access_range R, typename T,
          typename ReduceOp, typename TransformOp>
[[nodiscard]] T parallel_reduce(ThreadPool& pool, R&& range, T init,
                                ReduceOp&& reduce, TransformOp&& transform,
                                std::size_t chunk_size = 0)
{
    const auto sz = static_cast<std::size_t>(std::ranges::size(range));
    if (sz == 0) return init;
    if (chunk_size == 0)
        chunk_size = detail::default_chunk_size(sz, pool.worker_count());

    auto red = std::make_shared<std::decay_t<ReduceOp>>(std::forward<ReduceOp>(reduce));
    auto xfm = std::make_shared<std::decay_t<TransformOp>>(std::forward<TransformOp>(transform));
    auto it = std::ranges::begin(range);

    std::vector<std::future<T>> futures;
    futures.reserve((sz + chunk_size - 1) / chunk_size);

    for (std::size_t lo = 0; lo < sz; lo += chunk_size) {
        auto hi = std::min(lo + chunk_size, sz);
        futures.push_back(pool.submit(
            [red, xfm, it, lo, hi] {
                auto acc = (*xfm)(*(it + static_cast<std::ptrdiff_t>(lo)));
                for (std::size_t i = lo + 1; i < hi; ++i)
                    acc = (*red)(std::move(acc),
                                 (*xfm)(*(it + static_cast<std::ptrdiff_t>(i))));
                return acc;
            }
        ));
    }

    T result = std::move(init);
    for (auto& f : futures)
        result = (*red)(std::move(result), f.get());
    return result;
}

} // namespace utils
