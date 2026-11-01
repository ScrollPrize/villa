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

} // namespace utils
