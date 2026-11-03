// Fixed-size worker pool with a blocking parallel_for.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vcr {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads)
    {
        threads = threads ? threads : 1;
        workers_.reserve(threads);
        for (std::size_t i = 0; i < threads; ++i)
            workers_.emplace_back([this] { loop(); });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const noexcept { return workers_.size(); }

    // Runs fn(i) for every i in [0, n) on the pool and the calling thread, and
    // waits. The first exception stops the remaining items and is rethrown.
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn)
    {
        if (n == 0)
            return;
        if (n == 1) {
            fn(0);
            return;
        }
        auto job = std::make_shared<Job>();
        job->n = n;
        job->fn = &fn;
        job->remaining = n;
        const std::size_t helpers = std::min(n - 1, workers_.size());
        {
            std::lock_guard lock(mutex_);
            for (std::size_t i = 0; i < helpers; ++i)
                queue_.push_back(job);
        }
        cv_.notify_all();
        run(*job);
        // Wait for the items, not the helpers: a helper that wakes after every
        // item has been taken just finds nothing to do, and must not delay us.
        std::unique_lock lock(job->mutex);
        job->done.wait(lock, [&] { return job->remaining == 0; });
        if (job->error)
            std::rethrow_exception(job->error);
    }

private:
    struct Job {
        std::size_t n = 0;
        const std::function<void(std::size_t)>* fn = nullptr;  // valid while remaining > 0
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::size_t remaining = 0;  // items not yet finished; guarded by mutex
        std::exception_ptr error;  // guarded by mutex
        std::mutex mutex;
        std::condition_variable done;
    };

    static void run(Job& job)
    {
        for (;;) {
            const std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= job.n)
                return;
            if (!job.failed.load(std::memory_order_relaxed)) {  // after a failure, items are only counted
                try {
                    (*job.fn)(i);
                } catch (...) {
                    std::lock_guard lock(job.mutex);
                    if (!job.error)
                        job.error = std::current_exception();
                    job.failed = true;
                }
            }
            std::lock_guard lock(job.mutex);
            if (--job.remaining == 0)
                job.done.notify_all();
        }
    }

    void loop()
    {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty())
                    return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            run(*job);
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

}  // namespace vcr
