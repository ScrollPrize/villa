// Fixed-size worker pool with a blocking parallel_for. A pool of zero workers runs
// every item on the calling thread. Concurrent parallel_for calls share the workers
// item by item.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace vcr {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads)
    {
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
    // Workers take one item at a time from the active jobs in turn, so concurrent
    // callers share the workers instead of the first job taking all of them.
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
        if (!workers_.empty()) {
            {
                std::lock_guard lock(mutex_);
                active_.push_back(job);
                active_count_.store(active_.size(), std::memory_order_relaxed);
            }
            cv_.notify_all();
        }
        while (run_one(*job)) {}
        {
            // Wait for the items, not the workers: a worker that picks the job after
            // every item has been taken just finds nothing to do.
            std::unique_lock lock(job->mutex);
            job->done.wait(lock, [&] { return job->remaining == 0; });
        }
        if (!workers_.empty()) {
            std::lock_guard lock(mutex_);
            retire(job.get());
        }
        if (job->error)
            std::rethrow_exception(job->error);
    }

private:
    struct Job {
        std::size_t n = 0;
        const std::function<void(std::size_t)>* fn = nullptr;  // valid while items remain to be taken
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::size_t remaining = 0;  // items not yet finished; guarded by mutex
        std::exception_ptr error;  // guarded by mutex
        std::mutex mutex;
        std::condition_variable done;
    };

    // Runs one item of the job; false when every item has been taken.
    static bool run_one(Job& job)
    {
        const std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= job.n)
            return false;
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
        return true;
    }

    // Removes a job from the active ones (if still there). Holds mutex_.
    void retire(const Job* job)
    {
        for (std::size_t k = 0; k < active_.size(); ++k)
            if (active_[k].get() == job) {
                active_.erase(active_.begin() + std::ptrdiff_t(k));
                if (k < cursor_)
                    --cursor_;
                break;
            }
        active_count_.store(active_.size(), std::memory_order_relaxed);
    }

    // The next active job with items left, in turn; null when stopping.
    std::shared_ptr<Job> next_job(std::unique_lock<std::mutex>& lock)
    {
        for (;;) {
            cv_.wait(lock, [&] { return stopping_ || !active_.empty(); });
            if (active_.empty())
                return nullptr;
            if (cursor_ >= active_.size())
                cursor_ = 0;
            auto job = active_[cursor_];
            if (job->next.load(std::memory_order_relaxed) >= job->n) {
                retire(job.get());
                continue;
            }
            ++cursor_;
            return job;
        }
    }

    void loop()
    {
        std::shared_ptr<Job> job;
        for (;;) {
            // The only active job: keep taking its items without the pool's lock.
            if (job && active_count_.load(std::memory_order_relaxed) == 1 && run_one(*job))
                continue;
            {
                std::unique_lock lock(mutex_);
                job = next_job(lock);
                if (!job)
                    return;
            }
            run_one(*job);
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::shared_ptr<Job>> active_;  // jobs that may have items left; guarded by mutex_
    std::size_t cursor_ = 0;                    // the next of active_ to take from; guarded by mutex_
    std::atomic<std::size_t> active_count_{0};  // active_.size(), read without the lock
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

// Sampling, transfer and projection share a bounded pool rather than each
// retaining their own workers. Small callers can still execute synchronously.
//
// Never destroyed, like every function-local static pool here: on Windows the
// static destructors of a DLL run during process exit, after the OS has ended
// every other thread, and joining the workers there never returns (#1686).
inline ThreadPool& samplingPool()
{
    static auto* pool = new ThreadPool(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    return *pool;
}

// The workers of every array asking for `threads` workers: arrays open at the same
// time share them rather than each adding its own, so many open arrays do not
// multiply threads. Each parallel_for still gets up to `threads` helpers, and its
// caller runs items too, so nested reads (a computed array reading another array
// on the same pool) always progress. A pool lives while an array holds it. A forked
// child gets new pools: the parent's workers do not run there.
inline std::shared_ptr<ThreadPool> sharedPool(std::size_t threads)
{
    static std::mutex mutex;
    // Never destroyed, so arrays opened during static destruction still find it.
    static auto* pools = new std::map<std::pair<pid_t, std::size_t>, std::weak_ptr<ThreadPool>>();
    const std::pair<pid_t, std::size_t> key{getpid(), threads};
    std::lock_guard lock(mutex);
    for (auto it = pools->begin(); it != pools->end();)
        it = it->second.expired() || it->first.first != key.first ? pools->erase(it) : std::next(it);
    if (auto pool = (*pools)[key].lock())
        return pool;
    auto pool = std::make_shared<ThreadPool>(threads);
    (*pools)[key] = pool;
    return pool;
}

}  // namespace vcr
