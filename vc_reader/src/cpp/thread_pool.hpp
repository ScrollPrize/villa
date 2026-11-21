// Fixed-size worker pool with a blocking parallel_for. A pool of zero workers runs
// every item on the calling thread. Concurrent parallel_for calls share the workers
// item by item.
#pragma once

#include <algorithm>
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
#include <moodycamel/concurrentqueue.h>

#include <unistd.h>
#include "coordination_trace.hpp"

namespace vcr {

class ThreadPool {
    struct Job {
        std::shared_ptr<diagnostics::JobState> trace;
        diagnostics::SchedulerStage stage=diagnostics::SchedulerStage::Other;
        bool chunks=false;
        std::size_t n=0;
        const std::function<void(std::size_t)>* fn=nullptr;
        std::atomic<std::size_t> next{0},references{1};
        std::atomic<bool> failed{false};
        std::size_t remaining=0;  // guarded by mutex
        std::exception_ptr error;
        std::mutex mutex;
        std::condition_variable done;
        void retain() noexcept {references.fetch_add(1,std::memory_order_relaxed);}
        void release() noexcept {
            if(references.fetch_sub(1,std::memory_order_acq_rel)==1)delete this;
        }
    };
    struct ReleaseJob {void operator()(Job* job)const noexcept {job->release();}};
    using JobPtr=std::unique_ptr<Job,ReleaseJob>;

    static std::size_t checkedCapacity(std::size_t capacity) {
        if(!capacity)throw std::invalid_argument("reader queue capacity must be positive");
        return capacity;
    }
public:
    explicit ThreadPool(std::size_t threads,std::size_t queueCapacity=4096)
        :queue_(checkedCapacity(queueCapacity))
    {
        workers_.reserve(threads);
        try {
            for (std::size_t i = 0; i < threads; ++i)
                workers_.emplace_back([this] { loop(); });
        } catch(...) {stopWorkers();throw;}
    }

    ~ThreadPool()
    {
        stopWorkers();
        Job* job=nullptr;
        while(queue_.try_dequeue(job))job->release();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const noexcept { return workers_.size(); }
    bool queue_lock_free()const noexcept {return queue_.is_lock_free();}

    // Runs fn(i) for every i in [0, n) on the pool and the calling thread, and
    // waits. The first exception stops the remaining items and is rethrown.
    // Workers take one item at a time from the active jobs in turn, so concurrent
    // callers share the workers instead of the first job taking all of them.
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn)
    {
        diagnostics::schedulerRecord(diagnostics::SchedulerMetric::Calls,n);
        diagnostics::schedulerRecord(n==0?diagnostics::SchedulerMetric::Size0:n==1?diagnostics::SchedulerMetric::Size1:
            n<=4?diagnostics::SchedulerMetric::Size2to4:n<=16?diagnostics::SchedulerMetric::Size5to16:
            n<=64?diagnostics::SchedulerMetric::Size17to64:n<=256?diagnostics::SchedulerMetric::Size65to256:diagnostics::SchedulerMetric::Size257plus);
        if (n == 0)
            return;
        if (n == 1) {
            recordItem();
            diagnostics::Scope task(diagnostics::Task);
            fn(0);
            return;
        }
        run_job(n,fn);
    }

private:
    void run_job(std::size_t n,const std::function<void(std::size_t)>& fn) {
        JobPtr job(new Job);
        job->stage=diagnostics::schedulerStage;
        if(diagnostics::Trace::enabled()) {
            job->chunks=diagnostics::threadState()->flags.load(std::memory_order_relaxed)&diagnostics::ChunkBatch;
            // Detailed counters replace per-job trace registration, avoiding a
            // diagnostic registry lock on every measured submission.
            if(!diagnostics::schedulerProfiling())job->trace=diagnostics::Trace::instance().job(n,job->chunks);
        }
        job->n = n;
        job->fn = &fn;
        job->remaining = n;
        struct ActiveCall {
            std::atomic<std::size_t>& count;
            explicit ActiveCall(std::atomic<std::size_t>& c):count(c){count.fetch_add(1,std::memory_order_relaxed);}
            ~ActiveCall(){count.fetch_sub(1,std::memory_order_relaxed);}
        } active(active_calls_);
        if (!workers_.empty()) {
            if(publish(*job))diagnostics::schedulerRecord(diagnostics::SchedulerMetric::Submitted,n);
        }
        while (run_one(*job)) {}
        {
            // Wait for the items, not the workers: a worker that picks the job after
            // every item has been taken just finds nothing to do.
            diagnostics::SchedulerLock lock(job->mutex,diagnostics::SchedulerMetric::BatchLock);
            diagnostics::Scope waiting(diagnostics::BatchWait);
            lock.wait(job->done, [&] { return job->remaining == 0; },diagnostics::SchedulerMetric::BatchWait);
        }
        if (job->error)
            std::rethrow_exception(job->error);
    }

    static void recordItem() {
        if(diagnostics::schedulerProfiling())diagnostics::schedulerRecord(
            diagnostics::threadState()->flags.load(std::memory_order_relaxed)&diagnostics::Helper?
                diagnostics::SchedulerMetric::HelperItems:diagnostics::SchedulerMetric::CallerItems,1);
    }

    // Runs one item; false when every item has been taken.
    static bool run_one(Job& job)
    {
        const std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= job.n)
            return false;
        if(job.trace)job.trace->ready.fetch_sub(1,std::memory_order_relaxed);
        diagnostics::SchedulerStageScope stage(job.stage);
        recordItem();
        diagnostics::Scope task(diagnostics::Task|(job.chunks?diagnostics::ChunkTask:0));
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
        diagnostics::SchedulerLock lock(job.mutex,diagnostics::SchedulerMetric::CompletionLock);
        if (--job.remaining == 0)
            job.done.notify_all();
        return true;
    }

    // One ticket circulates for each published batch. Every ticket owns a
    // reference, including completed tickets awaiting dequeue. No caller waits
    // for queue space or removes its completed ticket from the queue.
    bool publish(Job& job) {
        diagnostics::SchedulerElapsed elapsed(diagnostics::SchedulerMetric::QueuePush);
        job.retain();
        if(!queue_.try_enqueue(&job)) {
            job.release();diagnostics::schedulerRecord(diagnostics::SchedulerMetric::QueueFull);return false;
        }
        available_.fetch_add(1,std::memory_order_release);
        available_.notify_one();return true;
    }

    void stopWorkers()noexcept {
        stopping_.store(true,std::memory_order_release);
        available_.fetch_add(1,std::memory_order_release);
        available_.notify_all();
        for(auto& worker:workers_)worker.join();
    }

    void loop()
    {
        moodycamel::ConsumerToken consumer(queue_);
        diagnostics::Scope helper(diagnostics::Helper);
        for (;;) {
            const auto observed=available_.load(std::memory_order_acquire);
            if(stopping_.load(std::memory_order_acquire))return;
            Job* raw=nullptr;
            bool found;
            {
                diagnostics::SchedulerElapsed elapsed(diagnostics::SchedulerMetric::QueuePop);
                found=queue_.try_dequeue(consumer,raw);
            }
            if(!found) {
                // Read the epoch before checking the queue: a concurrent publish
                // either supplies a ticket or changes the value we wait on.
                diagnostics::SchedulerElapsed idle(diagnostics::SchedulerMetric::HelperIdle);
                available_.wait(observed,std::memory_order_acquire);
                continue;
            }
            JobPtr job(raw);
            if(job->next.load(std::memory_order_relaxed)>=job->n) {
                diagnostics::schedulerRecord(diagnostics::SchedulerMetric::QueueStale);continue;
            }
            // Republish before executing so other helpers can join a single
            // large request. Atomic item claims, not tickets, own callbacks.
            publish(*job);
            while(run_one(*job)) {
                if(active_calls_.load(std::memory_order_relaxed)!=1)break;
            }
        }
    }

    std::vector<std::thread> workers_;
    // Implicit producer state belongs to the queue, not to caller-local tokens
    // that could outlive the pool. try_enqueue never grows the message blocks;
    // initial producer registration may allocate once per participating thread.
    moodycamel::ConcurrentQueue<Job*> queue_;
    std::atomic<std::uint64_t> available_{0};
    std::atomic<std::size_t> active_calls_{0};
    std::atomic<bool> stopping_{false};
};

// Sampling, transfer and projection share a bounded pool rather than each
// retaining their own workers. Small callers can still execute synchronously.
inline ThreadPool& samplingPool()
{
    static ThreadPool pool(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    return pool;
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
