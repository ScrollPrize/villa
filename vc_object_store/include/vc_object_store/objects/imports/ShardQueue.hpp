#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace vc::objects::imports {
// One bounded queue per destination, independent of source-reader partitioning.
// Batches keep synchronization out of the per-record SQLite hot path.
class StreamFailure {
    std::mutex mutex_;
    std::exception_ptr error_;
    std::atomic<bool> failed_{false};
public:
    void fail(std::exception_ptr error) { std::lock_guard lock(mutex_); if(!error_)error_=error; failed_.store(true,std::memory_order_release); }
    void check() { if(!failed_.load(std::memory_order_acquire))return; std::lock_guard lock(mutex_); std::rethrow_exception(error_); }
};
template<class T> class ShardQueues {
    struct Batch { std::vector<T> rows; size_t bytes; };
    struct Queue {
        std::mutex mutex; std::condition_variable cv;
        std::deque<Batch> batches; size_t bytes=0; bool closed=false;
    };
    std::vector<std::unique_ptr<Queue>> queues_;
    size_t budget_;
    StreamFailure& failure_;
public:
    ShardQueues(size_t count,size_t budget,StreamFailure& failure):budget_(budget),failure_(failure) {
        for(size_t i=0;i<count;++i)queues_.push_back(std::make_unique<Queue>());
    }
    void push(size_t shard,std::vector<T> rows,size_t bytes) {
        if(rows.empty())return;
        auto& q=*queues_.at(shard);std::unique_lock lock(q.mutex);
        // A single oversize batch must be able to make progress.
        while(q.bytes && (bytes>budget_ || q.bytes>budget_-bytes)) {
            failure_.check();q.cv.wait_for(lock,std::chrono::milliseconds(50));
        }
        failure_.check();q.bytes+=bytes;q.batches.push_back({std::move(rows),bytes});q.cv.notify_all();
    }
    template<class Check> bool pop(size_t shard,std::vector<T>& rows,Check&& check) {
        // Destruct previous payloads outside the queue lock. This can free
        // megabytes and must not serialize every producer targeting this shard.
        rows.clear();
        auto& q=*queues_.at(shard);std::unique_lock lock(q.mutex);
        while(q.batches.empty()&&!q.closed) {
            failure_.check();check();q.cv.wait_for(lock,std::chrono::milliseconds(50));
        }
        failure_.check();if(q.batches.empty())return false;
        auto batch=std::move(q.batches.front());q.batches.pop_front();q.bytes-=batch.bytes;
        rows=std::move(batch.rows);q.cv.notify_all();return true;
    }
    void close() { for(auto& p:queues_){std::lock_guard lock(p->mutex);p->closed=true;p->cv.notify_all();} }
};
}
