#pragma once

#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <utility>

namespace vc::core::util {

// Concurrent producers, one ordered consumer. The next required batch may
// bypass capacity by one: otherwise slow early work can deadlock a full queue.
// Payloads owned by blocked producers are additional, bounded by producer count.
template<class T> class OrderedBatchQueue {
    std::mutex mutex_;
    std::condition_variable_any ready_;
    std::map<std::size_t,T> pending_;
    std::size_t next_=0,capacity_,peak_=0;
    bool finished_=false;
    std::exception_ptr failure_;
public:
    explicit OrderedBatchQueue(std::size_t capacity):capacity_(capacity) {
        if(!capacity)throw std::invalid_argument("Ordered queue capacity must be positive");
    }
    void push(std::size_t index,T batch) {
        std::unique_lock lock(mutex_);
        ready_.wait(lock,[&]{return finished_ || failure_ || pending_.size()<capacity_ || index==next_;});
        if(failure_)std::rethrow_exception(failure_);
        if(finished_)throw std::runtime_error("Push after stream completion");
        if(index<next_ || !pending_.emplace(index,std::move(batch)).second)
            throw std::runtime_error("Duplicate streamed batch ID");
        peak_=std::max(peak_,pending_.size());
        ready_.notify_all();
    }
    std::optional<T> next(std::stop_token stop={}) {
        std::unique_lock lock(mutex_);
        if(!ready_.wait(lock,stop,[&]{return failure_ || finished_ || pending_.contains(next_);}))
            return std::nullopt;
        if(failure_)std::rethrow_exception(failure_);
        auto found=pending_.find(next_);
        if(found==pending_.end()) {
            if(!pending_.empty())throw std::runtime_error("Missing streamed batch ID");
            return std::nullopt;
        }
        T value=std::move(found->second);pending_.erase(found);++next_;
        ready_.notify_all();return value;
    }
    void finish(std::exception_ptr error={}) {
        std::lock_guard lock(mutex_);
        if(error && !failure_)failure_=error;
        finished_=true;ready_.notify_all();
    }
    std::size_t peakPending() {
        std::lock_guard lock(mutex_);return peak_;
    }
};
} // namespace vc::core::util
