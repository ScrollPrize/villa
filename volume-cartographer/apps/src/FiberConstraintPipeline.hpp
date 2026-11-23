#pragma once
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stop_token>

// Single producer/single consumer; cancellation wakes a producer when the writer
// throws. Keep this before the producer jthread in scope so the thread joins first.
template<class T> class FiberConstraintPipeline {
    std::mutex mutex_;
    std::condition_variable_any ready_;
    std::deque<T> queue_;
    bool done_=false;
    std::exception_ptr failure_;
public:
    bool waitForSpace(std::stop_token stop) {
        std::unique_lock lock(mutex_);
        return ready_.wait(lock,stop,[&]{return queue_.size()<2;});
    }
    void push(T value) {
        {std::lock_guard lock(mutex_);queue_.push_back(std::move(value));}
        ready_.notify_all();
    }
    void finish(std::exception_ptr failure={}) {
        {std::lock_guard lock(mutex_);failure_=failure;done_=true;}
        ready_.notify_all();
    }
    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock,[&]{return !queue_.empty() || done_;});
        if(queue_.empty()) {
            if(failure_)std::rethrow_exception(failure_);
            return std::nullopt;
        }
        auto value=std::move(queue_.front());queue_.pop_front();
        lock.unlock();ready_.notify_all();return value;
    }
};
