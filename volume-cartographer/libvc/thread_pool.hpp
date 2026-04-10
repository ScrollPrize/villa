#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vc {

class ThreadPool {
    std::queue<std::function<void()>> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::atomic<size_t> active_{0};
    std::vector<std::jthread> workers_;

    void loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] { return stop.stop_requested() || !queue_.empty(); });
                if (stop.stop_requested() && queue_.empty()) return;
                if (queue_.empty()) continue;
                task = std::move(queue_.front());
                queue_.pop();
                active_.fetch_add(1, std::memory_order_acq_rel);
            }
            task();
            active_.fetch_sub(1, std::memory_order_release);
            idle_cv_.notify_all();
        }
    }

public:
    explicit ThreadPool(size_t n = 0) {
        if (!n) n = std::max<size_t>(1, std::thread::hardware_concurrency());
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this](std::stop_token s) { loop(s); });
    }

    ~ThreadPool() {
        for (auto& w : workers_) w.request_stop();
        cv_.notify_all();
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind_front(std::forward<F>(f), std::forward<Args>(args)...));
        auto fut = task->get_future();
        { std::lock_guard lk(mu_); queue_.emplace([t = std::move(task)] { (*t)(); }); }
        cv_.notify_one();
        return fut;
    }

    template<typename F>
    void enqueue(F&& f) {
        { std::lock_guard lk(mu_); queue_.emplace(std::forward<F>(f)); }
        cv_.notify_one();
    }

    void wait_idle() {
        std::unique_lock lk(mu_);
        idle_cv_.wait(lk, [this] { return queue_.empty() && active_.load() == 0; });
    }

    size_t worker_count() const { return workers_.size(); }
    size_t pending() const { std::lock_guard lk(mu_); return queue_.size(); }
    size_t active() const { return active_.load(std::memory_order_relaxed); }
};

class PriorityThreadPool {
    struct Entry {
        int32_t priority;
        uint64_t seq, epoch;
        std::function<void()> func;
        bool operator>(const Entry& r) const {
            return priority != r.priority ? priority > r.priority : seq > r.seq;
        }
    };

    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::atomic<uint64_t> epoch_{0};
    uint64_t seq_ = 0;
    std::atomic<size_t> active_{0};
    std::vector<std::jthread> workers_;

    void loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] { return stop.stop_requested() || !queue_.empty(); });
                if (stop.stop_requested() && queue_.empty()) return;
                auto cur = epoch_.load(std::memory_order_acquire);
                while (!queue_.empty()) {
                    auto& top = queue_.top();
                    if (top.epoch != uint64_t(-1) && top.epoch < cur) { queue_.pop(); continue; }
                    break;
                }
                if (queue_.empty()) continue;
                task = std::move(const_cast<Entry&>(queue_.top()).func);
                queue_.pop();
                active_.fetch_add(1, std::memory_order_acq_rel);
            }
            task();
            active_.fetch_sub(1, std::memory_order_release);
            idle_cv_.notify_all();
        }
    }

public:
    explicit PriorityThreadPool(size_t n = 0) {
        if (!n) n = std::max<size_t>(1, std::thread::hardware_concurrency());
        workers_.reserve(n);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this](std::stop_token s) { loop(s); });
    }

    ~PriorityThreadPool() {
        for (auto& w : workers_) w.request_stop();
        cv_.notify_all();
    }

    template<typename F>
    void submit(int32_t priority, F&& f) {
        { std::lock_guard lk(mu_); queue_.push({priority, seq_++, uint64_t(-1), std::forward<F>(f)}); }
        cv_.notify_one();
    }

    template<typename F>
    void submit(int32_t priority, uint64_t epoch, F&& f) {
        { std::lock_guard lk(mu_); queue_.push({priority, seq_++, epoch, std::forward<F>(f)}); }
        cv_.notify_one();
    }

    void set_epoch(uint64_t e) { epoch_.store(e, std::memory_order_release); }
    uint64_t epoch() const { return epoch_.load(std::memory_order_acquire); }

    void cancel_pending() { std::lock_guard lk(mu_); queue_ = {}; }
    void wait_idle() {
        std::unique_lock lk(mu_);
        idle_cv_.wait(lk, [this] { return queue_.empty() && active_.load() == 0; });
    }

    size_t worker_count() const { return workers_.size(); }
    size_t pending() const { std::lock_guard lk(mu_); return queue_.size(); }
    size_t active() const { return active_.load(std::memory_order_relaxed); }
};

} // namespace vc
