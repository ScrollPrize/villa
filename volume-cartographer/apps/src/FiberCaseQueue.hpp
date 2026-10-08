#pragma once

// Stackful cases pinned to a bounded set of workers. Only synchronous callers
// outside a case use OS blocking; publication waits inside a case yield its stack.
#include <ucontext.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

class FiberCaseQueue {
    struct Task {
        ucontext_t context{};
        void* allocation = MAP_FAILED;
        size_t bytes = 0;
        std::function<bool()> ready;
        bool done = false;
        ~Task() { if (allocation != MAP_FAILED) munmap(allocation, bytes); }
    };
    struct Worker {
        ucontext_t scheduler{};
        Task* task = nullptr;
        const std::function<void()>* work = nullptr;
        std::exception_ptr error;
    };
    static inline thread_local Worker* current_ = nullptr;
    static inline std::mutex wakeMutex_;
    static inline std::condition_variable wake_;
    static inline std::atomic<size_t> epoch_{0};
    std::vector<std::jthread> workers_;
    std::mutex errorMutex_;
    std::exception_ptr error_;
    std::atomic<size_t> next_{0};
    std::atomic<bool> failed_{false};
    static void entry() {
        auto& w = *current_;
        try { (*w.work)(); } catch (...) { w.error = std::current_exception(); }
        w.task->done = true;
        // uc_link returns to the scheduler, on the same OS thread.
    }
public:
    static bool active() { return current_ != nullptr; }
    static void notify() { ++epoch_; wake_.notify_all(); }
    static void await(std::function<bool()> ready) {
        if (!active()) throw std::logic_error("case await outside scheduler");
        if (ready()) return;
        auto& w = *current_;
        w.task->ready = std::move(ready);
        if (swapcontext(&w.task->context, &w.scheduler) != 0)
            throw std::runtime_error("case context suspension failed");
        w.task->ready = {};
    }
    FiberCaseQueue(size_t workers, size_t tasks, std::function<void()> work,
                   size_t contextsPerWorker = 8) {
        if (!workers || !contextsPerWorker) throw std::invalid_argument("empty worker/context budget");
        for (size_t worker = 0; worker < std::min(workers, tasks); ++worker) {
            workers_.emplace_back([this, contextsPerWorker, tasks, work] {
                Worker state;
                state.work = &work;
                std::vector<std::unique_ptr<Task>> pending;
                try {
                    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
                    for (size_t i = 0; i < contextsPerWorker && next_.load() < tasks; ++i) {
                        auto task = std::make_unique<Task>();
                        constexpr size_t stackBytes = 8 * 1024 * 1024;
                        task->bytes = stackBytes + 2 * page;
                        task->allocation = mmap(nullptr, task->bytes, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                        if (task->allocation == MAP_FAILED ||
                            mprotect(static_cast<char*>(task->allocation) + page,
                                     stackBytes, PROT_READ | PROT_WRITE) != 0)
                            throw std::runtime_error("cannot reserve guarded case stack");
                        task->done = true; // First admission initializes this reusable stack.
                        pending.push_back(std::move(task));
                    }
                    while (!pending.empty() && !failed_.load()) {
                        const auto epoch = epoch_.load();
                        bool progressed = false;
                        for (auto& task : pending) {
                            if (task->done) {
                                if (next_.fetch_add(1) >= tasks) continue;
                                task->done = false;
                                if (getcontext(&task->context) != 0)
                                    throw std::runtime_error("cannot create case context");
                                task->context.uc_stack.ss_sp = static_cast<char*>(task->allocation) + page;
                                task->context.uc_stack.ss_size = task->bytes - 2 * page;
                                task->context.uc_link = &state.scheduler;
                                makecontext(&task->context, &FiberCaseQueue::entry, 0);
                            }
                            if (task->ready && !task->ready()) continue;
                            progressed = true;
                            state.task = task.get();
                            current_ = &state;
                            if (swapcontext(&state.scheduler, &task->context) != 0)
                                throw std::runtime_error("case context resume failed");
                            current_ = nullptr;
                            if (state.error) std::rethrow_exception(state.error);
                        }
                        if (next_.load() >= tasks)
                            std::erase_if(pending, [](const auto& task) { return task->done; });
                        if (!progressed && !pending.empty()) {
                            std::unique_lock lock(wakeMutex_);
                            wake_.wait_for(lock, std::chrono::milliseconds(10),
                                [&] { return epoch_.load() != epoch; });
                        }
                    }
                } catch (...) {
                    current_ = nullptr;
                    std::lock_guard lock(errorMutex_);
                    if (!error_) error_ = std::current_exception();
                    failed_ = true;
                    notify();
                }
            });
        }
    }
    void join() {
        workers_.clear();
        if (error_) std::rethrow_exception(error_);
    }
    ~FiberCaseQueue() { workers_.clear(); }
};
