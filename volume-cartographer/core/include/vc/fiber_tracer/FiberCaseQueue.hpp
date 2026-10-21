#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <string>

namespace vc::fiber_tracer {

class FiberConstructionCpuPermit;

// Bounded stackful cases pinned to their original OS worker. No GPU work.
// Dependencies must be externally completable or submitted with parallelFor;
// arbitrary dependencies on not-yet-admitted root cases are not supported.
class FiberCaseQueue {
public:
    FiberCaseQueue(size_t workers, size_t cases, std::function<void()> work,
                   size_t contextsPerWorker = 8);
    ~FiberCaseQueue();
    FiberCaseQueue(const FiberCaseQueue&) = delete;
    FiberCaseQueue& operator=(const FiberCaseQueue&) = delete;
    static bool supported() noexcept;
    static bool active() noexcept;
    static void notify();
    // Predicates must be nonblocking; successful predicates can consume a permit.
    static void await(std::function<bool()> ready);
    static void parallelFor(size_t count, const std::function<void(size_t)>& work);
    // Cooperative: interrupts await, not arbitrary user I/O/compute.
    void cancel(const std::string& reason = "fiber cases cancelled");
    void join();
private:
    friend class FiberConstructionCpuPermit;
    static FiberConstructionCpuPermit*& constructionPermit();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class FiberCaseCpuPermit {
public:
    explicit FiberCaseCpuPermit(std::counting_semaphore<>& pool) : pool_(pool) { acquire(); }
    ~FiberCaseCpuPermit() { if (held_) { pool_.release(); FiberCaseQueue::notify(); } }
    FiberCaseCpuPermit(const FiberCaseCpuPermit&) = delete;
    FiberCaseCpuPermit& operator=(const FiberCaseCpuPermit&) = delete;
    template<class Wait> void suspend(Wait&& wait) {
        if (!held_) throw std::logic_error("CPU permit already suspended");
        pool_.release();
        held_ = false;
        FiberCaseQueue::notify();
        // On failure unwind without reacquiring: another cancelled case must
        // not be required to release a permit before this one can unwind.
        wait();
        acquire();
    }
private:
    void acquire() {
        if (FiberCaseQueue::active()) FiberCaseQueue::await([this] { return pool_.try_acquire(); });
        else pool_.acquire();
        held_ = true;
    }
    std::counting_semaphore<>& pool_;
    bool held_ = false;
};

// Unlike ordinary TLS, this scope follows a suspended case, not its OS worker.
class FiberConstructionCpuPermit : public FiberCaseCpuPermit {
public:
    explicit FiberConstructionCpuPermit(std::counting_semaphore<>& pool)
        : FiberCaseCpuPermit(pool), previous_(current()) {
        FiberCaseQueue::constructionPermit() = this;
    }
    ~FiberConstructionCpuPermit() { FiberCaseQueue::constructionPermit() = previous_; }
    static FiberConstructionCpuPermit* current() { return FiberCaseQueue::constructionPermit(); }
private:
    FiberConstructionCpuPermit* previous_;
};
} // namespace vc::fiber_tracer
