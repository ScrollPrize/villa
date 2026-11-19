#pragma once
// Progress and cancellation for a scaffold run another thread watches.
//
// A measured run reads the volume along every fiber and takes a minute, so the
// caller that started it must be able to ask where it is and to take it back.
// Both go through one object the caller owns and the stages report into: the
// stage names itself and the work it is about to do, counts that work off, and
// checks for cancellation where stopping is free - between fibers, between
// components, between violated constraints. A cancelled run throws
// ScaffoldCancelled out of the stage it was in and writes nothing.
//
// The monitor is optional everywhere: a null monitor pointer in the options is
// an uninterruptible run, which is what the command line and the tests want.
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vc::fiber_graph {

// Thrown by a stage that found the run cancelled. It is not a failure: the
// caller asked for it, and nothing was written.
class ScaffoldCancelled : public std::runtime_error {
public:
    ScaffoldCancelled() : std::runtime_error("scaffold run: cancelled") {}
};

class ScaffoldRunMonitor {
public:
    struct Progress {
        // The stage's own name, as it is shown to a person.
        std::string stage;
        // Work done and the work the stage announced, in that stage's own
        // units (fibers, components, constraints). A zero total means the
        // stage cannot say in advance.
        std::int64_t done = 0, total = 0;
        // Which stage of how many: the coarse progress. `stageCount` is what
        // the caller announced with expectStages(); 0 when it announced
        // nothing, and `stageIndex` counts from 1 once a stage is entered.
        std::int64_t stageIndex = 0, stageCount = 0;
    };

    // Asks for the run back. Safe from any thread; the run stops at its next
    // checkpoint rather than immediately.
    void cancel() { _cancelled.store(true, std::memory_order_relaxed); }
    [[nodiscard]] bool cancelled() const { return _cancelled.load(std::memory_order_relaxed); }
    // Throws ScaffoldCancelled when the caller asked for the run back.
    void checkpoint() const
    {
        if (cancelled()) throw ScaffoldCancelled();
    }

    // How many stages this run will enter, so the caller can be told which
    // one it is in. Stated before the first stage; a run that enters more
    // than it announced reports the larger count.
    void expectStages(std::int64_t count) { _stageCount.store(count, std::memory_order_relaxed); }

    // Enter a stage: the name it reports under and the units of work it is
    // about to do (0 when it cannot say).
    void enter(std::string stage, std::int64_t total = 0)
    {
        _done.store(0, std::memory_order_relaxed);
        _total.store(total, std::memory_order_relaxed);
        const auto index = _stageIndex.fetch_add(1, std::memory_order_relaxed) + 1;
        if (index > _stageCount.load(std::memory_order_relaxed))
            _stageCount.store(index, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> guard(_mutex);
        _stage = std::move(stage);
    }
    // Set the work done of the current stage, for a caller that counts its
    // own progress rather than incrementing (a progress callback). Safe from
    // any thread; never moves the count backwards.
    void report(std::int64_t done, std::int64_t total = 0)
    {
        if (total > 0) _total.store(total, std::memory_order_relaxed);
        auto previous = _done.load(std::memory_order_relaxed);
        while (previous < done && !_done.compare_exchange_weak(previous, done, std::memory_order_relaxed)) {
        }
    }
    // Count work of the current stage off. Safe from any thread.
    void advance(std::int64_t units = 1) { _done.fetch_add(units, std::memory_order_relaxed); }

    [[nodiscard]] Progress progress() const
    {
        Progress out;
        out.done = _done.load(std::memory_order_relaxed);
        out.total = _total.load(std::memory_order_relaxed);
        out.stageIndex = _stageIndex.load(std::memory_order_relaxed);
        out.stageCount = _stageCount.load(std::memory_order_relaxed);
        const std::lock_guard<std::mutex> guard(_mutex);
        out.stage = _stage;
        return out;
    }

private:
    std::atomic<bool> _cancelled{false};
    std::atomic<std::int64_t> _done{0}, _total{0}, _stageIndex{0}, _stageCount{0};
    mutable std::mutex _mutex;
    std::string _stage;
};

} // namespace vc::fiber_graph
