#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace vc3d::line_annotation {

// One running immutable job and one latest replacement per session. The caller
// owns scheduling; workers retain this state, never the GUI session. Cancellation
// invalidates only our speculative job and never waits for its current I/O.
class LineModelWarmupQueue {
public:
    using Cancel = std::shared_ptr<std::atomic<bool>>;
    using Job = std::function<void(const Cancel&)>;

    // True means the caller must schedule one run(). Rapid replacements reuse
    // that worker, including while its previous job is opening model metadata.
    bool replace(Job job)
    {
        std::lock_guard lock(mutex_);
        if (activeCancel_)
            activeCancel_->store(true, std::memory_order_relaxed);
        pending_ = std::move(job);
        if (running_)
            return false;
        running_ = true;
        return true;
    }

    void cancel()
    {
        std::lock_guard lock(mutex_);
        if (activeCancel_)
            activeCancel_->store(true, std::memory_order_relaxed);
        pending_.reset();
    }

    void run() noexcept
    {
        for (;;) {
            Job job;
            Cancel cancel;
            {
                std::lock_guard lock(mutex_);
                if (!pending_) {
                    activeCancel_.reset();
                    running_ = false;
                    return;
                }
                job = std::move(*pending_);
                pending_.reset();
                cancel = std::make_shared<std::atomic<bool>>(false);
                activeCancel_ = cancel;
            }
            try {
                if (!cancel->load(std::memory_order_relaxed))
                    job(cancel);
            } catch (...) {
                // Failed speculation cannot poison the latest pending job or
                // escape the thread-pool entry point. Demand reports its errors.
            }
        }
    }

private:
    std::mutex mutex_;
    std::optional<Job> pending_;
    Cancel activeCancel_;
    bool running_ = false;
};

} // namespace vc3d::line_annotation
