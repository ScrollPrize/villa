#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace vc::render {

struct ChunkWorkPriority {
    bool interactive = false;
    bool activeView = false;
    int levelPriority = 0;
    float distanceSquared = 0.0f;
    int backgroundPriority = 0;
};

// Keyed worker queue used by regular chunk probes and fetch/decode work.
// Pending tasks can be reprioritized without submitting duplicate lambdas.
class ChunkRequestScheduler final {
public:
    using TaskId = std::uint64_t;
    using TaskGroup = std::uint64_t;

    explicit ChunkRequestScheduler(std::size_t workers,
                                   std::size_t interactiveBurst = 7);
    ~ChunkRequestScheduler();

    ChunkRequestScheduler(const ChunkRequestScheduler&) = delete;
    ChunkRequestScheduler& operator=(const ChunkRequestScheduler&) = delete;

    void submit(TaskId id,
                ChunkWorkPriority priority,
                TaskGroup group,
                std::uint64_t groupEpoch,
                std::function<void()> task);
    bool reprioritize(TaskId id, ChunkWorkPriority priority);
    void cancelGroupBefore(TaskGroup group, std::uint64_t minimumEpoch);

    [[nodiscard]] std::size_t pending() const;
    [[nodiscard]] std::size_t active() const noexcept;
    void waitIdle();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::render
