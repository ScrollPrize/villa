#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace vc::render {

// Shared by schedulers which must publish related queue changes atomically.
// Workers may finish running work while an update is published, but cannot
// select their next task until the complete update is visible.
class ChunkRequestSelectionGate final {
public:
    ChunkRequestSelectionGate();
    ~ChunkRequestSelectionGate();

    ChunkRequestSelectionGate(const ChunkRequestSelectionGate&) = delete;
    ChunkRequestSelectionGate& operator=(const ChunkRequestSelectionGate&) = delete;

    void publish(const std::function<void()>& update);

private:
    friend class ChunkRequestScheduler;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ChunkWorkPriority {
    bool interactive = false;
    bool activeView = false;
    // Interactive work uses view-relative level offsets; larger is coarser.
    int levelPriority = 0;
    float distanceSquared = 0.0f;
    int backgroundPriority = 0;
};

// Keyed worker queue shared by regular chunk probe, source-read, and decode
// stages.
// Pending tasks can be reprioritized without submitting duplicate lambdas.
class ChunkRequestScheduler final {
public:
    using TaskId = std::uint64_t;
    using TaskGroup = std::uint64_t;

    explicit ChunkRequestScheduler(std::size_t workers,
                                   std::size_t interactiveBurst = 7,
                                   std::shared_ptr<ChunkRequestSelectionGate> selectionGate = {});
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
