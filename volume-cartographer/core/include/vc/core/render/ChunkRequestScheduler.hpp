#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

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
    // ChunkCache adds 100 for a source's terminal pyramid level.
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

    struct AdaptiveConcurrency {
        std::size_t minimum = 2;
        std::size_t maximum = 64;
        // Rolling successful-transfer window used for displayed bandwidth.
        std::size_t successfulSamplesPerWorker = 4;
        double minimumEpochSeconds = 2.0;
        double maximumEpochSeconds = 5.0;
        double unstableProbeIntervalSeconds = 60.0;
        double stableProbeIntervalSeconds = 300.0;
        double minimumStabilityObservationSeconds = 300.0;
        double bandwidthChangeRatio = 2.0;
        double throughputGainRatio = 1.08;
        double maximumLatencyInflation = 1.75;
        std::size_t initialProbeMultiplier = 4;
        std::size_t refinementProbeMultiplier = 2;
        std::size_t continuousSearchTurns = 5;
        double lowerThroughputRetention = 0.95;
        double lowerLatencyRatio = 0.85;
    };

    // Stable capacity data that may be carried across process runs. Probe
    // phases, epochs, and stability timing are intentionally not included.
    struct AdaptiveState {
        std::size_t settledAdmissionLimit = 0;
        double longTermBytesPerSecond = 0.0;
        std::size_t maximumSaturatedParallelism = 0;
        double saturatedBytesPerSecondPerWorker = 0.0;
    };

    struct TransferStats {
        std::size_t admissionLimit = 0;
        double bytesPerSecond = 0.0;
        double averageChunkBytes = 0.0;
        std::size_t sampleCount = 0;
        bool adaptive = false;
        std::size_t targetAdmissionLimit = 0;
        double longTermBytesPerSecond = 0.0;
        double probeIntervalSeconds = 0.0;
        bool probing = false;
    };

    explicit ChunkRequestScheduler(std::size_t workers,
                                   std::size_t interactiveBurst = 7,
                                   std::shared_ptr<ChunkRequestSelectionGate> selectionGate = {},
                                   std::optional<AdaptiveConcurrency> adaptiveConcurrency = {},
                                   std::optional<AdaptiveState> initialAdaptiveState = {});
    ~ChunkRequestScheduler();

    ChunkRequestScheduler(const ChunkRequestScheduler&) = delete;
    ChunkRequestScheduler& operator=(const ChunkRequestScheduler&) = delete;

    // Changes admission on the existing worker pool. Pending and running work
    // is retained unchanged. The requested bounds must fit the physical worker
    // capacity supplied to the constructor.
    void configureConcurrency(
        std::size_t fixedAdmissionLimit,
        std::optional<AdaptiveConcurrency> adaptiveConcurrency = {});
    [[nodiscard]] std::size_t workerCapacity() const noexcept;

    void submit(TaskId id,
                ChunkWorkPriority priority,
                TaskGroup group,
                std::uint64_t groupEpoch,
                std::function<void()> task);
    bool reprioritize(TaskId id, ChunkWorkPriority priority);
    // Cancels a task only while it is still pending. Running work is allowed
    // to complete its current stage.
    bool cancel(TaskId id);
    void cancelGroupBefore(TaskGroup group, std::uint64_t minimumEpoch);

    [[nodiscard]] std::size_t pending() const;
    [[nodiscard]] std::size_t active() const noexcept;
    void recordSuccessfulTransfer(
        std::size_t encodedBytes,
        std::chrono::steady_clock::time_point started,
        std::chrono::steady_clock::time_point completed);
    [[nodiscard]] TransferStats transferStats() const;
    [[nodiscard]] std::optional<AdaptiveState> adaptiveState() const;
    void waitIdle();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::render
