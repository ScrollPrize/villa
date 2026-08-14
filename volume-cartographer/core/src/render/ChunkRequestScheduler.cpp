#include "vc/core/render/ChunkRequestScheduler.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <set>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vc::render {

namespace {

float finiteDistance(float value) noexcept
{
    return std::isfinite(value) ? value : std::numeric_limits<float>::infinity();
}

} // namespace

struct ChunkRequestSelectionGate::Impl {
    std::mutex mutex;
};

ChunkRequestSelectionGate::ChunkRequestSelectionGate()
    : impl_(std::make_unique<Impl>())
{
}

ChunkRequestSelectionGate::~ChunkRequestSelectionGate() = default;

void ChunkRequestSelectionGate::publish(const std::function<void()>& update)
{
    if (!update)
        return;
    std::lock_guard lock(impl_->mutex);
    update();
}

struct ChunkRequestScheduler::Impl {
    struct Item {
        TaskId id = 0;
        ChunkWorkPriority priority;
        TaskGroup group = 0;
        std::uint64_t groupEpoch = 0;
        std::uint64_t sequence = 0;
        std::function<void()> task;
    };

    struct GuiLess {
        bool operator()(const std::shared_ptr<Item>& lhs,
                        const std::shared_ptr<Item>& rhs) const noexcept
        {
            if (lhs->priority.levelPriority != rhs->priority.levelPriority)
                return lhs->priority.levelPriority > rhs->priority.levelPriority;
            if (lhs->priority.activeView != rhs->priority.activeView)
                return lhs->priority.activeView > rhs->priority.activeView;
            const float ld = finiteDistance(lhs->priority.distanceSquared);
            const float rd = finiteDistance(rhs->priority.distanceSquared);
            if (ld != rd)
                return ld < rd;
            if (lhs->sequence != rhs->sequence)
                return lhs->sequence < rhs->sequence;
            return lhs->id < rhs->id;
        }
    };

    struct BackgroundLess {
        bool operator()(const std::shared_ptr<Item>& lhs,
                        const std::shared_ptr<Item>& rhs) const noexcept
        {
            if (lhs->priority.backgroundPriority != rhs->priority.backgroundPriority)
                return lhs->priority.backgroundPriority < rhs->priority.backgroundPriority;
            if (lhs->sequence != rhs->sequence)
                return lhs->sequence < rhs->sequence;
            return lhs->id < rhs->id;
        }
    };

    using GuiQueue = std::set<std::shared_ptr<Item>, GuiLess>;
    using BackgroundQueue = std::set<std::shared_ptr<Item>, BackgroundLess>;

    struct Location {
        bool interactive = false;
        GuiQueue::iterator gui;
        BackgroundQueue::iterator background;
    };

    struct TransferSample {
        std::size_t bytes = 0;
        std::chrono::steady_clock::time_point started;
        std::chrono::steady_clock::time_point completed;
        std::size_t admission = 1;
        bool saturated = true;
    };

    struct EpochMeasurement {
        double bytesPerSecond = 0.0;
        double latencyP90Seconds = 0.0;
        double observationSeconds = 0.0;
        std::chrono::steady_clock::time_point completed;
    };

    enum class ProbePhase {
        Monitor,
        Up,
        BaselineAfterUp,
        Down,
        BaselineAfterDown,
    };

    explicit Impl(std::size_t workerCount,
                  std::size_t burst,
                  std::shared_ptr<ChunkRequestSelectionGate> gate,
                  std::optional<AdaptiveConcurrency> adaptiveConfig)
        : selectionGate(std::move(gate))
        , interactiveBurst(std::max<std::size_t>(1, burst))
        , maximumWorkers(std::max<std::size_t>(1, workerCount))
    {
        if (!selectionGate)
            selectionGate = std::make_shared<ChunkRequestSelectionGate>();
        workerCount = maximumWorkers;
        if (adaptiveConfig) {
            adaptive = true;
            adaptiveConfig->minimum = std::clamp(
                adaptiveConfig->minimum, std::size_t{1}, maximumWorkers);
            adaptiveConfig->maximum = std::clamp(
                adaptiveConfig->maximum, adaptiveConfig->minimum, maximumWorkers);
            adaptiveConfig->successfulSamplesPerWorker = std::max<std::size_t>(
                1, adaptiveConfig->successfulSamplesPerWorker);
            adaptiveConfig->minimumEpochSeconds = std::max(
                0.0, adaptiveConfig->minimumEpochSeconds);
            adaptiveConfig->maximumEpochSeconds = std::max(
                adaptiveConfig->minimumEpochSeconds,
                adaptiveConfig->maximumEpochSeconds);
            adaptiveConfig->unstableProbeIntervalSeconds = std::max(
                0.0, adaptiveConfig->unstableProbeIntervalSeconds);
            adaptiveConfig->stableProbeIntervalSeconds = std::max(
                adaptiveConfig->unstableProbeIntervalSeconds,
                adaptiveConfig->stableProbeIntervalSeconds);
            adaptiveConfig->minimumStabilityObservationSeconds = std::max(
                0.0, adaptiveConfig->minimumStabilityObservationSeconds);
            adaptiveConfig->bandwidthChangeRatio = std::max(
                1.000001, adaptiveConfig->bandwidthChangeRatio);
            adaptiveConfig->throughputGainRatio = std::max(
                1.0, adaptiveConfig->throughputGainRatio);
            adaptiveConfig->maximumLatencyInflation = std::max(
                1.0, adaptiveConfig->maximumLatencyInflation);
            adaptiveConfig->initialProbeMultiplier = std::max<std::size_t>(
                2, adaptiveConfig->initialProbeMultiplier);
            adaptiveConfig->refinementProbeMultiplier = std::max<std::size_t>(
                2, adaptiveConfig->refinementProbeMultiplier);
            adaptiveConfig->continuousSearchTurns = std::max<std::size_t>(
                1, adaptiveConfig->continuousSearchTurns);
            adaptiveConfig->lowerThroughputRetention = std::clamp(
                adaptiveConfig->lowerThroughputRetention, 0.0, 1.0);
            adaptiveConfig->lowerLatencyRatio = std::clamp(
                adaptiveConfig->lowerLatencyRatio, 0.0, 1.0);
            adaptiveOptions = *adaptiveConfig;
            admissionLimit = adaptiveOptions.minimum;
            targetAdmissionLimit = admissionLimit;
            settledAdmissionLimit = admissionLimit;
            probeMultiplier = adaptiveOptions.initialProbeMultiplier;
            currentProbeIntervalSeconds =
                adaptiveOptions.unstableProbeIntervalSeconds;
        } else {
            admissionLimit = maximumWorkers;
            adaptiveOptions.minimum = maximumWorkers;
            adaptiveOptions.maximum = maximumWorkers;
            targetAdmissionLimit = admissionLimit;
            settledAdmissionLimit = admissionLimit;
        }
        workers.reserve(workerCount);
        for (std::size_t i = 0; i < workerCount; ++i) {
            workers.emplace_back([this](std::stop_token stop) { workerLoop(stop); });
        }
    }

    ~Impl()
    {
        for (auto& worker : workers)
            worker.request_stop();
        cv.notify_all();
    }

    bool staleLocked(const Item& item) const
    {
        const auto found = minimumGroupEpoch.find(item.group);
        return item.group != 0 && found != minimumGroupEpoch.end() &&
               item.groupEpoch < found->second;
    }

    void eraseLocked(TaskId id)
    {
        const auto found = locations.find(id);
        if (found == locations.end())
            return;
        if (found->second.interactive)
            gui.erase(found->second.gui);
        else
            background.erase(found->second.background);
        locations.erase(found);
    }

    void insertLocked(const std::shared_ptr<Item>& item)
    {
        Location location;
        location.interactive = item->priority.interactive;
        if (location.interactive)
            location.gui = gui.insert(item).first;
        else
            location.background = background.insert(item).first;
        locations[item->id] = location;
    }

    std::shared_ptr<Item> popLocked()
    {
        const bool chooseGui = !gui.empty() &&
            (background.empty() || consecutiveInteractive < interactiveBurst);
        std::shared_ptr<Item> item;
        if (chooseGui) {
            auto it = gui.begin();
            item = *it;
            gui.erase(it);
            ++consecutiveInteractive;
        } else if (!background.empty()) {
            auto it = background.begin();
            item = *it;
            background.erase(it);
            consecutiveInteractive = 0;
        }
        if (item)
            locations.erase(item->id);
        return item;
    }

    bool canSelectLocked() const
    {
        return (!gui.empty() || !background.empty()) &&
               activeCount.load(std::memory_order_acquire) < admissionLimit;
    }

    void updateTransferEstimateLocked()
    {
        const std::size_t required = admissionLimit *
            adaptiveOptions.successfulSamplesPerWorker;
        if (transferSamples.empty())
            return;
        if (!transferSamples.back().saturated &&
            maximumSaturatedParallelism != 0) {
            estimatedBytesPerSecond = saturatedBytesPerSecondPerWorker *
                static_cast<double>(targetAdmissionLimit);
            return;
        }

        std::size_t available = 0;
        for (auto sample = transferSamples.rbegin();
             sample != transferSamples.rend() && available < required;
             ++sample) {
            if (!sample->saturated || sample->admission != admissionLimit)
                break;
            ++available;
        }
        if (available == 0)
            return;

        const auto first = transferSamples.end() -
            static_cast<std::ptrdiff_t>(available);
        auto earliestStart = first->started;
        auto latestCompletion = first->completed;
        std::size_t totalBytes = 0;
        for (auto sample = first; sample != transferSamples.end(); ++sample) {
            earliestStart = std::min(earliestStart, sample->started);
            latestCompletion = std::max(latestCompletion, sample->completed);
            totalBytes += sample->bytes;
        }
        const double elapsed = std::chrono::duration<double>(
            latestCompletion - earliestStart).count();
        if (elapsed <= 0.0 || totalBytes == 0)
            return;

        estimatedBytesPerSecond = static_cast<double>(totalBytes) / elapsed;
        averageChunkBytes = static_cast<double>(totalBytes) /
            static_cast<double>(available);
        estimateSampleCount = available;
    }

    void resetEpochLocked(std::chrono::steady_clock::time_point notBefore)
    {
        epochSamples.clear();
        epochNotBefore = notBefore;
    }

    void setAdmissionTargetLocked(
        std::size_t requested,
        std::chrono::steady_clock::time_point transition)
    {
        requested = std::clamp(
            requested, adaptiveOptions.minimum, adaptiveOptions.maximum);
        targetAdmissionLimit = requested;
        resetEpochLocked(transition);
        if (requested > admissionLimit) {
            // Grow by one permit now and one per subsequent completion. This
            // avoids opening a burst of new connections at a probe boundary.
            ++admissionLimit;
            rampingAdmission = admissionLimit < targetAdmissionLimit;
            cv.notify_all();
        } else {
            admissionLimit = requested;
            rampingAdmission = false;
        }
    }

    std::optional<EpochMeasurement> epochMeasurementLocked() const
    {
        if (epochSamples.empty())
            return std::nullopt;
        auto earliestStart = epochSamples.front().started;
        auto latestCompletion = epochSamples.front().completed;
        std::size_t totalBytes = 0;
        std::vector<double> latencies;
        latencies.reserve(epochSamples.size());
        for (const auto& sample : epochSamples) {
            earliestStart = std::min(earliestStart, sample.started);
            latestCompletion = std::max(latestCompletion, sample.completed);
            totalBytes += sample.bytes;
            latencies.push_back(std::chrono::duration<double>(
                sample.completed - sample.started).count());
        }
        const double elapsed = std::chrono::duration<double>(
            latestCompletion - earliestStart).count();
        if (elapsed <= 0.0 || totalBytes == 0)
            return std::nullopt;

        const auto p90 = latencies.begin() + static_cast<std::ptrdiff_t>(
            std::min(latencies.size() - 1,
                     static_cast<std::size_t>(std::ceil(
                         0.9 * static_cast<double>(latencies.size()))) - 1));
        std::nth_element(latencies.begin(), p90, latencies.end());
        return EpochMeasurement{
            static_cast<double>(totalBytes) / elapsed,
            *p90,
            elapsed,
            latestCompletion};
    }

    bool epochCompleteLocked() const
    {
        if (epochSamples.empty())
            return false;
        auto earliest = epochSamples.front().started;
        auto latest = epochSamples.front().completed;
        for (const auto& sample : epochSamples) {
            earliest = std::min(earliest, sample.started);
            latest = std::max(latest, sample.completed);
        }
        const double elapsed = std::chrono::duration<double>(
            latest - earliest).count();
        const std::size_t minimumSamples = std::max<std::size_t>(
            4, targetAdmissionLimit);
        return (elapsed >= adaptiveOptions.minimumEpochSeconds &&
                epochSamples.size() >= minimumSamples) ||
               elapsed >= adaptiveOptions.maximumEpochSeconds;
    }

    void observeSettledBandwidthLocked(const EpochMeasurement& measurement)
    {
        if (measurement.bytesPerSecond <= 0.0)
            return;
        if (longTermBytesPerSecond <= 0.0) {
            longTermBytesPerSecond = measurement.bytesPerSecond;
            bandwidthInstability = 0.0;
            stabilityObservedSeconds = measurement.observationSeconds;
            return;
        }

        const double logRange = std::log(adaptiveOptions.bandwidthChangeRatio);
        const double deviation = std::clamp(
            std::abs(std::log(measurement.bytesPerSecond /
                              longTermBytesPerSecond)) / logRange,
            0.0, 1.0);
        bandwidthInstability = 0.8 * bandwidthInstability + 0.2 * deviation;
        lastBandwidthDeviation = deviation;
        longTermBytesPerSecond = 0.9 * longTermBytesPerSecond +
            0.1 * measurement.bytesPerSecond;
        stabilityObservedSeconds += measurement.observationSeconds;
    }

    double explorationIntervalLocked() const
    {
        if (stabilityObservedSeconds <
                adaptiveOptions.minimumStabilityObservationSeconds) {
            return adaptiveOptions.unstableProbeIntervalSeconds;
        }
        const double instability = std::clamp(
            std::max(lastBandwidthDeviation, bandwidthInstability), 0.0, 1.0);
        return adaptiveOptions.stableProbeIntervalSeconds - instability *
            (adaptiveOptions.stableProbeIntervalSeconds -
             adaptiveOptions.unstableProbeIntervalSeconds);
    }

    static EpochMeasurement averageMeasurements(
        const EpochMeasurement& lhs,
        const EpochMeasurement& rhs)
    {
        return {
            0.5 * (lhs.bytesPerSecond + rhs.bytesPerSecond),
            0.5 * (lhs.latencyP90Seconds + rhs.latencyP90Seconds),
            lhs.observationSeconds + rhs.observationSeconds,
            std::max(lhs.completed, rhs.completed)};
    }

    static double latencyRatio(const EpochMeasurement& candidate,
                               const EpochMeasurement& baseline)
    {
        if (baseline.latencyP90Seconds <= 0.0)
            return 1.0;
        return candidate.latencyP90Seconds / baseline.latencyP90Seconds;
    }

    static double throughputRatio(const EpochMeasurement& candidate,
                                  const EpochMeasurement& baseline)
    {
        if (baseline.bytesPerSecond <= 0.0)
            return 0.0;
        return candidate.bytesPerSecond / baseline.bytesPerSecond;
    }

    void finishProbeCycleLocked(const EpochMeasurement& finalBaseline)
    {
        const std::size_t previousSettled = settledAdmissionLimit;
        std::size_t selected = settledAdmissionLimit;
        double selectedGain = 0.0;
        EpochMeasurement selectedMeasurement = finalBaseline;

        if (upMeasurement && baselineBeforeUp && baselineAfterUp) {
            const auto baseline = averageMeasurements(
                *baselineBeforeUp, *baselineAfterUp);
            const double throughput = throughputRatio(*upMeasurement, baseline);
            const double latency = latencyRatio(*upMeasurement, baseline);
            if (throughput >= adaptiveOptions.throughputGainRatio &&
                latency <= adaptiveOptions.maximumLatencyInflation) {
                const double gain = throughput - 1.0;
                if (gain > selectedGain) {
                    selected = upAdmissionLimit;
                    selectedGain = gain;
                    selectedMeasurement = *upMeasurement;
                }
            }
        }

        if (downMeasurement && baselineAfterUp) {
            const auto baseline = averageMeasurements(
                *baselineAfterUp, finalBaseline);
            const double throughput = throughputRatio(*downMeasurement, baseline);
            const double latency = latencyRatio(*downMeasurement, baseline);
            const bool preservesThroughputAndLatency =
                throughput >= adaptiveOptions.lowerThroughputRetention &&
                latency <= adaptiveOptions.lowerLatencyRatio;
            const bool improvesThroughput =
                throughput >= adaptiveOptions.throughputGainRatio &&
                latency <= adaptiveOptions.maximumLatencyInflation;
            if (preservesThroughputAndLatency || improvesThroughput) {
                const double gain = improvesThroughput
                    ? throughput - 1.0
                    : 0.5 * (1.0 - latency);
                if (gain > selectedGain) {
                    selected = downAdmissionLimit;
                    selectedGain = gain;
                    selectedMeasurement = *downMeasurement;
                }
            }
        }

        const bool changed = selected != previousSettled;
        const int direction = selected > previousSettled
            ? 1
            : (selected < previousSettled ? -1 : 0);
        if (continuousSearch) {
            if (direction == 0 ||
                (lastSearchDirection != 0 && direction != lastSearchDirection)) {
                ++searchTurns;
                probeMultiplier = adaptiveOptions.refinementProbeMultiplier;
            }
            if (direction != 0)
                lastSearchDirection = direction;
            if (searchTurns >= adaptiveOptions.continuousSearchTurns)
                continuousSearch = false;
        } else if (changed) {
            // A periodic probe found a new operating point. Refine around it
            // continuously until the local choice is confirmed again.
            continuousSearch = true;
            searchTurns = 0;
            lastSearchDirection = direction;
            probeMultiplier = adaptiveOptions.refinementProbeMultiplier;
        }
        settledAdmissionLimit = selected;
        phase = ProbePhase::Monitor;
        baselineBeforeUp.reset();
        baselineAfterUp.reset();
        upMeasurement.reset();
        downMeasurement.reset();
        if (changed) {
            // Continue initial/local discovery immediately around a newly
            // selected point. Stability timing begins once a probe retains C.
            longTermBytesPerSecond = selectedMeasurement.bytesPerSecond;
            bandwidthInstability = 0.0;
            lastBandwidthDeviation = 0.0;
            stabilityObservedSeconds = selectedMeasurement.observationSeconds;
        }
        if (!continuousSearch) {
            if (!changed)
                observeSettledBandwidthLocked(finalBaseline);
            currentProbeIntervalSeconds = explorationIntervalLocked();
            nextProbe = finalBaseline.completed + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(currentProbeIntervalSeconds));
        }
        setAdmissionTargetLocked(selected, finalBaseline.completed);
    }

    void beginProbeLocked(const EpochMeasurement& baseline)
    {
        baselineBeforeUp = baseline;
        upAdmissionLimit = settledAdmissionLimit >
                adaptiveOptions.maximum / probeMultiplier
            ? adaptiveOptions.maximum
            : std::min(adaptiveOptions.maximum,
                       settledAdmissionLimit * probeMultiplier);
        downAdmissionLimit = std::max(
            adaptiveOptions.minimum, settledAdmissionLimit / probeMultiplier);
        if (upAdmissionLimit > settledAdmissionLimit) {
            phase = ProbePhase::Up;
            setAdmissionTargetLocked(upAdmissionLimit, baseline.completed);
        } else if (downAdmissionLimit < settledAdmissionLimit) {
            baselineAfterUp = baseline;
            phase = ProbePhase::Down;
            setAdmissionTargetLocked(downAdmissionLimit, baseline.completed);
        } else {
            finishProbeCycleLocked(baseline);
        }
    }

    void completeEpochLocked(const EpochMeasurement& measurement)
    {
        if (targetAdmissionLimit >= maximumSaturatedParallelism) {
            maximumSaturatedParallelism = targetAdmissionLimit;
            saturatedBytesPerSecondPerWorker = measurement.bytesPerSecond /
                static_cast<double>(targetAdmissionLimit);
        }
        switch (phase) {
        case ProbePhase::Monitor:
            observeSettledBandwidthLocked(measurement);
            if (!continuousSearch && nextProbe != Clock::time_point::min()) {
                currentProbeIntervalSeconds = explorationIntervalLocked();
                const auto instabilityDeadline = measurement.completed +
                    std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(
                            currentProbeIntervalSeconds));
                nextProbe = std::min(nextProbe, instabilityDeadline);
            }
            if (continuousSearch || nextProbe == Clock::time_point::min() ||
                measurement.completed >= nextProbe) {
                beginProbeLocked(measurement);
            } else {
                resetEpochLocked(measurement.completed);
            }
            break;
        case ProbePhase::Up:
            upMeasurement = measurement;
            phase = ProbePhase::BaselineAfterUp;
            setAdmissionTargetLocked(settledAdmissionLimit,
                                     measurement.completed);
            break;
        case ProbePhase::BaselineAfterUp:
            baselineAfterUp = measurement;
            if (downAdmissionLimit < settledAdmissionLimit) {
                phase = ProbePhase::Down;
                setAdmissionTargetLocked(downAdmissionLimit,
                                         measurement.completed);
            } else {
                finishProbeCycleLocked(measurement);
            }
            break;
        case ProbePhase::Down:
            downMeasurement = measurement;
            phase = ProbePhase::BaselineAfterDown;
            setAdmissionTargetLocked(settledAdmissionLimit,
                                     measurement.completed);
            break;
        case ProbePhase::BaselineAfterDown:
            finishProbeCycleLocked(measurement);
            break;
        }
    }

    void updateAdaptiveControlLocked(const TransferSample& sample)
    {
        if (!adaptive)
            return;
        if (rampingAdmission) {
            if (admissionLimit < targetAdmissionLimit) {
                ++admissionLimit;
                cv.notify_all();
            }
            if (admissionLimit >= targetAdmissionLimit) {
                rampingAdmission = false;
                resetEpochLocked(sample.completed);
            }
            return;
        }
        if (sample.started < epochNotBefore)
            return;
        // Queue-drain samples describe demand, not connection capacity. Keep
        // the last fully occupied estimate and resume this epoch when enough
        // work is available again.
        if (!sample.saturated)
            return;
        epochSamples.push_back(sample);
        if (!epochCompleteLocked())
            return;
        const auto measurement = epochMeasurementLocked();
        if (measurement)
            completeEpochLocked(*measurement);
    }

    void workerLoop(std::stop_token stop)
    {
        while (!stop.stop_requested()) {
            std::shared_ptr<Item> item;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] {
                    return stop.stop_requested() || canSelectLocked();
                });
                if (stop.stop_requested() && gui.empty() && background.empty())
                    return;
            }
            {
                // Do not hold the queue mutex while waiting for publication:
                // the publisher updates queued items through reprioritize().
                std::lock_guard selectionLock(selectionGate->impl_->mutex);
                std::unique_lock lock(mutex);
                if (stop.stop_requested() && gui.empty() && background.empty())
                    return;
                if (!canSelectLocked())
                    continue;
                item = popLocked();
                if (!item)
                    continue;
                if (staleLocked(*item)) {
                    if (gui.empty() && background.empty() &&
                        activeCount.load(std::memory_order_acquire) == 0) {
                        idleCv.notify_all();
                    }
                    continue;
                }
                activeCount.fetch_add(1, std::memory_order_acq_rel);
            }
            item->task();
            activeCount.fetch_sub(1, std::memory_order_release);
            std::lock_guard lock(mutex);
            cv.notify_all();
            if (gui.empty() && background.empty() &&
                activeCount.load(std::memory_order_acquire) == 0) {
                idleCv.notify_all();
            }
        }
    }

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable idleCv;
    GuiQueue gui;
    BackgroundQueue background;
    std::unordered_map<TaskId, Location> locations;
    std::unordered_map<TaskGroup, std::uint64_t> minimumGroupEpoch;
    std::shared_ptr<ChunkRequestSelectionGate> selectionGate;
    // Declared after the gate so jthread destruction joins every worker before
    // the gate they may be waiting on is released.
    std::vector<std::jthread> workers;
    std::atomic_size_t activeCount{0};
    std::uint64_t nextSequence = 0;
    std::size_t consecutiveInteractive = 0;
    const std::size_t interactiveBurst;
    const std::size_t maximumWorkers;
    bool adaptive = false;
    AdaptiveConcurrency adaptiveOptions;
    std::size_t admissionLimit = 1;
    std::size_t targetAdmissionLimit = 1;
    std::size_t settledAdmissionLimit = 1;
    bool rampingAdmission = false;
    std::deque<TransferSample> transferSamples;
    std::vector<TransferSample> epochSamples;
    using Clock = std::chrono::steady_clock;
    Clock::time_point epochNotBefore = Clock::time_point::min();
    Clock::time_point nextProbe = Clock::time_point::min();
    ProbePhase phase = ProbePhase::Monitor;
    bool continuousSearch = true;
    std::size_t probeMultiplier = adaptiveOptions.initialProbeMultiplier;
    std::size_t searchTurns = 0;
    int lastSearchDirection = 0;
    std::size_t upAdmissionLimit = 1;
    std::size_t downAdmissionLimit = 1;
    std::optional<EpochMeasurement> baselineBeforeUp;
    std::optional<EpochMeasurement> baselineAfterUp;
    std::optional<EpochMeasurement> upMeasurement;
    std::optional<EpochMeasurement> downMeasurement;
    double estimatedBytesPerSecond = 0.0;
    double averageChunkBytes = 0.0;
    std::size_t estimateSampleCount = 0;
    double longTermBytesPerSecond = 0.0;
    double bandwidthInstability = 0.0;
    double lastBandwidthDeviation = 0.0;
    double currentProbeIntervalSeconds = 0.0;
    double stabilityObservedSeconds = 0.0;
    std::size_t maximumSaturatedParallelism = 0;
    double saturatedBytesPerSecondPerWorker = 0.0;
};

ChunkRequestScheduler::ChunkRequestScheduler(std::size_t workers,
                                             std::size_t interactiveBurst,
                                             std::shared_ptr<ChunkRequestSelectionGate> selectionGate,
                                             std::optional<AdaptiveConcurrency> adaptiveConcurrency)
    : impl_(std::make_unique<Impl>(workers, interactiveBurst,
                                  std::move(selectionGate),
                                  std::move(adaptiveConcurrency)))
{
}

ChunkRequestScheduler::~ChunkRequestScheduler() = default;

void ChunkRequestScheduler::submit(TaskId id,
                                   ChunkWorkPriority priority,
                                   TaskGroup group,
                                   std::uint64_t groupEpoch,
                                   std::function<void()> task)
{
    if (id == 0 || !task)
        return;
    std::lock_guard lock(impl_->mutex);
    const auto minimum = impl_->minimumGroupEpoch.find(group);
    if (group != 0 && minimum != impl_->minimumGroupEpoch.end() &&
        groupEpoch < minimum->second) {
        return;
    }
    impl_->eraseLocked(id);
    auto item = std::make_shared<Impl::Item>();
    item->id = id;
    item->priority = priority;
    item->group = group;
    item->groupEpoch = groupEpoch;
    item->sequence = impl_->nextSequence++;
    item->task = std::move(task);
    impl_->insertLocked(item);
    impl_->cv.notify_one();
}

bool ChunkRequestScheduler::reprioritize(TaskId id, ChunkWorkPriority priority)
{
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->locations.find(id);
    if (found == impl_->locations.end())
        return false;
    std::shared_ptr<Impl::Item> item = found->second.interactive
        ? *found->second.gui
        : *found->second.background;
    impl_->eraseLocked(id);
    item->priority = priority;
    impl_->insertLocked(item);
    impl_->cv.notify_one();
    return true;
}

bool ChunkRequestScheduler::cancel(TaskId id)
{
    if (id == 0)
        return false;
    std::lock_guard lock(impl_->mutex);
    if (impl_->locations.find(id) == impl_->locations.end())
        return false;
    impl_->eraseLocked(id);
    if (impl_->gui.empty() && impl_->background.empty() &&
        impl_->activeCount.load(std::memory_order_acquire) == 0) {
        impl_->idleCv.notify_all();
    }
    return true;
}

void ChunkRequestScheduler::cancelGroupBefore(TaskGroup group,
                                              std::uint64_t minimumEpoch)
{
    std::lock_guard lock(impl_->mutex);
    auto& accepted = impl_->minimumGroupEpoch[group];
    accepted = std::max(accepted, minimumEpoch);
    for (auto it = impl_->locations.begin(); it != impl_->locations.end();) {
        const auto& location = it->second;
        const auto item = location.interactive ? *location.gui : *location.background;
        if (item->group == group && item->groupEpoch < accepted) {
            const auto id = it->first;
            ++it;
            impl_->eraseLocked(id);
        } else {
            ++it;
        }
    }
    if (impl_->gui.empty() && impl_->background.empty() &&
        impl_->activeCount.load(std::memory_order_acquire) == 0) {
        impl_->idleCv.notify_all();
    }
}

std::size_t ChunkRequestScheduler::pending() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->locations.size();
}

std::size_t ChunkRequestScheduler::active() const noexcept
{
    return impl_->activeCount.load(std::memory_order_relaxed);
}

void ChunkRequestScheduler::recordSuccessfulTransfer(
    std::size_t encodedBytes,
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point completed)
{
    if (encodedBytes == 0 || completed <= started)
        return;
    std::lock_guard lock(impl_->mutex);
    const std::size_t availableWork =
        impl_->activeCount.load(std::memory_order_acquire) +
        impl_->locations.size();
    // Direct callers used by tests and non-worker integrations cannot expose
    // queue occupancy; treat those samples as capacity observations.
    const bool occupancyKnown = availableWork != 0;
    const bool saturated = !occupancyKnown ||
        availableWork >= impl_->targetAdmissionLimit;
    impl_->transferSamples.push_back(
        {encodedBytes, started, completed, impl_->admissionLimit, saturated});
    const std::size_t maximumSamples = impl_->adaptiveOptions.maximum *
        impl_->adaptiveOptions.successfulSamplesPerWorker;
    while (impl_->transferSamples.size() > maximumSamples)
        impl_->transferSamples.pop_front();
    const auto previousLimit = impl_->admissionLimit;
    impl_->updateTransferEstimateLocked();
    impl_->updateAdaptiveControlLocked(impl_->transferSamples.back());
    if (impl_->admissionLimit > previousLimit)
        impl_->cv.notify_all();
}

ChunkRequestScheduler::TransferStats ChunkRequestScheduler::transferStats() const
{
    std::lock_guard lock(impl_->mutex);
    return {
        impl_->admissionLimit,
        impl_->estimatedBytesPerSecond,
        impl_->averageChunkBytes,
        impl_->estimateSampleCount,
        impl_->adaptive,
        impl_->targetAdmissionLimit,
        impl_->longTermBytesPerSecond,
        impl_->currentProbeIntervalSeconds,
        impl_->phase != Impl::ProbePhase::Monitor || impl_->rampingAdmission};
}

void ChunkRequestScheduler::waitIdle()
{
    std::unique_lock lock(impl_->mutex);
    impl_->idleCv.wait(lock, [&] {
        return impl_->gui.empty() && impl_->background.empty() &&
               impl_->activeCount.load(std::memory_order_acquire) == 0;
    });
}

} // namespace vc::render
