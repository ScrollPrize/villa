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
#include <stdexcept>
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
    using Clock = std::chrono::steady_clock;

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

    struct MaintenanceLess {
        bool operator()(const std::shared_ptr<Item>& lhs,
                        const std::shared_ptr<Item>& rhs) const noexcept
        {
            if (lhs->priority.reserveForegroundSlot !=
                rhs->priority.reserveForegroundSlot) {
                return lhs->priority.reserveForegroundSlot <
                       rhs->priority.reserveForegroundSlot;
            }
            return BackgroundLess{}(lhs, rhs);
        }
    };

    using GuiQueue = std::set<std::shared_ptr<Item>, GuiLess>;
    using BackgroundQueue = std::set<std::shared_ptr<Item>, BackgroundLess>;
    using MaintenanceQueue = std::set<std::shared_ptr<Item>, MaintenanceLess>;

    struct Location {
        enum class Kind {
            Interactive,
            Background,
            Maintenance,
        };

        Kind kind = Kind::Background;
        GuiQueue::iterator gui;
        BackgroundQueue::iterator background;
        MaintenanceQueue::iterator maintenance;
    };

    struct TransferSample {
        std::chrono::steady_clock::time_point started;
        std::chrono::steady_clock::time_point completed;
        std::size_t admission = 1;
        bool saturated = true;
    };

    struct ActiveTransfer {
        std::chrono::steady_clock::time_point started;
        std::size_t admission = 1;
        std::uint64_t epochGeneration = 0;
        std::size_t streamedBytes = 0;
        bool streaming = false;
        bool epochStreaming = false;
    };

    struct StreamByteSample {
        double activeSeconds = 0.0;
        std::size_t bytes = 0;
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

    AdaptiveConcurrency normalizedAdaptiveOptions(
        AdaptiveConcurrency options) const
    {
        if (options.minimum == 0 || options.maximum == 0 ||
            options.minimum > options.maximum ||
            options.maximum > maximumWorkers) {
            throw std::invalid_argument(
                "adaptive concurrency bounds exceed scheduler worker capacity");
        }
        options.minimumEpochSeconds = std::max(
            0.0, options.minimumEpochSeconds);
        options.unstableProbeIntervalSeconds = std::max(
            0.0, options.unstableProbeIntervalSeconds);
        options.stableProbeIntervalSeconds = std::max(
            options.unstableProbeIntervalSeconds,
            options.stableProbeIntervalSeconds);
        options.minimumStabilityObservationSeconds = std::max(
            0.0, options.minimumStabilityObservationSeconds);
        options.bandwidthChangeRatio = std::max(
            1.000001, options.bandwidthChangeRatio);
        options.throughputGainRatio = std::max(
            1.0, options.throughputGainRatio);
        options.initialProbeMultiplier = std::max<std::size_t>(
            2, options.initialProbeMultiplier);
        options.refinementProbeMultiplier = std::max<std::size_t>(
            2, options.refinementProbeMultiplier);
        options.continuousSearchTurns = std::max<std::size_t>(
            1, options.continuousSearchTurns);
        options.lowerThroughputRetention = std::clamp(
            options.lowerThroughputRetention, 0.0, 1.0);
        options.lowerLatencyRatio = std::clamp(
            options.lowerLatencyRatio, 0.0, 1.0);
        return options;
    }

    explicit Impl(std::size_t workerCount,
                  std::size_t burst,
                  std::shared_ptr<ChunkRequestSelectionGate> gate,
                  std::optional<AdaptiveConcurrency> adaptiveConfig,
                  std::optional<AdaptiveState> initialAdaptiveState,
                  std::size_t reservedSlots)
        : selectionGate(std::move(gate))
        , interactiveBurst(std::max<std::size_t>(1, burst))
        , maximumWorkers(std::max<std::size_t>(1, workerCount))
        , maintenanceReservedSlots(std::min(reservedSlots, maximumWorkers))
    {
        if (!selectionGate)
            selectionGate = std::make_shared<ChunkRequestSelectionGate>();
        workerCount = maximumWorkers;
        if (adaptiveConfig) {
            adaptive = true;
            hasAdaptiveHistory = true;
            adaptiveOptions = normalizedAdaptiveOptions(*adaptiveConfig);
            admissionLimit = adaptiveOptions.minimum;
            targetAdmissionLimit = admissionLimit;
            settledAdmissionLimit = admissionLimit;
            probeMultiplier = adaptiveOptions.initialProbeMultiplier;
            currentProbeIntervalSeconds =
                adaptiveOptions.unstableProbeIntervalSeconds;
            if (initialAdaptiveState &&
                initialAdaptiveState->settledAdmissionLimit != 0) {
                settledAdmissionLimit = std::clamp(
                    initialAdaptiveState->settledAdmissionLimit,
                    adaptiveOptions.minimum, adaptiveOptions.maximum);
                admissionLimit = settledAdmissionLimit;
                targetAdmissionLimit = settledAdmissionLimit;
                if (std::isfinite(initialAdaptiveState->longTermBytesPerSecond) &&
                    initialAdaptiveState->longTermBytesPerSecond > 0.0) {
                    longTermBytesPerSecond =
                        initialAdaptiveState->longTermBytesPerSecond;
                }
                if (initialAdaptiveState->maximumSaturatedParallelism != 0 &&
                    std::isfinite(
                        initialAdaptiveState->saturatedBytesPerSecondPerWorker) &&
                    initialAdaptiveState->saturatedBytesPerSecondPerWorker > 0.0) {
                    maximumSaturatedParallelism = std::clamp(
                        initialAdaptiveState->maximumSaturatedParallelism,
                        adaptiveOptions.minimum, adaptiveOptions.maximum);
                    saturatedBytesPerSecondPerWorker =
                        initialAdaptiveState->saturatedBytesPerSecondPerWorker;
                }
            }
            retainedAdaptiveSettledAdmissionLimit = settledAdmissionLimit;
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
        {
            // Serialize the stop transition with the worker wait predicate.
            // Otherwise notify_all() can land after a worker's predicate check
            // but before it has entered the condition-variable wait.
            std::lock_guard lock(mutex);
            for (auto& worker : workers)
                worker.request_stop();
        }
        cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable())
                worker.join();
        }
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
        if (found->second.kind == Location::Kind::Interactive)
            gui.erase(found->second.gui);
        else if (found->second.kind == Location::Kind::Maintenance)
            maintenance.erase(found->second.maintenance);
        else
            background.erase(found->second.background);
        locations.erase(found);
    }

    std::shared_ptr<Item> itemAtLocked(const Location& location) const
    {
        if (location.kind == Location::Kind::Interactive)
            return *location.gui;
        if (location.kind == Location::Kind::Maintenance)
            return *location.maintenance;
        return *location.background;
    }

    void insertLocked(const std::shared_ptr<Item>& item)
    {
        Location location;
        if (item->priority.interactive) {
            location.kind = Location::Kind::Interactive;
            location.gui = gui.insert(item).first;
        } else if (item->priority.maintenance) {
            location.kind = Location::Kind::Maintenance;
            location.maintenance = maintenance.insert(item).first;
        } else {
            location.kind = Location::Kind::Background;
            location.background = background.insert(item).first;
        }
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
        } else if (gui.empty() && !maintenance.empty()) {
            auto it = maintenance.begin();
            item = *it;
            maintenance.erase(it);
            consecutiveInteractive = 0;
        }
        if (item)
            locations.erase(item->id);
        return item;
    }

    template <bool ReserveMaintenanceCapacity>
    bool canSelectLocked() const
    {
        if constexpr (!ReserveMaintenanceCapacity) {
            // Keep the ordinary scheduler predicate identical to the
            // pre-Delta3D path. The encoding choice is process-lifetime
            // state, so workers specialize once at startup instead of paying
            // a mode check for every selected cache task.
            return (!gui.empty() || !background.empty() ||
                    !maintenance.empty()) &&
                   activeCount.load(std::memory_order_acquire) <
                       admissionLimit;
        } else {
            std::size_t effectiveAdmission = admissionLimit;
            const bool onlyBulkMaintenance =
                gui.empty() && background.empty() && !maintenance.empty() &&
                (*maintenance.begin())->priority.reserveForegroundSlot;
            if (onlyBulkMaintenance && admissionLimit > 1) {
                effectiveAdmission = std::max<std::size_t>(
                    1, admissionLimit - maintenanceReservedSlots);
            }
            return (!gui.empty() || !background.empty() ||
                    !maintenance.empty()) &&
                   activeCount.load(std::memory_order_acquire) <
                       effectiveAdmission;
        }
    }

    void configureConcurrencyLocked(
        std::size_t fixedAdmissionLimit,
        std::optional<AdaptiveConcurrency> adaptiveConfig)
    {
        if (fixedAdmissionLimit == 0 || fixedAdmissionLimit > maximumWorkers) {
            throw std::invalid_argument(
                "concurrency admission exceeds scheduler worker capacity");
        }

        const auto now = Clock::now();
        if (!adaptiveConfig) {
            if (adaptive)
                retainedAdaptiveSettledAdmissionLimit = settledAdmissionLimit;
            adaptive = false;
            adaptiveOptions.minimum = fixedAdmissionLimit;
            adaptiveOptions.maximum = fixedAdmissionLimit;
            admissionLimit = fixedAdmissionLimit;
            targetAdmissionLimit = fixedAdmissionLimit;
            settledAdmissionLimit = fixedAdmissionLimit;
        } else {
            adaptiveConfig->maximum = fixedAdmissionLimit;
            adaptiveConfig->minimum = std::min(
                adaptiveConfig->minimum, adaptiveConfig->maximum);
            adaptiveOptions = normalizedAdaptiveOptions(*adaptiveConfig);
            const std::size_t previousSettled = adaptive
                ? settledAdmissionLimit
                : retainedAdaptiveSettledAdmissionLimit.value_or(
                      settledAdmissionLimit);
            adaptive = true;
            hasAdaptiveHistory = true;
            settledAdmissionLimit = std::clamp(
                previousSettled, adaptiveOptions.minimum,
                adaptiveOptions.maximum);
            retainedAdaptiveSettledAdmissionLimit = settledAdmissionLimit;
            admissionLimit = settledAdmissionLimit;
            targetAdmissionLimit = settledAdmissionLimit;
        }

        rampingAdmission = false;
        phase = ProbePhase::Monitor;
        continuousSearch = adaptive;
        probeMultiplier = adaptiveOptions.initialProbeMultiplier;
        searchTurns = 0;
        lastSearchDirection = 0;
        baselineBeforeUp.reset();
        baselineAfterUp.reset();
        upMeasurement.reset();
        downMeasurement.reset();
        currentProbeIntervalSeconds = adaptive
            ? adaptiveOptions.unstableProbeIntervalSeconds
            : 0.0;
        stabilityObservedSeconds = 0.0;
        nextProbe = Clock::time_point::min();
        resetEpochLocked(now);
        cv.notify_all();
    }

    double streamedActiveTimeLocked(Clock::time_point now) const
    {
        if (streamActiveCount == 0)
            return streamActiveSeconds;
        return streamActiveSeconds + std::chrono::duration<double>(
            now - streamActiveStarted).count();
    }

    void beginMeasuredTransferLocked(ActiveTransfer& transfer,
                                     Clock::time_point now)
    {
        if (transfer.streaming)
            return;
        transfer.streaming = true;
        if (streamActiveCount++ == 0) {
            streamActiveStarted = now;
            if (!streamMeasurementStarted) {
                streamMeasurementStartedActiveSeconds = streamActiveSeconds;
                streamMeasurementStarted = true;
            }
        }

        if (adaptive && transfer.epochGeneration == epochGeneration &&
            transfer.admission == targetAdmissionLimit &&
            transfer.started >= epochNotBefore) {
            transfer.epochStreaming = true;
            if (epochStreamActiveCount++ == 0)
                epochStreamActiveStarted = now;
        }
    }

    void recordStreamBytesLocked(ActiveTransfer& transfer,
                                 std::size_t bytes,
                                 Clock::time_point now)
    {
        if (bytes == 0)
            return;
        transfer.streamedBytes += bytes;

        const double activeNow = streamedActiveTimeLocked(now);
        streamByteSamples.push_back({activeNow, bytes});
        streamWindowBytes += bytes;
        updateStreamEstimateLocked(now);

        if (transfer.epochStreaming &&
            transfer.epochGeneration == epochGeneration) {
            epochStreamBytes += bytes;
        }
    }

    void updateStreamEstimateLocked(Clock::time_point now)
    {
        if (!streamMeasurementStarted)
            return;
        const double activeNow = streamedActiveTimeLocked(now);
        constexpr double kWindowSeconds = 5.0;
        while (!streamByteSamples.empty() &&
               streamByteSamples.front().activeSeconds <
                   activeNow - kWindowSeconds) {
            streamWindowBytes -= streamByteSamples.front().bytes;
            streamByteSamples.pop_front();
        }
        const double elapsed = std::min(
            kWindowSeconds,
            std::max(0.0, activeNow - streamMeasurementStartedActiveSeconds));
        if (elapsed > 0.0)
            estimatedBytesPerSecond = static_cast<double>(streamWindowBytes) / elapsed;
    }

    void finishStreamingLocked(ActiveTransfer& transfer, Clock::time_point now)
    {
        if (transfer.streaming) {
            if (streamActiveCount > 0 && --streamActiveCount == 0) {
                streamActiveSeconds += std::chrono::duration<double>(
                    now - streamActiveStarted).count();
            }
            transfer.streaming = false;
        }
        if (transfer.epochStreaming &&
            transfer.epochGeneration == epochGeneration) {
            if (epochStreamActiveCount > 0 && --epochStreamActiveCount == 0) {
                epochStreamActiveSeconds += std::chrono::duration<double>(
                    now - epochStreamActiveStarted).count();
            }
            transfer.epochStreaming = false;
        }
    }

    void resetEpochLocked(std::chrono::steady_clock::time_point notBefore)
    {
        epochSamples.clear();
        epochNotBefore = notBefore;
        ++epochGeneration;
        epochStreamBytes = 0;
        epochStreamActiveSeconds = 0.0;
        epochStreamActiveCount = 0;
        epochStreamActiveStarted = {};
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
        auto latestCompletion = epochSamples.front().completed;
        std::vector<double> latencies;
        latencies.reserve(epochSamples.size());
        for (const auto& sample : epochSamples) {
            latestCompletion = std::max(latestCompletion, sample.completed);
            const double latency = std::chrono::duration<double>(
                sample.completed - sample.started).count();
            latencies.push_back(latency);
        }
        if (epochStreamBytes == 0)
            return std::nullopt;

        double elapsed = epochStreamActiveSeconds;
        if (epochStreamActiveCount != 0) {
            elapsed += std::chrono::duration<double>(
                latestCompletion - epochStreamActiveStarted).count();
        }
        if (elapsed <= 0.0)
            return std::nullopt;
        const double bytesPerSecond = static_cast<double>(epochStreamBytes) / elapsed;
        if (bytesPerSecond <= 0.0)
            return std::nullopt;

        const auto p90 = latencies.begin() + static_cast<std::ptrdiff_t>(
            std::min(latencies.size() - 1,
                     static_cast<std::size_t>(std::ceil(
                         0.9 * static_cast<double>(latencies.size()))) - 1));
        std::nth_element(latencies.begin(), p90, latencies.end());
        return EpochMeasurement{
            bytesPerSecond,
            *p90,
            elapsed,
            latestCompletion};
    }

    bool epochCompleteLocked() const
    {
        if (epochSamples.empty())
            return false;
        double elapsed = epochStreamActiveSeconds;
        if (epochStreamActiveCount != 0) {
            elapsed += std::chrono::duration<double>(
                epochSamples.back().completed - epochStreamActiveStarted).count();
        }
        return elapsed >= adaptiveOptions.minimumEpochSeconds &&
               epochSamples.size() >= targetAdmissionLimit;
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

    void applyProbeSelectionLocked(
        std::size_t selected,
        const EpochMeasurement& selectedMeasurement,
        const EpochMeasurement& completedMeasurement)
    {
        const std::size_t previousSettled = settledAdmissionLimit;
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
                observeSettledBandwidthLocked(completedMeasurement);
            currentProbeIntervalSeconds = explorationIntervalLocked();
            nextProbe = completedMeasurement.completed + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(currentProbeIntervalSeconds));
        }
        setAdmissionTargetLocked(selected, completedMeasurement.completed);
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
            if (throughput >= adaptiveOptions.throughputGainRatio) {
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
                throughput >= adaptiveOptions.throughputGainRatio;
            if (preservesThroughputAndLatency || improvesThroughput) {
                const double gain = throughput - 1.0;
                if ((improvesThroughput && gain > selectedGain) ||
                    (!improvesThroughput && selected == previousSettled)) {
                    selected = downAdmissionLimit;
                    selectedGain = gain;
                    selectedMeasurement = *downMeasurement;
                }
            }
        }

        applyProbeSelectionLocked(
            selected, selectedMeasurement, finalBaseline);
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
            if (baselineBeforeUp &&
                throughputRatio(measurement, *baselineBeforeUp) >=
                    adaptiveOptions.throughputGainRatio) {
                // During discovery, a clear aggregate-goodput gain becomes the
                // next operating point immediately. Replaying the old baseline
                // and probing downward would only delay the continued climb.
                applyProbeSelectionLocked(
                    upAdmissionLimit, measurement, measurement);
                break;
            }
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

    void updateAdaptiveControlLocked(const TransferSample& sample,
                                     std::uint64_t sampleEpochGeneration)
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
        if (sampleEpochGeneration != epochGeneration)
            return;
        if (sample.started < epochNotBefore)
            return;
        // Queue-drain samples describe demand, not connection capacity. End
        // the epoch so later saturated work cannot inherit an idle gap or an
        // underfilled byte interval.
        if (!sample.saturated) {
            resetEpochLocked(sample.completed);
            return;
        }
        epochSamples.push_back(sample);
        if (!epochCompleteLocked())
            return;
        const auto measurement = epochMeasurementLocked();
        if (measurement)
            completeEpochLocked(*measurement);
    }

    template <bool ReserveMaintenanceCapacity>
    void workerLoopImpl(std::stop_token stop)
    {
        while (!stop.stop_requested()) {
            std::shared_ptr<Item> item;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] {
                    return stop.stop_requested() ||
                           canSelectLocked<ReserveMaintenanceCapacity>();
                });
                if (stop.stop_requested() && gui.empty() && background.empty() &&
                    maintenance.empty())
                    return;
            }
            {
                // Do not hold the queue mutex while waiting for publication:
                // the publisher updates queued items through reprioritize().
                std::lock_guard selectionLock(selectionGate->impl_->mutex);
                std::unique_lock lock(mutex);
                if (stop.stop_requested() && gui.empty() && background.empty() &&
                    maintenance.empty())
                    return;
                if (!canSelectLocked<ReserveMaintenanceCapacity>())
                    continue;
                item = popLocked();
                if (!item)
                    continue;
                if (staleLocked(*item)) {
                    if (gui.empty() && background.empty() && maintenance.empty() &&
                        activeCount.load(std::memory_order_acquire) == 0) {
                        idleCv.notify_all();
                    }
                    continue;
                }
                activeCount.fetch_add(1, std::memory_order_acq_rel);
            }
            item->task();
            // A task may retain scheduler-owned state, including a reference
            // to this scheduler for transfer accounting. Release all task
            // captures before publishing the worker as idle so teardown can
            // never leave their final destruction to this worker thread.
            item.reset();
            activeCount.fetch_sub(1, std::memory_order_release);
            std::lock_guard lock(mutex);
            cv.notify_all();
            if (gui.empty() && background.empty() && maintenance.empty() &&
                activeCount.load(std::memory_order_acquire) == 0) {
                idleCv.notify_all();
            }
        }
    }

    void workerLoop(std::stop_token stop)
    {
        if (maintenanceReservedSlots == 0)
            workerLoopImpl<false>(stop);
        else
            workerLoopImpl<true>(stop);
    }

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable idleCv;
    GuiQueue gui;
    BackgroundQueue background;
    MaintenanceQueue maintenance;
    std::unordered_map<TaskId, Location> locations;
    std::unordered_map<TaskGroup, std::uint64_t> minimumGroupEpoch;
    std::shared_ptr<ChunkRequestSelectionGate> selectionGate;
    // Workers are explicitly joined in ~Impl while all scheduler state remains
    // alive. Keep the gate declared first as an additional lifetime safeguard.
    std::vector<std::jthread> workers;
    std::atomic_size_t activeCount{0};
    std::uint64_t nextSequence = 0;
    std::size_t consecutiveInteractive = 0;
    const std::size_t interactiveBurst;
    const std::size_t maximumWorkers;
    const std::size_t maintenanceReservedSlots;
    bool adaptive = false;
    bool hasAdaptiveHistory = false;
    std::optional<std::size_t> retainedAdaptiveSettledAdmissionLimit;
    AdaptiveConcurrency adaptiveOptions;
    std::size_t admissionLimit = 1;
    std::size_t targetAdmissionLimit = 1;
    std::size_t settledAdmissionLimit = 1;
    bool rampingAdmission = false;
    std::unordered_map<TransferId, ActiveTransfer> activeTransfers;
    TransferId nextTransferId = 1;
    std::vector<TransferSample> epochSamples;
    std::deque<StreamByteSample> streamByteSamples;
    std::size_t streamWindowBytes = 0;
    std::size_t streamActiveCount = 0;
    Clock::time_point streamActiveStarted{};
    double streamActiveSeconds = 0.0;
    double streamMeasurementStartedActiveSeconds = 0.0;
    bool streamMeasurementStarted = false;
    Clock::time_point epochNotBefore = Clock::time_point::min();
    std::uint64_t epochGeneration = 1;
    std::size_t epochStreamBytes = 0;
    std::size_t epochStreamActiveCount = 0;
    Clock::time_point epochStreamActiveStarted{};
    double epochStreamActiveSeconds = 0.0;
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
                                             std::optional<AdaptiveConcurrency> adaptiveConcurrency,
                                             std::optional<AdaptiveState> initialAdaptiveState,
                                             std::size_t maintenanceReservedSlots)
    : impl_(std::make_unique<Impl>(workers, interactiveBurst,
                                  std::move(selectionGate),
                                  std::move(adaptiveConcurrency),
                                  std::move(initialAdaptiveState),
                                  maintenanceReservedSlots))
{
}

ChunkRequestScheduler::~ChunkRequestScheduler() = default;

void ChunkRequestScheduler::configureConcurrency(
    std::size_t fixedAdmissionLimit,
    std::optional<AdaptiveConcurrency> adaptiveConcurrency)
{
    std::lock_guard lock(impl_->mutex);
    impl_->configureConcurrencyLocked(
        fixedAdmissionLimit, std::move(adaptiveConcurrency));
}

std::size_t ChunkRequestScheduler::workerCapacity() const noexcept
{
    return impl_->maximumWorkers;
}

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
    std::shared_ptr<Impl::Item> item = impl_->itemAtLocked(found->second);
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
        impl_->maintenance.empty() &&
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
        const auto item = impl_->itemAtLocked(location);
        if (item->group == group && item->groupEpoch < accepted) {
            const auto id = it->first;
            ++it;
            impl_->eraseLocked(id);
        } else {
            ++it;
        }
    }
    if (impl_->gui.empty() && impl_->background.empty() &&
        impl_->maintenance.empty() &&
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

ChunkRequestScheduler::TransferMeasurement::TransferMeasurement(
    ChunkRequestScheduler* scheduler,
    TransferId id,
    std::chrono::steady_clock::time_point started)
    : scheduler_(scheduler)
    , id_(id)
    , lastFlush_(started)
{
}

ChunkRequestScheduler::TransferMeasurement::~TransferMeasurement()
{
    if (scheduler_)
        finish(false, 0);
}

ChunkRequestScheduler::TransferMeasurement::TransferMeasurement(
    TransferMeasurement&& other) noexcept
    : scheduler_(std::exchange(other.scheduler_, nullptr))
    , id_(std::exchange(other.id_, 0))
    , pendingBytes_(std::exchange(other.pendingBytes_, 0))
    , observedBytes_(std::exchange(other.observedBytes_, false))
    , lastFlush_(other.lastFlush_)
{
}

void ChunkRequestScheduler::TransferMeasurement::flush(
    std::chrono::steady_clock::time_point observed)
{
    if (!scheduler_ || pendingBytes_ == 0)
        return;
    scheduler_->recordTransferBytes(id_, pendingBytes_, observed);
    pendingBytes_ = 0;
    lastFlush_ = observed;
}

void ChunkRequestScheduler::TransferMeasurement::recordBytes(
    std::size_t encodedBytes,
    std::chrono::steady_clock::time_point observed)
{
    if (!scheduler_ || encodedBytes == 0)
        return;
    if (!observedBytes_) {
        observedBytes_ = true;
        scheduler_->recordTransferBytes(id_, encodedBytes, observed);
        lastFlush_ = observed;
        return;
    }
    pendingBytes_ += encodedBytes;
    constexpr std::size_t kProgressBatchBytes = 256 * 1024;
    constexpr auto kProgressBatchTime = std::chrono::milliseconds{100};
    if (pendingBytes_ >= kProgressBatchBytes ||
        observed - lastFlush_ >= kProgressBatchTime) {
        flush(observed);
    }
}

void ChunkRequestScheduler::TransferMeasurement::finish(
    bool successful,
    std::size_t encodedBytes,
    std::chrono::steady_clock::time_point completed)
{
    if (!scheduler_)
        return;
    flush(completed);
    auto* scheduler = std::exchange(scheduler_, nullptr);
    scheduler->finishTransfer(id_, successful, encodedBytes, completed);
    id_ = 0;
}

ChunkRequestScheduler::TransferMeasurement ChunkRequestScheduler::beginTransfer(
    std::chrono::steady_clock::time_point started)
{
    return TransferMeasurement(this, beginTransferId(started), started);
}

ChunkRequestScheduler::TransferId ChunkRequestScheduler::beginTransferId(
    std::chrono::steady_clock::time_point started)
{
    std::lock_guard lock(impl_->mutex);
    const TransferId id = impl_->nextTransferId++;
    auto [transfer, inserted] = impl_->activeTransfers.emplace(id, Impl::ActiveTransfer{
        started,
        impl_->admissionLimit,
        impl_->epochGeneration});
    (void)inserted;
    impl_->beginMeasuredTransferLocked(transfer->second, started);
    return id;
}

void ChunkRequestScheduler::recordTransferBytes(
    TransferId transfer,
    std::size_t encodedBytes,
    std::chrono::steady_clock::time_point observed)
{
    if (transfer == 0 || encodedBytes == 0)
        return;
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->activeTransfers.find(transfer);
    if (found == impl_->activeTransfers.end())
        return;
    impl_->recordStreamBytesLocked(found->second, encodedBytes, observed);
}

void ChunkRequestScheduler::finishTransfer(
    TransferId transfer,
    bool successful,
    std::size_t encodedBytes,
    std::chrono::steady_clock::time_point completed)
{
    if (transfer == 0)
        return;
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->activeTransfers.find(transfer);
    if (found == impl_->activeTransfers.end())
        return;

    auto measured = found->second;
    impl_->finishStreamingLocked(measured, completed);
    impl_->activeTransfers.erase(found);

    const std::size_t availableWork =
        impl_->activeCount.load(std::memory_order_acquire) +
        impl_->locations.size();
    const bool occupancyKnown = availableWork != 0;
    const bool saturated = !occupancyKnown ||
        availableWork >= measured.admission;

    const bool successfulPayload = successful && encodedBytes != 0 &&
        measured.streamedBytes != 0 && completed > measured.started;
    const auto previousLimit = impl_->admissionLimit;
    if (impl_->rampingAdmission && completed > measured.started) {
        // Ramp pacing limits connection bursts, so every terminal request can
        // release the next permit. Bandwidth evidence remains payload-only.
        impl_->updateAdaptiveControlLocked(
            {measured.started, completed, measured.admission, saturated},
            measured.epochGeneration);
        if (impl_->admissionLimit > previousLimit)
            impl_->cv.notify_all();
        return;
    }

    if (!successfulPayload) {
        // Sparse remote arrays routinely omit fill chunks. A missing or failed
        // request ends its own active interval without discarding valid
        // observations from other requests in the epoch.
        return;
    }

    impl_->updateAdaptiveControlLocked(
        {measured.started, completed, measured.admission, saturated},
        measured.epochGeneration);
    if (impl_->admissionLimit > previousLimit)
        impl_->cv.notify_all();
}

ChunkRequestScheduler::TransferStats ChunkRequestScheduler::transferStats() const
{
    std::lock_guard lock(impl_->mutex);
    impl_->updateStreamEstimateLocked(std::chrono::steady_clock::now());
    return {
        impl_->admissionLimit,
        impl_->estimatedBytesPerSecond,
        impl_->adaptive,
        impl_->targetAdmissionLimit,
        impl_->longTermBytesPerSecond,
        impl_->currentProbeIntervalSeconds,
        impl_->phase != Impl::ProbePhase::Monitor || impl_->rampingAdmission};
}

std::optional<ChunkRequestScheduler::AdaptiveState>
ChunkRequestScheduler::adaptiveState() const
{
    std::lock_guard lock(impl_->mutex);
    if (!impl_->hasAdaptiveHistory)
        return std::nullopt;
    return AdaptiveState{
        impl_->adaptive
            ? impl_->settledAdmissionLimit
            : impl_->retainedAdaptiveSettledAdmissionLimit.value_or(
                  impl_->settledAdmissionLimit),
        impl_->longTermBytesPerSecond,
        impl_->maximumSaturatedParallelism,
        impl_->saturatedBytesPerSecondPerWorker};
}

void ChunkRequestScheduler::waitIdle()
{
    std::unique_lock lock(impl_->mutex);
    impl_->idleCv.wait(lock, [&] {
        return impl_->gui.empty() && impl_->background.empty() &&
               impl_->maintenance.empty() &&
               impl_->activeCount.load(std::memory_order_acquire) == 0;
    });
}

} // namespace vc::render
