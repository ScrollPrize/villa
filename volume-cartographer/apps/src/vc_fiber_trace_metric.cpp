#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "FiberTraceCli.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using vc::fiber_tracer::FiberTraceConfig;

struct CliOptions {
    std::string fiberManifest;
    std::filesystem::path fiberJson;
    std::string normalManifest;
    std::filesystem::path remoteCacheDir;
    std::optional<double> voxelSizeUm;
    double errorThresholdBaseVoxels = 20.0;
    size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    int inferenceScaledownPower = 2;
    bool quiet = false;
    FiberTraceConfig trace;
};

[[noreturn]] void failOption(const std::string& message)
{
    throw std::invalid_argument(message);
}

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " <fiber.lasagna.json> <fiber.json>"
        << " --normal-manifest <lasagna.lasagna.json> [options]\n\n"
        << "Options:\n"
        << "  --normal-manifest PATH          required Lasagna normal manifest for tangent/normal smoothness\n"
        << "  --remote-cache-dir PATH         required for remote HTTP/S3 Lasagna manifests\n"
        << "  --voxel-size-um N               base-voxel size in micrometers for err/m output\n"
        << "  --inference-scaledown-power N   prediction output scaledown relative to trace voxels, as 2^N [2]\n"
        << "  --step-voxels N                 trace step in manifest trace voxels [4]\n"
        << "  --cone-angle-degrees N          candidate cone half-angle [25]\n"
        << "  --cone-angle-step-degrees N     candidate cone grid step [5]\n"
        << "  --cone-grid-size N              legacy square-to-disk grid size when cone step <= 0 [25]\n"
        << "  --beam-width N                  kept beams per step [8]\n"
        << "  --beam-prune-distance-voxels N  beam endpoint merge radius after lookahead [1]\n"
        << "  --beam-lookahead-steps N        expand this many steps before pruning [2]\n"
        << "  --lookahead-parent-cap N        final-lookahead parent cap, 0 is exact [32]\n"
        << "  --lookahead-retry-parent-cap N  retry failed segments at this cap, 0 disables [0]\n"
        << "  --exhaustive-lookahead          evaluate the full lookahead frontier\n"
        << "  --threads N                     candidate scoring threads, 0 uses worker default, 1 serial [0]\n"
        << "  --smoothness-weight N           smoothness scale [2]\n"
        << "  --smoothness-normal-weight N    normal-axis smoothness weight [0.1]\n"
        << "  --smoothness-tangent-weight N   tangent-plane smoothness weight [10]\n"
        << "  --smoothness-free-angle-degrees N free turn before smoothness penalty [0]\n"
        << "  --cumulative-smoothness-steps N history length for cumulative tangent smoothing [4]\n"
        << "  --cumulative-smoothness-tangent-weight N cumulative tangent smoothing weight [2]\n"
        << "  --max-step-factor N             max steps as factor of CP span [3]\n"
        << "  --error-threshold-base-voxels N restart threshold at target plane [20]\n"
        << "  --cache-gib N                   per-channel chunk-cache budget [8]\n"
        << "  --quiet                         suppress progress line\n";
}

double parseDouble(const std::string& value, const std::string& name)
{
    size_t parsed = 0;
    const double out = std::stod(value, &parsed);
    if (parsed != value.size() || !std::isfinite(out)) {
        failOption("--" + name + " requires a finite number");
    }
    return out;
}

int parseInt(const std::string& value, const std::string& name)
{
    size_t parsed = 0;
    const int out = std::stoi(value, &parsed);
    if (parsed != value.size()) {
        failOption("--" + name + " requires an integer");
    }
    return out;
}

size_t percentileCount(const std::vector<size_t>& values, double quantile)
{
    if (values.empty())
        return 0;
    std::vector<size_t> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t index = std::min(
        sorted.size() - 1,
        static_cast<size_t>(std::ceil(quantile * sorted.size())) - 1);
    return sorted[index];
}

size_t histogramPercentile(
    const std::array<uint64_t, 65>& histogram,
    double quantile)
{
    uint64_t total = 0;
    for (const uint64_t count : histogram)
        total += count;
    if (total == 0)
        return 0;
    const uint64_t target = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(quantile * total)));
    uint64_t cumulative = 0;
    for (size_t index = 0; index < histogram.size(); ++index) {
        cumulative += histogram[index];
        if (cumulative >= target)
            return index;
    }
    return histogram.size() - 1;
}

std::string requireValue(int& index, int argc, char** argv, const std::string& name)
{
    if (index + 1 >= argc) {
        failOption("--" + name + " requires a value");
    }
    return argv[++index];
}

CliOptions parseArgs(int argc, char** argv)
{
    if (argc >= 2) {
        const std::string first = argv[1];
        if (first == "--help" || first == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
    }
    if (argc < 3) {
        printUsage(argv[0]);
        std::exit(2);
    }
    CliOptions options;
    options.fiberManifest = argv[1];
    options.fiberJson = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--normal-manifest") {
            options.normalManifest =
                requireValue(i, argc, argv, "normal-manifest");
        } else if (arg == "--remote-cache-dir") {
            options.remoteCacheDir =
                requireValue(i, argc, argv, "remote-cache-dir");
        } else if (arg == "--voxel-size-um") {
            options.voxelSizeUm =
                parseDouble(requireValue(i, argc, argv, "voxel-size-um"),
                            "voxel-size-um");
        } else if (arg == "--inference-scaledown-power") {
            options.inferenceScaledownPower =
                parseInt(requireValue(i, argc, argv, "inference-scaledown-power"),
                         "inference-scaledown-power");
        } else if (vc::fiber_tracer::cli::parseTraceOption(
                       arg, i, argc, argv, options.trace)) {
            continue;
        } else if (arg == "--error-threshold-base-voxels") {
            options.errorThresholdBaseVoxels =
                parseDouble(
                    requireValue(i, argc, argv, "error-threshold-base-voxels"),
                    "error-threshold-base-voxels");
        } else if (arg == "--cache-gib") {
            const double gib =
                parseDouble(requireValue(i, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0))
                failOption("--cache-gib must be positive");
            options.cacheBytes = static_cast<size_t>(
                gib * 1024.0 * 1024.0 * 1024.0);
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else {
            failOption("unknown option: " + arg);
        }
    }
    vc::fiber_tracer::cli::validateTraceOptions(options.trace);
    if (!(options.errorThresholdBaseVoxels >= 0.0))
        failOption("--error-threshold-base-voxels must be non-negative");
    if (options.inferenceScaledownPower < 0 || options.inferenceScaledownPower > 30)
        failOption("--inference-scaledown-power must be in [0, 30]");
    if (options.normalManifest.empty()) {
        failOption(
            "--normal-manifest is required; pass the Lasagna normal manifest used for "
            "tangent/normal smoothness");
    }
    const bool usesRemoteManifest =
        vc::lasagna::isRemoteLasagnaLocation(options.fiberManifest) ||
        vc::lasagna::isRemoteLasagnaLocation(options.normalManifest);
    if (usesRemoteManifest && options.remoteCacheDir.empty()) {
        failOption("remote Lasagna manifests require --remote-cache-dir");
    }
    return options;
}

std::string formatDuration(double seconds)
{
    if (!(seconds > 0.0))
        return "0s";
    if (seconds < 60.0)
        return std::to_string(static_cast<int>(std::round(seconds))) + "s";
    const int minutes = static_cast<int>(seconds) / 60;
    const int secs = static_cast<int>(seconds) % 60;
    return std::to_string(minutes) + "m" + std::to_string(secs) + "s";
}

std::string progressBar(int done, int total)
{
    constexpr int width = 24;
    const double fraction =
        total > 0 ? std::clamp(static_cast<double>(done) / total, 0.0, 1.0) : 1.0;
    const int filled = static_cast<int>(std::round(fraction * width));
    return "[" + std::string(static_cast<size_t>(filled), '#') +
           std::string(static_cast<size_t>(width - filled), '-') + "]";
}

void clearProgressLine()
{
    std::cout << "\r" << std::string(180, ' ') << "\r";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parseArgs(argc, argv);

        vc::lasagna::LasagnaDatasetOpenOptions datasetOptions;
        datasetOptions.remoteCacheRoot = options.remoteCacheDir;

        const auto openedDataset = vc::lasagna::LasagnaDataset::openLocation(
            options.fiberManifest,
            datasetOptions);
        const auto traceScales =
            vc::fiber_tracer::resolveFiberPredictionTraceScales(
                openedDataset.manifest(),
                options.inferenceScaledownPower);
        const double workingToBaseScale = traceScales.traceToBaseScale;
        auto predictionManifest = openedDataset.manifest();
        predictionManifest.workingToBaseScale = workingToBaseScale;
        const vc::lasagna::LasagnaDataset dataset(std::move(predictionManifest));
        const vc::fiber_tracer::FiberPredictionField predictions(
            dataset,
            options.cacheBytes);

        std::optional<vc::lasagna::LasagnaDataset> normalDataset;
        std::optional<vc::lasagna::LasagnaNormalSampler> normalSampler;
        vc::lasagna::LasagnaDatasetOpenOptions normalDatasetOptions;
        normalDatasetOptions.workingToBaseScale = workingToBaseScale;
        normalDatasetOptions.remoteCacheRoot = options.remoteCacheDir;
        normalDataset.emplace(vc::lasagna::LasagnaDataset::openLocation(
            options.normalManifest,
            normalDatasetOptions));
        normalSampler.emplace(
            *normalDataset,
            vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});
        const vc::lasagna::NormalSampler* normalSamplerPtr = &*normalSampler;

        const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
        if (!options.quiet) {
            std::cout
                << "vc_fiber_trace_metric input fiber_manifest="
                << options.fiberManifest
                << " fiber_json=" << options.fiberJson
                << " control_points=" << fiber.controlPointsXyzBase.size()
                << " segments=" << (fiber.controlPointsXyzBase.size() - 1)
                << " derived_trace_to_base=" << traceScales.traceToBaseScale
                << " derived_prediction_to_base=" << traceScales.predictionToBaseScale
                << " derived_prediction_spacing_trace_voxels="
                << traceScales.predictionSpacingInTraceVoxels
                << " inference_scaledown_power=" << options.inferenceScaledownPower
                << " threads="
                << (options.trace.parallelThreads > 0
                        ? std::to_string(options.trace.parallelThreads)
                        : std::string("auto"))
                << " normal_sampler=" << (normalSamplerPtr != nullptr ? "on" : "off")
                << '\n';
        }

        vc::fiber_tracer::FiberTraceWholeFiberMetricRequest request;
        request.fiber = fiber;
        request.workingToBaseScale = workingToBaseScale;
        request.errorThresholdBaseVoxels = options.errorThresholdBaseVoxels;
        request.voxelSizeUm = options.voxelSizeUm;
        request.config = options.trace;
        vc::fiber_tracer::FiberTraceProfile profile;
        request.config.profile = &profile;

        using Clock = std::chrono::steady_clock;
        const auto wallStart = Clock::now();
        const std::clock_t cpuStart = std::clock();
        auto lastProgress = Clock::now();

        const auto progress = [&](const vc::fiber_tracer::FiberTraceWholeFiberProgress& event) {
            if (options.quiet)
                return;
            const auto now = Clock::now();
            const bool done = event.completedSegments >= event.segmentCount;
            if (!done &&
                std::chrono::duration<double>(now - lastProgress).count() < 0.5) {
                return;
            }
            lastProgress = now;
            const double elapsed = std::chrono::duration<double>(now - wallStart).count();
            const double rate =
                event.completedSegments > 0
                    ? elapsed / static_cast<double>(event.completedSegments)
                    : 0.0;
            const double eta =
                rate > 0.0
                    ? rate * static_cast<double>(
                          std::max(0, event.segmentCount - event.completedSegments))
                    : 0.0;
            clearProgressLine();
            std::cout << "native whole fiber "
                      << progressBar(event.completedSegments, event.segmentCount)
                      << ' ' << event.completedSegments << '/' << event.segmentCount
                      << " elapsed=" << formatDuration(elapsed)
                      << " eta=" << formatDuration(eta)
                      << " segment=" << event.currentSegment << '/'
                      << event.segmentCount
                      << " status=" << event.status
                      << " restarts=" << event.restartCount
                      << " err/kvx=" << std::fixed << std::setprecision(1)
                      << event.restartsPerKvx;
            if (event.restartsPerMeter.has_value()) {
                std::cout << " err/m=" << std::fixed << std::setprecision(1)
                          << *event.restartsPerMeter;
                if (event.referenceLengthMeters.has_value()) {
                    std::cout << " (" << std::fixed << std::setprecision(1)
                              << (*event.referenceLengthMeters * 1000.0) << "mm)";
                }
            }
            if (event.hasTraceProgress) {
                std::cout << " step=" << event.traceProgress.step
                          << '/' << event.traceProgress.maxSteps
                          << " reason=" << event.traceProgress.reason;
            }
            std::cout << std::flush;
            if (done)
                std::cout << '\n';
        };

        const auto result = vc::fiber_tracer::traceWholeFiberMetric(
            predictions,
            request,
            normalSamplerPtr,
            progress);
        const auto wallEnd = Clock::now();
        const std::clock_t cpuEnd = std::clock();
        const double wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();
        const double cpuSeconds =
            static_cast<double>(cpuEnd - cpuStart) / static_cast<double>(CLOCKS_PER_SEC);

        std::cout << "native_trace2cp_fiber err/kvx=" << std::fixed
                  << std::setprecision(1) << result.restartsPerKvx
                  << " restarts=" << result.restartCount
                  << " lookahead_retries=" << result.lookaheadRetryCount
                  << " lookahead_retry_recovered="
                  << result.lookaheadRetryRecoveredCount
                  << " segments=" << result.segmentCount << '\n';
        if (result.restartsPerMeter.has_value()) {
            std::cout << "native_trace2cp_fiber err/m=" << std::fixed
                      << std::setprecision(1) << *result.restartsPerMeter;
            if (result.referenceLengthMeters.has_value()) {
                std::cout << " (" << std::fixed << std::setprecision(1)
                          << (*result.referenceLengthMeters * 1000.0) << "mm)";
            }
            std::cout << '\n';
        }
        std::cout << "native_trace2cp_timing trace_wall_s=" << std::fixed
                  << std::setprecision(3) << wallSeconds
                  << " trace_cpu_s=" << cpuSeconds << '\n';
        std::cout << "native_trace2cp_profile"
                  << " one_way=" << profile.oneWayCalls
                  << " generations=" << profile.generations
                  << " candidates=" << profile.candidateTasks
                  << " avg_candidates_per_generation="
                  << (profile.generations > 0
                          ? static_cast<double>(profile.candidateTasks) /
                                static_cast<double>(profile.generations)
                          : 0.0)
                  << " lookahead_final_frontiers="
                  << profile.lookaheadFinalFrontiers
                  << " lookahead_total_parents=" << profile.lookaheadTotalParents
                  << " lookahead_required_parents="
                  << profile.lookaheadRequiredParents
                  << " lookahead_evaluated_parents="
                  << profile.lookaheadEvaluatedParents
                  << " lookahead_parent_retain_ratio="
                  << (profile.lookaheadTotalParents > 0
                          ? static_cast<double>(profile.lookaheadRequiredParents) /
                                static_cast<double>(profile.lookaheadTotalParents)
                          : 0.0)
                  << " lookahead_required_parent_mean="
                  << (profile.lookaheadFinalFrontiers > 0
                          ? static_cast<double>(profile.lookaheadRequiredParents) /
                                static_cast<double>(profile.lookaheadFinalFrontiers)
                          : 0.0)
                  << " lookahead_required_parent_p50="
                  << percentileCount(profile.lookaheadRequiredParentCounts, 0.50)
                  << " lookahead_required_parent_p95="
                  << percentileCount(profile.lookaheadRequiredParentCounts, 0.95)
                  << " lookahead_required_parent_max="
                  << (profile.lookaheadRequiredParentCounts.empty()
                          ? 0
                          : *std::max_element(
                                profile.lookaheadRequiredParentCounts.begin(),
                                profile.lookaheadRequiredParentCounts.end()))
                  << " lookahead_total_children="
                  << profile.lookaheadTotalChildCandidates
                  << " lookahead_required_children="
                  << profile.lookaheadRequiredChildCandidates
                  << " lookahead_evaluated_children="
                  << profile.lookaheadEvaluatedChildCandidates
                  << " depth1_batches=" << profile.candidateDepth1Batches
                  << " depth1_points=" << profile.candidateDepth1Points
                  << " depth1_batch_p50="
                  << percentileCount(profile.candidateDepth1BatchSizes, 0.50)
                  << " depth1_batch_p95="
                  << percentileCount(profile.candidateDepth1BatchSizes, 0.95)
                  << " depth2_batches=" << profile.candidateDepth2Batches
                  << " depth2_points=" << profile.candidateDepth2Points
                  << " depth2_batch_p50="
                  << percentileCount(profile.candidateDepth2BatchSizes, 0.50)
                  << " depth2_batch_p95="
                  << percentileCount(profile.candidateDepth2BatchSizes, 0.95)
                  << " corner_points=" << profile.cornerPointCount
                  << " corner_unique_cubes=" << profile.cornerUniqueVoxelCubes
                  << " corner_points_per_cube="
                  << (profile.cornerUniqueVoxelCubes > 0
                          ? static_cast<double>(profile.cornerPointCount) /
                                static_cast<double>(profile.cornerUniqueVoxelCubes)
                          : 0.0)
                  << " corner_cube_reuse_p50="
                  << histogramPercentile(profile.cornerCubeOccupancyHistogram, 0.50)
                  << " corner_cube_reuse_p95="
                  << histogramPercentile(profile.cornerCubeOccupancyHistogram, 0.95)
                  << " corner_cube_reuse_max="
                  << profile.cornerMaxCandidatesPerCube
                  << " corner_worker_tasks=" << profile.cornerWorkerTasks
                  << " depth_dependency_overlap="
                  << (profile.depthDependencyUnion > 0
                          ? static_cast<double>(profile.depthDependencyShared) /
                                static_cast<double>(profile.depthDependencyUnion)
                          : 0.0)
                  << " step_dependency_overlap="
                  << (profile.stepDependencyUnion > 0
                          ? static_cast<double>(profile.stepDependencyShared) /
                                static_cast<double>(profile.stepDependencyUnion)
                          : 0.0)
                  << " start_sample_s=" << profile.startSampleSeconds
                  << " task_build_s=" << profile.taskBuildSeconds
                  << " prediction_batch_s=" << profile.predictionBatchSeconds
                  << " prediction_prepare_s=" << profile.predictionPrepareSeconds
                  << " prediction_prefetch_s=" << profile.predictionPrefetchSeconds
                  << " prediction_assign_s=" << profile.predictionAssignSeconds
                  << " prediction_materialize_s=" << profile.predictionMaterializeSeconds
                  << " prediction_corner_s=" << profile.predictionCornerSeconds
                  << " prediction_corner_prepare_s="
                  << profile.predictionCornerPrepareSeconds
                  << " prediction_corner_layout_s="
                  << profile.predictionCornerLayoutSeconds
                  << " prediction_corner_pin_s="
                  << profile.predictionCornerPinSeconds
                  << " prediction_corner_gather_s="
                  << profile.predictionCornerGatherSeconds
                  << " prediction_corner_chunk_runs="
                  << profile.predictionCornerLayoutChunkRuns
                  << " prediction_corner_boundary_points="
                  << profile.predictionCornerBoundaryPoints
                  << " prediction_corner_dependencies="
                  << profile.predictionCornerDependencies
                  << " prediction_decode_s=" << profile.predictionDecodeSeconds
                  << " normal_decode_s=" << profile.normalDecodeSeconds
                  << " normal_batch_s=" << profile.normalBatchSeconds
                  << " normal_prefetch_s=" << profile.normalPrefetchSeconds
                  << " normal_materialize_s=" << profile.normalMaterializeSeconds
                  << " candidate_score_s=" << profile.candidateScoreSeconds
                  << " frontier_s=" << profile.frontierSeconds
                  << " prune_s=" << profile.pruneSeconds
                  << " lookahead_decision_s=" << profile.lookaheadDecisionSeconds
                  << " lookahead_parent_order_s="
                  << profile.lookaheadParentOrderSeconds
                  << " lookahead_frontier_storage_s="
                  << profile.lookaheadFrontierStorageSeconds
                  << " lookahead_frontier_allocated_slots="
                  << profile.lookaheadFrontierAllocatedSlots
                  << " lookahead_frontier_evaluated_slots="
                  << profile.lookaheadFrontierEvaluatedSlots
                  << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "vc_fiber_trace_metric error: " << exc.what() << '\n';
        return 1;
    }
}
