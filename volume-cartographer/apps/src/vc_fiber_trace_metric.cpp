#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationOptimizationDefaults.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/ModelPrefetch.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

namespace {

using vc::fiber_tracer::FiberTraceConfig;

struct CliOptions {
    std::string fiberManifest;
    std::filesystem::path fiberJson;
    std::string normalManifest;
    std::filesystem::path remoteCacheDir;
    std::filesystem::path traceOutput;
    std::string normalManifestIdentity;
    std::string fiberManifestIdentity;
    std::optional<double> voxelSizeUm;
    double errorThresholdBaseVoxels = 20.0;
    size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    int inferenceScaledownPower = 2;
    int prefetchWarmupMs = 0;
    int settlePrefetchMs = 0;
    std::optional<int> guiDirtySpan;
    std::optional<int> modelReaders;
    bool quiet = false;
    bool guiReoptimize = false;
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
        << "  --prefetch-warmup-ms N          optional model corridor warmup, separately timed [0; max 10000]\n"
        << "  --model-readers N              fixed shared model read concurrency [default adaptive; 1..64]\n"
        << "  --trace-output PATH            write exact trace/decision JSON to a new file (double bit patterns)\n"
        << "  --gui-reoptimize                run the actual GUI fiber optimizer, including 1200-base-voxel tails\n"
        << "  --gui-dirty-span N              re-solve one zero-based saved CP span; requires --gui-reoptimize\n"
        << "  --settle-prefetch-ms N          bounded post-timing drain for complete model counters [0; max 60000]\n"
        << "  --normal-manifest-identity ID   stable normal-model provenance in GUI output [manifest location]\n"
        << "  --fiber-manifest-identity ID    stable prediction-model provenance in GUI output [manifest location]\n"
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
        } else if (arg == "--prefetch-warmup-ms") {
            options.prefetchWarmupMs = parseInt(
                requireValue(i, argc, argv, "prefetch-warmup-ms"), "prefetch-warmup-ms");
            if (options.prefetchWarmupMs < 0 || options.prefetchWarmupMs > 10000)
                failOption("--prefetch-warmup-ms must be between 0 and 10000");
        } else if (arg == "--model-readers") {
            options.modelReaders = parseInt(
                requireValue(i, argc, argv, "model-readers"), "model-readers");
            if (*options.modelReaders < 1 || *options.modelReaders > 64)
                failOption("--model-readers must be between 1 and 64");
        } else if (arg == "--trace-output") {
            options.traceOutput = requireValue(i, argc, argv, "trace-output");
            if (options.traceOutput.empty() || std::filesystem::exists(options.traceOutput))
                failOption("--trace-output must name a new file");
        } else if (arg == "--gui-reoptimize") {
            options.guiReoptimize = true;
        } else if (arg == "--gui-dirty-span") {
            options.guiDirtySpan = parseInt(requireValue(i, argc, argv, "gui-dirty-span"), "gui-dirty-span");
            if (*options.guiDirtySpan < 0)
                failOption("--gui-dirty-span must be nonnegative");
        } else if (arg == "--settle-prefetch-ms") {
            options.settlePrefetchMs = parseInt(requireValue(i, argc, argv, "settle-prefetch-ms"), "settle-prefetch-ms");
            if (options.settlePrefetchMs < 0 || options.settlePrefetchMs > 60000)
                failOption("--settle-prefetch-ms must be between 0 and 60000");
        } else if (arg == "--normal-manifest-identity") {
            options.normalManifestIdentity = requireValue(i, argc, argv, "normal-manifest-identity");
        } else if (arg == "--fiber-manifest-identity") {
            options.fiberManifestIdentity = requireValue(i, argc, argv, "fiber-manifest-identity");
        } else if (arg == "--voxel-size-um") {
            options.voxelSizeUm =
                parseDouble(requireValue(i, argc, argv, "voxel-size-um"),
                            "voxel-size-um");
        } else if (arg == "--inference-scaledown-power") {
            options.inferenceScaledownPower =
                parseInt(requireValue(i, argc, argv, "inference-scaledown-power"),
                         "inference-scaledown-power");
        } else if (arg == "--step-voxels") {
            options.trace.stepVoxels =
                parseDouble(requireValue(i, argc, argv, "step-voxels"),
                            "step-voxels");
        } else if (arg == "--cone-angle-degrees") {
            options.trace.coneAngleDegrees =
                parseDouble(requireValue(i, argc, argv, "cone-angle-degrees"),
                            "cone-angle-degrees");
        } else if (arg == "--cone-angle-step-degrees") {
            options.trace.coneAngleStepDegrees =
                parseDouble(requireValue(i, argc, argv, "cone-angle-step-degrees"),
                            "cone-angle-step-degrees");
        } else if (arg == "--cone-grid-size") {
            options.trace.coneGridSize =
                parseInt(requireValue(i, argc, argv, "cone-grid-size"),
                         "cone-grid-size");
        } else if (arg == "--beam-width") {
            options.trace.beamWidth =
                parseInt(requireValue(i, argc, argv, "beam-width"),
                         "beam-width");
        } else if (arg == "--beam-prune-distance-voxels") {
            options.trace.beamPruneDistanceVoxels =
                parseDouble(requireValue(i, argc, argv, "beam-prune-distance-voxels"),
                            "beam-prune-distance-voxels");
        } else if (arg == "--beam-lookahead-steps") {
            options.trace.beamLookaheadSteps =
                parseInt(requireValue(i, argc, argv, "beam-lookahead-steps"),
                         "beam-lookahead-steps");
        } else if (arg == "--lookahead-parent-cap") {
            const int cap = parseInt(
                requireValue(i, argc, argv, "lookahead-parent-cap"),
                "lookahead-parent-cap");
            if (cap < 0)
                failOption("--lookahead-parent-cap must be non-negative");
            options.trace.lookaheadParentCap = static_cast<size_t>(cap);
        } else if (arg == "--lookahead-retry-parent-cap") {
            const int cap = parseInt(
                requireValue(i, argc, argv, "lookahead-retry-parent-cap"),
                "lookahead-retry-parent-cap");
            if (cap < 0)
                failOption("--lookahead-retry-parent-cap must be non-negative");
            options.trace.lookaheadRetryParentCap = static_cast<size_t>(cap);
        } else if (arg == "--exhaustive-lookahead") {
            options.trace.lazyLookahead = false;
        } else if (arg == "--threads") {
            options.trace.parallelThreads =
                parseInt(requireValue(i, argc, argv, "threads"), "threads");
        } else if (arg == "--smoothness-weight") {
            options.trace.smoothnessWeight =
                parseDouble(requireValue(i, argc, argv, "smoothness-weight"),
                            "smoothness-weight");
        } else if (arg == "--smoothness-normal-weight") {
            options.trace.smoothnessNormalWeight =
                parseDouble(requireValue(i, argc, argv, "smoothness-normal-weight"),
                            "smoothness-normal-weight");
        } else if (arg == "--smoothness-tangent-weight") {
            options.trace.smoothnessTangentWeight =
                parseDouble(requireValue(i, argc, argv, "smoothness-tangent-weight"),
                            "smoothness-tangent-weight");
        } else if (arg == "--smoothness-free-angle-degrees") {
            options.trace.smoothnessFreeAngleDegrees =
                parseDouble(requireValue(i, argc, argv, "smoothness-free-angle-degrees"),
                            "smoothness-free-angle-degrees");
        } else if (arg == "--cumulative-smoothness-steps") {
            options.trace.cumulativeSmoothnessSteps =
                parseInt(requireValue(i, argc, argv, "cumulative-smoothness-steps"),
                         "cumulative-smoothness-steps");
        } else if (arg == "--cumulative-smoothness-tangent-weight") {
            options.trace.cumulativeSmoothnessTangentWeight =
                parseDouble(requireValue(
                                i,
                                argc,
                                argv,
                                "cumulative-smoothness-tangent-weight"),
                            "cumulative-smoothness-tangent-weight");
        } else if (arg == "--max-step-factor") {
            options.trace.maxStepFactor =
                parseDouble(requireValue(i, argc, argv, "max-step-factor"),
                            "max-step-factor");
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
    if (!(options.trace.stepVoxels > 0.0))
        failOption("--step-voxels must be positive");
    if (!(options.trace.coneAngleDegrees >= 0.0))
        failOption("--cone-angle-degrees must be non-negative");
    if (options.trace.coneGridSize < 1)
        failOption("--cone-grid-size must be at least 1");
    if (options.trace.beamWidth < 1)
        failOption("--beam-width must be at least 1");
    if (!(options.trace.beamPruneDistanceVoxels >= 0.0))
        failOption("--beam-prune-distance-voxels must be non-negative");
    if (options.trace.beamLookaheadSteps < 1)
        failOption("--beam-lookahead-steps must be at least 1");
    if (options.trace.parallelThreads < 0)
        failOption("--threads must be non-negative");
    if (!(options.trace.smoothnessWeight >= 0.0) ||
        !(options.trace.smoothnessNormalWeight >= 0.0) ||
        !(options.trace.smoothnessTangentWeight >= 0.0) ||
        !(options.trace.cumulativeSmoothnessTangentWeight >= 0.0)) {
        failOption("smoothness weights must be non-negative");
    }
    if (!(options.trace.smoothnessFreeAngleDegrees >= 0.0))
        failOption("--smoothness-free-angle-degrees must be non-negative");
    if (options.trace.cumulativeSmoothnessSteps < 1)
        failOption("--cumulative-smoothness-steps must be at least 1");
    if (!(options.errorThresholdBaseVoxels >= 0.0))
        failOption("--error-threshold-base-voxels must be non-negative");
    if (options.inferenceScaledownPower < 0 || options.inferenceScaledownPower > 30)
        failOption("--inference-scaledown-power must be in [0, 30]");
    if (options.guiDirtySpan && !options.guiReoptimize)
        failOption("--gui-dirty-span requires --gui-reoptimize");
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

void warmModelCorridor(
    const vc::fiber_tracer::FiberPredictionField& predictions,
    const vc::lasagna::LasagnaNormalSampler& normals,
    const vc::fiber_tracer::FiberInput& fiber,
    double traceToBaseScale, int budgetMs)
{
    if (budgetMs == 0)
        return;
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto before = vc::lasagna::remoteStoreStats();
    vc::lasagna::ModelPrefetchReport report;
    bool done = false;
    bool failed = false;
    const bool enabled = vc::lasagna::modelPrefetchEnabled();
    if (enabled) {
        try {
            auto sources = predictions.prefetchSources();
            auto normalSources = normals.prefetchSources();
            sources.insert(sources.end(), normalSources.begin(), normalSources.end());
            for (auto& source : sources)
                source.spacing *= traceToBaseScale;
            std::erase_if(sources, [](const auto& source) { return !source.remote; });
            if (!sources.empty()) {
                vc::lasagna::ModelPrefetchPlan plan(std::move(sources), fiber.linePointsXyzBase);
                const auto deadline = start + std::chrono::milliseconds(budgetMs);
                while (Clock::now() < deadline) {
                    if (!done)
                        done = plan.pump();
                    // This standalone CLI has no other speculative producer
                    // before tracing. Settling is bounded by the same deadline;
                    // submitted downloads may continue into the trace phase.
                    if (done && vc::render::ChunkCache::speculativePrefetchStats().pendingRequests == 0)
                        break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                report = plan.report();
            } else {
                done = true;
            }
        } catch (const std::exception& error) {
            failed = true;
            std::cerr << "model corridor warmup skipped after error: " << error.what() << '\n';
        }
    }
    const auto after = vc::lasagna::remoteStoreStats();
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "native_trace2cp_warmup enabled=" << enabled
              << " budget_ms=" << budgetMs << " warmup_ms=" << std::fixed << std::setprecision(3) << elapsed
              << " planning_ms=" << report.planningMs << " planned=" << report.planned
              << " submitted=" << report.submitted << " rejected=" << report.rejected
              << " planned_bytes=" << report.plannedBytes << " truncated=" << report.truncated
              << " submission_done=" << done << " failed=" << failed
              << " pending_requests=" << vc::render::ChunkCache::speculativePrefetchStats().pendingRequests
              << " remote_owned=" << after.objectsOwned - before.objectsOwned
              << " remote_bytes=" << after.bytesOwned - before.bytesOwned
              << " remote_disk_hits=" << after.objectsFromDisk - before.objectsFromDisk << '\n';
}

// Exact bit payloads preserve signed zero, infinities, and NaN payloads as well
// as ordinary coordinates. Timings, cache state, and input paths are excluded
// so equal solver outputs compare byte-for-byte across storage conditions.
std::string doubleBits(double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16)
        << std::bit_cast<std::uint64_t>(value);
    return out.str();
}

nlohmann::json exactPoint(const cv::Vec3d& point)
{
    return {doubleBits(point[0]), doubleBits(point[1]), doubleBits(point[2])};
}

nlohmann::json exactPoints(const std::vector<cv::Vec3d>& points)
{
    auto out = nlohmann::json::array();
    for (const auto& point : points)
        out.push_back(exactPoint(point));
    return out;
}

nlohmann::json optionalDoubleBits(const std::optional<double>& value)
{
    return value ? nlohmann::json(doubleBits(*value)) : nlohmann::json(nullptr);
}

nlohmann::json exactJsonDoubles(nlohmann::json value)
{
    if (value.is_number_float())
        return doubleBits(value.get<double>());
    if (value.is_structured()) {
        for (auto& child : value)
            child = exactJsonDoubles(std::move(child));
    }
    return value;
}

void writeExactOutput(const std::filesystem::path& path, const nlohmann::json& output)
{
    if (std::filesystem::exists(path))
        throw std::runtime_error("trace output already exists: " + path.string());
    std::ofstream stream(path, std::ios::out | std::ios::noreplace);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << output.dump(2) << '\n';
}

void writeTraceOutput(const std::filesystem::path& path,
                      const vc::fiber_tracer::FiberTraceWholeFiberResult& result,
                      double traceToBaseScale)
{
    if (path.empty())
        return;
    auto segments = nlohmann::json::array();
    for (const auto& segment : result.segments) {
        const auto& trace = segment.trace;
        auto crossings = nlohmann::json::array();
        for (const auto& crossing : trace.targetPlaneCrossings) {
            crossings.push_back({{"name", crossing.name}, {"point", exactPoint(crossing.point)},
                                 {"in_plane_error_voxels", doubleBits(crossing.inPlaneErrorVoxels)}});
        }
        segments.push_back({
            {"start_control_point", segment.startControlPointIndex},
            {"target_control_point", segment.targetControlPointIndex},
            {"success", segment.success}, {"restart", segment.restart}, {"reason", segment.reason},
            {"in_plane_error_trace_voxels", doubleBits(segment.inPlaneErrorTraceVoxels)},
            {"in_plane_error_base_voxels", doubleBits(segment.inPlaneErrorBaseVoxels)},
            {"reference_arc_distance_voxels", doubleBits(segment.referenceArcDistanceVoxels)},
            {"trace", {{"points", exactPoints(trace.points)},
                       {"reached_target_plane", trace.reachedTargetPlane},
                       {"reached_trace_length", trace.reachedTraceLength},
                       {"reason", trace.reason}, {"steps", trace.steps},
                       {"target_plane_crossings", std::move(crossings)},
                       {"selected_target_plane_name", trace.selectedTargetPlaneName},
                       {"selected_target_plane_crossing", trace.selectedTargetPlaneCrossing
                           ? exactPoint(*trace.selectedTargetPlaneCrossing) : nlohmann::json(nullptr)},
                       {"selected_target_plane_error_voxels", doubleBits(trace.selectedTargetPlaneErrorVoxels)}}}});
    }
    const nlohmann::json output = {
        {"format", "vc_fiber_trace_metric_exact_v1"},
        {"double_encoding", "IEEE-754 binary64 bits as 16 hexadecimal digits"},
        {"point_coordinate_space", "trace_voxels"},
        {"trace_to_base_scale", doubleBits(traceToBaseScale)},
        {"segments", std::move(segments)}, {"stitched_trace", exactPoints(result.stitchedTrace)},
        {"restart_count", result.restartCount}, {"lookahead_retry_count", result.lookaheadRetryCount},
        {"lookahead_retry_recovered_count", result.lookaheadRetryRecoveredCount},
        {"segment_count", result.segmentCount}, {"restarts_per_kvx", doubleBits(result.restartsPerKvx)},
        {"reference_length_voxels", doubleBits(result.referenceLengthVoxels)},
        {"reference_length_meters", optionalDoubleBits(result.referenceLengthMeters)},
        {"restarts_per_meter", optionalDoubleBits(result.restartsPerMeter)}};
    writeExactOutput(path, output);
}

vc3d::line_annotation::FiberModeOptimizationRequest makeGuiRequest(
    const CliOptions& options, const vc::fiber_tracer::FiberInput& fiber,
    const vc::fiber_tracer::FiberPredictionField& predictions,
    const vc::lasagna::LasagnaNormalSampler& traceNormals,
    const vc::lasagna::LasagnaNormalSampler& baseNormals,
    double traceToBaseScale, vc::fiber_tracer::FiberTraceProfile& profile)
{
    namespace annotation = vc3d::line_annotation;
    const auto parsed = vc::fiber_tracer::parseVc3dFiberJson(
        nlohmann::json::parse(std::ifstream(options.fiberJson)), options.fiberJson.string());
    annotation::FiberModeOptimizationRequest request;
    request.linePointsBase = fiber.linePointsXyzBase;
    request.predictions = &predictions;
    request.baseNormalSampler = &baseNormals;
    request.traceNormalSampler = &traceNormals;
    request.traceToBaseScale = traceToBaseScale;
    request.traceConfig = options.trace;
    request.traceConfig.traceToBaseScale = traceToBaseScale;
    request.traceConfig.baseVoxelSizeUm = options.voxelSizeUm;
    request.traceConfig.profile = &profile;
    request.traceConfig.endpointAcceptThresholdBaseVoxels = options.errorThresholdBaseVoxels;
    request.normalManifestLocation = options.normalManifestIdentity.empty()
        ? options.normalManifest : options.normalManifestIdentity;
    request.fiberManifestLocation = options.fiberManifestIdentity.empty()
        ? options.fiberManifest : options.fiberManifestIdentity;
    request.globalMode = annotation::fiberOptimizationModeFromString(parsed.optimizationMode);
    request.retraceAll = !options.guiDirtySpan.has_value();
    if (options.guiDirtySpan) {
        if (static_cast<size_t>(*options.guiDirtySpan) + 1 >= fiber.controlPointsXyzBase.size())
            failOption("--gui-dirty-span is outside the saved fiber's control spans");
        request.dirtySegments = std::vector<size_t>{static_cast<size_t>(*options.guiDirtySpan)};
    }
    request.extrapolationDistanceBaseVoxels = annotation::kDefaultExtrapolationDistanceBaseVoxels;
    annotation::configureFiberModeLasagnaDefaults(
        request.lasagnaConfig, annotation::kDefaultExtrapolationDistanceBaseVoxels);
    const double center = static_cast<double>(fiber.linePointsXyzBase.size() - 1) * 0.5;
    size_t seedControl = 0;
    double seedDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < fiber.controlPointsXyzBase.size(); ++i) {
        const size_t lineIndex = fiber.controlPointLineIndices[i];
        annotation::LineControlPoint control{
            static_cast<double>(lineIndex), fiber.controlPointsXyzBase[i], false,
            static_cast<int>(lineIndex)};
        if (i < parsed.segmentMetadata.size() && !parsed.segmentMetadata[i].is_null())
            control.segmentToNext = annotation::fiberTraceSegmentMetadataFromJson(parsed.segmentMetadata[i]);
        request.controlPoints.push_back(std::move(control));
        const double distance = std::abs(static_cast<double>(lineIndex) - center);
        if (distance < seedDistance) {
            seedDistance = distance;
            seedControl = i;
        }
    }
    request.controlPoints[seedControl].isSeed = true;
    return request;
}

void writeGuiTraceOutput(const std::filesystem::path& path,
                         const vc3d::line_annotation::FiberModeOptimizationResult& result)
{
    if (path.empty())
        return;
    auto controls = nlohmann::json::array();
    for (const auto& control : result.controlPoints) {
        controls.push_back({{"point", exactPoint(control.volumePoint)},
                            {"line_position", doubleBits(control.linePosition)},
                            {"optimized_index", control.optimizedIndex}, {"is_seed", control.isSeed},
                            {"accepted_native", vc3d::line_annotation::isAcceptedNativeTrace(control.segmentToNext)},
                            {"segment_to_next", control.segmentToNext
                                ? exactJsonDoubles(vc3d::line_annotation::fiberTraceSegmentMetadataToJson(*control.segmentToNext))
                                : nlohmann::json(nullptr)}});
    }
    const auto normalJson = [](const vc::lasagna::NormalSample& sample) {
        return nlohmann::json{{"normal", exactPoint(sample.normal)},
                               {"valid", sample.valid}, {"reason", sample.reason}};
    };
    auto points = nlohmann::json::array();
    for (const auto& point : result.optimization.line.points) {
        points.push_back({{"position", exactPoint(point.position)},
                          {"sampled_normal", normalJson(point.sampledNormal)}, {"valid", point.valid}});
    }
    auto segmentSamples = nlohmann::json::array();
    for (const auto& segment : result.optimization.line.segmentSamples) {
        auto samples = nlohmann::json::array();
        for (const auto& sample : segment.samples) {
            samples.push_back({{"t", doubleBits(sample.t)}, {"position", exactPoint(sample.position)},
                               {"sampled_normal", normalJson(sample.sampledNormal)}});
        }
        segmentSamples.push_back(std::move(samples));
    }
    const auto& report = result.optimization.report;
    writeExactOutput(path, {
        {"format", "vc_gui_fiber_reoptimization_exact_v1"},
        {"double_encoding", "IEEE-754 binary64 bits as 16 hexadecimal digits"},
        {"point_coordinate_space", "base_voxels"},
        {"control_points", std::move(controls)}, {"points", std::move(points)},
        {"segment_samples", std::move(segmentSamples)},
        {"display_frame_anchor_index", result.optimization.line.displayFrameAnchorIndex},
        {"native_segments", result.nativeSegments}, {"lasagna_fallback_segments", result.lasagnaFallbackSegments},
        {"cspline_fallback_segments", result.csplineFallbackSegments},
        {"native_extrapolations", result.nativeExtrapolations},
        {"lasagna_fallback_extrapolations", result.lasagnaFallbackExtrapolations},
        {"optimization", {{"initial_cost", doubleBits(report.initialCost)}, {"final_cost", doubleBits(report.finalCost)},
                           {"initial_rms", doubleBits(report.initialRms)}, {"final_rms", doubleBits(report.finalRms)},
                           {"residuals", report.residuals}, {"iterations", report.iterations},
                           {"valid_normal_samples", report.validNormalSamples}, {"invalid_normal_samples", report.invalidNormalSamples},
                           {"converged", report.converged}, {"message", report.message}}}});
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parseArgs(argc, argv);
        if (options.modelReaders) {
            vc::render::processChunkCacheService()->configureFetchConcurrency(
                static_cast<std::size_t>(*options.modelReaders), false);
        }
        std::cout << "native_trace2cp_readers adaptive=" << !options.modelReaders.has_value()
                  << " configured_max=" << options.modelReaders.value_or(64) << '\n';
        const auto remoteBefore = vc::lasagna::remoteStoreStats();

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

        std::optional<vc::lasagna::LasagnaDataset> baseNormalDataset;
        std::optional<vc::lasagna::LasagnaNormalSampler> baseNormalSampler;
        if (options.guiReoptimize) {
            auto manifest = normalDataset->manifest();
            manifest.workingToBaseScale = 1.0;
            baseNormalDataset.emplace(std::move(manifest));
            baseNormalSampler.emplace(
                *baseNormalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});
        }

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
        std::optional<vc3d::line_annotation::FiberModeOptimizationRequest> guiRequest;
        if (options.guiReoptimize) {
            guiRequest.emplace(makeGuiRequest(options, fiber, predictions, *normalSampler,
                                              *baseNormalSampler, workingToBaseScale, profile));
        }

        warmModelCorridor(predictions, *normalSampler, fiber,
                          workingToBaseScale, options.prefetchWarmupMs);

        using Clock = std::chrono::steady_clock;
        const auto solveRemoteBefore = vc::lasagna::remoteStoreStats();
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

        vc::fiber_tracer::FiberTraceWholeFiberResult result;
        std::optional<vc3d::line_annotation::FiberModeOptimizationResult> guiResult;
        if (guiRequest) {
            guiResult.emplace(vc3d::line_annotation::optimizeFiberWithNativeFallback(
                std::move(*guiRequest)));
        } else {
            result = vc::fiber_tracer::traceWholeFiberMetric(
                predictions, request, normalSamplerPtr, progress);
        }
        const auto wallEnd = Clock::now();
        const std::clock_t cpuEnd = std::clock();
        const double wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();
        const double cpuSeconds =
            static_cast<double>(cpuEnd - cpuStart) / static_cast<double>(CLOCKS_PER_SEC);

        if (guiResult) {
            writeGuiTraceOutput(options.traceOutput, *guiResult);
            std::cout << "native_trace2cp_gui segments=" << guiResult->controlPoints.size() - 1
                      << " points=" << guiResult->optimization.line.points.size()
                      << " native_segments=" << guiResult->nativeSegments
                      << " lasagna_fallback_segments=" << guiResult->lasagnaFallbackSegments
                      << " cspline_fallback_segments=" << guiResult->csplineFallbackSegments
                      << " native_tails=" << guiResult->nativeExtrapolations
                      << " lasagna_tails=" << guiResult->lasagnaFallbackExtrapolations << '\n';
            std::cout << "native_trace2cp_gui_stages span_trace_ms=" << std::fixed << std::setprecision(3)
                      << guiResult->spanTraceMs << " reinit_ms=" << guiResult->reinitMs
                      << " tail_trace_ms=" << guiResult->tailTraceMs
                      << " tail_normal_pass_ms=" << guiResult->tailNormalPassMs
                      << " report_prefetch_ms=" << guiResult->optimization.report.normalChunkPrefetchMs +
                                                    guiResult->optimization.report.normalMaterializeMs
                      << " report_ceres_ms=" << guiResult->optimization.report.ceresSolveMs
                      << " span_remote_bytes=" << guiResult->spanRemoteBytes
                      << " reinit_remote_bytes=" << guiResult->reinitRemoteBytes
                      << " tail_remote_bytes=" << guiResult->tailRemoteBytes
                      << " dirty_span=" << options.guiDirtySpan.value_or(-1) << '\n';
        } else {
            writeTraceOutput(options.traceOutput, result, workingToBaseScale);
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
                  << " model_prefetch_ms=" << profile.modelPrefetchMs
                  << " model_prefetch_submitted=" << profile.modelPrefetchSubmitted
                  << " model_prefetch_rejected=" << profile.modelPrefetchRejected
                  << '\n';
        // Separate from optimizer timing. A benchmark can wait for optional
        // reads to finish before snapshotting process-wide model-store counters.
        const auto settleStart = Clock::now();
        const auto settleDeadline = settleStart + std::chrono::milliseconds(options.settlePrefetchMs);
        while (vc::render::ChunkCache::speculativePrefetchStats().pendingRequests != 0 &&
               Clock::now() < settleDeadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const auto remoteAfter = vc::lasagna::remoteStoreStats();
        std::cout << "native_trace2cp_remote remote_owned=" << remoteAfter.objectsOwned - remoteBefore.objectsOwned
                  << " remote_bytes=" << remoteAfter.bytesOwned - remoteBefore.bytesOwned
                  << " remote_disk_hits=" << remoteAfter.objectsFromDisk - remoteBefore.objectsFromDisk
                  << " remote_failures=" << remoteAfter.failures - remoteBefore.failures
                  << " remote_owner_ms_sum=" << (remoteAfter.ownerNanoseconds - remoteBefore.ownerNanoseconds) / 1e6
                  << " solve_and_drain_remote_bytes=" << remoteAfter.bytesOwned - solveRemoteBefore.bytesOwned
                  << " pending_requests=" << vc::render::ChunkCache::speculativePrefetchStats().pendingRequests
                  << " settle_ms=" << std::chrono::duration<double, std::milli>(Clock::now() - settleStart).count()
                  << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "vc_fiber_trace_metric error: " << exc.what() << '\n';
        return 1;
    }
}
