#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

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
    double errorThresholdVoxels = 10.0;
    size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
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
        << "  --step-voxels N                 trace step in manifest prediction voxels [4]\n"
        << "  --cone-angle-degrees N          candidate cone half-angle [25]\n"
        << "  --cone-angle-step-degrees N     candidate cone grid step [5]\n"
        << "  --cone-grid-size N              legacy square-to-disk grid size when cone step <= 0 [25]\n"
        << "  --beam-width N                  kept beams per step [8]\n"
        << "  --beam-prune-distance-voxels N  beam endpoint merge radius after lookahead [1]\n"
        << "  --beam-lookahead-steps N        expand this many steps before pruning [2]\n"
        << "  --smoothness-weight N           smoothness scale [2]\n"
        << "  --smoothness-normal-weight N    normal-axis smoothness weight [0.1]\n"
        << "  --smoothness-tangent-weight N   tangent-plane smoothness weight [10]\n"
        << "  --smoothness-free-angle-degrees N free turn before smoothness penalty [0]\n"
        << "  --cumulative-smoothness-steps N history length for cumulative tangent smoothing [4]\n"
        << "  --cumulative-smoothness-tangent-weight N cumulative tangent smoothing weight [2]\n"
        << "  --max-step-factor N             max steps as factor of CP span [3]\n"
        << "  --error-threshold-voxels N      restart threshold at target plane [10]\n"
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
        } else if (arg == "--error-threshold-voxels") {
            options.errorThresholdVoxels =
                parseDouble(requireValue(i, argc, argv, "error-threshold-voxels"),
                            "error-threshold-voxels");
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
    if (!(options.errorThresholdVoxels >= 0.0))
        failOption("--error-threshold-voxels must be non-negative");
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
        const double workingToBaseScale =
            vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(
                openedDataset.manifest());
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
                << " working_to_base_scale=" << workingToBaseScale
                << " working_to_base_scale_source=manifest"
                << " normal_sampler=" << (normalSamplerPtr != nullptr ? "on" : "off")
                << '\n';
        }

        vc::fiber_tracer::FiberTraceWholeFiberMetricRequest request;
        request.fiber = fiber;
        request.workingToBaseScale = workingToBaseScale;
        request.errorThresholdVoxels = options.errorThresholdVoxels;
        request.voxelSizeUm = options.voxelSizeUm;
        request.config = options.trace;

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
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "vc_fiber_trace_metric error: " << exc.what() << '\n';
        return 1;
    }
}
