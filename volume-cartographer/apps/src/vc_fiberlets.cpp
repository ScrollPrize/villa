#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "FiberTraceCli.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

enum class Command {
    Anchors,
    Paths,
    FiberReplay,
};

struct CliOptions {
    Command command = Command::Anchors;
    std::string manifestLocation;
    std::filesystem::path anchorArtifact;
    std::filesystem::path fiberJson;
    std::string normalManifestLocation;
    std::filesystem::path outputDirectory;
    std::filesystem::path remoteCacheDirectory;
    vc::fiber_tracer::FiberAnchorConfig anchors;
    vc::fiber_tracer::FiberletPathConfig paths;
    std::optional<vc::fiber_tracer::FiberAnchorCrop> baseCrop;
    std::optional<double> corridorRadiusBaseVoxels;
    std::optional<double> falloffSigmaBaseVoxels;
    std::optional<double> localWindowBaseVoxels;
    std::optional<double> baseVoxelSizeUm;
    double glyphLengthBaseVoxels = 16.0;
    size_t decodedCacheBytes = 512ULL * 1024ULL * 1024ULL;
    bool printStats = false;
    bool writePresenceSlices = true;
    int inferenceScaledownPower = 2;
    double failureThresholdBaseVoxels = 20.0;
    int postrollSteps = 100;
    double alongBaseVoxels = 512.0;
    double radiusBaseVoxels = 128.0;
    double matchRefineSteps = 1.0;
    vc::fiber_tracer::FiberTraceConfig trace;
    vc::fiber_tracer::cli::SeenOptions seenTraceOptions;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable)
{
    std::cerr << "Usage:\n"
              << "  " << executable << " anchors <fiber.lasagna.json-or-url> <output-dir> [options]\n"
              << "  " << executable
              << " paths <fiber.lasagna.json-or-url> <anchors.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "  " << executable
              << " fiber-replay <fiber.lasagna.json-or-url> <fiber.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "Common options:\n"
              << "  --threads N                   decode/search workers [hardware]\n"
              << "  --cache-gib N                 decoded chunk cache budget [0.5]\n"
              << "  --remote-cache-dir PATH       required for direct remote manifests\n"
              << "  --base-voxel-size-um N        optional physical reporting metadata\n\n"
              << "Anchor options:\n"
              << "  --cell-size N                 prediction-grid cell side, 2..8 [4]\n"
              << "  --falloff N                   normal-plane sigma in base voxels [cell-side/2]\n"
              << "  --window N                    refinement/NMS radius in base voxels [cell-side]\n"
              << "  --presence-floor N            inclusive observation floor [0.05]\n"
              << "  --minimum-support N           inclusive aligned support [0.05]\n"
              << "  --merge-angle-deg N           maximum duplicate-axis angle [10]\n"
              << "  --merge-abs-loss N            maximum normalized merge loss [0.01]\n"
              << "  --merge-rel-loss N            maximum relative merge loss [0.05]\n"
              << "  --maximum-seeds N             deterministic PCA seed count [8]\n"
              << "  --maximum-iterations N        assignment/PCA iteration limit [64]\n"
              << "  --crop X,Y,Z,W,H,D            base-volume box; selects intersected cells\n"
              << "  --glyph-length-base-voxels N  diagnostic anchor length [16]\n\n"
              << "Path options:\n"
              << "  --normal-manifest PATH        required regular Lasagna normals\n"
              << "  --cell-radius N               candidate cell-shell radius [4]\n"
              << "  --shell-half-width N          candidate shell half-width [0.5]\n"
              << "  --endpoint-angle-degrees N    endpoint/chord and attachment bound [45]\n"
              << "  --corridor-radius N           base voxels [one anchor-cell width]\n"
              << "  --invalid-prediction-cost N   invalid cost per prediction-grid voxel [4]\n"
              << "  --smoothness-weight N         invalid-normal isotropic weight [2]\n"
              << "  --smoothness-normal-weight N  normal-tilt weight [0.1]\n"
              << "  --smoothness-tangent-weight N tangent-plane turn weight [10]\n"
              << "  --smoothness-free-angle N     lattice free angle in degrees [45]\n"
              << "  --stats                       print path-count and score statistics\n"
              << "  --no-slices                   skip central presence-slice outputs\n";
    std::cerr << "\nReplay options:\n"
              << "  --fail N                      dense-reference failure distance in base voxels [20]\n"
              << "  --after N                     greedy steps retained after failure [100]\n"
              << "  --along N                     reference distance each side of failure [512]\n"
              << "  --radius N                    extraction tube radius in base voxels [128]\n"
              << "  --match-refine N              forward match refinement in trace steps [1]\n"
              << "  --inference-scaledown-power N prediction scaledown relative to trace voxels [2]\n";
}

std::string valueAfter(int& index, int argc, char** argv, const char* name)
{
    if (index + 1 >= argc)
        fail(std::string("--") + name + " requires a value");
    return argv[++index];
}

double parseDouble(const std::string& text, const char* name)
{
    size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value))
        fail(std::string("--") + name + " requires a finite number");
    return value;
}

int parseInt(const std::string& text, const char* name)
{
    size_t parsed = 0;
    const long long value = std::stoll(text, &parsed);
    if (parsed != text.size() || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        fail(std::string("--") + name + " requires an integer");
    }
    return static_cast<int>(value);
}

vc::fiber_tracer::FiberAnchorCrop parseCrop(const std::string& text)
{
    std::array<size_t, 6> values{};
    std::stringstream input(text);
    std::string token;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::getline(input, token, ',') || token.empty())
            fail("--crop requires X,Y,Z,W,H,D");
        if (token.front() == '-')
            fail("--crop contains an invalid integer");
        size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size() || value > std::numeric_limits<size_t>::max())
            fail("--crop contains an invalid integer");
        values[index] = static_cast<size_t>(value);
    }
    if (std::getline(input, token, ','))
        fail("--crop requires exactly six integers");
    return {{values[0], values[1], values[2]}, {values[3], values[4], values[5]}};
}

CliOptions parseArgs(int argc, char** argv)
{
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        std::exit(0);
    }
    if (argc < 4) {
        usage(argv[0]);
        std::exit(2);
    }
    CliOptions options;
    const std::string command = argv[1];
    int firstOption = 0;
    if (command == "anchors") {
        options.command = Command::Anchors;
        options.manifestLocation = argv[2];
        options.outputDirectory = argv[3];
        firstOption = 4;
    } else if (command == "paths") {
        if (argc < 5) {
            usage(argv[0]);
            std::exit(2);
        }
        options.command = Command::Paths;
        options.manifestLocation = argv[2];
        options.anchorArtifact = argv[3];
        options.outputDirectory = argv[4];
        firstOption = 5;
    } else if (command == "fiber-replay") {
        if (argc < 5) {
            usage(argv[0]);
            std::exit(2);
        }
        options.command = Command::FiberReplay;
        options.manifestLocation = argv[2];
        options.fiberJson = argv[3];
        options.outputDirectory = argv[4];
        firstOption = 5;
    } else {
        usage(argv[0]);
        std::exit(2);
    }
    const int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    options.anchors.parallelThreads = workers;
    options.paths.parallelThreads = workers;
    options.trace.parallelThreads = workers;
    for (int index = firstOption; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (argument == "--threads") {
            const int value = parseInt(valueAfter(index, argc, argv, "threads"), "threads");
            options.anchors.parallelThreads = value;
            options.paths.parallelThreads = value;
            options.trace.parallelThreads = value;
        } else if (argument == "--cache-gib") {
            const double gib = parseDouble(valueAfter(index, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0) || gib > 1024.0)
                fail("--cache-gib must be in (0, 1024]");
            options.decodedCacheBytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = valueAfter(index, argc, argv, "remote-cache-dir");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(valueAfter(index, argc, argv, "base-voxel-size-um"), "base-voxel-size-um");
        } else if (argument == "--normal-manifest" && options.command != Command::Anchors) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
        } else if (argument == "--fail" && options.command == Command::FiberReplay) {
            options.failureThresholdBaseVoxels = parseDouble(valueAfter(index, argc, argv, "fail"), "fail");
        } else if (argument == "--after" && options.command == Command::FiberReplay) {
            options.postrollSteps = parseInt(valueAfter(index, argc, argv, "after"), "after");
        } else if (argument == "--along" && options.command == Command::FiberReplay) {
            options.alongBaseVoxels = parseDouble(valueAfter(index, argc, argv, "along"), "along");
        } else if (argument == "--radius" && options.command == Command::FiberReplay) {
            options.radiusBaseVoxels = parseDouble(valueAfter(index, argc, argv, "radius"), "radius");
        } else if (argument == "--match-refine" && options.command == Command::FiberReplay) {
            options.matchRefineSteps = parseDouble(valueAfter(index, argc, argv, "match-refine"), "match-refine");
        } else if (argument == "--inference-scaledown-power" && options.command == Command::FiberReplay) {
            options.inferenceScaledownPower = parseInt(valueAfter(index, argc, argv, "inference-scaledown-power"), "inference-scaledown-power");
        } else if (argument == "--cell-size" && options.command != Command::Paths) {
            options.anchors.cellSizePredictionVoxels = parseInt(valueAfter(index, argc, argv, "cell-size"), "cell-size");
        } else if (argument == "--falloff" && options.command != Command::Paths) {
            options.falloffSigmaBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "falloff"), "falloff");
        } else if (argument == "--window" && options.command != Command::Paths) {
            options.localWindowBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "window"), "window");
        } else if (argument == "--presence-floor" && options.command != Command::Paths) {
            options.anchors.observationPresenceFloor = parseDouble(valueAfter(index, argc, argv, "presence-floor"), "presence-floor");
        } else if (argument == "--minimum-support" && options.command != Command::Paths) {
            options.anchors.minimumAlignedSupport = parseDouble(valueAfter(index, argc, argv, "minimum-support"), "minimum-support");
        } else if (argument == "--merge-angle-deg" && options.command != Command::Paths) {
            options.anchors.mergeMaximumAngleDegrees = parseDouble(valueAfter(index, argc, argv, "merge-angle-deg"), "merge-angle-deg");
        } else if (argument == "--merge-abs-loss" && options.command != Command::Paths) {
            options.anchors.mergeMaximumAbsoluteObjectiveLoss =
                parseDouble(valueAfter(index, argc, argv, "merge-abs-loss"), "merge-abs-loss");
        } else if (argument == "--merge-rel-loss" && options.command != Command::Paths) {
            options.anchors.mergeMaximumRelativeObjectiveLoss =
                parseDouble(valueAfter(index, argc, argv, "merge-rel-loss"), "merge-rel-loss");
        } else if (argument == "--maximum-seeds" && options.command != Command::Paths) {
            const int value = parseInt(valueAfter(index, argc, argv, "maximum-seeds"), "maximum-seeds");
            if (value <= 0)
                fail("--maximum-seeds must be positive");
            options.anchors.maximumSeedCount = static_cast<size_t>(value);
        } else if (argument == "--maximum-iterations" && options.command != Command::Paths) {
            options.anchors.maximumIterations = parseInt(valueAfter(index, argc, argv, "maximum-iterations"), "maximum-iterations");
        } else if (argument == "--crop" && options.command == Command::Anchors) {
            options.baseCrop = parseCrop(valueAfter(index, argc, argv, "crop"));
        } else if (argument == "--glyph-length-base-voxels" && options.command == Command::Anchors) {
            options.glyphLengthBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "glyph-length-base-voxels"), "glyph-length-base-voxels");
        } else if (argument == "--cell-radius" && options.command == Command::Paths) {
            options.paths.cellRadius = parseInt(valueAfter(index, argc, argv, "cell-radius"), "cell-radius");
        } else if (argument == "--shell-half-width" && options.command == Command::Paths) {
            options.paths.shellHalfWidthCells = parseDouble(valueAfter(index, argc, argv, "shell-half-width"), "shell-half-width");
        } else if (argument == "--endpoint-angle-degrees" && options.command == Command::Paths) {
            options.paths.maximumEndpointAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "endpoint-angle-degrees"), "endpoint-angle-degrees");
        } else if (argument == "--corridor-radius" && options.command == Command::Paths) {
            options.corridorRadiusBaseVoxels = parseDouble(valueAfter(index, argc, argv, "corridor-radius"), "corridor-radius");
            if (!(*options.corridorRadiusBaseVoxels > 0.0))
                fail("--corridor-radius must be positive");
        } else if (argument == "--stats" && options.command == Command::Paths) {
            options.printStats = true;
        } else if (argument == "--no-slices" && options.command == Command::Paths) {
            options.writePresenceSlices = false;
        } else if (argument == "--invalid-prediction-cost" && options.command == Command::Paths) {
            options.paths.invalidPredictionCostPerVoxel =
                parseDouble(valueAfter(index, argc, argv, "invalid-prediction-cost"), "invalid-prediction-cost");
        } else if (argument == "--smoothness-weight" && options.command == Command::Paths) {
            options.paths.smoothnessWeight = parseDouble(valueAfter(index, argc, argv, "smoothness-weight"), "smoothness-weight");
        } else if (argument == "--smoothness-normal-weight" && options.command == Command::Paths) {
            options.paths.smoothnessNormalWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-normal-weight"), "smoothness-normal-weight");
        } else if (argument == "--smoothness-tangent-weight" && options.command == Command::Paths) {
            options.paths.smoothnessTangentWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-tangent-weight"), "smoothness-tangent-weight");
        } else if (argument == "--smoothness-free-angle" && options.command == Command::Paths) {
            options.paths.smoothnessFreeAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "smoothness-free-angle"), "smoothness-free-angle");
        } else if (options.command == Command::FiberReplay &&
                   vc::fiber_tracer::cli::parseTraceOption(
                       argument, index, argc, argv, options.trace,
                       &options.seenTraceOptions)) {
            continue;
        } else {
            fail("unknown option for selected command: " + argument);
        }
    }
    if (options.baseVoxelSizeUm.has_value() && !(*options.baseVoxelSizeUm > 0.0))
        fail("--base-voxel-size-um must be positive");
    if (options.command != Command::Paths) {
        if (options.falloffSigmaBaseVoxels.has_value() &&
            !(*options.falloffSigmaBaseVoxels > 0.0))
            fail("--falloff must be positive");
        if (options.localWindowBaseVoxels.has_value() &&
            !(*options.localWindowBaseVoxels > 0.0))
            fail("--window must be positive");
    }
    if (options.command != Command::Anchors) {
        if (options.normalManifestLocation.empty())
            fail("paths and fiber-replay require --normal-manifest");
        vc::fiber_tracer::validateFiberletPathConfig(options.paths);
    }
    if (options.command == Command::FiberReplay) {
        if (!(options.failureThresholdBaseVoxels >= 0.0) || options.postrollSteps < 0 ||
            !(options.alongBaseVoxels >= 0.0) || !(options.radiusBaseVoxels > 0.0) ||
            !(options.matchRefineSteps >= 0.0) || options.inferenceScaledownPower < 0 ||
            options.inferenceScaledownPower > 30) {
            fail("fiber-replay options are outside their valid range");
        }
        vc::fiber_tracer::cli::validateTraceOptions(options.trace);
        if ((options.seenTraceOptions.beamWidth && options.trace.beamWidth != 1) ||
            (options.seenTraceOptions.beamLookahead && options.trace.beamLookaheadSteps != 1)) {
            fail("fiber-replay only supports --beam-width 1 and --beam-lookahead-steps 1");
        }
    }
    const bool remote = vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) ||
                        (options.command != Command::Anchors && vc::lasagna::isRemoteLasagnaLocation(options.normalManifestLocation));
    if (remote && options.remoteCacheDirectory.empty())
        fail("direct remote manifests require --remote-cache-dir");
    return options;
}

std::string fileHash(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot hash file: " + path.string());
    uint64_t hash = 14695981039346656037ULL;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(index)]);
            hash *= 1099511628211ULL;
        }
    }
    if (!input.eof())
        throw std::runtime_error("failed while hashing file: " + path.string());
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string stringHash(const std::string& value)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string credentialFreeLocator(std::string locator)
{
    const auto scheme = locator.find("://");
    if (scheme == std::string::npos)
        return locator;
    const size_t authorityBegin = scheme + 3;
    const size_t authorityEnd = locator.find('/', authorityBegin);
    const size_t at = locator.rfind('@', authorityEnd);
    if (at != std::string::npos && at >= authorityBegin)
        locator.erase(authorityBegin, at - authorityBegin + 1);
    const size_t suffix = locator.find_first_of("?#");
    if (suffix != std::string::npos)
        locator.erase(suffix);
    return locator;
}

std::string datasetLocator(const vc::lasagna::LasagnaDataset& dataset)
{
    const auto& manifest = dataset.manifest();
    return manifest.manifestIsRemote ? credentialFreeLocator(manifest.manifestLocation)
                                     : std::filesystem::absolute(manifest.manifestPath).lexically_normal().string();
}

void printRateProgress(
    const char* prefix,
    const std::string& phase,
    const char* rateName,
    size_t completed,
    size_t total,
    double elapsedSeconds)
{
    const double percent = total == 0
        ? 100.0
        : 100.0 * static_cast<double>(completed) / static_cast<double>(total);
    const double rate = elapsedSeconds > 0.0
        ? static_cast<double>(completed) / elapsedSeconds
        : 0.0;
    const double eta = completed >= total
        ? 0.0
        : rate > 0.0
            ? static_cast<double>(total - completed) / rate
            : std::numeric_limits<double>::infinity();
    std::ostringstream line;
    line << std::fixed << std::setprecision(1) << prefix;
    if (!phase.empty())
        line << " phase=" << phase;
    line << " completed=" << completed
         << " total=" << total
         << " percent=" << percent
         << " elapsed_seconds=" << elapsedSeconds
         << ' ' << rateName << '=' << rate
         << " eta_seconds=";
    if (std::isfinite(eta))
        line << eta;
    else
        line << "n/a";
    std::cerr << line.str() << '\n';
}

void printAnchorProgress(const vc::fiber_tracer::FiberAnchorProgress& progress)
{
    printRateProgress(
        "fiber_anchor_progress",
        progress.phase,
        "cells_per_second",
        progress.completed,
        progress.total,
        progress.elapsedSeconds);
}

void printFiberletProgress(const vc::fiber_tracer::FiberletPathProgress& progress)
{
    printRateProgress(
        "fiberlet_progress",
        {},
        "candidates_per_second",
        progress.completed,
        progress.total,
        progress.elapsedSeconds);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        CliOptions options = parseArgs(argc, argv);
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(options.manifestLocation, openOptions);
        const vc::fiber_tracer::FiberPredictionField field(dataset, options.decodedCacheBytes, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
        const auto grid = field.storedGridInfo();

        if (options.command == Command::FiberReplay) {
            const auto traceSetupStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=trace_setup status=started\n";
            const auto scales = vc::fiber_tracer::resolveFiberPredictionTraceScales(
                dataset.manifest(), options.inferenceScaledownPower);
            auto traceManifest = dataset.manifest();
            traceManifest.workingToBaseScale = scales.traceToBaseScale;
            const vc::lasagna::LasagnaDataset traceDataset(std::move(traceManifest));
            const vc::fiber_tracer::FiberPredictionField traceField(
                traceDataset,
                options.decodedCacheBytes,
                vc::fiber_tracer::FiberPredictionFieldBindingMode::TraceOptions);

            vc::lasagna::LasagnaDatasetOpenOptions traceNormalOptions;
            traceNormalOptions.workingToBaseScale = scales.traceToBaseScale;
            traceNormalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto traceNormalDataset = vc::lasagna::LasagnaDataset::openLocation(
                options.normalManifestLocation, traceNormalOptions);
            const vc::lasagna::LasagnaNormalSampler traceNormalSampler(
                traceNormalDataset,
                vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});

            const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
            const auto requestedTrace = options.trace;
            auto effectiveTrace = requestedTrace;
            effectiveTrace.beamWidth = 1;
            effectiveTrace.beamLookaheadSteps = 1;
            effectiveTrace.traceToBaseScale = scales.traceToBaseScale;
            vc::fiber_tracer::FiberReplayTraceRequest replayRequest;
            replayRequest.fiber = fiber;
            replayRequest.traceToBaseScale = scales.traceToBaseScale;
            replayRequest.errorThresholdBaseVoxels =
                options.failureThresholdBaseVoxels;
            replayRequest.matchRefineSteps = options.matchRefineSteps;
            replayRequest.postrollSteps = options.postrollSteps;
            replayRequest.config = effectiveTrace;

            std::cerr << "fiber_replay_stage stage=trace_setup status=completed"
                      << " elapsed_seconds="
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - traceSetupStart)
                             .count()
                      << '\n';
            const auto traceStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=trace status=started\n";
            const auto replay = vc::fiber_tracer::traceFiberReplay(
                traceField,
                replayRequest,
                &traceNormalSampler,
                [](const vc::fiber_tracer::FiberTraceProgress& event) {
                    if (event.step == event.maxSteps || event.step % 100 == 0) {
                        std::cerr << "fiber_replay_progress step=" << event.step
                                  << '/' << event.maxSteps
                                  << " reason=" << event.reason << '\n';
                    }
                });
            std::cerr << "fiber_replay_stage stage=trace status=completed"
                      << " elapsed_seconds="
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - traceStart)
                             .count()
                      << " result=" << vc::fiber_tracer::fiberReplayStatusName(
                             replay.status)
                      << '\n';

            const auto reference = vc::fiber_tracer::makePolylineArcGeometry(
                fiber.linePointsXyzBase);
            const double startArc = reference.vertexArcs.at(
                fiber.controlPointLineIndices.front());
            vc::fiber_tracer::FiberReplayBundleInput bundle;
            bundle.request = replayRequest;
            bundle.replay = replay;
            bundle.referenceGeometryBase = vc::fiber_tracer::slicePolylineArc(
                reference, startArc, reference.length());
            bundle.sources = {
                {"fiber_manifest", datasetLocator(dataset)},
                {"fiber_manifest_content_hash", fileHash(dataset.manifest().manifestPath)},
                {"normal_manifest", datasetLocator(traceNormalDataset)},
                {"normal_manifest_content_hash", fileHash(traceNormalDataset.manifest().manifestPath)},
                {"fiber_json", std::filesystem::absolute(options.fiberJson).lexically_normal().string()},
                {"fiber_json_content_hash", fileHash(options.fiberJson)},
            };
            bundle.traceBinding = {
                {"mode", "trace_options"},
                {"trace_to_base_scale", scales.traceToBaseScale},
                {"prediction_to_base_scale", scales.predictionToBaseScale},
                {"prediction_spacing_trace_voxels", scales.predictionSpacingInTraceVoxels},
            };
            bundle.predictionBinding = {
                {"mode", "canonical_stored_grid"},
                {"prediction_to_base_scale", grid.predictionToBaseScale},
                {"prediction_shape_zyx", grid.shapeZYX},
            };
            bundle.requestedTraceConfig =
                vc::fiber_tracer::cli::traceConfigJson(requestedTrace);
            bundle.effectiveTraceConfig =
                vc::fiber_tracer::cli::traceConfigJson(effectiveTrace);

            const bool failed =
                replay.status == vc::fiber_tracer::FiberReplayStatus::FailureWithPostroll ||
                replay.status == vc::fiber_tracer::FiberReplayStatus::FailureTruncated;
            if (failed) {
                const auto tubeStart = std::chrono::steady_clock::now();
                std::cerr << "fiber_replay_stage stage=tube status=started\n";
                const auto tube = vc::fiber_tracer::makeFiberReplayTube(
                    fiber.linePointsXyzBase,
                    *replay.failureReferenceArcBase,
                    options.alongBaseVoxels,
                    options.radiusBaseVoxels,
                    grid,
                    options.anchors.cellSizePredictionVoxels);
                bundle.tube = tube;
                bundle.referenceGeometryBase = tube.referenceIntervalBase;
                std::cerr << "fiber_replay_stage stage=tube status=completed"
                          << " elapsed_seconds="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - tubeStart)
                                 .count()
                          << " cells=" << tube.cellsZYX.size() << '\n';

                const double cellSideBase =
                    options.anchors.cellSizePredictionVoxels *
                    grid.predictionToBaseScale;
                options.anchors.gaussianSigmaPredictionVoxels =
                    options.falloffSigmaBaseVoxels.value_or(cellSideBase * 0.5) /
                    grid.predictionToBaseScale;
                options.anchors.localWindowRadiusPredictionVoxels =
                    options.localWindowBaseVoxels.value_or(cellSideBase) /
                    grid.predictionToBaseScale;
                options.anchors.axialSupportHalfWidthPredictionVoxels =
                    1.5 * options.anchors.cellSizePredictionVoxels;
                options.anchors.nmsLongitudinalRadiusPredictionVoxels =
                    0.5 * options.anchors.cellSizePredictionVoxels;
                options.anchors.nmsMaximumAngleDegrees =
                    options.anchors.mergeMaximumAngleDegrees;
                vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
                const auto anchorsStart = std::chrono::steady_clock::now();
                std::cerr << "fiber_replay_stage stage=anchors status=started"
                          << " cells=" << tube.cellsZYX.size() << '\n';
                bundle.anchors = vc::fiber_tracer::extractFiberAnchorsForCells(
                    grid,
                    options.anchors,
                    [&](const auto& indices, int threads, auto& samples) {
                        field.sampleStoredGridBatch(indices, threads, samples);
                    },
                    tube.cellsZYX,
                    [&](const vc::fiber_tracer::FiberAnchor& anchor) {
                        return tube.containsPredictionPoint(
                            anchor.positionPredictionXYZ,
                            grid.predictionToBaseScale);
                    },
                    printAnchorProgress);
                const auto anchorCount =
                    bundle.anchors->diagnostics.oneAnchorCells +
                    2 * bundle.anchors->diagnostics.twoAnchorCells;
                std::cerr << "fiber_replay_stage stage=anchors status=completed"
                          << " elapsed_seconds="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - anchorsStart)
                                 .count()
                          << " anchors=" << anchorCount << '\n';
                vc::fiber_tracer::FiberAnchorArtifactInfo anchorArtifact;
                anchorArtifact.sourceLocator = datasetLocator(dataset);
                anchorArtifact.manifestContentHash =
                    fileHash(dataset.manifest().manifestPath);
                anchorArtifact.glyphLengthBaseVoxels =
                    options.glyphLengthBaseVoxels;
                anchorArtifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
                bundle.anchorArtifact = anchorArtifact;

                vc::lasagna::LasagnaDatasetOpenOptions canonicalNormalOptions;
                canonicalNormalOptions.workingToBaseScale =
                    grid.predictionToBaseScale;
                canonicalNormalOptions.remoteCacheRoot =
                    options.remoteCacheDirectory;
                const auto canonicalNormalDataset =
                    vc::lasagna::LasagnaDataset::openLocation(
                        options.normalManifestLocation, canonicalNormalOptions);
                const vc::lasagna::LasagnaNormalSampler canonicalNormalSampler(
                    canonicalNormalDataset,
                    vc::lasagna::LasagnaNormalSamplerOptions{
                        options.decodedCacheBytes});
                vc::fiber_tracer::LoadedFiberAnchorArtifact loadedAnchors{
                    *bundle.anchors, anchorArtifact};
                const auto fiberletsStart = std::chrono::steady_clock::now();
                std::cerr << "fiber_replay_stage stage=fiberlets status=started"
                          << " anchors=" << anchorCount << '\n';
                bundle.paths = vc::fiber_tracer::traceFiberletPaths(
                    loadedAnchors,
                    grid,
                    options.paths,
                    [&](const auto& indices, int threads, auto& samples) {
                        field.sampleStoredGridBatch(indices, threads, samples);
                    },
                    canonicalNormalSampler,
                    printFiberletProgress,
                    [&](const cv::Vec3d& pointPrediction) {
                        return tube.containsPredictionPoint(
                            pointPrediction, grid.predictionToBaseScale);
                    });
                std::cerr << "fiber_replay_stage stage=fiberlets status=completed"
                          << " elapsed_seconds="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - fiberletsStart)
                                 .count()
                          << " searched="
                          << bundle.paths->diagnostics.searchedPairs
                          << " accepted="
                          << bundle.paths->diagnostics.successfulPaths
                          << " preloaded_voxels=" << bundle.paths->preloadedVoxels
                          << '\n';
                vc::fiber_tracer::FiberletArtifactInfo pathArtifact;
                pathArtifact.fiberManifestLocator = datasetLocator(dataset);
                pathArtifact.fiberManifestContentHash =
                    anchorArtifact.manifestContentHash;
                pathArtifact.normalManifestLocator =
                    datasetLocator(canonicalNormalDataset);
                pathArtifact.normalManifestContentHash =
                    fileHash(canonicalNormalDataset.manifest().manifestPath);
                pathArtifact.anchorArtifactLocator = "anchors/anchors.json";
                pathArtifact.anchorArtifactContentHash = stringHash(
                    vc::fiber_tracer::fiberAnchorReportJson(
                        *bundle.anchors, anchorArtifact).dump(2) + "\n");
                pathArtifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
                bundle.pathArtifact = pathArtifact;
            }

            const auto publishStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=publish status=started\n";
            const auto resultBundle = vc::fiber_tracer::writeFiberReplayBundle(
                options.outputDirectory, bundle);
            std::cerr << "fiber_replay_stage stage=publish status=completed"
                      << " elapsed_seconds="
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - publishStart)
                             .count()
                      << '\n';
            std::cout << "fiber_replay status="
                      << resultBundle.at("status").get<std::string>()
                      << " trace_points=" << replay.tracePointsBase.size()
                      << " matches=" << replay.matches.size()
                      << " postroll=" << replay.completedPostrollSteps;
            if (bundle.tube.has_value()) {
                const auto anchorCount = bundle.anchors->diagnostics.oneAnchorCells +
                    2 * bundle.anchors->diagnostics.twoAnchorCells;
                std::cout << " cells=" << bundle.tube->cellsZYX.size()
                          << " anchors=" << anchorCount
                          << " fiberlets=" << bundle.paths->diagnostics.successfulPaths
                          << " preloaded_voxels=" << bundle.paths->preloadedVoxels;
            }
            std::cout << '\n';
            return 0;
        }

        if (options.command == Command::Anchors) {
            const double cellSideBase =
                options.anchors.cellSizePredictionVoxels *
                grid.predictionToBaseScale;
            options.anchors.gaussianSigmaPredictionVoxels =
                options.falloffSigmaBaseVoxels.value_or(cellSideBase * 0.5) /
                grid.predictionToBaseScale;
            options.anchors.localWindowRadiusPredictionVoxels =
                options.localWindowBaseVoxels.value_or(cellSideBase) /
                grid.predictionToBaseScale;
            options.anchors.axialSupportHalfWidthPredictionVoxels =
                1.5 * options.anchors.cellSizePredictionVoxels;
            options.anchors.nmsLongitudinalRadiusPredictionVoxels =
                0.5 * options.anchors.cellSizePredictionVoxels;
            options.anchors.nmsMaximumAngleDegrees =
                options.anchors.mergeMaximumAngleDegrees;
            vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
            const auto crop = options.baseCrop.has_value()
                                  ? std::make_optional(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(*options.baseCrop, grid.predictionToBaseScale))
                                  : std::nullopt;
            const auto report = vc::fiber_tracer::extractFiberAnchors(
                grid,
                options.anchors,
                [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
                crop,
                printAnchorProgress);
            vc::fiber_tracer::FiberAnchorArtifactInfo artifact;
            artifact.sourceLocator = datasetLocator(dataset);
            artifact.manifestContentHash = fileHash(dataset.manifest().manifestPath);
            artifact.glyphLengthBaseVoxels = options.glyphLengthBaseVoxels;
            artifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
            vc::fiber_tracer::writeFiberAnchorArtifacts(options.outputDirectory, report, artifact);

            std::cout << "prediction_shape_zyx=" << grid.shapeZYX[0] << ',' << grid.shapeZYX[1] << ',' << grid.shapeZYX[2]
                      << " prediction_to_base=" << grid.predictionToBaseScale << " cell_side_base_voxels=" << cellSideBase
                      << " falloff_sigma_base_voxels=" << options.anchors.gaussianSigmaPredictionVoxels * grid.predictionToBaseScale
                      << " local_window_base_voxels=" << options.anchors.localWindowRadiusPredictionVoxels * grid.predictionToBaseScale
                      << " cell_diagonal_base_voxels=" << cellSideBase * std::sqrt(3.0) << " cells=" << report.diagnostics.totalCells
                      << " anchors=" << report.diagnostics.oneAnchorCells + 2 * report.diagnostics.twoAnchorCells
                      << " zero=" << report.diagnostics.zeroAnchorCells << " one=" << report.diagnostics.oneAnchorCells
                      << " two=" << report.diagnostics.twoAnchorCells << " merged=" << report.diagnostics.mergedComponentPairs
                      << " nms_suppressed=" << report.diagnostics.nmsSuppressedComponents
                      << " elapsed_seconds=" << report.elapsedSeconds << '\n';
            if (options.baseVoxelSizeUm.has_value()) {
                std::cout << "cell_side_um=" << cellSideBase * *options.baseVoxelSizeUm
                          << " cell_diagonal_um=" << cellSideBase * std::sqrt(3.0) * *options.baseVoxelSizeUm << '\n';
            }
            return 0;
        }

        if (options.corridorRadiusBaseVoxels.has_value()) {
            options.paths.corridorRadiusPredictionVoxels = *options.corridorRadiusBaseVoxels / grid.predictionToBaseScale;
            vc::fiber_tracer::validateFiberletPathConfig(options.paths);
        }

        const auto anchors = vc::fiber_tracer::loadFiberAnchorArtifact(options.anchorArtifact);
        const std::string manifestHash = fileHash(dataset.manifest().manifestPath);
        if (manifestHash != anchors.artifact.manifestContentHash)
            fail("fiber manifest content hash does not match anchors.json");
        if (grid.shapeZYX != anchors.report.grid.shapeZYX || std::abs(grid.predictionToBaseScale - anchors.report.grid.predictionToBaseScale) > 1.0e-12) {
            fail("fiber prediction grid does not match anchors.json");
        }

        vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
        normalOptions.workingToBaseScale = grid.predictionToBaseScale;
        normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, normalOptions);
        const vc::lasagna::LasagnaNormalSampler normalSampler(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});
        const auto report = vc::fiber_tracer::traceFiberletPaths(
            anchors,
            grid,
            options.paths,
            [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
            normalSampler,
            printFiberletProgress);
        std::optional<vc::fiber_tracer::FiberPresenceSliceReport> presenceSlices;
        if (options.writePresenceSlices) {
            const auto sliceCrop = vc::fiber_tracer::fiberAnchorCellCoverageCrop(anchors);
            presenceSlices = vc::fiber_tracer::sampleFiberPresenceSlices(
                sliceCrop,
                grid,
                [&](const auto& indices, int threads, auto& samples) { field.sampleStoredPresenceBatch(indices, threads, samples); },
                options.paths.parallelThreads);
        }
        vc::fiber_tracer::FiberletArtifactInfo artifact;
        artifact.fiberManifestLocator = datasetLocator(dataset);
        artifact.fiberManifestContentHash = manifestHash;
        artifact.normalManifestLocator = datasetLocator(normalDataset);
        artifact.normalManifestContentHash = fileHash(normalDataset.manifest().manifestPath);
        artifact.anchorArtifactLocator = std::filesystem::absolute(options.anchorArtifact).lexically_normal().string();
        artifact.anchorArtifactContentHash = fileHash(options.anchorArtifact);
        artifact.baseVoxelSizeUm = options.baseVoxelSizeUm.has_value() ? options.baseVoxelSizeUm : anchors.artifact.baseVoxelSizeUm;
        vc::fiber_tracer::writeFiberletPathArtifacts(options.outputDirectory, report, artifact);
        if (presenceSlices.has_value()) {
            vc::fiber_tracer::writeFiberPresenceSliceArtifacts(options.outputDirectory, *presenceSlices, grid);
        } else {
            vc::fiber_tracer::removeFiberPresenceSliceArtifacts(options.outputDirectory);
        }
        std::cout << "anchors=" << report.diagnostics.occupiedAnchors << " shell_offsets=" << report.diagnostics.shellOffsets
                  << " generated_pairs=" << report.diagnostics.generatedPairs << " axis_rejected=" << report.diagnostics.axisRejectedPairs
                  << " searched=" << report.diagnostics.searchedPairs << " successful=" << report.diagnostics.successfulPaths
                  << " no_path=" << report.diagnostics.noPathPairs << " preloaded_voxels=" << report.preloadedVoxels
                  << " estimated_preload_bytes=" << report.estimatedPreloadBytes << " candidate_workers=" << report.candidateWorkers
                  << " candidate_seconds=" << report.candidateGenerationSeconds << " preload_seconds=" << report.preloadSeconds
                  << " search_seconds=" << report.searchSeconds << " elapsed_seconds=" << report.elapsedSeconds << '\n';
        if (options.printStats) {
            const auto statistics = vc::fiber_tracer::fiberletPathStatistics(report);
            std::cout << "fiberlet_stats anchors=" << statistics.anchors << " total_candidate_pairs=" << statistics.candidates
                      << " pre_dp_rejections=" << statistics.preDpRejected << " dp_searches=" << statistics.dpSearched
                      << " searched_unscored=" << statistics.searchedUnscored << " scored_fiberlets=" << statistics.scored
                      << " accepted_fiberlets=" << statistics.accepted << " unscored_candidates=" << statistics.unscored;
            if (presenceSlices.has_value())
                std::cout << " slices=enabled slice_pixels=" << presenceSlices->pixelCount() << '\n';
            else
                std::cout << " slices=disabled\n";
            const auto printScores = [](const char* name, const vc::fiber_tracer::FiberletScoreStatistics& scores) {
                std::cout << name << " count=" << scores.count;
                if (scores.count == 0) {
                    std::cout << " min=n/a mean=n/a max=n/a\n";
                } else {
                    std::cout << " min=" << *scores.minimum << " mean=" << *scores.mean << " max=" << *scores.maximum << '\n';
                }
            };
            printScores("fiberlet_total_loss_all", statistics.allScores);
            printScores("fiberlet_total_loss_accepted", statistics.acceptedScores);
            printScores(
                "fiberlet_loss_per_prediction_voxel_accepted",
                statistics.acceptedLossDensities);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiberlets: " << error.what() << '\n';
        return 1;
    }
}
