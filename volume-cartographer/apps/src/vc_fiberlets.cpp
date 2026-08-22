#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"
#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "FiberTraceCli.hpp"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

enum class Command {
    Anchors,
    AnchorBenchmark,
    Benchmark,
    QuantizationBenchmark,
    Paths,
    FiberletReplay,
};

bool isReplayCommand(Command command)
{
    return command == Command::FiberletReplay;
}

bool isQuantizationCommand(Command command)
{
    return command == Command::QuantizationBenchmark;
}

bool usesGraphReplayOptions(Command command)
{
    return isReplayCommand(command) || isQuantizationCommand(command);
}

bool needsPathExtraction(Command command)
{
    return command == Command::Paths || command == Command::Benchmark || isReplayCommand(command) || isQuantizationCommand(command);
}

struct CliOptions {
    Command command = Command::Anchors;
    std::string manifestLocation;
    std::filesystem::path anchorArtifact;
    std::filesystem::path fiberJson;
    std::string normalManifestLocation;
    std::filesystem::path outputDirectory;
    std::filesystem::path volumeZarr;
    std::filesystem::path remoteCacheDirectory;
    vc::fiber_tracer::FiberAnchorConfig anchors;
    vc::fiber_tracer::FiberletPathConfig paths;
    std::optional<vc::fiber_tracer::FiberAnchorCrop> baseCrop;
    std::optional<double> corridorRadiusBaseVoxels;
    std::optional<double> falloffSigmaBaseVoxels;
    std::optional<double> peakSigmaBaseVoxels;
    std::optional<double> peakAxialSigmaBaseVoxels;
    std::optional<double> peakStepBaseVoxels;
    std::optional<double> localWindowBaseVoxels;
    std::optional<double> baseVoxelSizeUm;
    std::optional<double> replayLengthBaseVoxels;
    std::optional<double> replayBeginArcBaseVoxels;
    std::optional<vc::fiber_tracer::FiberletStorageKey> replayInitialSeedKey;
    double routeStatsFailureMarginBaseVoxels = 128.0;
    double glyphLengthBaseVoxels = 16.0;
    size_t decodedCacheBytes = 512ULL * 1024ULL * 1024ULL;
    bool printStats = false;
    bool writePresenceSlices = true;
    bool writeReplayVisualizations = false;
    bool alongSpecified = false;
    int inferenceScaledownPower = 2;
    double failureThresholdBaseVoxels = 20.0;
    double alongBaseVoxels = 128.0;
    double radiusBaseVoxels = 64.0;
    double matchRefineSteps = 1.0;
    vc::fiber_tracer::FiberTraceConfig trace;
    vc::fiber_tracer::FiberletGraphReplayConfig graphReplay;
    int storageChunkSideBaseVoxels = 512;
    std::optional<std::string> quantizationScenario;
    std::filesystem::path anchorCacheRoot;
    std::filesystem::path fiberletCacheRoot;
    bool eagerGraphReplay = false;
    std::size_t storageCompressionChunks = 0;
    std::uint64_t storageCompressionSeed = 1;
    int storageCompressionChunkSideBaseVoxels = 512;
    vc::fiber_tracer::cli::SeenOptions seenTraceOptions;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

double processCpuSeconds()
{
    const std::clock_t ticks = std::clock();
    return ticks == static_cast<std::clock_t>(-1) ? 0.0 : static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

double effectiveCores(double cpuSeconds, double wallSeconds)
{
    return wallSeconds > 0.0 ? cpuSeconds / wallSeconds : 0.0;
}

void usage(const char* executable)
{
    std::cerr << "Usage:\n"
              << "  " << executable << " anchors <fiber.lasagna.json-or-url> <output-dir> [options]\n"
              << "  " << executable << " anchor-benchmark <fiber.lasagna.json-or-url> <fiber.json> [options]\n"
              << "  " << executable
              << " benchmark <fiber.lasagna.json-or-url> <fiber.json>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n"
              << "  " << executable
              << " quantization-benchmark <fiber.lasagna.json-or-url> <fiber.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n"
              << "  " << executable
              << " paths <fiber.lasagna.json-or-url> <anchors.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "  " << executable
              << " fiberlet-replay <fiber.lasagna.json-or-url> <fiber.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "Common options:\n"
              << "  --threads N                   decode/search workers [hardware]\n"
              << "  --cache-gib N                 decoded chunk cache budget [0.5]\n"
              << "  --remote-cache-dir PATH       required for direct remote manifests\n"
              << "  --stats                       print detailed path/replay diagnostics\n"
              << "  --base-voxel-size-um N        optional physical reporting metadata\n\n"
              << "Anchor options:\n"
              << "  --cell-size N                 prediction-grid cell side, 2..8 [4]\n"
              << "  --falloff N                   normal-plane sigma in base voxels [cell-side/2]\n"
              << "  --peak-sigma N                transverse peak sigma in base voxels [1.5 prediction voxels]\n"
              << "  --axial-sigma N               along-fiber peak sigma in base voxels [1.5 cell-sides]\n"
              << "  --peak-step N                 local-peak grid step in base voxels [0.5 prediction voxels]\n"
              << "  --gradient-weight N           signed presence-gradient centering weight [1.0]\n"
              << "  --window N                    refinement radius in base voxels [cell-side]\n"
              << "  --presence-floor N            inclusive observation floor [0.05]\n"
              << "  --minimum-support N           inclusive aligned support [0.05]\n"
              << "  --robust-max-trim N           maximum trimmed evidence mass [0.20]\n"
              << "  --robust-mad-multiplier N     angular residual MAD multiplier [3]\n"
              << "  --robust-min-angle-deg N      angular noise floor [5]\n"
              << "  --nms-angle-deg N             maximum duplicate-axis angle [10]\n"
              << "  --maximum-seeds N             deterministic PCA seed count [8]\n"
              << "  --maximum-iterations N        anchor quality/speed pass limit [1]\n"
              << "  --crop X,Y,Z,W,H,D            base-volume box; selects intersected cells\n"
              << "  --glyph-length-base-voxels N  diagnostic anchor length [16]\n\n"
              << "Path options:\n"
              << "  --normal-manifest PATH        required regular Lasagna normals\n"
              << "  --cell-radius N               candidate neighborhood radius [4]\n"
              << "  --radius-margin N             outer neighborhood margin [0.5]\n"
              << "  --endpoint-angle-degrees N    endpoint/chord and attachment bound [45]\n"
              << "  --prediction-angle N          hard sampled-fiber deviation [25]\n"
              << "  --corridor-radius N           base voxels [one anchor-cell width]\n"
              << "  --invalid-prediction-cost N   invalid cost per prediction-grid voxel [4]\n"
              << "  --smoothness-weight N         invalid-normal isotropic weight [2]\n"
              << "  --smoothness-normal-weight N  normal-tilt weight [0.1]\n"
              << "  --smoothness-tangent-weight N tangent-plane turn weight [10]\n"
              << "  --smoothness-free-angle N     local curvature free angle in degrees [0]\n"
              << "  --batch N                     unique native coordinates per sampler call [65536]\n"
              << "  --no-slices                   skip central presence-slice outputs\n";
    std::cerr << "\nReplay options:\n"
              << "  --fail N                      Lasagna-normal failure radius in base voxels; tangent-plane radius is 4N [20]\n"
              << "  --length N                    compared reference length in base voxels [full]\n"
              << "  --arc N                       absolute reference-polyline start arc in base voxels\n"
              << "  --seed-key Z,Y,X,V            exact first graph-anchor key for a focused replay\n"
              << "  --route-stats-failure-margin N exclude this base-voxel distance around failures [128]\n"
              << "  --beam-step-distance N        rolling checkpoint step in base voxels [48]\n"
              << "  --lookahead-distance N        persistent beam lookahead in base voxels [384]\n"
              << "  --search-width N              approximate intermediate width; zero is exact [0]\n"
              << "  --prune-distance N            approximate-mode pruning interval in base voxels [48]\n"
              << "  --vis                         write indexed local failure visualizations\n"
              << "  --volume PATH                 required CT OME-Zarr array/group path for --vis\n"
              << "  --along N                     replay visualization half-width [128]; benchmark length [full]\n"
              << "  --radius N                    extraction tube radius in base voxels [64]\n"
              << "  --match-refine N              forward match refinement in trace steps [1]\n"
              << "  --inference-scaledown-power N prediction scaledown relative to trace voxels [2]\n";
    std::cerr << "  --beam N                      positive graph replay beam width [16]\n"
              << "  --anchor-cache PATH           generated anchor cache [output/cache/anchors.zarr]\n"
              << "  --fiberlet-cache PATH         generated fiberlet cache [output/cache/fiberlets.zarr]\n"
              << "  --eager-graph                 diagnostic corridor-wide graph extraction\n"
              << "  --storage-chunk-side N        storage chunk side in base voxels [512]\n"
              << "  --storage-compression-chunks N  benchmark compact storage on N complete spatial regions\n"
              << "  --storage-compression-seed N    deterministic chunk sample seed [1]\n"
              << "  --storage-compression-chunk-side N  extracted region side in base voxels [512]\n"
              << "  --scenario NAME|all           quantization scenario(s); baseline runs once\n";
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

vc::fiber_tracer::FiberletStorageKey parseStorageKey(const std::string& text)
{
    std::array<std::int64_t, 3> coordinate{};
    std::uint8_t variant = 0;
    std::stringstream input(text);
    std::string token;
    for (size_t index = 0; index < 4; ++index) {
        if (!std::getline(input, token, ',') || token.empty())
            fail("--seed-key requires Z,Y,X,V");
        size_t parsed = 0;
        const long long value = std::stoll(token, &parsed);
        if (parsed != token.size())
            fail("--seed-key requires four integers");
        if (index < 3) {
            coordinate[index] = static_cast<std::int64_t>(value);
        } else {
            if (value < 0 || value > 1)
                fail("--seed-key variant must be zero or one");
            variant = static_cast<std::uint8_t>(value);
        }
    }
    if (std::getline(input, token, ','))
        fail("--seed-key requires exactly four integers");
    return {coordinate, variant};
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
    } else if (command == "anchor-benchmark") {
        options.command = Command::AnchorBenchmark;
        options.manifestLocation = argv[2];
        options.fiberJson = argv[3];
        firstOption = 4;
    } else if (command == "benchmark") {
        options.command = Command::Benchmark;
        options.manifestLocation = argv[2];
        options.fiberJson = argv[3];
        firstOption = 4;
    } else if (command == "quantization-benchmark") {
        if (argc < 5) {
            usage(argv[0]);
            std::exit(2);
        }
        options.command = Command::QuantizationBenchmark;
        options.manifestLocation = argv[2];
        options.fiberJson = argv[3];
        options.outputDirectory = argv[4];
        options.quantizationScenario =
            "compact_axis_cost_sqrt_u16_max256";
        firstOption = 5;
    } else if (command == "fiberlet-replay") {
        if (argc < 5) {
            usage(argv[0]);
            std::exit(2);
        }
        options.command = Command::FiberletReplay;
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
    options.graphReplay.expansionThreads = static_cast<size_t>(workers);
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
            options.graphReplay.expansionThreads = static_cast<size_t>(value);
        } else if (argument == "--cache-gib") {
            const double gib = parseDouble(valueAfter(index, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0) || gib > 1024.0)
                fail("--cache-gib must be in (0, 1024]");
            options.decodedCacheBytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = valueAfter(index, argc, argv, "remote-cache-dir");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(valueAfter(index, argc, argv, "base-voxel-size-um"), "base-voxel-size-um");
        } else if (argument == "--normal-manifest" && needsPathExtraction(options.command)) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
        } else if (argument == "--fail" && usesGraphReplayOptions(options.command)) {
            options.failureThresholdBaseVoxels = parseDouble(valueAfter(index, argc, argv, "fail"), "fail");
        } else if (argument == "--length" && usesGraphReplayOptions(options.command)) {
            options.replayLengthBaseVoxels = parseDouble(valueAfter(index, argc, argv, "length"), "length");
        } else if (argument == "--arc" && isQuantizationCommand(options.command)) {
            options.replayBeginArcBaseVoxels = parseDouble(valueAfter(index, argc, argv, "arc"), "arc");
        } else if (argument == "--seed-key" && isQuantizationCommand(options.command)) {
            options.replayInitialSeedKey = parseStorageKey(valueAfter(index, argc, argv, "seed-key"));
        } else if (argument == "--route-stats-failure-margin" && isQuantizationCommand(options.command)) {
            options.routeStatsFailureMarginBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "route-stats-failure-margin"), "route-stats-failure-margin");
        } else if (argument == "--vis" && isReplayCommand(options.command)) {
            options.writeReplayVisualizations = true;
        } else if (argument == "--volume" && isReplayCommand(options.command)) {
            options.volumeZarr = valueAfter(index, argc, argv, "volume");
        } else if (argument == "--along" && (isReplayCommand(options.command) || options.command == Command::Benchmark)) {
            options.alongBaseVoxels = parseDouble(valueAfter(index, argc, argv, "along"), "along");
            options.alongSpecified = true;
        } else if (argument == "--radius" && (usesGraphReplayOptions(options.command) || options.command == Command::Benchmark)) {
            options.radiusBaseVoxels = parseDouble(valueAfter(index, argc, argv, "radius"), "radius");
        } else if (argument == "--match-refine" && usesGraphReplayOptions(options.command)) {
            options.matchRefineSteps = parseDouble(valueAfter(index, argc, argv, "match-refine"), "match-refine");
        } else if (argument == "--inference-scaledown-power" && isReplayCommand(options.command)) {
            options.inferenceScaledownPower =
                parseInt(valueAfter(index, argc, argv, "inference-scaledown-power"), "inference-scaledown-power");
        } else if (argument == "--anchor-cache" && usesGraphReplayOptions(options.command)) {
            options.anchorCacheRoot = valueAfter(index, argc, argv, "anchor-cache");
        } else if (argument == "--fiberlet-cache" && usesGraphReplayOptions(options.command)) {
            options.fiberletCacheRoot = valueAfter(index, argc, argv, "fiberlet-cache");
        } else if (argument == "--eager-graph" && isReplayCommand(options.command)) {
            options.eagerGraphReplay = true;
        } else if (argument == "--storage-compression-chunks" &&
                   isReplayCommand(options.command)) {
            const int value = parseInt(
                valueAfter(index, argc, argv, "storage-compression-chunks"),
                "storage-compression-chunks");
            if (value <= 0)
                fail("--storage-compression-chunks must be positive");
            options.storageCompressionChunks = static_cast<std::size_t>(value);
        } else if (argument == "--storage-compression-seed" &&
                   isReplayCommand(options.command)) {
            const int value = parseInt(
                valueAfter(index, argc, argv, "storage-compression-seed"),
                "storage-compression-seed");
            if (value < 0)
                fail("--storage-compression-seed must be nonnegative");
            options.storageCompressionSeed = static_cast<std::uint64_t>(value);
        } else if (argument == "--storage-compression-chunk-side" &&
                   isReplayCommand(options.command)) {
            const int value = parseInt(
                valueAfter(index, argc, argv,
                    "storage-compression-chunk-side"),
                "storage-compression-chunk-side");
            if (value <= 0)
                fail("--storage-compression-chunk-side must be positive");
            options.storageCompressionChunkSideBaseVoxels = value;
        } else if (argument == "--cell-size" && options.command != Command::Paths) {
            options.anchors.cellSizePredictionVoxels = parseInt(valueAfter(index, argc, argv, "cell-size"), "cell-size");
        } else if (argument == "--falloff" && options.command != Command::Paths) {
            options.falloffSigmaBaseVoxels = parseDouble(valueAfter(index, argc, argv, "falloff"), "falloff");
        } else if (argument == "--peak-sigma" && options.command != Command::Paths) {
            options.peakSigmaBaseVoxels = parseDouble(valueAfter(index, argc, argv, "peak-sigma"), "peak-sigma");
        } else if (argument == "--axial-sigma" && options.command != Command::Paths) {
            options.peakAxialSigmaBaseVoxels = parseDouble(valueAfter(index, argc, argv, "axial-sigma"), "axial-sigma");
        } else if (argument == "--peak-step" && options.command != Command::Paths) {
            options.peakStepBaseVoxels = parseDouble(valueAfter(index, argc, argv, "peak-step"), "peak-step");
        } else if (argument == "--gradient-weight" && options.command != Command::Paths) {
            options.anchors.peakGradientWeight = parseDouble(valueAfter(index, argc, argv, "gradient-weight"), "gradient-weight");
        } else if (argument == "--window" && options.command != Command::Paths) {
            options.localWindowBaseVoxels = parseDouble(valueAfter(index, argc, argv, "window"), "window");
        } else if (argument == "--presence-floor" && options.command != Command::Paths) {
            options.anchors.observationPresenceFloor = parseDouble(valueAfter(index, argc, argv, "presence-floor"), "presence-floor");
        } else if (argument == "--minimum-support" && options.command != Command::Paths) {
            options.anchors.minimumAlignedSupport = parseDouble(valueAfter(index, argc, argv, "minimum-support"), "minimum-support");
        } else if (argument == "--robust-max-trim" && options.command != Command::Paths) {
            options.anchors.robustMaximumTrimMassFraction = parseDouble(valueAfter(index, argc, argv, "robust-max-trim"), "robust-max-trim");
        } else if (argument == "--robust-mad-multiplier" && options.command != Command::Paths) {
            options.anchors.robustMadMultiplier =
                parseDouble(valueAfter(index, argc, argv, "robust-mad-multiplier"), "robust-mad-multiplier");
        } else if (argument == "--robust-min-angle-deg" && options.command != Command::Paths) {
            options.anchors.robustMinimumAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "robust-min-angle-deg"), "robust-min-angle-deg");
        } else if (argument == "--nms-angle-deg" && options.command != Command::Paths) {
            options.anchors.nmsMaximumAngleDegrees = parseDouble(valueAfter(index, argc, argv, "nms-angle-deg"), "nms-angle-deg");
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
        } else if (argument == "--cell-radius" && needsPathExtraction(options.command)) {
            options.paths.cellRadius = parseInt(valueAfter(index, argc, argv, "cell-radius"), "cell-radius");
        } else if (argument == "--radius-margin" && needsPathExtraction(options.command)) {
            options.paths.neighborhoodMarginCells = parseDouble(valueAfter(index, argc, argv, "radius-margin"), "radius-margin");
        } else if (argument == "--endpoint-angle-degrees" && needsPathExtraction(options.command)) {
            options.paths.maximumEndpointAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "endpoint-angle-degrees"), "endpoint-angle-degrees");
        } else if (argument == "--prediction-angle" && needsPathExtraction(options.command)) {
            options.paths.maximumPredictionDeviationDegrees =
                parseDouble(valueAfter(index, argc, argv, "prediction-angle"), "prediction-angle");
        } else if (argument == "--corridor-radius" && needsPathExtraction(options.command)) {
            options.corridorRadiusBaseVoxels = parseDouble(valueAfter(index, argc, argv, "corridor-radius"), "corridor-radius");
            if (!(*options.corridorRadiusBaseVoxels > 0.0))
                fail("--corridor-radius must be positive");
        } else if (argument == "--stats" && (options.command == Command::Paths || isReplayCommand(options.command))) {
            options.printStats = true;
        } else if (argument == "--no-slices" && options.command == Command::Paths) {
            options.writePresenceSlices = false;
        } else if (argument == "--invalid-prediction-cost" && needsPathExtraction(options.command)) {
            options.paths.invalidPredictionCostPerVoxel =
                parseDouble(valueAfter(index, argc, argv, "invalid-prediction-cost"), "invalid-prediction-cost");
        } else if (argument == "--smoothness-weight" && needsPathExtraction(options.command)) {
            options.paths.smoothnessWeight = parseDouble(valueAfter(index, argc, argv, "smoothness-weight"), "smoothness-weight");
        } else if (argument == "--smoothness-normal-weight" && needsPathExtraction(options.command)) {
            options.paths.smoothnessNormalWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-normal-weight"), "smoothness-normal-weight");
        } else if (argument == "--smoothness-tangent-weight" && needsPathExtraction(options.command)) {
            options.paths.smoothnessTangentWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-tangent-weight"), "smoothness-tangent-weight");
        } else if (argument == "--smoothness-free-angle" && needsPathExtraction(options.command)) {
            options.paths.smoothnessFreeAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "smoothness-free-angle"), "smoothness-free-angle");
        } else if (argument == "--batch" && needsPathExtraction(options.command)) {
            options.paths.samplingBatchCoordinates = parseInt(valueAfter(index, argc, argv, "batch"), "batch");
        } else if (argument == "--beam" && usesGraphReplayOptions(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "beam"), "beam");
            if (value < 1)
                fail("--beam must be positive");
            options.graphReplay.beamWidth = static_cast<size_t>(value);
        } else if (argument == "--beam-step-distance" && usesGraphReplayOptions(options.command)) {
            options.graphReplay.beamStepDistanceBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "beam-step-distance"), "beam-step-distance");
        } else if (argument == "--lookahead-distance" && usesGraphReplayOptions(options.command)) {
            options.graphReplay.lookaheadDistanceBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "lookahead-distance"), "lookahead-distance");
        } else if (argument == "--search-width" && usesGraphReplayOptions(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "search-width"), "search-width");
            if (value < 0)
                fail("--search-width must be non-negative");
            options.graphReplay.searchWidth = static_cast<size_t>(value);
        } else if (argument == "--prune-distance" && usesGraphReplayOptions(options.command)) {
            options.graphReplay.pruneDistanceBaseVoxels = parseDouble(valueAfter(index, argc, argv, "prune-distance"), "prune-distance");
        } else if (argument == "--storage-chunk-side" && (isQuantizationCommand(options.command) || isReplayCommand(options.command))) {
            options.storageChunkSideBaseVoxels = parseInt(valueAfter(index, argc, argv, "storage-chunk-side"), "storage-chunk-side");
        } else if (argument == "--scenario" && isQuantizationCommand(options.command)) {
            options.quantizationScenario = valueAfter(index, argc, argv, "scenario");
        } else if (
            isReplayCommand(options.command) &&
            vc::fiber_tracer::cli::parseTraceOption(argument, index, argc, argv, options.trace, &options.seenTraceOptions)) {
            continue;
        } else {
            fail("unknown option for selected command: " + argument);
        }
    }
    if (options.baseVoxelSizeUm.has_value() && !(*options.baseVoxelSizeUm > 0.0))
        fail("--base-voxel-size-um must be positive");
    if (options.command != Command::Paths) {
        if (options.falloffSigmaBaseVoxels.has_value() && !(*options.falloffSigmaBaseVoxels > 0.0))
            fail("--falloff must be positive");
        if (options.peakSigmaBaseVoxels.has_value() && !(*options.peakSigmaBaseVoxels > 0.0))
            fail("--peak-sigma must be positive");
        if (options.peakAxialSigmaBaseVoxels.has_value() && !(*options.peakAxialSigmaBaseVoxels > 0.0))
            fail("--axial-sigma must be positive");
        if (options.peakStepBaseVoxels.has_value() && !(*options.peakStepBaseVoxels > 0.0))
            fail("--peak-step must be positive");
        if (options.localWindowBaseVoxels.has_value() && !(*options.localWindowBaseVoxels > 0.0))
            fail("--window must be positive");
    }
    if (needsPathExtraction(options.command)) {
        if (options.normalManifestLocation.empty())
            fail("paths and replay commands require --normal-manifest");
        vc::fiber_tracer::validateFiberletPathConfig(options.paths);
    }
    if (usesGraphReplayOptions(options.command) &&
        (!(options.graphReplay.beamStepDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.beamStepDistanceBaseVoxels) ||
         !(options.graphReplay.lookaheadDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.lookaheadDistanceBaseVoxels) ||
         (options.graphReplay.searchWidth != 0 && options.graphReplay.searchWidth < options.graphReplay.beamWidth) ||
         !(options.graphReplay.pruneDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.pruneDistanceBaseVoxels))) {
        fail("graph search distances and widths are outside their valid range");
    }
    if (isReplayCommand(options.command)) {
        if (!(options.failureThresholdBaseVoxels >= 0.0) || !(options.alongBaseVoxels > 0.0) || !(options.radiusBaseVoxels > 0.0) ||
            !(options.matchRefineSteps >= 0.0) || options.storageChunkSideBaseVoxels <= 0 ||
            (options.replayLengthBaseVoxels.has_value() && (!(*options.replayLengthBaseVoxels > 0.0) || !std::isfinite(*options.replayLengthBaseVoxels))) ||
            options.inferenceScaledownPower < 0 || options.inferenceScaledownPower > 30) {
            fail("fiber-replay options are outside their valid range");
        }
        vc::fiber_tracer::cli::validateTraceOptions(options.trace);
        if ((options.seenTraceOptions.beamWidth && options.trace.beamWidth != 1) ||
            (options.seenTraceOptions.beamLookahead && options.trace.beamLookaheadSteps != 1)) {
            fail("fiber-replay only supports --beam-width 1 and --beam-lookahead-steps 1");
        }
        if (options.writeReplayVisualizations && options.volumeZarr.empty())
            fail("fiber-replay --vis requires --volume PATH for CT strip sampling");
        if (!options.writeReplayVisualizations && !options.volumeZarr.empty()) {
            fail("fiber-replay volume strip options are only valid together with --vis");
        }
        if (options.eagerGraphReplay && options.storageCompressionChunks > 0) {
            fail("--storage-compression-chunks requires the on-demand replay cache");
        }
    }
    if (isQuantizationCommand(options.command)) {
        if (!(options.failureThresholdBaseVoxels >= 0.0) || !(options.radiusBaseVoxels > 0.0) || !(options.matchRefineSteps >= 0.0) ||
            !(options.routeStatsFailureMarginBaseVoxels >= 0.0) || options.storageChunkSideBaseVoxels <= 0 ||
            (options.replayLengthBaseVoxels.has_value() && !(*options.replayLengthBaseVoxels > 0.0))) {
            fail("quantization-benchmark options are outside their valid range");
        }
        if (options.replayBeginArcBaseVoxels.has_value() && !(*options.replayBeginArcBaseVoxels >= 0.0)) {
            fail("quantization-benchmark --arc must be non-negative");
        }
        if (options.replayInitialSeedKey.has_value() && !options.replayBeginArcBaseVoxels.has_value()) {
            fail("quantization-benchmark --seed-key requires --arc");
        }
        if (options.quantizationScenario.has_value() && *options.quantizationScenario != "all") {
            const auto scenarios = vc::fiber_tracer::standardFiberletQuantizationScenarios();
            const bool known = std::any_of(scenarios.begin(), scenarios.end(), [&](const auto& scenario) {
                return scenario.name == *options.quantizationScenario;
            });
            if (!known) {
                fail("unknown fiberlet quantization scenario: " + *options.quantizationScenario);
            }
        }
    }
    if (options.command == Command::Benchmark && ((options.alongSpecified && !(options.alongBaseVoxels > 0.0)) || !(options.radiusBaseVoxels > 0.0)))
        fail("benchmark --along and --radius must be positive");
    const bool needsNormals = needsPathExtraction(options.command);
    const bool remote = vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) ||
                        (needsNormals && vc::lasagna::isRemoteLasagnaLocation(options.normalManifestLocation));
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

std::string datasetLocator(const vc::lasagna::LasagnaDataset& dataset);

std::array<std::uint8_t, 32> storageFingerprint(const std::string& value)
{
    std::array<std::uint8_t, 32> result{};
    for (std::size_t lane = 0; lane < 4; ++lane) {
        std::uint64_t hash = 14695981039346656037ULL ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(lane + 1));
        for (const unsigned char byte : value) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        for (std::size_t byte = 0; byte < 8; ++byte)
            result[lane * 8 + byte] = static_cast<std::uint8_t>(hash >> (byte * 8));
    }
    return result;
}

vc::fiber_tracer::FiberletDatasetMetadata replayDatasetMetadata(
    vc::fiber_tracer::FiberletDatasetKind kind,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid,
    const CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset,
    const std::vector<cv::Vec3d>& corridorReferenceBase,
    double corridorRadiusBaseVoxels,
    const vc::fiber_tracer::FiberletGeometryCacheProfile& cacheProfile = {})
{
    const double cellSideBase = static_cast<double>(options.anchors.cellSizePredictionVoxels) * grid.predictionToBaseScale;
    const auto roundedCellSideBase = std::llround(cellSideBase);
    if (!(roundedCellSideBase > 0) || std::abs(cellSideBase - static_cast<double>(roundedCellSideBase)) > 1.0e-9 ||
        options.storageChunkSideBaseVoxels % roundedCellSideBase != 0) {
        fail("storage chunk side must be an exact multiple of the anchor cell side in base voxels");
    }
    const std::int64_t unitsPerChunk = options.storageChunkSideBaseVoxels / roundedCellSideBase;
    std::array<std::int64_t, 3> cellShape{};
    std::array<std::int32_t, 3> chunkShape{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        cellShape[axis] = static_cast<std::int64_t>(
            (grid.shapeZYX[axis] + static_cast<std::size_t>(options.anchors.cellSizePredictionVoxels) - 1) /
            static_cast<std::size_t>(options.anchors.cellSizePredictionVoxels));
        const auto chunks = (cellShape[axis] + unitsPerChunk - 1) / unitsPerChunk;
        if (chunks > std::numeric_limits<std::int32_t>::max())
            fail("fiberlet storage chunk grid exceeds int32");
        chunkShape[axis] = static_cast<std::int32_t>(chunks);
    }
    std::int64_t maximumReach = 0;
    for (const auto& offset : vc::fiber_tracer::fiberletCellNeighborhoodOffsets(options.paths.cellRadius, options.paths.neighborhoodMarginCells)) {
        for (const auto coordinate : offset)
            maximumReach = std::max(maximumReach, static_cast<std::int64_t>(std::abs(coordinate)));
    }
    std::ostringstream identity;
    identity << std::setprecision(17) << "fiber=" << datasetLocator(fiberDataset) << ";fiber_hash=" << fileHash(fiberDataset.manifest().manifestPath)
             << ";normal=" << datasetLocator(normalDataset) << ";normal_hash=" << fileHash(normalDataset.manifest().manifestPath)
             << ";grid=" << grid.shapeZYX[0] << ',' << grid.shapeZYX[1] << ',' << grid.shapeZYX[2] << ";scale=" << grid.predictionToBaseScale
             << ";cell=" << options.anchors.cellSizePredictionVoxels << ";anchor_sigma=" << options.anchors.gaussianSigmaPredictionVoxels
             << ";peak_sigma=" << options.anchors.peakSigmaPredictionVoxels << ";axial_sigma=" << options.anchors.peakAxialSigmaPredictionVoxels
             << ";peak_step=" << options.anchors.peakGridStepPredictionVoxels << ";gradient=" << options.anchors.peakGradientWeight
             << ";window=" << options.anchors.localWindowRadiusPredictionVoxels << ";path_radius=" << options.paths.cellRadius
             << ";path_margin=" << options.paths.neighborhoodMarginCells << ";long_step=" << options.paths.longitudinalStepPredictionVoxels
             << ";transverse_step=" << options.paths.transverseStepPredictionVoxels << ";endpoint_angle=" << options.paths.maximumEndpointAngleDegrees
             << ";prediction_angle=" << options.paths.maximumPredictionDeviationDegrees
             << ";corridor=" << options.paths.corridorRadiusPredictionVoxels << ";invalid=" << options.paths.invalidPredictionCostPerVoxel
             << ";smooth=" << options.paths.smoothnessWeight << ',' << options.paths.smoothnessNormalWeight << ','
             << options.paths.smoothnessTangentWeight << ',' << options.paths.smoothnessFreeAngleDegrees << ";storage_schema=float_cache_v2"
             << ";corridor_selector=chunk_local_segment_aabb_v2"
             << ";corridor_radius_base=" << corridorRadiusBaseVoxels << ";corridor_reference=" << corridorReferenceBase.size();
    if (cacheProfile.enabled()) {
        identity << ";evaluation_quantization_v=2"
                 << ";position_quantum_base=" << cacheProfile.geometry.positionQuantumBaseVoxels
                 << ";compact_directions=" << (cacheProfile.geometry.compactDirections ? 1 : 0) << ";cost_bits=" << cacheProfile.compatibilityCostTagBits
                 << ";compact_owner_chunk_base=" << cacheProfile.storageChunkSideBaseVoxels;
    }
    for (const auto& point : corridorReferenceBase)
        identity << ';' << point[0] << ',' << point[1] << ',' << point[2];
    const auto identityText = identity.str();
    vc::fiber_tracer::FiberletDatasetMetadata metadata;
    metadata.kind = kind;
    metadata.profile = vc::fiber_tracer::FiberletStorageProfile::Float32Cache;
    metadata.chunkGridShapeZYX = chunkShape;
    metadata.coordinateOriginZYX = {0, 0, 0};
    metadata.coordinateUnitsPerChunkZYX = {unitsPerChunk, unitsPerChunk, unitsPerChunk};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {maximumReach, maximumReach, maximumReach};
    metadata.datasetFingerprint = storageFingerprint(identityText);
    metadata.spatialChunkSideBaseVoxels = static_cast<std::uint32_t>(options.storageChunkSideBaseVoxels);
    metadata.predictionToBaseScale = grid.predictionToBaseScale;
    metadata.algorithmFingerprint = stringHash(identityText);
    metadata.fiberManifest = datasetLocator(fiberDataset);
    metadata.fiberManifestHash = fileHash(fiberDataset.manifest().manifestPath);
    metadata.normalManifest = datasetLocator(normalDataset);
    metadata.normalManifestHash = fileHash(normalDataset.manifest().manifestPath);
    return metadata;
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

double resolveAnchorConfig(CliOptions& options, const vc::fiber_tracer::FiberPredictionGridInfo& grid)
{
    const double cellSideBase = options.anchors.cellSizePredictionVoxels * grid.predictionToBaseScale;
    options.anchors.gaussianSigmaPredictionVoxels = options.falloffSigmaBaseVoxels.value_or(cellSideBase * 0.5) / grid.predictionToBaseScale;
    options.anchors.peakSigmaPredictionVoxels = options.peakSigmaBaseVoxels.value_or(1.5 * grid.predictionToBaseScale) / grid.predictionToBaseScale;
    options.anchors.peakAxialSigmaPredictionVoxels = options.peakAxialSigmaBaseVoxels.value_or(1.5 * cellSideBase) / grid.predictionToBaseScale;
    options.anchors.peakGridStepPredictionVoxels = options.peakStepBaseVoxels.value_or(0.5 * grid.predictionToBaseScale) / grid.predictionToBaseScale;
    options.anchors.localWindowRadiusPredictionVoxels = options.localWindowBaseVoxels.value_or(cellSideBase) / grid.predictionToBaseScale;
    options.anchors.axialSupportHalfWidthPredictionVoxels = 1.5 * options.anchors.cellSizePredictionVoxels;
    vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
    return cellSideBase;
}

vc::fiber_tracer::ForwardPolylineArcInterval resolveQuantizationReplayInterval(
    const vc::fiber_tracer::PolylineArcGeometry& reference, size_t firstControlPointLineIndex, const CliOptions& options)
{
    const auto available = vc::fiber_tracer::selectForwardPolylineArcInterval(reference, firstControlPointLineIndex);
    if (!options.replayBeginArcBaseVoxels.has_value()) {
        return vc::fiber_tracer::selectForwardPolylineArcInterval(reference, firstControlPointLineIndex, options.replayLengthBaseVoxels);
    }
    const double begin = *options.replayBeginArcBaseVoxels;
    if (begin < available.beginArc - 1.0e-9 || begin >= available.endArc - 1.0e-9) {
        fail("quantization-benchmark --arc lies outside the first-CP reference interval");
    }
    const double end =
        options.replayLengthBaseVoxels.has_value() ? std::min(available.endArc, begin + *options.replayLengthBaseVoxels) : available.endArc;
    if (!(end > begin + 1.0e-9))
        fail("quantization-benchmark focused replay interval is empty");
    return {begin, end};
}

double quantizationExtractionArcPaddingBaseVoxels(const CliOptions& options, const vc::fiber_tracer::FiberPredictionGridInfo& grid)
{
    const double cellSideBase = static_cast<double>(options.anchors.cellSizePredictionVoxels) * grid.predictionToBaseScale;
    double maximumCellOffset = 0.0;
    for (const auto& offset : vc::fiber_tracer::fiberletCellNeighborhoodOffsets(options.paths.cellRadius, options.paths.neighborhoodMarginCells)) {
        maximumCellOffset =
            std::max(maximumCellOffset, std::sqrt(static_cast<double>(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])));
    }
    const double maximumFiberletReachBase = (maximumCellOffset + std::sqrt(3.0)) * cellSideBase;
    const double seedWindowBase = std::max(options.graphReplay.minimumResetAdvanceBaseVoxels, cellSideBase);
    return seedWindowBase + options.graphReplay.beamStepDistanceBaseVoxels + options.graphReplay.lookaheadDistanceBaseVoxels + maximumFiberletReachBase;
}

std::filesystem::path writeQuantizationReplay(
    const CliOptions& options,
    const vc::fiber_tracer::FiberletGraphReplayResult& replay,
    const vc::fiber_tracer::FiberletGraphReplayConfig& config,
    const vc::fiber_tracer::FiberletQuantizationScenario& scenario,
    std::string_view source,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset)
{
    std::ostringstream intervalIdentity;
    intervalIdentity << std::setprecision(17) << replay.referenceBeginArcBase << ',' << replay.referenceEndArcBase;
    std::string intervalHash = stringHash(intervalIdentity.str());
    std::replace(intervalHash.begin(), intervalHash.end(), ':', '-');
    const auto directory = options.outputDirectory / "quantization-replays" / intervalHash;
    auto json = vc::fiber_tracer::fiberletGraphReplayJson(replay, config);
    json["quantization"] = {
        {"source", source},
        {"scenario", scenario.name},
        {"position_quantum_base_voxels", scenario.positionQuantumBaseVoxels},
        {"compact_directions", scenario.compactAxes},
        {"cost_bits", scenario.costBits},
        {"cost_domain", vc::fiber_tracer::fiberletCostQuantizationDomainName(scenario.costDomain)},
        {"cost_density_maximum", scenario.costDensityMaximum},
    };
    json["inputs"] = {
        {"fiber_prediction_manifest", datasetLocator(fiberDataset)},
        {"fiber_prediction_manifest_hash", fileHash(fiberDataset.manifest().manifestPath)},
        {"normal_manifest", datasetLocator(normalDataset)},
        {"normal_manifest_hash", fileHash(normalDataset.manifest().manifestPath)},
        {"reference_fiber", std::filesystem::absolute(options.fiberJson).lexically_normal().string()},
        {"reference_fiber_hash", fileHash(options.fiberJson)},
    };
    const auto path = directory / (std::string(source) + "-" + scenario.name + ".json");
    vc::core::util::atomicWriteString(path, json.dump(2) + "\n");
    return path;
}

void printQuantizationFailureWindows(std::string_view source, const vc::fiber_tracer::FiberletGraphReplayResult& replay)
{
    const auto windows = vc::fiber_tracer::fiberletGraphReplayFailureWindows(replay);
    for (const auto& window : windows) {
        const auto& failure = replay.failures.at(window.failureIndex);
        std::cout << std::setprecision(17) << "fiberlet_quantization_failure_window"
                  << " source=" << source << " index=" << window.failureIndex << " segment=" << window.segmentIndex
                  << " reason=" << window.reason << " failure_arc_base=" << window.failureReferenceArcBase
                  << " arc=" << window.replayBeginArcBase << " length=" << window.replayEndArcBase - window.replayBeginArcBase;
        if (failure.evaluatorPointBase.has_value()) {
            std::cout << " evaluator_base_xyz=" << (*failure.evaluatorPointBase)[0] << ',' << (*failure.evaluatorPointBase)[1] << ','
                      << (*failure.evaluatorPointBase)[2];
        }
        if (failure.candidateIndex.has_value())
            std::cout << " candidate=" << *failure.candidateIndex;
        if (failure.arcIndex.has_value())
            std::cout << " graph_arc=" << *failure.arcIndex;
        if (failure.candidatePathPointIndex.has_value()) {
            std::cout << " path_point=" << *failure.candidatePathPointIndex;
        }
        if (window.seedKey.has_value()) {
            std::cout << " seed_key=" << window.seedKey->coordinateZYX[0] << ',' << window.seedKey->coordinateZYX[1] << ','
                      << window.seedKey->coordinateZYX[2] << ',' << static_cast<unsigned>(window.seedKey->variant);
        }
        std::cout << '\n';
    }
}

nlohmann::json replayCostJson(const vc::fiber_tracer::FiberletGraphReplayCost& cost)
{
    return {
        {"invalid_prediction", cost.invalidPrediction},
        {"alignment", cost.alignment},
        {"isotropic_smoothness", cost.isotropicSmoothness},
        {"tangent_smoothness", cost.tangentSmoothness},
        {"normal_smoothness", cost.normalSmoothness},
        {"total", cost.total()},
    };
}

nlohmann::json replayStorageKeyJson(const vc::fiber_tracer::FiberletStorageKey& key)
{
    return nlohmann::json::array({key.coordinateZYX[0], key.coordinateZYX[1], key.coordinateZYX[2], key.variant});
}

nlohmann::json replayArcIdJson(const vc::fiber_tracer::DirectedFiberletStorageId& arc)
{
    return nlohmann::json::array({replayStorageKeyJson(arc.fiberlet.first), replayStorageKeyJson(arc.fiberlet.second), arc.reverse});
}

std::vector<const vc::fiber_tracer::FiberletGraphReplayDecision*> replayDecisions(const vc::fiber_tracer::FiberletGraphReplayResult& replay)
{
    std::vector<const vc::fiber_tracer::FiberletGraphReplayDecision*> result;
    for (const auto& segment : replay.segments) {
        for (const auto& decision : segment.decisions)
            result.push_back(&decision);
    }
    return result;
}

const vc::fiber_tracer::FiberletGraphReplayDecisionRoute* selectedRoute(const vc::fiber_tracer::FiberletGraphReplayDecision& decision)
{
    if (!decision.selectedRouteIndex.has_value() || *decision.selectedRouteIndex >= decision.routes.size()) {
        return nullptr;
    }
    return &decision.routes[*decision.selectedRouteIndex];
}

nlohmann::json decisionRouteJson(const vc::fiber_tracer::FiberletGraphReplayDecisionRoute* route, std::optional<size_t> rank)
{
    if (route == nullptr)
        return nullptr;
    nlohmann::json arcs = nlohmann::json::array();
    for (const auto& arc : route->logicalArcs)
        arcs.push_back(replayArcIdJson(arc));
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : route->routePointsBaseXYZ)
        points.push_back(nlohmann::json::array({point[0], point[1], point[2]}));
    return {
        {"rank", rank.has_value() ? nlohmann::json(*rank) : nlohmann::json(nullptr)},
        {"logical_arcs", std::move(arcs)},
        {"route_points_base_xyz", std::move(points)},
        {"edge_cost", replayCostJson(route->edgeCost)},
        {"transition_cost", replayCostJson(route->transitionCost)},
        {"committed_edge_cost", replayCostJson(route->committedEdgeCost)},
        {"committed_transition_cost", replayCostJson(route->committedTransitionCost)},
        {"committed_path_length_prediction_voxels", route->committedPathLengthPredictionVoxels},
        {"total_loss", route->totalLoss},
        {"path_length_prediction_voxels", route->pathLengthPredictionVoxels},
        {"complete_path_length_prediction_voxels", route->completePathLengthPredictionVoxels},
        {"loss_per_prediction_voxel", route->lossPerPredictionVoxel},
    };
}

nlohmann::json decisionRouteDeltaJson(const vc::fiber_tracer::FiberletGraphReplayDecisionRoute* baseline, const vc::fiber_tracer::FiberletGraphReplayDecisionRoute* scenario)
{
    if (baseline == nullptr || scenario == nullptr)
        return nullptr;
    const auto costDelta = [](const auto& first, const auto& second) {
        return nlohmann::json{
            {"invalid_prediction", second.invalidPrediction - first.invalidPrediction},
            {"alignment", second.alignment - first.alignment},
            {"isotropic_smoothness", second.isotropicSmoothness - first.isotropicSmoothness},
            {"tangent_smoothness", second.tangentSmoothness - first.tangentSmoothness},
            {"normal_smoothness", second.normalSmoothness - first.normalSmoothness},
            {"total", second.total() - first.total()},
        };
    };
    return {
        {"edge_cost", costDelta(baseline->edgeCost, scenario->edgeCost)},
        {"transition_cost", costDelta(baseline->transitionCost, scenario->transitionCost)},
        {"total_loss", scenario->totalLoss - baseline->totalLoss},
        {"path_length_prediction_voxels", scenario->pathLengthPredictionVoxels - baseline->pathLengthPredictionVoxels},
        {"complete_path_length_prediction_voxels", scenario->completePathLengthPredictionVoxels - baseline->completePathLengthPredictionVoxels},
        {"loss_per_prediction_voxel", scenario->lossPerPredictionVoxel - baseline->lossPerPredictionVoxel},
    };
}

std::optional<size_t> decisionRouteRank(const vc::fiber_tracer::FiberletGraphReplayDecision& decision, const vc::fiber_tracer::FiberletGraphReplayDecisionRoute* target)
{
    if (target == nullptr)
        return std::nullopt;
    for (size_t index = 0; index < decision.routes.size(); ++index) {
        if (decision.routes[index].logicalArcs == target->logicalArcs)
            return index;
    }
    return std::nullopt;
}

double symmetricDecisionRouteDistance(const vc::fiber_tracer::FiberletGraphReplayDecisionRoute& first, const vc::fiber_tracer::FiberletGraphReplayDecisionRoute& second)
{
    if (first.routePointsBaseXYZ.size() < 2 || second.routePointsBaseXYZ.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto firstGeometry = vc::fiber_tracer::makePolylineArcGeometry(first.routePointsBaseXYZ);
    const auto secondGeometry = vc::fiber_tracer::makePolylineArcGeometry(second.routePointsBaseXYZ);
    const auto directed = [](const auto& source, const auto& target) {
        double maximum = 0.0;
        const size_t samples = std::max<size_t>(1, static_cast<size_t>(std::ceil(source.length())));
        for (size_t index = 0; index <= samples; ++index) {
            const double arc = source.length() * static_cast<double>(index) / static_cast<double>(samples);
            const auto point = vc::fiber_tracer::samplePolylineArc(source, arc).point;
            maximum = std::max(maximum, vc::fiber_tracer::projectPointToPolylineArc(target, point, 0.0, target.length()).distance);
        }
        return maximum;
    };
    return std::max(directed(firstGeometry, secondGeometry), directed(secondGeometry, firstGeometry));
}

nlohmann::json scalarStatisticsJson(std::vector<double> values)
{
    if (values.empty())
        return nullptr;
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
    return {
        {"count", values.size()},
        {"sum", sum},
        {"minimum", values.front()},
        {"mean", sum / static_cast<double>(values.size())},
        {"median", median},
        {"maximum", values.back()},
    };
}

nlohmann::json replayRouteCostStatisticsJson(
    const vc::fiber_tracer::FiberletGraphReplayResult& replay,
    std::optional<double> failureMarginBaseVoxels = std::nullopt,
    const vc::fiber_tracer::FiberletGraphReplayResult* failureSource = nullptr)
{
    std::vector<double> invalidPredictionDensity;
    std::vector<double> alignmentDensity;
    std::vector<double> isotropicSmoothnessDensity;
    std::vector<double> normalSmoothnessDensity;
    std::vector<double> tangentSmoothnessDensity;
    std::vector<double> edgeDensity;
    std::vector<double> transitionDensity;
    std::vector<double> combinedDensity;
    std::vector<double> failureArcs;
    if (failureMarginBaseVoxels.has_value()) {
        if (failureSource == nullptr)
            failureSource = &replay;
        failureArcs.reserve(failureSource->failures.size());
        for (const auto& failure : failureSource->failures)
            failureArcs.push_back(failure.referenceArcBase);
        std::sort(failureArcs.begin(), failureArcs.end());
    }
    size_t excludedFiberlets = 0;
    double totalPathLength = 0.0;
    double totalLoss = 0.0;
    for (const auto& segment : replay.segments) {
        for (const auto& step : segment.committedSteps) {
            if (!(step.pathLengthPredictionVoxels > 0.0))
                continue;
            if (failureMarginBaseVoxels.has_value()) {
                const double begin = std::min(step.referenceBeginArcBase, step.referenceEndArcBase) - *failureMarginBaseVoxels;
                const double end = std::max(step.referenceBeginArcBase, step.referenceEndArcBase) + *failureMarginBaseVoxels;
                const auto failure = std::lower_bound(failureArcs.begin(), failureArcs.end(), begin);
                if (failure != failureArcs.end() && *failure <= end) {
                    ++excludedFiberlets;
                    continue;
                }
            }
            const auto& edge = step.edgeCost;
            const auto& transition = step.transitionCost;
            const double inverseLength = 1.0 / step.pathLengthPredictionVoxels;
            invalidPredictionDensity.push_back((edge.invalidPrediction + transition.invalidPrediction) * inverseLength);
            alignmentDensity.push_back((edge.alignment + transition.alignment) * inverseLength);
            isotropicSmoothnessDensity.push_back((edge.isotropicSmoothness + transition.isotropicSmoothness) * inverseLength);
            normalSmoothnessDensity.push_back((edge.normalSmoothness + transition.normalSmoothness) * inverseLength);
            tangentSmoothnessDensity.push_back((edge.tangentSmoothness + transition.tangentSmoothness) * inverseLength);
            edgeDensity.push_back(edge.total() * inverseLength);
            transitionDensity.push_back(transition.total() * inverseLength);
            const double combined = edge.total() + transition.total();
            combinedDensity.push_back(combined * inverseLength);
            totalPathLength += step.pathLengthPredictionVoxels;
            totalLoss += combined;
        }
    }
    return {
        {"committed_fiberlets", combinedDensity.size()},
        {"excluded_fiberlets", excludedFiberlets},
        {"replay_failure_count", replay.failures.size()},
        {"exclusion_failure_count", failureArcs.size()},
        {"failure_exclusion_margin_base_voxels", failureMarginBaseVoxels.has_value() ? nlohmann::json(*failureMarginBaseVoxels) : nlohmann::json(nullptr)},
        {"total_path_length_prediction_voxels", totalPathLength},
        {"total_loss", totalLoss},
        {"whole_route_loss_per_prediction_voxel", totalPathLength > 0.0 ? nlohmann::json(totalLoss / totalPathLength) : nlohmann::json(nullptr)},
        {"combined_loss_per_prediction_voxel", scalarStatisticsJson(std::move(combinedDensity))},
        {"edge_loss_per_prediction_voxel", scalarStatisticsJson(std::move(edgeDensity))},
        {"transition_loss_per_prediction_voxel", scalarStatisticsJson(std::move(transitionDensity))},
        {"invalid_prediction_loss_per_prediction_voxel", scalarStatisticsJson(std::move(invalidPredictionDensity))},
        {"alignment_loss_per_prediction_voxel", scalarStatisticsJson(std::move(alignmentDensity))},
        {"isotropic_smoothness_loss_per_prediction_voxel", scalarStatisticsJson(std::move(isotropicSmoothnessDensity))},
        {"normal_smoothness_loss_per_prediction_voxel", scalarStatisticsJson(std::move(normalSmoothnessDensity))},
        {"tangent_smoothness_loss_per_prediction_voxel", scalarStatisticsJson(std::move(tangentSmoothnessDensity))},
    };
}

std::filesystem::path writeQuantizationRouteCostStatistics(
    const std::filesystem::path& directory,
    const vc::fiber_tracer::FiberletGraphReplayResult& baseline,
    const vc::fiber_tracer::FiberletGraphReplayResult& scenario,
    std::string_view scenarioName,
    double failureMarginBaseVoxels)
{
    const nlohmann::json root = {
        {"format", "vc_fiberlet_quantization_route_cost_statistics"},
        {"version", 1},
        {"scenario", scenarioName},
        {"baseline_all", replayRouteCostStatisticsJson(baseline)},
        {"baseline_away_from_failures", replayRouteCostStatisticsJson(baseline, failureMarginBaseVoxels, &baseline)},
        {"scenario_all", replayRouteCostStatisticsJson(scenario)},
        {"scenario_away_from_failures", replayRouteCostStatisticsJson(scenario, failureMarginBaseVoxels, &baseline)},
    };
    const auto path = directory / ("route-cost-statistics-" + std::string(scenarioName) + ".json");
    vc::core::util::atomicWriteString(path, root.dump(2) + "\n");
    std::cout << "fiberlet_quantization_route_cost_statistics"
              << " artifact=" << path.string() << " baseline_all=" << root["baseline_all"].dump()
              << " baseline_away_from_failures=" << root["baseline_away_from_failures"].dump() << '\n';
    return path;
}

std::filesystem::path writeQuantizationDecisionComparison(
    const std::filesystem::path& directory,
    const vc::fiber_tracer::FiberletGraphReplayResult& baseline,
    const vc::fiber_tracer::FiberletGraphReplayResult& scenario,
    std::string_view scenarioName)
{
    const auto baselineDecisions = replayDecisions(baseline);
    const auto scenarioDecisions = replayDecisions(scenario);
    const size_t commonCount = std::min(baselineDecisions.size(), scenarioDecisions.size());
    std::optional<size_t> firstDifference;
    std::optional<size_t> maximumDistanceIndex;
    double maximumDistance = -1.0;
    for (size_t index = 0; index < commonCount; ++index) {
        const auto& baselineDecision = *baselineDecisions[index];
        const auto& scenarioDecision = *scenarioDecisions[index];
        const auto* baselineSelected = selectedRoute(baselineDecision);
        const auto* scenarioSelected = selectedRoute(scenarioDecision);
        if (!firstDifference.has_value() && (baselineDecision.sourceKey != scenarioDecision.sourceKey || baselineSelected == nullptr ||
                                             scenarioSelected == nullptr || baselineSelected->logicalArcs != scenarioSelected->logicalArcs)) {
            firstDifference = index;
        }
        if (baselineSelected != nullptr && scenarioSelected != nullptr) {
            const double distance = symmetricDecisionRouteDistance(*baselineSelected, *scenarioSelected);
            if (std::isfinite(distance) && distance > maximumDistance) {
                maximumDistance = distance;
                maximumDistanceIndex = index;
            }
        }
    }
    if (!firstDifference.has_value() && baselineDecisions.size() != scenarioDecisions.size()) {
        firstDifference = commonCount;
    }

    nlohmann::json root = {
        {"format", "vc_fiberlet_quantization_decision_comparison"},
        {"version", 1},
        {"scenario", scenarioName},
        {"baseline_decisions", baselineDecisions.size()},
        {"scenario_decisions", scenarioDecisions.size()},
        {"first_selected_route_difference", nullptr},
        {"maximum_selected_route_distance", nullptr},
        {"baseline_route_cost_statistics", replayRouteCostStatisticsJson(baseline)},
        {"scenario_route_cost_statistics", replayRouteCostStatisticsJson(scenario)},
    };
    if (firstDifference.has_value() && *firstDifference < commonCount) {
        const auto& baselineDecision = *baselineDecisions[*firstDifference];
        const auto& scenarioDecision = *scenarioDecisions[*firstDifference];
        const auto* baselineSelected = selectedRoute(baselineDecision);
        const auto* scenarioSelected = selectedRoute(scenarioDecision);
        const auto baselineInScenario = decisionRouteRank(scenarioDecision, baselineSelected);
        const auto scenarioInBaseline = decisionRouteRank(baselineDecision, scenarioSelected);
        root["first_selected_route_difference"] = {
            {"decision_index", *firstDifference},
            {"baseline_reference_arc_base", baselineDecision.referenceArcBase},
            {"scenario_reference_arc_base", scenarioDecision.referenceArcBase},
            {"baseline_source_key", replayStorageKeyJson(baselineDecision.sourceKey)},
            {"scenario_source_key", replayStorageKeyJson(scenarioDecision.sourceKey)},
            {"baseline_selected_in_baseline", decisionRouteJson(baselineSelected, baselineDecision.selectedRouteIndex)},
            {"baseline_selected_in_scenario",
             decisionRouteJson(baselineInScenario.has_value() ? &scenarioDecision.routes[*baselineInScenario] : nullptr, baselineInScenario)},
            {"scenario_selected_in_baseline",
             decisionRouteJson(scenarioInBaseline.has_value() ? &baselineDecision.routes[*scenarioInBaseline] : nullptr, scenarioInBaseline)},
            {"scenario_selected_in_scenario", decisionRouteJson(scenarioSelected, scenarioDecision.selectedRouteIndex)},
            {"baseline_choice_scenario_minus_baseline",
             decisionRouteDeltaJson(baselineSelected, baselineInScenario.has_value() ? &scenarioDecision.routes[*baselineInScenario] : nullptr)},
            {"scenario_choice_scenario_minus_baseline",
             decisionRouteDeltaJson(scenarioInBaseline.has_value() ? &baselineDecision.routes[*scenarioInBaseline] : nullptr, scenarioSelected)},
        };
        if (baselineSelected != nullptr && scenarioInBaseline.has_value()) {
            root["first_selected_route_difference"]["baseline_choice_margin_loss_per_prediction_voxel"] =
                baselineDecision.routes[*scenarioInBaseline].lossPerPredictionVoxel - baselineSelected->lossPerPredictionVoxel;
        }
        if (scenarioSelected != nullptr && baselineInScenario.has_value()) {
            root["first_selected_route_difference"]["scenario_choice_margin_loss_per_prediction_voxel"] =
                scenarioDecision.routes[*baselineInScenario].lossPerPredictionVoxel - scenarioSelected->lossPerPredictionVoxel;
        }
    } else if (firstDifference.has_value()) {
        root["first_selected_route_difference"] = {
            {"decision_index", *firstDifference},
            {"reason", "decision_count_differs_after_common_prefix"},
        };
    }
    if (maximumDistanceIndex.has_value()) {
        root["maximum_selected_route_distance"] = {
            {"decision_index", *maximumDistanceIndex},
            {"symmetric_max_base_voxels", maximumDistance},
        };
    }
    const auto path = directory / ("decision-comparison-" + std::string(scenarioName) + ".json");
    vc::core::util::atomicWriteString(path, root.dump(2) + "\n");
    std::cout << std::setprecision(17) << "fiberlet_quantization_decision_comparison"
              << " artifact=" << path.string() << " baseline_decisions=" << baselineDecisions.size()
              << " scenario_decisions=" << scenarioDecisions.size();
    if (firstDifference.has_value())
        std::cout << " first_selected_route_difference=" << *firstDifference;
    else
        std::cout << " first_selected_route_difference=none";
    if (maximumDistanceIndex.has_value()) {
        std::cout << " maximum_route_distance_decision=" << *maximumDistanceIndex << " maximum_route_distance_base=" << maximumDistance;
    }
    std::cout << '\n';
    std::cout << "fiberlet_quantization_route_cost_statistics"
              << " source=baseline json=" << root["baseline_route_cost_statistics"].dump() << '\n';
    std::cout << "fiberlet_quantization_route_cost_statistics"
              << " source=scenario json=" << root["scenario_route_cost_statistics"].dump() << '\n';
    return path;
}

std::string progressDuration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
        return "n/a";
    const auto rounded = static_cast<std::uint64_t>(std::llround(seconds));
    const auto hours = rounded / 3600;
    const auto minutes = (rounded % 3600) / 60;
    const auto remainingSeconds = rounded % 60;
    std::ostringstream text;
    if (hours > 0)
        text << hours << 'h';
    if (hours > 0 || minutes > 0)
        text << minutes << 'm';
    text << remainingSeconds << 's';
    return text.str();
}

using ReplayChunkId = std::array<int, 4>;

struct ReplayPreprocessingSnapshot {
    size_t expectedAnchors = 0;
    size_t resolvedAnchors = 0;
    size_t expectedPrefixes = 0;
    size_t resolvedPrefixes = 0;
};

class ReplayPreprocessingProgress
{
public:
    void configure(std::set<ReplayChunkId> expectedAnchors, std::set<ReplayChunkId> expectedPrefixes)
    {
        std::lock_guard lock(mutex_);
        expectedAnchors_ = std::move(expectedAnchors);
        expectedPrefixes_ = std::move(expectedPrefixes);
        resolvedAnchors_.clear();
        resolvedPrefixes_.clear();
        enabled_ = true;
    }

    void resolve(vc::fiber_tracer::FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status)
    {
        if (status != vc::render::ChunkFetchStatus::Found)
            return;
        const ReplayChunkId id{key.level, key.iz, key.iy, key.ix};
        std::lock_guard lock(mutex_);
        if (!enabled_)
            return;
        if (kind == vc::fiber_tracer::FiberletStorageChunkKind::Anchors) {
            if (expectedAnchors_.contains(id))
                resolvedAnchors_.insert(id);
        } else if (kind == vc::fiber_tracer::FiberletStorageChunkKind::FiberletPrefix) {
            if (expectedPrefixes_.contains(id))
                resolvedPrefixes_.insert(id);
        }
    }

    [[nodiscard]] ReplayPreprocessingSnapshot snapshot() const
    {
        std::lock_guard lock(mutex_);
        return {
            expectedAnchors_.size(),
            resolvedAnchors_.size(),
            expectedPrefixes_.size(),
            resolvedPrefixes_.size(),
        };
    }

    void disable()
    {
        std::lock_guard lock(mutex_);
        enabled_ = false;
    }

private:
    mutable std::mutex mutex_;
    std::set<ReplayChunkId> expectedAnchors_;
    std::set<ReplayChunkId> expectedPrefixes_;
    std::set<ReplayChunkId> resolvedAnchors_;
    std::set<ReplayChunkId> resolvedPrefixes_;
    bool enabled_ = false;
};

class ReplayOverallProgress
{
public:
    explicit ReplayOverallProgress(bool enabled, std::string label = "fiber replay") : enabled_(enabled), label_(std::move(label))
    {
        if (enabled_)
            ticker_ = std::thread([this] { tickerLoop(); });
    }

    ~ReplayOverallProgress() { endLine(); }

    ReplayOverallProgress(const ReplayOverallProgress&) = delete;
    ReplayOverallProgress& operator=(const ReplayOverallProgress&) = delete;

    void updateGreedy(double fraction) { updateTracer(fraction, greedyFraction_, {}, {}); }
    void updateFiberlet(double fraction, std::optional<std::size_t> rolloutExpandedStateCount = {}, std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel = {})
    {
        updateTracer(fraction, fiberletFraction_, rolloutExpandedStateCount, minimumAppliedLocalPruneLossCutoffPerPredictionVoxel);
    }

    void beginTracing()
    {
        std::lock_guard lock(mutex_);
        startTracingLocked(std::chrono::steady_clock::now());
        renderLocked(true, false);
    }

    void printEventLine(const std::string& line)
    {
        std::lock_guard lock(mutex_);
        closeLineLocked();
        std::cerr << line << '\n';
        renderLocked(true, false);
    }

    void attachPreprocessing(std::shared_ptr<ReplayPreprocessingProgress> preprocessing)
    {
        const auto snapshot = preprocessing->snapshot();
        std::lock_guard lock(mutex_);
        preprocessing_ = std::move(preprocessing);
        if (!preprocessingStarted_) {
            preprocessingStarted_ = true;
            preprocessingStart_ = std::chrono::steady_clock::now();
        }
        updatePreprocessingLocked(snapshot);
        renderLocked(true, false);
    }

    void tracingComplete()
    {
        std::lock_guard lock(mutex_);
        if (preprocessing_)
            updatePreprocessingLocked(preprocessing_->snapshot());
        greedyFraction_ = 1.0;
        fiberletFraction_ = 1.0;
        startTracingLocked(std::chrono::steady_clock::now());
        renderLocked(true, true);
        traceDisplayComplete_ = true;
    }

    void setOutputStage(std::string stage, std::optional<std::size_t> completed = {}, std::optional<std::size_t> total = {})
    {
        std::lock_guard lock(mutex_);
        closeLineLocked();
        if (!outputStarted_) {
            outputStarted_ = true;
        }
        outputStart_ = std::chrono::steady_clock::now();
        outputStage_ = std::move(stage);
        outputCompleted_ = completed;
        outputTotal_ = total;
        renderLocked(true, false);
    }

    void updateOutputStage(std::size_t completed, std::size_t total)
    {
        std::lock_guard lock(mutex_);
        outputCompleted_ = std::max(outputCompleted_.value_or(0), completed);
        outputTotal_ = total;
        renderLocked(true, false);
    }

    void finish()
    {
        disablePreprocessing();
        stopTicker();
        std::lock_guard lock(mutex_);
        if (!enabled_ || finished_)
            return;
        renderLocked(true, true);
        finished_ = true;
        lineOpen_ = false;
    }

    void endLine()
    {
        disablePreprocessing();
        stopTicker();
        std::lock_guard lock(mutex_);
        closeLineLocked();
    }

private:
    void updateTracer(double fraction, double& current, std::optional<std::size_t> rolloutExpandedStateCount, std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel)
    {
        if (!std::isfinite(fraction))
            return;
        std::lock_guard lock(mutex_);
        startTracingLocked(std::chrono::steady_clock::now());
        current = std::max(current, std::clamp(fraction, 0.0, 1.0));
        if (rolloutExpandedStateCount.has_value())
            rolloutExpandedStateCount_ = rolloutExpandedStateCount;
        if (minimumAppliedLocalPruneLossCutoffPerPredictionVoxel.has_value()) {
            minimumAppliedLocalPruneLossCutoffPerPredictionVoxel_ = minimumAppliedLocalPruneLossCutoffPerPredictionVoxel;
        }
        renderLocked(false, false);
    }

    void startTracingLocked(std::chrono::steady_clock::time_point now)
    {
        if (traceStarted_)
            return;
        traceStarted_ = true;
        traceStart_ = now;
        traceSpeedSamples_.clear();
        traceSpeedSamples_.emplace_back(now, std::min(greedyFraction_, fiberletFraction_));
    }

    void updatePreprocessingLocked(const ReplayPreprocessingSnapshot& snapshot)
    {
        constexpr double prefixWeight = 16.0;
        const double expected = static_cast<double>(snapshot.expectedAnchors) + prefixWeight * static_cast<double>(snapshot.expectedPrefixes);
        const double resolved = static_cast<double>(snapshot.resolvedAnchors) + prefixWeight * static_cast<double>(snapshot.resolvedPrefixes);
        const double fraction = expected > 0.0 ? std::clamp(resolved / expected, 0.0, 1.0) : 1.0;
        preprocessingFraction_ = std::max(preprocessingFraction_, fraction);
    }

    void tickerLoop()
    {
        std::unique_lock waitLock(tickerMutex_);
        while (!tickerCv_.wait_for(waitLock, std::chrono::milliseconds(250), [this] { return tickerStop_; })) {
            waitLock.unlock();
            std::shared_ptr<ReplayPreprocessingProgress> preprocessing;
            {
                std::lock_guard lock(mutex_);
                preprocessing = preprocessing_;
            }
            const auto snapshot = preprocessing ? std::optional{preprocessing->snapshot()} : std::nullopt;
            {
                std::lock_guard lock(mutex_);
                if (!finished_) {
                    if (snapshot)
                        updatePreprocessingLocked(*snapshot);
                    sampleTraceSpeedLocked(std::chrono::steady_clock::now());
                    renderLocked(true, false);
                }
            }
            waitLock.lock();
        }
    }

    void stopTicker()
    {
        {
            std::lock_guard lock(tickerMutex_);
            tickerStop_ = true;
        }
        tickerCv_.notify_all();
        if (ticker_.joinable())
            ticker_.join();
    }

    void disablePreprocessing()
    {
        std::shared_ptr<ReplayPreprocessingProgress> preprocessing;
        {
            std::lock_guard lock(mutex_);
            preprocessing = preprocessing_;
        }
        if (preprocessing)
            preprocessing->disable();
    }

    static void appendProgressMetric(
        std::ostringstream& line,
        std::string_view name,
        double fraction,
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point now,
        std::optional<double> currentEta = {})
    {
        const double elapsed = std::chrono::duration<double>(now - start).count();
        const double eta = fraction > 0.0 && fraction < 1.0 ? elapsed * (1.0 - fraction) / fraction
                           : fraction >= 1.0                ? 0.0
                                                            : std::numeric_limits<double>::infinity();
        constexpr int width = 12;
        const int filled = std::clamp(static_cast<int>(std::floor(fraction * width)), 0, width);
        const double percent = 100.0 * fraction;
        line << name << " [" << std::string(filled, '#') << std::string(width - filled, '-') << "] " << std::fixed
             << std::setprecision(percent < 10.0 ? 2 : 1) << percent << '%';
        line << " eta=" << progressDuration(eta);
        if (currentEta.has_value())
            line << " eta_current=" << progressDuration(*currentEta);
    }

    void sampleTraceSpeedLocked(std::chrono::steady_clock::time_point now)
    {
        if (!traceStarted_)
            return;
        const double fraction = std::min(greedyFraction_, fiberletFraction_);
        traceSpeedSamples_.emplace_back(now, fraction);
        const auto windowBegin = now - std::chrono::seconds(10);
        while (traceSpeedSamples_.size() > 2 && traceSpeedSamples_[1].first <= windowBegin)
            traceSpeedSamples_.pop_front();
    }

    [[nodiscard]] double currentTraceEtaLocked(std::chrono::steady_clock::time_point now) const
    {
        if (traceSpeedSamples_.size() < 2)
            return std::numeric_limits<double>::infinity();
        const auto& first = traceSpeedSamples_.front();
        const double seconds = std::chrono::duration<double>(now - first.first).count();
        const double fraction = std::min(greedyFraction_, fiberletFraction_);
        const double rate = seconds > 0.0 ? (fraction - first.second) / seconds : 0.0;
        return rate > 0.0 ? (1.0 - fraction) / rate : std::numeric_limits<double>::infinity();
    }

    void renderLocked(bool force, bool final)
    {
        if (!enabled_ || (!preprocessingStarted_ && !traceStarted_ && !outputStarted_) || (traceDisplayComplete_ && !outputStarted_))
            return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && lastRender_.time_since_epoch().count() != 0 && now - lastRender_ < std::chrono::milliseconds(250))
            return;
        lastRender_ = now;
        std::ostringstream line;
        line << label_ << ' ';
        if (outputStarted_) {
            line << "output stage=" << outputStage_;
            if (outputCompleted_.has_value() && outputTotal_.has_value()) {
                const double fraction =
                    *outputTotal_ == 0 ? 1.0 : std::clamp(static_cast<double>(*outputCompleted_) / static_cast<double>(*outputTotal_), 0.0, 1.0);
                line << ' ';
                appendProgressMetric(line, "items", fraction, outputStart_, now);
                line << " (" << *outputCompleted_ << '/' << *outputTotal_ << ')';
            } else {
                line << " eta=n/a";
            }
        } else {
            bool needsSeparator = false;
            if (preprocessingStarted_ && preprocessingFraction_ < 1.0) {
                appendProgressMetric(line, "cache/prep", preprocessingFraction_, preprocessingStart_, now);
                needsSeparator = true;
            }
            if (needsSeparator && traceStarted_)
                line << " | ";
            if (traceStarted_) {
                appendProgressMetric(line, "trace", std::min(greedyFraction_, fiberletFraction_), traceStart_, now, currentTraceEtaLocked(now));
                if (rolloutExpandedStateCount_.has_value())
                    line << " fiberlet_rollout_expansions=" << *rolloutExpandedStateCount_;
                if (minimumAppliedLocalPruneLossCutoffPerPredictionVoxel_.has_value()) {
                    line << " fiberlet_local_cutoff_loss_per_vx_min=" << std::setprecision(6) << *minimumAppliedLocalPruneLossCutoffPerPredictionVoxel_;
                }
            }
        }
        line << " elapsed=" << progressDuration(std::chrono::duration<double>(now - start_).count());
        const auto rendered = line.str();
        std::cerr << '\r' << rendered;
        if (renderedWidth_ > rendered.size())
            std::cerr << std::string(renderedWidth_ - rendered.size(), ' ');
        if (final)
            std::cerr << '\n';
        else
            std::cerr << std::flush;
        lineOpen_ = !final;
        renderedWidth_ = final ? 0 : rendered.size();
    }

    void closeLineLocked()
    {
        if (enabled_ && lineOpen_) {
            std::cerr << '\n';
            lineOpen_ = false;
        }
        renderedWidth_ = 0;
    }

    const bool enabled_;
    const std::string label_;
    const std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::mutex mutex_;
    std::chrono::steady_clock::time_point lastRender_{};
    std::chrono::steady_clock::time_point preprocessingStart_{};
    std::chrono::steady_clock::time_point traceStart_{};
    std::chrono::steady_clock::time_point outputStart_{};
    double greedyFraction_ = 0.0;
    double fiberletFraction_ = 0.0;
    double preprocessingFraction_ = 0.0;
    std::shared_ptr<ReplayPreprocessingProgress> preprocessing_;
    std::string outputStage_;
    std::optional<std::size_t> outputCompleted_;
    std::optional<std::size_t> outputTotal_;
    std::optional<std::size_t> rolloutExpandedStateCount_;
    std::optional<double> minimumAppliedLocalPruneLossCutoffPerPredictionVoxel_;
    std::deque<std::pair<std::chrono::steady_clock::time_point, double>> traceSpeedSamples_;
    std::size_t renderedWidth_ = 0;
    bool preprocessingStarted_ = false;
    bool traceStarted_ = false;
    bool traceDisplayComplete_ = false;
    bool outputStarted_ = false;
    bool lineOpen_ = false;
    bool finished_ = false;
    std::mutex tickerMutex_;
    std::condition_variable tickerCv_;
    bool tickerStop_ = false;
    std::thread ticker_;
};

void printRateProgress(const char* prefix, const std::string& phase, const char* rateName, size_t completed, size_t total, double elapsedSeconds)
{
    const double percent = total == 0 ? 100.0 : 100.0 * static_cast<double>(completed) / static_cast<double>(total);
    const double rate = elapsedSeconds > 0.0 ? static_cast<double>(completed) / elapsedSeconds : 0.0;
    const double eta = completed >= total ? 0.0
                       : rate > 0.0       ? static_cast<double>(total - completed) / rate
                                          : std::numeric_limits<double>::infinity();
    std::ostringstream line;
    line << std::fixed << std::setprecision(1) << prefix;
    if (!phase.empty())
        line << " phase=" << phase;
    line << " completed=" << completed << " total=" << total << " percent=" << percent << " elapsed_seconds=" << elapsedSeconds << ' '
         << rateName << '=' << rate << " eta_seconds=";
    if (std::isfinite(eta))
        line << eta;
    else
        line << "n/a";
    std::cerr << line.str() << '\n';
}

void printAnchorProgress(const vc::fiber_tracer::FiberAnchorProgress& progress)
{
    printRateProgress("fiber_anchor_progress", progress.phase, "cells_per_second", progress.completed, progress.total, progress.elapsedSeconds);
}

void printFiberletProgress(const vc::fiber_tracer::FiberletPathProgress& progress)
{
    const char* rateName = progress.phase.find("sampling") != std::string::npos ? "voxels_per_second" : "candidates_per_second";
    printRateProgress("fiberlet_progress", progress.phase, rateName, progress.completed, progress.total, progress.elapsedSeconds);
}

void printQuantizationSummary(std::ostream& output, const char* name, const vc::fiber_tracer::FiberletQuantizationSummary& summary)
{
    output << ' ' << name << "_count=" << summary.count << ' ' << name << "_min=" << summary.minimum << ' ' << name
           << "_mean=" << summary.mean << ' ' << name << "_median=" << summary.median << ' ' << name << "_max=" << summary.maximum;
}

void printQuantizationProgress(const vc::fiber_tracer::FiberletQuantizationProgress& progress)
{
    printRateProgress(
        "fiberlet_quantization_progress",
        progress.phase + ":" + progress.scenario,
        "samples_per_second",
        progress.completed,
        progress.total,
        progress.elapsedSeconds);
}

struct TubeExtractionResult {
    vc::fiber_tracer::FiberReplayTube tube;
    vc::fiber_tracer::FiberAnchorExtractionReport anchors;
    vc::fiber_tracer::FiberletPathReport paths;
    double anchorSeconds = 0.0;
    double anchorCpuSeconds = 0.0;
    double fiberletSeconds = 0.0;
    double fiberletCpuSeconds = 0.0;
};

void printTubeExtractionProfile(std::ostream& output, const TubeExtractionResult& extraction)
{
    const auto previousPrecision = output.precision();
    const auto& anchor = extraction.anchors.profile;
    const auto& fit = anchor.fit;
    const auto& paths = extraction.paths;
    const double anchorProfiledSeconds = anchor.setupSeconds + anchor.tilePlanningSeconds + anchor.cellProcessingSeconds + anchor.selectionSeconds +
                                         anchor.initialDiagnosticsSeconds + anchor.duplicateSuppressionSeconds + anchor.finalizationSeconds;
    const double fiberletProfiledSeconds = paths.candidateGenerationSeconds + paths.preparationSeconds + paths.cornerMergeSeconds +
                                           paths.predictionSamplingSeconds + paths.normalSamplingSeconds +
                                           paths.samplingMaterializationSeconds + paths.searchSeconds;
    const double fitProfiledWorkSeconds = fit.setupWorkSeconds + fit.seedGenerationWorkSeconds + fit.seedPairRefinementWorkSeconds +
                                          fit.initializationWorkSeconds + fit.localRefinementWorkSeconds + fit.peakSearchWorkSeconds +
                                          fit.finalEvaluationWorkSeconds;
    const double localProfiledWorkSeconds = fit.robustObservationPreparationWorkSeconds + fit.localTensorProposalWorkSeconds +
                                            fit.localCentroidProposalWorkSeconds + fit.localStateEvaluationWorkSeconds;
    const auto depthCounts = [](const auto& counts) {
        std::ostringstream encoded;
        for (size_t depth = 0; depth < counts.size(); ++depth) {
            if (depth != 0)
                encoded << ',';
            encoded << counts[depth];
        }
        return encoded.str();
    };
    output << std::setprecision(17) << "fiberlet_extraction_profile version=30"
           << " anchor_elapsed_seconds=" << extraction.anchors.elapsedSeconds << " anchor_cpu_seconds=" << anchor.elapsedCpuSeconds
           << " anchor_profiled_seconds=" << anchorProfiledSeconds
           << " anchor_residual_seconds=" << std::max(0.0, extraction.anchors.elapsedSeconds - anchorProfiledSeconds)
           << " anchor_selected_cells=" << anchor.selectedCells << " anchor_context_cells=" << anchor.contextCells
           << " anchor_work_cells=" << anchor.workCells << " anchor_tiles=" << anchor.tiles
           << " anchor_sampling_partitions=" << anchor.samplingPartitions << " anchor_workers=" << anchor.workers
           << " anchor_sampler_calls=" << anchor.predictionSamplerCalls << " anchor_shared_sampling_batches=" << anchor.sharedSamplingBatches
           << " anchor_max_sampling_batch_voxels=" << anchor.maximumSamplingBatchVoxels
           << " anchor_submitted_prediction_voxels=" << anchor.submittedPredictionVoxels
           << " anchor_unique_tile_prediction_voxels=" << anchor.uniqueTilePredictionVoxels
           << " anchor_reused_prediction_voxels=" << anchor.reusedPredictionVoxels << " anchor_cell_result_handle_bytes=" << anchor.cellResultHandleBytes
           << " anchor_max_raw_interval_bytes=" << anchor.maximumRawIntervalBytes << " anchor_shared_observation_voxels=" << anchor.sharedObservationVoxels
           << " anchor_max_shared_sample_bytes=" << anchor.maximumSharedSampleBytes
           << " anchor_max_accounted_live_bytes=" << anchor.maximumAccountedLiveBytes << " anchor_candidate_observations=" << anchor.candidateObservations
           << " anchor_retained_observations=" << anchor.retainedObservations << " anchor_support_stencil_cells=" << anchor.supportStencilCells
           << " anchor_clipped_support_cells=" << anchor.clippedSupportCells << " anchor_gradient_attempts=" << anchor.gradientAttempts
           << " anchor_valid_gradients=" << anchor.validGradients << " anchor_gradient_computations=" << anchor.gradientComputations
           << " anchor_valid_gradient_computations=" << anchor.validGradientComputations
           << " anchor_retain_predicate_calls=" << anchor.retainPredicateCalls << " anchor_fit_iterations=" << anchor.fitIterations
           << " anchor_setup_seconds=" << anchor.setupSeconds << " anchor_tile_planning_seconds=" << anchor.tilePlanningSeconds
           << " anchor_interval_preparation_seconds=" << anchor.intervalPreparationSeconds
           << " anchor_interval_preparation_cpu_seconds=" << anchor.intervalPreparationCpuSeconds
           << " anchor_cell_processing_seconds=" << anchor.cellProcessingSeconds << " anchor_cell_processing_cpu_seconds=" << anchor.cellProcessingCpuSeconds
           << " anchor_shared_sampling_seconds=" << anchor.sharedSamplingSeconds << " anchor_shared_sampling_cpu_seconds=" << anchor.sharedSamplingCpuSeconds
           << " anchor_coordinate_construction_work_seconds=" << anchor.coordinateConstructionWorkSeconds
           << " anchor_prediction_sampling_work_seconds=" << anchor.predictionSamplingWorkSeconds
           << " anchor_shared_observation_construction_work_seconds=" << anchor.sharedObservationConstructionWorkSeconds
           << " anchor_tile_observation_index_work_seconds=" << anchor.tileObservationIndexWorkSeconds
           << " anchor_gradient_construction_work_seconds=" << anchor.gradientConstructionWorkSeconds
           << " anchor_observation_construction_work_seconds=" << anchor.observationConstructionWorkSeconds
           << " anchor_fitting_work_seconds=" << anchor.fittingWorkSeconds << " anchor_partition_p50_seconds=" << anchor.partitionP50Seconds
           << " anchor_partition_p95_seconds=" << anchor.partitionP95Seconds << " anchor_partition_max_seconds=" << anchor.partitionMaximumSeconds
           << " anchor_tile_preparation_p50_seconds=" << anchor.tilePreparationP50Seconds
           << " anchor_tile_preparation_p95_seconds=" << anchor.tilePreparationP95Seconds
           << " anchor_tile_preparation_max_seconds=" << anchor.tilePreparationMaximumSeconds
           << " anchor_cell_processing_p50_seconds=" << anchor.cellProcessingP50Seconds
           << " anchor_cell_processing_p95_seconds=" << anchor.cellProcessingP95Seconds
           << " anchor_cell_processing_max_seconds=" << anchor.cellProcessingMaximumSeconds << " anchor_fit_invocations=" << fit.invocations
           << " anchor_fit_nonempty_cells=" << fit.nonemptyCells << " anchor_fit_weighted_observations=" << fit.weightedObservations
           << " anchor_fit_owned_discovery_observation_visits=" << fit.ownedDiscoveryObservationVisits
           << " anchor_fit_owned_initialization_observation_visits=" << fit.ownedInitializationObservationVisits
           << " anchor_fit_avoided_owned_support_observation_visits=" << fit.avoidedOwnedSupportObservationVisits
           << " anchor_fit_seeds=" << fit.seeds << " anchor_fit_seed_generation_observation_visits=" << fit.seedGenerationObservationVisits
           << " anchor_fit_seed_pairs=" << fit.seedPairs << " anchor_fit_seed_pair_iterations=" << fit.seedPairIterations
           << " anchor_fit_seed_assignment_observation_visits=" << fit.seedAssignmentObservationVisits
           << " anchor_fit_seed_tensor_observation_visits=" << fit.seedTensorObservationVisits
           << " anchor_fit_seed_objective_observation_visits=" << fit.seedObjectiveObservationVisits
           << " anchor_fit_initialization_observation_visits=" << fit.initializationObservationVisits
           << " anchor_fit_local_refinement_attempts=" << fit.localRefinementAttempts
           << " anchor_fit_local_refinement_accepted_steps=" << fit.localRefinementAcceptedSteps
           << " anchor_fit_backtracking_evaluations=" << fit.backtrackingEvaluations
           << " anchor_fit_robust_components_without_outliers=" << fit.robustComponentsWithoutOutliers
           << " anchor_fit_robust_trimmed_components=" << fit.robustTrimmedComponents
           << " anchor_fit_robust_removed_nonunique_components=" << fit.robustRemovedNonuniqueComponents
           << " anchor_fit_robust_hard_limit_hits=" << fit.robustHardLimitHits << " anchor_fit_spatial_candidates_tested=" << fit.spatialCandidatesTested
           << " anchor_fit_robust_candidate_trimmed_mass=" << fit.robustCandidateTrimmedMass
           << " anchor_fit_robust_trimmed_mass=" << fit.robustTrimmedMass << " anchor_fit_robust_retained_mass=" << fit.robustRetainedMass
           << " anchor_fit_spatial_tested_by_depth=" << depthCounts(fit.spatialCandidatesTestedByDepth)
           << " anchor_fit_spatial_accepted_by_depth=" << depthCounts(fit.spatialCandidatesAcceptedByDepth)
           << " anchor_fit_local_tensor_observation_visits=" << fit.localTensorObservationVisits
           << " anchor_fit_robust_axis_proposal_calls=" << fit.robustAxisProposalCalls
           << " anchor_fit_robust_axis_logical_observation_visits=" << fit.robustAxisLogicalObservationVisits
           << " anchor_fit_robust_axis_eligible_observation_visits=" << fit.robustAxisEligibleObservationVisits
           << " anchor_fit_robust_axis_indexed_observation_visits=" << fit.robustAxisIndexedObservationVisits
           << " anchor_fit_robust_axis_cutoff_observation_visits=" << fit.robustAxisCutoffObservationVisits
           << " anchor_fit_robust_membership_proposal_calls=" << fit.robustMembershipProposalCalls
           << " anchor_fit_robust_membership_logical_observation_visits=" << fit.robustMembershipLogicalObservationVisits
           << " anchor_fit_robust_membership_eligible_observation_visits=" << fit.robustMembershipEligibleObservationVisits
           << " anchor_fit_robust_membership_indexed_observation_visits=" << fit.robustMembershipIndexedObservationVisits
           << " anchor_fit_robust_membership_cutoff_observation_visits=" << fit.robustMembershipCutoffObservationVisits
           << " anchor_fit_robust_proposal_buffer_initializations=" << fit.robustProposalBufferInitializations
           << " anchor_fit_robust_proposal_initialized_bytes=" << fit.robustProposalInitializedBytes
           << " anchor_fit_robust_evaluation_copied_bytes=" << fit.robustEvaluationCopiedBytes
           << " anchor_fit_robust_prepared_observation_records=" << fit.robustPreparedObservationRecords
           << " anchor_fit_robust_prepared_observation_record_bytes=" << fit.robustPreparedObservationRecordBytes
           << " anchor_fit_local_centroid_observation_visits=" << fit.localCentroidObservationVisits
           << " anchor_fit_local_centroid_indexed_observation_visits=" << fit.localCentroidIndexedObservationVisits
           << " anchor_fit_refined_evaluation_observation_visits=" << fit.refinedEvaluationObservationVisits
           << " anchor_fit_peak_components=" << fit.peakComponents << " anchor_fit_peak_preparation_observation_visits=" << fit.peakPreparationObservationVisits
           << " anchor_fit_peak_prepared_response_observations=" << fit.peakPreparedResponseObservations
           << " anchor_fit_peak_prepared_evidence_observations=" << fit.peakPreparedEvidenceObservations
           << " anchor_fit_peak_response_observation_record_bytes=" << fit.peakResponseObservationRecordBytes
           << " anchor_fit_peak_evidence_observation_record_bytes=" << fit.peakEvidenceObservationRecordBytes
           << " anchor_fit_peak_maximum_observation_storage_bytes=" << fit.peakMaximumObservationStorageBytes
           << " anchor_fit_peak_grid_response_requests=" << fit.peakGridResponseRequests
           << " anchor_fit_peak_computed_grid_responses=" << fit.peakComputedGridResponses
           << " anchor_fit_peak_acceptance_responses=" << fit.peakAcceptanceResponses
           << " anchor_fit_peak_response_observation_visits=" << fit.peakResponseObservationVisits
           << " anchor_fit_peak_response_radial_acceptances=" << fit.peakResponseRadialAcceptances
           << " anchor_fit_peak_response_evidence_observation_visits=" << fit.peakResponseEvidenceObservationVisits
           << " anchor_fit_final_evaluation_observation_visits=" << fit.finalEvaluationObservationVisits
           << " anchor_fit_setup_work_seconds=" << fit.setupWorkSeconds << " anchor_fit_seed_generation_work_seconds=" << fit.seedGenerationWorkSeconds
           << " anchor_fit_seed_pair_refinement_work_seconds=" << fit.seedPairRefinementWorkSeconds
           << " anchor_fit_initialization_work_seconds=" << fit.initializationWorkSeconds
           << " anchor_fit_local_refinement_work_seconds=" << fit.localRefinementWorkSeconds
           << " anchor_fit_local_tensor_proposal_work_seconds=" << fit.localTensorProposalWorkSeconds
           << " anchor_fit_robust_axis_proposal_work_seconds=" << fit.robustAxisProposalWorkSeconds
           << " anchor_fit_robust_membership_proposal_work_seconds=" << fit.robustMembershipProposalWorkSeconds
           << " anchor_fit_robust_observation_preparation_work_seconds=" << fit.robustObservationPreparationWorkSeconds
           << " anchor_fit_local_centroid_proposal_work_seconds=" << fit.localCentroidProposalWorkSeconds
           << " anchor_fit_local_state_evaluation_work_seconds=" << fit.localStateEvaluationWorkSeconds
           << " anchor_fit_local_profiled_work_seconds=" << localProfiledWorkSeconds
           << " anchor_fit_local_control_work_seconds=" << std::max(0.0, fit.localRefinementWorkSeconds - localProfiledWorkSeconds)
           << " anchor_fit_peak_search_work_seconds=" << fit.peakSearchWorkSeconds
           << " anchor_fit_final_evaluation_work_seconds=" << fit.finalEvaluationWorkSeconds << " anchor_fit_profiled_work_seconds=" << fitProfiledWorkSeconds
           << " anchor_fit_residual_work_seconds=" << std::max(0.0, anchor.fittingWorkSeconds - fitProfiledWorkSeconds)
           << " anchor_selection_seconds=" << anchor.selectionSeconds << " anchor_initial_diagnostics_seconds=" << anchor.initialDiagnosticsSeconds
           << " anchor_duplicate_suppression_seconds=" << anchor.duplicateSuppressionSeconds
           << " anchor_finalization_seconds=" << anchor.finalizationSeconds << " fiberlet_elapsed_seconds=" << paths.elapsedSeconds
           << " fiberlet_cpu_seconds=" << paths.elapsedCpuSeconds << " fiberlet_profiled_seconds=" << fiberletProfiledSeconds
           << " fiberlet_residual_seconds=" << std::max(0.0, paths.elapsedSeconds - fiberletProfiledSeconds)
           << " fiberlet_candidate_predicate_calls=" << paths.candidatePointPredicateCalls << " fiberlet_lattice_node_positions=" << paths.latticeNodePositions
           << " fiberlet_corridor_segment_tests=" << paths.corridorSegmentTests << " fiberlet_corridor_accepted_nodes=" << paths.corridorAcceptedNodes
           << " fiberlet_node_predicate_calls=" << paths.nodePointPredicateCalls << " fiberlet_retained_search_nodes=" << paths.retainedSearchNodes
           << " fiberlet_corner_insertion_attempts=" << paths.interpolationCornerInsertions
           << " fiberlet_corner_worker_unique_voxels=" << paths.cornerWorkerUniqueVoxels << " fiberlet_corner_worker_pages=" << paths.cornerWorkerPages
           << " fiberlet_corner_page_directory_probes=" << paths.cornerPageDirectoryProbes
           << " fiberlet_corner_same_page_hits=" << paths.cornerSamePageHits << " fiberlet_corner_cached_page_hits=" << paths.cornerCachedPageHits
           << " fiberlet_corner_merged_pages=" << paths.cornerMergedPages << " fiberlet_unique_sampled_voxels=" << paths.sampledVoxels
           << " fiberlet_interpolated_scoring_points=" << paths.interpolatedScoringPoints
           << " fiberlet_endpoint_scoring_interpolations=" << paths.endpointScoringInterpolations
           << " fiberlet_lazy_node_scoring_requests=" << paths.lazyNodeScoringRequests
           << " fiberlet_lazy_node_scoring_materializations=" << paths.lazyNodeScoringMaterializations
           << " fiberlet_lazy_node_scoring_cache_hits=" << paths.lazyNodeScoringCacheHits
           << " fiberlet_scoring_page_count=" << paths.scoringPageCount << " fiberlet_scoring_page_slots=" << paths.scoringPageSlots
           << " fiberlet_scoring_page_directory_probes=" << paths.scoringPageDirectoryProbes
           << " fiberlet_interpolation_profiled_points=" << paths.interpolationProfiledPoints
           << " fiberlet_interpolation_profiled_corners=" << paths.interpolationProfiledCorners
           << " fiberlet_interpolation_profiled_prediction_identical=" << paths.interpolationProfiledPredictionIdentical
           << " fiberlet_interpolation_profiled_normal_identical=" << paths.interpolationProfiledNormalIdentical
           << " fiberlet_interpolation_profiled_prediction_principal_solves=" << paths.interpolationProfiledPredictionPrincipalSolves
           << " fiberlet_interpolation_profiled_normal_principal_solves=" << paths.interpolationProfiledNormalPrincipalSolves
           << " fiberlet_interpolation_prediction_closed_form_resolutions=" << paths.interpolationPredictionClosedFormResolutions
           << " fiberlet_interpolation_normal_closed_form_resolutions=" << paths.interpolationNormalClosedFormResolutions
           << " fiberlet_interpolation_prediction_iterative_fallbacks=" << paths.interpolationPredictionIterativeFallbacks
           << " fiberlet_interpolation_normal_iterative_fallbacks=" << paths.interpolationNormalIterativeFallbacks
           << " fiberlet_dp_node_index_entries=" << paths.dpNodeIndexEntries << " fiberlet_dp_node_index_slots=" << paths.dpNodeIndexSlots
           << " fiberlet_dp_prepared_nodes=" << paths.dpPreparedNodes << " fiberlet_dp_max_prepared_node_bytes=" << paths.dpMaximumPreparedNodeBytes
           << " fiberlet_dp_max_lazy_cache_index_bytes=" << paths.dpMaximumLazyCacheIndexBytes
           << " fiberlet_dp_max_direct_index_bytes=" << paths.dpMaximumDirectIndexBytes << " fiberlet_dp_max_state_bytes=" << paths.dpMaximumStateBytes
           << " fiberlet_dp_shared_scoring_bytes=" << paths.dpSharedScoringBytes << " fiberlet_dp_reached_nodes=" << paths.dpReachedNodes
           << " fiberlet_dp_generated_edges=" << paths.dpGeneratedEdges << " fiberlet_dp_valid_edges=" << paths.dpValidEdges
           << " fiberlet_dp_reused_edges=" << paths.dpReusedEdges << " fiberlet_dp_transition_lookups=" << paths.dpTransitionLookups
           << " fiberlet_dp_reached_state_visits=" << paths.dpReachedStateVisits << " fiberlet_dp_relaxations=" << paths.dpRelaxations
           << " fiberlet_candidate_generation_seconds=" << paths.candidateGenerationSeconds
           << " fiberlet_candidate_generation_cpu_seconds=" << paths.candidateGenerationCpuSeconds
           << " fiberlet_preparation_seconds=" << paths.preparationSeconds << " fiberlet_preparation_cpu_seconds=" << paths.preparationCpuSeconds
           << " fiberlet_preparation_geometry_work_seconds=" << paths.preparationGeometryWorkSeconds
           << " fiberlet_node_enumeration_work_seconds=" << paths.preparationNodeEnumerationWorkSeconds
           << " fiberlet_corner_collection_work_seconds=" << paths.preparationCornerCollectionWorkSeconds
           << " fiberlet_corner_merge_seconds=" << paths.cornerMergeSeconds << " fiberlet_corner_merge_cpu_seconds=" << paths.cornerMergeCpuSeconds
           << " fiberlet_prediction_sampling_seconds=" << paths.predictionSamplingSeconds
           << " fiberlet_prediction_sampling_cpu_seconds=" << paths.predictionSamplingCpuSeconds
           << " fiberlet_normal_sampling_seconds=" << paths.normalSamplingSeconds << " fiberlet_normal_sampling_cpu_seconds=" << paths.normalSamplingCpuSeconds
           << " fiberlet_materialization_seconds=" << paths.samplingMaterializationSeconds
           << " fiberlet_materialization_cpu_seconds=" << paths.samplingMaterializationCpuSeconds
           << " fiberlet_scoring_index_seconds=" << paths.scoringIndexSeconds << " fiberlet_scoring_index_cpu_seconds=" << paths.scoringIndexCpuSeconds
           << " fiberlet_scoring_preparation_seconds=" << paths.scoringPreparationSeconds
           << " fiberlet_scoring_preparation_cpu_seconds=" << paths.scoringPreparationCpuSeconds
           << " fiberlet_interpolation_materialization_seconds=" << paths.interpolationMaterializationSeconds
           << " fiberlet_interpolation_materialization_cpu_seconds=" << paths.interpolationMaterializationCpuSeconds
           << " fiberlet_interpolation_profiled_lookup_seconds=" << paths.interpolationProfiledLookupSeconds
           << " fiberlet_interpolation_profiled_prediction_corner_seconds=" << paths.interpolationProfiledPredictionCornerSeconds
           << " fiberlet_interpolation_profiled_normal_corner_seconds=" << paths.interpolationProfiledNormalCornerSeconds
           << " fiberlet_interpolation_profiled_prediction_resolve_seconds=" << paths.interpolationProfiledPredictionResolveSeconds
           << " fiberlet_interpolation_profiled_normal_resolve_seconds=" << paths.interpolationProfiledNormalResolveSeconds
           << " fiberlet_search_seconds=" << paths.searchSeconds << " fiberlet_search_cpu_seconds=" << paths.searchCpuSeconds
           << " fiberlet_node_index_work_seconds=" << paths.searchNodeIndexWorkSeconds
           << " fiberlet_node_preparation_work_seconds=" << paths.searchNodePreparationWorkSeconds
           << " fiberlet_dp_work_seconds=" << paths.searchDpWorkSeconds << '\n';
    output.precision(previousPrecision);
}

TubeExtractionResult extractTubeFiberlets(
    const std::vector<cv::Vec3d>& referenceBase,
    double beginArcBase,
    double endArcBase,
    double radiusBaseVoxels,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid,
    const CliOptions& options,
    const vc::fiber_tracer::FiberPredictionField& field,
    const vc::lasagna::LasagnaNormalSampler& normalSampler,
    bool retainAnchorDiagnostics = true,
    bool reportDetailedProgress = true)
{
    if (!(endArcBase > beginArcBase))
        throw std::invalid_argument("fiberlet extraction interval must have positive length");
    TubeExtractionResult result;
    result.tube = vc::fiber_tracer::makeFiberReplayTube(
        referenceBase, 0.5 * (beginArcBase + endArcBase), 0.5 * (endArcBase - beginArcBase), radiusBaseVoxels, grid, options.anchors.cellSizePredictionVoxels);
    const auto containmentQuery = result.tube.makePredictionContainmentQuery(grid.predictionToBaseScale);
    const auto anchorStart = std::chrono::steady_clock::now();
    const double anchorCpuStart = processCpuSeconds();
    result.anchors = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid,
        options.anchors,
        [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
        result.tube.cellsZYX,
        [containmentQuery](const vc::fiber_tracer::FiberAnchor& anchor) { return containmentQuery.evaluatePredictionAnchor(anchor); },
        reportDetailedProgress ? vc::fiber_tracer::FiberAnchorProgressCallback{printAnchorProgress} : vc::fiber_tracer::FiberAnchorProgressCallback{},
        retainAnchorDiagnostics);
    result.anchorSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - anchorStart).count();
    result.anchorCpuSeconds = processCpuSeconds() - anchorCpuStart;

    vc::fiber_tracer::LoadedFiberAnchorArtifact loaded{result.anchors, {}};
    const auto fiberletStart = std::chrono::steady_clock::now();
    const double fiberletCpuStart = processCpuSeconds();
    result.paths = vc::fiber_tracer::traceFiberletPaths(
        loaded,
        grid,
        options.paths,
        [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
        normalSampler,
        reportDetailedProgress ? vc::fiber_tracer::FiberletPathProgressCallback{printFiberletProgress} : vc::fiber_tracer::FiberletPathProgressCallback{},
        [&](const cv::Vec3d& pointPrediction) { return containmentQuery.containsPredictionPoint(pointPrediction); });
    result.fiberletSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fiberletStart).count();
    result.fiberletCpuSeconds = processCpuSeconds() - fiberletCpuStart;
    return result;
}

struct StorageCompressionSample {
    std::size_t records = 0;
    std::size_t payloadBytes = 0;
    std::size_t outerZstdBytes = 0;
    std::size_t rawFieldBytes = 0;
    std::size_t wholeZstdBytes = 0;
};

std::size_t outerZstdSize(std::span<const std::byte> bytes)
{
    std::vector<std::byte> compressed(ZSTD_compressBound(bytes.size()));
    const std::size_t size = ZSTD_compress(
        compressed.data(), compressed.size(), bytes.data(), bytes.size(), 3);
    if (ZSTD_isError(size)) {
        throw std::runtime_error(
            std::string("fiberlet outer zstd encode failed: ") +
            ZSTD_getErrorName(size));
    }
    return size;
}

StorageCompressionSample storageCompressionSample(
    std::size_t records,
    std::span<const std::byte> payload)
{
    const auto rawFields = vc::fiber_tracer::materializeFiberletPayload(payload);
    return {
        records,
        payload.size(),
        outerZstdSize(payload),
        rawFields.size(),
        outerZstdSize(rawFields),
    };
}

vc::fiber_tracer::FiberletStorageCodecConfig compactCodec(
    const vc::fiber_tracer::FiberletDatasetMetadata& metadata,
    const std::array<int, 3>& owner)
{
    vc::fiber_tracer::FiberletStorageCodecConfig result;
    result.profile =
        vc::fiber_tracer::FiberletStorageProfile::CompactDirectionsFixedCost;
    result.chunkZYX = owner;
    result.datasetFingerprint = metadata.datasetFingerprint;
    std::int64_t maximumUnitsPerChunk = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        result.coordinateOriginZYX[axis] =
            static_cast<std::int64_t>(owner[axis]) *
            metadata.coordinateUnitsPerChunkZYX[axis];
        maximumUnitsPerChunk = std::max(
            maximumUnitsPerChunk,
            metadata.coordinateUnitsPerChunkZYX[axis]);
    }
    result.coordinateBits =
        maximumUnitsPerChunk <= 256
        ? 8
        : maximumUnitsPerChunk <= 65536 ? 16 : 32;
    result.deltaBits = 16;
    result.routeCountBits = 16;
    result.routeLatticeBits = 16;
    result.costBits = 16;
    result.positionQuantumBaseVoxels = 0;
    result.predictionToBaseScale = metadata.predictionToBaseScale;
    return result;
}

std::array<int, 3> storageOwnerChunk(
    const vc::fiber_tracer::FiberletStorageKey& key,
    const std::array<std::int64_t, 3>& unitsPerChunk)
{
    std::array<int, 3> result{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (key.coordinateZYX[axis] < 0 || unitsPerChunk[axis] <= 0)
            throw std::invalid_argument(
                "compact storage benchmark encountered an invalid key");
        const auto owner = key.coordinateZYX[axis] / unitsPerChunk[axis];
        if (owner > std::numeric_limits<int>::max())
            throw std::overflow_error(
                "compact storage benchmark chunk coordinate exceeds int");
        result[axis] = static_cast<int>(owner);
    }
    return result;
}

void printStorageCompressionSummary(
    const char* payload,
    std::vector<StorageCompressionSample> samples,
    int chunkSideBaseVoxels)
{
    if (samples.empty()) {
        std::cout << "fiberlet_storage_compression"
                  << " profile=compact_axis_cost_sqrt_u16_max256"
                  << " chunk_side_base=" << chunkSideBaseVoxels
                  << " internal_codec=field_zstd3"
                  << " payload=" << payload << " chunks=0\n";
        return;
    }
    std::uint64_t records = 0;
    std::uint64_t payloadBytes = 0;
    std::uint64_t outerBytes = 0;
    std::uint64_t rawFieldBytes = 0;
    std::uint64_t wholeZstdBytes = 0;
    std::vector<std::size_t> payloadSizes;
    std::vector<std::size_t> outerSizes;
    payloadSizes.reserve(samples.size());
    outerSizes.reserve(samples.size());
    for (const auto& sample : samples) {
        records += sample.records;
        payloadBytes += sample.payloadBytes;
        outerBytes += sample.outerZstdBytes;
        rawFieldBytes += sample.rawFieldBytes;
        wholeZstdBytes += sample.wholeZstdBytes;
        payloadSizes.push_back(sample.payloadBytes);
        outerSizes.push_back(sample.outerZstdBytes);
    }
    std::sort(payloadSizes.begin(), payloadSizes.end());
    std::sort(outerSizes.begin(), outerSizes.end());
    const auto percentile = [](const auto& values, double fraction) {
        const auto index = static_cast<std::size_t>(std::ceil(
            fraction * static_cast<double>(values.size()))) - 1;
        return values[std::min(index, values.size() - 1)];
    };
    std::cout << std::setprecision(17)
              << "fiberlet_storage_compression"
              << " profile=compact_axis_cost_sqrt_u16_max256"
              << " chunk_side_base=" << chunkSideBaseVoxels
              << " internal_codec=field_zstd3"
              << " payload=" << payload
              << " chunks=" << samples.size()
              << " records=" << records
              << " payload_bytes=" << payloadBytes
              << " outer_zstd_bytes=" << outerBytes
              << " outer_to_payload_ratio="
              << static_cast<double>(outerBytes) /
                     static_cast<double>(payloadBytes)
              << " raw_field_bytes=" << rawFieldBytes
              << " whole_zstd_bytes=" << wholeZstdBytes
              << " whole_zstd_to_payload_ratio="
              << static_cast<double>(wholeZstdBytes) /
                     static_cast<double>(payloadBytes)
              << " payload_mean_bytes="
              << static_cast<double>(payloadBytes) /
                     static_cast<double>(samples.size())
              << " payload_p50_bytes=" << percentile(payloadSizes, 0.50)
              << " payload_p95_bytes=" << percentile(payloadSizes, 0.95)
              << " payload_max_bytes=" << payloadSizes.back()
              << " outer_mean_bytes="
              << static_cast<double>(outerBytes) /
                     static_cast<double>(samples.size())
              << " outer_p50_bytes=" << percentile(outerSizes, 0.50)
              << " outer_p95_bytes=" << percentile(outerSizes, 0.95)
              << " outer_max_bytes=" << outerSizes.back() << '\n';
}

void benchmarkReplayStorageCompression(
    const std::shared_ptr<vc::fiber_tracer::FiberletOnDemandPreprocessor>&
        preprocessor,
    std::span<const vc::fiber_tracer::FiberletScheduledChunk> schedule,
    std::size_t requestedChunks,
    std::uint64_t seed,
    int chunkSideBaseVoxels,
    const std::set<std::array<int, 3>>& reportOwners = {})
{
    using namespace vc::fiber_tracer;
    if (!preprocessor || requestedChunks == 0)
        return;

    std::vector<FiberletScheduledChunk> randomized(schedule.begin(), schedule.end());
    std::mt19937_64 random(seed);
    std::shuffle(randomized.begin(), randomized.end(), random);

    struct StoredPair {
        FiberletStoredPrefix prefix;
        FiberletStoredRoute route;
        std::array<int, 3> sourceOwner{};
    };
    struct FloatStorageSize {
        std::size_t rawBytes = 0;
        std::size_t zstdBytes = 0;
    };
    std::map<FiberletStorageKey, FiberletStoredAnchor> sourceAnchors;
    std::vector<StoredPair> sourcePairs;
    std::map<std::array<int, 3>, FloatStorageSize> floatByOwner;
    std::size_t sourceChunks = 0;
    for (const auto& scheduled : randomized) {
        const std::array<int, 3> sourceOwner{
            scheduled.key.iz, scheduled.key.iy, scheduled.key.ix};
        const auto prefixChunk = preprocessor->fiberletCache()->getChunkBlocking(
            0, scheduled.key.iz, scheduled.key.iy, scheduled.key.ix);
        const auto prefixPayload = std::dynamic_pointer_cast<
            const FiberletPrefixChunkPayload>(prefixChunk.payload);
        if (prefixChunk.status != vc::render::ChunkStatus::Data ||
            !prefixPayload || prefixPayload->prefixes.empty()) {
            continue;
        }
        const auto routeChunk = preprocessor->fiberletCache()->getChunkBlocking(
            1, scheduled.key.iz, scheduled.key.iy, scheduled.key.ix);
        const auto routePayload = std::dynamic_pointer_cast<
            const FiberletRouteChunkPayload>(routeChunk.payload);
        if (routeChunk.status != vc::render::ChunkStatus::Data || !routePayload ||
            routePayload->routes.size() != prefixPayload->prefixes.size()) {
            throw std::runtime_error(
                "storage compression benchmark could not load a complete fiberlet chunk");
        }
        for (std::size_t index = 0; index < prefixPayload->prefixes.size(); ++index) {
            sourcePairs.push_back(
                {prefixPayload->prefixes[index], routePayload->routes[index],
                    sourceOwner});
        }
        for (const auto& dependency :
             preprocessor->anchorDependencies(scheduled.key)) {
            const auto anchorChunk = preprocessor->anchorCache()->getChunkBlocking(
                dependency.level, dependency.iz, dependency.iy, dependency.ix);
            const auto anchorPayload = std::dynamic_pointer_cast<
                const FiberletAnchorChunkPayload>(anchorChunk.payload);
            if (anchorChunk.status != vc::render::ChunkStatus::Data ||
                !anchorPayload) {
                throw std::runtime_error(
                    "storage compression benchmark could not load an anchor dependency");
            }
            for (const auto& anchor : anchorPayload->anchors) {
                const auto [found, inserted] =
                    sourceAnchors.emplace(anchor.key, anchor);
                if (!inserted &&
                    (found->second.positionPredictionXYZ !=
                         anchor.positionPredictionXYZ ||
                     found->second.fittedAxisXYZ != anchor.fittedAxisXYZ)) {
                    throw std::invalid_argument(
                        "storage compression benchmark found conflicting anchor records");
                }
            }
        }
        const auto ownedAnchorChunk =
            preprocessor->anchorCache()->getChunkBlocking(
                0, scheduled.key.iz, scheduled.key.iy, scheduled.key.ix);
        const auto ownedAnchorPayload = std::dynamic_pointer_cast<
            const FiberletAnchorChunkPayload>(ownedAnchorChunk.payload);
        if (ownedAnchorChunk.status != vc::render::ChunkStatus::Data ||
            !ownedAnchorPayload) {
            throw std::runtime_error(
                "storage compression benchmark could not load the owned anchor chunk");
        }
        const auto floatAnchors = serializeFiberletAnchors(
            ownedAnchorPayload->config, ownedAnchorPayload->anchors);
        const auto floatPrefixes = serializeFiberletPrefixes(
            prefixPayload->config, prefixPayload->prefixes);
        const auto floatRoutes = serializeFiberletRoutes(
            routePayload->config, routePayload->routes);
        auto& floatSize = floatByOwner[sourceOwner];
        floatSize.zstdBytes = floatAnchors.size() + floatPrefixes.size() +
            floatRoutes.size();
        floatSize.rawBytes =
            materializeFiberletPayload(floatAnchors).size() +
            materializeFiberletPayload(floatPrefixes).size() +
            materializeFiberletPayload(floatRoutes).size();
        if (++sourceChunks == requestedChunks)
            break;
    }
    if (sourceChunks == 0)
        throw std::runtime_error(
            "storage compression benchmark found no nonempty scheduled chunks");

    struct QuantizedAnchorRecord {
        FiberletStorageKey oldKey;
        FiberletStoredAnchor anchor;
        std::array<int, 3> sourceOwner{};
    };
    std::vector<QuantizedAnchorRecord> quantizedAnchors;
    const auto& metadata = preprocessor->anchorDataset()->metadata();
    for (const auto& [oldKey, source] : sourceAnchors) {
        QuantizedAnchorRecord record;
        record.oldKey = oldKey;
        record.anchor = source;
        record.sourceOwner = storageOwnerChunk(
            oldKey, metadata.coordinateUnitsPerChunkZYX);
        quantizedAnchors.push_back(std::move(record));
    }

    std::map<FiberletStorageKey, FiberletStorageKey> quantizedKeyByOldKey;
    for (const auto& record : quantizedAnchors)
        quantizedKeyByOldKey.emplace(record.oldKey, record.anchor.key);
    using AnchorGroups = std::map<std::array<int, 3>,
        std::vector<FiberletStoredAnchor>>;
    std::map<std::array<int, 3>, AnchorGroups> anchorGroupsBySource;
    for (auto& record : quantizedAnchors) {
        if (!reportOwners.contains(record.sourceOwner))
            continue;
        anchorGroupsBySource[record.sourceOwner]
            [storageOwnerChunk(
                record.anchor.key, metadata.coordinateUnitsPerChunkZYX)]
            .push_back(std::move(record.anchor));
    }
    for (auto& [sourceOwner, groups] : anchorGroupsBySource) {
        (void)sourceOwner;
        for (auto& [compactOwner, anchors] : groups) {
            (void)compactOwner;
            std::sort(anchors.begin(), anchors.end(), [](const auto& left,
                                                         const auto& right) {
                return left.key < right.key;
            });
        }
    }

    using FiberletGroups =
        std::map<std::array<int, 3>, std::vector<StoredPair>>;
    std::map<std::array<int, 3>, FiberletGroups> fiberletGroupsBySource;
    for (auto& pair : sourcePairs) {
        const auto first = quantizedKeyByOldKey.find(pair.prefix.id.first);
        const auto second = quantizedKeyByOldKey.find(pair.prefix.id.second);
        if (first == quantizedKeyByOldKey.end() ||
            second == quantizedKeyByOldKey.end()) {
            throw std::logic_error(
                "storage compression benchmark is missing a fiberlet endpoint");
        }
        pair.prefix.id = {first->second, second->second};
        if (pair.prefix.id.second < pair.prefix.id.first) {
            std::swap(pair.prefix.id.first, pair.prefix.id.second);
            std::swap(pair.prefix.entryUV, pair.prefix.exitUV);
            std::reverse(pair.route.middleUV.begin(), pair.route.middleUV.end());
            const cv::Vec3f firstStep = pair.prefix.firstStepBaseXYZ;
            pair.prefix.firstStepBaseXYZ = -pair.prefix.lastStepBaseXYZ;
            pair.prefix.lastStepBaseXYZ = -firstStep;
        }
        fiberletGroupsBySource[pair.sourceOwner]
            [storageOwnerChunk(
                pair.prefix.id.first, metadata.coordinateUnitsPerChunkZYX)]
            .push_back(std::move(pair));
    }
    for (auto& [sourceOwner, groups] : fiberletGroupsBySource) {
        (void)sourceOwner;
        for (auto& [compactOwner, pairs] : groups) {
            (void)compactOwner;
            std::sort(pairs.begin(), pairs.end(), [](const auto& left,
                                                     const auto& right) {
                return left.prefix.id < right.prefix.id;
            });
        }
    }

    std::vector<StorageCompressionSample> anchorSamples;
    std::map<std::array<int, 3>, StorageCompressionSample> anchorByOwner;
    for (const auto& sourceOwner : reportOwners) {
        StorageCompressionSample total;
        for (const auto& [compactOwner, anchors] :
            anchorGroupsBySource[sourceOwner]) {
            const auto codec = compactCodec(metadata, compactOwner);
            const auto bytes = serializeFiberletAnchors(codec, anchors);
            const auto sample = storageCompressionSample(anchors.size(), bytes);
            total.records += sample.records;
            total.payloadBytes += sample.payloadBytes;
            total.outerZstdBytes += sample.outerZstdBytes;
            total.rawFieldBytes += sample.rawFieldBytes;
            total.wholeZstdBytes += sample.wholeZstdBytes;
            anchorSamples.push_back(sample);
        }
        anchorByOwner.emplace(sourceOwner, total);
    }

    std::vector<StorageCompressionSample> prefixSamples;
    std::vector<StorageCompressionSample> routeSamples;
    std::map<std::array<int, 3>, StorageCompressionSample> prefixByOwner;
    std::map<std::array<int, 3>, StorageCompressionSample> routeByOwner;
    for (const auto& sourceOwner : reportOwners) {
        StorageCompressionSample prefixTotal;
        StorageCompressionSample routeTotal;
        for (const auto& [compactOwner, pairs] :
            fiberletGroupsBySource[sourceOwner]) {
            std::vector<FiberletStoredPrefix> prefixes;
            std::vector<FiberletStoredRoute> routes;
            prefixes.reserve(pairs.size());
            routes.reserve(pairs.size());
            for (const auto& pair : pairs) {
                prefixes.push_back(pair.prefix);
                routes.push_back(pair.route);
            }
            const auto codec = compactCodec(metadata, compactOwner);
            const auto prefixBytes = serializeFiberletPrefixes(codec, prefixes);
            const auto routeBytes = serializeFiberletRoutes(codec, routes);
            const auto prefixSample = storageCompressionSample(
                prefixes.size(), prefixBytes);
            const auto routeSample = storageCompressionSample(
                routes.size(), routeBytes);
            prefixTotal.records += prefixSample.records;
            prefixTotal.payloadBytes += prefixSample.payloadBytes;
            prefixTotal.outerZstdBytes += prefixSample.outerZstdBytes;
            prefixTotal.rawFieldBytes += prefixSample.rawFieldBytes;
            prefixTotal.wholeZstdBytes += prefixSample.wholeZstdBytes;
            routeTotal.records += routeSample.records;
            routeTotal.payloadBytes += routeSample.payloadBytes;
            routeTotal.outerZstdBytes += routeSample.outerZstdBytes;
            routeTotal.rawFieldBytes += routeSample.rawFieldBytes;
            routeTotal.wholeZstdBytes += routeSample.wholeZstdBytes;
            prefixSamples.push_back(prefixSample);
            routeSamples.push_back(routeSample);
        }
        prefixByOwner.emplace(sourceOwner, prefixTotal);
        routeByOwner.emplace(sourceOwner, routeTotal);
    }

    std::cout << "fiberlet_storage_compression_sample"
              << " profile=compact_axis_cost_sqrt_u16_max256"
              << " chunk_side_base=" << chunkSideBaseVoxels
              << " seed=" << seed
              << " scheduled_chunks=" << schedule.size()
              << " sampled_nonempty_source_chunks=" << sourceChunks
              << " source_anchors=" << sourceAnchors.size()
              << " source_fiberlets=" << sourcePairs.size() << '\n';
    if (!reportOwners.empty()) {
        for (const auto& owner : reportOwners) {
            const auto anchor = anchorByOwner.find(owner);
            const auto prefix = prefixByOwner.find(owner);
            const auto route = routeByOwner.find(owner);
            const StorageCompressionSample empty;
            const auto& anchorSample =
                anchor == anchorByOwner.end() ? empty : anchor->second;
            const auto& prefixSample =
                prefix == prefixByOwner.end() ? empty : prefix->second;
            const auto& routeSample =
                route == routeByOwner.end() ? empty : route->second;
            const std::size_t payloadBytes = anchorSample.payloadBytes +
                prefixSample.payloadBytes + routeSample.payloadBytes;
            const std::size_t rawBytes = anchorSample.rawFieldBytes +
                prefixSample.rawFieldBytes + routeSample.rawFieldBytes;
            const auto floatFound = floatByOwner.find(owner);
            const FloatStorageSize emptyFloat;
            const auto& floatSize = floatFound == floatByOwner.end()
                ? emptyFloat
                : floatFound->second;
            std::cout << "fiberlet_storage_chunk"
                      << " profile=compact_axis_cost_sqrt_u16_max256"
                      << " chunk_side_base=" << chunkSideBaseVoxels
                      << " owner=" << owner[0] << '/' << owner[1] << '/'
                      << owner[2]
                      << " anchors=" << anchorSample.records
                      << " fiberlets=" << prefixSample.records
                      << " float_raw_bytes=" << floatSize.rawBytes
                      << " float_zstd_bytes=" << floatSize.zstdBytes
                      << " raw_bytes=" << rawBytes
                      << " zstd_bytes=" << payloadBytes
                      << " zstd_to_raw_ratio="
                      << (rawBytes == 0 ? 0.0
                                        : static_cast<double>(payloadBytes) /
                                              static_cast<double>(rawBytes))
                      << '\n';
        }
    }
    std::vector<StorageCompressionSample> allSamples = anchorSamples;
    allSamples.insert(
        allSamples.end(), prefixSamples.begin(), prefixSamples.end());
    allSamples.insert(
        allSamples.end(), routeSamples.begin(), routeSamples.end());
    printStorageCompressionSummary("anchors", std::move(anchorSamples),
        chunkSideBaseVoxels);
    printStorageCompressionSummary("prefix", std::move(prefixSamples),
        chunkSideBaseVoxels);
    printStorageCompressionSummary("routes", std::move(routeSamples),
        chunkSideBaseVoxels);
    printStorageCompressionSummary("all", std::move(allSamples),
        chunkSideBaseVoxels);
}

void benchmarkFullRegionStorageCompression(
    const vc::fiber_tracer::FiberPredictionField& field,
    const std::shared_ptr<const vc::lasagna::NormalSampler>& normalSampler,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid,
    const CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset,
    const std::shared_ptr<vc::fiber_tracer::FiberletOnDemandPreprocessor>&
        replayPreprocessor,
    std::span<const vc::fiber_tracer::FiberletScheduledChunk> replaySchedule)
{
    using namespace vc::fiber_tracer;
    if (!replayPreprocessor || options.storageCompressionChunks == 0)
        return;

    const int targetSide = options.storageCompressionChunkSideBaseVoxels;
    const int sourceSide = static_cast<int>(
        replayPreprocessor->fiberletDataset()->metadata()
            .spatialChunkSideBaseVoxels);
    std::set<std::array<int, 3>> candidateOwners;
    for (const auto& scheduled : replaySchedule) {
        std::array<int, 3> owner{};
        const std::array<int, 3> source{
            scheduled.key.iz, scheduled.key.iy, scheduled.key.ix};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            owner[axis] = static_cast<int>(
                static_cast<std::int64_t>(source[axis]) * sourceSide /
                targetSide);
        }
        candidateOwners.insert(owner);
    }
    std::vector<std::array<int, 3>> randomizedOwners(
        candidateOwners.begin(), candidateOwners.end());
    std::mt19937_64 random(options.storageCompressionSeed);
    std::shuffle(randomizedOwners.begin(), randomizedOwners.end(), random);
    if (randomizedOwners.size() > options.storageCompressionChunks)
        randomizedOwners.resize(options.storageCompressionChunks);
    std::sort(randomizedOwners.begin(), randomizedOwners.end());
    const std::set<std::array<int, 3>> reportOwners(
        randomizedOwners.begin(), randomizedOwners.end());
    if (reportOwners.empty())
        throw std::runtime_error(
            "storage compression benchmark found no target regions");

    CliOptions fullOptions = options;
    fullOptions.storageChunkSideBaseVoxels = targetSide;
    auto anchorMetadata = replayDatasetMetadata(
        FiberletDatasetKind::Anchors, grid, fullOptions, fiberDataset,
        normalDataset, {}, 0.0);
    auto fiberletMetadata = anchorMetadata;
    fiberletMetadata.kind = FiberletDatasetKind::Fiberlets;

    const auto cellSide = static_cast<std::size_t>(
        options.anchors.cellSizePredictionVoxels);
    const std::array<std::size_t, 3> cellShape{
        (grid.shapeZYX[0] + cellSide - 1) / cellSide,
        (grid.shapeZYX[1] + cellSide - 1) / cellSide,
        (grid.shapeZYX[2] + cellSide - 1) / cellSide};
    std::array<int, 3> minimumOffset{0, 0, 0};
    std::array<int, 3> maximumOffset{0, 0, 0};
    for (const auto& offset : fiberletCellNeighborhoodOffsets(
             options.paths.cellRadius,
             options.paths.neighborhoodMarginCells)) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            minimumOffset[axis] =
                std::min(minimumOffset[axis], offset[axis]);
            maximumOffset[axis] =
                std::max(maximumOffset[axis], offset[axis]);
        }
    }
    const auto unitsPerChunk =
        anchorMetadata.coordinateUnitsPerChunkZYX[0];
    std::set<std::array<std::size_t, 3>> selectedCells;
    for (const auto& owner : reportOwners) {
        std::array<std::int64_t, 3> begin{};
        std::array<std::int64_t, 3> end{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            begin[axis] = std::max<std::int64_t>(0,
                static_cast<std::int64_t>(owner[axis]) * unitsPerChunk +
                    minimumOffset[axis]);
            end[axis] = std::min<std::int64_t>(cellShape[axis],
                (static_cast<std::int64_t>(owner[axis]) + 1) *
                        unitsPerChunk +
                    maximumOffset[axis]);
        }
        for (std::int64_t z = begin[0]; z < end[0]; ++z) {
            for (std::int64_t y = begin[1]; y < end[1]; ++y) {
                for (std::int64_t x = begin[2]; x < end[2]; ++x) {
                    selectedCells.insert({static_cast<std::size_t>(z),
                        static_cast<std::size_t>(y),
                        static_cast<std::size_t>(x)});
                }
            }
        }
    }

    std::ostringstream ownerNamespace;
    ownerNamespace << "full-region-v1-side-" << targetSide << "-seed-"
                   << options.storageCompressionSeed;
    for (const auto& owner : reportOwners)
        ownerNamespace << '-' << owner[0] << '_' << owner[1] << '_'
                       << owner[2];
    const auto cacheRoot = options.outputDirectory /
        "storage-compression-cache" / ownerNamespace.str();
    auto budget = std::make_shared<vc::render::DecodedChunkCacheBudget>(
        options.decodedCacheBytes);
    vc::render::ChunkCache::Options anchorCacheOptions;
    anchorCacheOptions.decodedByteCapacity = options.decodedCacheBytes;
    anchorCacheOptions.decodedByteBudget = budget;
    anchorCacheOptions.maxConcurrentReads = 1;
    FiberletOnDemandConfig onDemand;
    onDemand.anchorRoot = cacheRoot / "anchors.zarr";
    onDemand.fiberletRoot = cacheRoot / "fiberlets.zarr";
    onDemand.anchorMetadata = anchorMetadata;
    onDemand.fiberletMetadata = fiberletMetadata;
    onDemand.grid = grid;
    onDemand.anchorConfig = options.anchors;
    onDemand.pathConfig = options.paths;
    onDemand.predictionSampler =
        [&](const auto& indices, int threads, auto& samples) {
            field.sampleStoredGridBatch(indices, threads, samples);
        };
    onDemand.normalSampler = normalSampler;
    onDemand.selectedAnchorCellsZYX.assign(
        selectedCells.begin(), selectedCells.end());
    onDemand.anchorRetainPredicate = [](const FiberAnchor&) {
        return FiberAnchorRetainEvaluation{true, {}, {}};
    };
    onDemand.pointPredicate = [](const cv::Vec3d&) { return true; };
    onDemand.anchorCacheOptions = anchorCacheOptions;
    onDemand.fiberletCacheOptions = anchorCacheOptions;
    onDemand.progress = [](const FiberletOnDemandProgress& progress) {
        if (progress.status == "completed") {
            std::cerr << "fiberlet_storage_region"
                      << " stage=" << progress.stage
                      << " key=" << progress.key.iz << '/'
                      << progress.key.iy << '/' << progress.key.ix
                      << " inputs=" << progress.inputCount
                      << " outputs=" << progress.outputCount
                      << " elapsed_seconds=" << progress.elapsedSeconds
                      << '\n';
        }
    };
    auto fullPreprocessor = FiberletOnDemandPreprocessor::create(
        std::move(onDemand));
    std::vector<FiberletScheduledChunk> targetSchedule;
    targetSchedule.reserve(reportOwners.size());
    for (const auto& owner : reportOwners) {
        targetSchedule.push_back({
            {0, owner[0], owner[1], owner[2]}, 0.0, 0.0});
    }
    std::cout << "fiberlet_storage_full_region"
              << " chunk_side_base=" << targetSide
              << " chunks=" << targetSchedule.size()
              << " selected_cells_with_halo=" << selectedCells.size()
              << '\n';
    benchmarkReplayStorageCompression(fullPreprocessor, targetSchedule,
        targetSchedule.size(), options.storageCompressionSeed, targetSide,
        reportOwners);
}

struct CachedReplayContext {
    CachedReplayContext() = default;
    CachedReplayContext(const CachedReplayContext&) = delete;
    CachedReplayContext& operator=(const CachedReplayContext&) = delete;
    CachedReplayContext(CachedReplayContext&&) = default;
    CachedReplayContext& operator=(CachedReplayContext&&) = delete;
    ~CachedReplayContext()
    {
        graph.reset();
        if (preprocessor)
            preprocessor->shutdown();
    }

    std::shared_ptr<vc::fiber_tracer::FiberletOnDemandPreprocessor> preprocessor;
    std::unique_ptr<vc::fiber_tracer::FiberletCachedReplayGraphSource> graph;
    std::shared_ptr<ReplayPreprocessingProgress> preprocessingProgress;
    std::vector<vc::fiber_tracer::FiberletScheduledChunk> schedule;
    std::filesystem::path anchorRoot;
    std::filesystem::path fiberletRoot;
};

CachedReplayContext createCachedReplayContext(
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid,
    const CliOptions& options,
    const vc::fiber_tracer::FiberPredictionField& field,
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler,
    const std::vector<cv::Vec3d>& referenceBase,
    const vc::fiber_tracer::PolylineArcGeometry& reference,
    double processingBeginArcBase,
    double processingEndArcBase,
    double scheduleBeginArcBase,
    double scheduleEndArcBase,
    bool prefetchSchedule,
    const std::filesystem::path& cacheBaseRoot,
    const vc::fiber_tracer::FiberletGeometryCacheProfile& cacheProfile,
    const vc::fiber_tracer::FiberletEvaluationQuantization& replayQuantization,
    const std::filesystem::path& anchorRootOverride,
    const std::filesystem::path& fiberletRootOverride,
    ReplayOverallProgress& overallProgress)
{
    const auto processingTube = vc::fiber_tracer::makeFiberReplayTube(
        referenceBase,
        0.5 * (processingBeginArcBase + processingEndArcBase),
        0.5 * (processingEndArcBase - processingBeginArcBase),
        options.radiusBaseVoxels,
        grid,
        options.anchors.cellSizePredictionVoxels,
        false);
    const auto containmentQuery = processingTube.makePredictionContainmentQuery(grid.predictionToBaseScale);
    auto anchorMetadata = replayDatasetMetadata(
        vc::fiber_tracer::FiberletDatasetKind::Anchors,
        grid,
        options,
        fiberDataset,
        normalDataset,
        processingTube.referenceIntervalBase,
        processingTube.radiusBaseVoxels);
    auto fiberletMetadata = replayDatasetMetadata(
        vc::fiber_tracer::FiberletDatasetKind::Fiberlets,
        grid,
        options,
        fiberDataset,
        normalDataset,
        processingTube.referenceIntervalBase,
        processingTube.radiusBaseVoxels,
        cacheProfile);
    auto anchorNamespace = anchorMetadata.algorithmFingerprint;
    auto fiberletNamespace = fiberletMetadata.algorithmFingerprint;
    std::replace(anchorNamespace.begin(), anchorNamespace.end(), ':', '-');
    std::replace(fiberletNamespace.begin(), fiberletNamespace.end(), ':', '-');
    const auto anchorCacheRoot = cacheBaseRoot / anchorNamespace;
    const auto fiberletCacheRoot = cacheBaseRoot / fiberletNamespace;

    const auto evaluationAnchorCacheBytes = std::max<std::size_t>(1, options.decodedCacheBytes / 8);
    const auto decodedChunkCacheBytes = options.decodedCacheBytes > evaluationAnchorCacheBytes ? options.decodedCacheBytes - evaluationAnchorCacheBytes
                                                                                               : options.decodedCacheBytes;
    auto graphBudget = std::make_shared<vc::render::DecodedChunkCacheBudget>(decodedChunkCacheBytes);
    vc::render::ChunkCache::Options cacheOptions;
    cacheOptions.decodedByteCapacity = options.decodedCacheBytes;
    cacheOptions.decodedByteBudget = graphBudget;
    cacheOptions.maxConcurrentReads = 1;

    CachedReplayContext result;
    result.anchorRoot = anchorRootOverride.empty() ? anchorCacheRoot / "anchors.zarr" : anchorRootOverride;
    result.fiberletRoot = fiberletRootOverride.empty() ? fiberletCacheRoot / "fiberlets.zarr" : fiberletRootOverride;
    result.preprocessingProgress = std::make_shared<ReplayPreprocessingProgress>();
    vc::fiber_tracer::FiberletOnDemandConfig onDemand;
    onDemand.anchorRoot = result.anchorRoot;
    onDemand.fiberletRoot = result.fiberletRoot;
    onDemand.anchorMetadata = std::move(anchorMetadata);
    onDemand.fiberletMetadata = std::move(fiberletMetadata);
    onDemand.grid = grid;
    onDemand.anchorConfig = options.anchors;
    onDemand.pathConfig = options.paths;
    onDemand.geometryQuantization = cacheProfile.geometry;
    onDemand.evaluationAnchorCacheBytes = evaluationAnchorCacheBytes;
    onDemand.predictionSampler = [&field](const auto& indices, int threads, auto& samples) {
        field.sampleStoredGridBatch(indices, threads, samples);
    };
    onDemand.normalSampler = std::move(normalSampler);
    onDemand.anchorCellPredicate = [containmentQuery, grid, cellSize = options.anchors.cellSizePredictionVoxels](const std::array<size_t, 3>& cell) {
        return containmentQuery.intersectsPredictionCell(cell, grid, cellSize);
    };
    onDemand.anchorRetainPredicate = [containmentQuery](const vc::fiber_tracer::FiberAnchor& anchor) {
        return containmentQuery.evaluatePredictionAnchor(anchor);
    };
    onDemand.pointPredicate = [containmentQuery](const cv::Vec3d& pointPrediction) {
        return containmentQuery.containsPredictionPoint(pointPrediction);
    };
    onDemand.anchorCacheOptions = cacheOptions;
    onDemand.fiberletCacheOptions = std::move(cacheOptions);
    onDemand.chunkResolved =
        [progress = result.preprocessingProgress](vc::fiber_tracer::FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status) {
            progress->resolve(kind, key, status);
        };
    result.preprocessor = vc::fiber_tracer::FiberletOnDemandPreprocessor::create(std::move(onDemand));
    result.schedule = result.preprocessor->referenceChunkSchedule(reference, scheduleBeginArcBase, scheduleEndArcBase, options.radiusBaseVoxels);
    std::set<ReplayChunkId> expectedAnchors;
    std::set<ReplayChunkId> expectedPrefixes;
    for (const auto& scheduled : result.schedule) {
        expectedPrefixes.insert({scheduled.key.level, scheduled.key.iz, scheduled.key.iy, scheduled.key.ix});
        for (const auto& dependency : result.preprocessor->anchorDependencies(scheduled.key)) {
            expectedAnchors.insert({dependency.level, dependency.iz, dependency.iy, dependency.ix});
        }
    }
    result.preprocessingProgress->configure(std::move(expectedAnchors), std::move(expectedPrefixes));
    overallProgress.attachPreprocessing(result.preprocessingProgress);
    result.graph = std::make_unique<vc::fiber_tracer::FiberletCachedReplayGraphSource>(result.preprocessor, options.paths, replayQuantization);
    if (prefetchSchedule) {
        result.preprocessor->prefetchScheduled(result.schedule, 0, result.schedule.size(), false);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        CliOptions options = parseArgs(argc, argv);
        std::shared_ptr<Volume> replayCtVolume;
        std::string replayCtLocator;
        if (options.writeReplayVisualizations) {
            replayCtLocator = std::filesystem::absolute(options.volumeZarr).lexically_normal().string();
            replayCtVolume = Volume::New(options.volumeZarr);
            replayCtVolume->setIOThreads(options.paths.parallelThreads);
            replayCtVolume->setCacheBudget(options.decodedCacheBytes);
            (void)vc::fiber_tracer::validateFiberReplayStripCtVolume(*replayCtVolume, replayCtLocator);
        }
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(options.manifestLocation, openOptions);
        const vc::fiber_tracer::FiberPredictionField field(dataset, options.decodedCacheBytes, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
        const auto grid = field.storedGridInfo();

        if (options.command == Command::AnchorBenchmark) {
            const double cellSideBase = resolveAnchorConfig(options, grid);
            const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
            const auto cells =
                vc::fiber_tracer::fiberAnchorCellsNearPolyline(fiber.linePointsXyzBase, 0.0, grid, options.anchors.cellSizePredictionVoxels);
            std::cerr << "anchor_benchmark_stage stage=anchors status=started"
                      << " cells=" << cells.size() << '\n';
            const auto anchors = vc::fiber_tracer::extractRefinedFiberAnchorsForCells(
                grid,
                options.anchors,
                [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
                cells,
                printAnchorProgress);
            std::cerr << "anchor_benchmark_stage stage=anchors status=completed"
                      << " elapsed_seconds=" << anchors.elapsedSeconds << '\n';
            const auto benchmark = vc::fiber_tracer::benchmarkRefinedFiberAnchors(anchors, fiber.linePointsXyzBase, {4.0, 8.0});
            std::cout << std::setprecision(17) << "anchor_benchmark_extraction"
                      << " prediction_to_base=" << grid.predictionToBaseScale << " cell_side_base_voxels=" << cellSideBase
                      << " extraction_seconds=" << anchors.elapsedSeconds << '\n';
            const auto printOptional = [](const std::optional<double>& value) {
                if (value.has_value())
                    std::cout << *value;
                else
                    std::cout << "n/a";
            };
            const auto printStage = [&](const char* name, const auto& stage) {
                std::cout << "anchor_benchmark_population"
                          << " stage=" << name << " reference_cells=" << stage.referenceCells
                          << " cells_with_refined_anchors=" << stage.cellsWithRefinedAnchors << " refined_anchors=" << stage.refinedAnchors << '\n';
                const auto& distances = stage.anchorDistancesBaseVoxels;
                std::cout << "anchor_benchmark_distance_base_voxels"
                          << " stage=" << name << " count=" << distances.count << " min=";
                printOptional(distances.minimum);
                std::cout << " mean=";
                printOptional(distances.mean);
                std::cout << " median=";
                printOptional(distances.median);
                std::cout << " p95=";
                printOptional(distances.percentile95);
                std::cout << " max=";
                printOptional(distances.maximum);
                std::cout << '\n';
                for (const auto& threshold : stage.thresholds) {
                    std::cout << "anchor_benchmark_threshold"
                              << " stage=" << name << " threshold_base_voxels=" << threshold.thresholdBaseVoxels
                              << " anchor_hits=" << threshold.anchorHits << " anchor_total=" << stage.refinedAnchors << " anchor_hit_rate=";
                    printOptional(threshold.anchorHitRate);
                    std::cout << " cell_hits=" << threshold.cellHits << " cell_total=" << stage.referenceCells
                              << " cell_hit_rate=" << threshold.cellHitRate << '\n';
                }
            };
            printStage("discrete", benchmark.discrete);
            printStage("separable_1d", benchmark.separable1d);
            printStage("joint_2d", benchmark.joint2d);
            return 0;
        }

        if (options.command == Command::Benchmark) {
            resolveAnchorConfig(options, grid);
            if (options.corridorRadiusBaseVoxels.has_value()) {
                options.paths.corridorRadiusPredictionVoxels = *options.corridorRadiusBaseVoxels / grid.predictionToBaseScale;
                vc::fiber_tracer::validateFiberletPathConfig(options.paths);
            }
            const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
            if (fiber.controlPointLineIndices.empty() || fiber.controlPointLineIndices.front() >= fiber.linePointsXyzBase.size())
                fail("benchmark fiber has no valid first control point");
            const auto reference = vc::fiber_tracer::makePolylineArcGeometry(fiber.linePointsXyzBase);
            const auto interval = vc::fiber_tracer::selectForwardPolylineArcInterval(
                reference, fiber.controlPointLineIndices.front(), options.alongSpecified ? std::optional<double>{options.alongBaseVoxels} : std::nullopt);
            const double beginArc = interval.beginArc;
            const double endArc = interval.endArc;

            vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
            normalOptions.workingToBaseScale = grid.predictionToBaseScale;
            normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, normalOptions);
            const vc::lasagna::LasagnaNormalSampler normalSampler(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});
            const auto totalStart = std::chrono::steady_clock::now();
            const double totalCpuStart = processCpuSeconds();
            const auto extraction =
                extractTubeFiberlets(fiber.linePointsXyzBase, beginArc, endArc, options.radiusBaseVoxels, grid, options, field, normalSampler);
            const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - totalStart).count();
            const double totalCpuSeconds = processCpuSeconds() - totalCpuStart;
            const size_t anchors = extraction.anchors.diagnostics.oneAnchorCells + 2 * extraction.anchors.diagnostics.twoAnchorCells;
            const double cellRate =
                extraction.anchorSeconds > 0.0 ? static_cast<double>(extraction.tube.cellsZYX.size()) / extraction.anchorSeconds : 0.0;
            const double candidateRate = extraction.fiberletSeconds > 0.0
                                             ? static_cast<double>(extraction.paths.diagnostics.searchedPairs) / extraction.fiberletSeconds
                                             : 0.0;
            const double nodeRate =
                extraction.fiberletSeconds > 0.0 ? static_cast<double>(extraction.paths.evaluatedDpNodes) / extraction.fiberletSeconds : 0.0;
            const double meanBatchVoxels =
                extraction.paths.samplingCoordinateBatches > 0
                    ? static_cast<double>(extraction.paths.sampledVoxels) / static_cast<double>(extraction.paths.samplingCoordinateBatches)
                    : 0.0;
            std::cout << std::setprecision(17) << "fiberlet_extraction_benchmark"
                      << " threads=" << options.paths.parallelThreads << " reference_begin_arc_base=" << beginArc
                      << " reference_end_arc_base=" << endArc << " reference_length_base=" << endArc - beginArc
                      << " radius_base=" << options.radiusBaseVoxels << " cells=" << extraction.tube.cellsZYX.size() << " anchors=" << anchors
                      << " anchor_seconds=" << extraction.anchorSeconds << " anchor_cpu_seconds=" << extraction.anchorCpuSeconds
                      << " anchor_effective_cores=" << effectiveCores(extraction.anchorCpuSeconds, extraction.anchorSeconds)
                      << " anchor_cells_per_second=" << cellRate << " generated_pairs=" << extraction.paths.diagnostics.generatedPairs
                      << " searched=" << extraction.paths.diagnostics.searchedPairs << " successful=" << extraction.paths.diagnostics.successfulPaths
                      << " fiberlet_seconds=" << extraction.fiberletSeconds << " fiberlet_cpu_seconds=" << extraction.fiberletCpuSeconds
                      << " fiberlet_effective_cores=" << effectiveCores(extraction.fiberletCpuSeconds, extraction.fiberletSeconds)
                      << " searched_per_second=" << candidateRate << " sampling_coordinate_batches=" << extraction.paths.samplingCoordinateBatches
                      << " sampling_batch_coordinates=" << options.paths.samplingBatchCoordinates << " sampled_voxels=" << extraction.paths.sampledVoxels
                      << " mean_batch_voxels=" << meanBatchVoxels << " peak_batch_voxels=" << extraction.paths.peakCoordinateBatchVoxels
                      << " prepared_geometry_bytes=" << extraction.paths.preparedGeometryBytes
                      << " peak_search_transient_bytes=" << extraction.paths.peakSearchTransientBytes
                      << " estimated_peak_owned_bytes=" << extraction.paths.estimatedPeakOwnedBytes
                      << " evaluated_dp_nodes=" << extraction.paths.evaluatedDpNodes << " dp_nodes_per_second=" << nodeRate
                      << " candidate_generation_seconds=" << extraction.paths.candidateGenerationSeconds
                      << " candidate_generation_cpu_seconds=" << extraction.paths.candidateGenerationCpuSeconds << " candidate_generation_effective_cores="
                      << effectiveCores(extraction.paths.candidateGenerationCpuSeconds, extraction.paths.candidateGenerationSeconds)
                      << " preparation_seconds=" << extraction.paths.preparationSeconds << " preparation_cpu_seconds=" << extraction.paths.preparationCpuSeconds
                      << " preparation_effective_cores=" << effectiveCores(extraction.paths.preparationCpuSeconds, extraction.paths.preparationSeconds)
                      << " corner_merge_seconds=" << extraction.paths.cornerMergeSeconds
                      << " corner_merge_cpu_seconds=" << extraction.paths.cornerMergeCpuSeconds
                      << " corner_merge_effective_cores=" << effectiveCores(extraction.paths.cornerMergeCpuSeconds, extraction.paths.cornerMergeSeconds)
                      << " prediction_read_seconds=" << extraction.paths.predictionSamplingSeconds
                      << " prediction_read_calls=" << extraction.paths.predictionSamplingCalls
                      << " prediction_read_cpu_seconds=" << extraction.paths.predictionSamplingCpuSeconds << " prediction_read_effective_cores="
                      << effectiveCores(extraction.paths.predictionSamplingCpuSeconds, extraction.paths.predictionSamplingSeconds)
                      << " normal_read_seconds=" << extraction.paths.normalSamplingSeconds << " normal_read_calls=" << extraction.paths.normalSamplingCalls
                      << " normal_read_cpu_seconds=" << extraction.paths.normalSamplingCpuSeconds << " normal_read_effective_cores="
                      << effectiveCores(extraction.paths.normalSamplingCpuSeconds, extraction.paths.normalSamplingSeconds)
                      << " sampling_materialize_seconds=" << extraction.paths.samplingMaterializationSeconds
                      << " sampling_materialize_cpu_seconds=" << extraction.paths.samplingMaterializationCpuSeconds << " sampling_materialize_effective_cores="
                      << effectiveCores(extraction.paths.samplingMaterializationCpuSeconds, extraction.paths.samplingMaterializationSeconds)
                      << " search_seconds=" << extraction.paths.searchSeconds << " search_cpu_seconds=" << extraction.paths.searchCpuSeconds
                      << " search_effective_cores=" << effectiveCores(extraction.paths.searchCpuSeconds, extraction.paths.searchSeconds)
                      << " total_seconds=" << totalSeconds << " total_cpu_seconds=" << totalCpuSeconds
                      << " total_effective_cores=" << effectiveCores(totalCpuSeconds, totalSeconds) << '\n';
            printTubeExtractionProfile(std::cout, extraction);
            return 0;
        }

        if (options.command == Command::QuantizationBenchmark) {
            resolveAnchorConfig(options, grid);
            if (options.corridorRadiusBaseVoxels.has_value()) {
                options.paths.corridorRadiusPredictionVoxels = *options.corridorRadiusBaseVoxels / grid.predictionToBaseScale;
                vc::fiber_tracer::validateFiberletPathConfig(options.paths);
            }
            const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
            if (fiber.controlPointLineIndices.empty() || fiber.controlPointLineIndices.front() >= fiber.linePointsXyzBase.size()) {
                fail("quantization benchmark fiber has no valid first control point");
            }
            const auto reference = vc::fiber_tracer::makePolylineArcGeometry(fiber.linePointsXyzBase);
            const auto availableInterval = vc::fiber_tracer::selectForwardPolylineArcInterval(reference, fiber.controlPointLineIndices.front());
            const auto interval = resolveQuantizationReplayInterval(reference, fiber.controlPointLineIndices.front(), options);
            vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
            normalOptions.workingToBaseScale = grid.predictionToBaseScale;
            normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, normalOptions);
            auto normalSampler =
                std::make_shared<vc::lasagna::LasagnaNormalSampler>(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});
            options.graphReplay.errorThresholdBaseVoxels = options.failureThresholdBaseVoxels;
            options.graphReplay.matchRefineSteps = options.matchRefineSteps;
            options.graphReplay.minimumResetAdvanceBaseVoxels = options.paths.longitudinalStepPredictionVoxels * grid.predictionToBaseScale;
            options.graphReplay.referenceBeginArcBase = interval.beginArc;
            options.graphReplay.referenceEndArcBase = interval.endArc;
            options.graphReplay.initialSeedKey = options.replayInitialSeedKey;
            options.graphReplay.recordDecisionDiagnostics = options.replayBeginArcBaseVoxels.has_value();
            const double extractionPaddingBase = quantizationExtractionArcPaddingBaseVoxels(options, grid);
            const vc::fiber_tracer::ForwardPolylineArcInterval extractionInterval{
                std::max(availableInterval.beginArc, interval.beginArc - extractionPaddingBase),
                std::min(availableInterval.endArc, interval.endArc + extractionPaddingBase),
            };
            const bool focusedDiagnostics = options.replayBeginArcBaseVoxels.has_value();
            const auto cacheIdentityInterval = focusedDiagnostics ? availableInterval : extractionInterval;
            const auto scheduledInterval = focusedDiagnostics ? interval : extractionInterval;
            const auto scenarios = vc::fiber_tracer::standardFiberletQuantizationScenarios();
            std::vector<vc::fiber_tracer::FiberletQuantizationScenario> selectedScenarios;
            if (*options.quantizationScenario == "all") {
                for (const auto& scenario : scenarios) {
                    if (scenario.name != "baseline")
                        selectedScenarios.push_back(scenario);
                }
            } else {
                const auto scenarioIt = std::find_if(scenarios.begin(), scenarios.end(), [&](const auto& scenario) {
                    return scenario.name == *options.quantizationScenario;
                });
                if (scenarioIt == scenarios.end())
                    fail("unknown quantization scenario");
                selectedScenarios.push_back(*scenarioIt);
            }
            struct CachedRun {
                vc::fiber_tracer::FiberletGraphReplayResult replay;
                vc::fiber_tracer::FiberletLogicalProjectionStats projection;
                std::size_t decodedResidentBytes = 0;
                double wallSeconds = 0.0;
                double cpuSeconds = 0.0;
            };
            const auto run = [&](std::string label,
                                 const vc::fiber_tracer::FiberletGeometryCacheProfile& cacheProfile,
                                 const vc::fiber_tracer::FiberletEvaluationQuantization& replayQuantization) {
                ReplayOverallProgress progress(true, label);
                const auto start = std::chrono::steady_clock::now();
                const double cpuStart = processCpuSeconds();
                auto context = createCachedReplayContext(
                    dataset,
                    normalDataset,
                    grid,
                    options,
                    field,
                    normalSampler,
                    fiber.linePointsXyzBase,
                    reference,
                    cacheIdentityInterval.beginArc,
                    cacheIdentityInterval.endArc,
                    scheduledInterval.beginArc,
                    scheduledInterval.endArc,
                    !focusedDiagnostics,
                    options.outputDirectory / "cache",
                    cacheProfile,
                    replayQuantization,
                    options.anchorCacheRoot,
                    cacheProfile.enabled() ? std::filesystem::path{} : options.fiberletCacheRoot,
                    progress);
                progress.beginTracing();
                progress.updateGreedy(1.0);
                const auto failurePrinter = [&](const vc::fiber_tracer::FiberReplayFailure& event) {
                    std::ostringstream line;
                    line << std::setprecision(17) << "fiberlet_quantization_failure run=" << std::quoted(label) << " index=" << event.index
                         << " reference_arc_base=" << event.referenceArcBase << " reference_arc_fraction=" << event.referenceArcFraction
                         << " reason=" << event.reason;
                    if (event.thresholdMeasurement.has_value()) {
                        const auto& measurement = *event.thresholdMeasurement;
                        line << " euclidean_error_base_voxels=" << measurement.euclideanErrorBaseVoxels << " normal_error_base_voxels=";
                        if (measurement.normalErrorBaseVoxels.has_value())
                            line << *measurement.normalErrorBaseVoxels;
                        else
                            line << "n/a";
                        line << " tangential_error_base_voxels=";
                        if (measurement.tangentialErrorBaseVoxels.has_value())
                            line << *measurement.tangentialErrorBaseVoxels;
                        else
                            line << "n/a";
                        line << " threshold_error_base_voxels=" << measurement.thresholdErrorBaseVoxels
                             << " threshold_error_ratio=" << measurement.thresholdErrorRatio
                             << " local_normal_valid=" << (measurement.localNormalValid ? "true" : "false");
                    }
                    progress.printEventLine(line.str());
                };
                auto replay = vc::fiber_tracer::
                    traceFiberletGraphReplay(*context.graph, fiber.linePointsXyzBase, *normalSampler, grid.predictionToBaseScale, options.graphReplay, failurePrinter, [&](const auto& event) {
                        progress.updateFiberlet(event.referenceArcFraction, event.rolloutExpandedStateCount, event.minimumAppliedLocalPruneLossCutoffPerPredictionVoxel);
                    });
                progress.tracingComplete();
                const auto anchorStats = context.preprocessor->anchorCache()->stats();
                const auto fiberletStats = context.preprocessor->fiberletCache()->stats();
                CachedRun result;
                result.replay = std::move(replay);
                result.projection = context.graph->logicalProjectionStats();
                result.decodedResidentBytes = anchorStats.localDecodedBytes + fiberletStats.localDecodedBytes;
                result.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                result.cpuSeconds = processCpuSeconds() - cpuStart;
                progress.finish();
                return result;
            };

            const auto exactQuantization =
                vc::fiber_tracer::exactFiberletReplayQuantization(
                    options.storageChunkSideBaseVoxels);
            const auto baseline = run(
                "quantization baseline",
                vc::fiber_tracer::fiberletGeometryCacheProfile(
                    exactQuantization),
                exactQuantization);
            for (const auto& scenario : selectedScenarios) {
                vc::fiber_tracer::FiberletEvaluationQuantization replayQuantization{
                    scenario.positionQuantumBaseVoxels,
                    scenario.compactAxes,
                    scenario.costBits,
                    options.storageChunkSideBaseVoxels,
                };
                replayQuantization.costDomain = scenario.costDomain;
                replayQuantization.costDensityMaximum =
                    scenario.costDensityMaximum;
                const auto cacheProfile = vc::fiber_tracer::fiberletGeometryCacheProfile(replayQuantization);
                std::optional<CachedRun> measuredRun;
                if (scenario.name != "baseline") {
                    measuredRun = run("quantization " + scenario.name, cacheProfile, replayQuantization);
                }
                const auto& measured = measuredRun.has_value() ? *measuredRun : baseline;
                const auto comparison = vc::fiber_tracer::compareFiberletCachedReplays(
                    scenario, baseline.replay, measured.replay, fiber.linePointsXyzBase, *normalSampler, grid.predictionToBaseScale, options.failureThresholdBaseVoxels, printQuantizationProgress);
                const auto baselineReplayPath =
                    writeQuantizationReplay(options, baseline.replay, options.graphReplay, scenario, "baseline", dataset, normalDataset);
                const auto scenarioReplayPath =
                    writeQuantizationReplay(options, measured.replay, options.graphReplay, scenario, "scenario", dataset, normalDataset);
                (void)writeQuantizationRouteCostStatistics(
                    baselineReplayPath.parent_path(), baseline.replay, measured.replay, scenario.name, options.routeStatsFailureMarginBaseVoxels);
                if (options.graphReplay.recordDecisionDiagnostics) {
                    (void)writeQuantizationDecisionComparison(baselineReplayPath.parent_path(), baseline.replay, measured.replay, scenario.name);
                }
                std::cout << "fiberlet_quantization_replay_artifacts"
                          << " baseline=" << baselineReplayPath.string() << " scenario=" << scenarioReplayPath.string()
                          << " replay_arc_base=" << interval.beginArc << " replay_end_arc_base=" << interval.endArc
                          << " extraction_arc_base=" << extractionInterval.beginArc
                          << " extraction_end_arc_base=" << extractionInterval.endArc << " extraction_padding_base=" << extractionPaddingBase
                          << " cache_arc_base=" << cacheIdentityInterval.beginArc << " cache_end_arc_base=" << cacheIdentityInterval.endArc
                          << " cache_prefetch=" << (!focusedDiagnostics ? "true" : "false") << '\n';
                printQuantizationFailureWindows("baseline", baseline.replay);
                printQuantizationFailureWindows("scenario", measured.replay);
                std::cout << std::setprecision(17) << "fiberlet_cached_quantization"
                          << " scenario=" << scenario.name << " position_quantum_base=" << scenario.positionQuantumBaseVoxels
                          << " compact_directions=" << (scenario.compactAxes ? "true" : "false") << " cost_bits=" << scenario.costBits
                          << " cost_domain=" << vc::fiber_tracer::fiberletCostQuantizationDomainName(scenario.costDomain)
                          << " cost_density_maximum=" << scenario.costDensityMaximum
                          << " geometry_cache_cost_tag_bits=" << cacheProfile.compatibilityCostTagBits
                          << " radius_base=" << options.radiusBaseVoxels << " reference_length_base=" << interval.endArc - interval.beginArc
                          << " baseline_failures=" << comparison.baselineFailures << " scenario_failures=" << comparison.scenarioFailures
                          << " baseline_completed_fraction=" << comparison.baselineCompletedFraction
                          << " scenario_completed_fraction=" << comparison.scenarioCompletedFraction
                          << " line_distance_available=" << (comparison.lineDistanceAvailable ? "true" : "false")
                          << " line_distance_samples=" << comparison.lineDistanceSamples
                          << " line_distance_invalid_normal_samples=" << comparison.lineDistanceInvalidNormalSamples
                          << " projected_anchors=" << measured.projection.projectedAnchors
                          << " coincident_groups=" << measured.projection.coincidentPositionGroups
                          << " maximum_variants=" << measured.projection.maximumVariants << " compact_cost_chunks=" << measured.projection.compactCostChunks
                          << " baseline_decoded_resident_bytes=" << baseline.decodedResidentBytes
                          << " scenario_decoded_resident_bytes=" << measured.decodedResidentBytes
                          << " baseline_wall_seconds=" << baseline.wallSeconds << " baseline_cpu_seconds=" << baseline.cpuSeconds
                          << " scenario_wall_seconds=" << measured.wallSeconds << " scenario_cpu_seconds=" << measured.cpuSeconds;
                printQuantizationSummary(std::cout, "line_distance_base", comparison.lineDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "line_normal_distance_base", comparison.lineNormalDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "line_tangential_distance_base", comparison.lineTangentialDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "baseline_reference_distance_base", comparison.baselineReferenceDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "baseline_reference_normal_distance_base", comparison.baselineReferenceNormalDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "baseline_reference_tangential_distance_base", comparison.baselineReferenceTangentialDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "scenario_reference_distance_base", comparison.scenarioReferenceDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "scenario_reference_normal_distance_base", comparison.scenarioReferenceNormalDistanceBaseVoxels);
                printQuantizationSummary(std::cout, "scenario_reference_tangential_distance_base", comparison.scenarioReferenceTangentialDistanceBaseVoxels);
                std::cout << '\n' << std::flush;
            }
            return 0;
        }

        if (isReplayCommand(options.command)) {
            const auto traceSetupStart = std::chrono::steady_clock::now();
            ReplayOverallProgress overallProgress(!options.printStats);
            if (options.printStats)
                std::cerr << "fiber_replay_stage stage=trace_setup status=started\n";
            const auto scales = vc::fiber_tracer::resolveFiberPredictionTraceScales(dataset.manifest(), options.inferenceScaledownPower);
            auto traceManifest = dataset.manifest();
            traceManifest.workingToBaseScale = scales.traceToBaseScale;
            const vc::lasagna::LasagnaDataset traceDataset(std::move(traceManifest));
            const vc::fiber_tracer::FiberPredictionField traceField(traceDataset, options.decodedCacheBytes, vc::fiber_tracer::FiberPredictionFieldBindingMode::TraceOptions);

            vc::lasagna::LasagnaDatasetOpenOptions traceNormalOptions;
            traceNormalOptions.workingToBaseScale = scales.traceToBaseScale;
            traceNormalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto traceNormalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, traceNormalOptions);
            const vc::lasagna::LasagnaNormalSampler traceNormalSampler(traceNormalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});

            const auto fiber = vc::fiber_tracer::loadFiberJson(options.fiberJson);
            if (fiber.controlPointLineIndices.empty() || fiber.controlPointLineIndices.front() >= fiber.linePointsXyzBase.size()) {
                fail("fiber replay fiber has no valid first control point");
            }
            const auto reference = vc::fiber_tracer::makePolylineArcGeometry(fiber.linePointsXyzBase);
            const auto interval =
                vc::fiber_tracer::selectForwardPolylineArcInterval(reference, fiber.controlPointLineIndices.front(), options.replayLengthBaseVoxels);
            const double startArc = interval.beginArc;
            const double endArc = interval.endArc;
            const auto requestedTrace = options.trace;
            auto effectiveTrace = requestedTrace;
            effectiveTrace.beamWidth = 1;
            effectiveTrace.beamLookaheadSteps = 1;
            effectiveTrace.traceToBaseScale = scales.traceToBaseScale;
            vc::fiber_tracer::FiberReplayTraceRequest replayRequest;
            replayRequest.fiber = fiber;
            replayRequest.traceToBaseScale = scales.traceToBaseScale;
            replayRequest.errorThresholdBaseVoxels = options.failureThresholdBaseVoxels;
            replayRequest.matchRefineSteps = options.matchRefineSteps;
            replayRequest.referenceEndArcBase = endArc;
            const double nominalStepBaseVoxels = effectiveTrace.stepVoxels * scales.traceToBaseScale;
            replayRequest.config = effectiveTrace;

            if (options.printStats) {
                std::cerr << "fiber_replay_stage stage=trace_setup status=completed"
                          << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - traceSetupStart).count()
                          << '\n';
            }
            const auto referenceGeometry = vc::fiber_tracer::slicePolylineArc(reference, startArc, endArc);
            resolveAnchorConfig(options, grid);
            std::mutex outputMutex;

            vc::fiber_tracer::FiberAnchorArtifactInfo baseAnchorArtifact;
            baseAnchorArtifact.sourceLocator = datasetLocator(dataset);
            baseAnchorArtifact.manifestContentHash = fileHash(dataset.manifest().manifestPath);
            baseAnchorArtifact.glyphLengthBaseVoxels = options.glyphLengthBaseVoxels;
            baseAnchorArtifact.baseVoxelSizeUm = options.baseVoxelSizeUm;

            vc::lasagna::LasagnaDatasetOpenOptions canonicalNormalOptions;
            canonicalNormalOptions.workingToBaseScale = grid.predictionToBaseScale;
            canonicalNormalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto canonicalNormalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, canonicalNormalOptions);
            const auto canonicalNormalSampler =
                std::make_shared<vc::lasagna::LasagnaNormalSampler>(canonicalNormalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});
            const auto fiberletEvaluationQuantization = options.eagerGraphReplay
                ? vc::fiber_tracer::exactFiberletReplayQuantization(
                      options.storageChunkSideBaseVoxels)
                : vc::fiber_tracer::defaultFiberletReplayQuantization(
                      options.storageChunkSideBaseVoxels);
            const auto fiberletCacheProfile =
                vc::fiber_tracer::fiberletGeometryCacheProfile(
                    fiberletEvaluationQuantization);
            std::optional<vc::fiber_tracer::FiberletGraph> eagerGraph;
            std::shared_ptr<vc::fiber_tracer::FiberletOnDemandPreprocessor>
                preprocessor;
            std::vector<vc::fiber_tracer::FiberletScheduledChunk> chunkSchedule;
            std::unique_ptr<vc::fiber_tracer::FiberletCachedReplayGraphSource>
                cachedGraph;
            struct ReplayChunkProgressLocation {
                std::size_t scheduleIndex = 0;
                double referenceArcBase = 0.0;
            };
            std::map<ReplayChunkId, ReplayChunkProgressLocation> fiberletChunkProgressLocations;
            std::map<ReplayChunkId, ReplayChunkProgressLocation> anchorChunkProgressLocations;
            std::set<ReplayChunkId> completedFiberletChunks;
            std::set<ReplayChunkId> completedAnchorChunks;
            auto preprocessingProgress = std::make_shared<ReplayPreprocessingProgress>();
            if (options.eagerGraphReplay) {
                if (options.printStats)
                    std::cerr << "fiber_replay_stage stage=full_extraction status=started\n";
                auto fullExtraction = extractTubeFiberlets(
                    fiber.linePointsXyzBase,
                    startArc,
                    endArc,
                    options.radiusBaseVoxels,
                    grid,
                    options,
                    field,
                    *canonicalNormalSampler,
                    true,
                    options.printStats);
                const auto& fullTube = fullExtraction.tube;
                const size_t fullAnchorCount = fullExtraction.anchors.diagnostics.oneAnchorCells + 2 * fullExtraction.anchors.diagnostics.twoAnchorCells;
                if (options.printStats) {
                    std::cerr << "fiber_replay_stage stage=full_extraction status=completed"
                              << " cells=" << fullTube.cellsZYX.size() << " anchors=" << fullAnchorCount
                              << " anchor_seconds=" << fullExtraction.anchorSeconds << " fiberlet_seconds=" << fullExtraction.fiberletSeconds
                              << " searched=" << fullExtraction.paths.diagnostics.searchedPairs
                              << " accepted=" << fullExtraction.paths.diagnostics.successfulPaths << '\n';
                    printTubeExtractionProfile(std::cerr, fullExtraction);
                }
                eagerGraph.emplace(vc::fiber_tracer::buildFiberletGraph(fullExtraction.paths));
            } else {
                const auto processingTube = vc::fiber_tracer::makeFiberReplayTube(
                    fiber.linePointsXyzBase,
                    0.5 * (startArc + endArc),
                    0.5 * (endArc - startArc),
                    options.radiusBaseVoxels,
                    grid,
                    options.anchors.cellSizePredictionVoxels,
                    false);
                const auto containmentQuery = processingTube.makePredictionContainmentQuery(grid.predictionToBaseScale);
                auto anchorMetadata = replayDatasetMetadata(
                    vc::fiber_tracer::FiberletDatasetKind::Anchors,
                    grid,
                    options,
                    dataset,
                    canonicalNormalDataset,
                    processingTube.referenceIntervalBase,
                    processingTube.radiusBaseVoxels);
                auto fiberletMetadata = replayDatasetMetadata(
                    vc::fiber_tracer::FiberletDatasetKind::Fiberlets,
                    grid,
                    options,
                    dataset,
                    canonicalNormalDataset,
                    processingTube.referenceIntervalBase,
                    processingTube.radiusBaseVoxels,
                    fiberletCacheProfile);
                auto anchorNamespace = anchorMetadata.algorithmFingerprint;
                auto fiberletNamespace = fiberletMetadata.algorithmFingerprint;
                std::replace(anchorNamespace.begin(), anchorNamespace.end(), ':', '-');
                std::replace(fiberletNamespace.begin(), fiberletNamespace.end(), ':', '-');
                const auto anchorRoot = options.anchorCacheRoot.empty()
                    ? options.outputDirectory / "cache" / anchorNamespace /
                        "anchors.zarr"
                    : options.anchorCacheRoot;
                const auto fiberletRoot = options.fiberletCacheRoot.empty()
                    ? options.outputDirectory / "cache" / fiberletNamespace /
                        "fiberlets.zarr"
                    : options.fiberletCacheRoot;
                auto graphBudget = std::make_shared<vc::render::DecodedChunkCacheBudget>(options.decodedCacheBytes);
                vc::render::ChunkCache::Options anchorCacheOptions;
                anchorCacheOptions.decodedByteCapacity = options.decodedCacheBytes;
                anchorCacheOptions.decodedByteBudget = graphBudget;
                anchorCacheOptions.maxConcurrentReads = 1;
                vc::render::ChunkCache::Options fiberletCacheOptions = anchorCacheOptions;
                vc::fiber_tracer::FiberletOnDemandConfig onDemand;
                onDemand.anchorRoot = anchorRoot;
                onDemand.fiberletRoot = fiberletRoot;
                onDemand.anchorMetadata = anchorMetadata;
                onDemand.fiberletMetadata = fiberletMetadata;
                onDemand.grid = grid;
                onDemand.anchorConfig = options.anchors;
                onDemand.pathConfig = options.paths;
                onDemand.geometryQuantization = fiberletCacheProfile.geometry;
                onDemand.predictionSampler = [&](const auto& indices, int threads, auto& samples) {
                    field.sampleStoredGridBatch(indices, threads, samples);
                };
                onDemand.normalSampler = canonicalNormalSampler;
                onDemand.anchorCellPredicate =
                    [containmentQuery, grid, cellSize = options.anchors.cellSizePredictionVoxels](const std::array<size_t, 3>& cell) {
                        return containmentQuery.intersectsPredictionCell(cell, grid, cellSize);
                    };
                onDemand.anchorRetainPredicate = [containmentQuery](const vc::fiber_tracer::FiberAnchor& anchor) {
                    return containmentQuery.evaluatePredictionAnchor(anchor);
                };
                onDemand.pointPredicate = [containmentQuery](const cv::Vec3d& pointPrediction) {
                    return containmentQuery.containsPredictionPoint(pointPrediction);
                };
                onDemand.anchorCacheOptions = std::move(anchorCacheOptions);
                onDemand.fiberletCacheOptions = std::move(fiberletCacheOptions);
                onDemand.chunkResolved =
                    [preprocessingProgress](vc::fiber_tracer::FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, vc::render::ChunkFetchStatus status) {
                        preprocessingProgress->resolve(kind, key, status);
                    };
                if (options.printStats) {
                    onDemand.progress = [&](const auto& progress) {
                        std::lock_guard lock(outputMutex);
                        const ReplayChunkId id{progress.key.level, progress.key.iz, progress.key.iy, progress.key.ix};
                        const bool anchorStage = progress.stage == "anchors";
                        const auto& locations = anchorStage ? anchorChunkProgressLocations : fiberletChunkProgressLocations;
                        auto& completed = anchorStage ? completedAnchorChunks : completedFiberletChunks;
                        if (progress.status == "completed")
                            completed.insert(id);
                        const auto location = locations.find(id);
                        std::cerr << "fiber_replay_cache_chunk"
                                  << " stage=" << progress.stage << " status=" << progress.status << " key=" << progress.key.iz << ','
                                  << progress.key.iy << ',' << progress.key.ix << " inputs=" << progress.inputCount
                                  << " unfiltered_inputs=" << progress.unfilteredInputCount << " filtered_inputs="
                                  << (progress.unfilteredInputCount >= progress.inputCount ? progress.unfilteredInputCount - progress.inputCount : 0)
                                  << " outputs=" << progress.outputCount << " generated_chunks=" << completed.size()
                                  << " scheduled_chunks=" << locations.size();
                        if (location != locations.end()) {
                            std::cerr << " schedule_index=" << location->second.scheduleIndex + 1
                                      << " nearest_reference_arc_base=" << location->second.referenceArcBase << " nearest_reference_arc_fraction="
                                      << std::clamp((location->second.referenceArcBase - startArc) / (endArc - startArc), 0.0, 1.0);
                        }
                        if (!progress.phase.empty()) {
                            std::cerr << " phase=" << progress.phase << " phase_completed=" << progress.phaseCompleted
                                      << " phase_total=" << progress.phaseTotal;
                        }
                        std::cerr << " elapsed_seconds=" << progress.elapsedSeconds << " cpu_seconds=" << progress.cpuSeconds;
                        if (progress.candidateGenerationWorkers > 0) {
                            std::cerr << " candidate_generation_workers=" << progress.candidateGenerationWorkers
                                      << " candidate_generation_seconds=" << progress.candidateGenerationSeconds
                                      << " candidate_generation_cpu_seconds=" << progress.candidateGenerationCpuSeconds
                                      << " candidate_generation_effective_cores="
                                      << (progress.candidateGenerationSeconds > 0.0 ? progress.candidateGenerationCpuSeconds / progress.candidateGenerationSeconds
                                                                                    : 0.0);
                        }
                        std::cerr << '\n';
                    };
                    std::cerr << "fiber_replay_stage stage=cache_open status=started"
                              << " anchor_root=" << std::quoted(anchorRoot.string()) << " fiberlet_root=" << std::quoted(fiberletRoot.string())
                              << " decoded_budget_bytes=" << options.decodedCacheBytes << '\n';
                }
                preprocessor =
                    vc::fiber_tracer::FiberletOnDemandPreprocessor::create(
                        std::move(onDemand));
                chunkSchedule = preprocessor->referenceChunkSchedule(
                    reference, startArc, endArc,
                    options.radiusBaseVoxels);
                for (std::size_t index = 0; index < chunkSchedule.size(); ++index) {
                    const auto& scheduled = chunkSchedule[index];
                    const ReplayChunkId fiberletId{scheduled.key.level, scheduled.key.iz, scheduled.key.iy, scheduled.key.ix};
                    fiberletChunkProgressLocations.emplace(fiberletId, ReplayChunkProgressLocation{index, scheduled.nearestReferenceArcBase});
                    for (const auto& dependency : preprocessor->anchorDependencies(scheduled.key)) {
                        const ReplayChunkId anchorId{dependency.level, dependency.iz, dependency.iy, dependency.ix};
                        const ReplayChunkProgressLocation candidate{index, scheduled.nearestReferenceArcBase};
                        const auto found = anchorChunkProgressLocations.find(anchorId);
                        if (found == anchorChunkProgressLocations.end() || candidate.scheduleIndex < found->second.scheduleIndex) {
                            anchorChunkProgressLocations[anchorId] = candidate;
                        }
                    }
                }
                std::set<ReplayChunkId> expectedAnchors;
                std::set<ReplayChunkId> expectedPrefixes;
                for (const auto& [id, location] : anchorChunkProgressLocations) {
                    (void)location;
                    expectedAnchors.insert(id);
                }
                for (const auto& [id, location] : fiberletChunkProgressLocations) {
                    (void)location;
                    expectedPrefixes.insert(id);
                }
                preprocessingProgress->configure(std::move(expectedAnchors), std::move(expectedPrefixes));
                overallProgress.attachPreprocessing(preprocessingProgress);
                cachedGraph = std::make_unique<
                    vc::fiber_tracer::FiberletCachedReplayGraphSource>(
                    preprocessor, options.paths,
                    fiberletEvaluationQuantization);
                preprocessor->prefetchScheduled(chunkSchedule, 0, chunkSchedule.size(), false);
                if (options.printStats)
                    std::cerr << "fiber_replay_stage stage=cache_open status=completed\n";
            }

            options.graphReplay.errorThresholdBaseVoxels = options.failureThresholdBaseVoxels;
            options.graphReplay.matchRefineSteps = options.matchRefineSteps;
            options.graphReplay.minimumResetAdvanceBaseVoxels = nominalStepBaseVoxels;
            options.graphReplay.referenceBeginArcBase = startArc;
            options.graphReplay.referenceEndArcBase = endArc;

            size_t greedyFailureCount = 0;
            size_t fiberletFailureCount = 0;
            const auto failurePrinter = [&](vc::fiber_tracer::FiberReplayTracer tracer) {
                return [&, tracer](const vc::fiber_tracer::FiberReplayFailure& event) {
                    if (tracer == vc::fiber_tracer::FiberReplayTracer::Greedy)
                        overallProgress.updateGreedy(event.referenceArcFraction);
                    else
                        overallProgress.updateFiberlet(event.referenceArcFraction);
                    std::lock_guard lock(outputMutex);
                    auto& count = tracer == vc::fiber_tracer::FiberReplayTracer::Greedy ? greedyFailureCount : fiberletFailureCount;
                    count = event.index + 1;
                    std::ostringstream line;
                    line << std::setprecision(17) << "fiber_replay_failure tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer)
                         << " index=" << event.index << " reference_arc_base=" << event.referenceArcBase
                         << " reference_arc_fraction=" << event.referenceArcFraction << " reason=" << event.reason;
                    const auto printOptional = [&](const auto& value) {
                        if (value.has_value())
                            line << *value;
                        else
                            line << "n/a";
                    };
                    line << " euclidean_error_base_voxels=";
                    if (event.thresholdMeasurement.has_value())
                        line << event.thresholdMeasurement->euclideanErrorBaseVoxels;
                    else
                        line << "n/a";
                    line << " normal_error_base_voxels=";
                    printOptional(event.thresholdMeasurement.has_value() ? event.thresholdMeasurement->normalErrorBaseVoxels : std::optional<double>{});
                    line << " tangential_error_base_voxels=";
                    printOptional(event.thresholdMeasurement.has_value() ? event.thresholdMeasurement->tangentialErrorBaseVoxels : std::optional<double>{});
                    line << " threshold_error_base_voxels=";
                    if (event.thresholdMeasurement.has_value())
                        line << event.thresholdMeasurement->thresholdErrorBaseVoxels;
                    else
                        line << "n/a";
                    line << " threshold_error_ratio=";
                    if (event.thresholdMeasurement.has_value())
                        line << event.thresholdMeasurement->thresholdErrorRatio;
                    else
                        line << "n/a";
                    line << " local_normal_valid=";
                    if (event.thresholdMeasurement.has_value())
                        line << (event.thresholdMeasurement->localNormalValid ? "true" : "false");
                    else
                        line << "n/a";
                    line << " greedy_failures=" << greedyFailureCount << " fiberlet_failures=" << fiberletFailureCount;
                    overallProgress.printEventLine(line.str());
                };
            };

            const auto traceStart = std::chrono::steady_clock::now();
            overallProgress.beginTracing();
            if (options.printStats)
                std::cerr << "fiber_replay_stage stage=parallel_trace status=started\n";
            auto greedyFuture = std::async(std::launch::async, [&]() {
                try {
                    auto result = vc::fiber_tracer::traceFiberReplay(
                        traceField,
                        replayRequest,
                        traceNormalSampler,
                        scales.traceToBaseScale,
                        [&](const vc::fiber_tracer::FiberTraceProgress& event) {
                            overallProgress.updateGreedy(event.referenceArcFraction.value_or(0.0));
                            if (options.printStats && (event.step == event.maxSteps || event.step % 100 == 0)) {
                                std::lock_guard lock(outputMutex);
                                std::cerr << std::setprecision(17) << "fiber_replay_progress tracer=greedy"
                                          << " state=running"
                                          << " reference_arc_base=" << event.referenceArcBase.value_or(startArc)
                                          << " reference_arc_fraction=" << event.referenceArcFraction.value_or(0.0)
                                          << " segment=" << event.replaySegmentIndex.value_or(0) << " local_step=" << event.step
                                          << " local_budget=" << event.maxSteps << " local_reason=" << event.reason << '\n';
                            }
                        },
                        failurePrinter(vc::fiber_tracer::FiberReplayTracer::Greedy));
                    overallProgress.updateGreedy(1.0);
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << std::setprecision(17) << "fiber_replay_evaluator tracer=greedy"
                                  << " status=completed"
                                  << " reference_arc_base=" << result.completedReferenceArcBase << " reference_arc_fraction=1"
                                  << " failures=" << result.failures.size() << '\n';
                    }
                    return result;
                } catch (const std::exception& error) {
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << "fiber_replay_evaluator tracer=greedy"
                                  << " status=failed error=" << std::quoted(error.what()) << '\n';
                    }
                    throw;
                } catch (...) {
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << "fiber_replay_evaluator tracer=greedy"
                                  << " status=failed error=\"unknown exception\"\n";
                    }
                    throw;
                }
            });
            auto fiberletFuture = std::async(std::launch::async, [&]() {
                try {
                    const auto progress = [&](const vc::fiber_tracer::FiberletGraphReplayProgress& event) {
                        overallProgress.updateFiberlet(event.referenceArcFraction, event.rolloutExpandedStateCount, event.minimumAppliedLocalPruneLossCutoffPerPredictionVoxel);
                        if (!options.printStats)
                            return;
                        std::lock_guard lock(outputMutex);
                        std::cerr << std::setprecision(17) << "fiber_replay_progress tracer=fiberlet"
                                  << " state=" << event.state << " reference_arc_base=" << event.referenceArcBase
                                  << " reference_arc_fraction=" << event.referenceArcFraction << " segment=" << event.segmentIndex;
                        if (event.rolloutExpandedStateCount.has_value())
                            std::cerr << " fiberlet_rollout_expansions=" << *event.rolloutExpandedStateCount;
                        if (event.minimumAppliedLocalPruneLossCutoffPerPredictionVoxel.has_value()) {
                            std::cerr << " fiberlet_local_cutoff_loss_per_vx_min=" << *event.minimumAppliedLocalPruneLossCutoffPerPredictionVoxel;
                        }
                        std::cerr << '\n';
                    };
                    auto result = eagerGraph.has_value() ? vc::fiber_tracer::traceFiberletGraphReplay(
                                                               *eagerGraph,
                                                               fiber.linePointsXyzBase,
                                                               *canonicalNormalSampler,
                                                               grid.predictionToBaseScale,
                                                               options.graphReplay,
                                                               failurePrinter(vc::fiber_tracer::FiberReplayTracer::Fiberlet),
                                                               progress)
                                                         : vc::fiber_tracer::traceFiberletGraphReplay(
                                                               *cachedGraph,
                                                               fiber.linePointsXyzBase,
                                                               *canonicalNormalSampler,
                                                               grid.predictionToBaseScale,
                                                               options.graphReplay,
                                                               failurePrinter(vc::fiber_tracer::FiberReplayTracer::Fiberlet),
                                                               progress);
                    overallProgress.updateFiberlet(1.0);
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << std::setprecision(17) << "fiber_replay_evaluator tracer=fiberlet"
                                  << " status=completed"
                                  << " reference_arc_base=" << result.completedReferenceArcBase << " reference_arc_fraction=1"
                                  << " failures=" << result.failures.size() << '\n';
                    }
                    return result;
                } catch (const std::exception& error) {
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << "fiber_replay_evaluator tracer=fiberlet"
                                  << " status=failed error=" << std::quoted(error.what()) << '\n';
                    }
                    throw;
                } catch (...) {
                    if (options.printStats) {
                        std::lock_guard lock(outputMutex);
                        std::cerr << "fiber_replay_evaluator tracer=fiberlet"
                                  << " status=failed error=\"unknown exception\"\n";
                    }
                    throw;
                }
            });
            std::optional<vc::fiber_tracer::FiberReplayTraceResult> greedyReplay;
            std::optional<vc::fiber_tracer::FiberletGraphReplayResult> fiberletReplay;
            std::exception_ptr traceError;
            try {
                greedyReplay = greedyFuture.get();
            } catch (...) {
                traceError = std::current_exception();
            }
            try {
                fiberletReplay = fiberletFuture.get();
            } catch (...) {
                if (!traceError)
                    traceError = std::current_exception();
            }
            if (traceError) {
                overallProgress.endLine();
                std::rethrow_exception(traceError);
            }
            overallProgress.tracingComplete();
            if (preprocessor && options.printStats) {
                const auto anchorStats = preprocessor->anchorCache()->stats();
                const auto fiberletStats = preprocessor->fiberletCache()->stats();
                const auto anchorMaterialization = preprocessor->anchorDataset()->materializationStats();
                const auto fiberletMaterialization = preprocessor->fiberletDataset()->materializationStats();
                const std::size_t residentBytes = anchorStats.localDecodedBytes + fiberletStats.localDecodedBytes;
                std::cerr << "fiber_replay_cache"
                          << " anchor_decoded_bytes=" << anchorStats.localDecodedBytes << " fiberlet_decoded_bytes=" << fiberletStats.localDecodedBytes
                          << " total_decoded_bytes=" << residentBytes << " configured_budget_bytes=" << options.decodedCacheBytes
                          << " anchor_chunk_decodes=" << anchorMaterialization.anchorDecodes
                          << " prefix_chunk_decodes=" << fiberletMaterialization.prefixDecodes
                          << " route_chunk_decodes=" << fiberletMaterialization.routeDecodes << '\n';
            }
            if (options.printStats) {
                std::cerr << "fiber_replay_stage stage=parallel_trace status=completed"
                          << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - traceStart).count()
                          << " greedy_failures=" << greedyReplay->failures.size()
                          << " fiberlet_failures=" << fiberletReplay->failures.size() << '\n';
            }

            vc::fiber_tracer::FiberReplayBundleInput bundle;
            bundle.request = replayRequest;
            bundle.greedyReplay = std::move(*greedyReplay);
            bundle.fiberletReplay = std::move(*fiberletReplay);
            bundle.fiberletReplayConfig = options.graphReplay;
            bundle.requestedLengthBaseVoxels = options.replayLengthBaseVoxels;
            bundle.referenceGeometryBase = referenceGeometry;
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
            bundle.fiberletEvaluationProfile = {
                {"role", options.eagerGraphReplay ? "exact_float_oracle"
                                                   : "default_compact"},
                {"position_quantum_base_voxels",
                 fiberletEvaluationQuantization.positionQuantumBaseVoxels},
                {"compact_directions",
                 fiberletEvaluationQuantization.compactDirections},
                {"cost_bits", fiberletEvaluationQuantization.costBits},
                {"cost_domain",
                 vc::fiber_tracer::fiberletCostQuantizationDomainName(
                     fiberletEvaluationQuantization.costDomain)},
                {"cost_density_maximum",
                 fiberletEvaluationQuantization.costDensityMaximum},
                {"storage_chunk_side_base_voxels",
                 fiberletEvaluationQuantization.storageChunkSideBaseVoxels},
                {"persistent_payload_profile", "float32_cache"},
            };
            bundle.requestedTraceConfig = vc::fiber_tracer::cli::traceConfigJson(requestedTrace);
            bundle.effectiveTraceConfig = vc::fiber_tracer::cli::traceConfigJson(effectiveTrace);

            if (options.writeReplayVisualizations) {
                const auto overviewStart = std::chrono::steady_clock::now();
                overallProgress.setOutputStage("overview");
                if (options.printStats)
                    std::cerr << "fiber_replay_stage stage=overview status=started\n";
                bundle.overview = vc::fiber_tracer::renderFiberReplayOverview(
                    bundle.referenceGeometryBase,
                    bundle.greedyReplay,
                    bundle.fiberletReplay,
                    *canonicalNormalSampler,
                    grid.predictionToBaseScale,
                    options.paths.parallelThreads,
                    *replayCtVolume,
                    replayCtLocator);
                std::cout.flush();
                if (options.printStats) {
                    std::cerr << "fiber_replay_stage stage=overview status=completed"
                              << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - overviewStart).count()
                              << " reference_top_shape_yx=" << bundle.overview->referenceTopShapeYX[0] << ','
                              << bundle.overview->referenceTopShapeYX[1]
                              << " reference_side_shape_yx=" << bundle.overview->referenceSideShapeYX[0] << ','
                              << bundle.overview->referenceSideShapeYX[1] << " fiberlet_top_shape_yx=" << bundle.overview->fiberletTopShapeYX[0]
                              << ',' << bundle.overview->fiberletTopShapeYX[1]
                              << " fiberlet_side_shape_yx=" << bundle.overview->fiberletSideShapeYX[0] << ','
                              << bundle.overview->fiberletSideShapeYX[1] << " pages=" << bundle.overview->pages.size() << '\n';
                }
                const std::size_t visualizationCount = bundle.greedyReplay.failures.size() + bundle.fiberletReplay.failures.size();
                std::size_t completedVisualizations = 0;
                if (visualizationCount > 0)
                    overallProgress.setOutputStage("visualizations", completedVisualizations, visualizationCount);
                const auto addVisualizations = [&](vc::fiber_tracer::FiberReplayTracer tracer, const auto& failures) {
                    for (const auto& failure : failures) {
                        const auto visualStart = std::chrono::steady_clock::now();
                        if (options.printStats) {
                            std::cerr << "fiber_replay_stage stage=visualization status=started"
                                      << " tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer) << " index=" << failure.index << '\n';
                        }
                        vc::fiber_tracer::FiberReplayVisualizationInput visual;
                        visual.tracer = tracer;
                        visual.tracerFailureIndex = failure.index;
                        auto local = extractTubeFiberlets(
                            fiber.linePointsXyzBase,
                            std::max(startArc, failure.referenceArcBase - options.alongBaseVoxels),
                            std::min(endArc, failure.referenceArcBase + options.alongBaseVoxels),
                            options.radiusBaseVoxels,
                            grid,
                            options,
                            field,
                            *canonicalNormalSampler,
                            true,
                            options.printStats);
                        visual.tube = std::move(local.tube);
                        visual.anchors = std::move(local.anchors);
                        visual.paths = std::move(local.paths);
                        visual.anchorArtifact = baseAnchorArtifact;
                        visual.pathArtifact.fiberManifestLocator = datasetLocator(dataset);
                        visual.pathArtifact.fiberManifestContentHash = baseAnchorArtifact.manifestContentHash;
                        visual.pathArtifact.normalManifestLocator = datasetLocator(canonicalNormalDataset);
                        visual.pathArtifact.normalManifestContentHash = fileHash(canonicalNormalDataset.manifest().manifestPath);
                        visual.pathArtifact.anchorArtifactLocator = "anchors/anchors.json";
                        visual.pathArtifact.anchorArtifactContentHash =
                            stringHash(vc::fiber_tracer::fiberAnchorReportJson(visual.anchors, visual.anchorArtifact).dump(2) + "\n");
                        visual.pathArtifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
                        visual.strips = vc::fiber_tracer::makeFiberReplayStripSurfaces(
                            visual.tube,
                            bundle.greedyReplay,
                            bundle.fiberletReplay,
                            *canonicalNormalSampler,
                            grid.predictionToBaseScale,
                            options.paths.parallelThreads);
                        vc::fiber_tracer::renderFiberReplayStripTextures(*visual.strips, *replayCtVolume, replayCtLocator);
                        bundle.visualizations.push_back(std::move(visual));
                        ++completedVisualizations;
                        overallProgress.updateOutputStage(completedVisualizations, visualizationCount);
                        if (options.printStats) {
                            std::cerr << "fiber_replay_stage stage=visualization status=completed"
                                      << " tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer) << " index=" << failure.index << " elapsed_seconds="
                                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - visualStart).count() << '\n';
                        }
                    }
                };
                addVisualizations(vc::fiber_tracer::FiberReplayTracer::Greedy, bundle.greedyReplay.failures);
                addVisualizations(vc::fiber_tracer::FiberReplayTracer::Fiberlet, bundle.fiberletReplay.failures);
            }

            const auto publishStart = std::chrono::steady_clock::now();
            overallProgress.setOutputStage("publish");
            if (options.printStats)
                std::cerr << "fiber_replay_stage stage=publish status=started\n";
            const auto resultBundle = vc::fiber_tracer::writeFiberReplayBundle(options.outputDirectory, bundle);
            if (options.printStats) {
                std::cerr << "fiber_replay_stage stage=publish status=completed"
                          << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - publishStart).count() << '\n';
            }
            overallProgress.finish();
            if (options.storageCompressionChunks > 0) {
                benchmarkFullRegionStorageCompression(field,
                    canonicalNormalSampler, grid, options, dataset,
                    canonicalNormalDataset, preprocessor, chunkSchedule);
            }
            if (resultBundle.contains("overview")) {
                for (const auto& page : resultBundle.at("overview").at("pages")) {
                    std::cout << "fiber_replay_overview"
                              << " index=" << page.at("index") << " image="
                              << std::filesystem::absolute(options.outputDirectory / page.at("stable_path").get<std::string>())
                                     .lexically_normal()
                                     .string()
                              << '\n';
                }
            }
            for (const auto& visualization : resultBundle.at("visualizations")) {
                std::cout << "fiber_replay_visualization"
                          << " tracer=" << visualization.at("tracer")
                          << " tracer_failure_index=" << visualization.at("tracer_failure_index") << " manifest="
                          << std::filesystem::absolute(options.outputDirectory / visualization.at("manifest").at("path").get<std::string>())
                                 .lexically_normal()
                                 .string()
                          << '\n';
            }
            std::cout << "fiber_replay status=reference_end"
                      << " greedy_failures=" << resultBundle.at("failure_counts").at("greedy")
                      << " fiberlet_failures=" << resultBundle.at("failure_counts").at("fiberlet") << " greedy_reference_fraction=1"
                      << " fiberlet_reference_fraction=1"
                      << " visualizations=" << resultBundle.at("visualizations").size() << '\n';
            return 0;
        }

        if (options.command == Command::Anchors) {
            const double cellSideBase = resolveAnchorConfig(options, grid);
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
                      << " peak_sigma_base_voxels=" << options.anchors.peakSigmaPredictionVoxels * grid.predictionToBaseScale
                      << " axial_sigma_base_voxels=" << options.anchors.peakAxialSigmaPredictionVoxels * grid.predictionToBaseScale
                      << " peak_step_base_voxels=" << options.anchors.peakGridStepPredictionVoxels * grid.predictionToBaseScale
                      << " local_window_base_voxels=" << options.anchors.localWindowRadiusPredictionVoxels * grid.predictionToBaseScale
                      << " nms_transverse_radius_base_voxels=" << options.anchors.nmsTransverseRadiusPredictionVoxels * grid.predictionToBaseScale
                      << " nms_longitudinal_radius_base_voxels=" << options.anchors.nmsLongitudinalRadiusPredictionVoxels * grid.predictionToBaseScale
                      << " robust_max_trim=" << options.anchors.robustMaximumTrimMassFraction
                      << " robust_mad_multiplier=" << options.anchors.robustMadMultiplier << " robust_min_angle_deg=" << options.anchors.robustMinimumAngleDegrees
                      << " cell_diagonal_base_voxels=" << cellSideBase * std::sqrt(3.0) << " cells=" << report.diagnostics.totalCells
                      << " anchors=" << report.diagnostics.oneAnchorCells + 2 * report.diagnostics.twoAnchorCells
                      << " zero=" << report.diagnostics.zeroAnchorCells << " one=" << report.diagnostics.oneAnchorCells
                      << " two=" << report.diagnostics.twoAnchorCells << " merged=" << report.diagnostics.mergedComponentPairs
                      << " nms_suppressed=" << report.diagnostics.nmsSuppressedComponents << " elapsed_seconds=" << report.elapsedSeconds << '\n';
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
        std::cout << "anchors=" << report.diagnostics.occupiedAnchors << " neighborhood_offsets=" << report.diagnostics.neighborhoodOffsets
                  << " generated_pairs=" << report.diagnostics.generatedPairs << " axis_rejected=" << report.diagnostics.axisRejectedPairs
                  << " searched=" << report.diagnostics.searchedPairs << " successful=" << report.diagnostics.successfulPaths
                  << " no_path=" << report.diagnostics.noPathPairs << " sampling_batches=" << report.samplingCoordinateBatches
                  << " sampled_voxels=" << report.sampledVoxels << " peak_batch_voxels=" << report.peakCoordinateBatchVoxels
                  << " evaluated_dp_nodes=" << report.evaluatedDpNodes << " prepared_geometry_bytes=" << report.preparedGeometryBytes
                  << " peak_search_transient_bytes=" << report.peakSearchTransientBytes << " estimated_peak_owned_bytes=" << report.estimatedPeakOwnedBytes
                  << " candidate_generation_workers=" << report.candidateGenerationWorkers
                  << " candidate_preparation_workers=" << report.candidateWorkers << " candidate_generation_seconds=" << report.candidateGenerationSeconds
                  << " candidate_generation_cpu_seconds=" << report.candidateGenerationCpuSeconds << " preparation_seconds=" << report.preparationSeconds
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
            printScores("fiberlet_loss_per_prediction_voxel_accepted", statistics.acceptedLossDensities);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiberlets: " << error.what() << '\n';
        return 1;
    }
}
