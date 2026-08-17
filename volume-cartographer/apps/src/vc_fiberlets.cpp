#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

enum class Command {
    Anchors,
    AnchorBenchmark,
    Benchmark,
    Paths,
    FiberletReplay,
};

bool isReplayCommand(Command command)
{
    return command == Command::FiberletReplay;
}

bool needsPathExtraction(Command command)
{
    return command == Command::Paths || command == Command::Benchmark || isReplayCommand(command);
}

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
    std::optional<double> peakSigmaBaseVoxels;
    std::optional<double> peakAxialSigmaBaseVoxels;
    std::optional<double> peakStepBaseVoxels;
    std::optional<double> localWindowBaseVoxels;
    std::optional<double> baseVoxelSizeUm;
    std::optional<double> replayLengthBaseVoxels;
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
              << " paths <fiber.lasagna.json-or-url> <anchors.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "  " << executable
              << " fiberlet-replay <fiber.lasagna.json-or-url> <fiber.json> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "Common options:\n"
              << "  --threads N                   decode/search workers [hardware]\n"
              << "  --cache-gib N                 decoded chunk cache budget [0.5]\n"
              << "  --remote-cache-dir PATH       required for direct remote manifests\n"
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
              << "  --merge-angle-deg N           maximum duplicate-axis angle [10]\n"
              << "  --merge-abs-loss N            maximum normalized merge loss [0.01]\n"
              << "  --merge-rel-loss N            maximum relative merge loss [0.05]\n"
              << "  --maximum-seeds N             deterministic PCA seed count [8]\n"
              << "  --maximum-iterations N        assignment/PCA iteration limit [64]\n"
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
              << "  --stats                       print path-count and score statistics\n"
              << "  --no-slices                   skip central presence-slice outputs\n";
    std::cerr << "\nReplay options:\n"
              << "  --fail N                      dense-reference failure distance in base voxels [20]\n"
              << "  --length N                    compared reference length in base voxels [full]\n"
              << "  --vis                         write indexed local failure visualizations\n"
              << "  --along N                     replay visualization half-width [128]; benchmark length [full]\n"
              << "  --radius N                    extraction tube radius in base voxels [64]\n"
              << "  --match-refine N              forward match refinement in trace steps [1]\n"
              << "  --inference-scaledown-power N prediction scaledown relative to trace voxels [2]\n";
    std::cerr << "  --beam N                      graph replay beam width [16]\n"
              << "  --lookahead N                 graph replay lookahead edges [3]\n";
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
        } else if (argument == "--normal-manifest" && needsPathExtraction(options.command)) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
        } else if (argument == "--fail" && isReplayCommand(options.command)) {
            options.failureThresholdBaseVoxels = parseDouble(valueAfter(index, argc, argv, "fail"), "fail");
        } else if (argument == "--length" && isReplayCommand(options.command)) {
            options.replayLengthBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "length"), "length");
        } else if (argument == "--vis" && isReplayCommand(options.command)) {
            options.writeReplayVisualizations = true;
        } else if (argument == "--along" && (isReplayCommand(options.command) || options.command == Command::Benchmark)) {
            options.alongBaseVoxels = parseDouble(valueAfter(index, argc, argv, "along"), "along");
            options.alongSpecified = true;
        } else if (argument == "--radius" && (isReplayCommand(options.command) || options.command == Command::Benchmark)) {
            options.radiusBaseVoxels = parseDouble(valueAfter(index, argc, argv, "radius"), "radius");
        } else if (argument == "--match-refine" && isReplayCommand(options.command)) {
            options.matchRefineSteps = parseDouble(valueAfter(index, argc, argv, "match-refine"), "match-refine");
        } else if (argument == "--inference-scaledown-power" && isReplayCommand(options.command)) {
            options.inferenceScaledownPower =
                parseInt(valueAfter(index, argc, argv, "inference-scaledown-power"), "inference-scaledown-power");
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
        } else if (argument == "--stats" && options.command == Command::Paths) {
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
        } else if (argument == "--beam" && options.command == Command::FiberletReplay) {
            const int value = parseInt(valueAfter(index, argc, argv, "beam"), "beam");
            if (value < 1)
                fail("--beam must be positive");
            options.graphReplay.beamWidth = static_cast<size_t>(value);
        } else if (argument == "--lookahead" && options.command == Command::FiberletReplay) {
            const int value = parseInt(valueAfter(index, argc, argv, "lookahead"), "lookahead");
            if (value < 1)
                fail("--lookahead must be positive");
            options.graphReplay.lookaheadEdges = static_cast<size_t>(value);
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
    if (isReplayCommand(options.command)) {
        if (!(options.failureThresholdBaseVoxels >= 0.0) || !(options.alongBaseVoxels > 0.0) || !(options.radiusBaseVoxels > 0.0) ||
            !(options.matchRefineSteps >= 0.0) ||
            (options.replayLengthBaseVoxels.has_value() &&
             (!(*options.replayLengthBaseVoxels > 0.0) ||
              !std::isfinite(*options.replayLengthBaseVoxels))) ||
            options.inferenceScaledownPower < 0 || options.inferenceScaledownPower > 30) {
            fail("fiber-replay options are outside their valid range");
        }
        vc::fiber_tracer::cli::validateTraceOptions(options.trace);
        if ((options.seenTraceOptions.beamWidth && options.trace.beamWidth != 1) ||
            (options.seenTraceOptions.beamLookahead && options.trace.beamLookaheadSteps != 1)) {
            fail("fiber-replay only supports --beam-width 1 and --beam-lookahead-steps 1");
        }
    }
    if (options.command == Command::Benchmark &&
        ((options.alongSpecified && !(options.alongBaseVoxels > 0.0)) ||
         !(options.radiusBaseVoxels > 0.0)))
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
    options.anchors.nmsMaximumAngleDegrees = options.anchors.mergeMaximumAngleDegrees;
    vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
    return cellSideBase;
}

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

struct TubeExtractionResult {
    vc::fiber_tracer::FiberReplayTube tube;
    vc::fiber_tracer::FiberAnchorExtractionReport anchors;
    vc::fiber_tracer::FiberletPathReport paths;
    double anchorSeconds = 0.0;
    double anchorCpuSeconds = 0.0;
    double fiberletSeconds = 0.0;
    double fiberletCpuSeconds = 0.0;
};

void printTubeExtractionProfile(
    std::ostream& output,
    const TubeExtractionResult& extraction)
{
    const auto previousPrecision = output.precision();
    const auto& anchor = extraction.anchors.profile;
    const auto& fit = anchor.fit;
    const auto& paths = extraction.paths;
    const double anchorProfiledSeconds =
        anchor.setupSeconds + anchor.tilePlanningSeconds +
        anchor.cellProcessingSeconds + anchor.selectionSeconds +
        anchor.initialDiagnosticsSeconds +
        anchor.duplicateSuppressionSeconds + anchor.finalizationSeconds;
    const double fiberletProfiledSeconds =
        paths.candidateGenerationSeconds + paths.preparationSeconds +
        paths.cornerMergeSeconds + paths.predictionSamplingSeconds +
        paths.normalSamplingSeconds + paths.samplingMaterializationSeconds +
        paths.searchSeconds;
    const double fitProfiledWorkSeconds =
        fit.setupWorkSeconds + fit.seedGenerationWorkSeconds +
        fit.seedPairRefinementWorkSeconds + fit.initializationWorkSeconds +
        fit.localRefinementWorkSeconds + fit.peakSearchWorkSeconds +
        fit.finalEvaluationWorkSeconds;
    output << std::setprecision(17)
           << "fiberlet_extraction_profile version=2"
           << " anchor_elapsed_seconds=" << extraction.anchors.elapsedSeconds
           << " anchor_cpu_seconds=" << anchor.elapsedCpuSeconds
           << " anchor_profiled_seconds=" << anchorProfiledSeconds
           << " anchor_residual_seconds="
           << std::max(0.0, extraction.anchors.elapsedSeconds - anchorProfiledSeconds)
           << " anchor_selected_cells=" << anchor.selectedCells
           << " anchor_context_cells=" << anchor.contextCells
           << " anchor_work_cells=" << anchor.workCells
           << " anchor_tiles=" << anchor.tiles
           << " anchor_workers=" << anchor.workers
           << " anchor_sampler_calls=" << anchor.predictionSamplerCalls
           << " anchor_submitted_prediction_voxels="
           << anchor.submittedPredictionVoxels
           << " anchor_candidate_observations=" << anchor.candidateObservations
           << " anchor_retained_observations=" << anchor.retainedObservations
           << " anchor_gradient_attempts=" << anchor.gradientAttempts
           << " anchor_valid_gradients=" << anchor.validGradients
           << " anchor_retain_predicate_calls=" << anchor.retainPredicateCalls
           << " anchor_fit_iterations=" << anchor.fitIterations
           << " anchor_setup_seconds=" << anchor.setupSeconds
           << " anchor_tile_planning_seconds=" << anchor.tilePlanningSeconds
           << " anchor_cell_processing_seconds=" << anchor.cellProcessingSeconds
           << " anchor_cell_processing_cpu_seconds="
           << anchor.cellProcessingCpuSeconds
           << " anchor_coordinate_construction_work_seconds="
           << anchor.coordinateConstructionWorkSeconds
           << " anchor_prediction_sampling_work_seconds="
           << anchor.predictionSamplingWorkSeconds
           << " anchor_observation_construction_work_seconds="
           << anchor.observationConstructionWorkSeconds
           << " anchor_fitting_work_seconds=" << anchor.fittingWorkSeconds
           << " anchor_fit_invocations=" << fit.invocations
           << " anchor_fit_nonempty_cells=" << fit.nonemptyCells
           << " anchor_fit_weighted_observations=" << fit.weightedObservations
           << " anchor_fit_seeds=" << fit.seeds
           << " anchor_fit_seed_generation_observation_visits="
           << fit.seedGenerationObservationVisits
           << " anchor_fit_seed_pairs=" << fit.seedPairs
           << " anchor_fit_seed_pair_iterations=" << fit.seedPairIterations
           << " anchor_fit_seed_assignment_observation_visits="
           << fit.seedAssignmentObservationVisits
           << " anchor_fit_seed_tensor_observation_visits="
           << fit.seedTensorObservationVisits
           << " anchor_fit_seed_objective_observation_visits="
           << fit.seedObjectiveObservationVisits
           << " anchor_fit_initialization_observation_visits="
           << fit.initializationObservationVisits
           << " anchor_fit_local_refinement_attempts="
           << fit.localRefinementAttempts
           << " anchor_fit_local_refinement_accepted_steps="
           << fit.localRefinementAcceptedSteps
           << " anchor_fit_backtracking_evaluations="
           << fit.backtrackingEvaluations
           << " anchor_fit_local_tensor_observation_visits="
           << fit.localTensorObservationVisits
           << " anchor_fit_local_centroid_observation_visits="
           << fit.localCentroidObservationVisits
           << " anchor_fit_refined_evaluation_observation_visits="
           << fit.refinedEvaluationObservationVisits
           << " anchor_fit_peak_components=" << fit.peakComponents
           << " anchor_fit_peak_preparation_observation_visits="
           << fit.peakPreparationObservationVisits
           << " anchor_fit_peak_grid_response_requests="
           << fit.peakGridResponseRequests
           << " anchor_fit_peak_computed_grid_responses="
           << fit.peakComputedGridResponses
           << " anchor_fit_peak_acceptance_responses="
           << fit.peakAcceptanceResponses
           << " anchor_fit_peak_response_observation_visits="
           << fit.peakResponseObservationVisits
           << " anchor_fit_final_evaluation_observation_visits="
           << fit.finalEvaluationObservationVisits
           << " anchor_fit_setup_work_seconds=" << fit.setupWorkSeconds
           << " anchor_fit_seed_generation_work_seconds="
           << fit.seedGenerationWorkSeconds
           << " anchor_fit_seed_pair_refinement_work_seconds="
           << fit.seedPairRefinementWorkSeconds
           << " anchor_fit_initialization_work_seconds="
           << fit.initializationWorkSeconds
           << " anchor_fit_local_refinement_work_seconds="
           << fit.localRefinementWorkSeconds
           << " anchor_fit_peak_search_work_seconds="
           << fit.peakSearchWorkSeconds
           << " anchor_fit_final_evaluation_work_seconds="
           << fit.finalEvaluationWorkSeconds
           << " anchor_fit_profiled_work_seconds=" << fitProfiledWorkSeconds
           << " anchor_fit_residual_work_seconds="
           << std::max(0.0, anchor.fittingWorkSeconds - fitProfiledWorkSeconds)
           << " anchor_selection_seconds=" << anchor.selectionSeconds
           << " anchor_initial_diagnostics_seconds="
           << anchor.initialDiagnosticsSeconds
           << " anchor_duplicate_suppression_seconds="
           << anchor.duplicateSuppressionSeconds
           << " anchor_finalization_seconds=" << anchor.finalizationSeconds
           << " fiberlet_elapsed_seconds=" << paths.elapsedSeconds
           << " fiberlet_cpu_seconds=" << paths.elapsedCpuSeconds
           << " fiberlet_profiled_seconds=" << fiberletProfiledSeconds
           << " fiberlet_residual_seconds="
           << std::max(0.0, paths.elapsedSeconds - fiberletProfiledSeconds)
           << " fiberlet_candidate_predicate_calls="
           << paths.candidatePointPredicateCalls
           << " fiberlet_lattice_node_positions=" << paths.latticeNodePositions
           << " fiberlet_corridor_segment_tests=" << paths.corridorSegmentTests
           << " fiberlet_corridor_accepted_nodes=" << paths.corridorAcceptedNodes
           << " fiberlet_node_predicate_calls=" << paths.nodePointPredicateCalls
           << " fiberlet_retained_search_nodes=" << paths.retainedSearchNodes
           << " fiberlet_corner_insertion_attempts="
           << paths.interpolationCornerInsertions
           << " fiberlet_unique_sampled_voxels=" << paths.sampledVoxels
           << " fiberlet_interpolated_scoring_points="
           << paths.interpolatedScoringPoints
           << " fiberlet_dp_node_index_entries=" << paths.dpNodeIndexEntries
           << " fiberlet_dp_transition_lookups=" << paths.dpTransitionLookups
           << " fiberlet_dp_reached_state_visits=" << paths.dpReachedStateVisits
           << " fiberlet_dp_relaxations=" << paths.dpRelaxations
           << " fiberlet_candidate_generation_seconds="
           << paths.candidateGenerationSeconds
           << " fiberlet_candidate_generation_cpu_seconds="
           << paths.candidateGenerationCpuSeconds
           << " fiberlet_preparation_seconds=" << paths.preparationSeconds
           << " fiberlet_preparation_cpu_seconds="
           << paths.preparationCpuSeconds
           << " fiberlet_preparation_geometry_work_seconds="
           << paths.preparationGeometryWorkSeconds
           << " fiberlet_node_enumeration_work_seconds="
           << paths.preparationNodeEnumerationWorkSeconds
           << " fiberlet_corner_collection_work_seconds="
           << paths.preparationCornerCollectionWorkSeconds
           << " fiberlet_corner_merge_seconds=" << paths.cornerMergeSeconds
           << " fiberlet_corner_merge_cpu_seconds="
           << paths.cornerMergeCpuSeconds
           << " fiberlet_prediction_sampling_seconds="
           << paths.predictionSamplingSeconds
           << " fiberlet_prediction_sampling_cpu_seconds="
           << paths.predictionSamplingCpuSeconds
           << " fiberlet_normal_sampling_seconds=" << paths.normalSamplingSeconds
           << " fiberlet_normal_sampling_cpu_seconds="
           << paths.normalSamplingCpuSeconds
           << " fiberlet_materialization_seconds="
           << paths.samplingMaterializationSeconds
           << " fiberlet_materialization_cpu_seconds="
           << paths.samplingMaterializationCpuSeconds
           << " fiberlet_scoring_index_seconds=" << paths.scoringIndexSeconds
           << " fiberlet_scoring_index_cpu_seconds="
           << paths.scoringIndexCpuSeconds
           << " fiberlet_interpolation_materialization_seconds="
           << paths.interpolationMaterializationSeconds
           << " fiberlet_interpolation_materialization_cpu_seconds="
           << paths.interpolationMaterializationCpuSeconds
           << " fiberlet_search_seconds=" << paths.searchSeconds
           << " fiberlet_search_cpu_seconds=" << paths.searchCpuSeconds
           << " fiberlet_node_index_work_seconds="
           << paths.searchNodeIndexWorkSeconds
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
    const vc::lasagna::LasagnaNormalSampler& normalSampler)
{
    if (!(endArcBase > beginArcBase))
        throw std::invalid_argument("fiberlet extraction interval must have positive length");
    TubeExtractionResult result;
    result.tube = vc::fiber_tracer::makeFiberReplayTube(
        referenceBase, 0.5 * (beginArcBase + endArcBase), 0.5 * (endArcBase - beginArcBase), radiusBaseVoxels, grid, options.anchors.cellSizePredictionVoxels);
    const auto anchorStart = std::chrono::steady_clock::now();
    const double anchorCpuStart = processCpuSeconds();
    result.anchors = vc::fiber_tracer::extractFiberAnchorsForCells(
        grid,
        options.anchors,
        [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
        result.tube.cellsZYX,
        [&](const vc::fiber_tracer::FiberAnchor& anchor) {
            const double distance = result.tube.distanceToBasePoint(anchor.positionPredictionXYZ * grid.predictionToBaseScale);
            return vc::fiber_tracer::FiberAnchorRetainEvaluation{
                distance <= result.tube.radiusBaseVoxels + 1.0e-12,
                distance,
                result.tube.radiusBaseVoxels,
            };
        },
        printAnchorProgress);
    result.anchorSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - anchorStart).count();
    result.anchorCpuSeconds = processCpuSeconds() - anchorCpuStart;

    vc::fiber_tracer::LoadedFiberAnchorArtifact loaded{result.anchors, {}};
    const auto containmentQuery = result.tube.makePredictionContainmentQuery(
        grid.predictionToBaseScale);
    const auto fiberletStart = std::chrono::steady_clock::now();
    const double fiberletCpuStart = processCpuSeconds();
    result.paths = vc::fiber_tracer::traceFiberletPaths(
        loaded,
        grid,
        options.paths,
        [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
        normalSampler,
        printFiberletProgress,
        [&](const cv::Vec3d& pointPrediction) {
            return containmentQuery.containsPredictionPoint(pointPrediction);
        });
    result.fiberletSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fiberletStart).count();
    result.fiberletCpuSeconds = processCpuSeconds() - fiberletCpuStart;
    return result;
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
                reference,
                fiber.controlPointLineIndices.front(),
                options.alongSpecified
                    ? std::optional<double>{options.alongBaseVoxels}
                    : std::nullopt);
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

        if (isReplayCommand(options.command)) {
            const auto traceSetupStart = std::chrono::steady_clock::now();
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
            if (fiber.controlPointLineIndices.empty() ||
                fiber.controlPointLineIndices.front() >=
                    fiber.linePointsXyzBase.size()) {
                fail("fiber replay fiber has no valid first control point");
            }
            const auto reference = vc::fiber_tracer::makePolylineArcGeometry(
                fiber.linePointsXyzBase);
            const auto interval =
                vc::fiber_tracer::selectForwardPolylineArcInterval(
                    reference,
                    fiber.controlPointLineIndices.front(),
                    options.replayLengthBaseVoxels);
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

            std::cerr << "fiber_replay_stage stage=trace_setup status=completed"
                      << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - traceSetupStart).count() << '\n';
            const auto referenceGeometry = vc::fiber_tracer::slicePolylineArc(reference, startArc, endArc);
            resolveAnchorConfig(options, grid);

            vc::fiber_tracer::FiberAnchorArtifactInfo baseAnchorArtifact;
            baseAnchorArtifact.sourceLocator = datasetLocator(dataset);
            baseAnchorArtifact.manifestContentHash = fileHash(dataset.manifest().manifestPath);
            baseAnchorArtifact.glyphLengthBaseVoxels = options.glyphLengthBaseVoxels;
            baseAnchorArtifact.baseVoxelSizeUm = options.baseVoxelSizeUm;

            vc::lasagna::LasagnaDatasetOpenOptions canonicalNormalOptions;
            canonicalNormalOptions.workingToBaseScale = grid.predictionToBaseScale;
            canonicalNormalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto canonicalNormalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, canonicalNormalOptions);
            const vc::lasagna::LasagnaNormalSampler
                canonicalNormalSampler(canonicalNormalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});
            std::cerr << "fiber_replay_stage stage=full_extraction status=started\n";
            auto fullExtraction =
                extractTubeFiberlets(fiber.linePointsXyzBase, startArc, endArc, options.radiusBaseVoxels, grid, options, field, canonicalNormalSampler);
            const auto& fullTube = fullExtraction.tube;
            const size_t fullAnchorCount = fullExtraction.anchors.diagnostics.oneAnchorCells + 2 * fullExtraction.anchors.diagnostics.twoAnchorCells;
            std::cerr << "fiber_replay_stage stage=full_extraction status=completed"
                      << " cells=" << fullTube.cellsZYX.size() << " anchors=" << fullAnchorCount
                      << " anchor_seconds=" << fullExtraction.anchorSeconds << " fiberlet_seconds=" << fullExtraction.fiberletSeconds
                      << " searched=" << fullExtraction.paths.diagnostics.searchedPairs << " accepted=" << fullExtraction.paths.diagnostics.successfulPaths
                      << " sampling_batches=" << fullExtraction.paths.samplingCoordinateBatches
                      << " sampled_voxels=" << fullExtraction.paths.sampledVoxels << " peak_batch_voxels=" << fullExtraction.paths.peakCoordinateBatchVoxels
                      << " evaluated_dp_nodes=" << fullExtraction.paths.evaluatedDpNodes << '\n';
            printTubeExtractionProfile(std::cerr, fullExtraction);
            auto fullPaths = std::move(fullExtraction.paths);
            const auto graphStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=graph status=started\n";
            const auto graph = vc::fiber_tracer::buildFiberletGraph(fullPaths);
            std::cerr << "fiber_replay_stage stage=graph status=completed"
                      << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - graphStart).count()
                      << " nodes=" << graph.nodes.size() << " edges=" << graph.edges.size() << " transitions=" << graph.transitions.size() << '\n';
            fullPaths = {};

            options.graphReplay.errorThresholdBaseVoxels = options.failureThresholdBaseVoxels;
            options.graphReplay.matchRefineSteps = options.matchRefineSteps;
            options.graphReplay.minimumResetAdvanceBaseVoxels = nominalStepBaseVoxels;
            options.graphReplay.referenceBeginArcBase = startArc;
            options.graphReplay.referenceEndArcBase = endArc;

            std::mutex outputMutex;
            size_t greedyFailureCount = 0;
            size_t fiberletFailureCount = 0;
            const auto failurePrinter = [&](vc::fiber_tracer::FiberReplayTracer tracer) {
                return [&, tracer](const vc::fiber_tracer::FiberReplayFailure& event) {
                    std::lock_guard lock(outputMutex);
                    auto& count = tracer == vc::fiber_tracer::FiberReplayTracer::Greedy ? greedyFailureCount : fiberletFailureCount;
                    count = event.index + 1;
                    std::cerr << std::setprecision(17) << "fiber_replay_failure tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer)
                              << " index=" << event.index << " reference_arc_base=" << event.referenceArcBase
                              << " reference_arc_fraction=" << event.referenceArcFraction << " reason=" << event.reason << " error_base_voxels=";
                    if (event.errorBaseVoxels.has_value())
                        std::cerr << *event.errorBaseVoxels;
                    else
                        std::cerr << "n/a";
                    std::cerr << " greedy_failures=" << greedyFailureCount << " fiberlet_failures=" << fiberletFailureCount << '\n';
                };
            };

            const auto traceStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=parallel_trace status=started\n";
            auto greedyFuture = std::async(std::launch::async, [&]() {
                return vc::fiber_tracer::traceFiberReplay(
                    traceField,
                    replayRequest,
                    &traceNormalSampler,
                    [&](const vc::fiber_tracer::FiberTraceProgress& event) {
                        if (event.step == event.maxSteps || event.step % 100 == 0) {
                            std::lock_guard lock(outputMutex);
                            std::cerr << "fiber_replay_progress tracer=greedy step=" << event.step << '/' << event.maxSteps
                                      << " reason=" << event.reason << '\n';
                        }
                    },
                    failurePrinter(vc::fiber_tracer::FiberReplayTracer::Greedy));
            });
            auto fiberletFuture = std::async(std::launch::async, [&]() {
                return vc::fiber_tracer::traceFiberletGraphReplay(graph, fiber.linePointsXyzBase, options.graphReplay, failurePrinter(vc::fiber_tracer::FiberReplayTracer::Fiberlet));
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
            if (traceError)
                std::rethrow_exception(traceError);
            std::cerr << "fiber_replay_stage stage=parallel_trace status=completed"
                      << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - traceStart).count()
                      << " greedy_failures=" << greedyReplay->failures.size() << " fiberlet_failures=" << fiberletReplay->failures.size() << '\n';

            vc::fiber_tracer::FiberReplayBundleInput bundle;
            bundle.request = replayRequest;
            bundle.greedyReplay = std::move(*greedyReplay);
            bundle.fiberletReplay = std::move(*fiberletReplay);
            bundle.fiberletReplayConfig = options.graphReplay;
            bundle.requestedLengthBaseVoxels =
                options.replayLengthBaseVoxels;
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
            bundle.requestedTraceConfig = vc::fiber_tracer::cli::traceConfigJson(requestedTrace);
            bundle.effectiveTraceConfig = vc::fiber_tracer::cli::traceConfigJson(effectiveTrace);

            if (options.writeReplayVisualizations) {
                const auto addVisualizations = [&](vc::fiber_tracer::FiberReplayTracer tracer, const auto& failures) {
                    for (const auto& failure : failures) {
                        const auto visualStart = std::chrono::steady_clock::now();
                        std::cerr << "fiber_replay_stage stage=visualization status=started"
                                  << " tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer) << " index=" << failure.index << '\n';
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
                            canonicalNormalSampler);
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
                        bundle.visualizations.push_back(std::move(visual));
                        std::cerr << "fiber_replay_stage stage=visualization status=completed"
                                  << " tracer=" << vc::fiber_tracer::fiberReplayTracerName(tracer) << " index=" << failure.index
                                  << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - visualStart).count()
                                  << '\n';
                    }
                };
                addVisualizations(vc::fiber_tracer::FiberReplayTracer::Greedy, bundle.greedyReplay.failures);
                addVisualizations(vc::fiber_tracer::FiberReplayTracer::Fiberlet, bundle.fiberletReplay.failures);
            }

            const auto publishStart = std::chrono::steady_clock::now();
            std::cerr << "fiber_replay_stage stage=publish status=started\n";
            const auto resultBundle = vc::fiber_tracer::writeFiberReplayBundle(options.outputDirectory, bundle);
            std::cerr << "fiber_replay_stage stage=publish status=completed"
                      << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - publishStart).count() << '\n';
            for (const auto& visualization : resultBundle.at("visualizations")) {
                std::cout << "fiber_replay_visualization"
                          << " tracer=" << visualization.at("tracer")
                          << " tracer_failure_index="
                          << visualization.at("tracer_failure_index")
                          << " manifest="
                          << std::filesystem::absolute(
                                 options.outputDirectory /
                                 visualization.at("manifest").at("path")
                                     .get<std::string>())
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
                  << " evaluated_dp_nodes=" << report.evaluatedDpNodes
                  << " prepared_geometry_bytes=" << report.preparedGeometryBytes
                  << " peak_search_transient_bytes=" << report.peakSearchTransientBytes
                  << " estimated_peak_owned_bytes=" << report.estimatedPeakOwnedBytes
                  << " candidate_workers=" << report.candidateWorkers << " candidate_seconds=" << report.candidateGenerationSeconds
                  << " preparation_seconds=" << report.preparationSeconds << " search_seconds=" << report.searchSeconds
                  << " elapsed_seconds=" << report.elapsedSeconds << '\n';
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
