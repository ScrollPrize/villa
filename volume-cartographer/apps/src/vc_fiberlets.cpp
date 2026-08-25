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
#include "utils/thread_pool.hpp"

#include <zstd.h>

#include <algorithm>
#include <atomic>
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
#include <string_view>
#include <thread>
#include <utility>
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
    PreprocessVolume,
    ChunkRouteStats,
};

enum class ChunkRouteMode {
    Stats,
    Staged,
};

struct ChunkRouteStageSpec {
    int sideBaseVoxels = 0;
    std::array<int, 3> offsetBaseXYZ{0, 0, 0};
};

bool isReplayCommand(Command command)
{
    return command == Command::FiberletReplay;
}

bool isWholeVolumeCommand(Command command)
{
    return command == Command::PreprocessVolume;
}

bool isChunkRouteStatsCommand(Command command)
{
    return command == Command::ChunkRouteStats;
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
    return command == Command::Paths || command == Command::Benchmark || isReplayCommand(command) || isQuantizationCommand(command) ||
           isWholeVolumeCommand(command) || isChunkRouteStatsCommand(command);
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
    bool steppedReplayCostOptionSpecified = false;
    int storageChunkSideBaseVoxels = 512;
    std::optional<std::string> quantizationScenario;
    std::filesystem::path anchorCacheRoot;
    std::filesystem::path fiberletCacheRoot;
    std::filesystem::path sourceContextPath;
    bool eagerGraphReplay = false;
    std::size_t storageCompressionChunks = 0;
    std::uint64_t storageCompressionSeed = 1;
    int storageCompressionChunkSideBaseVoxels = 512;
    vc::fiber_tracer::cli::SeenOptions seenTraceOptions;
    std::array<int, 3> analysisChunkMinimumBaseXYZ{0, 0, 0};
    bool analysisChunkSpecified = false;
    int analysisChunkSizeBaseVoxels = 256;
    int analysisRegionSizeBaseVoxels = 0;
    ChunkRouteMode analysisMode = ChunkRouteMode::Stats;
    std::vector<ChunkRouteStageSpec> analysisStages;
    float analysisMaximumJoinAngleDegrees = 45.0F;
    std::optional<vc::fiber_tracer::FiberletChunkRouteEdgeCostView>
        analysisEdgeCostView;
    std::size_t analysisMaximumStatesPerEntry = 1'000'000;
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
              << "  " << executable
              << " preprocess-volume <fiber.lasagna.json-or-url> <output.zarr>"
                 " --normal-manifest <lasagna.json-or-url> [options]\n\n"
              << "  " << executable
              << " chunk-route-stats <fiber.lasagna.json-or-url> <output-dir>"
                 " --normal-manifest <lasagna.json-or-url> --chunk X,Y,Z [options]\n\n"
              << "Common options:\n"
              << "  --threads N                   decode/search workers [hardware]\n"
              << "  --cache-gib N                 decoded chunk cache budget [0.5]\n"
              << "  --remote-cache-dir PATH       required for direct remote manifests\n"
              << "  --source-context PATH         stable manager/Atlas source identities (preprocess-volume)\n"
              << "  --stats                       print detailed path/replay diagnostics\n"
              << "  --base-voxel-size-um N        optional physical reporting metadata\n\n"
              << "Chunk-route options:\n"
              << "  --chunk X,Y,Z                region minimum in base voxels\n"
              << "  --chunk-size N               cubic analysis-box side [256]\n"
              << "  --region-size N              cubic region side [chunk-size]\n"
              << "  --mode stats|staged          reduction mode [stats]\n"
              << "  --stage N,OX,OY,OZ           repeatable stage side/XYZ offset\n"
              << "  --join-angle N               maximum anchor join angle [45]\n"
              << "  --cost-profile stored|sqrt-u16 analysis cost view [sqrt-u16]\n"
              << "  --max-states N               search-state limit per entry [1000000]\n\n"
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
              << "  --cost-mode MODE              fiberlet or stepped [fiberlet]\n"
              << "  --cost-weight W               stepped: geometric weight per base voxel (0,1] [1]\n"
              << "  --cost-delay N                stepped: full-weight distance before decay [0]\n"
              << "  --cost-step N                 stepped: integration step in base voxels [16]\n"
              << "  --cost-profile-weight A       stepped: subsegment density blend in [0,1] [1]\n"
              << "  --decision-window BEGIN,END   retain --stats beam details only in this base-arc window; repeatable\n"
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
    std::cerr << "\nChunk-route statistics options:\n"
              << "  --chunk X,Y,Z                 analysis-box minimum in base voxels\n"
              << "  --chunk-size N                analysis chunk side in base voxels [256]\n"
              << "  --region-size N               selected staged bbox side\n"
              << "  --mode stats|staged           reduction mode [stats]\n"
              << "  --stage N,OX,OY,OZ            repeatable stage side/XYZ offset\n"
              << "  --join-angle N                regular strict maximum join angle\n"
              << "  --cost-profile PROFILE        stored or fixed sqrt uint16 [sqrt-u16]\n"
              << "  --max-states N                exact generated-state limit per entry [1000000]\n";
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

std::pair<double, double> parseDoublePair(const std::string& text, const char* name)
{
    const auto separator = text.find(',');
    if (separator == std::string::npos || text.find(',', separator + 1) != std::string::npos)
        fail(std::string("--") + name + " requires BEGIN,END");
    const double begin = parseDouble(text.substr(0, separator), name);
    const double end = parseDouble(text.substr(separator + 1), name);
    if (!(begin >= 0.0) || !(end >= begin))
        fail(std::string("--") + name + " requires 0 <= BEGIN <= END");
    return {begin, end};
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

std::array<int, 3> parseIntTriple(
    const std::string& text, const char* name)
{
    std::array<int, 3> result{};
    std::stringstream input(text);
    std::string token;
    for (std::size_t axis = 0; axis < result.size(); ++axis) {
        if (!std::getline(input, token, ',') || token.empty())
            fail(std::string("--") + name + " requires X,Y,Z");
        result[axis] = parseInt(token, name);
    }
    if (std::getline(input, token, ','))
        fail(std::string("--") + name + " requires exactly three integers");
    return result;
}

ChunkRouteStageSpec parseChunkRouteStage(const std::string& text)
{
    std::array<int, 4> values{};
    std::stringstream input(text);
    std::string token;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::getline(input, token, ',') || token.empty())
            fail("--stage requires SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z");
        values[index] = parseInt(token, "stage");
    }
    if (std::getline(input, token, ',')) {
        fail("--stage requires exactly four integers");
    }
    return {values[0], {values[1], values[2], values[3]}};
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
    } else if (command == "preprocess-volume") {
        options.command = Command::PreprocessVolume;
        options.manifestLocation = argv[2];
        options.outputDirectory = argv[3];
        firstOption = 4;
    } else if (command == "chunk-route-stats") {
        options.command = Command::ChunkRouteStats;
        options.manifestLocation = argv[2];
        options.outputDirectory = argv[3];
        options.analysisEdgeCostView = vc::fiber_tracer::
            FiberletChunkRouteEdgeCostView::SqrtUint16Max256;
        firstOption = 4;
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
        } else if (argument == "--chunk" &&
                   isChunkRouteStatsCommand(options.command)) {
            options.analysisChunkMinimumBaseXYZ = parseIntTriple(
                valueAfter(index, argc, argv, "chunk"), "chunk");
            options.analysisChunkSpecified = true;
        } else if (argument == "--chunk-size" &&
                   isChunkRouteStatsCommand(options.command)) {
            options.analysisChunkSizeBaseVoxels = parseInt(
                valueAfter(index, argc, argv, "chunk-size"), "chunk-size");
        } else if (argument == "--region-size" &&
                   isChunkRouteStatsCommand(options.command)) {
            options.analysisRegionSizeBaseVoxels = parseInt(
                valueAfter(index, argc, argv, "region-size"), "region-size");
        } else if (argument == "--mode" &&
                   isChunkRouteStatsCommand(options.command)) {
            const auto value = valueAfter(index, argc, argv, "mode");
            if (value == "stats")
                options.analysisMode = ChunkRouteMode::Stats;
            else if (value == "staged")
                options.analysisMode = ChunkRouteMode::Staged;
            else
                fail("--mode must be stats or staged");
        } else if (argument == "--stage" &&
                   isChunkRouteStatsCommand(options.command)) {
            options.analysisStages.push_back(parseChunkRouteStage(
                valueAfter(index, argc, argv, "stage")));
        } else if (argument == "--join-angle" &&
                   isChunkRouteStatsCommand(options.command)) {
            options.analysisMaximumJoinAngleDegrees = static_cast<float>(
                parseDouble(valueAfter(index, argc, argv, "join-angle"),
                            "join-angle"));
        } else if (argument == "--cost-profile" &&
                   isChunkRouteStatsCommand(options.command)) {
            const auto value = valueAfter(index, argc, argv, "cost-profile");
            if (value == "stored") {
                options.analysisEdgeCostView =
                    vc::fiber_tracer::FiberletChunkRouteEdgeCostView::Stored;
            } else if (value == "sqrt-u16") {
                options.analysisEdgeCostView = vc::fiber_tracer::
                    FiberletChunkRouteEdgeCostView::SqrtUint16Max256;
            } else {
                fail("--cost-profile must be stored or sqrt-u16");
            }
        } else if (argument == "--max-states" &&
                   isChunkRouteStatsCommand(options.command)) {
            const int value = parseInt(
                valueAfter(index, argc, argv, "max-states"), "max-states");
            if (value <= 0)
                fail("--max-states must be positive");
            options.analysisMaximumStatesPerEntry =
                static_cast<std::size_t>(value);
        } else if (argument == "--source-context" &&
                   isWholeVolumeCommand(options.command)) {
            options.sourceContextPath =
                valueAfter(index, argc, argv, "source-context");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(valueAfter(index, argc, argv, "base-voxel-size-um"), "base-voxel-size-um");
        } else if (argument == "--normal-manifest" && needsPathExtraction(options.command)) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
        } else if (argument == "--fail" && usesGraphReplayOptions(options.command)) {
            options.failureThresholdBaseVoxels = parseDouble(valueAfter(index, argc, argv, "fail"), "fail");
        } else if (argument == "--length" && usesGraphReplayOptions(options.command)) {
            options.replayLengthBaseVoxels = parseDouble(valueAfter(index, argc, argv, "length"), "length");
        } else if (argument == "--arc" && usesGraphReplayOptions(options.command)) {
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
        } else if (
            argument == "--anchor-cache" &&
            (usesGraphReplayOptions(options.command) || isWholeVolumeCommand(options.command) ||
             isChunkRouteStatsCommand(options.command))) {
            options.anchorCacheRoot = valueAfter(index, argc, argv, "anchor-cache");
        } else if (
            argument == "--fiberlet-cache" &&
            (usesGraphReplayOptions(options.command) || isChunkRouteStatsCommand(options.command))) {
            options.fiberletCacheRoot = valueAfter(index, argc, argv, "fiberlet-cache");
        } else if (argument == "--eager-graph" && isReplayCommand(options.command)) {
            options.eagerGraphReplay = true;
        } else if (argument == "--storage-compression-chunks" && isReplayCommand(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "storage-compression-chunks"), "storage-compression-chunks");
            if (value <= 0)
                fail("--storage-compression-chunks must be positive");
            options.storageCompressionChunks = static_cast<std::size_t>(value);
        } else if (argument == "--storage-compression-seed" && isReplayCommand(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "storage-compression-seed"), "storage-compression-seed");
            if (value < 0)
                fail("--storage-compression-seed must be nonnegative");
            options.storageCompressionSeed = static_cast<std::uint64_t>(value);
        } else if (argument == "--storage-compression-chunk-side" && isReplayCommand(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "storage-compression-chunk-side"), "storage-compression-chunk-side");
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
        } else if (
            argument == "--stats" && (options.command == Command::Paths || isReplayCommand(options.command) || isWholeVolumeCommand(options.command) ||
                                         isChunkRouteStatsCommand(options.command))) {
            options.printStats = true;
        } else if (argument == "--no-slices" && options.command == Command::Paths) {
            options.writePresenceSlices = false;
        } else if (argument == "--invalid-prediction-cost" &&
                   (needsPathExtraction(options.command) ||
                    isChunkRouteStatsCommand(options.command))) {
            options.paths.invalidPredictionCostPerVoxel =
                parseDouble(valueAfter(index, argc, argv, "invalid-prediction-cost"), "invalid-prediction-cost");
        } else if (argument == "--smoothness-weight" &&
                   (needsPathExtraction(options.command) ||
                    isChunkRouteStatsCommand(options.command))) {
            options.paths.smoothnessWeight = parseDouble(valueAfter(index, argc, argv, "smoothness-weight"), "smoothness-weight");
        } else if (argument == "--smoothness-normal-weight" &&
                   (needsPathExtraction(options.command) ||
                    isChunkRouteStatsCommand(options.command))) {
            options.paths.smoothnessNormalWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-normal-weight"), "smoothness-normal-weight");
        } else if (argument == "--smoothness-tangent-weight" &&
                   (needsPathExtraction(options.command) ||
                    isChunkRouteStatsCommand(options.command))) {
            options.paths.smoothnessTangentWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-tangent-weight"), "smoothness-tangent-weight");
        } else if (argument == "--smoothness-free-angle" &&
                   (needsPathExtraction(options.command) ||
                    isChunkRouteStatsCommand(options.command))) {
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
        } else if (argument == "--cost-mode" && usesGraphReplayOptions(options.command)) {
            const auto value = valueAfter(index, argc, argv, "cost-mode");
            if (value == "fiberlet") {
                options.graphReplay.costMode = vc::fiber_tracer::FiberletGraphReplayCostMode::Fiberlet;
            } else if (value == "stepped") {
                options.graphReplay.costMode = vc::fiber_tracer::FiberletGraphReplayCostMode::Stepped;
            } else {
                fail("--cost-mode must be fiberlet or stepped");
            }
        } else if (argument == "--cost-weight" && usesGraphReplayOptions(options.command)) {
            options.steppedReplayCostOptionSpecified = true;
            options.graphReplay.geometricCostWeightPerBaseVoxel =
                parseDouble(valueAfter(index, argc, argv, "cost-weight"), "cost-weight");
        } else if (argument == "--cost-delay" && usesGraphReplayOptions(options.command)) {
            options.steppedReplayCostOptionSpecified = true;
            options.graphReplay.geometricCostDelayBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "cost-delay"), "cost-delay");
        } else if (argument == "--cost-step" && usesGraphReplayOptions(options.command)) {
            options.steppedReplayCostOptionSpecified = true;
            options.graphReplay.costIntegrationStepBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "cost-step"), "cost-step");
        } else if (argument == "--cost-profile-weight" && usesGraphReplayOptions(options.command)) {
            options.steppedReplayCostOptionSpecified = true;
            options.graphReplay.costProfileWeight =
                parseDouble(valueAfter(index, argc, argv, "cost-profile-weight"), "cost-profile-weight");
        } else if (argument == "--decision-window" && isReplayCommand(options.command)) {
            options.graphReplay.decisionDiagnosticReferenceArcWindowsBase.push_back(
                parseDoublePair(valueAfter(index, argc, argv, "decision-window"), "decision-window"));
        } else if (argument == "--search-width" && usesGraphReplayOptions(options.command)) {
            const int value = parseInt(valueAfter(index, argc, argv, "search-width"), "search-width");
            if (value < 0)
                fail("--search-width must be non-negative");
            options.graphReplay.searchWidth = static_cast<size_t>(value);
        } else if (argument == "--prune-distance" && usesGraphReplayOptions(options.command)) {
            options.graphReplay.pruneDistanceBaseVoxels = parseDouble(valueAfter(index, argc, argv, "prune-distance"), "prune-distance");
        } else if (
            argument == "--storage-chunk-side" &&
            (isQuantizationCommand(options.command) || isReplayCommand(options.command) || isWholeVolumeCommand(options.command) ||
             isChunkRouteStatsCommand(options.command))) {
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
            fail("this command requires --normal-manifest");
        vc::fiber_tracer::validateFiberletPathConfig(options.paths);
    }
    if ((isWholeVolumeCommand(options.command) ||
         isChunkRouteStatsCommand(options.command)) &&
        options.storageChunkSideBaseVoxels <= 0) {
        fail("--storage-chunk-side must be positive");
    }
    if (isChunkRouteStatsCommand(options.command)) {
        if (!options.analysisChunkSpecified)
            fail("chunk-route-stats requires --chunk X,Y,Z");
        if (std::any_of(options.analysisChunkMinimumBaseXYZ.begin(),
                        options.analysisChunkMinimumBaseXYZ.end(),
                        [](int value) { return value < 0; })) {
            fail("--chunk base coordinates must be non-negative");
        }
        if (options.analysisChunkSizeBaseVoxels <= 0)
            fail("chunk-route-stats requires a positive --chunk-size");
        if (options.analysisRegionSizeBaseVoxels == 0) {
            options.analysisRegionSizeBaseVoxels =
                options.analysisChunkSizeBaseVoxels;
        }
        if (options.analysisRegionSizeBaseVoxels <= 0)
            fail("chunk-route-stats requires a positive --region-size");
        if (options.analysisMode == ChunkRouteMode::Staged) {
            if (options.analysisStages.empty())
                fail("staged chunk-route analysis requires --stage");
            for (const auto& stage : options.analysisStages) {
                if (stage.sideBaseVoxels <= 0)
                    fail("--stage side must be positive");
                if (std::any_of(
                        stage.offsetBaseXYZ.begin(),
                        stage.offsetBaseXYZ.end(),
                        [](int value) { return value < 0; })) {
                    fail("--stage offsets must be non-negative");
                }
                for (int axis = 0; axis < 3; ++axis) {
                    if (stage.offsetBaseXYZ[axis] +
                            stage.sideBaseVoxels >
                        options.analysisRegionSizeBaseVoxels) {
                        fail("--stage contains no complete box in the selected region");
                    }
                }
            }
        } else if (!options.analysisStages.empty()) {
            fail("--stage requires --mode staged");
        }
        if (!(options.analysisMaximumJoinAngleDegrees >= 0.0F) ||
            !(options.analysisMaximumJoinAngleDegrees <= 180.0F)) {
            fail("chunk-route-stats requires --join-angle in [0,180]");
        }
    }
    if (usesGraphReplayOptions(options.command)) {
        if (options.steppedReplayCostOptionSpecified &&
            options.graphReplay.costMode != vc::fiber_tracer::FiberletGraphReplayCostMode::Stepped) {
            fail("--cost-weight, --cost-delay, --cost-step, and --cost-profile-weight require --cost-mode stepped");
        }
        if (options.replayBeginArcBaseVoxels.has_value() &&
            !(*options.replayBeginArcBaseVoxels >= 0.0)) {
            fail("--arc must be non-negative");
        }
        if (!(options.graphReplay.geometricCostWeightPerBaseVoxel > 0.0) ||
            options.graphReplay.geometricCostWeightPerBaseVoxel > 1.0 ||
            !std::isfinite(options.graphReplay.geometricCostWeightPerBaseVoxel)) {
            fail("--cost-weight must be finite and in (0,1]");
        }
        if (!(options.graphReplay.costIntegrationStepBaseVoxels > 0.0) ||
            !std::isfinite(options.graphReplay.costIntegrationStepBaseVoxels)) {
            fail("--cost-step must be finite and positive");
        }
        if (!(options.graphReplay.costProfileWeight >= 0.0) ||
            options.graphReplay.costProfileWeight > 1.0 ||
            !std::isfinite(options.graphReplay.costProfileWeight)) {
            fail("--cost-profile-weight must be finite and in [0,1]");
        }
        if (!(options.graphReplay.geometricCostDelayBaseVoxels >= 0.0) ||
            !std::isfinite(options.graphReplay.geometricCostDelayBaseVoxels)) {
            fail("--cost-delay must be finite and non-negative");
        }
        if (!(options.graphReplay.beamStepDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.beamStepDistanceBaseVoxels) ||
            !(options.graphReplay.lookaheadDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.lookaheadDistanceBaseVoxels) ||
            (options.graphReplay.searchWidth != 0 && options.graphReplay.searchWidth < options.graphReplay.beamWidth) ||
            !(options.graphReplay.pruneDistanceBaseVoxels > 0.0) || !std::isfinite(options.graphReplay.pruneDistanceBaseVoxels)) {
            fail("graph search distances and widths are outside their valid range");
        }
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
        if (!options.graphReplay.decisionDiagnosticReferenceArcWindowsBase.empty() && !options.printStats)
            fail("fiber-replay --decision-window requires --stats");
    }
    if (isQuantizationCommand(options.command)) {
        if (!(options.failureThresholdBaseVoxels >= 0.0) || !(options.radiusBaseVoxels > 0.0) || !(options.matchRefineSteps >= 0.0) ||
            !(options.routeStatsFailureMarginBaseVoxels >= 0.0) || options.storageChunkSideBaseVoxels <= 0 ||
            (options.replayLengthBaseVoxels.has_value() && !(*options.replayLengthBaseVoxels > 0.0))) {
            fail("quantization-benchmark options are outside their valid range");
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
    const bool remote =
        vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) ||
        (needsNormals && vc::lasagna::isRemoteLasagnaLocation(
             options.normalManifestLocation));
    if (remote && options.remoteCacheDirectory.empty())
        fail("direct remote manifests require --remote-cache-dir");
    return options;
}

void printChunkRouteDistribution(
    const char* name,
    const vc::fiber_tracer::FiberletChunkRouteDistribution& values)
{
    std::cout << "fiberlet_chunk_route_distribution name=" << name
              << " count=" << values.count;
    if (values.count == 0) {
        std::cout << " min=n/a mean=n/a median=n/a max=n/a\n";
        return;
    }
    std::cout << " min=" << *values.minimum << " mean=" << *values.mean
              << " median=" << *values.median << " max=" << *values.maximum
              << '\n';
}

int runChunkRouteStats(
    CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::fiber_tracer::FiberPredictionField& field,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid);

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

std::string directoryHash(
    const std::filesystem::path& root,
    const std::shared_ptr<
        vc::fiber_tracer::FiberletChunkWriteBackCache>& writeBack = {})
{
    std::map<std::string, std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        const auto relative =
            std::filesystem::relative(entry.path(), root).generic_string();
        files.emplace(relative, entry.path());
    }
    std::map<std::string, std::shared_ptr<const std::vector<std::byte>>>
        memory;
    if (writeBack) {
        writeBack->waitForSpills();
        for (const auto& file : writeBack->logicalFiles(root)) {
            const auto relative =
                std::filesystem::relative(file.path, root).generic_string();
            files.try_emplace(relative, file.path);
            memory.insert_or_assign(relative, file.bytes);
        }
    }
    std::uint64_t hash = 14695981039346656037ULL;
    const auto append = [&](std::span<const char> bytes) {
        for (const unsigned char byte : bytes) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
    };
    std::array<char, 64 * 1024> buffer{};
    for (const auto& [relative, path] : files) {
        append(std::span{relative.data(), relative.size()});
        const char separator = '\0';
        append(std::span{&separator, std::size_t{1}});
        if (const auto found = memory.find(relative); found != memory.end()) {
            append(std::span{
                reinterpret_cast<const char*>(found->second->data()),
                found->second->size()});
            continue;
        }
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot hash file: " + path.string());
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            append(std::span{buffer.data(), static_cast<std::size_t>(input.gcount())});
        }
        if (!input.eof())
            throw std::runtime_error("failed while hashing file: " + path.string());
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
           << hash;
    return output.str();
}

std::string fiberletIdHash(
    std::span<const vc::fiber_tracer::FiberletStorageId> ids)
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto append = [&](std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash ^= static_cast<std::uint8_t>(value & 0xffU);
            hash *= 1099511628211ULL;
            value >>= 8;
        }
    };
    for (const auto& id : ids) {
        for (const auto& endpoint : {id.first, id.second}) {
            for (const auto coordinate : endpoint.coordinateZYX)
                append(static_cast<std::uint64_t>(coordinate));
            append(endpoint.variant);
        }
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
           << hash;
    return output.str();
}

std::string datasetLocator(const vc::lasagna::LasagnaDataset& dataset);

nlohmann::json fiberletSourceIdentity(
    const CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset)
{
    nlohmann::json sources = nlohmann::json::object();
    if (!options.sourceContextPath.empty()) {
        std::ifstream input(options.sourceContextPath);
        if (!input)
            fail("cannot open --source-context " +
                 options.sourceContextPath.string());
        input >> sources;
        if (!sources.is_object())
            fail("--source-context must contain one JSON object");
        for (const auto* field : {
                 "source_volume", "fiber_prediction", "normal_prediction"}) {
            if (!sources.contains(field) || !sources.at(field).is_object())
                fail(std::string("--source-context is missing object '") +
                     field + "'");
        }
    } else {
        // Manual native runs remain location-independent even without manager
        // catalog identities. The content hashes are stable identifiers.
        sources = {
            {"source_volume", {{"identity", "manifest_hashes_only"}}},
            {"fiber_prediction", nlohmann::json::object()},
            {"normal_prediction", nlohmann::json::object()},
        };
    }
    sources["fiber_prediction"]["manifest_content_hash"] =
        fileHash(fiberDataset.manifest().manifestPath);
    sources["normal_prediction"]["manifest_content_hash"] =
        fileHash(normalDataset.manifest().manifestPath);
    return sources;
}

vc::fiber_tracer::FiberletDatasetMetadata replayDatasetMetadata(
    vc::fiber_tracer::FiberletDatasetKind kind,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid,
    const CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::lasagna::LasagnaDataset& normalDataset,
    const std::vector<cv::Vec3d>& corridorReferenceBase,
    double corridorRadiusBaseVoxels,
    const vc::fiber_tracer::FiberletGeometryCacheProfile& cacheProfile = {},
    vc::fiber_tracer::FiberletStorageProfile storageProfile = vc::fiber_tracer::FiberletStorageProfile::Float32Cache,
    std::string_view selectionIdentity = "chunk_local_segment_aabb_v2")
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
    vc::fiber_tracer::FiberletDatasetMetadata metadata;
    metadata.kind = kind;
    metadata.profile = storageProfile;
    metadata.chunkGridShapeZYX = chunkShape;
    metadata.coordinateOriginZYX = {0, 0, 0};
    metadata.coordinateUnitsPerChunkZYX = {unitsPerChunk, unitsPerChunk, unitsPerChunk};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {maximumReach, maximumReach, maximumReach};
    metadata.spatialChunkSideBaseVoxels = static_cast<std::uint32_t>(options.storageChunkSideBaseVoxels);
    if (storageProfile == vc::fiber_tracer::FiberletStorageProfile::CompactDirectionsFixedCost) {
        metadata.costBits = 16;
    }
    metadata.predictionToBaseScale = grid.predictionToBaseScale;
    metadata.sources = fiberletSourceIdentity(
        options, fiberDataset, normalDataset);
    nlohmann::json corridor = nlohmann::json::array();
    for (const auto& point : corridorReferenceBase)
        corridor.push_back({point[0], point[1], point[2]});
    metadata.processing = {
        {"contract_version", 2},
        {"grid", {
            {"shape_zyx", grid.shapeZYX},
            {"prediction_to_base", grid.predictionToBaseScale},
            {"coordinate_order", "zyx_storage_xyz_vectors"},
        }},
        {"selection", {
            {"algorithm", selectionIdentity},
            {"corridor_radius_base", corridorRadiusBaseVoxels},
            {"corridor_reference_base_xyz", corridor},
        }},
        {"anchors", {
            {"cell_size_prediction", options.anchors.cellSizePredictionVoxels},
            {"gaussian_sigma_prediction", options.anchors.gaussianSigmaPredictionVoxels},
            {"peak_sigma_prediction", options.anchors.peakSigmaPredictionVoxels},
            {"peak_axial_sigma_prediction", options.anchors.peakAxialSigmaPredictionVoxels},
            {"peak_grid_step_prediction", options.anchors.peakGridStepPredictionVoxels},
            {"peak_gradient_weight", options.anchors.peakGradientWeight},
            {"peak_gradient_reliability_scale", options.anchors.peakGradientReliabilityScale},
            {"gaussian_cutoff_sigmas", options.anchors.gaussianCutoffSigmas},
            {"local_window_radius_prediction", options.anchors.localWindowRadiusPredictionVoxels},
            {"axial_support_half_width_prediction", options.anchors.axialSupportHalfWidthPredictionVoxels},
            {"position_convergence_tolerance_prediction", options.anchors.positionConvergenceTolerancePredictionVoxels},
            {"nms_maximum_angle_degrees", options.anchors.nmsMaximumAngleDegrees},
            {"nms_transverse_radius_prediction", options.anchors.nmsTransverseRadiusPredictionVoxels},
            {"nms_longitudinal_radius_prediction", options.anchors.nmsLongitudinalRadiusPredictionVoxels},
            {"observation_presence_floor", options.anchors.observationPresenceFloor},
            {"minimum_aligned_support", options.anchors.minimumAlignedSupport},
            {"robust_maximum_trim_mass_fraction", options.anchors.robustMaximumTrimMassFraction},
            {"robust_mad_multiplier", options.anchors.robustMadMultiplier},
            {"robust_minimum_angle_degrees", options.anchors.robustMinimumAngleDegrees},
            {"merge_maximum_angle_degrees", options.anchors.mergeMaximumAngleDegrees},
            {"merge_maximum_absolute_objective_loss", options.anchors.mergeMaximumAbsoluteObjectiveLoss},
            {"merge_maximum_relative_objective_loss", options.anchors.mergeMaximumRelativeObjectiveLoss},
            {"maximum_seed_count", options.anchors.maximumSeedCount},
            {"maximum_iterations", options.anchors.maximumIterations},
            {"convergence_tolerance", options.anchors.convergenceTolerance},
        }},
        {"paths", {
            {"cell_radius", options.paths.cellRadius},
            {"neighborhood_margin_cells", options.paths.neighborhoodMarginCells},
            {"longitudinal_step_prediction", options.paths.longitudinalStepPredictionVoxels},
            {"transverse_step_prediction", options.paths.transverseStepPredictionVoxels},
            {"maximum_endpoint_angle_degrees", options.paths.maximumEndpointAngleDegrees},
            {"maximum_prediction_deviation_degrees", options.paths.maximumPredictionDeviationDegrees},
            {"corridor_radius_prediction", options.paths.corridorRadiusPredictionVoxels},
            {"invalid_prediction_cost_per_voxel", options.paths.invalidPredictionCostPerVoxel},
            {"smoothness_weight", options.paths.smoothnessWeight},
            {"smoothness_normal_weight", options.paths.smoothnessNormalWeight},
            {"smoothness_tangent_weight", options.paths.smoothnessTangentWeight},
            {"smoothness_free_angle_degrees", options.paths.smoothnessFreeAngleDegrees},
        }},
        {"layout", {
            {"sparse", true},
            {"chunk_grid_shape_zyx", chunkShape},
            {"coordinate_units_per_chunk_zyx", metadata.coordinateUnitsPerChunkZYX},
            {"maximum_endpoint_reach_coordinate_units_zyx", metadata.maximumEndpointReachCoordinateUnitsZYX},
            {"arrays", {"anchors", "prefix", "routes"}},
        }},
        {"storage", {
            {"profile", static_cast<int>(storageProfile)},
            {"spatial_chunk_side_base", options.storageChunkSideBaseVoxels},
            {"coordinate_bits", metadata.coordinateBits},
            {"delta_bits", metadata.deltaBits},
            {"route_count_bits", metadata.routeCountBits},
            {"route_lattice_bits", metadata.routeLatticeBits},
            {"cost_bits", metadata.costBits},
            {"position_quantum_base", metadata.positionQuantumBaseVoxels == 0
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(metadata.positionQuantumBaseVoxels)},
            {"compact_directions", cacheProfile.geometry.compactDirections},
            {"route_cost_density_schema", kind == vc::fiber_tracer::FiberletDatasetKind::Anchors
                 ? nlohmann::json(nullptr)
                 : nlohmann::json("sqrt_u16_max256_v1")},
            {"codec_envelope", "vc_fiberlet_chunk_v2"},
        }},
    };
    vc::fiber_tracer::finalizeFiberletDatasetIdentity(metadata);
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

class ChunkRouteReplayProgress
{
public:
    explicit ChunkRouteReplayProgress(bool verbose);
    ~ChunkRouteReplayProgress();

    ChunkRouteReplayProgress(const ChunkRouteReplayProgress&) = delete;
    ChunkRouteReplayProgress& operator=(const ChunkRouteReplayProgress&) = delete;

    void configure(
        std::span<const vc::render::ChunkKey> anchorChunks,
        std::span<const vc::render::ChunkKey> fiberletChunks);
    void resolve(
        vc::fiber_tracer::FiberletStorageChunkKind kind,
        const vc::render::ChunkKey& key,
        vc::render::ChunkFetchStatus status);
    void event(const vc::fiber_tracer::FiberletOnDemandProgress& event);
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string fingerprintHex(const std::array<std::uint8_t, 32>& value);
void printRateProgress(
    const char* prefix,
    const std::string& phase,
    const char* rateName,
    size_t completed,
    size_t total,
    double elapsedSeconds);

struct ChunkRouteBox {
    vc::render::ChunkKey owner;
    vc::fiber_tracer::FiberletChunkRouteAnalysisConfig config;
};

vc::fiber_tracer::FiberletChunkRouteAnalysisConfig chunkRouteConfig(
    const CliOptions& options,
    const cv::Vec3d& minimum,
    double side)
{
    vc::fiber_tracer::FiberletChunkRouteAnalysisConfig config;
    config.minimumBaseXYZ = minimum;
    config.maximumBaseXYZ = minimum + cv::Vec3d{side, side, side};
    config.maximumJoinAngleDegrees = options.analysisMaximumJoinAngleDegrees;
    config.edgeCostView = *options.analysisEdgeCostView;
    config.parallelThreads = static_cast<std::size_t>(
        std::max(1, options.paths.parallelThreads));
    config.maximumGeneratedStatesPerEntry =
        options.analysisMaximumStatesPerEntry;
    return config;
}

int chunkRouteOwnerAxis(const vc::render::ChunkKey& owner, std::size_t zyx)
{
    if (zyx == 0)
        return owner.iz;
    if (zyx == 1)
        return owner.iy;
    return owner.ix;
}

cv::Vec3d chunkRouteOwnerMinimumExact(
    const vc::fiber_tracer::FiberletDatasetMetadata& metadata,
    const vc::render::ChunkKey& owner)
{
    cv::Vec3d result;
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double cellSide =
            static_cast<double>(metadata.spatialChunkSideBaseVoxels) /
            static_cast<double>(metadata.coordinateUnitsPerChunkZYX[zyx]);
        result[static_cast<int>(xyz)] =
            (static_cast<double>(metadata.coordinateOriginZYX[zyx]) +
             static_cast<double>(chunkRouteOwnerAxis(owner, zyx)) *
                 static_cast<double>(metadata.coordinateUnitsPerChunkZYX[zyx])) *
            cellSide;
    }
    return result;
}

std::vector<ChunkRouteBox> chunkRouteIntersectingBoxes(
    const CliOptions& options,
    const vc::fiber_tracer::FiberletDatasetMetadata& metadata)
{
    const cv::Vec3d regionMinimum{
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[0]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[1]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[2])};
    const double regionSide =
        static_cast<double>(options.analysisRegionSizeBaseVoxels);
    const cv::Vec3d regionMaximum =
        regionMinimum + cv::Vec3d{regionSide, regionSide, regionSide};
    std::array<int, 3> begin{};
    std::array<int, 3> end{};
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double side =
            static_cast<double>(metadata.spatialChunkSideBaseVoxels);
        const double origin =
            static_cast<double>(metadata.coordinateOriginZYX[zyx]) * side /
            static_cast<double>(metadata.coordinateUnitsPerChunkZYX[zyx]);
        begin[zyx] = static_cast<int>(std::floor(
            (regionMinimum[static_cast<int>(xyz)] - origin) / side));
        end[zyx] = static_cast<int>(std::floor(
            (std::nextafter(regionMaximum[static_cast<int>(xyz)],
                            -std::numeric_limits<double>::infinity()) -
             origin) /
            side));
        begin[zyx] = std::max(0, begin[zyx]);
        end[zyx] = std::min(
            metadata.chunkGridShapeZYX[zyx] - 1, end[zyx]);
    }
    std::vector<ChunkRouteBox> result;
    const double side =
        static_cast<double>(metadata.spatialChunkSideBaseVoxels);
    for (int z = begin[0]; z <= end[0]; ++z) {
        for (int y = begin[1]; y <= end[1]; ++y) {
            for (int x = begin[2]; x <= end[2]; ++x) {
                const vc::render::ChunkKey owner{0, z, y, x};
                result.push_back({
                    owner,
                    chunkRouteConfig(
                        options,
                        chunkRouteOwnerMinimumExact(metadata, owner), side)});
            }
        }
    }
    return result;
}

std::vector<vc::fiber_tracer::FiberletChunkRouteAnalysisConfig>
chunkRouteStageBoxes(
    const CliOptions& options,
    const ChunkRouteStageSpec& stage)
{
    const cv::Vec3d selectedMinimum{
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[0]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[1]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[2])};
    const int regionSide = options.analysisRegionSizeBaseVoxels;
    std::vector<vc::fiber_tracer::FiberletChunkRouteAnalysisConfig> result;
    for (int z = stage.offsetBaseXYZ[2];
         z + stage.sideBaseVoxels <= regionSide;
         z += stage.sideBaseVoxels) {
        for (int y = stage.offsetBaseXYZ[1];
             y + stage.sideBaseVoxels <= regionSide;
             y += stage.sideBaseVoxels) {
            for (int x = stage.offsetBaseXYZ[0];
                 x + stage.sideBaseVoxels <= regionSide;
                 x += stage.sideBaseVoxels) {
                result.push_back(chunkRouteConfig(
                    options,
                    selectedMinimum + cv::Vec3d{
                        static_cast<double>(x), static_cast<double>(y),
                        static_cast<double>(z)},
                    static_cast<double>(stage.sideBaseVoxels)));
            }
        }
    }
    return result;
}

vc::fiber_tracer::FiberletDatasetMetadata chunkRouteReducedMetadata(
    const CliOptions& options,
    const vc::fiber_tracer::FiberletDatasetMetadata& source)
{
    vc::fiber_tracer::FiberletDatasetMetadata result = source;
    const std::int64_t newSide = options.analysisChunkSizeBaseVoxels;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::int64_t oldSide = source.spatialChunkSideBaseVoxels;
        const std::int64_t numerator =
            newSide * source.coordinateUnitsPerChunkZYX[axis];
        if (numerator % oldSide != 0)
            fail("--chunk-size is not aligned to the anchor-cell grid");
        const std::int64_t units = numerator / oldSide;
        if (units <= 0)
            fail("--chunk-size is smaller than one anchor cell");
        const std::int64_t totalUnits =
            static_cast<std::int64_t>(source.chunkGridShapeZYX[axis]) *
            source.coordinateUnitsPerChunkZYX[axis];
        const std::int64_t chunks = (totalUnits + units - 1) / units;
        if (chunks > std::numeric_limits<std::int32_t>::max())
            fail("reduced Fiberlet chunk grid is too large");
        result.coordinateUnitsPerChunkZYX[axis] = units;
        result.chunkGridShapeZYX[axis] = static_cast<std::int32_t>(chunks);
    }
    result.spatialChunkSideBaseVoxels =
        static_cast<std::uint32_t>(newSide);
    result.processing["layout"]["chunk_grid_shape_zyx"] =
        result.chunkGridShapeZYX;
    result.processing["layout"]["coordinate_units_per_chunk_zyx"] =
        result.coordinateUnitsPerChunkZYX;
    result.processing["storage"]["spatial_chunk_side_base"] = newSide;
    result.processing["reduction"] = {
        {"contract", "exact_entry_to_first_exit_chunk_reduction_v1"},
        {"source_dataset_fingerprint",
         fingerprintHex(source.datasetFingerprint)},
        {"chunk_size_base_voxels", newSide},
        {"maximum_join_angle_degrees",
         options.analysisMaximumJoinAngleDegrees},
        {"edge_cost_view",
         *options.analysisEdgeCostView ==
                 vc::fiber_tracer::FiberletChunkRouteEdgeCostView::Stored
             ? "stored"
             : "sqrt_u16_max256"},
        {"maximum_generated_states_per_entry",
         options.analysisMaximumStatesPerEntry},
    };
    vc::fiber_tracer::finalizeFiberletDatasetIdentity(result);
    return result;
}

template <typename T>
std::vector<T> canonicalUnion(const std::vector<std::vector<T>>& sets)
{
    std::vector<T> result;
    for (const auto& values : sets)
        result.insert(result.end(), values.begin(), values.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

template <typename T>
std::vector<T> canonicalIntersection(
    const std::vector<T>& left, const std::vector<T>& right)
{
    std::vector<T> result;
    std::set_intersection(
        left.begin(), left.end(), right.begin(), right.end(),
        std::back_inserter(result));
    return result;
}

std::string fingerprintHex(const std::array<std::uint8_t, 32>& value)
{
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const auto byte : value)
        result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
}

double reductionPercent(std::size_t before, std::size_t after)
{
    return before == 0
        ? 0.0
        : 100.0 * static_cast<double>(before - after) /
              static_cast<double>(before);
}

std::string percentString(std::size_t before, std::size_t after)
{
    std::ostringstream result;
    result << std::fixed << std::setprecision(2)
           << reductionPercent(before, after) << '%';
    return result.str();
}

std::string optionalFixed(const std::optional<double>& value)
{
    if (!value)
        return "n/a";
    std::ostringstream result;
    result << std::fixed << std::setprecision(2) << *value;
    return result.str();
}

void printChunkRouteSimplification(
    std::size_t boxIndex,
    const vc::fiber_tracer::FiberletChunkRouteSimplificationReport& report)
{
    std::cout << "fiberlet_chunk_simplification box=" << boxIndex << '\n'
              << std::left
              << std::setw(18) << "anchors_before"
              << std::setw(17) << "anchors_after"
              << std::setw(17) << "anchors_removed"
              << std::setw(18) << "anchor_reduction"
              << "boundary_portals\n"
              << std::setw(18) << report.inputAnchors
              << std::setw(17) << report.retainedAnchors
              << std::setw(17) << report.unusedAnchorsRemoved
              << std::setw(18) << percentString(
                     report.inputAnchors, report.retainedAnchors)
              << report.boundaryPortals << '\n'
              << std::setw(20) << "fiberlets_before"
              << std::setw(17) << "fiberlets_live"
              << std::setw(18) << "macro_fiberlets"
              << std::setw(18) << "fiberlets_merged"
              << "macro_reduction\n"
              << std::setw(20) << report.inputPhysicalFiberlets
              << std::setw(17) << report.livePhysicalFiberlets
              << std::setw(18) << report.physicalMacros
              << std::setw(18) << report.physicalFiberletsMerged
              << percentString(
                     report.inputPhysicalFiberlets, report.physicalMacros)
              << '\n'
              << std::setw(18) << "directed_before"
              << std::setw(17) << "directed_live"
              << std::setw(18) << "directed_macros"
              << "directed_removed\n"
              << std::setw(18) << report.inputDirectedStates
              << std::setw(17) << report.liveDirectedStates
              << std::setw(18) << report.liveDirectedMacros
              << report.deadDirectedStatesRemoved << '\n'
              << std::setw(19) << "zero_continuation"
              << std::setw(20) << "forced_continuation"
              << std::setw(17) << "branching"
              << "forced_rollouts\n"
              << std::setw(19) << report.zeroContinuationStates
              << std::setw(20) << report.forcedContinuationStates
              << std::setw(17) << report.branchingStates
              << report.deterministicRollouts << '\n'
              << std::setw(23) << "directed_macros_before"
              << std::setw(22) << "directed_macros_after"
              << "directed_macros_merged\n"
              << std::setw(23) << report.liveDirectedMacros
              << std::setw(22) << report.directedChainMacros
              << report.directedMacrosMerged << '\n'
              << std::setw(28) << "distribution"
              << std::setw(10) << "count"
              << std::setw(10) << "mean"
              << std::setw(10) << "median"
              << "max\n"
              << std::setw(28) << "fiberlets_per_macro"
              << std::setw(10) << report.physicalFiberletsPerMacro.count
              << std::setw(10) << optionalFixed(
                     report.physicalFiberletsPerMacro.mean)
              << std::setw(10) << optionalFixed(
                     report.physicalFiberletsPerMacro.median)
              << optionalFixed(report.physicalFiberletsPerMacro.maximum)
              << '\n'
              << std::setw(28) << "macros_per_forced_rollout"
              << std::setw(10) << report.macrosPerDeterministicRollout.count
              << std::setw(10) << optionalFixed(
                     report.macrosPerDeterministicRollout.mean)
              << std::setw(10) << optionalFixed(
                     report.macrosPerDeterministicRollout.median)
              << optionalFixed(
                     report.macrosPerDeterministicRollout.maximum)
              << '\n';
}

struct ChunkRouteStagePopulation {
    std::size_t anchors = 0;
    std::size_t allFiberlets = 0;
    std::size_t interiorFiberlets = 0;
};

ChunkRouteStagePopulation collectChunkRouteStagePopulation(
    const vc::fiber_tracer::FiberletChunkGraphSource& graph,
    const std::vector<vc::fiber_tracer::FiberletChunkRouteAnalysisConfig>& boxes)
{
    using namespace vc::fiber_tracer;
    ChunkRouteStagePopulation result;
    std::vector<std::vector<FiberletStorageId>> allSets;
    std::vector<std::vector<FiberletStorageId>> internalSets;
    allSets.reserve(boxes.size());
    internalSets.reserve(boxes.size());
    for (const auto& box : boxes) {
        auto population = collectFiberletChunkRoutePopulation(graph, box);
        result.anchors += population.insideAnchors;
        allSets.push_back(std::move(population.physicalFiberletIds));
        internalSets.push_back(std::move(population.internalFiberletIds));
    }
    result.allFiberlets = canonicalUnion(allSets).size();
    result.interiorFiberlets = canonicalUnion(internalSets).size();
    return result;
}

int runStagedChunkRouteReduction(
    const CliOptions& options,
    const std::shared_ptr<vc::fiber_tracer::FiberletOnDemandPreprocessor>&
        preprocessor,
    const vc::fiber_tracer::FiberletChunkGraphSource& initialGraph,
    const vc::fiber_tracer::FiberletChunkCacheOptions& cacheOptions)
{
    using namespace vc::fiber_tracer;
    struct TemporaryTree {
        std::filesystem::path root;
        ~TemporaryTree()
        {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
    } temporary;
    std::filesystem::create_directories(options.outputDirectory);
    std::mt19937_64 random(std::random_device{}());
    do {
        temporary.root = options.outputDirectory /
            (".chunk-route-stages-" + std::to_string(random()));
    } while (std::filesystem::exists(temporary.root));
    std::filesystem::create_directories(temporary.root);

    const auto decodedBudget = cacheOptions.service.decodedByteBudget;
    const std::size_t writeBackBytes = decodedBudget
        ? decodedBudget->maximumBytes()
        : cacheOptions.service.decodedByteCapacity;
    auto writeBack = FiberletChunkWriteBackCache::create({
        writeBackBytes,
        1,
        decodedBudget,
        {},
    });

    auto viewMetadata = preprocessor->anchorDataset()->metadata();
    viewMetadata.processing["reduction_view"] = {
        {"contract", "evaluated_anchor_view_v1"},
        {"source_dataset_fingerprint",
         fingerprintHex(viewMetadata.datasetFingerprint)},
    };
    finalizeFiberletDatasetIdentity(viewMetadata);
    auto viewDataset = FiberletChunkDataset::createOrOpen(
        temporary.root / "initial-anchor-view.zarr", viewMetadata);
    auto viewCache = createGeneratedFiberletChunkCache(
        viewDataset,
        [preprocessor](
            FiberletStorageChunkKind kind,
            const vc::render::ChunkKey& requested,
            const FiberletStorageCodecConfig& codec) {
            if (kind != FiberletStorageChunkKind::Anchors)
                throw std::invalid_argument(
                    "evaluated anchor view received a path request");
            const auto loaded = preprocessor->anchorCache()->getChunkBlocking(
                requested.level, requested.iz, requested.iy, requested.ix);
            const auto payload = std::dynamic_pointer_cast<
                const FiberletAnchorChunkPayload>(loaded.payload);
            if (loaded.status != vc::render::ChunkStatus::Data || !payload) {
                throw std::runtime_error(
                    "evaluated anchor view could not load its source chunk");
            }
            const auto evaluated = preprocessor->evaluationAnchorChunk(
                requested, payload);
            FiberletDecodedAnchors decoded{codec, *evaluated};
            return FiberletChunkDataset::MaterializedChunk{
                {},
                std::make_shared<const FiberletAnchorChunkPayload>(
                    std::move(decoded)),
                true};
        },
        cacheOptions);

    struct Layer {
        std::shared_ptr<FiberletChunkDataset> anchors;
        std::shared_ptr<vc::render::ChunkCache> anchorCache;
        std::shared_ptr<FiberletChunkDataset> fiberlets;
        std::shared_ptr<vc::render::ChunkCache> fiberletCache;
    };
    Layer current{
        viewDataset, viewCache, preprocessor->fiberletDataset(),
        preprocessor->fiberletCache()};
    const cv::Vec3d selectedMinimum{
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[0]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[1]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[2])};
    const auto selectedConfig = chunkRouteConfig(
        options, selectedMinimum,
        static_cast<double>(options.analysisRegionSizeBaseVoxels));
    const auto initialPopulation = collectFiberletChunkRoutePopulation(
        initialGraph, selectedConfig);
    struct StageActivity {
        struct Phase {
            double wallSeconds = 0.0;
            double cpuSeconds = 0.0;
        };
        std::size_t boxes = 0;
        std::size_t anchorChunkWrites = 0;
        std::size_t fiberletChunkWrites = 0;
        Phase materialization;
        Phase analysis;
        Phase simplification;
        Phase write;
        Phase population;
        std::string idDigest;
        std::string payloadDigest;
        ChunkRouteStagePopulation original;
        ChunkRouteStagePopulation input;
        ChunkRouteStagePopulation output;
    };
    std::vector<StageActivity> activities;
    std::optional<FiberletChunkRoutePopulation> finalSelectedPopulation;

    for (std::size_t stageIndex = 0;
         stageIndex < options.analysisStages.size(); ++stageIndex) {
        const auto& specification = options.analysisStages[stageIndex];
        const auto boxes = chunkRouteStageBoxes(options, specification);
        if (boxes.empty())
            fail("staged chunk-route reduction produced an empty stage");
        FiberletChunkGraphSource inputGraph(
            current.anchors, current.anchorCache, current.fiberlets,
            current.fiberletCache, options.paths);
        StageActivity activity;
        activity.boxes = boxes.size();
        const auto populationWallStarted = std::chrono::steady_clock::now();
        const double populationCpuStarted = processCpuSeconds();
        activity.original = collectChunkRouteStagePopulation(
            initialGraph, boxes);
        activity.input = collectChunkRouteStagePopulation(
            inputGraph, boxes);
        activity.population.wallSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - populationWallStarted).count();
        activity.population.cpuSeconds +=
            processCpuSeconds() - populationCpuStarted;
        auto anchorMetadata = current.anchors->metadata();
        auto fiberletMetadata = current.fiberlets->metadata();
        const auto reduction = nlohmann::json{
            {"contract", "monotone_sparse_box_overlay_v1"},
            {"stage_index", stageIndex},
            {"side_base_voxels", specification.sideBaseVoxels},
            {"offset_base_xyz", specification.offsetBaseXYZ},
            {"selected_minimum_base_xyz",
             options.analysisChunkMinimumBaseXYZ},
            {"selected_side_base_voxels",
             options.analysisRegionSizeBaseVoxels},
            {"maximum_join_angle_degrees",
             options.analysisMaximumJoinAngleDegrees},
            {"edge_cost_view",
             *options.analysisEdgeCostView ==
                     FiberletChunkRouteEdgeCostView::Stored
                 ? "stored"
                 : "sqrt_u16_max256"},
            {"maximum_generated_states_per_entry",
             options.analysisMaximumStatesPerEntry},
        };
        anchorMetadata.processing["reduction"] = reduction;
        anchorMetadata.processing["reduction"]["source_dataset_fingerprint"] =
            fingerprintHex(current.anchors->metadata().datasetFingerprint);
        fiberletMetadata.processing["reduction"] = reduction;
        fiberletMetadata.processing["reduction"]
            ["source_dataset_fingerprint"] = fingerprintHex(
                current.fiberlets->metadata().datasetFingerprint);
        finalizeFiberletDatasetIdentity(anchorMetadata);
        finalizeFiberletDatasetIdentity(fiberletMetadata);
        const auto stageRoot = temporary.root /
            ("stage-" + std::to_string(stageIndex + 1));
        auto stageAnchors = FiberletChunkDataset::createOrOpen(
            stageRoot / "anchors.zarr", anchorMetadata, writeBack);
        auto stageFiberlets = FiberletChunkDataset::createOrOpen(
            stageRoot / "fiberlets.zarr", fiberletMetadata, writeBack);
        const auto started = std::chrono::steady_clock::now();
        std::vector<FiberletChunkRouteSimplificationReport>
            simplifications;
        for (std::size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) {
            auto stageAnchorCache = createOverlayFiberletAnchorChunkCache(
                stageAnchors, current.anchors, current.anchorCache,
                cacheOptions);
            auto stageFiberletCache = createOverlayFiberletPathChunkCache(
                stageFiberlets, current.fiberlets, current.fiberletCache,
                cacheOptions);
            {
                FiberletChunkGraphSource graph(
                    stageAnchors, stageAnchorCache, stageFiberlets,
                    stageFiberletCache, options.paths);
                const auto reduced = analyzeAndSimplifyFiberletChunkRoutes(
                    graph, boxes[boxIndex]);
                const auto& analyzed = reduced.analysis;
                const auto& simplified = reduced.simplification;
                activity.materialization.wallSeconds +=
                    reduced.materializationSeconds;
                activity.materialization.cpuSeconds +=
                    reduced.materializationCpuSeconds;
                activity.analysis.wallSeconds += reduced.analysisSeconds;
                activity.analysis.cpuSeconds += reduced.analysisCpuSeconds;
                activity.simplification.wallSeconds +=
                    reduced.simplificationSeconds;
                activity.simplification.cpuSeconds +=
                    reduced.simplificationCpuSeconds;
                const auto writeWallStarted =
                    std::chrono::steady_clock::now();
                const double writeCpuStarted = processCpuSeconds();
                const auto written = writeFiberletReductionOverlayBox(
                    graph, stageAnchors, stageFiberlets, boxes[boxIndex],
                    analyzed.physicalFiberletIds,
                    simplified.livePhysicalFiberletIds);
                activity.write.wallSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - writeWallStarted).count();
                activity.write.cpuSeconds +=
                    processCpuSeconds() - writeCpuStarted;
                activity.anchorChunkWrites += written.touchedAnchorChunks;
                activity.fiberletChunkWrites +=
                    written.touchedFiberletChunks;
                if (options.printStats)
                    simplifications.push_back(simplified);
            }
            stageAnchorCache->cancelPendingAndWait();
            stageFiberletCache->cancelPendingAndWait();
            printRateProgress(
                "fiberlet staged reduction",
                "stage" + std::to_string(stageIndex + 1),
                "boxes_per_second", boxIndex + 1, boxes.size(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count());
        }
        auto finalAnchorCache = createOverlayFiberletAnchorChunkCache(
            stageAnchors, current.anchors, current.anchorCache, cacheOptions);
        auto finalFiberletCache = createOverlayFiberletPathChunkCache(
            stageFiberlets, current.fiberlets, current.fiberletCache,
            cacheOptions);
        current = {
            stageAnchors, finalAnchorCache, stageFiberlets,
            finalFiberletCache};
        FiberletChunkGraphSource stageGraph(
            current.anchors, current.anchorCache, current.fiberlets,
            current.fiberletCache, options.paths);
        const auto outputPopulationWallStarted =
            std::chrono::steady_clock::now();
        const double outputPopulationCpuStarted = processCpuSeconds();
        const auto population = collectFiberletChunkRoutePopulation(
            stageGraph, selectedConfig);
        finalSelectedPopulation = population;
        activity.output = collectChunkRouteStagePopulation(
            stageGraph, boxes);
        activity.population.wallSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - outputPopulationWallStarted).count();
        activity.population.cpuSeconds +=
            processCpuSeconds() - outputPopulationCpuStarted;
        if (!std::includes(
                initialPopulation.physicalFiberletIds.begin(),
                initialPopulation.physicalFiberletIds.end(),
                population.physicalFiberletIds.begin(),
                population.physicalFiberletIds.end())) {
            throw std::runtime_error(
                "staged reduction restored a physical Fiberlet");
        }
        if (options.printStats) {
            activity.idDigest = fiberletIdHash(population.physicalFiberletIds);
            activity.payloadDigest = directoryHash(stageRoot, writeBack);
        }
        activities.push_back(activity);
        if (options.printStats) {
            std::cout << "fiberlet_stage_cache stage=" << stageIndex + 1
                      << " root=" << std::quoted(stageRoot.string())
                      << " anchor_chunk_writes="
                      << activity.anchorChunkWrites
                      << " fiberlet_chunk_writes="
                      << activity.fiberletChunkWrites
                      << " id_digest=" << activity.idDigest
                      << " payload_digest=" << activity.payloadDigest
                      << '\n';
            const auto printPhase = [&](const char* name,
                                        const StageActivity::Phase& phase) {
                std::cout << "fiberlet_stage_phase stage=" << stageIndex + 1
                          << " phase=" << name
                          << " wall_seconds=" << phase.wallSeconds
                          << " cpu_seconds=" << phase.cpuSeconds
                          << " effective_cores=" << effectiveCores(
                                 phase.cpuSeconds, phase.wallSeconds)
                          << '\n';
            };
            printPhase("materialization", activity.materialization);
            printPhase("analysis", activity.analysis);
            printPhase("simplification", activity.simplification);
            printPhase("write", activity.write);
            printPhase("population", activity.population);
            for (std::size_t index = 0;
                 index < simplifications.size(); ++index) {
                printChunkRouteSimplification(index, simplifications[index]);
            }
        }
    }

    std::cout << std::left
              << std::setw(8) << "stage"
              << std::setw(12) << "scope"
              << std::setw(14) << "original"
              << std::setw(14) << "input"
              << std::setw(14) << "output"
              << std::setw(18) << "stage_reduction"
              << std::setw(22) << "cumulative_reduction"
              << "boxes\n";
    const auto printStageRow = [&](std::size_t stage, const char* scope,
                                   std::size_t original, std::size_t input,
                                   std::size_t output) {
        std::cout << std::setw(8) << stage
                  << std::setw(12) << scope
                  << std::setw(14) << original
                  << std::setw(14) << input
                  << std::setw(14) << output
                  << std::setw(18) << percentString(input, output)
                  << std::setw(22) << percentString(original, output)
                  << activities[stage - 1].boxes << '\n';
    };
    for (std::size_t stage = 1; stage <= activities.size(); ++stage) {
        const auto& activity = activities[stage - 1];
        printStageRow(
            stage, "anchors", activity.original.anchors,
            activity.input.anchors, activity.output.anchors);
        printStageRow(
            stage, "all", activity.original.allFiberlets,
            activity.input.allFiberlets, activity.output.allFiberlets);
        printStageRow(
            stage, "interior", activity.original.interiorFiberlets,
            activity.input.interiorFiberlets,
            activity.output.interiorFiberlets);
    }
    if (!finalSelectedPopulation)
        throw std::logic_error("staged reduction has no final population");
    const auto& finalPopulation = *finalSelectedPopulation;
    std::cout << "\njoint reduction\n"
              << std::setw(12) << "scope"
              << std::setw(14) << "original"
              << std::setw(14) << "final"
              << "reduction\n";
    const auto printJoint = [&](const char* scope, std::size_t original,
                                std::size_t final) {
        std::cout << std::setw(12) << scope
                  << std::setw(14) << original
                  << std::setw(14) << final
                  << percentString(original, final) << '\n';
    };
    printJoint(
        "anchors", initialPopulation.insideAnchors,
        finalPopulation.insideAnchors);
    printJoint(
        "full", initialPopulation.physicalFiberletIds.size(),
        finalPopulation.physicalFiberletIds.size());
    printJoint(
        "interior", initialPopulation.internalFiberletIds.size(),
        finalPopulation.internalFiberletIds.size());
    current.anchorCache->cancelPendingAndWait();
    current.fiberletCache->cancelPendingAndWait();
    viewCache->cancelPendingAndWait();
    writeBack->waitForSpills();
    if (options.printStats) {
        const auto stats = writeBack->stats();
        std::cout << "fiberlet_write_back_cache"
                  << " resident_entries=" << stats.residentEntries
                  << " pending_entries=" << stats.pendingEntries
                  << " live_bytes=" << stats.liveBytes
                  << " peak_live_bytes=" << stats.peakLiveBytes
                  << " memory_hits=" << stats.memoryHits
                  << " spills=" << stats.spills
                  << " spilled_bytes=" << stats.spilledBytes << '\n';
    }
    writeBack->finish();
    return 0;
}

int runChunkRouteStats(
    CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::fiber_tracer::FiberPredictionField& field,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid)
{
    using namespace vc::fiber_tracer;
    resolveAnchorConfig(options, grid);
    validateFiberletPathConfig(options.paths);

    vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
    normalOptions.workingToBaseScale = grid.predictionToBaseScale;
    normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
    const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(
        options.normalManifestLocation, normalOptions);
    auto normalSampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(
        normalDataset,
        vc::lasagna::LasagnaNormalSamplerOptions{
            options.decodedCacheBytes});

    const auto evaluation = defaultFiberletReplayQuantization(
        options.storageChunkSideBaseVoxels);
    const auto cacheProfile = fiberletGeometryCacheProfile(evaluation);
    auto anchorMetadata = replayDatasetMetadata(
        FiberletDatasetKind::Anchors, grid, options, fiberDataset,
        normalDataset, {}, 0.0, {}, FiberletStorageProfile::Float32Cache,
        "whole_volume_presence_chunks_v1");
    auto fiberletMetadata = replayDatasetMetadata(
        FiberletDatasetKind::Fiberlets, grid, options, fiberDataset,
        normalDataset, {}, 0.0, cacheProfile,
        FiberletStorageProfile::Float32Cache,
        "whole_volume_presence_chunks_v1");
    const auto analysisMetadata = options.analysisMode == ChunkRouteMode::Stats
        ? chunkRouteReducedMetadata(options, fiberletMetadata)
        : fiberletMetadata;
    const auto statsBoxes = options.analysisMode == ChunkRouteMode::Stats
        ? chunkRouteIntersectingBoxes(options, analysisMetadata)
        : std::vector<ChunkRouteBox>{};
    if (options.analysisMode == ChunkRouteMode::Stats && statsBoxes.empty())
        fail("selected chunk-route region does not intersect the dataset");
    const cv::Vec3d selectedMinimum{
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[0]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[1]),
        static_cast<double>(options.analysisChunkMinimumBaseXYZ[2])};
    const double selectedSide =
        static_cast<double>(options.analysisRegionSizeBaseVoxels);
    const auto selectedMaximum =
        selectedMinimum + cv::Vec3d{selectedSide, selectedSide, selectedSide};
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double cellSide =
            static_cast<double>(analysisMetadata.spatialChunkSideBaseVoxels) /
            static_cast<double>(
                analysisMetadata.coordinateUnitsPerChunkZYX[zyx]);
        const double datasetMinimum =
            static_cast<double>(analysisMetadata.coordinateOriginZYX[zyx]) *
            cellSide;
        const double datasetMaximum = datasetMinimum +
            static_cast<double>(analysisMetadata.chunkGridShapeZYX[zyx]) *
                static_cast<double>(
                    analysisMetadata.spatialChunkSideBaseVoxels);
        if (selectedMinimum[static_cast<int>(xyz)] < datasetMinimum ||
            selectedMaximum[static_cast<int>(xyz)] > datasetMaximum) {
            fail("selected chunk-route region lies outside the dataset grid");
        }
    }
    auto anchorNamespace = anchorMetadata.algorithmFingerprint;
    auto fiberletNamespace = fiberletMetadata.algorithmFingerprint;
    std::replace(anchorNamespace.begin(), anchorNamespace.end(), ':', '-');
    std::replace(fiberletNamespace.begin(), fiberletNamespace.end(), ':', '-');
    const auto cacheBase = options.outputDirectory / "cache";
    const auto anchorRoot = options.anchorCacheRoot.empty()
        ? cacheBase / anchorNamespace / "anchors.zarr"
        : options.anchorCacheRoot;
    const auto fiberletRoot = options.fiberletCacheRoot.empty()
        ? cacheBase / fiberletNamespace / "fiberlets.zarr"
        : options.fiberletCacheRoot;
    const auto evaluationAnchorBytes =
        std::max<std::size_t>(1, options.decodedCacheBytes / 8);
    const auto decodedBytes = options.decodedCacheBytes > evaluationAnchorBytes
        ? options.decodedCacheBytes - evaluationAnchorBytes
        : options.decodedCacheBytes;
    auto budget = std::make_shared<vc::render::DecodedChunkCacheBudget>(
        decodedBytes);
    FiberletChunkCacheOptions cacheOptions;
    cacheOptions.service.decodedByteCapacity = options.decodedCacheBytes;
    cacheOptions.service.decodedByteBudget = budget;
    cacheOptions.service.fetchConcurrency.workerCapacity =
        static_cast<std::size_t>(std::max(1, options.paths.parallelThreads));
    cacheOptions.service.fetchConcurrency.maxConcurrentReads =
        static_cast<std::size_t>(std::max(1, options.paths.parallelThreads));

    FiberletOnDemandConfig onDemand;
    onDemand.anchorRoot = anchorRoot;
    onDemand.fiberletRoot = fiberletRoot;
    onDemand.anchorMetadata = std::move(anchorMetadata);
    onDemand.fiberletMetadata = std::move(fiberletMetadata);
    onDemand.grid = grid;
    onDemand.anchorConfig = options.anchors;
    onDemand.pathConfig = options.paths;
    // The existing scheduler parallelizes complete storage chunks. Each chunk
    // keeps its deterministic cell/source ordering and uses one inner worker.
    onDemand.anchorConfig.parallelThreads = 1;
    onDemand.pathConfig.parallelThreads = 1;
    onDemand.geometryQuantization = cacheProfile.geometry;
    onDemand.evaluationAnchorCacheBytes = evaluationAnchorBytes;
    onDemand.predictionSampler = [&field](
        const auto& indices, int threads, auto& samples) {
        field.sampleStoredGridBatch(indices, threads, samples);
    };
    onDemand.normalSampler = normalSampler;
    onDemand.anchorCellPredicate = [](const std::array<std::size_t, 3>&) {
        return true;
    };
    onDemand.anchorRetainPredicate = [](const FiberAnchor&) {
        return FiberAnchorRetainEvaluation{true, {}, {}};
    };
    onDemand.pointPredicate = [](const cv::Vec3d&) { return true; };
    onDemand.anchorCacheOptions = cacheOptions;
    onDemand.fiberletCacheOptions = cacheOptions;
    auto progress = std::make_shared<ChunkRouteReplayProgress>(
        options.printStats);
    onDemand.progress = [progress](const FiberletOnDemandProgress& event) {
        progress->event(event);
    };
    onDemand.chunkResolved =
        [progress](FiberletStorageChunkKind kind,
                   const vc::render::ChunkKey& key,
                   vc::render::ChunkFetchStatus status) {
            progress->resolve(kind, key, status);
        };
    auto preprocessor = FiberletOnDemandPreprocessor::create(
        std::move(onDemand));
    struct ShutdownGuard {
        std::shared_ptr<FiberletOnDemandPreprocessor> value;
        ~ShutdownGuard()
        {
            if (value)
                value->shutdown();
        }
    } shutdown{preprocessor};

    std::map<std::tuple<int, int, int>, vc::render::ChunkKey> sourceOwners;
    auto addSourcePrefetch = [&](const auto& config) {
        for (const auto& key : fiberletChunkRoutePrefetchChunks(
                 preprocessor->fiberletDataset()->metadata(), config)) {
            sourceOwners.try_emplace(
                std::tuple{key.iz, key.iy, key.ix}, key);
        }
    };
    if (options.analysisMode == ChunkRouteMode::Stats) {
        for (const auto& box : statsBoxes)
            addSourcePrefetch(box.config);
    } else {
        for (const auto& stage : options.analysisStages) {
            for (const auto& config : chunkRouteStageBoxes(options, stage))
                addSourcePrefetch(config);
        }
        addSourcePrefetch(chunkRouteConfig(
            options, selectedMinimum, selectedSide));
    }
    std::vector<FiberletScheduledChunk> prefetch;
    std::vector<vc::render::ChunkKey> anchorChunks;
    std::vector<vc::render::ChunkKey> fiberletChunks;
    for (const auto& [coordinate, key] : sourceOwners) {
        (void)coordinate;
        prefetch.push_back({key, 0.0, 0.0});
        fiberletChunks.push_back(key);
        const auto dependencies = preprocessor->anchorDependencies(key);
        anchorChunks.insert(
            anchorChunks.end(), dependencies.begin(), dependencies.end());
    }
    progress->configure(anchorChunks, fiberletChunks);
    preprocessor->prefetchScheduled(
        prefetch, 0, prefetch.size(), true);
    progress->finish();

    FiberletChunkGraphSource graph(
        preprocessor->anchorDataset(), preprocessor->anchorCache(),
        preprocessor->fiberletDataset(), preprocessor->fiberletCache(),
        options.paths,
        [preprocessor](const vc::render::ChunkKey& key,
                       std::shared_ptr<const FiberletAnchorChunkPayload> chunk) {
            return preprocessor->evaluationAnchorChunk(key, std::move(chunk));
        });
    std::cout << std::setprecision(17)
              << "fiberlet_chunk_route_config region_minimum_base_xyz="
              << options.analysisChunkMinimumBaseXYZ[0] << ','
              << options.analysisChunkMinimumBaseXYZ[1] << ','
              << options.analysisChunkMinimumBaseXYZ[2]
              << " region_size_base_voxels="
              << options.analysisRegionSizeBaseVoxels
              << " mode="
              << (options.analysisMode == ChunkRouteMode::Stats
                      ? "stats"
                      : "staged")
              << " stages=" << options.analysisStages.size()
              << " source_cache=" << std::quoted(fiberletRoot.string())
              << '\n';
    if (options.analysisMode == ChunkRouteMode::Stats) {
        std::cout << "fiberlet_chunk_route_stats_config chunk_size_base_voxels="
                  << options.analysisChunkSizeBaseVoxels << '\n';
    }

    if (options.analysisMode == ChunkRouteMode::Stats) {
        std::vector<std::vector<FiberletStorageId>> originalSets;
        std::vector<std::vector<FiberletStorageId>> retainedSets;
        std::vector<std::vector<FiberletStorageId>> originalInternalSets;
        std::vector<std::vector<FiberletStorageId>> retainedInternalSets;
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < statsBoxes.size(); ++index) {
            const auto report = analyzeFiberletChunkRoutes(
                graph, statsBoxes[index].config);
            originalSets.push_back(report.physicalFiberletIds);
            retainedSets.push_back(report.retainedPhysicalFiberlets);
            originalInternalSets.push_back(report.internalFiberletIds);
            retainedInternalSets.push_back(canonicalIntersection(
                report.internalFiberletIds,
                report.retainedPhysicalFiberlets));
            printRateProgress(
                "fiberlet chunk analysis", "stats", "boxes_per_second",
                index + 1, statsBoxes.size(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count());
            if (options.printStats) {
                std::cout << "fiberlet_chunk_route_box index=" << index
                          << " fiberlets_before="
                          << report.physicalFiberlets
                          << " fiberlets_after="
                          << report.usedPhysicalFiberlets
                          << " internal_before="
                          << report.internalFiberlets
                          << " internal_after="
                          << report.usedInternalFiberlets
                          << " elapsed_seconds=" << report.elapsedSeconds
                          << '\n';
            }
        }
        const auto original = canonicalUnion(originalSets);
        const auto retained = canonicalUnion(retainedSets);
        const auto originalInternal = canonicalUnion(originalInternalSets);
        const auto retainedInternal = canonicalUnion(retainedInternalSets);
        std::cout << std::left
                  << std::setw(12) << "scope"
                  << std::setw(19) << "fiberlets_before"
                  << std::setw(18) << "fiberlets_after"
                  << "fiberlets_reduction\n"
                  << std::setw(12) << "all"
                  << std::setw(19) << original.size()
                  << std::setw(18) << retained.size()
                  << percentString(original.size(), retained.size()) << '\n'
                  << std::setw(12) << "interior"
                  << std::setw(19) << originalInternal.size()
                  << std::setw(18) << retainedInternal.size()
                  << percentString(
                         originalInternal.size(), retainedInternal.size())
                  << '\n';
        return 0;
    }

    return runStagedChunkRouteReduction(
        options, preprocessor, graph, cacheOptions);

}

vc::fiber_tracer::ForwardPolylineArcInterval resolveReplayInterval(
    const vc::fiber_tracer::PolylineArcGeometry& reference, size_t firstControlPointLineIndex, const CliOptions& options)
{
    const auto available = vc::fiber_tracer::selectForwardPolylineArcInterval(reference, firstControlPointLineIndex);
    if (!options.replayBeginArcBaseVoxels.has_value()) {
        return vc::fiber_tracer::selectForwardPolylineArcInterval(reference, firstControlPointLineIndex, options.replayLengthBaseVoxels);
    }
    const double begin = *options.replayBeginArcBaseVoxels;
    if (begin < available.beginArc - 1.0e-9 || begin >= available.endArc - 1.0e-9) {
        fail("--arc lies outside the first-CP reference interval");
    }
    const double end =
        options.replayLengthBaseVoxels.has_value() ? std::min(available.endArc, begin + *options.replayLengthBaseVoxels) : available.endArc;
    if (!(end > begin + 1.0e-9))
        fail("focused replay interval is empty");
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
        std::optional<ReplayPreprocessingSnapshot> finalPreprocessing;
        {
            std::lock_guard lock(mutex_);
            if (preprocessing_)
                finalPreprocessing = preprocessing_->snapshot();
        }
        disablePreprocessing();
        stopTicker();
        std::lock_guard lock(mutex_);
        if (!enabled_ || finished_)
            return;
        if (finalPreprocessing)
            updatePreprocessingLocked(*finalPreprocessing);
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
        preprocessingSnapshot_ = snapshot;
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
            if (preprocessingStarted_ &&
                (preprocessingFraction_ < 1.0 || !traceStarted_)) {
                appendProgressMetric(line, "cache/prep", preprocessingFraction_, preprocessingStart_, now);
                line << " anchors=" << preprocessingSnapshot_.resolvedAnchors
                     << '/' << preprocessingSnapshot_.expectedAnchors
                     << " fiberlets="
                     << preprocessingSnapshot_.resolvedPrefixes << '/'
                     << preprocessingSnapshot_.expectedPrefixes;
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
    ReplayPreprocessingSnapshot preprocessingSnapshot_;
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

struct ChunkRouteReplayProgress::Impl {
    explicit Impl(bool verboseOutput)
        : preprocessing(std::make_shared<ReplayPreprocessingProgress>()),
          overall(true, "fiberlet chunk stats"),
          verbose(verboseOutput)
    {
    }

    std::shared_ptr<ReplayPreprocessingProgress> preprocessing;
    ReplayOverallProgress overall;
    bool verbose = false;
    bool finished = false;
};

ChunkRouteReplayProgress::ChunkRouteReplayProgress(bool verbose)
    : impl_(std::make_unique<Impl>(verbose))
{
}

ChunkRouteReplayProgress::~ChunkRouteReplayProgress() = default;

void ChunkRouteReplayProgress::configure(
    std::span<const vc::render::ChunkKey> anchorChunks,
    std::span<const vc::render::ChunkKey> fiberletChunks)
{
    std::set<ReplayChunkId> anchors;
    std::set<ReplayChunkId> fiberlets;
    for (const auto& key : anchorChunks)
        anchors.insert({key.level, key.iz, key.iy, key.ix});
    for (const auto& key : fiberletChunks)
        fiberlets.insert({key.level, key.iz, key.iy, key.ix});
    impl_->preprocessing->configure(
        std::move(anchors), std::move(fiberlets));
    impl_->overall.attachPreprocessing(impl_->preprocessing);
}

void ChunkRouteReplayProgress::resolve(
    vc::fiber_tracer::FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key,
    vc::render::ChunkFetchStatus status)
{
    impl_->preprocessing->resolve(kind, key, status);
}

void ChunkRouteReplayProgress::event(
    const vc::fiber_tracer::FiberletOnDemandProgress& event)
{
    if (!impl_->verbose || event.status != "completed")
        return;
    std::ostringstream line;
    line << "fiberlet_chunk_route_cache stage=" << event.stage
         << " status=" << event.status << " key=" << event.key.iz << ','
         << event.key.iy << ',' << event.key.ix
         << " inputs=" << event.inputCount
         << " outputs=" << event.outputCount
         << " elapsed_seconds=" << event.elapsedSeconds;
    impl_->overall.printEventLine(line.str());
}

void ChunkRouteReplayProgress::finish()
{
    if (impl_->finished)
        return;
    impl_->overall.finish();
    impl_->finished = true;
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
    vc::fiber_tracer::FiberletChunkCacheOptions anchorCacheOptions;
    anchorCacheOptions.service.decodedByteCapacity = options.decodedCacheBytes;
    anchorCacheOptions.service.decodedByteBudget = budget;
    anchorCacheOptions.service.fetchConcurrency.workerCapacity = 1;
    anchorCacheOptions.service.fetchConcurrency.maxConcurrentReads = 1;
    FiberletOnDemandConfig onDemand;
    onDemand.anchorRoot = cacheRoot / "anchors.zarr";
    onDemand.fiberletRoot = cacheRoot / "fiberlets.zarr";
    onDemand.anchorMetadata = anchorMetadata;
    onDemand.fiberletMetadata = fiberletMetadata;
    onDemand.grid = grid;
    onDemand.anchorConfig = options.anchors;
    onDemand.pathConfig = options.paths;
    onDemand.predictionSampler = [&](const auto& indices, int threads, auto& samples) {
        field.sampleStoredGridBatch(indices, threads, samples);
    };
    onDemand.normalSampler = normalSampler;
    onDemand.selectedAnchorCellsZYX.assign(selectedCells.begin(), selectedCells.end());
    onDemand.anchorRetainPredicate = [](const FiberAnchor&) { return FiberAnchorRetainEvaluation{true, {}, {}}; };
    onDemand.pointPredicate = [](const cv::Vec3d&) { return true; };
    onDemand.anchorCacheOptions = anchorCacheOptions;
    onDemand.fiberletCacheOptions = anchorCacheOptions;
    onDemand.progress = [](const FiberletOnDemandProgress& progress) {
        if (progress.status == "completed") {
            std::cerr << "fiberlet_storage_region"
                      << " stage=" << progress.stage << " key=" << progress.key.iz << '/' << progress.key.iy << '/' << progress.key.ix
                      << " inputs=" << progress.inputCount << " outputs=" << progress.outputCount
                      << " elapsed_seconds=" << progress.elapsedSeconds << '\n';
        }
    };
    auto fullPreprocessor = FiberletOnDemandPreprocessor::create(std::move(onDemand));
    std::vector<FiberletScheduledChunk> targetSchedule;
    targetSchedule.reserve(reportOwners.size());
    for (const auto& owner : reportOwners) {
        targetSchedule.push_back({{0, owner[0], owner[1], owner[2]}, 0.0, 0.0});
    }
    std::cout << "fiberlet_storage_full_region"
              << " chunk_side_base=" << targetSide << " chunks=" << targetSchedule.size()
              << " selected_cells_with_halo=" << selectedCells.size() << '\n';
    benchmarkReplayStorageCompression(fullPreprocessor, targetSchedule, targetSchedule.size(), options.storageCompressionSeed, targetSide, reportOwners);
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

std::filesystem::path defaultWholeVolumeAnchorRoot(const std::filesystem::path& outputRoot)
{
    const auto stem = outputRoot.stem().empty() ? outputRoot.filename().string() : outputRoot.stem().string();
    return outputRoot.parent_path() / (stem + ".anchors.zarr");
}

class AtomicTemporaryCleanup final
{
public:
    explicit AtomicTemporaryCleanup(std::vector<std::filesystem::path> roots)
        : roots_(std::move(roots))
    {
    }

    ~AtomicTemporaryCleanup()
    {
        for (const auto& root : roots_) {
            try {
                (void)vc::core::util::cleanupAtomicWriteTemporaryFiles(root);
            } catch (...) {
                // Cleanup must not mask the preprocessing result or exception.
            }
        }
    }

private:
    std::vector<std::filesystem::path> roots_;
};

std::string progressBytes(std::uint64_t bytes)
{
    constexpr double kib = 1024.0;
    constexpr double mib = kib * 1024.0;
    constexpr double gib = mib * 1024.0;
    std::ostringstream text;
    if (bytes >= static_cast<std::uint64_t>(gib))
        text << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << "GiB";
    else if (bytes >= static_cast<std::uint64_t>(mib))
        text << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / mib << "MiB";
    else if (bytes >= static_cast<std::uint64_t>(kib))
        text << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / kib << "KiB";
    else
        text << bytes << 'B';
    return text.str();
}

std::uint64_t existingFileSize(const std::filesystem::path& path)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error)
        throw std::filesystem::filesystem_error("cannot measure fiberlet payload", path, error);
    return static_cast<std::uint64_t>(size);
}

class WholeVolumePipelineProgress final
{
public:
    WholeVolumePipelineProgress(
        std::size_t anchorTotal,
        std::size_t initialAnchors,
        std::uint64_t initialAnchorBytes,
        std::size_t outputTotal,
        std::size_t initialOutputs,
        std::uint64_t initialOutputBytes,
        std::optional<int> currentOutputZ,
        int maximumOutputZ)
        : anchorTotal_(anchorTotal),
          initialAnchors_(initialAnchors),
          outputTotal_(outputTotal),
          initialOutputs_(initialOutputs),
          maximumOutputZ_(maximumOutputZ),
          anchorsCompleted_(initialAnchors),
          anchorBytes_(initialAnchorBytes),
          outputsCompleted_(initialOutputs),
          outputBytes_(initialOutputBytes),
          currentOutputZ_(currentOutputZ.value_or(-1)),
          ticker_([this] { tickerLoop(); })
    {
    }

    ~WholeVolumePipelineProgress() { finish(); }

    WholeVolumePipelineProgress(const WholeVolumePipelineProgress&) = delete;
    WholeVolumePipelineProgress& operator=(const WholeVolumePipelineProgress&) = delete;

    void update(
        std::size_t anchorsCompleted,
        std::uint64_t anchorBytes,
        std::size_t outputsCompleted,
        std::uint64_t outputBytes,
        std::optional<int> currentOutputZ)
    {
        anchorBytes_.store(anchorBytes, std::memory_order_relaxed);
        anchorsCompleted_.store(anchorsCompleted, std::memory_order_relaxed);
        outputBytes_.store(outputBytes, std::memory_order_relaxed);
        outputsCompleted_.store(outputsCompleted, std::memory_order_relaxed);
        currentOutputZ_.store(currentOutputZ.value_or(-1), std::memory_order_release);
    }

    void finish()
    {
        bool expected = false;
        if (!finished_.compare_exchange_strong(expected, true))
            return;
        {
            std::lock_guard lock(waitMutex_);
            stop_ = true;
        }
        waitCv_.notify_all();
        if (ticker_.joinable())
            ticker_.join();
        render(true);
    }

private:
    void tickerLoop()
    {
        render(false);
        std::unique_lock lock(waitMutex_);
        while (!waitCv_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_; })) {
            lock.unlock();
            render(false);
            lock.lock();
        }
    }

    void render(bool final)
    {
        const auto now = std::chrono::steady_clock::now();
        const auto anchorsCompleted = anchorsCompleted_.load(std::memory_order_relaxed);
        const auto anchorBytes = anchorBytes_.load(std::memory_order_relaxed);
        const auto outputsCompleted = outputsCompleted_.load(std::memory_order_relaxed);
        const auto outputBytes = outputBytes_.load(std::memory_order_relaxed);
        const auto currentOutputZ = currentOutputZ_.load(std::memory_order_acquire);
        const double elapsed = std::chrono::duration<double>(now - started_).count();
        const auto progressRate = [elapsed](std::size_t completed, std::size_t initial) {
            return elapsed > 0.0 ? static_cast<double>(completed - initial) / elapsed : 0.0;
        };
        const auto progressEta = [](std::size_t completed, std::size_t total, double rate) {
            if (completed >= total)
                return 0.0;
            return rate > 0.0 ? static_cast<double>(total - completed) / rate : std::numeric_limits<double>::infinity();
        };
        const auto projectedSize = [](std::uint64_t bytes, std::size_t completed, std::size_t total) {
            return completed > 0
                ? static_cast<std::uint64_t>(std::llround(
                      static_cast<double>(bytes) * static_cast<double>(total) / static_cast<double>(completed)))
                : std::uint64_t{0};
        };
        const double anchorRate = progressRate(anchorsCompleted, initialAnchors_);
        const double outputRate = progressRate(outputsCompleted, initialOutputs_);
        const double anchorPercent = anchorTotal_ == 0 ? 100.0 : 100.0 * static_cast<double>(anchorsCompleted) / static_cast<double>(anchorTotal_);
        const double outputPercent = outputTotal_ == 0 ? 100.0 : 100.0 * static_cast<double>(outputsCompleted) / static_cast<double>(outputTotal_);
        const double anchorEta = progressEta(anchorsCompleted, anchorTotal_, anchorRate);
        const double outputEta = progressEta(outputsCompleted, outputTotal_, outputRate);
        const auto projectedAnchorBytes = projectedSize(anchorBytes, anchorsCompleted, anchorTotal_);
        const auto projectedOutputBytes = projectedSize(outputBytes, outputsCompleted, outputTotal_);
        std::ostringstream line;
        line << "fiberlet_preprocess_progress z=";
        if (currentOutputZ >= 0)
            line << currentOutputZ << '/' << maximumOutputZ_;
        else
            line << "done";
        line << " anchors=" << anchorsCompleted << '/' << anchorTotal_ << '(' << std::fixed << std::setprecision(1) << anchorPercent << "%)"
             << " anchor_rate=" << std::fixed << std::setprecision(2) << anchorRate << "chunks/s"
             << " anchor_eta=" << progressDuration(anchorEta)
             << " anchor_size=" << progressBytes(anchorBytes) << '/'
             << (anchorsCompleted > 0 || anchorTotal_ == 0 ? progressBytes(projectedAnchorBytes) : "n/a")
             << " outputs=" << outputsCompleted << '/' << outputTotal_ << '(' << std::setprecision(1) << outputPercent << "%)"
             << std::setprecision(2)
             << " output_rate=" << outputRate << "chunks/s"
             << " output_eta=" << progressDuration(outputEta)
             << " output_size=" << progressBytes(outputBytes) << '/'
             << (outputsCompleted > 0 || outputTotal_ == 0 ? progressBytes(projectedOutputBytes) : "n/a")
             << " elapsed=" << progressDuration(elapsed);
        const auto rendered = line.str();
        std::cerr << '\r' << rendered;
        if (renderedWidth_ > rendered.size())
            std::cerr << std::string(renderedWidth_ - rendered.size(), ' ');
        const bool persistent = final || now - lastPersistent_ >= std::chrono::minutes(1);
        if (persistent) {
            std::cerr << '\n';
            renderedWidth_ = 0;
            lastPersistent_ = now;
        } else {
            std::cerr << std::flush;
            renderedWidth_ = rendered.size();
        }
    }

    const std::size_t anchorTotal_;
    const std::size_t initialAnchors_;
    const std::size_t outputTotal_;
    const std::size_t initialOutputs_;
    const int maximumOutputZ_;
    const std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    std::atomic_size_t anchorsCompleted_{0};
    std::atomic<std::uint64_t> anchorBytes_{0};
    std::atomic_size_t outputsCompleted_{0};
    std::atomic<std::uint64_t> outputBytes_{0};
    std::atomic_int currentOutputZ_{-1};
    std::atomic_bool finished_{false};
    std::mutex waitMutex_;
    std::condition_variable waitCv_;
    bool stop_ = false;
    std::chrono::steady_clock::time_point lastPersistent_ = started_;
    std::size_t renderedWidth_ = 0;
    std::thread ticker_;
};

std::uint64_t finalTupleSize(
    const vc::fiber_tracer::FiberletChunkDataset& dataset,
    const vc::render::ChunkKey& owner)
{
    auto route = owner;
    route.level = 1;
    return existingFileSize(dataset.chunkPath(vc::fiber_tracer::FiberletStorageChunkKind::Anchors, owner)) +
        existingFileSize(dataset.chunkPath(vc::fiber_tracer::FiberletStorageChunkKind::FiberletPrefix, owner)) +
        existingFileSize(dataset.chunkPath(vc::fiber_tracer::FiberletStorageChunkKind::FiberletRoutes, route));
}

int runWholeVolumePreprocessing(
    CliOptions& options,
    const vc::lasagna::LasagnaDataset& fiberDataset,
    const vc::fiber_tracer::FiberPredictionField& field,
    const vc::fiber_tracer::FiberPredictionGridInfo& grid)
{
    using namespace vc::fiber_tracer;
    const auto started = std::chrono::steady_clock::now();
    resolveAnchorConfig(options, grid);
    if (options.corridorRadiusBaseVoxels.has_value()) {
        options.paths.corridorRadiusPredictionVoxels = *options.corridorRadiusBaseVoxels / grid.predictionToBaseScale;
        validateFiberletPathConfig(options.paths);
    }

    vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
    normalOptions.workingToBaseScale = grid.predictionToBaseScale;
    normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
    const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, normalOptions);
    auto normalSampler =
        std::make_shared<vc::lasagna::LasagnaNormalSampler>(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.decodedCacheBytes});

    auto anchorMetadata = replayDatasetMetadata(
        FiberletDatasetKind::Anchors,
        grid,
        options,
        fiberDataset,
        normalDataset,
        {},
        0.0,
        {},
        FiberletStorageProfile::Float32Cache,
        "whole_volume_presence_chunks_v1");
    auto finalMetadata = replayDatasetMetadata(
        FiberletDatasetKind::Combined,
        grid,
        options,
        fiberDataset,
        normalDataset,
        {},
        0.0,
        FiberletGeometryCacheProfile{
            .geometry = {.positionQuantumBaseVoxels = 0.0, .compactDirections = true},
            .compatibilityCostTagBits = 16,
            .storageChunkSideBaseVoxels = options.storageChunkSideBaseVoxels},
        FiberletStorageProfile::CompactDirectionsFixedCost,
        "whole_volume_presence_chunks_v1");

    const auto anchorRoot = options.anchorCacheRoot.empty() ? defaultWholeVolumeAnchorRoot(options.outputDirectory) : options.anchorCacheRoot;
    std::filesystem::create_directories(options.outputDirectory);
    std::filesystem::create_directories(anchorRoot);
    const auto outputCanonical = std::filesystem::canonical(options.outputDirectory);
    const auto anchorCanonical = std::filesystem::canonical(anchorRoot);
    if (outputCanonical == anchorCanonical)
        throw std::invalid_argument("whole-volume anchor cache and final output must use different roots");
    const bool outputFirst = outputCanonical.string() < anchorCanonical.string();
    auto firstLock = std::make_unique<vc::core::util::ExclusiveDirectoryLock>(outputFirst ? outputCanonical : anchorCanonical);
    auto secondLock = std::make_unique<vc::core::util::ExclusiveDirectoryLock>(outputFirst ? anchorCanonical : outputCanonical);
    (void)vc::core::util::cleanupAtomicWriteTemporaryFiles(anchorRoot);
    (void)vc::core::util::cleanupAtomicWriteTemporaryFiles(options.outputDirectory);
    AtomicTemporaryCleanup temporaryCleanup{{anchorRoot, options.outputDirectory}};

    auto finalDataset = FiberletChunkDataset::createOrOpen(options.outputDirectory, finalMetadata);
    std::cerr << "fiberlet_preprocess_presence status=started\n";
    const auto scan = field.scanStoredPresenceChunks(options.anchors.parallelThreads);
    auto activeChunks = fiberletOutputChunksForNonemptyPresence(scan, finalMetadata, options.anchors.cellSizePredictionVoxels);
    finalDataset->configureExpectedChunks(activeChunks);
    const auto totalInput = scan.missingChunks + scan.emptyChunks + scan.nonemptyChunksZYX.size();
    std::cout << "fiberlet_preprocess_presence status=completed"
              << " input_chunks=" << totalInput << " missing_chunks=" << scan.missingChunks << " empty_chunks=" << scan.emptyChunks
              << " nonempty_chunks=" << scan.nonemptyChunksZYX.size() << " active_output_chunks=" << activeChunks.size() << '\n';

    auto budget = std::make_shared<vc::render::DecodedChunkCacheBudget>(options.decodedCacheBytes);
    vc::fiber_tracer::FiberletChunkCacheOptions cacheOptions;
    cacheOptions.service.decodedByteCapacity = options.decodedCacheBytes;
    cacheOptions.service.decodedByteBudget = budget;
    const auto workerCount = static_cast<std::size_t>(std::max(1, options.anchors.parallelThreads));
    cacheOptions.service.fetchConcurrency.workerCapacity = workerCount;
    cacheOptions.service.fetchConcurrency.maxConcurrentReads = workerCount;

    FiberletOnDemandConfig onDemand;
    onDemand.anchorRoot = anchorRoot;
    onDemand.fiberletRoot = options.outputDirectory;
    onDemand.anchorMetadata = anchorMetadata;
    onDemand.fiberletMetadata = finalMetadata;
    onDemand.grid = grid;
    onDemand.anchorConfig = options.anchors;
    onDemand.pathConfig = options.paths;
    // Whole-volume parallelism is across chunks so one global worker budget can
    // move dynamically between ready fiberlets and their anchor dependencies.
    onDemand.anchorConfig.parallelThreads = 1;
    onDemand.pathConfig.parallelThreads = 1;
    onDemand.geometryQuantization = {.positionQuantumBaseVoxels = 0.0, .compactDirections = true};
    onDemand.predictionSampler = [&field](const auto& indices, int threads, auto& samples) {
        field.sampleStoredGridBatch(indices, threads, samples);
    };
    onDemand.normalSampler = normalSampler;
    onDemand.anchorCellPredicate = [](const std::array<size_t, 3>&) { return true; };
    onDemand.anchorRetainPredicate = [](const FiberAnchor&) { return FiberAnchorRetainEvaluation{true, {}, {}}; };
    onDemand.pointPredicate = [](const cv::Vec3d&) { return true; };
    onDemand.anchorCacheOptions = cacheOptions;
    onDemand.fiberletCacheOptions = cacheOptions;
    if (options.printStats) {
        onDemand.progress = [](const FiberletOnDemandProgress& progress) {
            if (progress.status == "completed") {
                std::cerr << "fiberlet_preprocess_chunk"
                          << " stage=" << progress.stage << " key=" << progress.key.iz << '/' << progress.key.iy << '/' << progress.key.ix
                          << " inputs=" << progress.inputCount << " outputs=" << progress.outputCount
                          << " elapsed_seconds=" << progress.elapsedSeconds << '\n';
            }
        };
    }
    auto preprocessor = FiberletOnDemandPreprocessor::create(std::move(onDemand));

    std::set<std::array<int, 4>> dependencyCoordinates;
    std::vector<std::vector<vc::render::ChunkKey>> outputDependencies;
    outputDependencies.reserve(activeChunks.size());
    for (const auto& key : activeChunks) {
        auto dependencies = preprocessor->anchorDependencies(key);
        for (const auto& dependency : dependencies)
            dependencyCoordinates.insert({dependency.level, dependency.iz, dependency.iy, dependency.ix});
        outputDependencies.push_back(std::move(dependencies));
    }
    std::vector<vc::render::ChunkKey> anchorChunks;
    anchorChunks.reserve(dependencyCoordinates.size());
    for (const auto& coordinate : dependencyCoordinates)
        anchorChunks.push_back({coordinate[0], coordinate[1], coordinate[2], coordinate[3]});

    std::vector<vc::render::ChunkKey> completedOutputs;
    completedOutputs.reserve(activeChunks.size());
    std::uint64_t fiberletPayloadBytes = 0;
    for (const auto& key : activeChunks) {
        if (!finalDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberletPrefix, key))
            continue;
        completedOutputs.push_back(key);
        fiberletPayloadBytes += finalTupleSize(*finalDataset, key);
    }

    std::vector<vc::render::ChunkKey> availableAnchors;
    availableAnchors.reserve(anchorChunks.size());
    std::uint64_t anchorPayloadBytes = 0;
    for (const auto& key : anchorChunks) {
        if (!preprocessor->anchorDataset()->readMaterializedChunk(FiberletStorageChunkKind::Anchors, key))
            continue;
        availableAnchors.push_back(key);
        anchorPayloadBytes += existingFileSize(preprocessor->anchorDataset()->chunkPath(FiberletStorageChunkKind::Anchors, key));
    }

    FiberletPreprocessSchedule schedule(
        activeChunks,
        std::move(outputDependencies),
        completedOutputs,
        availableAnchors);
    const int maximumOutputZ = activeChunks.empty() ? -1 : activeChunks.back().iz;
    WholeVolumePipelineProgress progress(
        schedule.anchorTotal(),
        schedule.anchorsCompleted(),
        anchorPayloadBytes,
        schedule.outputTotal(),
        schedule.outputsCompleted(),
        fiberletPayloadBytes,
        schedule.currentOutputZ(),
        maximumOutputZ);

    struct RunningWork {
        FiberletPreprocessWork work;
        std::future<std::uint64_t> result;
    };
    utils::ThreadPool workers(workerCount);
    std::vector<RunningWork> running;
    running.reserve(workerCount);
    const auto submit = [&](const FiberletPreprocessWork& work) {
        return workers.submit([&, work] {
            if (work.kind == FiberletPreprocessWorkKind::Anchor) {
                const auto chunk = preprocessor->anchorCache()->getChunkBlocking(work.key.level, work.key.iz, work.key.iy, work.key.ix);
                if (chunk.status != vc::render::ChunkStatus::Data || !chunk.payload) {
                    throw std::runtime_error(
                        "whole-volume anchor chunk generation failed at " + std::to_string(work.key.iz) + '/' +
                        std::to_string(work.key.iy) + '/' + std::to_string(work.key.ix) +
                        (chunk.error.empty() ? std::string{} : ": " + chunk.error));
                }
                return existingFileSize(
                    preprocessor->anchorDataset()->chunkPath(FiberletStorageChunkKind::Anchors, work.key));
            }

            const auto anchorChunk =
                preprocessor->anchorCache()->getChunkBlocking(work.key.level, work.key.iz, work.key.iy, work.key.ix);
            const auto anchors = std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(anchorChunk.payload);
            if (anchorChunk.status != vc::render::ChunkStatus::Data || !anchors)
                throw std::runtime_error("whole-volume final anchor source is unavailable");
            const auto compactCodec = finalDataset->codecConfig(FiberletStorageChunkKind::Anchors, work.key);
            finalDataset->publishChunk(
                FiberletStorageChunkKind::Anchors,
                work.key,
                serializeFiberletAnchors(compactCodec, anchors->anchors));

            const auto fiberletChunk =
                preprocessor->fiberletCache()->getChunkBlocking(work.key.level, work.key.iz, work.key.iy, work.key.ix);
            if (fiberletChunk.status != vc::render::ChunkStatus::Data || !fiberletChunk.payload) {
                throw std::runtime_error(
                    "whole-volume fiberlet chunk generation failed at " + std::to_string(work.key.iz) + '/' +
                    std::to_string(work.key.iy) + '/' + std::to_string(work.key.ix) +
                    (fiberletChunk.error.empty() ? std::string{} : ": " + fiberletChunk.error));
            }
            if (!finalDataset->readMaterializedChunk(FiberletStorageChunkKind::FiberletPrefix, work.key))
                throw std::runtime_error("whole-volume final tuple is incomplete after publication");
            return finalTupleSize(*finalDataset, work.key);
        });
    };

    while (!schedule.done()) {
        while (running.size() < workerCount) {
            const auto work = schedule.takeNext();
            if (!work)
                break;
            running.push_back({*work, submit(*work)});
        }
        if (running.empty())
            throw std::logic_error("whole-volume preprocessing schedule stalled");

        auto completed = std::find_if(running.begin(), running.end(), [](auto& work) {
            return work.result.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        });
        if (completed == running.end()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto bytes = completed->result.get();
        schedule.complete(completed->work);
        if (completed->work.kind == FiberletPreprocessWorkKind::Anchor)
            anchorPayloadBytes += bytes;
        else
            fiberletPayloadBytes += bytes;
        progress.update(
            schedule.anchorsCompleted(),
            anchorPayloadBytes,
            schedule.outputsCompleted(),
            fiberletPayloadBytes,
            schedule.currentOutputZ());
        running.erase(completed);
    }
    progress.finish();
    preprocessor->shutdown();
    (void)vc::core::util::cleanupAtomicWriteTemporaryFiles(anchorRoot);
    (void)vc::core::util::cleanupAtomicWriteTemporaryFiles(options.outputDirectory);
    auto verifiedDataset = FiberletChunkDataset::openExisting(
        options.outputDirectory);
    verifiedDataset->configureExpectedChunks(activeChunks);
    if (!verifiedDataset->datasetComplete())
        throw std::runtime_error(
            "whole-volume Fiberlet output failed its final completeness check");
    std::cout << "fiberlet_preprocess_volume status=completed"
              << " active_chunks=" << activeChunks.size() << " anchor_chunks=" << anchorChunks.size()
              << " resumed_chunks=" << completedOutputs.size()
              << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << " anchor_cache=" << anchorRoot << " output=" << options.outputDirectory << '\n';
    return 0;
}

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
    vc::fiber_tracer::FiberletChunkCacheOptions cacheOptions;
    cacheOptions.service.decodedByteCapacity = options.decodedCacheBytes;
    cacheOptions.service.decodedByteBudget = graphBudget;
    cacheOptions.service.fetchConcurrency.workerCapacity = 1;
    cacheOptions.service.fetchConcurrency.maxConcurrentReads = 1;

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
            auto chunkService = vc::render::processChunkCacheService();
            chunkService->configureFetchConcurrency(
                static_cast<std::size_t>(
                    std::max(1, options.paths.parallelThreads)),
                false);
            chunkService->configureDecodedByteCapacity(
                options.decodedCacheBytes);
            replayCtVolume = Volume::New(options.volumeZarr);
            (void)vc::fiber_tracer::validateFiberReplayStripCtVolume(*replayCtVolume, replayCtLocator);
        }
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(options.manifestLocation, openOptions);
        const vc::fiber_tracer::FiberPredictionField field(dataset, options.decodedCacheBytes, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
        const auto grid = field.storedGridInfo();

        if (isChunkRouteStatsCommand(options.command))
            return runChunkRouteStats(options, dataset, field, grid);

        if (isWholeVolumeCommand(options.command))
            return runWholeVolumePreprocessing(options, dataset, field, grid);

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
            const auto interval = resolveReplayInterval(reference, fiber.controlPointLineIndices.front(), options);
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
            const auto availableInterval =
                vc::fiber_tracer::selectForwardPolylineArcInterval(
                    reference, fiber.controlPointLineIndices.front());
            const auto interval = resolveReplayInterval(
                reference, fiber.controlPointLineIndices.front(), options);
            const double startArc = interval.beginArc;
            const double endArc = interval.endArc;
            options.graphReplay.recordDecisionDiagnostics = options.printStats;
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
            replayRequest.referenceBeginArcBase = startArc;
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
                const auto processingInterval =
                    options.replayBeginArcBaseVoxels.has_value()
                    ? availableInterval
                    : interval;
                const auto processingTube = vc::fiber_tracer::makeFiberReplayTube(
                    fiber.linePointsXyzBase,
                    0.5 * (processingInterval.beginArc +
                           processingInterval.endArc),
                    0.5 * (processingInterval.endArc -
                           processingInterval.beginArc),
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
                vc::fiber_tracer::FiberletChunkCacheOptions anchorCacheOptions;
                anchorCacheOptions.service.decodedByteCapacity = options.decodedCacheBytes;
                anchorCacheOptions.service.decodedByteBudget = graphBudget;
                anchorCacheOptions.service.fetchConcurrency.workerCapacity = 1;
                anchorCacheOptions.service.fetchConcurrency.maxConcurrentReads = 1;
                vc::fiber_tracer::FiberletChunkCacheOptions fiberletCacheOptions = anchorCacheOptions;
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

            // The replay runs both evaluators concurrently. Treat --threads as
            // their shared worker budget rather than assigning the full budget
            // independently to each nested parallel search.
            const int replayThreadBudget = std::max(1, options.paths.parallelThreads);
            const int greedyReplayThreads = std::max(1, (replayThreadBudget + 1) / 2);
            const size_t fiberletReplayThreads = static_cast<size_t>(
                std::max(1, replayThreadBudget - greedyReplayThreads));
            replayRequest.config.parallelThreads = greedyReplayThreads;
            auto effectiveGraphReplay = options.graphReplay;
            effectiveGraphReplay.expansionThreads = fiberletReplayThreads;

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
                                                               effectiveGraphReplay,
                                                               failurePrinter(vc::fiber_tracer::FiberReplayTracer::Fiberlet),
                                                               progress)
                                                         : vc::fiber_tracer::traceFiberletGraphReplay(
                                                               *cachedGraph,
                                                               fiber.linePointsXyzBase,
                                                               *canonicalNormalSampler,
                                                               grid.predictionToBaseScale,
                                                               effectiveGraphReplay,
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
            bundle.fiberletReplayConfig = effectiveGraphReplay;
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
