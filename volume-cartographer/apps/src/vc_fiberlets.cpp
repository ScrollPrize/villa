#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"

#include "vc/fiber_tracer/FiberletQuantization.hpp"

#include "vc/fiber_tracer/FiberletOnDemand.hpp"
#include "vc/fiber_tracer/FiberletPipeline.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

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

namespace {
enum class Command { Anchors, Paths, PreprocessVolume };
bool isWholeVolumeCommand(Command command) { return command == Command::PreprocessVolume; }
bool needsPathExtraction(Command command) { return command != Command::Anchors; }
struct CliOptions {
    Command command = Command::Anchors;
    std::string manifestLocation;
    std::filesystem::path anchorArtifact;
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
    double glyphLengthBaseVoxels = 16.0;
    size_t decodedCacheBytes = 512ULL * 1024ULL * 1024ULL;
    bool printStats = false;
    bool writePresenceSlices = true;
    int storageChunkSideBaseVoxels = 512;
    std::filesystem::path anchorCacheRoot;
    std::filesystem::path sourceContextPath;
};
[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable) {
    std::cerr << "Usage:\n  " << executable << " anchors MANIFEST OUTPUT [--crop X,Y,Z,W,H,D]\n"
              << "  " << executable << " paths MANIFEST ANCHORS_JSON OUTPUT --normal-manifest MANIFEST\n"
              << "  " << executable << " preprocess-volume MANIFEST OUTPUT --normal-manifest MANIFEST\n"
              << "Common: --threads N --cache-gib N --remote-cache-dir DIR --stats\n"
              << "Preprocess: --anchor-cache DIR --source-context JSON --storage-chunk-side N\n"
              << "Fit: --cell-size N --falloff N --peak-sigma N --axial-sigma N --peak-step N\n"
              << "     --gradient-weight N --window N --presence-floor N --minimum-support N\n"
              << "     --robust-max-trim N --robust-mad-multiplier N --robust-min-angle-deg N\n"
              << "     --nms-angle-deg N --maximum-seeds N --maximum-iterations N\n"
              << "Paths: --cell-radius N --radius-margin N --endpoint-angle-degrees N\n"
              << "       --prediction-angle N --corridor-radius N --invalid-prediction-cost N\n"
              << "       --smoothness-weight N --smoothness-normal-weight N --smoothness-tangent-weight N\n"
              << "       --smoothness-free-angle N --batch N --no-slices\n"
              << "Distances are base voxels except --cell-size (prediction voxels).\n";
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

CliOptions parseArgs(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) { usage(argv[0]); std::exit(0); }
    if (argc < 4) { usage(argv[0]); fail("missing command arguments"); }
    CliOptions options;
    const std::string command = argv[1];
    int firstOption = 4;
    options.manifestLocation = argv[2];
    options.outputDirectory = argv[3];
    if (command == "anchors") options.command = Command::Anchors;
    else if (command == "preprocess-volume") options.command = Command::PreprocessVolume;
    else if (command == "paths") {
        if (argc < 5) fail("paths requires MANIFEST ANCHORS_JSON OUTPUT");
        options.command = Command::Paths;
        options.anchorArtifact = argv[3];
        options.outputDirectory = argv[4];
        firstOption = 5;
    } else fail("unknown command: " + command);
    const int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    options.anchors.parallelThreads = workers;
    options.paths.parallelThreads = workers;
    for (int index = firstOption; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (argument == "--cache-gib") {
            const double gib = parseDouble(valueAfter(index, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0) || gib > 1024.0)
                fail("--cache-gib must be in (0, 1024]");
            options.decodedCacheBytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = valueAfter(index, argc, argv, "remote-cache-dir");
        } else if (argument == "--source-context" &&
                   isWholeVolumeCommand(options.command)) {
            options.sourceContextPath =
                valueAfter(index, argc, argv, "source-context");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(valueAfter(index, argc, argv, "base-voxel-size-um"), "base-voxel-size-um");
        } else if (argument == "--normal-manifest" && needsPathExtraction(options.command)) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
        } else if (
            argument == "--anchor-cache" &&
            isWholeVolumeCommand(options.command)) {
            options.anchorCacheRoot = valueAfter(index, argc, argv, "anchor-cache");
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
            argument == "--stats" && needsPathExtraction(options.command)) {
            options.printStats = true;
        } else if (argument == "--no-slices" && options.command == Command::Paths) {
            options.writePresenceSlices = false;
        } else if (argument == "--invalid-prediction-cost" &&
                   needsPathExtraction(options.command)) {
            options.paths.invalidPredictionCostPerVoxel =
                parseDouble(valueAfter(index, argc, argv, "invalid-prediction-cost"), "invalid-prediction-cost");
        } else if (argument == "--smoothness-weight" &&
                   needsPathExtraction(options.command)) {
            options.paths.smoothnessWeight = parseDouble(valueAfter(index, argc, argv, "smoothness-weight"), "smoothness-weight");
        } else if (argument == "--smoothness-normal-weight" &&
                   needsPathExtraction(options.command)) {
            options.paths.smoothnessNormalWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-normal-weight"), "smoothness-normal-weight");
        } else if (argument == "--smoothness-tangent-weight" &&
                   needsPathExtraction(options.command)) {
            options.paths.smoothnessTangentWeight =
                parseDouble(valueAfter(index, argc, argv, "smoothness-tangent-weight"), "smoothness-tangent-weight");
        } else if (argument == "--smoothness-free-angle" &&
                   needsPathExtraction(options.command)) {
            options.paths.smoothnessFreeAngleDegrees =
                parseDouble(valueAfter(index, argc, argv, "smoothness-free-angle"), "smoothness-free-angle");
        } else if (argument == "--batch" && needsPathExtraction(options.command)) {
            options.paths.samplingBatchCoordinates = parseInt(valueAfter(index, argc, argv, "batch"), "batch");
        } else if (argument == "--threads") {
            const int value = parseInt(valueAfter(index, argc, argv, "threads"), "threads");
            if (value < 1) fail("--threads must be positive");
            options.anchors.parallelThreads = value;
            options.paths.parallelThreads = value;
        } else if (argument == "--storage-chunk-side") {
            options.storageChunkSideBaseVoxels = parseInt(valueAfter(index, argc, argv, "storage-chunk-side"), "storage-chunk-side");
        } else fail("unknown or inapplicable option: " + argument);
    }
    if (needsPathExtraction(options.command) && options.normalManifestLocation.empty()) fail("--normal-manifest is required");
    if (options.storageChunkSideBaseVoxels <= 0) fail("--storage-chunk-side must be positive");
    vc::fiber_tracer::validateFiberletPathConfig(options.paths);
    if ((vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) ||
         vc::lasagna::isRemoteLasagnaLocation(options.normalManifestLocation)) && options.remoteCacheDirectory.empty())
        fail("direct remote manifests require --remote-cache-dir");
    return options;
}
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
    }
    return vc::fiber_tracer::makeFiberletSourceIdentity(
        fiberDataset.manifest().manifestPath,
        normalDataset.manifest().manifestPath, std::move(sources));
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
    auto metadata = vc::fiber_tracer::makeFiberletDatasetMetadata(
        kind, grid, options.anchors, options.paths, options.storageChunkSideBaseVoxels,
        fiberletSourceIdentity(options, fiberDataset, normalDataset),
        corridorReferenceBase, corridorRadiusBaseVoxels, cacheProfile,
        storageProfile, selectionIdentity);
    // Producer diagnostics are intentionally excluded from scientific identity.
    metadata.processing["producer_toolchain"] = {
        {"compiler_id", VC_FIBERLET_COMPILER_ID},
        {"compiler_version", VC_FIBERLET_COMPILER_VERSION},
        {"build_config", VC_FIBERLET_BUILD_CONFIG}};
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
        std::make_shared<vc::lasagna::LasagnaNormalSampler>(normalDataset);

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

    // Whole-volume parallelism is across chunks so one global worker budget can
    // move dynamically between ready fiberlets and their anchor dependencies.
    auto onDemand = makeCanonicalFiberletOnDemandConfig(
        anchorRoot, options.outputDirectory, anchorMetadata, finalMetadata,
        grid, options.anchors, options.paths,
        [&field](const auto& indices, int threads, auto& samples) {
            field.sampleStoredGridBatch(indices, threads, samples);
        }, normalSampler, cacheOptions);
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
            if (anchors->anchors.empty())
                return std::uint64_t{0};
            const auto compactCodec = finalDataset->codecConfig(FiberletStorageChunkKind::Anchors, work.key);
            const auto compactAnchors = serializeFiberletAnchors(compactCodec, anchors->anchors);
            finalDataset->publishChunk(FiberletStorageChunkKind::Anchors, work.key, compactAnchors);
            const auto removeFinalTuple = [&] {
                auto routeKey = work.key;
                routeKey.level = 1;
                std::error_code ignored;
                std::filesystem::remove(
                    finalDataset->chunkPath(FiberletStorageChunkKind::Anchors, work.key), ignored);
                std::filesystem::remove(
                    finalDataset->chunkPath(FiberletStorageChunkKind::FiberletPrefix, work.key), ignored);
                std::filesystem::remove(
                    finalDataset->chunkPath(FiberletStorageChunkKind::FiberletRoutes, routeKey), ignored);
            };

            const auto fiberletChunk = [&] {
                try {
                    return preprocessor->fiberletCache()->getChunkBlocking(
                        work.key.level, work.key.iz, work.key.iy, work.key.ix);
                } catch (...) {
                    removeFinalTuple();
                    throw;
                }
            }();
            if (fiberletChunk.status != vc::render::ChunkStatus::Data || !fiberletChunk.payload) {
                removeFinalTuple();
                throw std::runtime_error(
                    "whole-volume fiberlet chunk generation failed at " + std::to_string(work.key.iz) + '/' +
                    std::to_string(work.key.iy) + '/' + std::to_string(work.key.ix) +
                    (fiberletChunk.error.empty() ? std::string{} : ": " + fiberletChunk.error));
            }
            const auto prefixes = std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(fiberletChunk.payload);
            if (!prefixes) {
                removeFinalTuple();
                throw std::runtime_error("whole-volume fiberlet chunk has the wrong decoded payload");
            }
            if (prefixes->prefixes.empty()) {
                removeFinalTuple();
                return std::uint64_t{0};
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
    for (const auto& key : activeChunks) {
        // Missing tuples are canonical sparse empties. Present tuples must
        // still decode and validate as a complete anchor/prefix/route unit.
        const bool anchorPresent = std::filesystem::exists(
            verifiedDataset->chunkPath(FiberletStorageChunkKind::Anchors, key));
        const auto pairPresence = verifiedDataset->pairPresence(key);
        if (!anchorPresent &&
            pairPresence == FiberletChunkDataset::PairPresence::Absent) {
            continue;
        }
        if (!anchorPresent ||
            pairPresence != FiberletChunkDataset::PairPresence::Complete ||
            !verifiedDataset->readMaterializedChunk(
                FiberletStorageChunkKind::FiberletPrefix, key)) {
            throw std::runtime_error(
                "whole-volume Fiberlet output contains a partial sparse tuple at " +
                std::to_string(key.iz) + '/' + std::to_string(key.iy) + '/' +
                std::to_string(key.ix));
        }
    }
    std::cout << "fiberlet_preprocess_volume status=completed"
              << " active_chunks=" << activeChunks.size() << " anchor_chunks=" << anchorChunks.size()
              << " resumed_chunks=" << completedOutputs.size()
              << " elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << " anchor_cache=" << anchorRoot << " output=" << options.outputDirectory << '\n';
    return 0;
}

} // namespace
int main(int argc, char** argv) {
    try {
        CliOptions options = parseArgs(argc, argv);
        vc::render::processChunkCacheService()->configureDecodedByteCapacity(options.decodedCacheBytes);
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(options.manifestLocation, openOptions);
        const vc::fiber_tracer::FiberPredictionField field(dataset, vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid, options.decodedCacheBytes);
        const auto grid = field.storedGridInfo();
        if (isWholeVolumeCommand(options.command)) return runWholeVolumePreprocessing(options, dataset, field, grid);
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
            artifact.manifestContentHash = vc::fiber_tracer::fiberletContentHash(dataset.manifest().manifestPath);
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
        const std::string manifestHash = vc::fiber_tracer::fiberletContentHash(dataset.manifest().manifestPath);
        if (manifestHash != anchors.artifact.manifestContentHash)
            fail("fiber manifest content hash does not match anchors.json");
        if (grid.shapeZYX != anchors.report.grid.shapeZYX || std::abs(grid.predictionToBaseScale - anchors.report.grid.predictionToBaseScale) > 1.0e-12) {
            fail("fiber prediction grid does not match anchors.json");
        }

        vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
        normalOptions.workingToBaseScale = grid.predictionToBaseScale;
        normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifestLocation, normalOptions);
        const vc::lasagna::LasagnaNormalSampler normalSampler(normalDataset);
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
        artifact.normalManifestContentHash = vc::fiber_tracer::fiberletContentHash(normalDataset.manifest().manifestPath);
        artifact.anchorArtifactLocator = std::filesystem::absolute(options.anchorArtifact).lexically_normal().string();
        artifact.anchorArtifactContentHash = vc::fiber_tracer::fiberletContentHash(options.anchorArtifact);
        artifact.baseVoxelSizeUm = options.baseVoxelSizeUm.has_value() ? options.baseVoxelSizeUm : anchors.artifact.baseVoxelSizeUm;
        vc::fiber_tracer::writeFiberletPathArtifacts(options.outputDirectory, report, artifact);
        if (presenceSlices.has_value()) {
            vc::fiber_tracer::writeFiberPresenceSliceArtifacts(options.outputDirectory, *presenceSlices, grid);
        } else {
            vc::fiber_tracer::removeFiberPresenceSliceArtifacts(options.outputDirectory);
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "vc_fiberlets: " << error.what() << '\n'; return 1; }
}
