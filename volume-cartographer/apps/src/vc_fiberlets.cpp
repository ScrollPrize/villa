#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberPaths.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
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
};

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
    std::optional<double> baseVoxelSizeUm;
    double glyphLengthBaseVoxels = 16.0;
    size_t decodedCacheBytes = 512ULL * 1024ULL * 1024ULL;
    bool sigmaExplicit = false;
    bool printStats = false;
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
              << "Common options:\n"
              << "  --threads N                   decode/search workers [hardware]\n"
              << "  --cache-gib N                 decoded chunk cache budget [0.5]\n"
              << "  --remote-cache-dir PATH       required for direct remote manifests\n"
              << "  --base-voxel-size-um N        optional physical reporting metadata\n\n"
              << "Anchor options:\n"
              << "  --cell-size N                 prediction-grid cell side, 2..8 [4]\n"
              << "  --gaussian-sigma N            Gaussian sigma [cell-size/2]\n"
              << "  --presence-floor N            inclusive observation floor [0.05]\n"
              << "  --minimum-support N           inclusive aligned support [0.05]\n"
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
              << "  --presence-weight N           low-presence cost weight [1]\n"
              << "  --direction-weight N          quantization-relative direction weight [1]\n"
              << "  --invalid-prediction-cost N   invalid cost per prediction-grid voxel [4]\n"
              << "  --smoothness-weight N         invalid-normal isotropic weight [2]\n"
              << "  --smoothness-normal-weight N  normal-tilt weight [0.1]\n"
              << "  --smoothness-tangent-weight N tangent-plane turn weight [10]\n"
              << "  --smoothness-free-angle N     lattice free angle in degrees [45]\n"
              << "  --stats                       print path-count and score statistics\n";
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
    } else {
        usage(argv[0]);
        std::exit(2);
    }
    const int workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    options.anchors.parallelThreads = workers;
    options.paths.parallelThreads = workers;
    for (int index = firstOption; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (argument == "--threads") {
            const int value = parseInt(valueAfter(index, argc, argv, "threads"), "threads");
            options.anchors.parallelThreads = value;
            options.paths.parallelThreads = value;
        } else if (argument == "--cache-gib") {
            const double gib = parseDouble(valueAfter(index, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0) || gib > 1024.0)
                fail("--cache-gib must be in (0, 1024]");
            options.decodedCacheBytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = valueAfter(index, argc, argv, "remote-cache-dir");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(valueAfter(index, argc, argv, "base-voxel-size-um"), "base-voxel-size-um");
        } else if (argument == "--cell-size" && options.command == Command::Anchors) {
            options.anchors.cellSizePredictionVoxels = parseInt(valueAfter(index, argc, argv, "cell-size"), "cell-size");
        } else if (argument == "--gaussian-sigma" && options.command == Command::Anchors) {
            options.anchors.gaussianSigmaPredictionVoxels = parseDouble(valueAfter(index, argc, argv, "gaussian-sigma"), "gaussian-sigma");
            options.sigmaExplicit = true;
        } else if (argument == "--presence-floor" && options.command == Command::Anchors) {
            options.anchors.observationPresenceFloor = parseDouble(valueAfter(index, argc, argv, "presence-floor"), "presence-floor");
        } else if (argument == "--minimum-support" && options.command == Command::Anchors) {
            options.anchors.minimumAlignedSupport = parseDouble(valueAfter(index, argc, argv, "minimum-support"), "minimum-support");
        } else if (argument == "--maximum-seeds" && options.command == Command::Anchors) {
            const int value = parseInt(valueAfter(index, argc, argv, "maximum-seeds"), "maximum-seeds");
            if (value <= 0)
                fail("--maximum-seeds must be positive");
            options.anchors.maximumSeedCount = static_cast<size_t>(value);
        } else if (argument == "--maximum-iterations" && options.command == Command::Anchors) {
            options.anchors.maximumIterations = parseInt(valueAfter(index, argc, argv, "maximum-iterations"), "maximum-iterations");
        } else if (argument == "--crop" && options.command == Command::Anchors) {
            options.baseCrop = parseCrop(valueAfter(index, argc, argv, "crop"));
        } else if (argument == "--glyph-length-base-voxels" && options.command == Command::Anchors) {
            options.glyphLengthBaseVoxels =
                parseDouble(valueAfter(index, argc, argv, "glyph-length-base-voxels"), "glyph-length-base-voxels");
        } else if (argument == "--normal-manifest" && options.command == Command::Paths) {
            options.normalManifestLocation = valueAfter(index, argc, argv, "normal-manifest");
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
        } else if (argument == "--presence-weight" && options.command == Command::Paths) {
            options.paths.presenceWeight = parseDouble(valueAfter(index, argc, argv, "presence-weight"), "presence-weight");
        } else if (argument == "--direction-weight" && options.command == Command::Paths) {
            options.paths.directionWeight = parseDouble(valueAfter(index, argc, argv, "direction-weight"), "direction-weight");
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
        } else {
            fail("unknown option for selected command: " + argument);
        }
    }
    if (options.baseVoxelSizeUm.has_value() && !(*options.baseVoxelSizeUm > 0.0))
        fail("--base-voxel-size-um must be positive");
    if (options.command == Command::Anchors) {
        if (!options.sigmaExplicit) {
            options.anchors.gaussianSigmaPredictionVoxels = static_cast<double>(options.anchors.cellSizePredictionVoxels) * 0.5;
        }
        vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
    } else {
        if (options.normalManifestLocation.empty())
            fail("paths requires --normal-manifest");
        vc::fiber_tracer::validateFiberletPathConfig(options.paths);
    }
    const bool remote = vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) ||
                        (options.command == Command::Paths && vc::lasagna::isRemoteLasagnaLocation(options.normalManifestLocation));
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

        if (options.command == Command::Anchors) {
            const auto crop = options.baseCrop.has_value()
                                  ? std::make_optional(vc::fiber_tracer::fiberAnchorCropFromBaseVoxels(*options.baseCrop, grid.predictionToBaseScale))
                                  : std::nullopt;
            const auto report = vc::fiber_tracer::extractFiberAnchors(
                grid,
                options.anchors,
                [&](const auto& indices, int threads, auto& samples) { field.sampleStoredGridBatch(indices, threads, samples); },
                crop);
            vc::fiber_tracer::FiberAnchorArtifactInfo artifact;
            artifact.sourceLocator = datasetLocator(dataset);
            artifact.manifestContentHash = fileHash(dataset.manifest().manifestPath);
            artifact.glyphLengthBaseVoxels = options.glyphLengthBaseVoxels;
            artifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
            vc::fiber_tracer::writeFiberAnchorArtifacts(options.outputDirectory, report, artifact);

            const double cellSideBase = options.anchors.cellSizePredictionVoxels * grid.predictionToBaseScale;
            std::cout << "prediction_shape_zyx=" << grid.shapeZYX[0] << ',' << grid.shapeZYX[1] << ',' << grid.shapeZYX[2]
                      << " prediction_to_base=" << grid.predictionToBaseScale << " cell_side_base_voxels=" << cellSideBase
                      << " cell_diagonal_base_voxels=" << cellSideBase * std::sqrt(3.0) << " cells=" << report.diagnostics.totalCells
                      << " anchors=" << report.diagnostics.oneAnchorCells + 2 * report.diagnostics.twoAnchorCells
                      << " zero=" << report.diagnostics.zeroAnchorCells << " one=" << report.diagnostics.oneAnchorCells
                      << " two=" << report.diagnostics.twoAnchorCells << " elapsed_seconds=" << report.elapsedSeconds << '\n';
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
            normalSampler);
        vc::fiber_tracer::FiberletArtifactInfo artifact;
        artifact.fiberManifestLocator = datasetLocator(dataset);
        artifact.fiberManifestContentHash = manifestHash;
        artifact.normalManifestLocator = datasetLocator(normalDataset);
        artifact.normalManifestContentHash = fileHash(normalDataset.manifest().manifestPath);
        artifact.anchorArtifactLocator = std::filesystem::absolute(options.anchorArtifact).lexically_normal().string();
        artifact.anchorArtifactContentHash = fileHash(options.anchorArtifact);
        artifact.baseVoxelSizeUm = options.baseVoxelSizeUm.has_value() ? options.baseVoxelSizeUm : anchors.artifact.baseVoxelSizeUm;
        vc::fiber_tracer::writeFiberletPathArtifacts(options.outputDirectory, report, artifact);
        std::cout << "anchors=" << report.diagnostics.occupiedAnchors << " shell_offsets=" << report.diagnostics.shellOffsets
                  << " generated_pairs=" << report.diagnostics.generatedPairs << " axis_rejected=" << report.diagnostics.axisRejectedPairs
                  << " searched=" << report.diagnostics.searchedPairs << " successful=" << report.diagnostics.successfulPaths
                  << " no_path=" << report.diagnostics.noPathPairs << " elapsed_seconds=" << report.elapsedSeconds << '\n';
        if (options.printStats) {
            const auto statistics = vc::fiber_tracer::fiberletPathStatistics(report);
            std::cout << "fiberlet_stats anchors=" << statistics.anchors << " total_candidate_pairs=" << statistics.candidates
                      << " pre_dp_rejections=" << statistics.preDpRejected << " dp_searches=" << statistics.dpSearched
                      << " searched_unscored=" << statistics.searchedUnscored << " scored_fiberlets=" << statistics.scored
                      << " accepted_fiberlets=" << statistics.accepted << " unscored_candidates=" << statistics.unscored << '\n';
            const auto printScores = [](const char* name, const vc::fiber_tracer::FiberletScoreStatistics& scores) {
                std::cout << name << " count=" << scores.count;
                if (scores.count == 0) {
                    std::cout << " min=n/a mean=n/a max=n/a\n";
                } else {
                    std::cout << " min=" << *scores.minimum << " mean=" << *scores.mean << " max=" << *scores.maximum << '\n';
                }
            };
            printScores("fiberlet_scores_all", statistics.allScores);
            printScores("fiberlet_scores_accepted", statistics.acceptedScores);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiberlets: " << error.what() << '\n';
        return 1;
    }
}
