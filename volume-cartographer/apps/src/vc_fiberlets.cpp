#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/lasagna/Dataset.hpp"

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
#include <vector>

namespace {

struct CliOptions {
    std::string manifestLocation;
    std::filesystem::path outputDirectory;
    std::filesystem::path remoteCacheDirectory;
    vc::fiber_tracer::FiberAnchorConfig anchors;
    std::optional<vc::fiber_tracer::FiberAnchorCrop> crop;
    std::optional<double> baseVoxelSizeUm;
    double glyphLengthBaseVoxels = 16.0;
    size_t decodedCacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    bool sigmaExplicit = false;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " anchors <fiber.lasagna.json-or-url> <output-dir> [options]\n\n"
        << "Options:\n"
        << "  --cell-size N                 prediction-grid cell side, 2..8 [4]\n"
        << "  --gaussian-sigma N            Gaussian sigma in prediction voxels [cell-size/2]\n"
        << "  --presence-floor N            inclusive observation presence floor [0.05]\n"
        << "  --minimum-support N           inclusive aligned-support threshold [0.05]\n"
        << "  --maximum-seeds N             deterministic PCA seed count [8]\n"
        << "  --maximum-iterations N        assignment/PCA iteration limit [64]\n"
        << "  --threads N                   stored-grid decode workers [hardware]\n"
        << "  --crop-prediction-xyzwhd V    comma-separated X,Y,Z,W,H,D whole-cell selector\n"
        << "  --cache-gib N                 decoded chunk cache budget [8]\n"
        << "  --remote-cache-dir PATH       required for a direct remote manifest\n"
        << "  --glyph-length-base-voxels N  diagnostic OBJ line length [16]\n"
        << "  --base-voxel-size-um N        optional physical reporting metadata\n";
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
    if (parsed != text.size() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
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
            fail("--crop-prediction-xyzwhd requires X,Y,Z,W,H,D");
        size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size() || value > std::numeric_limits<size_t>::max())
            fail("--crop-prediction-xyzwhd contains an invalid integer");
        values[index] = static_cast<size_t>(value);
    }
    if (std::getline(input, token, ','))
        fail("--crop-prediction-xyzwhd requires exactly six integers");
    return {{values[0], values[1], values[2]},
            {values[3], values[4], values[5]}};
}

CliOptions parseArgs(int argc, char** argv)
{
    if (argc == 2 && (std::string(argv[1]) == "--help" ||
                      std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        std::exit(0);
    }
    if (argc < 4 || std::string(argv[1]) != "anchors") {
        usage(argv[0]);
        std::exit(2);
    }
    CliOptions options;
    options.manifestLocation = argv[2];
    options.outputDirectory = argv[3];
    options.anchors.parallelThreads = static_cast<int>(std::max(
        1U, std::thread::hardware_concurrency()));
    for (int index = 4; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (argument == "--cell-size") {
            options.anchors.cellSizePredictionVoxels = parseInt(
                valueAfter(index, argc, argv, "cell-size"), "cell-size");
        } else if (argument == "--gaussian-sigma") {
            options.anchors.gaussianSigmaPredictionVoxels = parseDouble(
                valueAfter(index, argc, argv, "gaussian-sigma"), "gaussian-sigma");
            options.sigmaExplicit = true;
        } else if (argument == "--presence-floor") {
            options.anchors.observationPresenceFloor = parseDouble(
                valueAfter(index, argc, argv, "presence-floor"), "presence-floor");
        } else if (argument == "--minimum-support") {
            options.anchors.minimumAlignedSupport = parseDouble(
                valueAfter(index, argc, argv, "minimum-support"), "minimum-support");
        } else if (argument == "--maximum-seeds") {
            const int value = parseInt(
                valueAfter(index, argc, argv, "maximum-seeds"), "maximum-seeds");
            if (value < 0)
                fail("--maximum-seeds must be positive");
            options.anchors.maximumSeedCount = static_cast<size_t>(value);
        } else if (argument == "--maximum-iterations") {
            options.anchors.maximumIterations = parseInt(
                valueAfter(index, argc, argv, "maximum-iterations"),
                "maximum-iterations");
        } else if (argument == "--threads") {
            options.anchors.parallelThreads = parseInt(
                valueAfter(index, argc, argv, "threads"), "threads");
        } else if (argument == "--crop-prediction-xyzwhd") {
            options.crop = parseCrop(
                valueAfter(index, argc, argv, "crop-prediction-xyzwhd"));
        } else if (argument == "--cache-gib") {
            const double gib = parseDouble(
                valueAfter(index, argc, argv, "cache-gib"), "cache-gib");
            if (!(gib > 0.0) || gib > 1024.0)
                fail("--cache-gib must be in (0, 1024]");
            options.decodedCacheBytes = static_cast<size_t>(
                gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory =
                valueAfter(index, argc, argv, "remote-cache-dir");
        } else if (argument == "--glyph-length-base-voxels") {
            options.glyphLengthBaseVoxels = parseDouble(
                valueAfter(index, argc, argv, "glyph-length-base-voxels"),
                "glyph-length-base-voxels");
        } else if (argument == "--base-voxel-size-um") {
            options.baseVoxelSizeUm = parseDouble(
                valueAfter(index, argc, argv, "base-voxel-size-um"),
                "base-voxel-size-um");
        } else {
            fail("unknown option: " + argument);
        }
    }
    if (!options.sigmaExplicit) {
        options.anchors.gaussianSigmaPredictionVoxels =
            static_cast<double>(options.anchors.cellSizePredictionVoxels) * 0.5;
    }
    if (vc::lasagna::isRemoteLasagnaLocation(options.manifestLocation) &&
        options.remoteCacheDirectory.empty()) {
        fail("a direct remote manifest requires --remote-cache-dir");
    }
    if (options.baseVoxelSizeUm.has_value() &&
        !(*options.baseVoxelSizeUm > 0.0)) {
        fail("--base-voxel-size-um must be positive");
    }
    vc::fiber_tracer::validateFiberAnchorConfig(options.anchors);
    return options;
}

std::string fileHash(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot hash materialized manifest: " + path.string());
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
        throw std::runtime_error("failed while hashing manifest: " + path.string());
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

} // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parseArgs(argc, argv);
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
            options.manifestLocation, openOptions);
        const vc::fiber_tracer::FiberPredictionField field(
            dataset,
            options.decodedCacheBytes,
            vc::fiber_tracer::FiberPredictionFieldBindingMode::CanonicalStoredGrid);
        const auto grid = field.storedGridInfo();
        const auto report = vc::fiber_tracer::extractFiberAnchors(
            grid,
            options.anchors,
            [&](const auto& indices, int threads, auto& samples) {
                field.sampleStoredGridBatch(indices, threads, samples);
            },
            options.crop);

        const auto& manifest = dataset.manifest();
        const std::string sourceLocator = manifest.manifestIsRemote
            ? credentialFreeLocator(manifest.manifestLocation)
            : std::filesystem::absolute(manifest.manifestPath)
                  .lexically_normal().string();
        vc::fiber_tracer::FiberAnchorArtifactInfo artifact;
        artifact.sourceLocator = sourceLocator;
        artifact.manifestContentHash = fileHash(manifest.manifestPath);
        artifact.glyphLengthBaseVoxels = options.glyphLengthBaseVoxels;
        artifact.baseVoxelSizeUm = options.baseVoxelSizeUm;
        vc::fiber_tracer::writeFiberAnchorArtifacts(
            options.outputDirectory, report, artifact);

        const double cellSideBase =
            options.anchors.cellSizePredictionVoxels * grid.predictionToBaseScale;
        std::cout << "prediction_shape_zyx=" << grid.shapeZYX[0] << ','
                  << grid.shapeZYX[1] << ',' << grid.shapeZYX[2]
                  << " prediction_to_base=" << grid.predictionToBaseScale
                  << " cell_side_base_voxels=" << cellSideBase
                  << " cell_diagonal_base_voxels=" << cellSideBase * std::sqrt(3.0)
                  << " cells=" << report.diagnostics.totalCells
                  << " anchors=" << report.diagnostics.oneAnchorCells +
                         2 * report.diagnostics.twoAnchorCells
                  << " zero=" << report.diagnostics.zeroAnchorCells
                  << " one=" << report.diagnostics.oneAnchorCells
                  << " two=" << report.diagnostics.twoAnchorCells
                  << " elapsed_seconds=" << report.elapsedSeconds << '\n';
        if (options.baseVoxelSizeUm.has_value()) {
            std::cout << "cell_side_um=" << cellSideBase * *options.baseVoxelSizeUm
                      << " cell_diagonal_um="
                      << cellSideBase * std::sqrt(3.0) * *options.baseVoxelSizeUm
                      << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiberlets: " << error.what() << '\n';
        return 1;
    }
}
