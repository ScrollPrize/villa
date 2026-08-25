#include "vc/core/io/PolylineObj.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropVisualization.hpp"
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

struct Options {
    std::filesystem::path fiberlets;
    std::string normalManifest;
    std::filesystem::path remoteCacheDirectory;
    std::filesystem::path output;
    std::filesystem::path volume;
    int maximumTextureDimension = 4096;
    int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    std::size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    vc::fiber_tracer::FiberletCropTraceConfig trace;
    bool hasBounds = false;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable)
{
    std::cerr << "Usage: " << executable << " <fiberlets.zarr>"
              << " --normal-manifest PATH"
              << " --bbox X0 Y0 Z0 X1 Y1 Z1"
              << " --output lines.obj [options]\n\n"
              << "  --volume PATH              concrete uint8 CT Zarr group\n"
              << "  --remote-cache-dir PATH    cache for a remote normal manifest\n"
              << "  --threads N                graph preparation and trace workers [host CPUs]\n"
              << "  --cache-gib N              decoded graph/normal cache [8]\n"
              << "  --beam N                   retained lookahead candidates [16]\n"
              << "  --lookahead N              lookahead in base voxels [384]\n"
              << "  --coverage N               normal coverage radius in base voxels [20]\n"
              << "  --coverage-angle N         parallel-axis coverage angle [25]\n"
              << "  --max-attempts N           anchor attempt limit; zero is unlimited [0]\n"
              << "  --max-fibers N             accepted line limit; zero is unlimited [0]\n"
              << "  --texture-max N            maximum bbox texture dimension [4096]\n";
}

std::string value(int& index, int argc, char** argv, const char* option)
{
    if (++index >= argc)
        fail(std::string(option) + " requires a value");
    return argv[index];
}

double number(int& index, int argc, char** argv, const char* option)
{
    const std::string text = value(index, argc, argv, option);
    std::size_t parsed = 0;
    const double result = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(result))
        fail(std::string(option) + " requires a finite number");
    return result;
}

std::size_t count(int& index, int argc, char** argv, const char* option)
{
    const std::string text = value(index, argc, argv, option);
    if (text.starts_with('-'))
        fail(std::string(option) + " requires a non-negative integer");
    std::size_t parsed = 0;
    const auto result = std::stoull(text, &parsed);
    if (parsed != text.size())
        fail(std::string(option) + " requires a non-negative integer");
    return static_cast<std::size_t>(result);
}

Options parse(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage(argv[0]);
        std::exit(argc < 2 ? 2 : 0);
    }
    Options options;
    options.fiberlets = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--normal-manifest") {
            options.normalManifest = value(index, argc, argv, "--normal-manifest");
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = value(index, argc, argv, "--remote-cache-dir");
        } else if (argument == "--output") {
            options.output = value(index, argc, argv, "--output");
        } else if (argument == "--volume") {
            options.volume = value(index, argc, argv, "--volume");
        } else if (argument == "--bbox") {
            options.trace.minimumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.trace.maximumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.hasBounds = true;
        } else if (argument == "--threads") {
            options.threads = static_cast<int>(count(index, argc, argv, "--threads"));
        } else if (argument == "--cache-gib") {
            const double gib = number(index, argc, argv, "--cache-gib");
            if (!(gib > 0.0))
                fail("--cache-gib must be positive");
            options.cacheBytes = static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (argument == "--beam") {
            options.trace.beamWidth = count(index, argc, argv, "--beam");
        } else if (argument == "--lookahead") {
            options.trace.lookaheadDistanceBaseVoxels = number(index, argc, argv, "--lookahead");
        } else if (argument == "--coverage") {
            options.trace.coverageNormalRadiusBaseVoxels = number(index, argc, argv, "--coverage");
        } else if (argument == "--coverage-angle") {
            options.trace.coverageDirectionDegrees = number(index, argc, argv, "--coverage-angle");
        } else if (argument == "--max-attempts") {
            options.trace.maximumAttempts = count(index, argc, argv, "--max-attempts");
        } else if (argument == "--max-fibers") {
            options.trace.maximumFibers = count(index, argc, argv, "--max-fibers");
        } else if (argument == "--texture-max") {
            options.maximumTextureDimension = static_cast<int>(count(index, argc, argv, "--texture-max"));
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown option: " + argument);
        }
    }
    if (options.normalManifest.empty())
        fail("--normal-manifest is required");
    if (!options.hasBounds)
        fail("--bbox is required");
    if (options.output.empty())
        fail("--output is required");
    if (options.threads < 1)
        fail("--threads must be positive");
    if (options.maximumTextureDimension < 2)
        fail("--texture-max must be at least two");
    options.trace.parallelThreads = static_cast<std::size_t>(options.threads);
    if (vc::lasagna::isRemoteLasagnaLocation(options.normalManifest) && options.remoteCacheDirectory.empty()) {
        fail("a remote normal manifest requires --remote-cache-dir");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse(argc, argv);
        auto dataset = vc::fiber_tracer::FiberletChunkDataset::openExisting(options.fiberlets);
        if (dataset->metadata().kind != vc::fiber_tracer::FiberletDatasetKind::Combined) {
            fail("input must be a combined Fiberlet dataset");
        }

        vc::fiber_tracer::FiberletChunkCacheOptions cacheOptions;
        cacheOptions.service.decodedByteCapacity = options.cacheBytes;
        cacheOptions.service.fetchConcurrency.workerCapacity = static_cast<std::size_t>(options.threads);
        cacheOptions.service.fetchConcurrency.maxConcurrentReads = static_cast<std::size_t>(options.threads);
        vc::fiber_tracer::FiberletStoredReplayGraphSource graph(dataset, cacheOptions);

        vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
        normalOptions.workingToBaseScale = dataset->metadata().predictionToBaseScale;
        normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(options.normalManifest, normalOptions);
        vc::fiber_tracer::validateFiberletNormalDatasetCompatibility(
            dataset->metadata(), normalDataset);
        const vc::lasagna::LasagnaNormalSampler normals(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});

        const auto graphStarted = std::chrono::steady_clock::now();
        const auto graphCpuStarted = std::clock();
        auto materialized = graph.materializeBaseBox(
            options.trace.minimumBaseXYZ, options.trace.maximumBaseXYZ,
            static_cast<std::size_t>(options.threads));
        const double graphSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - graphStarted).count();
        const double graphCpuSeconds = static_cast<double>(
            std::clock() - graphCpuStarted) / CLOCKS_PER_SEC;
        std::cout << "fiberlet crop graph prepared"
                  << " anchors=" << materialized.insideAnchors.size()
                  << " prediction_to_base=" << dataset->metadata().predictionToBaseScale
                  << " elapsed_seconds=" << graphSeconds
                  << " cpu_seconds=" << graphCpuSeconds << '\n';
        const auto traceStarted = std::chrono::steady_clock::now();
        const auto traceCpuStarted = std::clock();
        const auto result =
            vc::fiber_tracer::traceFiberletCrop(*materialized.graph, std::move(materialized.insideAnchors), normals, dataset->metadata().predictionToBaseScale, options.trace, [](const auto& current, std::size_t remaining) {
                std::cout << "fiberlet crop attempted=" << current.attemptedAnchors << " accepted=" << current.lines.size()
                          << " covered=" << current.coveredAnchors << " remaining=" << remaining << '\n';
            });
        const double traceSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - traceStarted).count();
        const double traceCpuSeconds = static_cast<double>(
            std::clock() - traceCpuStarted) / CLOCKS_PER_SEC;

        std::filesystem::create_directories(options.output.parent_path().empty() ? std::filesystem::path{"."} : options.output.parent_path());
        std::vector<vc::core::io::NamedPolyline> lines;
        lines.reserve(result.lines.size());
        for (std::size_t index = 0; index < result.lines.size(); ++index) {
            std::ostringstream name;
            name << "fiber_" << std::setw(6) << std::setfill('0') << index << "_presence_" << std::fixed << std::setprecision(4)
                 << result.lines[index].seedPresence;
            lines.push_back({name.str(), result.lines[index].pointsBaseXYZ});
        }
        vc::core::io::writePolylinesObj(lines, options.output, "VC3D Fiberlet crop traces");

        if (!options.volume.empty()) {
            const std::string locator = std::filesystem::absolute(options.volume).lexically_normal().string();
            auto volume = Volume::New(options.volume);
            vc::fiber_tracer::writeFiberletCropBoxVisualization(
                *volume,
                locator,
                options.trace.minimumBaseXYZ,
                options.trace.maximumBaseXYZ,
                options.output.parent_path() / (options.output.stem().string() + "_volume_slices.obj"),
                options.maximumTextureDimension);
        }

        std::cout << "fiberlet crop completed"
                  << " candidates=" << result.candidateAnchors << " attempted=" << result.attemptedAnchors << " covered=" << result.coveredAnchors
                  << " computed=" << result.computedCandidates << " discarded=" << result.discardedCandidates
                  << " accepted=" << result.lines.size() << " no_edge=" << result.noEdgeAnchors << " one_sided=" << result.oneSidedLines
                  << " bidirectional=" << result.bidirectionalLines << " output=" << options.output << '\n';
        std::cout << "fiberlet crop timing"
                  << " graph_seconds=" << graphSeconds
                  << " graph_cpu_seconds=" << graphCpuSeconds
                  << " trace_seconds=" << traceSeconds
                  << " trace_cpu_seconds=" << traceCpuSeconds
                  << " candidate_batch_seconds=" << result.candidateBatchSeconds
                  << " candidate_batch_cpu_seconds=" << result.candidateBatchCpuSeconds
                  << " candidate_task_seconds=" << result.candidateTaskSeconds
                  << " candidate_task_max_seconds=" << result.maximumCandidateTaskSeconds
                  << " integration_seconds=" << result.integrationSeconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_trace_chunk: " << error.what() << '\n';
        return 1;
    }
}
