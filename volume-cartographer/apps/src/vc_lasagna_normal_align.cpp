#include "vc/fiber_tracer/LasagnaNormalAlignment.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Options {
    std::string manifest;
    std::filesystem::path remoteCacheDirectory;
    std::filesystem::path outputBase;
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    bool hasBounds = false;
    double spacingBaseVoxels = 0.0;
    int neighborRadius = 1;
    int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    std::size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    vc::fiber_tracer::LasagnaNormalAlignmentConfig alignment;
    double glyphBaseRadius = 0.0;
    double glyphDirectionLength = 0.0;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable)
{
    std::cerr << "Usage:\n  " << executable
              << " <normal-manifest> --bbox X0 Y0 Z0 X1 Y1 Z1"
                 " --output BASENAME [options]\n\n"
              << "Options:\n"
              << "  --remote-cache-dir PATH   required for a remote manifest\n"
              << "  --spacing N               globally anchored sample spacing [normal level]\n"
              << "  --neighbor-radius N       Chebyshev lattice radius [1]\n"
              << "  --threads N               Lasagna sampling and BP workers [host CPUs]\n"
              << "  --cache-gib N             decoded Lasagna cache [8]\n"
              << "  --temperature F           binary BP temperature [0.25]\n"
              << "  --message-iterations N    BP iteration limit [500]\n"
              << "  --damping F               BP message damping in (0,1] [0.5]\n"
              << "  --residual F              BP convergence tolerance [1e-8]\n"
              << "  --glyph-base-radius N     crossed-base half length [0.2*spacing]\n"
              << "  --glyph-direction-length N directed-stroke length [0.8*spacing]\n";
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
    options.manifest = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--bbox") {
            options.minimumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.maximumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.hasBounds = true;
        } else if (argument == "--output") {
            options.outputBase = value(index, argc, argv, "--output");
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = value(index, argc, argv, "--remote-cache-dir");
        } else if (argument == "--spacing") {
            options.spacingBaseVoxels = number(index, argc, argv, "--spacing");
        } else if (argument == "--neighbor-radius") {
            options.neighborRadius = static_cast<int>(count(index, argc, argv, "--neighbor-radius"));
        } else if (argument == "--threads") {
            options.threads = static_cast<int>(count(index, argc, argv, "--threads"));
        } else if (argument == "--cache-gib") {
            const double gib = number(index, argc, argv, "--cache-gib");
            if (!(gib > 0.0) || gib > static_cast<double>(std::numeric_limits<std::size_t>::max()) / static_cast<double>(1ULL << 30)) {
                fail("--cache-gib must be positive and representable");
            }
            options.cacheBytes = static_cast<std::size_t>(gib * static_cast<double>(1ULL << 30));
        } else if (argument == "--temperature") {
            options.alignment.beliefPropagation.temperature = number(index, argc, argv, "--temperature");
        } else if (argument == "--message-iterations") {
            options.alignment.beliefPropagation.maximumMessageIterations = count(index, argc, argv, "--message-iterations");
        } else if (argument == "--damping") {
            options.alignment.beliefPropagation.messageDamping = number(index, argc, argv, "--damping");
        } else if (argument == "--residual") {
            options.alignment.beliefPropagation.messageResidualTolerance = number(index, argc, argv, "--residual");
        } else if (argument == "--glyph-base-radius") {
            options.glyphBaseRadius = number(index, argc, argv, "--glyph-base-radius");
        } else if (argument == "--glyph-direction-length") {
            options.glyphDirectionLength = number(index, argc, argv, "--glyph-direction-length");
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown option: " + argument);
        }
    }
    if (!options.hasBounds)
        fail("--bbox is required");
    if (options.outputBase.empty())
        fail("--output is required");
    for (int axis = 0; axis < 3; ++axis) {
        if (!(options.maximumBaseXYZ[axis] > options.minimumBaseXYZ[axis]))
            fail("--bbox must have strictly increasing half-open XYZ bounds");
    }
    if (options.spacingBaseVoxels < 0.0)
        fail("--spacing must be positive");
    if (options.neighborRadius < 1)
        fail("--neighbor-radius must be positive");
    if (options.threads < 1)
        fail("--threads must be positive");
    if (vc::lasagna::isRemoteLasagnaLocation(options.manifest) && options.remoteCacheDirectory.empty()) {
        fail("a remote normal manifest requires --remote-cache-dir");
    }
    return options;
}

std::filesystem::path suffixedObj(std::filesystem::path base, const std::string& suffix)
{
    if (base.extension() == ".obj")
        base.replace_extension();
    return base.parent_path() / (base.filename().string() + suffix + ".obj");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse(argc, argv);
        vc::lasagna::LasagnaDatasetOpenOptions openOptions;
        openOptions.workingToBaseScale = 1.0;
        openOptions.remoteCacheRoot = options.remoteCacheDirectory;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(options.manifest, openOptions);
        vc::lasagna::validateLasagnaNormalDatasetStructure(dataset);
        const auto nx = vc::lasagna::bindLasagnaChannel(dataset.manifest(), "nx");
        const double spacing = options.spacingBaseVoxels > 0.0 ? options.spacingBaseVoxels : nx.spacing;
        if (!std::isfinite(spacing) || !(spacing > 0.0))
            fail("normal sample spacing must be finite and positive");

        const auto& baseShapeZYX = *dataset.manifest().baseShapeZYX;
        const cv::Vec3d baseMaximumXYZ{static_cast<double>(baseShapeZYX[2]), static_cast<double>(baseShapeZYX[1]), static_cast<double>(baseShapeZYX[0])};
        for (int axis = 0; axis < 3; ++axis) {
            if (options.minimumBaseXYZ[axis] < 0.0 || options.maximumBaseXYZ[axis] > baseMaximumXYZ[axis]) {
                fail("--bbox lies outside manifest base_shape_zyx");
            }
        }

        const vc::lasagna::LasagnaNormalSampler sampler(dataset, vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});
        const auto field = vc::fiber_tracer::sampleAndAlignLasagnaNormalLattice(
            sampler,
            options.minimumBaseXYZ,
            options.maximumBaseXYZ,
            spacing,
            options.neighborRadius,
            options.threads,
            options.alignment);
        const auto factors = vc::fiber_tracer::makeLasagnaNormalLatticeFactors(
            field.lattice,
            field.nodeByLatticeSample,
            field.rawNormals,
            options.neighborRadius);
        const auto& positions = field.positionsBaseXYZ;
        const auto& normals = field.rawNormals;
        const auto& alignment = field.alignment;
        std::size_t negativeBefore = 0;
        std::size_t negativeAfter = 0;
        double dotBeforeSum = 0.0;
        double dotAfterSum = 0.0;
        for (const auto& factor : factors) {
            const double dotBefore = factor.differentCost - factor.sameCost;
            const bool flippedA = alignment.flipProbability[factor.a] > 0.5;
            const bool flippedB = alignment.flipProbability[factor.b] > 0.5;
            const double dotAfter = flippedA == flippedB ? dotBefore : -dotBefore;
            negativeBefore += dotBefore < 0.0 ? 1 : 0;
            negativeAfter += dotAfter < 0.0 ? 1 : 0;
            dotBeforeSum += dotBefore;
            dotAfterSum += dotAfter;
        }
        const double baseRadius = options.glyphBaseRadius > 0.0 ? options.glyphBaseRadius : 0.2 * spacing;
        const double directionLength = options.glyphDirectionLength > 0.0 ? options.glyphDirectionLength : 0.8 * spacing;
        const vc::fiber_tracer::NormalGlyphObjConfig glyph{baseRadius, directionLength};
        const auto unalignedObj = suffixedObj(options.outputBase, "_unaligned");
        const auto alignedObj = suffixedObj(options.outputBase, "_aligned");
        vc::fiber_tracer::writeNormalGlyphObj(unalignedObj, positions, normals, glyph);
        vc::fiber_tracer::writeNormalGlyphObj(alignedObj, positions, alignment.alignedNormals, glyph);

        std::cout << "Lasagna normal BP alignment\n"
                  << "spacing_base=" << spacing << " candidates=" << field.candidateSamples << " valid=" << normals.size()
                  << " invalid=" << field.candidateSamples - normals.size() << " factors=" << factors.size() << " components=" << alignment.connectedComponents
                  << " isolated=" << alignment.isolatedSamples << " flipped=" << alignment.flippedSamples
                  << " negative_links_before=" << negativeBefore << " negative_links_after=" << negativeAfter
                  << " mean_neighbor_dot_before=" << (factors.empty() ? 0.0 : dotBeforeSum / factors.size())
                  << " mean_neighbor_dot_after=" << (factors.empty() ? 0.0 : dotAfterSum / factors.size())
                  << " iterations=" << alignment.beliefPropagation.messageIterations
                  << " residual=" << alignment.beliefPropagation.messageResidual << " converged=" << std::boolalpha
                  << alignment.beliefPropagation.messageConverged << " bp_workers=" << alignment.beliefPropagation.effectiveWorkers
                  << " bp_setup_ms=" << alignment.beliefPropagation.setupMilliseconds << " bp_totals_ms=" << alignment.beliefPropagation.nodeTotalMilliseconds
                  << " bp_updates_ms=" << alignment.beliefPropagation.messageUpdateMilliseconds
                  << " bp_solve_ms=" << alignment.beliefPropagation.solveMilliseconds << " bp_ms=" << alignment.beliefPropagation.elapsedMilliseconds
                  << " prefetch_ms=" << field.prefetchMilliseconds << " materialize_ms=" << field.materializeMilliseconds << '\n'
                  << "unaligned_obj=" << unalignedObj << '\n'
                  << "aligned_obj=" << alignedObj << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_lasagna_normal_align error: " << error.what() << '\n';
        return 1;
    }
}
