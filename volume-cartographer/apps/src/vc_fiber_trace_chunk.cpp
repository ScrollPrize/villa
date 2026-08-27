#include "vc/core/types/Volume.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberletCropVisualization.hpp"
#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/FiberTraceConsensus.hpp"
#include "vc/fiber_tracer/FiberTraceLabeling.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

enum class Mode {
    Trace,
    Visualize,
    Constraints,
    Consensus,
    DirectionDiagnostic,
    DirectionAblation,
};

enum class BpBalanceSelection {
    None,
    Soft,
    Tight,
    Both,
};

struct Options {
    Mode mode = Mode::Trace;
    std::filesystem::path input;
    std::string normalManifest;
    std::filesystem::path remoteCacheDirectory;
    std::filesystem::path output;
    std::filesystem::path obj;
    std::filesystem::path volume;
    int maximumTextureDimension = 4096;
    double directionDominance =
        vc::fiber_tracer::kFiberDirectionDominanceFraction;
    int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    std::size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    vc::fiber_tracer::FiberletCropTraceConfig trace;
    vc::fiber_tracer::FiberTraceConstraintConfig constraints;
    vc::fiber_tracer::FiberTraceLabelingConfig labeling;
    std::optional<std::size_t> maximumConstraintsPerFiber;
    std::size_t ablationStep = 5;
    std::optional<std::size_t> ablationLimit;
    std::size_t postIterations = 0;
    double postInfluence = 1.0;
    BpBalanceSelection bpBalance = BpBalanceSelection::None;
    vc::fiber_tracer::FiberTraceBeliefInference bpInference =
        vc::fiber_tracer::FiberTraceBeliefInference::MinSum;
    bool bpOnly = false;
    vc::fiber_tracer::FiberTraceBeliefPropagationConfig bp;
    bool hasBounds = false;
    bool hasTraceOnlyOption = false;
    bool hasConstraintOnlyOption = false;
    bool hasSolverOnlyOption = false;
    bool hasSharedRuntimeOption = false;
    bool hasDirectionVisualizationOption = false;
    bool hasHvOnlyOption = false;
    bool hasAblationOnlyOption = false;
    bool hasPostInfluenceOption = false;
    bool hasBpTuningOption = false;
    bool hasBpInferenceOption = false;
    bool hasBpBalanceTuningOption = false;
    bool hasBpMixedCostOption = false;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

void usage(const char* executable)
{
    std::cerr << "Usage:\n"
              << "  " << executable
              << " trace <fiberlets.zarr> --normal-manifest PATH"
                 " --bbox X0 Y0 Z0 X1 Y1 Z1 --output traces.zarr [options]\n"
              << "  " << executable << " visualize <traces.zarr> --output lines.obj\n\n"
              << "  " << executable
              << " constraints <traces.zarr> --normal-manifest PATH"
                 " [--output BASENAME] [--broken-cost-per-link COST]"
                 " [--mip-gap FRACTION] [--lp-relaxation]"
                 " [--lp-parallel] [--lp-solver NAME]"
                 " [--hv-only] [--exact-perpendicular-milp]"
                 " [--perpendicular-only]"
                 " [--exclude-parallel-separate-winding] [options]\n\n"
              << "  " << executable
              << " consensus <traces.zarr> --normal-manifest PATH"
                 " [--output BASENAME] [--broken-cost-per-link COST] [options]\n\n"
              << "  " << executable
              << " direction-diagnostic <traces.zarr> --normal-manifest PATH"
                 " [--output BASENAME] [--direction-dominance F] [options]\n\n"
              << "  " << executable
              << " direction-ablation <traces.zarr> --normal-manifest PATH"
                 " [--output BASENAME] [--direction-dominance F]"
                 " [--ablation-step N] [--ablation-limit N]"
                 " [--perpendicular-only] [--post-iterations N]"
                 " [--post-influence F] [--bp-only]"
                 " [--bp-balance MODE] [options]\n\n"
              << "Trace options:\n"
              << "  --obj PATH                 line OBJ; defaults beside trace Zarr\n"
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
              << "  --texture-max N            maximum bbox texture dimension [4096]\n\n"
              << "  --direction-dominance F   direction support/arc fraction in (0.5,1] [0.75]\n\n"
              << "  --ablation-step N         mixed fibers admitted per checkpoint [5]\n\n"
              << "  --ablation-limit N        stop after admitting N mixed fibers [all]\n\n"
              << "  --post-iterations N       perpendicular H/V consensus iterations [0]\n"
              << "  --post-influence F        neighbor confidence support width in (0,1] [1]\n\n"
              << "Belief-propagation options (direction-ablation only):\n"
              << "  --bp-only                 run only final-cohort BP; skip HiGHS\n"
              << "  --bp-inference MODE       min-sum, sum-product, or sum-product-mixed [min-sum]\n"
              << "  --bp-mixed-cost F         Mixed-state unary cost per piece [0.5]\n"
              << "  --bp-balance MODE         soft, tight, or both [disabled]\n"
              << "  --bp-target F             arc-weighted H fraction [0.5]\n"
              << "  --bp-soft-strength F      quadratic balance strength [1]\n"
              << "  --bp-temperature F        min-marginal decoding temperature [0.25]\n"
              << "  --bp-message-iterations N message update limit [500]\n"
              << "  --bp-balance-iterations N field update limit [64]\n"
              << "  --bp-damping F            message damping in (0,1] [0.5]\n"
              << "  --bp-residual F           message residual tolerance [1e-8]\n"
              << "  --bp-balance-tolerance F  target/field tolerance [1e-3]\n\n"
              << "Constraint options (all distances are base voxels):\n"
              << "  --output PATH              OBJ basename; defaults beside trace dataset\n"
              << "  --sample-step N            common trace resampling step [32]\n"
              << "  --piece-length N           target overlapping piece length [512]\n"
              << "  --piece-overlap N          neighboring piece overlap [128]\n"
              << "  --max-distance N           closest-pair threshold [128]\n"
              << "  --tangent-window N         centered tangent secant length [32]\n"
              << "  --winding-step N           Lasagna connector integration step [8]\n"
              << "  --winding-cutoff N         exclusive finite winding cutoff [1.5]\n"
              << "  --no-winding-cutoff        retain every finite winding measurement\n"
              << "  --constraints-per-fiber N mutual strongest-link cap per source fiber\n"
              << "  --lp-relaxation            solve continuous [0,1] label relaxation\n"
              << "  --lp-parallel              request HiGHS parallel LP execution\n"
              << "  --lp-solver NAME           choose, simplex, hipo, or ipm [choose]\n"
              << "  --hv-only                  solve active/broken and H/V only\n"
              << "  --exact-perpendicular-milp exact continuous H/V loss with binary activity\n"
              << "  --perpendicular-only      label from perpendicular measured links only\n"
              << "  --exclude-parallel-separate-winding\n"
              << "                              omit that measured class from labeling only\n"
              << "  --threads N                scoring workers [host CPUs]\n"
              << "  --cache-gib N              decoded normal cache [8]\n";
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
    if (argc < 3)
        fail("a mode and input dataset are required");
    Options options;
    const std::string mode = argv[1];
    if (mode == "trace")
        options.mode = Mode::Trace;
    else if (mode == "visualize")
        options.mode = Mode::Visualize;
    else if (mode == "constraints")
        options.mode = Mode::Constraints;
    else if (mode == "consensus")
        options.mode = Mode::Consensus;
    else if (mode == "direction-diagnostic")
        options.mode = Mode::DirectionDiagnostic;
    else if (mode == "direction-ablation")
        options.mode = Mode::DirectionAblation;
    else
        fail("mode must be 'trace', 'visualize', 'constraints', 'consensus', 'direction-diagnostic', or 'direction-ablation'");
    options.input = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output") {
            options.output = value(index, argc, argv, "--output");
        } else if (argument == "--obj") {
            options.obj = value(index, argc, argv, "--obj");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--normal-manifest") {
            options.normalManifest = value(index, argc, argv, "--normal-manifest");
        } else if (argument == "--remote-cache-dir") {
            options.remoteCacheDirectory = value(index, argc, argv, "--remote-cache-dir");
        } else if (argument == "--volume") {
            options.volume = value(index, argc, argv, "--volume");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--bbox") {
            options.trace.minimumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.trace.maximumBaseXYZ =
                {number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox"), number(index, argc, argv, "--bbox")};
            options.hasBounds = true;
            options.hasTraceOnlyOption = true;
        } else if (argument == "--threads") {
            options.threads = static_cast<int>(count(index, argc, argv, "--threads"));
            options.hasSharedRuntimeOption = true;
        } else if (argument == "--cache-gib") {
            const double gib = number(index, argc, argv, "--cache-gib");
            if (!(gib > 0.0))
                fail("--cache-gib must be positive");
            options.cacheBytes = static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0);
            options.hasSharedRuntimeOption = true;
        } else if (argument == "--beam") {
            options.trace.beamWidth = count(index, argc, argv, "--beam");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--lookahead") {
            options.trace.lookaheadDistanceBaseVoxels = number(index, argc, argv, "--lookahead");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--coverage") {
            options.trace.coverageNormalRadiusBaseVoxels = number(index, argc, argv, "--coverage");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--coverage-angle") {
            options.trace.coverageDirectionDegrees = number(index, argc, argv, "--coverage-angle");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--max-attempts") {
            options.trace.maximumAttempts = count(index, argc, argv, "--max-attempts");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--max-fibers") {
            options.trace.maximumFibers = count(index, argc, argv, "--max-fibers");
            options.hasTraceOnlyOption = true;
        } else if (argument == "--texture-max") {
            options.maximumTextureDimension = static_cast<int>(count(index, argc, argv, "--texture-max"));
            options.hasTraceOnlyOption = true;
        } else if (argument == "--direction-dominance") {
            options.directionDominance =
                number(index, argc, argv, "--direction-dominance");
            if (!(options.directionDominance > 0.5 &&
                  options.directionDominance <= 1.0)) {
                fail("--direction-dominance must be in (0.5, 1]");
            }
            options.hasDirectionVisualizationOption = true;
        } else if (argument == "--ablation-step") {
            options.ablationStep = count(
                index, argc, argv, "--ablation-step");
            if (options.ablationStep == 0)
                fail("--ablation-step must be positive");
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--ablation-limit") {
            options.ablationLimit = count(
                index, argc, argv, "--ablation-limit");
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--post-iterations") {
            options.postIterations = count(
                index, argc, argv, "--post-iterations");
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--post-influence") {
            options.postInfluence = number(
                index, argc, argv, "--post-influence");
            if (!(options.postInfluence > 0.0 &&
                  options.postInfluence <= 1.0)) {
                fail("--post-influence must be in (0, 1]");
            }
            options.hasPostInfluenceOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-only") {
            options.bpOnly = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-inference") {
            const std::string inference = value(
                index, argc, argv, "--bp-inference");
            if (inference == "min-sum") {
                options.bpInference = vc::fiber_tracer::
                    FiberTraceBeliefInference::MinSum;
            } else if (inference == "sum-product") {
                options.bpInference = vc::fiber_tracer::
                    FiberTraceBeliefInference::SumProduct;
            } else if (inference == "sum-product-mixed") {
                options.bpInference = vc::fiber_tracer::
                    FiberTraceBeliefInference::SumProductMixed;
            } else {
                fail("--bp-inference must be min-sum, sum-product, or sum-product-mixed");
            }
            options.hasBpInferenceOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-mixed-cost") {
            options.bp.mixedUnaryCost =
                number(index, argc, argv, "--bp-mixed-cost");
            if (!std::isfinite(options.bp.mixedUnaryCost) ||
                options.bp.mixedUnaryCost < 0.0) {
                fail("--bp-mixed-cost must be finite and nonnegative");
            }
            options.hasBpMixedCostOption = true;
            options.hasBpTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-balance") {
            const std::string mode = value(index, argc, argv, "--bp-balance");
            if (mode == "soft")
                options.bpBalance = BpBalanceSelection::Soft;
            else if (mode == "tight")
                options.bpBalance = BpBalanceSelection::Tight;
            else if (mode == "both")
                options.bpBalance = BpBalanceSelection::Both;
            else
                fail("--bp-balance must be soft, tight, or both");
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-target") {
            options.bp.targetHorizontalFraction =
                number(index, argc, argv, "--bp-target");
            if (options.bp.targetHorizontalFraction < 0.0 ||
                options.bp.targetHorizontalFraction > 1.0) {
                fail("--bp-target must be in [0, 1]");
            }
            options.hasBpTuningOption = true;
            options.hasBpBalanceTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-soft-strength") {
            options.bp.softBalanceStrength =
                number(index, argc, argv, "--bp-soft-strength");
            if (options.bp.softBalanceStrength < 0.0)
                fail("--bp-soft-strength must be nonnegative");
            options.hasBpTuningOption = true;
            options.hasBpBalanceTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-temperature") {
            options.bp.horizontalnessTemperature =
                number(index, argc, argv, "--bp-temperature");
            if (!(options.bp.horizontalnessTemperature > 0.0))
                fail("--bp-temperature must be positive");
            options.hasBpTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-message-iterations") {
            options.bp.maximumMessageIterations =
                count(index, argc, argv, "--bp-message-iterations");
            if (options.bp.maximumMessageIterations == 0)
                fail("--bp-message-iterations must be positive");
            options.hasBpTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-balance-iterations") {
            options.bp.maximumBalanceIterations =
                count(index, argc, argv, "--bp-balance-iterations");
            if (options.bp.maximumBalanceIterations == 0)
                fail("--bp-balance-iterations must be positive");
            options.hasBpTuningOption = true;
            options.hasBpBalanceTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-damping") {
            options.bp.messageDamping = number(index, argc, argv, "--bp-damping");
            if (!(options.bp.messageDamping > 0.0) ||
                options.bp.messageDamping > 1.0) {
                fail("--bp-damping must be in (0, 1]");
            }
            options.hasBpTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-residual") {
            options.bp.messageResidualTolerance =
                number(index, argc, argv, "--bp-residual");
            if (options.bp.messageResidualTolerance < 0.0)
                fail("--bp-residual must be nonnegative");
            options.hasBpTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--bp-balance-tolerance") {
            options.bp.balanceTolerance =
                number(index, argc, argv, "--bp-balance-tolerance");
            if (options.bp.balanceTolerance < 0.0)
                fail("--bp-balance-tolerance must be nonnegative");
            options.hasBpTuningOption = true;
            options.hasBpBalanceTuningOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--sample-step") {
            options.constraints.resampleSpacingBaseVoxels = number(index, argc, argv, "--sample-step");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--piece-length") {
            options.constraints.targetPieceLengthBaseVoxels = number(index, argc, argv, "--piece-length");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--piece-overlap") {
            options.constraints.pieceOverlapBaseVoxels = number(index, argc, argv, "--piece-overlap");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--max-distance") {
            options.constraints.maximumDistanceBaseVoxels = number(index, argc, argv, "--max-distance");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--tangent-window") {
            options.constraints.tangentWindowBaseVoxels = number(index, argc, argv, "--tangent-window");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-step") {
            options.constraints.windingIntegrationStepBaseVoxels = number(index, argc, argv, "--winding-step");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-cutoff") {
            options.constraints.maximumWindingDistance =
                number(index, argc, argv, "--winding-cutoff");
            if (!(options.constraints.maximumWindingDistance > 0.0))
                fail("--winding-cutoff must be positive");
            options.constraints.enforceMaximumWindingDistance = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--no-winding-cutoff") {
            options.constraints.enforceMaximumWindingDistance = false;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--constraints-per-fiber") {
            const std::size_t limit =
                count(index, argc, argv, "--constraints-per-fiber");
            if (limit == 0)
                fail("--constraints-per-fiber must be positive");
            options.maximumConstraintsPerFiber = limit;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--broken-cost-per-link") {
            options.labeling.brokenCostPerConstraint =
                number(index, argc, argv, "--broken-cost-per-link");
            if (options.labeling.brokenCostPerConstraint < 0.0)
                fail("--broken-cost-per-link must be nonnegative");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--mip-gap") {
            options.labeling.relativeMipGap = number(index, argc, argv, "--mip-gap");
            if (options.labeling.relativeMipGap < 0.0)
                fail("--mip-gap must be nonnegative");
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--lp-relaxation") {
            options.labeling.relaxIntegrality = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--lp-parallel") {
            options.labeling.lpParallel = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--lp-solver") {
            options.labeling.lpSolver = value(index, argc, argv, "--lp-solver");
            if (options.labeling.lpSolver != "choose" &&
                options.labeling.lpSolver != "simplex" &&
                options.labeling.lpSolver != "hipo" &&
                options.labeling.lpSolver != "ipm") {
                fail("--lp-solver must be choose, simplex, hipo, or ipm");
            }
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--exclude-parallel-separate-winding") {
            options.labeling.excludeParallelSeparateWinding = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--perpendicular-only") {
            options.labeling.perpendicularOnly = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--hv-only") {
            options.labeling.hvOnly = true;
            options.hasHvOnlyOption = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--exact-perpendicular-milp") {
            options.labeling.exactPerpendicularMilp = true;
            options.hasConstraintOnlyOption = true;
            options.hasSolverOnlyOption = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown option: " + argument);
        }
    }
    if (options.mode == Mode::Visualize) {
        if (options.output.empty())
            fail("--output is required");
        if (!options.obj.empty() || !options.normalManifest.empty() || !options.remoteCacheDirectory.empty() || !options.volume.empty() ||
            options.hasBounds || options.hasConstraintOnlyOption || options.hasSharedRuntimeOption) {
            fail("visualize accepts only a trace dataset and --output OBJ");
        }
        return options;
    }
    if (options.normalManifest.empty())
        fail("--normal-manifest is required");
    if (options.threads < 1)
        fail("--threads must be positive");
    if (vc::lasagna::isRemoteLasagnaLocation(options.normalManifest) && options.remoteCacheDirectory.empty()) {
        fail("a remote normal manifest requires --remote-cache-dir");
    }
    if (options.mode == Mode::Constraints ||
        options.mode == Mode::Consensus ||
        options.mode == Mode::DirectionDiagnostic ||
        options.mode == Mode::DirectionAblation) {
        if (options.hasTraceOnlyOption)
            fail("constraint processing does not accept trace-only options");
        if (options.hasDirectionVisualizationOption &&
            options.mode != Mode::DirectionDiagnostic &&
            options.mode != Mode::DirectionAblation) {
            fail("constraint processing does not accept visualization options");
        }
        if (options.hasAblationOnlyOption &&
            options.mode != Mode::DirectionAblation) {
            fail("ablation controls require direction-ablation");
        }
        if (options.mode == Mode::Consensus && options.hasSolverOnlyOption)
            fail("consensus does not accept HiGHS labeling options");
        if (options.mode == Mode::DirectionDiagnostic ||
            options.mode == Mode::DirectionAblation) {
            if (options.hasHvOnlyOption)
                fail("direction diagnostics imply H/V-only labeling; omit --hv-only");
            if (options.labeling.relaxIntegrality ||
                options.labeling.lpParallel ||
                options.labeling.lpSolver != "choose" ||
                options.labeling.exactPerpendicularMilp ||
                options.labeling.excludeParallelSeparateWinding) {
                fail("direction diagnostics require the ordinary discrete H/V-only MILP");
            }
            options.labeling.hvOnly = true;
        }
        if (!options.constraints.enforceMaximumWindingDistance &&
            options.mode == Mode::Constraints && !options.labeling.hvOnly) {
            fail("--no-winding-cutoff currently requires --hv-only");
        }
        if (!options.labeling.relaxIntegrality &&
            (options.labeling.lpParallel || options.labeling.lpSolver != "choose")) {
            fail("--lp-parallel and --lp-solver require --lp-relaxation");
        }
        if (options.labeling.exactPerpendicularMilp &&
            !options.labeling.hvOnly) {
            fail("--exact-perpendicular-milp requires --hv-only");
        }
        if (options.labeling.exactPerpendicularMilp &&
            options.labeling.relaxIntegrality) {
            fail("--exact-perpendicular-milp conflicts with --lp-relaxation");
        }
        if (options.labeling.perpendicularOnly &&
            options.labeling.excludeParallelSeparateWinding) {
            fail("--perpendicular-only conflicts with --exclude-parallel-separate-winding");
        }
        if (options.postIterations > 0 &&
            !options.labeling.perpendicularOnly) {
            fail("--post-iterations requires --perpendicular-only");
        }
        if (options.hasPostInfluenceOption && options.postIterations == 0) {
            fail("--post-influence requires positive --post-iterations");
        }
        if (options.hasBpTuningOption &&
            options.bpBalance == BpBalanceSelection::None &&
            !options.bpOnly) {
            fail("BP tuning controls require --bp-balance");
        }
        if (options.hasBpInferenceOption && !options.bpOnly)
            fail("--bp-inference requires --bp-only");
        if (options.bpInference !=
                vc::fiber_tracer::FiberTraceBeliefInference::MinSum &&
            options.bpBalance != BpBalanceSelection::None) {
            fail("sum-product BP does not support --bp-balance");
        }
        if (options.bpInference !=
                vc::fiber_tracer::FiberTraceBeliefInference::MinSum &&
            options.hasBpBalanceTuningOption) {
            fail("sum-product BP does not accept balance tuning controls");
        }
        if (options.hasBpMixedCostOption &&
            options.bpInference != vc::fiber_tracer::
                FiberTraceBeliefInference::SumProductMixed) {
            fail("--bp-mixed-cost requires --bp-inference sum-product-mixed");
        }
        if ((options.bpBalance == BpBalanceSelection::Tight ||
             options.bpBalance == BpBalanceSelection::Both) &&
            options.bp.maximumBalanceIterations < 2) {
            fail("tight BP requires at least two balance iterations");
        }
        if (options.output.empty()) {
            const std::string stem = options.input.has_extension()
                ? options.input.stem().string()
                : options.input.filename().string();
            options.output = options.input.parent_path() /
                (stem + (options.mode == Mode::Consensus
                             ? "_consensus"
                             : options.mode == Mode::DirectionDiagnostic
                                 ? "_direction_diagnostic"
                                 : options.mode == Mode::DirectionAblation
                                     ? "_direction_ablation"
                                 : "_constraints"));
        }
        options.constraints.parallelThreads = static_cast<std::size_t>(options.threads);
        options.labeling.parallelThreads = static_cast<std::size_t>(options.threads);
        return options;
    }
    if (options.output.empty())
        fail("--output is required");
    if (options.hasConstraintOnlyOption)
        fail("trace does not accept constraint extraction options");
    if (!options.hasBounds)
        fail("--bbox is required");
    if (options.obj.empty()) {
        options.obj = options.output;
        options.obj.replace_extension(".obj");
    }
    if (options.maximumTextureDimension < 2)
        fail("--texture-max must be at least two");
    options.trace.parallelThreads = static_cast<std::size_t>(options.threads);
    return options;
}

double quantile(std::vector<double> values, double fraction)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    const double blend = index - static_cast<double>(lower);
    return values[lower] * (1.0 - blend) + values[upper] * blend;
}

void printConstraintReport(
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const std::vector<vc::fiber_tracer::FiberTraceConstraint>& extractedConstraints,
    const vc::fiber_tracer::FiberTraceConstraintConfig& config,
    bool manifestMatches,
    double wallSeconds,
    double cpuSeconds)
{
    std::vector<double> distances;
    std::vector<double> parallel;
    std::vector<double> perpendicular;
    std::vector<double> winding;
    for (const auto& constraint : extractedConstraints) {
        if (constraint.hardContinuity)
            continue;
        distances.push_back(constraint.closestDistanceBaseVoxels);
        parallel.push_back(constraint.parallelScore);
        perpendicular.push_back(constraint.perpendicularScore);
        winding.push_back(constraint.windingDistance);
    }
    std::cout << std::setprecision(8)
              << "fiber trace constraint config\n"
              << "sample_step  piece_length  piece_overlap  max_distance  tangent_window  winding_step  winding_cutoff  threads\n"
              << config.resampleSpacingBaseVoxels << "  "
              << config.targetPieceLengthBaseVoxels << "  "
              << config.pieceOverlapBaseVoxels << "  "
              << config.maximumDistanceBaseVoxels << "  "
              << config.tangentWindowBaseVoxels << "  "
              << config.windingIntegrationStepBaseVoxels << "  "
              << (config.enforceMaximumWindingDistance
                      ? std::to_string(config.maximumWindingDistance)
                      : "off") << "  "
              << config.parallelThreads << '\n'
              << "fiber trace constraint counts\n"
              << "traces  degenerate  pieces  samples  spatial_hits  candidates  measured  hard  tangent_rejected  winding_invalid  winding_cutoff  manifest_matches_trace\n"
              << report.inputTraces << "  " << report.skippedDegenerateTraces << "  "
              << report.pieces.size() << "  " << report.resampledPoints << "  "
              << report.spatialHits << "  " << report.measuredCandidates << "  "
              << distances.size() << "  " << report.hardConstraints << "  "
              << report.rejectedTangents << "  " << report.rejectedWinding << "  "
              << report.rejectedWindingCutoff << "  "
              << std::boolalpha << manifestMatches << std::noboolalpha << '\n'
              << "fiber trace constraint quantiles (measured links only)\n"
              << "quantile  closest_distance  parallel_score  perpendicular_score  aligned_winding\n";
    for (int percentile = 0; percentile <= 100; percentile += 10) {
        const double fraction = static_cast<double>(percentile) / 100.0;
        std::cout << percentile << "  " << quantile(distances, fraction) << "  "
                  << quantile(parallel, fraction) << "  "
                  << quantile(perpendicular, fraction) << "  "
                  << quantile(winding, fraction) << '\n';
    }
    std::cout << "fiber trace constraint timing\n"
              << "prepare_seconds  search_seconds  orientation_seconds  winding_seconds  score_seconds  total_wall_seconds  total_cpu_seconds\n"
              << report.prepareSeconds << "  " << report.searchSeconds << "  "
              << report.orientationScoreSeconds << "  "
              << report.windingScoreSeconds << "  "
              << report.scoreSeconds << "  " << wallSeconds << "  " << cpuSeconds << '\n';
}

void printConstraintPruningReport(
    const vc::fiber_tracer::FiberTraceConstraintPruningReport& report)
{
    const auto row = [](const char* scope,
                        const vc::fiber_tracer::FiberTraceConstraintGraphStats& stats) {
        std::cout << std::left << std::setw(9) << scope << std::right
                  << std::setw(8) << stats.traces
                  << std::setw(9) << stats.crossTraceConstraints
                  << std::setw(8) << stats.minimumDegree
                  << std::setw(10) << std::fixed << std::setprecision(3)
                  << stats.meanDegree
                  << std::setw(9) << stats.medianDegree
                  << std::setw(8) << stats.maximumDegree
                  << std::setw(10) << stats.isolatedTraces
                  << std::setw(12) << stats.connectedComponents << '\n';
    };
    std::cout << "fiber trace constraint strength pruning\n"
              << "limit  input_total  retained_total  hard  zero_rejected  discarded  recovery_candidates  expected_bridges  recovery_bridges  cap_bridges  overflow_bridges  fibers_over_limit\n"
              << report.maximumConstraintsPerTrace << "  "
              << report.inputTotalConstraints << "  "
              << report.retainedTotalConstraints << "  "
              << report.hardConstraints << "  "
              << report.rejectedZeroStrength << "  "
              << report.rejectedNotMutual << "  "
              << report.recoveryCandidates << "  "
              << report.expectedRecoveryBridges << "  "
              << report.recoveryBridges << "  "
              << report.capRespectingRecoveryBridges << "  "
              << report.fallbackOverflowBridges << "  "
              << report.tracesAboveTargetDegree << '\n'
              << "scope      fibers    links  min_deg  mean_deg  median  max_deg  isolated  components\n";
    row("before", report.before);
    row("mutual", report.mutual);
    row("after", report.after);
    std::cout << std::defaultfloat;
}

void printLabelingReport(
    const vc::fiber_tracer::FiberTraceLabelingReport& report,
    const vc::fiber_tracer::FiberTraceLabelingConfig& config,
    const vc::fiber_tracer::FiberTraceLabelObjReport& objects)
{
    const std::array<const char*, 5> names{
        "h_even", "h_odd", "v_even", "v_odd", "broken"};
    const std::array<std::filesystem::path, 5> paths{
        objects.paths.hEven,
        objects.paths.hOdd,
        objects.paths.vEven,
        objects.paths.vOdd,
        objects.paths.broken,
    };
    std::cout << std::setprecision(8)
              << "fiber trace labeling optimization\n"
              << "status  hv_only  perpendicular_only  objective  orientation_cost  winding_cost  broken_cost  broken_cost_per_link  retained_links  excluded_non_perpendicular  excluded_parallel_separate_winding  requested_mip_gap  variables  integer_variables  rows  mip_nodes  mip_gap  solve_seconds\n"
              << report.modelStatus << "  " << (report.hvOnly ? "true" : "false")
              << "  " << (config.perpendicularOnly ? "true" : "false")
              << "  " << report.objective << "  "
              << report.orientationCost << "  " << report.windingCost << "  "
              << report.brokenCost << "  " << config.brokenCostPerConstraint
              << "  " << report.retainedConstraints << "  "
              << report.excludedNonPerpendicular << "  "
              << report.excludedParallelSeparateWinding << "  "
              << config.relativeMipGap << "  " << report.variables
              << "  " << report.integerVariables
              << "  " << report.rows << "  "
              << report.mipNodes << "  " << report.mipGap << "  "
              << report.solveSeconds << '\n'
              << "fiber trace label OBJ outputs\n"
              << "label  pieces  path\n";
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::cout << names[index] << "  " << objects.pieceCounts[index]
                  << "  " << paths[index] << '\n';
    }
}

const char* pieceLabelName(vc::fiber_tracer::FiberTracePieceLabel label)
{
    using Label = vc::fiber_tracer::FiberTracePieceLabel;
    switch (label) {
    case Label::HEven:
        return "h_even";
    case Label::HOdd:
        return "h_odd";
    case Label::VEven:
        return "v_even";
    case Label::VOdd:
        return "v_odd";
    case Label::Broken:
        return "broken";
    }
    return "invalid";
}

const char* directionGroupName(vc::fiber_tracer::FiberDirectionGroup group)
{
    using Group = vc::fiber_tracer::FiberDirectionGroup;
    switch (group) {
    case Group::Direction1:
        return "dir1";
    case Group::Direction2:
        return "dir2";
    case Group::Mixed:
        return "mixed";
    }
    return "invalid";
}

void printDirectionDiagnosticReport(
    const vc::fiber_tracer::FiberDirectionClassification& directions,
    std::span<const std::size_t> originalTraceIndices,
    const vc::fiber_tracer::FiberDirectionLabelComparisonReport& comparison)
{
    const std::size_t pieces = comparison.rawH + comparison.rawV +
        comparison.rawBroken;
    const std::size_t errors = comparison.orientationErrors +
        comparison.brokenErrors + comparison.defectActiveErrors;
    const double pieceErrorRate = pieces == 0
        ? 0.0
        : static_cast<double>(errors) / static_cast<double>(pieces);
    const double traceErrorRate = comparison.representedTraces == 0
        ? 0.0
        : static_cast<double>(comparison.errorTraces) /
            static_cast<double>(comparison.representedTraces);

    std::cout << "fiber direction MILP diagnostic population\n"
              << "input_fibers  dir1_retained  dir2_retained  mixed_removed  retained_fibers  represented_fibers  pieces\n"
              << directions.lines.size() << "  "
              << directions.groupCounts[0] << "  "
              << directions.groupCounts[1] << "  "
              << directions.groupCounts[2] << "  "
              << originalTraceIndices.size() << "  "
              << comparison.representedTraces << "  " << pieces << '\n'
              << "fiber direction MILP raw labels\n"
              << "h  v  broken  active_components  flipped_components\n"
              << comparison.rawH << "  " << comparison.rawV << "  "
              << comparison.rawBroken << "  "
              << comparison.activeComponents << "  "
              << comparison.flippedComponents << '\n'
              << "fiber direction MILP gauge-aligned confusion\n"
              << "initial  pieces  aligned_dir1  aligned_dir2  broken  errors  error_rate\n";
    for (std::size_t rowIndex = 0; rowIndex < comparison.confusion.size(); ++rowIndex) {
        const auto& row = comparison.confusion[rowIndex];
        const double rate = row.pieces == 0
            ? 0.0
            : static_cast<double>(row.errors) /
                static_cast<double>(row.pieces);
        std::cout << (rowIndex == 0 ? "dir1" : "dir2") << "  "
                  << row.pieces << "  " << row.alignedDirection1 << "  "
                  << row.alignedDirection2 << "  " << row.broken << "  "
                  << row.errors << "  " << std::fixed << std::setprecision(4)
                  << rate << '\n';
    }
    std::cout << "fiber direction MILP errors\n"
              << "orientation_errors  broken_errors  defect_active_errors  total_errors  piece_error_rate  error_fibers  represented_fibers  fiber_error_rate\n"
              << comparison.orientationErrors << "  "
              << comparison.brokenErrors << "  "
              << comparison.defectActiveErrors << "  " << errors << "  "
              << std::fixed << std::setprecision(4) << pieceErrorRate << "  "
              << comparison.errorTraces << "  "
              << comparison.representedTraces << "  " << traceErrorRate << '\n'
              << "fiber direction MILP error details\n"
              << "piece  filtered_trace  original_trace  trace_piece  begin_arc  end_arc  initial  raw_label  component  flipped  aligned  kind\n";
    for (const auto& error : comparison.errors) {
        if (error.filteredTraceIndex >= originalTraceIndices.size()) {
            throw std::logic_error(
                "direction diagnostic error has invalid filtered trace index");
        }
        std::cout << error.pieceIndex << "  " << error.filteredTraceIndex
                  << "  " << originalTraceIndices[error.filteredTraceIndex]
                  << "  " << error.tracePieceIndex << "  "
                  << std::fixed << std::setprecision(3)
                  << error.beginArcBaseVoxels << "  "
                  << error.endArcBaseVoxels << "  "
                  << directionGroupName(error.initialDirection) << "  "
                  << pieceLabelName(error.rawLabel) << "  ";
        if (error.kind ==
            vc::fiber_tracer::FiberDirectionLabelErrorKind::Broken) {
            std::cout << "-  -  -  broken\n";
        } else if (error.kind ==
                   vc::fiber_tracer::FiberDirectionLabelErrorKind::DefectActive) {
            std::cout << error.componentIndex << "  "
                      << (error.componentFlipped ? "yes" : "no")
                      << "  active  defect_active\n";
        } else {
            std::cout << error.componentIndex << "  "
                      << (error.componentFlipped ? "yes" : "no") << "  "
                      << directionGroupName(error.alignedDirection)
                      << "  orientation\n";
        }
    }
    std::cout << std::defaultfloat;
}

void printRelaxedLabelingReport(
    const vc::fiber_tracer::FiberTraceLabelingReport& report,
    const vc::fiber_tracer::FiberTraceLabelingConfig& config,
    const std::filesystem::path& csv,
    const vc::fiber_tracer::FiberTraceRelaxationObjReport& visualization)
{
    std::cout << std::setprecision(8)
              << "fiber trace labeling continuous values\n"
              << "status  mode  hv_only  perpendicular_only  requested_solver  requested_parallel  threads  objective  orientation_cost  winding_cost  broken_cost  broken_cost_per_link  retained_links  excluded_non_perpendicular  excluded_parallel_separate_winding  variables  integer_variables  perpendicular_branch_variables  rows  gauge_roots  triangles  triangle_rows  mip_nodes  mip_gap  solve_seconds  csv\n"
              << report.modelStatus << "  "
              << (report.exactPerpendicularMilp
                      ? "exact_perpendicular_milp"
                      : "lp_relaxation")
              << "  " << (report.hvOnly ? "true" : "false")
              << "  " << (config.perpendicularOnly ? "true" : "false")
              << "  " << config.lpSolver << "  "
              << (config.lpParallel ? "on" : "choose") << "  "
              << config.parallelThreads << "  " << report.objective << "  "
              << report.orientationCost << "  " << report.windingCost << "  "
              << report.brokenCost << "  " << config.brokenCostPerConstraint
              << "  " << report.retainedConstraints << "  "
              << report.excludedNonPerpendicular << "  "
              << report.excludedParallelSeparateWinding << "  "
              << report.variables << "  " << report.integerVariables << "  "
              << report.perpendicularBranchVariables << "  "
              << report.rows << "  "
              << report.gaugeRoots << "  " << report.triangles << "  "
              << report.triangleRows << "  "
              << report.mipNodes << "  " << report.mipGap << "  "
              << report.solveSeconds << "  " << csv << '\n'
              << "fiber trace labeling continuous variable quantiles\n"
              << "quantile  active  vertical  odd\n";
    for (int percentile = 0; percentile <= 100; percentile += 10) {
        const double fraction = static_cast<double>(percentile) / 100.0;
        std::cout << percentile << "  "
                  << quantile(report.activeValues, fraction) << "  "
                  << quantile(report.verticalValues, fraction) << "  "
                  << quantile(report.oddValues, fraction) << '\n';
    }
    const std::array<const char*, 5> names{
        "h_even", "h_odd", "v_even", "v_odd", "broken"};
    const std::array<std::filesystem::path, 5> paths{
        visualization.objects.paths.hEven,
        visualization.objects.paths.hOdd,
        visualization.objects.paths.vEven,
        visualization.objects.paths.vOdd,
        visualization.objects.paths.broken,
    };
    std::cout << "fiber trace labeling continuous threshold visualization\n"
              << "active_threshold  vertical_threshold  odd_threshold\n"
              << visualization.activeThreshold << "  0.5  0.5\n"
              << "label  pieces  path\n";
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::cout << names[index] << "  "
                  << visualization.objects.pieceCounts[index] << "  "
                  << paths[index] << '\n';
    }
}

void printConstraintObjReport(
    const vc::fiber_tracer::FiberTraceConstraintObjReport& report)
{
    std::cout << "fiber trace constraint OBJ outputs\n"
              << "class  lines  path\n"
              << "perpendicular_same_winding  "
              << report.perpendicularSameWinding << "  "
              << report.paths.perpendicularSameWinding << '\n'
              << "perpendicular_separate_winding  "
              << report.perpendicularSeparateWinding << "  "
              << report.paths.perpendicularSeparateWinding << '\n'
              << "parallel_same_winding  " << report.parallelSameWinding << "  "
              << report.paths.parallelSameWinding << '\n'
              << "parallel_separate_winding  "
              << report.parallelSeparateWinding << "  "
              << report.paths.parallelSeparateWinding << '\n';
}

const char* consensusLabelName(
    vc::fiber_tracer::FiberTraceConsensusLabel label)
{
    using Label = vc::fiber_tracer::FiberTraceConsensusLabel;
    switch (label) {
    case Label::H:
        return "h";
    case Label::V:
        return "v";
    case Label::Broken:
        return "broken";
    case Label::Unassigned:
        return "unassigned";
    }
    return "invalid";
}

void printConsensusReport(
    const vc::fiber_tracer::FiberTraceConsensusReport& report,
    const vc::fiber_tracer::FiberTraceConsensusConfig& config,
    const vc::fiber_tracer::FiberTraceConsensusObjReport& objects)
{
    using Label = vc::fiber_tracer::FiberTraceConsensusLabel;
    std::cout << std::setprecision(8)
              << "fiber trace iterative consensus final OBJ outputs\n"
              << "label  fibers  path\n"
              << "h  " << objects.hCount << "  " << objects.finalPaths.h << '\n'
              << "v  " << objects.vCount << "  " << objects.finalPaths.v << '\n'
              << "broken  " << objects.brokenCount << "  "
              << objects.finalPaths.broken << '\n';
    if (!objects.snapshots.empty()) {
        std::cout << "fiber trace iterative consensus snapshot OBJ outputs\n"
                  << "assignments  h_fibers  v_fibers  broken_fibers  h_path  v_path  broken_path\n";
        for (const auto& snapshot : objects.snapshots) {
            std::cout << snapshot.addedCount << "  " << snapshot.hCount
                      << "  " << snapshot.vCount << "  "
                      << snapshot.brokenCount << "  " << snapshot.paths.h
                      << "  " << snapshot.paths.v << "  "
                      << snapshot.paths.broken << '\n';
        }
    }
    std::cout << "fiber trace iterative consensus component seeds\n"
              << "assignments  component  trace  label  straightness  center_distance_base  arc_length_base\n";
    for (const auto& step : report.steps) {
        if (step.componentSeed) {
            std::cout << step.addedCount << "  " << step.componentIndex
                      << "  " << step.traceIndex << "  "
                      << consensusLabelName(step.label) << "  "
                      << step.seedStraightness << "  "
                      << step.seedCenterDistanceBaseVoxels << "  "
                      << step.seedArcLengthBaseVoxels << '\n';
        }
    }
    std::cout << "fiber trace iterative consensus first choices\n"
              << std::right
              << std::setw(7) << "step"
              << std::setw(7) << "trace"
              << std::setw(7) << "comp"
              << std::setw(7) << "seed"
              << std::setw(9) << "label"
              << std::setw(10) << "evidence"
              << std::setw(12) << "mean_dist"
              << std::setw(12) << "score"
              << std::setw(10) << "h_cost"
              << std::setw(10) << "v_cost"
              << std::setw(10) << "broken"
              << std::setw(10) << "selected" << '\n'
              << std::fixed << std::setprecision(3);
    const std::size_t choiceCount = std::min<std::size_t>(100, report.steps.size());
    for (std::size_t index = 0; index < choiceCount; ++index) {
        const auto& step = report.steps[index];
        std::cout << std::setw(7) << step.addedCount
                  << std::setw(7) << step.traceIndex
                  << std::setw(7) << step.componentIndex
                  << std::setw(7) << (step.componentSeed ? "yes" : "no")
                  << std::setw(9) << consensusLabelName(step.label)
                  << std::setw(10) << step.evidenceCount
                  << std::setw(12) << step.meanDistanceBaseVoxels
                  << std::setw(12) << step.connectivityScore
                  << std::setw(10) << step.hCost
                  << std::setw(10) << step.vCost
                  << std::setw(10) << step.brokenCost
                  << std::setw(10) << step.selectedCost << '\n';
    }
    const auto countStat = [](const char* name, std::size_t value) {
        std::cout << std::left << std::setw(42) << name << std::right
                  << std::setw(14) << value << '\n';
    };
    const auto valueStat = [](const char* name, double value) {
        std::cout << std::left << std::setw(42) << name << std::right
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << value << '\n';
    };
    std::cout << std::defaultfloat
              << "fiber trace iterative consensus final stats\n"
              << std::left << std::setw(42) << "metric" << std::right
              << std::setw(14) << "value" << '\n';
    countStat("assignments", report.steps.size());
    countStat("components", report.components);
    countStat("degenerate", report.degenerateTraces);
    countStat(
        "retained cross-trace constraints",
        report.retainedCrossTraceConstraints);
    countStat("H", report.labelCounts[static_cast<std::size_t>(Label::H)]);
    countStat("V", report.labelCounts[static_cast<std::size_t>(Label::V)]);
    countStat(
        "broken",
        report.labelCounts[static_cast<std::size_t>(Label::Broken)]);
    valueStat("orientation cost", report.orientationCost);
    valueStat("broken cost", report.brokenCost);
    valueStat("objective", report.objective);
    valueStat("broken cost per link", config.brokenCostPerConstraint);
    std::cout << std::defaultfloat;
}

void writeAndPrintBpReport(
    const vc::fiber_tracer::FiberTraceBeliefPropagationReport& report,
    vc::fiber_tracer::FiberTraceBalanceMode mode,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::fiber_tracer::FiberTraceConstraintReport& constraints,
    std::span<const std::size_t> originalTraceIndices,
    std::span<const vc::fiber_tracer::FiberDirectionGroup> directions,
    const std::filesystem::path& output)
{
    const bool sumProduct = report.inference !=
        vc::fiber_tracer::FiberTraceBeliefInference::MinSum;
    const bool mixedState = report.inference ==
        vc::fiber_tracer::FiberTraceBeliefInference::SumProductMixed;
    if (report.horizontalness.size() != lines.size() ||
        constraints.pieces.size() != lines.size() ||
        directions.size() != lines.size() ||
        originalTraceIndices.size() != lines.size() ||
        report.seedPieceIndex >= lines.size() ||
        (mixedState &&
         (report.verticalProbability.size() != lines.size() ||
          report.mixedProbability.size() != lines.size() ||
          report.horizontalProbability.size() != lines.size()))) {
        throw std::logic_error("BP report does not match represented pieces");
    }
    const std::string modeName =
        vc::fiber_tracer::fiberTraceBalanceModeName(mode);
    const std::string inferenceName =
        vc::fiber_tracer::fiberTraceBeliefInferenceName(report.inference);
    const std::string artifactName = sumProduct
        ? inferenceName
        : modeName;
    const auto outputBase = output.parent_path() /
        (output.stem().string() + "_bp_" + artifactName);
    const auto bands =
        vc::fiber_tracer::classifyFiberValues(report.horizontalness);
    const auto paths = vc::fiber_tracer::writeFiberletCropValueBandObjs(
        lines, bands, outputBase);
    std::optional<vc::fiber_tracer::FiberValueBandObjPaths>
        mixedPaths;
    if (mixedState) {
        const auto mixedBands = vc::fiber_tracer::classifyFiberValues(
            report.mixedProbability);
        mixedPaths = vc::fiber_tracer::writeFiberletCropValueBandObjs(
            lines, mixedBands,
            output.parent_path() /
                (output.stem().string() + "_bp_" + artifactName + "_mixed"));
    }
    const auto [minimum, maximum] = std::minmax_element(
        report.horizontalness.begin(), report.horizontalness.end());
    const double mean = std::accumulate(
        report.horizontalness.begin(), report.horizontalness.end(), 0.0) /
        static_cast<double>(report.horizontalness.size());
    const auto directionName = [](vc::fiber_tracer::FiberDirectionGroup group) {
        using Group = vc::fiber_tracer::FiberDirectionGroup;
        switch (group) {
        case Group::Direction1:
            return "dir1";
        case Group::Direction2:
            return "dir2";
        case Group::Mixed:
            return "mixed";
        }
        return "invalid";
    };
    const auto consistency = mixedState
        ? vc::fiber_tracer::analyzeMixedFiberTraceConstraintConsistency(
              constraints,
              report.verticalProbability,
              report.mixedProbability,
              report.horizontalProbability)
        : vc::fiber_tracer::analyzeFiberTraceConstraintConsistency(
              constraints, report.horizontalness);
    const auto csv = output.parent_path() /
        (output.stem().string() + "_bp_" + artifactName +
         "_consistency.csv");
    std::ofstream csvOutput(csv);
    if (!csvOutput)
        throw std::runtime_error("failed to open BP consistency CSV: " + csv.string());
    csvOutput << "piece,original_trace,source_piece,begin_arc_base,end_arc_base,reference,";
    if (sumProduct) {
        csvOutput << "bp_inference,bp_temperature,";
    }
    if (mixedState)
        csvOutput << "bp_mixed_unary_cost,p_v,p_mixed,p_h,";
    csvOutput
        << "bp_status,vertical_threshold,"
           "horizontal_threshold,"
        << (mixedState ? "orientation_projection" : "horizontalness")
        << ",degree,incident_measurements,"
           "total_strength,"
           "resolved_degree,resolved_strength,unresolved_degree,"
           "unresolved_strength,hard_mismatches,hard_mismatch_rate,"
           "weighted_hard_mismatch_rate,soft_mismatch_proxy,"
           "neighbor_support_balance,neighbor_certainty\n"
        << std::setprecision(17);
    const auto csvOptional = [&csvOutput](const std::optional<double>& value) {
        if (value)
            csvOutput << *value;
        else
            csvOutput << "NA";
    };
    for (std::size_t piece = 0; piece < consistency.pieces.size(); ++piece) {
        const auto& current = consistency.pieces[piece];
        const auto& descriptor = constraints.pieces[piece];
        csvOutput << piece << ',' << originalTraceIndices[piece] << ','
                  << descriptor.pieceIndex << ','
                  << descriptor.beginArcBaseVoxels << ','
                  << descriptor.endArcBaseVoxels << ','
                  << directionName(directions[piece]) << ',';
        if (sumProduct) {
            csvOutput << inferenceName << ',' << report.inferenceTemperature
                      << ',';
        }
        if (mixedState) {
            csvOutput << report.mixedUnaryCost << ','
                      << report.verticalProbability[piece] << ','
                      << report.mixedProbability[piece] << ','
                      << report.horizontalProbability[piece] << ',';
        }
        csvOutput << report.status << ',' << consistency.verticalThreshold << ','
                  << consistency.horizontalThreshold << ','
                  << report.horizontalness[piece] << ',' << current.degree
                  << ',' << current.incidentMeasurements << ','
                  << current.totalStrength << ','
                  << current.resolvedDegree << ',' << current.resolvedStrength
                  << ',' << current.unresolvedDegree << ','
                  << current.unresolvedStrength << ','
                  << current.hardMismatches << ',';
        csvOptional(current.hardMismatchRate);
        csvOutput << ',';
        csvOptional(current.weightedHardMismatchRate);
        csvOutput << ',';
        csvOptional(current.softMismatchProxy);
        csvOutput << ',';
        csvOptional(current.neighborSupportBalance);
        csvOutput << ',';
        csvOptional(current.neighborCertainty);
        csvOutput << '\n';
    }
    if (!csvOutput)
        throw std::runtime_error("failed to write BP consistency CSV: " + csv.string());
    std::cout << std::fixed << std::setprecision(6);
    if (mixedState) {
        std::cout
            << "fiber direction Mixed-state sum-product BP\n"
            << "inference  temperature  mixed_unary_cost  status  pieces  factors  measurements"
               "  neutral_factors  neutral_measurements"
               "  components  isolated_pieces  seed_piece  seed_original_trace"
               "  seed_source_piece  seed_ref  message_iterations"
               "  message_residual  achieved_orientation  min_orientation"
               "  mean_orientation  max_orientation  seconds\n"
            << inferenceName << "  " << report.inferenceTemperature << "  "
            << report.mixedUnaryCost << "  "
            << report.status << "  " << lines.size() << "  " << report.factors
            << "  " << report.mergedMeasurements << "  "
            << report.neutralFactors << "  " << report.neutralMeasurements << "  "
            << report.connectedComponents << "  " << report.isolatedPieces
            << "  " << report.seedPieceIndex << "  "
            << originalTraceIndices[report.seedPieceIndex] << "  "
            << constraints.pieces[report.seedPieceIndex].pieceIndex << "  "
            << directionName(directions[report.seedPieceIndex]) << "  "
            << report.messageIterations << "  " << report.messageResidual
            << "  " << report.achievedHorizontalFraction << "  " << *minimum
            << "  " << mean << "  " << *maximum << "  "
            << report.solveSeconds << '\n';
    } else if (sumProduct) {
        std::cout
            << "fiber direction sum-product BP\n"
            << "inference  temperature  status  pieces  factors  measurements"
               "  neutral_factors  neutral_measurements"
               "  components  isolated_pieces  seed_piece  seed_original_trace"
               "  seed_source_piece  seed_ref  message_iterations"
               "  message_residual  achieved_h  min_h  mean_h  max_h  seconds\n"
            << inferenceName << "  " << report.inferenceTemperature << "  "
            << report.status << "  " << lines.size() << "  " << report.factors
            << "  " << report.mergedMeasurements << "  "
            << report.neutralFactors << "  " << report.neutralMeasurements << "  "
            << report.connectedComponents << "  " << report.isolatedPieces
            << "  " << report.seedPieceIndex << "  "
            << originalTraceIndices[report.seedPieceIndex] << "  "
            << constraints.pieces[report.seedPieceIndex].pieceIndex << "  "
            << directionName(directions[report.seedPieceIndex]) << "  "
            << report.messageIterations << "  " << report.messageResidual
            << "  " << report.achievedHorizontalFraction << "  " << *minimum
            << "  " << mean << "  " << *maximum << "  "
            << report.solveSeconds << '\n';
    } else {
        std::cout
            << "fiber direction min-sum BP\n"
            << "mode  status  pieces  factors  measurements  neutral_factors"
               "  neutral_measurements  components"
               "  isolated_pieces  seed_piece  seed_original_trace  seed_source_piece"
               "  seed_ref  message_iterations"
               "  balance_iterations  message_residual  field  target_h"
               "  achieved_h  min_h  mean_h  max_h  seconds\n"
            << modeName << "  " << report.status << "  " << lines.size()
            << "  " << report.factors << "  " << report.mergedMeasurements
            << "  " << report.neutralFactors << "  "
            << report.neutralMeasurements
            << "  " << report.connectedComponents << "  "
            << report.isolatedPieces << "  " << report.seedPieceIndex << "  "
            << originalTraceIndices[report.seedPieceIndex] << "  "
            << constraints.pieces[report.seedPieceIndex].pieceIndex << "  "
            << directionName(directions[report.seedPieceIndex]) << "  "
            << report.messageIterations << "  " << report.balanceIterations
            << "  " << report.messageResidual << "  " << report.balanceField
            << "  " << report.targetHorizontalFraction << "  "
            << report.achievedHorizontalFraction << "  " << *minimum << "  "
            << mean << "  " << *maximum << "  " << report.solveSeconds << '\n';
    }
    std::cout << "band  count  dir1_ref  dir2_ref  mixed_ref  min  mean  max"
                 "  path\n";
    for (std::size_t band = 0; band < bands.bands.size(); ++band) {
        const auto& current = bands.bands[band];
        std::array<std::size_t, 3> references{};
        for (const std::size_t trace : current.lineIndices) {
            ++references[static_cast<std::size_t>(directions[trace])];
        }
        std::cout << 'p' << band << "  " << current.lineIndices.size()
                  << "  " << references[0] << "  " << references[1]
                  << "  " << references[2] << "  " << current.minimumValue
                  << "  " << current.meanValue << "  "
                  << current.maximumValue << "  " << paths.bands[band]
                  << '\n';
    }
    if (mixedState) {
        std::array<std::array<std::size_t, 4>, 3> confusion{};
        std::vector<vc::fiber_tracer::FiberTernaryState> predictions(
            lines.size(), vc::fiber_tracer::FiberTernaryState::Tie);
        std::vector<double> mixedReferences;
        std::vector<double> trustedReferences;
        for (std::size_t trace = 0; trace < lines.size(); ++trace) {
            const std::array probabilities{
                report.verticalProbability[trace],
                report.mixedProbability[trace],
                report.horizontalProbability[trace],
            };
            const double maximum = *std::max_element(
                probabilities.begin(), probabilities.end());
            const std::size_t ties = static_cast<std::size_t>(std::count(
                probabilities.begin(), probabilities.end(), maximum));
            const std::size_t prediction = ties == 1
                ? static_cast<std::size_t>(std::distance(
                      probabilities.begin(),
                      std::find(probabilities.begin(), probabilities.end(), maximum)))
                : 3;
            if (prediction < 3) {
                predictions[trace] = static_cast<
                    vc::fiber_tracer::FiberTernaryState>(prediction);
            }
            ++confusion[static_cast<std::size_t>(directions[trace])][prediction];
            if (directions[trace] ==
                vc::fiber_tracer::FiberDirectionGroup::Mixed) {
                mixedReferences.push_back(report.mixedProbability[trace]);
            } else {
                trustedReferences.push_back(report.mixedProbability[trace]);
            }
        }
        double favorable = 0.0;
        for (const double positive : mixedReferences) {
            for (const double negative : trustedReferences) {
                favorable += positive > negative
                    ? 1.0
                    : positive == negative ? 0.5 : 0.0;
            }
        }
        std::cout << "fiber direction explicit Mixed marginal\n"
                  << "reference  predicted_v  predicted_mixed  predicted_h  tie\n";
        for (std::size_t reference = 0; reference < confusion.size(); ++reference) {
            std::cout << directionName(
                             static_cast<vc::fiber_tracer::FiberDirectionGroup>(reference))
                      << "  " << confusion[reference][0]
                      << "  " << confusion[reference][1]
                      << "  " << confusion[reference][2]
                      << "  " << confusion[reference][3] << '\n';
        }
        std::cout << "metric  direction  mixed  trusted  auroc\n"
                  << "p_mixed  higher  " << mixedReferences.size() << "  "
                  << trustedReferences.size() << "  ";
        if (mixedReferences.empty() || trustedReferences.empty())
            std::cout << "NA\n";
        else
            std::cout << favorable / static_cast<double>(
                mixedReferences.size() * trustedReferences.size()) << '\n';
        std::cout << "fiber direction explicit state marginal summaries\n"
                  << "reference  state  count  min  mean  median  p90  max\n";
        const std::array<std::span<const double>, 3> stateValues{
            report.verticalProbability,
            report.mixedProbability,
            report.horizontalProbability,
        };
        constexpr std::array<const char*, 3> stateNames{"p_v", "p_mixed", "p_h"};
        for (std::size_t reference = 0; reference < 3; ++reference) {
            for (std::size_t state = 0; state < 3; ++state) {
                std::vector<double> values;
                for (std::size_t trace = 0; trace < lines.size(); ++trace) {
                    if (static_cast<std::size_t>(directions[trace]) == reference)
                        values.push_back(stateValues[state][trace]);
                }
                const double valueMean = std::accumulate(
                    values.begin(), values.end(), 0.0) /
                    static_cast<double>(values.size());
                std::cout << directionName(
                                 static_cast<vc::fiber_tracer::FiberDirectionGroup>(reference))
                          << "  " << stateNames[state] << "  " << values.size()
                          << "  " << quantile(values, 0.0)
                          << "  " << valueMean
                          << "  " << quantile(values, 0.5)
                          << "  " << quantile(values, 0.9)
                          << "  " << quantile(values, 1.0) << '\n';
            }
        }
        std::cout << "fiber direction explicit Mixed probability bands base="
                  << mixedPaths->bands.front().parent_path() /
                        (output.stem().string() + "_bp_" + artifactName + "_mixed")
                  << '\n';
        const auto statePaths = vc::fiber_tracer::
            writeFiberletCropTernaryStateObjs(
                lines,
                predictions,
                output.parent_path() / (output.stem().string() + "_bp"));
        std::cout << "fiber direction explicit state OBJ layers\n"
                  << "state  path\n"
                  << "v  " << statePaths.vertical << '\n'
                  << "mixed  " << statePaths.mixed << '\n'
                  << "h  " << statePaths.horizontal << '\n'
                  << "tie  " << statePaths.tie << '\n';
    }
    const auto printConsistencyMetric = [&] (
                                            const char* reference,
                                            const char* metric,
                                            std::vector<double> values) {
        if (values.empty()) {
            std::cout << reference << "  " << metric
                      << "  0  NA  NA  NA  NA  NA\n";
            return;
        }
        const double valueMean = std::accumulate(
            values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());
        std::cout << reference << "  " << metric << "  " << values.size()
                  << "  " << quantile(values, 0.0)
                  << "  " << valueMean
                  << "  " << quantile(values, 0.5)
                  << "  " << quantile(values, 0.9)
                  << "  " << quantile(values, 1.0) << '\n';
    };
    std::cout << "fiber direction BP constraint consistency\n"
              << "reference  metric  valid_pieces  min  mean  median  p90  max\n";
    constexpr std::array<const char*, 9> names{
        "degree",
        "measurements",
        "strength",
        "unresolved_rate",
        "hard_mismatch_rate",
        "weighted_hard_mismatch_rate",
        "soft_mismatch_proxy",
        "neighbor_support_balance",
        "neighbor_certainty",
    };
    const auto metricValue = [](
                                 const vc::fiber_tracer::
                                     FiberTraceConstraintConsistency& current,
                                 std::size_t metric) -> std::optional<double> {
        switch (metric) {
        case 0:
            return static_cast<double>(current.degree);
        case 1:
            return static_cast<double>(current.incidentMeasurements);
        case 2:
            return current.totalStrength;
        case 3:
            if (current.degree == 0)
                return std::nullopt;
            return static_cast<double>(current.unresolvedDegree) /
                static_cast<double>(current.degree);
        case 4:
            return current.hardMismatchRate;
        case 5:
            return current.weightedHardMismatchRate;
        case 6:
            return current.softMismatchProxy;
        case 7:
            return current.neighborSupportBalance;
        case 8:
            return current.neighborCertainty;
        default:
            return std::nullopt;
        }
    };
    for (std::size_t group = 0; group < 3; ++group) {
        std::array<std::vector<double>, names.size()> metrics;
        for (std::size_t trace = 0; trace < consistency.pieces.size(); ++trace) {
            if (static_cast<std::size_t>(directions[trace]) != group)
                continue;
            const auto& current = consistency.pieces[trace];
            for (std::size_t metric = 0; metric < metrics.size(); ++metric) {
                if (const auto value = metricValue(current, metric))
                    metrics[metric].push_back(*value);
            }
        }
        const auto reference = directionName(
            static_cast<vc::fiber_tracer::FiberDirectionGroup>(group));
        for (std::size_t metric = 0; metric < metrics.size(); ++metric)
            printConsistencyMetric(reference, names[metric], std::move(metrics[metric]));
    }
    std::cout << "fiber direction BP Mixed discrimination\n"
              << "metric  direction  mixed  trusted  auroc\n";
    for (std::size_t metric = 0; metric < names.size(); ++metric) {
        std::vector<double> mixed;
        std::vector<double> trusted;
        for (std::size_t trace = 0; trace < consistency.pieces.size(); ++trace) {
            const auto value = metricValue(consistency.pieces[trace], metric);
            if (!value)
                continue;
            if (directions[trace] ==
                vc::fiber_tracer::FiberDirectionGroup::Mixed) {
                mixed.push_back(*value);
            } else {
                trusted.push_back(*value);
            }
        }
        const bool lowerPredictsMixed = metric == 8;
        double favorable = 0.0;
        for (const double positive : mixed) {
            for (const double negative : trusted) {
                const double orientedPositive = lowerPredictsMixed
                    ? -positive
                    : positive;
                const double orientedNegative = lowerPredictsMixed
                    ? -negative
                    : negative;
                favorable += orientedPositive > orientedNegative
                    ? 1.0
                    : orientedPositive == orientedNegative ? 0.5 : 0.0;
            }
        }
        std::cout << names[metric] << "  "
                  << (lowerPredictsMixed ? "lower" : "higher") << "  "
                  << mixed.size() << "  " << trusted.size() << "  ";
        if (mixed.empty() || trusted.empty())
            std::cout << "NA\n";
        else
            std::cout << favorable /
                    static_cast<double>(mixed.size() * trusted.size())
                      << '\n';
    }
    std::cout << "fiber direction BP constraint consistency csv=" << csv << '\n';
    std::cout << std::defaultfloat;
}

struct VisualizationReport {
    vc::fiber_tracer::FiberDirectionClassification directions;
    vc::fiber_tracer::FiberQualityHistogram quality;
};

VisualizationReport visualize(
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const std::filesystem::path& output,
    double directionDominance)
{
    std::filesystem::create_directories(output.parent_path().empty() ? std::filesystem::path{"."} : output.parent_path());
    VisualizationReport report;
    report.directions = vc::fiber_tracer::classifyFiberletCropDirections(
        lines, directionDominance);
    vc::fiber_tracer::writeFiberletCropDirectionObjs(lines, report.directions, output);
    report.quality = vc::fiber_tracer::classifyFiberletCropQuality(lines);
    vc::fiber_tracer::writeFiberletCropQualityArtifacts(lines, report.quality, output);

    std::cout << "fiberlet crop quality histogram\n"
              << "decile  count  total_min  total_mean  total_max"
                 "  density_min  density_mean  density_max\n";
    std::cout << std::setprecision(8);
    for (std::size_t index = 0; index < report.quality.bins.size(); ++index) {
        const auto& bin = report.quality.bins[index];
        std::cout << std::setw(2) << index * 10 << '-' << std::setw(3) << (index + 1) * 10 << "  " << std::setw(5) << bin.lineIndices.size();
        if (bin.lineIndices.empty()) {
            std::cout << "\n";
        } else {
            std::cout << "  " << bin.minimumTotalMetricCost << "  " << bin.meanTotalMetricCost << "  " << bin.maximumTotalMetricCost << "  "
                      << bin.minimumCostDensity << "  " << bin.meanCostDensity << "  " << bin.maximumCostDensity << '\n';
        }
    }
    return report;
}

void printDirectionReport(const vc::fiber_tracer::FiberDirectionClassification& classification, const std::filesystem::path& output)
{
    const auto paths = vc::fiber_tracer::fiberDirectionObjPaths(output);
    std::cout << "fiberlet crop directions"
              << " dir1_xyz=" << classification.direction1BaseXYZ[0] << ',' << classification.direction1BaseXYZ[1] << ','
              << classification.direction1BaseXYZ[2] << " dir2_xyz=" << classification.direction2BaseXYZ[0] << ','
              << classification.direction2BaseXYZ[1] << ',' << classification.direction2BaseXYZ[2] << " analyzed_steps=" << classification.analyzedSteps
              << " analyzed_length_base=" << classification.analyzedLengthBaseVoxels << " dir1_fibers=" << classification.groupCounts[0]
              << " dir2_fibers=" << classification.groupCounts[1] << " mixed_fibers=" << classification.groupCounts[2]
              << " dominance_fraction=" << classification.dominanceFraction
              << " output=" << paths.all << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse(argc, argv);
        if (options.mode == Mode::Constraints ||
            options.mode == Mode::Consensus ||
            options.mode == Mode::DirectionDiagnostic ||
            options.mode == Mode::DirectionAblation) {
            const auto started = std::chrono::steady_clock::now();
            const auto cpuStarted = std::clock();
            const auto artifact =
                vc::fiber_tracer::readFiberletCropTraceArtifact(options.input);
            vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
            normalOptions.workingToBaseScale = 1.0;
            normalOptions.remoteCacheRoot = options.remoteCacheDirectory;
            const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(
                options.normalManifest, normalOptions);
            vc::fiber_tracer::validateFiberletCropTraceNormalDatasetCompatibility(
                artifact, normalDataset);
            const bool manifestMatches =
                artifact.metadata.sources.at("normal_manifest") ==
                normalDataset.manifest().raw;
            const vc::lasagna::LasagnaNormalSampler normals(
                normalDataset,
                vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});

            std::vector<vc::fiber_tracer::FiberletCropTraceLine>
                diagnosticLines;
            std::vector<std::size_t> diagnosticOriginalTraceIndices;
            std::vector<vc::fiber_tracer::FiberDirectionGroup>
                diagnosticDirections;
            std::optional<vc::fiber_tracer::FiberDirectionClassification>
                diagnosticClassification;
            const std::vector<vc::fiber_tracer::FiberletCropTraceLine>*
                constraintLines = &artifact.lines;
            if (options.mode == Mode::DirectionDiagnostic ||
                options.mode == Mode::DirectionAblation) {
                diagnosticClassification =
                    vc::fiber_tracer::classifyFiberletCropDirections(
                        artifact.lines, options.directionDominance);
                const auto outputDirectory = options.output.parent_path().empty()
                    ? std::filesystem::path{"."}
                    : options.output.parent_path();
                std::filesystem::create_directories(outputDirectory);
                const std::filesystem::path initialOutput = outputDirectory /
                    (options.output.stem().string() + "_initial.obj");
                vc::fiber_tracer::writeFiberletCropDirectionObjs(
                    artifact.lines, *diagnosticClassification, initialOutput);
                printDirectionReport(*diagnosticClassification, initialOutput);
            }
            if (options.mode == Mode::DirectionAblation) {
                const auto candidates =
                    vc::fiber_tracer::rankMixedFiberDirections(
                        *diagnosticClassification);
                const std::size_t admittedTarget = std::min(
                    candidates.size(),
                    options.ablationLimit.value_or(candidates.size()));
                std::vector<unsigned char> admitted(
                    artifact.lines.size(), 0);
                std::vector<std::size_t> admittedCounts;
                if (options.bpOnly) {
                    admittedCounts.push_back(admittedTarget);
                } else {
                    admittedCounts.push_back(0);
                    for (std::size_t count = options.ablationStep;
                         count < admittedTarget;) {
                        admittedCounts.push_back(count);
                        if (count > admittedTarget - options.ablationStep)
                            break;
                        count += options.ablationStep;
                    }
                    if (admittedCounts.back() != admittedTarget)
                        admittedCounts.push_back(admittedTarget);
                }
                std::cout
                    << "fiber direction mixed ablation checkpoints="
                    << admittedCounts.size()
                    << " trusted_fibers="
                    << diagnosticClassification->groupCounts[0] +
                        diagnosticClassification->groupCounts[1]
                    << " mixed_fibers=" << candidates.size()
                    << " admitted_target=" << admittedTarget << '\n';
                std::size_t admittedSoFar = 0;
                for (std::size_t checkpoint = 0;
                     checkpoint < admittedCounts.size();
                     ++checkpoint) {
                    const std::size_t admittedCount = admittedCounts[checkpoint];
                    while (admittedSoFar < admittedCount) {
                        admitted[candidates[admittedSoFar].lineIndex] = 1;
                        ++admittedSoFar;
                    }
                    diagnosticLines.clear();
                    diagnosticOriginalTraceIndices.clear();
                    diagnosticDirections.clear();
                    std::vector<std::uint8_t> trustedMask;
                    diagnosticLines.reserve(artifact.lines.size());
                    diagnosticOriginalTraceIndices.reserve(artifact.lines.size());
                    diagnosticDirections.reserve(artifact.lines.size());
                    trustedMask.reserve(artifact.lines.size());
                    for (std::size_t trace = 0;
                         trace < artifact.lines.size();
                         ++trace) {
                        const bool trusted =
                            diagnosticClassification->lines[trace].group !=
                            vc::fiber_tracer::FiberDirectionGroup::Mixed;
                        if (!trusted && admitted[trace] == 0)
                            continue;
                        diagnosticLines.push_back(artifact.lines[trace]);
                        diagnosticOriginalTraceIndices.push_back(trace);
                        diagnosticDirections.push_back(
                            diagnosticClassification->lines[trace].group);
                        trustedMask.push_back(trusted ? 1 : 0);
                    }

                    auto checkpointReport =
                        vc::fiber_tracer::extractFiberTraceConstraints(
                            diagnosticLines,
                            options.constraints,
                            [&normals](const cv::Vec3d& a,
                                       const cv::Vec3d& b,
                                       double step) {
                                return normals.normalAlignedWindingDistance(
                                    a, b, step);
                            },
                            [&normals](
                                const std::vector<std::pair<
                                    cv::Vec3d, cv::Vec3d>>& connectors,
                                double step,
                                int threads) {
                                return normals.normalAlignedWindingDistancesBatch(
                                    connectors, step, threads);
                            });
                    const auto extractedConstraints =
                        checkpointReport.constraints;
                    std::optional<
                        vc::fiber_tracer::FiberTraceConstraintPruningReport>
                        checkpointPruning;
                    if (options.maximumConstraintsPerFiber) {
                        auto pruning =
                            vc::fiber_tracer::pruneFiberTraceConstraintsByStrength(
                                checkpointReport,
                                options.constraints.maximumDistanceBaseVoxels,
                                *options.maximumConstraintsPerFiber);
                        checkpointReport.constraints =
                            std::move(pruning.constraints);
                        checkpointPruning = std::move(pruning.report);
                    }
                    if (options.bpOnly) {
                        auto bpConstraints = checkpointReport;
                        const auto selection = vc::fiber_tracer::
                            selectFiberTraceLabelConstraints(
                                checkpointReport, options.labeling);
                        bpConstraints.constraints.clear();
                        bpConstraints.constraints.reserve(
                            selection.retainedIndices.size());
                        for (const std::size_t index : selection.retainedIndices) {
                            bpConstraints.constraints.push_back(
                                checkpointReport.constraints.at(index));
                        }
                        std::cout
                            << "fiber direction BP-only cohort"
                            << " admitted=" << admittedCount
                            << " fibers=" << diagnosticLines.size()
                            << " pieces=" << bpConstraints.pieces.size()
                            << " selected_constraints="
                            << bpConstraints.constraints.size() << '\n';
                        const auto bpPieceLines = vc::fiber_tracer::
                            makeFiberTraceConstraintPieceLines(
                                diagnosticLines, bpConstraints);
                        std::vector<std::size_t> bpOriginalTraceIndices;
                        std::vector<vc::fiber_tracer::FiberDirectionGroup>
                            bpDirections;
                        bpOriginalTraceIndices.reserve(bpConstraints.pieces.size());
                        bpDirections.reserve(bpConstraints.pieces.size());
                        for (const auto& piece : bpConstraints.pieces) {
                            if (piece.traceIndex >= diagnosticLines.size()) {
                                throw std::logic_error(
                                    "BP piece references an invalid cohort source fiber");
                            }
                            bpOriginalTraceIndices.push_back(
                                diagnosticOriginalTraceIndices[piece.traceIndex]);
                            bpDirections.push_back(
                                diagnosticDirections[piece.traceIndex]);
                        }
                        const auto runBp = [&] (
                                               vc::fiber_tracer::
                                                   FiberTraceBalanceMode mode) {
                            auto config = options.bp;
                            config.balanceMode = mode;
                            config.cropMinimumBaseXYZ = artifact.minimumBaseXYZ;
                            config.cropMaximumBaseXYZ = artifact.maximumBaseXYZ;
                            vc::fiber_tracer::
                                FiberTraceBeliefPropagationReport report;
                            switch (options.bpInference) {
                            case vc::fiber_tracer::
                                FiberTraceBeliefInference::MinSum:
                                report = vc::fiber_tracer::
                                    solveFiberTraceBeliefPropagation(
                                        diagnosticLines,
                                        bpConstraints,
                                        config);
                                break;
                            case vc::fiber_tracer::
                                FiberTraceBeliefInference::SumProduct:
                                report = vc::fiber_tracer::
                                    solveFiberTraceSumProduct(
                                        diagnosticLines, bpConstraints, config);
                                break;
                            case vc::fiber_tracer::
                                FiberTraceBeliefInference::SumProductMixed:
                                report = vc::fiber_tracer::
                                    solveFiberTraceMixedSumProduct(
                                        diagnosticLines, bpConstraints, config);
                                break;
                            }
                            writeAndPrintBpReport(
                                report,
                                mode,
                                bpPieceLines,
                                bpConstraints,
                                bpOriginalTraceIndices,
                                bpDirections,
                                options.output);
                        };
                        if (options.bpBalance == BpBalanceSelection::None) {
                            runBp(vc::fiber_tracer::FiberTraceBalanceMode::None);
                        } else {
                            if (options.bpBalance == BpBalanceSelection::Soft ||
                                options.bpBalance == BpBalanceSelection::Both) {
                                runBp(vc::fiber_tracer::
                                          FiberTraceBalanceMode::Soft);
                            }
                            if (options.bpBalance == BpBalanceSelection::Tight ||
                                options.bpBalance == BpBalanceSelection::Both) {
                                runBp(vc::fiber_tracer::
                                          FiberTraceBalanceMode::Tight);
                            }
                        }
                        if (checkpointPruning)
                            printConstraintPruningReport(*checkpointPruning);
                        continue;
                    }
                    const auto checkpointLabeling =
                        vc::fiber_tracer::solveFiberTraceLabels(
                            checkpointReport, options.labeling);
                    auto lpConfig = options.labeling;
                    lpConfig.relaxIntegrality = true;
                    const auto checkpointLp =
                        vc::fiber_tracer::solveFiberTraceLabels(
                            checkpointReport, lpConfig);
                    const auto checkpointLpThresholded =
                        vc::fiber_tracer::thresholdFiberTraceLabeling(
                            checkpointLp);
                    auto comparisonReport = checkpointReport;
                    comparisonReport.constraints.clear();
                    comparisonReport.constraints.reserve(
                        checkpointLabeling.retainedConstraintIndices.size());
                    for (const std::size_t index :
                         checkpointLabeling.retainedConstraintIndices) {
                        comparisonReport.constraints.push_back(
                            checkpointReport.constraints.at(index));
                    }
                    const auto comparison =
                        vc::fiber_tracer::compareFiberDirectionLabels(
                            comparisonReport,
                            diagnosticDirections,
                            checkpointLabeling,
                            trustedMask);
                    const auto lpComparison =
                        vc::fiber_tracer::compareFiberDirectionLabels(
                            comparisonReport,
                            diagnosticDirections,
                            checkpointLpThresholded,
                            trustedMask);
                    const auto cohortPieces = [](const auto& cohort) {
                        return cohort.confusion[0].pieces +
                            cohort.confusion[1].pieces +
                            cohort.expectedDefectPieces;
                    };
                    const auto cohortErrors = [](const auto& cohort) {
                        return cohort.orientationErrors + cohort.brokenErrors +
                            cohort.defectActiveErrors;
                    };
                    std::cout << "fiber direction ablation checkpoint="
                              << checkpoint
                              << " admitted=" << admittedCount;
                    if (admittedCount == 0) {
                        std::cout << " latest_confidence=-";
                    } else {
                        const auto& latest = candidates[admittedCount - 1];
                        std::cout << " latest_confidence=" << std::fixed
                                  << std::setprecision(6) << latest.confidence
                                  << " latest_original_trace="
                                  << latest.lineIndex;
                    }
                    std::cout << " fibers=" << diagnosticLines.size()
                              << " pieces=" << checkpointReport.pieces.size()
                              << " constraints_extracted="
                              << checkpointReport.constraints.size()
                              << " constraints_retained="
                              << checkpointLabeling.retainedConstraints
                              << " excluded_non_perpendicular="
                              << checkpointLabeling.excludedNonPerpendicular
                              << " milp_status="
                              << checkpointLabeling.modelStatus
                              << " milp_gap=" << std::setprecision(6)
                              << checkpointLabeling.mipGap
                              << " milp_solve_seconds="
                              << checkpointLabeling.solveSeconds
                              << " milp_objective="
                              << checkpointLabeling.objective
                              << " lp_status=" << checkpointLp.modelStatus
                              << " lp_solve_seconds="
                              << checkpointLp.solveSeconds
                              << " lp_objective=" << checkpointLp.objective
                              << '\n';
                    const auto printErrors = [&](
                                                 const char* solver,
                                                 const auto& current) {
                        const std::size_t allPieces = current.rawH +
                            current.rawV + current.rawBroken;
                        const std::size_t allErrors =
                            current.orientationErrors + current.brokenErrors +
                            current.defectActiveErrors;
                        std::cout
                            << "fiber direction ablation errors checkpoint="
                            << checkpoint << " solver=" << solver
                            << " raw_h=" << current.rawH
                            << " raw_v=" << current.rawV
                            << " raw_broken=" << current.rawBroken
                            << " hv_orientation_errors="
                            << current.trusted.orientationErrors
                            << " hv_broken_errors="
                            << current.trusted.brokenErrors
                            << " hv_error_pieces="
                            << cohortErrors(current.trusted) << '/'
                            << cohortPieces(current.trusted)
                            << " hv_error_fibers="
                            << current.trusted.errorTraces << '/'
                            << current.trusted.representedTraces
                            << " mixed_active_errors="
                            << current.admitted.defectActiveErrors
                            << " mixed_broken_correct="
                            << current.admitted.defectBrokenPieces << '/'
                            << current.admitted.expectedDefectPieces
                            << " mixed_error_pieces="
                            << cohortErrors(current.admitted) << '/'
                            << cohortPieces(current.admitted)
                            << " mixed_error_fibers="
                            << current.admitted.errorTraces << '/'
                            << current.admitted.representedTraces
                            << " all_pieces=" << allErrors << '/'
                            << allPieces << " all_fibers="
                            << current.errorTraces << '/'
                            << current.representedTraces
                            << " components=" << current.activeComponents
                            << '\n';
                    };
                    printErrors("milp", comparison);
                    printErrors("lp_threshold_0.5", lpComparison);
                    std::cout << std::defaultfloat;

                    if (checkpoint + 1 == admittedCounts.size()) {
                        const auto objReport =
                            vc::fiber_tracer::writeFiberTraceConstraintObjs(
                                checkpointReport, options.output);
                        const auto labelObjReport =
                            vc::fiber_tracer::writeFiberTraceLabelObjs(
                                checkpointReport,
                                checkpointLabeling,
                                options.output);
                        const auto lpOutput = options.output.parent_path() /
                            (options.output.stem().string() +
                             "_lp_thresholded");
                        const auto lpLabelObjReport =
                            vc::fiber_tracer::writeFiberTraceLabelObjs(
                                checkpointReport,
                                checkpointLpThresholded,
                                lpOutput);
                        if (options.bpBalance != BpBalanceSelection::None) {
                            const auto bpPieceLines = vc::fiber_tracer::
                                makeFiberTraceConstraintPieceLines(
                                    diagnosticLines, comparisonReport);
                            std::vector<std::size_t> bpOriginalTraceIndices;
                            std::vector<vc::fiber_tracer::FiberDirectionGroup>
                                bpDirections;
                            bpOriginalTraceIndices.reserve(
                                comparisonReport.pieces.size());
                            bpDirections.reserve(comparisonReport.pieces.size());
                            for (const auto& piece : comparisonReport.pieces) {
                                bpOriginalTraceIndices.push_back(
                                    diagnosticOriginalTraceIndices.at(
                                        piece.traceIndex));
                                bpDirections.push_back(
                                    diagnosticDirections.at(piece.traceIndex));
                            }
                            const auto runBp = [&](
                                                   vc::fiber_tracer::
                                                       FiberTraceBalanceMode mode) {
                                auto config = options.bp;
                                config.balanceMode = mode;
                                config.cropMinimumBaseXYZ =
                                    artifact.minimumBaseXYZ;
                                config.cropMaximumBaseXYZ =
                                    artifact.maximumBaseXYZ;
                                const auto report = vc::fiber_tracer::
                                    solveFiberTraceBeliefPropagation(
                                        diagnosticLines,
                                        comparisonReport,
                                        config);
                                writeAndPrintBpReport(
                                    report,
                                    mode,
                                    bpPieceLines,
                                    comparisonReport,
                                    bpOriginalTraceIndices,
                                    bpDirections,
                                    options.output);
                            };
                            if (options.bpBalance == BpBalanceSelection::Soft ||
                                options.bpBalance == BpBalanceSelection::Both) {
                                runBp(vc::fiber_tracer::
                                          FiberTraceBalanceMode::Soft);
                            }
                            if (options.bpBalance == BpBalanceSelection::Tight ||
                                options.bpBalance == BpBalanceSelection::Both) {
                                runBp(vc::fiber_tracer::
                                          FiberTraceBalanceMode::Tight);
                            }
                        }
                        if (options.postIterations > 0) {
                            const auto values = vc::fiber_tracer::
                                postFilterPerpendicularFiberTraceLabels(
                                    checkpointReport,
                                    checkpointLabeling,
                                    diagnosticLines.size(),
                                    {options.postIterations,
                                     options.postInfluence});
                            const auto bands =
                                vc::fiber_tracer::classifyFiberValues(values);
                            const auto paths = vc::fiber_tracer::
                                writeFiberletCropValueBandObjs(
                                    diagnosticLines, bands, options.output);
                            std::vector<unsigned char> hErrors(values.size(), 0);
                            std::vector<unsigned char> vErrors(values.size(), 0);
                            std::vector<unsigned char> mixedErrors(
                                values.size(), 0);
                            for (const auto& error : comparison.errors) {
                                if (error.filteredTraceIndex >= values.size()) {
                                    throw std::logic_error(
                                        "post-filter error references an invalid fiber");
                                }
                                if (!error.trustedReference) {
                                    mixedErrors[error.filteredTraceIndex] = 1;
                                } else if (
                                    error.initialDirection ==
                                    vc::fiber_tracer::FiberDirectionGroup::Direction1) {
                                    hErrors[error.filteredTraceIndex] = 1;
                                } else {
                                    vErrors[error.filteredTraceIndex] = 1;
                                }
                            }
                            std::cout
                                << "fiber direction post-filter"
                                << " iterations=" << options.postIterations
                                << " influence=" << options.postInfluence
                                << " fibers=" << values.size() << '\n'
                                << "band  count  h_ref  v_ref  mixed_ref"
                                   "  h_errors  v_errors  mixed_errors"
                                   "  total_errors  min  mean  max  path\n"
                                << std::fixed << std::setprecision(6);
                            for (std::size_t band = 0;
                                 band < bands.bands.size();
                                 ++band) {
                                const auto& current = bands.bands[band];
                                std::size_t hReferences = 0;
                                std::size_t vReferences = 0;
                                std::size_t mixedReferences = 0;
                                std::size_t hErrorCount = 0;
                                std::size_t vErrorCount = 0;
                                std::size_t mixedErrorCount = 0;
                                for (const std::size_t trace :
                                     current.lineIndices) {
                                    const auto direction =
                                        diagnosticDirections.at(trace);
                                    hReferences += direction ==
                                        vc::fiber_tracer::FiberDirectionGroup::Direction1;
                                    vReferences += direction ==
                                        vc::fiber_tracer::FiberDirectionGroup::Direction2;
                                    mixedReferences += direction ==
                                        vc::fiber_tracer::FiberDirectionGroup::Mixed;
                                    hErrorCount += hErrors[trace];
                                    vErrorCount += vErrors[trace];
                                    mixedErrorCount += mixedErrors[trace];
                                }
                                std::cout << 'p' << band << "  "
                                          << current.lineIndices.size()
                                          << "  " << hReferences
                                          << "  " << vReferences
                                          << "  " << mixedReferences
                                          << "  " << hErrorCount
                                          << "  " << vErrorCount
                                          << "  " << mixedErrorCount
                                          << "  " << hErrorCount +
                                                vErrorCount + mixedErrorCount
                                          << "  " << current.minimumValue
                                          << "  " << current.meanValue
                                          << "  " << current.maximumValue
                                          << "  " << paths.bands[band]
                                          << '\n';
                            }
                            std::cout << std::defaultfloat;
                        }
                        printConstraintObjReport(objReport);
                        printLabelingReport(
                            checkpointLabeling,
                            options.labeling,
                            labelObjReport);
                        std::cout
                            << "fiber direction ablation lp_thresholded_output="
                            << lpLabelObjReport.paths.hEven.parent_path() /
                                lpOutput.stem()
                            << '\n';
                        if (checkpointPruning)
                            printConstraintPruningReport(*checkpointPruning);
                        std::cout
                            << "fiber direction ablation final_extracted_constraints="
                            << extractedConstraints.size() << '\n';
                    }
                }
                return 0;
            }
            if (options.mode == Mode::DirectionDiagnostic) {
                diagnosticLines.reserve(artifact.lines.size());
                diagnosticOriginalTraceIndices.reserve(artifact.lines.size());
                diagnosticDirections.reserve(artifact.lines.size());
                for (std::size_t trace = 0;
                     trace < artifact.lines.size();
                     ++trace) {
                    const auto group =
                        diagnosticClassification->lines[trace].group;
                    if (group == vc::fiber_tracer::FiberDirectionGroup::Mixed)
                        continue;
                    diagnosticLines.push_back(artifact.lines[trace]);
                    diagnosticOriginalTraceIndices.push_back(trace);
                    diagnosticDirections.push_back(group);
                }
                constraintLines = &diagnosticLines;
            }
            auto report = vc::fiber_tracer::extractFiberTraceConstraints(
                *constraintLines,
                options.constraints,
                [&normals](const cv::Vec3d& a,
                           const cv::Vec3d& b,
                           double step) {
                    return normals.normalAlignedWindingDistance(a, b, step);
                },
                [&normals](
                    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
                    double step,
                    int threads) {
                    return normals.normalAlignedWindingDistancesBatch(
                        connectors, step, threads);
                });
            std::vector<vc::fiber_tracer::FiberTraceConstraint>
                extractedConstraints;
            std::optional<vc::fiber_tracer::FiberTraceConstraintPruningReport>
                pruningReport;
            if (options.maximumConstraintsPerFiber) {
                extractedConstraints = report.constraints;
                auto pruning =
                    vc::fiber_tracer::pruneFiberTraceConstraintsByStrength(
                        report,
                        options.constraints.maximumDistanceBaseVoxels,
                        *options.maximumConstraintsPerFiber);
                report.constraints = std::move(pruning.constraints);
                pruningReport = std::move(pruning.report);
            }
            const auto& extractionConstraints = extractedConstraints.empty()
                ? report.constraints
                : extractedConstraints;
            if (options.mode == Mode::Consensus) {
                vc::fiber_tracer::FiberTraceConsensusConfig consensusConfig;
                consensusConfig.brokenCostPerConstraint =
                    options.labeling.brokenCostPerConstraint;
                consensusConfig.cropMinimumBaseXYZ = artifact.minimumBaseXYZ;
                consensusConfig.cropMaximumBaseXYZ = artifact.maximumBaseXYZ;
                const auto consensus = vc::fiber_tracer::growFiberTraceConsensus(
                    artifact.lines, report, consensusConfig);
                const auto consensusObjects =
                    vc::fiber_tracer::writeFiberTraceConsensusObjs(
                        artifact.lines, consensus, options.output);
                const double wallSeconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                const double cpuSeconds = static_cast<double>(
                    std::clock() - cpuStarted) / CLOCKS_PER_SEC;
                printConstraintReport(
                    report,
                    extractionConstraints,
                    options.constraints,
                    manifestMatches,
                    wallSeconds,
                    cpuSeconds);
                if (pruningReport)
                    printConstraintPruningReport(*pruningReport);
                printConsensusReport(
                    consensus, consensusConfig, consensusObjects);
                return 0;
            }
            const auto objReport =
                vc::fiber_tracer::writeFiberTraceConstraintObjs(
                    report, options.output);
            const auto labeling = vc::fiber_tracer::solveFiberTraceLabels(
                report, options.labeling);
            std::optional<vc::fiber_tracer::FiberTraceLabelObjReport> labelObjReport;
            std::optional<std::filesystem::path> relaxationCsv;
            std::optional<vc::fiber_tracer::FiberTraceRelaxationObjReport>
                relaxationVisualization;
            if (labeling.continuousPieceValues) {
                relaxationCsv = vc::fiber_tracer::writeFiberTraceLabelRelaxationCsv(
                    report, labeling, options.output);
                relaxationVisualization =
                    vc::fiber_tracer::writeFiberTraceLabelRelaxationObjs(
                        report, labeling, options.output);
            } else {
                labelObjReport = vc::fiber_tracer::writeFiberTraceLabelObjs(
                    report, labeling, options.output);
            }
            const double wallSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const double cpuSeconds = static_cast<double>(
                std::clock() - cpuStarted) / CLOCKS_PER_SEC;
            printConstraintReport(
                report,
                extractionConstraints,
                options.constraints,
                manifestMatches,
                wallSeconds,
                cpuSeconds);
            if (pruningReport)
                printConstraintPruningReport(*pruningReport);
            printConstraintObjReport(objReport);
            if (relaxationCsv) {
                printRelaxedLabelingReport(
                    labeling,
                    options.labeling,
                    *relaxationCsv,
                    *relaxationVisualization);
            } else {
                printLabelingReport(
                    labeling, options.labeling, *labelObjReport);
            }
            if (options.mode == Mode::DirectionDiagnostic) {
                const auto comparison =
                    vc::fiber_tracer::compareFiberDirectionLabels(
                        report, diagnosticDirections, labeling);
                printDirectionDiagnosticReport(
                    *diagnosticClassification,
                    diagnosticOriginalTraceIndices,
                    comparison);
            }
            return 0;
        }
        if (options.mode == Mode::Visualize) {
            const auto artifact = vc::fiber_tracer::readFiberletCropTraceArtifact(options.input);
            const auto report = visualize(
                artifact.lines, options.output, options.directionDominance);
            printDirectionReport(report.directions, options.output);
            std::cout << "fiberlet crop visualization completed"
                      << " traces=" << artifact.lines.size() << " input=" << options.input << " output=" << options.output << '\n';
            return 0;
        }

        auto dataset = vc::fiber_tracer::FiberletChunkDataset::openExisting(options.input);
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
        vc::fiber_tracer::validateFiberletNormalDatasetCompatibility(dataset->metadata(), normalDataset);
        const vc::lasagna::LasagnaNormalSampler normals(normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{options.cacheBytes});

        const auto graphStarted = std::chrono::steady_clock::now();
        const auto graphCpuStarted = std::clock();
        auto materialized =
            graph.materializeBaseBox(options.trace.minimumBaseXYZ, options.trace.maximumBaseXYZ, static_cast<std::size_t>(options.threads));
        const double graphSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - graphStarted).count();
        const double graphCpuSeconds = static_cast<double>(std::clock() - graphCpuStarted) / CLOCKS_PER_SEC;
        std::cout << "fiberlet crop graph prepared"
                  << " anchors=" << materialized.insideAnchors.size() << " prediction_to_base=" << dataset->metadata().predictionToBaseScale
                  << " elapsed_seconds=" << graphSeconds << " cpu_seconds=" << graphCpuSeconds << '\n';

        const auto traceStarted = std::chrono::steady_clock::now();
        const auto traceCpuStarted = std::clock();
        const auto result = vc::fiber_tracer::traceFiberletCrop(
            *materialized.graph,
            std::move(materialized.insideAnchors),
            normals,
            dataset->metadata().predictionToBaseScale,
            options.trace,
            [](const auto& current, std::size_t remaining) {
                std::cout << "fiberlet crop attempted=" << current.attemptedAnchors << " accepted=" << current.lines.size()
                          << " covered=" << current.coveredAnchors << " remaining=" << remaining << '\n';
            });
        const double traceSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - traceStarted).count();
        const double traceCpuSeconds = static_cast<double>(std::clock() - traceCpuStarted) / CLOCKS_PER_SEC;

        vc::fiber_tracer::
            writeFiberletCropTraceArtifact(options.output, dataset->metadata(), normalDataset.manifest().raw, options.trace, result.lines);
        const auto artifact = vc::fiber_tracer::readFiberletCropTraceArtifact(options.output);
        const auto visualization = visualize(
            artifact.lines, options.obj, options.directionDominance);

        if (!options.volume.empty()) {
            const std::string locator = std::filesystem::absolute(options.volume).lexically_normal().string();
            auto volume = Volume::New(options.volume);
            vc::fiber_tracer::writeFiberletCropBoxVisualization(
                *volume,
                locator,
                options.trace.minimumBaseXYZ,
                options.trace.maximumBaseXYZ,
                options.obj.parent_path() / (options.obj.stem().string() + "_volume_slices.obj"),
                options.maximumTextureDimension);
        }

        std::cout << "fiberlet crop completed"
                  << " candidates=" << result.candidateAnchors << " attempted=" << result.attemptedAnchors << " covered=" << result.coveredAnchors
                  << " computed=" << result.computedCandidates << " discarded=" << result.discardedCandidates
                  << " accepted=" << artifact.lines.size() << " no_edge=" << result.noEdgeAnchors << " one_sided=" << result.oneSidedLines
                  << " bidirectional=" << result.bidirectionalLines << " trace_output=" << options.output << " obj_output=" << options.obj << '\n';
        printDirectionReport(visualization.directions, options.obj);
        std::cout << "fiberlet crop timing"
                  << " graph_seconds=" << graphSeconds << " graph_cpu_seconds=" << graphCpuSeconds << " trace_seconds=" << traceSeconds
                  << " trace_cpu_seconds=" << traceCpuSeconds << " candidate_batch_seconds=" << result.candidateBatchSeconds
                  << " candidate_batch_cpu_seconds=" << result.candidateBatchCpuSeconds << " candidate_task_seconds=" << result.candidateTaskSeconds
                  << " candidate_task_max_seconds=" << result.maximumCandidateTaskSeconds
                  << " lookahead_route_nodes_max=" << result.maximumLookaheadRouteNodes << " lookahead_route_bytes_max=" << result.maximumLookaheadRouteBytes
                  << " integration_seconds=" << result.integrationSeconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_trace_chunk: " << error.what() << '\n';
        return 1;
    }
}
