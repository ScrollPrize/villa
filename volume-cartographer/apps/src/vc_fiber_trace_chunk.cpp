#include "vc/core/types/Volume.hpp"
#include "vc/core/io/PolylineObj.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberletCropVisualization.hpp"
#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/FiberTraceConsensus.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberTraceLabeling.hpp"
#include "vc/fiber_tracer/LasagnaNormalAlignment.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <numbers>
#include <optional>
#include <sstream>
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

constexpr double kDefaultWindingPhase = 0.5;
constexpr double kDefaultWindingMeasurementScale = 0.822;

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
    std::optional<double> qualityFraction;
    std::filesystem::path referenceFiberDirectory;
    std::string referenceFiberTag;
    std::size_t ablationStep = 5;
    std::optional<std::size_t> ablationLimit;
    std::size_t postIterations = 0;
    double postInfluence = 1.0;
    BpBalanceSelection bpBalance = BpBalanceSelection::None;
    vc::fiber_tracer::FiberTraceBeliefInference bpInference =
        vc::fiber_tracer::FiberTraceBeliefInference::MinSum;
    bool bpOnly = false;
    vc::fiber_tracer::FiberTraceBeliefPropagationConfig bp;
    vc::fiber_tracer::FiberTraceWindingSolver windingSolver =
        vc::fiber_tracer::FiberTraceWindingSolver::JointGrid;
    bool windingFixedOrientation = false;
    double windingDefectCost = 100.0;
    double pieceBreakCost = 0.0;
    bool hardSplitContinuity = true;
    std::optional<double> hardSignMinimumNormalAlignment =
        0.8660254037844386;
    std::optional<double> parallelWindingCutoff;
    bool enforcePerpendicularWindingSign = true;
    bool enforceParallelWindingSign = true;
    vc::fiber_tracer::FiberTraceWindingDecisionConfidence
        windingDecisionConfidence =
        vc::fiber_tracer::FiberTraceWindingDecisionConfidence::Cosine;
    vc::fiber_tracer::FiberTraceWindingNormalConfidence
        windingNormalConfidence =
        vc::fiber_tracer::FiberTraceWindingNormalConfidence::Linear;
    std::optional<double> windingSignCost = 44.0;
    std::array<double, 5> windingWeights =
        vc::fiber_tracer::kDefaultFiberTraceWindingClassWeights;
    std::array<double, 2> windingSignWeights =
        vc::fiber_tracer::kDefaultFiberTraceWindingSignWeights;
    std::optional<std::vector<double>> windingWeightSearch;
    bool windingWeightSearchLocal = false;
    vc::fiber_tracer::FiberTraceJointGridWindingConfig jointGrid;
    bool hasBounds = false;
    bool hasTraceOnlyOption = false;
    bool hasConstraintOnlyOption = false;
    bool hasSolverOnlyOption = false;
    bool hasSharedRuntimeOption = false;
    bool profileMemory = false;
    bool hasDirectionVisualizationOption = false;
    bool hasHvOnlyOption = false;
    bool hasAblationOnlyOption = false;
    bool hasPostInfluenceOption = false;
    bool hasBpTuningOption = false;
    bool hasBpInferenceOption = false;
    bool hasBpBalanceTuningOption = false;
    bool hasBpMixedCostOption = false;
    bool hasReferenceFiberDirectoryOption = false;
    bool hasReferenceFiberTagOption = false;
    bool hasWindingSolverOption = false;
    bool hasWindingOrientationOption = false;
    bool hasWindingDefectCostOption = false;
    bool hasPieceBreakCostOption = false;
    bool hasParallelWindingCutoffOption = false;
    bool hasWindingWeightOption = false;
    bool hasWindingSignWeightOption = false;
    bool hasWindingSignOption = false;
    bool hasWindingDecisionConfidenceOption = false;
    bool hasWindingNormalConfidenceOption = false;
    bool hasWindingSignCostOption = false;
    bool hasWindingWeightSearchOption = false;
    bool hasWindingWeightSearchLocalOption = false;
    bool hasJointGridOption = false;
    bool hasAdaptiveGridOption = false;
    bool hasFixedCalibrationOption = false;
    bool hasWindingCutoffOption = false;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

double orientationProjection(
    double horizontal,
    double mixed,
    double vertical,
    std::size_t piece)
{
    constexpr double tolerance = 1.0e-12;
    const std::array probabilities{horizontal, mixed, vertical};
    const double total = horizontal + mixed + vertical;
    if (std::any_of(
            probabilities.begin(), probabilities.end(),
            [](double probability) {
                return !std::isfinite(probability) ||
                    probability < -tolerance ||
                    probability > 1.0 + tolerance;
            }) ||
        !std::isfinite(total) || std::abs(total - 1.0) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Winding BP produced an invalid orientation marginal at piece "
                << piece << ": H=" << horizontal << " Mixed=" << mixed
                << " V=" << vertical << " total=" << total;
        throw std::runtime_error(message.str());
    }
    const double value = horizontal + 0.5 * mixed;
    if (!std::isfinite(value) || value < -tolerance ||
        value > 1.0 + tolerance) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Winding BP produced an invalid orientation projection at piece "
                << piece << ": value=" << value;
        throw std::runtime_error(message.str());
    }
    return std::clamp(value, 0.0, 1.0);
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
              << "Stored trace input options:\n"
              << "  --quality-fraction F      retain the best F fraction by cost density\n"
              << "  --reference-fiber-dir DIR tagged VC3D reference fiber JSON directory\n"
              << "  --reference-fiber-tag TAG exact tag selected from that directory\n\n"
              << "Trace options:\n"
              << "  --obj PATH                 line OBJ; defaults beside trace Zarr\n"
              << "  --volume PATH              concrete uint8 CT Zarr group\n"
              << "  --remote-cache-dir PATH    cache for a remote normal manifest\n"
              << "  --threads N                graph preparation and trace workers [host CPUs]\n"
              << "  --cache-gib N              decoded graph/normal cache [8]\n"
              << "  --profile-memory          print graph/cache memory counters every second\n"
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
              << "  --bp-mixed-cost F         orientation-prepass Mixed cost per constraint [1]\n"
              << "  --bp-balance MODE         soft, tight, or both [disabled]\n"
              << "  --bp-target F             arc-weighted H fraction [0.5]\n"
              << "  --bp-soft-strength F      quadratic balance strength [1]\n"
              << "  --bp-temperature F        min-marginal decoding temperature [1.25]\n"
              << "  --bp-message-iterations N message update limit [500]\n"
              << "  --bp-balance-iterations N field update limit [64]\n"
              << "  --bp-damping F            message damping in (0,1] [0.5]\n"
              << "  --bp-residual F           message residual tolerance [1e-8]\n"
              << "  --bp-balance-tolerance F  target/field tolerance [1e-3]\n"
              << "  --winding-solver MODE     joint-grid or alternating [joint-grid]\n"
              << "  --winding-fixed-orientation  solve H/V/Mixed first, then only winding\n"
              << "  --winding-defect-cost F   winding-stage Defect cost per constraint [100]\n"
              << "  --split-continuity MODE   hard or finite [hard]\n"
              << "  --piece-break-cost F      finite-mode same-trace boundary cost [0]\n"
              << "  --parallel-winding-cutoff F\n"
              << "                              exclusive parallel integer-distance cutoff [off]\n"
              << "  --winding-hard-signs MODE  none, perpendicular, parallel, or both [both]\n"
              << "  --winding-decision-confidence MODE\n"
              << "                              legacy, linear, or cosine [cosine]\n"
              << "  --winding-normal-confidence MODE\n"
              << "                              none, linear, or cosine [linear]\n"
              << "  --winding-sign-cost F|hard finite enabled-sign infringement cost [44]\n"
              << "  --winding-hard-sign-angle DEG|off\n"
              << "                              promote signs within DEG of normal to hard [30]\n"
              << "  --winding-weights P05,PFAR,P0,P1,P2\n"
              << "                              five nonnegative factor multipliers [0.5,0,1,2,1]\n"
              << "  --winding-sign-weights PERP,PARALLEL\n"
              << "                              two nonnegative sign multipliers [0.5,1]\n"
              << "  --winding-weight-search V0,V1,...\n"
              << "                              exhaustive seven-weight reference grid\n"
              << "  --winding-weight-search-local\n"
              << "                              repeat one-coordinate /2,*2 search to a local optimum\n"
              << "  --winding-fixed-phase F   fixed phase in [0,0.5] [0.5]\n"
              << "  --winding-fixed-scale F   fixed positive measurement scale [0.822]\n"
              << "  --winding-adaptive-calibration\n"
              << "                              infer phase and scale instead of fixed defaults\n"
              << "  --winding-gain-cells N    initial joint gain cells [5]\n"
              << "  --winding-phase-cells N   joint canonical phase cells [6]\n"
              << "  --winding-log-gain-step F joint log-gain lattice spacing [log(1.1)]\n"
              << "  --winding-boundary F      calibration boundary pressure [0.25]\n"
              << "  --winding-max-gain-cells N joint gain support guard [17]\n"
              << "  --winding-max-shifts N    joint sliding-window shift guard [32]\n\n"
              << "Constraint options (all distances are base voxels):\n"
              << "  --output PATH              OBJ basename; defaults beside trace dataset\n"
              << "  --sample-step N            common trace resampling step [32]\n"
              << "  --piece-length N           target overlapping piece length [512]\n"
              << "  --piece-overlap N          neighboring piece overlap [128]\n"
              << "  --max-distance N           closest-pair threshold [128]\n"
              << "  --tangent-window N         centered tangent secant length [32]\n"
              << "  --parallel-correspondence MODE\n"
              << "                              distance or perpendicular-grid [distance]\n"
              << "  --parallel-grid-step F     grid step as sample-step fraction [0.05]\n"
              << "  --parallel-grid-limit F    per-step grid range as sample-step fraction [0.25]\n"
              << "  --parallel-step-weight F   nonnegative advance-residual weight [1]\n"
              << "  --parallel-perp-weight F   nonnegative perpendicularity weight [1]\n"
              << "  --parallel-direction-weight F\n"
              << "                              nonnegative connector continuity weight [0]\n"
              << "  --parallel-length-weight F nonnegative length continuity weight [0]\n"
              << "  --parallel-diagnostics     collect correspondence geometry CSV fields\n"
              << "  --winding-step N           Lasagna connector integration step [8]\n"
              << "  --winding-cutoff N         exclusive finite winding cutoff [4 H/V; 1.5 parity]\n"
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

std::vector<double> numberList(
    int& index,
    int argc,
    char** argv,
    const char* option,
    bool allowZero = false)
{
    const std::string input = value(index, argc, argv, option);
    std::vector<double> result;
    std::size_t begin = 0;
    while (begin <= input.size()) {
        const std::size_t end = input.find(',', begin);
        const std::string item = input.substr(
            begin,
            end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty())
            fail(std::string(option) + " requires comma-separated numbers");
        std::size_t parsed = 0;
        const double parsedValue = std::stod(item, &parsed);
        if (parsed != item.size() || !std::isfinite(parsedValue) ||
            (allowZero ? parsedValue < 0.0 : !(parsedValue > 0.0))) {
            fail(std::string(option) +
                 (allowZero
                      ? " requires finite nonnegative comma-separated numbers"
                      : " requires finite positive comma-separated numbers"));
        }
        result.push_back(parsedValue);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    if (result.empty())
        fail(std::string(option) + " requires at least one number");
    return result;
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
        } else if (argument == "--quality-fraction") {
            options.qualityFraction = number(
                index, argc, argv, "--quality-fraction");
            if (!(*options.qualityFraction > 0.0) ||
                *options.qualityFraction > 1.0) {
                fail("--quality-fraction must be in (0, 1]");
            }
        } else if (argument == "--reference-fiber-dir") {
            options.referenceFiberDirectory = value(
                index, argc, argv, "--reference-fiber-dir");
            options.hasReferenceFiberDirectoryOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--reference-fiber-tag") {
            options.referenceFiberTag = value(
                index, argc, argv, "--reference-fiber-tag");
            options.hasReferenceFiberTagOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
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
        } else if (argument == "--profile-memory") {
            options.profileMemory = true;
            options.hasTraceOnlyOption = true;
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
        } else if (argument == "--winding-solver") {
            const std::string solver = value(
                index, argc, argv, "--winding-solver");
            if (solver == "joint-grid") {
                options.windingSolver = vc::fiber_tracer::
                    FiberTraceWindingSolver::JointGrid;
            } else if (solver == "alternating") {
                options.windingSolver = vc::fiber_tracer::
                    FiberTraceWindingSolver::Alternating;
            } else {
                fail("--winding-solver must be joint-grid or alternating");
            }
            options.hasWindingSolverOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-fixed-orientation") {
            options.windingFixedOrientation = true;
            options.hasWindingOrientationOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-defect-cost") {
            options.windingDefectCost = number(
                index, argc, argv, "--winding-defect-cost");
            if (!std::isfinite(options.windingDefectCost) ||
                options.windingDefectCost < 0.0) {
                fail("--winding-defect-cost must be finite and nonnegative");
            }
            options.hasWindingDefectCostOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--piece-break-cost") {
            options.pieceBreakCost = number(
                index, argc, argv, "--piece-break-cost");
            if (!std::isfinite(options.pieceBreakCost) ||
                options.pieceBreakCost < 0.0) {
                fail("--piece-break-cost must be finite and nonnegative");
            }
            options.hasPieceBreakCostOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--split-continuity") {
            const std::string mode = value(
                index, argc, argv, "--split-continuity");
            if (mode == "hard")
                options.hardSplitContinuity = true;
            else if (mode == "finite")
                options.hardSplitContinuity = false;
            else
                fail("--split-continuity must be hard or finite");
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-winding-cutoff") {
            options.parallelWindingCutoff = number(
                index, argc, argv, "--parallel-winding-cutoff");
            if (!std::isfinite(*options.parallelWindingCutoff) ||
                !(*options.parallelWindingCutoff > 0.0)) {
                fail("--parallel-winding-cutoff must be finite and positive");
            }
            options.hasParallelWindingCutoffOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-weights") {
            const auto values = numberList(
                index, argc, argv, "--winding-weights", true);
            if (values.size() != options.windingWeights.size()) {
                fail("--winding-weights requires exactly five values");
            }
            std::copy(
                values.begin(), values.end(), options.windingWeights.begin());
            options.hasWindingWeightOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-sign-weights") {
            const auto values = numberList(
                index, argc, argv, "--winding-sign-weights", true);
            if (values.size() != options.windingSignWeights.size()) {
                fail("--winding-sign-weights requires exactly two values");
            }
            std::copy(
                values.begin(), values.end(),
                options.windingSignWeights.begin());
            options.hasWindingSignWeightOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-hard-signs") {
            const std::string mode = value(
                index, argc, argv, "--winding-hard-signs");
            if (mode == "none") {
                options.enforcePerpendicularWindingSign = false;
                options.enforceParallelWindingSign = false;
            } else if (mode == "perpendicular") {
                options.enforcePerpendicularWindingSign = true;
                options.enforceParallelWindingSign = false;
            } else if (mode == "parallel") {
                options.enforcePerpendicularWindingSign = false;
                options.enforceParallelWindingSign = true;
            } else if (mode == "both") {
                options.enforcePerpendicularWindingSign = true;
                options.enforceParallelWindingSign = true;
            } else {
                fail("--winding-hard-signs must be none, perpendicular, parallel, or both");
            }
            options.hasWindingSignOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-decision-confidence") {
            const std::string mode = value(
                index, argc, argv, "--winding-decision-confidence");
            if (mode == "legacy") {
                options.windingDecisionConfidence = vc::fiber_tracer::
                    FiberTraceWindingDecisionConfidence::Legacy;
            } else if (mode == "linear") {
                options.windingDecisionConfidence = vc::fiber_tracer::
                    FiberTraceWindingDecisionConfidence::Linear;
            } else if (mode == "cosine") {
                options.windingDecisionConfidence = vc::fiber_tracer::
                    FiberTraceWindingDecisionConfidence::Cosine;
            } else {
                fail("--winding-decision-confidence must be legacy, linear, or cosine");
            }
            options.hasWindingDecisionConfidenceOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-normal-confidence") {
            const std::string mode = value(
                index, argc, argv, "--winding-normal-confidence");
            if (mode == "none") {
                options.windingNormalConfidence = vc::fiber_tracer::
                    FiberTraceWindingNormalConfidence::None;
            } else if (mode == "linear") {
                options.windingNormalConfidence = vc::fiber_tracer::
                    FiberTraceWindingNormalConfidence::Linear;
            } else if (mode == "cosine") {
                options.windingNormalConfidence = vc::fiber_tracer::
                    FiberTraceWindingNormalConfidence::Cosine;
            } else {
                fail("--winding-normal-confidence must be none, linear, or cosine");
            }
            options.hasWindingNormalConfidenceOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-sign-cost") {
            const std::string signCost = value(
                index, argc, argv, "--winding-sign-cost");
            if (signCost == "hard") {
                options.windingSignCost.reset();
            } else {
                std::size_t parsed = 0;
                try {
                    options.windingSignCost = std::stod(signCost, &parsed);
                } catch (const std::exception&) {
                    fail("--winding-sign-cost must be hard or finite and nonnegative");
                }
                if (parsed != signCost.size() ||
                    !std::isfinite(*options.windingSignCost) ||
                    *options.windingSignCost < 0.0) {
                    fail("--winding-sign-cost must be hard or finite and nonnegative");
                }
            }
            options.hasWindingSignCostOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-hard-sign-angle") {
            const std::string angle = value(
                index, argc, argv, "--winding-hard-sign-angle");
            if (angle == "off") {
                options.hardSignMinimumNormalAlignment.reset();
            } else {
                std::size_t parsed = 0;
                double degrees = 0.0;
                try {
                    degrees = std::stod(angle, &parsed);
                } catch (const std::exception&) {
                    fail("--winding-hard-sign-angle must be off or in [0, 90]");
                }
                if (parsed != angle.size() || !std::isfinite(degrees) ||
                    degrees < 0.0 || degrees > 90.0) {
                    fail("--winding-hard-sign-angle must be off or in [0, 90]");
                }
                options.hardSignMinimumNormalAlignment =
                    std::cos(degrees * std::numbers::pi / 180.0);
            }
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-weight-search") {
            options.windingWeightSearch = numberList(
                index, argc, argv, "--winding-weight-search");
            std::sort(
                options.windingWeightSearch->begin(),
                options.windingWeightSearch->end());
            options.windingWeightSearch->erase(
                std::unique(
                    options.windingWeightSearch->begin(),
                    options.windingWeightSearch->end()),
                options.windingWeightSearch->end());
            options.hasWindingWeightSearchOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-weight-search-local") {
            options.windingWeightSearchLocal = true;
            options.hasWindingWeightSearchLocalOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-gain-cells") {
            options.jointGrid.initialGainCells = count(
                index, argc, argv, "--winding-gain-cells");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-phase-cells") {
            options.jointGrid.phaseCells = count(
                index, argc, argv, "--winding-phase-cells");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-log-gain-step") {
            options.jointGrid.logGainStep = number(
                index, argc, argv, "--winding-log-gain-step");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-boundary") {
            options.jointGrid.calibrationBoundaryProbabilityThreshold = number(
                index, argc, argv, "--winding-boundary");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-max-gain-cells") {
            options.jointGrid.maximumGainCells = count(
                index, argc, argv, "--winding-max-gain-cells");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-max-shifts") {
            options.jointGrid.maximumGridShifts = count(
                index, argc, argv, "--winding-max-shifts");
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-fixed-phase") {
            options.jointGrid.fixedPhaseMagnitude = number(
                index, argc, argv, "--winding-fixed-phase");
            options.hasJointGridOption = true;
            options.hasFixedCalibrationOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-fixed-scale") {
            options.jointGrid.fixedMeasurementScale = number(
                index, argc, argv, "--winding-fixed-scale");
            options.hasJointGridOption = true;
            options.hasFixedCalibrationOption = true;
            options.hasAblationOnlyOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--winding-adaptive-calibration") {
            options.hasJointGridOption = true;
            options.hasAdaptiveGridOption = true;
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
        } else if (argument == "--parallel-correspondence") {
            const auto mode = value(
                index, argc, argv, "--parallel-correspondence");
            if (mode == "distance") {
                options.constraints.parallelCorrespondence =
                    vc::fiber_tracer::FiberTraceParallelCorrespondence::Distance;
            } else if (mode == "perpendicular-grid") {
                options.constraints.parallelCorrespondence =
                    vc::fiber_tracer::FiberTraceParallelCorrespondence::PerpendicularGrid;
            } else {
                fail("--parallel-correspondence must be distance or perpendicular-grid");
            }
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-grid-step") {
            options.constraints.correspondenceGridStepFraction =
                number(index, argc, argv, "--parallel-grid-step");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-grid-limit") {
            options.constraints.correspondenceGridLimitFraction =
                number(index, argc, argv, "--parallel-grid-limit");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-step-weight") {
            options.constraints.correspondenceGridStepWeight =
                number(index, argc, argv, "--parallel-step-weight");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-perp-weight") {
            options.constraints.correspondenceGridPerpendicularWeight =
                number(index, argc, argv, "--parallel-perp-weight");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-direction-weight") {
            options.constraints.correspondenceGridDirectionWeight =
                number(index, argc, argv, "--parallel-direction-weight");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-length-weight") {
            options.constraints.correspondenceGridLengthWeight =
                number(index, argc, argv, "--parallel-length-weight");
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--parallel-diagnostics") {
            options.constraints.collectParallelCorrespondenceDiagnostics = true;
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
            options.hasWindingCutoffOption = true;
            options.hasConstraintOnlyOption = true;
        } else if (argument == "--no-winding-cutoff") {
            options.constraints.enforceMaximumWindingDistance = false;
            options.hasWindingCutoffOption = true;
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
            fail(
                "visualize accepts only a trace dataset, --output OBJ, "
                "--direction-dominance, and --quality-fraction");
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
        if (options.hasReferenceFiberDirectoryOption !=
            options.hasReferenceFiberTagOption) {
            fail("--reference-fiber-dir and --reference-fiber-tag must be used together");
        }
        if (options.hasReferenceFiberTagOption &&
            options.referenceFiberTag.empty()) {
            fail("--reference-fiber-tag must not be empty");
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
        if (!options.hasWindingCutoffOption && options.labeling.hvOnly)
            options.constraints.maximumWindingDistance = 4.0;
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
        if ((options.hasWindingSolverOption || options.hasJointGridOption ||
             options.hasWindingOrientationOption ||
             options.hasWindingDefectCostOption ||
             options.hasPieceBreakCostOption ||
             options.hasParallelWindingCutoffOption ||
             options.hasWindingWeightOption ||
             options.hasWindingSignWeightOption ||
             options.hasWindingSignOption ||
             options.hasWindingDecisionConfidenceOption ||
             options.hasWindingNormalConfidenceOption ||
             options.hasWindingSignCostOption ||
             options.hasWindingWeightSearchOption ||
             options.hasWindingWeightSearchLocalOption) &&
            (!options.bpOnly ||
             options.bpInference != vc::fiber_tracer::
                 FiberTraceBeliefInference::SumProductMixed)) {
            fail("winding solver controls require --bp-only --bp-inference sum-product-mixed");
        }
        if (options.hasWindingWeightSearchOption &&
            (options.hasWindingWeightOption ||
             options.hasWindingSignWeightOption ||
             options.hasWindingWeightSearchLocalOption)) {
            fail("explicit winding value/sign-hardness weights conflict with --winding-weight-search");
        }
        if (options.hasWindingWeightSearchLocalOption &&
            !options.hasWindingWeightOption) {
            fail("--winding-weight-search-local requires --winding-weights");
        }
        if ((options.hasWindingWeightSearchOption ||
             options.hasWindingWeightSearchLocalOption) &&
            !options.hasReferenceFiberDirectoryOption) {
            fail("winding weight search requires reference fibers");
        }
        if (options.hasJointGridOption &&
            options.windingSolver != vc::fiber_tracer::
                FiberTraceWindingSolver::JointGrid) {
            fail("joint-grid controls require --winding-solver joint-grid");
        }
        const bool hasExplicitFixedPhase =
            options.jointGrid.fixedPhaseMagnitude.has_value();
        const bool hasExplicitFixedScale =
            options.jointGrid.fixedMeasurementScale.has_value();
        if (hasExplicitFixedPhase != hasExplicitFixedScale) {
            fail("--winding-fixed-phase and --winding-fixed-scale must be supplied together");
        }
        if (options.hasFixedCalibrationOption && options.hasAdaptiveGridOption) {
            fail("fixed winding calibration cannot be combined with adaptive-grid controls");
        }
        if (!hasExplicitFixedPhase && !options.hasAdaptiveGridOption &&
            options.mode == Mode::DirectionAblation &&
            options.windingSolver == vc::fiber_tracer::
                FiberTraceWindingSolver::JointGrid) {
            options.jointGrid.fixedPhaseMagnitude = kDefaultWindingPhase;
            options.jointGrid.fixedMeasurementScale =
                kDefaultWindingMeasurementScale;
        }
        const bool hasFixedPhase =
            options.jointGrid.fixedPhaseMagnitude.has_value();
        const bool hasFixedScale =
            options.jointGrid.fixedMeasurementScale.has_value();
        if (hasFixedPhase &&
            (!std::isfinite(*options.jointGrid.fixedPhaseMagnitude) ||
             *options.jointGrid.fixedPhaseMagnitude < 0.0 ||
             *options.jointGrid.fixedPhaseMagnitude > 0.5)) {
            fail("--winding-fixed-phase must be finite and in [0,0.5]");
        }
        if (hasFixedScale &&
            (!std::isfinite(*options.jointGrid.fixedMeasurementScale) ||
             !(*options.jointGrid.fixedMeasurementScale > 0.0))) {
            fail("--winding-fixed-scale must be positive and finite");
        }
        if (options.jointGrid.initialGainCells == 0 ||
            options.jointGrid.initialGainCells % 2 == 0 ||
            options.jointGrid.phaseCells < 2 ||
            !(options.jointGrid.logGainStep > 0.0) ||
            !(options.jointGrid.calibrationBoundaryProbabilityThreshold > 0.0) ||
            options.jointGrid.calibrationBoundaryProbabilityThreshold >= 1.0 ||
            options.jointGrid.maximumGainCells <
                options.jointGrid.initialGainCells) {
            fail("joint-grid winding controls are invalid");
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
    if (options.qualityFraction)
        fail("trace does not accept --quality-fraction");
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
        winding.push_back(
            vc::fiber_tracer::dominantFiberTraceConstraintWindingDistance(
                constraint));
    }
    std::cout << std::setprecision(8)
              << "fiber trace constraint config\n"
              << "sample_step  piece_length  piece_overlap  max_distance  tangent_window  correspondence  winding_step  winding_cutoff  threads\n"
              << config.resampleSpacingBaseVoxels << "  "
              << config.targetPieceLengthBaseVoxels << "  "
              << config.pieceOverlapBaseVoxels << "  "
              << config.maximumDistanceBaseVoxels << "  "
              << config.tangentWindowBaseVoxels << "  "
              << (config.parallelCorrespondence ==
                          vc::fiber_tracer::FiberTraceParallelCorrespondence::Distance
                      ? "distance"
                      : "perpendicular-grid")
              << "  "
              << config.windingIntegrationStepBaseVoxels << "  "
              << (config.enforceMaximumWindingDistance
                      ? std::to_string(config.maximumWindingDistance)
                      : "off") << "  "
              << config.parallelThreads << '\n'
              << "fiber trace constraint counts\n"
              << "traces  degenerate  pieces  samples  spatial_hits  candidates  measured  hard  tangent_rejected  winding_invalid  "
                 "winding_cutoff  manifest_matches_trace\n"
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
              << "prepare_seconds  search_seconds  orientation_seconds  winding_seconds  score_seconds  total_wall_seconds  "
                 "total_cpu_seconds\n"
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
              << "limit  input_total  retained_total  hard  zero_rejected  discarded  recovery_candidates  expected_bridges  "
                 "recovery_bridges  cap_bridges  overflow_bridges  fibers_over_limit\n"
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
              << "status  hv_only  perpendicular_only  objective  orientation_cost  winding_cost  broken_cost  broken_cost_per_link  "
                 "retained_links  excluded_non_perpendicular  excluded_parallel_separate_winding  requested_mip_gap  variables  "
                 "integer_variables  rows  mip_nodes  mip_gap  solve_seconds\n"
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
              << "orientation_errors  broken_errors  defect_active_errors  total_errors  piece_error_rate  error_fibers  "
                 "represented_fibers  fiber_error_rate\n"
              << comparison.orientationErrors << "  "
              << comparison.brokenErrors << "  "
              << comparison.defectActiveErrors << "  " << errors << "  "
              << std::fixed << std::setprecision(4) << pieceErrorRate << "  "
              << comparison.errorTraces << "  "
              << comparison.representedTraces << "  " << traceErrorRate << '\n'
              << "fiber direction MILP error details\n"
              << "piece  filtered_trace  original_trace  trace_piece  begin_arc  end_arc  initial  raw_label  component  flipped  aligned  "
                 "kind\n";
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
              << "status  mode  hv_only  perpendicular_only  requested_solver  requested_parallel  threads  objective  orientation_cost  "
                 "winding_cost  broken_cost  broken_cost_per_link  retained_links  excluded_non_perpendicular  "
                 "excluded_parallel_separate_winding  variables  integer_variables  perpendicular_branch_variables  rows  gauge_roots  "
                 "triangles  triangle_rows  mip_nodes  mip_gap  solve_seconds  csv\n"
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
    const vc::fiber_tracer::FiberTraceWindingBeliefPropagationReport& winding,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport* interleaved,
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
        winding.windingValid.size() != lines.size() ||
        winding.continuousWinding.size() != lines.size() ||
        winding.mapWinding.size() != lines.size() ||
        winding.posteriorMeanWinding.size() != lines.size() ||
        winding.mapProbability.size() != lines.size() ||
        winding.entropy.size() != lines.size() ||
        winding.candidateMinimum.size() != lines.size() ||
        winding.candidateMaximum.size() != lines.size() ||
        winding.componentByPiece.size() != lines.size() ||
        winding.integerGaugeByPiece.size() != lines.size() ||
        winding.incidentSignedConstraints.size() != lines.size() ||
        winding.incidentSkippedConstraints.size() != lines.size() ||
        (interleaved &&
         (interleaved->classAProbability.size() != lines.size() ||
          interleaved->mixedProbability.size() != lines.size() ||
          interleaved->classBProbability.size() != lines.size() ||
          interleaved->mapLatentCoordinate.size() != lines.size() ||
          interleaved->mapOrientationByPiece.size() != lines.size())) ||
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
    const auto outputBase = output.parent_path() / output.stem();

    std::vector<vc::fiber_tracer::FiberTernaryState> predictions(
        lines.size(), vc::fiber_tracer::FiberTernaryState::Tie);
    const bool fixedPrepass = interleaved &&
        interleaved->orientationMode == vc::fiber_tracer::
            FiberTraceWindingOrientationMode::FixedPrepass;
    if (fixedPrepass &&
        interleaved->fixedOrientationByPiece.size() != lines.size()) {
        throw std::logic_error(
            "Fixed-prepass winding report does not match represented pieces");
    }
    std::optional<vc::fiber_tracer::FiberTernaryStateObjPaths> prepassPaths;
    if (fixedPrepass) {
        std::vector<vc::fiber_tracer::FiberTernaryState> prepassStates;
        prepassStates.reserve(lines.size());
        for (const auto orientation : interleaved->fixedOrientationByPiece) {
            switch (orientation) {
            case vc::fiber_tracer::FiberTraceFixedOrientation::Horizontal:
                prepassStates.push_back(
                    vc::fiber_tracer::FiberTernaryState::Horizontal);
                break;
            case vc::fiber_tracer::FiberTraceFixedOrientation::Mixed:
                prepassStates.push_back(
                    vc::fiber_tracer::FiberTernaryState::Mixed);
                break;
            case vc::fiber_tracer::FiberTraceFixedOrientation::Vertical:
                prepassStates.push_back(
                    vc::fiber_tracer::FiberTernaryState::Vertical);
                break;
            }
        }
        prepassPaths = vc::fiber_tracer::writeFiberletCropTernaryStateObjs(
            lines,
            prepassStates,
            outputBase.parent_path() /
                (outputBase.stem().string() + "_prepass"));
    }
    const auto ternaryPrediction = [](double horizontal,
                                      double mixed,
                                      double vertical) {
        const std::array probabilities{vertical, mixed, horizontal};
        const double maximum = *std::max_element(
            probabilities.begin(), probabilities.end());
        if (std::count(
                probabilities.begin(), probabilities.end(), maximum) != 1) {
            return vc::fiber_tracer::FiberTernaryState::Tie;
        }
        return static_cast<vc::fiber_tracer::FiberTernaryState>(std::distance(
            probabilities.begin(),
            std::find(probabilities.begin(), probabilities.end(), maximum)));
    };
    const auto ternaryName = [](vc::fiber_tracer::FiberTernaryState state) {
        switch (state) {
        case vc::fiber_tracer::FiberTernaryState::Vertical:
            return "v";
        case vc::fiber_tracer::FiberTernaryState::Mixed:
            return "defect";
        case vc::fiber_tracer::FiberTernaryState::Horizontal:
            return "h";
        case vc::fiber_tracer::FiberTernaryState::Tie:
            return "tie";
        }
        return "invalid";
    };
    for (std::size_t piece = 0; piece < lines.size(); ++piece) {
        if (interleaved && winding.windingValid[piece] == 0) {
            predictions[piece] =
                vc::fiber_tracer::FiberTernaryState::Mixed;
        } else if (interleaved) {
            predictions[piece] = ternaryPrediction(
                interleaved->classAProbability[piece],
                interleaved->mixedProbability[piece],
                interleaved->classBProbability[piece]);
        } else if (mixedState) {
            predictions[piece] = ternaryPrediction(
                report.horizontalProbability[piece],
                report.mixedProbability[piece],
                report.verticalProbability[piece]);
        } else if (report.horizontalness[piece] <= 0.25) {
            predictions[piece] =
                vc::fiber_tracer::FiberTernaryState::Vertical;
        } else if (report.horizontalness[piece] >= 0.75) {
            predictions[piece] =
                vc::fiber_tracer::FiberTernaryState::Horizontal;
        } else {
            predictions[piece] = vc::fiber_tracer::FiberTernaryState::Mixed;
        }
    }

    std::optional<int> relativeWindingMinimum;
    std::optional<int> relativeWindingMaximum;
    for (std::size_t piece = 0; piece < lines.size(); ++piece) {
        if (winding.windingValid[piece] == 0)
            continue;
        relativeWindingMinimum = relativeWindingMinimum
            ? std::min(*relativeWindingMinimum, winding.mapWinding[piece])
            : winding.mapWinding[piece];
        relativeWindingMaximum = relativeWindingMaximum
            ? std::max(*relativeWindingMaximum, winding.mapWinding[piece])
            : winding.mapWinding[piece];
    }
    const int windingOutputOffset = relativeWindingMinimum
        ? -*relativeWindingMinimum
        : 0;
    std::map<int, std::vector<vc::core::io::NamedPolyline>> windingLines;
    std::map<int, std::vector<std::size_t>> windingPieceIndices;
    for (std::size_t piece = 0; piece < lines.size(); ++piece) {
        if (winding.windingValid[piece] == 0)
            continue;
        const int outputWinding =
            winding.mapWinding[piece] + windingOutputOffset;
        windingLines[outputWinding].push_back({
            "piece_" + std::to_string(piece),
            lines[piece].pointsBaseXYZ,
        });
        windingPieceIndices[outputWinding].push_back(piece);
    }
    std::vector<std::filesystem::path> windingPaths;
    windingPaths.reserve(windingLines.size());
    for (const auto& [label, polylines] : windingLines) {
        const std::string suffix = std::to_string(label);
        const auto path = outputBase.parent_path() /
            (outputBase.stem().string() + "_w_" + suffix + ".obj");
        const int relativeLabel = label - windingOutputOffset;
        vc::core::io::writePolylinesObj(
            polylines,
            path,
            "VC3D winding output " + std::to_string(label) +
                " (relative MAP " + std::to_string(relativeLabel) + ")");
        std::vector<vc::fiber_tracer::FiberletCropTraceLine> stateLines;
        std::vector<vc::fiber_tracer::FiberTernaryState> stateValues;
        stateLines.reserve(windingPieceIndices.at(label).size());
        stateValues.reserve(windingPieceIndices.at(label).size());
        for (const std::size_t piece : windingPieceIndices.at(label)) {
            stateLines.push_back(lines[piece]);
            stateValues.push_back(predictions[piece]);
        }
        (void)vc::fiber_tracer::writeFiberletCropTernaryStateObjs(
            stateLines,
            stateValues,
            outputBase.parent_path() /
                (outputBase.stem().string() + "_w_" + suffix));
        windingPaths.push_back(path);
    }

    const auto factorCsv = outputBase.parent_path() /
        (outputBase.stem().string() + "_winding_factors.csv");
    std::ofstream factorOutput(factorCsv);
    if (!factorOutput)
        throw std::runtime_error(
            "failed to open winding factor CSV: " + factorCsv.string());
    factorOutput
        << "constraint,piece_a,piece_b,node_a,node_b,parallel,perpendicular,"
           "parallel_winding_weight_multiplier,"
           "perpendicular_winding_weight_multiplier,"
           "effective_parallel_winding_weight,"
           "effective_perpendicular_winding_weight,"
           "decision_confidence,normal_confidence,"
           "effective_parallel_sign_penalty,"
           "effective_perpendicular_sign_penalty,"
           "parallel_sign_weight_multiplier,"
           "perpendicular_sign_weight_multiplier,"
           "parallel_winding_present,perpendicular_winding_present,"
           "parallel_sign_present,perpendicular_sign_present,"
           "perpendicular_normal_alignment,parallel_normal_alignment,"
           "original_signed_delta,canonical_raw_signed_delta,"
           "original_signed_parallel_delta,"
           "canonical_raw_signed_parallel_delta,"
           "effective_parallel_winding_distance,"
           "effective_signed_parallel_delta,parallel_winding_retained,"
           "effective_perpendicular_signed_delta,"
           "calibrated_perpendicular_signed_delta,normal_component,self_edge,"
           "hard_parallel_sign,hard_perpendicular_sign,"
           "parallel_sign_promoted_by_alignment,"
           "perpendicular_sign_promoted_by_alignment,"
           "correspondence_samples,advance_residual_fraction,"
           "connector_tangent_abs_dot,connector_length_change_fraction,"
           "connector_direction_change,limit_hit_fraction\n"
        << std::setprecision(17);
    const auto writeOptionalDouble = [](std::ostream& stream,
                                        const std::optional<double>& value) {
        if (value)
            stream << *value;
        else
            stream << "NA";
    };
    const auto writeOptionalSize = [](std::ostream& stream,
                                      const std::optional<std::size_t>& value) {
        if (value)
            stream << *value;
        else
            stream << "NA";
    };
    for (const auto& factor : winding.factorDiagnostics) {
        factorOutput << factor.constraintIndex << ',' << factor.pieceA << ','
                     << factor.pieceB << ',' << factor.canonicalNodeA << ','
                     << factor.canonicalNodeB << ',' << factor.parallelScore
                     << ',' << factor.perpendicularScore << ','
                     << factor.parallelWindingWeightMultiplier << ','
                     << factor.perpendicularWindingWeightMultiplier << ','
                     << factor.effectiveParallelWindingWeight << ','
                     << factor.effectivePerpendicularWindingWeight << ','
                     << factor.decisionConfidenceMultiplier << ','
                     << factor.normalConfidenceMultiplier << ','
                     << factor.effectiveParallelSignPenalty << ','
                     << factor.effectivePerpendicularSignPenalty << ','
                     << factor.parallelSignWeightMultiplier << ','
                     << factor.perpendicularSignWeightMultiplier << ','
                     << (factor.parallelMagnitudePresent ? 1 : 0) << ','
                     << (factor.perpendicularMagnitudePresent ? 1 : 0) << ','
                     << (factor.parallelSignPresent ? 1 : 0) << ','
                     << (factor.perpendicularSignPresent ? 1 : 0) << ',';
        writeOptionalDouble(
            factorOutput, factor.perpendicularNormalAlignment);
        factorOutput << ',';
        writeOptionalDouble(factorOutput, factor.parallelNormalAlignment);
        factorOutput << ',';
        writeOptionalDouble(factorOutput, factor.originalSignedDelta);
        factorOutput << ',';
        writeOptionalDouble(factorOutput, factor.canonicalSignedDelta);
        factorOutput << ',';
        writeOptionalDouble(
            factorOutput, factor.originalSignedParallelDelta);
        factorOutput << ',';
        writeOptionalDouble(
            factorOutput, factor.canonicalSignedParallelDelta);
        factorOutput << ',' << factor.effectiveParallelWindingDistance << ',';
        writeOptionalDouble(
            factorOutput, factor.effectiveSignedParallelDelta);
        factorOutput << ','
                     << (factor.parallelWindingRetained ? 1 : 0) << ',';
        writeOptionalDouble(
            factorOutput, factor.effectivePerpendicularSignedDelta);
        factorOutput << ',';
        if (factor.effectivePerpendicularSignedDelta && interleaved)
            factorOutput << *factor.effectivePerpendicularSignedDelta *
                interleaved->measurementScale;
        else
            writeOptionalDouble(
                factorOutput, factor.effectivePerpendicularSignedDelta);
        factorOutput << ',';
        writeOptionalSize(factorOutput, factor.normalComponent);
        const auto& sourceConstraint =
            constraints.constraints.at(factor.constraintIndex);
        factorOutput << ',' << (factor.selfEdge ? 1 : 0) << ','
                     << (factor.hardParallelSign ? 1 : 0) << ','
                     << (factor.hardPerpendicularSign ? 1 : 0) << ','
                     << (factor.parallelSignPromotedByAlignment ? 1 : 0) << ','
                     << (factor.perpendicularSignPromotedByAlignment ? 1 : 0) << ','
                     << sourceConstraint.parallelCorrespondenceSamples << ','
                     << sourceConstraint.parallelMeanAdvanceResidualFraction << ','
                     << sourceConstraint.parallelMeanConnectorTangentAbsDot << ','
                     << sourceConstraint.parallelMeanConnectorLengthChangeFraction << ','
                     << sourceConstraint.parallelMeanConnectorDirectionChange << ','
                     << sourceConstraint.parallelLimitHitFraction << '\n';
    }
    if (!factorOutput)
        throw std::runtime_error(
            "failed to write winding factor CSV: " + factorCsv.string());
    const auto bands =
        vc::fiber_tracer::classifyFiberValues(report.horizontalness);
    const auto paths = vc::fiber_tracer::writeFiberletCropValueBandObjs(
        lines,
        bands,
        outputBase.parent_path() /
            (outputBase.stem().string() + "_orientation"));
    std::optional<vc::fiber_tracer::FiberValueBandObjPaths>
        mixedPaths;
    if (mixedState) {
        const auto mixedBands = vc::fiber_tracer::classifyFiberValues(
            report.mixedProbability);
        mixedPaths = vc::fiber_tracer::writeFiberletCropValueBandObjs(
            lines, mixedBands,
            output.parent_path() /
                (output.stem().string() + "_error_probability"));
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
    std::optional<std::array<std::size_t, 4>> finalStateCounts;
    const auto csv = outputBase.parent_path() /
        (outputBase.stem().string() + "_consistency.csv");
    std::ofstream csvOutput(csv);
    if (!csvOutput)
        throw std::runtime_error("failed to open BP consistency CSV: " + csv.string());
    csvOutput << "piece,original_trace,source_piece,begin_arc_base,end_arc_base,reference,";
    if (sumProduct) {
        csvOutput << "bp_inference,bp_temperature,";
    }
    if (mixedState)
        csvOutput << "bp_mixed_cost_per_constraint,p_v,p_mixed,p_h,";
    csvOutput
        << "bp_status,vertical_threshold,"
           "horizontal_threshold,"
        << (mixedState ? "orientation_projection" : "horizontalness")
        << ",winding_component,winding_valid,winding_continuous,winding_relative_map,"
           "winding_output,"
           "winding_posterior_mean,winding_map_probability,winding_entropy,"
           "winding_candidate_min,winding_candidate_max,"
           "winding_signed_incident,winding_skipped_incident,"
           "degree,incident_measurements,"
           "total_strength,"
           "resolved_degree,resolved_strength,unresolved_degree,"
           "unresolved_strength,hard_mismatches,hard_mismatch_rate,"
           "weighted_hard_mismatch_rate,soft_mismatch_proxy,"
           "neighbor_support_balance,neighbor_certainty";
    if (interleaved) {
        csvOutput << ",winding_latent_mean,winding_phase,winding_scale,"
                     "winding_defect_cost_per_constraint,"
                     "winding_piece_break_cost,"
                     "winding_component_phase_sign,winding_solver,"
                     "winding_orientation_mode,winding_prepass_class,"
                     "winding_final_class,winding_final_p_h,"
                     "winding_final_p_defect,winding_final_p_v,"
                     "winding_calibration_mode,"
                     "winding_phase_mean,winding_scale_mean,"
                     "winding_component_positive_sign_probability";
    }
    csvOutput << '\n' << std::setprecision(17);
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
                  << report.horizontalness[piece] << ','
                  << winding.componentByPiece[piece] << ','
                  << (winding.windingValid[piece] != 0 ? 1 : 0) << ',';
        if (winding.windingValid[piece] != 0) {
            csvOutput << winding.continuousWinding[piece] << ','
                      << winding.mapWinding[piece] << ','
                      << winding.mapWinding[piece] + windingOutputOffset << ','
                      << winding.posteriorMeanWinding[piece] << ',';
        } else {
            csvOutput << "NA,NA,NA,NA,";
        }
        csvOutput << winding.mapProbability[piece] << ','
                  << winding.entropy[piece] << ',';
        if (winding.windingValid[piece] != 0) {
            csvOutput << winding.candidateMinimum[piece] << ','
                      << winding.candidateMaximum[piece] << ',';
        } else {
            csvOutput << "NA,NA,";
        }
        csvOutput << winding.incidentSignedConstraints[piece] << ','
                  << winding.incidentSkippedConstraints[piece] << ','
                  << current.degree
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
        if (interleaved) {
            const std::size_t component = winding.componentByPiece[piece];
            csvOutput << ',';
            if (winding.windingValid[piece] != 0)
                csvOutput << interleaved->posteriorMeanLatentCoordinate[piece];
            else
                csvOutput << "NA";
            csvOutput << ',' << interleaved->phaseMagnitude
                      << ',' << interleaved->measurementScale
                      << ',' << interleaved->defectUnaryCost
                      << ',' << interleaved->pieceBreakCost
                      << ',' << interleaved->componentPhaseSign.at(
                             component)
                      << ',' << vc::fiber_tracer::fiberTraceWindingSolverName(
                             interleaved->solver)
                      << ',' << vc::fiber_tracer::
                             fiberTraceWindingOrientationModeName(
                                 interleaved->orientationMode)
                      << ',';
            if (interleaved->orientationMode == vc::fiber_tracer::
                    FiberTraceWindingOrientationMode::FixedPrepass) {
                csvOutput << vc::fiber_tracer::fiberTraceFixedOrientationName(
                    interleaved->fixedOrientationByPiece.at(piece));
            } else {
                csvOutput << "NA";
            }
            csvOutput << ',' << ternaryName(predictions[piece])
                      << ',' << interleaved->classAProbability.at(piece)
                      << ',' << interleaved->mixedProbability.at(piece)
                      << ',' << interleaved->classBProbability.at(piece)
                      << ',' << vc::fiber_tracer::
                             fiberTraceWindingCalibrationModeName(
                                 interleaved->calibrationMode)
                      << ',' << interleaved->calibrationPhaseMean
                      << ',' << interleaved->calibrationScaleMean << ',';
            if (component < interleaved->
                    componentPositivePhaseSignProbability.size()) {
                csvOutput << interleaved->
                    componentPositivePhaseSignProbability[component];
            } else {
                csvOutput << "NA";
            }
        }
        csvOutput << '\n';
    }
    if (!csvOutput)
        throw std::runtime_error("failed to write BP consistency CSV: " + csv.string());
    double meanWindingConfidence = 0.0;
    std::size_t validWindingCount = 0;
    for (std::size_t piece = 0; piece < lines.size(); ++piece) {
        if (winding.windingValid[piece] == 0)
            continue;
        meanWindingConfidence += winding.mapProbability[piece];
        ++validWindingCount;
    }
    if (validWindingCount != 0)
        meanWindingConfidence /= static_cast<double>(validWindingCount);
    std::cout << std::fixed << std::setprecision(6)
              << "fiber winding BP\n"
              << "status  pieces  variables  factors  components"
                 "  continuous_rms  continuous_s  temperature  expansion_rounds"
                 "  message_iterations  message_residual  workers  candidate_states"
                 "  relative_min_w  relative_max_w  output_min_w  output_max_w"
                 "  mean_map_probability  discrete_s\n"
              << winding.status << "  " << lines.size() << "  "
              << winding.variables << "  " << winding.factors << "  "
              << winding.connectedComponents << "  "
              << winding.continuousRootMeanSquareResidual << "  "
              << winding.continuousSolveSeconds << "  "
              << winding.temperature << "  " << winding.expansionRounds << "  "
              << winding.messageIterations << "  " << winding.messageResidual
              << "  " << winding.effectiveWorkers << "  "
              << winding.totalCandidateStates << "  "
              << relativeWindingMinimum.value_or(0) << "  "
              << relativeWindingMaximum.value_or(0)
              << "  0  "
              << relativeWindingMaximum.value_or(0) + windingOutputOffset << "  "
              << meanWindingConfidence << "  " << winding.discreteSolveSeconds
              << '\n'
              << "winding_factor_csv=" << factorCsv
              << " winding_obj_layers=" << windingPaths.size() << '\n';
    if (interleaved) {
        std::cout << "fiber winding calibration\n";
        if (interleaved->solver == vc::fiber_tracer::
                FiberTraceWindingSolver::JointGrid) {
            std::cout
                << "solver  orientation_mode  calibration_mode  defect_cost_per_constraint  piece_break_cost  phase_map  phase_mean  scale_map  scale_mean"
                   "  grid_cells  grid_shifts  entropy  lower_boundary"
                   "  upper_boundary  min_gain  max_gain  converged"
                   "  decoded_energy  hard_sign_projected_defects\n"
                << vc::fiber_tracer::fiberTraceWindingSolverName(
                       interleaved->solver)
                << "  " << vc::fiber_tracer::
                       fiberTraceWindingOrientationModeName(
                           interleaved->orientationMode)
                << "  " << vc::fiber_tracer::
                       fiberTraceWindingCalibrationModeName(
                           interleaved->calibrationMode)
                << "  " << interleaved->defectUnaryCost
                << "  " << interleaved->pieceBreakCost
                << "  " << interleaved->phaseMagnitude
                << "  " << interleaved->calibrationPhaseMean
                << "  " << interleaved->measurementScale
                << "  " << interleaved->calibrationScaleMean
                << "  " << interleaved->calibrationGridCells
                << "  " << interleaved->calibrationGridShifts
                << "  " << interleaved->calibrationEntropy
                << "  " << interleaved->lowerGainBoundaryProbability
                << "  " << interleaved->upperGainBoundaryProbability
                << "  " << interleaved->minimumCalibrationGain
                << "  " << interleaved->maximumCalibrationGain
                << "  "
                << (interleaved->calibrationConverged ? "true" : "false")
                << "  " << interleaved->decodedEnergy
                << "  " << interleaved->hardSignProjectedDefects << '\n';
        } else {
            std::cout
                << "solver  orientation_mode  defect_cost_per_constraint  piece_break_cost  phase  scale  calibration_iterations"
                   "  calibration_converged  initialization"
                   "  rank_deficient_updates  decoded_energy"
                   "  hard_sign_projected_defects\n"
                << vc::fiber_tracer::fiberTraceWindingSolverName(
                       interleaved->solver)
                << "  " << vc::fiber_tracer::
                       fiberTraceWindingOrientationModeName(
                           interleaved->orientationMode)
                << "  " << interleaved->defectUnaryCost
                << "  " << interleaved->pieceBreakCost
                << "  " << interleaved->phaseMagnitude << "  "
                << interleaved->measurementScale << "  "
                << interleaved->calibrationIterations << "  "
                << (interleaved->calibrationConverged ? "true" : "false")
                << "  " << interleaved->selectedInitialization << "  "
                << interleaved->rankDeficientUpdates << "  "
                << interleaved->decodedEnergy << "  "
                << interleaved->hardSignProjectedDefects << '\n';
        }
        const auto agreement = vc::fiber_tracer::
            summarizeFiberTraceConstraintAgreement(
                constraints, *interleaved);
        std::cout
            << "fiber winding constraint agreement\n"
            << std::left << std::setw(16) << "class"
            << std::right << std::setw(10) << "prepared"
            << std::setw(10) << "active"
            << std::setw(12) << "neutralized"
            << std::setw(12) << "infringed"
            << std::setw(12) << "infringed_%" << '\n';
        const auto row = [](std::string_view name,
                            const vc::fiber_tracer::
                                FiberTraceConstraintAgreementCounts& counts) {
            std::cout << std::left << std::setw(16) << name
                      << std::right << std::setw(10) << counts.prepared
                      << std::setw(10) << counts.evaluated
                      << std::setw(12) << counts.defectNeutralized
                      << std::setw(12) << counts.infringed;
            if (counts.evaluated == 0) {
                std::cout << std::setw(12) << "NA";
            } else {
                std::ostringstream percent;
                percent << std::fixed << std::setprecision(2)
                        << 100.0 * static_cast<double>(counts.infringed) /
                               static_cast<double>(counts.evaluated)
                        << '%';
                std::cout << std::setw(12) << percent.str();
            }
            std::cout << '\n';
        };
        for (std::size_t index = 0;
             index < agreement.classes.size();
             ++index) {
            row(
                vc::fiber_tracer::fiberTraceConstraintAgreementClassName(
                    static_cast<vc::fiber_tracer::
                        FiberTraceConstraintAgreementClass>(index)),
                agreement.classes[index]);
        }
        row("sum", agreement.total);
    }
    std::cout << std::fixed << std::setprecision(6);
    if (mixedState) {
        std::cout
            << "fiber direction Mixed-state sum-product BP\n"
            << "inference  temperature  mixed_cost_per_constraint  status  pieces  factors  measurements"
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
        std::vector<double> mixedReferences;
        std::vector<double> trustedReferences;
        for (std::size_t trace = 0; trace < lines.size(); ++trace) {
            const std::size_t prediction =
                static_cast<std::size_t>(predictions[trace]);
            ++confusion[static_cast<std::size_t>(directions[trace])][prediction];
            if (directions[trace] ==
                vc::fiber_tracer::FiberDirectionGroup::Mixed) {
                mixedReferences.push_back(report.mixedProbability[trace]);
            } else {
                trustedReferences.push_back(report.mixedProbability[trace]);
            }
        }
        std::array<std::size_t, 4> counts{};
        for (const auto& row : confusion) {
            for (std::size_t state = 0; state < counts.size(); ++state)
                counts[state] += row[state];
        }
        finalStateCounts = counts;
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
                        (output.stem().string() + "_error_probability")
                  << '\n';
        const auto statePaths = vc::fiber_tracer::
            writeFiberletCropTernaryStateObjs(
                lines,
                predictions,
                outputBase);
        std::cout << "fiber direction explicit state OBJ layers\n"
                  << "state  path\n"
                  << "v  " << statePaths.vertical << '\n'
                  << "err  " << statePaths.mixed << '\n'
                  << "h  " << statePaths.horizontal << '\n'
                  << "tie  " << statePaths.tie << '\n';
        if (prepassPaths) {
            std::cout << "fiber direction fixed prepass OBJ layers\n"
                      << "state  path\n"
                      << "v  " << prepassPaths->vertical << '\n'
                      << "err  " << prepassPaths->mixed << '\n'
                      << "h  " << prepassPaths->horizontal << '\n'
                      << "tie  " << prepassPaths->tie << '\n';
        }
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
    if (finalStateCounts) {
        const double denominator = static_cast<double>(lines.size());
        const auto printState = [&] (const char* state, std::size_t count) {
            std::cout << std::left << std::setw(8) << state << std::right
                      << std::setw(10) << count
                      << std::setw(12) << std::fixed << std::setprecision(4)
                      << static_cast<double>(count) / denominator << '\n';
        };
        std::cout << "fiber direction final states"
                  << " pieces=" << lines.size()
                  << " ties=" << (*finalStateCounts)[3] << '\n'
                  << std::left << std::setw(8) << "state" << std::right
                  << std::setw(10) << "count"
                  << std::setw(12) << "fraction" << '\n';
        printState("H", (*finalStateCounts)[2]);
        printState("V", (*finalStateCounts)[0]);
        printState("Mix", (*finalStateCounts)[1]);
    }
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

vc::fiber_tracer::FiberTraceInterleavedWindingProgressCallback
makeInterleavedWindingProgressPrinter()
{
    using Phase = vc::fiber_tracer::FiberTraceInterleavedWindingProgressPhase;
    return [lastPrinted = std::chrono::steady_clock::time_point{},
            lastPhase = Phase::Complete,
            completedInitializations = std::size_t{0},
            meanInitializationSeconds = 0.0,
            completedInitializationElapsedSeconds = 0.0,
            completedCalibrations = std::size_t{0},
            meanCalibrationSeconds = 0.0,
            completedCalibrationElapsedSeconds = 0.0](
               const vc::fiber_tracer::FiberTraceInterleavedWindingProgress& p)
               mutable {
        const auto now = std::chrono::steady_clock::now();
        const bool transition = p.phase != lastPhase;
        const bool intervalElapsed = lastPrinted.time_since_epoch().count() == 0 ||
            now - lastPrinted >= std::chrono::seconds(1);
        if (!transition && !intervalElapsed && p.phase == Phase::MessagePassing)
            return;

        if (p.phase == Phase::InitializationComplete) {
            completedInitializations = p.initialization;
            meanInitializationSeconds = p.elapsedSeconds /
                static_cast<double>(completedInitializations);
            completedInitializationElapsedSeconds = p.elapsedSeconds;
        } else if (p.phase == Phase::Calibration) {
            ++completedCalibrations;
            meanCalibrationSeconds = p.elapsedSeconds /
                static_cast<double>(completedCalibrations);
            completedCalibrationElapsedSeconds = p.elapsedSeconds;
        }

        std::ostringstream line;
        line << std::fixed << std::setprecision(1);
        switch (p.phase) {
        case Phase::Preparing:
            line << "fiber winding BP status=preparing";
            break;
        case Phase::MessagePassing:
            line << "fiber winding BP"
                 << " init=" << p.initialization << '/' << p.initializationCount
                 << " calibration=" << p.calibrationIteration << '/'
                 << p.maximumCalibrationIterations
                 << " support=" << p.adaptiveSupportRound
                 << " message=" << p.messageIteration << '/'
                 << p.maximumMessageIterations
                 << " total_messages=" << p.accumulatedMessageIterations
                 << " states=" << p.candidateStates
                 << std::scientific << std::setprecision(3)
                 << " residual=" << p.messageResidual
                 << std::fixed << std::setprecision(4)
                 << " phase=" << p.phaseMagnitude
                 << " scale=" << p.measurementScale;
            break;
        case Phase::Calibration:
            line << "fiber winding BP status=calibration_update"
                 << " init=" << p.initialization << '/' << p.initializationCount
                 << " calibration=" << p.calibrationIteration << '/'
                 << p.maximumCalibrationIterations
                 << " proposed_phase=" << std::setprecision(4)
                 << p.phaseMagnitude
                 << " proposed_scale=" << p.measurementScale;
            break;
        case Phase::InitializationComplete:
            line << "fiber winding BP status=initialization_complete"
                 << " init=" << p.initialization << '/' << p.initializationCount
                 << " calibrations=" << p.calibrationIteration
                 << " total_messages=" << p.accumulatedMessageIterations;
            break;
        case Phase::Complete:
            line << "fiber winding BP status=complete"
                 << " initializations=" << p.initializationCount
                 << " total_messages=" << p.accumulatedMessageIterations
                 << " states=" << p.candidateStates
                 << " phase=" << std::setprecision(4) << p.phaseMagnitude
                 << " scale=" << p.measurementScale;
            break;
        }
        line << std::fixed << std::setprecision(1)
             << " elapsed=" << p.elapsedSeconds << 's';
        if (completedInitializations > 0 && p.phase != Phase::Complete) {
            const double currentInitializationSeconds =
                p.elapsedSeconds - completedInitializationElapsedSeconds;
            const double eta = std::max(
                0.0,
                meanInitializationSeconds * static_cast<double>(
                    p.initializationCount - completedInitializations) -
                    currentInitializationSeconds);
            line << " eta_est=" << eta << 's'
                 << " eta_basis=initialization";
        } else if (completedCalibrations > 0 && p.phase != Phase::Complete) {
            const std::size_t maximumCalibrations =
                p.initializationCount * p.maximumCalibrationIterations;
            const double currentCalibrationSeconds =
                p.elapsedSeconds - completedCalibrationElapsedSeconds;
            const double eta = std::max(
                0.0,
                meanCalibrationSeconds * static_cast<double>(
                    maximumCalibrations - completedCalibrations) -
                    currentCalibrationSeconds);
            line << " eta_est=" << eta << 's'
                 << " eta_basis=calibration_max";
        }
        std::cout << line.str() << '\n' << std::flush;
        lastPrinted = now;
        lastPhase = p.phase;
    };
}

vc::fiber_tracer::FiberTraceJointGridProgressCallback
makeJointGridWindingProgressPrinter()
{
    using Phase = vc::fiber_tracer::FiberTraceJointGridProgressPhase;
    return [lastPrinted = std::chrono::steady_clock::time_point{},
            lastPhase = Phase::Complete](
               const vc::fiber_tracer::FiberTraceJointGridProgress& p) mutable {
        const auto now = std::chrono::steady_clock::now();
        const bool transition = p.phase != lastPhase;
        const bool intervalElapsed = lastPrinted.time_since_epoch().count() == 0 ||
            now - lastPrinted >= std::chrono::seconds(1);
        if (!transition && !intervalElapsed && p.phase == Phase::MessagePassing)
            return;
        std::ostringstream line;
        line << "fiber winding joint-grid calibration="
             << vc::fiber_tracer::fiberTraceWindingCalibrationModeName(
                    p.calibrationMode);
        switch (p.phase) {
        case Phase::Preparing:
            line << " status=preparing";
            break;
        case Phase::MessagePassing:
            line << " message=" << p.messageIteration << '/'
                 << p.maximumMessageIterations;
            break;
        case Phase::SupportChanged:
            line << " status=support_changed"
                 << " message=" << p.messageIteration << '/'
                 << p.maximumMessageIterations;
            break;
        case Phase::Complete:
            line << " status=complete"
                 << " messages=" << p.messageIteration;
            break;
        }
        line << " states=" << p.candidateStates
             << " grid=" << p.gainCells << 'x' << p.phaseCells
             << " shifts=" << p.gridShifts
             << std::scientific << std::setprecision(3)
             << " residual=" << p.messageResidual
             << " calibration_residual=" << p.calibrationPosteriorResidual
             << std::fixed << std::setprecision(4)
             << " phase_map=" << p.phaseMap
             << " phase_mean=" << p.phaseMean
             << " scale_map=" << p.scaleMap
             << " scale_mean=" << p.scaleMean
             << " boundary=" << p.lowerGainBoundaryProbability << ','
             << p.upperGainBoundaryProbability
             << " gain=" << p.minimumGain << ':' << p.maximumGain
             << std::setprecision(1) << " elapsed=" << p.elapsedSeconds << 's';
        std::cout << line.str() << '\n' << std::flush;
        lastPrinted = now;
        lastPhase = p.phase;
    };
}

std::vector<std::size_t> applyQualityFilter(
    std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const std::optional<double>& fraction)
{
    std::vector<std::size_t> originalTraceIndices(lines.size());
    std::iota(
        originalTraceIndices.begin(), originalTraceIndices.end(),
        std::size_t{0});
    if (!fraction)
        return originalTraceIndices;

    const auto selection = vc::fiber_tracer::selectFiberletCropQuality(
        lines, *fraction);
    std::vector<vc::fiber_tracer::FiberletCropTraceLine> retained;
    retained.reserve(selection.lineIndices.size());
    for (const std::size_t index : selection.lineIndices)
        retained.push_back(std::move(lines.at(index)));
    lines = std::move(retained);
    originalTraceIndices = selection.lineIndices;

    std::cout << std::fixed << std::setprecision(6)
              << "fiber input quality filter"
              << " requested_fraction=" << selection.requestedFraction
              << " input=" << selection.inputLines
              << " retained=" << selection.lineIndices.size()
              << " effective_fraction=" << selection.effectiveFraction
              << " cutoff_cost_density=";
    if (selection.maximumRetainedCostDensity)
        std::cout << *selection.maximumRetainedCostDensity;
    else
        std::cout << "n/a";
    std::cout << '\n' << std::defaultfloat;
    return originalTraceIndices;
}

std::filesystem::path referenceFiberObjPath(
    const std::filesystem::path& output)
{
    return output.parent_path() /
        (output.stem().string() + "_reference.obj");
}

void removeReferenceFiberArtifact(const std::filesystem::path& path)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw std::runtime_error(
            "failed to inspect reference fiber artifact: " + error.message());
    }
    if (!exists)
        return;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error(
            "reference fiber artifact path is not a regular file: " +
            path.string());
    }
    if (!std::filesystem::remove(path, error) || error) {
        throw std::runtime_error(
            "failed to remove stale reference fiber artifact: " +
            path.string());
    }
}

struct ReferenceFiberDiagnostics {
    std::vector<vc::fiber_tracer::FiberletCropTraceLine> lines;
    std::vector<std::size_t> sourceIds;
    std::vector<std::string> sourceNames;
    std::vector<vc::fiber_tracer::FiberletCropTraceLine> pieceLines;
    std::vector<std::size_t> sourceIdsByPiece;
    std::optional<vc::fiber_tracer::FiberTraceConstraintReport>
        constraintReport;
    std::optional<vc::fiber_tracer::FiberTraceBeliefTopology>
        windingTopology;
    std::size_t sourceFibers = 0;
};

std::optional<ReferenceFiberDiagnostics> updateReferenceFiberArtifact(
    const Options& options,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ)
{
    const auto outputPath = referenceFiberObjPath(options.output);
    removeReferenceFiberArtifact(outputPath);
    if (!options.hasReferenceFiberDirectoryOption)
        return std::nullopt;

    const auto selection = vc::fiber_tracer::loadTaggedVc3dFiberJsonDirectory(
        options.referenceFiberDirectory, options.referenceFiberTag);
    if (selection.fibers.empty()) {
        throw std::runtime_error(
            "no VC3D fibers matched reference tag '" +
            options.referenceFiberTag + "' in " +
            options.referenceFiberDirectory.string());
    }

    std::vector<vc::core::io::NamedPolyline> objLines;
    ReferenceFiberDiagnostics diagnostics;
    diagnostics.sourceFibers = selection.fibers.size();
    diagnostics.sourceNames.reserve(selection.fibers.size());
    std::size_t retainedPoints = 0;
    for (std::size_t index = 0; index < selection.fibers.size(); ++index) {
        const auto& selected = selection.fibers[index];
        diagnostics.sourceNames.push_back(selected.path.stem().string());
        auto runs = vc::fiber_tracer::clipPolylineToHalfOpenBox(
            selected.fiber.linePoints, minimumBaseXYZ, maximumBaseXYZ);
        for (std::size_t run = 0; run < runs.size(); ++run) {
            const std::string name =
                "reference_" + std::to_string(index) + "_run_" +
                std::to_string(run) + "_" + selected.path.stem().string();
            retainedPoints += runs[run].size();
            objLines.push_back({name, runs[run]});
            vc::fiber_tracer::FiberletCropTraceLine diagnostic;
            diagnostic.seedBaseXYZ = runs[run].front();
            diagnostic.pointsBaseXYZ = std::move(runs[run]);
            diagnostics.lines.push_back(std::move(diagnostic));
            diagnostics.sourceIds.push_back(index);
        }
    }
    if (objLines.empty()) {
        throw std::runtime_error(
            "tagged VC3D reference fibers do not intersect the trace crop");
    }
    vc::core::io::writePolylinesObj(
        objLines, outputPath, "VC3D tagged reference fibers");
    std::cout << "fiber reference export"
              << " scanned_json=" << selection.scannedJsonFiles
              << " selected=" << selection.fibers.size()
              << " retained_runs=" << objLines.size()
              << " retained_points=" << retainedPoints
              << " tag=" << std::quoted(options.referenceFiberTag)
              << " directory=" << std::quoted(
                     options.referenceFiberDirectory.string())
              << " output=" << std::quoted(outputPath.string()) << '\n';
    return diagnostics;
}

std::string formatReferenceFiberConstraints(
    const ReferenceFiberDiagnostics& reference,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::fiber_tracer::FiberTraceReferenceWindingBenchmark& calibration,
    const vc::fiber_tracer::FiberTraceWindingBeliefPropagationConfig& config,
    double measurementScale)
{
    if (reference.sourceNames.size() != reference.sourceFibers) {
        throw std::invalid_argument(
            "Reference fiber names do not match selected sources");
    }

    const auto diagnostics = vc::fiber_tracer::
        makeFiberTraceReferenceConstraintDiagnosticReport(
            report, reference.sourceIds, calibration);
    if (!reference.windingTopology) {
        throw std::logic_error(
            "Reference constraint scale calibration has no BP topology");
    }
    const auto factorDiagnostics = vc::fiber_tracer::
        diagnoseFiberTraceWindingFactors(
            report,
            *reference.windingTopology,
            config,
            {},
            true,
            measurementScale);
    const auto scaleCalibration = vc::fiber_tracer::
        calibrateFiberTraceReferenceConstraintScales(
            diagnostics, factorDiagnostics);
    const auto phaseCalibration = vc::fiber_tracer::
        calibrateFiberTraceReferenceConstraintPhase(
            diagnostics, measurementScale);
    const auto stepStatistics = vc::fiber_tracer::
        summarizeFiberTraceReferenceConstraintSteps(diagnostics);
    using Row = vc::fiber_tracer::FiberTraceReferenceConstraintDiagnosticRow;
    std::vector<std::vector<const Row*>> perpendicular(
        reference.sourceFibers);
    std::vector<std::vector<const Row*>> parallel(reference.sourceFibers);
    for (const auto& row : diagnostics.rows) {
        if (row.targetSource >= reference.sourceFibers) {
            throw std::invalid_argument(
                "Reference constraint source is out of range");
        }
        auto& group = row.perpendicularDominant ? perpendicular : parallel;
        group[row.ownerSource].push_back(&row);
    }

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "reference constraint calibration global_sign="
           << calibration.globalSign << '\n';
    const bool evenReferenceIsHorizontal =
        !phaseCalibration.selectedGauge ||
        phaseCalibration.gauges[*phaseCalibration.selectedGauge]
            .evenReferenceIsHorizontal;
    const auto orientationName = [&](std::size_t parity) {
        const bool horizontal = parity == 0
            ? evenReferenceIsHorizontal
            : !evenReferenceIsHorizontal;
        return horizontal ? "H" : "V";
    };
    output << "reference raw signed step distributions"
           << " mapping=even:" << orientationName(0)
           << ",odd:" << orientationName(1)
           << " values=source-oriented-unweighted\n"
           << std::left
           << std::setw(10) << "class"
           << std::setw(8) << "from_to"
           << std::setw(9) << "gt_band"
           << std::right
           << std::setw(7) << "n"
           << std::setw(11) << "raw_min"
           << std::setw(11) << "raw_mean"
           << std::setw(11) << "raw_med"
           << std::setw(11) << "raw_max" << '\n';
    for (std::size_t relation = 0; relation < 2; ++relation) {
        for (std::size_t owner = 0; owner < 2; ++owner) {
            for (std::size_t target = 0; target < 2; ++target) {
                const bool oppositeParity = owner != target;
                const std::array<const char*, 3> bands = oppositeParity
                    ? std::array<const char*, 3>{"0.5", "1.5", "2.5+"}
                    : std::array<const char*, 3>{"1", "2", "3+"};
                for (std::size_t band = 0; band < 3; ++band) {
                    const auto& stats =
                        stepStatistics.groups[relation][owner][target][band];
                    if (stats.observations == 0)
                        continue;
                    const std::string transition =
                        std::string(orientationName(owner)) + "->" +
                        orientationName(target);
                    output << std::left
                           << std::setw(10)
                           << (relation == 0 ? "perp" : "parallel")
                           << std::setw(8) << transition
                           << std::setw(9) << bands[band]
                           << std::right
                           << std::setw(7) << stats.observations
                           << std::fixed << std::setprecision(3)
                           << std::setw(11) << stats.minimum
                           << std::setw(11) << stats.mean
                           << std::setw(11) << stats.median
                           << std::setw(11) << stats.maximum << '\n';
                }
            }
        }
    }
    output << '\n';
    output << "reference constraint phase calibration"
           << " objective=sum(abs(predicted/scale-raw_signed))"
           << " scale=" << std::fixed << std::setprecision(3)
           << phaseCalibration.measurementScale
           << " sign_penalties=excluded\n"
           << std::left
           << std::setw(5) << "dir"
           << std::setw(8) << "even"
           << std::right
           << std::setw(7) << "total"
           << std::setw(7) << "ident"
           << std::setw(7) << "used"
           << std::setw(8) << "p_same"
           << std::setw(8) << "q_same"
           << std::setw(8) << "q_opp"
           << std::setw(11) << "used_w"
           << std::setw(12) << "loss_p0"
           << std::setw(12) << "loss_p05"
           << std::setw(10) << "fit_p"
           << std::setw(12) << "fit_loss"
           << std::setw(10) << "reduce_%"
           << std::setw(10) << "sign_bad"
           << "  status\n";
    for (std::size_t index = 0;
         index < phaseCalibration.gauges.size();
         ++index) {
        const auto& fit = phaseCalibration.gauges[index];
        output << std::left
               << std::setw(5) << fit.windingDirection
               << std::setw(8)
               << (fit.evenReferenceIsHorizontal ? "H" : "V")
               << std::right
               << std::setw(7) << fit.totalRows
               << std::setw(7) << fit.identifyingRows
               << std::setw(7) << fit.usedRows
               << std::setw(8) << fit.perpendicularSameParityRows
               << std::setw(8) << fit.parallelSameParityRows
               << std::setw(8) << fit.parallelOppositeParityRows
               << std::fixed << std::setprecision(3)
               << std::setw(11) << fit.effectiveWeight
               << std::setw(12) << fit.lossAtZero
               << std::setw(12) << fit.lossAtHalf;
        if (fit.fittedPhase) {
            output << std::setw(10) << *fit.fittedPhase
                   << std::setw(12) << fit.fittedLoss;
            if (fit.lossAtHalf > 0.0) {
                output << std::setw(10)
                       << 100.0 *
                              (fit.lossAtHalf - fit.fittedLoss) /
                              fit.lossAtHalf;
            } else {
                output << std::setw(10) << "NA";
            }
            output << std::setw(10) << fit.fittedSignDisagreements;
        } else {
            output << std::setw(10) << "NA"
                   << std::setw(12) << "NA"
                   << std::setw(10) << "NA"
                   << std::setw(10) << "NA";
        }
        output << "  "
               << (phaseCalibration.selectedGauge &&
                           *phaseCalibration.selectedGauge == index
                       ? "selected"
                       : fit.fittedPhase ? "candidate" : "unidentifiable")
               << '\n';
    }
    output << "p_same=perpendicular same-parity (phase-independent);"
              " q_same=parallel same-parity; q_opp=parallel opposite-parity"
              " model contradictions\n\n";
    output << "reference constraint measurement-scale calibration"
           << " objective=sum(w*abs(gt/scale-target))"
           << " range=" << std::fixed << std::setprecision(2)
           << scaleCalibration.minimumScale << ':'
           << scaleCalibration.maximumScale << '\n'
           << "raw targets diagnose continuous measurement bias; canonical"
              " perpendicular_all is solver-compatible; parallel rows are"
              " counterfactual because current solver scale does not affect"
              " parallel integer targets\n"
           << std::left
           << std::setw(11) << "target"
           << std::setw(19) << "scope"
           << std::setw(15) << "scale_use"
           << std::right
           << std::setw(7) << "n"
           << std::setw(7) << "used"
           << std::setw(7) << "fit_n"
           << std::setw(11) << "sum_w"
           << std::setw(11) << "fit_w"
           << std::setw(12) << "loss_s1"
           << std::setw(11) << "fit_scale"
           << std::setw(12) << "fit_loss"
           << std::setw(11) << "reduce_%"
           << std::setw(9) << "bound" << '\n';
    const auto printScale = [&](
                                std::string_view target,
                                std::string_view scope,
                                std::string_view use,
                                const vc::fiber_tracer::
                                    FiberTraceReferenceScaleFit& fit) {
        output << std::left
               << std::setw(11) << target
               << std::setw(19) << scope
               << std::setw(15) << use
               << std::right
               << std::setw(7) << fit.observations
               << std::setw(7) << fit.admittedObservations
               << std::setw(7) << fit.informativeObservations
               << std::fixed << std::setprecision(3)
               << std::setw(11) << fit.effectiveWeight
               << std::setw(11) << fit.reciprocalScaleWeight
               << std::setw(12) << fit.unitScaleLoss;
        if (fit.fittedScale) {
            output << std::setw(11) << *fit.fittedScale
                   << std::setw(12) << fit.fittedLoss;
            if (fit.unitScaleLoss > 0.0) {
                output << std::setw(11)
                       << 100.0 *
                              (fit.unitScaleLoss - fit.fittedLoss) /
                              fit.unitScaleLoss;
            } else {
                output << std::setw(11) << "NA";
            }
            output << std::setw(9)
                   << (fit.atLowerBound ? "lower"
                       : fit.atUpperBound ? "upper"
                                          : "-");
        } else {
            output << std::setw(11) << "NA"
                   << std::setw(12) << "NA"
                   << std::setw(11) << "NA"
                   << std::setw(9) << "NA";
        }
        output << '\n';
    };
    printScale(
        "raw",
        "perpendicular_all",
        "measurement",
        scaleCalibration.rawPerpendicular);
    printScale(
        "canonical",
        "perpendicular_all",
        "solver",
        scaleCalibration.canonicalPerpendicular);
    printScale(
        "raw",
        "parallel_all",
        "counterfactual",
        scaleCalibration.rawParallel);
    printScale(
        "canonical",
        "parallel_all",
        "counterfactual",
        scaleCalibration.canonicalParallel);
    printScale(
        "raw",
        "all_constraints",
        "counterfactual",
        scaleCalibration.rawAll);
    printScale(
        "canonical",
        "all_constraints",
        "counterfactual",
        scaleCalibration.canonicalAll);
    constexpr std::size_t scaleGroupCount = static_cast<std::size_t>(
        vc::fiber_tracer::FiberTraceReferenceConstraintGroup::Count);
    for (std::size_t group = 0; group < scaleGroupCount; ++group) {
        const auto kind = static_cast<vc::fiber_tracer::
            FiberTraceReferenceConstraintGroup>(group);
        const bool parallel =
            kind == vc::fiber_tracer::
                        FiberTraceReferenceConstraintGroup::ParallelSame ||
            kind == vc::fiber_tracer::
                        FiberTraceReferenceConstraintGroup::ParallelOne ||
            kind == vc::fiber_tracer::
                        FiberTraceReferenceConstraintGroup::ParallelTwoPlus;
        printScale(
            "raw",
            vc::fiber_tracer::fiberTraceReferenceConstraintGroupName(kind),
            parallel ? "counterfactual" : "measurement",
            scaleCalibration.rawGroups[group]);
        printScale(
            "canonical",
            vc::fiber_tracer::fiberTraceReferenceConstraintGroupName(kind),
            parallel ? "counterfactual" : "class_diag",
            scaleCalibration.canonicalGroups[group]);
    }
    output << "scale<1 means selected targets exceed known latent separation;"
              " scale>1 means selected targets are smaller\n\n";
    const auto printTable = [&output](
                                std::string_view title,
                                const std::vector<const Row*>& rows) {
        output << title << '\n'
               << std::left
               << std::setw(16) << "target_winding"
               << std::setw(14) << "raw_step"
               << std::setw(18) << "calibrated_step"
               << std::setw(17) << "canonical_step"
               << std::setw(12) << "gt_step"
               << "calibrated_minus_gt" << '\n';
        if (rows.empty()) {
            output << "(none)\n";
            return;
        }
        for (const Row* row : rows) {
            const double targetWinding =
                0.5 * static_cast<double>(row->targetSource);
            output << std::fixed
                   << std::setprecision(1)
                   << std::setw(16) << targetWinding
                   << std::setprecision(3)
                   << std::setw(14) << row->rawStep
                   << std::setw(18) << row->calibratedStep
                   << std::setprecision(1)
                   << std::setw(17) << row->canonicalStep
                   << std::setw(12) << row->groundTruthStep
                   << std::setprecision(3)
                   << row->calibratedStep - row->groundTruthStep
                   << '\n';
        }
    };

    for (std::size_t source = 0; source < reference.sourceFibers; ++source) {
        output << '\n'
               << "reference fiber " << std::quoted(
                      reference.sourceNames[source])
               << " winding=" << std::fixed << std::setprecision(1)
               << 0.5 * static_cast<double>(source) << '\n';
        printTable("perpendicular constraints", perpendicular[source]);
        printTable("parallel constraints", parallel[source]);
    }
    output << '\n'
           << "reference constraint canonical summary\n"
           << std::left
           << std::setw(12) << "correct"
           << std::setw(12) << "false"
           << "total\n"
           << std::setw(12) << diagnostics.counts.correct
           << std::setw(12) << diagnostics.counts.falseCount
           << diagnostics.counts.total << '\n';
    return output.str();
}

void prepareReferenceFiberPieces(
    ReferenceFiberDiagnostics& reference,
    const vc::fiber_tracer::FiberTraceConstraintReport& report)
{
    if (report.inputTraces != reference.lines.size() ||
        reference.sourceIds.size() != reference.lines.size()) {
        throw std::invalid_argument(
            "Reference constraint report does not match cropped runs");
    }
    reference.pieceLines = vc::fiber_tracer::makeFiberTraceConstraintPieceLines(reference.lines, report);
    reference.sourceIdsByPiece.clear();
    reference.sourceIdsByPiece.reserve(report.pieces.size());
    for (const auto& piece : report.pieces) {
        if (piece.traceIndex >= reference.sourceIds.size()) {
            throw std::invalid_argument("Reference constraint piece source is out of range");
        }
        reference.sourceIdsByPiece.push_back(reference.sourceIds[piece.traceIndex]);
    }
}

struct ReferenceBpCrossConstraints {
    vc::fiber_tracer::FiberTraceConstraintReport report;
    std::size_t referencePieces = 0;
};

using WindingWeightTuple = std::array<double, 7>;

void setWindingClassWeights(
    vc::fiber_tracer::FiberTraceWindingBeliefPropagationConfig& config,
    const std::array<double, 5>& weights)
{
    config.perpendicularNextWeight = weights[0];
    config.perpendicularFarWeight = weights[1];
    config.parallelSameWeight = weights[2];
    config.parallelOneWeight = weights[3];
    config.parallelFarWeight = weights[4];
}

void setWindingWeights(
    vc::fiber_tracer::FiberTraceWindingBeliefPropagationConfig& config,
    const WindingWeightTuple& weights)
{
    setWindingClassWeights(
        config,
        {weights[0], weights[1], weights[2], weights[3], weights[4]});
    config.perpendicularSignWeight = weights[5];
    config.parallelSignWeight = weights[6];
}

WindingWeightTuple windingWeights(const Options& options)
{
    return {
        options.windingWeights[0],
        options.windingWeights[1],
        options.windingWeights[2],
        options.windingWeights[3],
        options.windingWeights[4],
        options.windingSignWeights[0],
        options.windingSignWeights[1],
    };
}

ReferenceBpCrossConstraints extractReferenceBpCrossConstraints(
    const ReferenceFiberDiagnostics& reference,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& bpPieceLines,
    const Options& options,
    const vc::lasagna::LasagnaNormalSampler& normals,
    const vc::fiber_tracer::LasagnaNormalAlignmentField& alignedNormals)
{
    if (reference.pieceLines.size() != reference.sourceIdsByPiece.size()) {
        throw std::invalid_argument("Reference diagnostic pieces do not match their source IDs");
    }
    std::vector<vc::fiber_tracer::FiberletCropTraceLine> combined;
    combined.reserve(reference.pieceLines.size() + bpPieceLines.size());
    combined.insert(combined.end(), reference.pieceLines.begin(), reference.pieceLines.end());
    combined.insert(combined.end(), bpPieceLines.begin(), bpPieceLines.end());

    auto config = options.constraints;
    config.preserveInputLinesAsPieces = true;
    const std::size_t referencePieces = reference.pieceLines.size();
    auto report = vc::fiber_tracer::extractFiberTraceConstraints(
        combined,
        config,
        [&normals](const cv::Vec3d& a, const cv::Vec3d& b, double step) {
            return normals.normalAlignedWindingDistance(a, b, step);
        },
        [&normals](
            const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
            double step,
            int threads) {
            return normals.normalAlignedWindingDistancesBatch(
                connectors, step, threads);
        },
        [referencePieces](std::size_t a, std::size_t b) {
            return (a < referencePieces) != (b < referencePieces);
        },
        &alignedNormals);
    if (report.pieces.size() != combined.size()) {
        throw std::logic_error("Reference/BP cross extraction did not preserve input pieces");
    }
    return {std::move(report), referencePieces};
}

std::vector<vc::fiber_tracer::FiberTraceReferenceWindingObservation>
makeReferenceBpWindingObservations(
    const ReferenceFiberDiagnostics& reference,
    const ReferenceBpCrossConstraints& cross,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport& winding,
    const vc::fiber_tracer::FiberTraceWindingBeliefPropagationConfig& config)
{
    if (cross.report.inputTraces < cross.referencePieces) {
        throw std::invalid_argument(
            "Reference/BP cross report has fewer traces than references");
    }
    const std::size_t bpPieces =
        cross.report.inputTraces - cross.referencePieces;
    if (reference.sourceIdsByPiece.size() != cross.referencePieces ||
        winding.windingValid.size() != bpPieces ||
        winding.mapLatentCoordinate.size() != bpPieces ||
        winding.mapOrientationByPiece.size() != bpPieces ||
        winding.integerGaugeByPiece.size() != bpPieces) {
        throw std::invalid_argument(
            "Reference/BP benchmark inputs do not match represented pieces");
    }

    std::vector<vc::fiber_tracer::FiberTraceReferenceWindingObservation>
        observations;
    observations.reserve(cross.report.constraints.size());
    for (const auto& constraint : cross.report.constraints) {
        if (constraint.hardContinuity)
            continue;
        if (constraint.pieceA >= cross.report.pieces.size() ||
            constraint.pieceB >= cross.report.pieces.size()) {
            throw std::invalid_argument(
                "Reference/BP constraint references an invalid piece");
        }
        const std::size_t traceA =
            cross.report.pieces[constraint.pieceA].traceIndex;
        const std::size_t traceB =
            cross.report.pieces[constraint.pieceB].traceIndex;
        const bool referenceIsA = traceA < cross.referencePieces;
        const bool referenceIsB = traceB < cross.referencePieces;
        if (referenceIsA == referenceIsB) {
            throw std::logic_error(
                "Reference/BP cross report contains a non-cross constraint");
        }
        const std::size_t referencePiece = referenceIsA ? traceA : traceB;
        const std::size_t bpPiece =
            (referenceIsA ? traceB : traceA) - cross.referencePieces;
        const std::size_t referenceSource =
            reference.sourceIdsByPiece[referencePiece];
        if (referenceSource >= reference.sourceNames.size()) {
            throw std::logic_error(
                "Reference/BP constraint has an invalid reference source");
        }
        auto observation = vc::fiber_tracer::
            makeFiberTraceReferenceWindingObservation(
                constraint,
                referenceIsA,
                0.5 * static_cast<double>(referenceSource),
                bpPiece,
                winding,
                config);
        observation.referenceSource = referenceSource;
        observations.push_back(std::move(observation));
    }
    return observations;
}

struct WindingWeightSearchScore {
    std::size_t exactReferences = 0;
    std::size_t wrongReferences = 0;
    std::size_t missingReferences = 0;
    std::size_t rightConstraints = 0;
    std::size_t wrongConstraints = 0;
    bool converged = false;
    double residual = std::numeric_limits<double>::infinity();
};

WindingWeightSearchScore scoreWindingWeightSearch(
    const ReferenceFiberDiagnostics& reference,
    const vc::fiber_tracer::FiberTraceReferenceWindingBenchmark& benchmark,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport& winding)
{
    constexpr double epsilon = 1.0e-12;
    WindingWeightSearchScore result;
    for (std::size_t source = 0; source < reference.sourceNames.size();
         ++source) {
        const auto estimated = source < benchmark.references.size()
            ? benchmark.references[source].estimatedWinding
            : std::nullopt;
        if (!estimated) {
            ++result.missingReferences;
            continue;
        }
        const double truth = 0.5 * static_cast<double>(source);
        if (std::abs(*estimated - truth) <= epsilon)
            ++result.exactReferences;
        else
            ++result.wrongReferences;
    }
    result.rightConstraints = benchmark.sum.right;
    result.wrongConstraints = benchmark.sum.wrong;
    result.converged = winding.messageConverged;
    result.residual = winding.messageResidual;
    return result;
}

bool betterWindingWeightSearchResult(
    const WindingWeightSearchScore& candidate,
    const WindingWeightTuple& candidateWeights,
    const WindingWeightSearchScore& current,
    const WindingWeightTuple& currentWeights)
{
    if (candidate.converged != current.converged)
        return candidate.converged;
    if (candidate.exactReferences != current.exactReferences)
        return candidate.exactReferences > current.exactReferences;
    if (candidate.missingReferences != current.missingReferences)
        return candidate.missingReferences < current.missingReferences;
    if (candidate.wrongReferences != current.wrongReferences)
        return candidate.wrongReferences < current.wrongReferences;
    if (candidate.rightConstraints != current.rightConstraints)
        return candidate.rightConstraints > current.rightConstraints;
    const std::size_t candidateTotal =
        candidate.rightConstraints + candidate.wrongConstraints;
    const std::size_t currentTotal =
        current.rightConstraints + current.wrongConstraints;
    if (candidateTotal != currentTotal)
        return candidateTotal > currentTotal;
    if (candidate.wrongConstraints != current.wrongConstraints)
        return candidate.wrongConstraints < current.wrongConstraints;
    if (candidate.residual != current.residual)
        return candidate.residual < current.residual;
    return candidateWeights < currentWeights;
}

bool strictlyBetterWindingWeightSearchQuality(
    const WindingWeightSearchScore& candidate,
    const WindingWeightSearchScore& current)
{
    if (candidate.converged != current.converged)
        return candidate.converged;
    if (candidate.exactReferences != current.exactReferences)
        return candidate.exactReferences > current.exactReferences;
    if (candidate.missingReferences != current.missingReferences)
        return candidate.missingReferences < current.missingReferences;
    if (candidate.wrongReferences != current.wrongReferences)
        return candidate.wrongReferences < current.wrongReferences;
    if (candidate.rightConstraints != current.rightConstraints)
        return candidate.rightConstraints > current.rightConstraints;
    const std::size_t candidateTotal =
        candidate.rightConstraints + candidate.wrongConstraints;
    const std::size_t currentTotal =
        current.rightConstraints + current.wrongConstraints;
    if (candidateTotal != currentTotal)
        return candidateTotal > currentTotal;
    if (candidate.wrongConstraints != current.wrongConstraints)
        return candidate.wrongConstraints < current.wrongConstraints;
    return false;
}

std::string formatWindingWeights(const WindingWeightTuple& weights)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(6);
    output << "winding=";
    for (std::size_t index = 0; index < 5; ++index) {
        if (index != 0)
            output << ',';
        output << weights[index];
    }
    output << " sign=" << weights[5] << ',' << weights[6];
    return output.str();
}

std::string formatReferenceBpWindingBenchmark(
    const ReferenceFiberDiagnostics& reference,
    const ReferenceBpCrossConstraints& cross,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport& winding,
    vc::fiber_tracer::FiberTraceBalanceMode balanceMode,
    const vc::fiber_tracer::FiberTraceWindingBeliefPropagationConfig& config)
{
    const auto observations = makeReferenceBpWindingObservations(
        reference, cross, winding, config);

    const auto benchmark = vc::fiber_tracer::calibrateFiberTraceReferenceWindings(observations);
    const auto orientationBenchmark = vc::fiber_tracer::
        benchmarkFiberTraceReferenceOrientations(observations);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    if (!reference.constraintReport) {
        throw std::logic_error(
            "Reference/BP benchmark has no reference constraint report");
    }
    output << formatReferenceFiberConstraints(
                  reference,
                  *reference.constraintReport,
                  benchmark,
                  config,
                  winding.measurementScale)
           << '\n'
           << "reference-to-BP winding benchmark"
           << " balance=" << vc::fiber_tracer::fiberTraceBalanceModeName(balanceMode)
           << " solver="
           << vc::fiber_tracer::fiberTraceWindingSolverName(winding.solver)
           << " status=" << winding.status
           << " tolerance=" << std::fixed << std::setprecision(3) << benchmark.tolerance
           << " class_weights="
           << config.perpendicularNextWeight << ','
           << config.perpendicularFarWeight << ','
           << config.parallelSameWeight << ','
           << config.parallelOneWeight << ','
           << config.parallelFarWeight
           << " sign_weights="
           << config.perpendicularSignWeight << ','
           << config.parallelSignWeight << '\n'
           << "global sign=" << benchmark.globalSign << '\n'
           << "gauge calibration\n"
           << std::left << std::setw(12) << "gauge"
           << std::setw(14) << "offset"
           << std::setw(16) << "exact_matches"
           << "estimate_votes\n";
    for (const auto& gauge : benchmark.gauges) {
        output << std::setw(12) << gauge.integerGauge
               << std::setw(14) << gauge.offset
               << std::setw(16) << gauge.exactMatches
               << gauge.estimateVotes << '\n';
    }
    const auto groupDiagnostics = vc::fiber_tracer::
        summarizeFiberTraceReferenceConstraintGroups(
            observations, benchmark);
    constexpr auto groupCount = static_cast<std::size_t>(
        vc::fiber_tracer::FiberTraceReferenceConstraintGroup::Count);
    output << "reference constraint groups BP winding energy"
           << " (dominant hypothesis; hard violations first)\n"
           << std::setw(8) << "winding"
           << std::setw(14) << "group"
           << std::setw(7) << "n"
           << std::setw(10) << "raw_w"
           << std::setw(10) << "used_w"
           << std::setw(9) << "true_h"
           << std::setw(11) << "true_L1"
           << std::setw(11) << "true_avg"
           << std::setw(10) << "infer_w"
           << std::setw(9) << "infer_h"
           << std::setw(11) << "infer_L1"
           << "infer_avg\n";
    for (std::size_t source = 0;
         source < reference.sourceNames.size();
         ++source) {
        for (std::size_t groupIndex = 0;
             groupIndex <= groupCount;
             ++groupIndex) {
            const vc::fiber_tracer::
                FiberTraceReferenceConstraintGroupDiagnostic empty;
            const auto& diagnostic = source >= groupDiagnostics.size()
                ? empty
                : groupIndex == groupCount
                ? groupDiagnostics[source].all
                : groupDiagnostics[source].groups[groupIndex];
            const char* groupName = groupIndex == groupCount
                ? "all"
                : vc::fiber_tracer::
                      fiberTraceReferenceConstraintGroupName(
                          static_cast<vc::fiber_tracer::
                              FiberTraceReferenceConstraintGroup>(
                                  groupIndex));
            output << std::setw(8) << std::setprecision(1)
                   << 0.5 * static_cast<double>(source)
                   << std::setw(14) << groupName
                   << std::setw(7) << diagnostic.observations
                   << std::setw(10) << std::setprecision(3)
                   << diagnostic.rawCoefficient
                   << std::setw(10) << diagnostic.admittedCoefficient
                   << std::setw(9) << diagnostic.truthHardViolations;
            if (!(diagnostic.admittedCoefficient > 0.0) ||
                !diagnostic.preferredWinding) {
                output << std::setw(11) << "NA"
                       << std::setw(11) << "NA"
                       << std::setw(10) << "NA"
                       << std::setw(9) << "NA"
                       << std::setw(11) << "NA"
                       << "NA\n";
                continue;
            }
            output << std::setw(11) << diagnostic.truthLoss
                   << std::setw(11)
                   << diagnostic.truthLoss /
                          diagnostic.admittedCoefficient
                   << std::setw(10) << std::setprecision(1)
                   << *diagnostic.preferredWinding
                   << std::setw(9)
                   << diagnostic.preferredHardViolations
                   << std::setw(11) << std::setprecision(3)
                   << diagnostic.preferredLoss
                   << diagnostic.preferredLoss /
                          diagnostic.admittedCoefficient
                   << '\n';
        }
    }
    const auto printOrientationCounts = [&output](
                                            const vc::fiber_tracer::
                                                FiberTraceReferenceOrientationCounts&
                                                    counts) {
        output << std::setw(8) << counts.right
               << std::setw(8) << counts.wrong;
        if (counts.total() == 0) {
            output << std::setw(8) << "NA";
        } else {
            output << std::setw(8) << std::setprecision(3)
                   << static_cast<double>(counts.right) /
                          static_cast<double>(counts.total());
        }
    };
    output << "reference H/V component calibration\n"
           << std::setw(12) << "component"
           << std::setw(14) << "even_ref"
           << std::setw(12) << "even_H_r"
           << "even_V_r\n";
    if (orientationBenchmark.components.empty()) {
        output << "(none)\n";
    } else {
        for (const auto& component : orientationBenchmark.components) {
            output << std::setw(12) << component.component
                   << std::setw(14)
                   << (component.evenReferenceIsHorizontal ? "H" : "V")
                   << std::setw(12) << component.evenHorizontalRight
                   << component.evenVerticalRight << '\n';
        }
    }
    output << "reference H/V endpoint consistency"
           << " excluded_inactive="
           << orientationBenchmark.excludedInactive << '\n'
           << std::setw(8) << "winding"
           << std::setw(8) << "perp_r"
           << std::setw(8) << "perp_w"
           << std::setw(8) << "perp_f"
           << std::setw(8) << "para_r"
           << std::setw(8) << "para_w"
           << std::setw(8) << "para_f"
           << std::setw(8) << "sum_r"
           << std::setw(8) << "sum_w"
           << "sum_f\n";
    constexpr std::size_t orientationRelationCount = static_cast<std::size_t>(
        vc::fiber_tracer::FiberTraceReferenceOrientationRelation::Count);
    for (std::size_t source = 0; source < reference.sourceNames.size(); ++source) {
        const vc::fiber_tracer::FiberTraceReferenceOrientationSourceBenchmark
            empty;
        const auto& current = source < orientationBenchmark.references.size()
            ? orientationBenchmark.references[source]
            : empty;
        output << std::setw(8) << std::setprecision(1)
               << 0.5 * static_cast<double>(source);
        for (std::size_t relation = 0;
             relation < orientationRelationCount;
             ++relation) {
            printOrientationCounts(current.relations[relation]);
        }
        printOrientationCounts(current.sum);
        output << '\n';
    }
    output << std::setw(8) << "sum";
    for (std::size_t relation = 0;
         relation < orientationRelationCount;
         ++relation) {
        printOrientationCounts(orientationBenchmark.relations[relation]);
    }
    printOrientationCounts(orientationBenchmark.sum);
    output << "\n\n";

    constexpr std::array<const char*, 6> names{
        "perp_winding",
        "perp_sign",
        "parallel_same_winding",
        "parallel_other_winding",
        "parallel_sign",
        "sum"};
    output << "reference fiber errors fraction=right/(right+wrong)\n"
           << std::setw(8) << "winding"
           << std::setw(8) << "est_w"
           << std::setw(10) << "parity_ok"
           << std::setw(8) << "pm_r" << std::setw(8) << "pm_w" << std::setw(8) << "pm_f"
           << std::setw(8) << "ps_r" << std::setw(8) << "ps_w" << std::setw(8) << "ps_f"
           << std::setw(8) << "sm_r" << std::setw(8) << "sm_w" << std::setw(8) << "sm_f"
           << std::setw(8) << "om_r" << std::setw(8) << "om_w" << std::setw(8) << "om_f"
           << std::setw(8) << "qs_r" << std::setw(8) << "qs_w" << std::setw(8) << "qs_f"
           << std::setw(8) << "sum_r" << std::setw(8) << "sum_w" << "sum_f\n";
    for (std::size_t source = 0; source < reference.sourceNames.size(); ++source) {
        const vc::fiber_tracer::FiberTraceReferenceSourceBenchmark empty;
        const auto& referenceCounts = source < benchmark.references.size()
            ? benchmark.references[source]
            : empty;
        output << std::setw(8) << std::setprecision(1)
               << 0.5 * static_cast<double>(source);
        if (referenceCounts.estimatedWinding)
            output << std::setw(8) << std::setprecision(1)
                   << *referenceCounts.estimatedWinding;
        else
            output << std::setw(8) << "NA";
        if (referenceCounts.estimatedParityMatches) {
            output << std::setw(10)
                   << (*referenceCounts.estimatedParityMatches ? "yes" : "no");
        } else {
            output << std::setw(10) << "NA";
        }
        for (std::size_t index = 0; index < names.size(); ++index) {
            const auto& counts = index < referenceCounts.classes.size()
                ? referenceCounts.classes[index]
                : referenceCounts.sum;
            output << std::setw(8) << counts.right
                   << std::setw(8) << counts.wrong;
            if (counts.total == 0)
                output << std::setw(8) << "NA";
            else
                output << std::setw(8) << std::setprecision(3)
                       << static_cast<double>(counts.right) /
                              static_cast<double>(counts.total);
        }
        output << '\n';
    }
    output << "constraint accuracy\n"
           << std::setw(20) << "class" << std::setw(12) << "right" << std::setw(12) << "wrong" << std::setw(12) << "total"
           << "right_percent\n";
    for (std::size_t index = 0; index < names.size(); ++index) {
        const auto& counts = index < benchmark.classes.size() ? benchmark.classes[index] : benchmark.sum;
        output << std::setw(20) << names[index]
               << std::setw(12) << counts.right
               << std::setw(12) << counts.wrong
               << std::setw(12) << counts.total;
        if (counts.total == 0)
            output << "NA\n";
        else
            output << 100.0 * static_cast<double>(counts.right) / static_cast<double>(counts.total) << "%\n";
    }
    return output.str();
}

std::string formatBpFinalStateCohorts(
    std::span<const unsigned char> sourcePieceOne,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport& winding)
{
    const auto summary = vc::fiber_tracer::summarizeFiberTraceFinalStates(
        winding.mapOrientationByPiece,
        winding.windingValid,
        sourcePieceOne);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "BP final states by source-piece cohort\n"
           << std::left
           << std::setw(22) << "cohort"
           << std::right
           << std::setw(9) << "pieces"
           << std::setw(9) << "h"
           << std::setw(9) << "v"
           << std::setw(9) << "active"
           << std::setw(9) << "defect"
           << std::setw(11) << "defect_%" << '\n';
    const auto row = [&output](
                         std::string_view name,
                         const vc::fiber_tracer::FiberTraceFinalStateCounts& counts) {
        output << std::left << std::setw(22) << name
               << std::right << std::setw(9) << counts.pieces
               << std::setw(9) << counts.horizontal
               << std::setw(9) << counts.vertical
               << std::setw(9) << counts.active()
               << std::setw(9) << counts.defect;
        if (counts.pieces == 0)
            output << std::setw(11) << "NA";
        else
            output << std::setw(10) << std::fixed << std::setprecision(2)
                   << 100.0 * static_cast<double>(counts.defect) /
                          static_cast<double>(counts.pieces)
                   << '%';
        output << '\n';
    };
    row("central(source_piece=1)", summary.selected);
    row("non-central", summary.other);
    row("total", summary.total);
    return output.str();
}

std::string formatBpConstraintEvidenceCohorts(
    std::span<const unsigned char> sourcePieceOne,
    const vc::fiber_tracer::FiberTraceConstraintReport& constraints,
    const vc::fiber_tracer::FiberTraceInterleavedWindingReport& winding)
{
    const auto summary = vc::fiber_tracer::summarizeFiberTraceConstraintEvidence(
        constraints,
        winding.factorDiagnostics,
        winding.mapOrientationByPiece,
        winding.windingValid,
        sourcePieceOne);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "BP admitted winding evidence by source-piece cohort\n"
           << std::left
           << std::setw(13) << "cohort"
           << std::setw(16) << "class"
           << std::right
           << std::setw(9) << "inc"
           << std::setw(10) << "act_i/p"
           << std::setw(10) << "def_i/p"
           << std::setw(11) << "coeff"
           << std::setw(11) << "act_c/p"
           << std::setw(11) << "def_c/p" << '\n';
    const auto ratio = [](
        double numerator,
        std::size_t denominator) -> std::optional<double> {
        if (denominator == 0)
            return std::nullopt;
        return numerator / static_cast<double>(denominator);
    };
    const auto row = [&] (
        std::string_view cohortName,
        std::string_view className,
        const vc::fiber_tracer::FiberTraceConstraintEvidenceCounts& counts,
        const vc::fiber_tracer::FiberTraceFinalStateCounts& states,
        bool hardSignOnly = false) {
        const auto field = [&output](
            std::optional<double> value,
            int width,
            int precision = 2) {
            if (!value)
                output << std::setw(width) << "NA";
            else
                output << std::setw(width) << std::fixed
                       << std::setprecision(precision) << *value;
        };
        const std::size_t incidences = hardSignOnly
            ? counts.hardSignIncidences
            : counts.incidences;
        const std::size_t activeIncidences = hardSignOnly
            ? counts.activeHardSignIncidences
            : counts.activeIncidences;
        const std::size_t defectIncidences = hardSignOnly
            ? counts.defectHardSignIncidences
            : counts.defectIncidences;
        output << std::left << std::setw(13) << cohortName
               << std::setw(16) << className
               << std::right << std::setw(9) << incidences;
        field(
            ratio(
                static_cast<double>(activeIncidences),
                states.active()),
            10);
        field(
            ratio(
                static_cast<double>(defectIncidences),
                states.defect),
            10);
        if (hardSignOnly) {
            output << std::setw(11) << "NA"
                   << std::setw(11) << "NA"
                   << std::setw(11) << "NA";
        } else {
            output << std::setw(11) << std::fixed << std::setprecision(2)
                   << counts.effectiveWeight;
            field(ratio(counts.activeEffectiveWeight, states.active()), 11);
            field(ratio(counts.defectEffectiveWeight, states.defect), 11);
        }
        output << '\n';
    };
    constexpr std::array evidenceClasses{
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::Continuity,
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::PerpendicularMagnitude,
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::PerpendicularSign,
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::ParallelSameMagnitude,
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::ParallelOtherMagnitude,
        vc::fiber_tracer::FiberTraceConstraintEvidenceClass::ParallelSign,
    };
    const auto cohort = [&] (
        std::string_view name,
        const vc::fiber_tracer::FiberTraceConstraintEvidenceCohort& counts) {
        row(name, "measurement", counts.total, counts.states);
        for (const auto evidenceClass : evidenceClasses) {
            const auto& evidence =
                counts.classes[static_cast<std::size_t>(evidenceClass)];
            row(
                name,
                vc::fiber_tracer::fiberTraceConstraintEvidenceClassName(
                    evidenceClass),
                evidence,
                counts.states);
            if (evidenceClass == vc::fiber_tracer::
                    FiberTraceConstraintEvidenceClass::PerpendicularSign) {
                row(name, "perp_sign_hard", evidence, counts.states, true);
            } else if (evidenceClass == vc::fiber_tracer::
                    FiberTraceConstraintEvidenceClass::ParallelSign) {
                row(name, "parallel_sign_hard", evidence, counts.states, true);
            }
        }
    };
    cohort("central", summary.selected);
    cohort("non-central", summary.other);
    cohort("total", summary.total);
    return output.str();
}

struct ProcessMemoryStats {
    std::optional<std::size_t> residentBytes;
    std::optional<std::size_t> peakResidentBytes;
};

ProcessMemoryStats processMemoryStats()
{
    ProcessMemoryStats result;
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        const auto parseKilobytes = [&](std::string_view prefix) {
            if (!line.starts_with(prefix))
                return std::optional<std::size_t>{};
            std::istringstream field(line.substr(prefix.size()));
            std::size_t kilobytes = 0;
            if (!(field >> kilobytes))
                return std::optional<std::size_t>{};
            return std::optional<std::size_t>{kilobytes * 1024ULL};
        };
        if (const auto value = parseKilobytes("VmRSS:"))
            result.residentBytes = value;
        else if (const auto value = parseKilobytes("VmHWM:"))
            result.peakResidentBytes = value;
    }
    return result;
}

std::string_view materializationPhaseName(
    vc::fiber_tracer::FiberletGraphMaterializationPhase phase)
{
    using Phase = vc::fiber_tracer::FiberletGraphMaterializationPhase;
    switch (phase) {
    case Phase::SeedChunks:
        return "seed_chunks";
    case Phase::PrefixChunks:
        return "prefix_chunks";
    case Phase::EndpointChunks:
        return "endpoint_chunks";
    case Phase::Edges:
        return "edges";
    case Phase::Transitions:
        return "transitions";
    case Phase::ImmutableGraph:
        return "immutable_graph";
    case Phase::Complete:
        return "complete";
    case Phase::Tracing:
        return "tracing";
    }
    return "unknown";
}

std::string formatProgressDuration(double seconds)
{
    if (!(seconds >= 0.0) || !std::isfinite(seconds))
        return "n/a";
    const auto rounded = static_cast<std::uint64_t>(std::ceil(seconds));
    const auto hours = rounded / 3600;
    const auto minutes = (rounded % 3600) / 60;
    const auto remainder = rounded % 60;
    std::ostringstream output;
    if (hours != 0)
        output << hours << 'h';
    if (hours != 0 || minutes != 0)
        output << minutes << 'm';
    output << remainder << 's';
    return output.str();
}

std::string_view normalAlignmentPhaseName(
    vc::fiber_tracer::LasagnaNormalAlignmentProgressPhase phase)
{
    using Phase = vc::fiber_tracer::LasagnaNormalAlignmentProgressPhase;
    switch (phase) {
    case Phase::Sampling:
        return "sampling";
    case Phase::Factors:
        return "factors";
    case Phase::Components:
        return "components";
    case Phase::Messages:
        return "messages";
    case Phase::Finalize:
        return "finalize";
    case Phase::Complete:
        return "complete";
    }
    return "unknown";
}

vc::fiber_tracer::LasagnaNormalAlignmentProgressCallback
makeNormalAlignmentProgressPrinter()
{
    using Phase = vc::fiber_tracer::LasagnaNormalAlignmentProgressPhase;
    return [started = std::chrono::steady_clock::now(),
            phaseStarted = std::chrono::steady_clock::now(),
            lastPrinted = std::chrono::steady_clock::time_point{},
            lastPhase = Phase::Complete](
               const vc::fiber_tracer::LasagnaNormalAlignmentProgress& p)
               mutable {
        const auto now = std::chrono::steady_clock::now();
        const bool transition = p.phase != lastPhase;
        if (transition)
            phaseStarted = now;
        const bool phaseComplete = p.total != 0 && p.completed == p.total;
        const bool intervalElapsed =
            lastPrinted.time_since_epoch().count() == 0 ||
            now - lastPrinted >= std::chrono::seconds(1);
        if (!transition && !phaseComplete && !intervalElapsed)
            return;

        const double elapsed = std::chrono::duration<double>(
            now - started).count();
        const double phaseElapsed = std::chrono::duration<double>(
            now - phaseStarted).count();
        const double percent = p.total == 0
            ? 0.0
            : 100.0 * static_cast<double>(p.completed) /
                  static_cast<double>(p.total);
        double eta = std::numeric_limits<double>::infinity();
        if (p.total != 0 && p.completed == p.total) {
            eta = 0.0;
        } else if (p.phase != Phase::Sampling && p.completed != 0 &&
                   p.completed < p.total) {
            eta = phaseElapsed * static_cast<double>(p.total - p.completed) /
                static_cast<double>(p.completed);
        }

        std::ostringstream line;
        line << std::fixed << std::setprecision(1)
             << "fiber normal alignment phase="
             << normalAlignmentPhaseName(p.phase)
             << " completed=" << p.completed << '/' << p.total
             << " percent=" << percent
             << " elapsed=" << formatProgressDuration(elapsed);
        if (p.phase == Phase::Messages) {
            line << std::scientific << std::setprecision(3)
                 << " residual=" << p.messageResidual
                 << std::fixed
                 << " eta_to_limit=" << formatProgressDuration(eta);
        } else if (p.phase == Phase::Sampling) {
            line << " eta="
                 << (phaseComplete ? "0s" : "n/a")
                 << " eta_basis=opaque_batch";
        } else if (p.phase != Phase::Complete) {
            line << " eta_phase=" << formatProgressDuration(eta);
        }
        std::cout << line.str() << '\n' << std::flush;
        lastPrinted = now;
        lastPhase = p.phase;
    };
}

std::string formatGraphMemoryProfile(
    double elapsedSeconds,
    const vc::fiber_tracer::FiberletGraphMaterializationDiagnostics& diagnostics,
    const vc::fiber_tracer::FiberletStoredReplayCacheStats& caches,
    const vc::lasagna::LasagnaChannelChunkCache::Stats& normalCache,
    const ProcessMemoryStats& memory)
{
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    const auto gib = [](std::size_t bytes) {
        return static_cast<double>(bytes) / bytesPerGiB;
    };
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << "fiberlet graph profile elapsed=" << elapsedSeconds
           << "s phase="
           << materializationPhaseName(
                  diagnostics.phase.load(std::memory_order_relaxed));
    if (memory.residentBytes)
        output << " rss_gib=" << gib(*memory.residentBytes);
    else
        output << " rss_gib=NA";
    if (memory.peakResidentBytes)
        output << " peak_gib=" << gib(*memory.peakResidentBytes);
    else
        output << " peak_gib=NA";
    output << '\n'
           << "  cache_gib anchor=" << gib(caches.anchorDecodedBytes)
           << '/' << gib(caches.anchorDecodedByteCapacity)
           << " path=" << gib(caches.pathDecodedBytes)
           << '/' << gib(caches.pathDecodedByteCapacity)
           << " normal=" << gib(normalCache.cachedBytes)
           << '/' << gib(normalCache.capacityBytes)
           << " pending anchor="
           << caches.anchorPendingDecodes + caches.anchorUnresolvedFetches
           << " path="
           << caches.pathPendingDecodes + caches.pathUnresolvedFetches
           << " normal=" << normalCache.loadsInFlight << '\n'
           << "  chunks seed="
           << diagnostics.seedChunksLoaded.load(std::memory_order_relaxed)
           << '/'
           << diagnostics.seedChunksTotal.load(std::memory_order_relaxed)
           << " prefix="
           << diagnostics.prefixChunksLoaded.load(std::memory_order_relaxed)
           << '/'
           << diagnostics.prefixChunksTotal.load(std::memory_order_relaxed)
           << " endpoint="
           << diagnostics.endpointChunksLoaded.load(std::memory_order_relaxed)
           << '/'
           << diagnostics.endpointChunksTotal.load(std::memory_order_relaxed)
           << " anchors inside="
           << diagnostics.insideAnchors.load(std::memory_order_relaxed)
           << " required="
           << diagnostics.requiredAnchors.load(std::memory_order_relaxed)
           << " materialized="
           << diagnostics.materializedAnchors.load(std::memory_order_relaxed)
           << '\n'
           << "  graph fiberlets="
           << diagnostics.physicalFiberlets.load(std::memory_order_relaxed)
           << " arcs="
           << diagnostics.directedArcs.load(std::memory_order_relaxed)
           << " route_points="
           << diagnostics.routePoints.load(std::memory_order_relaxed)
           << " profile_segments="
           << diagnostics.profileSegments.load(std::memory_order_relaxed)
           << " transition_arcs="
           << diagnostics.transitionInputArcsProcessed.load(
                  std::memory_order_relaxed)
           << '/'
           << diagnostics.transitionInputArcsTotal.load(
                  std::memory_order_relaxed)
           << " successors="
           << diagnostics.successors.load(std::memory_order_relaxed)
           << " replay_transitions="
           << diagnostics.replayTransitions.load(std::memory_order_relaxed)
           << " final="
           << diagnostics.finalAnchors.load(std::memory_order_relaxed)
           << ',' << diagnostics.finalEdges.load(std::memory_order_relaxed)
           << ','
           << diagnostics.finalTransitions.load(std::memory_order_relaxed)
           << '\n';
    return output.str();
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
            auto artifact =
                vc::fiber_tracer::readFiberletCropTraceArtifact(options.input);
            const auto retainedOriginalTraceIndices = applyQualityFilter(
                artifact.lines, options.qualityFraction);
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
            std::optional<vc::fiber_tracer::LasagnaNormalAlignmentField>
                alignedNormalField;
            if (options.mode == Mode::DirectionAblation &&
                (options.bpOnly ||
                 options.bpBalance != BpBalanceSelection::None)) {
                const auto nx = vc::lasagna::bindLasagnaChannel(
                    normalDataset.manifest(), "nx");
                const auto& shapeZYX = *normalDataset.manifest().baseShapeZYX;
                const cv::Vec3d volumeMaximumBaseXYZ{
                    static_cast<double>(shapeZYX[2]),
                    static_cast<double>(shapeZYX[1]),
                    static_cast<double>(shapeZYX[0]),
                };
                cv::Vec3d alignmentMinimum = artifact.minimumBaseXYZ;
                cv::Vec3d alignmentMaximum = artifact.maximumBaseXYZ;
                for (int axis = 0; axis < 3; ++axis) {
                    alignmentMinimum[axis] = std::max(
                        0.0, alignmentMinimum[axis] - nx.spacing);
                    alignmentMaximum[axis] = std::min(
                        volumeMaximumBaseXYZ[axis],
                        alignmentMaximum[axis] + nx.spacing);
                }
                vc::fiber_tracer::LasagnaNormalAlignmentConfig alignmentConfig;
                alignmentConfig.beliefPropagation.temperature =
                    options.bp.horizontalnessTemperature;
                alignmentConfig.beliefPropagation.messageDamping =
                    options.bp.messageDamping;
                alignmentConfig.beliefPropagation.messageResidualTolerance =
                    options.bp.messageResidualTolerance;
                alignmentConfig.beliefPropagation.maximumMessageIterations =
                    options.bp.maximumMessageIterations;
                alignmentConfig.beliefPropagation.parallelWorkers =
                    static_cast<std::size_t>(options.threads);
                alignedNormalField.emplace(
                    vc::fiber_tracer::sampleAndAlignLasagnaNormalLattice(
                        normals,
                        alignmentMinimum,
                        alignmentMaximum,
                        nx.spacing,
                        1,
                        options.threads,
                        alignmentConfig,
                        makeNormalAlignmentProgressPrinter()));
                std::cout
                    << "fiber winding normal alignment"
                    << " spacing_base=" << nx.spacing
                    << " candidates=" << alignedNormalField->candidateSamples
                    << " valid=" << alignedNormalField->rawNormals.size()
                    << " components="
                    << alignedNormalField->alignment.connectedComponents
                    << " message_iterations="
                    << alignedNormalField->alignment.beliefPropagation.messageIterations
                    << " message_residual="
                    << alignedNormalField->alignment.beliefPropagation.messageResidual
                    << " converged="
                    << (alignedNormalField->alignment.beliefPropagation.messageConverged
                            ? "true"
                            : "false")
                    << '\n';
            }

            std::vector<vc::fiber_tracer::FiberletCropTraceLine>
                diagnosticLines;
            std::vector<std::size_t> diagnosticOriginalTraceIndices;
            std::vector<vc::fiber_tracer::FiberDirectionGroup>
                diagnosticDirections;
            std::optional<vc::fiber_tracer::FiberDirectionClassification>
                diagnosticClassification;
            std::optional<ReferenceFiberDiagnostics> referenceDiagnostics;
            std::vector<std::string> deferredReferenceDiagnostics;
            std::vector<std::string> deferredBpStateDiagnostics;
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
                if (options.mode == Mode::DirectionAblation) {
                    referenceDiagnostics = updateReferenceFiberArtifact(
                        options,
                        artifact.minimumBaseXYZ,
                        artifact.maximumBaseXYZ);
                }
                if (referenceDiagnostics) {
                    const auto referenceConstraints =
                        vc::fiber_tracer::extractFiberTraceConstraints(
                            referenceDiagnostics->lines,
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
                            },
                            {},
                            alignedNormalField
                                ? &*alignedNormalField
                                : nullptr);
                    referenceDiagnostics->windingTopology =
                        vc::fiber_tracer::prepareFiberTraceBeliefTopology(
                            referenceDiagnostics->lines,
                            referenceConstraints,
                            artifact.minimumBaseXYZ,
                            artifact.maximumBaseXYZ);
                    prepareReferenceFiberPieces(
                        *referenceDiagnostics, referenceConstraints);
                    referenceDiagnostics->constraintReport =
                        std::move(referenceConstraints);
                }
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
                        diagnosticOriginalTraceIndices.push_back(
                            retainedOriginalTraceIndices.at(trace));
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
                            },
                            {},
                            alignedNormalField
                                ? &*alignedNormalField
                                : nullptr);
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
                        auto selectedBpConstraints = checkpointReport;
                        const auto selection = vc::fiber_tracer::
                            selectFiberTraceLabelConstraints(
                                checkpointReport, options.labeling);
                        selectedBpConstraints.constraints.clear();
                        selectedBpConstraints.constraints.reserve(
                            selection.retainedIndices.size());
                        for (const std::size_t index : selection.retainedIndices) {
                            selectedBpConstraints.constraints.push_back(
                                checkpointReport.constraints.at(index));
                        }
                        std::cout
                            << "fiber direction BP-only cohort"
                            << " admitted=" << admittedCount
                            << " fibers=" << diagnosticLines.size()
                            << " pieces=" << selectedBpConstraints.pieces.size()
                            << " selected_constraints="
                            << selectedBpConstraints.constraints.size() << '\n';
                        const auto runBp = [&] (
                                               vc::fiber_tracer::
                                                   FiberTraceBalanceMode mode) {
                            auto config = options.bp;
                            config.enforceHardSplitContinuity =
                                options.hardSplitContinuity;
                            config.balanceMode = mode;
                            config.cropMinimumBaseXYZ = artifact.minimumBaseXYZ;
                            config.cropMaximumBaseXYZ = artifact.maximumBaseXYZ;
                            vc::fiber_tracer::
                                FiberTraceBeliefPropagationReport report;
                            vc::fiber_tracer::
                                FiberTraceWindingBeliefPropagationConfig
                                    windingConfig;
                            windingConfig.temperature =
                                options.bpInference == vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProductMixed
                                ? 0.25
                                : options.bp.horizontalnessTemperature;
                            windingConfig.messageDamping = options.bp.messageDamping;
                            windingConfig.messageResidualTolerance =
                                options.bp.messageResidualTolerance;
                            windingConfig.maximumMessageIterations =
                                options.bp.maximumMessageIterations;
                            windingConfig.parallelWorkers =
                                static_cast<std::size_t>(options.threads);
                            windingConfig.parallelWindingDistanceCutoff =
                                options.parallelWindingCutoff;
                            windingConfig.enforcePerpendicularWindingSign =
                                options.enforcePerpendicularWindingSign;
                            windingConfig.enforceParallelWindingSign =
                                options.enforceParallelWindingSign;
                            windingConfig.enforceHardSplitContinuity =
                                options.hardSplitContinuity;
                            windingConfig.hardSignMinimumNormalAlignment =
                                options.hardSignMinimumNormalAlignment;
                            windingConfig.decisionConfidence =
                                options.windingDecisionConfidence;
                            windingConfig.normalConfidence =
                                options.windingNormalConfidence;
                            windingConfig.finiteSignInfringementCost =
                                options.windingSignCost;
                            windingConfig.perpendicularSignWeight =
                                options.windingSignWeights[0];
                            windingConfig.parallelSignWeight =
                                options.windingSignWeights[1];
                            setWindingClassWeights(
                                windingConfig, options.windingWeights);
                            auto bpConstraints = selectedBpConstraints;
                            auto bpSourceLines = diagnosticLines;
                            auto bpSourceOriginalTraceIndices = diagnosticOriginalTraceIndices;
                            auto bpSourceDirections = diagnosticDirections;
                            std::vector<unsigned char> bpSourcePieceOne;
                            bpSourcePieceOne.reserve(bpConstraints.pieces.size());
                            for (const auto& piece : bpConstraints.pieces) {
                                bpSourcePieceOne.push_back(
                                    piece.pieceIndex == 1 ? 1 : 0);
                            }
                            std::optional<vc::fiber_tracer::
                                FiberTraceWindingBeliefPropagationReport>
                                    independentWinding;
                            std::optional<vc::fiber_tracer::
                                FiberTraceInterleavedWindingReport>
                                    interleavedWinding;
                            const bool jointGrid =
                                options.bpInference == vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProductMixed &&
                                options.windingSolver == vc::fiber_tracer::
                                    FiberTraceWindingSolver::JointGrid;
                            auto bpTopology = vc::fiber_tracer::
                                prepareFiberTraceBeliefTopology(bpSourceLines, bpConstraints, artifact.minimumBaseXYZ, artifact.maximumBaseXYZ);
                            std::vector<vc::fiber_tracer::
                                FiberTraceFixedOrientation> fixedOrientations;
                            const auto solveOrientation = [&] {
                                const auto orientationStarted =
                                    std::chrono::steady_clock::now();
                                std::cout
                                    << "fiber orientation BP status=started\n"
                                    << std::flush;
                                switch (options.bpInference) {
                                case vc::fiber_tracer::
                                    FiberTraceBeliefInference::MinSum:
                                    report = vc::fiber_tracer::
                                        solveFiberTraceBeliefPropagation(bpSourceLines,
                                            bpConstraints,
                                            config);
                                    break;
                                case vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProduct:
                                    report = vc::fiber_tracer::
                                        solveFiberTraceSumProduct(bpSourceLines,
                                            bpConstraints,
                                            config);
                                    break;
                                case vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProductMixed:
                                    report = vc::fiber_tracer::
                                        solveFiberTraceMixedSumProduct(bpSourceLines,
                                            bpConstraints,
                                            config);
                                    break;
                                }
                                std::cout
                                    << "fiber orientation BP status=complete elapsed="
                                    << std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() -
                                           orientationStarted)
                                           .count()
                                    << "s\n"
                                    << std::flush;
                            };
                            const std::size_t inputPieces =
                                bpConstraints.pieces.size();
                            const std::size_t inputTraces = bpConstraints.inputTraces;
                            const std::size_t inputConstraints =
                                bpConstraints.constraints.size();
                            std::size_t componentRounds = 0;
                            while (true) {
                                if (bpConstraints.pieces.empty()) {
                                    throw std::runtime_error("BP main winding component is empty");
                                }
                                fixedOrientations.clear();
                                if (options.windingFixedOrientation) {
                                    solveOrientation();
                                    fixedOrientations = vc::fiber_tracer::
                                        fixedFiberTraceOrientations(report);
                                }
                                const auto component = vc::fiber_tracer::
                                    selectLargestFiberTraceWindingComponent(
                                        bpConstraints,
                                        bpTopology,
                                        windingConfig,
                                        fixedOrientations,
                                        options.bpInference == vc::fiber_tracer::
                                            FiberTraceBeliefInference::SumProductMixed,
                                        bpTopology.centralSeedPiece,
                                        jointGrid
                                            ? options.jointGrid
                                                  .fixedMeasurementScale
                                                  .value_or(1.0)
                                            : 1.0);
                                if (component.components <= 1)
                                    break;
                                const auto subset = vc::fiber_tracer::
                                    subsetFiberTraceConstraintReport(
                                        bpConstraints,
                                        component.retainedPieceIndices);
                                std::vector<vc::fiber_tracer::FiberletCropTraceLine> nextLines;
                                std::vector<std::size_t> nextOriginalIndices;
                                std::vector<vc::fiber_tracer::FiberDirectionGroup> nextDirections;
                                std::vector<unsigned char> nextSourcePieceOne;
                                nextLines.reserve(subset.retainedTraceIndices.size());
                                nextOriginalIndices.reserve(subset.retainedTraceIndices.size());
                                nextDirections.reserve(subset.retainedTraceIndices.size());
                                nextSourcePieceOne.reserve(
                                    subset.retainedPieceIndices.size());
                                for (const std::size_t oldPiece :
                                     subset.retainedPieceIndices) {
                                    nextSourcePieceOne.push_back(
                                        bpSourcePieceOne.at(oldPiece));
                                }
                                for (const std::size_t oldTrace : subset.retainedTraceIndices) {
                                    nextLines.push_back(bpSourceLines.at(oldTrace));
                                    nextOriginalIndices.push_back(bpSourceOriginalTraceIndices.at(oldTrace));
                                    nextDirections.push_back(bpSourceDirections.at(oldTrace));
                                }
                                bpConstraints = subset.report;
                                bpSourceLines = std::move(nextLines);
                                bpSourceOriginalTraceIndices = std::move(nextOriginalIndices);
                                bpSourceDirections = std::move(nextDirections);
                                bpSourcePieceOne = std::move(nextSourcePieceOne);
                                bpTopology = vc::fiber_tracer::
                                    prepareFiberTraceBeliefTopology(
                                        bpSourceLines,
                                        bpConstraints,
                                        artifact.minimumBaseXYZ,
                                        artifact.maximumBaseXYZ);
                                ++componentRounds;
                                if (componentRounds > inputPieces) {
                                    throw std::logic_error("BP component filtering did not converge");
                                }
                            }
                            if (!options.windingFixedOrientation && !jointGrid)
                                solveOrientation();
                            std::cout
                                << "fiber BP main winding component"
                                << " rounds=" << componentRounds
                                << " traces=" << inputTraces << "->"
                                << bpConstraints.inputTraces
                                << " pieces=" << inputPieces << "->"
                                << bpConstraints.pieces.size()
                                << " constraints=" << inputConstraints << "->"
                                << bpConstraints.constraints.size() << '\n';

                            const auto bpPieceLines = vc::fiber_tracer::makeFiberTraceConstraintPieceLines(bpSourceLines, bpConstraints);
                            std::vector<std::size_t> bpOriginalTraceIndices;
                            std::vector<vc::fiber_tracer::FiberDirectionGroup> bpDirections;
                            bpOriginalTraceIndices.reserve(bpConstraints.pieces.size());
                            bpDirections.reserve(bpConstraints.pieces.size());
                            for (const auto& piece : bpConstraints.pieces) {
                                if (piece.traceIndex >= bpSourceLines.size()) {
                                    throw std::logic_error("BP piece references an invalid filtered source fiber");
                                }
                                bpOriginalTraceIndices.push_back(bpSourceOriginalTraceIndices.at(piece.traceIndex));
                                bpDirections.push_back(bpSourceDirections.at(piece.traceIndex));
                            }
                            std::optional<ReferenceBpCrossConstraints> referenceBpConstraints;
                            if (referenceDiagnostics && options.bpInference == vc::fiber_tracer::FiberTraceBeliefInference::SumProductMixed) {
                                if (!alignedNormalField) {
                                    throw std::logic_error("Reference/BP benchmark requires aligned normals");
                                }
                                referenceBpConstraints =
                                    extractReferenceBpCrossConstraints(*referenceDiagnostics, bpPieceLines, options, normals, *alignedNormalField);
                            }
                            const auto solveInterleaved = [&] (
                                const WindingWeightTuple& weights,
                                bool showProgress) {
                                auto weightedConfig = windingConfig;
                                setWindingWeights(weightedConfig, weights);
                                if (jointGrid) {
                                    auto joint = options.jointGrid;
                                    static_cast<vc::fiber_tracer::
                                        FiberTraceWindingBeliefPropagationConfig&>(joint) =
                                            weightedConfig;
                                    joint.mixedUnaryCost = options.windingDefectCost;
                                    joint.pieceBreakCost = options.pieceBreakCost;
                                    joint.orientationTemperature =
                                        options.bp.horizontalnessTemperature;
                                    return vc::fiber_tracer::
                                        solveFiberTraceJointGridWindingBeliefPropagation(
                                            bpConstraints,
                                            bpTopology,
                                            joint,
                                            showProgress
                                                ? makeJointGridWindingProgressPrinter()
                                                : vc::fiber_tracer::
                                                      FiberTraceJointGridProgressCallback{},
                                            fixedOrientations);
                                }
                                vc::fiber_tracer::
                                    FiberTraceInterleavedWindingConfig joint;
                                static_cast<vc::fiber_tracer::
                                    FiberTraceWindingBeliefPropagationConfig&>(joint) =
                                        weightedConfig;
                                joint.mixedUnaryCost = options.windingDefectCost;
                                joint.pieceBreakCost = options.pieceBreakCost;
                                joint.orientationTemperature =
                                    options.bp.horizontalnessTemperature;
                                joint.temperature = 0.25;
                                return vc::fiber_tracer::
                                    solveFiberTraceInterleavedWindingBeliefPropagation(
                                        bpConstraints,
                                        bpTopology,
                                        report,
                                        joint,
                                        showProgress
                                            ? makeInterleavedWindingProgressPrinter()
                                            : vc::fiber_tracer::
                                                  FiberTraceInterleavedWindingProgressCallback{},
                                        fixedOrientations);
                            };

                            if (options.windingWeightSearch ||
                                options.windingWeightSearchLocal) {
                                if (!referenceDiagnostics ||
                                    !referenceBpConstraints) {
                                    throw std::logic_error(
                                        "Winding weight search has no reference benchmark");
                                }
                                struct SearchRow {
                                    WindingWeightTuple weights;
                                    WindingWeightSearchScore score;
                                    double seconds = 0.0;
                                    std::string status;
                                };
                                std::vector<SearchRow> rows;
                                std::optional<WindingWeightSearchScore> bestScore;
                                WindingWeightTuple bestWeights{};
                                const auto searchStarted =
                                    std::chrono::steady_clock::now();
                                std::size_t evaluatedScenarios = 0;

                                const auto evaluate = [&] (
                                    const WindingWeightTuple& weights,
                                    const std::string& phase,
                                    const std::size_t completed,
                                    const std::size_t workTotal)
                                    -> std::optional<std::size_t> {
                                    const auto started =
                                        std::chrono::steady_clock::now();
                                    try {
                                        auto candidate =
                                            solveInterleaved(weights, false);
                                        auto candidateConfig = windingConfig;
                                        setWindingWeights(
                                            candidateConfig, weights);
                                        const auto observations =
                                            makeReferenceBpWindingObservations(
                                                *referenceDiagnostics,
                                                *referenceBpConstraints,
                                                candidate,
                                                candidateConfig);
                                        const auto benchmark = vc::fiber_tracer::
                                            calibrateFiberTraceReferenceWindings(
                                                observations);
                                        const auto score =
                                            scoreWindingWeightSearch(
                                                *referenceDiagnostics,
                                                benchmark,
                                                candidate);
                                        const double seconds =
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                started).count();
                                        rows.push_back({
                                            weights, score, seconds,
                                            candidate.status});
                                        const std::size_t row = rows.size() - 1;
                                        if (!bestScore ||
                                            betterWindingWeightSearchResult(
                                                score, weights, *bestScore,
                                                bestWeights)) {
                                            bestScore = score;
                                            bestWeights = weights;
                                        }
                                        ++evaluatedScenarios;
                                        const double elapsed =
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                searchStarted).count();
                                        const double eta = elapsed /
                                            static_cast<double>(evaluatedScenarios) *
                                            static_cast<double>(
                                                workTotal - completed);
                                        const std::size_t constraintTotal =
                                            score.rightConstraints +
                                            score.wrongConstraints;
                                        std::cout
                                            << "winding weight search phase="
                                            << phase << ' ' << completed << '/'
                                            << workTotal << " weights="
                                            << formatWindingWeights(weights)
                                            << " ref_exact="
                                            << score.exactReferences << '/'
                                            << referenceDiagnostics->sourceNames.size()
                                            << " ref_wrong="
                                            << score.wrongReferences
                                            << " ref_missing="
                                            << score.missingReferences
                                            << " constraints="
                                            << score.rightConstraints << '/'
                                            << constraintTotal
                                            << " fraction="
                                            << (constraintTotal == 0 ? 0.0 :
                                                static_cast<double>(
                                                    score.rightConstraints) /
                                                static_cast<double>(
                                                    constraintTotal))
                                            << " converged="
                                            << std::boolalpha << score.converged
                                            << std::noboolalpha
                                            << " seconds=" << seconds
                                            << " eta_seconds=" << eta << '\n'
                                            << std::flush;
                                        return row;
                                    } catch (const std::exception& error) {
                                        ++evaluatedScenarios;
                                        const double seconds =
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                started).count();
                                        std::cout
                                            << "winding weight search phase="
                                            << phase << ' ' << completed << '/'
                                            << workTotal << " weights="
                                            << formatWindingWeights(weights)
                                            << " status=failed seconds="
                                            << seconds << " error="
                                            << error.what() << '\n'
                                            << std::flush;
                                        return std::nullopt;
                                    }
                                };

                                if (options.windingWeightSearch) {
                                    constexpr std::size_t maximumScenarios =
                                        100'000;
                                    std::size_t scenarioCount = 1;
                                    for (std::size_t dimension = 0;
                                         dimension < 7;
                                         ++dimension) {
                                        if (scenarioCount > maximumScenarios /
                                            options.windingWeightSearch->size()) {
                                            throw std::invalid_argument(
                                                "Winding weight search grid exceeds 100000 scenarios");
                                        }
                                        scenarioCount *=
                                            options.windingWeightSearch->size();
                                    }
                                    rows.reserve(scenarioCount);
                                    std::size_t scenario = 0;
                                    for (const double p05 : *options.windingWeightSearch)
                                    for (const double pFar : *options.windingWeightSearch)
                                    for (const double p0 : *options.windingWeightSearch)
                                    for (const double p1 : *options.windingWeightSearch)
                                    for (const double pFarParallel : *options.windingWeightSearch)
                                    for (const double perpendicularSign : *options.windingWeightSearch)
                                    for (const double parallelSign : *options.windingWeightSearch) {
                                        ++scenario;
                                        (void)evaluate(
                                            {p05, pFar, p0, p1, pFarParallel,
                                             perpendicularSign, parallelSign},
                                            "grid", scenario, scenarioCount);
                                    }
                                } else {
                                    constexpr int exponentLimit = 16;
                                    constexpr int zeroCoordinate =
                                        exponentLimit + 1;
                                    constexpr std::size_t maximumIterations =
                                        160;
                                    using WeightTuple = WindingWeightTuple;
                                    using SearchState = std::array<int, 7>;
                                    std::map<SearchState,
                                             std::optional<std::size_t>> cache;
                                    WeightTuple positiveAnchor{};
                                    const WeightTuple initialWeights =
                                        windingWeights(options);
                                    for (std::size_t dimension = 0;
                                         dimension < positiveAnchor.size();
                                         ++dimension) {
                                        positiveAnchor[dimension] =
                                            initialWeights[dimension] > 0.0
                                            ? initialWeights[dimension]
                                            : 1.0;
                                    }
                                    const auto weightsFor = [&] (
                                        const SearchState& state) {
                                        WeightTuple weights{};
                                        for (std::size_t dimension = 0;
                                             dimension < weights.size();
                                             ++dimension) {
                                            weights[dimension] =
                                                state[dimension] == zeroCoordinate
                                                ? 0.0
                                                : std::ldexp(
                                                      positiveAnchor[dimension],
                                                      state[dimension]);
                                            if (!std::isfinite(weights[dimension]) ||
                                                weights[dimension] < 0.0) {
                                                throw std::overflow_error(
                                                    "Local winding weight is outside the finite nonnegative domain");
                                            }
                                        }
                                        return weights;
                                    };
                                    const auto evaluateState = [&] (
                                        const SearchState& state,
                                        const std::string& phase,
                                        const std::size_t completed,
                                        const std::size_t workTotal) {
                                        const auto found = cache.find(state);
                                        if (found != cache.end()) {
                                            std::cout
                                                << "winding weight search phase="
                                                << phase << ' ' << completed
                                                << '/' << workTotal
                                                << " weights="
                                                << formatWindingWeights(
                                                       weightsFor(state))
                                                << " status=cached\n"
                                                << std::flush;
                                            return found->second;
                                        }
                                        const WeightTuple weights =
                                            weightsFor(state);
                                        const auto result = evaluate(
                                            weights, phase,
                                            completed, workTotal);
                                        cache.emplace(state, result);
                                        return result;
                                    };

                                    SearchState currentState{};
                                    for (std::size_t dimension = 0;
                                         dimension < currentState.size();
                                         ++dimension) {
                                        if (initialWeights[dimension] ==
                                            0.0) {
                                            currentState[dimension] =
                                                zeroCoordinate;
                                        }
                                    }
                                    const auto initial = evaluateState(
                                        currentState, "local_start", 1, 1);
                                    if (!initial) {
                                        throw std::runtime_error(
                                            "Initial local winding weight scenario failed");
                                    }
                                    std::size_t currentRow = *initial;
                                    bool localOptimum = false;
                                    for (std::size_t iteration = 1;
                                         iteration <= maximumIterations;
                                         ++iteration) {
                                        std::vector<SearchState> neighbors;
                                        neighbors.reserve(21);
                                        const auto appendNeighbor = [&] (
                                            const SearchState& candidate) {
                                            if (candidate == currentState ||
                                                std::find(
                                                    neighbors.begin(),
                                                    neighbors.end(),
                                                    candidate) != neighbors.end()) {
                                                return;
                                            }
                                            neighbors.push_back(candidate);
                                        };
                                        for (std::size_t dimension = 0;
                                             dimension < currentState.size();
                                             ++dimension) {
                                            if (currentState[dimension] !=
                                                zeroCoordinate) {
                                                SearchState zero = currentState;
                                                zero[dimension] = zeroCoordinate;
                                                appendNeighbor(zero);
                                                for (const int delta : {-1, 1}) {
                                                    SearchState neighbor =
                                                        currentState;
                                                    neighbor[dimension] += delta;
                                                    if (neighbor[dimension] <
                                                            -exponentLimit ||
                                                        neighbor[dimension] >
                                                            exponentLimit) {
                                                        continue;
                                                    }
                                                    appendNeighbor(neighbor);
                                                }
                                            } else {
                                                for (const int exponent :
                                                     {-1, 0, 1}) {
                                                    SearchState neighbor =
                                                        currentState;
                                                    neighbor[dimension] = exponent;
                                                    appendNeighbor(neighbor);
                                                }
                                            }
                                        }

                                        std::optional<std::size_t> nextRow;
                                        SearchState nextState{};
                                        for (std::size_t neighbor = 0;
                                             neighbor < neighbors.size();
                                             ++neighbor) {
                                            const auto candidate =
                                                evaluateState(
                                                    neighbors[neighbor],
                                                    "local_" +
                                                        std::to_string(iteration),
                                                    neighbor + 1,
                                                    neighbors.size());
                                            if (!candidate ||
                                                !strictlyBetterWindingWeightSearchQuality(
                                                    rows[*candidate].score,
                                                    rows[currentRow].score)) {
                                                continue;
                                            }
                                            if (!nextRow ||
                                                betterWindingWeightSearchResult(
                                                    rows[*candidate].score,
                                                    rows[*candidate].weights,
                                                    rows[*nextRow].score,
                                                    rows[*nextRow].weights)) {
                                                nextRow = *candidate;
                                                nextState =
                                                    neighbors[neighbor];
                                            }
                                        }
                                        if (!nextRow) {
                                            localOptimum = true;
                                            std::cout
                                                << "winding weight search status=local_optimum iterations="
                                                << iteration - 1
                                                << " evaluated=" << rows.size()
                                                << " selected="
                                                << formatWindingWeights(
                                                       rows[currentRow].weights)
                                                << '\n';
                                            break;
                                        }
                                        currentRow = *nextRow;
                                        currentState = nextState;
                                        const auto& selectedScore =
                                            rows[currentRow].score;
                                        const std::size_t selectedTotal =
                                            selectedScore.rightConstraints +
                                            selectedScore.wrongConstraints;
                                        std::cout
                                            << "winding weight search status=move iteration="
                                            << iteration << " selected="
                                            << formatWindingWeights(
                                                   rows[currentRow].weights)
                                            << " ref_exact="
                                            << selectedScore.exactReferences
                                            << '/'
                                            << referenceDiagnostics->sourceNames.size()
                                            << " constraints="
                                            << selectedScore.rightConstraints
                                            << '/' << selectedTotal
                                            << " fraction="
                                            << (selectedTotal == 0 ? 0.0 :
                                                static_cast<double>(
                                                    selectedScore.rightConstraints) /
                                                static_cast<double>(
                                                    selectedTotal))
                                            << '\n';
                                    }
                                    if (!localOptimum) {
                                        throw std::runtime_error(
                                            "Local winding weight search reached its iteration limit before a local optimum");
                                    }
                                    bestScore = rows[currentRow].score;
                                    bestWeights = rows[currentRow].weights;
                                }

                                if (!bestScore) {
                                    throw std::runtime_error(
                                        "Every winding weight search scenario failed");
                                }
                                std::sort(
                                    rows.begin(), rows.end(),
                                    [](const SearchRow& a, const SearchRow& b) {
                                        return betterWindingWeightSearchResult(
                                            a.score, a.weights,
                                            b.score, b.weights);
                                    });
                                std::cout
                                    << "winding weight search ranking\n"
                                    << "rank  weights  ref_exact  ref_wrong  ref_missing  right  wrong  fraction  converged  seconds\n";
                                for (std::size_t rank = 0;
                                     rank < rows.size();
                                     ++rank) {
                                    const auto& row = rows[rank];
                                    const std::size_t total =
                                        row.score.rightConstraints +
                                        row.score.wrongConstraints;
                                    std::cout
                                        << rank + 1 << "  "
                                        << formatWindingWeights(row.weights)
                                        << "  " << row.score.exactReferences
                                        << "  " << row.score.wrongReferences
                                        << "  " << row.score.missingReferences
                                        << "  " << row.score.rightConstraints
                                        << "  " << row.score.wrongConstraints
                                        << "  " << (total == 0 ? 0.0 :
                                            static_cast<double>(
                                                row.score.rightConstraints) /
                                            static_cast<double>(total))
                                        << "  " << std::boolalpha
                                        << row.score.converged
                                        << std::noboolalpha
                                        << "  " << row.seconds << '\n';
                                }
                                setWindingWeights(
                                    windingConfig, bestWeights);
                                interleavedWinding = solveInterleaved(
                                    bestWeights, false);
                                std::cout
                                    << "winding weight search selected="
                                    << formatWindingWeights(bestWeights)
                                    << '\n';
                            } else {
                                interleavedWinding = solveInterleaved(
                                    windingWeights(options), true);
                            }

                            if (interleavedWinding &&
                                !options.windingFixedOrientation) {
                                report.horizontalProbability =
                                    interleavedWinding->classAProbability;
                                report.mixedProbability =
                                    interleavedWinding->mixedProbability;
                                report.verticalProbability =
                                    interleavedWinding->classBProbability;
                                report.normalizedArcWeights =
                                    bpTopology.normalizedArcWeights;
                                report.seedPieceIndex =
                                    bpTopology.centralSeedPiece;
                                report.factors = interleavedWinding->factors;
                                report.mergedMeasurements =
                                    bpConstraints.constraints.size();
                                report.connectedComponents =
                                    interleavedWinding->connectedComponents;
                                report.targetHorizontalFraction =
                                    config.targetHorizontalFraction;
                                report.inference = vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProductMixed;
                                report.inferenceTemperature =
                                    options.bp.horizontalnessTemperature;
                                report.mixedUnaryCost = options.windingDefectCost;
                                report.messageIterations =
                                    interleavedWinding->messageIterations;
                                report.messageResidual =
                                    interleavedWinding->messageResidual;
                                report.messageConverged =
                                    interleavedWinding->messageConverged;
                                report.solveSeconds =
                                    interleavedWinding->continuousSolveSeconds +
                                    interleavedWinding->discreteSolveSeconds;
                                report.status = interleavedWinding->status;
                                report.horizontalness.resize(
                                    interleavedWinding->classAProbability.size());
                                std::vector<std::size_t> degree(
                                    report.horizontalness.size(), 0);
                                for (std::size_t piece = 0;
                                     piece < report.horizontalness.size();
                                     ++piece) {
                                    report.horizontalness[piece] =
                                        orientationProjection(
                                            report.horizontalProbability[piece],
                                            report.mixedProbability[piece],
                                            report.verticalProbability[piece],
                                            piece);
                                }
                                for (const auto& factor :
                                     interleavedWinding->factorDiagnostics) {
                                    ++degree.at(factor.pieceA);
                                    ++degree.at(factor.pieceB);
                                    if (std::abs(
                                            factor.parallelScore -
                                            factor.perpendicularScore) <=
                                        1.0e-12) {
                                        ++report.neutralFactors;
                                    }
                                }
                                report.neutralMeasurements =
                                    report.neutralFactors;
                                report.isolatedPieces =
                                    static_cast<std::size_t>(std::count(
                                        degree.begin(), degree.end(), 0));
                            }
                            if (!jointGrid &&
                                options.bpInference != vc::fiber_tracer::
                                    FiberTraceBeliefInference::SumProductMixed) {
                                independentWinding = vc::fiber_tracer::
                                    solveFiberTraceWindingBeliefPropagation(
                                        bpConstraints, bpTopology, windingConfig);
                            }
                            const double arcWeight = std::accumulate(
                                report.normalizedArcWeights.begin(),
                                report.normalizedArcWeights.end(),
                                0.0);
                            report.achievedHorizontalFraction = arcWeight > 0.0
                                ? std::inner_product(
                                      report.horizontalness.begin(),
                                      report.horizontalness.end(),
                                      report.normalizedArcWeights.begin(),
                                      0.0) / arcWeight
                                : std::accumulate(
                                      report.horizontalness.begin(),
                                      report.horizontalness.end(),
                                      0.0) /
                                      static_cast<double>(
                                          report.horizontalness.size());
                            const auto& winding = interleavedWinding
                                ? static_cast<const vc::fiber_tracer::
                                      FiberTraceWindingBeliefPropagationReport&>(
                                      *interleavedWinding)
                                : *independentWinding;
                            if (referenceBpConstraints && interleavedWinding) {
                                deferredReferenceDiagnostics.push_back(
                                    formatBpFinalStateCohorts(
                                        bpSourcePieceOne,
                                        *interleavedWinding) +
                                    formatBpConstraintEvidenceCohorts(
                                        bpSourcePieceOne,
                                        bpConstraints,
                                        *interleavedWinding) +
                                    formatReferenceBpWindingBenchmark(
                                        *referenceDiagnostics,
                                        *referenceBpConstraints,
                                        *interleavedWinding,
                                        mode,
                                        windingConfig));
                            } else if (interleavedWinding) {
                                deferredBpStateDiagnostics.push_back(
                                    formatBpFinalStateCohorts(
                                        bpSourcePieceOne,
                                        *interleavedWinding) +
                                    formatBpConstraintEvidenceCohorts(
                                        bpSourcePieceOne,
                                        bpConstraints,
                                        *interleavedWinding));
                            }
                            writeAndPrintBpReport(
                                report,
                                winding,
                                interleavedWinding
                                    ? &*interleavedWinding
                                    : nullptr,
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
                            const auto bpTopology = vc::fiber_tracer::
                                prepareFiberTraceBeliefTopology(
                                    diagnosticLines,
                                    comparisonReport,
                                    artifact.minimumBaseXYZ,
                                    artifact.maximumBaseXYZ);
                            vc::fiber_tracer::
                                FiberTraceWindingBeliefPropagationConfig
                                    windingConfig;
                            windingConfig.temperature =
                                options.bp.horizontalnessTemperature;
                            windingConfig.messageDamping =
                                options.bp.messageDamping;
                            windingConfig.messageResidualTolerance =
                                options.bp.messageResidualTolerance;
                            windingConfig.maximumMessageIterations =
                                options.bp.maximumMessageIterations;
                            windingConfig.parallelWorkers =
                                static_cast<std::size_t>(options.threads);
                            windingConfig.parallelWindingDistanceCutoff =
                                options.parallelWindingCutoff;
                            windingConfig.enforcePerpendicularWindingSign =
                                options.enforcePerpendicularWindingSign;
                            windingConfig.enforceParallelWindingSign =
                                options.enforceParallelWindingSign;
                            windingConfig.enforceHardSplitContinuity =
                                options.hardSplitContinuity;
                            windingConfig.hardSignMinimumNormalAlignment =
                                options.hardSignMinimumNormalAlignment;
                            windingConfig.decisionConfidence =
                                options.windingDecisionConfidence;
                            windingConfig.normalConfidence =
                                options.windingNormalConfidence;
                            windingConfig.finiteSignInfringementCost =
                                options.windingSignCost;
                            windingConfig.perpendicularSignWeight =
                                options.windingSignWeights[0];
                            windingConfig.parallelSignWeight =
                                options.windingSignWeights[1];
                            const auto winding = vc::fiber_tracer::
                                solveFiberTraceWindingBeliefPropagation(
                                    comparisonReport,
                                    bpTopology,
                                    windingConfig);
                            const auto runBp = [&](
                                                   vc::fiber_tracer::
                                                       FiberTraceBalanceMode mode) {
                                auto config = options.bp;
                                config.enforceHardSplitContinuity =
                                    options.hardSplitContinuity;
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
                                    winding,
                                    nullptr,
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
                if (!deferredBpStateDiagnostics.empty()) {
                    std::cout << "\nBP final diagnostics\n";
                    for (const auto& diagnostic : deferredBpStateDiagnostics) {
                        std::cout << diagnostic;
                        if (diagnostic.empty() || diagnostic.back() != '\n')
                            std::cout << '\n';
                    }
                }
                if (!deferredReferenceDiagnostics.empty()) {
                    std::cout << "\nreference diagnostics\n";
                    for (const auto& diagnostic : deferredReferenceDiagnostics) {
                        std::cout << diagnostic;
                        if (diagnostic.empty() || diagnostic.back() != '\n')
                            std::cout << '\n';
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
                    diagnosticOriginalTraceIndices.push_back(
                        retainedOriginalTraceIndices.at(trace));
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
            auto artifact =
                vc::fiber_tracer::readFiberletCropTraceArtifact(options.input);
            applyQualityFilter(artifact.lines, options.qualityFraction);
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

        vc::fiber_tracer::FiberletGraphMaterializationDiagnostics
            graphDiagnostics;
        const auto normalChunkCache =
            vc::lasagna::sharedLasagnaChannelChunkCache(
                options.cacheBytes);
        const auto profileStarted = std::chrono::steady_clock::now();
        std::jthread profileThread;
        if (options.profileMemory) {
            profileThread = std::jthread([&](std::stop_token stop) {
                while (!stop.stop_requested()) {
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - profileStarted)
                                               .count();
                    const auto line = formatGraphMemoryProfile(
                        elapsed, graphDiagnostics, graph.cacheStats(),
                        normalChunkCache->stats(), processMemoryStats());
                    std::cout << line << std::flush;
                    for (int tenth = 0;
                         tenth < 10 && !stop.stop_requested(); ++tenth) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(100));
                    }
                }
            });
        }

        const auto graphStarted = std::chrono::steady_clock::now();
        const auto graphCpuStarted = std::clock();
        const auto searchBox =
            vc::fiber_tracer::fiberletCropTraceSearchBox(options.trace);
        auto materialized =
            graph.materializeBaseBoxForSeeds(
                searchBox.minimumBaseXYZ,
                searchBox.maximumBaseXYZ,
                options.trace.minimumBaseXYZ,
                options.trace.maximumBaseXYZ,
                static_cast<std::size_t>(options.threads),
                options.profileMemory ? &graphDiagnostics : nullptr);
        const double graphSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - graphStarted).count();
        const double graphCpuSeconds = static_cast<double>(std::clock() - graphCpuStarted) / CLOCKS_PER_SEC;
        std::cout << "fiberlet crop graph prepared"
                  << " anchors=" << materialized.insideAnchors.size() << " prediction_to_base=" << dataset->metadata().predictionToBaseScale
                  << " crop_min_xyz=" << options.trace.minimumBaseXYZ[0] << ',' << options.trace.minimumBaseXYZ[1] << ',' << options.trace.minimumBaseXYZ[2]
                  << " crop_max_xyz=" << options.trace.maximumBaseXYZ[0] << ',' << options.trace.maximumBaseXYZ[1] << ',' << options.trace.maximumBaseXYZ[2]
                  << " search_min_xyz=" << searchBox.minimumBaseXYZ[0] << ',' << searchBox.minimumBaseXYZ[1] << ',' << searchBox.minimumBaseXYZ[2]
                  << " search_max_xyz=" << searchBox.maximumBaseXYZ[0] << ',' << searchBox.maximumBaseXYZ[1] << ',' << searchBox.maximumBaseXYZ[2]
                  << " search_padding_base=" << options.trace.lookaheadDistanceBaseVoxels
                  << " elapsed_seconds=" << graphSeconds << " cpu_seconds=" << graphCpuSeconds << '\n';

        const auto traceStarted = std::chrono::steady_clock::now();
        const auto traceCpuStarted = std::clock();
        std::deque<std::pair<
            std::chrono::steady_clock::time_point, std::size_t>>
            traceProgressSamples{{traceStarted, 0}};
        graphDiagnostics.phase.store(
            vc::fiber_tracer::FiberletGraphMaterializationPhase::Tracing,
            std::memory_order_relaxed);
        const auto result = vc::fiber_tracer::traceFiberletCrop(
            *materialized.graph,
            std::move(materialized.insideAnchors),
            normals,
            dataset->metadata().predictionToBaseScale,
            options.trace,
            [&](const auto& current, std::size_t remaining) {
                const auto now = std::chrono::steady_clock::now();
                const std::size_t resolved =
                    current.candidateAnchors >= remaining
                    ? current.candidateAnchors - remaining
                    : 0;
                traceProgressSamples.emplace_back(now, resolved);
                const auto currentWindowBegin = now - std::chrono::seconds(10);
                while (traceProgressSamples.size() > 2 &&
                       traceProgressSamples[1].first <= currentWindowBegin) {
                    traceProgressSamples.pop_front();
                }
                const double elapsed = std::chrono::duration<double>(
                    now - traceStarted).count();
                const double averageRate = elapsed > 0.0
                    ? static_cast<double>(resolved) / elapsed
                    : 0.0;
                const auto& currentBegin = traceProgressSamples.front();
                const double currentSeconds = std::chrono::duration<double>(
                    now - currentBegin.first).count();
                const double currentRate = currentSeconds > 0.0 &&
                        resolved >= currentBegin.second
                    ? static_cast<double>(resolved - currentBegin.second) /
                        currentSeconds
                    : 0.0;
                const double averageEta = remaining == 0
                    ? 0.0
                    : averageRate > 0.0
                    ? static_cast<double>(remaining) / averageRate
                    : std::numeric_limits<double>::infinity();
                const double currentEta = remaining == 0
                    ? 0.0
                    : currentRate > 0.0
                    ? static_cast<double>(remaining) / currentRate
                    : std::numeric_limits<double>::infinity();
                std::cout << "fiberlet crop attempted=" << current.attemptedAnchors << " accepted=" << current.lines.size()
                          << " covered=" << current.coveredAnchors
                          << " remaining=" << remaining
                          << " elapsed=" << formatProgressDuration(elapsed)
                          << " eta_current="
                          << formatProgressDuration(currentEta)
                          << " eta_avg="
                          << formatProgressDuration(averageEta) << '\n';
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
        if (profileThread.joinable()) {
            profileThread.request_stop();
            profileThread.join();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_trace_chunk: " << error.what() << '\n';
        return 1;
    }
}
