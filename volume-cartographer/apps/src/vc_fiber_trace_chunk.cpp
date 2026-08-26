#include "vc/core/types/Volume.hpp"
#include "vc/fiber_tracer/FiberletChunkGraph.hpp"
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberletCropVisualization.hpp"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

enum class Mode { Trace, Visualize, Constraints, Consensus };

struct Options {
    Mode mode = Mode::Trace;
    std::filesystem::path input;
    std::string normalManifest;
    std::filesystem::path remoteCacheDirectory;
    std::filesystem::path output;
    std::filesystem::path obj;
    std::filesystem::path volume;
    int maximumTextureDimension = 4096;
    int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    std::size_t cacheBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    vc::fiber_tracer::FiberletCropTraceConfig trace;
    vc::fiber_tracer::FiberTraceConstraintConfig constraints;
    vc::fiber_tracer::FiberTraceLabelingConfig labeling;
    bool hasBounds = false;
    bool hasTraceOnlyOption = false;
    bool hasConstraintOnlyOption = false;
    bool hasSolverOnlyOption = false;
    bool hasSharedRuntimeOption = false;
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
                 " [--exclude-parallel-separate-winding] [options]\n\n"
              << "  " << executable
              << " consensus <traces.zarr> --normal-manifest PATH"
                 " [--output BASENAME] [--broken-cost-per-link COST] [options]\n\n"
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
              << "  --lp-relaxation            solve continuous [0,1] label relaxation\n"
              << "  --lp-parallel              request HiGHS parallel LP execution\n"
              << "  --lp-solver NAME           choose, simplex, hipo, or ipm [choose]\n"
              << "  --hv-only                  solve active/broken and H/V only\n"
              << "  --exact-perpendicular-milp exact continuous H/V loss with binary activity\n"
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
    else
        fail("mode must be 'trace', 'visualize', 'constraints', or 'consensus'");
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
        } else if (argument == "--hv-only") {
            options.labeling.hvOnly = true;
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
    if (options.mode == Mode::Constraints || options.mode == Mode::Consensus) {
        if (options.hasTraceOnlyOption)
            fail("constraint processing does not accept trace-only options");
        if (options.mode == Mode::Consensus && options.hasSolverOnlyOption)
            fail("consensus does not accept HiGHS labeling options");
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
        if (options.output.empty()) {
            const std::string stem = options.input.has_extension()
                ? options.input.stem().string()
                : options.input.filename().string();
            options.output = options.input.parent_path() /
                (stem + (options.mode == Mode::Consensus
                             ? "_consensus"
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
    const vc::fiber_tracer::FiberTraceConstraintConfig& config,
    bool manifestMatches,
    double wallSeconds,
    double cpuSeconds)
{
    std::vector<double> distances;
    std::vector<double> parallel;
    std::vector<double> perpendicular;
    std::vector<double> winding;
    for (const auto& constraint : report.constraints) {
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
              << "status  hv_only  objective  orientation_cost  winding_cost  broken_cost  broken_cost_per_link  retained_links  excluded_parallel_separate_winding  requested_mip_gap  variables  integer_variables  rows  mip_nodes  mip_gap  solve_seconds\n"
              << report.modelStatus << "  " << (report.hvOnly ? "true" : "false")
              << "  " << report.objective << "  "
              << report.orientationCost << "  " << report.windingCost << "  "
              << report.brokenCost << "  " << config.brokenCostPerConstraint
              << "  " << report.retainedConstraints << "  "
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

void printRelaxedLabelingReport(
    const vc::fiber_tracer::FiberTraceLabelingReport& report,
    const vc::fiber_tracer::FiberTraceLabelingConfig& config,
    const std::filesystem::path& csv,
    const vc::fiber_tracer::FiberTraceRelaxationObjReport& visualization)
{
    std::cout << std::setprecision(8)
              << "fiber trace labeling continuous values\n"
              << "status  mode  hv_only  requested_solver  requested_parallel  threads  objective  orientation_cost  winding_cost  broken_cost  broken_cost_per_link  retained_links  excluded_parallel_separate_winding  variables  integer_variables  perpendicular_branch_variables  rows  gauge_roots  triangles  triangle_rows  mip_nodes  mip_gap  solve_seconds  csv\n"
              << report.modelStatus << "  "
              << (report.exactPerpendicularMilp
                      ? "exact_perpendicular_milp"
                      : "lp_relaxation")
              << "  " << (report.hvOnly ? "true" : "false")
              << "  " << config.lpSolver << "  "
              << (config.lpParallel ? "on" : "choose") << "  "
              << config.parallelThreads << "  " << report.objective << "  "
              << report.orientationCost << "  " << report.windingCost << "  "
              << report.brokenCost << "  " << config.brokenCostPerConstraint
              << "  " << report.retainedConstraints << "  "
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

struct VisualizationReport {
    vc::fiber_tracer::FiberDirectionClassification directions;
    vc::fiber_tracer::FiberQualityHistogram quality;
};

VisualizationReport visualize(const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines, const std::filesystem::path& output)
{
    std::filesystem::create_directories(output.parent_path().empty() ? std::filesystem::path{"."} : output.parent_path());
    VisualizationReport report;
    report.directions = vc::fiber_tracer::classifyFiberletCropDirections(lines);
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
              << " dominance_fraction=" << vc::fiber_tracer::kFiberDirectionDominanceFraction << " output=" << paths.all << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse(argc, argv);
        if (options.mode == Mode::Constraints ||
            options.mode == Mode::Consensus) {
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
            const auto report = vc::fiber_tracer::extractFiberTraceConstraints(
                artifact.lines,
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
                    options.constraints,
                    manifestMatches,
                    wallSeconds,
                    cpuSeconds);
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
                options.constraints,
                manifestMatches,
                wallSeconds,
                cpuSeconds);
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
            return 0;
        }
        if (options.mode == Mode::Visualize) {
            const auto artifact = vc::fiber_tracer::readFiberletCropTraceArtifact(options.input);
            const auto report = visualize(artifact.lines, options.output);
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
        const auto visualization = visualize(artifact.lines, options.obj);

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
