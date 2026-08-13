#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kLineSegmentLength = 32.0;
using Clock = std::chrono::steady_clock;

struct Options {
    fs::path debugJson;
    fs::path manifest;
    bool reverse = false;
    int maxControls = -1;
    int maxIterations = 1000;
    int segmentsPerSide = 200;
    std::vector<int> threadCounts{1, 2, 4};
    cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
};

struct StageTiming {
    std::string stage;
    int threads = 1;
    int controls = 0;
    int pointsBefore = 0;
    int pointsAfterUpdate = 0;
    int pointsAfterSolve = 0;
    int activeStart = -1;
    int activeEnd = -1;
    double updateMs = 0.0;
    double prefetchPrepMs = 0.0;
    double ceresSolveMs = 0.0;
    double solveOverallMs = 0.0;
    double wallMs = 0.0;
    int iterations = 0;
};

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double norm(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    const double n = norm(v);
    if (n <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finiteDirection(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
           norm(v) > kEpsilon;
}

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " --manifest <manifest.lasagna.json> <line_edit_debug.json> [options]\n"
        << "\n"
        << "Replays VC3D line editing from one side using the control points in a debug JSON.\n"
        << "\n"
        << "Options:\n"
        << "  --reverse                 Start from the highest line_position instead.\n"
        << "  --max-controls N          Replay at most N controls from the ordered list.\n"
        << "  --max-iterations N        Ceres iterations per solve (default 1000).\n"
        << "  --segments-per-side N     Initial/open extension segments (default 200).\n"
        << "  --threads A,B,C           Solver thread counts to replay (default 1,2,4).\n"
        << "  --source-normal x,y,z     VC3D source slice normal for tangent guides (default 0,0,1).\n";
}

int parseInt(const std::string& value, const char* name)
{
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return parsed;
}

cv::Vec3d parseVec3(const std::string& value, const char* name)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    char trailing = '\0';
    if (std::sscanf(value.c_str(), "%lf,%lf,%lf%c", &x, &y, &z, &trailing) != 3) {
        throw std::invalid_argument(std::string(name) + " must be formatted as x,y,z");
    }
    return {x, y, z};
}

std::vector<int> parseThreadCounts(const std::string& value)
{
    std::vector<int> counts;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string token = value.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);
        if (token.empty()) {
            throw std::invalid_argument("--threads must be a comma-separated list of positive integers");
        }
        const int count = parseInt(token, "--threads");
        if (count <= 0) {
            throw std::invalid_argument("--threads values must be positive");
        }
        counts.push_back(count);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    return counts;
}

Options parseArgs(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--manifest") {
            options.manifest = requireValue("--manifest");
        } else if (arg == "--reverse") {
            options.reverse = true;
        } else if (arg == "--max-controls") {
            options.maxControls = parseInt(requireValue("--max-controls"), "--max-controls");
        } else if (arg == "--max-iterations") {
            options.maxIterations = parseInt(requireValue("--max-iterations"), "--max-iterations");
        } else if (arg == "--segments-per-side") {
            options.segmentsPerSide = parseInt(requireValue("--segments-per-side"), "--segments-per-side");
        } else if (arg == "--threads") {
            options.threadCounts = parseThreadCounts(requireValue("--threads"));
        } else if (arg == "--source-normal") {
            options.sourceSliceNormal = parseVec3(requireValue("--source-normal"), "--source-normal");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        } else if (options.debugJson.empty()) {
            options.debugJson = arg;
        } else {
            throw std::invalid_argument("unexpected positional argument: " + arg);
        }
    }

    if (options.debugJson.empty()) {
        throw std::invalid_argument("missing debug JSON path");
    }
    if (options.manifest.empty()) {
        throw std::invalid_argument("missing --manifest; real VC3D replay requires the Lasagna normal source");
    }
    if (options.maxControls == 0 || options.maxControls < -1) {
        throw std::invalid_argument("--max-controls must be positive");
    }
    if (options.maxIterations < 0) {
        throw std::invalid_argument("--max-iterations must be non-negative");
    }
    if (options.segmentsPerSide <= 0) {
        throw std::invalid_argument("--segments-per-side must be positive");
    }
    if (options.threadCounts.empty()) {
        throw std::invalid_argument("--threads must contain at least one value");
    }
    return options;
}

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("expected point as [x, y, z]");
    }
    return {value.at(0).get<double>(),
            value.at(1).get<double>(),
            value.at(2).get<double>()};
}

std::vector<vc::lasagna::LineControlPoint> loadControls(const fs::path& debugJson)
{
    std::ifstream input(debugJson);
    if (!input.good()) {
        throw std::runtime_error("could not open debug JSON: " + debugJson.string());
    }
    nlohmann::json root;
    input >> root;
    if (!root.contains("control_points") || !root.at("control_points").is_array()) {
        throw std::runtime_error("debug JSON does not contain control_points[]");
    }

    std::vector<vc::lasagna::LineControlPoint> controls;
    for (const auto& item : root.at("control_points")) {
        vc::lasagna::LineControlPoint control;
        control.linePosition = item.at("line_position").get<double>();
        control.optimizedIndex = item.value("optimized_index", -1);
        control.isSeed = item.value("is_seed", false);
        control.volumePoint = pointFromJson(item.at("xyz"));
        controls.push_back(control);
    }
    std::stable_sort(controls.begin(),
                     controls.end(),
                     [](const auto& a, const auto& b) {
                         return a.linePosition < b.linePosition;
                     });
    return controls;
}

std::vector<cv::Vec3d> linePointsFromModel(const vc::lasagna::LineModel& line)
{
    std::vector<cv::Vec3d> points;
    points.reserve(line.points.size());
    for (const auto& point : line.points) {
        points.push_back(point.position);
    }
    return points;
}

double nearestLinePosition(const std::vector<cv::Vec3d>& points, const cv::Vec3d& query)
{
    if (points.size() < 2) {
        return 0.0;
    }
    double bestPosition = 0.0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        const cv::Vec3d a = points[i];
        const cv::Vec3d b = points[i + 1];
        const cv::Vec3d ab = b - a;
        const double denom = ab.dot(ab);
        const double t = denom <= kEpsilon
            ? 0.0
            : std::clamp((query - a).dot(ab) / denom, 0.0, 1.0);
        const cv::Vec3d closest = a + ab * t;
        const double distance = norm(query - closest);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = static_cast<double>(i) + t;
        }
    }
    return bestPosition;
}

cv::Vec3d initialTangentForVc3dSideways(
    const cv::Vec3d& sourceSliceNormal,
    const vc::lasagna::NormalSample& seedNormal)
{
    const cv::Vec3d sliceNormal = normalizedOrZero(sourceSliceNormal);
    const cv::Vec3d gtNormal = seedNormal.valid
        ? normalizedOrZero(seedNormal.normal)
        : cv::Vec3d{0.0, 0.0, 0.0};
    if (!finiteDirection(sliceNormal) || !finiteDirection(gtNormal)) {
        return {0.0, 0.0, 0.0};
    }
    return normalizedOrZero(sliceNormal.cross(gtNormal));
}

vc::lasagna::LineOptimizationConfig configLikeVc3d(
    const vc::lasagna::NormalSampler& sampler,
    const cv::Vec3d& seedPoint,
    const Options& options,
    int threads)
{
    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = options.segmentsPerSide;
    config.segmentLength = kLineSegmentLength;
    config.straightnessWeight = 0.1;
    config.tangentStraightnessWeight = 5.0;
    config.samplesPerSegment = 1;
    config.maxIterations = options.maxIterations;
    config.differentiableNormalSampling = true;
    config.initialTangent = initialTangentForVc3dSideways(
        options.sourceSliceNormal,
        sampler.sampleNormal(seedPoint));
    config.useInitialTangent = finiteDirection(config.initialTangent);
    config.tangentGuideVector = normalizedOrZero(options.sourceSliceNormal);
    config.tangentGuideWeight = 1.0;
    config.tangentGuideMode =
        vc::lasagna::LineOptimizationConfig::TangentGuideMode::CrossVectorWithNormal;
    config.printSolverProgress = false;
    config.numThreads = threads;
    return config;
}

std::vector<int> fixedIndicesFromControls(
    const std::vector<vc::lasagna::LineControlPoint>& controls,
    int maxIndex)
{
    std::vector<int> indices;
    indices.reserve(controls.size());
    for (const auto& control : controls) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        indices.push_back(std::clamp(static_cast<int>(std::llround(control.linePosition)),
                                     0,
                                     maxIndex));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

int seedDisplayIndex(const std::vector<vc::lasagna::LineControlPoint>& controls,
                     int fallback,
                     int maxIndex)
{
    for (const auto& control : controls) {
        if (control.isSeed && std::isfinite(control.linePosition)) {
            return std::clamp(static_cast<int>(std::llround(control.linePosition)),
                              0,
                              maxIndex);
        }
    }
    return std::clamp(fallback, 0, maxIndex);
}

void remapControlsToLine(std::vector<vc::lasagna::LineControlPoint>& controls,
                         const vc::lasagna::LineModel& line)
{
    for (auto& control : controls) {
        double bestDistance = std::numeric_limits<double>::infinity();
        int bestIndex = -1;
        for (size_t i = 0; i < line.points.size(); ++i) {
            const double distance = norm(line.points[i].position - control.volumePoint);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(i);
            }
        }
        if (bestIndex >= 0) {
            control.optimizedIndex = bestIndex;
            control.linePosition = static_cast<double>(bestIndex);
            control.volumePoint = line.points[static_cast<size_t>(bestIndex)].position;
        }
    }
}

void printStageHeader()
{
    std::cout << std::left << std::setw(12) << "stage"
              << std::right << std::setw(5) << "thr"
              << std::right << std::setw(6) << "ctrls"
              << std::setw(8) << "pts_in"
              << std::setw(10) << "pts_upd"
              << std::setw(10) << "pts_out"
              << std::setw(9) << "active_s"
              << std::setw(9) << "active_e"
              << std::setw(11) << "update"
              << std::setw(11) << "prefetch"
              << std::setw(11) << "ceres"
              << std::setw(11) << "solve"
              << std::setw(11) << "wall"
              << std::setw(7) << "iters"
              << '\n';
    std::cout << std::string(132, '-') << '\n';
}

void printStage(const StageTiming& stage)
{
    std::cout << std::left << std::setw(12) << stage.stage
              << std::right << std::setw(5) << stage.threads
              << std::right << std::setw(6) << stage.controls
              << std::setw(8) << stage.pointsBefore
              << std::setw(10) << stage.pointsAfterUpdate
              << std::setw(10) << stage.pointsAfterSolve
              << std::setw(9) << stage.activeStart
              << std::setw(9) << stage.activeEnd
              << std::fixed << std::setprecision(3)
              << std::setw(11) << stage.updateMs
              << std::setw(11) << stage.prefetchPrepMs
              << std::setw(11) << stage.ceresSolveMs
              << std::setw(11) << stage.solveOverallMs
              << std::setw(11) << stage.wallMs
              << std::setw(7) << stage.iterations
              << '\n';
    std::cout.flush();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseArgs(argc, argv);
        std::vector<vc::lasagna::LineControlPoint> replayControls =
            loadControls(options.debugJson);
        if (replayControls.empty()) {
            throw std::runtime_error("debug JSON contains no control points");
        }
        if (options.reverse) {
            std::reverse(replayControls.begin(), replayControls.end());
        }
        if (options.maxControls > 0 &&
            options.maxControls < static_cast<int>(replayControls.size())) {
            replayControls.resize(static_cast<size_t>(options.maxControls));
        }

        std::cout.imbue(std::locale::classic());
        std::cout << "debug_json=" << options.debugJson.string() << '\n'
                  << "manifest=" << options.manifest.string() << '\n'
                  << "controls=" << replayControls.size()
                  << " reverse=" << (options.reverse ? "true" : "false")
                  << " max_iterations=" << options.maxIterations
                  << " segments_per_side=" << options.segmentsPerSide
                  << " threads=";
        for (size_t i = 0; i < options.threadCounts.size(); ++i) {
            if (i > 0) {
                std::cout << ',';
            }
            std::cout << options.threadCounts[i];
        }
        std::cout << '\n';
        printStageHeader();

        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(options.manifest);
        for (const int threads : options.threadCounts) {
            vc::lasagna::LasagnaNormalSampler sampler(dataset);
            vc::lasagna::LineOptimizer optimizer(sampler);

            std::cout << "running thread_count=" << threads << " seed solve..." << std::endl;
            std::vector<StageTiming> timings;
            const auto benchmarkStart = Clock::now();

            std::vector<vc::lasagna::LineControlPoint> activeControls;
            activeControls.push_back({0.0, replayControls.front().volumePoint, true, -1});
            const auto seedStageStart = Clock::now();
            vc::lasagna::LineOptimizationConfig config =
                configLikeVc3d(sampler, activeControls.front().volumePoint, options, threads);
            vc::lasagna::LineOptimizationResult result =
                optimizer.optimizeFromControlPoints(activeControls, config);
            remapControlsToLine(activeControls, result.line);
            StageTiming seedTiming{
                "seed",
                threads,
                static_cast<int>(activeControls.size()),
                0,
                0,
                static_cast<int>(result.line.points.size()),
                0,
                static_cast<int>(result.line.points.empty() ? 0 : result.line.points.size() - 1),
                0.0,
                result.report.normalChunkPrefetchMs + result.report.normalMaterializeMs,
                result.report.ceresSolveMs,
                result.report.totalMs,
                elapsedMs(seedStageStart, Clock::now()),
                result.report.iterations,
            };
            timings.push_back(seedTiming);
            printStage(seedTiming);

            for (size_t i = 1; i < replayControls.size(); ++i) {
                std::cout << "running thread_count=" << threads
                          << " add_" << i
                          << " controls=" << (i + 1) << "..." << std::endl;
                const auto stageStart = Clock::now();
                std::vector<cv::Vec3d> currentLinePoints = linePointsFromModel(result.line);
                const int pointsBefore = static_cast<int>(currentLinePoints.size());
                vc::lasagna::LineControlPoint next;
                next.volumePoint = replayControls[i].volumePoint;
                next.linePosition = nearestLinePosition(currentLinePoints, next.volumePoint);
                next.isSeed = false;
                next.optimizedIndex = -1;
                activeControls.push_back(next);
                const size_t changedControlIndex = activeControls.size() - 1;

                const auto updateStart = Clock::now();
                vc::lasagna::LineControlPointUpdateResult update =
                    vc::lasagna::updateExistingLineControlPoint(std::move(currentLinePoints),
                                                                std::move(activeControls),
                                                                changedControlIndex,
                                                                sampler,
                                                                config);
                const double updateMs = elapsedMs(updateStart, Clock::now());
                activeControls = std::move(update.controlPoints);

                const int maxIndex = static_cast<int>(update.linePoints.size()) - 1;
                const std::vector<int> fixedIndices = fixedIndicesFromControls(activeControls, maxIndex);
                const int displayIndex = seedDisplayIndex(activeControls,
                                                          maxIndex / 2,
                                                          maxIndex);
                result = optimizer.optimizeExistingLine(std::move(update.linePoints),
                                                        fixedIndices,
                                                        displayIndex,
                                                        config,
                                                        update.activeStart,
                                                        update.activeEnd,
                                                        "bench-existing-line+local");
                remapControlsToLine(activeControls, result.line);

                StageTiming addTiming{
                    "add_" + std::to_string(i),
                    threads,
                    static_cast<int>(activeControls.size()),
                    pointsBefore,
                    maxIndex + 1,
                    static_cast<int>(result.line.points.size()),
                    update.activeStart,
                    update.activeEnd,
                    updateMs,
                    result.report.normalChunkPrefetchMs + result.report.normalMaterializeMs,
                    result.report.ceresSolveMs,
                    result.report.totalMs,
                    elapsedMs(stageStart, Clock::now()),
                    result.report.iterations,
                };
                timings.push_back(addTiming);
                printStage(addTiming);
            }

            const double totalWallMs = elapsedMs(benchmarkStart, Clock::now());
            double totalUpdateMs = 0.0;
            double totalPrefetchPrepMs = 0.0;
            double totalCeresMs = 0.0;
            double totalSolveOverallMs = 0.0;
            for (const auto& timing : timings) {
                totalUpdateMs += timing.updateMs;
                totalPrefetchPrepMs += timing.prefetchPrepMs;
                totalCeresMs += timing.ceresSolveMs;
                totalSolveOverallMs += timing.solveOverallMs;
            }

            std::cout << std::fixed << std::setprecision(3)
                      << "summary threads=" << threads
                      << " total_wall_ms=" << totalWallMs
                      << " total_update_ms=" << totalUpdateMs
                      << " total_prefetch_prep_ms=" << totalPrefetchPrepMs
                      << " total_ceres_solve_ms=" << totalCeresMs
                      << " total_solve_overall_ms=" << totalSolveOverallMs
                      << " final_points=" << result.line.points.size()
                      << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "bench_lasagna_line_edit: " << ex.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
    return 0;
}
