#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEpsilon = 1.0e-12;

cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    const double n = std::sqrt(v.dot(v));
    if (n <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

bool finiteDirection(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
           std::sqrt(v.dot(v)) > kEpsilon;
}

void printUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " <manifest.lasagna.json> [--constant-normal-jacobian] [--benchmark-solvers] [--benchmark-threads] [--trace-init] [--segments-per-side=N] [--seed=x,y,z]\n"
              << "Runs line annotation optimization at seed "
              << "[17955,15141,37891] with initial z-axis mode.\n";
}

cv::Vec3d parseSeed(const std::string& value)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    char trailing = '\0';
    if (std::sscanf(value.c_str(), "%lf,%lf,%lf%c", &x, &y, &z, &trailing) != 3) {
        throw std::invalid_argument("seed must be formatted as x,y,z");
    }
    return {x, y, z};
}

int parsePositiveInt(const std::string& value, const char* name)
{
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return parsed;
}

cv::Vec3d initialZInOutTangent(const cv::Vec3d& sourceSliceNormal,
                               const vc::lasagna::NormalSample& seedNormal)
{
    const cv::Vec3d sliceNormal = normalizedOrZero(sourceSliceNormal);
    const cv::Vec3d gtNormal = seedNormal.valid
        ? normalizedOrZero(seedNormal.normal)
        : cv::Vec3d{0.0, 0.0, 0.0};
    if (!finiteDirection(sliceNormal) || !finiteDirection(gtNormal)) {
        return {0.0, 0.0, 0.0};
    }
    return normalizedOrZero(sliceNormal - gtNormal * sliceNormal.dot(gtNormal));
}

void printLosses(const vc::lasagna::LineOptimizationReport& report)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Final loss breakdown:\n"
              << "term                 n      weight    raw_cost weighted_cost\n";
    for (const auto& loss : report.finalLosses) {
        std::cout << std::left << std::setw(18) << loss.name
                  << std::right << std::setw(6) << loss.residuals
                  << std::setw(12) << loss.weight
                  << std::setw(12) << loss.rawCost
                  << std::setw(14) << loss.weightedCost
                  << '\n';
    }
}

const char* solverName(vc::lasagna::LineOptimizationConfig::LinearSolver solver)
{
    using LinearSolver = vc::lasagna::LineOptimizationConfig::LinearSolver;
    switch (solver) {
    case LinearSolver::DenseQR:
        return "dense_qr";
    case LinearSolver::DenseNormalCholesky:
        return "dense_normal_cholesky";
    case LinearSolver::SparseNormalCholesky:
        return "sparse_normal_cholesky";
    case LinearSolver::DenseSchur:
        return "dense_schur";
    case LinearSolver::SparseSchur:
        return "sparse_schur";
    case LinearSolver::IterativeSchur:
        return "iterative_schur";
    case LinearSolver::CGNR:
        return "cgnr";
    }
    return "unknown";
}

double normalAlignmentCost(const vc::lasagna::LineOptimizationReport& report)
{
    for (const auto& loss : report.finalLosses) {
        if (loss.name == "normal_alignment") {
            return loss.weightedCost;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void printLineViewDiagnostics(const vc::lasagna::LineModel& line)
{
    const auto diagnostics = vc::lasagna::diagnoseLineViewFrames(line);
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Line view frame diagnostics:\n"
              << "frames=" << diagnostics.frameCount
              << " issues=" << diagnostics.issues.size()
              << " max_abs_roll_delta_rad=" << diagnostics.maxAbsRollDeltaRadians
              << " min_normal_dot=" << diagnostics.minNormalContinuityDot
              << " min_side_dot=" << diagnostics.minSideContinuityDot
              << " min_sampled_axis_dot=" << diagnostics.minSampledAxisContinuityDot
              << " min_mesh_to_sampled_axis_dot=" << diagnostics.minMeshToSampledAxisDot
              << " max_abs_display_up_roll_delta_rad=" << diagnostics.maxAbsDisplayUpRollDeltaRadians
              << " min_display_up_dot=" << diagnostics.minDisplayUpContinuityDot
              << '\n';
    if (!diagnostics.issues.empty()) {
        std::cout << "idx     roll_delta  normal_dot    side_dot sampled_axis mesh_sampled_axis display_roll display_up reason\n";
        for (const auto& issue : diagnostics.issues) {
            std::cout << std::setw(3) << issue.index
                      << std::setw(15) << issue.rollDeltaRadians
                      << std::setw(12) << issue.normalContinuityDot
                      << std::setw(12) << issue.sideContinuityDot
                      << std::setw(13) << issue.sampledAxisContinuityDot
                      << std::setw(18) << issue.meshToSampledAxisDot
                      << std::setw(13) << issue.displayUpRollDeltaRadians
                      << std::setw(11) << issue.displayUpContinuityDot
                      << ' ' << issue.reason << '\n';
        }
    }
}

bool isSchurSolver(vc::lasagna::LineOptimizationConfig::LinearSolver solver)
{
    using LinearSolver = vc::lasagna::LineOptimizationConfig::LinearSolver;
    return solver == LinearSolver::DenseSchur ||
           solver == LinearSolver::SparseSchur ||
           solver == LinearSolver::IterativeSchur;
}

cv::Vec3d projectDirectionToNormalPlane(const cv::Vec3d& direction,
                                        const cv::Vec3d& normal)
{
    cv::Vec3d projected = direction - normal * direction.dot(normal);
    projected = normalizedOrZero(projected);
    const cv::Vec3d normalizedDirection = normalizedOrZero(direction);
    if (finiteDirection(projected) &&
        finiteDirection(normalizedDirection) &&
        projected.dot(normalizedDirection) < 0.0) {
        projected *= -1.0;
    }
    return finiteDirection(projected) ? projected : normalizedDirection;
}

cv::Vec3d rotateAroundAxis(const cv::Vec3d& vector,
                           const cv::Vec3d& axis,
                           double angle)
{
    const cv::Vec3d unitAxis = normalizedOrZero(axis);
    if (!finiteDirection(unitAxis)) {
        return vector;
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return vector * c + unitAxis.cross(vector) * s +
           unitAxis * (unitAxis.dot(vector) * (1.0 - c));
}

double angleDegrees(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const cv::Vec3d an = normalizedOrZero(a);
    const cv::Vec3d bn = normalizedOrZero(b);
    if (!finiteDirection(an) || !finiteDirection(bn)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::acos(std::clamp(an.dot(bn), -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
}

void printInitTraceDirection(const char* label,
                             int sign,
                             const cv::Vec3d& seedPoint,
                             const cv::Vec3d& seedTangent,
                             const vc::lasagna::NormalSampler& sampler,
                             const vc::lasagna::LineOptimizationConfig& config)
{
    cv::Vec3d point = seedPoint;
    cv::Vec3d direction = normalizedOrZero(seedTangent) * static_cast<double>(sign);
    vc::lasagna::NormalSample previousSample = sampler.sampleNormal(point);
    cv::Vec3d previousNormal = normalizedOrZero(previousSample.normal);
    if (!previousSample.valid || !finiteDirection(previousNormal)) {
        previousNormal = {0.0, 0.0, 0.0};
    }

    std::cout << "Init trace " << label << ":\n"
              << "step raw_normal_dot flip normal_angle_deg dir_turn_deg pred_to_actual normal_dot step_end[x,y,z]\n";
    for (int step = 1; step <= config.segmentsPerSide; ++step) {
        const cv::Vec3d oldDirection = direction;
        const cv::Vec3d predicted = point + oldDirection * config.segmentLength;
        const auto sample = sampler.sampleNormal(predicted);
        cv::Vec3d normal = normalizedOrZero(sample.normal);
        const bool validNormal = sample.valid && finiteDirection(normal);

        double rawNormalDot = std::numeric_limits<double>::quiet_NaN();
        bool flipped = false;
        double normalAngleDeg = 0.0;
        if (validNormal && finiteDirection(previousNormal)) {
            rawNormalDot = previousNormal.dot(normal);
            const auto transportedFor = [&](const cv::Vec3d& candidateNormal) {
                cv::Vec3d candidateDirection = oldDirection;
                const cv::Vec3d axis = previousNormal.cross(candidateNormal);
                const double sinAngle = std::sqrt(axis.dot(axis));
                const double cosAngle = std::clamp(previousNormal.dot(candidateNormal), -1.0, 1.0);
                if (sinAngle > kEpsilon) {
                    candidateDirection = rotateAroundAxis(oldDirection,
                                                          axis,
                                                          std::atan2(sinAngle, cosAngle));
                }
                return projectDirectionToNormalPlane(candidateDirection, candidateNormal);
            };
            const cv::Vec3d sameDirection = transportedFor(normal);
            const cv::Vec3d flippedNormal = normal * -1.0;
            const cv::Vec3d flippedDirection = transportedFor(flippedNormal);
            if (flippedDirection.dot(oldDirection) > sameDirection.dot(oldDirection)) {
                normal = flippedNormal;
                direction = flippedDirection;
                flipped = true;
            } else {
                direction = sameDirection;
            }
            normalAngleDeg = angleDegrees(previousNormal, normal);
            previousNormal = normal;
        } else if (validNormal) {
            direction = projectDirectionToNormalPlane(direction, normal);
            previousNormal = normal;
        }

        const cv::Vec3d next = point + direction * config.segmentLength;
        const double dirTurnDeg = angleDegrees(oldDirection, direction);
        const double predToActual = std::sqrt((next - predicted).dot(next - predicted));
        const double normalDot = validNormal ? direction.dot(normal) : std::numeric_limits<double>::quiet_NaN();
        std::cout << std::setw(4) << step
                  << std::setw(15) << rawNormalDot
                  << std::setw(5) << (flipped ? "yes" : "no")
                  << std::setw(17) << normalAngleDeg
                  << std::setw(13) << dirTurnDeg
                  << std::setw(15) << predToActual
                  << std::setw(11) << normalDot
                  << " [" << next[0] << ", " << next[1] << ", " << next[2] << "]\n";
        point = next;
    }
}

void printInitTrace(const cv::Vec3d& seedPoint,
                    const cv::Vec3d& seedTangent,
                    const vc::lasagna::NormalSampler& sampler,
                    const vc::lasagna::LineOptimizationConfig& config)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3);
    printInitTraceDirection("forward", 1, seedPoint, seedTangent, sampler, config);
    printInitTraceDirection("backward", -1, seedPoint, seedTangent, sampler, config);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 2;
    }
    bool differentiableNormalSampling = true;
    bool benchmarkSolvers = false;
    bool benchmarkThreads = false;
    bool traceInit = false;
    int segmentsPerSide = 200;
    cv::Vec3d seedPoint{17955.0, 15141.0, 37891.0};
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--constant-normal-jacobian") {
            differentiableNormalSampling = false;
        } else if (arg == "--benchmark-solvers") {
            benchmarkSolvers = true;
        } else if (arg == "--benchmark-threads") {
            benchmarkThreads = true;
        } else if (arg == "--trace-init") {
            traceInit = true;
        } else if (arg.rfind("--segments-per-side=", 0) == 0) {
            segmentsPerSide = parsePositiveInt(arg.substr(20), "segments-per-side");
        } else if (arg.rfind("--seed=", 0) == 0) {
            seedPoint = parseSeed(arg.substr(7));
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    try {
        const std::filesystem::path manifestPath = argv[1];
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        vc::lasagna::LineOptimizer optimizer(sampler);

        const cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
        const auto seedNormal = sampler.sampleNormal(seedPoint);

        vc::lasagna::LineOptimizationConfig config;
        config.segmentsPerSide = segmentsPerSide;
        config.segmentLength = 32.0;
        config.straightnessWeight = 0.1;
        config.tangentStraightnessWeight = 5.0;
        config.normalStraightnessWeight = 0.005;
        config.samplesPerSegment = 1;
        config.maxIterations = 1000;
        config.differentiableNormalSampling = differentiableNormalSampling;
        config.initialTangent = initialZInOutTangent(sourceSliceNormal, seedNormal);
        config.useInitialTangent = finiteDirection(config.initialTangent);
        config.tangentGuideVector = normalizedOrZero(sourceSliceNormal);
        config.tangentGuideWeight = 1.0;
        config.tangentGuideMode =
            vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;

        std::cout << "Seed: [" << seedPoint[0] << ", " << seedPoint[1] << ", " << seedPoint[2] << "]\n";
        std::cout << "Source direction: [0, 0, 1]\n";
        std::cout << "Seed normal valid=" << seedNormal.valid
                  << " normal=[" << seedNormal.normal[0]
                  << ", " << seedNormal.normal[1]
                  << ", " << seedNormal.normal[2] << "]\n";
        std::cout << "Initial tangent valid=" << config.useInitialTangent
                  << " tangent=[" << config.initialTangent[0]
                  << ", " << config.initialTangent[1]
                  << ", " << config.initialTangent[2] << "]\n";
        std::cout << "Differentiable normal sampling=" << config.differentiableNormalSampling << "\n";
        if (traceInit) {
            printInitTrace(seedPoint, config.initialTangent, sampler, config);
        }

        if (benchmarkThreads) {
            const std::vector<int> threadCounts{1, 2, 4, 8, 16, 32};
            std::cout.imbue(std::locale::classic());
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "Thread benchmark:\n"
                      << "threads   run       ms  ceres_ms prefetch_ms materialize_ms prefetch_calls chunks_read chunks_requested  iters   final_cost normal_cost status\n";
            for (const int threads : threadCounts) {
                auto trialConfig = config;
                trialConfig.numThreads = threads;
                trialConfig.printSolverProgress = false;
                (void)optimizer.optimizeFromSeed(seedPoint, trialConfig);
                for (int run = 1; run <= 3; ++run) {
                    const auto start = std::chrono::steady_clock::now();
                    const auto result = optimizer.optimizeFromSeed(seedPoint, trialConfig);
                    const auto end = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
                    std::cout << std::right << std::setw(7) << threads
                              << std::setw(6) << run
                              << std::setw(10) << ms
                              << std::setw(10) << result.report.ceresSolveMs
                              << std::setw(12) << result.report.normalChunkPrefetchMs
                              << std::setw(14) << result.report.normalMaterializeMs
                              << std::setw(15) << result.report.normalPrefetchCalls
                              << std::setw(12) << result.report.normalPrefetchChunksRead
                              << std::setw(17) << result.report.normalPrefetchRequestedChunks
                              << std::setw(7) << result.report.iterations
                              << std::setw(13) << result.report.finalCost
                              << std::setw(12) << normalAlignmentCost(result.report)
                              << " " << (result.report.converged ? "ok" : "not_converged") << '\n';
                }
            }
            return 0;
        }

        if (benchmarkSolvers) {
            using LinearSolver = vc::lasagna::LineOptimizationConfig::LinearSolver;
            const std::vector<LinearSolver> solvers{
                LinearSolver::DenseQR,
                LinearSolver::DenseNormalCholesky,
                LinearSolver::SparseNormalCholesky,
                LinearSolver::CGNR,
                LinearSolver::DenseSchur,
                LinearSolver::SparseSchur,
                LinearSolver::IterativeSchur,
            };
            const std::vector<int> threadCounts{1, 2, 4, 8};
            std::cout.imbue(std::locale::classic());
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "Solver benchmark:\n"
                      << "solver                    threads   run       ms  ceres_ms prefetch_ms materialize_ms prefetch_calls chunks_read chunks_requested  iters   final_cost normal_cost status\n";
            for (const auto solver : solvers) {
                for (const int threads : threadCounts) {
                    if (isSchurSolver(solver)) {
                        std::cout << std::left << std::setw(25) << solverName(solver)
                                  << std::right << std::setw(7) << threads
                                  << "    --        --        --          --            --              --          --               --     --          --          -- unsupported_residual_graph\n";
                        continue;
                    }
                    auto trialConfig = config;
                    trialConfig.linearSolver = solver;
                    trialConfig.numThreads = threads;
                    trialConfig.printSolverProgress = false;
                    for (int run = 1; run <= 2; ++run) {
                        const auto start = std::chrono::steady_clock::now();
                        const auto result = optimizer.optimizeFromSeed(seedPoint, trialConfig);
                        const auto end = std::chrono::steady_clock::now();
                        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
                        std::cout << std::left << std::setw(25) << solverName(solver)
                                  << std::right << std::setw(7) << threads
                                  << std::setw(6) << (run == 1 ? "cold" : "warm")
                                  << std::setw(10) << ms
                                  << std::setw(10) << result.report.ceresSolveMs
                                  << std::setw(12) << result.report.normalChunkPrefetchMs
                                  << std::setw(14) << result.report.normalMaterializeMs
                                  << std::setw(15) << result.report.normalPrefetchCalls
                                  << std::setw(12) << result.report.normalPrefetchChunksRead
                                  << std::setw(17) << result.report.normalPrefetchRequestedChunks
                                  << std::setw(7) << result.report.iterations
                                  << std::setw(13) << result.report.finalCost
                                  << std::setw(12) << normalAlignmentCost(result.report)
                                  << " " << (result.report.converged ? "ok" : "not_converged") << '\n';
                    }
                }
            }
            return 0;
        }

        const auto result = optimizer.optimizeFromSeed(seedPoint, config);

        std::cout << "Optimization complete: points=" << result.line.points.size()
                  << " iterations=" << result.report.iterations
                  << " initial_cost=" << result.report.initialCost
                  << " final_cost=" << result.report.finalCost
                  << " valid_normals=" << result.report.validNormalSamples
                  << " invalid_normals=" << result.report.invalidNormalSamples
                  << " converged=" << result.report.converged << "\n";
        if (!result.report.message.empty()) {
            std::cout << result.report.message << '\n';
        }
        printLosses(result.report);
        printLineViewDiagnostics(result.line);
    } catch (const std::exception& ex) {
        std::cerr << "vc_lasagna_line_probe failed: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
