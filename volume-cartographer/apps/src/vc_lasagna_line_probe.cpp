#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr int kTiffCompressionNone = 1;

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
    std::cerr << "Usage: " << argv0 << " <manifest.lasagna.json> [--constant-normal-jacobian] [--benchmark-solvers] [--benchmark-threads] [--trace-init] [--segments-per-side=N] [--seed=x,y,z] [--verbose]\n"
              << "       " << argv0 << " <manifest.lasagna.json> --fiber <fiber.json> [--reopt|--reinit-reopt] [--output <fiber.json>] [--obj-output-dir <dir>] [--reinit-debug-obj-output-dir <dir>] [--strip-output-dir <dir>] [--texture-zarr <zarr>] [--texture-level N] [--strip-render-scale N] [--constant-normal-jacobian] [--verbose]\n"
              << "Runs line annotation optimization at seed "
              << "[17955,15141,37891] with initial z-axis mode, or loads/reoptimizes a saved VC3D fiber.\n";
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

int parseNonNegativeInt(const std::string& value, const char* name)
{
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    if (consumed != value.size() || parsed < 0) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    return parsed;
}

struct FiberInput {
    nlohmann::json root;
    std::vector<cv::Vec3d> controlPoints;
    std::vector<cv::Vec3d> linePoints;
};

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("point must be [x, y, z]");
    }
    cv::Vec3d point{
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>(),
    };
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        throw std::runtime_error("point contains non-finite coordinates");
    }
    return point;
}

std::vector<cv::Vec3d> readPointArray(const nlohmann::json& root, const char* key)
{
    if (!root.contains(key) || !root.at(key).is_array()) {
        throw std::runtime_error(std::string("fiber JSON is missing array ") + key);
    }
    std::vector<cv::Vec3d> points;
    points.reserve(root.at(key).size());
    for (const auto& value : root.at(key)) {
        points.push_back(pointFromJson(value));
    }
    return points;
}

FiberInput loadFiberInput(const std::filesystem::path& fiberPath)
{
    std::ifstream input(fiberPath);
    if (!input.good()) {
        throw std::runtime_error("could not open fiber JSON: " + fiberPath.string());
    }
    FiberInput fiber;
    fiber.root = nlohmann::json::parse(input);
    if (fiber.root.value("type", std::string{}) != "vc3d_fiber") {
        throw std::runtime_error("fiber JSON type is not vc3d_fiber");
    }
    if (fiber.root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported vc3d_fiber version");
    }

    fiber.controlPoints = readPointArray(fiber.root, "control_points");
    fiber.linePoints = readPointArray(fiber.root, "line_points");
    if (fiber.controlPoints.empty()) {
        throw std::runtime_error("fiber JSON has no control_points");
    }
    if (fiber.linePoints.size() < 2) {
        throw std::runtime_error("fiber JSON needs at least two line_points for reoptimization");
    }
    return fiber;
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

void saveReoptimizedFiber(const FiberInput& fiber,
                          const std::vector<cv::Vec3d>& optimizedLinePoints,
                          const std::filesystem::path& outputPath)
{
    if (outputPath.empty()) {
        throw std::runtime_error("output path is empty");
    }
    nlohmann::json root = fiber.root;
    root["line_points"] = nlohmann::json::array();
    for (const auto& point : optimizedLinePoints) {
        root["line_points"].push_back(pointToJson(point));
    }
    root["generation"] = std::max<uint64_t>(uint64_t{1},
                                            root.value("generation", uint64_t{1})) +
                         uint64_t{1};

    const std::filesystem::path parent = outputPath.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("failed to create output directory " +
                                     parent.string() + ": " + ec.message());
        }
    }

    const std::filesystem::path tempPath = outputPath.string() + ".tmp";
    {
        std::ofstream output(tempPath);
        if (!output.good()) {
            throw std::runtime_error("could not open output temp file: " + tempPath.string());
        }
        output << root.dump(2) << '\n';
    }

    std::filesystem::rename(tempPath, outputPath, ec);
    if (ec) {
        std::filesystem::remove(tempPath);
        throw std::runtime_error("failed to replace output file " +
                                 outputPath.string() + ": " + ec.message());
    }
}

double distance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const cv::Vec3d delta = a - b;
    return std::sqrt(delta.dot(delta));
}

int nearestLinePointIndex(const std::vector<cv::Vec3d>& linePoints,
                          const cv::Vec3d& point)
{
    if (linePoints.empty()) {
        return -1;
    }
    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < linePoints.size(); ++i) {
        const double candidate = distance(linePoints[i], point);
        if (candidate < bestDistance) {
            bestDistance = candidate;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

int middleControlIndex(const std::vector<cv::Vec3d>& controlPoints,
                       const std::vector<cv::Vec3d>& linePoints)
{
    const double center = static_cast<double>(linePoints.size() - 1) * 0.5;
    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        const int lineIndex = nearestLinePointIndex(linePoints, controlPoints[i]);
        const double candidate = std::abs(static_cast<double>(lineIndex) - center);
        if (candidate < bestDistance) {
            bestDistance = candidate;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

std::vector<int> fixedIndicesForControls(const FiberInput& fiber)
{
    std::vector<int> fixedIndices;
    fixedIndices.reserve(fiber.controlPoints.size());
    for (const auto& control : fiber.controlPoints) {
        fixedIndices.push_back(nearestLinePointIndex(fiber.linePoints, control));
    }
    std::sort(fixedIndices.begin(), fixedIndices.end());
    fixedIndices.erase(std::unique(fixedIndices.begin(), fixedIndices.end()),
                       fixedIndices.end());
    return fixedIndices;
}

double lineLength(const std::vector<cv::Vec3d>& points)
{
    double length = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        length += distance(points[i - 1], points[i]);
    }
    return length;
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

vc::lasagna::LineModel makeLineModelFromPoints(
    const std::vector<cv::Vec3d>& points,
    const vc::lasagna::NormalSampler& sampler,
    int displayFrameAnchorIndex)
{
    vc::lasagna::LineModel model;
    model.displayFrameAnchorIndex = displayFrameAnchorIndex;
    model.points.reserve(points.size());
    for (const auto& point : points) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = point;
        linePoint.sampledNormal = sampler.sampleNormal(point);
        linePoint.sampledNormal.normal = normalizedOrZero(linePoint.sampledNormal.normal);
        linePoint.sampledNormal.valid =
            linePoint.sampledNormal.valid && finiteDirection(linePoint.sampledNormal.normal);
        linePoint.valid = linePoint.sampledNormal.valid;
        model.points.push_back(std::move(linePoint));
    }
    return model;
}

struct PointMotionStats {
    size_t compared = 0;
    size_t inputPoints = 0;
    size_t outputPoints = 0;
    double min = 0.0;
    double avg = 0.0;
    double max = 0.0;
};

PointMotionStats pointMotionStats(const std::vector<cv::Vec3d>& inputPoints,
                                  const std::vector<cv::Vec3d>& outputPoints)
{
    PointMotionStats stats;
    stats.inputPoints = inputPoints.size();
    stats.outputPoints = outputPoints.size();
    stats.compared = std::min(inputPoints.size(), outputPoints.size());
    if (stats.compared == 0) {
        return stats;
    }

    stats.min = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (size_t i = 0; i < stats.compared; ++i) {
        const double motion = distance(inputPoints[i], outputPoints[i]);
        stats.min = std::min(stats.min, motion);
        stats.max = std::max(stats.max, motion);
        sum += motion;
    }
    stats.avg = sum / static_cast<double>(stats.compared);
    return stats;
}

PointMotionStats pointMotionStatsForRange(const std::vector<cv::Vec3d>& inputPoints,
                                          const std::vector<cv::Vec3d>& outputPoints,
                                          size_t start,
                                          size_t endInclusive)
{
    PointMotionStats stats;
    stats.inputPoints = inputPoints.size();
    stats.outputPoints = outputPoints.size();
    const size_t compared = std::min(inputPoints.size(), outputPoints.size());
    if (compared == 0 || start >= compared) {
        return stats;
    }
    endInclusive = std::min(endInclusive, compared - 1);
    if (endInclusive < start) {
        return stats;
    }

    stats.compared = endInclusive - start + 1;
    stats.min = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (size_t i = start; i <= endInclusive; ++i) {
        const double motion = distance(inputPoints[i], outputPoints[i]);
        stats.min = std::min(stats.min, motion);
        stats.max = std::max(stats.max, motion);
        sum += motion;
    }
    stats.avg = sum / static_cast<double>(stats.compared);
    return stats;
}

void printPointMotionStats(const PointMotionStats& stats)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Point motion: compared=" << stats.compared
              << " input_points=" << stats.inputPoints
              << " output_points=" << stats.outputPoints
              << " avg=" << stats.avg
              << " min=" << stats.min
              << " max=" << stats.max << '\n';
}

void printSegmentMotionTable(const std::vector<cv::Vec3d>& inputPoints,
                             const std::vector<cv::Vec3d>& outputPoints,
                             std::vector<int> fixedIndices)
{
    const size_t compared = std::min(inputPoints.size(), outputPoints.size());
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Segment motion:\n"
              << "segment              start      end   points          avg          min          max\n";
    if (compared == 0) {
        return;
    }

    fixedIndices.erase(std::remove_if(fixedIndices.begin(),
                                      fixedIndices.end(),
                                      [compared](int index) {
                                          return index < 0 ||
                                                 static_cast<size_t>(index) >= compared;
                                      }),
                       fixedIndices.end());
    std::sort(fixedIndices.begin(), fixedIndices.end());
    fixedIndices.erase(std::unique(fixedIndices.begin(), fixedIndices.end()),
                       fixedIndices.end());

    auto printRow = [&](const std::string& label, size_t start, size_t endInclusive) {
        const PointMotionStats stats =
            pointMotionStatsForRange(inputPoints, outputPoints, start, endInclusive);
        std::cout << std::left << std::setw(18) << label
                  << std::right << std::setw(8) << start
                  << std::setw(9) << endInclusive
                  << std::setw(9) << stats.compared
                  << std::setw(13) << stats.avg
                  << std::setw(13) << stats.min
                  << std::setw(13) << stats.max << '\n';
    };

    if (fixedIndices.empty()) {
        printRow("whole_line", 0, compared - 1);
        return;
    }

    const size_t first = static_cast<size_t>(fixedIndices.front());
    if (first > 0) {
        printRow("open_left", 0, first);
    }
    for (size_t i = 0; i + 1 < fixedIndices.size(); ++i) {
        const size_t start = static_cast<size_t>(fixedIndices[i]);
        const size_t end = static_cast<size_t>(fixedIndices[i + 1]);
        printRow("control_" + std::to_string(i) + "_" + std::to_string(i + 1),
                 start,
                 end);
    }
    const size_t last = static_cast<size_t>(fixedIndices.back());
    if (last + 1 < compared) {
        printRow("open_right", last, compared - 1);
    }
}

const vc::lasagna::LineReinitializationCandidateReport* spanCandidateReport(
    const vc::lasagna::LineReinitializationSpanReport& span,
    const char* name)
{
    for (const auto& candidate : span.candidates) {
        if (candidate.name == name) {
            return &candidate;
        }
    }
    return nullptr;
}

double initialRmsForCandidate(const vc::lasagna::LineReinitializationSpanReport& span,
                              const char* name)
{
    const auto* candidate = spanCandidateReport(span, name);
    return candidate ? candidate->initialRms : 0.0;
}

double finalRmsForCandidate(const vc::lasagna::LineReinitializationSpanReport& span,
                            const char* name)
{
    const auto* candidate = spanCandidateReport(span, name);
    return candidate ? candidate->finalRms : 0.0;
}

void printReinitSpanTable(const std::vector<vc::lasagna::LineReinitializationSpanReport>& spans)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(1);
    std::cout << "Reinit segment candidates:\n"
              << std::right
              << std::setw(3) << "seg"
              << std::setw(4) << "lcp"
              << std::setw(4) << "rcp"
              << std::setw(4) << "pts"
              << std::setw(5) << "lsgn"
              << std::setw(8) << "lnear"
              << std::setw(5) << "rsgn"
              << std::setw(8) << "rnear"
              << std::setw(8) << "clnear"
              << std::setw(8) << "crnear"
              << std::setw(9) << "lirms"
              << std::setw(9) << "lfrms"
              << std::setw(9) << "rirms"
              << std::setw(9) << "rfrms"
              << std::setw(9) << "clfrms"
              << std::setw(9) << "crfrms"
              << std::setw(8) << "mxstep"
              << std::setw(8) << "mxtan"
              << std::setw(8) << "mxnorm"
              << std::setw(8) << "mxalign"
              << ' ' << "pick" << '\n';
    for (const auto& span : spans) {
        std::cout << std::right << std::setw(3) << span.segmentIndex
                  << std::setw(4) << span.leftControlIndex
                  << std::setw(4) << span.rightControlIndex
                  << std::setw(4) << span.points
                  << std::setw(5) << span.candLeftSelectedSign
                  << std::setw(8) << span.candLeftClosestTargetDistance
                  << std::setw(5) << span.candRightSelectedSign
                  << std::setw(8) << span.candRightClosestTargetDistance
                  << std::setw(8) << span.candContinueLeftClosestTargetDistance
                  << std::setw(8) << span.candContinueRightClosestTargetDistance
                  << std::setw(9) << initialRmsForCandidate(span, "left")
                  << std::setw(9) << finalRmsForCandidate(span, "left")
                  << std::setw(9) << initialRmsForCandidate(span, "right")
                  << std::setw(9) << finalRmsForCandidate(span, "right")
                  << std::setw(9) << finalRmsForCandidate(span, "continue-left")
                  << std::setw(9) << finalRmsForCandidate(span, "continue-right")
                  << std::setw(8) << span.chosenMaxEvenStepDeviation
                  << std::setw(8) << span.chosenMaxTangentSmoothDeviation
                  << std::setw(8) << span.chosenMaxNormalSmoothDeviation
                  << std::setw(8) << span.chosenMaxNormalAlignmentAbs
                  << ' ' << span.chosen << '\n';
    }
}

void printReinitCandidateTable(const std::vector<vc::lasagna::LineReinitializationSpanReport>& spans)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Reinit candidate details (pick=min avg(mean,p95,max) alignment):\n"
              << std::right
              << std::setw(3) << "seg"
              << std::setw(4) << "lcp"
              << std::setw(4) << "rcp"
              << ' ' << std::left << std::setw(15) << "cand"
              << std::right
              << std::setw(5) << "sgn"
              << std::setw(7) << "steps"
              << std::setw(7) << "trunc"
              << std::setw(6) << "pts"
              << std::setw(7) << "iters"
              << std::setw(7) << "res"
              << std::setw(4) << "ok"
              << ' '
              << std::setw(12) << "near"
              << std::setw(11) << "meanaln"
              << std::setw(11) << "p95aln"
              << std::setw(11) << "maxaln"
              << std::setw(11) << "alnscore"
              << std::setw(12) << "init_rms"
              << std::setw(12) << "final_rms"
              << std::setw(11) << "dscore"
              << std::setw(6) << "pick" << '\n';
    for (const auto& span : spans) {
        for (const auto& cand : span.candidates) {
            std::cout << std::right << std::setw(3) << span.segmentIndex
                      << std::setw(4) << span.leftControlIndex
                      << std::setw(4) << span.rightControlIndex
                      << ' ' << std::left << std::setw(15) << cand.name
                      << std::right
                      << std::setw(5) << cand.selectedSign
                      << std::setw(7) << cand.rolloutSteps
                      << std::setw(7) << cand.truncatedPoints
                      << std::setw(6) << cand.points
                      << std::setw(7) << cand.iterations
                      << std::setw(7) << cand.residuals
                      << std::setw(4) << (cand.usable ? "yes" : "no")
                      << ' '
                      << std::setw(12) << cand.closestTargetDistance
                      << std::setw(11) << cand.avgNormalAlignmentAbs
                      << std::setw(11) << cand.p95NormalAlignmentAbs
                      << std::setw(11) << cand.maxNormalAlignmentAbs
                      << std::setw(11) << cand.alignmentChoiceScore
                      << std::setw(12) << cand.initialRms
                      << std::setw(12) << cand.finalRms
                      << std::setw(11) << cand.alignmentChoiceScoreDelta
                      << std::setw(6) << (cand.chosen ? "*" : "") << '\n';
        }
    }
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
              << "term                 n      weight     raw_rms  weighted_rms\n";
    for (const auto& loss : report.finalLosses) {
        const double rawRms = loss.residuals > 0
            ? std::sqrt(std::max(0.0, 2.0 * loss.rawCost) /
                        static_cast<double>(loss.residuals))
            : 0.0;
        const double weightedRms = loss.residuals > 0
            ? std::sqrt(std::max(0.0, 2.0 * loss.weightedCost) /
                        static_cast<double>(loss.residuals))
            : 0.0;
        std::cout << std::left << std::setw(18) << loss.name
                  << std::right << std::setw(6) << loss.residuals
                  << std::setw(12) << loss.weight
                  << std::setw(12) << rawRms
                  << std::setw(14) << weightedRms
                  << '\n';
    }
}

void printResidualSummary(const vc::lasagna::LineOptimizationReport& report)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(3)
              << "Residual RMS: initial=" << report.initialRms
              << " optimized=" << report.finalRms
              << " residuals=" << report.residuals
              << '\n';
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

double normalAlignmentRms(const vc::lasagna::LineOptimizationReport& report)
{
    for (const auto& loss : report.finalLosses) {
        if (loss.name == "normal_alignment") {
            if (loss.residuals <= 0) {
                return 0.0;
            }
            return std::sqrt(std::max(0.0, 2.0 * loss.weightedCost) /
                             static_cast<double>(loss.residuals));
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

void ensureDirectory(const std::filesystem::path& dir)
{
    if (dir.empty()) {
        throw std::runtime_error("output directory is empty");
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("failed to create output directory " +
                                 dir.string() + ": " + ec.message());
    }
}

bool validSurfacePoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f &&
           std::isfinite(point[0]) &&
           std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

void writeLineObj(const std::vector<cv::Vec3d>& points,
                  const std::filesystem::path& outputPath)
{
    std::ofstream output(outputPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# VC3D fiber line\n";
    for (const auto& point : points) {
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    if (!points.empty()) {
        output << "l";
        for (size_t i = 0; i < points.size(); ++i) {
            output << ' ' << (i + 1);
        }
        output << '\n';
    }
}

std::string objElementName(std::string name)
{
    if (name.empty()) {
        return "polyline";
    }
    for (char& ch : name) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (!std::isalnum(value) && ch != '_' && ch != '-') {
            ch = '_';
        }
    }
    return name;
}

cv::Vec3d perpendicularUnitVector(const cv::Vec3d& direction)
{
    const cv::Vec3d unit = normalizedOrZero(direction);
    if (!finiteDirection(unit)) {
        return {0.0, 1.0, 0.0};
    }
    const cv::Vec3d reference = std::abs(unit[0]) < 0.9
        ? cv::Vec3d{1.0, 0.0, 0.0}
        : cv::Vec3d{0.0, 1.0, 0.0};
    const cv::Vec3d perpendicular = normalizedOrZero(unit.cross(reference));
    return finiteDirection(perpendicular) ? perpendicular : cv::Vec3d{0.0, 1.0, 0.0};
}

struct PlaneSliceBasis {
    cv::Vec3d origin{0.0, 0.0, 0.0};
    cv::Vec3d u{1.0, 0.0, 0.0};
    cv::Vec3d v{0.0, 1.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 1.0};
    double minU = -1.0;
    double maxU = 1.0;
    double minV = -1.0;
    double maxV = 1.0;
};

PlaneSliceBasis fitControlPointPlaneSlice(const std::vector<cv::Vec3d>& points)
{
    if (points.empty()) {
        throw std::runtime_error("cannot write control plane slice without control points");
    }

    PlaneSliceBasis basis;
    for (const auto& point : points) {
        basis.origin += point;
    }
    basis.origin *= 1.0 / static_cast<double>(points.size());

    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (const auto& point : points) {
        const cv::Vec3d centered = point - basis.origin;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                covariance(row, col) += centered[row] * centered[col];
            }
        }
    }
    covariance *= 1.0 / static_cast<double>(points.size());

    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    if (cv::eigen(cv::Mat(covariance), eigenvalues, eigenvectors) &&
        eigenvectors.rows == 3 && eigenvectors.cols == 3) {
        basis.u = normalizedOrZero({eigenvectors.at<double>(0, 0),
                                    eigenvectors.at<double>(0, 1),
                                    eigenvectors.at<double>(0, 2)});
        basis.v = normalizedOrZero({eigenvectors.at<double>(1, 0),
                                    eigenvectors.at<double>(1, 1),
                                    eigenvectors.at<double>(1, 2)});
        basis.normal = normalizedOrZero({eigenvectors.at<double>(2, 0),
                                         eigenvectors.at<double>(2, 1),
                                         eigenvectors.at<double>(2, 2)});
    }

    if (!finiteDirection(basis.u) && points.size() >= 2) {
        basis.u = normalizedOrZero(points.back() - points.front());
    }
    if (!finiteDirection(basis.u)) {
        basis.u = {1.0, 0.0, 0.0};
    }
    if (!finiteDirection(basis.normal)) {
        basis.normal = normalizedOrZero(basis.u.cross({0.0, 0.0, 1.0}));
        if (!finiteDirection(basis.normal)) {
            basis.normal = perpendicularUnitVector(basis.u);
        }
    }
    basis.v = normalizedOrZero(basis.normal.cross(basis.u));
    if (!finiteDirection(basis.v)) {
        basis.v = perpendicularUnitVector(basis.u);
    }
    basis.normal = normalizedOrZero(basis.u.cross(basis.v));
    basis.v = normalizedOrZero(basis.normal.cross(basis.u));

    basis.minU = basis.minV = std::numeric_limits<double>::infinity();
    basis.maxU = basis.maxV = -std::numeric_limits<double>::infinity();
    for (const auto& point : points) {
        const cv::Vec3d delta = point - basis.origin;
        const double du = delta.dot(basis.u);
        const double dv = delta.dot(basis.v);
        basis.minU = std::min(basis.minU, du);
        basis.maxU = std::max(basis.maxU, du);
        basis.minV = std::min(basis.minV, dv);
        basis.maxV = std::max(basis.maxV, dv);
    }

    const double rangeU = std::max(0.0, basis.maxU - basis.minU);
    const double rangeV = std::max(0.0, basis.maxV - basis.minV);
    const double margin = std::max(64.0, 0.05 * std::max(rangeU, rangeV));
    basis.minU -= margin;
    basis.maxU += margin;
    basis.minV -= margin;
    basis.maxV += margin;
    if (basis.maxU - basis.minU <= kEpsilon) {
        basis.minU -= margin;
        basis.maxU += margin;
    }
    if (basis.maxV - basis.minV <= kEpsilon) {
        basis.minV -= margin;
        basis.maxV += margin;
    }
    return basis;
}

void writePlaneSliceObj(const std::vector<cv::Vec3d>& points,
                        const std::filesystem::path& outputPath)
{
    const PlaneSliceBasis basis = fitControlPointPlaneSlice(points);
    const auto corner = [&](double u, double v) {
        return basis.origin + basis.u * u + basis.v * v;
    };
    const std::vector<cv::Vec3d> corners{
        corner(basis.minU, basis.minV),
        corner(basis.maxU, basis.minV),
        corner(basis.maxU, basis.maxV),
        corner(basis.minU, basis.maxV),
    };

    std::ofstream output(outputPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# Reinit control-point fitted plane slice\n";
    output << "# origin " << basis.origin[0] << ' ' << basis.origin[1] << ' ' << basis.origin[2] << '\n';
    output << "# normal " << basis.normal[0] << ' ' << basis.normal[1] << ' ' << basis.normal[2] << '\n';
    for (const auto& point : corners) {
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    output << "f 1 2 3 4\n"
           << "l 1 2 3 4 1\n";
}

cv::Mat_<cv::Vec3f> makePlaneSliceGrid(const PlaneSliceBasis& basis,
                                       double targetSpacing,
                                       int renderScale)
{
    const double width = std::max(kEpsilon, basis.maxU - basis.minU);
    const double height = std::max(kEpsilon, basis.maxV - basis.minV);
    targetSpacing = std::max(1.0, targetSpacing);
    renderScale = std::max(1, renderScale);

    const int maxRenderedDim = 4096;
    const int maxBaseDim = std::max(2, maxRenderedDim / renderScale);
    const int cols = std::clamp(static_cast<int>(std::ceil(width / targetSpacing)) + 1,
                                2,
                                maxBaseDim);
    const int rows = std::clamp(static_cast<int>(std::ceil(height / targetSpacing)) + 1,
                                2,
                                maxBaseDim);

    cv::Mat_<cv::Vec3f> coords(rows, cols);
    for (int row = 0; row < rows; ++row) {
        const double tv = rows == 1
            ? 0.0
            : static_cast<double>(row) / static_cast<double>(rows - 1);
        const double v = basis.minV * (1.0 - tv) + basis.maxV * tv;
        for (int col = 0; col < cols; ++col) {
            const double tu = cols == 1
                ? 0.0
                : static_cast<double>(col) / static_cast<double>(cols - 1);
            const double u = basis.minU * (1.0 - tu) + basis.maxU * tu;
            const cv::Vec3d point = basis.origin + basis.u * u + basis.v * v;
            coords(row, col) = {
                static_cast<float>(point[0]),
                static_cast<float>(point[1]),
                static_cast<float>(point[2]),
            };
        }
    }
    return coords;
}

void writeTexturedPlaneSliceObj(const cv::Mat_<cv::Vec3f>& coords,
                                const std::filesystem::path& objPath,
                                const std::string& materialName,
                                const std::string& mtlName)
{
    if (coords.empty()) {
        throw std::runtime_error("cannot write empty control plane slice OBJ");
    }

    std::ofstream output(objPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + objPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# Textured reinit control-point fitted plane slice\n"
           << "mtllib " << mtlName << '\n'
           << "usemtl " << materialName << '\n';

    for (int row = 0; row < coords.rows; ++row) {
        for (int col = 0; col < coords.cols; ++col) {
            const cv::Vec3f& point = coords(row, col);
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        }
    }

    const double colDenom = std::max(1, coords.cols - 1);
    const double rowDenom = std::max(1, coords.rows - 1);
    for (int row = 0; row < coords.rows; ++row) {
        for (int col = 0; col < coords.cols; ++col) {
            output << "vt "
                   << (static_cast<double>(col) / colDenom) << ' '
                   << (1.0 - static_cast<double>(row) / rowDenom) << '\n';
        }
    }

    const auto index = [&](int row, int col) {
        return row * coords.cols + col + 1;
    };
    for (int row = 0; row + 1 < coords.rows; ++row) {
        for (int col = 0; col + 1 < coords.cols; ++col) {
            const int v00 = index(row, col);
            const int v01 = index(row, col + 1);
            const int v10 = index(row + 1, col);
            const int v11 = index(row + 1, col + 1);
            output << "f "
                   << v00 << '/' << v00 << ' '
                   << v01 << '/' << v01 << ' '
                   << v11 << '/' << v11 << ' '
                   << v10 << '/' << v10 << '\n';
        }
    }
}

void writeControlPointsObj(const std::vector<cv::Vec3d>& points,
                           const std::filesystem::path& outputPath)
{
    std::ofstream output(outputPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# Reinit control points\n";
    for (const auto& point : points) {
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    }
    if (!points.empty()) {
        output << "p";
        for (size_t i = 0; i < points.size(); ++i) {
            output << ' ' << (i + 1);
        }
        output << '\n';
    }
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        output << "l " << (i + 1) << ' ' << (i + 2) << '\n';
    }
}

void writeNamedPolylinesObj(const std::vector<vc::lasagna::LineDebugPolyline>& lines,
                            const std::filesystem::path& outputPath)
{
    std::ofstream output(outputPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# Reinit continuation rollout candidates\n";

    size_t nextVertex = 1;
    for (const auto& line : lines) {
        output << "o " << objElementName(line.name) << '\n';
        if (line.points.empty()) {
            continue;
        }
        const size_t firstVertex = nextVertex;
        for (const auto& point : line.points) {
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
            ++nextVertex;
        }
        if (line.points.size() >= 2) {
            output << "l";
            for (size_t index = 0; index < line.points.size(); ++index) {
                output << ' ' << (firstVertex + index);
            }
            output << '\n';
        } else {
            output << "p " << firstVertex << '\n';
        }
    }
}

void writeSurfaceMtl(const std::filesystem::path& path,
                     const std::string& materialName,
                     const std::string& textureName)
{
    std::ofstream output(path);
    if (!output.good()) {
        throw std::runtime_error("could not open MTL output: " + path.string());
    }
    output << "newmtl " << materialName << '\n'
           << "Ka 1 1 1\n"
           << "Kd 1 1 1\n"
           << "Ks 0 0 0\n"
           << "d 1\n"
           << "illum 1\n"
           << "map_Kd " << textureName << '\n';
}

void writeSurfaceObj(const QuadSurface& surface,
                     const std::filesystem::path& objPath,
                     const std::string& materialName,
                     const std::string& mtlName)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        throw std::runtime_error("surface has no points for OBJ export");
    }

    std::ofstream output(objPath);
    if (!output.good()) {
        throw std::runtime_error("could not open OBJ output: " + objPath.string());
    }
    output.imbue(std::locale::classic());
    output << "# VC3D line-view strip\n"
           << "mtllib " << mtlName << '\n'
           << "usemtl " << materialName << '\n';

    std::vector<int> vertexIndex(static_cast<size_t>(points->rows * points->cols), 0);
    int nextVertex = 1;
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f& point = (*points)(row, col);
            if (!validSurfacePoint(point)) {
                continue;
            }
            vertexIndex[static_cast<size_t>(row * points->cols + col)] = nextVertex++;
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        }
    }

    std::vector<int> uvIndex(static_cast<size_t>(points->rows * points->cols), 0);
    int nextUv = 1;
    const double colDenom = std::max(1, points->cols - 1);
    const double rowDenom = std::max(1, points->rows - 1);
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f& point = (*points)(row, col);
            if (!validSurfacePoint(point)) {
                continue;
            }
            uvIndex[static_cast<size_t>(row * points->cols + col)] = nextUv++;
            output << "vt "
                   << (static_cast<double>(col) / colDenom) << ' '
                   << (1.0 - static_cast<double>(row) / rowDenom) << '\n';
        }
    }

    for (int row = 0; row + 1 < points->rows; ++row) {
        for (int col = 0; col + 1 < points->cols; ++col) {
            const auto idx = [&](int r, int c) {
                return static_cast<size_t>(r * points->cols + c);
            };
            const int v00 = vertexIndex[idx(row, col)];
            const int v01 = vertexIndex[idx(row, col + 1)];
            const int v10 = vertexIndex[idx(row + 1, col)];
            const int v11 = vertexIndex[idx(row + 1, col + 1)];
            if (v00 == 0 || v01 == 0 || v10 == 0 || v11 == 0) {
                continue;
            }
            const int t00 = uvIndex[idx(row, col)];
            const int t01 = uvIndex[idx(row, col + 1)];
            const int t10 = uvIndex[idx(row + 1, col)];
            const int t11 = uvIndex[idx(row + 1, col + 1)];
            output << "f "
                   << v00 << '/' << t00 << ' '
                   << v01 << '/' << t01 << ' '
                   << v11 << '/' << t11 << ' '
                   << v10 << '/' << t10 << '\n';
        }
    }
}

cv::Mat renderCoordsTexture(const cv::Mat_<cv::Vec3f>& baseCoords,
                            Volume& textureVolume,
                            int textureLevel,
                            int renderScale,
                            const char* label)
{
    if (baseCoords.empty()) {
        throw std::runtime_error("surface has no points for texture rendering");
    }
    renderScale = std::max(1, renderScale);
    cv::Mat_<cv::Vec3f> coords;
    if (renderScale == 1) {
        coords = baseCoords.clone();
    } else {
        cv::resize(baseCoords,
                   coords,
                   cv::Size(baseCoords.cols * renderScale, baseCoords.rows * renderScale),
                   0.0,
                   0.0,
                   cv::INTER_LINEAR);
    }

    vc::render::IChunkedArray* cache = textureVolume.chunkedCache();
    if (!cache) {
        throw std::runtime_error("texture volume has no chunk cache");
    }
    if (cache->dtype() != vc::render::ChunkDtype::UInt8) {
        throw std::runtime_error(
            "line probe strip rendering uses the VC3D uint8 chunk sampler; "
            "choose a uint8 texture zarr");
    }

    cv::Mat_<uint8_t> sampled(coords.rows, coords.cols, uint8_t(0));
    cv::Mat_<uint8_t> coverage(coords.rows, coords.cols, uint8_t(0));
    const vc::render::ChunkedPlaneSampler::Options options(vc::Sampling::Trilinear, 32);
    const int startLevel = std::clamp(textureLevel, 0, cache->numLevels() - 1);

    for (int level = startLevel; level < cache->numLevels(); ++level) {
        std::vector<vc::render::ChunkKey> keys =
            vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
                *cache, level, coords, coverage, options);
        if (!keys.empty()) {
            cache->prefetchChunks(keys, true);
        }
    }

    const auto stats = vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
        *cache, startLevel, coords, sampled, coverage, options);
    const int covered = cv::countNonZero(coverage);
    const int total = coverage.rows * coverage.cols;
    std::cout << label << ": start_level=" << startLevel
              << " render_scale=" << renderScale
              << " size=" << sampled.cols << "x" << sampled.rows
              << " covered=" << covered << "/" << total
              << " requested_chunks=" << stats.requestedChunks
              << " error_chunks=" << stats.errorChunks << '\n';
    return sampled;
}

cv::Mat renderSurfaceTexture(const QuadSurface& surface,
                             Volume& textureVolume,
                             int textureLevel,
                             int renderScale)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        throw std::runtime_error("surface has no points for texture rendering");
    }
    return renderCoordsTexture(*points,
                               textureVolume,
                               textureLevel,
                               renderScale,
                               "Strip texture sampling");
}

void writeTif(const std::filesystem::path& path, const cv::Mat& image);

vc::lasagna::LineModel lineModelSliceFromFixedIndices(
    const vc::lasagna::LineModel& line,
    const std::vector<int>& fixedPointIndices)
{
    if (line.points.size() < 2) {
        throw std::runtime_error("cannot slice line with fewer than two points");
    }

    int first = std::numeric_limits<int>::max();
    int last = std::numeric_limits<int>::min();
    const int maxIndex = static_cast<int>(line.points.size()) - 1;
    for (const int index : fixedPointIndices) {
        if (index < 0 || index > maxIndex) {
            continue;
        }
        first = std::min(first, index);
        last = std::max(last, index);
    }
    if (first == std::numeric_limits<int>::max() || last - first < 1) {
        throw std::runtime_error("cannot slice line without at least two fixed control indices");
    }

    vc::lasagna::LineModel slice;
    slice.points.reserve(static_cast<size_t>(last - first + 1));
    for (int index = first; index <= last; ++index) {
        slice.points.push_back(line.points[static_cast<size_t>(index)]);
    }

    if (line.displayFrameAnchorIndex >= first && line.displayFrameAnchorIndex <= last) {
        slice.displayFrameAnchorIndex = line.displayFrameAnchorIndex - first;
    } else {
        slice.displayFrameAnchorIndex = static_cast<int>(slice.points.size() / 2);
    }
    return slice;
}

std::vector<cv::Vec3d> linePointsSliceFromFixedIndices(
    const vc::lasagna::LineModel& line,
    const std::vector<int>& fixedPointIndices)
{
    return linePointsFromModel(lineModelSliceFromFixedIndices(line, fixedPointIndices));
}

void writePartialSideSliceObjOutput(const vc::lasagna::LineModel& line,
                                    const std::vector<int>& fixedPointIndices,
                                    Volume& textureVolume,
                                    int textureLevel,
                                    int renderScale,
                                    const std::filesystem::path& outputDir)
{
    const vc::lasagna::LineModel partialLine =
        lineModelSliceFromFixedIndices(line, fixedPointIndices);
    const auto surfaces = vc::lasagna::buildLineViewSurfaces(partialLine);
    if (!surfaces.lineSideSlice) {
        throw std::runtime_error("line view builder did not produce partial side strip");
    }

    const cv::Mat sideSliceTexture =
        renderSurfaceTexture(*surfaces.lineSideSlice, textureVolume, textureLevel, renderScale);
    writeTif(outputDir / "partial_side_slice.tif", sideSliceTexture);
    writeSurfaceMtl(outputDir / "partial_side_slice.mtl",
                    "partial_side_slice_texture",
                    "partial_side_slice.tif");
    writeSurfaceObj(*surfaces.lineSideSlice,
                    outputDir / "partial_side_slice.obj",
                    "partial_side_slice_texture",
                    "partial_side_slice.mtl");
}

void writeTif(const std::filesystem::path& path, const cv::Mat& image)
{
    if (image.empty()) {
        throw std::runtime_error("cannot write empty image: " + path.string());
    }
    const std::vector<int> params{
        cv::IMWRITE_TIFF_COMPRESSION,
        kTiffCompressionNone,
    };
    if (!cv::imwrite(path.string(), image, params)) {
        throw std::runtime_error("failed to write image: " + path.string());
    }
}

void writeReinitDebugObjOutput(
    const std::vector<cv::Vec3d>& controlPoints,
    const vc::lasagna::LineModel& finalLine,
    const std::vector<int>& fixedPointIndices,
    const std::vector<vc::lasagna::LineDebugPolyline>& continuationCandidateLines,
    const std::filesystem::path& outputDir,
    Volume* textureVolume,
    int textureLevel,
    int renderScale)
{
    ensureDirectory(outputDir);
    if (textureVolume != nullptr) {
        const PlaneSliceBasis basis = fitControlPointPlaneSlice(controlPoints);
        const cv::Mat_<cv::Vec3f> planeCoords =
            makePlaneSliceGrid(basis, 32.0, renderScale);
        const cv::Mat planeTexture = renderCoordsTexture(planeCoords,
                                                         *textureVolume,
                                                         textureLevel,
                                                         renderScale,
                                                         "Control plane texture sampling");
        writeTif(outputDir / "control_plane_slice.tif", planeTexture);
        writeSurfaceMtl(outputDir / "control_plane_slice.mtl",
                        "control_plane_slice_texture",
                        "control_plane_slice.tif");
        writeTexturedPlaneSliceObj(planeCoords,
                                   outputDir / "control_plane_slice.obj",
                                   "control_plane_slice_texture",
                                   "control_plane_slice.mtl");
        writePartialSideSliceObjOutput(finalLine,
                                       fixedPointIndices,
                                       *textureVolume,
                                       textureLevel,
                                       renderScale,
                                       outputDir);
    } else {
        writePlaneSliceObj(controlPoints, outputDir / "control_plane_slice.obj");
    }
    writeControlPointsObj(controlPoints, outputDir / "control_points.obj");
    const std::vector<cv::Vec3d> boundedLinePoints =
        linePointsSliceFromFixedIndices(finalLine, fixedPointIndices);
    writeLineObj(boundedLinePoints, outputDir / "final_line.obj");
    writeNamedPolylinesObj(continuationCandidateLines,
                           outputDir / "continuation_candidates.obj");
}

int scaledControlColumn(int sourceIndex, int renderScale, int cols)
{
    const double scaled = (static_cast<double>(sourceIndex) + 0.5) *
                          static_cast<double>(std::max(1, renderScale)) -
                          0.5;
    return std::clamp(static_cast<int>(std::lround(scaled)), 0, std::max(0, cols - 1));
}

cv::Mat makeStripOverlay(const cv::Mat& grayscale,
                         const std::vector<int>& fixedIndices,
                         int renderScale)
{
    cv::Mat image8;
    if (grayscale.depth() == CV_8U) {
        image8 = grayscale;
    } else {
        grayscale.convertTo(image8, CV_8U);
    }
    cv::Mat overlay;
    cv::cvtColor(image8, overlay, cv::COLOR_GRAY2BGR);

    const int centerRow = overlay.rows / 2;
    cv::line(overlay,
             cv::Point(0, centerRow),
             cv::Point(std::max(0, overlay.cols - 1), centerRow),
             cv::Scalar(0, 255, 255),
             1,
             cv::LINE_AA);

    for (const int index : fixedIndices) {
        if (index < 0) {
            continue;
        }
        const int col = scaledControlColumn(index, renderScale, overlay.cols);
        cv::line(overlay,
                 cv::Point(col, 0),
                 cv::Point(col, std::max(0, overlay.rows - 1)),
                 cv::Scalar(0, 0, 255),
                 1,
                 cv::LINE_AA);
        cv::circle(overlay,
                   cv::Point(col, centerRow),
                   3,
                   cv::Scalar(0, 255, 0),
                   cv::FILLED,
                   cv::LINE_AA);
    }
    return overlay;
}

struct RenderedStrips {
    cv::Mat lineSurface;
    cv::Mat lineSideSlice;
};

RenderedStrips renderLineViewStrips(const vc::lasagna::LineModel& line,
                                    Volume& textureVolume,
                                    int textureLevel,
                                    int renderScale)
{
    const auto surfaces = vc::lasagna::buildLineViewSurfaces(line);
    if (!surfaces.lineSurface || !surfaces.lineSideSlice) {
        throw std::runtime_error("line view builder did not produce both strips");
    }
    return {
        renderSurfaceTexture(*surfaces.lineSurface, textureVolume, textureLevel, renderScale),
        renderSurfaceTexture(*surfaces.lineSideSlice, textureVolume, textureLevel, renderScale),
    };
}

void writeStripImages(const vc::lasagna::LineModel& line,
                      const std::vector<int>& fixedIndices,
                      Volume& textureVolume,
                      int textureLevel,
                      int renderScale,
                      const std::filesystem::path& outputDir)
{
    ensureDirectory(outputDir);
    const RenderedStrips strips = renderLineViewStrips(line, textureVolume, textureLevel, renderScale);
    writeTif(outputDir / "line_surface.tif", strips.lineSurface);
    writeTif(outputDir / "line_surface_overlay.tif",
             makeStripOverlay(strips.lineSurface, fixedIndices, renderScale));
    writeTif(outputDir / "side_slice.tif", strips.lineSideSlice);
    writeTif(outputDir / "side_slice_overlay.tif",
             makeStripOverlay(strips.lineSideSlice, fixedIndices, renderScale));
}

void writeLineViewObjOutput(const vc::lasagna::LineModel& line,
                            const std::vector<cv::Vec3d>& linePoints,
                            Volume& textureVolume,
                            int textureLevel,
                            int renderScale,
                            const std::filesystem::path& outputDir)
{
    ensureDirectory(outputDir);
    const auto surfaces = vc::lasagna::buildLineViewSurfaces(line);
    if (!surfaces.lineSurface || !surfaces.lineSideSlice) {
        throw std::runtime_error("line view builder did not produce both strips");
    }

    writeLineObj(linePoints, outputDir / "line.obj");

    const cv::Mat lineSurfaceTexture =
        renderSurfaceTexture(*surfaces.lineSurface, textureVolume, textureLevel, renderScale);
    const cv::Mat sideSliceTexture =
        renderSurfaceTexture(*surfaces.lineSideSlice, textureVolume, textureLevel, renderScale);
    writeTif(outputDir / "line_surface.tif", lineSurfaceTexture);
    writeTif(outputDir / "side_slice.tif", sideSliceTexture);

    writeSurfaceMtl(outputDir / "line_surface.mtl",
                    "line_surface_texture",
                    "line_surface.tif");
    writeSurfaceMtl(outputDir / "side_slice.mtl",
                    "side_slice_texture",
                    "side_slice.tif");
    writeSurfaceObj(*surfaces.lineSurface,
                    outputDir / "line_surface.obj",
                    "line_surface_texture",
                    "line_surface.mtl");
    writeSurfaceObj(*surfaces.lineSideSlice,
                    outputDir / "side_slice.obj",
                    "side_slice_texture",
                    "side_slice.mtl");
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
    bool reopt = false;
    bool reinitReopt = false;
    bool verbose = false;
    std::filesystem::path fiberPath;
    std::filesystem::path outputPath;
    std::filesystem::path objOutputDir;
    std::filesystem::path reinitDebugObjOutputDir;
    std::filesystem::path stripOutputDir;
    std::filesystem::path textureZarrPath;
    int textureLevel = 0;
    int stripRenderScale = 4;
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
        } else if (arg == "--reopt") {
            reopt = true;
        } else if (arg == "--reinit-reopt") {
            reinitReopt = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--fiber") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--fiber requires a path");
            }
            fiberPath = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            outputPath = argv[++i];
        } else if (arg == "--obj-output-dir") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--obj-output-dir requires a directory");
            }
            objOutputDir = argv[++i];
        } else if (arg == "--reinit-debug-obj-output-dir") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--reinit-debug-obj-output-dir requires a directory");
            }
            reinitDebugObjOutputDir = argv[++i];
        } else if (arg == "--strip-output-dir") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--strip-output-dir requires a directory");
            }
            stripOutputDir = argv[++i];
        } else if (arg == "--texture-zarr") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--texture-zarr requires a zarr path");
            }
            textureZarrPath = argv[++i];
        } else if (arg == "--texture-level") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--texture-level requires an integer");
            }
            textureLevel = parseNonNegativeInt(argv[++i], "texture-level");
        } else if (arg == "--strip-render-scale") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--strip-render-scale requires a positive integer");
            }
            stripRenderScale = parsePositiveInt(argv[++i], "strip-render-scale");
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
        const bool wantsObjOutput = !objOutputDir.empty();
        const bool wantsReinitDebugObjOutput = !reinitDebugObjOutputDir.empty();
        const bool wantsStripOutput = !stripOutputDir.empty();
        const bool reinitDebugFailureMode =
            reinitReopt && (wantsObjOutput || wantsReinitDebugObjOutput || wantsStripOutput);
        std::filesystem::path effectiveOutputPath = outputPath;
        if (reinitDebugFailureMode && !effectiveOutputPath.empty()) {
            if (verbose) {
                std::cout << "Ignoring --output for reinit debug-output run; partial debug output is enabled\n";
            }
            effectiveOutputPath.clear();
        }
        const bool wantsTextureOutput = wantsObjOutput ||
                                        wantsStripOutput ||
                                        (wantsReinitDebugObjOutput && !textureZarrPath.empty());
        if (!outputPath.empty() && fiberPath.empty()) {
            throw std::invalid_argument("--output requires --fiber <fiber.json>");
        }
        if (reopt && fiberPath.empty()) {
            throw std::invalid_argument("--reopt requires --fiber <fiber.json>");
        }
        if (reinitReopt && fiberPath.empty()) {
            throw std::invalid_argument("--reinit-reopt requires --fiber <fiber.json>");
        }
        if (reopt && reinitReopt) {
            throw std::invalid_argument("--reopt and --reinit-reopt are mutually exclusive");
        }
        if (wantsTextureOutput && fiberPath.empty()) {
            throw std::invalid_argument("--obj-output-dir/--strip-output-dir require --fiber <fiber.json>");
        }
        if (wantsTextureOutput && textureZarrPath.empty()) {
            throw std::invalid_argument("--obj-output-dir/--strip-output-dir require --texture-zarr <zarr>");
        }
        if (wantsReinitDebugObjOutput && !reinitReopt) {
            throw std::invalid_argument("--reinit-debug-obj-output-dir requires --reinit-reopt");
        }
        if (wantsReinitDebugObjOutput && fiberPath.empty()) {
            throw std::invalid_argument("--reinit-debug-obj-output-dir requires --fiber <fiber.json>");
        }
        if (!textureZarrPath.empty() && !wantsTextureOutput && !wantsReinitDebugObjOutput) {
            throw std::invalid_argument("--texture-zarr requires --obj-output-dir, --strip-output-dir, or --reinit-debug-obj-output-dir");
        }
        if ((reopt || reinitReopt) && (benchmarkSolvers || benchmarkThreads || traceInit)) {
            throw std::invalid_argument("--reopt/--reinit-reopt cannot be combined with benchmark or seed trace modes");
        }
        if (!fiberPath.empty() && (benchmarkSolvers || benchmarkThreads || traceInit)) {
            throw std::invalid_argument("--fiber cannot be combined with benchmark or seed trace modes");
        }

        const std::filesystem::path manifestPath = argv[1];
        vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        vc::lasagna::LineOptimizer optimizer(sampler);
        std::shared_ptr<Volume> textureVolume;
        if (wantsTextureOutput) {
            textureVolume = Volume::New(textureZarrPath);
            if (!textureVolume->hasScaleLevel(textureLevel)) {
                throw std::runtime_error("texture zarr has no scale level " +
                                         std::to_string(textureLevel));
            }
        }

        const cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
        const auto seedNormal = sampler.sampleNormal(seedPoint);

        vc::lasagna::LineOptimizationConfig config;
        config.segmentsPerSide = segmentsPerSide;
        config.segmentLength = 32.0;
        config.straightnessWeight = 0.1;
        config.tangentStraightnessWeight = 5.0;
        config.samplesPerSegment = 1;
        config.maxIterations = 1000;
        config.differentiableNormalSampling = differentiableNormalSampling;
        config.initialTangent = initialZInOutTangent(sourceSliceNormal, seedNormal);
        config.useInitialTangent = finiteDirection(config.initialTangent);
        config.tangentGuideVector = normalizedOrZero(sourceSliceNormal);
        config.tangentGuideWeight = 1.0;
        config.tangentGuideMode =
            vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;

        if (!fiberPath.empty()) {
            const FiberInput fiber = loadFiberInput(fiberPath);
            const int seedControlIndex = middleControlIndex(fiber.controlPoints, fiber.linePoints);
            const int displayFrameAnchorIndex = nearestLinePointIndex(
                fiber.linePoints,
                fiber.controlPoints[static_cast<size_t>(seedControlIndex)]);
            const auto fiberSeedNormal =
                sampler.sampleNormal(fiber.controlPoints[static_cast<size_t>(seedControlIndex)]);
            config.initialTangent = initialZInOutTangent(sourceSliceNormal, fiberSeedNormal);
            config.useInitialTangent = finiteDirection(config.initialTangent);
            config.initialLinePoints = fiber.linePoints;
            config.printSolverProgress = false;

            const auto fixedIndices = fixedIndicesForControls(fiber);
            if (fixedIndices.empty()) {
                throw std::runtime_error("fiber has no usable fixed control indices");
            }

            if (verbose) {
                std::cout << "Fiber input:\n"
                          << "fiber=" << fiberPath.string() << '\n'
                          << "manifest=" << manifestPath.string() << '\n'
                          << "output=" << (effectiveOutputPath.empty() ? std::string{"<none>"} : effectiveOutputPath.string()) << '\n'
                          << "line_points=" << fiber.linePoints.size()
                          << " control_points=" << fiber.controlPoints.size()
                          << " fixed_points=" << fixedIndices.size()
                          << " seed_control_index=" << seedControlIndex
                          << " display_frame_anchor_index=" << displayFrameAnchorIndex
                          << " line_length=" << lineLength(fiber.linePoints)
                          << '\n';
                std::cout << "Seed normal valid=" << fiberSeedNormal.valid
                          << " normal=[" << fiberSeedNormal.normal[0]
                          << ", " << fiberSeedNormal.normal[1]
                          << ", " << fiberSeedNormal.normal[2] << "]\n";
                std::cout << "Initial tangent valid=" << config.useInitialTangent
                          << " tangent=[" << config.initialTangent[0]
                          << ", " << config.initialTangent[1]
                          << ", " << config.initialTangent[2] << "]\n";
                std::cout << "Differentiable normal sampling=" << config.differentiableNormalSampling << "\n";
            }

            std::vector<cv::Vec3d> outputLinePoints = fiber.linePoints;
            vc::lasagna::LineModel outputLine =
                makeLineModelFromPoints(fiber.linePoints, sampler, displayFrameAnchorIndex);
            std::vector<int> outputFixedIndices = fixedIndices;
            bool reinitFailedWithoutDebugOutput = false;

            if (reinitReopt) {
                std::vector<vc::lasagna::LineControlPoint> controls;
                controls.reserve(fiber.controlPoints.size());
                for (size_t controlIndex = 0; controlIndex < fiber.controlPoints.size(); ++controlIndex) {
                    const int lineIndex = nearestLinePointIndex(fiber.linePoints,
                                                                fiber.controlPoints[controlIndex]);
                    controls.push_back({
                        static_cast<double>(lineIndex),
                        fiber.controlPoints[controlIndex],
                        static_cast<int>(controlIndex) == seedControlIndex,
                        lineIndex,
                    });
                }
                const auto result =
                    optimizer.reinitializeAndOptimizeExistingLine(fiber.linePoints,
                                                                  std::move(controls),
                                                                  fixedIndices,
                                                                  displayFrameAnchorIndex,
                                                                  config);
                outputLine = result.optimization.line;
                outputLinePoints = linePointsFromModel(result.optimization.line);
                outputFixedIndices = result.fixedPointIndices;
                if (!result.failed) {
                    printResidualSummary(result.optimization.report);
                }
                if (verbose) {
                    const PointMotionStats motionStats = pointMotionStats(fiber.linePoints, outputLinePoints);
                    printReinitSpanTable(result.spans);
                    printReinitCandidateTable(result.spans);
                    std::cout << "max_segment_candidate_alignment_score_diff="
                              << result.maxSegmentCandidateAlignmentScoreDiff << '\n';
                    if (result.failed) {
                        std::cout << "Fiber reinit stopped at span "
                                  << result.failedSegmentIndex
                                  << ": " << result.failureReason << '\n';
                    } else {
                        printSegmentMotionTable(fiber.linePoints, outputLinePoints, fixedIndices);
                        std::cout << "Fiber reinit reoptimization complete: points="
                                  << result.optimization.line.points.size()
                                  << " iterations=" << result.optimization.report.iterations
                                  << " initial_rms=" << result.optimization.report.initialRms
                                  << " final_rms=" << result.optimization.report.finalRms
                                  << " residuals=" << result.optimization.report.residuals
                                  << " valid_normals=" << result.optimization.report.validNormalSamples
                                  << " invalid_normals=" << result.optimization.report.invalidNormalSamples
                                  << " converged=" << result.optimization.report.converged
                                  << " line_length=" << lineLength(outputLinePoints)
                                  << "\n";
                        printPointMotionStats(motionStats);
                        printLosses(result.optimization.report);
                    }
                }
                std::vector<std::filesystem::path> debugObjDirs;
                if (wantsReinitDebugObjOutput) {
                    debugObjDirs.push_back(reinitDebugObjOutputDir);
                }
                if (wantsObjOutput) {
                    debugObjDirs.push_back(objOutputDir / "reinit_debug");
                }
                const std::vector<cv::Vec3d>& debugControlPoints =
                    result.debugControlPoints.empty() ? fiber.controlPoints : result.debugControlPoints;
                for (const auto& debugObjDir : debugObjDirs) {
                    writeReinitDebugObjOutput(debugControlPoints,
                                              outputLine,
                                              result.fixedPointIndices,
                                              result.continuationCandidateLines,
                                              debugObjDir,
                                              textureVolume.get(),
                                              textureLevel,
                                              stripRenderScale);
                    if (verbose) {
                        std::cout << "Saved reinit debug OBJ output to "
                                  << debugObjDir.string() << '\n';
                    }
                }
                if (result.failed) {
                    if (reinitDebugFailureMode) {
                        if (wantsStripOutput) {
                            writeStripImages(outputLine,
                                             outputFixedIndices,
                                             *textureVolume,
                                             textureLevel,
                                             stripRenderScale,
                                             stripOutputDir);
                            if (verbose) {
                                std::cout << "Saved partial strip render output to "
                                          << stripOutputDir.string() << '\n';
                            }
                        }
                        throw std::runtime_error(result.failureReason);
                    }
                    outputLine = makeLineModelFromPoints(fiber.linePoints,
                                                         sampler,
                                                         displayFrameAnchorIndex);
                    outputLinePoints = fiber.linePoints;
                    outputFixedIndices = fixedIndices;
                    reinitFailedWithoutDebugOutput = true;
                    if (verbose) {
                        std::cout << "Keeping original fiber line because reinit failed without debug output\n";
                    }
                }
            } else if (reopt) {
                const auto result = optimizer.optimizeExistingLine(fiber.linePoints,
                                                                   fixedIndices,
                                                                   displayFrameAnchorIndex,
                                                                   config,
                                                                   -1,
                                                                   -1,
                                                                   "fiber-reopt+global");
                outputLine = result.line;
                outputLinePoints = linePointsFromModel(result.line);
                printResidualSummary(result.report);
                if (verbose) {
                    const PointMotionStats motionStats = pointMotionStats(fiber.linePoints, outputLinePoints);
                    printSegmentMotionTable(fiber.linePoints, outputLinePoints, fixedIndices);
                    std::cout << "Fiber reoptimization complete: points=" << result.line.points.size()
                              << " iterations=" << result.report.iterations
                              << " initial_rms=" << result.report.initialRms
                              << " final_rms=" << result.report.finalRms
                              << " residuals=" << result.report.residuals
                              << " valid_normals=" << result.report.validNormalSamples
                              << " invalid_normals=" << result.report.invalidNormalSamples
                              << " converged=" << result.report.converged
                              << " line_length=" << lineLength(outputLinePoints)
                              << "\n";
                    printPointMotionStats(motionStats);
                    printLosses(result.report);
                }
            } else {
                if (verbose) {
                    std::cout << "Loaded fiber without reoptimization: points="
                              << outputLine.points.size()
                              << " line_length=" << lineLength(outputLinePoints)
                              << '\n';
                }
            }
            if (verbose) {
                printLineViewDiagnostics(outputLine);
            }
            if (!effectiveOutputPath.empty() && !reinitFailedWithoutDebugOutput) {
                saveReoptimizedFiber(fiber, outputLinePoints, effectiveOutputPath);
                if (verbose) {
                    std::cout << "Saved " << (reinitReopt ? "reinitialized/reoptimized" :
                                              (reopt ? "reoptimized" : "original"))
                              << " fiber to " << effectiveOutputPath.string() << '\n';
                }
            } else if (!effectiveOutputPath.empty() && verbose) {
                std::cout << "Skipped --output because reinit failed without debug output\n";
            }
            if (wantsObjOutput) {
                const std::vector<cv::Vec3d> lineObjPoints = reinitReopt
                    ? linePointsSliceFromFixedIndices(outputLine, outputFixedIndices)
                    : outputLinePoints;
                writeLineViewObjOutput(outputLine,
                                       lineObjPoints,
                                       *textureVolume,
                                       textureLevel,
                                       stripRenderScale,
                                       objOutputDir);
                if (verbose) {
                    std::cout << "Saved OBJ line-view output to " << objOutputDir.string() << '\n';
                }
            }
            if (wantsStripOutput) {
                writeStripImages(outputLine,
                                 outputFixedIndices,
                                 *textureVolume,
                                 textureLevel,
                                 stripRenderScale,
                                 stripOutputDir);
                if (verbose) {
                    std::cout << "Saved strip render output to " << stripOutputDir.string() << '\n';
                }
            }
            return 0;
        }

        if (verbose) {
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
        }
        if (traceInit) {
            printInitTrace(seedPoint, config.initialTangent, sampler, config);
        }

        if (benchmarkThreads) {
            const std::vector<int> threadCounts{1, 2, 4, 8, 16, 32};
            std::cout.imbue(std::locale::classic());
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "Thread benchmark:\n"
                      << "threads   run       ms  ceres_ms prefetch_ms materialize_ms prefetch_calls chunks_read chunks_requested  iters    final_rms  normal_rms status\n";
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
                              << std::setw(13) << result.report.finalRms
                              << std::setw(12) << normalAlignmentRms(result.report)
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
                      << "solver                    threads   run       ms  ceres_ms prefetch_ms materialize_ms prefetch_calls chunks_read chunks_requested  iters    final_rms  normal_rms status\n";
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
                                  << std::setw(13) << result.report.finalRms
                                  << std::setw(12) << normalAlignmentRms(result.report)
                                  << " " << (result.report.converged ? "ok" : "not_converged") << '\n';
                    }
                }
            }
            return 0;
        }

        const auto result = optimizer.optimizeFromSeed(seedPoint, config);

        printResidualSummary(result.report);
        if (verbose) {
            std::cout << "Optimization complete: points=" << result.line.points.size()
                      << " iterations=" << result.report.iterations
                      << " initial_rms=" << result.report.initialRms
                      << " final_rms=" << result.report.finalRms
                      << " residuals=" << result.report.residuals
                      << " valid_normals=" << result.report.validNormalSamples
                      << " invalid_normals=" << result.report.invalidNormalSamples
                      << " converged=" << result.report.converged << "\n";
            if (!result.report.message.empty()) {
                std::cout << result.report.message << '\n';
            }
            printLosses(result.report);
            printLineViewDiagnostics(result.line);
        }
    } catch (const std::exception& ex) {
        std::cerr << "vc_lasagna_line_probe failed: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
