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
    std::cerr << "Usage: " << argv0 << " <manifest.lasagna.json> [--constant-normal-jacobian] [--benchmark-solvers] [--benchmark-threads] [--trace-init] [--segments-per-side=N] [--seed=x,y,z]\n"
              << "       " << argv0 << " <manifest.lasagna.json> --fiber <fiber.json> [--reopt|--reinit-reopt] [--output <fiber.json>] [--obj-output-dir <dir>] [--strip-output-dir <dir>] [--texture-zarr <zarr>] [--texture-level N] [--strip-render-scale N] [--constant-normal-jacobian]\n"
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

void printReinitSpanTable(const std::vector<vc::lasagna::LineReinitializationSpanReport>& spans)
{
    std::cout.imbue(std::locale::classic());
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Reinit segment candidates:\n"
              << "seg lcp rcp pts lsgn lnear rsgn rnear linit lfinal rinit rfinal mxstep mxtan mxnorm mxalign pick\n";
    for (const auto& span : spans) {
        std::cout << std::right << std::setw(3) << span.segmentIndex
                  << std::setw(8) << span.leftControlIndex
                  << std::setw(9) << span.rightControlIndex
                  << std::setw(7) << span.points
                  << std::setw(10) << span.candLeftSelectedSign
                  << std::setw(13) << span.candLeftClosestTargetDistance
                  << std::setw(11) << span.candRightSelectedSign
                  << std::setw(14) << span.candRightClosestTargetDistance
                  << std::setw(15) << span.candLeftInitialCost
                  << std::setw(16) << span.candLeftFinalCost
                  << std::setw(16) << span.candRightInitialCost
                  << std::setw(17) << span.candRightFinalCost
                  << std::setw(14) << span.chosenMaxEvenStepDeviation
                  << std::setw(15) << span.chosenMaxTangentSmoothDeviation
                  << std::setw(16) << span.chosenMaxNormalSmoothDeviation
                  << std::setw(15) << span.chosenMaxNormalAlignmentAbs
                  << ' ' << span.chosen << '\n';
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

cv::Mat renderSurfaceTexture(const QuadSurface& surface,
                             Volume& textureVolume,
                             int textureLevel,
                             int renderScale)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        throw std::runtime_error("surface has no points for texture rendering");
    }
    renderScale = std::max(1, renderScale);
    cv::Mat_<cv::Vec3f> coords;
    if (renderScale == 1) {
        coords = points->clone();
    } else {
        cv::resize(*points,
                   coords,
                   cv::Size(points->cols * renderScale, points->rows * renderScale),
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
    std::cout << "Strip texture sampling: start_level=" << startLevel
              << " render_scale=" << renderScale
              << " size=" << sampled.cols << "x" << sampled.rows
              << " covered=" << covered << "/" << total
              << " requested_chunks=" << stats.requestedChunks
              << " error_chunks=" << stats.errorChunks << '\n';
    return sampled;
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
    std::filesystem::path fiberPath;
    std::filesystem::path outputPath;
    std::filesystem::path objOutputDir;
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
        const bool wantsStripOutput = !stripOutputDir.empty();
        const bool wantsTextureOutput = wantsObjOutput || wantsStripOutput;
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
        if (!textureZarrPath.empty() && !wantsTextureOutput) {
            throw std::invalid_argument("--texture-zarr requires --obj-output-dir or --strip-output-dir");
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
        config.normalStraightnessWeight = 0.05;
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

            std::cout << "Fiber input:\n"
                      << "fiber=" << fiberPath.string() << '\n'
                      << "manifest=" << manifestPath.string() << '\n'
                      << "output=" << (outputPath.empty() ? std::string{"<none>"} : outputPath.string()) << '\n'
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

            std::vector<cv::Vec3d> outputLinePoints = fiber.linePoints;
            vc::lasagna::LineModel outputLine =
                makeLineModelFromPoints(fiber.linePoints, sampler, displayFrameAnchorIndex);
            std::vector<int> outputFixedIndices = fixedIndices;

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
                const PointMotionStats motionStats = pointMotionStats(fiber.linePoints, outputLinePoints);
                printReinitSpanTable(result.spans);
                std::cout << "max_segment_candidate_final_cost_diff="
                          << result.maxSegmentCandidateFinalCostDiff << '\n';
                printSegmentMotionTable(fiber.linePoints, outputLinePoints, fixedIndices);
                std::cout << "Fiber reinit reoptimization complete: points="
                          << result.optimization.line.points.size()
                          << " iterations=" << result.optimization.report.iterations
                          << " initial_cost=" << result.optimization.report.initialCost
                          << " final_cost=" << result.optimization.report.finalCost
                          << " valid_normals=" << result.optimization.report.validNormalSamples
                          << " invalid_normals=" << result.optimization.report.invalidNormalSamples
                          << " converged=" << result.optimization.report.converged
                          << " line_length=" << lineLength(outputLinePoints)
                          << "\n";
                std::cout << "Fit: initial=" << result.optimization.report.initialCost
                          << " final=" << result.optimization.report.finalCost
                          << " change=" << (result.optimization.report.initialCost -
                                            result.optimization.report.finalCost)
                          << '\n';
                printPointMotionStats(motionStats);
                printLosses(result.optimization.report);
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
                const PointMotionStats motionStats = pointMotionStats(fiber.linePoints, outputLinePoints);
                printSegmentMotionTable(fiber.linePoints, outputLinePoints, fixedIndices);
                std::cout << "Fiber reoptimization complete: points=" << result.line.points.size()
                          << " iterations=" << result.report.iterations
                          << " initial_cost=" << result.report.initialCost
                          << " final_cost=" << result.report.finalCost
                          << " valid_normals=" << result.report.validNormalSamples
                          << " invalid_normals=" << result.report.invalidNormalSamples
                          << " converged=" << result.report.converged
                          << " line_length=" << lineLength(outputLinePoints)
                          << "\n";
                std::cout << "Fit: initial=" << result.report.initialCost
                          << " final=" << result.report.finalCost
                          << " change=" << (result.report.initialCost - result.report.finalCost)
                          << '\n';
                printPointMotionStats(motionStats);
                printLosses(result.report);
            } else {
                std::cout << "Loaded fiber without reoptimization: points="
                          << outputLine.points.size()
                          << " line_length=" << lineLength(outputLinePoints)
                          << '\n';
            }
            printLineViewDiagnostics(outputLine);
            if (!outputPath.empty()) {
                saveReoptimizedFiber(fiber, outputLinePoints, outputPath);
                std::cout << "Saved " << (reinitReopt ? "reinitialized/reoptimized" :
                                          (reopt ? "reoptimized" : "original"))
                          << " fiber to " << outputPath.string() << '\n';
            }
            if (wantsObjOutput) {
                writeLineViewObjOutput(outputLine,
                                       outputLinePoints,
                                       *textureVolume,
                                       textureLevel,
                                       stripRenderScale,
                                       objOutputDir);
                std::cout << "Saved OBJ line-view output to " << objOutputDir.string() << '\n';
            }
            if (wantsStripOutput) {
                writeStripImages(outputLine,
                                 outputFixedIndices,
                                 *textureVolume,
                                 textureLevel,
                                 stripRenderScale,
                                 stripOutputDir);
                std::cout << "Saved strip render output to " << stripOutputDir.string() << '\n';
            }
            return 0;
        }

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
