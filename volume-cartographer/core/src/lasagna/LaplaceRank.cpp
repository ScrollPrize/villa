#include "vc/lasagna/LaplaceRank.hpp"

#include "vc/core/util/Tiff.hpp"
#include "vc/lasagna/LaplaceAmgx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vc::lasagna {
namespace {

constexpr double kDefaultThreshold = 110.0;
constexpr int64_t kDefaultMarginBaseVoxels = 1000;
constexpr int kDefaultSourceDepth = 0;
constexpr double kDefaultAdaptiveStartLambda = 1.0 / 2048.0;
constexpr double kDefaultSolverTolerance = 1.0e-8;
constexpr double kDefaultConfidenceFactor = 100.0;

struct RankOptions {
    uint8_t threshold = static_cast<uint8_t>(kDefaultThreshold);
    int64_t marginBaseVoxels = kDefaultMarginBaseVoxels;
    int sourceDepth = kDefaultSourceDepth;
    double adaptiveStartLambda = kDefaultAdaptiveStartLambda;
    double solverTolerance = kDefaultSolverTolerance;
    double confidenceFactor = kDefaultConfidenceFactor;
    std::filesystem::path amgxConfig;
    std::filesystem::path debugDir;
    std::string requestId;
};

struct PairSetJob {
    std::string id;
    std::vector<MaxflowDouble3> sideA;
    std::vector<MaxflowDouble3> sideB;
};

[[nodiscard]] std::string sideKey(LaplaceRankSide side)
{
    return side == LaplaceRankSide::SideA ? "side_a" : "side_b";
}

[[nodiscard]] const std::vector<MaxflowDouble3>& pointsForSide(
    const PairSetJob& job,
    LaplaceRankSide side)
{
    return side == LaplaceRankSide::SideA ? job.sideA : job.sideB;
}

[[nodiscard]] const std::vector<MaxflowTerminal>& terminalsForSide(
    const MaxflowGraphBuildReport& report,
    LaplaceRankSide side)
{
    return side == LaplaceRankSide::SideA ? report.sources : report.sinks;
}

[[nodiscard]] double requireFiniteDouble(const nlohmann::json& value, const char* name)
{
    if (!value.is_number()) {
        throw std::invalid_argument(std::string(name) + " must be a number");
    }
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
    return parsed;
}

[[nodiscard]] int64_t requireInt64(const nlohmann::json& value, const char* name)
{
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return value.get<int64_t>();
}

[[nodiscard]] MaxflowDouble3 parsePoint(const nlohmann::json& value, std::string_view name)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(std::string(name) + " point must be [x, y, z]");
    }
    return {
        requireFiniteDouble(value[0], "point x"),
        requireFiniteDouble(value[1], "point y"),
        requireFiniteDouble(value[2], "point z"),
    };
}

[[nodiscard]] std::vector<MaxflowDouble3> parsePoints(
    const nlohmann::json& value,
    std::string_view name)
{
    if (!value.is_array()) {
        throw std::invalid_argument(std::string(name) + " must be an array of points");
    }
    std::vector<MaxflowDouble3> points;
    points.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        points.push_back(parsePoint(value[i], name));
    }
    if (points.empty()) {
        throw std::invalid_argument(std::string(name) + " must contain at least one point");
    }
    return points;
}

[[nodiscard]] std::string generatedRequestId()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

[[nodiscard]] std::string safePathComponent(const std::string& raw, const std::string& fallback)
{
    std::string out;
    out.reserve(raw.size());
    for (char ch : raw) {
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') ||
                        ch == '.' || ch == '_' || ch == '-';
        out.push_back(ok ? ch : '_');
    }
    if (out.empty()) {
        return fallback;
    }
    return out;
}

[[nodiscard]] nlohmann::json pointJson(const MaxflowDouble3& point)
{
    return nlohmann::json::array({point.x, point.y, point.z});
}

[[nodiscard]] nlohmann::json voxelJson(const MaxflowInt3& point)
{
    return nlohmann::json::array({point.x, point.y, point.z});
}

[[nodiscard]] nlohmann::json boxJson(const MaxflowBox3& box)
{
    return nlohmann::json::array({voxelJson(box.begin), voxelJson(box.end)});
}

[[nodiscard]] nlohmann::json pointsJson(const std::vector<MaxflowDouble3>& points)
{
    nlohmann::json out = nlohmann::json::array();
    for (const auto& point : points) {
        out.push_back(pointJson(point));
    }
    return out;
}

[[nodiscard]] nlohmann::json terminalsJson(const std::vector<MaxflowTerminal>& terminals)
{
    nlohmann::json out = nlohmann::json::array();
    for (const auto& terminal : terminals) {
        out.push_back(voxelJson(terminal.predVoxelXYZ));
    }
    return out;
}

[[nodiscard]] nlohmann::json terminalNodesJson(const std::vector<MaxflowTerminal>& terminals)
{
    nlohmann::json out = nlohmann::json::array();
    for (const auto& terminal : terminals) {
        out.push_back(terminal.node);
    }
    return out;
}

[[nodiscard]] std::array<size_t, 3> cropShapeZYX(const MaxflowPredDtVolume& pred)
{
    return {
        static_cast<size_t>(pred.cropPredXYZ.end.z - pred.cropPredXYZ.begin.z),
        static_cast<size_t>(pred.cropPredXYZ.end.y - pred.cropPredXYZ.begin.y),
        static_cast<size_t>(pred.cropPredXYZ.end.x - pred.cropPredXYZ.begin.x),
    };
}

[[nodiscard]] size_t passableOffset(
    size_t z,
    size_t y,
    size_t x,
    const std::array<size_t, 3>& shapeZYX)
{
    return (z * shapeZYX[1] + y) * shapeZYX[2] + x;
}

[[nodiscard]] int64_t clampPlane(int64_t value, int64_t begin, int64_t end)
{
    if (end <= begin) {
        return begin;
    }
    return std::min(std::max(value, begin), end - 1);
}

void overlayXY(cv::Mat& image, const MaxflowPredDtVolume& pred, const MaxflowInt3& point, int64_t z)
{
    if (point.z != z) {
        return;
    }
    const int64_t x = point.x - pred.cropPredXYZ.begin.x;
    const int64_t y = point.y - pred.cropPredXYZ.begin.y;
    if (x >= 0 && y >= 0 && x < image.cols && y < image.rows) {
        image.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = 255;
    }
}

void overlayXZ(cv::Mat& image, const MaxflowPredDtVolume& pred, const MaxflowInt3& point, int64_t y)
{
    if (point.y != y) {
        return;
    }
    const int64_t x = point.x - pred.cropPredXYZ.begin.x;
    const int64_t z = point.z - pred.cropPredXYZ.begin.z;
    if (x >= 0 && z >= 0 && x < image.cols && z < image.rows) {
        image.at<uint8_t>(static_cast<int>(z), static_cast<int>(x)) = 255;
    }
}

void overlayYZ(cv::Mat& image, const MaxflowPredDtVolume& pred, const MaxflowInt3& point, int64_t x)
{
    if (point.x != x) {
        return;
    }
    const int64_t y = point.y - pred.cropPredXYZ.begin.y;
    const int64_t z = point.z - pred.cropPredXYZ.begin.z;
    if (y >= 0 && z >= 0 && y < image.cols && z < image.rows) {
        image.at<uint8_t>(static_cast<int>(z), static_cast<int>(y)) = 255;
    }
}

[[nodiscard]] cv::Mat sliceXY(
    const MaxflowPredDtVolume& pred,
    const std::array<size_t, 3>& shape,
    int64_t globalZ)
{
    const auto z = static_cast<size_t>(globalZ - pred.cropPredXYZ.begin.z);
    cv::Mat image(static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_8UC1);
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[2]; ++x) {
            image.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) =
                pred.passable[passableOffset(z, y, x, shape)] ? 64 : 0;
        }
    }
    return image;
}

[[nodiscard]] cv::Mat sliceXZ(
    const MaxflowPredDtVolume& pred,
    const std::array<size_t, 3>& shape,
    int64_t globalY)
{
    const auto y = static_cast<size_t>(globalY - pred.cropPredXYZ.begin.y);
    cv::Mat image(static_cast<int>(shape[0]), static_cast<int>(shape[2]), CV_8UC1);
    for (size_t z = 0; z < shape[0]; ++z) {
        for (size_t x = 0; x < shape[2]; ++x) {
            image.at<uint8_t>(static_cast<int>(z), static_cast<int>(x)) =
                pred.passable[passableOffset(z, y, x, shape)] ? 64 : 0;
        }
    }
    return image;
}

[[nodiscard]] cv::Mat sliceYZ(
    const MaxflowPredDtVolume& pred,
    const std::array<size_t, 3>& shape,
    int64_t globalX)
{
    const auto x = static_cast<size_t>(globalX - pred.cropPredXYZ.begin.x);
    cv::Mat image(static_cast<int>(shape[0]), static_cast<int>(shape[1]), CV_8UC1);
    for (size_t z = 0; z < shape[0]; ++z) {
        for (size_t y = 0; y < shape[1]; ++y) {
            image.at<uint8_t>(static_cast<int>(z), static_cast<int>(y)) =
                pred.passable[passableOffset(z, y, x, shape)] ? 64 : 0;
        }
    }
    return image;
}

void writeDebugTiff(const std::filesystem::path& path, const cv::Mat& image)
{
    writeTiff(path, image, CV_8UC1, 0, 0);
}

void overlayAll(
    cv::Mat& image,
    const MaxflowPredDtVolume& pred,
    const std::vector<MaxflowTerminal>& sideA,
    const std::vector<MaxflowTerminal>& sideB,
    char plane,
    int64_t coord)
{
    const auto overlay = [&](const MaxflowTerminal& terminal) {
        if (plane == 'z') overlayXY(image, pred, terminal.predVoxelXYZ, coord);
        if (plane == 'y') overlayXZ(image, pred, terminal.predVoxelXYZ, coord);
        if (plane == 'x') overlayYZ(image, pred, terminal.predVoxelXYZ, coord);
    };
    for (const auto& terminal : sideA) overlay(terminal);
    for (const auto& terminal : sideB) overlay(terminal);
}

void writePointSlices(
    const std::filesystem::path& dir,
    const std::string& side,
    const MaxflowPredDtVolume& pred,
    const std::array<size_t, 3>& shape,
    const std::vector<MaxflowTerminal>& terminals)
{
    for (size_t i = 0; i < terminals.size(); ++i) {
        const auto& point = terminals[i].predVoxelXYZ;
        std::ostringstream prefix;
        prefix << side << "_" << std::setw(3) << std::setfill('0') << i;

        auto xy = sliceXY(pred, shape, clampPlane(point.z, pred.cropPredXYZ.begin.z, pred.cropPredXYZ.end.z));
        overlayXY(xy, pred, point, point.z);
        writeDebugTiff(dir / (prefix.str() + "_xy.tif"), xy);

        auto xz = sliceXZ(pred, shape, clampPlane(point.y, pred.cropPredXYZ.begin.y, pred.cropPredXYZ.end.y));
        overlayXZ(xz, pred, point, point.y);
        writeDebugTiff(dir / (prefix.str() + "_xz.tif"), xz);

        auto yz = sliceYZ(pred, shape, clampPlane(point.x, pred.cropPredXYZ.begin.x, pred.cropPredXYZ.end.x));
        overlayYZ(yz, pred, point, point.x);
        writeDebugTiff(dir / (prefix.str() + "_yz.tif"), yz);
    }
}

void writeDebugArtifacts(
    const std::filesystem::path& dir,
    const MaxflowGraphBuildReport& report,
    const nlohmann::json& meta)
{
    std::filesystem::create_directories(dir);
    const auto shape = cropShapeZYX(report.predDt);
    if (shape[0] != 0 && shape[1] != 0 && shape[2] != 0) {
        const int64_t zmid = report.predDt.cropPredXYZ.begin.z + static_cast<int64_t>(shape[0] / 2);
        const int64_t ymid = report.predDt.cropPredXYZ.begin.y + static_cast<int64_t>(shape[1] / 2);
        const int64_t xmid = report.predDt.cropPredXYZ.begin.x + static_cast<int64_t>(shape[2] / 2);

        auto xy = sliceXY(report.predDt, shape, zmid);
        overlayAll(xy, report.predDt, report.sources, report.sinks, 'z', zmid);
        writeDebugTiff(dir / "passable_xy_zmid.tif", xy);

        auto xz = sliceXZ(report.predDt, shape, ymid);
        overlayAll(xz, report.predDt, report.sources, report.sinks, 'y', ymid);
        writeDebugTiff(dir / "passable_xz_ymid.tif", xz);

        auto yz = sliceYZ(report.predDt, shape, xmid);
        overlayAll(yz, report.predDt, report.sources, report.sinks, 'x', xmid);
        writeDebugTiff(dir / "passable_yz_xmid.tif", yz);

        writePointSlices(dir, "side_a", report.predDt, shape, report.sources);
        writePointSlices(dir, "side_b", report.predDt, shape, report.sinks);
    }
    std::ofstream metaOut(dir / "meta.json");
    metaOut << meta.dump(2) << '\n';
}

[[nodiscard]] nlohmann::json optionsJson(const RankOptions& options)
{
    return {
        {"threshold", options.threshold},
        {"margin_base_voxels", options.marginBaseVoxels},
        {"source_depth", options.sourceDepth},
        {"adaptive_start_lambda", options.adaptiveStartLambda},
        {"confidence_factor", options.confidenceFactor},
        {"solver_tolerance", options.solverTolerance},
        {"amgx_config", options.amgxConfig.empty() ? nlohmann::json(nullptr)
                                                    : nlohmann::json(options.amgxConfig.string())},
        {"debug_dir", options.debugDir.empty() ? nlohmann::json(nullptr)
                                                : nlohmann::json(options.debugDir.string())},
        {"request_id", options.requestId},
    };
}

[[nodiscard]] nlohmann::json graphJson(const MaxflowGraphBuildReport& report)
{
    return {
        {"crop_base_xyz", boxJson(report.predDt.cropBaseXYZ)},
        {"crop_pred_xyz", boxJson(report.predDt.cropPredXYZ)},
        {"passable_voxels", report.graph.stats.passableVoxels},
        {"graph_nodes", report.graph.stats.graphNodes},
        {"directed_edges", report.graph.stats.directedEdges},
    };
}

[[nodiscard]] RankOptions parseOptions(const nlohmann::json& request)
{
    RankOptions options;
    options.requestId = generatedRequestId();
    const nlohmann::json empty = nlohmann::json::object();
    const auto& raw = request.contains("options") && !request["options"].is_null()
        ? request["options"]
        : empty;
    if (!raw.is_object()) {
        throw std::invalid_argument("options must be an object");
    }
    if (raw.contains("threshold") && !raw["threshold"].is_null()) {
        const auto threshold = requireInt64(raw["threshold"], "threshold");
        if (threshold < 0 || threshold > 255) {
            throw std::invalid_argument("threshold must be in [0, 255]");
        }
        options.threshold = static_cast<uint8_t>(threshold);
    }
    if (raw.contains("margin_base_voxels") && !raw["margin_base_voxels"].is_null()) {
        options.marginBaseVoxels = requireInt64(raw["margin_base_voxels"], "margin_base_voxels");
        if (options.marginBaseVoxels < 0) {
            throw std::invalid_argument("margin_base_voxels must be non-negative");
        }
    }
    if (raw.contains("source_depth") && !raw["source_depth"].is_null()) {
        const auto sourceDepth = requireInt64(raw["source_depth"], "source_depth");
        if (sourceDepth < 0 || sourceDepth > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("source_depth must be a non-negative int32 value");
        }
        options.sourceDepth = static_cast<int>(sourceDepth);
    }
    for (const char* key : {"adaptive_start_lambda", "solver_tolerance", "confidence_factor"}) {
        if (raw.contains(key)) {
            throw std::invalid_argument(
                std::string(key) + " is server-owned and must not be supplied");
        }
    }
    if (raw.contains("amgx_config") && !raw["amgx_config"].is_null()) {
        if (!raw["amgx_config"].is_string()) {
            throw std::invalid_argument("amgx_config must be a path string or null");
        }
        options.amgxConfig = raw["amgx_config"].get<std::string>();
    }
    if (raw.contains("debug_dir") && !raw["debug_dir"].is_null()) {
        if (!raw["debug_dir"].is_string()) {
            throw std::invalid_argument("debug_dir must be a path string or null");
        }
        options.debugDir = raw["debug_dir"].get<std::string>();
    }
    if (raw.contains("request_id") && !raw["request_id"].is_null()) {
        if (!raw["request_id"].is_string()) {
            throw std::invalid_argument("request_id must be a string");
        }
        options.requestId = safePathComponent(raw["request_id"].get<std::string>(), options.requestId);
    }
    return options;
}

[[nodiscard]] std::vector<PairSetJob> parseJobs(const nlohmann::json& request)
{
    if (!request.contains("jobs") || !request["jobs"].is_array()) {
        throw std::invalid_argument("jobs must be an array");
    }
    std::vector<PairSetJob> jobs;
    jobs.reserve(request["jobs"].size());
    for (size_t i = 0; i < request["jobs"].size(); ++i) {
        const auto& raw = request["jobs"][i];
        if (!raw.is_object()) {
            throw std::invalid_argument("each job must be an object");
        }
        PairSetJob job;
        job.id = raw.value("id", "");
        if (job.id.empty()) {
            job.id = "job_" + std::to_string(i);
        }
        if (!raw.contains("side_a") || !raw.contains("side_b")) {
            throw std::invalid_argument("each job must contain side_a and side_b");
        }
        job.sideA = parsePoints(raw["side_a"], "side_a");
        job.sideB = parsePoints(raw["side_b"], "side_b");
        jobs.push_back(std::move(job));
    }
    return jobs;
}

[[nodiscard]] nlohmann::json sideResultJson(
    const std::vector<MaxflowDouble3>& inputPoints,
    const std::vector<MaxflowTerminal>& terminals)
{
    return {
        {"input_points", pointsJson(inputPoints)},
        {"resolved_pred_voxels", terminalsJson(terminals)},
        {"graph_nodes", terminalNodesJson(terminals)},
    };
}

[[nodiscard]] LaplaceRankEvaluation evaluatePairSet(
    const MaxflowGraph& graph,
    const std::vector<MaxflowTerminal>& solveTerminals,
    const std::vector<MaxflowTerminal>& targetTerminals,
    double lambda,
    const RankOptions& options)
{
    LaplaceRankEvaluation evaluation;
    evaluation.lambda = lambda;
    evaluation.solved = true;
    evaluation.status = "success";
    evaluation.values.reserve(solveTerminals.size() * targetTerminals.size());

    for (const auto& solveTerminal : solveTerminals) {
        LaplaceAmgxOptions laplaceOptions;
        laplaceOptions.lambda = lambda;
        laplaceOptions.sourceDepth = options.sourceDepth;
        laplaceOptions.configPath = options.amgxConfig;
        const auto result = solveScreenedLaplaceAmgx(graph, solveTerminal.node, laplaceOptions);
        evaluation.solveSecondsTotal += result.solveSeconds;
        if (!result.success) {
            evaluation.solved = false;
            evaluation.status = result.status.empty() ? "failed" : result.status;
            break;
        }
        for (const auto& targetTerminal : targetTerminals) {
            if (targetTerminal.node < 0 ||
                static_cast<uint64_t>(targetTerminal.node) >= graph.stats.graphNodes) {
                throw std::runtime_error("Laplace target node is outside graph");
            }
            const double value = result.valuesByNode[static_cast<size_t>(targetTerminal.node)];
            evaluation.values.push_back(value);
            evaluation.maxAbsValue = std::max(evaluation.maxAbsValue, std::abs(value));
        }
    }
    return evaluation;
}

[[nodiscard]] nlohmann::json valuesJson(
    const LaplaceRankEvaluation& evaluation,
    const LaplaceRankRoles& roles)
{
    nlohmann::json values = nlohmann::json::array();
    size_t offset = 0;
    for (size_t solveIndex = 0; solveIndex < roles.solveCount; ++solveIndex) {
        for (size_t targetIndex = 0; targetIndex < roles.targetCount; ++targetIndex) {
            values.push_back({
                {"solve_side", sideKey(roles.solveSide)},
                {"solve_index", solveIndex},
                {"target_side", sideKey(roles.targetSide)},
                {"target_index", targetIndex},
                {"value", evaluation.values.at(offset++)},
            });
        }
    }
    return values;
}

[[nodiscard]] nlohmann::json errorResult(
    const PairSetJob& job,
    std::string code,
    std::string message)
{
    return {
        {"id", job.id},
        {"status", "error"},
        {"error", {
            {"code", std::move(code)},
            {"message", std::move(message)},
        }},
    };
}

[[nodiscard]] std::string rankResultErrorCode(const nlohmann::json& result)
{
    if (result.contains("error") && result["error"].is_object()) {
        return result["error"].value("code", std::string{});
    }
    return {};
}

void logRankJobProgress(size_t index,
                        size_t count,
                        const nlohmann::json& result,
                        double elapsedSeconds)
{
    std::cerr << "[fit-service] /laplace/rank pair "
              << (index + 1) << "/" << count
              << " id=" << result.value("id", std::string{"<missing>"})
              << " status=" << result.value("status", std::string{"<missing>"});
    const std::string code = rankResultErrorCode(result);
    if (!code.empty()) {
        std::cerr << " code=" << code;
    }
    if (result.contains("graph") && result["graph"].is_object()) {
        const auto& graph = result["graph"];
        if (graph.contains("graph_nodes") && graph["graph_nodes"].is_number_unsigned()) {
            std::cerr << " nodes=" << graph["graph_nodes"].get<uint64_t>();
        }
        if (graph.contains("passable_voxels") && graph["passable_voxels"].is_number_unsigned()) {
            std::cerr << " passable=" << graph["passable_voxels"].get<uint64_t>();
        }
    }
    if (result.contains("selected_lambda") && result["selected_lambda"].is_number()) {
        std::cerr << " lambda=" << result["selected_lambda"].get<double>();
    } else if (result.contains("rejected_lambda") && result["rejected_lambda"].is_number()) {
        std::cerr << " rejected_lambda=" << result["rejected_lambda"].get<double>();
    }
    if (result.contains("max_abs_value") && result["max_abs_value"].is_number()) {
        std::cerr << " max_abs=" << result["max_abs_value"].get<double>();
    }
    if (result.contains("solve_seconds_total") &&
        result["solve_seconds_total"].is_number()) {
        std::cerr << " solve_s=" << result["solve_seconds_total"].get<double>();
    }
    if (result.contains("debug_dir") && result["debug_dir"].is_string()) {
        std::cerr << " debug_dir=" << result["debug_dir"].get<std::string>();
    }
    std::cerr << " elapsed_s=" << elapsedSeconds << std::endl;
}

[[nodiscard]] bool isAmgxRuntimeUnavailable(std::string_view message)
{
    return message.find("AMGX_") != std::string_view::npos ||
           message.find("AMGX support is disabled") != std::string_view::npos ||
           message.find("no CUDA-capable device") != std::string_view::npos ||
           message.find("Cuda failure") != std::string_view::npos;
}

void addResolvedContext(
    nlohmann::json& out,
    const PairSetJob& job,
    const MaxflowGraphBuildReport& report,
    const LaplaceRankRoles& roles,
    double confidenceFloor)
{
    out["confidence_floor"] = confidenceFloor;
    out["roles"] = {
        {"solve_side", sideKey(roles.solveSide)},
        {"target_side", sideKey(roles.targetSide)},
    };
    out["side_a"] = sideResultJson(job.sideA, report.sources);
    out["side_b"] = sideResultJson(job.sideB, report.sinks);
    out["graph"] = graphJson(report);
}

[[nodiscard]] nlohmann::json rankJob(
    const std::filesystem::path& manifestPath,
    const PairSetJob& job,
    const RankOptions& options,
    size_t jobIndex)
{
    const double confidenceFloor = options.solverTolerance * options.confidenceFactor;
    const auto roles = selectLaplaceRankRoles(job.sideA.size(), job.sideB.size());
    const std::filesystem::path debugJobDir = options.debugDir.empty()
        ? std::filesystem::path()
        : options.debugDir / options.requestId /
              safePathComponent(job.id, "job_" + std::to_string(jobIndex));

    if (!laplaceAmgxAvailable()) {
        auto out = errorResult(
            job,
            "amgx_unavailable",
            "AMGX support is disabled; configure volume-cartographer with -DVC_ENABLE_AMGX=ON");
        if (!debugJobDir.empty()) {
            std::filesystem::create_directories(debugJobDir);
            std::ofstream metaOut(debugJobDir / "meta.json");
            metaOut << nlohmann::json({
                {"id", job.id},
                {"status", "error"},
                {"options", optionsJson(options)},
                {"error", out["error"]},
            }).dump(2) << '\n';
            out["debug_dir"] = debugJobDir.string();
        }
        return out;
    }

    MaxflowManifestBuildOptions buildOptions;
    buildOptions.sourcesBase = job.sideA;
    buildOptions.sinksBase = job.sideB;
    buildOptions.sourceBase = job.sideA.front();
    buildOptions.sinkBase = job.sideB.front();
    buildOptions.marginBaseVoxels = options.marginBaseVoxels;
    buildOptions.threshold = options.threshold;

    const auto report = buildMaxflowGraphFromManifest(manifestPath, buildOptions);
    const auto& solveTerminals = terminalsForSide(report, roles.solveSide);
    const auto& targetTerminals = terminalsForSide(report, roles.targetSide);
    const auto evaluator = [&](double lambda) {
        return evaluatePairSet(report.graph, solveTerminals, targetTerminals, lambda, options);
    };
    LaplaceRankEvaluation selected;
    try {
        selected = selectAdaptiveLaplaceRankLambda(
            options.adaptiveStartLambda,
            confidenceFloor,
            evaluator);
    } catch (const std::exception& e) {
        auto out = errorResult(
            job,
            isAmgxRuntimeUnavailable(e.what()) ? "amgx_unavailable" : "rank_failed",
            e.what());
        out["selected_lambda"] = nullptr;
        addResolvedContext(out, job, report, roles, confidenceFloor);
        if (!debugJobDir.empty()) {
            out["debug_dir"] = debugJobDir.string();
            writeDebugArtifacts(debugJobDir, report, out);
        }
        return out;
    }
    if (!laplaceRankAccepted(selected, confidenceFloor)) {
        std::ostringstream message;
        message << "adaptive lambda search found no accepted value"
                << "; selected_status=" << (selected.status.empty() ? "<empty>" : selected.status)
                << "; selected_lambda=" << selected.lambda
                << "; max_abs_value=" << selected.maxAbsValue
                << "; confidence_floor=" << confidenceFloor
                << "; solved=" << (selected.solved ? "true" : "false");
        auto out = errorResult(
            job,
            "no_accepted_lambda",
            message.str());
        out["selected_lambda"] = nullptr;
        out["rejected_lambda"] = selected.lambda;
        out["max_abs_value"] = selected.maxAbsValue;
        out["solve_seconds_total"] = selected.solveSecondsTotal;
        addResolvedContext(out, job, report, roles, confidenceFloor);
        if (!debugJobDir.empty()) {
            out["debug_dir"] = debugJobDir.string();
            writeDebugArtifacts(debugJobDir, report, out);
        }
        return out;
    }

    nlohmann::json out = {
        {"id", job.id},
        {"status", "success"},
        {"selected_lambda", selected.lambda},
        {"confidence_floor", confidenceFloor},
        {"roles", {
            {"solve_side", sideKey(roles.solveSide)},
            {"target_side", sideKey(roles.targetSide)},
        }},
        {"side_a", sideResultJson(job.sideA, report.sources)},
        {"side_b", sideResultJson(job.sideB, report.sinks)},
        {"values", valuesJson(selected, roles)},
        {"max_abs_value", selected.maxAbsValue},
        {"solve_seconds_total", selected.solveSecondsTotal},
        {"graph", graphJson(report)},
    };
    if (!debugJobDir.empty()) {
        out["debug_dir"] = debugJobDir.string();
        nlohmann::json meta = out;
        meta["options"] = optionsJson(options);
        writeDebugArtifacts(debugJobDir, report, meta);
    }
    return out;
}

} // namespace

const char* toString(LaplaceRankSide side) noexcept
{
    switch (side) {
    case LaplaceRankSide::SideA:
        return "side_a";
    case LaplaceRankSide::SideB:
        return "side_b";
    }
    return "unknown";
}

LaplaceRankRoles selectLaplaceRankRoles(size_t sideACount, size_t sideBCount)
{
    if (sideACount == 0 || sideBCount == 0) {
        throw std::invalid_argument("side_a and side_b must both contain at least one point");
    }
    if (sideACount >= sideBCount) {
        return {LaplaceRankSide::SideA, LaplaceRankSide::SideB, sideACount, sideBCount};
    }
    return {LaplaceRankSide::SideB, LaplaceRankSide::SideA, sideBCount, sideACount};
}

bool laplaceRankAccepted(const LaplaceRankEvaluation& evaluation, double confidenceFloor)
{
    return evaluation.solved && evaluation.maxAbsValue >= confidenceFloor;
}

LaplaceRankEvaluation selectAdaptiveLaplaceRankLambda(
    double startLambda,
    double confidenceFloor,
    const LaplaceRankEvaluator& evaluator)
{
    if (!(startLambda > 0.0) || !std::isfinite(startLambda)) {
        throw std::invalid_argument("adaptive start lambda must be finite and positive");
    }
    if (!(confidenceFloor > 0.0) || !std::isfinite(confidenceFloor)) {
        throw std::invalid_argument("confidence floor must be finite and positive");
    }

    constexpr double bracketFactor = 16.0;
    constexpr double minRefineFactor = 2.0;
    constexpr int maxProbeSteps = 16;
    constexpr int maxRefineSteps = 16;

    LaplaceRankEvaluation best;
    bool haveBest = false;
    double rejectedHigh = 0.0;
    double acceptedLow = 0.0;

    auto start = evaluator(startLambda);
    if (laplaceRankAccepted(start, confidenceFloor)) {
        best = start;
        haveBest = true;
        acceptedLow = start.lambda;
        double probeLambda = start.lambda;
        for (int step = 0; step < maxProbeSteps; ++step) {
            probeLambda *= bracketFactor;
            auto probe = evaluator(probeLambda);
            if (laplaceRankAccepted(probe, confidenceFloor)) {
                best = probe;
                acceptedLow = probe.lambda;
            } else {
                rejectedHigh = probe.lambda;
                break;
            }
        }
    } else {
        rejectedHigh = start.lambda;
        double probeLambda = start.lambda;
        for (int step = 0; step < maxProbeSteps; ++step) {
            probeLambda /= bracketFactor;
            auto probe = evaluator(probeLambda);
            if (laplaceRankAccepted(probe, confidenceFloor)) {
                best = probe;
                haveBest = true;
                acceptedLow = probe.lambda;
                break;
            }
        }
    }

    if (haveBest && rejectedHigh > acceptedLow && acceptedLow > 0.0) {
        for (int step = 0;
             step < maxRefineSteps && rejectedHigh / acceptedLow > minRefineFactor;
             ++step) {
            const double midLambda = std::sqrt(rejectedHigh * acceptedLow);
            auto refine = evaluator(midLambda);
            if (laplaceRankAccepted(refine, confidenceFloor)) {
                best = refine;
                acceptedLow = refine.lambda;
            } else {
                rejectedHigh = refine.lambda;
            }
        }
    }

    return haveBest ? best : start;
}

nlohmann::json rankSnapPairsJson(const nlohmann::json& request)
{
    if (!request.is_object()) {
        throw std::invalid_argument("request must be an object");
    }
    if (!request.contains("manifest") || !request["manifest"].is_string()) {
        throw std::invalid_argument("manifest must be a path string");
    }
    const std::filesystem::path manifestPath = request["manifest"].get<std::string>();
    const RankOptions options = parseOptions(request);
    const auto jobs = parseJobs(request);

    nlohmann::json results = nlohmann::json::array();
    std::cerr << "[fit-service] /laplace/rank batch start"
              << " request_id=" << options.requestId
              << " jobs=" << jobs.size()
              << " manifest=" << manifestPath.string()
              << " threshold=" << static_cast<int>(options.threshold)
              << " margin_base_voxels=" << options.marginBaseVoxels
              << " source_depth=" << options.sourceDepth
              << " adaptive_start_lambda=" << options.adaptiveStartLambda
              << " confidence_floor=" << (options.solverTolerance * options.confidenceFactor)
              << " debug_dir="
              << (options.debugDir.empty() ? std::string{"<none>"} : options.debugDir.string())
              << std::endl;
    const auto batchStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto jobStart = std::chrono::steady_clock::now();
        nlohmann::json result;
        try {
            result = rankJob(manifestPath, jobs[i], options, i);
        } catch (const std::exception& e) {
            result = errorResult(jobs[i], "rank_failed", e.what());
        }
        const double elapsedSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - jobStart).count();
        logRankJobProgress(i, jobs.size(), result, elapsedSeconds);
        results.push_back(std::move(result));
    }
    const double batchSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - batchStart).count();
    std::cerr << "[fit-service] /laplace/rank batch finished"
              << " request_id=" << options.requestId
              << " jobs=" << jobs.size()
              << " elapsed_s=" << batchSeconds
              << std::endl;
    return {
        {"results", results},
        {"options", optionsJson(options)},
        {"request_id", options.requestId},
    };
}

} // namespace vc::lasagna
