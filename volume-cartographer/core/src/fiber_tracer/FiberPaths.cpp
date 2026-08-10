#include "vc/fiber_tracer/FiberPaths.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

using Clock = std::chrono::steady_clock;
using Voxel = std::array<int64_t, 3>;  // XYZ

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct VoxelHash {
    size_t operator()(const Voxel& voxel) const noexcept
    {
        size_t hash = 1469598103934665603ULL;
        for (const int64_t value : voxel) {
            hash ^= std::hash<int64_t>{}(value);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

struct FlatAnchor {
    FiberletAnchorId id;
    FiberAnchor anchor;
};

struct Move {
    Voxel delta{0, 0, 0};
    cv::Vec3d direction{1.0, 0.0, 0.0};
    double length = 1.0;
};

struct Attachment {
    Voxel voxel{0, 0, 0};
    cv::Vec3d direction{1.0, 0.0, 0.0};
    double length = 0.0;
};

struct SearchNode {
    Voxel voxel{0, 0, 0};
    double progress = 0.0;
    FiberStoredPredictionSample prediction;
    vc::lasagna::NormalSample normal;
    double quantizationFloorRadians = 0.0;
};

struct BackPointer {
    int64_t node = -1;
    int state = -1;
};

struct DpState {
    bool reached = false;
    FiberletPathCost cost;
    BackPointer previous;
};

double vectorLength(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3d normalized(const cv::Vec3d& value)
{
    const double length = vectorLength(value);
    if (!(length > kEpsilon) || !std::isfinite(length))
        return {0.0, 0.0, 0.0};
    return value / length;
}

cv::Vec3d voxelPoint(const Voxel& voxel)
{
    return {static_cast<double>(voxel[0]), static_cast<double>(voxel[1]), static_cast<double>(voxel[2])};
}

double axisAngle(const cv::Vec3d& left, const cv::Vec3d& right)
{
    const cv::Vec3d a = normalized(left);
    const cv::Vec3d b = normalized(right);
    return std::acos(std::clamp(std::abs(a.dot(b)), 0.0, 1.0));
}

double directedAngle(const cv::Vec3d& left, const cv::Vec3d& right)
{
    const cv::Vec3d a = normalized(left);
    const cv::Vec3d b = normalized(right);
    return std::acos(std::clamp(a.dot(b), -1.0, 1.0));
}

bool finiteVector(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool insideGrid(const Voxel& voxel, const FiberPredictionGridInfo& grid)
{
    return voxel[0] >= 0 && voxel[1] >= 0 && voxel[2] >= 0 && static_cast<uint64_t>(voxel[0]) < grid.shapeZYX[2] &&
           static_cast<uint64_t>(voxel[1]) < grid.shapeZYX[1] && static_cast<uint64_t>(voxel[2]) < grid.shapeZYX[0];
}

std::array<size_t, 3> storedIndex(const Voxel& voxel)
{
    return {static_cast<size_t>(voxel[2]), static_cast<size_t>(voxel[1]), static_cast<size_t>(voxel[0])};
}

std::array<size_t, 3> jsonSize3(const nlohmann::json& value, const char* name)
{
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error(std::string(name) + " must contain three integers");
    std::array<size_t, 3> out{};
    for (size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number_unsigned() && !value[index].is_number_integer())
            throw std::runtime_error(std::string(name) + " must contain integers");
        const int64_t item = value[index].get<int64_t>();
        if (item < 0)
            throw std::runtime_error(std::string(name) + " must be non-negative");
        out[index] = static_cast<size_t>(item);
    }
    return out;
}

cv::Vec3d jsonVec3(const nlohmann::json& value, const char* name)
{
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error(std::string(name) + " must contain three numbers");
    cv::Vec3d out;
    for (size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number())
            throw std::runtime_error(std::string(name) + " must contain numbers");
        out[static_cast<int>(index)] = value[index].get<double>();
    }
    if (!finiteVector(out))
        throw std::runtime_error(std::string(name) + " must be finite");
    return out;
}

double finiteNumber(const nlohmann::json& value, const char* name)
{
    if (!value.is_number())
        throw std::runtime_error(std::string(name) + " must be numeric");
    const double out = value.get<double>();
    if (!std::isfinite(out))
        throw std::runtime_error(std::string(name) + " must be finite");
    return out;
}

std::vector<FlatAnchor> flattenAnchors(const FiberAnchorExtractionReport& report)
{
    std::vector<FlatAnchor> anchors;
    for (const auto& cell : report.nonEmptyCells) {
        for (size_t component = 0; component < cell.components.size(); ++component) {
            if (!cell.components[component].retained)
                continue;
            anchors.push_back({{cell.cellZYX, component}, cell.components[component].anchor});
        }
    }
    return anchors;
}

std::vector<cv::Vec3d> hermitePolyline(const cv::Vec3d& start, const cv::Vec3d& target, const cv::Vec3d& startAxis, const cv::Vec3d& targetAxis)
{
    const double distance = vectorLength(target - start);
    const size_t segments = static_cast<size_t>(std::max(8.0, std::ceil(4.0 * distance)));
    std::vector<cv::Vec3d> points;
    points.reserve(segments + 1);
    const cv::Vec3d firstDerivative = startAxis * distance;
    const cv::Vec3d secondDerivative = targetAxis * distance;
    for (size_t index = 0; index <= segments; ++index) {
        const double t = static_cast<double>(index) / static_cast<double>(segments);
        const double t2 = t * t;
        const double t3 = t2 * t;
        points.push_back(
            start * (2.0 * t3 - 3.0 * t2 + 1.0) + firstDerivative * (t3 - 2.0 * t2 + t) + target * (-2.0 * t3 + 3.0 * t2) +
            secondDerivative * (t3 - t2));
    }
    return points;
}

double pointSegmentDistanceSquared(const cv::Vec3d& point, const cv::Vec3d& start, const cv::Vec3d& target)
{
    const cv::Vec3d delta = target - start;
    const double denominator = delta.dot(delta);
    if (!(denominator > kEpsilon))
        return (point - start).dot(point - start);
    const double t = std::clamp((point - start).dot(delta) / denominator, 0.0, 1.0);
    const cv::Vec3d residual = point - (start + delta * t);
    return residual.dot(residual);
}

bool insideCorridor(const cv::Vec3d& point, const std::vector<cv::Vec3d>& reference, double radiusSquared)
{
    for (size_t index = 1; index < reference.size(); ++index) {
        if (pointSegmentDistanceSquared(point, reference[index - 1], reference[index]) <= radiusSquared) {
            return true;
        }
    }
    return false;
}

std::vector<Attachment> endpointAttachments(
    const cv::Vec3d& endpoint, const cv::Vec3d& orientedAxis, const cv::Vec3d& chord, bool source, double maximumAngleRadians, const FiberPredictionGridInfo& grid)
{
    Voxel lower{};
    Voxel upper{};
    for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = static_cast<int64_t>(std::floor(endpoint[axis])) - 1;
        upper[axis] = static_cast<int64_t>(std::ceil(endpoint[axis])) + 1;
    }
    constexpr double maximumAttachmentLengthSquared = 3.0;
    std::vector<Attachment> out;
    for (int64_t z = lower[2]; z <= upper[2]; ++z) {
        for (int64_t y = lower[1]; y <= upper[1]; ++y) {
            for (int64_t x = lower[0]; x <= upper[0]; ++x) {
                const Voxel voxel{x, y, z};
                if (!insideGrid(voxel, grid))
                    continue;
                cv::Vec3d delta = source ? voxelPoint(voxel) - endpoint : endpoint - voxelPoint(voxel);
                const double length = vectorLength(delta);
                if (length * length > maximumAttachmentLengthSquared + kEpsilon)
                    continue;
                if (!(length > kEpsilon)) {
                    out.push_back({voxel, orientedAxis, 0.0});
                    continue;
                }
                const cv::Vec3d direction = delta / length;
                if (!(direction.dot(chord) > kEpsilon) || directedAngle(direction, orientedAxis) > maximumAngleRadians + kEpsilon) {
                    continue;
                }
                out.push_back({voxel, direction, length});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Attachment& left, const Attachment& right) { return left.voxel < right.voxel; });
    out.erase(std::unique(out.begin(), out.end(), [](const Attachment& left, const Attachment& right) { return left.voxel == right.voxel; }), out.end());
    return out;
}

std::vector<Move> forwardMoves(const cv::Vec3d& chord)
{
    std::vector<Move> moves;
    for (int64_t z = -1; z <= 1; ++z) {
        for (int64_t y = -1; y <= 1; ++y) {
            for (int64_t x = -1; x <= 1; ++x) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const cv::Vec3d delta{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
                const double length = vectorLength(delta);
                const cv::Vec3d direction = delta / length;
                if (direction.dot(chord) > kEpsilon)
                    moves.push_back({{x, y, z}, direction, length});
            }
        }
    }
    return moves;
}

FiberletPathCost dataCost(
    const SearchNode& node, const cv::Vec3d& direction, double edgeLength, const FiberletPathConfig& config, std::optional<double> quantizationFloor = std::nullopt)
{
    FiberletPathCost cost;
    if (!node.prediction.valid || !finiteVector(node.prediction.direction) || !std::isfinite(node.prediction.presence)) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel * edgeLength;
        return cost;
    }
    const cv::Vec3d predicted = normalized(node.prediction.direction);
    if (predicted.dot(predicted) <= kEpsilon) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel * edgeLength;
        return cost;
    }
    const double angle = axisAngle(predicted, direction);
    const double excess = std::max(0.0, angle - quantizationFloor.value_or(node.quantizationFloorRadians));
    cost.direction = config.directionWeight * excess * excess * edgeLength;
    cost.presence = config.presenceWeight * (1.0 - std::clamp(node.prediction.presence, 0.0, 1.0)) * edgeLength;
    return cost;
}

FiberletPathCost smoothnessCost(
    const cv::Vec3d& previousDirection,
    double previousLength,
    const cv::Vec3d& candidateDirection,
    double candidateLength,
    const vc::lasagna::NormalSample& normal,
    const FiberletPathConfig& config)
{
    const auto local = fiberLocalSmoothnessCost(
        cv::Vec3f(previousDirection),
        cv::Vec3f(candidateDirection),
        cv::Vec3f(normal.normal),
        normal.valid,
        FiberLocalSmoothnessConfig{
            static_cast<float>(config.smoothnessWeight),
            static_cast<float>(config.smoothnessNormalWeight),
            static_cast<float>(config.smoothnessTangentWeight),
            static_cast<float>(config.smoothnessFreeAngleDegrees * kPi / 180.0)});
    const double effectiveLength = std::max(1.0, (previousLength + candidateLength) * 0.5);
    FiberletPathCost cost;
    cost.isotropicSmoothness = local.isotropic / effectiveLength;
    cost.tangentSmoothness = local.tangent / effectiveLength;
    cost.normalSmoothness = local.normal / effectiveLength;
    return cost;
}

bool betterCost(double candidate, double current)
{
    return candidate < current;
}

FiberletCandidateResult solveCandidate(
    FiberletCandidateResult candidate,
    const FiberPredictionGridInfo& grid,
    int cellSize,
    const FiberletPathConfig& config,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler)
{
    candidate.searched = true;
    const cv::Vec3d chordVector = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
    const double chordLength = vectorLength(chordVector);
    if (!(chordLength > kEpsilon)) {
        candidate.reason = "zero_length";
        return candidate;
    }
    const cv::Vec3d chord = chordVector / chordLength;
    const double maximumAngle = config.maximumEndpointAngleDegrees * kPi / 180.0;
    const auto sources = endpointAttachments(candidate.startPositionPredictionXYZ, candidate.startAxisXYZ, chord, true, maximumAngle, grid);
    if (sources.empty()) {
        candidate.reason = "no_source_attachment";
        return candidate;
    }
    const auto targets = endpointAttachments(candidate.targetPositionPredictionXYZ, candidate.targetAxisXYZ, chord, false, maximumAngle, grid);
    if (targets.empty()) {
        candidate.reason = "no_target_attachment";
        return candidate;
    }
    const auto moves = forwardMoves(chord);
    if (moves.empty()) {
        candidate.reason = "no_forward_moves";
        return candidate;
    }

    const double corridorRadius = config.corridorRadiusPredictionVoxels > 0.0 ? config.corridorRadiusPredictionVoxels : static_cast<double>(cellSize);
    const auto reference =
        hermitePolyline(candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ, candidate.startAxisXYZ, candidate.targetAxisXYZ);
    cv::Vec3d minimum = reference.front();
    cv::Vec3d maximum = reference.front();
    for (const auto& point : reference) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    Voxel begin{};
    Voxel end{};
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    for (int axis = 0; axis < 3; ++axis) {
        begin[axis] = std::max<int64_t>(0, static_cast<int64_t>(std::floor(minimum[axis] - corridorRadius)));
        end[axis] = std::min<int64_t>(static_cast<int64_t>(shapeXYZ[axis]) - 1, static_cast<int64_t>(std::ceil(maximum[axis] + corridorRadius)));
    }
    std::set<Voxel> voxelSet;
    const double radiusSquared = corridorRadius * corridorRadius;
    for (int64_t z = begin[2]; z <= end[2]; ++z) {
        for (int64_t y = begin[1]; y <= end[1]; ++y) {
            for (int64_t x = begin[0]; x <= end[0]; ++x) {
                const Voxel voxel{x, y, z};
                if (insideCorridor(voxelPoint(voxel), reference, radiusSquared))
                    voxelSet.insert(voxel);
            }
        }
    }
    for (const auto& attachment : sources)
        voxelSet.insert(attachment.voxel);
    for (const auto& attachment : targets)
        voxelSet.insert(attachment.voxel);
    if (voxelSet.empty()) {
        candidate.reason = "empty_corridor";
        return candidate;
    }

    std::vector<SearchNode> nodes;
    nodes.reserve(voxelSet.size());
    for (const auto& voxel : voxelSet) {
        nodes.push_back({voxel, (voxelPoint(voxel) - candidate.startPositionPredictionXYZ).dot(chord)});
    }
    std::sort(nodes.begin(), nodes.end(), [](const SearchNode& left, const SearchNode& right) {
        if (left.progress != right.progress)
            return left.progress < right.progress;
        return left.voxel < right.voxel;
    });
    std::unordered_map<Voxel, size_t, VoxelHash> nodeIndex;
    nodeIndex.reserve(nodes.size() * 2);
    std::vector<std::array<size_t, 3>> indices;
    std::vector<cv::Vec3d> normalPoints;
    indices.reserve(nodes.size());
    normalPoints.reserve(nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        nodeIndex.emplace(nodes[index].voxel, index);
        indices.push_back(storedIndex(nodes[index].voxel));
        normalPoints.push_back(voxelPoint(nodes[index].voxel));
    }
    std::vector<FiberStoredPredictionSample> predictions;
    predictionSampler(indices, config.parallelThreads, predictions);
    if (predictions.size() != nodes.size())
        throw std::runtime_error("fiberlet prediction sampler returned the wrong number of samples");
    std::vector<vc::lasagna::NormalSampleWithDerivative> normals;
    (void)normalSampler.sampleNormalBatch(normalPoints, false, normals);
    if (normals.size() != nodes.size())
        throw std::runtime_error("fiberlet normal sampler returned the wrong number of samples");
    for (size_t index = 0; index < nodes.size(); ++index) {
        nodes[index].prediction = predictions[index];
        nodes[index].normal = normals[index].sample;
        if (!predictions[index].valid || !finiteVector(predictions[index].direction))
            continue;
        double floor = std::numeric_limits<double>::infinity();
        for (const auto& move : moves)
            floor = std::min(floor, axisAngle(predictions[index].direction, move.direction));
        nodes[index].quantizationFloorRadians = std::isfinite(floor) ? floor : 0.0;
    }

    const size_t stateCount = moves.size() + 1;
    const size_t sourceState = moves.size();
    std::vector<DpState> states(nodes.size() * stateCount);
    std::vector<cv::Vec3d> sourceDirections(nodes.size(), {0.0, 0.0, 0.0});
    std::vector<double> sourceLengths(nodes.size(), 0.0);
    double sourceQuantizationFloor = std::numeric_limits<double>::infinity();
    for (const auto& attachment : sources) {
        if (!(attachment.length > kEpsilon))
            continue;
        const auto found = nodeIndex.find(attachment.voxel);
        if (found == nodeIndex.end())
            continue;
        const auto& prediction = nodes[found->second].prediction;
        if (!prediction.valid || !finiteVector(prediction.direction))
            continue;
        sourceQuantizationFloor = std::min(sourceQuantizationFloor, axisAngle(prediction.direction, attachment.direction));
    }
    if (!std::isfinite(sourceQuantizationFloor))
        sourceQuantizationFloor = 0.0;
    for (const auto& attachment : sources) {
        const auto found = nodeIndex.find(attachment.voxel);
        if (found == nodeIndex.end())
            continue;
        const size_t node = found->second;
        FiberletPathCost initial;
        if (attachment.length > kEpsilon) {
            initial += dataCost(nodes[node], attachment.direction, attachment.length, config, sourceQuantizationFloor);
            initial += smoothnessCost(candidate.startAxisXYZ, attachment.length, attachment.direction, attachment.length, nodes[node].normal, config);
        }
        auto& state = states[node * stateCount + sourceState];
        if (!state.reached || betterCost(initial.total(), state.cost.total())) {
            state.reached = true;
            state.cost = initial;
            state.previous = {-1, -1};
            sourceDirections[node] = attachment.direction;
            sourceLengths[node] = attachment.length;
        }
    }

    for (size_t node = 0; node < nodes.size(); ++node) {
        for (size_t previousState = 0; previousState < stateCount; ++previousState) {
            const auto& currentState = states[node * stateCount + previousState];
            if (!currentState.reached)
                continue;
            const bool sourceAttachment = previousState == sourceState;
            const cv::Vec3d previousDirection = sourceAttachment ? sourceDirections[node] : moves[previousState].direction;
            const double previousLength = sourceAttachment ? sourceLengths[node] : moves[previousState].length;
            for (size_t moveIndex = 0; moveIndex < moves.size(); ++moveIndex) {
                const auto& move = moves[moveIndex];
                if (sourceAttachment && previousLength <= kEpsilon && directedAngle(candidate.startAxisXYZ, move.direction) > maximumAngle + kEpsilon) {
                    continue;
                }
                const Voxel nextVoxel{nodes[node].voxel[0] + move.delta[0], nodes[node].voxel[1] + move.delta[1], nodes[node].voxel[2] + move.delta[2]};
                const auto found = nodeIndex.find(nextVoxel);
                if (found == nodeIndex.end())
                    continue;
                const size_t next = found->second;
                if (!(nodes[next].progress > nodes[node].progress + kEpsilon))
                    continue;
                FiberletPathCost nextCost = currentState.cost;
                nextCost += dataCost(nodes[next], move.direction, move.length, config);
                nextCost += smoothnessCost(previousDirection, previousLength, move.direction, move.length, nodes[next].normal, config);
                auto& destination = states[next * stateCount + moveIndex];
                if (!destination.reached || betterCost(nextCost.total(), destination.cost.total())) {
                    destination.reached = true;
                    destination.cost = nextCost;
                    destination.previous = {static_cast<int64_t>(node), static_cast<int>(previousState)};
                }
            }
        }
    }

    bool foundPath = false;
    size_t bestNode = 0;
    size_t bestState = 0;
    FiberletPathCost bestCost;
    for (const auto& attachment : targets) {
        const auto found = nodeIndex.find(attachment.voxel);
        if (found == nodeIndex.end())
            continue;
        const size_t node = found->second;
        for (size_t stateIndex = 0; stateIndex < stateCount; ++stateIndex) {
            const auto& state = states[node * stateCount + stateIndex];
            if (!state.reached)
                continue;
            const bool sourceAttachment = stateIndex == sourceState;
            const cv::Vec3d incoming = sourceAttachment ? sourceDirections[node] : moves[stateIndex].direction;
            const double incomingLength = sourceAttachment ? sourceLengths[node] : moves[stateIndex].length;
            if (attachment.length <= kEpsilon && directedAngle(incoming, candidate.targetAxisXYZ) > maximumAngle + kEpsilon) {
                continue;
            }
            FiberletPathCost finalized = state.cost;
            const cv::Vec3d finalDirection = attachment.length > kEpsilon ? attachment.direction : candidate.targetAxisXYZ;
            finalized += smoothnessCost(incoming, incomingLength, finalDirection, attachment.length, nodes[node].normal, config);
            if (!foundPath || betterCost(finalized.total(), bestCost.total())) {
                foundPath = true;
                bestNode = node;
                bestState = stateIndex;
                bestCost = finalized;
            }
        }
    }
    if (!foundPath) {
        candidate.reason = "no_path";
        return candidate;
    }

    std::vector<cv::Vec3d> reversed;
    size_t node = bestNode;
    size_t state = bestState;
    while (true) {
        reversed.push_back(voxelPoint(nodes[node].voxel));
        const auto previous = states[node * stateCount + state].previous;
        if (previous.node < 0)
            break;
        node = static_cast<size_t>(previous.node);
        state = static_cast<size_t>(previous.state);
    }
    std::reverse(reversed.begin(), reversed.end());
    candidate.pointsPredictionXYZ.push_back(candidate.startPositionPredictionXYZ);
    for (const auto& point : reversed) {
        if (vectorLength(point - candidate.pointsPredictionXYZ.back()) > kEpsilon)
            candidate.pointsPredictionXYZ.push_back(point);
    }
    if (vectorLength(candidate.targetPositionPredictionXYZ - candidate.pointsPredictionXYZ.back()) > kEpsilon) {
        candidate.pointsPredictionXYZ.push_back(candidate.targetPositionPredictionXYZ);
    }
    if (!std::isfinite(bestCost.total()))
        throw std::runtime_error("fiberlet DP produced a non-finite path score");
    candidate.cost = bestCost;
    candidate.scoreValid = true;
    candidate.success = true;
    candidate.reason = "success";
    return candidate;
}

nlohmann::json anchorIdJson(const FiberletAnchorId& id)
{
    return {{"cell_zyx", id.cellZYX}, {"component", id.componentIndex}};
}

nlohmann::json pointJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

}  // namespace

double FiberletPathCost::total() const noexcept
{
    return invalidPrediction + presence + direction + isotropicSmoothness + tangentSmoothness + normalSmoothness;
}

FiberletPathCost& FiberletPathCost::operator+=(const FiberletPathCost& other) noexcept
{
    invalidPrediction += other.invalidPrediction;
    presence += other.presence;
    direction += other.direction;
    isotropicSmoothness += other.isotropicSmoothness;
    tangentSmoothness += other.tangentSmoothness;
    normalSmoothness += other.normalSmoothness;
    return *this;
}

FiberletPathStatistics fiberletPathStatistics(const FiberletPathReport& report)
{
    FiberletPathStatistics statistics;
    statistics.anchors = report.diagnostics.occupiedAnchors;
    statistics.candidates = report.candidates.size();
    double allSum = 0.0;
    double acceptedSum = 0.0;
    for (const auto& candidate : report.candidates) {
        if (candidate.searched)
            ++statistics.dpSearched;
        else
            ++statistics.preDpRejected;
        if (candidate.success && !candidate.scoreValid)
            throw std::logic_error("accepted fiberlet has no score");
        if (!candidate.scoreValid) {
            ++statistics.unscored;
            if (candidate.searched)
                ++statistics.searchedUnscored;
            continue;
        }
        const double score = candidate.cost.total();
        if (!std::isfinite(score))
            throw std::runtime_error("fiberlet statistics encountered a non-finite path score");
        ++statistics.scored;
        ++statistics.allScores.count;
        allSum += score;
        statistics.allScores.minimum = statistics.allScores.minimum.has_value() ? std::min(*statistics.allScores.minimum, score) : score;
        statistics.allScores.maximum = statistics.allScores.maximum.has_value() ? std::max(*statistics.allScores.maximum, score) : score;
        if (!candidate.success)
            continue;
        ++statistics.accepted;
        ++statistics.acceptedScores.count;
        acceptedSum += score;
        statistics.acceptedScores.minimum = statistics.acceptedScores.minimum.has_value() ? std::min(*statistics.acceptedScores.minimum, score) : score;
        statistics.acceptedScores.maximum = statistics.acceptedScores.maximum.has_value() ? std::max(*statistics.acceptedScores.maximum, score) : score;
    }
    if (statistics.allScores.count > 0)
        statistics.allScores.mean = allSum / static_cast<double>(statistics.allScores.count);
    if (statistics.acceptedScores.count > 0)
        statistics.acceptedScores.mean = acceptedSum / static_cast<double>(statistics.acceptedScores.count);
    return statistics;
}

void validateFiberletPathConfig(const FiberletPathConfig& config)
{
    const auto finiteNonnegative = [](double value) { return std::isfinite(value) && value >= 0.0; };
    if (config.cellRadius < 1 || config.cellRadius > 64)
        throw std::invalid_argument("fiberlet cell radius must be in [1, 64]");
    if (!(config.shellHalfWidthCells > 0.0) || !std::isfinite(config.shellHalfWidthCells)) {
        throw std::invalid_argument("fiberlet shell half width must be positive and finite");
    }
    if (!finiteNonnegative(config.maximumEndpointAngleDegrees) || config.maximumEndpointAngleDegrees > 90.0) {
        throw std::invalid_argument("fiberlet endpoint angle must be in [0, 90]");
    }
    if (!finiteNonnegative(config.corridorRadiusPredictionVoxels))
        throw std::invalid_argument("fiberlet corridor radius must be non-negative");
    if (!finiteNonnegative(config.presenceWeight) || !finiteNonnegative(config.directionWeight) ||
        !finiteNonnegative(config.invalidPredictionCostPerVoxel) || !finiteNonnegative(config.smoothnessWeight) ||
        !finiteNonnegative(config.smoothnessNormalWeight) || !finiteNonnegative(config.smoothnessTangentWeight) ||
        !finiteNonnegative(config.smoothnessFreeAngleDegrees)) {
        throw std::invalid_argument("fiberlet objective weights and angles must be finite and non-negative");
    }
    if (config.parallelThreads < 1)
        throw std::invalid_argument("fiberlet thread count must be positive");
}

LoadedFiberAnchorArtifact loadFiberAnchorArtifact(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open fiber anchor artifact: " + path.string());
    nlohmann::json root;
    try {
        input >> root;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("cannot parse fiber anchor artifact: " + std::string(error.what()));
    }
    if (!root.is_object() || root.value("format", "") != "vc_fiberlet_anchors" || root.value("version", 0) != 1) {
        throw std::runtime_error("fiber anchor artifact must be vc_fiberlet_anchors version 1");
    }
    LoadedFiberAnchorArtifact loaded;
    const auto& source = root.at("source");
    loaded.artifact.sourceLocator = source.at("manifest").get<std::string>();
    loaded.artifact.manifestContentHash = source.at("manifest_content_hash").get<std::string>();
    if (loaded.artifact.sourceLocator.empty() || loaded.artifact.manifestContentHash.empty()) {
        throw std::runtime_error("fiber anchor artifact source identity must not be empty");
    }
    const auto& coordinates = root.at("coordinates");
    if (coordinates.at("position_order") != "XYZ" || coordinates.at("cell_index_order") != "ZYX" ||
        coordinates.at("position_space") != "base_volume") {
        throw std::runtime_error("fiber anchor artifact coordinate contract is unsupported");
    }
    loaded.report.grid.shapeZYX = jsonSize3(coordinates.at("prediction_shape_zyx"), "prediction_shape_zyx");
    loaded.report.grid.predictionToBaseScale = finiteNumber(coordinates.at("prediction_to_base_scale"), "prediction_to_base_scale");
    if (!(loaded.report.grid.predictionToBaseScale > 0.0))
        throw std::runtime_error("fiber anchor prediction-to-base scale must be positive");
    if (coordinates.contains("base_voxel_size_um")) {
        loaded.artifact.baseVoxelSizeUm = finiteNumber(coordinates.at("base_voxel_size_um"), "base_voxel_size_um");
        if (!(*loaded.artifact.baseVoxelSizeUm > 0.0))
            throw std::runtime_error("fiber anchor base voxel size must be positive");
    }
    const auto& selection = root.at("selection");
    const cv::Vec3d cropOriginBase = jsonVec3(selection.at("prediction_interval_origin_base_xyz"), "prediction_interval_origin_base_xyz");
    const cv::Vec3d cropSizeBase = jsonVec3(selection.at("prediction_interval_size_base_xyz"), "prediction_interval_size_base_xyz");
    for (size_t axis = 0; axis < 3; ++axis) {
        const auto predictionIndex = [&](double baseValue, const char* name) {
            if (baseValue < 0.0)
                throw std::runtime_error(std::string(name) + " must be non-negative");
            const double scaled = baseValue / loaded.report.grid.predictionToBaseScale;
            const double rounded = std::round(scaled);
            const double tolerance = 1.0e-9 * std::max(1.0, std::abs(scaled));
            if (std::abs(scaled - rounded) > tolerance || rounded > static_cast<double>(std::numeric_limits<size_t>::max()))
                throw std::runtime_error(std::string(name) + " must align with the prediction grid");
            return static_cast<size_t>(rounded);
        };
        loaded.report.selectedCrop.originXYZ[axis] =
            predictionIndex(cropOriginBase[static_cast<int>(axis)], "prediction_interval_origin_base_xyz");
        loaded.report.selectedCrop.sizeXYZ[axis] =
            predictionIndex(cropSizeBase[static_cast<int>(axis)], "prediction_interval_size_base_xyz");
    }
    loaded.report.selectedCellBeginZYX = jsonSize3(selection.at("cell_begin_zyx"), "cell_begin_zyx");
    loaded.report.selectedCellEndZYX = jsonSize3(selection.at("cell_end_zyx"), "cell_end_zyx");
    const auto& parameters = root.at("parameters");
    auto& config = loaded.report.config;
    config.cellSizePredictionVoxels = parameters.at("cell_size_prediction_voxels").get<int>();
    config.gaussianSigmaPredictionVoxels =
        finiteNumber(parameters.at("gaussian_sigma_prediction_voxels"), "gaussian_sigma_prediction_voxels");
    config.observationPresenceFloor = finiteNumber(parameters.at("observation_presence_floor"), "observation_presence_floor");
    config.minimumAlignedSupport = finiteNumber(parameters.at("minimum_aligned_support"), "minimum_aligned_support");
    config.maximumSeedCount = parameters.at("maximum_seed_count").get<size_t>();
    config.maximumIterations = parameters.at("maximum_iterations").get<int>();
    config.convergenceTolerance = finiteNumber(parameters.at("convergence_tolerance"), "convergence_tolerance");
    config.parallelThreads = 1;
    validateFiberAnchorConfig(config);
    const auto& diagnostics = root.at("diagnostics");
    loaded.report.diagnostics.totalCells = diagnostics.at("total_cells").get<size_t>();
    loaded.report.diagnostics.zeroAnchorCells = diagnostics.at("zero_anchor_cells").get<size_t>();
    loaded.report.diagnostics.oneAnchorCells = diagnostics.at("one_anchor_cells").get<size_t>();
    loaded.report.diagnostics.twoAnchorCells = diagnostics.at("two_anchor_cells").get<size_t>();
    loaded.report.diagnostics.emptyComponents = diagnostics.at("empty_components").get<size_t>();
    loaded.report.diagnostics.degenerateComponents = diagnostics.at("degenerate_components").get<size_t>();
    loaded.report.diagnostics.belowSupportComponents = diagnostics.at("below_support_components").get<size_t>();

    std::optional<std::array<size_t, 3>> previousCell;
    const auto& cells = root.at("cells");
    if (!cells.is_array())
        throw std::runtime_error("fiber anchor cells must be an array");
    for (const auto& cellJson : cells) {
        FiberCellAnchorResult cell;
        cell.cellZYX = jsonSize3(cellJson.at("cell_zyx"), "cell_zyx");
        if (previousCell.has_value() && !(previousCell.value() < cell.cellZYX))
            throw std::runtime_error("fiber anchor cells must be strictly ordered");
        previousCell = cell.cellZYX;
        cell.objective = finiteNumber(cellJson.at("objective"), "cell objective");
        const auto& components = cellJson.at("components");
        if (!components.is_array() || components.size() != 2)
            throw std::runtime_error("fiber anchor cell must contain exactly two components");
        for (size_t index = 0; index < 2; ++index) {
            auto& component = cell.components[index];
            const auto& componentJson = components[index];
            component.retained = componentJson.at("retained").get<bool>();
            component.assignedObservationCount = componentJson.at("assigned_observations").get<size_t>();
            component.anchor.cellZYX = cell.cellZYX;
            if (!component.retained) {
                component.rejectionReason = componentJson.at("reason").get<std::string>();
                if (component.rejectionReason.empty())
                    throw std::runtime_error("rejected fiber anchor component needs a reason");
                continue;
            }
            component.anchor.positionPredictionXYZ =
                jsonVec3(componentJson.at("position_base_xyz"), "position_base_xyz") / loaded.report.grid.predictionToBaseScale;
            component.anchor.axisXYZ = jsonVec3(componentJson.at("axis_xyz"), "axis_xyz");
            const double axisLength = vectorLength(component.anchor.axisXYZ);
            if (std::abs(axisLength - 1.0) > 1.0e-6)
                throw std::runtime_error("fiber anchor axis must be unit length");
            component.anchor.alignedSupport = finiteNumber(componentJson.at("aligned_support"), "aligned_support");
            component.anchor.directionalCoherence = finiteNumber(componentJson.at("directional_coherence"), "directional_coherence");
            for (int axis = 0; axis < 3; ++axis) {
                const size_t cellAxis = static_cast<size_t>(2 - axis);
                const double lower = static_cast<double>(cell.cellZYX[cellAxis] * static_cast<size_t>(config.cellSizePredictionVoxels));
                const double upper = static_cast<double>(
                    std::min((cell.cellZYX[cellAxis] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[cellAxis]));
                const double position = component.anchor.positionPredictionXYZ[axis];
                if (position < lower - kEpsilon || position >= upper)
                    throw std::runtime_error("fiber anchor position is outside its owned cell");
            }
            ++cell.retainedAnchorCount;
        }
        if (cell.retainedAnchorCount == 0)
            throw std::runtime_error("fiber anchor artifact must not store empty cells");
        loaded.report.nonEmptyCells.push_back(std::move(cell));
    }
    return loaded;
}

std::vector<std::array<int, 3>> fiberletCellShellOffsets(int radius, double halfWidth)
{
    if (radius < 1 || !(halfWidth > 0.0) || !std::isfinite(halfWidth))
        throw std::invalid_argument("fiberlet cell shell requires positive radius and half width");
    const int limit = static_cast<int>(std::ceil(radius + halfWidth));
    const double lower = std::max(0.0, static_cast<double>(radius) - halfWidth);
    const double upper = static_cast<double>(radius) + halfWidth;
    std::vector<std::array<int, 3>> offsets;
    for (int z = -limit; z <= limit; ++z) {
        for (int y = -limit; y <= limit; ++y) {
            for (int x = -limit; x <= limit; ++x) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const double length = std::sqrt(static_cast<double>(x * x + y * y + z * z));
                if (length >= lower && length < upper)
                    offsets.push_back({z, y, x});
            }
        }
    }
    return offsets;
}

FiberletPathReport traceFiberletPaths(
    const LoadedFiberAnchorArtifact& anchors,
    const FiberPredictionGridInfo& grid,
    const FiberletPathConfig& inputConfig,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler)
{
    validateFiberletPathConfig(inputConfig);
    if (!predictionSampler)
        throw std::invalid_argument("fiberlet path tracing requires a prediction sampler");
    if (grid.shapeZYX != anchors.report.grid.shapeZYX || std::abs(grid.predictionToBaseScale - anchors.report.grid.predictionToBaseScale) > 1.0e-12) {
        throw std::invalid_argument("fiberlet prediction grid does not match anchor artifact");
    }
    FiberletPathReport report;
    report.grid = grid;
    report.anchorCellSizePredictionVoxels = anchors.report.config.cellSizePredictionVoxels;
    report.config = inputConfig;
    if (!(report.config.corridorRadiusPredictionVoxels > 0.0)) {
        report.config.corridorRadiusPredictionVoxels = static_cast<double>(report.anchorCellSizePredictionVoxels);
    }
    const auto startTime = Clock::now();
    const auto flat = flattenAnchors(anchors.report);
    report.diagnostics.occupiedAnchors = flat.size();
    const auto offsets = fiberletCellShellOffsets(report.config.cellRadius, report.config.shellHalfWidthCells);
    report.diagnostics.shellOffsets = offsets.size();
    std::map<std::array<size_t, 3>, std::vector<const FlatAnchor*>> byCell;
    for (const auto& anchor : flat)
        byCell[anchor.id.cellZYX].push_back(&anchor);
    const size_t cellSize = static_cast<size_t>(report.anchorCellSizePredictionVoxels);
    const std::array<size_t, 3>
        cellShape{(grid.shapeZYX[0] + cellSize - 1) / cellSize, (grid.shapeZYX[1] + cellSize - 1) / cellSize, (grid.shapeZYX[2] + cellSize - 1) / cellSize};
    const double minimumAxisDot = std::cos(report.config.maximumEndpointAngleDegrees * kPi / 180.0);
    for (const auto& source : flat) {
        for (const auto& offset : offsets) {
            std::array<size_t, 3> targetCell{};
            bool inside = true;
            for (size_t axis = 0; axis < 3; ++axis) {
                const int64_t value = static_cast<int64_t>(source.id.cellZYX[axis]) + static_cast<int64_t>(offset[axis]);
                if (value < 0 || static_cast<uint64_t>(value) >= cellShape[axis]) {
                    inside = false;
                    break;
                }
                targetCell[axis] = static_cast<size_t>(value);
            }
            if (!inside) {
                ++report.diagnostics.shellTargetsOutOfGrid;
                continue;
            }
            const auto targetAnchors = byCell.find(targetCell);
            if (targetAnchors == byCell.end())
                continue;
            for (const FlatAnchor* target : targetAnchors->second) {
                if (!(source.id < target->id))
                    continue;
                FiberletCandidateResult candidate;
                candidate.start = source.id;
                candidate.target = target->id;
                candidate.startPositionPredictionXYZ = source.anchor.positionPredictionXYZ;
                candidate.targetPositionPredictionXYZ = target->anchor.positionPredictionXYZ;
                const cv::Vec3d chordVector = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
                const double distance = vectorLength(chordVector);
                ++report.diagnostics.generatedPairs;
                if (!(distance > kEpsilon)) {
                    candidate.reason = "zero_length";
                    ++report.diagnostics.zeroLengthPairs;
                    report.candidates.push_back(std::move(candidate));
                    continue;
                }
                const cv::Vec3d chord = chordVector / distance;
                candidate.startAxisXYZ = normalized(source.anchor.axisXYZ);
                candidate.targetAxisXYZ = normalized(target->anchor.axisXYZ);
                if (candidate.startAxisXYZ.dot(chord) < 0.0)
                    candidate.startAxisXYZ *= -1.0;
                if (candidate.targetAxisXYZ.dot(chord) < 0.0)
                    candidate.targetAxisXYZ *= -1.0;
                if (candidate.startAxisXYZ.dot(chord) + kEpsilon < minimumAxisDot || candidate.targetAxisXYZ.dot(chord) + kEpsilon < minimumAxisDot) {
                    candidate.reason = "axis_mismatch";
                    ++report.diagnostics.axisRejectedPairs;
                    report.candidates.push_back(std::move(candidate));
                    continue;
                }
                candidate = solveCandidate(std::move(candidate), grid, report.anchorCellSizePredictionVoxels, report.config, predictionSampler, normalSampler);
                ++report.diagnostics.searchedPairs;
                if (candidate.success)
                    ++report.diagnostics.successfulPaths;
                else
                    ++report.diagnostics.noPathPairs;
                report.candidates.push_back(std::move(candidate));
            }
        }
    }
    report.elapsedSeconds = std::chrono::duration<double>(Clock::now() - startTime).count();
    return report;
}

nlohmann::json fiberletPathReportJson(const FiberletPathReport& report, const FiberletArtifactInfo& artifact)
{
    if (artifact.fiberManifestLocator.empty() || artifact.fiberManifestContentHash.empty() || artifact.normalManifestLocator.empty() ||
        artifact.normalManifestContentHash.empty() || artifact.anchorArtifactLocator.empty() || artifact.anchorArtifactContentHash.empty()) {
        throw std::invalid_argument("fiberlet artifacts require complete source identities");
    }
    nlohmann::json root = {
        {"format", "vc_fiberlets"},
        {"version", 1},
        {"sources",
         {
             {"fiber_manifest", artifact.fiberManifestLocator},
             {"fiber_manifest_content_hash", artifact.fiberManifestContentHash},
             {"normal_manifest", artifact.normalManifestLocator},
             {"normal_manifest_content_hash", artifact.normalManifestContentHash},
             {"anchor_artifact", artifact.anchorArtifactLocator},
             {"anchor_artifact_content_hash", artifact.anchorArtifactContentHash},
         }},
        {"coordinates",
         {
             {"position_order", "XYZ"},
             {"cell_index_order", "ZYX"},
             {"position_space", "base_volume"},
             {"prediction_shape_zyx", report.grid.shapeZYX},
             {"prediction_to_base_scale", report.grid.predictionToBaseScale},
         }},
        {"parameters",
         {
             {"anchor_cell_size_prediction_voxels", report.anchorCellSizePredictionVoxels},
             {"cell_radius", report.config.cellRadius},
             {"shell_half_width_cells", report.config.shellHalfWidthCells},
             {"maximum_endpoint_angle_degrees", report.config.maximumEndpointAngleDegrees},
             {"corridor_radius_base_voxels", report.config.corridorRadiusPredictionVoxels * report.grid.predictionToBaseScale},
             {"presence_weight", report.config.presenceWeight},
             {"direction_weight", report.config.directionWeight},
             {"invalid_prediction_cost_per_prediction_voxel", report.config.invalidPredictionCostPerVoxel},
             {"smoothness_weight", report.config.smoothnessWeight},
             {"smoothness_normal_weight", report.config.smoothnessNormalWeight},
             {"smoothness_tangent_weight", report.config.smoothnessTangentWeight},
             {"smoothness_free_angle_degrees", report.config.smoothnessFreeAngleDegrees},
         }},
        {"diagnostics",
         {
             {"occupied_anchors", report.diagnostics.occupiedAnchors},
             {"shell_offsets", report.diagnostics.shellOffsets},
             {"shell_targets_out_of_grid", report.diagnostics.shellTargetsOutOfGrid},
             {"generated_pairs", report.diagnostics.generatedPairs},
             {"zero_length_pairs", report.diagnostics.zeroLengthPairs},
             {"axis_rejected_pairs", report.diagnostics.axisRejectedPairs},
             {"searched_pairs", report.diagnostics.searchedPairs},
             {"successful_paths", report.diagnostics.successfulPaths},
             {"no_path_pairs", report.diagnostics.noPathPairs},
         }},
        {"candidates", nlohmann::json::array()},
    };
    if (artifact.baseVoxelSizeUm.has_value())
        root["coordinates"]["base_voxel_size_um"] = *artifact.baseVoxelSizeUm;
    for (const auto& candidate : report.candidates) {
        nlohmann::json item = {
            {"start", anchorIdJson(candidate.start)},
            {"target", anchorIdJson(candidate.target)},
            {"start_position_base_xyz", pointJson(candidate.startPositionPredictionXYZ * report.grid.predictionToBaseScale)},
            {"target_position_base_xyz", pointJson(candidate.targetPositionPredictionXYZ * report.grid.predictionToBaseScale)},
            {"start_axis_xyz", pointJson(candidate.startAxisXYZ)},
            {"target_axis_xyz", pointJson(candidate.targetAxisXYZ)},
            {"searched", candidate.searched},
            {"score_valid", candidate.scoreValid},
            {"success", candidate.success},
            {"reason", candidate.reason},
        };
        if (candidate.scoreValid) {
            item["cost"] = {
                {"total", candidate.cost.total()},
                {"invalid_prediction", candidate.cost.invalidPrediction},
                {"presence", candidate.cost.presence},
                {"direction", candidate.cost.direction},
                {"isotropic_smoothness", candidate.cost.isotropicSmoothness},
                {"tangent_smoothness", candidate.cost.tangentSmoothness},
                {"normal_smoothness", candidate.cost.normalSmoothness},
            };
        }
        if (candidate.success) {
            item["points_base_xyz"] = nlohmann::json::array();
            for (const auto& point : candidate.pointsPredictionXYZ)
                item["points_base_xyz"].push_back(pointJson(point * report.grid.predictionToBaseScale));
        }
        root["candidates"].push_back(std::move(item));
    }
    return root;
}

std::string fiberletPathReportObj(const FiberletPathReport& report)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "# vc_fiberlets version 1\n";
    size_t vertex = 1;
    for (const auto& candidate : report.candidates) {
        if (!candidate.success || candidate.pointsPredictionXYZ.size() < 2)
            continue;
        output << "g fiberlet_" << candidate.start.cellZYX[0] << '_' << candidate.start.cellZYX[1] << '_' << candidate.start.cellZYX[2]
               << '_' << candidate.start.componentIndex << "__" << candidate.target.cellZYX[0] << '_' << candidate.target.cellZYX[1] << '_'
               << candidate.target.cellZYX[2] << '_' << candidate.target.componentIndex << '\n';
        for (const auto& point : candidate.pointsPredictionXYZ) {
            const cv::Vec3d base = point * report.grid.predictionToBaseScale;
            output << "v " << base[0] << ' ' << base[1] << ' ' << base[2] << '\n';
        }
        for (size_t index = 1; index < candidate.pointsPredictionXYZ.size(); ++index)
            output << "l " << vertex + index - 1 << ' ' << vertex + index << '\n';
        vertex += candidate.pointsPredictionXYZ.size();
    }
    return output.str();
}

void writeFiberletPathArtifacts(const std::filesystem::path& outputDirectory, const FiberletPathReport& report, const FiberletArtifactInfo& artifact)
{
    if (outputDirectory.empty())
        throw std::invalid_argument("fiberlet output directory must not be empty");
    std::filesystem::create_directories(outputDirectory);
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.json", fiberletPathReportJson(report, artifact).dump(2) + "\n");
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.obj", fiberletPathReportObj(report));
}

}  // namespace vc::fiber_tracer
