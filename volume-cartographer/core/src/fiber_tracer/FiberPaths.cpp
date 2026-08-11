#include "vc/fiber_tracer/FiberPaths.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/TexturedMesh.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>

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
};

struct SearchCorridor {
    std::vector<cv::Vec3d> reference;
    Voxel begin{0, 0, 0};
    Voxel end{-1, -1, -1};
    double radiusSquared = 0.0;
};

struct ScoringVoxel {
    FiberStoredPredictionSample prediction;
    cv::Vec3d normal{0.0, 0.0, 0.0};
    bool normalValid = false;
};

struct DenseScoringVolume {
    Voxel begin{0, 0, 0};
    std::array<size_t, 3> sizeXYZ{0, 0, 0};
    std::vector<ScoringVoxel> voxels;

    [[nodiscard]] const ScoringVoxel& at(const Voxel& voxel) const
    {
        std::array<size_t, 3> local{};
        for (size_t axis = 0; axis < 3; ++axis) {
            if (voxel[axis] < begin[axis])
                throw std::out_of_range("fiberlet scoring voxel is below the preloaded region");
            const uint64_t delta = static_cast<uint64_t>(voxel[axis] - begin[axis]);
            if (delta >= sizeXYZ[axis])
                throw std::out_of_range("fiberlet scoring voxel is above the preloaded region");
            local[axis] = static_cast<size_t>(delta);
        }
        const size_t index = (local[2] * sizeXYZ[1] + local[1]) * sizeXYZ[0] + local[0];
        return voxels[index];
    }
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

SearchCorridor makeSearchCorridor(const FiberletCandidateResult& candidate, const FiberPredictionGridInfo& grid, int cellSize, const FiberletPathConfig& config)
{
    SearchCorridor corridor;
    const double radius = config.corridorRadiusPredictionVoxels > 0.0 ? config.corridorRadiusPredictionVoxels : static_cast<double>(cellSize);
    corridor.radiusSquared = radius * radius;
    corridor.reference =
        hermitePolyline(candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ, candidate.startAxisXYZ, candidate.targetAxisXYZ);
    cv::Vec3d minimum = corridor.reference.front();
    cv::Vec3d maximum = corridor.reference.front();
    for (const auto& point : corridor.reference) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    for (const cv::Vec3d* endpoint : {&candidate.startPositionPredictionXYZ, &candidate.targetPositionPredictionXYZ}) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], std::floor((*endpoint)[axis]) - 1.0);
            maximum[axis] = std::max(maximum[axis], std::ceil((*endpoint)[axis]) + 1.0);
        }
    }
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    for (int axis = 0; axis < 3; ++axis) {
        if (shapeXYZ[axis] == 0 || shapeXYZ[axis] > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("fiberlet prediction shape exceeds signed search indexing");
        }
        const double rawBegin = std::floor(minimum[axis] - radius);
        const double rawEnd = std::ceil(maximum[axis] + radius);
        const int64_t gridEnd = static_cast<int64_t>(shapeXYZ[axis]) - 1;
        if (!std::isfinite(rawBegin) || !std::isfinite(rawEnd))
            throw std::overflow_error("fiberlet corridor bound is not finite");
        corridor.begin[axis] = rawBegin <= 0.0 ? 0 : rawBegin >= static_cast<double>(gridEnd) ? gridEnd : static_cast<int64_t>(rawBegin);
        corridor.end[axis] = rawEnd >= static_cast<double>(gridEnd) ? gridEnd : rawEnd <= 0.0 ? 0 : static_cast<int64_t>(rawEnd);
    }
    return corridor;
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

size_t checkedProduct(size_t left, size_t right, const char* description)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
        throw std::overflow_error(std::string(description) + " overflows size_t");
    return left * right;
}

size_t checkedSum(size_t left, size_t right, const char* description)
{
    if (left > std::numeric_limits<size_t>::max() - right)
        throw std::overflow_error(std::string(description) + " overflows size_t");
    return left + right;
}

DenseScoringVolume preloadScoringVolume(
    const std::vector<FiberletCandidateResult>& candidates,
    const std::vector<size_t>& searchCandidateIndices,
    const FiberPredictionGridInfo& grid,
    int cellSize,
    const FiberletPathConfig& config,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler,
    size_t& estimatedPreloadBytes)
{
    DenseScoringVolume volume;
    estimatedPreloadBytes = 0;
    if (searchCandidateIndices.empty())
        return volume;

    bool initialized = false;
    Voxel unionBegin{};
    Voxel unionEnd{};
    for (const size_t candidateIndex : searchCandidateIndices) {
        const SearchCorridor corridor = makeSearchCorridor(candidates.at(candidateIndex), grid, cellSize, config);
        if (!initialized) {
            unionBegin = corridor.begin;
            unionEnd = corridor.end;
            initialized = true;
            continue;
        }
        for (size_t axis = 0; axis < 3; ++axis) {
            unionBegin[axis] = std::min(unionBegin[axis], corridor.begin[axis]);
            unionEnd[axis] = std::max(unionEnd[axis], corridor.end[axis]);
        }
    }
    volume.begin = unionBegin;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (unionEnd[axis] < unionBegin[axis])
            throw std::logic_error("fiberlet preload produced an empty scoring bound");
        const uint64_t extent = static_cast<uint64_t>(unionEnd[axis] - unionBegin[axis]) + 1;
        if (extent > std::numeric_limits<size_t>::max())
            throw std::overflow_error("fiberlet preload extent exceeds size_t");
        volume.sizeXYZ[axis] = static_cast<size_t>(extent);
    }
    const size_t xy = checkedProduct(volume.sizeXYZ[0], volume.sizeXYZ[1], "fiberlet preload XY extent");
    const size_t count = checkedProduct(xy, volume.sizeXYZ[2], "fiberlet preload voxel count");
    if (count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("fiberlet preload exceeds sampler index range");

    std::vector<std::array<size_t, 3>> indices;
    std::vector<cv::Vec3d> normalPoints;
    std::vector<FiberStoredPredictionSample> predictions;
    std::vector<vc::lasagna::NormalSampleWithDerivative> normals;
    if (count > volume.voxels.max_size() || count > indices.max_size() || count > normalPoints.max_size() ||
        count > predictions.max_size() || count > normals.max_size()) {
        throw std::length_error("fiberlet preload exceeds a working vector capacity");
    }
    const size_t bytesPerVoxel = checkedSum(
        checkedSum(
            checkedSum(sizeof(ScoringVoxel), sizeof(std::array<size_t, 3>), "fiberlet preload byte estimate"),
            sizeof(cv::Vec3d),
            "fiberlet preload byte estimate"),
        checkedSum(sizeof(FiberStoredPredictionSample), sizeof(vc::lasagna::NormalSampleWithDerivative), "fiberlet preload byte estimate"),
        "fiberlet preload byte estimate");
    estimatedPreloadBytes = checkedProduct(count, bytesPerVoxel, "fiberlet preload byte estimate");

    indices.reserve(count);
    normalPoints.reserve(count);
    for (int64_t z = unionBegin[2]; z <= unionEnd[2]; ++z) {
        for (int64_t y = unionBegin[1]; y <= unionEnd[1]; ++y) {
            for (int64_t x = unionBegin[0]; x <= unionEnd[0]; ++x) {
                const Voxel voxel{x, y, z};
                indices.push_back(storedIndex(voxel));
                normalPoints.push_back(voxelPoint(voxel));
            }
        }
    }
    if (indices.size() != count || normalPoints.size() != count)
        throw std::logic_error("fiberlet preload enumeration is incomplete");
    predictionSampler(indices, config.parallelThreads, predictions);
    if (predictions.size() != count)
        throw std::runtime_error("fiberlet prediction sampler returned the wrong preload sample count");
    (void)normalSampler.sampleNormalBatch(normalPoints, false, config.parallelThreads, normals);
    if (normals.size() != count)
        throw std::runtime_error("fiberlet normal sampler returned the wrong preload sample count");

    volume.voxels.resize(count);
    for (size_t index = 0; index < count; ++index) {
        volume.voxels[index].prediction = predictions[index];
        volume.voxels[index].normal = normals[index].sample.normal;
        volume.voxels[index].normalValid = normals[index].sample.valid;
    }
    return volume;
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

bool usablePrediction(const FiberStoredPredictionSample& prediction)
{
    const double normSquared = prediction.direction.dot(prediction.direction);
    return prediction.valid && finiteVector(prediction.direction) &&
           std::isfinite(prediction.presence) &&
           std::isfinite(normSquared) && normSquared > kEpsilon;
}

cv::Vec3d alignedAxis(const cv::Vec3d& axis, const cv::Vec3d& reference)
{
    cv::Vec3d aligned = normalized(axis);
    if (aligned.dot(normalized(reference)) < 0.0)
        aligned *= -1.0;
    return aligned;
}

cv::Vec3f floatVector(const cv::Vec3d& value)
{
    return {
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2]),
    };
}

FiberletPathCost alignmentCost(
    const FiberStoredPredictionSample* currentPrediction,
    const FiberStoredPredictionSample& candidatePrediction,
    const cv::Vec3d& previousStep,
    const cv::Vec3d& candidateStep,
    double edgeLength,
    const FiberletPathConfig& config)
{
    FiberletPathCost cost;
    if (!usablePrediction(candidatePrediction)) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel * edgeLength;
        return cost;
    }
    const cv::Vec3d previous = normalized(previousStep);
    const cv::Vec3d outgoing = normalized(candidateStep);
    const cv::Vec3d current = currentPrediction != nullptr &&
            usablePrediction(*currentPrediction)
        ? alignedAxis(currentPrediction->direction, previous)
        : previous;
    const cv::Vec3d candidate =
        alignedAxis(candidatePrediction.direction, outgoing);
    cost.alignment = static_cast<double>(fiberLocalAlignmentLoss(
                         static_cast<float>(candidatePrediction.presence),
                         floatVector(previous),
                         floatVector(outgoing),
                         floatVector(current),
                         floatVector(candidate))) *
        edgeLength;
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
    FiberletCandidateResult candidate, const FiberPredictionGridInfo& grid, int cellSize, const FiberletPathConfig& config, const DenseScoringVolume& scoringVolume)
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

    const SearchCorridor corridor = makeSearchCorridor(candidate, grid, cellSize, config);
    std::set<Voxel> voxelSet;
    for (int64_t z = corridor.begin[2]; z <= corridor.end[2]; ++z) {
        for (int64_t y = corridor.begin[1]; y <= corridor.end[1]; ++y) {
            for (int64_t x = corridor.begin[0]; x <= corridor.end[0]; ++x) {
                const Voxel voxel{x, y, z};
                if (insideCorridor(voxelPoint(voxel), corridor.reference, corridor.radiusSquared)) {
                    voxelSet.insert(voxel);
                }
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
    for (size_t index = 0; index < nodes.size(); ++index) {
        nodeIndex.emplace(nodes[index].voxel, index);
        const auto& scoring = scoringVolume.at(nodes[index].voxel);
        nodes[index].prediction = scoring.prediction;
        nodes[index].normal = {scoring.normal, scoring.normalValid, scoring.normalValid ? std::string{} : "invalid preloaded normal"};
    }

    const size_t stateCount = moves.size() + 1;
    const size_t sourceState = moves.size();
    std::vector<DpState> states(nodes.size() * stateCount);
    std::vector<cv::Vec3d> sourceDirections(nodes.size(), {0.0, 0.0, 0.0});
    std::vector<double> sourceLengths(nodes.size(), 0.0);
    for (const auto& attachment : sources) {
        const auto found = nodeIndex.find(attachment.voxel);
        if (found == nodeIndex.end())
            continue;
        const size_t node = found->second;
        FiberletPathCost initial;
        if (attachment.length > kEpsilon) {
            initial += alignmentCost(
                nullptr,
                nodes[node].prediction,
                candidate.startAxisXYZ,
                attachment.direction,
                attachment.length,
                config);
            initial += smoothnessCost(candidate.startAxisXYZ, attachment.length, attachment.direction, attachment.length, nodes[node].normal, config);
        }
        auto& state = states[node * stateCount + sourceState];
        if (!state.reached || betterCost(initial.total(), state.cost.total())) {
            state.reached = true;
            state.cost = initial;
            state.previous = {-1, -1};
            sourceDirections[node] = attachment.length > kEpsilon
                ? attachment.direction
                : candidate.startAxisXYZ;
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
                nextCost += alignmentCost(
                    sourceAttachment && previousLength <= kEpsilon
                        ? nullptr
                        : &nodes[node].prediction,
                    nodes[next].prediction,
                    previousDirection,
                    move.direction,
                    move.length,
                    config);
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
            if (attachment.length > kEpsilon) {
                const FiberStoredPredictionSample targetPrediction{
                    candidate.targetAxisXYZ, 1.0, true};
                finalized += alignmentCost(
                    &nodes[node].prediction,
                    targetPrediction,
                    incoming,
                    finalDirection,
                    attachment.length,
                    config);
            }
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

std::string fiberletId(const FiberletCandidateResult& candidate)
{
    std::ostringstream output;
    output << "fiberlet_" << candidate.start.cellZYX[0] << '_'
           << candidate.start.cellZYX[1] << '_' << candidate.start.cellZYX[2]
           << '_' << candidate.start.componentIndex << "__"
           << candidate.target.cellZYX[0] << '_' << candidate.target.cellZYX[1]
           << '_' << candidate.target.cellZYX[2] << '_'
           << candidate.target.componentIndex;
    return output.str();
}

double fiberletPathLength(const FiberletCandidateResult& candidate)
{
    if (candidate.pointsPredictionXYZ.size() < 2)
        throw std::runtime_error("successful fiberlet has fewer than two path points");
    double length = 0.0;
    for (size_t index = 1; index < candidate.pointsPredictionXYZ.size(); ++index) {
        if (!finiteVector(candidate.pointsPredictionXYZ[index - 1]) ||
            !finiteVector(candidate.pointsPredictionXYZ[index])) {
            throw std::runtime_error("successful fiberlet has a non-finite path point");
        }
        const double segment = vectorLength(
            candidate.pointsPredictionXYZ[index] -
            candidate.pointsPredictionXYZ[index - 1]);
        if (!std::isfinite(segment))
            throw std::runtime_error("successful fiberlet has a non-finite path segment");
        length += segment;
    }
    if (!(length > 0.0) || !std::isfinite(length))
        throw std::runtime_error("successful fiberlet has non-positive path length");
    return length;
}

}  // namespace

double FiberletPathCost::total() const noexcept
{
    return invalidPrediction + alignment + isotropicSmoothness + tangentSmoothness + normalSmoothness;
}

FiberletPathCost& FiberletPathCost::operator+=(const FiberletPathCost& other) noexcept
{
    invalidPrediction += other.invalidPrediction;
    alignment += other.alignment;
    isotropicSmoothness += other.isotropicSmoothness;
    tangentSmoothness += other.tangentSmoothness;
    normalSmoothness += other.normalSmoothness;
    return *this;
}

FiberletPathVisualReport fiberletPathVisualMetrics(const FiberletPathReport& report)
{
    if (!(report.grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(report.grid.predictionToBaseScale)) {
        throw std::runtime_error(
            "fiberlet visualization requires a positive prediction-to-base scale");
    }

    FiberletPathVisualReport visual;
    std::set<std::string> identifiers;
    for (size_t candidateIndex = 0;
         candidateIndex < report.candidates.size(); ++candidateIndex) {
        const auto& candidate = report.candidates[candidateIndex];
        if (candidate.success && !candidate.scoreValid)
            throw std::runtime_error("successful fiberlet has no valid score");
        if (!candidate.success || !candidate.scoreValid)
            continue;
        const std::string identifier = fiberletId(candidate);
        if (!identifiers.insert(identifier).second)
            throw std::runtime_error("fiberlet visualization has a duplicate path identifier");
        const std::array componentLosses{
            candidate.cost.invalidPrediction,
            candidate.cost.alignment,
            candidate.cost.isotropicSmoothness,
            candidate.cost.tangentSmoothness,
            candidate.cost.normalSmoothness,
        };
        if (std::any_of(componentLosses.begin(), componentLosses.end(),
                [](double value) {
                    return !(value >= 0.0) || !std::isfinite(value);
                })) {
            throw std::runtime_error(
                "successful fiberlet has an invalid component loss");
        }
        const double totalLoss = candidate.cost.total();
        if (!(totalLoss >= 0.0) || !std::isfinite(totalLoss))
            throw std::runtime_error("successful fiberlet has invalid total loss");
        const double pathLength = fiberletPathLength(candidate);
        const double density = totalLoss / pathLength;
        if (!(density >= 0.0) || !std::isfinite(density))
            throw std::runtime_error("successful fiberlet has invalid loss density");
        visual.paths.push_back({
            candidateIndex,
            pathLength,
            totalLoss,
            density,
            0.0,
        });
        visual.minimumLossPerPredictionVoxel =
            visual.minimumLossPerPredictionVoxel.has_value()
            ? std::min(*visual.minimumLossPerPredictionVoxel, density)
            : density;
        visual.maximumLossPerPredictionVoxel =
            visual.maximumLossPerPredictionVoxel.has_value()
            ? std::max(*visual.maximumLossPerPredictionVoxel, density)
            : density;
    }
    for (auto& path : visual.paths) {
        path.relativeQuality =
            *visual.minimumLossPerPredictionVoxel ==
                *visual.maximumLossPerPredictionVoxel
            ? 1.0
            : (*visual.maximumLossPerPredictionVoxel -
                  path.lossPerPredictionVoxel) /
                (*visual.maximumLossPerPredictionVoxel -
                  *visual.minimumLossPerPredictionVoxel);
        path.relativeQuality = std::clamp(path.relativeQuality, 0.0, 1.0);
    }
    return visual;
}

FiberletPathStatistics fiberletPathStatistics(const FiberletPathReport& report)
{
    FiberletPathStatistics statistics;
    statistics.anchors = report.diagnostics.occupiedAnchors;
    statistics.candidates = report.candidates.size();
    double allSum = 0.0;
    double acceptedSum = 0.0;
    for (size_t candidateIndex = 0;
         candidateIndex < report.candidates.size(); ++candidateIndex) {
        const auto& candidate = report.candidates[candidateIndex];
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
    const auto visual = fiberletPathVisualMetrics(report);
    double densitySum = 0.0;
    for (const auto& path : visual.paths) {
        ++statistics.acceptedLossDensities.count;
        densitySum += path.lossPerPredictionVoxel;
        statistics.acceptedLossDensities.minimum =
            statistics.acceptedLossDensities.minimum.has_value()
            ? std::min(*statistics.acceptedLossDensities.minimum,
                  path.lossPerPredictionVoxel)
            : path.lossPerPredictionVoxel;
        statistics.acceptedLossDensities.maximum =
            statistics.acceptedLossDensities.maximum.has_value()
            ? std::max(*statistics.acceptedLossDensities.maximum,
                  path.lossPerPredictionVoxel)
            : path.lossPerPredictionVoxel;
    }
    if (statistics.acceptedLossDensities.count > 0) {
        statistics.acceptedLossDensities.mean = densitySum /
            static_cast<double>(statistics.acceptedLossDensities.count);
    }
    return statistics;
}

size_t FiberPresenceSliceReport::pixelCount() const noexcept
{
    size_t count = 0;
    for (const auto& plane : planes)
        count += plane.pixels.size();
    return count;
}

FiberAnchorCrop fiberAnchorCellCoverageCrop(const LoadedFiberAnchorArtifact& anchors)
{
    const size_t cellSize = static_cast<size_t>(anchors.report.config.cellSizePredictionVoxels);
    if (cellSize == 0)
        throw std::invalid_argument("fiber presence slice cell size must be positive");
    std::array<size_t, 3> beginZYX{};
    std::array<size_t, 3> endZYX{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t shape = anchors.report.grid.shapeZYX[axis];
        const size_t cellCount = shape / cellSize + (shape % cellSize != 0 ? 1 : 0);
        const size_t cellBegin = anchors.report.selectedCellBeginZYX[axis];
        const size_t cellEnd = anchors.report.selectedCellEndZYX[axis];
        if (cellBegin >= cellEnd || cellEnd > cellCount)
            throw std::invalid_argument("fiber presence slice cell bounds are invalid");
        beginZYX[axis] = cellBegin * cellSize;
        endZYX[axis] = cellEnd == cellCount ? shape : cellEnd * cellSize;
    }
    return {
        {beginZYX[2], beginZYX[1], beginZYX[0]},
        {endZYX[2] - beginZYX[2], endZYX[1] - beginZYX[1], endZYX[0] - beginZYX[0]},
    };
}

FiberPresenceSliceReport sampleFiberPresenceSlices(
    const FiberAnchorCrop& cropPredictionXYZ, const FiberPredictionGridInfo& grid, const FiberStoredPresenceBatchSampler& presenceSampler, int parallelThreads)
{
    if (!presenceSampler)
        throw std::invalid_argument("fiber presence slices require a presence sampler");
    if (parallelThreads < 1)
        throw std::invalid_argument("fiber presence slices require a positive thread count");
    if (!(grid.predictionToBaseScale > 0.0) || !std::isfinite(grid.predictionToBaseScale))
        throw std::invalid_argument("fiber presence slices require a valid prediction-to-base scale");
    for (size_t xyz = 0; xyz < 3; ++xyz) {
        const size_t zyx = 2 - xyz;
        const size_t origin = cropPredictionXYZ.originXYZ[xyz];
        const size_t extent = cropPredictionXYZ.sizeXYZ[xyz];
        if (extent == 0 || origin > grid.shapeZYX[zyx] || extent > grid.shapeZYX[zyx] - origin)
            throw std::invalid_argument("fiber presence slice crop must be non-empty and inside the prediction grid");
    }
    const auto checkedProduct = [](size_t left, size_t right) {
        if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
            throw std::overflow_error("fiber presence slice pixel count overflows");
        return left * right;
    };
    const auto checkedAdd = [](size_t left, size_t right) {
        if (left > std::numeric_limits<size_t>::max() - right)
            throw std::overflow_error("fiber presence slice pixel count overflows");
        return left + right;
    };
    const auto& size = cropPredictionXYZ.sizeXYZ;
    const size_t xyCount = checkedProduct(size[0], size[1]);
    const size_t xzCount = checkedProduct(size[0], size[2]);
    const size_t yzCount = checkedProduct(size[1], size[2]);
    const size_t totalCount = checkedAdd(checkedAdd(xyCount, xzCount), yzCount);
    constexpr size_t kMaximumSlicePixels = 1'000'000;
    if (totalCount > kMaximumSlicePixels) {
        throw std::invalid_argument("fiber presence slices exceed 1000000 pixels; rerun paths with --no-slices");
    }

    FiberPresenceSliceReport report;
    report.cropPredictionXYZ = cropPredictionXYZ;
    const auto& origin = cropPredictionXYZ.originXYZ;
    const std::array<size_t, 3> centerXYZ{
        origin[0] + (size[0] - 1) / 2,
        origin[1] + (size[1] - 1) / 2,
        origin[2] + (size[2] - 1) / 2,
    };
    report.planes = {
        {"xy", {0, 1}, 2, centerXYZ[2], size[0], size[1], {}},
        {"xz", {0, 2}, 1, centerXYZ[1], size[0], size[2], {}},
        {"yz", {1, 2}, 0, centerXYZ[0], size[1], size[2], {}},
    };

    const auto appendPlane = [&](FiberPresenceSlice& plane, auto&& generateIndices, size_t pixelCount) {
        plane.pixels.reserve(pixelCount);
        constexpr size_t kBatchSize = 64 * 1024;
        std::vector<std::array<size_t, 3>> indices;
        indices.reserve(std::min(pixelCount, kBatchSize));
        const auto flush = [&]() {
            if (indices.empty())
                return;
            std::vector<FiberStoredPresenceSample> samples;
            presenceSampler(indices, parallelThreads, samples);
            if (samples.size() != indices.size())
                throw std::runtime_error("fiber presence sampler returned the wrong number of samples");
            for (size_t index = 0; index < indices.size(); ++index) {
                double presence = 0.0;
                if (samples[index].valid) {
                    if (!std::isfinite(samples[index].presence))
                        throw std::runtime_error("fiber presence sampler returned non-finite presence");
                    presence = std::clamp(samples[index].presence, 0.0, 1.0);
                }
                plane.pixels.push_back({indices[index], presence});
            }
            indices.clear();
        };
        generateIndices([&](const std::array<size_t, 3>& indexZYX) {
            indices.push_back(indexZYX);
            if (indices.size() == kBatchSize)
                flush();
        });
        flush();
    };

    appendPlane(
        report.planes[0],
        [&](auto&& emit) {
            for (size_t y = origin[1]; y < origin[1] + size[1]; ++y)
                for (size_t x = origin[0]; x < origin[0] + size[0]; ++x)
                    emit({centerXYZ[2], y, x});
        },
        xyCount);
    appendPlane(
        report.planes[1],
        [&](auto&& emit) {
            for (size_t z = origin[2]; z < origin[2] + size[2]; ++z)
                for (size_t x = origin[0]; x < origin[0] + size[0]; ++x)
                    emit({z, centerXYZ[1], x});
        },
        xzCount);
    appendPlane(
        report.planes[2],
        [&](auto&& emit) {
            for (size_t z = origin[2]; z < origin[2] + size[2]; ++z)
                for (size_t y = origin[1]; y < origin[1] + size[1]; ++y)
                    emit({z, y, centerXYZ[0]});
        },
        yzCount);
    return report;
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
    if (!finiteNonnegative(config.invalidPredictionCostPerVoxel) || !finiteNonnegative(config.smoothnessWeight) ||
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
    config.gaussianCutoffSigmas =
        finiteNumber(parameters.at("gaussian_cutoff_sigmas"), "gaussian_cutoff_sigmas");
    config.localWindowRadiusPredictionVoxels =
        finiteNumber(parameters.at("local_window_radius_prediction_voxels"), "local_window_radius_prediction_voxels");
    config.axialSupportHalfWidthPredictionVoxels =
        finiteNumber(parameters.at("axial_support_half_width_prediction_voxels"), "axial_support_half_width_prediction_voxels");
    config.positionConvergenceTolerancePredictionVoxels =
        finiteNumber(parameters.at("position_convergence_tolerance_prediction_voxels"), "position_convergence_tolerance_prediction_voxels");
    config.nmsMaximumAngleDegrees =
        finiteNumber(parameters.at("nms_maximum_angle_degrees"), "nms_maximum_angle_degrees");
    config.nmsLongitudinalRadiusPredictionVoxels =
        finiteNumber(parameters.at("nms_longitudinal_radius_prediction_voxels"), "nms_longitudinal_radius_prediction_voxels");
    config.observationPresenceFloor = finiteNumber(parameters.at("observation_presence_floor"), "observation_presence_floor");
    config.minimumAlignedSupport = finiteNumber(parameters.at("minimum_aligned_support"), "minimum_aligned_support");
    config.mergeMaximumAngleDegrees = finiteNumber(parameters.at("merge_maximum_angle_degrees"), "merge_maximum_angle_degrees");
    config.mergeMaximumAbsoluteObjectiveLoss =
        finiteNumber(parameters.at("merge_maximum_absolute_objective_loss"), "merge_maximum_absolute_objective_loss");
    config.mergeMaximumRelativeObjectiveLoss =
        finiteNumber(parameters.at("merge_maximum_relative_objective_loss"), "merge_maximum_relative_objective_loss");
    config.maximumSeedCount = parameters.at("maximum_seed_count").get<size_t>();
    config.maximumIterations = parameters.at("maximum_iterations").get<int>();
    config.convergenceTolerance = finiteNumber(parameters.at("convergence_tolerance"), "convergence_tolerance");
    config.parallelThreads = 1;
    validateFiberAnchorConfig(config);
    size_t selectedCellCount = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t totalCells =
            (loaded.report.grid.shapeZYX[axis] +
                static_cast<size_t>(config.cellSizePredictionVoxels) - 1) /
            static_cast<size_t>(config.cellSizePredictionVoxels);
        if (loaded.report.selectedCellBeginZYX[axis] >=
                loaded.report.selectedCellEndZYX[axis] ||
            loaded.report.selectedCellEndZYX[axis] > totalCells) {
            throw std::runtime_error("fiber anchor selected cell range is invalid");
        }
        const size_t extent = loaded.report.selectedCellEndZYX[axis] -
            loaded.report.selectedCellBeginZYX[axis];
        if (selectedCellCount > std::numeric_limits<size_t>::max() / extent)
            throw std::runtime_error("fiber anchor selected cell count overflows");
        selectedCellCount *= extent;
    }
    const auto& diagnostics = root.at("diagnostics");
    loaded.report.diagnostics.totalCells = diagnostics.at("total_cells").get<size_t>();
    loaded.report.diagnostics.zeroAnchorCells = diagnostics.at("zero_anchor_cells").get<size_t>();
    loaded.report.diagnostics.oneAnchorCells = diagnostics.at("one_anchor_cells").get<size_t>();
    loaded.report.diagnostics.twoAnchorCells = diagnostics.at("two_anchor_cells").get<size_t>();
    loaded.report.diagnostics.emptyComponents = diagnostics.at("empty_components").get<size_t>();
    loaded.report.diagnostics.degenerateComponents = diagnostics.at("degenerate_components").get<size_t>();
    loaded.report.diagnostics.belowSupportComponents = diagnostics.at("below_support_components").get<size_t>();
    loaded.report.diagnostics.mergedComponentPairs = diagnostics.at("merged_component_pairs").get<size_t>();
    loaded.report.diagnostics.nmsSuppressedComponents = diagnostics.at("nms_suppressed_components").get<size_t>();

    std::optional<std::array<size_t, 3>> previousCell;
    size_t storedMergedComponentPairs = 0;
    size_t storedNmsSuppressedComponents = 0;
    const auto& cells = root.at("cells");
    if (!cells.is_array())
        throw std::runtime_error("fiber anchor cells must be an array");
    for (const auto& cellJson : cells) {
        FiberCellAnchorResult cell;
        cell.cellZYX = jsonSize3(cellJson.at("cell_zyx"), "cell_zyx");
        for (size_t axis = 0; axis < 3; ++axis) {
            if (cell.cellZYX[axis] < loaded.report.selectedCellBeginZYX[axis] ||
                cell.cellZYX[axis] >= loaded.report.selectedCellEndZYX[axis]) {
                throw std::runtime_error("fiber anchor cell lies outside the selected cell range");
            }
        }
        if (previousCell.has_value() && !(previousCell.value() < cell.cellZYX))
            throw std::runtime_error("fiber anchor cells must be strictly ordered");
        previousCell = cell.cellZYX;
        cell.objective = finiteNumber(cellJson.at("objective"), "cell objective");
        if (cellJson.contains("merge_evaluation")) {
            const auto& mergeJson = cellJson.at("merge_evaluation");
            FiberAnchorMergeEvaluation merge;
            merge.angleDegrees = finiteNumber(mergeJson.at("angle_degrees"), "merge angle_degrees");
            merge.jointObjective = finiteNumber(mergeJson.at("joint_objective"), "merge joint_objective");
            merge.splitObjective = finiteNumber(mergeJson.at("split_objective"), "merge split_objective");
            merge.objectiveLoss = finiteNumber(mergeJson.at("objective_loss"), "merge objective_loss");
            merge.allowedObjectiveLoss = finiteNumber(mergeJson.at("allowed_objective_loss"), "merge allowed_objective_loss");
            merge.merged = mergeJson.at("merged").get<bool>();
            if (merge.angleDegrees < 0.0 || merge.angleDegrees > 90.0 || merge.jointObjective < 0.0 || merge.jointObjective > 1.0 ||
                merge.splitObjective < 0.0 || merge.splitObjective > 1.0 || merge.objectiveLoss < 0.0 || merge.objectiveLoss > 1.0 ||
                merge.allowedObjectiveLoss < 0.0 || merge.allowedObjectiveLoss > 1.0) {
                throw std::runtime_error("fiber anchor merge evaluation is outside its valid range");
            }
            const double expectedLoss = std::max(0.0, merge.splitObjective - merge.jointObjective);
            const double expectedAllowed =
                std::max(config.mergeMaximumAbsoluteObjectiveLoss, config.mergeMaximumRelativeObjectiveLoss * merge.jointObjective);
            const double tolerance = 1.0e-12;
            if (std::abs(merge.objectiveLoss - expectedLoss) > tolerance || std::abs(merge.allowedObjectiveLoss - expectedAllowed) > tolerance ||
                merge.merged != (merge.angleDegrees <= config.mergeMaximumAngleDegrees && merge.objectiveLoss <= merge.allowedObjectiveLoss)) {
                throw std::runtime_error("fiber anchor merge evaluation is inconsistent");
            }
            cell.mergeEvaluation = merge;
        }
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
                const std::set<std::string> validReasons{"empty", "degenerate", "below_support", "merged_same_direction", "nms_suppressed"};
                if (!validReasons.contains(component.rejectionReason))
                    throw std::runtime_error("rejected fiber anchor component has an unsupported reason");
                if (component.rejectionReason == "nms_suppressed")
                    ++storedNmsSuppressedComponents;
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
            component.anchor.refinementScore = finiteNumber(
                componentJson.at("refinement_score"), "refinement_score");
            component.anchor.refinementIterations =
                componentJson.at("refinement_iterations").get<size_t>();
            if (component.anchor.alignedSupport < 0.0 ||
                component.anchor.alignedSupport > 1.0 ||
                component.anchor.directionalCoherence < 0.0 ||
                component.anchor.directionalCoherence > 1.0 ||
                component.anchor.refinementScore < 0.0 ||
                component.anchor.refinementScore > 1.0 ||
                std::abs(component.anchor.refinementScore -
                    component.anchor.alignedSupport) > 1.0e-12 ||
                component.anchor.refinementIterations >
                    static_cast<size_t>(config.maximumIterations)) {
                throw std::runtime_error("fiber anchor refinement values are inconsistent");
            }
            const cv::Vec3d pivot{
                (static_cast<double>(cell.cellZYX[2] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                    static_cast<double>(std::min(
                        (cell.cellZYX[2] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels),
                        loaded.report.grid.shapeZYX[2])) - 1.0) * 0.5,
                (static_cast<double>(cell.cellZYX[1] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                    static_cast<double>(std::min(
                        (cell.cellZYX[1] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels),
                        loaded.report.grid.shapeZYX[1])) - 1.0) * 0.5,
                (static_cast<double>(cell.cellZYX[0] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                    static_cast<double>(std::min(
                        (cell.cellZYX[0] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels),
                        loaded.report.grid.shapeZYX[0])) - 1.0) * 0.5,
            };
            const cv::Vec3d pivotOffset =
                component.anchor.positionPredictionXYZ - pivot;
            const double planeResidual = std::abs(
                pivotOffset.dot(component.anchor.axisXYZ));
            const double pivotDistance = vectorLength(pivotOffset);
            for (int axis = 0; axis < 3; ++axis) {
                const double position = component.anchor.positionPredictionXYZ[axis];
                const size_t gridAxis = static_cast<size_t>(2 - axis);
                if (position < -kEpsilon ||
                    position > static_cast<double>(loaded.report.grid.shapeZYX[gridAxis] - 1) + kEpsilon) {
                    throw std::runtime_error("fiber anchor position is outside the prediction grid");
                }
            }
            if (planeResidual > 1.0e-6 ||
                pivotDistance > config.localWindowRadiusPredictionVoxels + 1.0e-6) {
                throw std::runtime_error("fiber anchor position violates its rotating-plane window");
            }
            ++cell.retainedAnchorCount;
        }
        const size_t mergedReasons = static_cast<size_t>(std::count_if(cell.components.begin(), cell.components.end(), [](const auto& component) {
            return component.rejectionReason == "merged_same_direction";
        }));
        const bool merged = cell.mergeEvaluation.has_value() && cell.mergeEvaluation->merged;
        if (merged) {
            if (mergedReasons != 1 || cell.retainedAnchorCount != 1) {
                throw std::runtime_error("merged fiber anchor cell is inconsistent");
            }
            ++storedMergedComponentPairs;
        } else if (mergedReasons != 0) {
            throw std::runtime_error("unmerged fiber anchor cell has a merged component reason");
        }
        if (cell.retainedAnchorCount == 0)
            throw std::runtime_error("fiber anchor artifact must not store empty cells");
        loaded.report.nonEmptyCells.push_back(std::move(cell));
    }
    if (storedMergedComponentPairs > loaded.report.diagnostics.mergedComponentPairs)
        throw std::runtime_error("fiber anchor merged-component diagnostics are inconsistent");
    if (storedNmsSuppressedComponents >
        loaded.report.diagnostics.nmsSuppressedComponents) {
        throw std::runtime_error("fiber anchor NMS diagnostics are inconsistent");
    }
    if (loaded.report.diagnostics.zeroAnchorCells +
            loaded.report.diagnostics.oneAnchorCells +
            loaded.report.diagnostics.twoAnchorCells !=
        loaded.report.diagnostics.totalCells) {
        throw std::runtime_error("fiber anchor cell-count diagnostics are inconsistent");
    }
    if (loaded.report.diagnostics.totalCells != selectedCellCount)
        throw std::runtime_error("fiber anchor total cells disagree with the selected lattice");
    if (loaded.report.nonEmptyCells.size() !=
        loaded.report.diagnostics.oneAnchorCells +
            loaded.report.diagnostics.twoAnchorCells) {
        throw std::runtime_error("fiber anchor stored cells disagree with diagnostics");
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
    const vc::lasagna::NormalSampler& normalSampler,
    const FiberletPathProgressCallback& progressCallback)
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
    std::vector<size_t> searchCandidateIndices;
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
                searchCandidateIndices.push_back(report.candidates.size());
                report.candidates.push_back(std::move(candidate));
            }
        }
    }
    const auto candidateGenerationEnd = Clock::now();
    report.candidateGenerationSeconds = std::chrono::duration<double>(candidateGenerationEnd - startTime).count();

    DenseScoringVolume scoringVolume;
    if (!searchCandidateIndices.empty()) {
        scoringVolume = preloadScoringVolume(
            report.candidates,
            searchCandidateIndices,
            grid,
            report.anchorCellSizePredictionVoxels,
            report.config,
            predictionSampler,
            normalSampler,
            report.estimatedPreloadBytes);
        report.preloadedVoxels = scoringVolume.voxels.size();
    }
    const auto preloadEnd = Clock::now();
    report.preloadSeconds = std::chrono::duration<double>(preloadEnd - candidateGenerationEnd).count();

    std::vector<std::exception_ptr> errors(searchCandidateIndices.size());
    std::atomic<size_t> nextSearch{0};
    std::atomic<size_t> completedSearches{0};
    const size_t workerCount = std::min(searchCandidateIndices.size(), static_cast<size_t>(report.config.parallelThreads));
    report.candidateWorkers = workerCount;
    std::mutex progressMutex;
    size_t lastReportedCompleted = 0;
    auto lastProgressTime = preloadEnd;
    std::exception_ptr progressError;
    const auto reportProgress = [&](bool terminal) noexcept {
        if (!progressCallback)
            return;
        std::lock_guard lock(progressMutex);
        if (progressError)
            return;
        const size_t completed = completedSearches.load(std::memory_order_relaxed);
        const size_t total = searchCandidateIndices.size();
        const auto now = Clock::now();
        if (!terminal) {
            if (completed <= lastReportedCompleted || completed >= total || now - lastProgressTime < std::chrono::seconds(1)) {
                return;
            }
        } else if (total > 0 && lastReportedCompleted >= total) {
            return;
        }
        try {
            progressCallback({
                completed,
                total,
                std::chrono::duration<double>(now - preloadEnd).count(),
            });
            lastReportedCompleted = completed;
            lastProgressTime = now;
        } catch (...) {
            progressError = std::current_exception();
        }
    };
    const auto worker = [&]() {
        while (true) {
            const size_t searchIndex = nextSearch.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= searchCandidateIndices.size())
                return;
            const size_t candidateIndex = searchCandidateIndices[searchIndex];
            try {
                report.candidates[candidateIndex] =
                    solveCandidate(report.candidates[candidateIndex], grid, report.anchorCellSizePredictionVoxels, report.config, scoringVolume);
            } catch (...) {
                errors[searchIndex] = std::current_exception();
            }
            completedSearches.fetch_add(1, std::memory_order_relaxed);
            reportProgress(false);
        }
    };
    if (workerCount == 1) {
        worker();
    } else if (workerCount > 1) {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();
    }
    reportProgress(true);
    for (const auto& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
    if (progressError)
        std::rethrow_exception(progressError);
    for (const size_t candidateIndex : searchCandidateIndices) {
        const auto& candidate = report.candidates[candidateIndex];
        ++report.diagnostics.searchedPairs;
        if (candidate.success)
            ++report.diagnostics.successfulPaths;
        else
            ++report.diagnostics.noPathPairs;
    }
    const auto searchEnd = Clock::now();
    report.searchSeconds = std::chrono::duration<double>(searchEnd - preloadEnd).count();
    report.elapsedSeconds = std::chrono::duration<double>(searchEnd - startTime).count();
    return report;
}

nlohmann::json fiberletPathReportJson(const FiberletPathReport& report, const FiberletArtifactInfo& artifact)
{
    if (artifact.fiberManifestLocator.empty() || artifact.fiberManifestContentHash.empty() || artifact.normalManifestLocator.empty() ||
        artifact.normalManifestContentHash.empty() || artifact.anchorArtifactLocator.empty() || artifact.anchorArtifactContentHash.empty()) {
        throw std::invalid_argument("fiberlet artifacts require complete source identities");
    }
    const auto visual = fiberletPathVisualMetrics(report);
    std::vector<const FiberletPathVisualMetric*> visualByCandidate(
        report.candidates.size(), nullptr);
    for (const auto& path : visual.paths)
        visualByCandidate[path.candidateIndex] = &path;
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
        {"trace_quality_visualization",
         {
             {"population", "successful_scored_fiberlets"},
             {"loss_density_unit", "prediction_voxel"},
             {"relative_quality_formula", "inverse_min_max_low_loss_is_one"},
             {"count", visual.paths.size()},
             {"minimum_loss_per_prediction_voxel",
              visual.minimumLossPerPredictionVoxel.has_value()
                  ? nlohmann::json(*visual.minimumLossPerPredictionVoxel)
                  : nlohmann::json(nullptr)},
             {"maximum_loss_per_prediction_voxel",
              visual.maximumLossPerPredictionVoxel.has_value()
                  ? nlohmann::json(*visual.maximumLossPerPredictionVoxel)
                  : nlohmann::json(nullptr)},
         }},
        {"candidates", nlohmann::json::array()},
    };
    if (artifact.baseVoxelSizeUm.has_value())
        root["coordinates"]["base_voxel_size_um"] = *artifact.baseVoxelSizeUm;
    for (size_t candidateIndex = 0;
         candidateIndex < report.candidates.size(); ++candidateIndex) {
        const auto& candidate = report.candidates[candidateIndex];
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
                {"alignment", candidate.cost.alignment},
                {"isotropic_smoothness", candidate.cost.isotropicSmoothness},
                {"tangent_smoothness", candidate.cost.tangentSmoothness},
                {"normal_smoothness", candidate.cost.normalSmoothness},
            };
        }
        if (candidate.success) {
            const auto* metrics = visualByCandidate[candidateIndex];
            if (metrics == nullptr)
                throw std::logic_error("successful fiberlet has no visualization metrics");
            item["path_length_base_voxels"] =
                metrics->pathLengthPredictionVoxels *
                report.grid.predictionToBaseScale;
            item["loss_per_prediction_voxel"] =
                metrics->lossPerPredictionVoxel;
            item["relative_visual_quality"] = metrics->relativeQuality;
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
    const auto visual = fiberletPathVisualMetrics(report);
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "# vc_fiberlets version 1\n";
    output << "# trace_quality_population successful_scored_fiberlets\n";
    output << "# trace_loss_density_unit prediction_voxel\n";
    output << "# trace_quality_formula inverse_min_max_low_loss_is_one\n";
    output << "# trace_quality_count " << visual.paths.size() << '\n';
    output << "# trace_loss_density_min ";
    if (visual.minimumLossPerPredictionVoxel.has_value())
        output << *visual.minimumLossPerPredictionVoxel;
    else
        output << "none";
    output << '\n';
    output << "# trace_loss_density_max ";
    if (visual.maximumLossPerPredictionVoxel.has_value())
        output << *visual.maximumLossPerPredictionVoxel;
    else
        output << "none";
    output << '\n';
    size_t vertex = 1;
    for (const auto& path : visual.paths) {
        const auto& candidate = report.candidates[path.candidateIndex];
        output << "g " << fiberletId(candidate) << '\n';
        output << "# trace_loss_total " << path.totalLoss << '\n';
        output << "# trace_loss_per_prediction_voxel "
               << path.lossPerPredictionVoxel << '\n';
        output << "# trace_quality_relative " << path.relativeQuality << '\n';
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
    std::error_code error;
    std::filesystem::remove(outputDirectory / "fiberlets.mtl", error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "cannot remove stale fiberlet material artifact",
            outputDirectory / "fiberlets.mtl",
            error);
    }
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.json", fiberletPathReportJson(report, artifact).dump(2) + "\n");
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.obj", fiberletPathReportObj(report));
}

namespace
{

struct PendingPresenceSliceArtifact {
    std::filesystem::path png;
    std::filesystem::path mtl;
    std::filesystem::path obj;
    std::filesystem::path temporaryPng;
    std::filesystem::path temporaryMtl;
    std::filesystem::path temporaryObj;
};

void removeSliceFile(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::remove(path, error);
    if (error)
        throw std::filesystem::filesystem_error("cannot remove fiber presence slice artifact", path, error);
}

PendingPresenceSliceArtifact presenceSliceArtifactPaths(const std::filesystem::path& outputDirectory, const std::string& plane)
{
    const std::string stem = "fiber_presence_" + plane;
    return {
        outputDirectory / (stem + ".png"),
        outputDirectory / (stem + ".mtl"),
        outputDirectory / (stem + ".obj"),
        outputDirectory / (stem + ".tmp.png"),
        outputDirectory / (stem + ".tmp.mtl"),
        outputDirectory / (stem + ".tmp.obj"),
    };
}

void validatePresenceSlice(const FiberPresenceSliceReport& report, const FiberPresenceSlice& plane)
{
    const std::array<std::string, 3> names{"xy", "xz", "yz"};
    const std::array<std::array<size_t, 2>, 3> varyingAxes{{{0, 1}, {0, 2}, {1, 2}}};
    const std::array<size_t, 3> fixedAxes{2, 1, 0};
    const auto found = std::find(names.begin(), names.end(), plane.name);
    if (found == names.end())
        throw std::invalid_argument("fiber presence slice has an unknown plane name");
    const size_t planeIndex = static_cast<size_t>(std::distance(names.begin(), found));
    if (plane.varyingAxesXYZ != varyingAxes[planeIndex] || plane.fixedAxisXYZ != fixedAxes[planeIndex]) {
        throw std::invalid_argument("fiber presence slice axes do not match its plane name");
    }
    const auto& origin = report.cropPredictionXYZ.originXYZ;
    const auto& size = report.cropPredictionXYZ.sizeXYZ;
    if (plane.width == 0 || plane.height == 0 || plane.height > std::numeric_limits<size_t>::max() / plane.width ||
        plane.width > static_cast<size_t>(std::numeric_limits<int>::max()) || plane.height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        plane.width != size[plane.varyingAxesXYZ[0]] || plane.height != size[plane.varyingAxesXYZ[1]] ||
        plane.fixedIndex != origin[plane.fixedAxisXYZ] + (size[plane.fixedAxisXYZ] - 1) / 2 || plane.pixels.size() != plane.width * plane.height) {
        throw std::invalid_argument("fiber presence slice dimensions do not match its crop");
    }
    for (size_t row = 0; row < plane.height; ++row) {
        for (size_t column = 0; column < plane.width; ++column) {
            const auto& pixel = plane.pixels[row * plane.width + column];
            std::array<size_t, 3> expectedXYZ = origin;
            expectedXYZ[plane.varyingAxesXYZ[0]] += column;
            expectedXYZ[plane.varyingAxesXYZ[1]] += row;
            expectedXYZ[plane.fixedAxisXYZ] = plane.fixedIndex;
            const std::array<size_t, 3> expectedZYX{expectedXYZ[2], expectedXYZ[1], expectedXYZ[0]};
            if (pixel.indexZYX != expectedZYX)
                throw std::invalid_argument("fiber presence slice pixel order is inconsistent");
            if (!std::isfinite(pixel.presence) || pixel.presence < 0.0 || pixel.presence > 1.0) {
                throw std::invalid_argument("fiber presence slice pixel must be in [0, 1]");
            }
        }
    }
}

vc::core::util::TexturedMesh presenceSliceMesh(const FiberPresenceSliceReport& report, const FiberPresenceSlice& plane, const FiberPredictionGridInfo& grid)
{
    const auto& origin = report.cropPredictionXYZ.originXYZ;
    const auto& size = report.cropPredictionXYZ.sizeXYZ;
    const size_t firstAxis = plane.varyingAxesXYZ[0];
    const size_t secondAxis = plane.varyingAxesXYZ[1];
    const double scale = grid.predictionToBaseScale;
    cv::Vec3d minimum{0.0, 0.0, 0.0};
    cv::Vec3d maximum{0.0, 0.0, 0.0};
    minimum[plane.fixedAxisXYZ] = static_cast<double>(plane.fixedIndex) * scale;
    maximum[plane.fixedAxisXYZ] = minimum[plane.fixedAxisXYZ];
    minimum[firstAxis] = (static_cast<double>(origin[firstAxis]) - 0.5) * scale;
    maximum[firstAxis] = (static_cast<double>(origin[firstAxis] + size[firstAxis]) - 0.5) * scale;
    minimum[secondAxis] = (static_cast<double>(origin[secondAxis]) - 0.5) * scale;
    maximum[secondAxis] = (static_cast<double>(origin[secondAxis] + size[secondAxis]) - 0.5) * scale;

    cv::Vec3d lowerRight = minimum;
    lowerRight[firstAxis] = maximum[firstAxis];
    cv::Vec3d upperRight = maximum;
    cv::Vec3d upperLeft = minimum;
    upperLeft[secondAxis] = maximum[secondAxis];
    vc::core::util::TexturedMesh mesh;
    mesh.vertices = {minimum, lowerRight, upperRight, upperLeft};
    mesh.textureCoordinates = {{0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}};
    mesh.quads.push_back({{0, 1, 2, 3}, {0, 1, 2, 3}});
    return mesh;
}

cv::Mat presenceSliceImage(const FiberPresenceSlice& plane)
{
    cv::Mat image(static_cast<int>(plane.height), static_cast<int>(plane.width), CV_8UC1);
    for (size_t row = 0; row < plane.height; ++row) {
        auto* target = image.ptr<uint8_t>(static_cast<int>(row));
        for (size_t column = 0; column < plane.width; ++column) {
            target[column] = static_cast<uint8_t>(std::lround(plane.pixels[row * plane.width + column].presence * 255.0));
        }
    }
    return image;
}

}  // namespace

void writeFiberPresenceSliceArtifacts(const std::filesystem::path& outputDirectory, const FiberPresenceSliceReport& report, const FiberPredictionGridInfo& grid)
{
    if (outputDirectory.empty())
        throw std::invalid_argument("fiber presence slice output directory must not be empty");
    if (!(grid.predictionToBaseScale > 0.0) || !std::isfinite(grid.predictionToBaseScale)) {
        throw std::invalid_argument("fiber presence slice output requires a valid prediction-to-base scale");
    }
    if (report.planes.size() != 3)
        throw std::invalid_argument("fiber presence slice output requires exactly three planes");
    std::filesystem::create_directories(outputDirectory);
    removeSliceFile(outputDirectory / "fiber_presence_slices.obj");

    std::vector<PendingPresenceSliceArtifact> pending;
    pending.reserve(report.planes.size());
    try {
        std::set<std::string> planeNames;
        for (const auto& plane : report.planes) {
            validatePresenceSlice(report, plane);
            if (!planeNames.insert(plane.name).second)
                throw std::invalid_argument("fiber presence slice plane name is duplicated");
            const auto paths = presenceSliceArtifactPaths(outputDirectory, plane.name);
            removeSliceFile(paths.temporaryPng);
            removeSliceFile(paths.temporaryMtl);
            removeSliceFile(paths.temporaryObj);
            const std::string material = "fiber_presence_" + plane.name;
            const std::string pngName = paths.png.filename().string();
            const std::string mtlName = paths.mtl.filename().string();
            const auto mesh = presenceSliceMesh(report, plane, grid);
            const std::string obj = vc::core::util::texturedMeshObj(mesh, "vc_fiber_presence_slice version 1", mtlName, material, material);
            const std::string mtl = vc::core::util::textureMaterialMtl(material, pngName);
            if (!cv::imwrite(paths.temporaryPng.string(), presenceSliceImage(plane), {cv::IMWRITE_PNG_COMPRESSION, 1})) {
                throw std::runtime_error("failed to write fiber presence texture: " + paths.temporaryPng.string());
            }
            vc::core::util::atomicWriteString(paths.temporaryMtl, mtl);
            vc::core::util::atomicWriteString(paths.temporaryObj, obj);
            pending.push_back(paths);
        }
        for (const auto& paths : pending) {
            vc::core::util::replaceFileAtomically(paths.temporaryPng, paths.png);
            vc::core::util::replaceFileAtomically(paths.temporaryMtl, paths.mtl);
            vc::core::util::replaceFileAtomically(paths.temporaryObj, paths.obj);
        }
    } catch (...) {
        for (const auto& plane : report.planes) {
            const auto paths = presenceSliceArtifactPaths(outputDirectory, plane.name);
            std::error_code ignored;
            std::filesystem::remove(paths.temporaryPng, ignored);
            std::filesystem::remove(paths.temporaryMtl, ignored);
            std::filesystem::remove(paths.temporaryObj, ignored);
        }
        throw;
    }
}

void removeFiberPresenceSliceArtifacts(const std::filesystem::path& outputDirectory)
{
    if (outputDirectory.empty())
        throw std::invalid_argument("fiber presence slice output directory must not be empty");
    removeSliceFile(outputDirectory / "fiber_presence_slices.obj");
    for (const std::string plane : {"xy", "xz", "yz"}) {
        const auto paths = presenceSliceArtifactPaths(outputDirectory, plane);
        removeSliceFile(paths.png);
        removeSliceFile(paths.mtl);
        removeSliceFile(paths.obj);
        removeSliceFile(paths.temporaryPng);
        removeSliceFile(paths.temporaryMtl);
        removeSliceFile(paths.temporaryObj);
    }
}

}  // namespace vc::fiber_tracer
