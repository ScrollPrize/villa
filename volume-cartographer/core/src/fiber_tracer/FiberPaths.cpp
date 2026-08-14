#include "vc/fiber_tracer/FiberPaths.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/TexturedMesh.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
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
#include <unordered_set>
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

double processCpuSeconds()
{
    const std::clock_t ticks = std::clock();
    return ticks == static_cast<std::clock_t>(-1) ? 0.0 : static_cast<double>(ticks) / static_cast<double>(CLOCKS_PER_SEC);
}

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

struct LocalNodeKey {
    size_t layer = 0;
    int transverseU = 0;
    int transverseV = 0;
};

struct LocalNodeKeyLayout {
    uint32_t transverseWidth = 0;
    int transverseLimit = 0;
};

struct CurvedLayer {
    double arc = 0.0;
    cv::Vec3d center{0.0, 0.0, 0.0};
    cv::Vec3d tangent{1.0, 0.0, 0.0};
    cv::Vec3d transverseU{0.0, 1.0, 0.0};
    cv::Vec3d transverseV{0.0, 0.0, 1.0};
};

struct CurvedDomain {
    std::vector<CurvedLayer> layers;
    double length = 0.0;
};

struct SearchNode {
    cv::Vec3f point{0.0f, 0.0f, 0.0f};
    uint32_t key = 0;
    std::array<uint8_t, 2> predictionAxis{128, 128};
    std::array<uint8_t, 2> normalAxis{128, 128};
    uint8_t presence = 0;
    uint8_t flags = 0;
};

static_assert(sizeof(SearchNode) == 24);

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

struct PreparedCandidate {
    CurvedDomain domain;
    std::vector<SearchNode> nodes;
    LocalNodeKeyLayout keyLayout;
    ScoringVoxel startScoring;
    ScoringVoxel targetScoring;
};

struct BackPointer {
    int64_t node = -1;
    int state = -1;
};

struct DpState {
    bool reached = false;
    FiberletPathCost cost;
    BackPointer previous;
    cv::Vec3d incomingDirection{0.0, 0.0, 0.0};
    double incomingLength = 0.0;
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

cv::Vec3d nativeVoxelPoint(const Voxel& voxel)
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

bool insidePredictionGrid(const cv::Vec3d& point, const FiberPredictionGridInfo& grid)
{
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    for (size_t axis = 0; axis < 3; ++axis) {
        if (shapeXYZ[axis] == 0 || !std::isfinite(point[axis]) || point[axis] < 0.0)
            return false;
        if (point[axis] > static_cast<double>(shapeXYZ[axis] - 1))
            return false;
    }
    return true;
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

cv::Vec3d hermitePoint(const cv::Vec3d& start, const cv::Vec3d& target, const cv::Vec3d& startDerivative, const cv::Vec3d& targetDerivative, double t)
{
    const double t2 = t * t;
    const double t3 = t2 * t;
    return start * (2.0 * t3 - 3.0 * t2 + 1.0) + startDerivative * (t3 - 2.0 * t2 + t) + target * (-2.0 * t3 + 3.0 * t2) +
           targetDerivative * (t3 - t2);
}

cv::Vec3d hermiteDerivative(const cv::Vec3d& start, const cv::Vec3d& target, const cv::Vec3d& startDerivative, const cv::Vec3d& targetDerivative, double t)
{
    const double t2 = t * t;
    return start * (6.0 * t2 - 6.0 * t) + startDerivative * (3.0 * t2 - 4.0 * t + 1.0) + target * (-6.0 * t2 + 6.0 * t) +
           targetDerivative * (3.0 * t2 - 2.0 * t);
}

cv::Vec3d rotateMinimal(const cv::Vec3d& value, const cv::Vec3d& from, const cv::Vec3d& to)
{
    const cv::Vec3d cross = from.cross(to);
    const double sine = vectorLength(cross);
    const double cosine = std::clamp(from.dot(to), -1.0, 1.0);
    if (sine <= kEpsilon) {
        if (cosine >= 0.0)
            return value;
        cv::Vec3d axis = normalized(value - from * value.dot(from));
        if (vectorLength(axis) <= kEpsilon)
            axis = normalized(cv::Vec3d{1.0, 0.0, 0.0} - from * from[0]);
        return axis * (2.0 * axis.dot(value)) - value;
    }
    const cv::Vec3d axis = cross / sine;
    return value * cosine + axis.cross(value) * sine + axis * axis.dot(value) * (1.0 - cosine);
}

CurvedDomain makeCurvedDomain(const FiberletCandidateResult& candidate, const FiberletPathConfig& config)
{
    const cv::Vec3d chord = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
    const double chordLength = vectorLength(chord);
    if (!(chordLength > kEpsilon))
        throw std::invalid_argument("fiberlet curved domain requires distinct endpoints");
    const cv::Vec3d startDerivative = candidate.startAxisXYZ * chordLength;
    const cv::Vec3d targetDerivative = candidate.targetAxisXYZ * chordLength;
    const size_t samples = static_cast<size_t>(std::max(64.0, std::ceil(chordLength * 16.0)));
    std::vector<double> sampleArcs(samples + 1, 0.0);
    std::vector<cv::Vec3d> samplePoints(samples + 1);
    for (size_t index = 0; index <= samples; ++index) {
        const double t = static_cast<double>(index) / static_cast<double>(samples);
        samplePoints[index] =
            hermitePoint(candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ, startDerivative, targetDerivative, t);
        if (index > 0) {
            sampleArcs[index] = sampleArcs[index - 1] + vectorLength(samplePoints[index] - samplePoints[index - 1]);
        }
    }
    CurvedDomain domain;
    domain.length = sampleArcs.back();
    if (!(domain.length > kEpsilon) || !std::isfinite(domain.length))
        throw std::runtime_error("fiberlet Hermite centerline has invalid length");
    std::vector<double> layerArcs{0.0};
    for (double arc = config.longitudinalStepPredictionVoxels; arc < domain.length - kEpsilon; arc += config.longitudinalStepPredictionVoxels) {
        layerArcs.push_back(arc);
    }
    layerArcs.push_back(domain.length);
    domain.layers.reserve(layerArcs.size());
    for (const double arc : layerArcs) {
        double t = 0.0;
        if (arc >= domain.length) {
            t = 1.0;
        } else if (arc > 0.0) {
            const auto upper = std::lower_bound(sampleArcs.begin(), sampleArcs.end(), arc);
            const size_t upperIndex = static_cast<size_t>(std::distance(sampleArcs.begin(), upper));
            const size_t lowerIndex = upperIndex - 1;
            const double span = sampleArcs[upperIndex] - sampleArcs[lowerIndex];
            const double fraction = span > kEpsilon ? (arc - sampleArcs[lowerIndex]) / span : 0.0;
            t = (static_cast<double>(lowerIndex) + fraction) / static_cast<double>(samples);
        }
        CurvedLayer layer;
        layer.arc = arc;
        layer.center =
            hermitePoint(candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ, startDerivative, targetDerivative, t);
        layer.tangent = normalized(
            hermiteDerivative(candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ, startDerivative, targetDerivative, t));
        if (vectorLength(layer.tangent) <= kEpsilon)
            throw std::runtime_error("fiberlet Hermite centerline has a degenerate tangent");
        if (domain.layers.empty()) {
            const std::array<cv::Vec3d, 3> axes{cv::Vec3d{1.0, 0.0, 0.0}, cv::Vec3d{0.0, 1.0, 0.0}, cv::Vec3d{0.0, 0.0, 1.0}};
            const auto reference = *std::min_element(axes.begin(), axes.end(), [&](const auto& left, const auto& right) {
                return std::abs(left.dot(layer.tangent)) < std::abs(right.dot(layer.tangent));
            });
            layer.transverseU = normalized(reference - layer.tangent * reference.dot(layer.tangent));
        } else {
            layer.transverseU = rotateMinimal(domain.layers.back().transverseU, domain.layers.back().tangent, layer.tangent);
            layer.transverseU = normalized(layer.transverseU - layer.tangent * layer.transverseU.dot(layer.tangent));
        }
        if (vectorLength(layer.transverseU) <= kEpsilon)
            throw std::runtime_error("fiberlet transverse frame is degenerate");
        layer.transverseV = normalized(layer.tangent.cross(layer.transverseU));
        domain.layers.push_back(layer);
    }
    domain.layers.front().center = candidate.startPositionPredictionXYZ;
    domain.layers.back().center = candidate.targetPositionPredictionXYZ;
    return domain;
}

std::vector<cv::Vec3d> domainCenterline(const CurvedDomain& domain)
{
    std::vector<cv::Vec3d> points;
    points.reserve(domain.layers.size());
    for (const auto& layer : domain.layers)
        points.push_back(layer.center);
    return points;
}

cv::Vec3d localNodePoint(const CurvedDomain& domain, const LocalNodeKey& key, const FiberletPathConfig& config)
{
    const auto& layer = domain.layers.at(key.layer);
    return layer.center + layer.transverseU * (static_cast<double>(key.transverseU) * config.transverseStepPredictionVoxels) +
           layer.transverseV * (static_cast<double>(key.transverseV) * config.transverseStepPredictionVoxels);
}

LocalNodeKeyLayout makeLocalNodeKeyLayout(
    const CurvedDomain& domain,
    double radius,
    double transverseStep)
{
    const double rawLimit = std::ceil(radius / transverseStep);
    if (!std::isfinite(rawLimit) || rawLimit < 0.0 ||
        rawLimit > static_cast<double>((std::numeric_limits<int>::max() - 1) / 2)) {
        throw std::overflow_error("fiberlet transverse node range exceeds packed-key limits");
    }
    LocalNodeKeyLayout layout;
    layout.transverseLimit = static_cast<int>(rawLimit);
    const uint64_t width =
        static_cast<uint64_t>(layout.transverseLimit) * 2ULL + 1ULL;
    if (width > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("fiberlet transverse node width exceeds packed-key limits");
    layout.transverseWidth = static_cast<uint32_t>(width);
    const uint64_t plane = width * width;
    constexpr uint64_t keyCapacity =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL;
    if (plane == 0 || plane > keyCapacity ||
        domain.layers.size() > keyCapacity / plane) {
        throw std::overflow_error("fiberlet local node lattice exceeds packed-key limits");
    }
    return layout;
}

uint32_t packLocalNodeKey(const LocalNodeKey& key, const LocalNodeKeyLayout& layout)
{
    if (layout.transverseWidth == 0 ||
        key.transverseU < -layout.transverseLimit ||
        key.transverseU > layout.transverseLimit ||
        key.transverseV < -layout.transverseLimit ||
        key.transverseV > layout.transverseLimit) {
        throw std::overflow_error("fiberlet local node lies outside packed-key layout");
    }
    const uint64_t width = layout.transverseWidth;
    const uint64_t plane = width * width;
    const uint64_t packed = static_cast<uint64_t>(key.layer) * plane +
        static_cast<uint64_t>(key.transverseU + layout.transverseLimit) * width +
        static_cast<uint64_t>(key.transverseV + layout.transverseLimit);
    if (packed > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("fiberlet local node key exceeds 32 bits");
    return static_cast<uint32_t>(packed);
}

LocalNodeKey unpackLocalNodeKey(uint32_t packed, const LocalNodeKeyLayout& layout)
{
    const uint64_t width = layout.transverseWidth;
    const uint64_t plane = width * width;
    if (width == 0 || plane == 0)
        throw std::logic_error("fiberlet packed-key layout is empty");
    LocalNodeKey key;
    key.layer = static_cast<size_t>(packed / plane);
    const uint64_t remainder = packed % plane;
    key.transverseU = static_cast<int>(remainder / width) -
        layout.transverseLimit;
    key.transverseV = static_cast<int>(remainder % width) -
        layout.transverseLimit;
    return key;
}

cv::Vec3d nodePoint(const SearchNode& node)
{
    return {
        static_cast<double>(node.point[0]),
        static_cast<double>(node.point[1]),
        static_cast<double>(node.point[2]),
    };
}

SearchCorridor makeSearchCorridor(const CurvedDomain& domain, const FiberPredictionGridInfo& grid, int cellSize, const FiberletPathConfig& config)
{
    SearchCorridor corridor;
    const double radius = config.corridorRadiusPredictionVoxels > 0.0 ? config.corridorRadiusPredictionVoxels : static_cast<double>(cellSize);
    corridor.radiusSquared = radius * radius;
    corridor.reference = domainCenterline(domain);
    cv::Vec3d minimum = corridor.reference.front();
    cv::Vec3d maximum = corridor.reference.front();
    for (const auto& point : corridor.reference) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    for (const cv::Vec3d* endpoint : {&domain.layers.front().center, &domain.layers.back().center}) {
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

std::vector<SearchNode> enumerateLocalNodes(
    const CurvedDomain& domain,
    const SearchCorridor& corridor,
    const FiberletPathConfig& config,
    const FiberPredictionGridInfo& grid,
    const FiberletPointPredicate& pointPredicate,
    const LocalNodeKeyLayout& layout)
{
    std::vector<SearchNode> nodes;
    if (domain.layers.size() <= 2)
        return nodes;
    for (size_t layer = 1; layer + 1 < domain.layers.size(); ++layer) {
        for (int u = -layout.transverseLimit; u <= layout.transverseLimit; ++u) {
            for (int v = -layout.transverseLimit; v <= layout.transverseLimit; ++v) {
                const LocalNodeKey key{layer, u, v};
                const cv::Vec3d mapped = localNodePoint(domain, key, config);
                const cv::Vec3f stored{
                    static_cast<float>(mapped[0]),
                    static_cast<float>(mapped[1]),
                    static_cast<float>(mapped[2]),
                };
                const cv::Vec3d point{
                    static_cast<double>(stored[0]),
                    static_cast<double>(stored[1]),
                    static_cast<double>(stored[2]),
                };
                if (!finiteVector(point))
                    throw std::overflow_error("fiberlet local node is not finite as float32");
                if (insidePredictionGrid(point, grid) && insideCorridor(point, corridor.reference, corridor.radiusSquared) &&
                    (!pointPredicate || pointPredicate(point))) {
                    SearchNode node;
                    node.point = stored;
                    node.key = packLocalNodeKey(key, layout);
                    nodes.push_back(node);
                }
            }
        }
    }
    return nodes;
}

template <typename Callback>
void forEachInterpolationCorner(
    const cv::Vec3d& point,
    const FiberPredictionGridInfo& grid,
    Callback&& callback)
{
    if (!insidePredictionGrid(point, grid))
        throw std::out_of_range("fiberlet sample point is outside the prediction volume");
    Voxel lower{};
    std::array<double, 3> fraction{};
    for (size_t axis = 0; axis < 3; ++axis) {
        lower[axis] = static_cast<int64_t>(std::floor(point[axis]));
        fraction[axis] = point[axis] - static_cast<double>(lower[axis]);
    }
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    Voxel upper{};
    for (size_t axis = 0; axis < 3; ++axis)
        upper[axis] = std::min<int64_t>(lower[axis] + 1, static_cast<int64_t>(shapeXYZ[axis] - 1));
    for (int z = 0; z <= 1; ++z) {
        const double wz = z == 0 ? 1.0 - fraction[2] : fraction[2];
        if (!(wz > 0.0))
            continue;
        for (int y = 0; y <= 1; ++y) {
            const double wy = y == 0 ? 1.0 - fraction[1] : fraction[1];
            if (!(wy > 0.0))
                continue;
            for (int x = 0; x <= 1; ++x) {
                const double wx = x == 0 ? 1.0 - fraction[0] : fraction[0];
                const double weight = wx * wy * wz;
                if (weight > 0.0)
                    callback(Voxel{x == 0 ? lower[0] : upper[0], y == 0 ? lower[1] : upper[1], z == 0 ? lower[2] : upper[2]}, weight);
            }
        }
    }
}

template <typename Lookup>
ScoringVoxel interpolateScoringPoint(const cv::Vec3d& point, const FiberPredictionGridInfo& grid, Lookup&& lookup)
{
    ScoringVoxel output;
    cv::Matx33d predictionTensor = cv::Matx33d::zeros();
    cv::Matx33d normalTensor = cv::Matx33d::zeros();
    bool predictionValid = true;
    bool normalValid = true;
    bool predictionAxesIdentical = true;
    bool normalAxesIdentical = true;
    std::optional<cv::Vec3d> firstPredictionAxis;
    std::optional<cv::Vec3d> firstNormalAxis;
    double presence = 0.0;
    forEachInterpolationCorner(point, grid, [&](const Voxel& corner, double weight) {
        const auto& sample = lookup(corner);
        const double directionNorm2 = sample.prediction.direction.dot(sample.prediction.direction);
        if (!sample.prediction.valid || !finiteVector(sample.prediction.direction) || !std::isfinite(sample.prediction.presence) ||
            !(directionNorm2 > kEpsilon) || !std::isfinite(directionNorm2)) {
            predictionValid = false;
        } else {
            const cv::Vec3d axis = sample.prediction.direction / std::sqrt(directionNorm2);
            if (!firstPredictionAxis.has_value())
                firstPredictionAxis = axis;
            else if (std::abs(firstPredictionAxis->dot(axis)) < 1.0 - 1.0e-12)
                predictionAxesIdentical = false;
            predictionTensor += fiberAxisTensor(axis, weight);
            presence += weight * sample.prediction.presence;
        }

        const double normalNorm2 = sample.normal.dot(sample.normal);
        if (!sample.normalValid || !finiteVector(sample.normal) || !(normalNorm2 > kEpsilon) || !std::isfinite(normalNorm2)) {
            normalValid = false;
        } else {
            const cv::Vec3d axis = sample.normal / std::sqrt(normalNorm2);
            if (!firstNormalAxis.has_value())
                firstNormalAxis = axis;
            else if (std::abs(firstNormalAxis->dot(axis)) < 1.0 - 1.0e-12)
                normalAxesIdentical = false;
            normalTensor += fiberAxisTensor(axis, weight);
        }
    });
    if (predictionValid && std::isfinite(presence)) {
        const auto principal = principalFiberAxis(predictionTensor);
        if (predictionAxesIdentical && firstPredictionAxis.has_value()) {
            output.prediction.direction = canonicalFiberAxis(*firstPredictionAxis);
            output.prediction.presence = presence;
            output.prediction.presenceValid = true;
            output.prediction.valid = true;
        } else if (principal.unique) {
            output.prediction.direction = principal.axis;
            output.prediction.presence = presence;
            output.prediction.presenceValid = true;
            output.prediction.valid = true;
        }
    }
    if (normalValid) {
        const auto principal = principalFiberAxis(normalTensor);
        if (normalAxesIdentical && firstNormalAxis.has_value()) {
            output.normal = canonicalFiberAxis(*firstNormalAxis);
            output.normalValid = true;
        } else if (principal.unique) {
            output.normal = principal.axis;
            output.normalValid = true;
        }
    }
    return output;
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

bool storedVoxelLess(const Voxel& left, const Voxel& right)
{
    return storedIndex(left) < storedIndex(right);
}

void addInterpolationCorners(
    const cv::Vec3d& point,
    const FiberPredictionGridInfo& grid,
    std::unordered_set<Voxel, VoxelHash>& corners)
{
    forEachInterpolationCorner(point, grid, [&](const Voxel& corner, double) {
        corners.insert(corner);
    });
}

PreparedCandidate prepareCandidate(
    const FiberletCandidateResult& candidate,
    const FiberPredictionGridInfo& grid,
    int cellSize,
    const FiberletPathConfig& config,
    const FiberletPointPredicate& pointPredicate,
    std::unordered_set<Voxel, VoxelHash>& corners)
{
    PreparedCandidate prepared;
    prepared.domain = makeCurvedDomain(candidate, config);
    const SearchCorridor corridor = makeSearchCorridor(prepared.domain, grid, cellSize, config);
    prepared.keyLayout = makeLocalNodeKeyLayout(
        prepared.domain,
        std::sqrt(corridor.radiusSquared),
        config.transverseStepPredictionVoxels);
    prepared.nodes = enumerateLocalNodes(
        prepared.domain,
        corridor,
        config,
        grid,
        pointPredicate,
        prepared.keyLayout);
    addInterpolationCorners(candidate.startPositionPredictionXYZ, grid, corners);
    addInterpolationCorners(candidate.targetPositionPredictionXYZ, grid, corners);
    for (const auto& node : prepared.nodes)
        addInterpolationCorners(nodePoint(node), grid, corners);
    return prepared;
}

std::vector<Voxel> mergeSortedUnique(const std::vector<Voxel>& left, const std::vector<Voxel>& right)
{
    std::vector<Voxel> merged;
    merged.reserve(checkedSum(left.size(), right.size(), "fiberlet corner merge size"));
    std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(merged), storedVoxelLess);
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

size_t preparedPayloadBytes(const std::vector<PreparedCandidate>& prepared)
{
    size_t bytes = checkedProduct(prepared.capacity(), sizeof(PreparedCandidate), "fiberlet prepared byte estimate");
    for (const auto& item : prepared) {
        bytes = checkedSum(
            bytes,
            checkedSum(
                checkedProduct(item.domain.layers.capacity(), sizeof(CurvedLayer), "fiberlet prepared byte estimate"),
                checkedProduct(item.nodes.capacity(), sizeof(SearchNode), "fiberlet prepared byte estimate"),
                "fiberlet prepared byte estimate"),
            "fiberlet prepared byte estimate");
    }
    return bytes;
}

constexpr uint8_t kNodePredictionValid = 1U << 0U;
constexpr uint8_t kNodePresenceValid = 1U << 1U;
constexpr uint8_t kNodeNormalValid = 1U << 2U;

void storeNodeScoring(SearchNode& node, const ScoringVoxel& scoring)
{
    node.flags = 0;
    if (scoring.prediction.presenceValid &&
        std::isfinite(scoring.prediction.presence)) {
        const double presence = std::clamp(scoring.prediction.presence, 0.0, 1.0);
        node.presence = static_cast<uint8_t>(std::lround(presence * 255.0));
        node.flags |= kNodePresenceValid;
    }
    if (scoring.prediction.valid) {
        const auto encoded =
            vc::lasagna::encodeCompactNormalToRaw(scoring.prediction.direction);
        if (encoded.has_value()) {
            node.predictionAxis = *encoded;
            node.flags |= kNodePredictionValid;
        }
    }
    if (scoring.normalValid) {
        const auto encoded =
            vc::lasagna::encodeCompactNormalToRaw(scoring.normal);
        if (encoded.has_value()) {
            node.normalAxis = *encoded;
            node.flags |= kNodeNormalValid;
        }
    }
}

FiberStoredPredictionSample nodePrediction(const SearchNode& node)
{
    FiberStoredPredictionSample prediction;
    prediction.presence = static_cast<double>(node.presence) / 255.0;
    prediction.presenceValid = (node.flags & kNodePresenceValid) != 0;
    prediction.valid = (node.flags & kNodePredictionValid) != 0;
    if (prediction.valid) {
        prediction.direction = vc::lasagna::decodeCompactNormalFromRaw(
            node.predictionAxis[0], node.predictionAxis[1]);
    }
    return prediction;
}

vc::lasagna::NormalSample nodeNormal(const SearchNode& node)
{
    const bool valid = (node.flags & kNodeNormalValid) != 0;
    return {
        valid ? vc::lasagna::decodeCompactNormalFromRaw(
                    node.normalAxis[0], node.normalAxis[1])
              : cv::Vec3d{0.0, 0.0, 0.0},
        valid,
        {},
    };
}

bool usablePrediction(const FiberStoredPredictionSample& prediction)
{
    const double normSquared = prediction.direction.dot(prediction.direction);
    return prediction.valid && finiteVector(prediction.direction) && std::isfinite(prediction.presence) && std::isfinite(normSquared) &&
           normSquared > kEpsilon;
}

bool withinPredictionDeviation(const cv::Vec3d& direction, const FiberStoredPredictionSample& prediction, double maximumDeviationRadians)
{
    if (!usablePrediction(prediction))
        return false;
    const cv::Vec3d unitDirection = normalized(direction);
    const cv::Vec3d unitPrediction = normalized(prediction.direction);
    if (vectorLength(unitDirection) <= kEpsilon || vectorLength(unitPrediction) <= kEpsilon) {
        return false;
    }
    return std::abs(unitDirection.dot(unitPrediction)) > std::cos(maximumDeviationRadians);
}

cv::Vec3f floatVector(const cv::Vec3d& value)
{
    return {
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2]),
    };
}

FiberLocalMetricSample localSample(const FiberStoredPredictionSample& sample)
{
    return {
        floatVector(sample.direction),
        static_cast<float>(sample.presence),
        usablePrediction(sample),
    };
}

FiberletPathCost pathStepCost(
    const FiberStoredPredictionSample* currentPrediction,
    const FiberStoredPredictionSample& candidatePrediction,
    const cv::Vec3d& previousDirection,
    double previousLength,
    const cv::Vec3d& candidateDirection,
    double candidateLength,
    const vc::lasagna::NormalSample& normal,
    const FiberletPathConfig& config)
{
    const auto current = currentPrediction != nullptr ? std::make_optional(localSample(*currentPrediction)) : std::nullopt;
    const auto local = fiberLocalMetricCost(
        current.has_value() ? &*current : nullptr,
        localSample(candidatePrediction),
        floatVector(previousDirection),
        static_cast<float>(previousLength),
        floatVector(candidateDirection),
        static_cast<float>(candidateLength),
        floatVector(normal.normal),
        normal.valid,
        FiberLocalMetricConfig{
            static_cast<float>(config.invalidPredictionCostPerVoxel),
            FiberLocalSmoothnessConfig{
                static_cast<float>(config.smoothnessWeight),
                static_cast<float>(config.smoothnessNormalWeight),
                static_cast<float>(config.smoothnessTangentWeight),
                static_cast<float>(config.smoothnessFreeAngleDegrees * kPi / 180.0)}});
    FiberletPathCost cost;
    cost.invalidPrediction = local.invalidPrediction;
    cost.alignment = local.alignment;
    cost.isotropicSmoothness = local.isotropicSmoothness;
    cost.tangentSmoothness = local.tangentSmoothness;
    cost.normalSmoothness = local.normalSmoothness;
    return cost;
}

bool betterCost(double candidate, double current)
{
    return candidate < current;
}

FiberletCandidateResult solveCandidate(FiberletCandidateResult candidate, const FiberletPathConfig& config, const PreparedCandidate& prepared)
{
    candidate.searched = true;
    const cv::Vec3d chordVector = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
    const double chordLength = vectorLength(chordVector);
    if (!(chordLength > kEpsilon)) {
        candidate.reason = "zero_length";
        return candidate;
    }
    const double maximumAngle = config.maximumEndpointAngleDegrees * kPi / 180.0;
    const double maximumPredictionDeviation = config.maximumPredictionDeviationDegrees * kPi / 180.0;
    const auto& domain = prepared.domain;
    const auto& nodes = prepared.nodes;
    const auto& startScoring = prepared.startScoring;
    const auto& targetScoring = prepared.targetScoring;
    candidate.startPrediction = startScoring.prediction;
    candidate.targetPrediction = targetScoring.prediction;
    candidate.startNormalXYZ = startScoring.normal;
    candidate.targetNormalXYZ = targetScoring.normal;
    candidate.startNormalValid = startScoring.normalValid;
    candidate.targetNormalValid = targetScoring.normalValid;

    FiberStoredPredictionSample targetProxy;
    targetProxy.direction = candidate.targetAxisXYZ;
    targetProxy.presence = 1.0;
    targetProxy.valid = true;
    targetProxy.presenceValid = true;

    if (domain.layers.size() == 2) {
        const cv::Vec3d direction = chordVector / chordLength;
        if (directedAngle(candidate.startAxisXYZ, direction) > maximumAngle + kEpsilon ||
            directedAngle(direction, candidate.targetAxisXYZ) > maximumAngle + kEpsilon ||
            !withinPredictionDeviation(direction, targetScoring.prediction, maximumPredictionDeviation)) {
            candidate.reason = "no_path";
            return candidate;
        }
        const vc::lasagna::NormalSample targetNormal{targetScoring.normal, targetScoring.normalValid, targetScoring.normalValid ? std::string{} : "invalid"};
        candidate.cost =
            pathStepCost(nullptr, targetScoring.prediction, candidate.startAxisXYZ, chordLength, direction, chordLength, targetNormal, config);
        candidate.cost +=
            pathStepCost(&targetScoring.prediction, targetProxy, direction, chordLength, candidate.targetAxisXYZ, 0.0, targetNormal, config);
        candidate.pointsPredictionXYZ = {candidate.startPositionPredictionXYZ, candidate.targetPositionPredictionXYZ};
        candidate.scoreValid = true;
        candidate.success = true;
        candidate.reason = "success";
        return candidate;
    }
    if (nodes.empty()) {
        candidate.reason = "empty_corridor";
        return candidate;
    }
    std::unordered_map<uint32_t, size_t> nodeIndex;
    nodeIndex.reserve(checkedProduct(nodes.size(), 2, "fiberlet DP node hash capacity"));
    for (size_t index = 0; index < nodes.size(); ++index)
        nodeIndex.emplace(nodes[index].key, index);

    constexpr size_t transitionStateCount = 9;
    constexpr size_t sourceState = transitionStateCount;
    constexpr size_t stateCount = transitionStateCount + 1;
    std::vector<DpState> states(checkedProduct(nodes.size(), stateCount, "fiberlet DP state count"));
    for (size_t node = 0; node < nodes.size(); ++node) {
        const LocalNodeKey key =
            unpackLocalNodeKey(nodes[node].key, prepared.keyLayout);
        if (key.layer != 1)
            continue;
        const cv::Vec3d point = nodePoint(nodes[node]);
        const cv::Vec3d delta = point - candidate.startPositionPredictionXYZ;
        const double stepLength = vectorLength(delta);
        if (!(stepLength > kEpsilon))
            continue;
        const cv::Vec3d direction = delta / stepLength;
        if (directedAngle(candidate.startAxisXYZ, direction) > maximumAngle + kEpsilon ||
            !withinPredictionDeviation(
                direction, nodePrediction(nodes[node]), maximumPredictionDeviation)) {
            continue;
        }
        const FiberStoredPredictionSample prediction = nodePrediction(nodes[node]);
        const vc::lasagna::NormalSample normal = nodeNormal(nodes[node]);
        auto& state = states[node * stateCount + sourceState];
        state.reached = true;
        state.cost = pathStepCost(
            nullptr, prediction, candidate.startAxisXYZ, stepLength,
            direction, stepLength, normal, config);
        state.previous = {-1, -1};
        state.incomingDirection = direction;
        state.incomingLength = stepLength;
    }

    for (size_t node = 0; node < nodes.size(); ++node) {
        const LocalNodeKey currentKey =
            unpackLocalNodeKey(nodes[node].key, prepared.keyLayout);
        const cv::Vec3d currentPoint = nodePoint(nodes[node]);
        const FiberStoredPredictionSample currentPrediction =
            nodePrediction(nodes[node]);
        for (size_t previousState = 0; previousState < stateCount; ++previousState) {
            const auto& currentState = states[node * stateCount + previousState];
            if (!currentState.reached)
                continue;
            for (int deltaU = -1; deltaU <= 1; ++deltaU) {
                for (int deltaV = -1; deltaV <= 1; ++deltaV) {
                    const LocalNodeKey nextKey{
                        currentKey.layer + 1,
                        currentKey.transverseU + deltaU,
                        currentKey.transverseV + deltaV};
                    if (nextKey.transverseU < -prepared.keyLayout.transverseLimit ||
                        nextKey.transverseU > prepared.keyLayout.transverseLimit ||
                        nextKey.transverseV < -prepared.keyLayout.transverseLimit ||
                        nextKey.transverseV > prepared.keyLayout.transverseLimit) {
                        continue;
                    }
                    const auto found = nodeIndex.find(
                        packLocalNodeKey(nextKey, prepared.keyLayout));
                    if (found == nodeIndex.end())
                        continue;
                    const size_t next = found->second;
                    const cv::Vec3d delta = nodePoint(nodes[next]) - currentPoint;
                    const double stepLength = vectorLength(delta);
                    if (!(stepLength > kEpsilon))
                        continue;
                    const cv::Vec3d direction = delta / stepLength;
                    const FiberStoredPredictionSample nextPrediction =
                        nodePrediction(nodes[next]);
                    if (!withinPredictionDeviation(
                            direction, nextPrediction, maximumPredictionDeviation)) {
                        continue;
                    }
                    const vc::lasagna::NormalSample nextNormal =
                        nodeNormal(nodes[next]);
                    FiberletPathCost nextCost = currentState.cost;
                    nextCost += pathStepCost(
                        &currentPrediction,
                        nextPrediction,
                        currentState.incomingDirection,
                        currentState.incomingLength,
                        direction,
                        stepLength,
                        nextNormal,
                        config);
                    const size_t transitionState = static_cast<size_t>((deltaU + 1) * 3 + (deltaV + 1));
                    auto& destination = states[next * stateCount + transitionState];
                    if (!destination.reached || betterCost(nextCost.total(), destination.cost.total())) {
                        destination.reached = true;
                        destination.cost = nextCost;
                        destination.previous = {static_cast<int64_t>(node), static_cast<int>(previousState)};
                        destination.incomingDirection = direction;
                        destination.incomingLength = stepLength;
                    }
                }
            }
        }
    }

    bool foundPath = false;
    size_t bestNode = 0;
    size_t bestState = 0;
    FiberletPathCost bestCost;
    const size_t finalInteriorLayer = domain.layers.size() - 2;
    for (size_t node = 0; node < nodes.size(); ++node) {
        const LocalNodeKey key =
            unpackLocalNodeKey(nodes[node].key, prepared.keyLayout);
        if (key.layer != finalInteriorLayer)
            continue;
        const cv::Vec3d point = nodePoint(nodes[node]);
        const FiberStoredPredictionSample prediction = nodePrediction(nodes[node]);
        const vc::lasagna::NormalSample normal = nodeNormal(nodes[node]);
        for (size_t stateIndex = 0; stateIndex < stateCount; ++stateIndex) {
            const auto& state = states[node * stateCount + stateIndex];
            if (!state.reached)
                continue;
            const cv::Vec3d delta = candidate.targetPositionPredictionXYZ - point;
            const double finalLength = vectorLength(delta);
            if (!(finalLength > kEpsilon))
                continue;
            const cv::Vec3d finalDirection = delta / finalLength;
            if (directedAngle(finalDirection, candidate.targetAxisXYZ) > maximumAngle + kEpsilon) {
                continue;
            }
            FiberletPathCost finalized = state.cost;
            finalized += pathStepCost(
                &prediction,
                targetProxy,
                state.incomingDirection,
                state.incomingLength,
                finalDirection,
                finalLength,
                normal,
                config);
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
        reversed.push_back(nodePoint(nodes[node]));
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
    output << "fiberlet_" << candidate.start.cellZYX[0] << '_' << candidate.start.cellZYX[1] << '_' << candidate.start.cellZYX[2] << '_'
           << candidate.start.componentIndex << "__" << candidate.target.cellZYX[0] << '_' << candidate.target.cellZYX[1] << '_'
           << candidate.target.cellZYX[2] << '_' << candidate.target.componentIndex;
    return output.str();
}

double fiberletPathLength(const FiberletCandidateResult& candidate)
{
    if (candidate.pointsPredictionXYZ.size() < 2)
        throw std::runtime_error("successful fiberlet has fewer than two path points");
    double length = 0.0;
    for (size_t index = 1; index < candidate.pointsPredictionXYZ.size(); ++index) {
        if (!finiteVector(candidate.pointsPredictionXYZ[index - 1]) || !finiteVector(candidate.pointsPredictionXYZ[index])) {
            throw std::runtime_error("successful fiberlet has a non-finite path point");
        }
        const double segment = vectorLength(candidate.pointsPredictionXYZ[index] - candidate.pointsPredictionXYZ[index - 1]);
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
    if (!(report.grid.predictionToBaseScale > 0.0) || !std::isfinite(report.grid.predictionToBaseScale)) {
        throw std::runtime_error("fiberlet visualization requires a positive prediction-to-base scale");
    }

    FiberletPathVisualReport visual;
    std::set<std::string> identifiers;
    for (size_t candidateIndex = 0; candidateIndex < report.candidates.size(); ++candidateIndex) {
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
        if (std::any_of(componentLosses.begin(), componentLosses.end(), [](double value) { return !(value >= 0.0) || !std::isfinite(value); })) {
            throw std::runtime_error("successful fiberlet has an invalid component loss");
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
            visual.minimumLossPerPredictionVoxel.has_value() ? std::min(*visual.minimumLossPerPredictionVoxel, density) : density;
        visual.maximumLossPerPredictionVoxel =
            visual.maximumLossPerPredictionVoxel.has_value() ? std::max(*visual.maximumLossPerPredictionVoxel, density) : density;
    }
    for (auto& path : visual.paths) {
        path.relativeQuality = *visual.minimumLossPerPredictionVoxel == *visual.maximumLossPerPredictionVoxel
                                   ? 1.0
                                   : (*visual.maximumLossPerPredictionVoxel - path.lossPerPredictionVoxel) /
                                         (*visual.maximumLossPerPredictionVoxel - *visual.minimumLossPerPredictionVoxel);
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
    for (size_t candidateIndex = 0; candidateIndex < report.candidates.size(); ++candidateIndex) {
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
        statistics.acceptedLossDensities.minimum = statistics.acceptedLossDensities.minimum.has_value()
                                                       ? std::min(*statistics.acceptedLossDensities.minimum, path.lossPerPredictionVoxel)
                                                       : path.lossPerPredictionVoxel;
        statistics.acceptedLossDensities.maximum = statistics.acceptedLossDensities.maximum.has_value()
                                                       ? std::max(*statistics.acceptedLossDensities.maximum, path.lossPerPredictionVoxel)
                                                       : path.lossPerPredictionVoxel;
    }
    if (statistics.acceptedLossDensities.count > 0) {
        statistics.acceptedLossDensities.mean = densitySum / static_cast<double>(statistics.acceptedLossDensities.count);
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
    if (!(config.neighborhoodMarginCells > 0.0) || !std::isfinite(config.neighborhoodMarginCells)) {
        throw std::invalid_argument("fiberlet neighborhood margin must be positive and finite");
    }
    if (!(config.longitudinalStepPredictionVoxels > 0.0) || !std::isfinite(config.longitudinalStepPredictionVoxels) ||
        !(config.transverseStepPredictionVoxels > 0.0) || !std::isfinite(config.transverseStepPredictionVoxels)) {
        throw std::invalid_argument("fiberlet local-grid steps must be positive and finite");
    }
    if (!finiteNonnegative(config.maximumEndpointAngleDegrees) || config.maximumEndpointAngleDegrees > 90.0) {
        throw std::invalid_argument("fiberlet endpoint angle must be in [0, 90]");
    }
    if (!finiteNonnegative(config.maximumPredictionDeviationDegrees) || config.maximumPredictionDeviationDegrees > 90.0) {
        throw std::invalid_argument("fiberlet prediction deviation must be in [0, 90]");
    }
    if (!finiteNonnegative(config.corridorRadiusPredictionVoxels))
        throw std::invalid_argument("fiberlet corridor radius must be non-negative");
    if (!finiteNonnegative(config.invalidPredictionCostPerVoxel) || !finiteNonnegative(config.smoothnessWeight) ||
        !finiteNonnegative(config.smoothnessNormalWeight) || !finiteNonnegative(config.smoothnessTangentWeight) ||
        !finiteNonnegative(config.smoothnessFreeAngleDegrees)) {
        throw std::invalid_argument("fiberlet objective weights and angles must be finite and non-negative");
    }
    if (config.samplingBatchCoordinates < 1)
        throw std::invalid_argument("fiberlet sampling batch size must be positive");
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
    if (selection.contains("cells_zyx")) {
        const auto& cells = selection.at("cells_zyx");
        if (!cells.is_array() || cells.empty())
            throw std::runtime_error("fiber anchor explicit cells must be a non-empty array");
        for (const auto& cell : cells)
            loaded.report.selectedCellsZYX.push_back(jsonSize3(cell, "cells_zyx"));
        if (!std::is_sorted(loaded.report.selectedCellsZYX.begin(), loaded.report.selectedCellsZYX.end()) ||
            std::adjacent_find(loaded.report.selectedCellsZYX.begin(), loaded.report.selectedCellsZYX.end()) !=
                loaded.report.selectedCellsZYX.end()) {
            throw std::runtime_error("fiber anchor explicit cells must be strictly ordered");
        }
    }
    const auto& parameters = root.at("parameters");
    const std::set<std::string> parameterKeys{
        "cell_size_prediction_voxels",
        "gaussian_sigma_prediction_voxels",
        "peak_sigma_prediction_voxels",
        "peak_axial_sigma_prediction_voxels",
        "peak_grid_step_prediction_voxels",
        "peak_gradient_weight",
        "peak_gradient_reliability_scale",
        "gaussian_cutoff_sigmas",
        "local_window_radius_prediction_voxels",
        "axial_support_half_width_prediction_voxels",
        "position_convergence_tolerance_prediction_voxels",
        "nms_maximum_angle_degrees",
        "nms_transverse_radius_prediction_voxels",
        "nms_longitudinal_radius_prediction_voxels",
        "observation_presence_floor",
        "minimum_aligned_support",
        "merge_maximum_angle_degrees",
        "merge_maximum_absolute_objective_loss",
        "merge_maximum_relative_objective_loss",
        "maximum_seed_count",
        "maximum_iterations",
        "convergence_tolerance",
    };
    if (!parameters.is_object())
        throw std::runtime_error("fiber anchor parameters must be an object");
    std::set<std::string> storedParameterKeys;
    for (const auto& item : parameters.items())
        storedParameterKeys.insert(item.key());
    if (storedParameterKeys != parameterKeys)
        throw std::runtime_error("fiber anchor parameters do not match the version-1 schema");
    auto& config = loaded.report.config;
    config.cellSizePredictionVoxels = parameters.at("cell_size_prediction_voxels").get<int>();
    config.gaussianSigmaPredictionVoxels =
        finiteNumber(parameters.at("gaussian_sigma_prediction_voxels"), "gaussian_sigma_prediction_voxels");
    config.peakSigmaPredictionVoxels = finiteNumber(parameters.at("peak_sigma_prediction_voxels"), "peak_sigma_prediction_voxels");
    config.peakAxialSigmaPredictionVoxels =
        finiteNumber(parameters.at("peak_axial_sigma_prediction_voxels"), "peak_axial_sigma_prediction_voxels");
    config.peakGridStepPredictionVoxels =
        finiteNumber(parameters.at("peak_grid_step_prediction_voxels"), "peak_grid_step_prediction_voxels");
    config.peakGradientWeight = finiteNumber(parameters.at("peak_gradient_weight"), "peak_gradient_weight");
    config.peakGradientReliabilityScale = finiteNumber(parameters.at("peak_gradient_reliability_scale"), "peak_gradient_reliability_scale");
    config.gaussianCutoffSigmas = finiteNumber(parameters.at("gaussian_cutoff_sigmas"), "gaussian_cutoff_sigmas");
    config.localWindowRadiusPredictionVoxels =
        finiteNumber(parameters.at("local_window_radius_prediction_voxels"), "local_window_radius_prediction_voxels");
    config.axialSupportHalfWidthPredictionVoxels =
        finiteNumber(parameters.at("axial_support_half_width_prediction_voxels"), "axial_support_half_width_prediction_voxels");
    config.positionConvergenceTolerancePredictionVoxels =
        finiteNumber(parameters.at("position_convergence_tolerance_prediction_voxels"), "position_convergence_tolerance_prediction_voxels");
    config.nmsMaximumAngleDegrees = finiteNumber(parameters.at("nms_maximum_angle_degrees"), "nms_maximum_angle_degrees");
    config.nmsTransverseRadiusPredictionVoxels =
        finiteNumber(parameters.at("nms_transverse_radius_prediction_voxels"), "nms_transverse_radius_prediction_voxels");
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
    size_t selectedCellCount = loaded.report.selectedCellsZYX.empty() ? 1 : loaded.report.selectedCellsZYX.size();
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t totalCells = (loaded.report.grid.shapeZYX[axis] + static_cast<size_t>(config.cellSizePredictionVoxels) - 1) /
                                  static_cast<size_t>(config.cellSizePredictionVoxels);
        if (loaded.report.selectedCellBeginZYX[axis] >= loaded.report.selectedCellEndZYX[axis] || loaded.report.selectedCellEndZYX[axis] > totalCells) {
            throw std::runtime_error("fiber anchor selected cell range is invalid");
        }
        if (loaded.report.selectedCellsZYX.empty()) {
            const size_t extent = loaded.report.selectedCellEndZYX[axis] - loaded.report.selectedCellBeginZYX[axis];
            if (selectedCellCount > std::numeric_limits<size_t>::max() / extent)
                throw std::runtime_error("fiber anchor selected cell count overflows");
            selectedCellCount *= extent;
        }
    }
    const std::set<std::array<size_t, 3>> explicitCells(loaded.report.selectedCellsZYX.begin(), loaded.report.selectedCellsZYX.end());
    for (const auto& cell : loaded.report.selectedCellsZYX) {
        for (size_t axis = 0; axis < 3; ++axis) {
            if (cell[axis] < loaded.report.selectedCellBeginZYX[axis] || cell[axis] >= loaded.report.selectedCellEndZYX[axis]) {
                throw std::runtime_error("fiber anchor explicit cell lies outside its bounds");
            }
        }
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
    loaded.report.diagnostics.outsideSelectionComponents = diagnostics.value("outside_selection_components", size_t{0});

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
            if (cell.cellZYX[axis] < loaded.report.selectedCellBeginZYX[axis] || cell.cellZYX[axis] >= loaded.report.selectedCellEndZYX[axis]) {
                throw std::runtime_error("fiber anchor cell lies outside the selected cell range");
            }
        }
        if (!explicitCells.empty() && !explicitCells.contains(cell.cellZYX))
            throw std::runtime_error("fiber anchor cell is not in the explicit selection");
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
                const std::set<std::string>
                    validReasons{"empty", "degenerate", "below_support", "merged_same_direction", "nms_suppressed", "outside_selection"};
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
            component.anchor.refinementScore = finiteNumber(componentJson.at("refinement_score"), "refinement_score");
            component.anchor.refinementIterations = componentJson.at("refinement_iterations").get<size_t>();
            if (component.anchor.alignedSupport < 0.0 || component.anchor.alignedSupport > 1.0 || component.anchor.directionalCoherence < 0.0 ||
                component.anchor.directionalCoherence > 1.0 || component.anchor.refinementScore < 0.0 || component.anchor.refinementScore > 1.0 ||
                std::abs(component.anchor.refinementScore - component.anchor.alignedSupport) > 1.0e-12 ||
                component.anchor.refinementIterations > static_cast<size_t>(config.maximumIterations)) {
                throw std::runtime_error("fiber anchor refinement values are inconsistent");
            }
            const cv::Vec3d pivot{
                (static_cast<double>(cell.cellZYX[2] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<double>(
                     std::min((cell.cellZYX[2] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[2])) -
                 1.0) *
                    0.5,
                (static_cast<double>(cell.cellZYX[1] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<double>(
                     std::min((cell.cellZYX[1] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[1])) -
                 1.0) *
                    0.5,
                (static_cast<double>(cell.cellZYX[0] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<double>(
                     std::min((cell.cellZYX[0] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[0])) -
                 1.0) *
                    0.5,
            };
            const cv::Vec3d pivotOffset = component.anchor.positionPredictionXYZ - pivot;
            const double planeResidual = std::abs(pivotOffset.dot(component.anchor.axisXYZ));
            const double pivotDistance = vectorLength(pivotOffset);
            for (int axis = 0; axis < 3; ++axis) {
                const double position = component.anchor.positionPredictionXYZ[axis];
                const size_t gridAxis = static_cast<size_t>(2 - axis);
                if (position < -kEpsilon || position > static_cast<double>(loaded.report.grid.shapeZYX[gridAxis] - 1) + kEpsilon) {
                    throw std::runtime_error("fiber anchor position is outside the prediction grid");
                }
                const double cellBegin = static_cast<double>(cell.cellZYX[gridAxis] * static_cast<size_t>(config.cellSizePredictionVoxels));
                const double cellEnd = static_cast<double>(
                    std::min((cell.cellZYX[gridAxis] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[gridAxis]));
                const double ownerLower = cellBegin == 0.0 ? 0.0 : cellBegin - 0.5;
                const double ownerUpper = cellEnd == static_cast<double>(loaded.report.grid.shapeZYX[gridAxis])
                                              ? cellEnd - 1.0
                                              : std::nextafter(cellEnd - 0.5, -std::numeric_limits<double>::infinity());
                if (position < ownerLower - kEpsilon || position > ownerUpper + kEpsilon) {
                    throw std::runtime_error("fiber anchor position is outside its owning cell");
                }
            }
            if (planeResidual > 1.0e-6 || pivotDistance > config.localWindowRadiusPredictionVoxels + 1.0e-6) {
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
    if (storedNmsSuppressedComponents > loaded.report.diagnostics.nmsSuppressedComponents) {
        throw std::runtime_error("fiber anchor NMS diagnostics are inconsistent");
    }
    if (loaded.report.diagnostics.zeroAnchorCells + loaded.report.diagnostics.oneAnchorCells + loaded.report.diagnostics.twoAnchorCells !=
        loaded.report.diagnostics.totalCells) {
        throw std::runtime_error("fiber anchor cell-count diagnostics are inconsistent");
    }
    if (loaded.report.diagnostics.totalCells != selectedCellCount)
        throw std::runtime_error("fiber anchor total cells disagree with the selected lattice");
    if (loaded.report.nonEmptyCells.size() != loaded.report.diagnostics.oneAnchorCells + loaded.report.diagnostics.twoAnchorCells) {
        throw std::runtime_error("fiber anchor stored cells disagree with diagnostics");
    }
    return loaded;
}

std::vector<std::array<int, 3>> fiberletCellNeighborhoodOffsets(int radius, double margin)
{
    if (radius < 1 || !(margin > 0.0) || !std::isfinite(margin))
        throw std::invalid_argument("fiberlet cell neighborhood requires positive radius and margin");
    const int limit = static_cast<int>(std::ceil(radius + margin));
    const double upper = static_cast<double>(radius) + margin;
    std::vector<std::array<int, 3>> offsets;
    for (int z = -limit; z <= limit; ++z) {
        for (int y = -limit; y <= limit; ++y) {
            for (int x = -limit; x <= limit; ++x) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const double length = std::sqrt(static_cast<double>(x * x + y * y + z * z));
                if (length < upper)
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
    const FiberletPathProgressCallback& progressCallback,
    const FiberletPointPredicate& pointPredicate)
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
    const double startCpuSeconds = processCpuSeconds();
    const auto flat = flattenAnchors(anchors.report);
    report.diagnostics.occupiedAnchors = flat.size();
    const auto offsets = fiberletCellNeighborhoodOffsets(report.config.cellRadius, report.config.neighborhoodMarginCells);
    report.diagnostics.neighborhoodOffsets = offsets.size();
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
                ++report.diagnostics.neighborhoodTargetsOutOfGrid;
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
                if (pointPredicate && (!pointPredicate(candidate.startPositionPredictionXYZ) || !pointPredicate(candidate.targetPositionPredictionXYZ))) {
                    candidate.reason = "outside_selection";
                    report.candidates.push_back(std::move(candidate));
                    continue;
                }
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
    report.candidateGenerationCpuSeconds = processCpuSeconds() - startCpuSeconds;

    std::mutex progressMutex;
    std::exception_ptr progressError;
    std::string progressPhase;
    size_t lastReportedCompleted = 0;
    auto lastProgressTime = candidateGenerationEnd;
    const auto reportProgress = [&](const char* phase, size_t completed, size_t total, const Clock::time_point& phaseStart, bool terminal) noexcept {
        if (!progressCallback)
            return;
        std::lock_guard lock(progressMutex);
        if (progressError)
            return;
        const auto now = Clock::now();
        const bool phaseChanged = progressPhase != phase;
        if (phaseChanged) {
            progressPhase = phase;
            lastReportedCompleted = 0;
            lastProgressTime = phaseStart;
        }
        if (!phaseChanged) {
            if (!terminal) {
                if (completed <= lastReportedCompleted || completed >= total || now - lastProgressTime < std::chrono::seconds(1)) {
                    return;
                }
            } else if (total > 0 && completed == lastReportedCompleted) {
                return;
            }
        }
        try {
            progressCallback({
                phase,
                completed,
                total,
                std::chrono::duration<double>(now - phaseStart).count(),
            });
            lastReportedCompleted = completed;
            lastProgressTime = now;
        } catch (...) {
            progressError = std::current_exception();
        }
    };
    const size_t workerCount = std::min(searchCandidateIndices.size(), static_cast<size_t>(report.config.parallelThreads));
    report.candidateWorkers = workerCount;
    std::vector<PreparedCandidate> prepared(searchCandidateIndices.size());
    std::vector<std::unordered_set<Voxel, VoxelHash>> workerCorners(workerCount);
    std::vector<std::exception_ptr> errors(searchCandidateIndices.size());

    const auto preparationStart = Clock::now();
    const double preparationCpuStart = processCpuSeconds();
    reportProgress("preparation", 0, searchCandidateIndices.size(), preparationStart, true);
    std::atomic<size_t> nextPreparation{0};
    std::atomic<size_t> completedPreparation{0};
    const auto preparationWorker = [&](size_t workerIndex) {
        auto& corners = workerCorners[workerIndex];
        while (true) {
            const size_t searchIndex = nextPreparation.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= searchCandidateIndices.size())
                return;
            try {
                prepared[searchIndex] = prepareCandidate(
                    report.candidates[searchCandidateIndices[searchIndex]], grid, report.anchorCellSizePredictionVoxels, report.config, pointPredicate, corners);
            } catch (...) {
                errors[searchIndex] = std::current_exception();
            }
            const size_t completed = completedPreparation.fetch_add(1, std::memory_order_relaxed) + 1;
            reportProgress("preparation", completed, searchCandidateIndices.size(), preparationStart, false);
        }
    };
    if (workerCount == 1) {
        preparationWorker(0);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(preparationWorker, index);
        for (auto& thread : workers)
            thread.join();
    }
    reportProgress("preparation", searchCandidateIndices.size(), searchCandidateIndices.size(), preparationStart, true);
    report.preparationSeconds = std::chrono::duration<double>(Clock::now() - preparationStart).count();
    report.preparationCpuSeconds = processCpuSeconds() - preparationCpuStart;
    for (const auto& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
    report.preparedCandidates = prepared.size();
    for (const auto& item : prepared) {
        report.evaluatedDpNodes = checkedSum(
            report.evaluatedDpNodes, checkedSum(item.nodes.size(), 2, "fiberlet evaluated node count"), "fiberlet evaluated node count");
    }
    size_t preparedBytes = preparedPayloadBytes(prepared);
    report.preparedGeometryBytes = preparedBytes;
    size_t maximumSearchTransientBytes = 0;
    constexpr size_t stateCount = 10;
    for (const auto& item : prepared) {
        const size_t stateBytes = checkedProduct(
            checkedProduct(item.nodes.size(), stateCount,
                "fiberlet DP state byte estimate"),
            sizeof(DpState), "fiberlet DP state byte estimate");
        const size_t nodeIndexBytes = checkedProduct(
            checkedProduct(item.nodes.size(), 2,
                "fiberlet DP index byte estimate"),
            sizeof(uint32_t) + sizeof(size_t),
            "fiberlet DP index byte estimate");
        maximumSearchTransientBytes = std::max(
            maximumSearchTransientBytes,
            checkedSum(stateBytes, nodeIndexBytes,
                "fiberlet search transient byte estimate"));
    }
    report.peakSearchTransientBytes = checkedProduct(
        maximumSearchTransientBytes, workerCount,
        "fiberlet concurrent search transient byte estimate");
    size_t workerCornerBytes = 0;
    for (const auto& corners : workerCorners) {
        workerCornerBytes = checkedSum(
            workerCornerBytes,
            checkedProduct(corners.size(), sizeof(Voxel), "fiberlet corner byte estimate"),
            "fiberlet corner byte estimate");
    }
    report.estimatedPeakOwnedBytes = checkedSum(preparedBytes, workerCornerBytes, "fiberlet peak owned byte estimate");
    report.estimatedPeakOwnedBytes = std::max(
        report.estimatedPeakOwnedBytes,
        checkedSum(preparedBytes, report.peakSearchTransientBytes,
            "fiberlet peak search byte estimate"));

    const auto cornerMergeStart = Clock::now();
    const double cornerMergeCpuStart = processCpuSeconds();
    reportProgress("corner_merge", 0, workerCorners.size(), cornerMergeStart, true);
    std::vector<std::vector<Voxel>> cornerVectors(workerCorners.size());
    for (size_t index = 0; index < workerCorners.size(); ++index) {
        cornerVectors[index].assign(workerCorners[index].begin(), workerCorners[index].end());
        std::sort(cornerVectors[index].begin(), cornerVectors[index].end(), storedVoxelLess);
    }
    size_t sortedCornerBytes = 0;
    for (const auto& corners : cornerVectors) {
        sortedCornerBytes = checkedSum(
            sortedCornerBytes,
            checkedProduct(corners.capacity(), sizeof(Voxel), "fiberlet sorted corner byte estimate"),
            "fiberlet sorted corner byte estimate");
    }
    report.estimatedPeakOwnedBytes = std::
        max(report.estimatedPeakOwnedBytes,
            checkedSum(
                checkedSum(preparedBytes, workerCornerBytes, "fiberlet peak owned byte estimate"),
                sortedCornerBytes,
                "fiberlet peak owned byte estimate"));
    workerCorners.clear();
    workerCorners.shrink_to_fit();
    while (cornerVectors.size() > 1) {
        const size_t pairCount = cornerVectors.size() / 2;
        std::vector<std::vector<Voxel>> next((cornerVectors.size() + 1) / 2);
        std::atomic<size_t> nextPair{0};
        std::vector<std::exception_ptr> mergeErrors(pairCount);
        const auto mergeWorker = [&]() {
            while (true) {
                const size_t pair = nextPair.fetch_add(1, std::memory_order_relaxed);
                if (pair >= pairCount)
                    return;
                try {
                    next[pair] = mergeSortedUnique(cornerVectors[pair * 2], cornerVectors[pair * 2 + 1]);
                } catch (...) {
                    mergeErrors[pair] = std::current_exception();
                }
            }
        };
        const size_t mergeWorkers = std::min(pairCount, static_cast<size_t>(report.config.parallelThreads));
        if (mergeWorkers == 1) {
            mergeWorker();
        } else {
            std::vector<std::thread> workers;
            workers.reserve(mergeWorkers);
            for (size_t index = 0; index < mergeWorkers; ++index)
                workers.emplace_back(mergeWorker);
            for (auto& thread : workers)
                thread.join();
        }
        for (const auto& error : mergeErrors) {
            if (error)
                std::rethrow_exception(error);
        }
        if (cornerVectors.size() % 2 != 0)
            next.back() = std::move(cornerVectors.back());
        cornerVectors = std::move(next);
    }
    std::vector<Voxel> orderedVoxels;
    if (!cornerVectors.empty())
        orderedVoxels = std::move(cornerVectors.front());
    reportProgress("corner_merge", workerCount, workerCount, cornerMergeStart, true);
    report.cornerMergeSeconds = std::chrono::duration<double>(Clock::now() - cornerMergeStart).count();
    report.cornerMergeCpuSeconds = processCpuSeconds() - cornerMergeCpuStart;
    report.sampledVoxels = orderedVoxels.size();

    const size_t coordinateBatchSize = static_cast<size_t>(report.config.samplingBatchCoordinates);
    report.peakCoordinateBatchVoxels = std::min(coordinateBatchSize, orderedVoxels.size());
    report.samplingCoordinateBatches = orderedVoxels.empty() ? 0 : (orderedVoxels.size() - 1) / coordinateBatchSize + 1;
    std::vector<ScoringVoxel> scoringVoxels(orderedVoxels.size());
    const size_t sampledArrayBytes = checkedSum(
        checkedProduct(orderedVoxels.capacity(), sizeof(Voxel), "fiberlet sampled byte estimate"),
        checkedProduct(scoringVoxels.capacity(), sizeof(ScoringVoxel), "fiberlet sampled byte estimate"),
        "fiberlet sampled byte estimate");
    report.estimatedPeakOwnedBytes =
        std::max(report.estimatedPeakOwnedBytes, checkedSum(preparedBytes, sampledArrayBytes, "fiberlet peak owned byte estimate"));
    const auto predictionStart = Clock::now();
    const double predictionCpuStart = processCpuSeconds();
    reportProgress("prediction_sampling", 0, orderedVoxels.size(), predictionStart, true);
    for (size_t begin = 0; begin < orderedVoxels.size(); begin += coordinateBatchSize) {
        const size_t end = std::min(orderedVoxels.size(), begin + coordinateBatchSize);
        std::vector<std::array<size_t, 3>> indices;
        indices.reserve(end - begin);
        for (size_t index = begin; index < end; ++index)
            indices.push_back(storedIndex(orderedVoxels[index]));
        std::vector<FiberStoredPredictionSample> samples;
        predictionSampler(indices, report.config.parallelThreads, samples);
        if (samples.size() != indices.size())
            throw std::runtime_error("fiberlet prediction sampler returned the wrong coordinate batch sample count");
        for (size_t index = 0; index < samples.size(); ++index)
            scoringVoxels[begin + index].prediction = samples[index];
        ++report.predictionSamplingCalls;
        reportProgress("prediction_sampling", end, orderedVoxels.size(), predictionStart, false);
    }
    reportProgress("prediction_sampling", orderedVoxels.size(), orderedVoxels.size(), predictionStart, true);
    report.predictionSamplingSeconds = std::chrono::duration<double>(Clock::now() - predictionStart).count();
    report.predictionSamplingCpuSeconds = processCpuSeconds() - predictionCpuStart;

    const auto normalStart = Clock::now();
    const double normalCpuStart = processCpuSeconds();
    reportProgress("normal_sampling", 0, orderedVoxels.size(), normalStart, true);
    for (size_t begin = 0; begin < orderedVoxels.size(); begin += coordinateBatchSize) {
        const size_t end = std::min(orderedVoxels.size(), begin + coordinateBatchSize);
        std::vector<cv::Vec3d> points;
        points.reserve(end - begin);
        for (size_t index = begin; index < end; ++index)
            points.push_back(nativeVoxelPoint(orderedVoxels[index]));
        std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
        (void)normalSampler.sampleNormalBatch(points, false, report.config.parallelThreads, samples);
        if (samples.size() != points.size())
            throw std::runtime_error("fiberlet normal sampler returned the wrong coordinate batch sample count");
        for (size_t index = 0; index < samples.size(); ++index) {
            scoringVoxels[begin + index].normal = samples[index].sample.normal;
            scoringVoxels[begin + index].normalValid = samples[index].sample.valid;
        }
        ++report.normalSamplingCalls;
        reportProgress("normal_sampling", end, orderedVoxels.size(), normalStart, false);
    }
    reportProgress("normal_sampling", orderedVoxels.size(), orderedVoxels.size(), normalStart, true);
    report.normalSamplingSeconds = std::chrono::duration<double>(Clock::now() - normalStart).count();
    report.normalSamplingCpuSeconds = processCpuSeconds() - normalCpuStart;

    const auto materializationStart = Clock::now();
    const double materializationCpuStart = processCpuSeconds();
    reportProgress("materialization", 0, prepared.size(), materializationStart, true);
    std::unordered_map<Voxel, size_t, VoxelHash> voxelIndices;
    voxelIndices.reserve(checkedProduct(orderedVoxels.size(), 2, "fiberlet scoring index capacity"));
    for (size_t index = 0; index < orderedVoxels.size(); ++index)
        voxelIndices.emplace(orderedVoxels[index], index);
    const size_t scoringIndexPayloadBytes = checkedProduct(
        voxelIndices.size(),
        checkedSum(sizeof(Voxel), sizeof(size_t), "fiberlet scoring index byte estimate"),
        "fiberlet scoring index byte estimate");
    report.estimatedPeakOwnedBytes = std::
        max(report.estimatedPeakOwnedBytes,
            checkedSum(
                checkedSum(preparedBytes, sampledArrayBytes, "fiberlet peak owned byte estimate"),
                scoringIndexPayloadBytes,
                "fiberlet peak owned byte estimate"));
    errors.assign(prepared.size(), {});
    std::atomic<size_t> nextMaterialization{0};
    std::atomic<size_t> completedMaterialization{0};
    const auto materializationWorker = [&]() {
        while (true) {
            const size_t searchIndex = nextMaterialization.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= prepared.size())
                return;
            try {
                auto& item = prepared[searchIndex];
                const auto lookup = [&](const Voxel& voxel) -> const ScoringVoxel& {
                    const auto found = voxelIndices.find(voxel);
                    if (found == voxelIndices.end())
                        throw std::logic_error("prepared fiberlet point references an unsampled voxel");
                    return scoringVoxels[found->second];
                };
                const auto& candidate =
                    report.candidates[searchCandidateIndices[searchIndex]];
                item.startScoring = interpolateScoringPoint(
                    candidate.startPositionPredictionXYZ, grid, lookup);
                item.targetScoring = interpolateScoringPoint(
                    candidate.targetPositionPredictionXYZ, grid, lookup);
                for (size_t node = 0; node < item.nodes.size(); ++node) {
                    const ScoringVoxel scoring = interpolateScoringPoint(
                        nodePoint(item.nodes[node]), grid, lookup);
                    storeNodeScoring(item.nodes[node], scoring);
                }
            } catch (...) {
                errors[searchIndex] = std::current_exception();
            }
            const size_t completed = completedMaterialization.fetch_add(1, std::memory_order_relaxed) + 1;
            reportProgress("materialization", completed, prepared.size(), materializationStart, false);
        }
    };
    if (workerCount == 1) {
        materializationWorker();
    } else {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(materializationWorker);
        for (auto& thread : workers)
            thread.join();
    }
    reportProgress("materialization", prepared.size(), prepared.size(), materializationStart, true);
    report.samplingMaterializationSeconds = std::chrono::duration<double>(Clock::now() - materializationStart).count();
    report.samplingMaterializationCpuSeconds = processCpuSeconds() - materializationCpuStart;
    for (const auto& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
    voxelIndices.clear();
    voxelIndices.rehash(0);
    orderedVoxels.clear();
    orderedVoxels.shrink_to_fit();
    scoringVoxels.clear();
    scoringVoxels.shrink_to_fit();

    const auto searchStart = Clock::now();
    const double searchCpuStart = processCpuSeconds();
    reportProgress("search", 0, searchCandidateIndices.size(), searchStart, true);
    errors.assign(searchCandidateIndices.size(), {});
    std::atomic<size_t> nextSearch{0};
    std::atomic<size_t> completedSearches{0};
    const auto searchWorker = [&]() {
        while (true) {
            const size_t searchIndex = nextSearch.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= searchCandidateIndices.size())
                return;
            const size_t candidateIndex = searchCandidateIndices[searchIndex];
            try {
                report.candidates[candidateIndex] = solveCandidate(report.candidates[candidateIndex], report.config, prepared[searchIndex]);
            } catch (...) {
                errors[searchIndex] = std::current_exception();
            }
            const size_t completed = completedSearches.fetch_add(1, std::memory_order_relaxed) + 1;
            reportProgress("search", completed, searchCandidateIndices.size(), searchStart, false);
        }
    };
    if (workerCount == 1) {
        searchWorker();
    } else {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(searchWorker);
        for (auto& thread : workers)
            thread.join();
    }
    reportProgress("search", searchCandidateIndices.size(), searchCandidateIndices.size(), searchStart, true);
    report.searchSeconds = std::chrono::duration<double>(Clock::now() - searchStart).count();
    report.searchCpuSeconds = processCpuSeconds() - searchCpuStart;
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
    report.elapsedSeconds = std::chrono::duration<double>(searchEnd - startTime).count();
    report.elapsedCpuSeconds = processCpuSeconds() - startCpuSeconds;
    return report;
}

nlohmann::json fiberletPathReportJson(const FiberletPathReport& report, const FiberletArtifactInfo& artifact)
{
    if (artifact.fiberManifestLocator.empty() || artifact.fiberManifestContentHash.empty() || artifact.normalManifestLocator.empty() ||
        artifact.normalManifestContentHash.empty() || artifact.anchorArtifactLocator.empty() || artifact.anchorArtifactContentHash.empty()) {
        throw std::invalid_argument("fiberlet artifacts require complete source identities");
    }
    const auto visual = fiberletPathVisualMetrics(report);
    std::vector<const FiberletPathVisualMetric*> visualByCandidate(report.candidates.size(), nullptr);
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
             {"neighborhood_margin_cells", report.config.neighborhoodMarginCells},
             {"maximum_endpoint_angle_degrees", report.config.maximumEndpointAngleDegrees},
             {"maximum_prediction_deviation_degrees", report.config.maximumPredictionDeviationDegrees},
             {"dp_longitudinal_step_prediction_voxels", report.config.longitudinalStepPredictionVoxels},
             {"dp_longitudinal_step_base_voxels", report.config.longitudinalStepPredictionVoxels * report.grid.predictionToBaseScale},
             {"dp_transverse_step_prediction_voxels", report.config.transverseStepPredictionVoxels},
             {"dp_transverse_step_base_voxels", report.config.transverseStepPredictionVoxels * report.grid.predictionToBaseScale},
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
             {"neighborhood_offsets", report.diagnostics.neighborhoodOffsets},
             {"neighborhood_targets_out_of_grid", report.diagnostics.neighborhoodTargetsOutOfGrid},
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
              visual.minimumLossPerPredictionVoxel.has_value() ? nlohmann::json(*visual.minimumLossPerPredictionVoxel) : nlohmann::json(nullptr)},
             {"maximum_loss_per_prediction_voxel",
              visual.maximumLossPerPredictionVoxel.has_value() ? nlohmann::json(*visual.maximumLossPerPredictionVoxel) : nlohmann::json(nullptr)},
         }},
        {"candidates", nlohmann::json::array()},
    };
    if (artifact.baseVoxelSizeUm.has_value())
        root["coordinates"]["base_voxel_size_um"] = *artifact.baseVoxelSizeUm;
    for (size_t candidateIndex = 0; candidateIndex < report.candidates.size(); ++candidateIndex) {
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
            item["path_length_base_voxels"] = metrics->pathLengthPredictionVoxels * report.grid.predictionToBaseScale;
            item["loss_per_prediction_voxel"] = metrics->lossPerPredictionVoxel;
            item["relative_visual_quality"] = metrics->relativeQuality;
            item["start_prediction"] = {
                {"direction_xyz", pointJson(candidate.startPrediction.direction)},
                {"presence", candidate.startPrediction.presence},
                {"valid", candidate.startPrediction.valid},
                {"normal_xyz", pointJson(candidate.startNormalXYZ)},
                {"normal_valid", candidate.startNormalValid},
            };
            item["target_prediction"] = {
                {"direction_xyz", pointJson(candidate.targetPrediction.direction)},
                {"presence", candidate.targetPrediction.presence},
                {"valid", candidate.targetPrediction.valid},
                {"normal_xyz", pointJson(candidate.targetNormalXYZ)},
                {"normal_valid", candidate.targetNormalValid},
            };
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
        output << "# trace_loss_per_prediction_voxel " << path.lossPerPredictionVoxel << '\n';
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
        throw std::filesystem::filesystem_error("cannot remove stale fiberlet material artifact", outputDirectory / "fiberlets.mtl", error);
    }
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.json", fiberletPathReportJson(report, artifact).dump(2) + "\n");
    vc::core::util::atomicWriteString(outputDirectory / "fiberlets.obj", fiberletPathReportObj(report));
    auto graphJson = fiberletGraphJson(buildFiberletGraph(report));
    graphJson["source"] = {
        {"fiber_manifest", artifact.fiberManifestLocator},
        {"fiber_manifest_content_hash", artifact.fiberManifestContentHash},
        {"normal_manifest", artifact.normalManifestLocator},
        {"normal_manifest_content_hash", artifact.normalManifestContentHash},
        {"anchor_artifact", artifact.anchorArtifactLocator},
        {"anchor_artifact_content_hash", artifact.anchorArtifactContentHash},
    };
    vc::core::util::atomicWriteString(outputDirectory / "fiberlet_graph.json", graphJson.dump(2) + "\n");
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
