#include "vc/fiber_tracer/FiberPaths.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/TexturedMesh.hpp"
#include "vc/fiber_tracer/FiberAxisTensor.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"
#include "vc/fiber_tracer/FiberGraph.hpp"
#include "FiberFloatGeometry.hpp"
#include "FiberLocalScoringInternal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
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
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace vc::fiber_tracer
{
namespace
{

using Clock = std::chrono::steady_clock;
using Voxel = std::array<int64_t, 3>;  // XYZ

constexpr float kEpsilon = 1.0e-6F;
constexpr float kPi = 3.14159265358979323846F;

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
    size_t keyCount = 0;
};

struct CurvedLayer {
    float arc = 0.0F;
    cv::Vec3f center{0.0F, 0.0F, 0.0F};
    cv::Vec3f tangent{1.0F, 0.0F, 0.0F};
    cv::Vec3f transverseU{0.0F, 1.0F, 0.0F};
    cv::Vec3f transverseV{0.0F, 0.0F, 1.0F};
};

struct CurvedDomain {
    std::vector<CurvedLayer> layers;
    float length = 0.0F;
};

struct SearchNode {
    cv::Vec3f point{0.0f, 0.0f, 0.0f};
    uint32_t key = 0;
};

static_assert(sizeof(SearchNode) == 16);

struct CompactNodeScoring {
    std::array<uint8_t, 2> predictionAxis{128, 128};
    std::array<uint8_t, 2> normalAxis{128, 128};
    uint8_t presence = 0;
    uint8_t flags = 0;
};

static_assert(sizeof(CompactNodeScoring) == 6);

struct SearchCorridor {
    std::vector<cv::Vec3f> reference;
    Voxel begin{0, 0, 0};
    Voxel end{-1, -1, -1};
    float radius = 0.0F;
    float radiusSquared = 0.0f;
};

struct ScoringVoxel {
    FiberletPredictionSample prediction;
    cv::Vec3f normal{0.0F, 0.0F, 0.0F};
    bool normalValid = false;
};

struct SymmetricAxisTensor {
    // xx, xy, xz, yy, yz, zz
    std::array<float, 6> values{};
};

struct PreparedScoringVoxel {
    cv::Vec3f predictionAxis{0.0f, 0.0f, 0.0f};
    cv::Vec3f normalAxis{0.0f, 0.0f, 0.0f};
    SymmetricAxisTensor predictionTensor;
    SymmetricAxisTensor normalTensor;
    float presence = 0.0f;
    uint8_t flags = 0;
};

static_assert(std::is_same_v<decltype(PreparedScoringVoxel::presence), float>);

constexpr uint8_t kPreparedPredictionValid = 1U << 0;
constexpr uint8_t kPreparedNormalValid = 1U << 1;

bool finiteVector(const cv::Vec3f& value);

SymmetricAxisTensor compactAxisTensor(const cv::Vec3f& axis)
{
    return {{
        axis[0] * axis[0], axis[0] * axis[1], axis[0] * axis[2],
        axis[1] * axis[1], axis[1] * axis[2], axis[2] * axis[2],
    }};
}

void prepareAxis(
    const cv::Vec3f& input,
    bool valid,
    cv::Vec3f& axis,
    SymmetricAxisTensor& tensor,
    uint8_t flag,
    uint8_t& flags)
{
    const float norm2 = input.dot(input);
    if (!valid || !finiteVector(input) || !(norm2 > kEpsilon) ||
        !std::isfinite(norm2)) {
        return;
    }
    const cv::Vec3f normalized = input / std::sqrt(norm2);
    axis = normalized;
    tensor = compactAxisTensor(normalized);
    flags |= flag;
}

PreparedScoringVoxel prepareScoringVoxel(const ScoringVoxel& input)
{
    PreparedScoringVoxel output;
    const bool predictionValid = input.prediction.valid &&
        std::isfinite(input.prediction.presence);
    prepareAxis(
        input.prediction.direction, predictionValid,
        output.predictionAxis, output.predictionTensor,
        kPreparedPredictionValid, output.flags);
    prepareAxis(
        input.normal, input.normalValid,
        output.normalAxis, output.normalTensor,
        kPreparedNormalValid, output.flags);
    output.presence = input.prediction.presence;
    return output;
}

void accumulateTensor(
    cv::Matx33f& output,
    const SymmetricAxisTensor& input,
    float weight)
{
    const auto& value = input.values;
    output(0, 0) += weight * value[0];
    output(0, 1) += weight * value[1];
    output(1, 0) += weight * value[1];
    output(0, 2) += weight * value[2];
    output(2, 0) += weight * value[2];
    output(1, 1) += weight * value[3];
    output(1, 2) += weight * value[4];
    output(2, 1) += weight * value[4];
    output(2, 2) += weight * value[5];
}

struct PreparedCandidate {
    CurvedDomain domain;
    std::vector<SearchNode> nodes;
    LocalNodeKeyLayout keyLayout;
    size_t maximumActiveLayerNodes = 0;
    ScoringVoxel startScoring;
    ScoringVoxel targetScoring;
};

struct PreparationProfile {
    size_t latticeNodePositions = 0;
    size_t corridorSegmentTests = 0;
    size_t corridorAcceptedNodes = 0;
    size_t pointPredicateCalls = 0;
    size_t retainedNodes = 0;
    size_t interpolationCornerInsertions = 0;
    double geometrySeconds = 0.0;
    double nodeEnumerationSeconds = 0.0;
    double cornerCollectionSeconds = 0.0;
};

struct SolveProfile {
    size_t nodeIndexEntries = 0;
    size_t nodeIndexSlots = 0;
    size_t preparedNodes = 0;
    size_t preparedNodeBytes = 0;
    size_t lazyCacheIndexBytes = 0;
    size_t directIndexBytes = 0;
    size_t stateBytes = 0;
    size_t lazyNodeRequests = 0;
    size_t lazyNodeCacheHits = 0;
    size_t scoringPageDirectoryProbes = 0;
    size_t interpolationProfiledPoints = 0;
    size_t interpolationProfiledCorners = 0;
    size_t interpolationProfiledPredictionIdentical = 0;
    size_t interpolationProfiledNormalIdentical = 0;
    size_t interpolationProfiledPredictionPrincipalSolves = 0;
    size_t interpolationProfiledNormalPrincipalSolves = 0;
    size_t interpolationPredictionClosedFormResolutions = 0;
    size_t interpolationNormalClosedFormResolutions = 0;
    size_t interpolationPredictionIterativeFallbacks = 0;
    size_t interpolationNormalIterativeFallbacks = 0;
    size_t reachedNodes = 0;
    size_t generatedEdges = 0;
    size_t validEdges = 0;
    size_t reusedEdges = 0;
    size_t transitionLookups = 0;
    size_t reachedStateVisits = 0;
    size_t relaxations = 0;
    double nodeIndexSeconds = 0.0;
    double nodePreparationSeconds = 0.0;
    double dpSeconds = 0.0;
    double interpolationProfiledLookupSeconds = 0.0;
    double interpolationProfiledPredictionCornerSeconds = 0.0;
    double interpolationProfiledNormalCornerSeconds = 0.0;
    double interpolationProfiledPredictionResolveSeconds = 0.0;
    double interpolationProfiledNormalResolveSeconds = 0.0;
};

struct DpNodeScoring {
    cv::Vec3f predictionAxis{0.0F, 0.0F, 0.0F};
    FiberLocalMetricSample metricPrediction;
    cv::Vec3f metricNormal{0.0f, 0.0f, 0.0f};
    std::array<uint8_t, 2> normalAxis{128, 128};
    uint8_t flags = 0;
};

struct DpEdge {
    uint32_t next = std::numeric_limits<uint32_t>::max();
    uint32_t scoring = std::numeric_limits<uint32_t>::max();
    float metricLength = 0.0f;
    detail::FiberLocalPreparedCandidateMetric candidateMetric;
};

struct DpAccumulatedCost {
    float invalidPrediction = 0.0f;
    float alignment = 0.0f;
    float isotropicSmoothness = 0.0f;
    float tangentSmoothness = 0.0f;
    float normalSmoothness = 0.0f;

    float total() const noexcept
    {
        return invalidPrediction + alignment + isotropicSmoothness +
            tangentSmoothness + normalSmoothness;
    }

    DpAccumulatedCost& operator+=(const FiberletPathCost& other) noexcept
    {
        invalidPrediction += static_cast<float>(other.invalidPrediction);
        alignment += static_cast<float>(other.alignment);
        isotropicSmoothness += static_cast<float>(other.isotropicSmoothness);
        tangentSmoothness += static_cast<float>(other.tangentSmoothness);
        normalSmoothness += static_cast<float>(other.normalSmoothness);
        return *this;
    }

    DpAccumulatedCost& operator+=(const FiberLocalMetricCost& other) noexcept
    {
        invalidPrediction += other.invalidPrediction;
        alignment += other.alignment;
        isotropicSmoothness += other.isotropicSmoothness;
        tangentSmoothness += other.tangentSmoothness;
        normalSmoothness += other.normalSmoothness;
        return *this;
    }
};

DpAccumulatedCost dpAccumulatedCost(const FiberletPathCost& cost)
{
    DpAccumulatedCost output;
    output += cost;
    return output;
}

FiberletPathCost fiberletPathCost(const DpAccumulatedCost& cost)
{
    FiberletPathCost output;
    output.invalidPrediction = cost.invalidPrediction;
    output.alignment = cost.alignment;
    output.isotropicSmoothness = cost.isotropicSmoothness;
    output.tangentSmoothness = cost.tangentSmoothness;
    output.normalSmoothness = cost.normalSmoothness;
    return output;
}

struct DpLayerState {
    DpAccumulatedCost cost;
    bool reached = false;
};

struct NodeRange {
    size_t begin = 0;
    size_t end = 0;

    size_t size() const noexcept { return end - begin; }
};

struct DpIncoming {
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float length = 0.0F;
    cv::Vec3f metricDirection{0.0f, 0.0f, 0.0f};
    float metricLength = 0.0f;
};

float vectorLength(const cv::Vec3f& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const float length = vectorLength(value);
    if (!(length > kEpsilon) || !std::isfinite(length))
        return {0.0F, 0.0F, 0.0F};
    return value / length;
}

cv::Vec3f nativeVoxelPoint(const Voxel& voxel)
{
    return {static_cast<float>(voxel[0]), static_cast<float>(voxel[1]), static_cast<float>(voxel[2])};
}

float directedAngle(const cv::Vec3f& left, const cv::Vec3f& right)
{
    const cv::Vec3f a = normalized(left);
    const cv::Vec3f b = normalized(right);
    return std::acos(std::clamp(a.dot(b), -1.0F, 1.0F));
}

bool finiteVector(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool insidePredictionGrid(const cv::Vec3f& point, const FiberPredictionGridInfo& grid)
{
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    for (size_t axis = 0; axis < 3; ++axis) {
        if (shapeXYZ[axis] == 0 || !std::isfinite(point[axis]) || point[axis] < 0.0)
            return false;
        if (point[axis] > static_cast<float>(shapeXYZ[axis] - 1))
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

cv::Vec3d jsonVec3d(const nlohmann::json& value, const char* name)
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

cv::Vec3f checkedVec3f(const cv::Vec3d& parsed, const char* name)
{
    constexpr double lowest =
        static_cast<double>(std::numeric_limits<float>::lowest());
    constexpr double maximum =
        static_cast<double>(std::numeric_limits<float>::max());
    for (int axis = 0; axis < 3; ++axis) {
        const double component = parsed[axis];
        if (component < lowest || component > maximum) {
            throw std::runtime_error(
                std::string(name) + " must be representable as finite float32");
        }
    }
    const cv::Vec3f out(parsed);
    if (!finiteVector(out))
        throw std::runtime_error(std::string(name) + " must be finite float32");
    return out;
}

cv::Vec3f jsonVec3f(const nlohmann::json& value, const char* name)
{
    return checkedVec3f(jsonVec3d(value, name), name);
}

float finiteFloat(const nlohmann::json& value, const char* name)
{
    if (!value.is_number())
        throw std::runtime_error(std::string(name) + " must be numeric");
    const double parsed = value.get<double>();
    const float out = static_cast<float>(parsed);
    if (!std::isfinite(parsed) || !std::isfinite(out))
        throw std::runtime_error(std::string(name) + " must be finite float32");
    return out;
}

double finiteDouble(const nlohmann::json& value, const char* name)
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

cv::Vec3f hermitePoint(const cv::Vec3f& start, const cv::Vec3f& target, const cv::Vec3f& startDerivative, const cv::Vec3f& targetDerivative, float t)
{
    const float t2 = t * t;
    const float t3 = t2 * t;
    return start * (2.0F * t3 - 3.0F * t2 + 1.0F) + startDerivative * (t3 - 2.0F * t2 + t) + target * (-2.0F * t3 + 3.0F * t2) +
           targetDerivative * (t3 - t2);
}

cv::Vec3f hermiteDerivative(const cv::Vec3f& start, const cv::Vec3f& target, const cv::Vec3f& startDerivative, const cv::Vec3f& targetDerivative, float t)
{
    const float t2 = t * t;
    return start * (6.0F * t2 - 6.0F * t) + startDerivative * (3.0F * t2 - 4.0F * t + 1.0F) + target * (-6.0F * t2 + 6.0F * t) +
           targetDerivative * (3.0F * t2 - 2.0F * t);
}

cv::Vec3f rotateMinimal(const cv::Vec3f& value, const cv::Vec3f& from, const cv::Vec3f& to)
{
    const cv::Vec3f cross = from.cross(to);
    const float sine = vectorLength(cross);
    const float cosine = std::clamp(from.dot(to), -1.0F, 1.0F);
    if (sine <= kEpsilon) {
        if (cosine >= 0.0F)
            return value;
        cv::Vec3f axis = normalized(value - from * value.dot(from));
        if (vectorLength(axis) <= kEpsilon)
            axis = normalized(cv::Vec3f{1.0F, 0.0F, 0.0F} - from * from[0]);
        return axis * (2.0F * axis.dot(value)) - value;
    }
    const cv::Vec3f axis = cross / sine;
    return value * cosine + axis.cross(value) * sine + axis * axis.dot(value) * (1.0F - cosine);
}

CurvedDomain makeCurvedDomain(const FiberletCandidateResult& candidate, const FiberletPathConfig& config)
{
    const cv::Vec3f chord = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
    const float chordLength = vectorLength(chord);
    if (!(chordLength > kEpsilon))
        throw std::invalid_argument("fiberlet curved domain requires distinct endpoints");
    const cv::Vec3f startDerivative = candidate.startAxisXYZ * chordLength;
    const cv::Vec3f targetDerivative = candidate.targetAxisXYZ * chordLength;
    const size_t samples = static_cast<size_t>(std::max(64.0F, std::ceil(chordLength * 16.0F)));
    std::vector<float> sampleArcs(samples + 1, 0.0F);
    std::vector<cv::Vec3f> samplePoints(samples + 1);
    for (size_t index = 0; index <= samples; ++index) {
        const float t = static_cast<float>(index) / static_cast<float>(samples);
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
    std::vector<float> layerArcs{0.0F};
    for (float arc = config.longitudinalStepPredictionVoxels; arc < domain.length - kEpsilon; arc += config.longitudinalStepPredictionVoxels) {
        layerArcs.push_back(arc);
    }
    layerArcs.push_back(domain.length);
    domain.layers.reserve(layerArcs.size());
    for (const float arc : layerArcs) {
        float t = 0.0F;
        if (arc >= domain.length) {
            t = 1.0F;
        } else if (arc > 0.0F) {
            const auto upper = std::lower_bound(sampleArcs.begin(), sampleArcs.end(), arc);
            const size_t upperIndex = static_cast<size_t>(std::distance(sampleArcs.begin(), upper));
            const size_t lowerIndex = upperIndex - 1;
            const float span = sampleArcs[upperIndex] - sampleArcs[lowerIndex];
            const float fraction = span > kEpsilon ? (arc - sampleArcs[lowerIndex]) / span : 0.0F;
            t = (static_cast<float>(lowerIndex) + fraction) / static_cast<float>(samples);
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
            const std::array<cv::Vec3f, 3> axes{cv::Vec3f{1.0F, 0.0F, 0.0F}, cv::Vec3f{0.0F, 1.0F, 0.0F}, cv::Vec3f{0.0F, 0.0F, 1.0F}};
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

std::vector<cv::Vec3f> domainCenterline(const CurvedDomain& domain)
{
    std::vector<cv::Vec3f> points;
    points.reserve(domain.layers.size());
    for (const auto& layer : domain.layers)
        points.push_back(layer.center);
    return points;
}

cv::Vec3f localNodePoint(const CurvedDomain& domain, const LocalNodeKey& key, const FiberletPathConfig& config)
{
    const auto& layer = domain.layers.at(key.layer);
    return layer.center + layer.transverseU * (static_cast<float>(key.transverseU) * config.transverseStepPredictionVoxels) +
           layer.transverseV * (static_cast<float>(key.transverseV) * config.transverseStepPredictionVoxels);
}

LocalNodeKeyLayout makeLocalNodeKeyLayout(
    const CurvedDomain& domain,
    float radius,
    float transverseStep)
{
    const float rawLimit = std::ceil(radius / transverseStep);
    if (!std::isfinite(rawLimit) || rawLimit < 0.0 ||
        rawLimit > static_cast<float>((std::numeric_limits<int>::max() - 1) / 2)) {
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
    const uint64_t keyCount =
        static_cast<uint64_t>(domain.layers.size()) * plane;
    if (keyCount > std::numeric_limits<size_t>::max())
        throw std::overflow_error("fiberlet local node lattice exceeds addressable memory");
    layout.keyCount = static_cast<size_t>(keyCount);
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

cv::Vec3f nodePoint(const SearchNode& node)
{
    return node.point;
}

SearchCorridor makeSearchCorridor(const CurvedDomain& domain, const FiberPredictionGridInfo& grid, int cellSize, const FiberletPathConfig& config)
{
    SearchCorridor corridor;
    const float radius = config.corridorRadiusPredictionVoxels > 0.0F ? config.corridorRadiusPredictionVoxels : static_cast<float>(cellSize);
    if (!(radius > 0.0F) || !std::isfinite(radius) ||
        !std::isfinite(radius * radius)) {
        throw std::overflow_error("fiberlet corridor radius is not finite as float32");
    }
    corridor.radius = radius;
    corridor.radiusSquared = radius * radius;
    const auto reference = domainCenterline(domain);
    corridor.reference.reserve(reference.size());
    for (const auto& point : reference) {
        if (!finiteVector(point)) {
            throw std::overflow_error(
                "fiberlet corridor reference is not finite as float32");
        }
        corridor.reference.push_back(point);
    }
    cv::Vec3f minimum = reference.front();
    cv::Vec3f maximum = reference.front();
    for (const auto& point : reference) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    for (const cv::Vec3f* endpoint : {&domain.layers.front().center, &domain.layers.back().center}) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], std::floor((*endpoint)[axis]) - 1.0F);
            maximum[axis] = std::max(maximum[axis], std::ceil((*endpoint)[axis]) + 1.0F);
        }
    }
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    for (int axis = 0; axis < 3; ++axis) {
        if (shapeXYZ[axis] == 0 || shapeXYZ[axis] > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("fiberlet prediction shape exceeds signed search indexing");
        }
        const float rawBegin = std::floor(minimum[axis] - radius);
        const float rawEnd = std::ceil(maximum[axis] + radius);
        const int64_t gridEnd = static_cast<int64_t>(shapeXYZ[axis]) - 1;
        if (!std::isfinite(rawBegin) || !std::isfinite(rawEnd))
            throw std::overflow_error("fiberlet corridor bound is not finite");
        corridor.begin[axis] = rawBegin <= 0.0F ? 0 : rawBegin >= static_cast<float>(gridEnd) ? gridEnd : static_cast<int64_t>(rawBegin);
        corridor.end[axis] = rawEnd >= static_cast<float>(gridEnd) ? gridEnd : rawEnd <= 0.0F ? 0 : static_cast<int64_t>(rawEnd);
    }
    return corridor;
}

float pointSegmentDistanceSquared(const cv::Vec3f& point, const cv::Vec3f& start, const cv::Vec3f& target)
{
    const cv::Vec3f delta = target - start;
    const float denominator = delta.dot(delta);
    if (!(denominator > 0.0f))
        return (point - start).dot(point - start);
    const float t = std::clamp(
        (point - start).dot(delta) / denominator, 0.0f, 1.0f);
    const cv::Vec3f residual = point - (start + delta * t);
    return residual.dot(residual);
}

bool insideCorridor(
    const cv::Vec3f& point,
    const std::vector<cv::Vec3f>& reference,
    float radiusSquared,
    std::optional<size_t> adjacentSegment,
    size_t& segmentTests)
{
    const auto passes = [&](size_t segment) {
        ++segmentTests;
        return pointSegmentDistanceSquared(
                   point, reference[segment], reference[segment + 1]) <=
            radiusSquared;
    };
    if (adjacentSegment.has_value() && *adjacentSegment + 1 < reference.size() &&
        passes(*adjacentSegment)) {
        return true;
    }
    for (size_t segment = 0; segment + 1 < reference.size(); ++segment) {
        if (adjacentSegment.has_value() && segment == *adjacentSegment)
            continue;
        if (passes(segment)) {
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
    const LocalNodeKeyLayout& layout,
    size_t& maximumActiveLayerNodes,
    PreparationProfile& profile)
{
    std::vector<SearchNode> nodes;
    maximumActiveLayerNodes = 0;
    if (domain.layers.size() <= 2)
        return nodes;
    size_t previousLayerNodes = 0;
    for (size_t layer = 1; layer + 1 < domain.layers.size(); ++layer) {
        const size_t layerBegin = nodes.size();
        for (int u = -layout.transverseLimit; u <= layout.transverseLimit; ++u) {
            for (int v = -layout.transverseLimit; v <= layout.transverseLimit; ++v) {
                ++profile.latticeNodePositions;
                const LocalNodeKey key{layer, u, v};
                const cv::Vec3f point = localNodePoint(domain, key, config);
                if (!finiteVector(point))
                    throw std::overflow_error("fiberlet local node is not finite as float32");
                if (!insidePredictionGrid(point, grid) ||
                    !insideCorridor(
                        point, corridor.reference, corridor.radiusSquared,
                        layer - 1,
                        profile.corridorSegmentTests)) {
                    continue;
                }
                ++profile.corridorAcceptedNodes;
                bool selected = true;
                if (pointPredicate) {
                    ++profile.pointPredicateCalls;
                    selected = pointPredicate(point);
                }
                if (selected) {
                    SearchNode node;
                    node.point = point;
                    node.key = packLocalNodeKey(key, layout);
                    nodes.push_back(node);
                }
            }
        }
        const size_t currentLayerNodes = nodes.size() - layerBegin;
        maximumActiveLayerNodes = std::max(
            maximumActiveLayerNodes,
            previousLayerNodes + currentLayerNodes);
        previousLayerNodes = currentLayerNodes;
    }
    return nodes;
}

template <typename Callback>
void forEachInterpolationCorner(
    const cv::Vec3f& point,
    const FiberPredictionGridInfo& grid,
    Callback&& callback)
{
    if (!insidePredictionGrid(point, grid))
        throw std::out_of_range("fiberlet sample point is outside the prediction volume");
    Voxel lower{};
    std::array<float, 3> fraction{};
    for (size_t axis = 0; axis < 3; ++axis) {
        lower[axis] = static_cast<int64_t>(std::floor(point[axis]));
        fraction[axis] = point[axis] - static_cast<float>(lower[axis]);
    }
    const std::array<size_t, 3> shapeXYZ{grid.shapeZYX[2], grid.shapeZYX[1], grid.shapeZYX[0]};
    Voxel upper{};
    for (size_t axis = 0; axis < 3; ++axis)
        upper[axis] = std::min<int64_t>(lower[axis] + 1, static_cast<int64_t>(shapeXYZ[axis] - 1));
    for (int z = 0; z <= 1; ++z) {
        const float wz = z == 0 ? 1.0F - fraction[2] : fraction[2];
        if (!(wz > 0.0F))
            continue;
        for (int y = 0; y <= 1; ++y) {
            const float wy = y == 0 ? 1.0F - fraction[1] : fraction[1];
            if (!(wy > 0.0F))
                continue;
            for (int x = 0; x <= 1; ++x) {
                const float wx = x == 0 ? 1.0F - fraction[0] : fraction[0];
                const float weight = wx * wy * wz;
                if (weight > 0.0F)
                    callback(Voxel{x == 0 ? lower[0] : upper[0], y == 0 ? lower[1] : upper[1], z == 0 ? lower[2] : upper[2]}, weight);
            }
        }
    }
}

struct InterpolationProfileSample {
    size_t points = 0;
    size_t corners = 0;
    size_t predictionIdentical = 0;
    size_t normalIdentical = 0;
    size_t predictionPrincipalSolves = 0;
    size_t normalPrincipalSolves = 0;
    double lookupSeconds = 0.0;
    double predictionCornerSeconds = 0.0;
    double normalCornerSeconds = 0.0;
    double predictionResolveSeconds = 0.0;
    double normalResolveSeconds = 0.0;
};

struct InterpolationResolutionStats {
    size_t predictionClosedFormResolutions = 0;
    size_t normalClosedFormResolutions = 0;
    size_t predictionIterativeFallbacks = 0;
    size_t normalIterativeFallbacks = 0;
};

template <typename Lookup>
ScoringVoxel interpolateScoringPoint(
    const cv::Vec3f& point,
    const FiberPredictionGridInfo& grid,
    Lookup&& lookup,
    InterpolationProfileSample* profile = nullptr,
    InterpolationResolutionStats* resolutionStats = nullptr)
{
    if (profile != nullptr)
        ++profile->points;
    ScoringVoxel output;
    cv::Matx33f predictionTensor = cv::Matx33f::zeros();
    cv::Matx33f normalTensor = cv::Matx33f::zeros();
    bool predictionValid = true;
    bool normalValid = true;
    bool predictionAxesIdentical = true;
    bool normalAxesIdentical = true;
    std::optional<cv::Vec3f> firstPredictionAxis;
    std::optional<cv::Vec3f> firstNormalAxis;
    float presence = 0.0F;
    forEachInterpolationCorner(point, grid, [&](const Voxel& corner, float weight) {
        const auto lookupStart = profile == nullptr ? Clock::time_point{} : Clock::now();
        const auto& sample = lookup(corner);
        if (profile != nullptr) {
            ++profile->corners;
            profile->lookupSeconds += std::chrono::duration<double>(
                Clock::now() - lookupStart).count();
        }
        const auto predictionStart = profile == nullptr ? Clock::time_point{} : Clock::now();
        if ((sample.flags & kPreparedPredictionValid) == 0) {
            predictionValid = false;
        } else {
            const cv::Vec3f axis = sample.predictionAxis;
            if (!firstPredictionAxis.has_value())
                firstPredictionAxis = axis;
            else if (std::abs(firstPredictionAxis->dot(axis)) < 1.0F - 1.0e-5F)
                predictionAxesIdentical = false;
            accumulateTensor(predictionTensor, sample.predictionTensor, weight);
            presence += weight * sample.presence;
        }
        if (profile != nullptr) {
            profile->predictionCornerSeconds += std::chrono::duration<double>(
                Clock::now() - predictionStart).count();
        }

        const auto normalStart = profile == nullptr ? Clock::time_point{} : Clock::now();
        if ((sample.flags & kPreparedNormalValid) == 0) {
            normalValid = false;
        } else {
            const cv::Vec3f axis = sample.normalAxis;
            if (!firstNormalAxis.has_value())
                firstNormalAxis = axis;
            else if (std::abs(firstNormalAxis->dot(axis)) < 1.0F - 1.0e-5F)
                normalAxesIdentical = false;
            accumulateTensor(normalTensor, sample.normalTensor, weight);
        }
        if (profile != nullptr) {
            profile->normalCornerSeconds += std::chrono::duration<double>(
                Clock::now() - normalStart).count();
        }
    });
    if (predictionValid && std::isfinite(presence)) {
        const auto resolveStart = profile == nullptr ? Clock::time_point{} : Clock::now();
        bool usedIterativeFallback = false;
        const auto principal = principalFiberAxisClosedFormF(
            predictionTensor, &usedIterativeFallback);
        if (resolutionStats != nullptr) {
            ++resolutionStats->predictionClosedFormResolutions;
            if (usedIterativeFallback)
                ++resolutionStats->predictionIterativeFallbacks;
        }
        if (profile != nullptr) {
            ++profile->predictionPrincipalSolves;
            if (predictionAxesIdentical && firstPredictionAxis.has_value())
                ++profile->predictionIdentical;
            profile->predictionResolveSeconds += std::chrono::duration<double>(
                Clock::now() - resolveStart).count();
        }
        if (predictionAxesIdentical && firstPredictionAxis.has_value()) {
            output.prediction.direction = canonicalFiberAxisF(*firstPredictionAxis);
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
        const auto resolveStart = profile == nullptr ? Clock::time_point{} : Clock::now();
        bool usedIterativeFallback = false;
        const auto principal = principalFiberAxisClosedFormF(
            normalTensor, &usedIterativeFallback);
        if (resolutionStats != nullptr) {
            ++resolutionStats->normalClosedFormResolutions;
            if (usedIterativeFallback)
                ++resolutionStats->normalIterativeFallbacks;
        }
        if (profile != nullptr) {
            ++profile->normalPrincipalSolves;
            if (normalAxesIdentical && firstNormalAxis.has_value())
                ++profile->normalIdentical;
            profile->normalResolveSeconds += std::chrono::duration<double>(
                Clock::now() - resolveStart).count();
        }
        if (normalAxesIdentical && firstNormalAxis.has_value()) {
            output.normal = canonicalFiberAxisF(*firstNormalAxis);
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

class PagedScoringIndex {
    struct Page;

public:
    static constexpr int64_t pageSize = 16;
    static constexpr size_t pageSlotCount =
        static_cast<size_t>(pageSize * pageSize * pageSize);
    static constexpr uint32_t missing =
        std::numeric_limits<uint32_t>::max();

    explicit PagedScoringIndex(const std::vector<Voxel>& voxels)
    {
        if (voxels.size() > static_cast<size_t>(missing))
            throw std::overflow_error("fiberlet scoring index exceeds 32 bits");
        for (size_t index = 0; index < voxels.size(); ++index) {
            const Voxel key = pageKey(voxels[index]);
            auto [page, inserted] = pages_.try_emplace(key);
            (void)inserted;
            auto& slot = page->second.indices[localOffset(voxels[index])];
            if (slot != missing)
                throw std::logic_error("fiberlet scoring index received a duplicate voxel");
            slot = static_cast<uint32_t>(index);
        }
    }

    class Lookup {
    public:
        Lookup(
            const PagedScoringIndex& index,
            const std::vector<PreparedScoringVoxel>& scoring,
            size_t& directoryProbes)
            : index_(index), scoring_(scoring), directoryProbes_(directoryProbes)
        {
        }

        const PreparedScoringVoxel& operator()(const Voxel& voxel)
        {
            const Voxel key = pageKey(voxel);
            const Page* page = nullptr;
            for (size_t cached = 0; cached < cacheSize_; ++cached) {
                if (cacheKeys_[cached] == key) {
                    page = cachePages_[cached];
                    break;
                }
            }
            if (page == nullptr) {
                ++directoryProbes_;
                const auto found = index_.pages_.find(key);
                if (found == index_.pages_.end()) {
                    throw std::logic_error(
                        "prepared fiberlet point references an unsampled page");
                }
                page = &found->second;
                if (cacheSize_ >= cacheKeys_.size())
                    throw std::logic_error("fiberlet interpolation spans too many pages");
                cacheKeys_[cacheSize_] = key;
                cachePages_[cacheSize_] = page;
                ++cacheSize_;
            }
            const uint32_t scoringIndex = page->indices[localOffset(voxel)];
            if (scoringIndex == missing) {
                throw std::logic_error(
                    "prepared fiberlet point references an unsampled voxel");
            }
            return scoring_[scoringIndex];
        }

    private:
        const PagedScoringIndex& index_;
        const std::vector<PreparedScoringVoxel>& scoring_;
        size_t& directoryProbes_;
        std::array<Voxel, 8> cacheKeys_{};
        std::array<const Page*, 8> cachePages_{};
        size_t cacheSize_ = 0;
    };

    Lookup lookup(
        const std::vector<PreparedScoringVoxel>& scoring,
        size_t& directoryProbes) const
    {
        return Lookup(*this, scoring, directoryProbes);
    }

    size_t pageCount() const noexcept { return pages_.size(); }

    size_t slotCount() const
    {
        return checkedProduct(
            pages_.size(), pageSlotCount,
            "fiberlet scoring page slot count");
    }

    size_t payloadBytes() const
    {
        return checkedSum(
            checkedProduct(
                slotCount(), sizeof(uint32_t),
                "fiberlet scoring page byte estimate"),
            checkedProduct(
                pages_.size(), sizeof(Voxel),
                "fiberlet scoring page key byte estimate"),
            "fiberlet scoring index byte estimate");
    }

    void clear()
    {
        pages_.clear();
        pages_.rehash(0);
    }

private:
    struct Page {
        Page() { indices.fill(missing); }
        std::array<uint32_t, pageSlotCount> indices;
    };

    static Voxel pageKey(const Voxel& voxel)
    {
        if (voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0)
            throw std::logic_error("fiberlet scoring voxel is negative");
        return {
            voxel[0] / pageSize,
            voxel[1] / pageSize,
            voxel[2] / pageSize,
        };
    }

    static size_t localOffset(const Voxel& voxel)
    {
        return static_cast<size_t>(voxel[2] % pageSize) *
                static_cast<size_t>(pageSize * pageSize) +
            static_cast<size_t>(voxel[1] % pageSize) *
                static_cast<size_t>(pageSize) +
            static_cast<size_t>(voxel[0] % pageSize);
    }

    std::unordered_map<Voxel, Page, VoxelHash> pages_;
};

bool storedVoxelLess(const Voxel& left, const Voxel& right)
{
    return storedIndex(left) < storedIndex(right);
}

class SparseCornerBitmap {
public:
    static constexpr int64_t pageSize = 16;
    static constexpr size_t pageSlotCount =
        static_cast<size_t>(pageSize * pageSize * pageSize);
    static constexpr size_t pageWordCount = pageSlotCount / 64;

    struct Page {
        std::array<uint64_t, pageWordCount> words{};
    };

    void insert(const Voxel& voxel)
    {
        const Voxel key = pageKey(voxel);
        Page* page = nullptr;
        if (lastPage_ != nullptr && key == lastKey_) {
            ++samePageHits_;
            page = lastPage_;
        } else {
            for (size_t index = 0; index < cacheSize_; ++index) {
                if (cacheKeys_[index] == key) {
                    ++cachedPageHits_;
                    page = cachePages_[index];
                    break;
                }
            }
            if (page == nullptr) {
                ++directoryProbes_;
                auto [found, inserted] = pages_.try_emplace(key);
                (void)inserted;
                page = &found->second;
                cacheKeys_[nextCacheSlot_] = key;
                cachePages_[nextCacheSlot_] = page;
                nextCacheSlot_ = (nextCacheSlot_ + 1) % cacheKeys_.size();
                cacheSize_ = std::min(cacheSize_ + 1, cacheKeys_.size());
            }
            lastKey_ = key;
            lastPage_ = page;
        }
        const size_t offset = localOffset(voxel);
        const uint64_t bit = uint64_t{1} << (offset % 64);
        uint64_t& word = page->words[offset / 64];
        if ((word & bit) == 0) {
            word |= bit;
            ++uniqueVoxels_;
        }
    }

    size_t uniqueVoxels() const noexcept { return uniqueVoxels_; }
    size_t pageCount() const noexcept { return pages_.size(); }
    size_t directoryProbes() const noexcept { return directoryProbes_; }
    size_t samePageHits() const noexcept { return samePageHits_; }
    size_t cachedPageHits() const noexcept { return cachedPageHits_; }

    size_t payloadBytes() const
    {
        return checkedProduct(
            pages_.size(), sizeof(std::pair<const Voxel, Page>),
            "fiberlet corner page byte estimate");
    }

    const std::unordered_map<Voxel, Page, VoxelHash>& pages() const noexcept
    {
        return pages_;
    }

    void clear()
    {
        pages_.clear();
        pages_.rehash(0);
        lastPage_ = nullptr;
        cacheSize_ = 0;
        nextCacheSlot_ = 0;
        uniqueVoxels_ = 0;
    }

    static size_t localOffset(const Voxel& voxel)
    {
        const auto local = [](int64_t value) {
            int64_t remainder = value % pageSize;
            if (remainder < 0)
                remainder += pageSize;
            return static_cast<size_t>(remainder);
        };
        return local(voxel[2]) *
                static_cast<size_t>(pageSize * pageSize) +
            local(voxel[1]) *
                static_cast<size_t>(pageSize) +
            local(voxel[0]);
    }

private:
    static Voxel pageKey(const Voxel& voxel)
    {
        const auto page = [](int64_t value) {
            int64_t quotient = value / pageSize;
            if (value % pageSize < 0)
                --quotient;
            return quotient;
        };
        return {
            page(voxel[0]),
            page(voxel[1]),
            page(voxel[2]),
        };
    }

    std::unordered_map<Voxel, Page, VoxelHash> pages_;
    Voxel lastKey_{};
    Page* lastPage_ = nullptr;
    std::array<Voxel, 8> cacheKeys_{};
    std::array<Page*, 8> cachePages_{};
    size_t cacheSize_ = 0;
    size_t nextCacheSlot_ = 0;
    size_t uniqueVoxels_ = 0;
    size_t directoryProbes_ = 0;
    size_t samePageHits_ = 0;
    size_t cachedPageHits_ = 0;
};

void addInterpolationCorners(
    const cv::Vec3f& point,
    const FiberPredictionGridInfo& grid,
    SparseCornerBitmap& corners,
    size_t& insertionAttempts)
{
    forEachInterpolationCorner(point, grid, [&](const Voxel& corner, float) {
        ++insertionAttempts;
        corners.insert(corner);
    });
}

PreparedCandidate prepareCandidate(
    const FiberletCandidateResult& candidate,
    const FiberPredictionGridInfo& grid,
    int cellSize,
    const FiberletPathConfig& config,
    const FiberletPointPredicate& pointPredicate,
    SparseCornerBitmap& corners,
    PreparationProfile& profile)
{
    PreparedCandidate prepared;
    const auto geometryStart = Clock::now();
    prepared.domain = makeCurvedDomain(candidate, config);
    const SearchCorridor corridor = makeSearchCorridor(prepared.domain, grid, cellSize, config);
    prepared.keyLayout = makeLocalNodeKeyLayout(
        prepared.domain,
        corridor.radius,
        config.transverseStepPredictionVoxels);
    profile.geometrySeconds = std::chrono::duration<double>(
        Clock::now() - geometryStart).count();
    const auto nodeEnumerationStart = Clock::now();
    prepared.nodes = enumerateLocalNodes(
        prepared.domain,
        corridor,
        config,
        grid,
        pointPredicate,
        prepared.keyLayout,
        prepared.maximumActiveLayerNodes,
        profile);
    profile.retainedNodes = prepared.nodes.size();
    profile.nodeEnumerationSeconds = std::chrono::duration<double>(
        Clock::now() - nodeEnumerationStart).count();
    const auto cornerCollectionStart = Clock::now();
    addInterpolationCorners(
        candidate.startPositionPredictionXYZ, grid, corners,
        profile.interpolationCornerInsertions);
    addInterpolationCorners(
        candidate.targetPositionPredictionXYZ, grid, corners,
        profile.interpolationCornerInsertions);
    for (const auto& node : prepared.nodes)
        addInterpolationCorners(
            nodePoint(node), grid, corners,
            profile.interpolationCornerInsertions);
    profile.cornerCollectionSeconds = std::chrono::duration<double>(
        Clock::now() - cornerCollectionStart).count();
    return prepared;
}

struct FinalizedCorners {
    std::vector<Voxel> voxels;
    size_t mergedPages = 0;
    size_t peakTransientBytes = 0;
};

FinalizedCorners finalizeCornerSets(
    std::vector<SparseCornerBitmap>& cornerSets,
    size_t cornerSetBytes)
{
    FinalizedCorners result;
    if (cornerSets.empty())
        return result;
    struct StoredPageLess {
        bool operator()(const Voxel& left, const Voxel& right) const
        {
            return storedVoxelLess(left, right);
        }
    };
    std::map<Voxel, SparseCornerBitmap::Page, StoredPageLess> mergedPages;
    for (const auto& cornerSet : cornerSets) {
        for (const auto& [key, page] : cornerSet.pages()) {
            auto [found, inserted] = mergedPages.try_emplace(key);
            if (inserted) {
                found->second = page;
            } else {
                for (size_t word = 0; word < page.words.size(); ++word)
                    found->second.words[word] |= page.words[word];
            }
        }
    }
    result.mergedPages = mergedPages.size();
    const size_t mergedPageBytes = checkedProduct(
        mergedPages.size(),
        sizeof(std::pair<const Voxel, SparseCornerBitmap::Page>),
        "fiberlet merged corner page byte estimate");
    result.peakTransientBytes = checkedSum(
        cornerSetBytes, mergedPageBytes,
        "fiberlet corner finalization byte estimate");
    for (auto& cornerSet : cornerSets)
        cornerSet.clear();
    cornerSets.clear();
    cornerSets.shrink_to_fit();

    size_t uniqueVoxels = 0;
    for (const auto& [key, page] : mergedPages) {
        (void)key;
        for (const uint64_t word : page.words)
            uniqueVoxels = checkedSum(
                uniqueVoxels, static_cast<size_t>(std::popcount(word)),
                "fiberlet unique corner count");
    }
    result.voxels.reserve(uniqueVoxels);
    for (const auto& [key, page] : mergedPages) {
        for (size_t wordIndex = 0; wordIndex < page.words.size(); ++wordIndex) {
            uint64_t bits = page.words[wordIndex];
            while (bits != 0) {
                const size_t bit = static_cast<size_t>(std::countr_zero(bits));
                const size_t offset = wordIndex * 64 + bit;
                const int64_t x = static_cast<int64_t>(offset % SparseCornerBitmap::pageSize);
                const int64_t y = static_cast<int64_t>(
                    (offset / SparseCornerBitmap::pageSize) %
                    SparseCornerBitmap::pageSize);
                const int64_t z = static_cast<int64_t>(
                    offset /
                    (SparseCornerBitmap::pageSize * SparseCornerBitmap::pageSize));
                result.voxels.push_back({
                    key[0] * SparseCornerBitmap::pageSize + x,
                    key[1] * SparseCornerBitmap::pageSize + y,
                    key[2] * SparseCornerBitmap::pageSize + z,
                });
                bits &= bits - 1;
            }
        }
    }
    std::sort(result.voxels.begin(), result.voxels.end(), storedVoxelLess);
    result.peakTransientBytes = std::max(
        result.peakTransientBytes,
        checkedSum(
            mergedPageBytes,
            checkedProduct(
                result.voxels.capacity(), sizeof(Voxel),
                "fiberlet finalized corner byte estimate"),
            "fiberlet corner finalization byte estimate"));
    return result;
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

std::optional<std::array<uint8_t, 2>> encodeCompactAxis(cv::Vec3f axis)
{
    axis = normalized(axis);
    if (vectorLength(axis) <= kEpsilon || !finiteVector(axis))
        return std::nullopt;
    if (axis[2] < 0.0F)
        axis *= -1.0F;
    const auto encode = [](float component) {
        const long raw = std::lround(component * 127.0F + 128.0F);
        return static_cast<uint8_t>(std::clamp(raw, 0L, 255L));
    };
    return std::array<uint8_t, 2>{encode(axis[0]), encode(axis[1])};
}

cv::Vec3f decodeCompactAxis(const std::array<uint8_t, 2>& raw)
{
    const float x = (static_cast<float>(raw[0]) - 128.0F) / 127.0F;
    const float y = (static_cast<float>(raw[1]) - 128.0F) / 127.0F;
    return normalized({x, y, std::sqrt(std::max(0.0F, 1.0F - x * x - y * y))});
}

CompactNodeScoring compactNodeScoring(const ScoringVoxel& scoring)
{
    CompactNodeScoring compact;
    if (scoring.prediction.presenceValid &&
        std::isfinite(scoring.prediction.presence)) {
        const float presence = std::clamp(scoring.prediction.presence, 0.0F, 1.0F);
        compact.presence = static_cast<uint8_t>(std::lround(presence * 255.0F));
        compact.flags |= kNodePresenceValid;
    }
    if (scoring.prediction.valid) {
        const auto encoded = encodeCompactAxis(scoring.prediction.direction);
        if (encoded.has_value()) {
            compact.predictionAxis = *encoded;
            compact.flags |= kNodePredictionValid;
        }
    }
    if (scoring.normalValid) {
        const auto encoded = encodeCompactAxis(scoring.normal);
        if (encoded.has_value()) {
            compact.normalAxis = *encoded;
            compact.flags |= kNodeNormalValid;
        }
    }
    return compact;
}

DpNodeScoring prepareDpNodeScoring(const CompactNodeScoring& node)
{
    DpNodeScoring scoring;
    const float presence = static_cast<float>(node.presence) / 255.0F;
    scoring.flags = node.flags;
    if ((node.flags & kNodePredictionValid) != 0) {
        scoring.predictionAxis = decodeCompactAxis(node.predictionAxis);
        scoring.metricPrediction.direction =
            prepareFiberLocalUnitDirection(scoring.predictionAxis);
        scoring.metricPrediction.valid = true;
    }
    scoring.metricPrediction.presence = presence;
    if ((node.flags & kNodeNormalValid) != 0) {
        scoring.normalAxis = node.normalAxis;
        const cv::Vec3f normalAxis = decodeCompactAxis(node.normalAxis);
        scoring.metricNormal =
            prepareFiberLocalUnitDirection(normalAxis);
    }
    return scoring;
}

FiberletPredictionSample nodePrediction(const DpNodeScoring& node)
{
    return {
        node.predictionAxis,
        node.metricPrediction.presence,
        (node.flags & kNodePredictionValid) != 0,
        (node.flags & kNodePresenceValid) != 0,
    };
}

bool usablePrediction(const FiberletPredictionSample& prediction)
{
    const float normSquared = prediction.direction.dot(prediction.direction);
    return prediction.valid && finiteVector(prediction.direction) && std::isfinite(prediction.presence) && std::isfinite(normSquared) &&
           normSquared > kEpsilon;
}

bool withinPredictionDeviation(const cv::Vec3f& direction, const FiberletPredictionSample& prediction, float maximumDeviationRadians)
{
    if (!usablePrediction(prediction))
        return false;
    const cv::Vec3f unitDirection = normalized(direction);
    const cv::Vec3f unitPrediction = normalized(prediction.direction);
    if (vectorLength(unitDirection) <= kEpsilon || vectorLength(unitPrediction) <= kEpsilon) {
        return false;
    }
    return std::abs(unitDirection.dot(unitPrediction)) > std::cos(maximumDeviationRadians);
}

FiberLocalMetricSample localSample(const FiberletPredictionSample& sample)
{
    return {
        sample.direction,
        sample.presence,
        usablePrediction(sample),
    };
}

FiberLocalMetricConfig localMetricConfig(const FiberletPathConfig& config)
{
    return {
        config.invalidPredictionCostPerVoxel,
        FiberLocalSmoothnessConfig{
            config.smoothnessWeight,
            config.smoothnessNormalWeight,
            config.smoothnessTangentWeight,
            config.smoothnessFreeAngleDegrees * kPi / 180.0F,
        },
    };
}

FiberletPathCost fiberletPathCost(const FiberLocalMetricCost& local)
{
    FiberletPathCost cost;
    cost.invalidPrediction = local.invalidPrediction;
    cost.alignment = local.alignment;
    cost.isotropicSmoothness = local.isotropicSmoothness;
    cost.tangentSmoothness = local.tangentSmoothness;
    cost.normalSmoothness = local.normalSmoothness;
    return cost;
}

FiberletPathCost pathStepCost(
    const FiberletPredictionSample* currentPrediction,
    const FiberletPredictionSample& candidatePrediction,
    const cv::Vec3f& previousDirection,
    float previousLength,
    const cv::Vec3f& candidateDirection,
    float candidateLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberletPathConfig& config)
{
    const auto current = currentPrediction != nullptr ? std::make_optional(localSample(*currentPrediction)) : std::nullopt;
    const auto local = fiberLocalMetricCost(
        current.has_value() ? &*current : nullptr,
        localSample(candidatePrediction),
        previousDirection,
        previousLength,
        candidateDirection,
        candidateLength,
        normal,
        normalValid,
        localMetricConfig(config));
    return fiberletPathCost(local);
}

bool betterCost(float candidate, float current)
{
    return candidate < current;
}

uint32_t predecessorNode(
    size_t node,
    size_t incomingState,
    const std::vector<SearchNode>& nodes,
    const std::vector<uint32_t>& nodeIndex,
    const LocalNodeKeyLayout& layout,
    uint32_t missingNode)
{
    if (incomingState >= 9)
        throw std::logic_error("fiberlet source state has no predecessor node");
    const LocalNodeKey current = unpackLocalNodeKey(nodes[node].key, layout);
    if (current.layer == 0)
        throw std::logic_error("fiberlet interior state is in layer zero");
    const int deltaU = static_cast<int>(incomingState / 3) - 1;
    const int deltaV = static_cast<int>(incomingState % 3) - 1;
    const LocalNodeKey previous{
        current.layer - 1,
        current.transverseU - deltaU,
        current.transverseV - deltaV,
    };
    if (previous.transverseU < -layout.transverseLimit ||
        previous.transverseU > layout.transverseLimit ||
        previous.transverseV < -layout.transverseLimit ||
        previous.transverseV > layout.transverseLimit) {
        throw std::logic_error("fiberlet reached state derives an out-of-range predecessor");
    }
    const uint32_t found = nodeIndex[packLocalNodeKey(previous, layout)];
    if (found == missingNode)
        throw std::logic_error("fiberlet reached state derives a missing predecessor");
    return found;
}

DpIncoming incomingForState(
    size_t node,
    size_t state,
    const FiberletCandidateResult& candidate,
    const std::vector<SearchNode>& nodes,
    const std::vector<uint32_t>& nodeIndex,
    const LocalNodeKeyLayout& layout,
    uint32_t missingNode)
{
    const cv::Vec3f point = nodePoint(nodes[node]);
    const cv::Vec3f previousPoint = state == 9
        ? candidate.startPositionPredictionXYZ
        : nodePoint(nodes[predecessorNode(
              node, state, nodes, nodeIndex, layout, missingNode)]);
    const cv::Vec3f delta = point - previousPoint;
    const float length = vectorLength(delta);
    if (!(length > kEpsilon))
        throw std::logic_error("fiberlet reached state has a zero-length incoming edge");
    const cv::Vec3f direction = delta / length;
    return {
        direction,
        length,
        prepareFiberLocalUnitDirection(direction),
        static_cast<float>(length),
    };
}

class LazyNodeScoringCache {
public:
    LazyNodeScoringCache(
        const std::vector<SearchNode>& nodes,
        const FiberPredictionGridInfo& grid,
        const PagedScoringIndex& scoringIndex,
        const std::vector<PreparedScoringVoxel>& scoring,
        size_t candidateIndex,
        SolveProfile& profile)
        : nodes_(nodes),
          grid_(grid),
          scoringIndex_(scoringIndex),
          scoring_(scoring),
          candidateIndex_(candidateIndex),
          profile_(profile),
          nodeToCache_(nodes.size(), missing)
    {
        profile_.lazyCacheIndexBytes = checkedProduct(
            nodeToCache_.capacity(), sizeof(uint32_t),
            "fiberlet lazy scoring-cache index byte count");
    }

    ~LazyNodeScoringCache() noexcept
    {
        profile_.preparedNodes = cache_.size();
        profile_.preparedNodeBytes =
            cache_.capacity() * sizeof(DpNodeScoring);
        profile_.interpolationProfiledPoints = sampled_.points;
        profile_.interpolationProfiledCorners = sampled_.corners;
        profile_.interpolationProfiledPredictionIdentical =
            sampled_.predictionIdentical;
        profile_.interpolationProfiledNormalIdentical =
            sampled_.normalIdentical;
        profile_.interpolationProfiledPredictionPrincipalSolves =
            sampled_.predictionPrincipalSolves;
        profile_.interpolationProfiledNormalPrincipalSolves =
            sampled_.normalPrincipalSolves;
        profile_.interpolationProfiledLookupSeconds = sampled_.lookupSeconds;
        profile_.interpolationProfiledPredictionCornerSeconds =
            sampled_.predictionCornerSeconds;
        profile_.interpolationProfiledNormalCornerSeconds =
            sampled_.normalCornerSeconds;
        profile_.interpolationProfiledPredictionResolveSeconds =
            sampled_.predictionResolveSeconds;
        profile_.interpolationProfiledNormalResolveSeconds =
            sampled_.normalResolveSeconds;
        profile_.interpolationPredictionClosedFormResolutions =
            resolutions_.predictionClosedFormResolutions;
        profile_.interpolationNormalClosedFormResolutions =
            resolutions_.normalClosedFormResolutions;
        profile_.interpolationPredictionIterativeFallbacks =
            resolutions_.predictionIterativeFallbacks;
        profile_.interpolationNormalIterativeFallbacks =
            resolutions_.normalIterativeFallbacks;
    }

    uint32_t materialize(size_t node)
    {
        ++profile_.lazyNodeRequests;
        uint32_t& cached = nodeToCache_.at(node);
        if (cached != missing) {
            ++profile_.lazyNodeCacheHits;
            return cached;
        }
        if (cache_.size() >= static_cast<size_t>(missing))
            throw std::overflow_error("fiberlet lazy scoring cache exceeds 32 bits");
        InterpolationProfileSample* sampled =
            selectedForProfile(nodes_[node].key) ? &sampled_ : nullptr;
        auto lookup = scoringIndex_.lookup(
            scoring_, profile_.scoringPageDirectoryProbes);
        const ScoringVoxel interpolated = interpolateScoringPoint(
            nodePoint(nodes_[node]), grid_, lookup, sampled, &resolutions_);
        cached = static_cast<uint32_t>(cache_.size());
        cache_.push_back(prepareDpNodeScoring(compactNodeScoring(interpolated)));
        return cached;
    }

    const DpNodeScoring& at(uint32_t cached) const
    {
        if (cached >= cache_.size())
            throw std::logic_error("fiberlet lazy scoring cache index is invalid");
        return cache_[cached];
    }

    uint32_t existing(size_t node) const
    {
        const uint32_t cached = nodeToCache_.at(node);
        if (cached == missing)
            throw std::logic_error("fiberlet reached node has no cached scoring");
        return cached;
    }

private:
    bool selectedForProfile(uint32_t nodeKey) const noexcept
    {
        uint64_t value =
            (static_cast<uint64_t>(candidateIndex_) << 32U) | nodeKey;
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        return (value & 4095U) == 0;
    }

    static constexpr uint32_t missing =
        std::numeric_limits<uint32_t>::max();
    const std::vector<SearchNode>& nodes_;
    const FiberPredictionGridInfo& grid_;
    const PagedScoringIndex& scoringIndex_;
    const std::vector<PreparedScoringVoxel>& scoring_;
    size_t candidateIndex_ = 0;
    SolveProfile& profile_;
    std::vector<uint32_t> nodeToCache_;
    std::vector<DpNodeScoring> cache_;
    InterpolationProfileSample sampled_;
    InterpolationResolutionStats resolutions_;
};

FiberletCandidateResult solveCandidate(
    FiberletCandidateResult candidate,
    const FiberletPathConfig& config,
    const PreparedCandidate& prepared,
    const FiberPredictionGridInfo& grid,
    const PagedScoringIndex& scoringIndex,
    const std::vector<PreparedScoringVoxel>& scoring,
    size_t candidateIndex,
    SolveProfile& profile)
{
    candidate.searched = true;
    const cv::Vec3f chordVector = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
    const float chordLength = vectorLength(chordVector);
    if (!(chordLength > kEpsilon)) {
        candidate.reason = "zero_length";
        return candidate;
    }
    const float maximumAngle = config.maximumEndpointAngleDegrees * kPi / 180.0F;
    const float maximumPredictionDeviation = config.maximumPredictionDeviationDegrees * kPi / 180.0F;
    const FiberLocalMetricConfig metricConfig = localMetricConfig(config);
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

    FiberletPredictionSample targetProxy;
    targetProxy.direction = candidate.targetAxisXYZ;
    targetProxy.presence = 1.0F;
    targetProxy.valid = true;
    targetProxy.presenceValid = true;

    if (domain.layers.size() == 2) {
        const cv::Vec3f direction = chordVector / chordLength;
        if (directedAngle(candidate.startAxisXYZ, direction) > maximumAngle + kEpsilon ||
            directedAngle(direction, candidate.targetAxisXYZ) > maximumAngle + kEpsilon ||
            !withinPredictionDeviation(direction, targetScoring.prediction, maximumPredictionDeviation)) {
            candidate.reason = "no_path";
            return candidate;
        }
        candidate.cost =
            pathStepCost(nullptr, targetScoring.prediction, candidate.startAxisXYZ, chordLength, direction, chordLength, targetScoring.normal, targetScoring.normalValid, config);
        candidate.cost +=
            pathStepCost(&targetScoring.prediction, targetProxy, direction, chordLength, candidate.targetAxisXYZ, 0.0F, targetScoring.normal, targetScoring.normalValid, config);
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
    const auto nodeIndexStart = Clock::now();
    constexpr uint32_t missingNode = std::numeric_limits<uint32_t>::max();
    if (nodes.size() > static_cast<size_t>(missingNode))
        throw std::overflow_error("fiberlet DP node index exceeds 32 bits");
    std::vector<uint32_t> nodeIndex(prepared.keyLayout.keyCount, missingNode);
    for (size_t index = 0; index < nodes.size(); ++index) {
        auto& slot = nodeIndex[nodes[index].key];
        if (slot == missingNode)
            slot = static_cast<uint32_t>(index);
    }
    profile.nodeIndexEntries = nodes.size();
    profile.nodeIndexSlots = nodeIndex.size();
    profile.directIndexBytes = checkedProduct(
        nodeIndex.capacity(), sizeof(uint32_t),
        "fiberlet DP direct-index byte count");
    profile.nodeIndexSeconds = std::chrono::duration<double>(
        Clock::now() - nodeIndexStart).count();

    const auto nodePreparationStart = Clock::now();
    LazyNodeScoringCache dpNodes(
        nodes, grid, scoringIndex, scoring, candidateIndex, profile);
    profile.nodePreparationSeconds = std::chrono::duration<double>(
        Clock::now() - nodePreparationStart).count();

    const auto dpStart = Clock::now();
    constexpr size_t transitionStateCount = 9;
    constexpr size_t sourceState = transitionStateCount;
    constexpr size_t stateCount = transitionStateCount + 1;
    constexpr uint8_t missingState = std::numeric_limits<uint8_t>::max();
    std::vector<NodeRange> layerRanges(domain.layers.size());
    size_t rangeCursor = 0;
    for (size_t layer = 0; layer < layerRanges.size(); ++layer) {
        layerRanges[layer].begin = rangeCursor;
        while (rangeCursor < nodes.size()) {
            const LocalNodeKey key = unpackLocalNodeKey(
                nodes[rangeCursor].key, prepared.keyLayout);
            if (key.layer < layer)
                throw std::logic_error("fiberlet DP nodes are not layer ordered");
            if (key.layer != layer)
                break;
            ++rangeCursor;
        }
        layerRanges[layer].end = rangeCursor;
    }
    if (rangeCursor != nodes.size())
        throw std::logic_error("fiberlet DP node layer exceeds its domain");

    std::vector<uint8_t> previousStates(
        checkedProduct(nodes.size(), stateCount,
            "fiberlet DP backpointer-state count"),
        missingState);
    const size_t backpointerBytes = checkedProduct(
        previousStates.capacity(), sizeof(uint8_t),
        "fiberlet DP backpointer-state byte count");
    const size_t finalInteriorLayer = domain.layers.size() - 2;
    NodeRange currentRange = layerRanges[1];
    std::vector<DpLayerState> currentStates(
        checkedProduct(currentRange.size(), stateCount,
            "fiberlet DP layer-state count"));
    profile.stateBytes = checkedSum(
        backpointerBytes,
        checkedProduct(currentStates.capacity(), sizeof(DpLayerState),
            "fiberlet DP layer-state byte count"),
        "fiberlet DP state byte count");
    for (size_t node = currentRange.begin; node < currentRange.end; ++node) {
        const cv::Vec3f point = nodePoint(nodes[node]);
        const cv::Vec3f delta = point - candidate.startPositionPredictionXYZ;
        const float stepLength = vectorLength(delta);
        if (!(stepLength > kEpsilon))
            continue;
        const cv::Vec3f direction = delta / stepLength;
        if (directedAngle(candidate.startAxisXYZ, direction) > maximumAngle + kEpsilon) {
            continue;
        }
        const uint32_t scoringIndex = dpNodes.materialize(node);
        const auto& scoring = dpNodes.at(scoringIndex);
        if (!withinPredictionDeviation(
                direction, nodePrediction(scoring), maximumPredictionDeviation)) {
            continue;
        }
        const FiberletPredictionSample prediction = nodePrediction(scoring);
        auto& state = currentStates[
            (node - currentRange.begin) * stateCount + sourceState];
        state.reached = true;
        state.cost = dpAccumulatedCost(pathStepCost(
            nullptr, prediction, candidate.startAxisXYZ, stepLength,
            direction, stepLength, scoring.metricNormal,
            (scoring.flags & kNodeNormalValid) != 0, config));
    }

    for (size_t layer = 1; layer < finalInteriorLayer; ++layer) {
        const NodeRange nextRange = layerRanges[layer + 1];
        std::vector<DpLayerState> nextStates(
            checkedProduct(nextRange.size(), stateCount,
                "fiberlet DP layer-state count"));
        const size_t activeLayerBytes = checkedSum(
            checkedProduct(currentStates.capacity(), sizeof(DpLayerState),
                "fiberlet DP layer-state byte count"),
            checkedProduct(nextStates.capacity(), sizeof(DpLayerState),
                "fiberlet DP layer-state byte count"),
            "fiberlet DP rolling-state byte count");
        profile.stateBytes = std::max(
            profile.stateBytes,
            checkedSum(backpointerBytes, activeLayerBytes,
                "fiberlet DP state byte count"));

        for (size_t node = currentRange.begin; node < currentRange.end; ++node) {
            const LocalNodeKey currentKey =
                unpackLocalNodeKey(nodes[node].key, prepared.keyLayout);
            const cv::Vec3f currentPoint = nodePoint(nodes[node]);
            const size_t currentOffset =
                (node - currentRange.begin) * stateCount;
            size_t reachedStateCount = 0;
            for (size_t state = 0; state < stateCount; ++state) {
                reachedStateCount +=
                    currentStates[currentOffset + state].reached ? 1U : 0U;
            }
            if (reachedStateCount == 0)
                continue;
            ++profile.reachedNodes;
            const uint32_t currentScoringIndex = dpNodes.existing(node);

            std::array<DpEdge, transitionStateCount> outgoing;
            detail::FiberLocalPreparedCandidateAlignmentBatch<
                transitionStateCount> outgoingAlignment;
            size_t generatedEdges = 0;
            size_t validEdges = 0;
            for (int deltaU = -1; deltaU <= 1; ++deltaU) {
                for (int deltaV = -1; deltaV <= 1; ++deltaV) {
                    const size_t transitionState =
                        static_cast<size_t>((deltaU + 1) * 3 + (deltaV + 1));
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
                    ++generatedEdges;
                    const uint32_t found = nodeIndex[
                        packLocalNodeKey(nextKey, prepared.keyLayout)];
                    if (found == missingNode)
                        continue;
                    if (found < nextRange.begin || found >= nextRange.end)
                        throw std::logic_error("fiberlet DP edge leaves the next layer");
                    const cv::Vec3f delta = nodePoint(nodes[found]) - currentPoint;
                    const float stepLength = vectorLength(delta);
                    if (!(stepLength > kEpsilon))
                        continue;
                    const cv::Vec3f direction = delta / stepLength;
                    const uint32_t nextScoringIndex =
                        dpNodes.materialize(found);
                    if (!withinPredictionDeviation(
                            direction, nodePrediction(dpNodes.at(nextScoringIndex)),
                            maximumPredictionDeviation)) {
                        continue;
                    }
                    const auto& nextScoring = dpNodes.at(nextScoringIndex);
                    const cv::Vec3f metricDirection =
                        prepareFiberLocalUnitDirection(direction);
                    const auto candidateMetric =
                        detail::prepareFiberLocalCandidateMetricInline(
                            nextScoring.metricPrediction,
                            metricDirection,
                            nextScoring.metricNormal,
                            (nextScoring.flags & kNodeNormalValid) != 0);
                    outgoing[transitionState] = {
                        found,
                        nextScoringIndex,
                        stepLength,
                        candidateMetric,
                    };
                    detail::appendFiberLocalCandidateAlignmentInline(
                        outgoingAlignment,
                        static_cast<std::uint8_t>(transitionState),
                        candidateMetric);
                    ++validEdges;
                }
            }
            profile.generatedEdges += generatedEdges;
            profile.validEdges += validEdges;
            profile.reusedEdges += validEdges * (reachedStateCount - 1);
            const auto& currentScoring = dpNodes.at(currentScoringIndex);
            for (size_t previousState = 0; previousState < stateCount; ++previousState) {
                const auto& currentState =
                    currentStates[currentOffset + previousState];
                if (!currentState.reached)
                    continue;
                ++profile.reachedStateVisits;
                profile.transitionLookups += generatedEdges;
                const DpIncoming incoming = incomingForState(
                    node, previousState, candidate, nodes, nodeIndex,
                    prepared.keyLayout, missingNode);
                const auto incomingMetric =
                    detail::prepareFiberLocalIncomingAlignmentInline(
                        &currentScoring.metricPrediction,
                        incoming.metricDirection);
                std::array<float, transitionStateCount> alignmentLosses;
                detail::fiberLocalAlignmentLossPreparedBatchInline(
                    incomingMetric, outgoingAlignment, alignmentLosses);
                for (size_t lane = 0; lane < outgoingAlignment.count; ++lane) {
                    const size_t transitionState =
                        outgoingAlignment.slotOfLane[lane];
                    const auto& edge = outgoing[transitionState];
                    const size_t next = edge.next;
                    DpAccumulatedCost nextCost = currentState.cost;
                    nextCost +=
                        detail::fiberLocalMetricCostFromPreparedAlignmentInline(
                            alignmentLosses[lane],
                            incomingMetric,
                            incoming.metricLength,
                            edge.metricLength,
                            edge.candidateMetric,
                            metricConfig);
                    auto& destination = nextStates[
                        (next - nextRange.begin) * stateCount +
                        transitionState];
                    if (!destination.reached || betterCost(
                            nextCost.total(), destination.cost.total())) {
                        ++profile.relaxations;
                        destination.reached = true;
                        destination.cost = nextCost;
                        previousStates[next * stateCount + transitionState] =
                            static_cast<uint8_t>(previousState);
                    }
                }
            }
        }
        currentRange = nextRange;
        currentStates = std::move(nextStates);
    }

    bool foundPath = false;
    size_t bestNode = 0;
    size_t bestState = 0;
    FiberletPathCost bestCost;
    if (currentRange.begin != layerRanges[finalInteriorLayer].begin ||
        currentRange.end != layerRanges[finalInteriorLayer].end) {
        throw std::logic_error("fiberlet DP did not finish on its final layer");
    }
    for (size_t node = currentRange.begin; node < currentRange.end; ++node) {
        const cv::Vec3f point = nodePoint(nodes[node]);
        bool reached = false;
        for (size_t stateIndex = 0; stateIndex < stateCount; ++stateIndex) {
            reached = reached || currentStates[
                (node - currentRange.begin) * stateCount + stateIndex].reached;
        }
        if (!reached)
            continue;
        const uint32_t scoringIndex = dpNodes.existing(node);
        const FiberletPredictionSample prediction =
            nodePrediction(dpNodes.at(scoringIndex));
        const auto& nodeScoring = dpNodes.at(scoringIndex);
        for (size_t stateIndex = 0; stateIndex < stateCount; ++stateIndex) {
            const auto& state = currentStates[
                (node - currentRange.begin) * stateCount + stateIndex];
            if (!state.reached)
                continue;
            const DpIncoming incoming = incomingForState(
                node, stateIndex, candidate, nodes, nodeIndex,
                prepared.keyLayout, missingNode);
            const cv::Vec3f delta = candidate.targetPositionPredictionXYZ - point;
            const float finalLength = vectorLength(delta);
            if (!(finalLength > kEpsilon))
                continue;
            const cv::Vec3f finalDirection = delta / finalLength;
            if (directedAngle(finalDirection, candidate.targetAxisXYZ) > maximumAngle + kEpsilon) {
                continue;
            }
            FiberletPathCost finalized = fiberletPathCost(state.cost);
            finalized += pathStepCost(
                &prediction,
                targetProxy,
                incoming.direction,
                incoming.length,
                finalDirection,
                finalLength,
                nodeScoring.metricNormal,
                (nodeScoring.flags & kNodeNormalValid) != 0,
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
        profile.dpSeconds = std::chrono::duration<double>(
            Clock::now() - dpStart).count();
        candidate.reason = "no_path";
        return candidate;
    }

    std::vector<cv::Vec3f> reversed;
    size_t node = bestNode;
    size_t state = bestState;
    while (true) {
        reversed.push_back(nodePoint(nodes[node]));
        if (state == sourceState)
            break;
        const uint8_t previousState =
            previousStates[node * stateCount + state];
        if (previousState >= stateCount)
            throw std::logic_error("fiberlet DP state has an invalid predecessor state");
        node = predecessorNode(
            node, state, nodes, nodeIndex, prepared.keyLayout, missingNode);
        state = previousState;
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
    profile.dpSeconds = std::chrono::duration<double>(
        Clock::now() - dpStart).count();
    return candidate;
}

nlohmann::json anchorIdJson(const FiberletAnchorId& id)
{
    return {{"cell_zyx", id.cellZYX}, {"component", id.componentIndex}};
}

template <typename T>
nlohmann::json pointJson(const cv::Vec<T, 3>& point)
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

float fiberletPathLength(const FiberletCandidateResult& candidate)
{
    if (candidate.pointsPredictionXYZ.size() < 2)
        throw std::runtime_error("successful fiberlet has fewer than two path points");
    float length = 0.0F;
    for (size_t index = 1; index < candidate.pointsPredictionXYZ.size(); ++index) {
        if (!finiteVector(candidate.pointsPredictionXYZ[index - 1]) || !finiteVector(candidate.pointsPredictionXYZ[index])) {
            throw std::runtime_error("successful fiberlet has a non-finite path point");
        }
        const float segment = vectorLength(candidate.pointsPredictionXYZ[index] - candidate.pointsPredictionXYZ[index - 1]);
        if (!std::isfinite(segment))
            throw std::runtime_error("successful fiberlet has a non-finite path segment");
        length += segment;
    }
    if (!(length > 0.0F) || !std::isfinite(length))
        throw std::runtime_error("successful fiberlet has non-positive path length");
    return length;
}

}  // namespace

float FiberletPathCost::total() const noexcept
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
        if (std::any_of(componentLosses.begin(), componentLosses.end(), [](float value) { return !(value >= 0.0F) || !std::isfinite(value); })) {
            throw std::runtime_error("successful fiberlet has an invalid component loss");
        }
        const float totalLoss = candidate.cost.total();
        if (!(totalLoss >= 0.0) || !std::isfinite(totalLoss))
            throw std::runtime_error("successful fiberlet has invalid total loss");
        const float pathLength = fiberletPathLength(candidate);
        const float density = totalLoss / pathLength;
        if (!(density >= 0.0) || !std::isfinite(density))
            throw std::runtime_error("successful fiberlet has invalid loss density");
        visual.paths.push_back({
            candidateIndex,
            pathLength,
            totalLoss,
            density,
            0.0F,
        });
        visual.minimumLossPerPredictionVoxel =
            visual.minimumLossPerPredictionVoxel.has_value() ? std::min(*visual.minimumLossPerPredictionVoxel, density) : density;
        visual.maximumLossPerPredictionVoxel =
            visual.maximumLossPerPredictionVoxel.has_value() ? std::max(*visual.maximumLossPerPredictionVoxel, density) : density;
    }
    for (auto& path : visual.paths) {
        path.relativeQuality = *visual.minimumLossPerPredictionVoxel == *visual.maximumLossPerPredictionVoxel
                                   ? 1.0F
                                   : (*visual.maximumLossPerPredictionVoxel - path.lossPerPredictionVoxel) /
                                         (*visual.maximumLossPerPredictionVoxel - *visual.minimumLossPerPredictionVoxel);
        path.relativeQuality = std::clamp(path.relativeQuality, 0.0F, 1.0F);
    }
    return visual;
}

FiberletPathStatistics fiberletPathStatistics(const FiberletPathReport& report)
{
    FiberletPathStatistics statistics;
    statistics.anchors = report.diagnostics.occupiedAnchors;
    statistics.candidates = report.candidates.size();
    float allSum = 0.0F;
    float acceptedSum = 0.0F;
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
        const float score = candidate.cost.total();
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
        statistics.allScores.mean = allSum / static_cast<float>(statistics.allScores.count);
    if (statistics.acceptedScores.count > 0)
        statistics.acceptedScores.mean = acceptedSum / static_cast<float>(statistics.acceptedScores.count);
    const auto visual = fiberletPathVisualMetrics(report);
    float densitySum = 0.0F;
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
        statistics.acceptedLossDensities.mean = densitySum / static_cast<float>(statistics.acceptedLossDensities.count);
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
                float presence = 0.0F;
                if (samples[index].valid) {
                    if (!std::isfinite(samples[index].presence))
                        throw std::runtime_error("fiber presence sampler returned non-finite presence");
                    presence = static_cast<float>(std::clamp(samples[index].presence, 0.0, 1.0));
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
    const auto finiteNonnegative = [](float value) { return std::isfinite(value) && value >= 0.0F; };
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
    const int artifactVersion = root.is_object() ? root.value("version", 0) : 0;
    if (!root.is_object() ||
        root.value("format", "") != "vc_fiberlet_anchors" ||
        (artifactVersion != 1 && artifactVersion != 2)) {
        throw std::runtime_error(
            "fiber anchor artifact must be vc_fiberlet_anchors version 1 or 2");
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
    if (!detail::floatGridShapeExactlyRepresentable(
            loaded.report.grid.shapeZYX)) {
        throw std::runtime_error(
            "fiber anchor prediction grid is not exactly representable in float32");
    }
    loaded.report.grid.predictionToBaseScale = finiteDouble(coordinates.at("prediction_to_base_scale"), "prediction_to_base_scale");
    if (!(loaded.report.grid.predictionToBaseScale > 0.0))
        throw std::runtime_error("fiber anchor prediction-to-base scale must be positive");
    if (coordinates.contains("base_voxel_size_um")) {
        loaded.artifact.baseVoxelSizeUm = finiteDouble(coordinates.at("base_voxel_size_um"), "base_voxel_size_um");
        if (!(*loaded.artifact.baseVoxelSizeUm > 0.0))
            throw std::runtime_error("fiber anchor base voxel size must be positive");
    }
    const auto& selection = root.at("selection");
    const cv::Vec3d cropOriginBase = jsonVec3d(selection.at("prediction_interval_origin_base_xyz"), "prediction_interval_origin_base_xyz");
    const cv::Vec3d cropSizeBase = jsonVec3d(selection.at("prediction_interval_size_base_xyz"), "prediction_interval_size_base_xyz");
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
        "robust_maximum_trim_mass_fraction",
        "robust_mad_multiplier",
        "robust_minimum_angle_degrees",
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
    auto legacyParameterKeys = parameterKeys;
    legacyParameterKeys.erase("robust_maximum_trim_mass_fraction");
    legacyParameterKeys.erase("robust_mad_multiplier");
    legacyParameterKeys.erase("robust_minimum_angle_degrees");
    const bool hasRobustParameters = artifactVersion == 2;
    const auto& expectedParameterKeys =
        hasRobustParameters ? parameterKeys : legacyParameterKeys;
    if (storedParameterKeys != expectedParameterKeys) {
        throw std::runtime_error(
            "fiber anchor parameters do not match the version-" +
            std::to_string(artifactVersion) + " schema");
    }
    auto& config = loaded.report.config;
    config.cellSizePredictionVoxels = parameters.at("cell_size_prediction_voxels").get<int>();
    config.gaussianSigmaPredictionVoxels =
        finiteFloat(parameters.at("gaussian_sigma_prediction_voxels"), "gaussian_sigma_prediction_voxels");
    config.peakSigmaPredictionVoxels = finiteFloat(parameters.at("peak_sigma_prediction_voxels"), "peak_sigma_prediction_voxels");
    config.peakAxialSigmaPredictionVoxels =
        finiteFloat(parameters.at("peak_axial_sigma_prediction_voxels"), "peak_axial_sigma_prediction_voxels");
    config.peakGridStepPredictionVoxels =
        finiteFloat(parameters.at("peak_grid_step_prediction_voxels"), "peak_grid_step_prediction_voxels");
    config.peakGradientWeight = finiteFloat(parameters.at("peak_gradient_weight"), "peak_gradient_weight");
    config.peakGradientReliabilityScale = finiteFloat(parameters.at("peak_gradient_reliability_scale"), "peak_gradient_reliability_scale");
    config.gaussianCutoffSigmas = finiteFloat(parameters.at("gaussian_cutoff_sigmas"), "gaussian_cutoff_sigmas");
    config.localWindowRadiusPredictionVoxels =
        finiteFloat(parameters.at("local_window_radius_prediction_voxels"), "local_window_radius_prediction_voxels");
    config.axialSupportHalfWidthPredictionVoxels =
        finiteFloat(parameters.at("axial_support_half_width_prediction_voxels"), "axial_support_half_width_prediction_voxels");
    config.positionConvergenceTolerancePredictionVoxels =
        finiteFloat(parameters.at("position_convergence_tolerance_prediction_voxels"), "position_convergence_tolerance_prediction_voxels");
    config.nmsMaximumAngleDegrees = finiteFloat(parameters.at("nms_maximum_angle_degrees"), "nms_maximum_angle_degrees");
    config.nmsTransverseRadiusPredictionVoxels =
        finiteFloat(parameters.at("nms_transverse_radius_prediction_voxels"), "nms_transverse_radius_prediction_voxels");
    config.nmsLongitudinalRadiusPredictionVoxels =
        finiteFloat(parameters.at("nms_longitudinal_radius_prediction_voxels"), "nms_longitudinal_radius_prediction_voxels");
    config.observationPresenceFloor = finiteFloat(parameters.at("observation_presence_floor"), "observation_presence_floor");
    config.minimumAlignedSupport = finiteFloat(parameters.at("minimum_aligned_support"), "minimum_aligned_support");
    if (hasRobustParameters) {
        config.robustMaximumTrimMassFraction = finiteFloat(
            parameters.at("robust_maximum_trim_mass_fraction"),
            "robust_maximum_trim_mass_fraction");
        config.robustMadMultiplier = finiteFloat(
            parameters.at("robust_mad_multiplier"), "robust_mad_multiplier");
        config.robustMinimumAngleDegrees = finiteFloat(
            parameters.at("robust_minimum_angle_degrees"),
            "robust_minimum_angle_degrees");
    }
    config.mergeMaximumAngleDegrees = finiteFloat(parameters.at("merge_maximum_angle_degrees"), "merge_maximum_angle_degrees");
    config.mergeMaximumAbsoluteObjectiveLoss =
        finiteFloat(parameters.at("merge_maximum_absolute_objective_loss"), "merge_maximum_absolute_objective_loss");
    config.mergeMaximumRelativeObjectiveLoss =
        finiteFloat(parameters.at("merge_maximum_relative_objective_loss"), "merge_maximum_relative_objective_loss");
    config.maximumSeedCount = parameters.at("maximum_seed_count").get<size_t>();
    config.maximumIterations = parameters.at("maximum_iterations").get<int>();
    config.convergenceTolerance = finiteFloat(parameters.at("convergence_tolerance"), "convergence_tolerance");
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
        cell.objective = finiteFloat(cellJson.at("objective"), "cell objective");
        if (cellJson.contains("merge_evaluation")) {
            const auto& mergeJson = cellJson.at("merge_evaluation");
            FiberAnchorMergeEvaluation merge;
            merge.angleDegrees = finiteFloat(mergeJson.at("angle_degrees"), "merge angle_degrees");
            merge.jointObjective = finiteFloat(mergeJson.at("joint_objective"), "merge joint_objective");
            merge.splitObjective = finiteFloat(mergeJson.at("split_objective"), "merge split_objective");
            merge.objectiveLoss = finiteFloat(mergeJson.at("objective_loss"), "merge objective_loss");
            merge.allowedObjectiveLoss = finiteFloat(mergeJson.at("allowed_objective_loss"), "merge allowed_objective_loss");
            merge.merged = mergeJson.at("merged").get<bool>();
            if (merge.angleDegrees < 0.0 || merge.angleDegrees > 90.0 || merge.jointObjective < 0.0 || merge.jointObjective > 1.0 ||
                merge.splitObjective < 0.0 || merge.splitObjective > 1.0 || merge.objectiveLoss < 0.0 || merge.objectiveLoss > 1.0 ||
                merge.allowedObjectiveLoss < 0.0 || merge.allowedObjectiveLoss > 1.0) {
                throw std::runtime_error("fiber anchor merge evaluation is outside its valid range");
            }
            const float expectedLoss = std::max(0.0F, merge.splitObjective - merge.jointObjective);
            const float expectedAllowed =
                std::max(config.mergeMaximumAbsoluteObjectiveLoss, config.mergeMaximumRelativeObjectiveLoss * merge.jointObjective);
            const float tolerance = 1.0e-5F;
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
            const cv::Vec3d positionPrediction =
                jsonVec3d(componentJson.at("position_base_xyz"),
                    "position_base_xyz") /
                loaded.report.grid.predictionToBaseScale;
            component.anchor.positionPredictionXYZ =
                checkedVec3f(positionPrediction,
                    "position_base_xyz prediction coordinates");
            component.anchor.axisXYZ = jsonVec3f(componentJson.at("axis_xyz"), "axis_xyz");
            const float axisLength = vectorLength(component.anchor.axisXYZ);
            if (std::abs(axisLength - 1.0F) > 1.0e-5F)
                throw std::runtime_error("fiber anchor axis must be unit length");
            component.anchor.alignedSupport = finiteFloat(componentJson.at("aligned_support"), "aligned_support");
            component.anchor.directionalCoherence = finiteFloat(componentJson.at("directional_coherence"), "directional_coherence");
            component.anchor.refinementScore = finiteFloat(componentJson.at("refinement_score"), "refinement_score");
            component.anchor.refinementIterations = componentJson.at("refinement_iterations").get<size_t>();
            if (component.anchor.alignedSupport < 0.0 || component.anchor.alignedSupport > 1.0 || component.anchor.directionalCoherence < 0.0 ||
                component.anchor.directionalCoherence > 1.0 || component.anchor.refinementScore < 0.0 || component.anchor.refinementScore > 1.0 ||
                std::abs(component.anchor.refinementScore - component.anchor.alignedSupport) > 1.0e-5F ||
                component.anchor.refinementIterations > static_cast<size_t>(config.maximumIterations)) {
                throw std::runtime_error("fiber anchor refinement values are inconsistent");
            }
            const cv::Vec3f pivot{
                (static_cast<float>(cell.cellZYX[2] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<float>(
                     std::min((cell.cellZYX[2] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[2])) -
                 1.0F) *
                    0.5F,
                (static_cast<float>(cell.cellZYX[1] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<float>(
                     std::min((cell.cellZYX[1] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[1])) -
                 1.0F) *
                    0.5F,
                (static_cast<float>(cell.cellZYX[0] * static_cast<size_t>(config.cellSizePredictionVoxels)) +
                 static_cast<float>(
                     std::min((cell.cellZYX[0] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[0])) -
                 1.0F) *
                    0.5F,
            };
            const cv::Vec3f pivotOffset = component.anchor.positionPredictionXYZ - pivot;
            const float planeResidual = std::abs(pivotOffset.dot(component.anchor.axisXYZ));
            const float pivotDistance = vectorLength(pivotOffset);
            for (int axis = 0; axis < 3; ++axis) {
                const float position = component.anchor.positionPredictionXYZ[axis];
                const size_t gridAxis = static_cast<size_t>(2 - axis);
                if (position < -kEpsilon || position > static_cast<float>(loaded.report.grid.shapeZYX[gridAxis] - 1) + kEpsilon) {
                    throw std::runtime_error("fiber anchor position is outside the prediction grid");
                }
                const float cellBegin = static_cast<float>(cell.cellZYX[gridAxis] * static_cast<size_t>(config.cellSizePredictionVoxels));
                const float cellEnd = static_cast<float>(
                    std::min((cell.cellZYX[gridAxis] + 1) * static_cast<size_t>(config.cellSizePredictionVoxels), loaded.report.grid.shapeZYX[gridAxis]));
                const float ownerLower = cellBegin == 0.0F ? 0.0F : cellBegin - 0.5F;
                const float ownerUpper = cellEnd == static_cast<float>(loaded.report.grid.shapeZYX[gridAxis])
                                              ? cellEnd - 1.0F
                                              : std::nextafter(cellEnd - 0.5F, -std::numeric_limits<float>::infinity());
                if (position < ownerLower - kEpsilon || position > ownerUpper + kEpsilon) {
                    throw std::runtime_error("fiber anchor position is outside its owning cell");
                }
            }
            if (planeResidual > 1.0e-5F || pivotDistance > config.localWindowRadiusPredictionVoxels + 1.0e-5F) {
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

std::vector<std::array<int, 3>> fiberletCellNeighborhoodOffsets(int radius, float margin)
{
    if (radius < 1 || !(margin > 0.0) || !std::isfinite(margin))
        throw std::invalid_argument("fiberlet cell neighborhood requires positive radius and margin");
    const int limit = static_cast<int>(std::ceil(radius + margin));
    const float upper = static_cast<float>(radius) + margin;
    std::vector<std::array<int, 3>> offsets;
    for (int z = -limit; z <= limit; ++z) {
        for (int y = -limit; y <= limit; ++y) {
            for (int x = -limit; x <= limit; ++x) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const float length = std::sqrt(static_cast<float>(x * x + y * y + z * z));
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
    if (!detail::floatGridShapeExactlyRepresentable(grid.shapeZYX)) {
        throw std::invalid_argument(
            "fiberlet prediction grid is not exactly representable in float32");
    }
    if (grid.shapeZYX != anchors.report.grid.shapeZYX || std::abs(grid.predictionToBaseScale - anchors.report.grid.predictionToBaseScale) > 1.0e-12) {
        throw std::invalid_argument("fiberlet prediction grid does not match anchor artifact");
    }
    FiberletPathReport report;
    report.grid = grid;
    report.anchorCellSizePredictionVoxels = anchors.report.config.cellSizePredictionVoxels;
    report.config = inputConfig;
    if (!(report.config.corridorRadiusPredictionVoxels > 0.0F)) {
        report.config.corridorRadiusPredictionVoxels = static_cast<float>(report.anchorCellSizePredictionVoxels);
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
    const float minimumAxisDot = std::cos(report.config.maximumEndpointAngleDegrees * kPi / 180.0F);
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
                const cv::Vec3f chordVector = candidate.targetPositionPredictionXYZ - candidate.startPositionPredictionXYZ;
                const float distance = vectorLength(chordVector);
                ++report.diagnostics.generatedPairs;
                if (!(distance > kEpsilon)) {
                    candidate.reason = "zero_length";
                    ++report.diagnostics.zeroLengthPairs;
                    report.candidates.push_back(std::move(candidate));
                    continue;
                }
                const cv::Vec3f chord = chordVector / distance;
                candidate.startAxisXYZ = normalized(source.anchor.axisXYZ);
                candidate.targetAxisXYZ = normalized(target->anchor.axisXYZ);
                bool endpointsSelected = true;
                if (pointPredicate) {
                    ++report.candidatePointPredicateCalls;
                    endpointsSelected =
                        pointPredicate(candidate.startPositionPredictionXYZ);
                    if (endpointsSelected) {
                        ++report.candidatePointPredicateCalls;
                        endpointsSelected = pointPredicate(
                            candidate.targetPositionPredictionXYZ);
                    }
                }
                if (!endpointsSelected) {
                    candidate.reason = "outside_selection";
                    report.candidates.push_back(std::move(candidate));
                    continue;
                }
                if (candidate.startAxisXYZ.dot(chord) < 0.0F)
                    candidate.startAxisXYZ *= -1.0F;
                if (candidate.targetAxisXYZ.dot(chord) < 0.0F)
                    candidate.targetAxisXYZ *= -1.0F;
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
    std::vector<PreparationProfile> preparationProfiles(
        searchCandidateIndices.size());
    std::vector<SparseCornerBitmap> workerCorners(workerCount);
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
                    report.candidates[searchCandidateIndices[searchIndex]], grid,
                    report.anchorCellSizePredictionVoxels, report.config,
                    pointPredicate, corners, preparationProfiles[searchIndex]);
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
    for (size_t index = 0; index < prepared.size(); ++index) {
        const auto& item = prepared[index];
        const auto& profile = preparationProfiles[index];
        report.evaluatedDpNodes = checkedSum(
            report.evaluatedDpNodes, checkedSum(item.nodes.size(), 2, "fiberlet evaluated node count"), "fiberlet evaluated node count");
        report.latticeNodePositions = checkedSum(
            report.latticeNodePositions, profile.latticeNodePositions,
            "fiberlet lattice node count");
        report.corridorSegmentTests = checkedSum(
            report.corridorSegmentTests, profile.corridorSegmentTests,
            "fiberlet corridor segment test count");
        report.corridorAcceptedNodes = checkedSum(
            report.corridorAcceptedNodes, profile.corridorAcceptedNodes,
            "fiberlet corridor accepted node count");
        report.nodePointPredicateCalls = checkedSum(
            report.nodePointPredicateCalls, profile.pointPredicateCalls,
            "fiberlet node predicate call count");
        report.retainedSearchNodes = checkedSum(
            report.retainedSearchNodes, profile.retainedNodes,
            "fiberlet retained node count");
        report.interpolationCornerInsertions = checkedSum(
            report.interpolationCornerInsertions,
            profile.interpolationCornerInsertions,
            "fiberlet interpolation corner insertion count");
        report.preparationGeometryWorkSeconds += profile.geometrySeconds;
        report.preparationNodeEnumerationWorkSeconds +=
            profile.nodeEnumerationSeconds;
        report.preparationCornerCollectionWorkSeconds +=
            profile.cornerCollectionSeconds;
    }
    size_t preparedBytes = preparedPayloadBytes(prepared);
    report.preparedGeometryBytes = preparedBytes;
    size_t maximumSearchTransientBytes = 0;
    constexpr size_t stateCount = 10;
    for (const auto& item : prepared) {
        const size_t backpointerBytes = checkedProduct(
            checkedProduct(item.nodes.size(), stateCount,
                "fiberlet DP backpointer byte estimate"),
            sizeof(uint8_t), "fiberlet DP backpointer byte estimate");
        const size_t activeStateBytes = checkedProduct(
            checkedProduct(item.maximumActiveLayerNodes, stateCount,
                "fiberlet DP rolling-state byte estimate"),
            sizeof(DpLayerState), "fiberlet DP rolling-state byte estimate");
        const size_t stateBytes = checkedSum(
            backpointerBytes, activeStateBytes,
            "fiberlet DP state byte estimate");
        const size_t nodeIndexBytes = checkedProduct(
            item.keyLayout.keyCount, sizeof(uint32_t),
            "fiberlet DP index byte estimate");
        const size_t preparedNodeBytes = checkedProduct(
            item.nodes.size(), sizeof(DpNodeScoring),
            "fiberlet prepared DP-node byte estimate");
        const size_t lazyCacheIndexBytes = checkedProduct(
            item.nodes.size(), sizeof(uint32_t),
            "fiberlet lazy scoring-cache index byte estimate");
        maximumSearchTransientBytes = std::max(
            maximumSearchTransientBytes,
            checkedSum(
                checkedSum(
                    checkedSum(stateBytes, nodeIndexBytes,
                        "fiberlet search transient byte estimate"),
                    lazyCacheIndexBytes,
                    "fiberlet search transient byte estimate"),
                preparedNodeBytes,
                "fiberlet search transient byte estimate"));
    }
    report.peakSearchTransientBytes = checkedProduct(
        maximumSearchTransientBytes, workerCount,
        "fiberlet concurrent search transient byte estimate");
    size_t workerCornerBytes = 0;
    for (const auto& corners : workerCorners) {
        report.cornerWorkerUniqueVoxels = checkedSum(
            report.cornerWorkerUniqueVoxels, corners.uniqueVoxels(),
            "fiberlet worker unique corner count");
        report.cornerWorkerPages = checkedSum(
            report.cornerWorkerPages, corners.pageCount(),
            "fiberlet worker corner page count");
        report.cornerPageDirectoryProbes = checkedSum(
            report.cornerPageDirectoryProbes, corners.directoryProbes(),
            "fiberlet corner page directory probe count");
        report.cornerSamePageHits = checkedSum(
            report.cornerSamePageHits, corners.samePageHits(),
            "fiberlet corner same-page hit count");
        report.cornerCachedPageHits = checkedSum(
            report.cornerCachedPageHits, corners.cachedPageHits(),
            "fiberlet corner cached-page hit count");
        workerCornerBytes = checkedSum(
            workerCornerBytes,
            corners.payloadBytes(),
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
    auto finalizedCorners = finalizeCornerSets(
        workerCorners, workerCornerBytes);
    report.estimatedPeakOwnedBytes = std::
        max(report.estimatedPeakOwnedBytes,
            checkedSum(
                preparedBytes, finalizedCorners.peakTransientBytes,
                "fiberlet peak owned byte estimate"));
    std::vector<Voxel> orderedVoxels = std::move(finalizedCorners.voxels);
    report.cornerMergedPages = finalizedCorners.mergedPages;
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
        for (size_t index = 0; index < samples.size(); ++index) {
            const auto& sample = samples[index];
            FiberletPredictionSample stored;
            stored.valid = sample.valid;
            stored.presenceValid = sample.presenceValid;
            if (sample.valid) {
                stored.direction = cv::Vec3f(sample.direction);
                if (!std::isfinite(sample.direction[0]) ||
                    !std::isfinite(sample.direction[1]) ||
                    !std::isfinite(sample.direction[2]) ||
                    !finiteVector(stored.direction)) {
                    throw std::runtime_error(
                        "fiberlet prediction direction is not finite float32");
                }
            }
            if (std::isfinite(sample.presence)) {
                stored.presence = static_cast<float>(sample.presence);
                if (!std::isfinite(stored.presence)) {
                    throw std::runtime_error(
                        "fiberlet prediction presence is not finite float32");
                }
            } else {
                stored.presence = std::numeric_limits<float>::quiet_NaN();
            }
            scoringVoxels[begin + index].prediction = stored;
        }
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
            points.emplace_back(nativeVoxelPoint(orderedVoxels[index]));
        std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
        (void)normalSampler.sampleNormalBatch(points, false, report.config.parallelThreads, samples);
        if (samples.size() != points.size())
            throw std::runtime_error("fiberlet normal sampler returned the wrong coordinate batch sample count");
        for (size_t index = 0; index < samples.size(); ++index) {
            const auto& sample = samples[index].sample;
            if (sample.valid) {
                const cv::Vec3f normal(sample.normal);
                if (!std::isfinite(sample.normal[0]) ||
                    !std::isfinite(sample.normal[1]) ||
                    !std::isfinite(sample.normal[2]) || !finiteVector(normal)) {
                    throw std::runtime_error(
                        "fiberlet normal is not finite float32");
                }
                scoringVoxels[begin + index].normal = normal;
            }
            scoringVoxels[begin + index].normalValid = sample.valid;
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
    const auto scoringPreparationStart = Clock::now();
    const double scoringPreparationCpuStart = processCpuSeconds();
    std::vector<PreparedScoringVoxel> preparedScoringVoxels;
    preparedScoringVoxels.reserve(scoringVoxels.size());
    for (const auto& scoring : scoringVoxels)
        preparedScoringVoxels.push_back(prepareScoringVoxel(scoring));
    report.scoringPreparationSeconds = std::chrono::duration<double>(
        Clock::now() - scoringPreparationStart).count();
    report.scoringPreparationCpuSeconds =
        processCpuSeconds() - scoringPreparationCpuStart;
    const size_t preparedScoringArrayBytes = checkedSum(
        checkedProduct(
            orderedVoxels.capacity(), sizeof(Voxel),
            "fiberlet prepared scoring byte estimate"),
        checkedProduct(
            preparedScoringVoxels.capacity(), sizeof(PreparedScoringVoxel),
            "fiberlet prepared scoring byte estimate"),
        "fiberlet prepared scoring byte estimate");
    report.estimatedPeakOwnedBytes = std::max(
        report.estimatedPeakOwnedBytes,
        checkedSum(
            checkedSum(preparedBytes, sampledArrayBytes,
                "fiberlet scoring preparation byte estimate"),
            checkedProduct(
                preparedScoringVoxels.capacity(), sizeof(PreparedScoringVoxel),
                "fiberlet scoring preparation byte estimate"),
            "fiberlet scoring preparation byte estimate"));
    scoringVoxels.clear();
    scoringVoxels.shrink_to_fit();
    const auto scoringIndexStart = Clock::now();
    const double scoringIndexCpuStart = processCpuSeconds();
    PagedScoringIndex scoringIndex(orderedVoxels);
    report.scoringPageCount = scoringIndex.pageCount();
    report.scoringPageSlots = scoringIndex.slotCount();
    report.scoringIndexSeconds = std::chrono::duration<double>(
        Clock::now() - scoringIndexStart).count();
    report.scoringIndexCpuSeconds =
        processCpuSeconds() - scoringIndexCpuStart;
    const size_t scoringIndexPayloadBytes = scoringIndex.payloadBytes();
    const size_t retainedPreparedScoringBytes = checkedProduct(
        preparedScoringVoxels.capacity(), sizeof(PreparedScoringVoxel),
        "fiberlet retained prepared-scoring byte estimate");
    report.dpSharedScoringBytes = checkedSum(
        retainedPreparedScoringBytes, scoringIndexPayloadBytes,
        "fiberlet shared search-scoring byte estimate");
    report.estimatedPeakOwnedBytes = std::
        max(report.estimatedPeakOwnedBytes,
            checkedSum(
                checkedSum(
                    preparedBytes, preparedScoringArrayBytes,
                    "fiberlet peak owned byte estimate"),
                scoringIndexPayloadBytes,
                "fiberlet peak owned byte estimate"));
    errors.assign(prepared.size(), {});
    report.endpointScoringInterpolations = checkedProduct(
        prepared.size(), 2, "fiberlet endpoint interpolation count");
    report.interpolatedScoringPoints = report.endpointScoringInterpolations;
    std::atomic<size_t> nextMaterialization{0};
    std::atomic<size_t> completedMaterialization{0};
    std::vector<size_t> pageDirectoryProbes(workerCount);
    std::vector<InterpolationProfileSample> interpolationProfiles(workerCount);
    std::vector<InterpolationResolutionStats> interpolationResolutionStats(
        workerCount);
    const auto materializationWorker = [&](size_t workerIndex) {
        size_t localPageDirectoryProbes = 0;
        size_t localInterpolationCount = 0;
        while (true) {
            const size_t searchIndex = nextMaterialization.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= prepared.size())
                break;
            try {
                auto& item = prepared[searchIndex];
                const auto interpolate = [&](const cv::Vec3f& point) {
                    auto lookup = scoringIndex.lookup(
                        preparedScoringVoxels, localPageDirectoryProbes);
                    InterpolationProfileSample* profile = nullptr;
                    if ((localInterpolationCount++ & 4095U) == 0)
                        profile = &interpolationProfiles[workerIndex];
                    return interpolateScoringPoint(
                        point, grid, lookup, profile,
                        &interpolationResolutionStats[workerIndex]);
                };
                const auto& candidate =
                    report.candidates[searchCandidateIndices[searchIndex]];
                item.startScoring = interpolate(
                    candidate.startPositionPredictionXYZ);
                item.targetScoring = interpolate(
                    candidate.targetPositionPredictionXYZ);
            } catch (...) {
                errors[searchIndex] = std::current_exception();
            }
            const size_t completed = completedMaterialization.fetch_add(1, std::memory_order_relaxed) + 1;
            reportProgress("materialization", completed, prepared.size(), materializationStart, false);
        }
        pageDirectoryProbes[workerIndex] = localPageDirectoryProbes;
    };
    const auto interpolationMaterializationStart = Clock::now();
    const double interpolationMaterializationCpuStart = processCpuSeconds();
    if (workerCount == 1) {
        materializationWorker(0);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(materializationWorker, index);
        for (auto& thread : workers)
            thread.join();
    }
    reportProgress("materialization", prepared.size(), prepared.size(), materializationStart, true);
    report.interpolationMaterializationSeconds =
        std::chrono::duration<double>(
            Clock::now() - interpolationMaterializationStart).count();
    report.interpolationMaterializationCpuSeconds =
        processCpuSeconds() - interpolationMaterializationCpuStart;
    for (const size_t probes : pageDirectoryProbes) {
        report.scoringPageDirectoryProbes = checkedSum(
            report.scoringPageDirectoryProbes, probes,
            "fiberlet scoring page probe count");
    }
    for (const auto& profile : interpolationProfiles) {
        report.interpolationProfiledPoints += profile.points;
        report.interpolationProfiledCorners += profile.corners;
        report.interpolationProfiledPredictionIdentical += profile.predictionIdentical;
        report.interpolationProfiledNormalIdentical += profile.normalIdentical;
        report.interpolationProfiledPredictionPrincipalSolves +=
            profile.predictionPrincipalSolves;
        report.interpolationProfiledNormalPrincipalSolves +=
            profile.normalPrincipalSolves;
        report.interpolationProfiledLookupSeconds += profile.lookupSeconds;
        report.interpolationProfiledPredictionCornerSeconds +=
            profile.predictionCornerSeconds;
        report.interpolationProfiledNormalCornerSeconds +=
            profile.normalCornerSeconds;
        report.interpolationProfiledPredictionResolveSeconds +=
            profile.predictionResolveSeconds;
        report.interpolationProfiledNormalResolveSeconds +=
            profile.normalResolveSeconds;
    }
    for (const auto& stats : interpolationResolutionStats) {
        report.interpolationPredictionClosedFormResolutions +=
            stats.predictionClosedFormResolutions;
        report.interpolationNormalClosedFormResolutions +=
            stats.normalClosedFormResolutions;
        report.interpolationPredictionIterativeFallbacks +=
            stats.predictionIterativeFallbacks;
        report.interpolationNormalIterativeFallbacks +=
            stats.normalIterativeFallbacks;
    }
    report.samplingMaterializationSeconds = std::chrono::duration<double>(Clock::now() - materializationStart).count();
    report.samplingMaterializationCpuSeconds = processCpuSeconds() - materializationCpuStart;
    for (const auto& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
    orderedVoxels.clear();
    orderedVoxels.shrink_to_fit();
    report.estimatedPeakOwnedBytes = std::max(
        report.estimatedPeakOwnedBytes,
        checkedSum(
            checkedSum(preparedBytes, report.dpSharedScoringBytes,
                "fiberlet peak lazy-search byte estimate"),
            report.peakSearchTransientBytes,
            "fiberlet peak lazy-search byte estimate"));

    const auto searchStart = Clock::now();
    const double searchCpuStart = processCpuSeconds();
    reportProgress("search", 0, searchCandidateIndices.size(), searchStart, true);
    errors.assign(searchCandidateIndices.size(), {});
    std::vector<SolveProfile> solveProfiles(searchCandidateIndices.size());
    std::atomic<size_t> nextSearch{0};
    std::atomic<size_t> completedSearches{0};
    const auto searchWorker = [&]() {
        while (true) {
            const size_t searchIndex = nextSearch.fetch_add(1, std::memory_order_relaxed);
            if (searchIndex >= searchCandidateIndices.size())
                return;
            const size_t candidateIndex = searchCandidateIndices[searchIndex];
            try {
                report.candidates[candidateIndex] = solveCandidate(
                    report.candidates[candidateIndex], report.config,
                    prepared[searchIndex], grid, scoringIndex,
                    preparedScoringVoxels, searchIndex,
                    solveProfiles[searchIndex]);
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
    for (const auto& profile : solveProfiles) {
        report.dpNodeIndexEntries = checkedSum(
            report.dpNodeIndexEntries, profile.nodeIndexEntries,
            "fiberlet DP node-index entry count");
        report.dpNodeIndexSlots = checkedSum(
            report.dpNodeIndexSlots, profile.nodeIndexSlots,
            "fiberlet DP node-index slot count");
        report.dpPreparedNodes = checkedSum(
            report.dpPreparedNodes, profile.preparedNodes,
            "fiberlet prepared DP-node count");
        report.lazyNodeScoringMaterializations = checkedSum(
            report.lazyNodeScoringMaterializations, profile.preparedNodes,
            "fiberlet lazy node materialization count");
        report.interpolatedScoringPoints = checkedSum(
            report.interpolatedScoringPoints, profile.preparedNodes,
            "fiberlet interpolated scoring point count");
        report.lazyNodeScoringRequests = checkedSum(
            report.lazyNodeScoringRequests, profile.lazyNodeRequests,
            "fiberlet lazy node scoring request count");
        report.lazyNodeScoringCacheHits = checkedSum(
            report.lazyNodeScoringCacheHits, profile.lazyNodeCacheHits,
            "fiberlet lazy node scoring cache-hit count");
        report.dpMaximumPreparedNodeBytes = std::max(
            report.dpMaximumPreparedNodeBytes, profile.preparedNodeBytes);
        report.dpMaximumLazyCacheIndexBytes = std::max(
            report.dpMaximumLazyCacheIndexBytes,
            profile.lazyCacheIndexBytes);
        report.dpMaximumDirectIndexBytes = std::max(
            report.dpMaximumDirectIndexBytes, profile.directIndexBytes);
        report.dpMaximumStateBytes = std::max(
            report.dpMaximumStateBytes, profile.stateBytes);
        report.dpReachedNodes = checkedSum(
            report.dpReachedNodes, profile.reachedNodes,
            "fiberlet reached DP-node count");
        report.dpGeneratedEdges = checkedSum(
            report.dpGeneratedEdges, profile.generatedEdges,
            "fiberlet generated DP-edge count");
        report.dpValidEdges = checkedSum(
            report.dpValidEdges, profile.validEdges,
            "fiberlet valid DP-edge count");
        report.dpReusedEdges = checkedSum(
            report.dpReusedEdges, profile.reusedEdges,
            "fiberlet reused DP-edge count");
        report.dpTransitionLookups = checkedSum(
            report.dpTransitionLookups, profile.transitionLookups,
            "fiberlet DP transition lookup count");
        report.dpReachedStateVisits = checkedSum(
            report.dpReachedStateVisits, profile.reachedStateVisits,
            "fiberlet DP reached-state visit count");
        report.dpRelaxations = checkedSum(
            report.dpRelaxations, profile.relaxations,
            "fiberlet DP relaxation count");
        report.scoringPageDirectoryProbes = checkedSum(
            report.scoringPageDirectoryProbes,
            profile.scoringPageDirectoryProbes,
            "fiberlet scoring page probe count");
        report.interpolationProfiledPoints +=
            profile.interpolationProfiledPoints;
        report.interpolationProfiledCorners +=
            profile.interpolationProfiledCorners;
        report.interpolationProfiledPredictionIdentical +=
            profile.interpolationProfiledPredictionIdentical;
        report.interpolationProfiledNormalIdentical +=
            profile.interpolationProfiledNormalIdentical;
        report.interpolationProfiledPredictionPrincipalSolves +=
            profile.interpolationProfiledPredictionPrincipalSolves;
        report.interpolationProfiledNormalPrincipalSolves +=
            profile.interpolationProfiledNormalPrincipalSolves;
        report.interpolationPredictionClosedFormResolutions +=
            profile.interpolationPredictionClosedFormResolutions;
        report.interpolationNormalClosedFormResolutions +=
            profile.interpolationNormalClosedFormResolutions;
        report.interpolationPredictionIterativeFallbacks +=
            profile.interpolationPredictionIterativeFallbacks;
        report.interpolationNormalIterativeFallbacks +=
            profile.interpolationNormalIterativeFallbacks;
        report.interpolationProfiledLookupSeconds +=
            profile.interpolationProfiledLookupSeconds;
        report.interpolationProfiledPredictionCornerSeconds +=
            profile.interpolationProfiledPredictionCornerSeconds;
        report.interpolationProfiledNormalCornerSeconds +=
            profile.interpolationProfiledNormalCornerSeconds;
        report.interpolationProfiledPredictionResolveSeconds +=
            profile.interpolationProfiledPredictionResolveSeconds;
        report.interpolationProfiledNormalResolveSeconds +=
            profile.interpolationProfiledNormalResolveSeconds;
        report.searchNodeIndexWorkSeconds += profile.nodeIndexSeconds;
        report.searchNodePreparationWorkSeconds +=
            profile.nodePreparationSeconds;
        report.searchDpWorkSeconds += profile.dpSeconds;
    }
    scoringIndex.clear();
    preparedScoringVoxels.clear();
    preparedScoringVoxels.shrink_to_fit();
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
    const double predictionToBaseScale = report.grid.predictionToBaseScale;
    if (!(predictionToBaseScale > 0.0) ||
        !std::isfinite(predictionToBaseScale) ||
        predictionToBaseScale > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(
            "fiberlet artifact scale is not finite float32");
    }
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
             {"dp_longitudinal_step_base_voxels", detail::checkedScaleFloatValue(report.config.longitudinalStepPredictionVoxels, predictionToBaseScale, "fiberlet base longitudinal step")},
             {"dp_transverse_step_prediction_voxels", report.config.transverseStepPredictionVoxels},
             {"dp_transverse_step_base_voxels", detail::checkedScaleFloatValue(report.config.transverseStepPredictionVoxels, predictionToBaseScale, "fiberlet base transverse step")},
             {"corridor_radius_base_voxels", detail::checkedScaleFloatValue(report.config.corridorRadiusPredictionVoxels, predictionToBaseScale, "fiberlet base corridor radius")},
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
            {"start_position_base_xyz", pointJson(detail::checkedScaleFloatPosition(
                candidate.startPositionPredictionXYZ, predictionToBaseScale,
                "fiberlet start base position"))},
            {"target_position_base_xyz", pointJson(detail::checkedScaleFloatPosition(
                candidate.targetPositionPredictionXYZ, predictionToBaseScale,
                "fiberlet target base position"))},
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
            item["path_length_base_voxels"] = detail::checkedScaleFloatValue(
                metrics->pathLengthPredictionVoxels, predictionToBaseScale,
                "fiberlet base path length");
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
                item["points_base_xyz"].push_back(pointJson(
                    detail::checkedScaleFloatPosition(
                        point, predictionToBaseScale,
                        "fiberlet base path position")));
        }
        root["candidates"].push_back(std::move(item));
    }
    return root;
}

std::string fiberletPathReportObj(const FiberletPathReport& report)
{
    const auto visual = fiberletPathVisualMetrics(report);
    const double predictionToBaseScale = report.grid.predictionToBaseScale;
    if (!(predictionToBaseScale > 0.0) ||
        !std::isfinite(predictionToBaseScale) ||
        predictionToBaseScale > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(
            "fiberlet OBJ scale is not finite float32");
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
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
            const cv::Vec3f base = detail::checkedScaleFloatPosition(
                point, predictionToBaseScale, "fiberlet OBJ base position");
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

#ifdef VC_TESTING
namespace testing
{

FiberletCorridorContainmentDebug debugFiberletCorridorContains(
    const cv::Vec3f& point,
    const std::vector<cv::Vec3f>& reference,
    float radius,
    std::optional<size_t> adjacentSegment)
{
    if (!(radius > 0.0f) || !std::isfinite(radius) || reference.size() < 2)
        throw std::invalid_argument("fiberlet corridor test input is invalid");
    FiberletCorridorContainmentDebug result;
    result.inside = insideCorridor(
        point, reference, radius * radius, adjacentSegment,
        result.segmentTests);
    return result;
}

std::vector<std::array<int64_t, 3>> debugFinalizeFiberletCornerSets(
    const std::vector<std::vector<std::array<int64_t, 3>>>& cornerSets)
{
    std::vector<SparseCornerBitmap> uniqueSets(cornerSets.size());
    size_t cornerSetBytes = 0;
    for (size_t index = 0; index < cornerSets.size(); ++index) {
        for (const auto& corner : cornerSets[index])
            uniqueSets[index].insert(corner);
        cornerSetBytes = checkedSum(
            cornerSetBytes, uniqueSets[index].payloadBytes(),
            "fiberlet test corner byte estimate");
    }
    return finalizeCornerSets(uniqueSets, cornerSetBytes)
        .voxels;
}

}  // namespace testing
#endif

}  // namespace vc::fiber_tracer
