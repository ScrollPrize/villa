#include "vc/fiber_tracer/FiberTrace.hpp"

#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vc::fiber_tracer {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

using TraceClock = std::chrono::steady_clock;

[[nodiscard]] double elapsedSeconds(TraceClock::time_point start)
{
    return std::chrono::duration<double>(TraceClock::now() - start).count();
}

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] cv::Vec3d normalizedOr(const cv::Vec3d& v, const cv::Vec3d& fallback)
{
    const double len = length(v);
    if (!(len > kEpsilon) || !std::isfinite(len))
        return fallback;
    return v / len;
}

[[nodiscard]] cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    return normalizedOr(v, {0.0, 0.0, 0.0});
}

[[nodiscard]] double clamp01(double value)
{
    if (!std::isfinite(value))
        return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

[[nodiscard]] double clampSignedUnit(double value)
{
    if (!std::isfinite(value))
        return 0.0;
    return std::clamp(value, -1.0, 1.0);
}

[[nodiscard]] double clampedPositiveDot(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return clamp01(normalizedOrZero(a).dot(normalizedOrZero(b)));
}

[[nodiscard]] double angleBetweenUnit(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return std::acos(clampSignedUnit(normalizedOrZero(a).dot(normalizedOrZero(b))));
}

[[nodiscard]] cv::Vec3d alignTo(const cv::Vec3d& direction, const cv::Vec3d& reference)
{
    cv::Vec3d out = normalizedOrZero(direction);
    const cv::Vec3d ref = normalizedOrZero(reference);
    if (length(out) <= kEpsilon)
        return out;
    if (length(ref) > kEpsilon && out.dot(ref) < 0.0)
        out *= -1.0;
    return out;
}

[[nodiscard]] bool endsWith(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::vector<std::string> fiberPredictionPrefixes(
    const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    std::vector<std::string> prefixes;
    if (manifest.groupForChannel("presence") != nullptr &&
        manifest.groupForChannel("nx") != nullptr &&
        manifest.groupForChannel("ny") != nullptr) {
        prefixes.push_back({});
    }
    for (const auto& group : manifest.groups) {
        for (const auto& channel : group.channels) {
            constexpr std::string_view suffix = "_presence";
            if (!endsWith(channel, suffix))
                continue;
            const std::string prefix = channel.substr(0, channel.size() - suffix.size());
            if (manifest.groupForChannel(prefix + "_nx") != nullptr &&
                manifest.groupForChannel(prefix + "_ny") != nullptr) {
                prefixes.push_back(prefix);
            }
        }
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    return prefixes;
}

[[nodiscard]] std::array<std::string, 3> predictionChannelNames(
    const std::string& prefix)
{
    if (prefix.empty())
        return {"presence", "nx", "ny"};
    return {prefix + "_presence", prefix + "_nx", prefix + "_ny"};
}

[[nodiscard]] const vc::lasagna::LasagnaChannelGroup& predictionChannelGroup(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    const std::string& channel)
{
    const auto* group = manifest.groupForChannel(channel);
    if (group == nullptr) {
        throw std::runtime_error(
            "fiber inference dataset is missing required channel '" +
            channel + "'");
    }
    return *group;
}

[[nodiscard]] double predictionChannelEffectiveScale(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    const std::string& channel)
{
    const auto& group = predictionChannelGroup(manifest, channel);
    const double scale =
        manifest.sourceToBase * static_cast<double>(group.scaleFactor());
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::runtime_error(
            "fiber inference channel '" + channel +
            "' has a non-positive or non-finite effective scale");
    }
    return scale;
}

[[nodiscard]] bool nearlySameScale(double a, double b)
{
    const double tolerance =
        1.0e-9 * std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tolerance;
}

[[nodiscard]] std::array<cv::Vec3d, 2> orthonormalBasis(const cv::Vec3d& direction)
{
    const cv::Vec3d axis = normalizedOr(direction, {1.0, 0.0, 0.0});
    std::array<cv::Vec3d, 3> refs = {
        cv::Vec3d{1.0, 0.0, 0.0},
        cv::Vec3d{0.0, 1.0, 0.0},
        cv::Vec3d{0.0, 0.0, 1.0},
    };
    size_t refIndex = 0;
    double bestAbsDot = std::abs(axis.dot(refs[0]));
    for (size_t index = 1; index < refs.size(); ++index) {
        const double value = std::abs(axis.dot(refs[index]));
        if (value < bestAbsDot) {
            bestAbsDot = value;
            refIndex = index;
        }
    }
    const cv::Vec3d b0 = normalizedOrZero(axis.cross(refs[refIndex]));
    const cv::Vec3d b1 = normalizedOrZero(axis.cross(b0));
    return {b0, b1};
}

struct ConeOffset {
    double u = 0.0;
    double v = 0.0;
    double radius2 = 0.0;
    size_t order = 0;
};

[[nodiscard]] std::vector<ConeOffset> angleStepConeOffsets(
    double maxAngleDegrees,
    double angleStepDegrees)
{
    const double maxAngle = std::max(0.0, maxAngleDegrees);
    if (maxAngle <= 0.0)
        return {{0.0, 0.0, 0.0, 0}};
    const double step = angleStepDegrees;
    if (!std::isfinite(step) || step <= 0.0)
        throw std::invalid_argument("cone angle step must be positive");

    const int maxSteps = static_cast<int>(std::floor(maxAngle / step + 1.0e-6));
    std::vector<ConeOffset> offsets;
    bool hasCenter = false;
    size_t order = 0;
    for (int vStep = -maxSteps; vStep <= maxSteps; ++vStep) {
        for (int uStep = -maxSteps; uStep <= maxSteps; ++uStep) {
            const double uDeg = static_cast<double>(uStep) * step;
            const double vDeg = static_cast<double>(vStep) * step;
            const double radius2 = uDeg * uDeg + vDeg * vDeg;
            if (radius2 > maxAngle * maxAngle + 1.0e-5)
                continue;
            if (uDeg == 0.0 && vDeg == 0.0)
                hasCenter = true;
            offsets.push_back({uDeg, vDeg, radius2, order++});
        }
    }
    if (!hasCenter)
        offsets.push_back({0.0, 0.0, 0.0, order++});
    std::sort(offsets.begin(), offsets.end(), [](const auto& a, const auto& b) {
        if (a.radius2 != b.radius2)
            return a.radius2 < b.radius2;
        if (a.u != b.u)
            return a.u < b.u;
        if (a.v != b.v)
            return a.v < b.v;
        return a.order < b.order;
    });
    for (auto& offset : offsets) {
        offset.u = std::tan(offset.u * kPi / 180.0);
        offset.v = std::tan(offset.v * kPi / 180.0);
    }
    return offsets;
}

[[nodiscard]] std::vector<ConeOffset> legacyGridConeOffsets(
    double maxAngleDegrees,
    int gridSize)
{
    const double maxAngle = std::max(0.0, maxAngleDegrees) * kPi / 180.0;
    if (gridSize <= 0)
        throw std::invalid_argument("cone grid size must be positive");
    if (maxAngle <= 0.0 || gridSize == 1)
        return {{0.0, 0.0, 0.0, 0}};

    std::vector<ConeOffset> offsets;
    offsets.reserve(static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize));
    const double tangentScale = std::tan(maxAngle);
    size_t centerIndex = 0;
    double centerRadius2 = std::numeric_limits<double>::infinity();
    for (int y = 0; y < gridSize; ++y) {
        const double b = gridSize == 1
            ? 0.0
            : -1.0 + 2.0 * static_cast<double>(y) / static_cast<double>(gridSize - 1);
        for (int x = 0; x < gridSize; ++x) {
            const double a = gridSize == 1
                ? 0.0
                : -1.0 + 2.0 * static_cast<double>(x) / static_cast<double>(gridSize - 1);
            double diskX = 0.0;
            double diskY = 0.0;
            if (a != 0.0 || b != 0.0) {
                double r = 0.0;
                double phi = 0.0;
                if (std::abs(a) > std::abs(b)) {
                    r = a;
                    phi = (kPi / 4.0) * b / a;
                } else {
                    r = b;
                    phi = kPi / 2.0 - (kPi / 4.0) * a / b;
                }
                diskX = r * std::cos(phi);
                diskY = r * std::sin(phi);
            }
            const double radius2 = diskX * diskX + diskY * diskY;
            if (radius2 < centerRadius2) {
                centerRadius2 = radius2;
                centerIndex = offsets.size();
            }
            offsets.push_back({
                tangentScale * diskX,
                tangentScale * diskY,
                radius2,
                offsets.size(),
            });
        }
    }
    if (centerIndex < offsets.size() && centerIndex != 0) {
        const ConeOffset center = offsets[centerIndex];
        offsets.erase(offsets.begin() + static_cast<std::ptrdiff_t>(centerIndex));
        offsets.insert(offsets.begin(), center);
    }
    return offsets;
}

[[nodiscard]] std::vector<cv::Vec3d> candidateDirections(
    const cv::Vec3d& reference,
    const std::vector<ConeOffset>& offsets)
{
    const cv::Vec3d forward = normalizedOr(reference, {1.0, 0.0, 0.0});
    const auto basis = orthonormalBasis(forward);
    std::vector<cv::Vec3d> out;
    out.reserve(offsets.size());
    for (const auto& offset : offsets)
        out.push_back(normalizedOr(
            forward + basis[0] * offset.u + basis[1] * offset.v,
            forward));
    if (out.empty())
        out.push_back(forward);
    return out;
}

[[nodiscard]] std::vector<cv::Vec3d> candidateDirections(
    const cv::Vec3d& reference,
    const FiberTraceConfig& config)
{
    const auto offsets = config.coneAngleStepDegrees > 0.0
        ? angleStepConeOffsets(config.coneAngleDegrees, config.coneAngleStepDegrees)
        : legacyGridConeOffsets(config.coneAngleDegrees, config.coneGridSize);
    return candidateDirections(reference, offsets);
}

[[nodiscard]] std::vector<double> arclengths(const std::vector<cv::Vec3d>& points)
{
    std::vector<double> out(points.size(), 0.0);
    for (size_t i = 1; i < points.size(); ++i)
        out[i] = out[i - 1] + length(points[i] - points[i - 1]);
    return out;
}

[[nodiscard]] double pointToPlaneSigned(
    const cv::Vec3d& point,
    const cv::Vec3d& planePoint,
    const cv::Vec3d& planeNormal)
{
    return (point - planePoint).dot(planeNormal);
}

[[nodiscard]] double endpointPlaneError(
    const cv::Vec3d& point,
    const cv::Vec3d& target,
    const cv::Vec3d& planeNormal)
{
    const cv::Vec3d delta = point - target;
    return length(delta - planeNormal * delta.dot(planeNormal));
}

[[nodiscard]] bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) &&
           std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

[[nodiscard]] cv::Vec3d pointFromJson(
    const nlohmann::json& value,
    const std::filesystem::path& path,
    const char* key)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(
            "fiber JSON point in '" + std::string(key) +
            "' must be a three-element array: " + path.string());
    }
    cv::Vec3d point{
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>(),
    };
    if (!finitePoint(point)) {
        throw std::runtime_error(
            "fiber JSON point in '" + std::string(key) +
            "' contains non-finite coordinates: " + path.string());
    }
    return point;
}

[[nodiscard]] std::vector<cv::Vec3d> pointArrayFromJson(
    const nlohmann::json& root,
    const std::filesystem::path& path,
    const char* key)
{
    if (!root.contains(key) || !root.at(key).is_array()) {
        throw std::runtime_error(
            "fiber JSON is missing array '" + std::string(key) +
            "': " + path.string());
    }
    std::vector<cv::Vec3d> points;
    points.reserve(root.at(key).size());
    for (const auto& value : root.at(key)) {
        points.push_back(pointFromJson(value, path, key));
    }
    return points;
}

[[nodiscard]] bool pointsExactlyEqual(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

[[nodiscard]] size_t exactLineIndexForControlPoint(
    const std::vector<cv::Vec3d>& line,
    const cv::Vec3d& control,
    size_t controlIndex,
    const std::filesystem::path& path)
{
    for (size_t index = 0; index < line.size(); ++index) {
        if (pointsExactlyEqual(line[index], control)) {
            return index;
        }
    }
    throw std::runtime_error(
        "fiber JSON control point " + std::to_string(controlIndex) +
        " is not an exact line point; refusing to guess line arc position: " +
        path.string());
}

[[nodiscard]] std::vector<cv::Vec3d> scaledPoints(
    const std::vector<cv::Vec3d>& points,
    double divisor)
{
    std::vector<cv::Vec3d> out;
    out.reserve(points.size());
    for (const auto& point : points)
        out.push_back(point / divisor);
    return out;
}

[[nodiscard]] double referenceLengthMeters(
    double referenceLengthWorkingVoxels,
    double workingToBaseScale,
    double voxelSizeUm)
{
    if (!(referenceLengthWorkingVoxels > 0.0) ||
        !(workingToBaseScale > 0.0) ||
        !(voxelSizeUm > 0.0)) {
        return 0.0;
    }
    return referenceLengthWorkingVoxels * workingToBaseScale * voxelSizeUm * 1.0e-6;
}

[[nodiscard]] cv::Vec3d terminalTraceDirection(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& fallback)
{
    if (points.size() < 2)
        return normalizedOr(fallback, {1.0, 0.0, 0.0});
    for (size_t offset = 1; offset < points.size(); ++offset) {
        const size_t index = points.size() - 1 - offset;
        const cv::Vec3d direction = normalizedOrZero(points.back() - points[index]);
        if (length(direction) > kEpsilon)
            return direction;
    }
    return normalizedOr(fallback, {1.0, 0.0, 0.0});
}

struct ScoredDirection {
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double presence = 0.0;
    bool valid = false;
};

[[nodiscard]] ScoredDirection bestAlignedPrediction(
    const FiberPredictionSample& sample,
    const cv::Vec3d& referenceDirection,
    bool weightByPresence)
{
    ScoredDirection best;
    double bestScore = -std::numeric_limits<double>::infinity();
    const cv::Vec3d reference = normalizedOrZero(referenceDirection);
    for (const auto& option : sample.options) {
        if (!option.valid)
            continue;
        const cv::Vec3d direction = alignTo(option.direction, referenceDirection);
        double score = clamp01(direction.dot(reference));
        const double presence = clamp01(option.presence);
        if (weightByPresence)
            score *= presence;
        if (score > bestScore) {
            bestScore = score;
            best = {direction, presence, true};
        }
    }
    return best;
}

[[nodiscard]] ScoredDirection bestAlignedPrediction(
    const FiberPredictionSource& predictions,
    const cv::Vec3d& point,
    const cv::Vec3d& referenceDirection,
    bool weightByPresence)
{
    return bestAlignedPrediction(
        predictions.sample(point, referenceDirection),
        referenceDirection,
        weightByPresence);
}

[[nodiscard]] bool normalAwareSmoothnessEnabled(const FiberTraceConfig& config)
{
    return config.smoothnessNormalWeight > 0.0 ||
           config.smoothnessTangentWeight > 0.0 ||
           config.cumulativeSmoothnessTangentWeight > 0.0;
}

void requireNormalSamplerForNormalAwareSmoothness(
    const FiberTraceConfig& config,
    const vc::lasagna::NormalSampler* normalSampler)
{
    if (normalSampler != nullptr || !normalAwareSmoothnessEnabled(config))
        return;
    throw std::invalid_argument(
        "Lasagna normal sampler is required for tangent/normal fiber trace smoothness");
}

void validateTraceConfig(const FiberTraceConfig& config)
{
    auto requireFinite = [](double value, const char* name) {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(name) + " must be finite");
    };
    requireFinite(config.stepVoxels, "step voxels");
    requireFinite(config.coneAngleDegrees, "cone angle degrees");
    requireFinite(config.coneAngleStepDegrees, "cone angle step degrees");
    requireFinite(config.beamPruneDistanceVoxels, "beam prune distance");
    if (config.parallelThreads < 0)
        throw std::invalid_argument("parallel threads must be non-negative");
    requireFinite(config.smoothnessWeight, "smoothness weight");
    requireFinite(config.smoothnessNormalWeight, "smoothness normal weight");
    requireFinite(config.smoothnessTangentWeight, "smoothness tangent weight");
    requireFinite(config.smoothnessFreeAngleDegrees, "smoothness free angle");
    requireFinite(
        config.cumulativeSmoothnessTangentWeight,
        "cumulative smoothness tangent weight");
    requireFinite(config.maxStepFactor, "max step factor");
    requireFinite(config.fusionGapFactor, "fusion gap factor");
    requireFinite(
        config.endpointAcceptThresholdBaseVoxels,
        "endpoint accept threshold in base voxels");
    requireFinite(config.traceToBaseScale, "trace-to-base scale");
    if (config.baseVoxelSizeUm.has_value())
        requireFinite(*config.baseVoxelSizeUm, "base voxel size");
    if (!(config.stepVoxels > 0.0))
        throw std::invalid_argument("step voxels must be positive");
    if (config.coneAngleDegrees < 0.0)
        throw std::invalid_argument("cone angle degrees must be non-negative");
    if (config.coneGridSize < 1)
        throw std::invalid_argument("cone grid size must be at least 1");
    if (config.beamWidth < 1)
        throw std::invalid_argument("beam width must be at least 1");
    if (config.beamPruneDistanceVoxels < 0.0)
        throw std::invalid_argument("beam prune distance must be non-negative");
    if (config.beamLookaheadSteps < 1)
        throw std::invalid_argument("beam lookahead steps must be at least 1");
    if (config.smoothnessWeight < 0.0 ||
        config.smoothnessNormalWeight < 0.0 ||
        config.smoothnessTangentWeight < 0.0 ||
        config.cumulativeSmoothnessTangentWeight < 0.0) {
        throw std::invalid_argument("smoothness weights must be non-negative");
    }
    if (config.smoothnessFreeAngleDegrees < 0.0)
        throw std::invalid_argument("smoothness free angle must be non-negative");
    if (config.cumulativeSmoothnessSteps < 1)
        throw std::invalid_argument("cumulative smoothness steps must be at least 1");
    if (config.maxStepFactor < 0.0)
        throw std::invalid_argument("max step factor must be non-negative");
    if (config.endpointAcceptThresholdBaseVoxels < 0.0)
        throw std::invalid_argument(
            "endpoint accept threshold in base voxels must be non-negative");
    if (!(config.traceToBaseScale > 0.0))
        throw std::invalid_argument("trace-to-base scale must be positive");
    if (config.baseVoxelSizeUm.has_value() && !(*config.baseVoxelSizeUm > 0.0))
        throw std::invalid_argument("base voxel size must be positive when provided");
}

[[nodiscard]] double excessAngleSquared(double angle, double freeAngle)
{
    return std::pow(std::max(0.0, angle - freeAngle), 2.0);
}

[[nodiscard]] double isotropicSmoothnessLoss(
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& candidateStepDirection,
    const FiberTraceConfig& config)
{
    const double freeAngle = config.smoothnessFreeAngleDegrees * kPi / 180.0;
    return config.smoothnessWeight *
           excessAngleSquared(
               angleBetweenUnit(previousStepDirection, candidateStepDirection),
               freeAngle);
}

[[nodiscard]] double smoothnessLoss(
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& candidateStepDirection,
    const cv::Vec3d& normal,
    bool normalValid,
    const FiberTraceConfig& config)
{
    const cv::Vec3d prev = normalizedOrZero(previousStepDirection);
    const cv::Vec3d cand = normalizedOrZero(candidateStepDirection);
    if (length(prev) <= kEpsilon || length(cand) <= kEpsilon)
        return 0.0;

    const double isotropic = isotropicSmoothnessLoss(prev, cand, config);
    const cv::Vec3d n = normalizedOrZero(normal);
    const double normalWeight = config.smoothnessNormalWeight;
    const double tangentWeight = config.smoothnessTangentWeight;
    if (!normalValid || length(n) <= kEpsilon)
        return isotropic;

    const double prevN = clampSignedUnit(prev.dot(n));
    const double candN = clampSignedUnit(cand.dot(n));
    const cv::Vec3d prevT = normalizedOrZero(prev - n * prevN);
    const cv::Vec3d candT = normalizedOrZero(cand - n * candN);
    const bool tangentOk = length(prevT) > kEpsilon && length(candT) > kEpsilon;
    const double tangentAngle = tangentOk
        ? angleBetweenUnit(prevT, candT)
        : angleBetweenUnit(prev, cand);
    const double normalAngle = std::abs(std::asin(candN) - std::asin(prevN));
    const double freeAngle = config.smoothnessFreeAngleDegrees * kPi / 180.0;
    return tangentWeight * excessAngleSquared(tangentAngle, freeAngle) +
           normalWeight * excessAngleSquared(normalAngle, freeAngle);
}

[[nodiscard]] double cumulativeTangentSmoothnessLoss(
    const cv::Vec3d& historyDirection,
    const cv::Vec3d& candidateStepDirection,
    const cv::Vec3d& normal,
    bool normalValid,
    const FiberTraceConfig& config)
{
    const double weight = config.cumulativeSmoothnessTangentWeight;
    if (!(weight > 0.0))
        return 0.0;
    const cv::Vec3d history = normalizedOrZero(historyDirection);
    const cv::Vec3d cand = normalizedOrZero(candidateStepDirection);
    const cv::Vec3d n = normalizedOrZero(normal);
    if (!normalValid ||
        length(history) <= kEpsilon ||
        length(cand) <= kEpsilon ||
        length(n) <= kEpsilon) {
        return 0.0;
    }
    const double historyN = clampSignedUnit(history.dot(n));
    const double candN = clampSignedUnit(cand.dot(n));
    const cv::Vec3d historyT = normalizedOrZero(history - n * historyN);
    const cv::Vec3d candT = normalizedOrZero(cand - n * candN);
    if (length(historyT) <= kEpsilon || length(candT) <= kEpsilon)
        return 0.0;
    const double freeAngle = config.smoothnessFreeAngleDegrees * kPi / 180.0;
    return weight * excessAngleSquared(angleBetweenUnit(historyT, candT), freeAngle);
}

[[nodiscard]] cv::Vec3d updateHistoryDirection(
    const cv::Vec3d& historyDirection,
    const cv::Vec3d& chosenDirection,
    int depth,
    int cumulativeSmoothnessSteps)
{
    const cv::Vec3d chosen = normalizedOrZero(chosenDirection);
    if (depth <= 0 || cumulativeSmoothnessSteps <= 1)
        return chosen;
    const double count = static_cast<double>(
        std::clamp(depth, 1, cumulativeSmoothnessSteps - 1));
    return normalizedOr(chosen + normalizedOrZero(historyDirection) * count, chosen);
}

struct BeamState {
    struct PathNode {
        cv::Vec3d point{0.0, 0.0, 0.0};
        std::shared_ptr<const PathNode> previous;
        size_t length = 1;
    };

    std::shared_ptr<const PathNode> path;
    cv::Vec3d previousStepDirection{0.0, 0.0, 0.0};
    cv::Vec3d currentSampleDirection{0.0, 0.0, 0.0};
    cv::Vec3d historyDirection{0.0, 0.0, 0.0};
    double loss = 0.0;
    double tracedLength = 0.0;
    int depth = 0;
    bool reached = false;
    std::string reason;
};

[[nodiscard]] std::shared_ptr<const BeamState::PathNode> appendBeamPathPoint(
    std::shared_ptr<const BeamState::PathNode> previous,
    const cv::Vec3d& point)
{
    auto node = std::make_shared<BeamState::PathNode>();
    node->point = point;
    node->previous = std::move(previous);
    node->length = node->previous ? node->previous->length + 1 : 1;
    return node;
}

[[nodiscard]] cv::Vec3d beamEndpoint(const BeamState& state)
{
    return state.path ? state.path->point : cv::Vec3d{0.0, 0.0, 0.0};
}

[[nodiscard]] size_t beamPointCount(const BeamState& state)
{
    return state.path ? state.path->length : 0;
}

[[nodiscard]] std::vector<cv::Vec3d> beamPathPoints(const BeamState& state)
{
    std::vector<cv::Vec3d> out;
    out.reserve(beamPointCount(state));
    for (auto node = state.path; node; node = node->previous)
        out.push_back(node->point);
    std::reverse(out.begin(), out.end());
    return out;
}

struct CandidateScore {
    double loss = std::numeric_limits<double>::infinity();
    cv::Vec3d selectedCurrentDirection{0.0, 0.0, 0.0};
    double selectedPresence = 0.0;
    bool valid = false;
};

struct CandidateTask {
    size_t beamIndex = 0;
    cv::Vec3d direction{0.0, 0.0, 0.0};
    cv::Vec3d candidatePoint{0.0, 0.0, 0.0};
};

struct FrontierCandidate {
    size_t beamIndex = 0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d previousStepDirection{0.0, 0.0, 0.0};
    cv::Vec3d currentSampleDirection{0.0, 0.0, 0.0};
    cv::Vec3d historyDirection{0.0, 0.0, 0.0};
    double loss = 0.0;
    double tracedLength = 0.0;
    int depth = 0;
    bool reached = false;
    size_t order = 0;
};

struct CandidateScoringScratch {
    std::vector<CandidateScore> scores;
    std::vector<cv::Vec3d> candidatePoints;
    std::vector<cv::Vec3d> referenceDirections;
    std::vector<FiberPredictionSample> predictionSamples;
    std::vector<vc::lasagna::NormalSampleWithDerivative> normalSamplesWithDerivative;
    std::vector<vc::lasagna::NormalSample> normalSamples;
    std::vector<FrontierCandidate> frontierCandidates;
};

void appendCandidateTasks(
    std::vector<CandidateTask>& tasks,
    size_t beamIndex,
    const BeamState& beam,
    const std::vector<ConeOffset>& offsets,
    double step)
{
    const cv::Vec3d currentPoint = beamEndpoint(beam);
    const cv::Vec3d forward = normalizedOr(
        beam.currentSampleDirection, {1.0, 0.0, 0.0});
    const auto basis = orthonormalBasis(forward);
    if (offsets.empty()) {
        tasks.push_back({beamIndex, forward, currentPoint + forward * step});
        return;
    }
    for (const auto& offset : offsets) {
        const cv::Vec3d direction = normalizedOr(
            forward + basis[0] * offset.u + basis[1] * offset.v,
            forward);
        tasks.push_back({beamIndex, direction, currentPoint + direction * step});
    }
}

[[nodiscard]] CandidateScore candidateLossFromSample(
    const FiberPredictionSample& candidateSample,
    const BeamState& beam,
    const cv::Vec3d& candidateDirection,
    const FiberTraceConfig& config,
    const vc::lasagna::NormalSample* precomputedNormal = nullptr)
{
    const auto selectedCurrent =
        bestAlignedPrediction(candidateSample, candidateDirection, true);
    if (!selectedCurrent.valid)
        return {};

    cv::Vec3d smoothNormal{0.0, 0.0, 0.0};
    bool smoothNormalValid = false;
    if (normalAwareSmoothnessEnabled(config) && precomputedNormal != nullptr) {
        if (precomputedNormal->valid) {
            smoothNormal = precomputedNormal->normal;
            smoothNormalValid = true;
        }
    }

    double bestLoss = std::numeric_limits<double>::infinity();
    for (const auto& option : candidateSample.options) {
        if (!option.valid)
            continue;
        const cv::Vec3d candidateSampleDirection =
            alignTo(option.direction, candidateDirection);
        const double presence = clamp01(option.presence);

        const cv::Vec3d prevStep = normalizedOrZero(beam.previousStepDirection);
        const cv::Vec3d currentSample = normalizedOrZero(beam.currentSampleDirection);
        const cv::Vec3d currentStep = normalizedOrZero(candidateDirection);

        double score = presence;
        score *= clampedPositiveDot(prevStep, currentStep);
        score *= clampedPositiveDot(prevStep, currentSample);
        score *= clampedPositiveDot(prevStep, candidateSampleDirection);
        score *= clampedPositiveDot(currentSample, currentStep);
        score *= clampedPositiveDot(currentSample, candidateSampleDirection);
        score *= clampedPositiveDot(currentStep, candidateSampleDirection);

        const double loss = (1.0 - score) +
            smoothnessLoss(prevStep, currentStep, smoothNormal, smoothNormalValid, config) +
            cumulativeTangentSmoothnessLoss(
                beam.historyDirection,
                currentStep,
                smoothNormal,
                smoothNormalValid,
                config);
        if (loss < bestLoss)
            bestLoss = loss;
    }
    return {bestLoss, selectedCurrent.direction, selectedCurrent.presence,
            std::isfinite(bestLoss)};
}

[[nodiscard]] CandidateScore candidateLoss(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const BeamState& beam,
    const cv::Vec3d& candidateDirection,
    const cv::Vec3d& candidatePoint,
    const FiberTraceConfig& config)
{
    const auto candidateSample =
        predictions.sample(candidatePoint, candidateDirection);
    vc::lasagna::NormalSample normalSample;
    const vc::lasagna::NormalSample* normal = nullptr;
    if (normalAwareSmoothnessEnabled(config) && normalSampler != nullptr) {
        normalSample = normalSampler->sampleNormal(candidatePoint);
        normal = &normalSample;
    }
    return candidateLossFromSample(
        candidateSample,
        beam,
        candidateDirection,
        config,
        normal);
}

[[nodiscard]] int traceWorkerCount(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceConfig& config,
    size_t taskCount)
{
    if (taskCount < 2 || !predictions.supportsConcurrentSampling())
        return 1;
    if (normalSampler != nullptr && !normalSampler->supportsConcurrentSampling())
        return 1;
    const int requested = config.parallelThreads;
    if (requested == 1)
        return 1;
#ifdef _OPENMP
    const int available = requested > 0 ? requested : omp_get_max_threads();
#else
    const int available = requested > 0 ? requested : 1;
#endif
    return std::clamp(available, 1, static_cast<int>(taskCount));
}

void sampleCandidateNormals(
    const vc::lasagna::NormalSampler* normalSampler,
    const std::vector<cv::Vec3d>& candidatePoints,
    const FiberTraceConfig& config,
    int parallelThreads,
    std::vector<vc::lasagna::NormalSampleWithDerivative>& samples,
    std::vector<vc::lasagna::NormalSample>& out,
    FiberTraceProfile* profile)
{
    out.clear();
    if (normalSampler == nullptr ||
        candidatePoints.empty() ||
        !normalAwareSmoothnessEnabled(config)) {
        return;
    }
    vc::lasagna::NormalBatchReport report;
    if (const auto* lasagnaSampler =
            dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(normalSampler)) {
        report = lasagnaSampler->sampleNormalBatch(
            candidatePoints,
            false,
            parallelThreads,
            samples);
    } else {
        report = normalSampler->sampleNormalBatch(candidatePoints, false, samples);
    }
    if (profile != nullptr) {
        profile->normalPrefetchSeconds += report.prefetchMs / 1000.0;
        profile->normalMaterializeSeconds += report.materializeMs / 1000.0;
    }
    out.resize(samples.size());
    for (size_t index = 0; index < samples.size(); ++index)
        out[index] = samples[index].sample;
}

[[nodiscard]] const std::vector<CandidateScore>& scoreCandidateTasks(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const std::vector<BeamState>& beams,
    const std::vector<CandidateTask>& tasks,
    const FiberTraceConfig& config,
    const std::vector<vc::lasagna::NormalSample>& normals,
    CandidateScoringScratch& scratch,
    FiberTraceProfile* profile)
{
    auto& scores = scratch.scores;
    scores.clear();
    scores.resize(tasks.size());
    if (tasks.empty())
        return scores;
    if (profile != nullptr)
        profile->candidateTasks += tasks.size();

    const int workers = traceWorkerCount(
        predictions,
        normalSampler,
        config,
        tasks.size());
    if (workers <= 1) {
        const auto scoreStart = TraceClock::now();
        for (size_t index = 0; index < tasks.size(); ++index) {
            const auto& task = tasks[index];
            scores[index] = candidateLoss(
                predictions,
                normalSampler,
                beams[task.beamIndex],
                task.direction,
                task.candidatePoint,
                config);
        }
        if (profile != nullptr)
            profile->candidateScoreSeconds += elapsedSeconds(scoreStart);
        return scores;
    }

    auto& candidatePoints = scratch.candidatePoints;
    auto& referenceDirections = scratch.referenceDirections;
    candidatePoints.clear();
    referenceDirections.clear();
    candidatePoints.reserve(tasks.size());
    referenceDirections.reserve(tasks.size());
    for (const auto& task : tasks) {
        candidatePoints.push_back(task.candidatePoint);
        referenceDirections.push_back(task.direction);
    }
    auto& predictionSamples = scratch.predictionSamples;
    const auto predictionStart = TraceClock::now();
    if (const auto* field = dynamic_cast<const FiberPredictionField*>(&predictions)) {
        field->sampleBatch(
            candidatePoints,
            referenceDirections,
            workers,
            predictionSamples,
            profile);
    } else {
        predictions.sampleBatch(
            candidatePoints,
            referenceDirections,
            workers,
            predictionSamples);
    }
    if (profile != nullptr)
        profile->predictionBatchSeconds += elapsedSeconds(predictionStart);
    if (predictionSamples.size() != tasks.size()) {
        throw std::runtime_error(
            "fiber prediction batch returned the wrong number of samples");
    }
    const auto normalStart = TraceClock::now();
    sampleCandidateNormals(
        normalSampler,
        candidatePoints,
        config,
        workers,
        scratch.normalSamplesWithDerivative,
        scratch.normalSamples,
        profile);
    if (profile != nullptr)
        profile->normalBatchSeconds += elapsedSeconds(normalStart);
    const auto& normalsForScoring =
        normals.empty() ? scratch.normalSamples : normals;

    const auto scoreStart = TraceClock::now();
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    auto scoreOne = [&](size_t index) {
        const auto& task = tasks[index];
        const vc::lasagna::NormalSample* normal =
            index < normalsForScoring.size() ? &normalsForScoring[index] : nullptr;
        scores[index] = candidateLossFromSample(
            predictionSamples[index],
            beams[task.beamIndex],
            task.direction,
            config,
            normal);
    };

#ifdef _OPENMP
    const auto count = static_cast<std::ptrdiff_t>(tasks.size());
    #pragma omp parallel for schedule(dynamic, 8) num_threads(workers)
    for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
        if (failed.load(std::memory_order_relaxed))
            continue;
        try {
            scoreOne(static_cast<size_t>(rawIndex));
        } catch (...) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                #pragma omp critical(fiber_trace_candidate_error)
                {
                    if (!firstError)
                        firstError = std::current_exception();
                }
            }
        }
    }
    if (firstError)
        std::rethrow_exception(firstError);
#else
    (void)workers;
    for (size_t index = 0; index < tasks.size(); ++index)
        scoreOne(index);
#endif
    if (profile != nullptr)
        profile->candidateScoreSeconds += elapsedSeconds(scoreStart);
    return scores;
}

[[nodiscard]] std::optional<cv::Vec3d> interpolatePlaneCrossing(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& planePoint,
    const cv::Vec3d& planeNormal)
{
    const double d0 = pointToPlaneSigned(start, planePoint, planeNormal);
    const double d1 = pointToPlaneSigned(end, planePoint, planeNormal);
    if (d0 == 0.0)
        return start;
    if (d0 * d1 > 0.0)
        return std::nullopt;
    const double denom = d0 - d1;
    if (std::abs(denom) <= kEpsilon)
        return end;
    const double t = std::clamp(d0 / denom, 0.0, 1.0);
    return start * (1.0 - t) + end * t;
}

[[nodiscard]] FrontierCandidate makeFrontierCandidate(
    const BeamState& beam,
    size_t beamIndex,
    size_t order,
    const cv::Vec3d& taskDirection,
    const cv::Vec3d& candidatePoint,
    const CandidateScore& candidateScore,
    const cv::Vec3d& target,
    const cv::Vec3d& targetPlaneNormal,
    const FiberTraceConfig& config)
{
    const cv::Vec3d currentPoint = beamEndpoint(beam);
    const auto crossing = interpolatePlaneCrossing(
        currentPoint,
        candidatePoint,
        target,
        targetPlaneNormal);
    const cv::Vec3d nextPoint = crossing.value_or(candidatePoint);
    return {
        beamIndex,
        nextPoint,
        taskDirection,
        candidateScore.selectedCurrentDirection,
        updateHistoryDirection(
            beam.historyDirection,
            taskDirection,
            beam.depth,
            config.cumulativeSmoothnessSteps),
        beam.loss + candidateScore.loss,
        beam.tracedLength + length(nextPoint - currentPoint),
        beam.depth + 1,
        crossing.has_value(),
        order,
    };
}

[[nodiscard]] double beamPruneScore(const BeamState& state)
{
    return state.loss + static_cast<double>(state.depth) * 1.0e-12;
}

[[nodiscard]] bool beamSearchLess(const BeamState& a, const BeamState& b)
{
    if (a.loss != b.loss)
        return a.loss < b.loss;
    return a.depth < b.depth;
}

[[nodiscard]] std::vector<BeamState> pruneBeamStates(
    std::vector<BeamState> states,
    int beamWidth,
    double pruneDistanceVoxels)
{
    if (states.empty())
        return {};
    const size_t keep = static_cast<size_t>(std::max(1, beamWidth));
    const double distance = std::max(0.0, pruneDistanceVoxels);
    std::vector<size_t> selected;
    selected.reserve(std::min(keep, states.size()));
    std::vector<unsigned char> unavailable(states.size(), 0);
    const double distance2 = distance * distance;

    while (selected.size() < keep) {
        std::optional<size_t> best;
        double bestScore = 0.0;
        for (size_t index = 0; index < states.size(); ++index) {
            if (unavailable[index])
                continue;
            const double score = beamPruneScore(states[index]);
            if (!std::isfinite(score))
                continue;
            if (distance > 0.0) {
                const cv::Vec3d point = beamEndpoint(states[index]);
                bool tooClose = false;
                for (const size_t existingIndex : selected) {
                    const cv::Vec3d delta = point - beamEndpoint(states[existingIndex]);
                    if (delta.dot(delta) < distance2) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose)
                    continue;
            }
            if (!best.has_value() || score < bestScore) {
                best = index;
                bestScore = score;
            }
        }
        if (!best.has_value())
            break;
        unavailable[*best] = 1;
        selected.push_back(*best);
    }

    std::vector<BeamState> out;
    out.reserve(selected.size());
    for (const size_t index : selected)
        out.push_back(std::move(states[index]));
    if (!out.empty())
        return out;
    return {std::move(states.front())};
}

[[nodiscard]] double frontierPruneScore(const FrontierCandidate& candidate)
{
    return candidate.loss + static_cast<double>(candidate.depth) * 1.0e-12;
}

[[nodiscard]] std::vector<size_t> selectFrontierCandidateIndices(
    const std::vector<FrontierCandidate>& candidates,
    int beamWidth,
    double pruneDistanceVoxels)
{
    if (candidates.empty())
        return {};
    const size_t keep = static_cast<size_t>(std::max(1, beamWidth));
    const double distance = std::max(0.0, pruneDistanceVoxels);
    const double distance2 = distance * distance;
    std::vector<size_t> selected;
    selected.reserve(std::min(keep, candidates.size()));
    std::vector<unsigned char> unavailable(candidates.size(), 0);

    while (selected.size() < keep) {
        std::optional<size_t> best;
        double bestScore = 0.0;
        for (size_t index = 0; index < candidates.size(); ++index) {
            if (unavailable[index])
                continue;
            const double score = frontierPruneScore(candidates[index]);
            if (!std::isfinite(score))
                continue;
            if (distance > 0.0) {
                bool tooClose = false;
                for (const size_t existingIndex : selected) {
                    const cv::Vec3d delta =
                        candidates[index].point - candidates[existingIndex].point;
                    if (delta.dot(delta) < distance2) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose)
                    continue;
            }
            if (!best.has_value() || score < bestScore) {
                best = index;
                bestScore = score;
            }
        }
        if (!best.has_value())
            break;
        unavailable[*best] = 1;
        selected.push_back(*best);
    }

    if (selected.empty())
        selected.push_back(0);
    return selected;
}

[[nodiscard]] BeamState beamStateFromFrontierCandidate(
    const std::vector<BeamState>& parents,
    const FrontierCandidate& candidate)
{
    const BeamState& parent = parents[candidate.beamIndex];
    BeamState out = parent;
    out.path = appendBeamPathPoint(parent.path, candidate.point);
    out.previousStepDirection = candidate.previousStepDirection;
    out.currentSampleDirection = candidate.currentSampleDirection;
    out.historyDirection = candidate.historyDirection;
    out.loss = candidate.loss;
    out.tracedLength = candidate.tracedLength;
    out.depth = candidate.depth;
    out.reached = candidate.reached;
    out.reason = candidate.reached ? "target_plane" : std::string{};
    return out;
}

[[nodiscard]] std::vector<BeamState> pruneFrontierCandidates(
    const std::vector<FrontierCandidate>& candidates,
    const std::vector<BeamState>& parents,
    int beamWidth,
    double pruneDistanceVoxels)
{
    if (candidates.empty())
        return {};
    const std::vector<size_t> selected = selectFrontierCandidateIndices(
        candidates,
        beamWidth,
        pruneDistanceVoxels);
    std::vector<BeamState> out;
    out.reserve(selected.size());
    for (const size_t index : selected) {
        out.push_back(beamStateFromFrontierCandidate(parents, candidates[index]));
    }
    return out;
}

[[nodiscard]] std::optional<size_t> bestReachedFrontierCandidateIndex(
    const std::vector<FrontierCandidate>& candidates)
{
    std::optional<size_t> best;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!candidate.reached)
            continue;
        if (!best.has_value() || candidate.loss < candidates[*best].loss)
            best = index;
    }
    return best;
}

[[nodiscard]] std::optional<size_t> bestReachedStateIndexPythonParity(
    const std::vector<BeamState>& states)
{
    std::optional<size_t> best;
    for (size_t index = 0; index < states.size(); ++index) {
        const auto& state = states[index];
        if (!state.reached)
            continue;
        if (!best.has_value() || state.loss < states[*best].loss)
            best = index;
    }
    return best;
}

[[nodiscard]] FiberTraceOneWayResult traceOneWayCore(
    const FiberPredictionSource& predictions,
    const FiberTraceOneWayRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress,
    std::string phase)
{
    const cv::Vec3d start = request.startPoint;
    const cv::Vec3d target = request.targetPoint;
    const cv::Vec3d targetPlaneNormal = normalizedOr(
        request.targetPlaneNormal,
        normalizedOr(target - start, {1.0, 0.0, 0.0}));
    const cv::Vec3d referenceStartDirection = normalizedOr(
        request.initialDirection,
        normalizedOr(target - start, {1.0, 0.0, 0.0}));
    FiberTraceProfile* profile = request.config.profile;
    if (profile != nullptr)
        ++profile->oneWayCalls;
    const auto startSampleStart = TraceClock::now();
    const ScoredDirection startPrediction =
        bestAlignedPrediction(predictions, start, referenceStartDirection, false);
    if (profile != nullptr)
        profile->startSampleSeconds += elapsedSeconds(startSampleStart);
    if (!startPrediction.valid) {
        throw std::invalid_argument(
            "fiber trace start point has no valid prediction direction");
    }
    const cv::Vec3d startDirection = startPrediction.direction;

    const double distance = request.budgetSpanVoxels > 0.0
        ? request.budgetSpanVoxels
        : length(target - start);
    const double step = std::max(1.0e-3, request.config.stepVoxels);
    const int maxSteps = std::max(
        1,
        static_cast<int>(std::ceil(
            distance * request.config.maxStepFactor / step)));

    BeamState initial;
    initial.path = appendBeamPathPoint(nullptr, start);
    initial.previousStepDirection = startDirection;
    initial.currentSampleDirection = startDirection;
    initial.historyDirection = startDirection;
    std::vector<BeamState> beams{std::move(initial)};
    CandidateScoringScratch scoringScratch;
    std::string reason = "max_step_factor";

    const int lookaheadSteps = request.config.beamWidth <= 1
        ? 1
        : std::max(1, request.config.beamLookaheadSteps);
    const auto coneOffsets = request.config.coneAngleStepDegrees > 0.0
        ? angleStepConeOffsets(
              request.config.coneAngleDegrees,
              request.config.coneAngleStepDegrees)
        : legacyGridConeOffsets(
              request.config.coneAngleDegrees,
              request.config.coneGridSize);
    const size_t candidateCount = std::max<size_t>(1, coneOffsets.size());
    int stepIndex = 0;
    while (stepIndex < maxSteps) {
        std::vector<BeamState> expanded = beams;
        int advanced = 0;
        bool prunedFinalFrontier = false;
        for (; advanced < lookaheadSteps && stepIndex + advanced < maxSteps; ++advanced) {
            if (profile != nullptr)
                ++profile->generations;
            const auto taskBuildStart = TraceClock::now();
            std::vector<CandidateTask> tasks;
            tasks.reserve(expanded.size() * candidateCount);
            for (size_t beamIndex = 0; beamIndex < expanded.size(); ++beamIndex) {
                appendCandidateTasks(
                    tasks,
                    beamIndex,
                    expanded[beamIndex],
                    coneOffsets,
                    step);
            }
            if (profile != nullptr)
                profile->taskBuildSeconds += elapsedSeconds(taskBuildStart);

            const auto& scores = scoreCandidateTasks(
                predictions,
                normalSampler,
                expanded,
                tasks,
                request.config,
                {},
                scoringScratch,
                profile);

            const bool finalLookaheadGeneration =
                advanced + 1 >= lookaheadSteps ||
                stepIndex + advanced + 1 >= maxSteps;
            if (finalLookaheadGeneration) {
                const auto frontierStart = TraceClock::now();
                auto& frontier = scoringScratch.frontierCandidates;
                frontier.clear();
                frontier.reserve(tasks.size());
                for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
                    const auto& task = tasks[taskIndex];
                    const CandidateScore& candidateScore = scores[taskIndex];
                    if (!candidateScore.valid || !std::isfinite(candidateScore.loss))
                        continue;
                    frontier.push_back(makeFrontierCandidate(
                        expanded[task.beamIndex],
                        task.beamIndex,
                        taskIndex,
                        task.direction,
                        task.candidatePoint,
                        candidateScore,
                        target,
                        targetPlaneNormal,
                        request.config));
                }
                if (profile != nullptr)
                    profile->frontierSeconds += elapsedSeconds(frontierStart);
                if (frontier.empty()) {
                    reason = "no_valid_candidates";
                    expanded.clear();
                    break;
                }
                const auto bestReachedIndex =
                    bestReachedFrontierCandidateIndex(frontier);
                if (bestReachedIndex.has_value()) {
                    const BeamState bestReached = beamStateFromFrontierCandidate(
                        expanded,
                        frontier[*bestReachedIndex]);
                    if (progress) {
                        FiberTraceProgress event;
                        event.phase = phase;
                        event.step = stepIndex + advanced + 1;
                        event.maxSteps = maxSteps;
                        event.targetPlaneProgress = 1.0;
                        event.reason = "target_plane";
                        progress(event);
                    }
                    return {beamPathPoints(bestReached),
                            true,
                            bestReached.reason,
                            static_cast<int>(
                                beamPointCount(bestReached) > 0
                                    ? beamPointCount(bestReached) - 1
                                    : 0)};
                }
                const auto pruneStart = TraceClock::now();
                beams = pruneFrontierCandidates(
                    frontier,
                    expanded,
                    request.config.beamWidth,
                    request.config.beamPruneDistanceVoxels);
                if (profile != nullptr)
                    profile->pruneSeconds += elapsedSeconds(pruneStart);
                prunedFinalFrontier = true;
                ++advanced;
                break;
            }

            const auto frontierStart = TraceClock::now();
            std::vector<BeamState> nextFrontier;
            nextFrontier.reserve(tasks.size());
            for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
                const auto& task = tasks[taskIndex];
                const CandidateScore& candidateScore = scores[taskIndex];
                if (!candidateScore.valid || !std::isfinite(candidateScore.loss))
                    continue;
                const BeamState& beam = expanded[task.beamIndex];
                const cv::Vec3d currentPoint = beamEndpoint(beam);
                const auto crossing = interpolatePlaneCrossing(
                    currentPoint,
                    task.candidatePoint,
                    target,
                    targetPlaneNormal);
                const cv::Vec3d nextPoint = crossing.value_or(task.candidatePoint);
                BeamState next = beam;
                next.path = appendBeamPathPoint(beam.path, nextPoint);
                next.tracedLength += length(nextPoint - currentPoint);
                next.loss += candidateScore.loss;
                next.previousStepDirection = task.direction;
                next.currentSampleDirection = candidateScore.selectedCurrentDirection;
                next.historyDirection = updateHistoryDirection(
                    beam.historyDirection,
                    task.direction,
                    beam.depth,
                    request.config.cumulativeSmoothnessSteps);
                next.depth = beam.depth + 1;
                next.reached = crossing.has_value();
                if (next.reached) {
                    next.reason = "target_plane";
                }
                nextFrontier.push_back(std::move(next));
            }
            if (profile != nullptr)
                profile->frontierSeconds += elapsedSeconds(frontierStart);
            expanded = std::move(nextFrontier);
            if (expanded.empty()) {
                reason = "no_valid_candidates";
                break;
            }
            const auto bestReachedIndex = bestReachedStateIndexPythonParity(expanded);
            if (bestReachedIndex.has_value()) {
                const auto& bestReached = expanded[*bestReachedIndex];
                if (progress) {
                    FiberTraceProgress event;
                    event.phase = phase;
                    event.step = stepIndex + advanced + 1;
                    event.maxSteps = maxSteps;
                    event.targetPlaneProgress = 1.0;
                    event.reason = "target_plane";
                    progress(event);
                }
                return {beamPathPoints(bestReached), true, bestReached.reason,
                        static_cast<int>(
                            beamPointCount(bestReached) > 0
                                ? beamPointCount(bestReached) - 1
                                : 0)};
            }
        }
        if (expanded.empty() && !prunedFinalFrontier)
            break;

        if (!prunedFinalFrontier) {
            const auto pruneStart = TraceClock::now();
            beams = pruneBeamStates(
                std::move(expanded),
                request.config.beamWidth,
                request.config.beamPruneDistanceVoxels);
            if (profile != nullptr)
                profile->pruneSeconds += elapsedSeconds(pruneStart);
        }
        stepIndex += std::max(1, advanced);

        if (progress) {
            const double signedDistance =
                std::abs(pointToPlaneSigned(beamEndpoint(beams.front()), target, targetPlaneNormal));
            FiberTraceProgress event;
            event.phase = phase;
            event.step = stepIndex;
            event.maxSteps = maxSteps;
            event.targetPlaneProgress =
                distance > kEpsilon ? 1.0 - std::min(1.0, signedDistance / distance) : 1.0;
            event.reason = beams.front().reached ? beams.front().reason : reason;
            progress(event);
        }
    }

    if (beams.empty()) {
        return {{start}, false, reason, 0};
    }

    const auto best = std::min_element(
        beams.begin(), beams.end(), beamSearchLess);
    return {beamPathPoints(*best), best->reached, best->reached ? best->reason : reason,
            static_cast<int>(beamPointCount(*best) > 0 ? beamPointCount(*best) - 1 : 0)};
}

[[nodiscard]] std::vector<cv::Vec3d> fuseTraces(
    const std::vector<cv::Vec3d>& forward,
    const std::vector<cv::Vec3d>& reverse,
    double gapFactor)
{
    if (forward.empty())
        return reverse;
    if (reverse.empty())
        return forward;

    const auto forwardLengths = arclengths(forward);
    const auto reverseLengths = arclengths(reverse);
    double bestScore = std::numeric_limits<double>::infinity();
    size_t bestI = forward.size() - 1;
    size_t bestJ = reverse.size() - 1;
    for (size_t i = 0; i < forward.size(); ++i) {
        for (size_t j = 0; j < reverse.size(); ++j) {
            const double gap = length(forward[i] - reverse[j]);
            const double score = gapFactor * gap + forwardLengths[i] + reverseLengths[j];
            if (score < bestScore) {
                bestScore = score;
                bestI = i;
                bestJ = j;
            }
        }
    }

    const cv::Vec3d midpoint = (forward[bestI] + reverse[bestJ]) * 0.5;
    std::vector<cv::Vec3d> fused;
    fused.reserve(bestI + bestJ + 3);
    for (size_t i = 0; i <= bestI; ++i)
        fused.push_back(forward[i]);
    fused.push_back(midpoint);
    for (size_t count = 0; count <= bestJ; ++count) {
        const size_t j = bestJ - count;
        fused.push_back(reverse[j]);
        if (j == 0)
            break;
    }
    if (!fused.empty()) {
        fused.front() = forward.front();
        fused.back() = reverse.front();
    }
    return fused;
}

} // namespace

#ifdef VC_TESTING
namespace testing {

namespace {

[[nodiscard]] std::vector<BeamState> debugStatesToBeamStates(
    const std::vector<BeamDebugState>& states)
{
    std::vector<BeamState> out;
    out.reserve(states.size());
    for (size_t index = 0; index < states.size(); ++index) {
        const auto& state = states[index];
        BeamState beam;
        beam.path = appendBeamPathPoint(nullptr, state.point);
        beam.loss = state.loss;
        beam.depth = state.depth;
        beam.tracedLength = state.tracedLength;
        beam.reached = state.reached;
        beam.reason = std::to_string(index);
        out.push_back(std::move(beam));
    }
    return out;
}

} // namespace

std::vector<size_t> debugPruneBeamStateIndices(
    const std::vector<BeamDebugState>& states,
    int beamWidth,
    double pruneDistanceVoxels)
{
    std::vector<size_t> indices;
    auto pruned = pruneBeamStates(
        debugStatesToBeamStates(states),
        beamWidth,
        pruneDistanceVoxels);
    indices.reserve(pruned.size());
    for (const auto& state : pruned)
        indices.push_back(static_cast<size_t>(std::stoull(state.reason)));
    return indices;
}

std::optional<size_t> debugBestReachedBeamStateIndex(
    const std::vector<BeamDebugState>& states)
{
    const auto beamStates = debugStatesToBeamStates(states);
    return bestReachedStateIndexPythonParity(beamStates);
}

namespace {

class DebugPredictionSource final : public FiberPredictionSource {
public:
    explicit DebugPredictionSource(bool concurrent)
        : concurrent_(concurrent)
    {
    }

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return concurrent_;
    }

    [[nodiscard]] FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d&) const override
    {
        return {};
    }

private:
    bool concurrent_ = false;
};

class DebugNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit DebugNormalSampler(bool concurrent)
        : concurrent_(concurrent)
    {
    }

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return concurrent_;
    }

    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d&) const override
    {
        return {};
    }

private:
    bool concurrent_ = false;
};

} // namespace

int debugTraceWorkerCount(
    bool predictionConcurrent,
    bool normalConcurrent,
    bool hasNormalSampler,
    int parallelThreads,
    size_t taskCount)
{
    FiberTraceConfig config;
    config.parallelThreads = parallelThreads;
    DebugPredictionSource predictions(predictionConcurrent);
    DebugNormalSampler normals(normalConcurrent);
    return traceWorkerCount(
        predictions,
        hasNormalSampler ? &normals : nullptr,
        config,
        taskCount);
}

} // namespace testing
#endif

FiberTraceCoordinateAdapter::FiberTraceCoordinateAdapter(
    double traceToBaseScaleValue)
    : traceToBaseScale(traceToBaseScaleValue)
{
    if (!(traceToBaseScale > 0.0) || !std::isfinite(traceToBaseScale)) {
        throw std::invalid_argument(
            "fiber trace-to-base scale must be positive and finite");
    }
}

cv::Vec3d FiberTraceCoordinateAdapter::baseToTrace(
    const cv::Vec3d& point) const
{
    return point / traceToBaseScale;
}

cv::Vec3d FiberTraceCoordinateAdapter::traceToBase(
    const cv::Vec3d& point) const
{
    return point * traceToBaseScale;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::baseToTrace(
    const std::vector<cv::Vec3d>& points) const
{
    std::vector<cv::Vec3d> converted;
    converted.reserve(points.size());
    for (const auto& point : points)
        converted.push_back(baseToTrace(point));
    return converted;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::traceToBase(
    const std::vector<cv::Vec3d>& points) const
{
    std::vector<cv::Vec3d> converted;
    converted.reserve(points.size());
    for (const auto& point : points)
        converted.push_back(traceToBase(point));
    return converted;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::traceSegmentToBase(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& exactStartBase,
    const cv::Vec3d& exactTargetBase) const
{
    if (points.size() < 2) {
        throw std::invalid_argument(
            "fiber trace segment must contain at least two points");
    }
    auto converted = traceToBase(points);
    converted.front() = exactStartBase;
    converted.back() = exactTargetBase;
    return converted;
}

double FiberTraceCoordinateAdapter::baseDistanceToTrace(
    double distanceBaseVoxels) const
{
    if (std::isnan(distanceBaseVoxels))
        throw std::invalid_argument("base distance must not be NaN");
    return distanceBaseVoxels / traceToBaseScale;
}

double FiberTraceCoordinateAdapter::traceDistanceToBase(
    double distanceTraceVoxels) const
{
    if (std::isnan(distanceTraceVoxels))
        throw std::invalid_argument("trace distance must not be NaN");
    return distanceTraceVoxels * traceToBaseScale;
}

FiberPredictionTraceScales resolveFiberPredictionTraceScales(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    int inferenceScaledownPower)
{
    if (inferenceScaledownPower < 0 || inferenceScaledownPower > 30) {
        throw std::runtime_error(
            "fiber inference scaledown power must be in [0, 30]");
    }
    const double inferenceScaledown =
        static_cast<double>(1 << inferenceScaledownPower);

    if (!manifest.raw.empty()) {
        const auto sourceIt = manifest.raw.find("source_to_base");
        if (sourceIt == manifest.raw.end() || !sourceIt->is_number()) {
            throw std::runtime_error(
                "fiber inference manifest must contain numeric source_to_base");
        }
    }
    if (!(manifest.sourceToBase > 0.0) || !std::isfinite(manifest.sourceToBase)) {
        throw std::runtime_error(
            "fiber inference manifest source_to_base must be positive and finite");
    }

    const auto prefixes = fiberPredictionPrefixes(manifest);
    if (prefixes.empty()) {
        throw std::runtime_error(
            "fiber inference dataset must contain presence/nx/ny channels");
    }

    std::optional<double> predictionToBaseScale;
    std::optional<double> predictionGroupScaleFactor;
    std::optional<std::string> inferredChannel;
    for (const auto& prefix : prefixes) {
        for (const auto& channel : predictionChannelNames(prefix)) {
            const auto& group = predictionChannelGroup(manifest, channel);
            const double scale = predictionChannelEffectiveScale(manifest, channel);
            const double groupScaleFactor =
                static_cast<double>(group.scaleFactor());
            if (!predictionToBaseScale.has_value()) {
                predictionToBaseScale = scale;
                predictionGroupScaleFactor = groupScaleFactor;
                inferredChannel = channel;
            } else if (!nearlySameScale(*predictionToBaseScale, scale)) {
                throw std::runtime_error(
                    "fiber inference prediction channels must share one effective "
                    "prediction-to-base scale; channel '" + channel +
                    "' has scale " + std::to_string(scale) + " but channel '" +
                    *inferredChannel + "' has scale " +
                    std::to_string(*predictionToBaseScale));
            } else if (!nearlySameScale(
                           *predictionGroupScaleFactor,
                           groupScaleFactor)) {
                throw std::runtime_error(
                    "fiber inference prediction channels must share one "
                    "manifest group scale factor; channel '" + channel +
                    "' has factor " +
                    std::to_string(groupScaleFactor) +
                    " but channel '" + *inferredChannel +
                    "' has factor " +
                    std::to_string(*predictionGroupScaleFactor));
            }
        }
    }

    const double traceToBaseScale = *predictionToBaseScale / inferenceScaledown;
    if (!(traceToBaseScale > 0.0) || !std::isfinite(traceToBaseScale)) {
        throw std::runtime_error(
            "fiber inference manifest derived trace scale must be positive and finite");
    }

    return {
        traceToBaseScale,
        *predictionToBaseScale,
        inferenceScaledown,
    };
}

double inferFiberPredictionWorkingToBaseScale(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    int inferenceScaledownPower)
{
    return resolveFiberPredictionTraceScales(
        manifest,
        inferenceScaledownPower).traceToBaseScale;
}

class FiberPredictionField::Impl {
public:
    struct Option {
        std::string name;
        vc::lasagna::LasagnaChannelBinding presence;
        vc::lasagna::LasagnaChannelBinding nx;
        vc::lasagna::LasagnaChannelBinding ny;
    };

    struct PreparedOptionSample {
        vc::lasagna::LasagnaCubeRequest presence;
        vc::lasagna::LasagnaCubeRequest nx;
        vc::lasagna::LasagnaCubeRequest ny;
    };

    struct OptionChunkMaps {
        vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap presence;
        vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap nx;
        vc::lasagna::LasagnaChannelChunkCache::ResolvedChunkMap ny;
    };

    struct OptionSamplingGrid {
        bool sharedPresenceNxNy = false;
    };

    Impl(const vc::lasagna::LasagnaDataset& dataset, size_t maxCachedBytes)
        : cache_(vc::lasagna::sharedLasagnaChannelChunkCache(maxCachedBytes))
    {
        const auto& manifest = dataset.manifest();
        const auto prefixes = fiberPredictionPrefixes(manifest);
        if (prefixes.empty())
            throw std::runtime_error(
                "fiber inference dataset must contain presence/nx/ny channels");

        options_.reserve(prefixes.size());
        for (const auto& prefix : prefixes) {
            const auto channels = predictionChannelNames(prefix);
            options_.push_back({
                prefix.empty() ? std::string("option_000") : prefix,
                vc::lasagna::bindLasagnaChannel(manifest, channels[0]),
                vc::lasagna::bindLasagnaChannel(manifest, channels[1]),
                vc::lasagna::bindLasagnaChannel(manifest, channels[2]),
            });
        }
        optionGrids_.reserve(options_.size());
        for (const auto& option : options_) {
            optionGrids_.push_back({
                vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.nx) &&
                vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.ny),
            });
        }
    }

    [[nodiscard]] vc::lasagna::NormalPrefetchReport prefetchSamples(
        const std::vector<cv::Vec3d>& volumePoints) const
    {
        std::vector<vc::lasagna::LasagnaChannelChunkCache::PrefetchRequest> requests;
        requests.reserve(volumePoints.size() * options_.size() * 24);
        std::vector<vc::lasagna::LasagnaChannelChunkKey> keys;
        keys.reserve(volumePoints.size() * 8);
        for (const auto& point : volumePoints) {
            for (const auto& option : options_) {
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(
                    option.presence, point, keys);
                for (const auto& key : keys)
                    requests.emplace_back(&option.presence, key);
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(
                    option.nx, point, keys);
                for (const auto& key : keys)
                    requests.emplace_back(&option.nx, key);
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(
                    option.ny, point, keys);
                for (const auto& key : keys)
                    requests.emplace_back(&option.ny, key);
            }
        }
        return cache_->prefetchInterleaved(requests);
    }

    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples,
        FiberTraceProfile* profile) const
    {
        if (volumePoints.size() != referenceDirections.size()) {
            throw std::invalid_argument(
                "fiber prediction batch points and reference directions size mismatch");
        }
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty())
            return;

        const size_t optionCount = options_.size();
        const int workers =
            std::clamp(parallelThreads, 1, static_cast<int>(volumePoints.size()));
        const auto directStart = TraceClock::now();
        auto materializeDirect = [&](
            size_t pointIndex,
            std::vector<vc::lasagna::LasagnaLocalChunkResolver>& presenceResolvers,
            std::vector<vc::lasagna::LasagnaLocalChunkResolver>& nxResolvers,
            std::vector<vc::lasagna::LasagnaLocalChunkResolver>& nyResolvers) {
            auto& out = samples[pointIndex];
            out.options.clear();
            out.options.reserve(optionCount);
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const auto& option = options_[optionIndex];
                auto presenceRequest = vc::lasagna::prepareLasagnaCubeRequest(
                    option.presence, volumePoints[pointIndex]);
                vc::lasagna::LasagnaCubeRequest nxRequest;
                vc::lasagna::LasagnaCubeRequest nyRequest;
                if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                    nxRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                        presenceRequest,
                        option.nx);
                    nyRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                        presenceRequest,
                        option.ny);
                } else {
                    nxRequest = vc::lasagna::prepareLasagnaCubeRequest(
                        option.nx, volumePoints[pointIndex]);
                    nyRequest = vc::lasagna::prepareLasagnaCubeRequest(
                        option.ny, volumePoints[pointIndex]);
                }
                presenceResolvers[optionIndex].resolve(presenceRequest);
                nxResolvers[optionIndex].resolve(nxRequest);
                nyResolvers[optionIndex].resolve(nyRequest);

                const auto rawPresence =
                    vc::lasagna::sampleLasagnaChannel(option.presence, presenceRequest);
                const auto direction =
                    vc::lasagna::sampleLasagnaCompactAxisTensor(
                        option.nx, option.ny, nxRequest, nyRequest);
                if (!rawPresence.has_value() || !direction.has_value()) {
                    out.options.push_back({});
                    continue;
                }
                out.options.push_back({
                    alignTo(*direction, referenceDirections[pointIndex]),
                    clamp01(*rawPresence / 255.0),
                    true,
                });
            }
        };
        auto makeResolvers = [&](
            const auto& bindingSelector) {
            std::vector<vc::lasagna::LasagnaLocalChunkResolver> resolvers;
            resolvers.reserve(optionCount);
            for (const auto& option : options_)
                resolvers.emplace_back(bindingSelector(option), *cache_);
            return resolvers;
        };

        if (workers <= 1) {
            auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& {
                return option.presence;
            });
            auto nxResolvers = makeResolvers([](const Option& option) -> const auto& {
                return option.nx;
            });
            auto nyResolvers = makeResolvers([](const Option& option) -> const auto& {
                return option.ny;
            });
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex) {
                materializeDirect(
                    pointIndex,
                    presenceResolvers,
                    nxResolvers,
                    nyResolvers);
            }
        } else {
#ifdef _OPENMP
            std::atomic<bool> failed{false};
            std::exception_ptr firstError;
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel num_threads(workers)
            {
                auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& {
                    return option.presence;
                });
                auto nxResolvers = makeResolvers([](const Option& option) -> const auto& {
                    return option.nx;
                });
                auto nyResolvers = makeResolvers([](const Option& option) -> const auto& {
                    return option.ny;
                });
                #pragma omp for schedule(static)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                    if (failed.load(std::memory_order_relaxed))
                        continue;
                    try {
                        materializeDirect(
                            static_cast<size_t>(rawIndex),
                            presenceResolvers,
                            nxResolvers,
                            nyResolvers);
                    } catch (...) {
                        bool expected = false;
                        if (failed.compare_exchange_strong(expected, true)) {
                            #pragma omp critical(fiber_prediction_direct_error)
                            {
                                if (!firstError)
                                    firstError = std::current_exception();
                            }
                        }
                    }
                }
            }
            if (firstError)
                std::rethrow_exception(firstError);
#else
            std::vector<std::future<void>> futures;
            futures.reserve(static_cast<size_t>(workers));
            std::atomic<size_t> next{0};
            for (int worker = 0; worker < workers; ++worker) {
                futures.push_back(std::async(std::launch::async, [&]() {
                    auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& {
                        return option.presence;
                    });
                    auto nxResolvers = makeResolvers([](const Option& option) -> const auto& {
                        return option.nx;
                    });
                    auto nyResolvers = makeResolvers([](const Option& option) -> const auto& {
                        return option.ny;
                    });
                    while (true) {
                        const size_t pointIndex = next.fetch_add(1);
                        if (pointIndex >= volumePoints.size())
                            return;
                        materializeDirect(
                            pointIndex,
                            presenceResolvers,
                            nxResolvers,
                            nyResolvers);
                    }
                }));
            }
            for (auto& future : futures)
                future.get();
#endif
        }
        if (profile != nullptr)
            profile->predictionMaterializeSeconds += elapsedSeconds(directStart);
        return;

        const auto prepareStart = TraceClock::now();
        std::vector<PreparedOptionSample> prepared(volumePoints.size() * optionCount);
        std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> presenceKeys(optionCount);
        std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> nxKeys(optionCount);
        std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> nyKeys(optionCount);
        const auto preparePoint = [&](size_t pointIndex,
                                      std::vector<vc::lasagna::LasagnaChannelChunkKey>* localPresenceKeys,
                                      std::vector<vc::lasagna::LasagnaChannelChunkKey>* localNxKeys,
                                      std::vector<vc::lasagna::LasagnaChannelChunkKey>* localNyKeys) {
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const auto& option = options_[optionIndex];
                auto& point = prepared[pointIndex * optionCount + optionIndex];
                point.presence = vc::lasagna::prepareLasagnaCubeRequest(
                    option.presence, volumePoints[pointIndex]);
                if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                    point.nx = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                        point.presence,
                        option.nx);
                    point.ny = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                        point.presence,
                        option.ny);
                } else {
                    point.nx = vc::lasagna::prepareLasagnaCubeRequest(
                        option.nx, volumePoints[pointIndex]);
                    point.ny = vc::lasagna::prepareLasagnaCubeRequest(
                        option.ny, volumePoints[pointIndex]);
                }
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(
                    point.presence,
                    localPresenceKeys[optionIndex]);
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(
                    point.nx,
                    localNxKeys[optionIndex]);
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(
                    point.ny,
                    localNyKeys[optionIndex]);
            }
        };

#ifdef _OPENMP
        if (workers > 1) {
            const size_t workerCount = static_cast<size_t>(workers);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>>
                presenceKeysByWorker(workerCount * optionCount);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>>
                nxKeysByWorker(workerCount * optionCount);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>>
                nyKeysByWorker(workerCount * optionCount);
            const size_t reservePerWorker =
                volumePoints.size() / workerCount + 16;
            for (size_t worker = 0; worker < workerCount; ++worker) {
                for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                    const size_t slot = worker * optionCount + optionIndex;
                    presenceKeysByWorker[slot].reserve(reservePerWorker);
                    nxKeysByWorker[slot].reserve(reservePerWorker);
                    nyKeysByWorker[slot].reserve(reservePerWorker);
                }
            }
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel num_threads(workers)
            {
                const size_t worker = static_cast<size_t>(omp_get_thread_num());
                const size_t slotOffset = worker * optionCount;
                #pragma omp for schedule(static)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                    preparePoint(
                        static_cast<size_t>(rawIndex),
                        presenceKeysByWorker.data() + slotOffset,
                        nxKeysByWorker.data() + slotOffset,
                        nyKeysByWorker.data() + slotOffset);
                }
                for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(
                        presenceKeysByWorker[slotOffset + optionIndex]);
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(
                        nxKeysByWorker[slotOffset + optionIndex]);
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(
                        nyKeysByWorker[slotOffset + optionIndex]);
                }
            }
            auto mergeKeys = [&](std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>>& out,
                                 std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>>& byWorker) {
                for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                    size_t total = 0;
                    for (size_t worker = 0; worker < workerCount; ++worker) {
                        total += byWorker[worker * optionCount + optionIndex].size();
                    }
                    out[optionIndex].reserve(total);
                    for (size_t worker = 0; worker < workerCount; ++worker) {
                        auto& local = byWorker[worker * optionCount + optionIndex];
                        out[optionIndex].insert(
                            out[optionIndex].end(),
                            std::make_move_iterator(local.begin()),
                            std::make_move_iterator(local.end()));
                    }
                }
            };
            mergeKeys(presenceKeys, presenceKeysByWorker);
            mergeKeys(nxKeys, nxKeysByWorker);
            mergeKeys(nyKeys, nyKeysByWorker);
        } else
#endif
        {
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                presenceKeys[optionIndex].reserve(volumePoints.size());
                nxKeys[optionIndex].reserve(volumePoints.size());
                nyKeys[optionIndex].reserve(volumePoints.size());
            }
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex) {
                preparePoint(
                    pointIndex,
                    presenceKeys.data(),
                    nxKeys.data(),
                    nyKeys.data());
            }
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                vc::lasagna::deduplicateLasagnaChunkKeysInPlace(presenceKeys[optionIndex]);
                vc::lasagna::deduplicateLasagnaChunkKeysInPlace(nxKeys[optionIndex]);
                vc::lasagna::deduplicateLasagnaChunkKeysInPlace(nyKeys[optionIndex]);
            }
        }
        if (profile != nullptr)
            profile->predictionPrepareSeconds += elapsedSeconds(prepareStart);

        const auto prefetchStart = TraceClock::now();
        std::vector<OptionChunkMaps> chunks(optionCount);
        const size_t readWorkers = vc::lasagna::lasagnaReadWorkersPerChannel();
        std::vector<std::future<vc::lasagna::NormalPrefetchReport>> prefetches;
        prefetches.reserve(optionCount * 3);
        for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
            prefetches.push_back(std::async(std::launch::async, [&, optionIndex]() {
                const auto& option = options_[optionIndex];
                return cache_->prefetchResolved(
                    option.presence,
                    *option.presence.array,
                    presenceKeys[optionIndex],
                    readWorkers,
                    chunks[optionIndex].presence);
            }));
            prefetches.push_back(std::async(std::launch::async, [&, optionIndex]() {
                const auto& option = options_[optionIndex];
                return cache_->prefetchResolved(
                    option.nx,
                    *option.nx.array,
                    nxKeys[optionIndex],
                    readWorkers,
                    chunks[optionIndex].nx);
            }));
            prefetches.push_back(std::async(std::launch::async, [&, optionIndex]() {
                const auto& option = options_[optionIndex];
                return cache_->prefetchResolved(
                    option.ny,
                    *option.ny.array,
                    nyKeys[optionIndex],
                    readWorkers,
                    chunks[optionIndex].ny);
            }));
        }
        for (auto& future : prefetches)
            (void)future.get();
        if (profile != nullptr)
            profile->predictionPrefetchSeconds += elapsedSeconds(prefetchStart);

        const auto assignStart = TraceClock::now();
        const auto assignPointChunks = [&](size_t pointIndex) {
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                auto& point = prepared[pointIndex * optionCount + optionIndex];
                const auto& maps = chunks[optionIndex];
                vc::lasagna::assignResolvedLasagnaCubeRequestChunks(
                    point.presence,
                    maps.presence);
                vc::lasagna::assignResolvedLasagnaCubeRequestChunks(point.nx, maps.nx);
                vc::lasagna::assignResolvedLasagnaCubeRequestChunks(point.ny, maps.ny);
            }
        };

#ifdef _OPENMP
        if (workers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                assignPointChunks(static_cast<size_t>(rawIndex));
            }
        } else
#endif
        {
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex) {
                assignPointChunks(pointIndex);
            }
        }
        if (profile != nullptr)
            profile->predictionAssignSeconds += elapsedSeconds(assignStart);

        const auto materializeStart = TraceClock::now();
        auto materializeOne = [&](size_t pointIndex) {
            auto& out = samples[pointIndex];
            out.options.clear();
            out.options.reserve(optionCount);
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const auto& option = options_[optionIndex];
                const auto& point = prepared[pointIndex * optionCount + optionIndex];
                const auto rawPresence =
                    vc::lasagna::sampleLasagnaChannel(option.presence, point.presence);
                const auto direction =
                    vc::lasagna::sampleLasagnaCompactAxisTensor(
                        option.nx, option.ny, point.nx, point.ny);
                if (!rawPresence.has_value() || !direction.has_value()) {
                    out.options.push_back({});
                    continue;
                }
                out.options.push_back({
                    alignTo(*direction, referenceDirections[pointIndex]),
                    clamp01(*rawPresence / 255.0),
                    true,
                });
            }
        };

        if (workers <= 1) {
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex)
                materializeOne(pointIndex);
        } else {
#ifdef _OPENMP
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                materializeOne(static_cast<size_t>(rawIndex));
#else
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex)
                materializeOne(pointIndex);
#endif
        }
        if (profile != nullptr)
            profile->predictionMaterializeSeconds += elapsedSeconds(materializeStart);
    }

    [[nodiscard]] FiberPredictionSample sample(
        const cv::Vec3d& volumePoint,
        const cv::Vec3d& referenceDirection) const
    {
        FiberPredictionSample out;
        out.options.reserve(options_.size());
        for (const auto& option : options_) {
            const auto rawPresence =
                vc::lasagna::sampleLasagnaChannel(option.presence, *cache_, volumePoint);
            const auto direction =
                vc::lasagna::sampleLasagnaCompactAxisTensor(
                    option.nx, option.ny, *cache_, volumePoint);
            if (!rawPresence.has_value() || !direction.has_value()) {
                out.options.push_back({});
                continue;
            }
            out.options.push_back({
                alignTo(*direction, referenceDirection),
                clamp01(*rawPresence / 255.0),
                true,
            });
        }
        return out;
    }

    [[nodiscard]] size_t optionCount() const noexcept { return options_.size(); }

private:
    std::vector<Option> options_;
    std::vector<OptionSamplingGrid> optionGrids_;
    std::shared_ptr<vc::lasagna::LasagnaChannelChunkCache> cache_;
};

FiberPredictionField::FiberPredictionField(
    const vc::lasagna::LasagnaDataset& dataset,
    size_t maxCachedBytes)
    : impl_(std::make_unique<Impl>(dataset, maxCachedBytes))
{
}

FiberPredictionField::~FiberPredictionField() = default;

vc::lasagna::NormalPrefetchReport FiberPredictionField::prefetchSamples(
    const std::vector<cv::Vec3d>& volumePoints) const
{
    return impl_->prefetchSamples(volumePoints);
}

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    const std::vector<cv::Vec3d>& referenceDirections,
    int parallelThreads,
    std::vector<FiberPredictionSample>& samples) const
{
    impl_->sampleBatch(
        volumePoints,
        referenceDirections,
        parallelThreads,
        samples,
        nullptr);
}

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    const std::vector<cv::Vec3d>& referenceDirections,
    int parallelThreads,
    std::vector<FiberPredictionSample>& samples,
    FiberTraceProfile* profile) const
{
    impl_->sampleBatch(
        volumePoints,
        referenceDirections,
        parallelThreads,
        samples,
        profile);
}

FiberPredictionSample FiberPredictionField::sample(
    const cv::Vec3d& volumePoint,
    const cv::Vec3d& referenceDirection) const
{
    return impl_->sample(volumePoint, referenceDirection);
}

size_t FiberPredictionField::optionCount() const noexcept
{
    return impl_->optionCount();
}

FiberInput loadFiberJson(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input.good()) {
        throw std::runtime_error("could not open fiber JSON: " + path.string());
    }
    const nlohmann::json root = nlohmann::json::parse(input);
    if (root.value("type", std::string{}) != "vc3d_fiber") {
        throw std::runtime_error("fiber JSON type is not vc3d_fiber: " + path.string());
    }
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported vc3d_fiber version: " + path.string());
    }

    FiberInput fiber;
    fiber.path = path;
    fiber.linePointsXyzBase = pointArrayFromJson(root, path, "line_points");
    fiber.controlPointsXyzBase = pointArrayFromJson(root, path, "control_points");
    if (fiber.linePointsXyzBase.size() < 2) {
        throw std::runtime_error(
            "fiber JSON needs at least two line_points: " + path.string());
    }
    if (fiber.controlPointsXyzBase.size() < 2) {
        throw std::runtime_error(
            "fiber JSON needs at least two control_points: " + path.string());
    }
    fiber.controlPointLineIndices.reserve(fiber.controlPointsXyzBase.size());
    for (size_t index = 0; index < fiber.controlPointsXyzBase.size(); ++index) {
        fiber.controlPointLineIndices.push_back(exactLineIndexForControlPoint(
            fiber.linePointsXyzBase, fiber.controlPointsXyzBase[index], index, path));
    }
    for (size_t index = 1; index < fiber.controlPointLineIndices.size(); ++index) {
        if (fiber.controlPointLineIndices[index] <=
            fiber.controlPointLineIndices[index - 1]) {
            throw std::runtime_error(
                "fiber JSON control_points are not strictly increasing along "
                "line_points: " + path.string());
        }
    }
    return fiber;
}

cv::Vec3d referenceTangentToward(
    const std::vector<cv::Vec3d>& line,
    size_t startIndex,
    size_t targetIndex)
{
    if (line.empty() || startIndex >= line.size() || targetIndex >= line.size())
        return {1.0, 0.0, 0.0};
    if (startIndex == targetIndex)
        return {1.0, 0.0, 0.0};
    if (targetIndex > startIndex) {
        for (size_t i = startIndex + 1; i < line.size(); ++i) {
            const cv::Vec3d dir = normalizedOrZero(line[i] - line[startIndex]);
            if (length(dir) > kEpsilon)
                return dir;
            if (i >= targetIndex)
                break;
        }
    } else {
        size_t i = startIndex;
        while (i > 0) {
            --i;
            const cv::Vec3d dir = normalizedOrZero(line[i] - line[startIndex]);
            if (length(dir) > kEpsilon)
                return dir;
            if (i <= targetIndex)
                break;
        }
    }
    return normalizedOr(line[targetIndex] - line[startIndex], {1.0, 0.0, 0.0});
}

FiberTraceOneWayResult traceFiberOneWay(
    const FiberPredictionSource& predictions,
    const FiberTraceOneWayRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress)
{
    if (!finitePoint(request.startPoint) || !finitePoint(request.targetPoint)) {
        throw std::invalid_argument("fiber trace one-way request has non-finite endpoint");
    }
    if (length(request.targetPoint - request.startPoint) <= kEpsilon) {
        throw std::invalid_argument("fiber trace one-way request endpoints must differ");
    }
    if (length(request.targetPlaneNormal) <= kEpsilon) {
        throw std::invalid_argument("fiber trace one-way request target plane normal is degenerate");
    }
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);
    return traceOneWayCore(
        predictions, request, normalSampler, progress, "trace");
}

FiberTraceSegmentResult traceFiberSegment(
    const FiberPredictionSource& predictions,
    const FiberTraceSegmentRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress)
{
    if (request.referenceLine.empty())
        throw std::invalid_argument("fiber trace request has no reference line");
    if (request.startIndex >= request.referenceLine.size() ||
        request.targetIndex >= request.referenceLine.size()) {
        throw std::invalid_argument("fiber trace request control-point index is out of range");
    }
    if (request.startIndex == request.targetIndex)
        throw std::invalid_argument("fiber trace request start and target indices must differ");
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);

    FiberTraceSegmentRequest forwardRequest = request;
    forwardRequest.targetPlaneNormal =
        normalizedOr(request.targetPlaneNormal.value_or(
                         request.referenceLine[request.targetIndex] -
                         request.referenceLine[request.startIndex]),
                     {1.0, 0.0, 0.0});
    FiberTraceSegmentRequest reverseRequest = request;
    reverseRequest.targetPlaneNormal = -*forwardRequest.targetPlaneNormal;

    FiberTraceSegmentResult result;
    const cv::Vec3d start = request.referenceLine[request.startIndex];
    const cv::Vec3d target = request.referenceLine[request.targetIndex];
    const double span = length(target - start);
    FiberTraceOneWayRequest forwardOneWay;
    forwardOneWay.startPoint = start;
    forwardOneWay.targetPoint = target;
    forwardOneWay.initialDirection =
        referenceTangentToward(request.referenceLine, request.startIndex, request.targetIndex);
    forwardOneWay.targetPlaneNormal = *forwardRequest.targetPlaneNormal;
    forwardOneWay.budgetSpanVoxels = span;
    forwardOneWay.config = request.config;
    FiberTraceOneWayRequest reverseOneWay;
    reverseOneWay.startPoint = target;
    reverseOneWay.targetPoint = start;
    reverseOneWay.initialDirection =
        referenceTangentToward(request.referenceLine, request.targetIndex, request.startIndex);
    reverseOneWay.targetPlaneNormal = *reverseRequest.targetPlaneNormal;
    reverseOneWay.budgetSpanVoxels = span;
    reverseOneWay.config = request.config;

    result.forward = traceOneWayCore(
        predictions, forwardOneWay, normalSampler, progress, "forward");
    result.reverse = traceOneWayCore(
        predictions, reverseOneWay, normalSampler, progress, "reverse");

    result.fusedLine = fuseTraces(
        result.forward.points, result.reverse.points,
        std::max(0.0, request.config.fusionGapFactor));
    if (!result.fusedLine.empty()) {
        result.fusedLine.front() = request.referenceLine[request.startIndex];
        result.fusedLine.back() = request.referenceLine[request.targetIndex];
    }

    const cv::Vec3d targetPlaneNormal = *forwardRequest.targetPlaneNormal;
    if (!result.forward.points.empty()) {
        result.forwardEndpointErrorTraceVoxels =
            endpointPlaneError(result.forward.points.back(), target, targetPlaneNormal);
    }
    if (!result.reverse.points.empty()) {
        result.reverseEndpointErrorTraceVoxels =
            endpointPlaneError(result.reverse.points.back(), start, targetPlaneNormal);
    }
    result.maxEndpointErrorTraceVoxels = std::max(
        result.forwardEndpointErrorTraceVoxels,
        result.reverseEndpointErrorTraceVoxels);
    const FiberTraceCoordinateAdapter coordinates(request.config.traceToBaseScale);
    result.maxEndpointErrorBaseVoxels =
        coordinates.traceDistanceToBase(result.maxEndpointErrorTraceVoxels);
    if (request.config.baseVoxelSizeUm.has_value()) {
        result.maxEndpointErrorUm =
            result.maxEndpointErrorBaseVoxels * *request.config.baseVoxelSizeUm;
    }
    result.accepted = result.forward.reachedTargetPlane &&
                      result.reverse.reachedTargetPlane &&
                      result.maxEndpointErrorBaseVoxels <=
                          request.config.endpointAcceptThresholdBaseVoxels;
    if (!result.forward.reachedTargetPlane || !result.reverse.reachedTargetPlane)
        result.reason = "target_plane_not_reached";
    else if (!result.accepted)
        result.reason = "endpoint_error_threshold";
    else
        result.reason = "ok";
    return result;
}

FiberTraceWholeFiberResult traceWholeFiberMetric(
    const FiberPredictionSource& predictions,
    const FiberTraceWholeFiberMetricRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceWholeFiberProgressCallback& progress)
{
    if (!(request.workingToBaseScale > 0.0) ||
        !std::isfinite(request.workingToBaseScale)) {
        throw std::invalid_argument("working-to-base scale must be positive");
    }
    if (!(request.errorThresholdBaseVoxels >= 0.0) ||
        !std::isfinite(request.errorThresholdBaseVoxels)) {
        throw std::invalid_argument("error threshold must be finite and non-negative");
    }
    if (request.fiber.controlPointsXyzBase.size() < 2) {
        throw std::invalid_argument("whole-fiber metric needs at least two control points");
    }
    if (request.fiber.linePointsXyzBase.size() < 2) {
        throw std::invalid_argument("whole-fiber metric needs at least two line points");
    }
    if (request.fiber.controlPointLineIndices.size() !=
        request.fiber.controlPointsXyzBase.size()) {
        throw std::invalid_argument(
            "whole-fiber metric control-point line-index count mismatch");
    }
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);

    const auto lineWorking =
        scaledPoints(request.fiber.linePointsXyzBase, request.workingToBaseScale);
    const auto cpWorking =
        scaledPoints(request.fiber.controlPointsXyzBase, request.workingToBaseScale);
    const auto lineLengths = arclengths(lineWorking);

    std::vector<double> cpArcs;
    cpArcs.reserve(request.fiber.controlPointLineIndices.size());
    for (const size_t lineIndex : request.fiber.controlPointLineIndices) {
        if (lineIndex >= lineLengths.size()) {
            throw std::invalid_argument(
                "whole-fiber metric control-point line index out of range");
        }
        cpArcs.push_back(lineLengths[lineIndex]);
    }

    FiberTraceWholeFiberResult result;
    result.segmentCount = static_cast<int>(cpWorking.size() - 1);
    result.referenceLengthVoxels = cpArcs.back() - cpArcs.front();
    if (request.voxelSizeUm.has_value() && *request.voxelSizeUm > 0.0) {
        result.referenceLengthMeters = referenceLengthMeters(
            result.referenceLengthVoxels,
            request.workingToBaseScale,
            *request.voxelSizeUm);
    }

    auto updateMetricFields = [&]() {
        if (result.referenceLengthVoxels > kEpsilon) {
            result.restartsPerKvx =
                static_cast<double>(result.restartCount) * 1000.0 /
                result.referenceLengthVoxels;
        }
        if (result.referenceLengthMeters.has_value() &&
            *result.referenceLengthMeters > 0.0) {
            result.restartsPerMeter =
                static_cast<double>(result.restartCount) /
                *result.referenceLengthMeters;
        }
    };

    auto emitProgress = [&](int completed,
                            int currentSegment,
                            std::string status,
                            const FiberTraceProgress* traceEvent = nullptr) {
        if (!progress)
            return;
        updateMetricFields();
        FiberTraceWholeFiberProgress event;
        event.completedSegments = completed;
        event.segmentCount = result.segmentCount;
        event.currentSegment = currentSegment;
        event.restartCount = result.restartCount;
        event.restartsPerKvx = result.restartsPerKvx;
        event.restartsPerMeter = result.restartsPerMeter;
        event.referenceLengthMeters = result.referenceLengthMeters;
        event.status = std::move(status);
        if (traceEvent != nullptr) {
            event.traceProgress = *traceEvent;
            event.hasTraceProgress = true;
        }
        progress(event);
    };

    cv::Vec3d currentPoint = cpWorking.front();
    cv::Vec3d currentDirection = referenceTangentToward(
        lineWorking,
        request.fiber.controlPointLineIndices[0],
        request.fiber.controlPointLineIndices[1]);
    result.stitchedTrace.push_back(currentPoint);

    emitProgress(0, 1, "start");
    for (size_t cpIndex = 0; cpIndex + 1 < cpWorking.size(); ++cpIndex) {
        const size_t targetCpIndex = cpIndex + 1;
        const cv::Vec3d target = cpWorking[targetCpIndex];
        const cv::Vec3d referenceStart = cpWorking[cpIndex];
        const cv::Vec3d targetPlaneNormal = normalizedOr(
            target - referenceStart,
            normalizedOr(target - currentPoint, {1.0, 0.0, 0.0}));
        const double budgetSpan = length(target - referenceStart);

        FiberTraceOneWayRequest oneWay;
        oneWay.startPoint = currentPoint;
        oneWay.targetPoint = target;
        oneWay.initialDirection = currentDirection;
        oneWay.targetPlaneNormal = targetPlaneNormal;
        oneWay.budgetSpanVoxels = budgetSpan;
        oneWay.config = request.config;

        FiberTraceWholeFiberSegmentResult segment;
        segment.startControlPointIndex = cpIndex;
        segment.targetControlPointIndex = targetCpIndex;
        segment.referenceArcDistanceVoxels = cpArcs[targetCpIndex] - cpArcs[cpIndex];

        const auto segmentProgress = [&](const FiberTraceProgress& traceEvent) {
            emitProgress(
                static_cast<int>(cpIndex),
                static_cast<int>(targetCpIndex),
                "tracing",
                &traceEvent);
        };
        segment.trace = traceOneWayCore(
            predictions, oneWay, normalSampler, segmentProgress, "fiber");

        if (!segment.trace.points.empty()) {
            segment.inPlaneErrorTraceVoxels =
                endpointPlaneError(segment.trace.points.back(), target, targetPlaneNormal);
        } else {
            segment.inPlaneErrorTraceVoxels = std::numeric_limits<double>::infinity();
        }
        const FiberTraceCoordinateAdapter coordinates(request.workingToBaseScale);
        segment.inPlaneErrorBaseVoxels =
            coordinates.traceDistanceToBase(segment.inPlaneErrorTraceVoxels);
        segment.success = segment.trace.reachedTargetPlane &&
                          segment.inPlaneErrorBaseVoxels <=
                              request.errorThresholdBaseVoxels;
        segment.restart = !segment.success;
        if (segment.success) {
            segment.reason = "ok";
            currentPoint = segment.trace.points.empty()
                ? target
                : segment.trace.points.back();
            currentDirection = terminalTraceDirection(segment.trace.points, currentDirection);
        } else {
            ++result.restartCount;
            segment.reason = segment.trace.reachedTargetPlane
                ? "in_plane_error"
                : segment.trace.reason;
            currentPoint = target;
            if (targetCpIndex + 1 < cpWorking.size()) {
                currentDirection = referenceTangentToward(
                    lineWorking,
                    request.fiber.controlPointLineIndices[targetCpIndex],
                    request.fiber.controlPointLineIndices[targetCpIndex + 1]);
            }
        }

        if (!segment.trace.points.empty()) {
            for (const auto& point : segment.trace.points) {
                if (result.stitchedTrace.empty() ||
                    length(result.stitchedTrace.back() - point) > kEpsilon) {
                    result.stitchedTrace.push_back(point);
                }
            }
        }
        if (result.stitchedTrace.empty() ||
            length(result.stitchedTrace.back() - currentPoint) > kEpsilon) {
            result.stitchedTrace.push_back(currentPoint);
        }

        result.segments.push_back(std::move(segment));
        const auto& saved = result.segments.back();
        emitProgress(
            static_cast<int>(targetCpIndex),
            static_cast<int>(targetCpIndex),
            saved.restart ? "restart:" + saved.reason : "ok");
    }
    updateMetricFields();
    emitProgress(result.segmentCount, result.segmentCount, "done");
    return result;
}

} // namespace vc::fiber_tracer
