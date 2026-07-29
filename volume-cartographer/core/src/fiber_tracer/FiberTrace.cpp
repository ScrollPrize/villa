#include "vc/fiber_tracer/FiberTrace.hpp"

#include "vc/lasagna/ChannelSampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

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
    const FiberTraceConfig& config)
{
    const cv::Vec3d forward = normalizedOr(reference, {1.0, 0.0, 0.0});
    const auto basis = orthonormalBasis(forward);
    const auto offsets = config.coneAngleStepDegrees > 0.0
        ? angleStepConeOffsets(config.coneAngleDegrees, config.coneAngleStepDegrees)
        : legacyGridConeOffsets(config.coneAngleDegrees, config.coneGridSize);
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
    requireFinite(config.smoothnessWeight, "smoothness weight");
    requireFinite(config.smoothnessNormalWeight, "smoothness normal weight");
    requireFinite(config.smoothnessTangentWeight, "smoothness tangent weight");
    requireFinite(config.smoothnessFreeAngleDegrees, "smoothness free angle");
    requireFinite(
        config.cumulativeSmoothnessTangentWeight,
        "cumulative smoothness tangent weight");
    requireFinite(config.maxStepFactor, "max step factor");
    requireFinite(config.fusionGapFactor, "fusion gap factor");
    requireFinite(config.endpointAcceptThresholdUm, "endpoint accept threshold");
    requireFinite(config.voxelSizeUm, "voxel size");
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
    std::vector<cv::Vec3d> points;
    cv::Vec3d previousStepDirection{0.0, 0.0, 0.0};
    cv::Vec3d currentSampleDirection{0.0, 0.0, 0.0};
    cv::Vec3d historyDirection{0.0, 0.0, 0.0};
    double loss = 0.0;
    double tracedLength = 0.0;
    int depth = 0;
    bool reached = false;
    std::string reason;
};

struct CandidateScore {
    double loss = std::numeric_limits<double>::infinity();
    cv::Vec3d selectedCurrentDirection{0.0, 0.0, 0.0};
    double selectedPresence = 0.0;
    bool valid = false;
};

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
    const auto selectedCurrent =
        bestAlignedPrediction(candidateSample, candidateDirection, true);
    if (!selectedCurrent.valid)
        return {};

    cv::Vec3d smoothNormal{0.0, 0.0, 0.0};
    bool smoothNormalValid = false;
    if (normalSampler != nullptr) {
        const auto normalSample = normalSampler->sampleNormal(candidatePoint);
        if (normalSample.valid) {
            smoothNormal = normalSample.normal;
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

[[nodiscard]] bool beamStateLess(const BeamState& a, const BeamState& b)
{
    if (a.loss != b.loss)
        return a.loss < b.loss;
    if (a.depth != b.depth)
        return a.depth < b.depth;
    return a.tracedLength < b.tracedLength;
}

[[nodiscard]] std::vector<BeamState> pruneBeamStates(
    std::vector<BeamState> states,
    int beamWidth,
    double pruneDistanceVoxels)
{
    if (states.empty())
        return {};
    std::sort(states.begin(), states.end(), beamStateLess);
    const size_t keep = static_cast<size_t>(std::max(1, beamWidth));
    const double distance = std::max(0.0, pruneDistanceVoxels);
    if (distance <= 0.0) {
        if (states.size() > keep)
            states.resize(keep);
        return states;
    }

    std::vector<BeamState> out;
    out.reserve(std::min(keep, states.size()));
    for (auto& state : states) {
        const cv::Vec3d point = state.points.empty()
            ? cv::Vec3d{0.0, 0.0, 0.0}
            : state.points.back();
        bool tooClose = false;
        for (const auto& existing : out) {
            const cv::Vec3d existingPoint = existing.points.empty()
                ? cv::Vec3d{0.0, 0.0, 0.0}
                : existing.points.back();
            if (length(point - existingPoint) < distance) {
                tooClose = true;
                break;
            }
        }
        if (tooClose)
            continue;
        out.push_back(std::move(state));
        if (out.size() >= keep)
            break;
    }
    if (!out.empty())
        return out;
    out.push_back(std::move(states.front()));
    return out;
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
    const ScoredDirection startPrediction =
        bestAlignedPrediction(predictions, start, referenceStartDirection, false);
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
    initial.points.push_back(start);
    initial.previousStepDirection = startDirection;
    initial.currentSampleDirection = startDirection;
    initial.historyDirection = startDirection;
    std::vector<BeamState> beams{std::move(initial)};
    std::string reason = "max_step_factor";

    const int lookaheadSteps = request.config.beamWidth <= 1
        ? 1
        : std::max(1, request.config.beamLookaheadSteps);
    int stepIndex = 0;
    while (stepIndex < maxSteps) {
        std::vector<BeamState> expanded = beams;
        int advanced = 0;
        for (; advanced < lookaheadSteps && stepIndex + advanced < maxSteps; ++advanced) {
            std::vector<BeamState> nextFrontier;
            nextFrontier.reserve(expanded.size() * 81);
            for (const auto& beam : expanded) {
                const auto directions = candidateDirections(
                    beam.currentSampleDirection, request.config);
                for (const auto& direction : directions) {
                    BeamState next = beam;
                    const cv::Vec3d currentPoint = beam.points.back();
                    const cv::Vec3d candidatePoint = currentPoint + direction * step;
                    const CandidateScore candidateScore = candidateLoss(
                        predictions,
                        normalSampler,
                        beam,
                        direction,
                        candidatePoint,
                        request.config);
                    if (!candidateScore.valid || !std::isfinite(candidateScore.loss))
                        continue;
                    const auto crossing = interpolatePlaneCrossing(
                        currentPoint,
                        candidatePoint,
                        target,
                        targetPlaneNormal);
                    const cv::Vec3d nextPoint = crossing.value_or(candidatePoint);
                    next.points.push_back(nextPoint);
                    next.tracedLength += length(nextPoint - currentPoint);
                    next.loss += candidateScore.loss;
                    next.previousStepDirection = direction;
                    next.currentSampleDirection = candidateScore.selectedCurrentDirection;
                    next.historyDirection = updateHistoryDirection(
                        beam.historyDirection,
                        direction,
                        beam.depth,
                        request.config.cumulativeSmoothnessSteps);
                    next.depth = beam.depth + 1;
                    next.reached = crossing.has_value();
                    if (next.reached) {
                        next.reason = "target_plane";
                    }
                    nextFrontier.push_back(std::move(next));
                }
            }
            expanded = std::move(nextFrontier);
            if (expanded.empty()) {
                reason = "no_valid_candidates";
                break;
            }
            const auto reachedBegin = std::partition(
                expanded.begin(), expanded.end(), [](const auto& state) {
                    return state.reached;
                });
            if (reachedBegin != expanded.begin()) {
                const auto bestReached = std::min_element(
                    expanded.begin(), reachedBegin, beamStateLess);
                if (progress) {
                    FiberTraceProgress event;
                    event.phase = phase;
                    event.step = stepIndex + advanced + 1;
                    event.maxSteps = maxSteps;
                    event.targetPlaneProgress = 1.0;
                    event.reason = "target_plane";
                    progress(event);
                }
                return {bestReached->points, true, bestReached->reason,
                        static_cast<int>(
                            bestReached->points.size() > 0
                                ? bestReached->points.size() - 1
                                : 0)};
            }
        }
        if (expanded.empty())
            break;

        beams = pruneBeamStates(
            std::move(expanded),
            request.config.beamWidth,
            request.config.beamPruneDistanceVoxels);
        stepIndex += std::max(1, advanced);

        if (progress) {
            const double signedDistance =
                std::abs(pointToPlaneSigned(beams.front().points.back(), target, targetPlaneNormal));
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
        beams.begin(), beams.end(), beamStateLess);
    return {best->points, best->reached, best->reached ? best->reason : reason,
            static_cast<int>(best->points.size() > 0 ? best->points.size() - 1 : 0)};
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
    std::shared_ptr<vc::lasagna::LasagnaChannelChunkCache> cache_;
};

FiberPredictionField::FiberPredictionField(
    const vc::lasagna::LasagnaDataset& dataset,
    size_t maxCachedBytes)
    : impl_(std::make_unique<Impl>(dataset, maxCachedBytes))
{
}

FiberPredictionField::~FiberPredictionField() = default;

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
        result.forwardEndpointErrorVoxels =
            endpointPlaneError(result.forward.points.back(), target, targetPlaneNormal);
    }
    if (!result.reverse.points.empty()) {
        result.reverseEndpointErrorVoxels =
            endpointPlaneError(result.reverse.points.back(), start, targetPlaneNormal);
    }
    result.maxEndpointErrorVoxels = std::max(
        result.forwardEndpointErrorVoxels, result.reverseEndpointErrorVoxels);
    result.maxEndpointErrorUm =
        request.config.voxelSizeUm > 0.0
            ? result.maxEndpointErrorVoxels * request.config.voxelSizeUm
            : result.maxEndpointErrorVoxels;
    const double thresholdVoxels =
        request.config.voxelSizeUm > 0.0
            ? request.config.endpointAcceptThresholdUm / request.config.voxelSizeUm
            : request.config.endpointAcceptThresholdUm;
    result.accepted = result.forward.reachedTargetPlane &&
                      result.reverse.reachedTargetPlane &&
                      result.maxEndpointErrorVoxels <= thresholdVoxels;
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
    if (!(request.errorThresholdVoxels >= 0.0) ||
        !std::isfinite(request.errorThresholdVoxels)) {
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
            segment.inPlaneErrorVoxels =
                endpointPlaneError(segment.trace.points.back(), target, targetPlaneNormal);
        } else {
            segment.inPlaneErrorVoxels = std::numeric_limits<double>::infinity();
        }
        segment.success = segment.trace.reachedTargetPlane &&
                          segment.inPlaneErrorVoxels <= request.errorThresholdVoxels;
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
