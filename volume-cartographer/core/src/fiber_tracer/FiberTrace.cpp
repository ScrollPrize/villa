#include "vc/fiber_tracer/FiberTrace.hpp"

#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/render/DecodedChunkCacheBudget.hpp"

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
constexpr float kTraceEpsilon = 1.0e-6f;
constexpr float kTracePi = 3.14159265358979323846f;

using TraceVec = cv::Vec3f;

using TraceClock = std::chrono::steady_clock;

[[nodiscard]] double elapsedSeconds(TraceClock::time_point start)
{
    return std::chrono::duration<double>(TraceClock::now() - start).count();
}

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] float traceLength(const TraceVec& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] TraceVec traceNormalizedOr(
    const TraceVec& v,
    const TraceVec& fallback)
{
    const float len = traceLength(v);
    if (!(len > kTraceEpsilon) || !std::isfinite(len))
        return fallback;
    return v / len;
}

[[nodiscard]] TraceVec traceNormalizedOrZero(const TraceVec& v)
{
    return traceNormalizedOr(v, {0.0f, 0.0f, 0.0f});
}

[[nodiscard]] float traceClamp01(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

[[nodiscard]] float traceClampSignedUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, -1.0f, 1.0f);
}

[[nodiscard]] float traceClampedPositiveDot(const TraceVec& a, const TraceVec& b)
{
    return traceClamp01(a.dot(b));
}

[[nodiscard]] float traceAngleBetweenUnit(const TraceVec& a, const TraceVec& b)
{
    return std::acos(traceClampSignedUnit(a.dot(b)));
}

[[nodiscard]] TraceVec traceAlignTo(
    const TraceVec& direction,
    const TraceVec& reference)
{
    TraceVec out = traceNormalizedOrZero(direction);
    const TraceVec ref = traceNormalizedOrZero(reference);
    if (traceLength(out) <= kTraceEpsilon)
        return out;
    if (traceLength(ref) > kTraceEpsilon && out.dot(ref) < 0.0f)
        out *= -1.0f;
    return out;
}

[[nodiscard]] TraceVec toTraceVec(const cv::Vec3d& value)
{
    return {
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2])};
}

[[nodiscard]] cv::Vec3d toVec3d(const TraceVec& value)
{
    return {value[0], value[1], value[2]};
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

[[nodiscard]] std::array<TraceVec, 2> traceOrthonormalBasis(
    const TraceVec& direction)
{
    const TraceVec axis = traceNormalizedOr(direction, {1.0f, 0.0f, 0.0f});
    const std::array<TraceVec, 3> refs = {
        TraceVec{1.0f, 0.0f, 0.0f},
        TraceVec{0.0f, 1.0f, 0.0f},
        TraceVec{0.0f, 0.0f, 1.0f},
    };
    size_t refIndex = 0;
    float bestAbsDot = std::abs(axis.dot(refs[0]));
    for (size_t index = 1; index < refs.size(); ++index) {
        const float value = std::abs(axis.dot(refs[index]));
        if (value < bestAbsDot) {
            bestAbsDot = value;
            refIndex = index;
        }
    }
    const TraceVec b0 = traceNormalizedOrZero(axis.cross(refs[refIndex]));
    const TraceVec b1 = traceNormalizedOrZero(axis.cross(b0));
    return {b0, b1};
}

struct ConeOffset {
    float u = 0.0f;
    float v = 0.0f;
    float radius2 = 0.0f;
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
    TraceVec direction{0.0f, 0.0f, 0.0f};
    float presence = 0.0f;
    bool valid = false;
};

[[nodiscard]] ScoredDirection bestAlignedPrediction(
    const FiberPredictionSample& sample,
    const TraceVec& referenceDirection,
    bool weightByPresence)
{
    ScoredDirection best;
    float bestScore = -std::numeric_limits<float>::infinity();
    const TraceVec reference = traceNormalizedOrZero(referenceDirection);
    for (const auto& option : sample.options) {
        if (!option.valid)
            continue;
        const TraceVec direction = traceAlignTo(option.direction, referenceDirection);
        float score = traceClamp01(direction.dot(reference));
        const float presence = traceClamp01(option.presence);
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
    const TraceVec& referenceDirection,
    bool weightByPresence)
{
    return bestAlignedPrediction(
        predictions.sample(point, toVec3d(referenceDirection)),
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

[[nodiscard]] float traceExcessAngleSquared(float angle, float freeAngle)
{
    const float excess = std::max(0.0f, angle - freeAngle);
    return excess * excess;
}

[[nodiscard]] float isotropicSmoothnessLoss(
    const TraceVec& previousStepDirection,
    const TraceVec& candidateStepDirection,
    const FiberTraceConfig& config)
{
    const float freeAngle =
        static_cast<float>(config.smoothnessFreeAngleDegrees) * kTracePi / 180.0f;
    return static_cast<float>(config.smoothnessWeight) *
           traceExcessAngleSquared(
               traceAngleBetweenUnit(previousStepDirection, candidateStepDirection),
               freeAngle);
}

[[nodiscard]] float smoothnessLoss(
    const TraceVec& previousStepDirection,
    const TraceVec& candidateStepDirection,
    const TraceVec& normal,
    bool normalValid,
    const FiberTraceConfig& config)
{
    const TraceVec& prev = previousStepDirection;
    const TraceVec& cand = candidateStepDirection;
    constexpr float kTraceEpsilon2 = kTraceEpsilon * kTraceEpsilon;
    if (prev.dot(prev) <= kTraceEpsilon2 || cand.dot(cand) <= kTraceEpsilon2)
        return 0.0f;

    const float isotropic = isotropicSmoothnessLoss(prev, cand, config);
    const TraceVec& n = normal;
    const float normalWeight = static_cast<float>(config.smoothnessNormalWeight);
    const float tangentWeight = static_cast<float>(config.smoothnessTangentWeight);
    if (!normalValid || n.dot(n) <= kTraceEpsilon2)
        return isotropic;

    const float prevN = traceClampSignedUnit(prev.dot(n));
    const float candN = traceClampSignedUnit(cand.dot(n));
    const TraceVec prevT = traceNormalizedOrZero(prev - n * prevN);
    const TraceVec candT = traceNormalizedOrZero(cand - n * candN);
    const bool tangentOk =
        prevT.dot(prevT) > kTraceEpsilon2 && candT.dot(candT) > kTraceEpsilon2;
    const float tangentAngle = tangentOk
        ? traceAngleBetweenUnit(prevT, candT)
        : traceAngleBetweenUnit(prev, cand);
    const float normalAngle = std::abs(std::asin(candN) - std::asin(prevN));
    const float freeAngle =
        static_cast<float>(config.smoothnessFreeAngleDegrees) * kTracePi / 180.0f;
    return tangentWeight * traceExcessAngleSquared(tangentAngle, freeAngle) +
           normalWeight * traceExcessAngleSquared(normalAngle, freeAngle);
}

[[nodiscard]] float cumulativeTangentSmoothnessLoss(
    const TraceVec& historyDirection,
    const TraceVec& candidateStepDirection,
    const TraceVec& normal,
    bool normalValid,
    const FiberTraceConfig& config)
{
    const float weight = static_cast<float>(config.cumulativeSmoothnessTangentWeight);
    if (!(weight > 0.0f))
        return 0.0f;
    const TraceVec& history = historyDirection;
    const TraceVec& cand = candidateStepDirection;
    const TraceVec& n = normal;
    constexpr float kTraceEpsilon2 = kTraceEpsilon * kTraceEpsilon;
    if (!normalValid ||
        history.dot(history) <= kTraceEpsilon2 ||
        cand.dot(cand) <= kTraceEpsilon2 ||
        n.dot(n) <= kTraceEpsilon2) {
        return 0.0f;
    }
    const float historyN = traceClampSignedUnit(history.dot(n));
    const float candN = traceClampSignedUnit(cand.dot(n));
    const TraceVec historyT = traceNormalizedOrZero(history - n * historyN);
    const TraceVec candT = traceNormalizedOrZero(cand - n * candN);
    if (historyT.dot(historyT) <= kTraceEpsilon2 ||
        candT.dot(candT) <= kTraceEpsilon2) {
        return 0.0f;
    }
    const float freeAngle =
        static_cast<float>(config.smoothnessFreeAngleDegrees) * kTracePi / 180.0f;
    return weight * traceExcessAngleSquared(
        traceAngleBetweenUnit(historyT, candT), freeAngle);
}

[[nodiscard]] TraceVec updateHistoryDirection(
    const TraceVec& historyDirection,
    const TraceVec& chosenDirection,
    int depth,
    int cumulativeSmoothnessSteps)
{
    const TraceVec chosen = traceNormalizedOrZero(chosenDirection);
    if (depth <= 0 || cumulativeSmoothnessSteps <= 1)
        return chosen;
    const float count = static_cast<float>(
        std::clamp(depth, 1, cumulativeSmoothnessSteps - 1));
    return traceNormalizedOr(
        chosen + traceNormalizedOrZero(historyDirection) * count, chosen);
}

struct BeamState {
    struct PathNode {
        TraceVec point{0.0f, 0.0f, 0.0f};
        std::shared_ptr<const PathNode> previous;
        size_t length = 1;
    };

    std::shared_ptr<const PathNode> path;
    TraceVec previousStepDirection{0.0f, 0.0f, 0.0f};
    TraceVec currentSampleDirection{0.0f, 0.0f, 0.0f};
    TraceVec historyDirection{0.0f, 0.0f, 0.0f};
    float loss = 0.0f;
    float tracedLength = 0.0f;
    int depth = 0;
    bool valid = false;
    bool reached = false;
    std::string reason;
};

[[nodiscard]] std::shared_ptr<const BeamState::PathNode> appendBeamPathPoint(
    std::shared_ptr<const BeamState::PathNode> previous,
    const TraceVec& point)
{
    auto node = std::make_shared<BeamState::PathNode>();
    node->point = point;
    node->previous = std::move(previous);
    node->length = node->previous ? node->previous->length + 1 : 1;
    return node;
}

[[nodiscard]] TraceVec beamEndpoint(const BeamState& state)
{
    return state.path ? state.path->point : TraceVec{0.0f, 0.0f, 0.0f};
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
        out.push_back(toVec3d(node->point));
    std::reverse(out.begin(), out.end());
    return out;
}

struct CandidateScore {
    float loss = std::numeric_limits<float>::infinity();
    TraceVec selectedCurrentDirection{0.0f, 0.0f, 0.0f};
    float selectedPresence = 0.0f;
    bool valid = false;
};

struct CandidateTask {
    uint32_t beamIndex = 0;
    TraceVec direction{0.0f, 0.0f, 0.0f};
};

struct FrontierCandidate {
    TraceVec point{0.0f, 0.0f, 0.0f};
    float loss = 0.0f;
    int depth = 0;
    size_t originalIndex = std::numeric_limits<size_t>::max();
    bool valid = false;
    bool reached = false;
};

struct CandidateScoringScratch {
    std::vector<CandidateTask> tasks;
    std::vector<CandidateScore> scores;
    std::vector<CandidateTask> lookaheadTasks;
    std::vector<CandidateScore> lookaheadScores;
    std::vector<size_t> lookaheadGlobalIndices;
    std::vector<TraceVec> candidatePoints;
    std::vector<TraceVec> referenceDirections;
    std::vector<FiberPredictionSample> predictionSamples;
    std::vector<vc::lasagna::NormalSampleWithDerivative> normalSamplesWithDerivative;
    std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample> floatNormalSamples;
    std::vector<vc::lasagna::NormalSample> fallbackNormalSamples;
    std::vector<FrontierCandidate> frontierCandidates;
};

[[nodiscard]] FrontierCandidate makeFrontierCandidate(
    const BeamState& beam,
    const TraceVec& candidatePoint,
    const CandidateScore& candidateScore,
    const TraceVec& target,
    const TraceVec& targetPlaneNormal,
    size_t originalIndex);

struct FrontierScoreOutput {
    std::vector<FrontierCandidate>* candidates = nullptr;
    const TraceVec* target = nullptr;
    const TraceVec* targetPlaneNormal = nullptr;
    const std::vector<size_t>* originalIndices = nullptr;
    size_t offset = 0;
};

void storeScoredFrontierCandidate(
    const FrontierScoreOutput* output,
    const std::vector<BeamState>& beams,
    const CandidateTask& task,
    const TraceVec& candidatePoint,
    const CandidateScore& score,
    size_t taskIndex)
{
    if (output == nullptr || !score.valid || !std::isfinite(score.loss))
        return;
    const size_t originalIndex = output->originalIndices != nullptr
        ? (*output->originalIndices)[taskIndex]
        : taskIndex;
    (*output->candidates)[output->offset + taskIndex] = makeFrontierCandidate(
        beams[task.beamIndex],
        candidatePoint,
        score,
        *output->target,
        *output->targetPlaneNormal,
        originalIndex);
}

void fillCandidateTasksForBeam(
    CandidateTask* out,
    TraceVec* outPoints,
    const std::vector<BeamState>& beams,
    size_t beamIndex,
    const std::vector<ConeOffset>& offsets,
    float step)
{
    const BeamState& beam = beams[beamIndex];
    const TraceVec currentPoint = beamEndpoint(beam);
    const TraceVec forward = traceNormalizedOr(
        beam.currentSampleDirection, {1.0f, 0.0f, 0.0f});
    const auto basis = traceOrthonormalBasis(forward);
    if (offsets.empty()) {
        out[0] = {static_cast<uint32_t>(beamIndex), forward};
        outPoints[0] = currentPoint + forward * step;
        return;
    }
    for (size_t offsetIndex = 0; offsetIndex < offsets.size(); ++offsetIndex) {
        const auto& offset = offsets[offsetIndex];
        const TraceVec direction = traceNormalizedOr(
            forward + basis[0] * offset.u + basis[1] * offset.v,
            forward);
        out[offsetIndex] = {static_cast<uint32_t>(beamIndex), direction};
        outPoints[offsetIndex] = currentPoint + direction * step;
    }
}

void buildCandidateTasks(
    std::vector<CandidateTask>& tasks,
    std::vector<TraceVec>& candidatePoints,
    const std::vector<BeamState>& beams,
    const std::vector<ConeOffset>& offsets,
    float step)
{
    if (beams.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::overflow_error("fiber trace has too many beam states");
    const size_t candidatesPerBeam = std::max<size_t>(1, offsets.size());
    tasks.resize(beams.size() * candidatesPerBeam);
    candidatePoints.resize(tasks.size());
    for (size_t beamIndex = 0; beamIndex < beams.size(); ++beamIndex) {
        fillCandidateTasksForBeam(
            tasks.data() + beamIndex * candidatesPerBeam,
            candidatePoints.data() + beamIndex * candidatesPerBeam,
            beams,
            beamIndex,
            offsets,
            step);
    }
}

void buildCandidateTasksForOrderedParents(
    std::vector<CandidateTask>& tasks,
    std::vector<TraceVec>& candidatePoints,
    std::vector<size_t>& globalIndices,
    const std::vector<BeamState>& beams,
    const std::vector<ConeOffset>& offsets,
    float step,
    const std::vector<size_t>& parentOrder,
    size_t orderBegin,
    size_t orderEnd)
{
    if (beams.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::overflow_error("fiber trace has too many beam states");
    const size_t candidatesPerBeam = std::max<size_t>(1, offsets.size());
    const size_t parentCount = orderEnd - orderBegin;
    tasks.resize(parentCount * candidatesPerBeam);
    candidatePoints.resize(tasks.size());
    globalIndices.resize(tasks.size());
    for (size_t localParent = 0; localParent < parentCount; ++localParent) {
        const size_t parentIndex = parentOrder[orderBegin + localParent];
        const size_t firstLocalTask = localParent * candidatesPerBeam;
        fillCandidateTasksForBeam(
            tasks.data() + firstLocalTask,
            candidatePoints.data() + firstLocalTask,
            beams,
            parentIndex,
            offsets,
            step);
        for (size_t offsetIndex = 0; offsetIndex < candidatesPerBeam; ++offsetIndex) {
            globalIndices[firstLocalTask + offsetIndex] =
                parentIndex * candidatesPerBeam + offsetIndex;
        }
    }
}

[[nodiscard]] CandidateScore candidateLossFromSample(
    const FiberPredictionSample& candidateSample,
    const BeamState& beam,
    const TraceVec& candidateDirection,
    const FiberTraceConfig& config,
    const vc::lasagna::LasagnaNormalSampler::FloatNormalSample* precomputedNormal = nullptr)
{
    const auto selectedCurrent =
        bestAlignedPrediction(candidateSample, candidateDirection, true);
    if (!selectedCurrent.valid)
        return {};

    TraceVec smoothNormal{0.0f, 0.0f, 0.0f};
    bool smoothNormalValid = false;
    if (normalAwareSmoothnessEnabled(config) && precomputedNormal != nullptr) {
        if (precomputedNormal->valid) {
            smoothNormal = precomputedNormal->normal;
            smoothNormalValid = true;
        }
    }

    float bestLoss = std::numeric_limits<float>::infinity();
    for (const auto& option : candidateSample.options) {
        if (!option.valid)
            continue;
        const TraceVec candidateSampleDirection =
            traceAlignTo(option.direction, candidateDirection);
        const float presence = traceClamp01(option.presence);

        const TraceVec& prevStep = beam.previousStepDirection;
        const TraceVec& currentSample = beam.currentSampleDirection;
        const TraceVec& currentStep = candidateDirection;

        float score = presence;
        score *= traceClampedPositiveDot(prevStep, currentStep);
        score *= traceClampedPositiveDot(prevStep, currentSample);
        score *= traceClampedPositiveDot(prevStep, candidateSampleDirection);
        score *= traceClampedPositiveDot(currentSample, currentStep);
        score *= traceClampedPositiveDot(currentSample, candidateSampleDirection);
        score *= traceClampedPositiveDot(currentStep, candidateSampleDirection);

        const float loss = (1.0f - score) +
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

void decodePredictionAndNormalCornerPoint(
    const vc::lasagna::LasagnaCornerBatch& corners,
    size_t firstVolume,
    size_t optionCount,
    size_t pointIndex,
    const TraceVec& referenceDirection,
    FiberPredictionSample& sample,
    size_t normalFirstVolume = std::numeric_limits<size_t>::max(),
    vc::lasagna::LasagnaNormalSampler::FloatNormalSample* normal = nullptr)
{
    sample.options.clear();
    sample.options.reserve(optionCount);
    const bool valid = corners.valid[pointIndex] != 0;
    const auto weights = vc::lasagna::lasagnaCornerWeights(
        corners.fractionsXYZ[pointIndex]);
    for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
        const size_t volume = firstVolume + optionIndex * 3;
        if (!valid) {
            sample.options.push_back({});
            continue;
        }
        TraceVec direction = vc::lasagna::interpolateLasagnaCompactAxisCorners(
            corners.values[volume + 1][pointIndex],
            corners.values[volume + 2][pointIndex],
            weights,
            referenceDirection);
        if (direction.dot(referenceDirection) < 0.0f)
            direction *= -1.0f;
        const float rawPresence = vc::lasagna::interpolateLasagnaCorners(
            corners.values[volume][pointIndex], weights);
        sample.options.push_back({
            direction,
            std::clamp(rawPresence / 255.0f, 0.0f, 1.0f),
            direction.dot(direction) > 1.0e-12f});
    }
    if (normal == nullptr)
        return;
    if (!valid ||
        !(vc::lasagna::interpolateLasagnaCorners(
              corners.values[normalFirstVolume][pointIndex], weights) > 0.0f)) {
        *normal = {};
        return;
    }
    normal->normal = vc::lasagna::interpolateLasagnaCompactAxisCorners(
        corners.values[normalFirstVolume + 1][pointIndex],
        corners.values[normalFirstVolume + 2][pointIndex],
        weights);
    normal->valid = normal->normal.dot(normal->normal) > 1.0e-12f;
}

[[nodiscard]] CandidateScore candidateLossFromCornerValues(
    std::span<const std::array<uint8_t, 8>> volumeCorners,
    const cv::Vec3f& fractionXYZ,
    bool valid,
    size_t optionCount,
    const BeamState& beam,
    const TraceVec& candidateDirection,
    const FiberTraceConfig& config)
{
    if (!valid)
        return {};

    const auto weights = vc::lasagna::lasagnaCornerWeights(fractionXYZ);
    const size_t normalFirstVolume = optionCount * 3;
    TraceVec smoothNormal{0.0f, 0.0f, 0.0f};
    bool smoothNormalValid = false;
    if (vc::lasagna::interpolateLasagnaCorners(
            volumeCorners[normalFirstVolume], weights) > 0.0f) {
        smoothNormal = vc::lasagna::interpolateLasagnaCompactAxisCorners(
            volumeCorners[normalFirstVolume + 1],
            volumeCorners[normalFirstVolume + 2],
            weights);
        smoothNormalValid = smoothNormal.dot(smoothNormal) > 1.0e-12f;
    }

    const TraceVec& reference = candidateDirection;
    const TraceVec& prevStep = beam.previousStepDirection;
    const TraceVec& currentSample = beam.currentSampleDirection;
    const TraceVec& currentStep = candidateDirection;
    float bestAlignmentScore = -std::numeric_limits<float>::infinity();
    ScoredDirection selectedCurrent;
    float bestLoss = std::numeric_limits<float>::infinity();
    for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
        const size_t volume = optionIndex * 3;
        TraceVec direction = vc::lasagna::interpolateLasagnaCompactAxisCorners(
            volumeCorners[volume + 1],
            volumeCorners[volume + 2],
            weights,
            candidateDirection);
        if (!(direction.dot(direction) > 1.0e-12f))
            continue;

        const TraceVec alignedDirection = direction;
        const float presence = traceClamp01(
            vc::lasagna::interpolateLasagnaCorners(
                volumeCorners[volume], weights) /
            255.0f);
        const float alignmentScore =
            traceClamp01(alignedDirection.dot(reference)) * presence;
        if (alignmentScore > bestAlignmentScore) {
            bestAlignmentScore = alignmentScore;
            selectedCurrent = {alignedDirection, presence, true};
        }

        float score = presence;
        score *= traceClampedPositiveDot(prevStep, currentStep);
        score *= traceClampedPositiveDot(prevStep, currentSample);
        score *= traceClampedPositiveDot(prevStep, alignedDirection);
        score *= traceClampedPositiveDot(currentSample, currentStep);
        score *= traceClampedPositiveDot(currentSample, alignedDirection);
        score *= traceClampedPositiveDot(currentStep, alignedDirection);

        const float loss = (1.0f - score) +
            smoothnessLoss(
                prevStep,
                currentStep,
                smoothNormal,
                smoothNormalValid,
                config) +
            cumulativeTangentSmoothnessLoss(
                beam.historyDirection,
                currentStep,
                smoothNormal,
                smoothNormalValid,
                config);
        if (loss < bestLoss)
            bestLoss = loss;
    }
    return {
        bestLoss,
        selectedCurrent.direction,
        selectedCurrent.presence,
        std::isfinite(bestLoss)};
}

[[nodiscard]] CandidateScore candidateLossFromCorners(
    const vc::lasagna::LasagnaCornerBatch& corners,
    size_t optionCount,
    size_t pointIndex,
    const BeamState& beam,
    const TraceVec& candidateDirection,
    const FiberTraceConfig& config)
{
    std::vector<std::array<uint8_t, 8>> pointCorners(corners.values.size());
    for (size_t volumeIndex = 0; volumeIndex < corners.values.size(); ++volumeIndex)
        pointCorners[volumeIndex] = corners.values[volumeIndex][pointIndex];
    return candidateLossFromCornerValues(
        pointCorners,
        corners.fractionsXYZ[pointIndex],
        corners.valid[pointIndex] != 0,
        optionCount,
        beam,
        candidateDirection,
        config);
}

[[nodiscard]] CandidateScore candidateLoss(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const BeamState& beam,
    const TraceVec& candidateDirection,
    const TraceVec& candidatePoint,
    const FiberTraceConfig& config)
{
    const auto candidateSample =
        predictions.sample(toVec3d(candidatePoint), toVec3d(candidateDirection));
    vc::lasagna::NormalSample normalSample;
    const vc::lasagna::NormalSample* normal = nullptr;
    if (normalAwareSmoothnessEnabled(config) && normalSampler != nullptr) {
        normalSample = normalSampler->sampleNormal(toVec3d(candidatePoint));
        normal = &normalSample;
    }
    vc::lasagna::LasagnaNormalSampler::FloatNormalSample floatNormal;
    const vc::lasagna::LasagnaNormalSampler::FloatNormalSample* traceNormal = nullptr;
    if (normal != nullptr) {
        floatNormal = {toTraceVec(normal->normal), normal->valid};
        traceNormal = &floatNormal;
    }
    return candidateLossFromSample(
        candidateSample,
        beam,
        candidateDirection,
        config,
        traceNormal);
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
    const std::vector<TraceVec>& candidatePoints,
    const FiberTraceConfig& config,
    int parallelThreads,
    std::vector<vc::lasagna::NormalSampleWithDerivative>& samples,
    std::vector<vc::lasagna::NormalSample>& fallbackSamples,
    std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample>& out,
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
            parallelThreads,
            out);
    } else {
        std::vector<cv::Vec3d> doublePoints;
        doublePoints.reserve(candidatePoints.size());
        for (const auto& point : candidatePoints)
            doublePoints.push_back(toVec3d(point));
        report = normalSampler->sampleNormalBatch(doublePoints, false, samples);
        fallbackSamples.resize(samples.size());
        out.resize(samples.size());
        for (size_t index = 0; index < samples.size(); ++index) {
            fallbackSamples[index] = samples[index].sample;
            out[index] = {
                toTraceVec(fallbackSamples[index].normal),
                fallbackSamples[index].valid};
        }
    }
    if (profile != nullptr) {
        profile->normalPrefetchSeconds += report.prefetchMs / 1000.0;
        profile->normalMaterializeSeconds += report.materializeMs / 1000.0;
    }
}

[[nodiscard]] const std::vector<CandidateScore>& scoreCandidateTasks(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const std::vector<BeamState>& beams,
    const std::vector<CandidateTask>& tasks,
    const std::vector<TraceVec>& candidatePoints,
    const FiberTraceConfig& config,
    int lookaheadDepth,
    const std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample>& normals,
    const FrontierScoreOutput* frontierOutput,
    CandidateScoringScratch& scratch,
    FiberTraceProfile* profile)
{
    if (candidatePoints.size() != tasks.size())
        throw std::invalid_argument("fiber candidate task/point size mismatch");
    auto& scores = scratch.scores;
    scores.clear();
    scores.resize(tasks.size());
    if (tasks.empty())
        return scores;
    if (profile != nullptr) {
        profile->candidateTasks += tasks.size();
        if (lookaheadDepth <= 1) {
            ++profile->candidateDepth1Batches;
            profile->candidateDepth1Points += tasks.size();
            profile->candidateDepth1BatchSizes.push_back(tasks.size());
        } else {
            ++profile->candidateDepth2Batches;
            profile->candidateDepth2Points += tasks.size();
            profile->candidateDepth2BatchSizes.push_back(tasks.size());
        }
    }

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
                candidatePoints[index],
                config);
            storeScoredFrontierCandidate(
                frontierOutput,
                beams,
                task,
                candidatePoints[index],
                scores[index],
                index);
        }
        if (profile != nullptr)
            profile->candidateScoreSeconds += elapsedSeconds(scoreStart);
        return scores;
    }

    auto& referenceDirections = scratch.referenceDirections;
    referenceDirections.clear();
    auto& predictionSamples = scratch.predictionSamples;
    const auto predictionStart = TraceClock::now();
    const auto* field = dynamic_cast<const FiberPredictionField*>(&predictions);
    const auto* lasagnaSampler =
        dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(normalSampler);
    struct CornerScoreContext {
        const std::vector<BeamState>* beams;
        const std::vector<CandidateTask>* tasks;
        const std::vector<TraceVec>* candidatePoints;
        const FiberTraceConfig* config;
        std::vector<CandidateScore>* scores;
        const FrontierScoreOutput* frontierOutput;
        size_t optionCount;
    } cornerContext{
        &beams,
        &tasks,
        &candidatePoints,
        &config,
        &scores,
        frontierOutput,
        field ? field->optionCount() : 0};
    const auto scoreCornerPoint = +[](
        void* rawContext,
        size_t pointIndex,
        const cv::Vec3f& fractionXYZ,
        bool valid,
        std::span<const std::array<uint8_t, 8>> volumeCorners) {
        const auto& context = *static_cast<CornerScoreContext*>(rawContext);
        const CandidateTask& task = (*context.tasks)[pointIndex];
        if (valid && volumeCorners.size() != context.optionCount * 3 + 3) {
            throw std::runtime_error(
                "fiber prediction corner visitor returned inconsistent dimensions");
        }
        CandidateScore& score = (*context.scores)[pointIndex];
        score = candidateLossFromCornerValues(
            volumeCorners,
            fractionXYZ,
            valid,
            context.optionCount,
            (*context.beams)[task.beamIndex],
            task.direction,
            *context.config);
        storeScoredFrontierCandidate(
            context.frontierOutput,
            *context.beams,
            task,
            (*context.candidatePoints)[pointIndex],
            score,
            pointIndex);
    };
    const bool fusedCornerScoring =
        field != nullptr && lasagnaSampler != nullptr && normals.empty() &&
        normalAwareSmoothnessEnabled(config) &&
        field->visitCornerBatchWithNormals(
            *lasagnaSampler,
            candidatePoints,
            workers,
            &cornerContext,
            scoreCornerPoint,
            lookaheadDepth,
            profile);
    if (fusedCornerScoring) {
        if (profile != nullptr)
            profile->predictionBatchSeconds += elapsedSeconds(predictionStart);
        return scores;
    }

    referenceDirections.reserve(tasks.size());
    for (const auto& task : tasks)
        referenceDirections.push_back(task.direction);
    if (field != nullptr) {
        field->sampleBatch(
            candidatePoints,
            referenceDirections,
            workers,
            predictionSamples,
            profile);
    } else {
        std::vector<cv::Vec3d> doublePoints;
        std::vector<cv::Vec3d> doubleDirections;
        doublePoints.reserve(candidatePoints.size());
        doubleDirections.reserve(referenceDirections.size());
        for (const auto& point : candidatePoints)
            doublePoints.push_back(toVec3d(point));
        for (const auto& direction : referenceDirections)
            doubleDirections.push_back(toVec3d(direction));
        predictions.sampleBatch(
            doublePoints,
            doubleDirections,
            workers,
            predictionSamples);
    }
    if (profile != nullptr)
        profile->predictionBatchSeconds += elapsedSeconds(predictionStart);
    if (predictionSamples.size() < tasks.size()) {
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
        scratch.fallbackNormalSamples,
        scratch.floatNormalSamples,
        profile);
    if (profile != nullptr)
        profile->normalBatchSeconds += elapsedSeconds(normalStart);
    const auto& normalsForScoring =
        normals.empty() ? scratch.floatNormalSamples : normals;

    const auto scoreStart = TraceClock::now();
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    auto scoreOne = [&](size_t index) {
        const auto& task = tasks[index];
        const vc::lasagna::LasagnaNormalSampler::FloatNormalSample* normal =
            index < normalsForScoring.size() ? &normalsForScoring[index] : nullptr;
        scores[index] = candidateLossFromSample(
            predictionSamples[index],
            beams[task.beamIndex],
            task.direction,
            config,
            normal);
        storeScoredFrontierCandidate(
            frontierOutput,
            beams,
            task,
            candidatePoints[index],
            scores[index],
            index);
    };

#ifdef _OPENMP
    const auto count = static_cast<std::ptrdiff_t>(tasks.size());
    #pragma omp parallel for schedule(dynamic, 64) num_threads(workers)
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

[[nodiscard]] float tracePointToPlaneSigned(
    const TraceVec& point,
    const TraceVec& planePoint,
    const TraceVec& planeNormal)
{
    return (point - planePoint).dot(planeNormal);
}

[[nodiscard]] std::optional<TraceVec> interpolatePlaneCrossing(
    const TraceVec& start,
    const TraceVec& end,
    const TraceVec& planePoint,
    const TraceVec& planeNormal)
{
    const float d0 = tracePointToPlaneSigned(start, planePoint, planeNormal);
    const float d1 = tracePointToPlaneSigned(end, planePoint, planeNormal);
    if (d0 == 0.0f)
        return start;
    if (d0 * d1 > 0.0f)
        return std::nullopt;
    const float denom = d0 - d1;
    if (std::abs(denom) <= kTraceEpsilon)
        return end;
    const float t = std::clamp(d0 / denom, 0.0f, 1.0f);
    return start * (1.0f - t) + end * t;
}

[[nodiscard]] FrontierCandidate makeFrontierCandidate(
    const BeamState& beam,
    const TraceVec& candidatePoint,
    const CandidateScore& candidateScore,
    const TraceVec& target,
    const TraceVec& targetPlaneNormal,
    size_t originalIndex)
{
    const TraceVec currentPoint = beamEndpoint(beam);
    const auto crossing = interpolatePlaneCrossing(
        currentPoint,
        candidatePoint,
        target,
        targetPlaneNormal);
    const TraceVec nextPoint = crossing.value_or(candidatePoint);
    return {
        nextPoint,
        beam.loss + candidateScore.loss,
        beam.depth + 1,
        originalIndex,
        true,
        crossing.has_value(),
    };
}

[[nodiscard]] float beamPruneScore(const BeamState& state)
{
    return state.loss;
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
    const float distance = std::max(0.0f, static_cast<float>(pruneDistanceVoxels));
    std::vector<size_t> selected;
    selected.reserve(std::min(keep, states.size()));
    std::vector<unsigned char> unavailable(states.size(), 0);
    const float distance2 = distance * distance;

    while (selected.size() < keep) {
        std::optional<size_t> best;
        float bestScore = 0.0f;
        for (size_t index = 0; index < states.size(); ++index) {
            if (unavailable[index])
                continue;
            const float score = beamPruneScore(states[index]);
            if (!std::isfinite(score))
                continue;
            if (distance > 0.0f) {
                const TraceVec point = beamEndpoint(states[index]);
                bool tooClose = false;
                for (const size_t existingIndex : selected) {
                    const TraceVec delta = point - beamEndpoint(states[existingIndex]);
                    if (delta.dot(delta) < distance2) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose)
                    continue;
            }
            if (!best.has_value() || score < bestScore ||
                (score == bestScore &&
                 states[index].depth < states[*best].depth)) {
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

[[nodiscard]] float frontierPruneScore(const FrontierCandidate& candidate)
{
    return candidate.loss;
}

[[nodiscard]] std::vector<size_t> selectFrontierCandidateIndices(
    const std::vector<FrontierCandidate>& candidates,
    int beamWidth,
    double pruneDistanceVoxels)
{
    if (candidates.empty())
        return {};
    const size_t keep = static_cast<size_t>(std::max(1, beamWidth));
    const float distance = std::max(0.0f, static_cast<float>(pruneDistanceVoxels));
    const float distance2 = distance * distance;
    std::vector<size_t> selected;
    selected.reserve(std::min(keep, candidates.size()));
    std::vector<size_t> ordered;
    ordered.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (candidates[index].valid &&
            std::isfinite(frontierPruneScore(candidates[index]))) {
            ordered.push_back(index);
        }
    }
    const auto better = [&](size_t a, size_t b) {
        const float aScore = frontierPruneScore(candidates[a]);
        const float bScore = frontierPruneScore(candidates[b]);
        if (aScore != bScore)
            return aScore < bScore;
        if (candidates[a].depth != candidates[b].depth)
            return candidates[a].depth < candidates[b].depth;
        return candidates[a].originalIndex < candidates[b].originalIndex;
    };
    const auto heapCompare = [&](size_t a, size_t b) {
        return better(b, a);
    };
    std::make_heap(ordered.begin(), ordered.end(), heapCompare);
    while (selected.size() < keep && !ordered.empty()) {
        std::pop_heap(ordered.begin(), ordered.end(), heapCompare);
        const size_t index = ordered.back();
        ordered.pop_back();
        bool tooClose = false;
        if (distance > 0.0f) {
            for (const size_t existingIndex : selected) {
                const TraceVec delta =
                    candidates[index].point - candidates[existingIndex].point;
                if (delta.dot(delta) < distance2) {
                    tooClose = true;
                    break;
                }
            }
        }
        if (!tooClose)
            selected.push_back(index);
    }

    return selected;
}

[[nodiscard]] BeamState beamStateFromFrontierCandidate(
    const std::vector<BeamState>& parents,
    const std::vector<CandidateTask>& tasks,
    const std::vector<CandidateScore>& scores,
    size_t candidateIndex,
    const FrontierCandidate& candidate,
    const FiberTraceConfig& config)
{
    const CandidateTask& task = tasks[candidateIndex];
    const CandidateScore& score = scores[candidateIndex];
    const BeamState& parent = parents[task.beamIndex];
    BeamState out = parent;
    out.path = appendBeamPathPoint(parent.path, candidate.point);
    out.previousStepDirection = task.direction;
    out.currentSampleDirection = score.selectedCurrentDirection;
    out.historyDirection = updateHistoryDirection(
        parent.historyDirection,
        task.direction,
        parent.depth,
        config.cumulativeSmoothnessSteps);
    out.loss = candidate.loss;
    out.tracedLength = parent.tracedLength +
        traceLength(candidate.point - beamEndpoint(parent));
    out.depth = candidate.depth;
    out.reached = candidate.reached;
    out.reason = candidate.reached ? "target_plane" : std::string{};
    return out;
}

[[nodiscard]] std::vector<BeamState> pruneFrontierCandidates(
    const std::vector<FrontierCandidate>& candidates,
    const std::vector<BeamState>& parents,
    const std::vector<CandidateTask>& tasks,
    const std::vector<CandidateScore>& scores,
    int beamWidth,
    double pruneDistanceVoxels,
    const FiberTraceConfig& config,
    std::vector<size_t>* selectedIndices = nullptr)
{
    if (candidates.empty())
        return {};
    const std::vector<size_t> selected = selectFrontierCandidateIndices(
        candidates,
        beamWidth,
        pruneDistanceVoxels);
    if (selectedIndices != nullptr)
        *selectedIndices = selected;
    std::vector<BeamState> out;
    out.reserve(selected.size());
    for (const size_t index : selected) {
        out.push_back(beamStateFromFrontierCandidate(
            parents,
            tasks,
            scores,
            index,
            candidates[index],
            config));
    }
    return out;
}

[[nodiscard]] size_t exactLookaheadRequiredParentCount(
    const std::vector<BeamState>& parents,
    std::optional<float> resultThreshold,
    bool finalBeamSetComplete)
{
    if (!resultThreshold.has_value() || !finalBeamSetComplete)
        return parents.size();
    return static_cast<size_t>(std::count_if(
        parents.begin(), parents.end(), [&](const BeamState& parent) {
            return parent.loss <= *resultThreshold;
        }));
}

void recordExactLookaheadPotential(
    FiberTraceProfile* profile,
    const std::vector<BeamState>& parents,
    size_t childCandidateCount,
    size_t evaluatedParentCount,
    size_t evaluatedChildCandidateCount,
    std::optional<float> resultThreshold,
    bool finalBeamSetComplete)
{
    if (profile == nullptr || parents.empty())
        return;
    const size_t requiredParents = exactLookaheadRequiredParentCount(
        parents, resultThreshold, finalBeamSetComplete);
    const size_t candidatesPerParent = childCandidateCount / parents.size();
    ++profile->lookaheadFinalFrontiers;
    profile->lookaheadTotalParents += parents.size();
    profile->lookaheadRequiredParents += requiredParents;
    profile->lookaheadEvaluatedParents += evaluatedParentCount;
    profile->lookaheadTotalChildCandidates += childCandidateCount;
    profile->lookaheadRequiredChildCandidates +=
        requiredParents * candidatesPerParent;
    profile->lookaheadEvaluatedChildCandidates += evaluatedChildCandidateCount;
    profile->lookaheadParentCounts.push_back(parents.size());
    profile->lookaheadRequiredParentCounts.push_back(requiredParents);
}

struct LazyLookaheadEvaluation {
    size_t evaluatedParents = 0;
    size_t evaluatedChildCandidates = 0;
    size_t totalChildCandidates = 0;
};

template <typename LossAt>
[[nodiscard]] std::vector<size_t> orderedIndexPrefix(
    size_t count,
    size_t limit,
    LossAt&& lossAt)
{
    limit = std::min(limit, count);
    if (limit == 0)
        return {};
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), size_t{0});
    const auto better = [&](size_t a, size_t b) {
        const auto aLoss = lossAt(a);
        const auto bLoss = lossAt(b);
        if (aLoss != bLoss)
            return aLoss < bLoss;
        return a < b;
    };
    if (limit < count) {
        std::nth_element(
            order.begin(), order.begin() + static_cast<std::ptrdiff_t>(limit),
            order.end(), better);
        order.resize(limit);
    }
    std::sort(order.begin(), order.end(), better);
    return order;
}

[[nodiscard]] bool shouldRetryLookahead(
    bool lazyLookahead,
    size_t parentCap,
    size_t retryParentCap,
    bool segmentSuccess)
{
    return !segmentSuccess && lazyLookahead && parentCap > 0 &&
        retryParentCap > parentCap;
}

[[nodiscard]] std::optional<size_t> bestReachedFrontierCandidateIndex(
    const std::vector<FrontierCandidate>& candidates);

[[nodiscard]] LazyLookaheadEvaluation evaluateLazyFinalFrontier(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const std::vector<BeamState>& parents,
    const std::vector<ConeOffset>& offsets,
    float step,
    const TraceVec& target,
    const TraceVec& targetPlaneNormal,
    const FiberTraceConfig& config,
    int lookaheadDepth,
    CandidateScoringScratch& scratch,
    FiberTraceProfile* profile)
{
    constexpr size_t kInitialParentBatch = 256;
    constexpr size_t kAdditionalParentBatch = 64;
    const size_t candidatesPerParent = std::max<size_t>(1, offsets.size());
    const size_t totalCandidates = parents.size() * candidatesPerParent;
    const size_t parentLimit = config.lookaheadParentCap > 0
        ? std::min(config.lookaheadParentCap, parents.size())
        : parents.size();
    auto& fullTasks = scratch.lookaheadTasks;
    auto& fullScores = scratch.lookaheadScores;
    auto& frontier = scratch.frontierCandidates;
    const auto storageStart = TraceClock::now();
    fullTasks.clear();
    fullScores.clear();
    frontier.clear();
    const size_t initialCandidates =
        std::min(kInitialParentBatch, parentLimit) * candidatesPerParent;
    fullTasks.reserve(initialCandidates);
    fullScores.reserve(initialCandidates);
    frontier.reserve(initialCandidates);
    if (profile != nullptr)
        profile->lookaheadFrontierStorageSeconds += elapsedSeconds(storageStart);

    const auto parentOrderStart = TraceClock::now();
    const std::vector<size_t> parentOrder = orderedIndexPrefix(
        parents.size(), parentLimit, [&](size_t index) {
            return parents[index].loss;
        });
    if (profile != nullptr)
        profile->lookaheadParentOrderSeconds += elapsedSeconds(parentOrderStart);

    size_t orderBegin = 0;
    size_t orderEnd = std::min(kInitialParentBatch, parentLimit);
    while (orderBegin < orderEnd) {
        const auto taskBuildStart = TraceClock::now();
        buildCandidateTasksForOrderedParents(
            scratch.tasks,
            scratch.candidatePoints,
            scratch.lookaheadGlobalIndices,
            parents,
            offsets,
            step,
            parentOrder,
            orderBegin,
            orderEnd);
        if (profile != nullptr)
            profile->taskBuildSeconds += elapsedSeconds(taskBuildStart);

        const auto appendStart = TraceClock::now();
        const size_t compactBegin = fullTasks.size();
        frontier.resize(compactBegin + scratch.tasks.size());
        if (profile != nullptr)
            profile->lookaheadFrontierStorageSeconds += elapsedSeconds(appendStart);
        const FrontierScoreOutput frontierOutput{
            &frontier,
            &target,
            &targetPlaneNormal,
            &scratch.lookaheadGlobalIndices,
            compactBegin};
        const auto& batchScores = scoreCandidateTasks(
            predictions,
            normalSampler,
            parents,
            scratch.tasks,
            scratch.candidatePoints,
            config,
            lookaheadDepth,
            {},
            &frontierOutput,
            scratch,
            profile);
        const auto scoreStorageStart = TraceClock::now();
        fullTasks.insert(
            fullTasks.end(), scratch.tasks.begin(), scratch.tasks.end());
        fullScores.insert(
            fullScores.end(), batchScores.begin(), batchScores.end());
        if (profile != nullptr)
            profile->lookaheadFrontierStorageSeconds +=
                elapsedSeconds(scoreStorageStart);

        orderBegin = orderEnd;
        if (orderBegin >= parentLimit)
            break;

        const auto decisionStart = TraceClock::now();
        std::optional<float> threshold;
        if (const auto reached = bestReachedFrontierCandidateIndex(frontier)) {
            threshold = frontier[*reached].loss;
        } else {
            const auto selected = selectFrontierCandidateIndices(
                frontier,
                config.beamWidth,
                config.beamPruneDistanceVoxels);
            if (selected.size() >= static_cast<size_t>(config.beamWidth)) {
                threshold = 0.0f;
                for (const size_t index : selected) {
                    threshold = std::max(
                        *threshold, frontierPruneScore(frontier[index]));
                }
            }
        }
        const float nextLowerBound = parents[parentOrder[orderBegin]].loss;
        const bool exactResultEstablished =
            threshold.has_value() && nextLowerBound > *threshold;
        if (profile != nullptr)
            profile->lookaheadDecisionSeconds += elapsedSeconds(decisionStart);
        if (exactResultEstablished)
            break;
        orderEnd = std::min(
            orderBegin + kAdditionalParentBatch, parentLimit);
    }

    if (profile != nullptr) {
        profile->lookaheadFrontierAllocatedSlots += frontier.size();
        profile->lookaheadFrontierEvaluatedSlots +=
            orderBegin * candidatesPerParent;
    }
    return {
        orderBegin,
        orderBegin * candidatesPerParent,
        totalCandidates,
    };
}

[[nodiscard]] std::optional<size_t> bestReachedFrontierCandidateIndex(
    const std::vector<FrontierCandidate>& candidates)
{
    std::optional<size_t> best;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!candidate.valid || !candidate.reached)
            continue;
        if (!best.has_value() || candidate.loss < candidates[*best].loss ||
            (candidate.loss == candidates[*best].loss &&
             candidate.originalIndex < candidates[*best].originalIndex)) {
            best = index;
        }
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
    const TraceVec start = toTraceVec(request.startPoint);
    const TraceVec target = toTraceVec(request.targetPoint);
    const TraceVec targetPlaneNormal = traceNormalizedOr(
        toTraceVec(request.targetPlaneNormal),
        traceNormalizedOr(target - start, {1.0f, 0.0f, 0.0f}));
    const TraceVec referenceStartDirection = traceNormalizedOr(
        toTraceVec(request.initialDirection),
        traceNormalizedOr(target - start, {1.0f, 0.0f, 0.0f}));
    FiberTraceProfile* profile = request.config.profile;
    if (profile != nullptr) {
        ++profile->oneWayCalls;
        profile->localityCurrentDepth1Dependencies.clear();
        profile->localityPreviousStepDependencies.clear();
    }
    const auto startSampleStart = TraceClock::now();
    const ScoredDirection startPrediction =
        bestAlignedPrediction(predictions, toVec3d(start), referenceStartDirection, false);
    if (profile != nullptr)
        profile->startSampleSeconds += elapsedSeconds(startSampleStart);
    if (!startPrediction.valid) {
        throw std::invalid_argument(
            "fiber trace start point has no valid prediction direction");
    }
    const TraceVec startDirection = startPrediction.direction;

    const float distance = request.budgetSpanVoxels > 0.0
        ? static_cast<float>(request.budgetSpanVoxels)
        : traceLength(target - start);
    const float step = std::max(1.0e-3f, static_cast<float>(request.config.stepVoxels));
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
    int stepIndex = 0;
    while (stepIndex < maxSteps) {
        std::vector<BeamState> expanded = beams;
        int advanced = 0;
        bool prunedFinalFrontier = false;
        for (; advanced < lookaheadSteps && stepIndex + advanced < maxSteps; ++advanced) {
            if (profile != nullptr)
                ++profile->generations;
            const bool finalLookaheadGeneration =
                advanced + 1 >= lookaheadSteps ||
                stepIndex + advanced + 1 >= maxSteps;
            const bool lazyFinalGeneration =
                finalLookaheadGeneration && advanced > 0 &&
                request.config.lazyLookahead;
            const std::vector<CandidateTask>* tasksPtr = nullptr;
            const std::vector<TraceVec>* candidatePointsPtr = nullptr;
            const std::vector<CandidateScore>* scoresPtr = nullptr;
            size_t evaluatedParents = expanded.size();
            size_t evaluatedChildCandidates = 0;
            size_t totalChildCandidates = 0;
            if (lazyFinalGeneration) {
                const LazyLookaheadEvaluation evaluation =
                    evaluateLazyFinalFrontier(
                        predictions,
                        normalSampler,
                        expanded,
                        coneOffsets,
                        step,
                        target,
                        targetPlaneNormal,
                        request.config,
                        advanced + 1,
                        scoringScratch,
                        profile);
                tasksPtr = &scoringScratch.lookaheadTasks;
                scoresPtr = &scoringScratch.lookaheadScores;
                evaluatedParents = evaluation.evaluatedParents;
                evaluatedChildCandidates = evaluation.evaluatedChildCandidates;
                totalChildCandidates = evaluation.totalChildCandidates;
            } else {
                const auto taskBuildStart = TraceClock::now();
                buildCandidateTasks(
                    scoringScratch.tasks,
                    scoringScratch.candidatePoints,
                    expanded,
                    coneOffsets,
                    step);
                if (profile != nullptr)
                    profile->taskBuildSeconds += elapsedSeconds(taskBuildStart);
                tasksPtr = &scoringScratch.tasks;
                candidatePointsPtr = &scoringScratch.candidatePoints;
                FrontierScoreOutput frontierOutput;
                const FrontierScoreOutput* frontierOutputPtr = nullptr;
                if (finalLookaheadGeneration) {
                    auto& frontier = scoringScratch.frontierCandidates;
                    frontier.clear();
                    frontier.resize(scoringScratch.tasks.size());
                    frontierOutput = {
                        &frontier,
                        &target,
                        &targetPlaneNormal,
                        nullptr,
                        0};
                    frontierOutputPtr = &frontierOutput;
                }
                scoresPtr = &scoreCandidateTasks(
                    predictions,
                    normalSampler,
                    expanded,
                    scoringScratch.tasks,
                    scoringScratch.candidatePoints,
                    request.config,
                    advanced + 1,
                    {},
                    frontierOutputPtr,
                    scoringScratch,
                    profile);
                evaluatedChildCandidates = scoringScratch.tasks.size();
                totalChildCandidates = scoringScratch.tasks.size();
            }
            const auto& tasks = *tasksPtr;
            const auto& scores = *scoresPtr;

            if (finalLookaheadGeneration) {
                auto& frontier = scoringScratch.frontierCandidates;
                const auto bestReachedIndex =
                    bestReachedFrontierCandidateIndex(frontier);
                if (bestReachedIndex.has_value()) {
                    if (advanced > 0) {
                        recordExactLookaheadPotential(
                            profile,
                            expanded,
                            totalChildCandidates,
                            evaluatedParents,
                            evaluatedChildCandidates,
                            frontier[*bestReachedIndex].loss,
                            true);
                    }
                    const BeamState bestReached = beamStateFromFrontierCandidate(
                        expanded,
                        tasks,
                        scores,
                        *bestReachedIndex,
                        frontier[*bestReachedIndex],
                        request.config);
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
                std::vector<size_t> selectedIndices;
                beams = pruneFrontierCandidates(
                    frontier,
                    expanded,
                    tasks,
                    scores,
                    request.config.beamWidth,
                    request.config.beamPruneDistanceVoxels,
                    request.config,
                    advanced > 0 ? &selectedIndices : nullptr);
                if (advanced > 0) {
                    const bool complete = selectedIndices.size() >=
                        static_cast<size_t>(request.config.beamWidth);
                    std::optional<float> threshold;
                    if (complete) {
                        threshold = 0.0f;
                        for (const size_t index : selectedIndices) {
                            threshold = std::max(
                                *threshold,
                                frontierPruneScore(frontier[index]));
                        }
                    }
                    recordExactLookaheadPotential(
                        profile,
                        expanded,
                        totalChildCandidates,
                        evaluatedParents,
                        evaluatedChildCandidates,
                        threshold,
                        complete);
                }
                if (profile != nullptr)
                    profile->pruneSeconds += elapsedSeconds(pruneStart);
                if (beams.empty()) {
                    reason = "no_valid_candidates";
                    expanded.clear();
                    break;
                }
                prunedFinalFrontier = true;
                ++advanced;
                break;
            }

            const auto frontierStart = TraceClock::now();
            const auto& candidatePoints = *candidatePointsPtr;
            std::vector<BeamState> nextFrontier;
            nextFrontier.reserve(tasks.size());
            for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
                const auto& task = tasks[taskIndex];
                const CandidateScore& candidateScore = scores[taskIndex];
                if (!candidateScore.valid || !std::isfinite(candidateScore.loss))
                    continue;
                const BeamState& beam = expanded[task.beamIndex];
                const TraceVec currentPoint = beamEndpoint(beam);
                const auto crossing = interpolatePlaneCrossing(
                    currentPoint,
                    candidatePoints[taskIndex],
                    target,
                    targetPlaneNormal);
                const TraceVec nextPoint = crossing.value_or(
                    candidatePoints[taskIndex]);
                BeamState next = beam;
                next.path = appendBeamPathPoint(beam.path, nextPoint);
                next.tracedLength += traceLength(nextPoint - currentPoint);
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
            const float signedDistance = std::abs(
                tracePointToPlaneSigned(beamEndpoint(beams.front()), target, targetPlaneNormal));
            FiberTraceProgress event;
            event.phase = phase;
            event.step = stepIndex;
            event.maxSteps = maxSteps;
            event.targetPlaneProgress =
                distance > kTraceEpsilon
                    ? 1.0 - std::min(1.0f, signedDistance / distance)
                    : 1.0;
            event.reason = beams.front().reached ? beams.front().reason : reason;
            progress(event);
        }
    }

    if (beams.empty()) {
        return {{toVec3d(start)}, false, reason, 0};
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

CandidateScoreDebug debugCandidateLossFromCorners(
    const vc::lasagna::LasagnaCornerBatch& corners,
    size_t optionCount,
    size_t pointIndex,
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& currentSampleDirection,
    const cv::Vec3d& historyDirection,
    const cv::Vec3d& candidateDirection,
    const FiberTraceConfig& config)
{
    BeamState beam;
    beam.previousStepDirection = toTraceVec(previousStepDirection);
    beam.currentSampleDirection = toTraceVec(currentSampleDirection);
    beam.historyDirection = toTraceVec(historyDirection);
    const CandidateScore score = candidateLossFromCorners(
        corners,
        optionCount,
        pointIndex,
        beam,
        toTraceVec(candidateDirection),
        config);
    return {
        score.loss,
        toVec3d(score.selectedCurrentDirection),
        score.selectedPresence,
        score.valid};
}

size_t debugExactLookaheadRequiredParentCount(
    const std::vector<double>& parentLowerBounds,
    std::optional<double> resultThreshold,
    bool finalBeamSetComplete)
{
    std::vector<BeamState> parents(parentLowerBounds.size());
    for (size_t index = 0; index < parentLowerBounds.size(); ++index)
        parents[index].loss = static_cast<float>(parentLowerBounds[index]);
    const std::optional<float> floatThreshold = resultThreshold.has_value()
        ? std::optional<float>{static_cast<float>(*resultThreshold)}
        : std::nullopt;
    return exactLookaheadRequiredParentCount(
        parents, floatThreshold, finalBeamSetComplete);
}

std::vector<size_t> debugOrderedIndexPrefix(
    const std::vector<double>& losses,
    size_t limit)
{
    return orderedIndexPrefix(losses.size(), limit, [&](size_t index) {
        return losses[index];
    });
}

bool debugShouldRetryLookahead(
    bool lazyLookahead,
    size_t parentCap,
    size_t retryParentCap,
    bool segmentSuccess)
{
    return shouldRetryLookahead(
        lazyLookahead,
        parentCap,
        retryParentCap,
        segmentSuccess);
}

} // namespace testing
#endif

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
        std::unique_ptr<vc::lasagna::LasagnaChannelCornerSampler> presenceCorners;
        std::unique_ptr<vc::lasagna::LasagnaChannelCornerSampler> nxCorners;
        std::unique_ptr<vc::lasagna::LasagnaChannelCornerSampler> nyCorners;
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

        cornerBudget_ = std::make_shared<vc::render::DecodedChunkCacheBudget>(
            maxCachedBytes);
        options_.reserve(prefixes.size());
        for (const auto& prefix : prefixes) {
            const auto channels = predictionChannelNames(prefix);
            Option option;
            option.name = prefix.empty() ? std::string("option_000") : prefix;
            option.presence = vc::lasagna::bindLasagnaChannel(manifest, channels[0]);
            option.nx = vc::lasagna::bindLasagnaChannel(manifest, channels[1]);
            option.ny = vc::lasagna::bindLasagnaChannel(manifest, channels[2]);
            const bool cornerCompatible =
                !option.presence.hasChannelDimension &&
                !option.nx.hasChannelDimension &&
                !option.ny.hasChannelDimension &&
                vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.nx) &&
                vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.ny);
            if (cornerCompatible) {
                option.presenceCorners =
                    std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(
                        option.presence, maxCachedBytes, cornerBudget_);
                option.nxCorners =
                    std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(
                        option.nx, maxCachedBytes, cornerBudget_);
                option.nyCorners =
                    std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(
                        option.ny, maxCachedBytes, cornerBudget_);
            } else {
                cornerSamplingAvailable_ = false;
            }
            options_.push_back(std::move(option));
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

    void materializeGroupedPredictionCorners(
        const std::vector<std::vector<vc::lasagna::LasagnaCornerSample>>& corners,
        size_t firstVolume,
        const std::vector<cv::Vec3f>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples,
        bool retainStorage = false) const
    {
        const size_t optionCount = options_.size();
        if (firstVolume + optionCount * 3 > corners.size()) {
            throw std::invalid_argument(
                "prediction corner batch is missing channel volumes");
        }
        if (retainStorage) {
            if (samples.size() < referenceDirections.size())
                samples.resize(referenceDirections.size());
        } else {
            samples.resize(referenceDirections.size());
        }
        const int workers = std::clamp(
            parallelThreads, 1,
            static_cast<int>(std::max<size_t>(1, referenceDirections.size())));
        auto materialize = [&](size_t pointIndex) {
            auto& out = samples[pointIndex];
            out.options.clear();
            out.options.reserve(optionCount);
            const cv::Vec3f reference = referenceDirections[pointIndex];
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const size_t volume = firstVolume + optionIndex * 3;
                const auto& presence = corners[volume][pointIndex];
                const auto& nx = corners[volume + 1][pointIndex];
                const auto& ny = corners[volume + 2][pointIndex];
                if (!presence.valid || !nx.valid || !ny.valid) {
                    out.options.push_back({});
                    continue;
                }
                cv::Vec3f direction =
                    vc::lasagna::interpolateLasagnaCompactAxisCorners(
                        nx, ny, reference);
                if (direction.dot(reference) < 0.0f)
                    direction *= -1.0f;
                const float rawPresence =
                    vc::lasagna::interpolateLasagnaCorners(presence);
                out.options.push_back({
                    direction,
                    std::clamp(rawPresence / 255.0f, 0.0f, 1.0f),
                    direction.dot(direction) > 1.0e-12f,
                });
            }
        };
#ifdef _OPENMP
        if (workers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(referenceDirections.size());
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                materialize(static_cast<size_t>(rawIndex));
            return;
        }
#endif
        for (size_t pointIndex = 0; pointIndex < referenceDirections.size(); ++pointIndex)
            materialize(pointIndex);
    }

    void materializeGroupedPredictionCorners(
        const vc::lasagna::LasagnaCornerBatch& corners,
        size_t firstVolume,
        const std::vector<cv::Vec3f>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples,
        bool retainStorage = false,
        size_t normalFirstVolume = std::numeric_limits<size_t>::max(),
        std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample>* normals =
            nullptr) const
    {
        const size_t optionCount = options_.size();
        if (firstVolume + optionCount * 3 > corners.values.size() ||
            corners.fractionsXYZ.size() != referenceDirections.size() ||
            corners.valid.size() != referenceDirections.size()) {
            throw std::invalid_argument(
                "prediction corner batch has inconsistent channel volumes");
        }
        if (normals != nullptr && normalFirstVolume + 3 > corners.values.size())
            throw std::invalid_argument("normal corner batch is missing channel volumes");
        if (retainStorage) {
            if (samples.size() < referenceDirections.size())
                samples.resize(referenceDirections.size());
        } else {
            samples.resize(referenceDirections.size());
        }
        if (normals != nullptr)
            normals->resize(referenceDirections.size());
        const int workers = std::clamp(
            parallelThreads, 1,
            static_cast<int>(std::max<size_t>(1, referenceDirections.size())));
        auto materialize = [&](size_t pointIndex) {
            decodePredictionAndNormalCornerPoint(
                corners,
                firstVolume,
                optionCount,
                pointIndex,
                referenceDirections[pointIndex],
                samples[pointIndex],
                normalFirstVolume,
                normals != nullptr ? &(*normals)[pointIndex] : nullptr);
        };
#ifdef _OPENMP
        if (workers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(referenceDirections.size());
            #pragma omp parallel for schedule(dynamic, 64) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                materialize(static_cast<size_t>(rawIndex));
            return;
        }
#endif
        for (size_t pointIndex = 0; pointIndex < referenceDirections.size(); ++pointIndex)
            materialize(pointIndex);
    }

    [[nodiscard]] bool groupedCornerSamplersWithNormals(
        const vc::lasagna::LasagnaNormalSampler& normalSampler,
        std::vector<const vc::lasagna::LasagnaChannelCornerSampler*>& samplers) const
    {
        if (!cornerSamplingAvailable_)
            return false;
        const auto normalSamplers = normalSampler.groupedCornerSamplers();
        if (std::any_of(
                normalSamplers.begin(), normalSamplers.end(),
                [](const auto* sampler) { return sampler == nullptr; })) {
            return false;
        }
        samplers.clear();
        samplers.reserve(options_.size() * 3 + normalSamplers.size());
        for (const auto& option : options_) {
            samplers.push_back(option.presenceCorners.get());
            samplers.push_back(option.nxCorners.get());
            samplers.push_back(option.nyCorners.get());
        }
        samplers.insert(samplers.end(), normalSamplers.begin(), normalSamplers.end());
        return true;
    }

    static void recordCornerReport(
        const vc::lasagna::NormalPrefetchReport& cornerReport,
        double cornerSeconds,
        int lookaheadDepth,
        FiberTraceProfile* profile)
    {
        if (profile != nullptr) {
            profile->predictionCornerSeconds += cornerSeconds;
            profile->predictionCornerPrepareSeconds +=
                cornerReport.cornerPrepareSeconds;
            profile->predictionCornerLayoutSeconds +=
                cornerReport.cornerLayoutSeconds;
            profile->predictionCornerPinSeconds += cornerReport.cornerPinSeconds;
            profile->predictionCornerGatherSeconds += cornerReport.cornerGatherSeconds;
            profile->predictionCornerLayoutChunkRuns +=
                cornerReport.cornerLayoutChunkRuns;
            profile->predictionCornerBoundaryPoints +=
                cornerReport.cornerBoundaryPoints;
            profile->predictionCornerDependencies += cornerReport.cornerDependencies;
            profile->cornerPointCount += cornerReport.cornerPointCount;
            profile->cornerUniqueVoxelCubes += cornerReport.cornerUniqueVoxelCubes;
            profile->cornerWorkerTasks += cornerReport.cornerWorkerTasks;
            profile->cornerMaxCandidatesPerCube = std::max(
                profile->cornerMaxCandidatesPerCube,
                cornerReport.cornerMaxCandidatesPerCube);
            for (size_t index = 0;
                 index < profile->cornerCubeOccupancyHistogram.size();
                 ++index) {
                profile->cornerCubeOccupancyHistogram[index] +=
                    cornerReport.cornerCubeOccupancyHistogram[index];
            }
            const auto overlap = [](const std::vector<uint64_t>& a,
                                    const std::vector<uint64_t>& b) {
                size_t aIndex = 0;
                size_t bIndex = 0;
                uint64_t shared = 0;
                while (aIndex < a.size() && bIndex < b.size()) {
                    if (a[aIndex] < b[bIndex]) {
                        ++aIndex;
                    } else if (b[bIndex] < a[aIndex]) {
                        ++bIndex;
                    } else {
                        ++shared;
                        ++aIndex;
                        ++bIndex;
                    }
                }
                return std::pair<uint64_t, uint64_t>{
                    shared,
                    static_cast<uint64_t>(a.size() + b.size()) - shared};
            };
            if (lookaheadDepth <= 1) {
                if (!profile->localityPreviousStepDependencies.empty()) {
                    const auto [shared, combined] = overlap(
                        profile->localityPreviousStepDependencies,
                        cornerReport.cornerDependencyIds);
                    profile->stepDependencyShared += shared;
                    profile->stepDependencyUnion += combined;
                }
                profile->localityCurrentDepth1Dependencies =
                    cornerReport.cornerDependencyIds;
            } else if (!profile->localityCurrentDepth1Dependencies.empty()) {
                const auto [shared, combined] = overlap(
                    profile->localityCurrentDepth1Dependencies,
                    cornerReport.cornerDependencyIds);
                profile->depthDependencyShared += shared;
                profile->depthDependencyUnion += combined;
            }
            profile->localityPreviousStepDependencies =
                cornerReport.cornerDependencyIds;
        }
    }

    [[nodiscard]] bool sampleCornerBatchWithNormals(
        const vc::lasagna::LasagnaNormalSampler& normalSampler,
        const std::vector<cv::Vec3f>& volumePoints,
        int parallelThreads,
        vc::lasagna::LasagnaCornerBatch* cornerScratch,
        FiberTraceProfile* profile) const
    {
        if (cornerScratch == nullptr)
            return false;
        std::vector<const vc::lasagna::LasagnaChannelCornerSampler*> samplers;
        if (!groupedCornerSamplersWithNormals(normalSampler, samplers))
            return false;
        const auto cornerStart = TraceClock::now();
        vc::lasagna::NormalPrefetchReport cornerReport;
        try {
            cornerReport = vc::lasagna::sampleLasagnaChannelCornerBatch(
                samplers, volumePoints, *cornerScratch, parallelThreads);
        } catch (const std::invalid_argument&) {
            return false;
        }
        recordCornerReport(
            cornerReport, elapsedSeconds(cornerStart), 0, profile);
        return true;
    }

    [[nodiscard]] bool visitCornerBatchWithNormals(
        const vc::lasagna::LasagnaNormalSampler& normalSampler,
        const std::vector<cv::Vec3f>& volumePoints,
        int parallelThreads,
        void* visitorContext,
        vc::lasagna::LasagnaCornerPointVisitor visitor,
        int lookaheadDepth,
        FiberTraceProfile* profile) const
    {
        std::vector<const vc::lasagna::LasagnaChannelCornerSampler*> samplers;
        if (!groupedCornerSamplersWithNormals(normalSampler, samplers))
            return false;
        const auto cornerStart = TraceClock::now();
        vc::lasagna::NormalPrefetchReport cornerReport;
        try {
            cornerReport = vc::lasagna::visitLasagnaChannelCorners(
                samplers,
                volumePoints,
                visitorContext,
                visitor,
                parallelThreads,
                profile != nullptr);
        } catch (const std::invalid_argument&) {
            return false;
        }
        recordCornerReport(
            cornerReport, elapsedSeconds(cornerStart), lookaheadDepth, profile);
        return true;
    }

    void sampleBatch(
        const std::vector<cv::Vec3f>& volumePoints,
        const std::vector<cv::Vec3f>& referenceDirections,
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

        if (!cornerSamplingAvailable_) {
            std::vector<cv::Vec3d> doublePoints;
            std::vector<cv::Vec3d> doubleDirections;
            doublePoints.reserve(volumePoints.size());
            doubleDirections.reserve(referenceDirections.size());
            for (const auto& point : volumePoints)
                doublePoints.emplace_back(point[0], point[1], point[2]);
            for (const auto& direction : referenceDirections)
                doubleDirections.emplace_back(direction[0], direction[1], direction[2]);
            sampleBatch(
                doublePoints,
                doubleDirections,
                parallelThreads,
                samples,
                profile);
            return;
        }

        const auto batchStart = TraceClock::now();
        const size_t optionCount = options_.size();
        const int workers =
            std::clamp(parallelThreads, 1, static_cast<int>(volumePoints.size()));
        std::vector<const vc::lasagna::LasagnaChannelCornerSampler*> samplers;
        samplers.reserve(optionCount * 3);
        for (const auto& option : options_) {
            samplers.push_back(option.presenceCorners.get());
            samplers.push_back(option.nxCorners.get());
            samplers.push_back(option.nyCorners.get());
        }
        std::vector<std::vector<vc::lasagna::LasagnaCornerSample>> corners;
        (void)vc::lasagna::sampleLasagnaChannelCornerBatch(
            samplers, volumePoints, corners, workers);
        materializeGroupedPredictionCorners(
            corners, 0, referenceDirections, workers, samples);
        if (profile != nullptr)
            profile->predictionMaterializeSeconds += elapsedSeconds(batchStart);
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

        if (cornerSamplingAvailable_) {
        std::vector<cv::Vec3f> floatPoints;
        std::vector<cv::Vec3f> floatReferences;
        floatPoints.reserve(volumePoints.size());
        floatReferences.reserve(referenceDirections.size());
        for (const auto& point : volumePoints) {
            floatPoints.emplace_back(
                static_cast<float>(point[0]),
                static_cast<float>(point[1]),
                static_cast<float>(point[2]));
        }
        for (const auto& direction : referenceDirections) {
            floatReferences.emplace_back(
                static_cast<float>(direction[0]),
                static_cast<float>(direction[1]),
                static_cast<float>(direction[2]));
        }
        sampleBatch(
            floatPoints,
            floatReferences,
            parallelThreads,
            samples,
            profile);
        return;
        }

        const size_t optionCount = options_.size();
        const int workers =
            std::clamp(parallelThreads, 1, static_cast<int>(volumePoints.size()));

        {
            const auto batchStart = TraceClock::now();
            std::vector<cv::Vec3f> floatPoints;
            floatPoints.reserve(volumePoints.size());
            for (const auto& point : volumePoints) {
                floatPoints.push_back({
                    static_cast<float>(point[0]),
                    static_cast<float>(point[1]),
                    static_cast<float>(point[2])});
            }
            struct OptionCorners {
                std::vector<vc::lasagna::LasagnaCornerSample> presence;
                std::vector<vc::lasagna::LasagnaCornerSample> nx;
                std::vector<vc::lasagna::LasagnaCornerSample> ny;
            };
            std::vector<OptionCorners> corners(optionCount);
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const auto& option = options_[optionIndex];
                std::vector<std::vector<vc::lasagna::LasagnaCornerSample>> grouped;
                (void)vc::lasagna::sampleLasagnaChannelCornerBatch(
                    {option.presenceCorners.get(),
                     option.nxCorners.get(),
                     option.nyCorners.get()},
                    floatPoints,
                    grouped,
                    workers);
                corners[optionIndex].presence = std::move(grouped[0]);
                corners[optionIndex].nx = std::move(grouped[1]);
                corners[optionIndex].ny = std::move(grouped[2]);
            }

            auto materializeCornerSample = [&](size_t pointIndex) {
                auto& out = samples[pointIndex];
                out.options.clear();
                out.options.reserve(optionCount);
                const cv::Vec3f reference{
                    static_cast<float>(referenceDirections[pointIndex][0]),
                    static_cast<float>(referenceDirections[pointIndex][1]),
                    static_cast<float>(referenceDirections[pointIndex][2])};
                for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                    const auto& optionCorners = corners[optionIndex];
                    const auto& presence = optionCorners.presence[pointIndex];
                    const auto& nx = optionCorners.nx[pointIndex];
                    const auto& ny = optionCorners.ny[pointIndex];
                    if (!presence.valid || !nx.valid || !ny.valid) {
                        out.options.push_back({});
                        continue;
                    }
                    cv::Vec3f direction =
                        vc::lasagna::interpolateLasagnaCompactAxisCorners(
                            nx, ny, reference);
                    if (direction.dot(reference) < 0.0f)
                        direction *= -1.0f;
                    const float rawPresence =
                        vc::lasagna::interpolateLasagnaCorners(presence);
                    out.options.push_back({
                        {direction[0], direction[1], direction[2]},
                        static_cast<double>(std::clamp(rawPresence / 255.0f, 0.0f, 1.0f)),
                        direction.dot(direction) > 1.0e-12f,
                    });
                }
            };

#ifdef _OPENMP
            if (workers > 1) {
                const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
                #pragma omp parallel for schedule(static) num_threads(workers)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                    materializeCornerSample(static_cast<size_t>(rawIndex));
            } else
#endif
            {
                for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex)
                    materializeCornerSample(pointIndex);
            }
            if (profile != nullptr)
                profile->predictionMaterializeSeconds += elapsedSeconds(batchStart);
            return;
        }

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
        for (size_t optionIndex = 0; optionIndex < options_.size(); ++optionIndex) {
            const auto& option = options_[optionIndex];
            auto presenceRequest = vc::lasagna::prepareLasagnaCubeRequest(
                option.presence, volumePoint);
            vc::lasagna::LasagnaCubeRequest nxRequest;
            vc::lasagna::LasagnaCubeRequest nyRequest;
            if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                nxRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                    presenceRequest, option.nx);
                nyRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(
                    presenceRequest, option.ny);
            } else {
                nxRequest = vc::lasagna::prepareLasagnaCubeRequest(
                    option.nx, volumePoint);
                nyRequest = vc::lasagna::prepareLasagnaCubeRequest(
                    option.ny, volumePoint);
            }
            vc::lasagna::LasagnaLocalChunkResolver presenceResolver(
                option.presence, *cache_);
            vc::lasagna::LasagnaLocalChunkResolver nxResolver(option.nx, *cache_);
            vc::lasagna::LasagnaLocalChunkResolver nyResolver(option.ny, *cache_);
            presenceResolver.resolve(presenceRequest);
            nxResolver.resolve(nxRequest);
            nyResolver.resolve(nyRequest);
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
    std::shared_ptr<vc::render::DecodedChunkCacheBudget> cornerBudget_;
    bool cornerSamplingAvailable_ = true;
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

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3f>& volumePoints,
    const std::vector<cv::Vec3f>& referenceDirections,
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

bool FiberPredictionField::sampleCornerBatchWithNormals(
    const vc::lasagna::LasagnaNormalSampler& normalSampler,
    const std::vector<cv::Vec3f>& volumePoints,
    int parallelThreads,
    vc::lasagna::LasagnaCornerBatch* cornerScratch,
    FiberTraceProfile* profile) const
{
    return impl_->sampleCornerBatchWithNormals(
        normalSampler,
        volumePoints,
        parallelThreads,
        cornerScratch,
        profile);
}

bool FiberPredictionField::visitCornerBatchWithNormals(
    const vc::lasagna::LasagnaNormalSampler& normalSampler,
    const std::vector<cv::Vec3f>& volumePoints,
    int parallelThreads,
    void* visitorContext,
    vc::lasagna::LasagnaCornerPointVisitor visitor,
    int lookaheadDepth,
    FiberTraceProfile* profile) const
{
    return impl_->visitCornerBatchWithNormals(
        normalSampler,
        volumePoints,
        parallelThreads,
        visitorContext,
        visitor,
        lookaheadDepth,
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

        const auto setSegmentOutcome = [&](FiberTraceOneWayResult trace) {
            segment.trace = std::move(trace);
            if (!segment.trace.points.empty()) {
                segment.inPlaneErrorVoxels = endpointPlaneError(
                    segment.trace.points.back(), target, targetPlaneNormal);
            } else {
                segment.inPlaneErrorVoxels =
                    std::numeric_limits<double>::infinity();
            }
            segment.success = segment.trace.reachedTargetPlane &&
                segment.inPlaneErrorVoxels <= request.errorThresholdVoxels;
        };
        setSegmentOutcome(std::move(segment.trace));
        if (shouldRetryLookahead(
                request.config.lazyLookahead,
                request.config.lookaheadParentCap,
                request.config.lookaheadRetryParentCap,
                segment.success)) {
            ++result.lookaheadRetryCount;
            FiberTraceOneWayRequest retry = oneWay;
            retry.config.lookaheadParentCap =
                request.config.lookaheadRetryParentCap;
            FiberTraceOneWayResult retryTrace = traceOneWayCore(
                predictions,
                retry,
                normalSampler,
                segmentProgress,
                "fiber_retry");
            double retryError = std::numeric_limits<double>::infinity();
            if (!retryTrace.points.empty()) {
                retryError = endpointPlaneError(
                    retryTrace.points.back(), target, targetPlaneNormal);
            }
            const bool retrySuccess = retryTrace.reachedTargetPlane &&
                retryError <= request.errorThresholdVoxels;
            if (retrySuccess) {
                ++result.lookaheadRetryRecoveredCount;
                setSegmentOutcome(std::move(retryTrace));
            }
        }
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
