#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/render/DecodedChunkCacheBudget.hpp"
#include <utils/zarr.hpp>

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

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kTraceEpsilon = 1.0e-6f;
constexpr float kTracePi = 3.14159265358979323846f;
constexpr size_t kMaxTargetPlanes = 3;

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

[[nodiscard]] TraceVec traceNormalizedOr(const TraceVec& v, const TraceVec& fallback)
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

[[nodiscard]] float traceAngleBetweenUnit(const TraceVec& a, const TraceVec& b)
{
    return std::acos(traceClampSignedUnit(a.dot(b)));
}

[[nodiscard]] TraceVec traceAlignTo(const TraceVec& direction, const TraceVec& reference)
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
    return {static_cast<float>(value[0]), static_cast<float>(value[1]), static_cast<float>(value[2])};
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
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::vector<std::string> fiberPredictionPrefixes(const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    std::vector<std::string> prefixes;
    if (manifest.groupForChannel("presence") != nullptr && manifest.groupForChannel("nx") != nullptr && manifest.groupForChannel("ny") != nullptr) {
        prefixes.push_back({});
    }
    for (const auto& group : manifest.groups) {
        for (const auto& channel : group.channels) {
            constexpr std::string_view suffix = "_presence";
            if (!endsWith(channel, suffix))
                continue;
            const std::string prefix = channel.substr(0, channel.size() - suffix.size());
            if (manifest.groupForChannel(prefix + "_nx") != nullptr && manifest.groupForChannel(prefix + "_ny") != nullptr) {
                prefixes.push_back(prefix);
            }
        }
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    return prefixes;
}

[[nodiscard]] std::array<std::string, 3> predictionChannelNames(const std::string& prefix)
{
    if (prefix.empty())
        return {"presence", "nx", "ny"};
    return {prefix + "_presence", prefix + "_nx", prefix + "_ny"};
}

[[nodiscard]] const vc::lasagna::LasagnaChannelGroup& predictionChannelGroup(const vc::lasagna::LasagnaDatasetManifest& manifest, const std::string& channel)
{
    const auto* group = manifest.groupForChannel(channel);
    if (group == nullptr) {
        throw std::runtime_error("fiber inference dataset is missing required channel '" + channel + "'");
    }
    return *group;
}

[[nodiscard]] double predictionChannelEffectiveScale(const vc::lasagna::LasagnaDatasetManifest& manifest, const std::string& channel)
{
    const auto& group = predictionChannelGroup(manifest, channel);
    const double scale = manifest.sourceToBase * static_cast<double>(group.scaleFactor());
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::runtime_error("fiber inference channel '" + channel + "' has a non-positive or non-finite effective scale");
    }
    return scale;
}

[[nodiscard]] bool nearlySameScale(double a, double b)
{
    const double tolerance = 1.0e-9 * std::max({1.0, std::abs(a), std::abs(b)});
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

[[nodiscard]] std::array<TraceVec, 2> traceOrthonormalBasis(const TraceVec& direction)
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

[[nodiscard]] std::vector<ConeOffset> angleStepConeOffsets(double maxAngleDegrees, double angleStepDegrees)
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
            offsets.push_back({static_cast<float>(uDeg), static_cast<float>(vDeg), static_cast<float>(radius2), order++});
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

[[nodiscard]] std::vector<ConeOffset> legacyGridConeOffsets(double maxAngleDegrees, int gridSize)
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
        const double b = gridSize == 1 ? 0.0 : -1.0 + 2.0 * static_cast<double>(y) / static_cast<double>(gridSize - 1);
        for (int x = 0; x < gridSize; ++x) {
            const double a = gridSize == 1 ? 0.0 : -1.0 + 2.0 * static_cast<double>(x) / static_cast<double>(gridSize - 1);
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
                static_cast<float>(tangentScale * diskX),
                static_cast<float>(tangentScale * diskY),
                static_cast<float>(radius2),
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

[[nodiscard]] std::vector<cv::Vec3d> candidateDirections(const cv::Vec3d& reference, const std::vector<ConeOffset>& offsets)
{
    const cv::Vec3d forward = normalizedOr(reference, {1.0, 0.0, 0.0});
    const auto basis = orthonormalBasis(forward);
    std::vector<cv::Vec3d> out;
    out.reserve(offsets.size());
    for (const auto& offset : offsets)
        out.push_back(normalizedOr(forward + basis[0] * offset.u + basis[1] * offset.v, forward));
    if (out.empty())
        out.push_back(forward);
    return out;
}

[[nodiscard]] std::vector<cv::Vec3d> candidateDirections(const cv::Vec3d& reference, const FiberTraceConfig& config)
{
    const auto offsets = config.coneAngleStepDegrees > 0.0 ? angleStepConeOffsets(config.coneAngleDegrees, config.coneAngleStepDegrees)
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

[[nodiscard]] double pointToPlaneSigned(const cv::Vec3d& point, const cv::Vec3d& planePoint, const cv::Vec3d& planeNormal)
{
    return (point - planePoint).dot(planeNormal);
}

[[nodiscard]] bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

[[nodiscard]] bool pointsExactlyEqual(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

[[nodiscard]] size_t exactLineIndexForControlPoint(const std::vector<cv::Vec3d>& line, const cv::Vec3d& control, size_t controlIndex, const std::filesystem::path& path)
{
    for (size_t index = 0; index < line.size(); ++index) {
        if (pointsExactlyEqual(line[index], control)) {
            return index;
        }
    }
    throw std::runtime_error(
        "fiber JSON control point " + std::to_string(controlIndex) +
        " is not an exact line point; refusing to guess line arc position: " + path.string());
}

[[nodiscard]] std::vector<cv::Vec3d> scaledPoints(const std::vector<cv::Vec3d>& points, double divisor)
{
    std::vector<cv::Vec3d> out;
    out.reserve(points.size());
    for (const auto& point : points)
        out.push_back(point / divisor);
    return out;
}

[[nodiscard]] double referenceLengthMeters(double referenceLengthWorkingVoxels, double workingToBaseScale, double voxelSizeUm)
{
    if (!(referenceLengthWorkingVoxels > 0.0) || !(workingToBaseScale > 0.0) || !(voxelSizeUm > 0.0)) {
        return 0.0;
    }
    return referenceLengthWorkingVoxels * workingToBaseScale * voxelSizeUm * 1.0e-6;
}

[[nodiscard]] cv::Vec3d terminalTraceDirection(const std::vector<cv::Vec3d>& points, const cv::Vec3d& fallback)
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

[[nodiscard]] ScoredDirection bestAlignedPrediction(const FiberPredictionSample& sample, const TraceVec& referenceDirection, bool weightByPresence)
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

[[nodiscard]] ScoredDirection bestAlignedPrediction(const FiberPredictionSource& predictions, const cv::Vec3d& point, const TraceVec& referenceDirection, bool weightByPresence)
{
    return bestAlignedPrediction(predictions.sample(point, toVec3d(referenceDirection)), referenceDirection, weightByPresence);
}

[[nodiscard]] bool normalAwareSmoothnessEnabled(const FiberTraceConfig& config)
{
    return config.smoothnessNormalWeight > 0.0 || config.smoothnessTangentWeight > 0.0 || config.cumulativeSmoothnessTangentWeight > 0.0;
}

void requireNormalSamplerForNormalAwareSmoothness(const FiberTraceConfig& config, const vc::lasagna::NormalSampler* normalSampler)
{
    if (normalSampler != nullptr || !normalAwareSmoothnessEnabled(config))
        return;
    throw std::invalid_argument("Lasagna normal sampler is required for tangent/normal fiber trace smoothness");
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
    requireFinite(config.cumulativeSmoothnessTangentWeight, "cumulative smoothness tangent weight");
    requireFinite(config.maxStepFactor, "max step factor");
    requireFinite(config.meetingAcceptMaxErrorRatio, "meeting accept maximum error ratio");
    requireFinite(config.endpointAcceptThresholdBaseVoxels, "endpoint accept threshold in base voxels");
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
    if (config.smoothnessWeight < 0.0 || config.smoothnessNormalWeight < 0.0 || config.smoothnessTangentWeight < 0.0 ||
        config.cumulativeSmoothnessTangentWeight < 0.0) {
        throw std::invalid_argument("smoothness weights must be non-negative");
    }
    if (config.smoothnessFreeAngleDegrees < 0.0)
        throw std::invalid_argument("smoothness free angle must be non-negative");
    if (config.cumulativeSmoothnessSteps < 1)
        throw std::invalid_argument("cumulative smoothness steps must be at least 1");
    if (config.maxStepFactor < 0.0)
        throw std::invalid_argument("max step factor must be non-negative");
    if (config.meetingAcceptMaxErrorRatio < 0.0 || config.meetingAcceptMaxErrorRatio > 1.0) {
        throw std::invalid_argument("meeting accept maximum error ratio must be in [0, 1]");
    }
    if (config.endpointAcceptThresholdBaseVoxels < 0.0)
        throw std::invalid_argument("endpoint accept threshold in base voxels must be non-negative");
    if (!(config.traceToBaseScale > 0.0))
        throw std::invalid_argument("trace-to-base scale must be positive");
    if (config.baseVoxelSizeUm.has_value() && !(*config.baseVoxelSizeUm > 0.0))
        throw std::invalid_argument("base voxel size must be positive when provided");
}

[[nodiscard]] float traceExcessAngleSquared(float angle, float freeAngle)
{
    const float excess = std::max(0.0f, angle - freeAngle);
    return excess * excess;
}

[[nodiscard]] float smoothnessLoss(
    const TraceVec& previousStepDirection, const TraceVec& candidateStepDirection, const TraceVec& normal, bool normalValid, const FiberTraceConfig& config)
{
    return fiberLocalSmoothnessCost(
               previousStepDirection,
               candidateStepDirection,
               normal,
               normalValid,
               FiberLocalSmoothnessConfig{
                   static_cast<float>(config.smoothnessWeight),
                   static_cast<float>(config.smoothnessNormalWeight),
                   static_cast<float>(config.smoothnessTangentWeight),
                   static_cast<float>(config.smoothnessFreeAngleDegrees) * kTracePi / 180.0f})
        .total();
}

[[nodiscard]] float cumulativeTangentSmoothnessLoss(
    const TraceVec& historyDirection, const TraceVec& candidateStepDirection, const TraceVec& normal, bool normalValid, const FiberTraceConfig& config)
{
    const float weight = static_cast<float>(config.cumulativeSmoothnessTangentWeight);
    if (!(weight > 0.0f))
        return 0.0f;
    const TraceVec& history = historyDirection;
    const TraceVec& cand = candidateStepDirection;
    const TraceVec& n = normal;
    constexpr float kTraceEpsilon2 = kTraceEpsilon * kTraceEpsilon;
    if (!normalValid || history.dot(history) <= kTraceEpsilon2 || cand.dot(cand) <= kTraceEpsilon2 || n.dot(n) <= kTraceEpsilon2) {
        return 0.0f;
    }
    const float historyN = traceClampSignedUnit(history.dot(n));
    const float candN = traceClampSignedUnit(cand.dot(n));
    const TraceVec historyT = traceNormalizedOrZero(history - n * historyN);
    const TraceVec candT = traceNormalizedOrZero(cand - n * candN);
    if (historyT.dot(historyT) <= kTraceEpsilon2 || candT.dot(candT) <= kTraceEpsilon2) {
        return 0.0f;
    }
    const float freeAngle = static_cast<float>(config.smoothnessFreeAngleDegrees) * kTracePi / 180.0f;
    return weight * traceExcessAngleSquared(traceAngleBetweenUnit(historyT, candT), freeAngle);
}

[[nodiscard]] TraceVec updateHistoryDirection(const TraceVec& historyDirection, const TraceVec& chosenDirection, int depth, int cumulativeSmoothnessSteps)
{
    const TraceVec chosen = traceNormalizedOrZero(chosenDirection);
    if (depth <= 0 || cumulativeSmoothnessSteps <= 1)
        return chosen;
    const float count = static_cast<float>(std::clamp(depth, 1, cumulativeSmoothnessSteps - 1));
    return traceNormalizedOr(chosen + traceNormalizedOrZero(historyDirection) * count, chosen);
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
    uint8_t crossedTargetPlaneMask = 0;
    std::array<TraceVec, kMaxTargetPlanes> targetPlaneCrossings{};
    std::string reason;
};

[[nodiscard]] std::shared_ptr<const BeamState::PathNode> appendBeamPathPoint(std::shared_ptr<const BeamState::PathNode> previous, const TraceVec& point)
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
    uint8_t crossedTargetPlaneMask = 0;
    std::array<TraceVec, kMaxTargetPlanes> targetPlaneCrossings{};
};

struct TraceTargetPlaneSet {
    std::array<TraceVec, kMaxTargetPlanes> points{};
    std::array<TraceVec, kMaxTargetPlanes> normals{};
    std::array<std::string, kMaxTargetPlanes> names{};
    size_t count = 0;
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
    const TraceTargetPlaneSet& targetPlanes,
    std::optional<float> acceptThresholdVoxels,
    size_t originalIndex);

struct FrontierScoreOutput {
    std::vector<FrontierCandidate>* candidates = nullptr;
    const TraceTargetPlaneSet* targetPlanes = nullptr;
    std::optional<float> acceptThresholdVoxels;
    const std::vector<size_t>* originalIndices = nullptr;
    size_t offset = 0;
};

void storeScoredFrontierCandidate(
    const FrontierScoreOutput* output, const std::vector<BeamState>& beams, const CandidateTask& task, const TraceVec& candidatePoint, const CandidateScore& score, size_t taskIndex)
{
    if (output == nullptr || !score.valid || !std::isfinite(score.loss))
        return;
    const size_t originalIndex = output->originalIndices != nullptr ? (*output->originalIndices)[taskIndex] : taskIndex;
    (*output->candidates)[output->offset + taskIndex] =
        makeFrontierCandidate(beams[task.beamIndex], candidatePoint, score, *output->targetPlanes, output->acceptThresholdVoxels, originalIndex);
}

void fillCandidateTasksForBeam(
    CandidateTask* out, TraceVec* outPoints, const std::vector<BeamState>& beams, size_t beamIndex, const std::vector<ConeOffset>& offsets, float step)
{
    const BeamState& beam = beams[beamIndex];
    const TraceVec currentPoint = beamEndpoint(beam);
    const TraceVec forward = traceNormalizedOr(beam.currentSampleDirection, {1.0f, 0.0f, 0.0f});
    const auto basis = traceOrthonormalBasis(forward);
    if (offsets.empty()) {
        out[0] = {static_cast<uint32_t>(beamIndex), forward};
        outPoints[0] = currentPoint + forward * step;
        return;
    }
    for (size_t offsetIndex = 0; offsetIndex < offsets.size(); ++offsetIndex) {
        const auto& offset = offsets[offsetIndex];
        const TraceVec direction = traceNormalizedOr(forward + basis[0] * offset.u + basis[1] * offset.v, forward);
        out[offsetIndex] = {static_cast<uint32_t>(beamIndex), direction};
        outPoints[offsetIndex] = currentPoint + direction * step;
    }
}

void buildCandidateTasks(
    std::vector<CandidateTask>& tasks, std::vector<TraceVec>& candidatePoints, const std::vector<BeamState>& beams, const std::vector<ConeOffset>& offsets, float step)
{
    if (beams.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::overflow_error("fiber trace has too many beam states");
    const size_t candidatesPerBeam = std::max<size_t>(1, offsets.size());
    tasks.resize(beams.size() * candidatesPerBeam);
    candidatePoints.resize(tasks.size());
    for (size_t beamIndex = 0; beamIndex < beams.size(); ++beamIndex) {
        fillCandidateTasksForBeam(tasks.data() + beamIndex * candidatesPerBeam, candidatePoints.data() + beamIndex * candidatesPerBeam, beams, beamIndex, offsets, step);
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
        fillCandidateTasksForBeam(tasks.data() + firstLocalTask, candidatePoints.data() + firstLocalTask, beams, parentIndex, offsets, step);
        for (size_t offsetIndex = 0; offsetIndex < candidatesPerBeam; ++offsetIndex) {
            globalIndices[firstLocalTask + offsetIndex] = parentIndex * candidatesPerBeam + offsetIndex;
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
    const auto selectedCurrent = bestAlignedPrediction(candidateSample, candidateDirection, true);
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
        const TraceVec candidateSampleDirection = traceAlignTo(option.direction, candidateDirection);
        const float presence = traceClamp01(option.presence);

        const TraceVec& prevStep = beam.previousStepDirection;
        const TraceVec& currentSample = beam.currentSampleDirection;
        const TraceVec& currentStep = candidateDirection;

        const float loss = fiberLocalAlignmentLoss(presence, prevStep, currentStep, currentSample, candidateSampleDirection) +
                           smoothnessLoss(prevStep, currentStep, smoothNormal, smoothNormalValid, config) +
                           cumulativeTangentSmoothnessLoss(beam.historyDirection, currentStep, smoothNormal, smoothNormalValid, config);
        if (loss < bestLoss)
            bestLoss = loss;
    }
    return {bestLoss, selectedCurrent.direction, selectedCurrent.presence, std::isfinite(bestLoss)};
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
    const auto weights = vc::lasagna::lasagnaCornerWeights(corners.fractionsXYZ[pointIndex]);
    for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
        const size_t volume = firstVolume + optionIndex * 3;
        if (!valid) {
            sample.options.push_back({});
            continue;
        }
        TraceVec direction =
            vc::lasagna::interpolateLasagnaCompactAxisCorners(corners.values[volume + 1][pointIndex], corners.values[volume + 2][pointIndex], weights, referenceDirection);
        if (direction.dot(referenceDirection) < 0.0f)
            direction *= -1.0f;
        const float rawPresence = vc::lasagna::interpolateLasagnaCorners(corners.values[volume][pointIndex], weights);
        sample.options.push_back({direction, std::clamp(rawPresence / 255.0f, 0.0f, 1.0f), direction.dot(direction) > 1.0e-12f});
    }
    if (normal == nullptr)
        return;
    if (!valid || !(vc::lasagna::interpolateLasagnaCorners(corners.values[normalFirstVolume][pointIndex], weights) > 0.0f)) {
        *normal = {};
        return;
    }
    normal->normal =
        vc::lasagna::interpolateLasagnaCompactAxisCorners(corners.values[normalFirstVolume + 1][pointIndex], corners.values[normalFirstVolume + 2][pointIndex], weights);
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
    if (vc::lasagna::interpolateLasagnaCorners(volumeCorners[normalFirstVolume], weights) > 0.0f) {
        smoothNormal =
            vc::lasagna::interpolateLasagnaCompactAxisCorners(volumeCorners[normalFirstVolume + 1], volumeCorners[normalFirstVolume + 2], weights);
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
        TraceVec direction =
            vc::lasagna::interpolateLasagnaCompactAxisCorners(volumeCorners[volume + 1], volumeCorners[volume + 2], weights, candidateDirection);
        if (!(direction.dot(direction) > 1.0e-12f))
            continue;

        const TraceVec alignedDirection = direction;
        const float presence = traceClamp01(vc::lasagna::interpolateLasagnaCorners(volumeCorners[volume], weights) / 255.0f);
        const float alignmentScore = traceClamp01(alignedDirection.dot(reference)) * presence;
        if (alignmentScore > bestAlignmentScore) {
            bestAlignmentScore = alignmentScore;
            selectedCurrent = {alignedDirection, presence, true};
        }

        const float loss = fiberLocalAlignmentLoss(presence, prevStep, currentStep, currentSample, alignedDirection) +
                           smoothnessLoss(prevStep, currentStep, smoothNormal, smoothNormalValid, config) +
                           cumulativeTangentSmoothnessLoss(beam.historyDirection, currentStep, smoothNormal, smoothNormalValid, config);
        if (loss < bestLoss)
            bestLoss = loss;
    }
    return {bestLoss, selectedCurrent.direction, selectedCurrent.presence, std::isfinite(bestLoss)};
}

[[nodiscard]] CandidateScore candidateLossFromCorners(
    const vc::lasagna::LasagnaCornerBatch& corners, size_t optionCount, size_t pointIndex, const BeamState& beam, const TraceVec& candidateDirection, const FiberTraceConfig& config)
{
    std::vector<std::array<uint8_t, 8>> pointCorners(corners.values.size());
    for (size_t volumeIndex = 0; volumeIndex < corners.values.size(); ++volumeIndex)
        pointCorners[volumeIndex] = corners.values[volumeIndex][pointIndex];
    return candidateLossFromCornerValues(pointCorners, corners.fractionsXYZ[pointIndex], corners.valid[pointIndex] != 0, optionCount, beam, candidateDirection, config);
}

[[nodiscard]] CandidateScore candidateLoss(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const BeamState& beam,
    const TraceVec& candidateDirection,
    const TraceVec& candidatePoint,
    const FiberTraceConfig& config)
{
    const auto candidateSample = predictions.sample(toVec3d(candidatePoint), toVec3d(candidateDirection));
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
    return candidateLossFromSample(candidateSample, beam, candidateDirection, config, traceNormal);
}

[[nodiscard]] int traceWorkerCount(const FiberPredictionSource& predictions, const vc::lasagna::NormalSampler* normalSampler, const FiberTraceConfig& config, size_t taskCount)
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
    if (normalSampler == nullptr || candidatePoints.empty() || !normalAwareSmoothnessEnabled(config)) {
        return;
    }
    vc::lasagna::NormalBatchReport report;
    if (const auto* lasagnaSampler = dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(normalSampler)) {
        report = lasagnaSampler->sampleNormalBatch(candidatePoints, parallelThreads, out);
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
            out[index] = {toTraceVec(fallbackSamples[index].normal), fallbackSamples[index].valid};
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

    const int workers = traceWorkerCount(predictions, normalSampler, config, tasks.size());
    if (workers <= 1) {
        const auto scoreStart = TraceClock::now();
        for (size_t index = 0; index < tasks.size(); ++index) {
            const auto& task = tasks[index];
            scores[index] = candidateLoss(predictions, normalSampler, beams[task.beamIndex], task.direction, candidatePoints[index], config);
            storeScoredFrontierCandidate(frontierOutput, beams, task, candidatePoints[index], scores[index], index);
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
    const auto* lasagnaSampler = dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(normalSampler);
    struct CornerScoreContext {
        const std::vector<BeamState>* beams;
        const std::vector<CandidateTask>* tasks;
        const std::vector<TraceVec>* candidatePoints;
        const FiberTraceConfig* config;
        std::vector<CandidateScore>* scores;
        const FrontierScoreOutput* frontierOutput;
        size_t optionCount;
    } cornerContext{&beams, &tasks, &candidatePoints, &config, &scores, frontierOutput, field ? field->optionCount() : 0};
    const auto scoreCornerPoint =
        +[](void* rawContext, size_t pointIndex, const cv::Vec3f& fractionXYZ, bool valid, std::span<const std::array<uint8_t, 8>> volumeCorners) {
            const auto& context = *static_cast<CornerScoreContext*>(rawContext);
            const CandidateTask& task = (*context.tasks)[pointIndex];
            if (valid && volumeCorners.size() != context.optionCount * 3 + 3) {
                throw std::runtime_error("fiber prediction corner visitor returned inconsistent dimensions");
            }
            CandidateScore& score = (*context.scores)[pointIndex];
            score = candidateLossFromCornerValues(
                volumeCorners, fractionXYZ, valid, context.optionCount, (*context.beams)[task.beamIndex], task.direction, *context.config);
            storeScoredFrontierCandidate(context.frontierOutput, *context.beams, task, (*context.candidatePoints)[pointIndex], score, pointIndex);
        };
    const bool fusedCornerScoring =
        field != nullptr && lasagnaSampler != nullptr && normals.empty() && normalAwareSmoothnessEnabled(config) &&
        field->visitCornerBatchWithNormals(*lasagnaSampler, candidatePoints, workers, &cornerContext, scoreCornerPoint, lookaheadDepth, profile);
    if (fusedCornerScoring) {
        if (profile != nullptr)
            profile->predictionBatchSeconds += elapsedSeconds(predictionStart);
        return scores;
    }

    referenceDirections.reserve(tasks.size());
    for (const auto& task : tasks)
        referenceDirections.push_back(task.direction);
    if (field != nullptr) {
        field->sampleBatch(candidatePoints, referenceDirections, workers, predictionSamples, profile);
    } else {
        std::vector<cv::Vec3d> doublePoints;
        std::vector<cv::Vec3d> doubleDirections;
        doublePoints.reserve(candidatePoints.size());
        doubleDirections.reserve(referenceDirections.size());
        for (const auto& point : candidatePoints)
            doublePoints.push_back(toVec3d(point));
        for (const auto& direction : referenceDirections)
            doubleDirections.push_back(toVec3d(direction));
        predictions.sampleBatch(doublePoints, doubleDirections, workers, predictionSamples);
    }
    if (profile != nullptr)
        profile->predictionBatchSeconds += elapsedSeconds(predictionStart);
    if (predictionSamples.size() < tasks.size()) {
        throw std::runtime_error("fiber prediction batch returned the wrong number of samples");
    }
    const auto normalStart = TraceClock::now();
    sampleCandidateNormals(
        normalSampler, candidatePoints, config, workers, scratch.normalSamplesWithDerivative, scratch.fallbackNormalSamples, scratch.floatNormalSamples, profile);
    if (profile != nullptr)
        profile->normalBatchSeconds += elapsedSeconds(normalStart);
    const auto& normalsForScoring = normals.empty() ? scratch.floatNormalSamples : normals;

    const auto scoreStart = TraceClock::now();
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    auto scoreOne = [&](size_t index) {
        const auto& task = tasks[index];
        const vc::lasagna::LasagnaNormalSampler::FloatNormalSample* normal = index < normalsForScoring.size() ? &normalsForScoring[index] : nullptr;
        scores[index] = candidateLossFromSample(predictionSamples[index], beams[task.beamIndex], task.direction, config, normal);
        storeScoredFrontierCandidate(frontierOutput, beams, task, candidatePoints[index], scores[index], index);
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

[[nodiscard]] float tracePointToPlaneSigned(const TraceVec& point, const TraceVec& planePoint, const TraceVec& planeNormal)
{
    return (point - planePoint).dot(planeNormal);
}

[[nodiscard]] std::optional<TraceVec> interpolatePlaneCrossing(const TraceVec& start, const TraceVec& end, const TraceVec& planePoint, const TraceVec& planeNormal)
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

[[nodiscard]] TraceTargetPlaneSet makeTraceTargetPlaneSet(const std::vector<FiberTraceTargetPlane>& planes, bool allowEmpty = false)
{
    if (planes.empty() && !allowEmpty)
        throw std::invalid_argument("fiber trace requires at least one target plane");
    if (planes.size() > kMaxTargetPlanes) {
        throw std::invalid_argument("fiber trace supports at most three target-local planes");
    }

    TraceTargetPlaneSet out;
    out.count = planes.size();
    std::set<std::string> names;
    for (size_t index = 0; index < planes.size(); ++index) {
        const auto& plane = planes[index];
        if (plane.name.empty())
            throw std::invalid_argument("fiber target plane name must not be empty");
        if (!names.insert(plane.name).second) {
            throw std::invalid_argument("fiber target plane names must be unique: " + plane.name);
        }
        const TraceVec point = toTraceVec(plane.point);
        const TraceVec normal = toTraceVec(plane.normal);
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
            throw std::invalid_argument("fiber target plane point must be finite");
        }
        if (!(traceLength(normal) > kTraceEpsilon) || !std::isfinite(traceLength(normal))) {
            throw std::invalid_argument("fiber target plane normal must be finite and non-degenerate");
        }
        out.points[index] = point;
        out.normals[index] = traceNormalizedOrZero(normal);
        out.names[index] = plane.name;
    }
    return out;
}

[[nodiscard]] float targetPlaneInPlaneError(const TraceVec& point, const TraceVec& planePoint, const TraceVec& planeNormal)
{
    const TraceVec delta = point - planePoint;
    return traceLength(delta - planeNormal * delta.dot(planeNormal));
}

void updateTargetPlaneCrossings(
    const TraceVec& start, const TraceVec& end, const TraceTargetPlaneSet& planes, uint8_t& crossedMask, std::array<TraceVec, kMaxTargetPlanes>& crossings)
{
    for (size_t index = 0; index < planes.count; ++index) {
        const auto crossing = interpolatePlaneCrossing(start, end, planes.points[index], planes.normals[index]);
        if (!crossing.has_value())
            continue;
        const uint8_t bit = static_cast<uint8_t>(1U << index);
        const bool alreadyCrossed = (crossedMask & bit) != 0;
        if (alreadyCrossed) {
            const float oldError = targetPlaneInPlaneError(crossings[index], planes.points[index], planes.normals[index]);
            const float newError = targetPlaneInPlaneError(*crossing, planes.points[index], planes.normals[index]);
            if (!(newError < oldError))
                continue;
        }
        crossedMask = static_cast<uint8_t>(crossedMask | bit);
        crossings[index] = *crossing;
    }
}

[[nodiscard]] uint8_t allTargetPlaneMask(const TraceTargetPlaneSet& planes)
{
    return static_cast<uint8_t>((1U << planes.count) - 1U);
}

[[nodiscard]] std::optional<size_t> selectedTargetPlaneIndex(const TraceTargetPlaneSet& planes, uint8_t crossedMask, const std::array<TraceVec, kMaxTargetPlanes>& crossings)
{
    std::optional<size_t> selected;
    float bestError = std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < planes.count; ++index) {
        if ((crossedMask & static_cast<uint8_t>(1U << index)) == 0)
            continue;
        const float error = targetPlaneInPlaneError(crossings[index], planes.points[index], planes.normals[index]);
        if (error < bestError) {
            bestError = error;
            selected = index;
        }
    }
    return selected;
}

[[nodiscard]] bool targetPlanesReached(
    const TraceTargetPlaneSet& planes, uint8_t crossedMask, const std::array<TraceVec, kMaxTargetPlanes>& crossings, std::optional<float> acceptThresholdVoxels)
{
    if ((crossedMask & allTargetPlaneMask(planes)) != allTargetPlaneMask(planes))
        return false;
    const auto selected = selectedTargetPlaneIndex(planes, crossedMask, crossings);
    if (!selected.has_value())
        return false;
    return !acceptThresholdVoxels.has_value() ||
           targetPlaneInPlaneError(crossings[*selected], planes.points[*selected], planes.normals[*selected]) <= *acceptThresholdVoxels;
}

[[nodiscard]] FrontierCandidate makeFrontierCandidate(
    const BeamState& beam,
    const TraceVec& candidatePoint,
    const CandidateScore& candidateScore,
    const TraceTargetPlaneSet& targetPlanes,
    std::optional<float> acceptThresholdVoxels,
    size_t originalIndex)
{
    const TraceVec currentPoint = beamEndpoint(beam);
    FrontierCandidate out;
    out.point = candidatePoint;
    out.loss = beam.loss + candidateScore.loss;
    out.depth = beam.depth + 1;
    out.originalIndex = originalIndex;
    out.valid = true;
    out.crossedTargetPlaneMask = beam.crossedTargetPlaneMask;
    out.targetPlaneCrossings = beam.targetPlaneCrossings;
    updateTargetPlaneCrossings(currentPoint, candidatePoint, targetPlanes, out.crossedTargetPlaneMask, out.targetPlaneCrossings);
    out.reached = targetPlanesReached(targetPlanes, out.crossedTargetPlaneMask, out.targetPlaneCrossings, acceptThresholdVoxels);
    return out;
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

[[nodiscard]] std::vector<BeamState> pruneBeamStates(std::vector<BeamState> states, int beamWidth, double pruneDistanceVoxels)
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
            if (!best.has_value() || score < bestScore || (score == bestScore && states[index].depth < states[*best].depth)) {
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

[[nodiscard]] std::vector<size_t> selectFrontierCandidateIndices(const std::vector<FrontierCandidate>& candidates, int beamWidth, double pruneDistanceVoxels)
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
        if (candidates[index].valid && std::isfinite(frontierPruneScore(candidates[index]))) {
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
    const auto heapCompare = [&](size_t a, size_t b) { return better(b, a); };
    std::make_heap(ordered.begin(), ordered.end(), heapCompare);
    while (selected.size() < keep && !ordered.empty()) {
        std::pop_heap(ordered.begin(), ordered.end(), heapCompare);
        const size_t index = ordered.back();
        ordered.pop_back();
        bool tooClose = false;
        if (distance > 0.0f) {
            for (const size_t existingIndex : selected) {
                const TraceVec delta = candidates[index].point - candidates[existingIndex].point;
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
    out.historyDirection = updateHistoryDirection(parent.historyDirection, task.direction, parent.depth, config.cumulativeSmoothnessSteps);
    out.loss = candidate.loss;
    out.tracedLength = parent.tracedLength + traceLength(candidate.point - beamEndpoint(parent));
    out.depth = candidate.depth;
    out.reached = candidate.reached;
    out.crossedTargetPlaneMask = candidate.crossedTargetPlaneMask;
    out.targetPlaneCrossings = candidate.targetPlaneCrossings;
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
    const std::vector<size_t> selected = selectFrontierCandidateIndices(candidates, beamWidth, pruneDistanceVoxels);
    if (selectedIndices != nullptr)
        *selectedIndices = selected;
    std::vector<BeamState> out;
    out.reserve(selected.size());
    for (const size_t index : selected) {
        out.push_back(beamStateFromFrontierCandidate(parents, tasks, scores, index, candidates[index], config));
    }
    return out;
}

[[nodiscard]] size_t exactLookaheadRequiredParentCount(const std::vector<BeamState>& parents, std::optional<float> resultThreshold, bool finalBeamSetComplete)
{
    if (!resultThreshold.has_value() || !finalBeamSetComplete)
        return parents.size();
    return static_cast<size_t>(
        std::count_if(parents.begin(), parents.end(), [&](const BeamState& parent) { return parent.loss <= *resultThreshold; }));
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
    const size_t requiredParents = exactLookaheadRequiredParentCount(parents, resultThreshold, finalBeamSetComplete);
    const size_t candidatesPerParent = childCandidateCount / parents.size();
    ++profile->lookaheadFinalFrontiers;
    profile->lookaheadTotalParents += parents.size();
    profile->lookaheadRequiredParents += requiredParents;
    profile->lookaheadEvaluatedParents += evaluatedParentCount;
    profile->lookaheadTotalChildCandidates += childCandidateCount;
    profile->lookaheadRequiredChildCandidates += requiredParents * candidatesPerParent;
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
[[nodiscard]] std::vector<size_t> orderedIndexPrefix(size_t count, size_t limit, LossAt&& lossAt)
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
        std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(limit), order.end(), better);
        order.resize(limit);
    }
    std::sort(order.begin(), order.end(), better);
    return order;
}

[[nodiscard]] bool shouldRetryLookahead(bool lazyLookahead, size_t parentCap, size_t retryParentCap, bool segmentSuccess)
{
    return !segmentSuccess && lazyLookahead && parentCap > 0 && retryParentCap > parentCap;
}

[[nodiscard]] std::optional<size_t> bestReachedFrontierCandidateIndex(const std::vector<FrontierCandidate>& candidates);

[[nodiscard]] LazyLookaheadEvaluation evaluateLazyFinalFrontier(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const std::vector<BeamState>& parents,
    const std::vector<ConeOffset>& offsets,
    float step,
    const TraceTargetPlaneSet& targetPlanes,
    std::optional<float> acceptThresholdVoxels,
    const FiberTraceConfig& config,
    int lookaheadDepth,
    CandidateScoringScratch& scratch,
    FiberTraceProfile* profile)
{
    constexpr size_t kInitialParentBatch = 256;
    constexpr size_t kAdditionalParentBatch = 64;
    const size_t candidatesPerParent = std::max<size_t>(1, offsets.size());
    const size_t totalCandidates = parents.size() * candidatesPerParent;
    const size_t parentLimit = config.lookaheadParentCap > 0 ? std::min(config.lookaheadParentCap, parents.size()) : parents.size();
    auto& fullTasks = scratch.lookaheadTasks;
    auto& fullScores = scratch.lookaheadScores;
    auto& frontier = scratch.frontierCandidates;
    const auto storageStart = TraceClock::now();
    fullTasks.clear();
    fullScores.clear();
    frontier.clear();
    const size_t initialCandidates = std::min(kInitialParentBatch, parentLimit) * candidatesPerParent;
    fullTasks.reserve(initialCandidates);
    fullScores.reserve(initialCandidates);
    frontier.reserve(initialCandidates);
    if (profile != nullptr)
        profile->lookaheadFrontierStorageSeconds += elapsedSeconds(storageStart);

    const auto parentOrderStart = TraceClock::now();
    const std::vector<size_t> parentOrder = orderedIndexPrefix(parents.size(), parentLimit, [&](size_t index) { return parents[index].loss; });
    if (profile != nullptr)
        profile->lookaheadParentOrderSeconds += elapsedSeconds(parentOrderStart);

    size_t orderBegin = 0;
    size_t orderEnd = std::min(kInitialParentBatch, parentLimit);
    while (orderBegin < orderEnd) {
        const auto taskBuildStart = TraceClock::now();
        buildCandidateTasksForOrderedParents(scratch.tasks, scratch.candidatePoints, scratch.lookaheadGlobalIndices, parents, offsets, step, parentOrder, orderBegin, orderEnd);
        if (profile != nullptr)
            profile->taskBuildSeconds += elapsedSeconds(taskBuildStart);

        const auto appendStart = TraceClock::now();
        const size_t compactBegin = fullTasks.size();
        frontier.resize(compactBegin + scratch.tasks.size());
        if (profile != nullptr)
            profile->lookaheadFrontierStorageSeconds += elapsedSeconds(appendStart);
        const FrontierScoreOutput frontierOutput{&frontier, &targetPlanes, acceptThresholdVoxels, &scratch.lookaheadGlobalIndices, compactBegin};
        const auto& batchScores =
            scoreCandidateTasks(predictions, normalSampler, parents, scratch.tasks, scratch.candidatePoints, config, lookaheadDepth, {}, &frontierOutput, scratch, profile);
        const auto scoreStorageStart = TraceClock::now();
        fullTasks.insert(fullTasks.end(), scratch.tasks.begin(), scratch.tasks.end());
        fullScores.insert(fullScores.end(), batchScores.begin(), batchScores.end());
        if (profile != nullptr)
            profile->lookaheadFrontierStorageSeconds += elapsedSeconds(scoreStorageStart);

        orderBegin = orderEnd;
        if (orderBegin >= parentLimit)
            break;

        const auto decisionStart = TraceClock::now();
        std::optional<float> threshold;
        if (const auto reached = bestReachedFrontierCandidateIndex(frontier)) {
            threshold = frontier[*reached].loss;
        } else {
            const auto selected = selectFrontierCandidateIndices(frontier, config.beamWidth, config.beamPruneDistanceVoxels);
            if (selected.size() >= static_cast<size_t>(config.beamWidth)) {
                threshold = 0.0f;
                for (const size_t index : selected) {
                    threshold = std::max(*threshold, frontierPruneScore(frontier[index]));
                }
            }
        }
        const float nextLowerBound = parents[parentOrder[orderBegin]].loss;
        const bool exactResultEstablished = threshold.has_value() && nextLowerBound > *threshold;
        if (profile != nullptr)
            profile->lookaheadDecisionSeconds += elapsedSeconds(decisionStart);
        if (exactResultEstablished)
            break;
        orderEnd = std::min(orderBegin + kAdditionalParentBatch, parentLimit);
    }

    if (profile != nullptr) {
        profile->lookaheadFrontierAllocatedSlots += frontier.size();
        profile->lookaheadFrontierEvaluatedSlots += orderBegin * candidatesPerParent;
    }
    return {
        orderBegin,
        orderBegin * candidatesPerParent,
        totalCandidates,
    };
}

[[nodiscard]] std::optional<size_t> bestReachedFrontierCandidateIndex(const std::vector<FrontierCandidate>& candidates)
{
    std::optional<size_t> best;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!candidate.valid || !candidate.reached)
            continue;
        if (!best.has_value() || candidate.loss < candidates[*best].loss ||
            (candidate.loss == candidates[*best].loss && candidate.originalIndex < candidates[*best].originalIndex)) {
            best = index;
        }
    }
    return best;
}

[[nodiscard]] std::optional<size_t> bestReachedStateIndexPythonParity(const std::vector<BeamState>& states)
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

[[nodiscard]] std::string targetPlaneFailureReason(std::string baseReason, const TraceTargetPlaneSet& planes, const BeamState& state, std::optional<float> acceptThresholdVoxels)
{
    const uint8_t required = allTargetPlaneMask(planes);
    if ((state.crossedTargetPlaneMask & required) == required) {
        if (acceptThresholdVoxels.has_value())
            return "target_plane_error_threshold";
        return baseReason;
    }
    baseReason += ":missing_target_planes=";
    bool first = true;
    for (size_t index = 0; index < planes.count; ++index) {
        if ((state.crossedTargetPlaneMask & static_cast<uint8_t>(1U << index)) != 0)
            continue;
        if (!first)
            baseReason += ',';
        baseReason += planes.names[index];
        first = false;
    }
    return baseReason;
}

[[nodiscard]] FiberTraceOneWayResult oneWayResultFromState(
    const BeamState& state,
    const TraceTargetPlaneSet& planes,
    bool reached,
    std::string reason,
    std::optional<float> acceptThresholdVoxels,
    bool snapTraceToSelectedCrossing,
    std::optional<double> traceLengthLimitVoxels)
{
    FiberTraceOneWayResult result;
    result.points = beamPathPoints(state);
    result.reachedTargetPlane = reached && !traceLengthLimitVoxels.has_value();
    result.reachedTraceLength = reached && traceLengthLimitVoxels.has_value();
    result.reason = reached ? std::move(reason) : targetPlaneFailureReason(std::move(reason), planes, state, acceptThresholdVoxels);
    result.steps = static_cast<int>(beamPointCount(state) > 0 ? beamPointCount(state) - 1 : 0);
    for (size_t index = 0; index < planes.count; ++index) {
        if ((state.crossedTargetPlaneMask & static_cast<uint8_t>(1U << index)) == 0)
            continue;
        result.targetPlaneCrossings.push_back({
            planes.names[index],
            toVec3d(state.targetPlaneCrossings[index]),
            targetPlaneInPlaneError(state.targetPlaneCrossings[index], planes.points[index], planes.normals[index]),
        });
    }
    const auto selected = selectedTargetPlaneIndex(planes, state.crossedTargetPlaneMask, state.targetPlaneCrossings);
    if (selected.has_value()) {
        result.selectedTargetPlaneName = planes.names[*selected];
        result.selectedTargetPlaneCrossing = toVec3d(state.targetPlaneCrossings[*selected]);
        result.selectedTargetPlaneErrorVoxels =
            targetPlaneInPlaneError(state.targetPlaneCrossings[*selected], planes.points[*selected], planes.normals[*selected]);
        if (snapTraceToSelectedCrossing && !result.points.empty())
            result.points.back() = *result.selectedTargetPlaneCrossing;
    }
    return result;
}

[[nodiscard]] FiberTraceOneWayResult traceOneWayCore(
    const FiberPredictionSource& predictions,
    const FiberTraceOneWayRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress,
    std::string phase,
    std::optional<double> traceLengthLimitVoxels = std::nullopt,
    std::optional<int> maximumStepsOverride = std::nullopt,
    const std::function<bool(const cv::Vec3d&, double, int)>& committedStep = {})
{
    const TraceVec start = toTraceVec(request.startPoint);
    const TraceVec target = toTraceVec(request.targetPoint);
    const TraceTargetPlaneSet targetPlanes = makeTraceTargetPlaneSet(request.targetPlanes, traceLengthLimitVoxels.has_value());
    std::optional<float> acceptThresholdVoxels;
    if (request.targetPlaneAcceptThresholdVoxels.has_value()) {
        const double threshold = *request.targetPlaneAcceptThresholdVoxels;
        if (!(threshold >= 0.0) || !std::isfinite(threshold)) {
            throw std::invalid_argument("target-plane acceptance threshold must be finite and non-negative");
        }
        acceptThresholdVoxels = static_cast<float>(threshold);
    }
    if (traceLengthLimitVoxels.has_value()) {
        if (!request.targetPlanes.empty()) {
            throw std::invalid_argument("fiber trace length completion cannot use target planes");
        }
        if (!(*traceLengthLimitVoxels > 0.0) || !std::isfinite(*traceLengthLimitVoxels)) {
            throw std::invalid_argument("fiber trace length limit must be finite and positive");
        }
    }
    const TraceVec referenceStartDirection =
        traceNormalizedOr(toTraceVec(request.initialDirection), traceNormalizedOr(target - start, {1.0f, 0.0f, 0.0f}));
    FiberTraceProfile* profile = request.config.profile;
    if (profile != nullptr) {
        ++profile->oneWayCalls;
        profile->localityCurrentDepth1Dependencies.clear();
        profile->localityPreviousStepDependencies.clear();
    }
    const auto startSampleStart = TraceClock::now();
    const ScoredDirection startPrediction = bestAlignedPrediction(predictions, toVec3d(start), referenceStartDirection, false);
    if (profile != nullptr)
        profile->startSampleSeconds += elapsedSeconds(startSampleStart);
    if (!startPrediction.valid) {
        throw std::invalid_argument("fiber trace start point has no valid prediction direction");
    }
    const TraceVec startDirection = startPrediction.direction;

    const float distance = traceLengthLimitVoxels.has_value()
                               ? static_cast<float>(*traceLengthLimitVoxels)
                               : (request.budgetSpanVoxels > 0.0 ? static_cast<float>(request.budgetSpanVoxels) : traceLength(target - start));
    const float step = std::max(1.0e-3f, static_cast<float>(request.config.stepVoxels));
    const double stepBudget = traceLengthLimitVoxels.has_value()
                                  ? *traceLengthLimitVoxels / static_cast<double>(step)
                                  : static_cast<double>(distance) * request.config.maxStepFactor / static_cast<double>(step);
    const int maxSteps = maximumStepsOverride.value_or(std::max(1, static_cast<int>(std::ceil(stepBudget))));
    if (maxSteps < 1)
        throw std::invalid_argument("fiber trace maximum step override must be positive");

    BeamState initial;
    initial.path = appendBeamPathPoint(nullptr, start);
    initial.previousStepDirection = startDirection;
    initial.currentSampleDirection = startDirection;
    initial.historyDirection = startDirection;
    updateTargetPlaneCrossings(start, start, targetPlanes, initial.crossedTargetPlaneMask, initial.targetPlaneCrossings);
    initial.reached = targetPlanesReached(targetPlanes, initial.crossedTargetPlaneMask, initial.targetPlaneCrossings, acceptThresholdVoxels);
    std::vector<BeamState> beams{initial};
    BeamState lastValidState = initial;
    const auto retainBestValidState = [&lastValidState](const std::vector<BeamState>& states) {
        if (states.empty())
            return;
        lastValidState = *std::min_element(states.begin(), states.end(), beamSearchLess);
    };
    if (initial.reached) {
        return oneWayResultFromState(initial, targetPlanes, true, "target_plane", acceptThresholdVoxels, request.snapTraceToSelectedCrossing, traceLengthLimitVoxels);
    }
    CandidateScoringScratch scoringScratch;
    std::string reason = traceLengthLimitVoxels.has_value() ? "trace_distance_not_reached" : "max_step_factor";

    const int lookaheadSteps = request.config.beamWidth <= 1 ? 1 : std::max(1, request.config.beamLookaheadSteps);
    const auto coneOffsets = request.config.coneAngleStepDegrees > 0.0
                                 ? angleStepConeOffsets(request.config.coneAngleDegrees, request.config.coneAngleStepDegrees)
                                 : legacyGridConeOffsets(request.config.coneAngleDegrees, request.config.coneGridSize);
    int stepIndex = 0;
    while (stepIndex < maxSteps) {
        std::vector<BeamState> expanded = beams;
        int advanced = 0;
        bool prunedFinalFrontier = false;
        for (; advanced < lookaheadSteps && stepIndex + advanced < maxSteps; ++advanced) {
            if (profile != nullptr)
                ++profile->generations;
            float generationStep = step;
            if (traceLengthLimitVoxels.has_value()) {
                const double nominalTracedLength = static_cast<double>(stepIndex + advanced) * static_cast<double>(step);
                generationStep = static_cast<float>(
                    std::min(static_cast<double>(step), std::max(static_cast<double>(kTraceEpsilon), *traceLengthLimitVoxels - nominalTracedLength)));
            }
            const bool finalLookaheadGeneration = advanced + 1 >= lookaheadSteps || stepIndex + advanced + 1 >= maxSteps;
            const bool lazyFinalGeneration = finalLookaheadGeneration && advanced > 0 && request.config.lazyLookahead;
            const std::vector<CandidateTask>* tasksPtr = nullptr;
            const std::vector<TraceVec>* candidatePointsPtr = nullptr;
            const std::vector<CandidateScore>* scoresPtr = nullptr;
            size_t evaluatedParents = expanded.size();
            size_t evaluatedChildCandidates = 0;
            size_t totalChildCandidates = 0;
            if (lazyFinalGeneration) {
                const LazyLookaheadEvaluation evaluation = evaluateLazyFinalFrontier(
                    predictions, normalSampler, expanded, coneOffsets, generationStep, targetPlanes, acceptThresholdVoxels, request.config, advanced + 1, scoringScratch, profile);
                tasksPtr = &scoringScratch.lookaheadTasks;
                scoresPtr = &scoringScratch.lookaheadScores;
                evaluatedParents = evaluation.evaluatedParents;
                evaluatedChildCandidates = evaluation.evaluatedChildCandidates;
                totalChildCandidates = evaluation.totalChildCandidates;
            } else {
                const auto taskBuildStart = TraceClock::now();
                buildCandidateTasks(scoringScratch.tasks, scoringScratch.candidatePoints, expanded, coneOffsets, generationStep);
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
                    frontierOutput = {&frontier, &targetPlanes, acceptThresholdVoxels, nullptr, 0};
                    frontierOutputPtr = &frontierOutput;
                }
                scoresPtr =
                    &scoreCandidateTasks(predictions, normalSampler, expanded, scoringScratch.tasks, scoringScratch.candidatePoints, request.config, advanced + 1, {}, frontierOutputPtr, scoringScratch, profile);
                evaluatedChildCandidates = scoringScratch.tasks.size();
                totalChildCandidates = scoringScratch.tasks.size();
            }
            const auto& tasks = *tasksPtr;
            const auto& scores = *scoresPtr;

            if (finalLookaheadGeneration) {
                auto& frontier = scoringScratch.frontierCandidates;
                const auto bestReachedIndex = bestReachedFrontierCandidateIndex(frontier);
                if (bestReachedIndex.has_value()) {
                    if (advanced > 0) {
                        recordExactLookaheadPotential(
                            profile, expanded, totalChildCandidates, evaluatedParents, evaluatedChildCandidates, frontier[*bestReachedIndex].loss, true);
                    }
                    const BeamState bestReached =
                        beamStateFromFrontierCandidate(expanded, tasks, scores, *bestReachedIndex, frontier[*bestReachedIndex], request.config);
                    if (progress) {
                        FiberTraceProgress event;
                        event.phase = phase;
                        event.step = stepIndex + advanced + 1;
                        event.maxSteps = maxSteps;
                        event.targetPlaneProgress = 1.0;
                        event.reason = "target_plane";
                        progress(event);
                    }
                    return oneWayResultFromState(bestReached, targetPlanes, true, bestReached.reason, acceptThresholdVoxels, request.snapTraceToSelectedCrossing, traceLengthLimitVoxels);
                }
                const auto pruneStart = TraceClock::now();
                std::vector<size_t> selectedIndices;
                beams = pruneFrontierCandidates(
                    frontier, expanded, tasks, scores, request.config.beamWidth, request.config.beamPruneDistanceVoxels, request.config, advanced > 0 ? &selectedIndices : nullptr);
                retainBestValidState(beams);
                if (advanced > 0) {
                    const bool complete = selectedIndices.size() >= static_cast<size_t>(request.config.beamWidth);
                    std::optional<float> threshold;
                    if (complete) {
                        threshold = 0.0f;
                        for (const size_t index : selectedIndices) {
                            threshold = std::max(*threshold, frontierPruneScore(frontier[index]));
                        }
                    }
                    recordExactLookaheadPotential(profile, expanded, totalChildCandidates, evaluatedParents, evaluatedChildCandidates, threshold, complete);
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
                const FrontierCandidate candidate =
                    makeFrontierCandidate(beam, candidatePoints[taskIndex], candidateScore, targetPlanes, acceptThresholdVoxels, taskIndex);
                nextFrontier.push_back(beamStateFromFrontierCandidate(expanded, tasks, scores, taskIndex, candidate, request.config));
            }
            if (profile != nullptr)
                profile->frontierSeconds += elapsedSeconds(frontierStart);
            expanded = std::move(nextFrontier);
            retainBestValidState(expanded);
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
                return oneWayResultFromState(bestReached, targetPlanes, true, bestReached.reason, acceptThresholdVoxels, request.snapTraceToSelectedCrossing, traceLengthLimitVoxels);
            }
        }
        if (expanded.empty() && !prunedFinalFrontier)
            break;

        if (!prunedFinalFrontier) {
            const auto pruneStart = TraceClock::now();
            beams = pruneBeamStates(std::move(expanded), request.config.beamWidth, request.config.beamPruneDistanceVoxels);
            retainBestValidState(beams);
            if (profile != nullptr)
                profile->pruneSeconds += elapsedSeconds(pruneStart);
        }
        stepIndex += std::max(1, advanced);

        if (committedStep) {
            const auto best = std::min_element(beams.begin(), beams.end(), beamSearchLess);
            if (!committedStep(toVec3d(beamEndpoint(*best)), static_cast<double>(best->loss), stepIndex)) {
                return oneWayResultFromState(*best, targetPlanes, false, "observer_stop", acceptThresholdVoxels, request.snapTraceToSelectedCrossing, traceLengthLimitVoxels);
            }
        }

        if (progress) {
            FiberTraceProgress event;
            event.phase = phase;
            event.step = stepIndex;
            event.maxSteps = maxSteps;
            if (traceLengthLimitVoxels.has_value()) {
                event.targetPlaneProgress = maxSteps > 0 ? std::min(1.0, static_cast<double>(stepIndex) / static_cast<double>(maxSteps)) : 1.0;
            } else {
                const float targetDistance = traceLength(beamEndpoint(beams.front()) - target);
                event.targetPlaneProgress = distance > kTraceEpsilon ? 1.0 - std::min(1.0f, targetDistance / distance) : 1.0;
            }
            const bool reachedTraceLength = traceLengthLimitVoxels.has_value() && stepIndex >= maxSteps;
            event.reason = reachedTraceLength ? "trace_distance" : (beams.front().reached ? beams.front().reason : reason);
            progress(event);
        }
    }

    if (beams.empty()) {
        return oneWayResultFromState(lastValidState, targetPlanes, false, reason, acceptThresholdVoxels, request.snapTraceToSelectedCrossing, traceLengthLimitVoxels);
    }

    const auto best = std::min_element(beams.begin(), beams.end(), beamSearchLess);
    const bool reachedTraceLength = traceLengthLimitVoxels.has_value() && stepIndex >= maxSteps;
    const bool reached = best->reached || reachedTraceLength;
    return oneWayResultFromState(
        *best,
        targetPlanes,
        reached,
        reachedTraceLength ? "trace_distance" : (best->reached ? best->reason : reason),
        acceptThresholdVoxels,
        request.snapTraceToSelectedCrossing,
        traceLengthLimitVoxels);
}

struct ResampledTrace {
    std::vector<cv::Vec3d> points;
    std::vector<double> lengths;
};

[[nodiscard]] std::vector<cv::Vec3d> finiteDeduplicatedTrace(const std::vector<cv::Vec3d>& points)
{
    std::vector<cv::Vec3d> out;
    out.reserve(points.size());
    for (const auto& point : points) {
        if (!finitePoint(point))
            return {};
        if (out.empty() || length(point - out.back()) > 1.0e-8)
            out.push_back(point);
    }
    return out;
}

[[nodiscard]] ResampledTrace resampleTraceWithLengths(const std::vector<cv::Vec3d>& input, double stepVoxels)
{
    ResampledTrace out;
    const auto points = finiteDeduplicatedTrace(input);
    if (points.empty())
        return out;
    const auto sourceLengths = arclengths(points);
    const double total = sourceLengths.back();
    if (points.size() == 1 || !(total > 1.0e-8) || !std::isfinite(total)) {
        out.points = points;
        out.lengths.assign(points.size(), 0.0);
        return out;
    }

    const double stride = std::max(1.0e-6, stepVoxels);
    const size_t sampleCount = static_cast<size_t>(std::floor(total / stride)) + 1;
    out.points.reserve(sampleCount + 1);
    out.lengths.reserve(sampleCount + 1);
    size_t segment = 0;
    const auto appendAt = [&](double sampleLength, size_t& sourceSegment, ResampledTrace& result) {
        while (sourceSegment + 1 < sourceLengths.size() && sourceLengths[sourceSegment + 1] < sampleLength) {
            ++sourceSegment;
        }
        sourceSegment = std::min(sourceSegment, points.size() - 2);
        const double begin = sourceLengths[sourceSegment];
        const double end = sourceLengths[sourceSegment + 1];
        const double t = end > begin ? std::clamp((sampleLength - begin) / (end - begin), 0.0, 1.0) : 0.0;
        result.points.push_back(points[sourceSegment] * (1.0 - t) + points[sourceSegment + 1] * t);
        result.lengths.push_back(sampleLength);
    };
    for (size_t index = 0; index < sampleCount; ++index)
        appendAt(std::min(total, static_cast<double>(index) * stride), segment, out);
    if (out.lengths.empty() || total - out.lengths.back() > 1.0e-8)
        appendAt(total, segment, out);
    else {
        out.points.back() = points.back();
        out.lengths.back() = total;
    }
    out.points.front() = points.front();
    out.points.back() = points.back();
    return out;
}

[[nodiscard]] cv::Vec3d localTraceTangent(const std::vector<cv::Vec3d>& points, size_t index)
{
    if (points.size() < 2 || index >= points.size())
        return {};
    for (size_t radius = 1; radius < points.size(); ++radius) {
        const size_t first = index > radius ? index - radius : 0;
        const size_t last = std::min(points.size() - 1, index + radius);
        const cv::Vec3d tangent = normalizedOrZero(points[last] - points[first]);
        if (length(tangent) > kEpsilon)
            return tangent;
        if (first == 0 && last + 1 == points.size())
            break;
    }
    return {};
}

struct SegmentPlaneIntersection {
    cv::Vec3d point{0.0, 0.0, 0.0};
    double segmentFraction = 0.0;
};

[[nodiscard]] std::optional<SegmentPlaneIntersection> segmentPlaneIntersection(
    const cv::Vec3d& start, const cv::Vec3d& end, const cv::Vec3d& planePoint, const cv::Vec3d& planeNormal)
{
    constexpr double epsilon = 1.0e-9;
    const double d0 = pointToPlaneSigned(start, planePoint, planeNormal);
    const double d1 = pointToPlaneSigned(end, planePoint, planeNormal);
    if (!std::isfinite(d0) || !std::isfinite(d1))
        return std::nullopt;
    const cv::Vec3d delta = end - start;
    const double lengthSquared = delta.dot(delta);
    if (std::abs(d0) <= epsilon && std::abs(d1) <= epsilon) {
        const double t = lengthSquared > epsilon ? std::clamp((planePoint - start).dot(delta) / lengthSquared, 0.0, 1.0) : 0.0;
        return SegmentPlaneIntersection{start + delta * t, t};
    }
    if (d0 * d1 > 0.0)
        return std::nullopt;
    const double denominator = d0 - d1;
    if (std::abs(denominator) <= epsilon)
        return std::nullopt;
    const double t = std::clamp(d0 / denominator, 0.0, 1.0);
    return SegmentPlaneIntersection{start + delta * t, t};
}

struct TraceArcProjection {
    cv::Vec3d point{0.0, 0.0, 0.0};
    double arcLength = 0.0;
    double distance = std::numeric_limits<double>::infinity();
};

[[nodiscard]] std::optional<TraceArcProjection> projectPointToTrace(const ResampledTrace& trace, const cv::Vec3d& point)
{
    if (trace.points.empty() || trace.points.size() != trace.lengths.size())
        return std::nullopt;
    if (trace.points.size() == 1) {
        return TraceArcProjection{trace.points.front(), 0.0, length(trace.points.front() - point)};
    }
    TraceArcProjection best;
    for (size_t index = 0; index + 1 < trace.points.size(); ++index) {
        const cv::Vec3d delta = trace.points[index + 1] - trace.points[index];
        const double lengthSquared = delta.dot(delta);
        const double t = lengthSquared > 1.0e-16 ? std::clamp((point - trace.points[index]).dot(delta) / lengthSquared, 0.0, 1.0) : 0.0;
        const cv::Vec3d projected = trace.points[index] + delta * t;
        const double distance = length(projected - point);
        if (distance < best.distance) {
            best.point = projected;
            best.arcLength = trace.lengths[index] + (trace.lengths[index + 1] - trace.lengths[index]) * t;
            best.distance = distance;
        }
    }
    return best;
}

struct TraceMeetingCandidate {
    cv::Vec3d forwardPoint{0.0, 0.0, 0.0};
    cv::Vec3d reversePoint{0.0, 0.0, 0.0};
    double forwardArcLength = 0.0;
    double reverseArcLength = 0.0;
    double error = std::numeric_limits<double>::infinity();
    std::string source;
    size_t stableIndex = 0;
};

[[nodiscard]] bool meetingCandidateLess(const TraceMeetingCandidate& lhs, const TraceMeetingCandidate& rhs)
{
    if (lhs.error != rhs.error)
        return lhs.error < rhs.error;
    const double lhsBalanced = std::min(lhs.forwardArcLength, lhs.reverseArcLength);
    const double rhsBalanced = std::min(rhs.forwardArcLength, rhs.reverseArcLength);
    if (lhsBalanced != rhsBalanced)
        return lhsBalanced > rhsBalanced;
    const double lhsCombined = lhs.forwardArcLength + lhs.reverseArcLength;
    const double rhsCombined = rhs.forwardArcLength + rhs.reverseArcLength;
    if (lhsCombined != rhsCombined)
        return lhsCombined > rhsCombined;
    return lhs.stableIndex < rhs.stableIndex;
}

void appendMovingPlaneCandidates(const ResampledTrace& source, const ResampledTrace& opposite, bool sourceIsForward, std::vector<TraceMeetingCandidate>& candidates)
{
    if (source.points.size() < 2 || opposite.points.empty())
        return;
    for (size_t sourceIndex = 0; sourceIndex < source.points.size(); ++sourceIndex) {
        const cv::Vec3d tangent = localTraceTangent(source.points, sourceIndex);
        if (length(tangent) <= kEpsilon)
            continue;
        if (opposite.points.size() == 1) {
            if (std::abs(pointToPlaneSigned(opposite.points.front(), source.points[sourceIndex], tangent)) > 1.0e-9) {
                continue;
            }
            TraceMeetingCandidate candidate;
            candidate.forwardPoint = sourceIsForward ? source.points[sourceIndex] : opposite.points.front();
            candidate.reversePoint = sourceIsForward ? opposite.points.front() : source.points[sourceIndex];
            candidate.forwardArcLength = sourceIsForward ? source.lengths[sourceIndex] : opposite.lengths.front();
            candidate.reverseArcLength = sourceIsForward ? opposite.lengths.front() : source.lengths[sourceIndex];
            candidate.error = length(candidate.forwardPoint - candidate.reversePoint);
            candidate.source = sourceIsForward ? "forward_moving_plane" : "reverse_moving_plane";
            candidate.stableIndex = candidates.size();
            candidates.push_back(std::move(candidate));
            continue;
        }
        for (size_t segment = 0; segment + 1 < opposite.points.size(); ++segment) {
            const auto crossing = segmentPlaneIntersection(opposite.points[segment], opposite.points[segment + 1], source.points[sourceIndex], tangent);
            if (!crossing)
                continue;
            const double oppositeArc = opposite.lengths[segment] + (opposite.lengths[segment + 1] - opposite.lengths[segment]) * crossing->segmentFraction;
            TraceMeetingCandidate candidate;
            candidate.forwardPoint = sourceIsForward ? source.points[sourceIndex] : crossing->point;
            candidate.reversePoint = sourceIsForward ? crossing->point : source.points[sourceIndex];
            candidate.forwardArcLength = sourceIsForward ? source.lengths[sourceIndex] : oppositeArc;
            candidate.reverseArcLength = sourceIsForward ? oppositeArc : source.lengths[sourceIndex];
            candidate.error = length(candidate.forwardPoint - candidate.reversePoint);
            candidate.source = sourceIsForward ? "forward_moving_plane" : "reverse_moving_plane";
            candidate.stableIndex = candidates.size();
            if (std::isfinite(candidate.error))
                candidates.push_back(std::move(candidate));
        }
    }
}

[[nodiscard]] std::vector<cv::Vec3d> tracePrefixAtArc(const ResampledTrace& trace, double arcLength)
{
    if (trace.points.empty())
        return {};
    const double cut = std::clamp(arcLength, 0.0, trace.lengths.back());
    std::vector<cv::Vec3d> out;
    out.reserve(trace.points.size());
    out.push_back(trace.points.front());
    for (size_t index = 1; index < trace.points.size(); ++index) {
        if (trace.lengths[index] < cut - 1.0e-9) {
            if (length(trace.points[index] - out.back()) > 1.0e-8)
                out.push_back(trace.points[index]);
            continue;
        }
        const double begin = trace.lengths[index - 1];
        const double end = trace.lengths[index];
        const double t = end > begin ? std::clamp((cut - begin) / (end - begin), 0.0, 1.0) : 0.0;
        const cv::Vec3d point = trace.points[index - 1] * (1.0 - t) + trace.points[index] * t;
        if (length(point - out.back()) > 1.0e-8)
            out.push_back(point);
        else
            out.back() = point;
        return out;
    }
    if (length(trace.points.back() - out.back()) > 1.0e-8)
        out.push_back(trace.points.back());
    return out;
}

[[nodiscard]] std::vector<cv::Vec3d> warpTracePrefixToMidpoint(
    std::vector<cv::Vec3d> partial, const cv::Vec3d& anchor, const cv::Vec3d& sourceMeeting, const cv::Vec3d& midpoint)
{
    if (partial.empty())
        return {};
    if (partial.size() == 1)
        partial.push_back(sourceMeeting);
    partial.front() = anchor;
    partial.back() = sourceMeeting;
    const auto lengths = arclengths(partial);
    const double total = lengths.back();
    const cv::Vec3d delta = midpoint - sourceMeeting;
    for (size_t index = 0; index < partial.size(); ++index) {
        const double blend = total > 1.0e-8 ? std::clamp(lengths[index] / total, 0.0, 1.0)
                                            : (partial.size() > 1 ? static_cast<double>(index) / static_cast<double>(partial.size() - 1) : 1.0);
        partial[index] += delta * blend;
    }
    partial.front() = anchor;
    partial.back() = midpoint;
    return partial;
}

struct TraceMeetingFusion {
    std::vector<cv::Vec3d> fusedLine;
    double errorTraceVoxels = std::numeric_limits<double>::infinity();
    double errorRatio = std::numeric_limits<double>::infinity();
    double traceLengthTraceVoxels = 0.0;
    std::string source;
    std::string reason = "invalid_trace_path";
    std::string detail;
};

[[nodiscard]] TraceMeetingFusion fuseTraceMeetings(const FiberTraceOneWayResult& forwardResult, const FiberTraceOneWayResult& reverseResult, const FiberTraceConfig& config)
{
    TraceMeetingFusion result;
    const double searchStep = std::max(1.0e-6, config.stepVoxels * 0.5);
    const ResampledTrace forward = resampleTraceWithLengths(forwardResult.points, searchStep);
    const ResampledTrace reverse = resampleTraceWithLengths(reverseResult.points, searchStep);
    if (forward.points.empty() || reverse.points.empty()) {
        result.detail = "forward or reverse trace is empty or non-finite";
        return result;
    }

    std::vector<TraceMeetingCandidate> candidates;
    const double endpointThresholdTrace = config.endpointAcceptThresholdBaseVoxels / config.traceToBaseScale;
    const auto appendEndpointCandidates = [&](const FiberTraceOneWayResult& traced, const ResampledTrace& tracedPath, bool forwardEndpoint) {
        for (const auto& crossing : traced.targetPlaneCrossings) {
            if (!(crossing.inPlaneErrorVoxels <= endpointThresholdTrace) || !finitePoint(crossing.point)) {
                continue;
            }
            const auto projected = projectPointToTrace(tracedPath, crossing.point);
            if (!projected)
                continue;
            TraceMeetingCandidate candidate;
            candidate.forwardPoint = forwardEndpoint ? crossing.point : forward.points.front();
            candidate.reversePoint = forwardEndpoint ? reverse.points.front() : crossing.point;
            candidate.forwardArcLength = forwardEndpoint ? projected->arcLength : 0.0;
            candidate.reverseArcLength = forwardEndpoint ? 0.0 : projected->arcLength;
            candidate.error = length(candidate.forwardPoint - candidate.reversePoint);
            candidate.source = std::string(forwardEndpoint ? "forward_endpoint:" : "reverse_endpoint:") + crossing.name;
            candidate.stableIndex = candidates.size();
            if (std::isfinite(candidate.error))
                candidates.push_back(std::move(candidate));
        }
    };
    appendEndpointCandidates(forwardResult, forward, true);
    appendEndpointCandidates(reverseResult, reverse, false);
    appendMovingPlaneCandidates(forward, reverse, true, candidates);
    appendMovingPlaneCandidates(reverse, forward, false, candidates);

    candidates.erase(
        std::remove_if(
            candidates.begin(),
            candidates.end(),
            [](const auto& candidate) {
                const double tracedLength = candidate.forwardArcLength + candidate.reverseArcLength;
                return !std::isfinite(candidate.error) || !std::isfinite(tracedLength) || !(tracedLength > 1.0e-8);
            }),
        candidates.end());
    if (candidates.empty()) {
        result.reason = "no_trace_plane_intersection";
        result.detail = "forward=" + forwardResult.reason + " reverse=" + reverseResult.reason;
        return result;
    }

    const auto best = std::min_element(candidates.begin(), candidates.end(), meetingCandidateLess);
    result.errorTraceVoxels = best->error;
    result.traceLengthTraceVoxels = best->forwardArcLength + best->reverseArcLength;
    result.errorRatio = result.errorTraceVoxels / result.traceLengthTraceVoxels;
    result.source = best->source;

    auto forwardPartial = tracePrefixAtArc(forward, best->forwardArcLength);
    auto reversePartial = tracePrefixAtArc(reverse, best->reverseArcLength);
    const cv::Vec3d midpoint = (best->forwardPoint + best->reversePoint) * 0.5;
    forwardPartial = warpTracePrefixToMidpoint(std::move(forwardPartial), forward.points.front(), best->forwardPoint, midpoint);
    reversePartial = warpTracePrefixToMidpoint(std::move(reversePartial), reverse.points.front(), best->reversePoint, midpoint);
    std::reverse(reversePartial.begin(), reversePartial.end());
    std::vector<cv::Vec3d> fusedDense = std::move(forwardPartial);
    if (!reversePartial.empty()) {
        const size_t begin = fusedDense.empty() ? 0 : 1;
        fusedDense.insert(fusedDense.end(), reversePartial.begin() + static_cast<std::ptrdiff_t>(begin), reversePartial.end());
    }
    result.fusedLine = resampleTraceWithLengths(fusedDense, config.stepVoxels).points;
    if (result.fusedLine.size() < 2) {
        result.reason = "fusion_failed";
        result.detail = "selected meeting did not produce a CP-to-CP polyline";
        result.fusedLine.clear();
        return result;
    }
    result.fusedLine.front() = forward.points.front();
    result.fusedLine.back() = reverse.points.front();
    const double errorBaseVoxels = result.errorTraceVoxels * config.traceToBaseScale;
    const double traceLengthBaseVoxels = result.traceLengthTraceVoxels * config.traceToBaseScale;
    const double acceptThresholdBaseVoxels = std::max(10.0, config.meetingAcceptMaxErrorRatio * traceLengthBaseVoxels);
    if (!(errorBaseVoxels <= acceptThresholdBaseVoxels)) {
        result.reason = "meeting_error_threshold";
        std::ostringstream detail;
        detail << "error_base_voxels=" << errorBaseVoxels << " threshold_base_voxels=" << acceptThresholdBaseVoxels
               << " ratio=" << result.errorRatio << " ratio_threshold=" << config.meetingAcceptMaxErrorRatio << " source=" << result.source;
        result.detail = detail.str();
        return result;
    }
    result.reason = "ok";
    return result;
}

}  // namespace

#ifdef VC_TESTING
namespace testing
{

namespace
{

[[nodiscard]] std::vector<BeamState> debugStatesToBeamStates(const std::vector<BeamDebugState>& states)
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

}  // namespace

std::vector<size_t> debugPruneBeamStateIndices(const std::vector<BeamDebugState>& states, int beamWidth, double pruneDistanceVoxels)
{
    std::vector<size_t> indices;
    auto pruned = pruneBeamStates(debugStatesToBeamStates(states), beamWidth, pruneDistanceVoxels);
    indices.reserve(pruned.size());
    for (const auto& state : pruned)
        indices.push_back(static_cast<size_t>(std::stoull(state.reason)));
    return indices;
}

std::optional<size_t> debugBestReachedBeamStateIndex(const std::vector<BeamDebugState>& states)
{
    const auto beamStates = debugStatesToBeamStates(states);
    return bestReachedStateIndexPythonParity(beamStates);
}

namespace
{

class DebugPredictionSource final : public FiberPredictionSource
{
public:
    explicit DebugPredictionSource(bool concurrent) : concurrent_(concurrent) {}

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override { return concurrent_; }

    [[nodiscard]] FiberPredictionSample sample(const cv::Vec3d&, const cv::Vec3d&) const override { return {}; }

private:
    bool concurrent_ = false;
};

class DebugNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    explicit DebugNormalSampler(bool concurrent) : concurrent_(concurrent) {}

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override { return concurrent_; }

    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {}; }

private:
    bool concurrent_ = false;
};

}  // namespace

int debugTraceWorkerCount(bool predictionConcurrent, bool normalConcurrent, bool hasNormalSampler, int parallelThreads, size_t taskCount)
{
    FiberTraceConfig config;
    config.parallelThreads = parallelThreads;
    DebugPredictionSource predictions(predictionConcurrent);
    DebugNormalSampler normals(normalConcurrent);
    return traceWorkerCount(predictions, hasNormalSampler ? &normals : nullptr, config, taskCount);
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
    const CandidateScore score = candidateLossFromCorners(corners, optionCount, pointIndex, beam, toTraceVec(candidateDirection), config);
    return {score.loss, toVec3d(score.selectedCurrentDirection), score.selectedPresence, score.valid};
}

CandidateScoreDebug debugCandidateLossFromSample(
    const FiberPredictionSample& sample,
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
    const CandidateScore score = candidateLossFromSample(sample, beam, toTraceVec(candidateDirection), config);
    return {score.loss, toVec3d(score.selectedCurrentDirection), score.selectedPresence, score.valid};
}

size_t debugExactLookaheadRequiredParentCount(const std::vector<double>& parentLowerBounds, std::optional<double> resultThreshold, bool finalBeamSetComplete)
{
    std::vector<BeamState> parents(parentLowerBounds.size());
    for (size_t index = 0; index < parentLowerBounds.size(); ++index)
        parents[index].loss = static_cast<float>(parentLowerBounds[index]);
    const std::optional<float> floatThreshold = resultThreshold.has_value() ? std::optional<float>{static_cast<float>(*resultThreshold)} : std::nullopt;
    return exactLookaheadRequiredParentCount(parents, floatThreshold, finalBeamSetComplete);
}

std::vector<size_t> debugOrderedIndexPrefix(const std::vector<double>& losses, size_t limit)
{
    return orderedIndexPrefix(losses.size(), limit, [&](size_t index) { return losses[index]; });
}

bool debugShouldRetryLookahead(bool lazyLookahead, size_t parentCap, size_t retryParentCap, bool segmentSuccess)
{
    return shouldRetryLookahead(lazyLookahead, parentCap, retryParentCap, segmentSuccess);
}

FiberTraceSegmentResult debugFuseTraceSegment(const std::vector<cv::Vec3d>& forward, const std::vector<cv::Vec3d>& reverse, const FiberTraceConfig& config)
{
    validateTraceConfig(config);
    FiberTraceSegmentResult result;
    result.forward.points = forward;
    result.forward.reason = "debug_forward";
    result.reverse.points = reverse;
    result.reverse.reason = "debug_reverse";
    const TraceMeetingFusion fusion = fuseTraceMeetings(result.forward, result.reverse, config);
    result.fusedLine = fusion.fusedLine;
    result.meetingErrorTraceVoxels = fusion.errorTraceVoxels;
    const FiberTraceCoordinateAdapter coordinates(config.traceToBaseScale);
    result.meetingErrorBaseVoxels = coordinates.traceDistanceToBase(fusion.errorTraceVoxels);
    result.meetingErrorRatio = fusion.errorRatio;
    result.meetingTraceLengthTraceVoxels = fusion.traceLengthTraceVoxels;
    result.meetingSource = fusion.source;
    result.reason = fusion.reason;
    result.detail = fusion.detail;
    result.accepted = result.reason == "ok";
    return result;
}

}  // namespace testing
#endif

FiberTraceCoordinateAdapter::FiberTraceCoordinateAdapter(double traceToBaseScaleValue) : traceToBaseScale(traceToBaseScaleValue)
{
    if (!(traceToBaseScale > 0.0) || !std::isfinite(traceToBaseScale)) {
        throw std::invalid_argument("fiber trace-to-base scale must be positive and finite");
    }
}

cv::Vec3d FiberTraceCoordinateAdapter::baseToTrace(const cv::Vec3d& point) const
{
    return point / traceToBaseScale;
}

cv::Vec3d FiberTraceCoordinateAdapter::traceToBase(const cv::Vec3d& point) const
{
    return point * traceToBaseScale;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::baseToTrace(const std::vector<cv::Vec3d>& points) const
{
    std::vector<cv::Vec3d> converted;
    converted.reserve(points.size());
    for (const auto& point : points)
        converted.push_back(baseToTrace(point));
    return converted;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::traceToBase(const std::vector<cv::Vec3d>& points) const
{
    std::vector<cv::Vec3d> converted;
    converted.reserve(points.size());
    for (const auto& point : points)
        converted.push_back(traceToBase(point));
    return converted;
}

std::vector<cv::Vec3d> FiberTraceCoordinateAdapter::traceSegmentToBase(
    const std::vector<cv::Vec3d>& points, const cv::Vec3d& exactStartBase, const cv::Vec3d& exactTargetBase) const
{
    if (points.size() < 2) {
        throw std::invalid_argument("fiber trace segment must contain at least two points");
    }
    auto converted = traceToBase(points);
    converted.front() = exactStartBase;
    converted.back() = exactTargetBase;
    return converted;
}

double FiberTraceCoordinateAdapter::baseDistanceToTrace(double distanceBaseVoxels) const
{
    if (std::isnan(distanceBaseVoxels))
        throw std::invalid_argument("base distance must not be NaN");
    return distanceBaseVoxels / traceToBaseScale;
}

double FiberTraceCoordinateAdapter::traceDistanceToBase(double distanceTraceVoxels) const
{
    if (std::isnan(distanceTraceVoxels))
        throw std::invalid_argument("trace distance must not be NaN");
    return distanceTraceVoxels * traceToBaseScale;
}

FiberPredictionTraceScales resolveFiberPredictionTraceScales(const vc::lasagna::LasagnaDatasetManifest& manifest, int inferenceScaledownPower)
{
    if (inferenceScaledownPower < 0 || inferenceScaledownPower > 30) {
        throw std::runtime_error("fiber inference scaledown power must be in [0, 30]");
    }
    const double inferenceScaledown = static_cast<double>(1 << inferenceScaledownPower);

    if (!manifest.raw.empty()) {
        const auto sourceIt = manifest.raw.find("source_to_base");
        if (sourceIt == manifest.raw.end() || !sourceIt->is_number()) {
            throw std::runtime_error("fiber inference manifest must contain numeric source_to_base");
        }
    }
    if (!(manifest.sourceToBase > 0.0) || !std::isfinite(manifest.sourceToBase)) {
        throw std::runtime_error("fiber inference manifest source_to_base must be positive and finite");
    }

    const auto prefixes = fiberPredictionPrefixes(manifest);
    if (prefixes.empty()) {
        throw std::runtime_error("fiber inference dataset must contain presence/nx/ny channels");
    }

    std::optional<double> predictionToBaseScale;
    std::optional<double> predictionGroupScaleFactor;
    std::optional<std::string> inferredChannel;
    for (const auto& prefix : prefixes) {
        for (const auto& channel : predictionChannelNames(prefix)) {
            const auto& group = predictionChannelGroup(manifest, channel);
            const double scale = predictionChannelEffectiveScale(manifest, channel);
            const double groupScaleFactor = static_cast<double>(group.scaleFactor());
            if (!predictionToBaseScale.has_value()) {
                predictionToBaseScale = scale;
                predictionGroupScaleFactor = groupScaleFactor;
                inferredChannel = channel;
            } else if (!nearlySameScale(*predictionToBaseScale, scale)) {
                throw std::runtime_error(
                    "fiber inference prediction channels must share one effective "
                    "prediction-to-base scale; channel '" +
                    channel + "' has scale " + std::to_string(scale) + " but channel '" + *inferredChannel + "' has scale " +
                    std::to_string(*predictionToBaseScale));
            } else if (!nearlySameScale(*predictionGroupScaleFactor, groupScaleFactor)) {
                throw std::runtime_error(
                    "fiber inference prediction channels must share one "
                    "manifest group scale factor; channel '" +
                    channel + "' has factor " + std::to_string(groupScaleFactor) + " but channel '" + *inferredChannel + "' has factor " +
                    std::to_string(*predictionGroupScaleFactor));
            }
        }
    }

    const double traceToBaseScale = *predictionToBaseScale / inferenceScaledown;
    if (!(traceToBaseScale > 0.0) || !std::isfinite(traceToBaseScale)) {
        throw std::runtime_error("fiber inference manifest derived trace scale must be positive and finite");
    }

    return {
        traceToBaseScale,
        *predictionToBaseScale,
        inferenceScaledown,
    };
}

double inferFiberPredictionWorkingToBaseScale(const vc::lasagna::LasagnaDatasetManifest& manifest, int inferenceScaledownPower)
{
    return resolveFiberPredictionTraceScales(manifest, inferenceScaledownPower).traceToBaseScale;
}

class FiberPredictionField::Impl
{
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

    Impl(const vc::lasagna::LasagnaDataset& dataset, size_t maxCachedBytes, FiberPredictionFieldBindingMode bindingMode)
        : cache_(vc::lasagna::sharedLasagnaChannelChunkCache(maxCachedBytes))
    {
        const auto& manifest = dataset.manifest();
        sourceScaleExplicit_ = manifest.raw.empty() || (manifest.raw.contains("source_to_base") && manifest.raw.at("source_to_base").is_number());
        const auto availablePrefixes = fiberPredictionPrefixes(manifest);
        const std::vector<std::string> prefixes =
            bindingMode == FiberPredictionFieldBindingMode::CanonicalStoredGrid
                ? (std::find(availablePrefixes.begin(), availablePrefixes.end(), std::string{}) != availablePrefixes.end()
                       ? std::vector<std::string>{std::string{}}
                       : std::vector<std::string>{})
                : availablePrefixes;
        if (prefixes.empty())
            throw std::runtime_error(
                bindingMode == FiberPredictionFieldBindingMode::CanonicalStoredGrid
                    ? "anchor extraction requires canonical presence/nx/ny channels"
                    : "fiber inference dataset must contain presence/nx/ny channels");

        cornerBudget_ = std::make_shared<vc::render::DecodedChunkCacheBudget>(maxCachedBytes);
        options_.reserve(prefixes.size());
        for (const auto& prefix : prefixes) {
            const auto channels = predictionChannelNames(prefix);
            Option option;
            option.name = prefix.empty() ? std::string("option_000") : prefix;
            option.presence = vc::lasagna::bindLasagnaChannel(manifest, channels[0]);
            option.nx = vc::lasagna::bindLasagnaChannel(manifest, channels[1]);
            option.ny = vc::lasagna::bindLasagnaChannel(manifest, channels[2]);
            const bool cornerCompatible = vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.nx) &&
                                          vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.ny);
            if (cornerCompatible) {
                option.presenceCorners = std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(option.presence, maxCachedBytes, cornerBudget_);
                option.nxCorners = std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(option.nx, maxCachedBytes, cornerBudget_);
                option.nyCorners = std::make_unique<vc::lasagna::LasagnaChannelCornerSampler>(option.ny, maxCachedBytes, cornerBudget_);
            } else {
                cornerSamplingAvailable_ = false;
            }
            options_.push_back(std::move(option));
            if (prefix.empty())
                canonicalOptionIndex_ = options_.size() - 1;
        }
        if (canonicalOptionIndex_.has_value()) {
            const auto& canonical = options_[*canonicalOptionIndex_];
            const auto sameShape = canonical.presence.shapeZYX == canonical.nx.shapeZYX && canonical.presence.shapeZYX == canonical.ny.shapeZYX;
            const auto closeSpacing = [](double a, double b) {
                const double tolerance = 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
                return std::abs(a - b) <= tolerance;
            };
            if (!sameShape || !closeSpacing(canonical.presence.spacing, canonical.nx.spacing) ||
                !closeSpacing(canonical.presence.spacing, canonical.ny.spacing)) {
                throw std::runtime_error("canonical fiber prediction presence/nx/ny channels must share shape and spacing");
            }
            canonicalGridInfo_.shapeZYX = canonical.presence.shapeZYX;
            canonicalGridInfo_.predictionToBaseScale = manifest.sourceToBase * static_cast<double>(canonical.presence.group->scaleFactor());
        }
        optionGrids_.reserve(options_.size());
        for (const auto& option : options_) {
            optionGrids_.push_back({
                vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.nx) &&
                    vc::lasagna::sameLasagnaSamplingGrid(option.presence, option.ny),
            });
        }
    }

    [[nodiscard]] vc::lasagna::NormalPrefetchReport prefetchSamples(const std::vector<cv::Vec3d>& volumePoints) const
    {
        std::vector<vc::lasagna::LasagnaChannelChunkCache::PrefetchRequest> requests;
        requests.reserve(volumePoints.size() * options_.size() * 24);
        std::vector<vc::lasagna::LasagnaChannelChunkKey> keys;
        keys.reserve(volumePoints.size() * 8);
        for (const auto& point : volumePoints) {
            for (const auto& option : options_) {
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(option.presence, point, keys);
                for (const auto& key : keys)
                    requests.emplace_back(&option.presence, key);
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(option.nx, point, keys);
                for (const auto& key : keys)
                    requests.emplace_back(&option.nx, key);
                keys.clear();
                vc::lasagna::appendLasagnaInterpolationChunkKeys(option.ny, point, keys);
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
            throw std::invalid_argument("prediction corner batch is missing channel volumes");
        }
        if (retainStorage) {
            if (samples.size() < referenceDirections.size())
                samples.resize(referenceDirections.size());
        } else {
            samples.resize(referenceDirections.size());
        }
        const int workers = std::clamp(parallelThreads, 1, static_cast<int>(std::max<size_t>(1, referenceDirections.size())));
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
                cv::Vec3f direction = vc::lasagna::interpolateLasagnaCompactAxisCorners(nx, ny, reference);
                if (direction.dot(reference) < 0.0f)
                    direction *= -1.0f;
                const float rawPresence = vc::lasagna::interpolateLasagnaCorners(presence);
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
        std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample>* normals = nullptr) const
    {
        const size_t optionCount = options_.size();
        if (firstVolume + optionCount * 3 > corners.values.size() || corners.fractionsXYZ.size() != referenceDirections.size() ||
            corners.valid.size() != referenceDirections.size()) {
            throw std::invalid_argument("prediction corner batch has inconsistent channel volumes");
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
        const int workers = std::clamp(parallelThreads, 1, static_cast<int>(std::max<size_t>(1, referenceDirections.size())));
        auto materialize = [&](size_t pointIndex) {
            decodePredictionAndNormalCornerPoint(corners, firstVolume, optionCount, pointIndex, referenceDirections[pointIndex], samples[pointIndex], normalFirstVolume, normals != nullptr ? &(*normals)[pointIndex] : nullptr);
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
        const vc::lasagna::LasagnaNormalSampler& normalSampler, std::vector<const vc::lasagna::LasagnaChannelCornerSampler*>& samplers) const
    {
        if (!cornerSamplingAvailable_)
            return false;
        const auto normalSamplers = normalSampler.groupedCornerSamplers();
        if (std::any_of(normalSamplers.begin(), normalSamplers.end(), [](const auto* sampler) { return sampler == nullptr; })) {
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

    static void recordCornerReport(const vc::lasagna::NormalPrefetchReport& cornerReport, double cornerSeconds, int lookaheadDepth, FiberTraceProfile* profile)
    {
        if (profile != nullptr) {
            profile->predictionCornerSeconds += cornerSeconds;
            profile->predictionCornerPrepareSeconds += cornerReport.cornerPrepareSeconds;
            profile->predictionCornerLayoutSeconds += cornerReport.cornerLayoutSeconds;
            profile->predictionCornerPinSeconds += cornerReport.cornerPinSeconds;
            profile->predictionCornerGatherSeconds += cornerReport.cornerGatherSeconds;
            profile->predictionCornerLayoutChunkRuns += cornerReport.cornerLayoutChunkRuns;
            profile->predictionCornerBoundaryPoints += cornerReport.cornerBoundaryPoints;
            profile->predictionCornerDependencies += cornerReport.cornerDependencies;
            profile->cornerPointCount += cornerReport.cornerPointCount;
            profile->cornerUniqueVoxelCubes += cornerReport.cornerUniqueVoxelCubes;
            profile->cornerWorkerTasks += cornerReport.cornerWorkerTasks;
            profile->cornerMaxCandidatesPerCube = std::max(profile->cornerMaxCandidatesPerCube, cornerReport.cornerMaxCandidatesPerCube);
            for (size_t index = 0; index < profile->cornerCubeOccupancyHistogram.size(); ++index) {
                profile->cornerCubeOccupancyHistogram[index] += cornerReport.cornerCubeOccupancyHistogram[index];
            }
            const auto overlap = [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
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
                return std::pair<uint64_t, uint64_t>{shared, static_cast<uint64_t>(a.size() + b.size()) - shared};
            };
            if (lookaheadDepth <= 1) {
                if (!profile->localityPreviousStepDependencies.empty()) {
                    const auto [shared, combined] = overlap(profile->localityPreviousStepDependencies, cornerReport.cornerDependencyIds);
                    profile->stepDependencyShared += shared;
                    profile->stepDependencyUnion += combined;
                }
                profile->localityCurrentDepth1Dependencies = cornerReport.cornerDependencyIds;
            } else if (!profile->localityCurrentDepth1Dependencies.empty()) {
                const auto [shared, combined] = overlap(profile->localityCurrentDepth1Dependencies, cornerReport.cornerDependencyIds);
                profile->depthDependencyShared += shared;
                profile->depthDependencyUnion += combined;
            }
            profile->localityPreviousStepDependencies = cornerReport.cornerDependencyIds;
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
            cornerReport = vc::lasagna::sampleLasagnaChannelCornerBatch(samplers, volumePoints, *cornerScratch, parallelThreads);
        } catch (const std::invalid_argument&) {
            return false;
        }
        recordCornerReport(cornerReport, elapsedSeconds(cornerStart), 0, profile);
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
            cornerReport = vc::lasagna::visitLasagnaChannelCorners(samplers, volumePoints, visitorContext, visitor, parallelThreads, profile != nullptr);
        } catch (const std::invalid_argument&) {
            return false;
        }
        recordCornerReport(cornerReport, elapsedSeconds(cornerStart), lookaheadDepth, profile);
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
            throw std::invalid_argument("fiber prediction batch points and reference directions size mismatch");
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
            sampleBatch(doublePoints, doubleDirections, parallelThreads, samples, profile);
            return;
        }

        const auto batchStart = TraceClock::now();
        const size_t optionCount = options_.size();
        const int workers = std::clamp(parallelThreads, 1, static_cast<int>(volumePoints.size()));
        std::vector<const vc::lasagna::LasagnaChannelCornerSampler*> samplers;
        samplers.reserve(optionCount * 3);
        for (const auto& option : options_) {
            samplers.push_back(option.presenceCorners.get());
            samplers.push_back(option.nxCorners.get());
            samplers.push_back(option.nyCorners.get());
        }
        std::vector<std::vector<vc::lasagna::LasagnaCornerSample>> corners;
        (void)vc::lasagna::sampleLasagnaChannelCornerBatch(samplers, volumePoints, corners, workers);
        materializeGroupedPredictionCorners(corners, 0, referenceDirections, workers, samples);
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
            throw std::invalid_argument("fiber prediction batch points and reference directions size mismatch");
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
                floatPoints.emplace_back(static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]));
            }
            for (const auto& direction : referenceDirections) {
                floatReferences.emplace_back(static_cast<float>(direction[0]), static_cast<float>(direction[1]), static_cast<float>(direction[2]));
            }
            sampleBatch(floatPoints, floatReferences, parallelThreads, samples, profile);
            return;
        }

        const size_t optionCount = options_.size();
        const int workers = std::clamp(parallelThreads, 1, static_cast<int>(volumePoints.size()));

        {
            const auto batchStart = TraceClock::now();
            std::vector<cv::Vec3f> floatPoints;
            floatPoints.reserve(volumePoints.size());
            for (const auto& point : volumePoints) {
                floatPoints.push_back({static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])});
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
                (void)vc::lasagna::sampleLasagnaChannelCornerBatch({option.presenceCorners.get(), option.nxCorners.get(), option.nyCorners.get()}, floatPoints, grouped, workers);
                corners[optionIndex].presence = std::move(grouped[0]);
                corners[optionIndex].nx = std::move(grouped[1]);
                corners[optionIndex].ny = std::move(grouped[2]);
            }

            auto materializeCornerSample = [&](size_t pointIndex) {
                auto& out = samples[pointIndex];
                out.options.clear();
                out.options.reserve(optionCount);
                const cv::Vec3f
                    reference{static_cast<float>(referenceDirections[pointIndex][0]), static_cast<float>(referenceDirections[pointIndex][1]), static_cast<float>(referenceDirections[pointIndex][2])};
                for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                    const auto& optionCorners = corners[optionIndex];
                    const auto& presence = optionCorners.presence[pointIndex];
                    const auto& nx = optionCorners.nx[pointIndex];
                    const auto& ny = optionCorners.ny[pointIndex];
                    if (!presence.valid || !nx.valid || !ny.valid) {
                        out.options.push_back({});
                        continue;
                    }
                    cv::Vec3f direction = vc::lasagna::interpolateLasagnaCompactAxisCorners(nx, ny, reference);
                    if (direction.dot(reference) < 0.0f)
                        direction *= -1.0f;
                    const float rawPresence = vc::lasagna::interpolateLasagnaCorners(presence);
                    out.options.push_back({
                        {direction[0], direction[1], direction[2]},
                        std::clamp(rawPresence / 255.0f, 0.0f, 1.0f),
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
        auto materializeDirect = [&](size_t pointIndex,
                                     std::vector<vc::lasagna::LasagnaLocalChunkResolver>& presenceResolvers,
                                     std::vector<vc::lasagna::LasagnaLocalChunkResolver>& nxResolvers,
                                     std::vector<vc::lasagna::LasagnaLocalChunkResolver>& nyResolvers) {
            auto& out = samples[pointIndex];
            out.options.clear();
            out.options.reserve(optionCount);
            for (size_t optionIndex = 0; optionIndex < optionCount; ++optionIndex) {
                const auto& option = options_[optionIndex];
                auto presenceRequest = vc::lasagna::prepareLasagnaCubeRequest(option.presence, volumePoints[pointIndex]);
                vc::lasagna::LasagnaCubeRequest nxRequest;
                vc::lasagna::LasagnaCubeRequest nyRequest;
                if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                    nxRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(presenceRequest, option.nx);
                    nyRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(presenceRequest, option.ny);
                } else {
                    nxRequest = vc::lasagna::prepareLasagnaCubeRequest(option.nx, volumePoints[pointIndex]);
                    nyRequest = vc::lasagna::prepareLasagnaCubeRequest(option.ny, volumePoints[pointIndex]);
                }
                presenceResolvers[optionIndex].resolve(presenceRequest);
                nxResolvers[optionIndex].resolve(nxRequest);
                nyResolvers[optionIndex].resolve(nyRequest);

                const auto rawPresence = vc::lasagna::sampleLasagnaChannel(option.presence, presenceRequest);
                const auto direction = vc::lasagna::sampleLasagnaCompactAxisTensor(option.nx, option.ny, nxRequest, nyRequest);
                if (!rawPresence.has_value() || !direction.has_value()) {
                    out.options.push_back({});
                    continue;
                }
                out.options.push_back({
                    alignTo(*direction, referenceDirections[pointIndex]),
                    static_cast<float>(clamp01(*rawPresence / 255.0)),
                    true,
                });
            }
        };
        auto makeResolvers = [&](const auto& bindingSelector) {
            std::vector<vc::lasagna::LasagnaLocalChunkResolver> resolvers;
            resolvers.reserve(optionCount);
            for (const auto& option : options_)
                resolvers.emplace_back(bindingSelector(option), *cache_);
            return resolvers;
        };

        if (workers <= 1) {
            auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& { return option.presence; });
            auto nxResolvers = makeResolvers([](const Option& option) -> const auto& { return option.nx; });
            auto nyResolvers = makeResolvers([](const Option& option) -> const auto& { return option.ny; });
            for (size_t pointIndex = 0; pointIndex < volumePoints.size(); ++pointIndex) {
                materializeDirect(pointIndex, presenceResolvers, nxResolvers, nyResolvers);
            }
        } else {
#ifdef _OPENMP
            std::atomic<bool> failed{false};
            std::exception_ptr firstError;
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
#pragma omp parallel num_threads(workers)
            {
                auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& { return option.presence; });
                auto nxResolvers = makeResolvers([](const Option& option) -> const auto& { return option.nx; });
                auto nyResolvers = makeResolvers([](const Option& option) -> const auto& { return option.ny; });
#pragma omp for schedule(static)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                    if (failed.load(std::memory_order_relaxed))
                        continue;
                    try {
                        materializeDirect(static_cast<size_t>(rawIndex), presenceResolvers, nxResolvers, nyResolvers);
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
                    auto presenceResolvers = makeResolvers([](const Option& option) -> const auto& { return option.presence; });
                    auto nxResolvers = makeResolvers([](const Option& option) -> const auto& { return option.nx; });
                    auto nyResolvers = makeResolvers([](const Option& option) -> const auto& { return option.ny; });
                    while (true) {
                        const size_t pointIndex = next.fetch_add(1);
                        if (pointIndex >= volumePoints.size())
                            return;
                        materializeDirect(pointIndex, presenceResolvers, nxResolvers, nyResolvers);
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
                point.presence = vc::lasagna::prepareLasagnaCubeRequest(option.presence, volumePoints[pointIndex]);
                if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                    point.nx = vc::lasagna::cloneLasagnaCubeRequestForBinding(point.presence, option.nx);
                    point.ny = vc::lasagna::cloneLasagnaCubeRequestForBinding(point.presence, option.ny);
                } else {
                    point.nx = vc::lasagna::prepareLasagnaCubeRequest(option.nx, volumePoints[pointIndex]);
                    point.ny = vc::lasagna::prepareLasagnaCubeRequest(option.ny, volumePoints[pointIndex]);
                }
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(point.presence, localPresenceKeys[optionIndex]);
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(point.nx, localNxKeys[optionIndex]);
                vc::lasagna::appendUniqueLasagnaCubeRequestChunkKeys(point.ny, localNyKeys[optionIndex]);
            }
        };

#ifdef _OPENMP
        if (workers > 1) {
            const size_t workerCount = static_cast<size_t>(workers);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> presenceKeysByWorker(workerCount * optionCount);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> nxKeysByWorker(workerCount * optionCount);
            std::vector<std::vector<vc::lasagna::LasagnaChannelChunkKey>> nyKeysByWorker(workerCount * optionCount);
            const size_t reservePerWorker = volumePoints.size() / workerCount + 16;
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
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(presenceKeysByWorker[slotOffset + optionIndex]);
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(nxKeysByWorker[slotOffset + optionIndex]);
                    vc::lasagna::deduplicateLasagnaChunkKeysInPlace(nyKeysByWorker[slotOffset + optionIndex]);
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
                        out[optionIndex]
                            .insert(out[optionIndex].end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
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
                preparePoint(pointIndex, presenceKeys.data(), nxKeys.data(), nyKeys.data());
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
                return cache_
                    ->prefetchResolved(option.presence, *option.presence.array, presenceKeys[optionIndex], readWorkers, chunks[optionIndex].presence);
            }));
            prefetches.push_back(std::async(std::launch::async, [&, optionIndex]() {
                const auto& option = options_[optionIndex];
                return cache_->prefetchResolved(option.nx, *option.nx.array, nxKeys[optionIndex], readWorkers, chunks[optionIndex].nx);
            }));
            prefetches.push_back(std::async(std::launch::async, [&, optionIndex]() {
                const auto& option = options_[optionIndex];
                return cache_->prefetchResolved(option.ny, *option.ny.array, nyKeys[optionIndex], readWorkers, chunks[optionIndex].ny);
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
                vc::lasagna::assignResolvedLasagnaCubeRequestChunks(point.presence, maps.presence);
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
                const auto rawPresence = vc::lasagna::sampleLasagnaChannel(option.presence, point.presence);
                const auto direction = vc::lasagna::sampleLasagnaCompactAxisTensor(option.nx, option.ny, point.nx, point.ny);
                if (!rawPresence.has_value() || !direction.has_value()) {
                    out.options.push_back({});
                    continue;
                }
                out.options.push_back({
                    alignTo(*direction, referenceDirections[pointIndex]),
                    static_cast<float>(clamp01(*rawPresence / 255.0)),
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

    [[nodiscard]] FiberPredictionSample sample(const cv::Vec3d& volumePoint, const cv::Vec3d& referenceDirection) const
    {
        FiberPredictionSample out;
        out.options.reserve(options_.size());
        for (size_t optionIndex = 0; optionIndex < options_.size(); ++optionIndex) {
            const auto& option = options_[optionIndex];
            auto presenceRequest = vc::lasagna::prepareLasagnaCubeRequest(option.presence, volumePoint);
            vc::lasagna::LasagnaCubeRequest nxRequest;
            vc::lasagna::LasagnaCubeRequest nyRequest;
            if (optionGrids_[optionIndex].sharedPresenceNxNy) {
                nxRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(presenceRequest, option.nx);
                nyRequest = vc::lasagna::cloneLasagnaCubeRequestForBinding(presenceRequest, option.ny);
            } else {
                nxRequest = vc::lasagna::prepareLasagnaCubeRequest(option.nx, volumePoint);
                nyRequest = vc::lasagna::prepareLasagnaCubeRequest(option.ny, volumePoint);
            }
            vc::lasagna::LasagnaLocalChunkResolver presenceResolver(option.presence, *cache_);
            vc::lasagna::LasagnaLocalChunkResolver nxResolver(option.nx, *cache_);
            vc::lasagna::LasagnaLocalChunkResolver nyResolver(option.ny, *cache_);
            presenceResolver.resolve(presenceRequest);
            nxResolver.resolve(nxRequest);
            nyResolver.resolve(nyRequest);
            const auto rawPresence = vc::lasagna::sampleLasagnaChannel(option.presence, presenceRequest);
            const auto direction = vc::lasagna::sampleLasagnaCompactAxisTensor(option.nx, option.ny, nxRequest, nyRequest);
            if (!rawPresence.has_value() || !direction.has_value()) {
                out.options.push_back({});
                continue;
            }
            out.options.push_back({
                alignTo(*direction, referenceDirection),
                static_cast<float>(clamp01(*rawPresence / 255.0)),
                true,
            });
        }
        return out;
    }

    [[nodiscard]] size_t optionCount() const noexcept { return options_.size(); }

    [[nodiscard]] FiberPredictionGridInfo storedGridInfo() const
    {
        if (!canonicalOptionIndex_.has_value()) {
            throw std::runtime_error("anchor extraction requires canonical presence/nx/ny channels");
        }
        if (!sourceScaleExplicit_) {
            throw std::runtime_error("anchor extraction requires an explicit numeric source_to_base");
        }
        if (!(canonicalGridInfo_.predictionToBaseScale > 0.0) || !std::isfinite(canonicalGridInfo_.predictionToBaseScale)) {
            throw std::runtime_error("anchor extraction requires a positive finite prediction-to-base scale");
        }
        return canonicalGridInfo_;
    }

    [[nodiscard]] FiberPresenceChunkScanReport scanStoredPresenceChunks(int parallelThreads) const
    {
        (void)storedGridInfo();
        const auto& binding = options_[*canonicalOptionIndex_].presence;
        FiberPresenceChunkScanReport report;
        report.shapeZYX = binding.shapeZYX;
        report.chunksZYX = binding.chunksZYX;
        size_t total = 1;
        for (size_t axis = 0; axis < 3; ++axis) {
            report.chunkGridShapeZYX[axis] = (binding.shapeZYX[axis] + binding.chunksZYX[axis] - 1) / binding.chunksZYX[axis];
            if (report.chunkGridShapeZYX[axis] > std::numeric_limits<size_t>::max() / total) {
                throw std::overflow_error("fiber presence chunk grid size overflows size_t");
            }
            total *= report.chunkGridShapeZYX[axis];
        }

        struct Partial {
            size_t missing = 0;
            size_t empty = 0;
            std::vector<std::array<size_t, 3>> nonempty;
        };
        const size_t workerCount = std::min<size_t>(total, static_cast<size_t>(std::max(1, parallelThreads)));
        std::vector<std::future<Partial>> futures;
        futures.reserve(workerCount);
        for (size_t worker = 0; worker < workerCount; ++worker) {
            const size_t begin = total * worker / workerCount;
            const size_t end = total * (worker + 1) / workerCount;
            futures.push_back(std::async(std::launch::async, [&, begin, end]() {
                Partial partial;
                partial.nonempty.reserve((end - begin) / 4);
                const size_t yx = report.chunkGridShapeZYX[1] * report.chunkGridShapeZYX[2];
                for (size_t linear = begin; linear < end; ++linear) {
                    const size_t z = linear / yx;
                    const size_t remainder = linear % yx;
                    const size_t y = remainder / report.chunkGridShapeZYX[2];
                    const size_t x = remainder % report.chunkGridShapeZYX[2];
                    const std::array<size_t, 3> key{z, y, x};
                    const auto bytes = binding.array->read_chunk(key);
                    if (!bytes.has_value()) {
                        ++partial.missing;
                        continue;
                    }
                    const bool nonempty = std::any_of(bytes->begin(), bytes->end(), [](std::byte value) { return value != std::byte{0}; });
                    if (nonempty)
                        partial.nonempty.push_back(key);
                    else
                        ++partial.empty;
                }
                return partial;
            }));
        }
        for (auto& future : futures) {
            auto partial = future.get();
            report.missingChunks += partial.missing;
            report.emptyChunks += partial.empty;
            report.nonemptyChunksZYX.insert(
                report.nonemptyChunksZYX.end(), std::make_move_iterator(partial.nonempty.begin()), std::make_move_iterator(partial.nonempty.end()));
        }
        std::sort(report.nonemptyChunksZYX.begin(), report.nonemptyChunksZYX.end());
        return report;
    }

    [[nodiscard]] static cv::Vec3d storedGridPoint(const Option& option, const std::array<size_t, 3>& zyx)
    {
        if (zyx[0] >= option.presence.shapeZYX[0] || zyx[1] >= option.presence.shapeZYX[1] || zyx[2] >= option.presence.shapeZYX[2]) {
            throw std::out_of_range("fiber stored-grid sample index is outside the prediction volume");
        }
        return {
            static_cast<double>(zyx[2]) * option.presence.spacing,
            static_cast<double>(zyx[1]) * option.presence.spacing,
            static_cast<double>(zyx[0]) * option.presence.spacing,
        };
    }

    template <typename ResolverFactory, typename SampleOne>
    static void sampleStoredIndices(size_t count, int parallelThreads, ResolverFactory&& resolverFactory, SampleOne&& sampleOne)
    {
        const int workers =
            std::clamp(parallelThreads > 0 ? parallelThreads : 1, 1, static_cast<int>(std::min<size_t>(std::max<size_t>(1, count), static_cast<size_t>(std::numeric_limits<int>::max()))));
        if (count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            throw std::length_error("fiber stored-grid sample batch is too large");
#ifdef _OPENMP
        if (workers > 1) {
            std::exception_ptr firstError;
            std::atomic<bool> failed{false};
            const auto signedCount = static_cast<std::ptrdiff_t>(count);
#pragma omp parallel num_threads(workers)
            {
                auto resolvers = resolverFactory();
#pragma omp for schedule(static)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < signedCount; ++rawIndex) {
                    if (failed.load(std::memory_order_relaxed))
                        continue;
                    try {
                        sampleOne(static_cast<size_t>(rawIndex), resolvers);
                    } catch (...) {
                        bool expected = false;
                        if (failed.compare_exchange_strong(expected, true)) {
#pragma omp critical(fiber_stored_grid_error)
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
            return;
        }
#endif
        auto resolvers = resolverFactory();
        for (size_t index = 0; index < count; ++index)
            sampleOne(index, resolvers);
    }

    void sampleStoredGridBatch(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPredictionSample>& samples) const
    {
        (void)storedGridInfo();
        const auto& option = options_[*canonicalOptionIndex_];
        samples.assign(indicesZYX.size(), {});
        const auto sampleOne = [&](size_t index, auto& resolvers) {
            const auto& zyx = indicesZYX[index];
            const cv::Vec3d point = storedGridPoint(option, zyx);
            auto presenceRequest = vc::lasagna::prepareLasagnaCubeRequest(option.presence, point);
            auto nxRequest = vc::lasagna::prepareLasagnaCubeRequest(option.nx, point);
            auto nyRequest = vc::lasagna::prepareLasagnaCubeRequest(option.ny, point);
            resolvers[0].resolve(presenceRequest);
            resolvers[1].resolve(nxRequest);
            resolvers[2].resolve(nyRequest);
            const auto rawPresence = vc::lasagna::sampleLasagnaChannel(option.presence, presenceRequest);
            const auto direction = vc::lasagna::sampleLasagnaCompactAxisTensor(option.nx, option.ny, nxRequest, nyRequest);
            if (!rawPresence.has_value())
                return;
            samples[index].presence = clamp01(*rawPresence / 255.0);
            samples[index].presenceValid = true;
            if (!direction.has_value())
                return;
            const double norm2 = direction->dot(*direction);
            if (!(norm2 > kEpsilon * kEpsilon) || !std::isfinite(norm2))
                return;
            samples[index].direction = *direction / std::sqrt(norm2);
            samples[index].valid = true;
        };
        const auto makeResolvers = [&]() {
            return std::array{
                vc::lasagna::LasagnaLocalChunkResolver(option.presence, *cache_),
                vc::lasagna::LasagnaLocalChunkResolver(option.nx, *cache_),
                vc::lasagna::LasagnaLocalChunkResolver(option.ny, *cache_),
            };
        };
        sampleStoredIndices(indicesZYX.size(), parallelThreads, makeResolvers, sampleOne);
    }

    void sampleStoredPresenceBatch(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPresenceSample>& samples) const
    {
        (void)storedGridInfo();
        const auto& option = options_[*canonicalOptionIndex_];
        samples.assign(indicesZYX.size(), {});
        const auto sampleOne = [&](size_t index, auto& presenceResolver) {
            const cv::Vec3d point = storedGridPoint(option, indicesZYX[index]);
            auto request = vc::lasagna::prepareLasagnaCubeRequest(option.presence, point);
            presenceResolver.resolve(request);
            const auto rawPresence = vc::lasagna::sampleLasagnaChannel(option.presence, request);
            if (rawPresence.has_value())
                samples[index] = {clamp01(*rawPresence / 255.0), true};
        };
        const auto makeResolver = [&]() { return vc::lasagna::LasagnaLocalChunkResolver(option.presence, *cache_); };
        sampleStoredIndices(indicesZYX.size(), parallelThreads, makeResolver, sampleOne);
    }

private:
    std::vector<Option> options_;
    std::vector<OptionSamplingGrid> optionGrids_;
    std::shared_ptr<vc::lasagna::LasagnaChannelChunkCache> cache_;
    std::shared_ptr<vc::render::DecodedChunkCacheBudget> cornerBudget_;
    bool cornerSamplingAvailable_ = true;
    bool sourceScaleExplicit_ = false;
    std::optional<size_t> canonicalOptionIndex_;
    FiberPredictionGridInfo canonicalGridInfo_;
};

FiberPredictionField::FiberPredictionField(const vc::lasagna::LasagnaDataset& dataset, size_t maxCachedBytes, FiberPredictionFieldBindingMode bindingMode)
    : impl_(std::make_unique<Impl>(dataset, maxCachedBytes, bindingMode))
{
}

FiberPredictionField::~FiberPredictionField() = default;

vc::lasagna::NormalPrefetchReport FiberPredictionField::prefetchSamples(const std::vector<cv::Vec3d>& volumePoints) const
{
    return impl_->prefetchSamples(volumePoints);
}

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3d>& volumePoints, const std::vector<cv::Vec3d>& referenceDirections, int parallelThreads, std::vector<FiberPredictionSample>& samples) const
{
    impl_->sampleBatch(volumePoints, referenceDirections, parallelThreads, samples, nullptr);
}

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    const std::vector<cv::Vec3d>& referenceDirections,
    int parallelThreads,
    std::vector<FiberPredictionSample>& samples,
    FiberTraceProfile* profile) const
{
    impl_->sampleBatch(volumePoints, referenceDirections, parallelThreads, samples, profile);
}

void FiberPredictionField::sampleBatch(
    const std::vector<cv::Vec3f>& volumePoints,
    const std::vector<cv::Vec3f>& referenceDirections,
    int parallelThreads,
    std::vector<FiberPredictionSample>& samples,
    FiberTraceProfile* profile) const
{
    impl_->sampleBatch(volumePoints, referenceDirections, parallelThreads, samples, profile);
}

bool FiberPredictionField::sampleCornerBatchWithNormals(
    const vc::lasagna::LasagnaNormalSampler& normalSampler,
    const std::vector<cv::Vec3f>& volumePoints,
    int parallelThreads,
    vc::lasagna::LasagnaCornerBatch* cornerScratch,
    FiberTraceProfile* profile) const
{
    return impl_->sampleCornerBatchWithNormals(normalSampler, volumePoints, parallelThreads, cornerScratch, profile);
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
    return impl_->visitCornerBatchWithNormals(normalSampler, volumePoints, parallelThreads, visitorContext, visitor, lookaheadDepth, profile);
}

FiberPredictionSample FiberPredictionField::sample(const cv::Vec3d& volumePoint, const cv::Vec3d& referenceDirection) const
{
    return impl_->sample(volumePoint, referenceDirection);
}

FiberPredictionGridInfo FiberPredictionField::storedGridInfo() const
{
    return impl_->storedGridInfo();
}

FiberPresenceChunkScanReport FiberPredictionField::scanStoredPresenceChunks(int parallelThreads) const
{
    return impl_->scanStoredPresenceChunks(parallelThreads);
}

void FiberPredictionField::sampleStoredGridBatch(
    const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPredictionSample>& samples) const
{
    impl_->sampleStoredGridBatch(indicesZYX, parallelThreads, samples);
}

void FiberPredictionField::sampleStoredPresenceBatch(
    const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPresenceSample>& samples) const
{
    impl_->sampleStoredPresenceBatch(indicesZYX, parallelThreads, samples);
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
    const auto parsed = parseVc3dFiberJson(root, path.string());

    FiberInput fiber;
    fiber.path = path;
    fiber.linePointsXyzBase = parsed.linePoints;
    fiber.controlPointsXyzBase = parsed.controlPoints;
    if (fiber.linePointsXyzBase.size() < 2) {
        throw std::runtime_error("fiber JSON needs at least two line_points: " + path.string());
    }
    if (fiber.controlPointsXyzBase.size() < 2) {
        throw std::runtime_error("fiber JSON needs at least two control_points: " + path.string());
    }
    fiber.controlPointLineIndices.reserve(fiber.controlPointsXyzBase.size());
    for (size_t index = 0; index < fiber.controlPointsXyzBase.size(); ++index) {
        fiber.controlPointLineIndices.push_back(exactLineIndexForControlPoint(fiber.linePointsXyzBase, fiber.controlPointsXyzBase[index], index, path));
    }
    for (size_t index = 1; index < fiber.controlPointLineIndices.size(); ++index) {
        if (fiber.controlPointLineIndices[index] <= fiber.controlPointLineIndices[index - 1]) {
            throw std::runtime_error(
                "fiber JSON control_points are not strictly increasing along "
                "line_points: " +
                path.string());
        }
    }
    return fiber;
}

cv::Vec3d referenceTangentToward(const std::vector<cv::Vec3d>& line, size_t startIndex, size_t targetIndex)
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

[[nodiscard]] std::vector<FiberTraceTargetPlane> targetLocalPlanes(
    const FiberPredictionSource& predictions, const std::vector<cv::Vec3d>& referenceLine, size_t targetLineIndex, size_t sourceLineIndex, const cv::Vec3d& targetPoint)
{
    if (targetLineIndex >= referenceLine.size() || sourceLineIndex >= referenceLine.size()) {
        throw std::invalid_argument("fiber target-local plane line index is out of range");
    }
    std::vector<FiberTraceTargetPlane> planes;
    planes.reserve(kMaxTargetPlanes);
    const cv::Vec3d lineCenter = referenceLine[targetLineIndex];
    const auto appendNeighbor = [&](std::string name, size_t neighborIndex) {
        const cv::Vec3d normal = normalizedOrZero(referenceLine[neighborIndex] - lineCenter);
        if (length(normal) <= kEpsilon) {
            throw std::invalid_argument("fiber target-local plane has a degenerate " + name + " normal");
        }
        planes.push_back({std::move(name), targetPoint, normal});
    };
    if (targetLineIndex + 1 < referenceLine.size())
        appendNeighbor("line_next", targetLineIndex + 1);
    if (targetLineIndex > 0)
        appendNeighbor("line_prev", targetLineIndex - 1);
    if (planes.empty()) {
        throw std::invalid_argument("fiber target control point has no line-neighbor target planes");
    }

    const cv::Vec3d targetTangent = referenceTangentToward(referenceLine, targetLineIndex, sourceLineIndex);
    const ScoredDirection inferred = bestAlignedPrediction(predictions, targetPoint, toTraceVec(targetTangent), false);
    if (!inferred.valid) {
        throw std::invalid_argument("fiber target control point has no valid inferred-direction plane");
    }
    planes.push_back({
        "inferred_direction",
        targetPoint,
        toVec3d(traceNormalizedOrZero(inferred.direction)),
    });
    return planes;
}

FiberTraceOneWayResult traceFiberOneWay(
    const FiberPredictionSource& predictions, const FiberTraceOneWayRequest& request, const vc::lasagna::NormalSampler* normalSampler, const FiberTraceProgressCallback& progress)
{
    if (!finitePoint(request.startPoint) || !finitePoint(request.targetPoint)) {
        throw std::invalid_argument("fiber trace one-way request has non-finite endpoint");
    }
    if (length(request.targetPoint - request.startPoint) <= kEpsilon) {
        throw std::invalid_argument("fiber trace one-way request endpoints must differ");
    }
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);
    return traceOneWayCore(predictions, request, normalSampler, progress, "trace");
}

FiberTraceOneWayResult traceFiberExtrapolation(
    const FiberPredictionSource& predictions,
    const cv::Vec3d& startPoint,
    const cv::Vec3d& outwardDirection,
    double distanceVoxels,
    const FiberTraceConfig& config,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress)
{
    if (!finitePoint(startPoint) || !finitePoint(outwardDirection)) {
        throw std::invalid_argument("fiber extrapolation request contains a non-finite point or direction");
    }
    if (!(distanceVoxels > 0.0) || !std::isfinite(distanceVoxels)) {
        throw std::invalid_argument("fiber extrapolation distance must be finite and positive");
    }
    const cv::Vec3d direction = normalizedOrZero(outwardDirection);
    if (length(direction) <= kEpsilon) {
        throw std::invalid_argument("fiber extrapolation direction must be non-degenerate");
    }

    FiberTraceOneWayRequest request;
    request.startPoint = startPoint;
    request.targetPoint = startPoint + direction * distanceVoxels;
    request.initialDirection = direction;
    request.budgetSpanVoxels = distanceVoxels;
    request.config = config;
    return traceOneWayCore(predictions, request, normalSampler, progress, "extrapolation", distanceVoxels);
}

FiberTraceSegmentResult traceFiberSegment(
    const FiberPredictionSource& predictions, const FiberTraceSegmentRequest& request, const vc::lasagna::NormalSampler* normalSampler, const FiberTraceProgressCallback& progress)
{
    if (request.referenceLine.empty())
        throw std::invalid_argument("fiber trace request has no reference line");
    if (request.startIndex >= request.referenceLine.size() || request.targetIndex >= request.referenceLine.size()) {
        throw std::invalid_argument("fiber trace request control-point index is out of range");
    }
    if (request.startIndex == request.targetIndex)
        throw std::invalid_argument("fiber trace request start and target indices must differ");
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);

    FiberTraceSegmentResult result;
    const cv::Vec3d start = request.referenceLine[request.startIndex];
    const cv::Vec3d target = request.referenceLine[request.targetIndex];
    const double span = length(target - start);
    FiberTraceOneWayRequest forwardOneWay;
    forwardOneWay.startPoint = start;
    forwardOneWay.targetPoint = target;
    forwardOneWay.initialDirection = referenceTangentToward(request.referenceLine, request.startIndex, request.targetIndex);
    forwardOneWay.targetPlanes = targetLocalPlanes(predictions, request.referenceLine, request.targetIndex, request.startIndex, target);
    forwardOneWay.targetPlaneAcceptThresholdVoxels = request.config.endpointAcceptThresholdBaseVoxels / request.config.traceToBaseScale;
    forwardOneWay.snapTraceToSelectedCrossing = false;
    forwardOneWay.budgetSpanVoxels = span;
    forwardOneWay.config = request.config;
    FiberTraceOneWayRequest reverseOneWay;
    reverseOneWay.startPoint = target;
    reverseOneWay.targetPoint = start;
    reverseOneWay.initialDirection = referenceTangentToward(request.referenceLine, request.targetIndex, request.startIndex);
    reverseOneWay.targetPlanes = targetLocalPlanes(predictions, request.referenceLine, request.startIndex, request.targetIndex, start);
    reverseOneWay.targetPlaneAcceptThresholdVoxels = request.config.endpointAcceptThresholdBaseVoxels / request.config.traceToBaseScale;
    reverseOneWay.snapTraceToSelectedCrossing = false;
    reverseOneWay.budgetSpanVoxels = span;
    reverseOneWay.config = request.config;

    result.forward = traceOneWayCore(predictions, forwardOneWay, normalSampler, progress, "forward");
    result.reverse = traceOneWayCore(predictions, reverseOneWay, normalSampler, progress, "reverse");

    const TraceMeetingFusion fusion = fuseTraceMeetings(result.forward, result.reverse, request.config);
    result.fusedLine = fusion.fusedLine;
    if (!result.fusedLine.empty()) {
        result.fusedLine.front() = request.referenceLine[request.startIndex];
        result.fusedLine.back() = request.referenceLine[request.targetIndex];
    }

    result.forwardEndpointErrorTraceVoxels = result.forward.selectedTargetPlaneErrorVoxels;
    result.reverseEndpointErrorTraceVoxels = result.reverse.selectedTargetPlaneErrorVoxels;
    result.maxEndpointErrorTraceVoxels = std::max(result.forwardEndpointErrorTraceVoxels, result.reverseEndpointErrorTraceVoxels);
    const FiberTraceCoordinateAdapter coordinates(request.config.traceToBaseScale);
    result.maxEndpointErrorBaseVoxels = coordinates.traceDistanceToBase(result.maxEndpointErrorTraceVoxels);
    if (request.config.baseVoxelSizeUm.has_value()) {
        if (std::isfinite(result.maxEndpointErrorBaseVoxels)) {
            result.maxEndpointErrorUm = result.maxEndpointErrorBaseVoxels * *request.config.baseVoxelSizeUm;
        }
    }
    result.meetingErrorTraceVoxels = fusion.errorTraceVoxels;
    result.meetingErrorBaseVoxels = coordinates.traceDistanceToBase(fusion.errorTraceVoxels);
    result.meetingErrorRatio = fusion.errorRatio;
    result.meetingTraceLengthTraceVoxels = fusion.traceLengthTraceVoxels;
    if (request.config.baseVoxelSizeUm.has_value() && std::isfinite(result.meetingErrorBaseVoxels)) {
        result.meetingErrorUm = result.meetingErrorBaseVoxels * *request.config.baseVoxelSizeUm;
    }
    result.meetingSource = fusion.source;
    result.reason = fusion.reason;
    result.detail = fusion.detail;
    result.accepted = result.reason == "ok";
    return result;
}

FiberTraceWholeFiberResult traceWholeFiberMetric(
    const FiberPredictionSource& predictions,
    const FiberTraceWholeFiberMetricRequest& request,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceWholeFiberProgressCallback& progress)
{
    if (!(request.workingToBaseScale > 0.0) || !std::isfinite(request.workingToBaseScale)) {
        throw std::invalid_argument("working-to-base scale must be positive");
    }
    if (!(request.errorThresholdBaseVoxels >= 0.0) || !std::isfinite(request.errorThresholdBaseVoxels)) {
        throw std::invalid_argument("error threshold must be finite and non-negative");
    }
    if (request.fiber.controlPointsXyzBase.size() < 2) {
        throw std::invalid_argument("whole-fiber metric needs at least two control points");
    }
    if (request.fiber.linePointsXyzBase.size() < 2) {
        throw std::invalid_argument("whole-fiber metric needs at least two line points");
    }
    if (request.fiber.controlPointLineIndices.size() != request.fiber.controlPointsXyzBase.size()) {
        throw std::invalid_argument("whole-fiber metric control-point line-index count mismatch");
    }
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, normalSampler);

    const auto lineWorking = scaledPoints(request.fiber.linePointsXyzBase, request.workingToBaseScale);
    const auto cpWorking = scaledPoints(request.fiber.controlPointsXyzBase, request.workingToBaseScale);
    const auto lineLengths = arclengths(lineWorking);

    std::vector<double> cpArcs;
    cpArcs.reserve(request.fiber.controlPointLineIndices.size());
    for (const size_t lineIndex : request.fiber.controlPointLineIndices) {
        if (lineIndex >= lineLengths.size()) {
            throw std::invalid_argument("whole-fiber metric control-point line index out of range");
        }
        cpArcs.push_back(lineLengths[lineIndex]);
    }

    FiberTraceWholeFiberResult result;
    result.segmentCount = static_cast<int>(cpWorking.size() - 1);
    result.referenceLengthVoxels = cpArcs.back() - cpArcs.front();
    if (request.voxelSizeUm.has_value() && *request.voxelSizeUm > 0.0) {
        result.referenceLengthMeters = referenceLengthMeters(result.referenceLengthVoxels, request.workingToBaseScale, *request.voxelSizeUm);
    }

    auto updateMetricFields = [&]() {
        if (result.referenceLengthVoxels > kEpsilon) {
            result.restartsPerKvx = static_cast<double>(result.restartCount) * 1000.0 / result.referenceLengthVoxels;
        }
        if (result.referenceLengthMeters.has_value() && *result.referenceLengthMeters > 0.0) {
            result.restartsPerMeter = static_cast<double>(result.restartCount) / *result.referenceLengthMeters;
        }
    };

    auto emitProgress = [&](int completed, int currentSegment, std::string status, const FiberTraceProgress* traceEvent = nullptr) {
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
    cv::Vec3d currentDirection =
        referenceTangentToward(lineWorking, request.fiber.controlPointLineIndices[0], request.fiber.controlPointLineIndices[1]);
    result.stitchedTrace.push_back(currentPoint);

    emitProgress(0, 1, "start");
    for (size_t cpIndex = 0; cpIndex + 1 < cpWorking.size(); ++cpIndex) {
        const size_t targetCpIndex = cpIndex + 1;
        const cv::Vec3d target = cpWorking[targetCpIndex];
        const cv::Vec3d referenceStart = cpWorking[cpIndex];
        const double budgetSpan = length(target - referenceStart);

        FiberTraceOneWayRequest oneWay;
        oneWay.startPoint = currentPoint;
        oneWay.targetPoint = target;
        oneWay.initialDirection = currentDirection;
        oneWay.targetPlanes =
            targetLocalPlanes(predictions, lineWorking, request.fiber.controlPointLineIndices[targetCpIndex], request.fiber.controlPointLineIndices[cpIndex], target);
        oneWay.targetPlaneAcceptThresholdVoxels = request.errorThresholdBaseVoxels / request.workingToBaseScale;
        oneWay.snapTraceToSelectedCrossing = false;
        oneWay.budgetSpanVoxels = budgetSpan;
        oneWay.config = request.config;

        FiberTraceWholeFiberSegmentResult segment;
        segment.startControlPointIndex = cpIndex;
        segment.targetControlPointIndex = targetCpIndex;
        segment.referenceArcDistanceVoxels = cpArcs[targetCpIndex] - cpArcs[cpIndex];

        const auto segmentProgress = [&](const FiberTraceProgress& traceEvent) {
            emitProgress(static_cast<int>(cpIndex), static_cast<int>(targetCpIndex), "tracing", &traceEvent);
        };
        segment.trace = traceOneWayCore(predictions, oneWay, normalSampler, segmentProgress, "fiber");

        const FiberTraceCoordinateAdapter coordinates(request.workingToBaseScale);
        const auto setSegmentOutcome = [&](FiberTraceOneWayResult trace) {
            segment.trace = std::move(trace);
            segment.inPlaneErrorTraceVoxels = segment.trace.selectedTargetPlaneErrorVoxels;
            segment.inPlaneErrorBaseVoxels = coordinates.traceDistanceToBase(segment.inPlaneErrorTraceVoxels);
            segment.success = segment.trace.reachedTargetPlane && segment.inPlaneErrorBaseVoxels <= request.errorThresholdBaseVoxels;
        };
        setSegmentOutcome(std::move(segment.trace));
        if (shouldRetryLookahead(
                request.config.lazyLookahead, request.config.lookaheadParentCap, request.config.lookaheadRetryParentCap, segment.success)) {
            ++result.lookaheadRetryCount;
            FiberTraceOneWayRequest retry = oneWay;
            retry.config.lookaheadParentCap = request.config.lookaheadRetryParentCap;
            FiberTraceOneWayResult retryTrace = traceOneWayCore(predictions, retry, normalSampler, segmentProgress, "fiber_retry");
            const bool retrySuccess = retryTrace.reachedTargetPlane;
            if (retrySuccess) {
                ++result.lookaheadRetryRecoveredCount;
                setSegmentOutcome(std::move(retryTrace));
            }
        }
        segment.restart = !segment.success;
        if (segment.success) {
            segment.reason = "ok";
            currentPoint = segment.trace.points.empty() ? target : segment.trace.points.back();
            currentDirection = terminalTraceDirection(segment.trace.points, currentDirection);
        } else {
            ++result.restartCount;
            segment.reason = segment.trace.reachedTargetPlane ? "in_plane_error" : segment.trace.reason;
            currentPoint = target;
            if (targetCpIndex + 1 < cpWorking.size()) {
                currentDirection =
                    referenceTangentToward(lineWorking, request.fiber.controlPointLineIndices[targetCpIndex], request.fiber.controlPointLineIndices[targetCpIndex + 1]);
            }
        }

        if (!segment.trace.points.empty()) {
            for (const auto& point : segment.trace.points) {
                if (result.stitchedTrace.empty() || length(result.stitchedTrace.back() - point) > kEpsilon) {
                    result.stitchedTrace.push_back(point);
                }
            }
        }
        if (result.stitchedTrace.empty() || length(result.stitchedTrace.back() - currentPoint) > kEpsilon) {
            result.stitchedTrace.push_back(currentPoint);
        }

        result.segments.push_back(std::move(segment));
        const auto& saved = result.segments.back();
        emitProgress(static_cast<int>(targetCpIndex), static_cast<int>(targetCpIndex), saved.restart ? "restart:" + saved.reason : "ok");
    }
    updateMetricFields();
    emitProgress(result.segmentCount, result.segmentCount, "done");
    return result;
}

FiberReplayTraceResult traceFiberReplay(
    const FiberPredictionSource& predictions,
    const FiberReplayTraceRequest& request,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberTraceProgressCallback& progress,
    const FiberReplayFailureCallback& failureCallback)
{
    if (!(request.traceToBaseScale > 0.0) || !std::isfinite(request.traceToBaseScale)) {
        throw std::invalid_argument("fiber replay trace-to-base scale must be finite and positive");
    }
    if (!(request.errorThresholdBaseVoxels >= 0.0) || !std::isfinite(request.errorThresholdBaseVoxels)) {
        throw std::invalid_argument("fiber replay failure threshold must be finite and non-negative");
    }
    if (!(normalWorkingToBaseScale > 0.0) ||
        !std::isfinite(normalWorkingToBaseScale)) {
        throw std::invalid_argument(
            "fiber replay normal working-to-base scale must be finite and positive");
    }
    if (!(request.matchRefineSteps >= 0.0) || !std::isfinite(request.matchRefineSteps)) {
        throw std::invalid_argument("fiber replay match refinement must be finite and non-negative");
    }
    if (request.config.beamWidth != 1 || request.config.beamLookaheadSteps != 1) {
        throw std::invalid_argument("fiber replay requires greedy tracing with beam width and lookahead equal to one");
    }
    validateTraceConfig(request.config);
    requireNormalSamplerForNormalAwareSmoothness(request.config, &normalSampler);
    if (request.startControlPointIndex >= request.fiber.controlPointsXyzBase.size() ||
        request.startControlPointIndex >= request.fiber.controlPointLineIndices.size()) {
        throw std::invalid_argument("fiber replay start control point is out of range");
    }
    if (request.startControlPointIndex + 1 >= request.fiber.controlPointsXyzBase.size())
        throw std::invalid_argument("fiber replay start control point has no following control point");
    const size_t startLineIndex = request.fiber.controlPointLineIndices[request.startControlPointIndex];
    const size_t nextLineIndex = request.fiber.controlPointLineIndices[request.startControlPointIndex + 1];
    if (startLineIndex >= request.fiber.linePointsXyzBase.size() || nextLineIndex >= request.fiber.linePointsXyzBase.size() ||
        nextLineIndex <= startLineIndex) {
        throw std::invalid_argument("fiber replay requires strictly forward control-point line indices");
    }

    const auto reference = makePolylineArcGeometry(request.fiber.linePointsXyzBase);
    const double startArcBase = reference.vertexArcs[startLineIndex];
    if (request.referenceEndArcBase.has_value() &&
        (!std::isfinite(*request.referenceEndArcBase) ||
         *request.referenceEndArcBase > reference.length() + kEpsilon)) {
        throw std::invalid_argument(
            "fiber replay reference end must be finite and within the reference");
    }
    const double endArcBase = request.referenceEndArcBase.has_value()
        ? *request.referenceEndArcBase
        : reference.length();
    const double remainingArcBase = endArcBase - startArcBase;
    if (!(remainingArcBase > kEpsilon))
        throw std::invalid_argument("fiber replay reference has no usable forward extent");
    const FiberTraceCoordinateAdapter coordinates(request.traceToBaseScale);
    const double nominalStepBase = coordinates.traceDistanceToBase(request.config.stepVoxels);
    if (!(nominalStepBase > kEpsilon) || !std::isfinite(nominalStepBase))
        throw std::invalid_argument("fiber replay nominal step must be finite and positive");

    FiberReplayTraceResult result;
    result.referenceBeginArcBase = startArcBase;
    result.referenceEndArcBase = endArcBase;
    result.completedReferenceArcBase = startArcBase;
    const size_t maximumSegments = static_cast<size_t>(std::ceil(remainingArcBase / nominalStepBase)) + 1;

    double segmentStartArc = startArcBase;
    for (size_t iteration = 0; iteration < maximumSegments &&
         segmentStartArc < endArcBase - kEpsilon; ++iteration) {
        FiberReplayTraceSegment segment;
        segment.startReferenceArcBase = segmentStartArc;
        segment.endReferenceArcBase = segmentStartArc;
        const auto start = samplePolylineArc(reference, segmentStartArc);
        segment.tracePointsBase.push_back(start.point);
        segment.cumulativeLosses.push_back(0.0);

        double previousArcBase = segmentStartArc;
        bool referenceExhausted = false;
        bool distanceFailed = false;
        const auto observe = [&](const cv::Vec3d& pointTrace, double cumulativeLoss, int /*step*/) {
            if (previousArcBase >= endArcBase - kEpsilon) {
                referenceExhausted = true;
                return false;
            }
            const cv::Vec3d pointBase = coordinates.traceToBase(pointTrace);
            const auto forwardMatch = matchForwardPolylinePoint(
                reference, pointBase, previousArcBase, nominalStepBase,
                request.matchRefineSteps, endArcBase);
            if (!(forwardMatch.searchEndArc > previousArcBase + kEpsilon)) {
                referenceExhausted = true;
                return false;
            }
            const auto& match = forwardMatch.projection;
            const auto thresholdMeasurement = measureFiberReplayThreshold(
                pointBase, match.point, normalSampler,
                normalWorkingToBaseScale,
                request.errorThresholdBaseVoxels);
            segment.tracePointsBase.push_back(pointBase);
            segment.cumulativeLosses.push_back(cumulativeLoss);
            segment.matches.push_back({
                segment.tracePointsBase.size() - 1,
                forwardMatch.predictedArc,
                match.arc,
                match.point,
                previousArcBase,
                forwardMatch.searchEndArc,
                thresholdMeasurement,
            });
            previousArcBase = match.arc;
            segment.endReferenceArcBase = match.arc;
            distanceFailed = fiberReplayThresholdExceeded(
                thresholdMeasurement, request.errorThresholdBaseVoxels);
            if (previousArcBase >= endArcBase - kEpsilon)
                referenceExhausted = true;
            return !distanceFailed && !referenceExhausted;
        };

        FiberTraceOneWayRequest oneWay;
        oneWay.startPoint = coordinates.baseToTrace(start.point);
        oneWay.initialDirection = start.tangent;
        oneWay.targetPoint = coordinates.baseToTrace(
            samplePolylineArc(reference, endArcBase).point);
        oneWay.budgetSpanVoxels =
            (endArcBase - segmentStartArc) / request.traceToBaseScale;
        oneWay.config = request.config;
        const int maximumSteps =
            std::max(1, static_cast<int>(std::ceil(
                request.config.maxStepFactor *
                (endArcBase - segmentStartArc) / nominalStepBase)) + 1);

        std::string terminationReason;
        try {
            FiberTraceProgressCallback segmentProgress;
            if (progress) {
                const size_t segmentIndex = result.segments.size();
                segmentProgress = [&](const FiberTraceProgress& event) {
                    auto replayEvent = event;
                    replayEvent.replaySegmentIndex = segmentIndex;
                    replayEvent.referenceArcBase = previousArcBase;
                    replayEvent.referenceArcFraction = std::clamp(
                        (previousArcBase - startArcBase) / remainingArcBase,
                        0.0, 1.0);
                    progress(replayEvent);
                };
            }
            const auto native =
                traceOneWayCore(predictions, oneWay, &normalSampler,
                    segmentProgress, "replay",
                    static_cast<double>(maximumSteps) *
                        request.config.stepVoxels,
                    maximumSteps, observe);
            terminationReason = native.reason;
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()) != "fiber trace start point has no valid prediction direction") {
                throw;
            }
            terminationReason = "invalid_initial_prediction";
        }

        if (referenceExhausted && !distanceFailed) {
            segment.endReferenceArcBase = endArcBase;
            segment.terminationReason = "reference_end";
            result.segments.push_back(std::move(segment));
            result.completedReferenceArcBase = endArcBase;
            break;
        }

        FiberReplayFailure replayFailure;
        replayFailure.index = result.failures.size();
        replayFailure.segmentIndex = result.segments.size();
        replayFailure.reason = distanceFailed ? "distance_above_threshold" : terminationReason;
        replayFailure.referenceArcBase = previousArcBase;
        replayFailure.referenceArcFraction = std::clamp(
            (previousArcBase - startArcBase) / remainingArcBase, 0.0, 1.0);
        replayFailure.referencePointBase = samplePolylineArc(reference, previousArcBase).point;
        if (distanceFailed) {
            const auto& match = segment.matches.back();
            replayFailure.evaluatorPointBase = segment.tracePointsBase.back();
            replayFailure.segmentPointIndex = segment.tracePointsBase.size() - 1;
            replayFailure.thresholdMeasurement =
                match.thresholdMeasurement;
        }
        segment.terminationReason = replayFailure.reason;
        result.segments.push_back(std::move(segment));
        result.failures.push_back(replayFailure);
        if (failureCallback)
            failureCallback(result.failures.back());

        const double resetArc = std::min(
            endArcBase,
            std::max(previousArcBase, segmentStartArc + nominalStepBase));
        if (!(resetArc > segmentStartArc + kEpsilon))
            throw std::logic_error("fiber replay reset did not advance");
        segmentStartArc = resetArc;
        result.completedReferenceArcBase = resetArc;
    }
    if (result.completedReferenceArcBase < endArcBase - kEpsilon)
        throw std::logic_error("fiber replay exceeded its deterministic reset bound");
    return result;
}

}  // namespace vc::fiber_tracer
