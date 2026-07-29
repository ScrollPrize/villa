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

[[nodiscard]] double clampedPositiveDot(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return clamp01(normalizedOrZero(a).dot(normalizedOrZero(b)));
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

[[nodiscard]] double predictionChannelEffectiveScale(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    const std::string& channel)
{
    const auto* group = manifest.groupForChannel(channel);
    if (group == nullptr) {
        throw std::runtime_error(
            "fiber inference dataset is missing required channel '" +
            channel + "'");
    }
    const double scale =
        manifest.sourceToBase * static_cast<double>(group->scaleFactor());
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

[[nodiscard]] cv::Vec3d arbitraryPerpendicular(const cv::Vec3d& direction)
{
    const cv::Vec3d d = normalizedOr(direction, {1.0, 0.0, 0.0});
    cv::Vec3d axis = std::abs(d[0]) < 0.8 ? cv::Vec3d{1.0, 0.0, 0.0}
                                          : cv::Vec3d{0.0, 1.0, 0.0};
    return normalizedOrZero(axis.cross(d));
}

[[nodiscard]] std::vector<cv::Vec3d> candidateDirections(
    const cv::Vec3d& reference,
    const FiberTraceConfig& config)
{
    const cv::Vec3d forward = normalizedOr(reference, {1.0, 0.0, 0.0});
    const cv::Vec3d axis0 = arbitraryPerpendicular(forward);
    const cv::Vec3d axis1 = normalizedOrZero(forward.cross(axis0));
    const double maxAngle = std::max(0.0, config.coneAngleDegrees);
    const double angleStep = std::max(0.25, config.coneAngleStepDegrees);

    std::vector<cv::Vec3d> out;
    for (double ay = -maxAngle; ay <= maxAngle + 1.0e-6; ay += angleStep) {
        for (double ax = -maxAngle; ax <= maxAngle + 1.0e-6; ax += angleStep) {
            const double tx = std::tan(ax * kPi / 180.0);
            const double ty = std::tan(ay * kPi / 180.0);
            out.push_back(normalizedOr(forward + axis0 * tx + axis1 * ty, forward));
        }
    }
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
    const FiberPredictionSource& predictions,
    const cv::Vec3d& point,
    const cv::Vec3d& referenceDirection)
{
    const auto sample = predictions.sample(point, referenceDirection);
    ScoredDirection best;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (const auto& option : sample.options) {
        if (!option.valid)
            continue;
        const cv::Vec3d direction = alignTo(option.direction, referenceDirection);
        const double score = direction.dot(normalizedOrZero(referenceDirection));
        if (score > bestScore) {
            bestScore = score;
            best = {direction, clamp01(option.presence), true};
        }
    }
    return best;
}

[[nodiscard]] double smoothnessLoss(
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& candidateStepDirection,
    const cv::Vec3d& normal,
    const FiberTraceConfig& config,
    bool firstStep)
{
    const cv::Vec3d prev = normalizedOrZero(previousStepDirection);
    const cv::Vec3d cand = alignTo(candidateStepDirection, prev);
    if (length(prev) <= kEpsilon || length(cand) <= kEpsilon)
        return 0.0;

    const cv::Vec3d n = normalizedOrZero(normal);
    const double normalWeight = config.smoothnessNormalWeight;
    const double tangentWeight = firstStep ? 0.0 : config.smoothnessTangentWeight;
    if (length(n) <= kEpsilon) {
        return config.smoothnessWeight * (normalWeight + tangentWeight) *
               (1.0 - clampedPositiveDot(prev, cand));
    }

    const double prevN = prev.dot(n);
    const double candN = cand.dot(n);
    const double normalLoss = (candN - prevN) * (candN - prevN);
    const cv::Vec3d prevT = normalizedOrZero(prev - n * prevN);
    const cv::Vec3d candT = normalizedOrZero(cand - n * candN);
    const double tangentLoss =
        length(prevT) > kEpsilon && length(candT) > kEpsilon
            ? 1.0 - clampedPositiveDot(prevT, candT)
            : 0.0;
    return config.smoothnessWeight *
           (normalWeight * normalLoss + tangentWeight * tangentLoss);
}

struct BeamState {
    std::vector<cv::Vec3d> points;
    cv::Vec3d previousStepDirection{0.0, 0.0, 0.0};
    cv::Vec3d currentSampleDirection{0.0, 0.0, 0.0};
    double loss = 0.0;
    double tracedLength = 0.0;
    bool reached = false;
    std::string reason;
};

[[nodiscard]] double candidateLoss(
    const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler* normalSampler,
    const BeamState& beam,
    const cv::Vec3d& candidateDirection,
    const cv::Vec3d& candidatePoint,
    const FiberTraceConfig& config)
{
    const auto candidateSample =
        predictions.sample(candidatePoint, candidateDirection);
    const bool firstStep = beam.points.size() <= 1;

    cv::Vec3d smoothNormal{0.0, 0.0, 0.0};
    if (normalSampler != nullptr) {
        const auto normalSample = normalSampler->sampleNormal(candidatePoint);
        if (normalSample.valid)
            smoothNormal = normalSample.normal;
    }

    double bestLoss = std::numeric_limits<double>::infinity();
    for (const auto& option : candidateSample.options) {
        if (!option.valid)
            continue;
        const cv::Vec3d candidateSampleDirection =
            alignTo(option.direction, candidateDirection);
        const double presence = clamp01(option.presence);

        const cv::Vec3d prevStep = normalizedOrZero(beam.previousStepDirection);
        const cv::Vec3d currentSample =
            alignTo(beam.currentSampleDirection, candidateDirection);
        const cv::Vec3d currentStep = normalizedOrZero(candidateDirection);

        double score = presence;
        score *= clampedPositiveDot(prevStep, currentStep);
        score *= clampedPositiveDot(prevStep, currentSample);
        score *= clampedPositiveDot(prevStep, candidateSampleDirection);
        score *= clampedPositiveDot(currentSample, currentStep);
        score *= clampedPositiveDot(currentSample, candidateSampleDirection);
        score *= clampedPositiveDot(currentStep, candidateSampleDirection);

        const double loss = (1.0 - score) +
            smoothnessLoss(prevStep, currentStep, smoothNormal, config, firstStep);
        if (loss < bestLoss)
            bestLoss = loss;
    }
    return bestLoss;
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
        bestAlignedPrediction(predictions, start, referenceStartDirection);
    const cv::Vec3d startDirection = startPrediction.valid
        ? startPrediction.direction
        : referenceStartDirection;

    const double distance = request.budgetSpanVoxels > 0.0
        ? request.budgetSpanVoxels
        : length(target - start);
    const double step = std::max(1.0e-3, request.config.stepVoxels);
    const int maxSteps = std::max(
        1,
        static_cast<int>(std::ceil(
            distance * std::max(1.0, request.config.maxStepFactor) / step)));

    BeamState initial;
    initial.points.push_back(start);
    initial.previousStepDirection = startDirection;
    initial.currentSampleDirection = startDirection;
    std::vector<BeamState> beams{std::move(initial)};
    std::string reason = "max_step_factor";

    const double startSigned = pointToPlaneSigned(start, target, targetPlaneNormal);
    const int lookaheadSteps = std::max(1, request.config.beamLookaheadSteps);
    int stepIndex = 0;
    while (stepIndex < maxSteps) {
        std::vector<BeamState> expanded = beams;
        int advanced = 0;
        for (; advanced < lookaheadSteps && stepIndex + advanced < maxSteps; ++advanced) {
            std::vector<BeamState> nextFrontier;
            nextFrontier.reserve(expanded.size() * 16);
            for (const auto& beam : expanded) {
                if (beam.reached) {
                    nextFrontier.push_back(beam);
                    continue;
                }
                const auto directions = candidateDirections(
                    beam.currentSampleDirection, request.config);
                for (const auto& direction : directions) {
                    BeamState next = beam;
                    const cv::Vec3d candidatePoint = beam.points.back() + direction * step;
                    const double loss = candidateLoss(
                        predictions,
                        normalSampler,
                        beam,
                        direction,
                        candidatePoint,
                        request.config);
                    if (!std::isfinite(loss))
                        continue;
                    next.points.push_back(candidatePoint);
                    next.tracedLength += step;
                    next.loss += loss;
                    next.previousStepDirection = direction;
                    const auto currentPrediction =
                        bestAlignedPrediction(predictions, candidatePoint, direction);
                    next.currentSampleDirection =
                        currentPrediction.valid ? currentPrediction.direction : direction;
                    const double signedDistance =
                        pointToPlaneSigned(candidatePoint, target, targetPlaneNormal);
                    next.reached = startSigned <= 0.0 ? signedDistance >= 0.0
                                                      : signedDistance <= 0.0;
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
        }
        if (expanded.empty())
            break;

        std::sort(expanded.begin(), expanded.end(), [](const auto& a, const auto& b) {
            if (a.reached != b.reached)
                return a.reached > b.reached;
            if (a.loss != b.loss)
                return a.loss < b.loss;
            return a.tracedLength < b.tracedLength;
        });
        const size_t keep = static_cast<size_t>(std::max(1, request.config.beamWidth));
        if (expanded.size() > keep)
            expanded.resize(keep);
        beams = std::move(expanded);
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
        if (beams.front().reached) {
            reason = beams.front().reason;
            break;
        }
    }

    if (beams.empty()) {
        return {{start}, false, reason, 0};
    }

    const auto best = std::min_element(
        beams.begin(), beams.end(), [](const auto& a, const auto& b) {
            if (a.reached != b.reached)
                return a.reached > b.reached;
            return a.loss < b.loss;
        });
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

double inferFiberPredictionWorkingToBaseScale(
    const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (!(manifest.sourceToBase > 0.0) || !std::isfinite(manifest.sourceToBase)) {
        throw std::runtime_error(
            "fiber inference manifest source_to_base must be positive and finite");
    }

    const auto prefixes = fiberPredictionPrefixes(manifest);
    if (prefixes.empty()) {
        throw std::runtime_error(
            "fiber inference dataset must contain presence/nx/ny channels");
    }

    std::optional<double> inferredScale;
    std::optional<std::string> inferredChannel;
    for (const auto& prefix : prefixes) {
        for (const auto& channel : predictionChannelNames(prefix)) {
            const double scale = predictionChannelEffectiveScale(manifest, channel);
            if (!inferredScale.has_value()) {
                inferredScale = scale;
                inferredChannel = channel;
                continue;
            }
            if (!nearlySameScale(*inferredScale, scale)) {
                throw std::runtime_error(
                    "fiber inference prediction channels must share one effective "
                    "working-to-base scale; channel '" + channel +
                    "' has scale " + std::to_string(scale) + " but channel '" +
                    *inferredChannel + "' has scale " +
                    std::to_string(*inferredScale));
            }
        }
    }
    return *inferredScale;
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
