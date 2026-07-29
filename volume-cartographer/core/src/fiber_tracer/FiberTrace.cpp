#include "vc/fiber_tracer/FiberTrace.hpp"

#include "vc/lasagna/ChannelSampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

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

[[nodiscard]] FiberTraceOneWayResult traceOneWay(
    const FiberPredictionSource& predictions,
    const FiberTraceSegmentRequest& request,
    size_t startIndex,
    size_t targetIndex,
    const vc::lasagna::NormalSampler* normalSampler,
    const FiberTraceProgressCallback& progress,
    std::string phase)
{
    const auto& line = request.referenceLine;
    const cv::Vec3d start = line.at(startIndex);
    const cv::Vec3d target = line.at(targetIndex);
    const cv::Vec3d targetPlaneNormal = normalizedOr(
        request.targetPlaneNormal.value_or(target - start),
        normalizedOr(target - start, {1.0, 0.0, 0.0}));
    const cv::Vec3d referenceStartDirection =
        referenceTangentToward(line, startIndex, targetIndex);
    const ScoredDirection startPrediction =
        bestAlignedPrediction(predictions, start, referenceStartDirection);
    const cv::Vec3d startDirection = startPrediction.valid
        ? startPrediction.direction
        : referenceStartDirection;

    const double distance = length(target - start);
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
    for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex) {
        std::vector<BeamState> expanded;
        expanded.reserve(beams.size() * 16);
        for (const auto& beam : beams) {
            if (beam.reached) {
                expanded.push_back(beam);
                continue;
            }
            const auto directions = candidateDirections(
                beam.currentSampleDirection, request.config);
            for (const auto& direction : directions) {
                BeamState next = beam;
                const cv::Vec3d candidatePoint = beam.points.back() + direction * step;
                const double loss = candidateLoss(
                    predictions, normalSampler, beam, direction, candidatePoint, request.config);
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
                expanded.push_back(std::move(next));
            }
        }
        if (expanded.empty()) {
            reason = "no_valid_candidates";
            break;
        }

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

        if (progress) {
            const double signedDistance =
                std::abs(pointToPlaneSigned(beams.front().points.back(), target, targetPlaneNormal));
            FiberTraceProgress event;
            event.phase = phase;
            event.step = stepIndex + 1;
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
        if (prefixes.empty())
            throw std::runtime_error(
                "fiber inference dataset must contain presence/nx/ny channels");

        options_.reserve(prefixes.size());
        for (const auto& prefix : prefixes) {
            const std::string presenceName = prefix.empty() ? "presence" : prefix + "_presence";
            const std::string nxName = prefix.empty() ? "nx" : prefix + "_nx";
            const std::string nyName = prefix.empty() ? "ny" : prefix + "_ny";
            options_.push_back({
                prefix.empty() ? std::string("option_000") : prefix,
                vc::lasagna::bindLasagnaChannel(manifest, presenceName),
                vc::lasagna::bindLasagnaChannel(manifest, nxName),
                vc::lasagna::bindLasagnaChannel(manifest, nyName),
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
    result.forward = traceOneWay(
        predictions, forwardRequest, request.startIndex, request.targetIndex,
        normalSampler, progress, "forward");
    result.reverse = traceOneWay(
        predictions, reverseRequest, request.targetIndex, request.startIndex,
        normalSampler, progress, "reverse");

    result.fusedLine = fuseTraces(
        result.forward.points, result.reverse.points,
        std::max(0.0, request.config.fusionGapFactor));
    if (!result.fusedLine.empty()) {
        result.fusedLine.front() = request.referenceLine[request.startIndex];
        result.fusedLine.back() = request.referenceLine[request.targetIndex];
    }

    const cv::Vec3d start = request.referenceLine[request.startIndex];
    const cv::Vec3d target = request.referenceLine[request.targetIndex];
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

} // namespace vc::fiber_tracer
