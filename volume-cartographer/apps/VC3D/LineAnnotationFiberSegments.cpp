#include "LineAnnotationFiberSegments.hpp"

#include "vc/lasagna/NormalAlignment.hpp"

#include "vc/lasagna/LineSpline.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace vc3d::line_annotation
{

bool shouldRunNativeSeedTrace(
    FiberOptimizationMode mode,
    bool hasSelectedFiberInferenceDataset,
    size_t attachedFiberInferenceDatasetCount) noexcept
{
    return mode == FiberOptimizationMode::NativeFiberTrace3d &&
        (hasSelectedFiberInferenceDataset || attachedFiberInferenceDatasetCount == 1);
}

namespace
{

constexpr std::string_view kOptimizer = "native_fiber_trace3d";

void requireFinitePositive(double value, std::string_view name)
{
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and positive");
    }
}

void requireFiniteNonNegative(double value, std::string_view name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and non-negative");
    }
}

void rejectUnknownKeys(const nlohmann::json& json, const std::unordered_set<std::string>& allowed, std::string_view context)
{
    for (const auto& [key, value] : json.items()) {
        (void)value;
        if (!allowed.contains(key)) {
            throw std::runtime_error(std::string(context) + " contains unknown field: " + key);
        }
    }
}

cv::Vec3d pointFromJson(const nlohmann::json& json)
{
    if (!json.is_array() || json.size() != 3) {
        throw std::runtime_error("control point position must contain exactly three numbers");
    }
    cv::Vec3d point;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!json[axis].is_number()) {
            throw std::runtime_error("control point position must contain exactly three numbers");
        }
        point[static_cast<int>(axis)] = json[axis].get<double>();
        if (!std::isfinite(point[static_cast<int>(axis)])) {
            throw std::runtime_error("control point position must be finite");
        }
    }
    return point;
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return {point[0], point[1], point[2]};
}

double pointDistance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return cv::norm(a - b);
}

cv::Vec3d normalizedFiberEndpointDirection(
    const cv::Vec3d& controlPoint,
    const std::vector<cv::Vec3d>& nativeSpan,
    bool fromStart)
{
    if (nativeSpan.size() < 2) {
        throw std::runtime_error(
            "native fiber span has fewer than two dense points");
    }
    for (size_t offset = 1; offset < nativeSpan.size(); ++offset) {
        const size_t index = fromStart ? offset : nativeSpan.size() - 1 - offset;
        const cv::Vec3d direction = nativeSpan[index] - controlPoint;
        const double distance = cv::norm(direction);
        if (distance > 1.0e-12 && std::isfinite(distance) &&
            std::isfinite(direction[0]) && std::isfinite(direction[1]) &&
            std::isfinite(direction[2])) {
            return direction * (1.0 / distance);
        }
    }
    throw std::runtime_error(
        "native fiber span has no distinct endpoint-adjacent direction point");
}

size_t nearestPointIndex(const std::vector<cv::Vec3d>& points, const cv::Vec3d& target)
{
    if (points.empty()) {
        throw std::invalid_argument("cannot resolve a control point on an empty line");
    }
    size_t best = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < points.size(); ++index) {
        const double distance = pointDistance(points[index], target);
        if (distance < bestDistance) {
            best = index;
            bestDistance = distance;
        }
    }
    return best;
}

std::vector<cv::Vec3d> inclusiveLineSpan(
    const std::vector<cv::Vec3d>& points, size_t first, size_t last)
{
    if (first > last || last >= points.size()) {
        throw std::invalid_argument("invalid inclusive line span");
    }
    return {points.begin() + static_cast<std::ptrdiff_t>(first),
            points.begin() + static_cast<std::ptrdiff_t>(last + 1)};
}

}  // namespace

std::string fiberOptimizationModeToString(FiberOptimizationMode mode)
{
    switch (mode) {
    case FiberOptimizationMode::Lasagna:
        return "lasagna";
    case FiberOptimizationMode::NativeFiberTrace3d:
        return "native_fiber_trace3d";
    }
    throw std::runtime_error("unsupported fiber optimization mode");
}

FiberOptimizationMode fiberOptimizationModeFromString(const std::string& value)
{
    if (value == "lasagna") {
        return FiberOptimizationMode::Lasagna;
    }
    if (value == "native_fiber_trace3d") {
        return FiberOptimizationMode::NativeFiberTrace3d;
    }
    throw std::runtime_error("unsupported fiber optimization mode: " + value);
}

std::string segmentInterpolationGoalToString(SegmentInterpolationGoal goal)
{
    switch (goal) {
    case SegmentInterpolationGoal::Global: return "global";
    case SegmentInterpolationGoal::Cspline: return "cspline";
    case SegmentInterpolationGoal::Lasagna: return "lasagna";
    case SegmentInterpolationGoal::Trace: return "trace";
    }
    throw std::runtime_error("unsupported segment interpolation goal");
}

SegmentInterpolationGoal segmentInterpolationGoalFromString(const std::string& value)
{
    if (value == "global") return SegmentInterpolationGoal::Global;
    if (value == "cspline") return SegmentInterpolationGoal::Cspline;
    if (value == "lasagna") return SegmentInterpolationGoal::Lasagna;
    if (value == "trace") return SegmentInterpolationGoal::Trace;
    throw std::runtime_error("unsupported segment interpolation goal: " + value);
}

std::string segmentInterpolationModeToString(SegmentInterpolationMode mode)
{
    switch (mode) {
    case SegmentInterpolationMode::Cspline: return "cspline";
    case SegmentInterpolationMode::Lasagna: return "lasagna";
    case SegmentInterpolationMode::Trace: return "trace";
    }
    throw std::runtime_error("unsupported segment interpolation mode");
}

SegmentInterpolationMode segmentInterpolationModeFromString(const std::string& value)
{
    if (value == "cspline") return SegmentInterpolationMode::Cspline;
    if (value == "lasagna") return SegmentInterpolationMode::Lasagna;
    if (value == "trace") return SegmentInterpolationMode::Trace;
    throw std::runtime_error("unsupported segment interpolation mode: " + value);
}

char segmentInterpolationModeMarker(SegmentInterpolationMode mode) noexcept
{
    switch (mode) {
    case SegmentInterpolationMode::Cspline: return 'C';
    case SegmentInterpolationMode::Lasagna: return 'L';
    case SegmentInterpolationMode::Trace: return 'T';
    }
    return '?';
}

SegmentInterpolationMode resolveSegmentInterpolationMode(
    SegmentInterpolationGoal goal,
    FiberOptimizationMode globalMode,
    double endpointDistanceBaseVoxels)
{
    if (!std::isfinite(endpointDistanceBaseVoxels) ||
        endpointDistanceBaseVoxels < 0.0) {
        throw std::invalid_argument("segment endpoint distance must be finite and non-negative");
    }
    if (goal == SegmentInterpolationGoal::Cspline ||
        (goal == SegmentInterpolationGoal::Global &&
         endpointDistanceBaseVoxels < 100.0)) {
        return SegmentInterpolationMode::Cspline;
    }
    if (goal == SegmentInterpolationGoal::Lasagna ||
        (goal == SegmentInterpolationGoal::Global &&
         globalMode == FiberOptimizationMode::Lasagna)) {
        return SegmentInterpolationMode::Lasagna;
    }
    return SegmentInterpolationMode::Trace;
}

bool isAcceptedNativeTrace(const FiberTraceSegmentMetadata& metadata) noexcept
{
    return metadata.interpMode == SegmentInterpolationMode::Trace;
}

bool isAcceptedNativeTrace(
    const std::optional<FiberTraceSegmentMetadata>& metadata) noexcept
{
    return metadata.has_value() && isAcceptedNativeTrace(*metadata);
}

namespace
{

template <typename ControlPointT>
bool anyAcceptedTraceSpan(const std::vector<ControlPointT>& controls) noexcept
{
    return std::any_of(controls.begin(), controls.end(),
                       [](const ControlPointT& control) {
                           return isAcceptedNativeTrace(control.segmentToNext);
                       });
}

}  // namespace

bool hasAcceptedTraceSpan(const std::vector<LineControlPoint>& controls) noexcept
{
    return anyAcceptedTraceSpan(controls);
}

bool hasAcceptedTraceSpan(const std::vector<StoredControlPoint>& controls) noexcept
{
    return anyAcceptedTraceSpan(controls);
}

FiberTraceState deriveTraceState(
    FiberOptimizationMode mode,
    const std::vector<StoredControlPoint>& controls) noexcept
{
    if (!hasAcceptedTraceSpan(controls)) {
        return FiberTraceState::Legacy;
    }
    return mode == FiberOptimizationMode::NativeFiberTrace3d
        ? FiberTraceState::Predictions
        : FiberTraceState::Mixed;
}

FiberTraceSegmentMetadata fiberTraceSegmentMetadataForResult(
    std::string normalManifestLocation,
    std::string fiberManifestLocation,
    double traceToBaseScale,
    const vc::fiber_tracer::FiberTraceConfig& config,
    const vc::fiber_tracer::FiberTraceSegmentResult& result)
{
    FiberTraceSegmentMetadata metadata;
    metadata.outcome = result.accepted
        ? FiberTraceSegmentMetadata::Outcome::AcceptedNative
        : FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
    metadata.normalManifestLocation = std::move(normalManifestLocation);
    metadata.fiberManifestLocation = std::move(fiberManifestLocation);
    metadata.traceToBaseScale = traceToBaseScale;
    metadata.config = config;
    metadata.config.baseVoxelSizeUm.reset();
    metadata.config.profile = nullptr;
    if (result.accepted) {
        metadata.interpMode = SegmentInterpolationMode::Trace;
        if (std::isfinite(result.meetingErrorBaseVoxels))
            metadata.meetingErrorBaseVoxels = result.meetingErrorBaseVoxels;
        if (std::isfinite(result.meetingErrorRatio))
            metadata.meetingErrorRatio = result.meetingErrorRatio;
        metadata.meetingSource = result.meetingSource;
        metadata.metric = metadata.meetingErrorBaseVoxels;
        metadata.message = "trace";
    } else {
        metadata.interpMode = SegmentInterpolationMode::Lasagna;
        metadata.failureCode = result.reason.empty()
            ? "fusion_failed"
            : result.reason;
        metadata.failureDetail = result.detail;
        if (metadata.failureCode == "meeting_error_threshold" &&
            std::isfinite(result.meetingErrorBaseVoxels)) {
            const double traceLengthBase =
                result.meetingTraceLengthTraceVoxels * traceToBaseScale;
            const double ratioThreshold = config.meetingAcceptMaxErrorRatio * traceLengthBase;
            std::ostringstream message;
            if (ratioThreshold > config.endpointAcceptThresholdBaseVoxels) {
                message << "trace gap " << result.meetingErrorRatio * 100.0
                        << "% exceeds " << config.meetingAcceptMaxErrorRatio * 100.0 << '%';
            } else {
                message << "trace gap " << result.meetingErrorBaseVoxels
                        << " vx exceeds " << config.endpointAcceptThresholdBaseVoxels << " vx";
            }
            metadata.message = message.str();
        } else {
            metadata.message = "trace: " + metadata.failureCode;
        }
    }
    return metadata;
}

FiberTraceSegmentMetadata fiberTraceSegmentMetadataForException(
    std::string normalManifestLocation,
    std::string fiberManifestLocation,
    double traceToBaseScale,
    const vc::fiber_tracer::FiberTraceConfig& config,
    std::string detail)
{
    FiberTraceSegmentMetadata metadata;
    metadata.outcome = FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
    metadata.normalManifestLocation = std::move(normalManifestLocation);
    metadata.fiberManifestLocation = std::move(fiberManifestLocation);
    metadata.traceToBaseScale = traceToBaseScale;
    metadata.config = config;
    metadata.config.baseVoxelSizeUm.reset();
    metadata.config.profile = nullptr;
    metadata.interpMode = SegmentInterpolationMode::Lasagna;
    metadata.failureCode = "trace_exception";
    metadata.failureDetail = std::move(detail);
    metadata.message = "trace: trace_exception";
    return metadata;
}

namespace {

void appendFiberModeReport(FiberModeOptimizationResult& output)
{
    std::ostringstream message;
    message << output.optimization.report.message
            << "\nfiber_mode native_segments=" << output.nativeSegments
            << " lasagna_fallback_segments=" << output.lasagnaFallbackSegments
            << " native_extrapolations=" << output.nativeExtrapolations
            << " lasagna_fallback_extrapolations="
            << output.lasagnaFallbackExtrapolations;
    output.optimization.report.message = message.str();
}

bool usableNativeExtrapolation(
    const vc::fiber_tracer::FiberTraceOneWayResult& traced)
{
    return traced.points.size() >= 2 &&
        (traced.reachedTraceLength ||
         traced.reason.starts_with("no_valid_candidates"));
}

void reportExtrapolationFallback(
    const FiberModeOptimizationRequest& request,
    FiberExtrapolationFallbackDiagnostic::Side side,
    const std::optional<vc::fiber_tracer::FiberTraceOneWayResult>& traced,
    const std::string& exception)
{
    if (!request.extrapolationFallbackCallback)
        return;
    FiberExtrapolationFallbackDiagnostic diagnostic;
    diagnostic.side = side;
    diagnostic.tracePointCount = traced ? traced->points.size() : 0;
    diagnostic.fromException = !exception.empty();
    if (!exception.empty()) {
        diagnostic.reason = exception;
    } else if (traced && !traced->reason.empty()) {
        diagnostic.reason = traced->reason;
    } else {
        diagnostic.reason = "native extrapolation returned no failure reason";
    }
    request.extrapolationFallbackCallback(diagnostic);
}

void replaceOpenTailsWithNative(
    const FiberModeOptimizationRequest& request,
    const vc::fiber_tracer::FiberTraceCoordinateAdapter& coordinates,
    int firstControl,
    int lastControl,
    FiberModeOptimizationResult& output)
{
    std::vector<cv::Vec3d> finalPoints;
    finalPoints.reserve(output.optimization.line.points.size());
    for (const auto& point : output.optimization.line.points) {
        finalPoints.push_back(point.position);
    }
    if (firstControl < 0 || lastControl < firstControl ||
        lastControl >= static_cast<int>(finalPoints.size())) {
        throw std::runtime_error(
            "Lasagna fallback returned invalid endpoint control indices");
    }

    std::vector<cv::Vec3d> leftTail(
        finalPoints.begin(), finalPoints.begin() + firstControl + 1);
    std::vector<cv::Vec3d> rightTail(
        finalPoints.begin() + lastControl, finalPoints.end());
    if (request.extrapolationDistanceBaseVoxels == 0.0) {
        leftTail = {finalPoints[static_cast<size_t>(firstControl)]};
        rightTail = {finalPoints[static_cast<size_t>(lastControl)]};
    } else {
        if (firstControl + 1 >= static_cast<int>(finalPoints.size()) ||
            lastControl == 0) {
            throw std::runtime_error(
                "Lasagna fallback did not provide both endpoint directions");
        }
        const double extrapolationTrace = coordinates.baseDistanceToTrace(
            request.extrapolationDistanceBaseVoxels);
        const auto traceTail = [&](int endpoint, int inner) {
            return vc::fiber_tracer::traceFiberExtrapolation(
                *request.predictions,
                coordinates.baseToTrace(finalPoints[static_cast<size_t>(endpoint)]),
                coordinates.baseToTrace(finalPoints[static_cast<size_t>(endpoint)]) -
                    coordinates.baseToTrace(finalPoints[static_cast<size_t>(inner)]),
                extrapolationTrace,
                request.traceConfig,
                request.traceNormalSampler);
        };
        std::optional<vc::fiber_tracer::FiberTraceOneWayResult> left;
        std::string leftException;
        try {
            left = traceTail(firstControl, firstControl + 1);
        } catch (const std::exception& ex) {
            leftException = ex.what();
            left.reset();
        } catch (...) {
            leftException = "unknown native fiber extrapolation exception";
            left.reset();
        }
        if (left && usableNativeExtrapolation(*left)) {
            leftTail = coordinates.traceToBase(left->points);
            leftTail.front() = finalPoints[static_cast<size_t>(firstControl)];
            std::reverse(leftTail.begin(), leftTail.end());
            ++output.nativeExtrapolations;
        } else {
            reportExtrapolationFallback(
                request,
                FiberExtrapolationFallbackDiagnostic::Side::Left,
                left,
                leftException);
            ++output.lasagnaFallbackExtrapolations;
        }
        std::optional<vc::fiber_tracer::FiberTraceOneWayResult> right;
        std::string rightException;
        try {
            right = traceTail(lastControl, lastControl - 1);
        } catch (const std::exception& ex) {
            rightException = ex.what();
            right.reset();
        } catch (...) {
            rightException = "unknown native fiber extrapolation exception";
            right.reset();
        }
        if (right && usableNativeExtrapolation(*right)) {
            rightTail = coordinates.traceToBase(right->points);
            rightTail.front() = finalPoints[static_cast<size_t>(lastControl)];
            ++output.nativeExtrapolations;
        } else {
            reportExtrapolationFallback(
                request,
                FiberExtrapolationFallbackDiagnostic::Side::Right,
                right,
                rightException);
            ++output.lasagnaFallbackExtrapolations;
        }
    }

    std::vector<cv::Vec3d> combined;
    combined.reserve(leftTail.size() + finalPoints.size() + rightTail.size());
    combined.insert(combined.end(), leftTail.begin(), leftTail.end());
    if (lastControl > firstControl) {
        combined.insert(combined.end(),
                        finalPoints.begin() + firstControl + 1,
                        finalPoints.begin() + lastControl);
        combined.insert(combined.end(), rightTail.begin(), rightTail.end());
    } else if (rightTail.size() > 1) {
        combined.insert(combined.end(), rightTail.begin() + 1, rightTail.end());
    }

    output.optimization.line.points.clear();
    output.optimization.line.segmentSamples.clear();
    output.optimization.line.displayFrameAnchorIndex =
        static_cast<int>(combined.size() / 2);
    output.optimization.line.points.reserve(combined.size());
    for (const auto& point : combined) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = point;
        linePoint.sampledNormal = request.baseNormalSampler->sampleNormal(point);
        output.optimization.line.points.push_back(std::move(linePoint));
    }
}

}  // namespace

FiberModeOptimizationResult optimizeFiberWithNativeFallback(
    FiberModeOptimizationRequest request)
{
    if (!request.baseNormalSampler) {
        throw std::invalid_argument(
            "fiber-mode optimization requires a base normal sampler");
    }
    if (request.controlPoints.empty() || request.linePointsBase.size() < 2) {
        throw std::invalid_argument(
            "fiber-mode optimization requires a control point and at least two line points");
    }
    if (!(request.traceToBaseScale > 0.0) ||
        !std::isfinite(request.traceToBaseScale)) {
        throw std::invalid_argument("trace-to-base scale must be finite and positive");
    }
    if (!(request.extrapolationDistanceBaseVoxels >= 0.0) ||
        !std::isfinite(request.extrapolationDistanceBaseVoxels)) {
        throw std::invalid_argument(
            "extrapolation distance must be finite and non-negative");
    }

    std::stable_sort(request.controlPoints.begin(), request.controlPoints.end(),
                     [](const LineControlPoint& lhs, const LineControlPoint& rhs) {
                         return lhs.linePosition < rhs.linePosition;
                     });
    for (size_t index = 0; index + 1 < request.controlPoints.size(); ++index) {
        if (!request.controlPoints[index].segmentToNext) {
            FiberTraceSegmentMetadata metadata;
            metadata.interpMode = SegmentInterpolationMode::Lasagna;
            metadata.message = "lasagna";
            request.controlPoints[index].segmentToNext = std::move(metadata);
        }
    }
    request.controlPoints.back().segmentToNext.reset();

    const vc::fiber_tracer::FiberTraceCoordinateAdapter coordinates(
        request.traceToBaseScale);
    request.traceConfig.traceToBaseScale = request.traceToBaseScale;
    FiberModeOptimizationResult output;
    if (request.controlPoints.size() == 1) {
        auto config = request.lasagnaConfig;
        const size_t inputControlIndex = nearestPointIndex(
            request.linePointsBase, request.controlPoints.front().volumePoint);
        cv::Vec3d tangent;
        if (inputControlIndex > 0 &&
            inputControlIndex + 1 < request.linePointsBase.size()) {
            tangent = request.linePointsBase[inputControlIndex + 1] -
                request.linePointsBase[inputControlIndex - 1];
        } else if (inputControlIndex + 1 < request.linePointsBase.size()) {
            tangent = request.linePointsBase[inputControlIndex + 1] -
                request.linePointsBase[inputControlIndex];
        } else {
            tangent = request.linePointsBase[inputControlIndex] -
                request.linePointsBase[inputControlIndex - 1];
        }
        const double tangentLength = cv::norm(tangent);
        if (tangentLength > 1.0e-12 && std::isfinite(tangentLength)) {
            config.initialTangent = tangent * (1.0 / tangentLength);
            config.useInitialTangent = true;
        }

        vc::lasagna::LineOptimizer optimizer(*request.baseNormalSampler);
        output.optimization = optimizer.optimizeFromControlPoints(
            optimizerControlPoints(request.controlPoints), config);
        std::vector<cv::Vec3d> baselinePoints;
        baselinePoints.reserve(output.optimization.line.points.size());
        for (const auto& point : output.optimization.line.points) {
            baselinePoints.push_back(point.position);
        }
        const int controlIndex = static_cast<int>(nearestPointIndex(
            baselinePoints, request.controlPoints.front().volumePoint));
        request.controlPoints.front().optimizedIndex = controlIndex;
        request.controlPoints.front().linePosition =
            static_cast<double>(controlIndex);
        if (request.globalMode == FiberOptimizationMode::NativeFiberTrace3d) {
            replaceOpenTailsWithNative(
                request, coordinates, controlIndex, controlIndex, output);
        }
        appendFiberModeReport(output);
        output.controlPoints = std::move(request.controlPoints);
        return output;
    }

    const std::vector<cv::Vec3d> referenceTrace =
        coordinates.baseToTrace(request.linePointsBase);

    std::vector<size_t> originalControlIndices;
    originalControlIndices.reserve(request.controlPoints.size());
    for (const auto& control : request.controlPoints) {
        originalControlIndices.push_back(
            nearestPointIndex(request.linePointsBase, control.volumePoint));
    }
    for (size_t index = 1; index < originalControlIndices.size(); ++index) {
        if (originalControlIndices[index] <= originalControlIndices[index - 1]) {
            throw std::runtime_error(
                "fiber control points do not resolve in strict line order");
        }
    }

    std::vector<std::vector<cv::Vec3d>> spans(request.controlPoints.size() - 1);
    std::vector<SegmentInterpolationMode> modes(
        spans.size(), SegmentInterpolationMode::Lasagna);
    std::vector<bool> attempt(spans.size(), true);
    if (request.globalGoalsOnly) {
        for (size_t index = 0; index < spans.size(); ++index) {
            attempt[index] = request.controlPoints[index].segmentToNext->interpGoal ==
                SegmentInterpolationGoal::Global;
        }
    }
    if (request.dirtySegments) {
        std::fill(attempt.begin(), attempt.end(), false);
        const auto joinsSplineRun = [&](size_t index) {
            const auto& metadata = *request.controlPoints[index].segmentToNext;
            return metadata.interpGoal == SegmentInterpolationGoal::Cspline ||
                   metadata.interpMode == SegmentInterpolationMode::Cspline;
        };
        for (const size_t dirty : *request.dirtySegments) {
            if (dirty >= spans.size())
                throw std::invalid_argument(
                    "dirty interpolation segment is out of range");
            size_t begin = dirty;
            size_t end = dirty;
            if (joinsSplineRun(dirty)) {
                while (begin > 0 && joinsSplineRun(begin - 1))
                    --begin;
                while (end + 1 < spans.size() && joinsSplineRun(end + 1))
                    ++end;
            }
            for (size_t index = begin; index <= end; ++index)
                attempt[index] = true;
        }
    }
    std::vector<bool> fixedSpan(spans.size(), false);
    for (size_t spanIndex = 0; spanIndex < spans.size(); ++spanIndex) {
        auto& owner = request.controlPoints[spanIndex];
        const size_t first = originalControlIndices[spanIndex];
        const size_t last = originalControlIndices[spanIndex + 1];
        spans[spanIndex] = inclusiveLineSpan(request.linePointsBase, first, last);
        auto& metadata = *owner.segmentToNext;
        const SegmentInterpolationGoal goal = metadata.interpGoal;
        if (!attempt[spanIndex]) {
            modes[spanIndex] = metadata.interpMode;
            fixedSpan[spanIndex] = true;
            continue;
        }
        const SegmentInterpolationMode requestedMode = resolveSegmentInterpolationMode(
            goal,
            request.globalMode,
            pointDistance(owner.volumePoint,
                          request.controlPoints[spanIndex + 1].volumePoint));
        if (requestedMode == SegmentInterpolationMode::Cspline) {
            modes[spanIndex] = SegmentInterpolationMode::Cspline;
            metadata.interpMode = SegmentInterpolationMode::Cspline;
            metadata.metric.reset();
            metadata.message = goal == SegmentInterpolationGoal::Global
                ? "short span"
                : "cspline";
            metadata.meetingErrorBaseVoxels.reset();
            metadata.meetingErrorRatio.reset();
            metadata.meetingSource.clear();
            metadata.failureCode.clear();
            metadata.failureDetail.clear();
            metadata.lasagnaFailureCode.clear();
            metadata.lasagnaFailureDetail.clear();
            metadata.normalManifestLocation.clear();
            metadata.fiberManifestLocation.clear();
            continue;
        }
        const bool wantsTrace = requestedMode == SegmentInterpolationMode::Trace;
        if (!wantsTrace) {
            modes[spanIndex] = SegmentInterpolationMode::Lasagna;
            metadata.interpMode = SegmentInterpolationMode::Lasagna;
            metadata.metric.reset();
            metadata.message = "lasagna";
            metadata.meetingErrorBaseVoxels.reset();
            metadata.meetingErrorRatio.reset();
            metadata.meetingSource.clear();
            metadata.failureCode.clear();
            metadata.failureDetail.clear();
            metadata.lasagnaFailureCode.clear();
            metadata.lasagnaFailureDetail.clear();
            metadata.normalManifestLocation = request.normalManifestLocation;
            metadata.fiberManifestLocation.clear();
            continue;
        }

        vc::fiber_tracer::FiberTraceSegmentRequest traceRequest;
        traceRequest.referenceLine = referenceTrace;
        traceRequest.startIndex = first;
        traceRequest.targetIndex = last;
        traceRequest.config = request.traceConfig;
        std::optional<vc::fiber_tracer::FiberTraceSegmentResult> traced;
        std::string traceException;
        try {
            if (!request.predictions || !request.traceNormalSampler) {
                throw std::runtime_error(
                    "fiber prediction or trace-normal sampler is unavailable");
            }
            traced = vc::fiber_tracer::traceFiberSegment(
                *request.predictions,
                traceRequest,
                request.traceNormalSampler);
        } catch (const std::exception& ex) {
            traceException = ex.what();
            traced.reset();
        } catch (...) {
            traceException = "unknown native fiber trace exception";
            traced.reset();
        }
        if (!traced || !traced->accepted || traced->fusedLine.size() < 2) {
            spans[spanIndex] = inclusiveLineSpan(request.linePointsBase, first, last);
            owner.segmentToNext = traced
                ? fiberTraceSegmentMetadataForResult(
                      request.normalManifestLocation,
                      request.fiberManifestLocation,
                      request.traceToBaseScale,
                      request.traceConfig,
                      *traced)
                : fiberTraceSegmentMetadataForException(
                      request.normalManifestLocation,
                      request.fiberManifestLocation,
                      request.traceToBaseScale,
                      request.traceConfig,
                      std::move(traceException));
            owner.segmentToNext->interpGoal = goal;
            modes[spanIndex] = SegmentInterpolationMode::Lasagna;
            ++output.lasagnaFallbackSegments;
            continue;
        }

        spans[spanIndex] = coordinates.traceSegmentToBase(
            traced->fusedLine,
            owner.volumePoint,
            request.controlPoints[spanIndex + 1].volumePoint);
        owner.segmentToNext = fiberTraceSegmentMetadataForResult(
            request.normalManifestLocation,
            request.fiberManifestLocation,
            request.traceToBaseScale,
            request.traceConfig,
            *traced);
        owner.segmentToNext->interpGoal = goal;
        modes[spanIndex] = SegmentInterpolationMode::Trace;
        ++output.nativeSegments;
    }

    const auto generateSplineRuns = [&]() {
        size_t begin = 0;
        while (begin < spans.size()) {
            if (modes[begin] != SegmentInterpolationMode::Cspline || fixedSpan[begin]) {
                ++begin;
                continue;
            }
            size_t end = begin;
            while (end + 1 < spans.size() &&
                   modes[end + 1] == SegmentInterpolationMode::Cspline) {
                if (fixedSpan[end + 1])
                    break;
                ++end;
            }
            vc::lasagna::LineSplineRequest splineRequest;
            for (size_t control = begin; control <= end + 1; ++control)
                splineRequest.controlPoints.push_back(request.controlPoints[control].volumePoint);
            splineRequest.sampleSpacing = std::max(0.1, request.lasagnaConfig.segmentLength);
            if (begin > 0 && spans[begin - 1].size() >= 2) {
                splineRequest.leftDirection =
                    spans[begin - 1].back() - spans[begin - 1][spans[begin - 1].size() - 2];
            }
            if (end + 1 < spans.size() && spans[end + 1].size() >= 2) {
                splineRequest.rightDirection = spans[end + 1][1] - spans[end + 1][0];
            }
            const auto spline = vc::lasagna::interpolateLineControlPoints(splineRequest);
            for (size_t local = 0; local <= end - begin; ++local) {
                const int first = spline.controlPointIndices[local];
                const int last = spline.controlPointIndices[local + 1];
                spans[begin + local] = {
                    spline.points.begin() + first,
                    spline.points.begin() + last + 1};
                auto& metadata = *request.controlPoints[begin + local].segmentToNext;
                metadata.interpMode = SegmentInterpolationMode::Cspline;
                metadata.metric.reset();
                if (metadata.message.empty())
                    metadata.message = "cspline";
            }
            begin = end + 1;
        }
    };

    const auto stitch = [&]() {
        std::pair<std::vector<cv::Vec3d>, std::vector<int>> value;
        auto& [points, indices] = value;
        points.insert(points.end(), request.linePointsBase.begin(),
                      request.linePointsBase.begin() +
                          static_cast<std::ptrdiff_t>(originalControlIndices.front()));
        indices.reserve(request.controlPoints.size());
        for (size_t spanIndex = 0; spanIndex < spans.size(); ++spanIndex) {
            const auto& span = spans[spanIndex];
            if (span.size() < 2)
                throw std::runtime_error("interpolation produced an invalid span");
            if (spanIndex == 0) {
                indices.push_back(static_cast<int>(points.size()));
                points.insert(points.end(), span.begin(), span.end());
            } else {
                points.insert(points.end(), span.begin() + 1, span.end());
            }
            indices.push_back(static_cast<int>(points.size()) - 1);
        }
        points.insert(points.end(),
                      request.linePointsBase.begin() +
                          static_cast<std::ptrdiff_t>(originalControlIndices.back() + 1),
                      request.linePointsBase.end());
        return value;
    };

    vc::lasagna::LineOptimizer optimizer(*request.baseNormalSampler);
    vc::lasagna::LineReinitializationOptimizationResult reinitialized;
    std::vector<int> controlIndices;
    while (true) {
        generateSplineRuns();
        auto [stitched, indices] = stitch();
        controlIndices = std::move(indices);
        for (size_t index = 0; index < request.controlPoints.size(); ++index) {
            request.controlPoints[index].optimizedIndex = controlIndices[index];
            request.controlPoints[index].linePosition = controlIndices[index];
        }

        std::vector<std::pair<int, int>> protectedSpans;
        std::vector<vc::lasagna::LineControlPointHardDirectionConstraint> hardDirections;
        std::vector<bool> protectedMode(modes.size(), false);
        for (size_t index = 0; index < modes.size(); ++index) {
            protectedMode[index] = fixedSpan[index] ||
                modes[index] != SegmentInterpolationMode::Lasagna;
        }
        for (size_t index = 0; index < modes.size(); ++index) {
            if (!protectedMode[index])
                continue;
            protectedSpans.emplace_back(static_cast<int>(index), static_cast<int>(index + 1));
            const cv::Vec3d leftDirection = normalizedFiberEndpointDirection(
                request.controlPoints[index].volumePoint, spans[index], true);
            const cv::Vec3d rightDirection = normalizedFiberEndpointDirection(
                request.controlPoints[index + 1].volumePoint, spans[index], false);
            if (index == 0 || !protectedMode[index - 1]) {
                hardDirections.push_back({static_cast<int>(index),
                                          vc::lasagna::LineControlPointSide::Before,
                                          -leftDirection});
            }
            if (index + 1 == modes.size() || !protectedMode[index + 1]) {
                hardDirections.push_back({static_cast<int>(index + 1),
                                          vc::lasagna::LineControlPointSide::After,
                                          -rightDirection});
            }
        }
        reinitialized = optimizer.reinitializeAndOptimizeExistingLine(
            std::move(stitched), optimizerControlPoints(request.controlPoints),
            controlIndices, controlIndices[controlIndices.size() / 2],
            request.lasagnaConfig, std::move(protectedSpans),
            std::move(hardDirections));
        if (!reinitialized.failed)
            break;
        const int failed = reinitialized.failedSegmentIndex;
        if (failed < 0 || static_cast<size_t>(failed) >= modes.size() ||
            modes[static_cast<size_t>(failed)] != SegmentInterpolationMode::Lasagna) {
            throw std::runtime_error("Lasagna interpolation failed structurally: " +
                                     reinitialized.failureReason);
        }
        modes[static_cast<size_t>(failed)] = SegmentInterpolationMode::Cspline;
        auto& metadata = *request.controlPoints[static_cast<size_t>(failed)].segmentToNext;
        metadata.interpMode = SegmentInterpolationMode::Cspline;
        metadata.metric.reset();
        metadata.lasagnaFailureCode = "no_usable_candidate";
        metadata.lasagnaFailureDetail = reinitialized.failureReason;
        if (!metadata.message.empty())
            metadata.message += " -> ";
        metadata.message += "lasagna: " + reinitialized.failureReason;
    }

    for (size_t spanIndex = 0; spanIndex < modes.size(); ++spanIndex) {
        auto& metadata = *request.controlPoints[spanIndex].segmentToNext;
        if (fixedSpan[spanIndex])
            continue;
        metadata.interpMode = modes[spanIndex];
        if (modes[spanIndex] != SegmentInterpolationMode::Lasagna)
            continue;
        metadata.outcome = FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
        metadata.meetingErrorBaseVoxels.reset();
        metadata.meetingErrorRatio.reset();
        metadata.meetingSource.clear();
        metadata.metric.reset();
        if (spanIndex < reinitialized.spans.size()) {
            metadata.metric = vc::lasagna::normalAlignmentMagnitudeErrorDegrees(
                reinitialized.spans[spanIndex].chosenMaxNormalAlignmentAbs);
        }
        if (metadata.message.empty() || metadata.message == "lasagna")
            metadata.message = "lasagna";
        else if (metadata.message.find("lasagna") == std::string::npos)
            metadata.message += " -> lasagna";
        metadata.lasagnaFailureCode.clear();
        metadata.lasagnaFailureDetail.clear();
    }

    if (reinitialized.fixedPointIndices.size() != request.controlPoints.size()) {
        throw std::runtime_error(
            "Lasagna fallback returned an invalid control-point index map");
    }
    const int firstControl = reinitialized.fixedPointIndices.front();
    const int lastControl = reinitialized.fixedPointIndices.back();
    output.optimization = std::move(reinitialized.optimization);
    if (request.globalMode == FiberOptimizationMode::NativeFiberTrace3d) {
        replaceOpenTailsWithNative(
            request, coordinates, firstControl, lastControl, output);
    }
    appendFiberModeReport(output);
    output.controlPoints = std::move(request.controlPoints);
    return output;
}

nlohmann::json fiberTraceSegmentMetadataToJson(const FiberTraceSegmentMetadata& metadata)
{
    const auto& config = metadata.config;
    const bool acceptedNative = isAcceptedNativeTrace(metadata);
    nlohmann::json json = {
        {"optimizer", kOptimizer},
        {"metadata_version", FiberTraceSegmentMetadata::MetadataVersion},
        {"tracer_version", FiberTraceSegmentMetadata::TracerVersion},
        {"interp_goal", segmentInterpolationGoalToString(metadata.interpGoal)},
        {"interp_mode", segmentInterpolationModeToString(metadata.interpMode)},
        {"metric", metadata.metric ? nlohmann::json(*metadata.metric) : nlohmann::json(nullptr)},
        {"msg", metadata.message},
        {"normal_manifest", metadata.normalManifestLocation},
        {"fiber_manifest", metadata.fiberManifestLocation},
        {"trace_to_base_scale", metadata.traceToBaseScale},
        {"meeting_error_base_voxels", acceptedNative && metadata.meetingErrorBaseVoxels
             ? nlohmann::json(*metadata.meetingErrorBaseVoxels)
             : nlohmann::json(nullptr)},
        {"meeting_error_ratio", acceptedNative && metadata.meetingErrorRatio
             ? nlohmann::json(*metadata.meetingErrorRatio)
             : nlohmann::json(nullptr)},
        {"meeting_source", acceptedNative ? metadata.meetingSource : std::string{}},
        {"failure_code", metadata.failureCode},
        {"failure_detail", metadata.failureDetail},
        {"lasagna_failure_code", metadata.lasagnaFailureCode},
        {"lasagna_failure_detail", metadata.lasagnaFailureDetail},
        {"config",
         {
             {"step_voxels", config.stepVoxels},
             {"cone_angle_degrees", config.coneAngleDegrees},
             {"cone_angle_step_degrees", config.coneAngleStepDegrees},
             {"cone_grid_size", config.coneGridSize},
             {"beam_width", config.beamWidth},
             {"beam_prune_distance_voxels", config.beamPruneDistanceVoxels},
             {"beam_lookahead_steps", config.beamLookaheadSteps},
             {"smoothness_weight", config.smoothnessWeight},
             {"smoothness_normal_weight", config.smoothnessNormalWeight},
             {"smoothness_tangent_weight", config.smoothnessTangentWeight},
             {"smoothness_free_angle_degrees", config.smoothnessFreeAngleDegrees},
             {"cumulative_smoothness_steps", config.cumulativeSmoothnessSteps},
             {"cumulative_smoothness_tangent_weight", config.cumulativeSmoothnessTangentWeight},
             {"initial_free_angle_degrees", config.initialFreeAngleDegrees},
             {"max_step_factor", config.maxStepFactor},
             {"meeting_accept_max_error_ratio", config.meetingAcceptMaxErrorRatio},
             {"endpoint_accept_threshold_base_voxels", config.endpointAcceptThresholdBaseVoxels},
         }},
    };
    return json;
}

FiberTraceSegmentMetadata fiberTraceSegmentMetadataFromJson(const nlohmann::json& json)
{
    if (!json.is_object()) {
        throw std::runtime_error("segment_to_next must be an object");
    }
    if (json.at("optimizer").get<std::string>() != kOptimizer) {
        throw std::runtime_error("unsupported segment_to_next optimizer");
    }
    const int metadataVersion = json.at("metadata_version").get<int>();
    const int tracerVersion = json.at("tracer_version").get<int>();
    const bool currentVersion =
        metadataVersion == FiberTraceSegmentMetadata::MetadataVersion &&
        tracerVersion == FiberTraceSegmentMetadata::TracerVersion;
    if (!currentVersion)
        throw std::runtime_error("unsupported segment_to_next metadata/tracer version");
    rejectUnknownKeys(
        json,
        {"optimizer", "metadata_version", "tracer_version",
         "interp_goal", "interp_mode", "metric", "msg",
         "normal_manifest", "fiber_manifest", "trace_to_base_scale",
         "meeting_error_base_voxels", "meeting_error_ratio",
         "meeting_source", "failure_code", "failure_detail",
         "lasagna_failure_code", "lasagna_failure_detail", "config"},
        "segment_to_next");

    FiberTraceSegmentMetadata metadata;
    metadata.normalManifestLocation = json.at("normal_manifest").get<std::string>();
    metadata.fiberManifestLocation = json.at("fiber_manifest").get<std::string>();
    metadata.traceToBaseScale = json.at("trace_to_base_scale").get<double>();
    metadata.interpGoal = segmentInterpolationGoalFromString(
        json.at("interp_goal").get<std::string>());
    metadata.interpMode = segmentInterpolationModeFromString(
        json.at("interp_mode").get<std::string>());
    metadata.outcome = metadata.interpMode == SegmentInterpolationMode::Trace
        ? FiberTraceSegmentMetadata::Outcome::AcceptedNative
        : FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
    if (!json.at("metric").is_null())
        metadata.metric = json.at("metric").get<double>();
    metadata.message = json.at("msg").get<std::string>();
    if (!json.at("meeting_error_base_voxels").is_null())
        metadata.meetingErrorBaseVoxels = json.at("meeting_error_base_voxels").get<double>();
    if (!json.at("meeting_error_ratio").is_null())
        metadata.meetingErrorRatio = json.at("meeting_error_ratio").get<double>();
    metadata.meetingSource = json.at("meeting_source").get<std::string>();
    metadata.failureCode = json.at("failure_code").get<std::string>();
    metadata.failureDetail = json.at("failure_detail").get<std::string>();
    metadata.lasagnaFailureCode = json.at("lasagna_failure_code").get<std::string>();
    metadata.lasagnaFailureDetail = json.at("lasagna_failure_detail").get<std::string>();

    const auto& configJson = json.at("config");
    if (!configJson.is_object()) {
        throw std::runtime_error("segment_to_next config must be an object");
    }
    const std::unordered_set<std::string> configKeys{
         "step_voxels",
         "cone_angle_degrees",
         "cone_angle_step_degrees",
         "cone_grid_size",
         "beam_width",
         "beam_prune_distance_voxels",
         "beam_lookahead_steps",
         "smoothness_weight",
         "smoothness_normal_weight",
         "smoothness_tangent_weight",
         "smoothness_free_angle_degrees",
         "cumulative_smoothness_steps",
         "cumulative_smoothness_tangent_weight",
         "initial_free_angle_degrees",
         "max_step_factor",
         "meeting_accept_max_error_ratio",
         "endpoint_accept_threshold_base_voxels"};
    rejectUnknownKeys(configJson, configKeys, "segment_to_next config");
    auto& config = metadata.config;
    config.stepVoxels = configJson.at("step_voxels").get<double>();
    config.coneAngleDegrees = configJson.at("cone_angle_degrees").get<double>();
    config.coneAngleStepDegrees = configJson.at("cone_angle_step_degrees").get<double>();
    config.coneGridSize = configJson.at("cone_grid_size").get<int>();
    config.beamWidth = configJson.at("beam_width").get<int>();
    config.beamPruneDistanceVoxels = configJson.at("beam_prune_distance_voxels").get<double>();
    config.beamLookaheadSteps = configJson.at("beam_lookahead_steps").get<int>();
    config.smoothnessWeight = configJson.at("smoothness_weight").get<double>();
    config.smoothnessNormalWeight = configJson.at("smoothness_normal_weight").get<double>();
    config.smoothnessTangentWeight = configJson.at("smoothness_tangent_weight").get<double>();
    config.smoothnessFreeAngleDegrees = configJson.at("smoothness_free_angle_degrees").get<double>();
    config.cumulativeSmoothnessSteps = configJson.at("cumulative_smoothness_steps").get<int>();
    config.cumulativeSmoothnessTangentWeight = configJson.at("cumulative_smoothness_tangent_weight").get<double>();
    config.initialFreeAngleDegrees = configJson.at("initial_free_angle_degrees").get<double>();
    config.maxStepFactor = configJson.at("max_step_factor").get<double>();
    config.meetingAcceptMaxErrorRatio =
        configJson.at("meeting_accept_max_error_ratio").get<double>();
    config.endpointAcceptThresholdBaseVoxels = configJson.at("endpoint_accept_threshold_base_voxels").get<double>();
    config.traceToBaseScale = metadata.traceToBaseScale;

    requireFinitePositive(metadata.traceToBaseScale, "trace_to_base_scale");
    if (metadata.meetingErrorBaseVoxels)
        requireFiniteNonNegative(*metadata.meetingErrorBaseVoxels, "meeting_error_base_voxels");
    if (metadata.meetingErrorRatio) {
        requireFiniteNonNegative(*metadata.meetingErrorRatio, "meeting_error_ratio");
    }
    if (metadata.metric)
        requireFiniteNonNegative(*metadata.metric, "metric");
    if (metadata.interpMode == SegmentInterpolationMode::Trace) {
        if (!metadata.metric || !metadata.meetingErrorBaseVoxels ||
            !metadata.meetingErrorRatio || metadata.meetingSource.empty() ||
            !metadata.failureCode.empty() || !metadata.failureDetail.empty() ||
            metadata.normalManifestLocation.empty() ||
            metadata.fiberManifestLocation.empty()) {
            throw std::runtime_error("trace segment_to_next is inconsistent");
        }
    } else if (metadata.meetingErrorBaseVoxels || metadata.meetingErrorRatio ||
               !metadata.meetingSource.empty()) {
        throw std::runtime_error(
            "non-trace segment_to_next cannot contain meeting diagnostics");
    }
    if (metadata.interpMode == SegmentInterpolationMode::Cspline && metadata.metric) {
        throw std::runtime_error("cspline segment_to_next cannot contain metric");
    }
    requireFinitePositive(config.stepVoxels, "step_voxels");
    requireFinitePositive(config.coneAngleDegrees, "cone_angle_degrees");
    requireFinitePositive(config.coneAngleStepDegrees, "cone_angle_step_degrees");
    requireFinitePositive(config.beamPruneDistanceVoxels, "beam_prune_distance_voxels");
    requireFiniteNonNegative(config.smoothnessWeight, "smoothness_weight");
    requireFiniteNonNegative(config.smoothnessNormalWeight, "smoothness_normal_weight");
    requireFiniteNonNegative(config.smoothnessTangentWeight, "smoothness_tangent_weight");
    requireFiniteNonNegative(config.smoothnessFreeAngleDegrees, "smoothness_free_angle_degrees");
    requireFiniteNonNegative(config.cumulativeSmoothnessTangentWeight, "cumulative_smoothness_tangent_weight");
    requireFiniteNonNegative(config.initialFreeAngleDegrees, "initial_free_angle_degrees");
    requireFinitePositive(config.maxStepFactor, "max_step_factor");
    requireFiniteNonNegative(
        config.meetingAcceptMaxErrorRatio,
        "meeting_accept_max_error_ratio");
    if (config.meetingAcceptMaxErrorRatio > 1.0)
        throw std::runtime_error("meeting_accept_max_error_ratio must be at most one");
    requireFinitePositive(config.endpointAcceptThresholdBaseVoxels, "endpoint_accept_threshold_base_voxels");
    if (config.coneGridSize <= 0 || config.beamWidth <= 0 || config.beamLookaheadSteps < 0 || config.cumulativeSmoothnessSteps < 0) {
        throw std::runtime_error("segment_to_next config contains invalid integer values");
    }
    return metadata;
}

nlohmann::json storedControlPointToJson(const StoredControlPoint& control)
{
    nlohmann::json json{{"position", pointToJson(control)}};
    if (control.segmentToNext) {
        json["segment_to_next"] = fiberTraceSegmentMetadataToJson(*control.segmentToNext);
    }
    return json;
}

StoredControlPoint storedControlPointFromJson(const nlohmann::json& json, int fiberVersion)
{
    if (fiberVersion == 1) {
        return StoredControlPoint{pointFromJson(json)};
    }
    if (fiberVersion != 3 || !json.is_object()) {
        throw std::runtime_error("version-3 control point entries must be objects");
    }
    rejectUnknownKeys(json, {"position", "segment_to_next"}, "control point");
    StoredControlPoint control{pointFromJson(json.at("position"))};
    if (json.contains("segment_to_next")) {
        control.segmentToNext = fiberTraceSegmentMetadataFromJson(json.at("segment_to_next"));
    }
    return control;
}

void validateStoredControlPoints(const std::vector<StoredControlPoint>& controls)
{
    if (!controls.empty() && controls.back().segmentToNext) {
        throw std::runtime_error("the final control point cannot contain segment_to_next");
    }
}

std::vector<cv::Vec3d> storedControlPointPositions(const std::vector<StoredControlPoint>& controls)
{
    std::vector<cv::Vec3d> positions;
    positions.reserve(controls.size());
    for (const auto& control : controls) {
        positions.emplace_back(control);
    }
    return positions;
}

std::vector<vc::lasagna::LineControlPoint> optimizerControlPoints(const std::vector<LineControlPoint>& controls)
{
    std::vector<vc::lasagna::LineControlPoint> result;
    result.reserve(controls.size());
    for (const auto& control : controls) {
        result.push_back(control);
    }
    return result;
}

std::vector<LineControlPoint> mergeOptimizerControlPoints(std::vector<vc::lasagna::LineControlPoint> optimized, const std::vector<LineControlPoint>& original)
{
    std::vector<LineControlPoint> result;
    result.reserve(optimized.size());
    for (auto& control : optimized) {
        LineControlPoint merged{control};
        const auto found = std::find_if(original.begin(), original.end(), [&control](const LineControlPoint& candidate) {
            return candidate.volumePoint == control.volumePoint;
        });
        if (found != original.end()) {
            merged.segmentToNext = found->segmentToNext;
        }
        result.push_back(std::move(merged));
    }
    return result;
}

ControlPointCollapseResult collapseControlPointsAtClick(
    const std::vector<LineControlPoint>& controls,
    std::vector<size_t> collapsedIndices,
    double clickedLinePosition,
    const cv::Vec3d& clickedPoint)
{
    if (!std::isfinite(clickedLinePosition) ||
        !std::isfinite(clickedPoint[0]) ||
        !std::isfinite(clickedPoint[1]) ||
        !std::isfinite(clickedPoint[2])) {
        throw std::invalid_argument(
            "control-point collapse requires a finite clicked position");
    }

    std::sort(collapsedIndices.begin(), collapsedIndices.end());
    collapsedIndices.erase(
        std::unique(collapsedIndices.begin(), collapsedIndices.end()),
        collapsedIndices.end());
    if (!collapsedIndices.empty() && collapsedIndices.back() >= controls.size()) {
        throw std::out_of_range("collapsed control-point index is out of range");
    }

    std::vector<bool> collapsed(controls.size(), false);
    for (const size_t index : collapsedIndices) {
        collapsed[index] = true;
    }

    LineControlPoint replacement;
    replacement.linePosition = clickedLinePosition;
    replacement.volumePoint = clickedPoint;
    replacement.optimizedIndex = -1;
    if (!collapsedIndices.empty()) {
        size_t rightmost = collapsedIndices.front();
        for (const size_t index : collapsedIndices) {
            replacement.isSeed = replacement.isSeed || controls[index].isSeed;
            if (controls[index].linePosition > controls[rightmost].linePosition) {
                rightmost = index;
            }
        }
        replacement.segmentToNext = controls[rightmost].segmentToNext;
    }

    struct PendingControl {
        LineControlPoint control;
        std::optional<size_t> oldIndex;
        bool replacement = false;
    };
    std::vector<PendingControl> pending;
    pending.reserve(controls.size() + (collapsedIndices.empty() ? 1 : 0));
    for (size_t index = 0; index < controls.size(); ++index) {
        if (!collapsed[index]) {
            pending.push_back({controls[index], index, false});
        }
    }
    pending.push_back({std::move(replacement), std::nullopt, true});
    std::stable_sort(pending.begin(), pending.end(),
                     [](const PendingControl& lhs, const PendingControl& rhs) {
                         return lhs.control.linePosition < rhs.control.linePosition;
                     });

    ControlPointCollapseResult result;
    result.oldToNewIndices.resize(controls.size());
    result.collapsedOldIndices = std::move(collapsedIndices);
    result.controlPoints.reserve(pending.size());
    for (size_t newIndex = 0; newIndex < pending.size(); ++newIndex) {
        auto& item = pending[newIndex];
        if (item.replacement) {
            result.replacementIndex = newIndex;
        } else {
            result.oldToNewIndices[*item.oldIndex] = newIndex;
        }
        result.controlPoints.push_back(std::move(item.control));
    }
    for (const size_t oldIndex : result.collapsedOldIndices) {
        result.oldToNewIndices[oldIndex] = result.replacementIndex;
    }

    if (!result.replacedExisting() && result.replacementIndex > 0) {
        result.controlPoints[result.replacementIndex].segmentToNext =
            result.controlPoints[result.replacementIndex - 1].segmentToNext;
    }
    if (result.replacementIndex + 1 == result.controlPoints.size()) {
        result.controlPoints[result.replacementIndex].segmentToNext.reset();
    }
    if (result.replacementIndex > 0) {
        result.dirtySegmentIndices.push_back(result.replacementIndex - 1);
    }
    if (result.replacementIndex + 1 < result.controlPoints.size()) {
        result.dirtySegmentIndices.push_back(result.replacementIndex);
    }
    return result;
}

void invalidateSegmentsAdjacentToControl(std::vector<LineControlPoint>& controls, size_t controlIndex)
{
    if (controlIndex >= controls.size()) {
        return;
    }
    std::vector<size_t> order(controls.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&controls](size_t lhs, size_t rhs) {
        return controls[lhs].linePosition < controls[rhs].linePosition;
    });
    const auto found = std::find(order.begin(), order.end(), controlIndex);
    // Goals are CP-owned policy and survive edits. The coordinator always
    // resolves dirty spans from interpGoal, so dropping the descriptor here
    // would silently turn an explicit goal back into global.
    (void)found;
}

namespace
{

// The loader's exact-membership scan (kControlPointMatchEpsilon in
// core/src/Atlas.cpp): controls must be an ordered subset of the dense
// line, so geometry sliced at these indices reloads cleanly.
std::optional<std::vector<size_t>> orderedControlLineIndices(
    const std::vector<cv::Vec3d>& controlPoints,
    const std::vector<cv::Vec3d>& linePoints)
{
    constexpr double kMatchEpsilon = 1.0e-8;
    constexpr double kMaxDistanceSq = kMatchEpsilon * kMatchEpsilon;
    std::vector<size_t> lineIndices;
    lineIndices.reserve(controlPoints.size());
    size_t nextLineIndex = 0;
    for (const auto& control : controlPoints) {
        std::optional<size_t> matched;
        for (size_t j = nextLineIndex; j < linePoints.size(); ++j) {
            const cv::Vec3d delta = control - linePoints[j];
            if (delta.dot(delta) <= kMaxDistanceSq) {
                matched = j;
                break;
            }
        }
        if (!matched) {
            return std::nullopt;
        }
        lineIndices.push_back(*matched);
        nextLineIndex = *matched + 1;
    }
    return lineIndices;
}

}  // namespace

std::optional<FiberSplitPlan> computeFiberSplitPlan(
    const std::vector<cv::Vec3d>& controlPoints,
    const std::vector<cv::Vec3d>& linePoints,
    size_t splitAfterControlIndex)
{
    // Both halves must keep at least 2 control points.
    if (splitAfterControlIndex < 1 ||
        splitAfterControlIndex + 3 > controlPoints.size()) {
        return std::nullopt;
    }
    const auto scannedIndices = orderedControlLineIndices(controlPoints, linePoints);
    if (!scannedIndices) {
        return std::nullopt;
    }
    const std::vector<size_t>& lineIndices = *scannedIndices;

    FiberSplitPlan plan;
    plan.splitAfterControlIndex = splitAfterControlIndex;
    plan.prefixControlCount = splitAfterControlIndex + 1;
    plan.prefixLineCount = lineIndices[splitAfterControlIndex] + 1;
    plan.suffixControlBegin = splitAfterControlIndex + 1;
    plan.suffixLineBegin = lineIndices[splitAfterControlIndex + 1];
    return plan;
}

std::optional<std::pair<bool, int>> remappedSplitControlPointIndex(
    const FiberSplitPlan& plan, int controlPointIndex)
{
    if (controlPointIndex < 0) {
        return std::nullopt;
    }
    const auto index = static_cast<size_t>(controlPointIndex);
    if (index < plan.suffixControlBegin) {
        return std::make_pair(false, controlPointIndex);
    }
    return std::make_pair(true,
                          static_cast<int>(index - plan.suffixControlBegin));
}

std::vector<StoredControlPoint> reversedStoredControlPoints(
    const std::vector<StoredControlPoint>& controls)
{
    const size_t count = controls.size();
    std::vector<StoredControlPoint> reversed;
    reversed.reserve(count);
    for (size_t j = 0; j < count; ++j) {
        StoredControlPoint control{
            static_cast<const cv::Vec3d&>(controls[count - 1 - j])};
        // Span j of the reversed fiber is span (n-2-j) of the original run
        // in the opposite direction; its descriptor travels with it. The
        // new final CP carries none.
        if (j + 1 < count) {
            control.segmentToNext = controls[count - 2 - j].segmentToNext;
        }
        reversed.push_back(std::move(control));
    }
    return reversed;
}

std::optional<FiberMergeGeometry> computeFiberMergeGeometry(
    const std::vector<StoredControlPoint>& aControls,
    const std::vector<cv::Vec3d>& aLine,
    const std::vector<StoredControlPoint>& bControls,
    const std::vector<cv::Vec3d>& bLine)
{
    if (aControls.size() < 2 || bControls.size() < 2) {
        return std::nullopt;
    }
    const auto aIndices =
        orderedControlLineIndices(storedControlPointPositions(aControls), aLine);
    const auto bIndices =
        orderedControlLineIndices(storedControlPointPositions(bControls), bLine);
    if (!aIndices || !bIndices) {
        return std::nullopt;
    }

    FiberMergeGeometry merged;
    merged.controlPoints.reserve(aControls.size() + bControls.size());
    merged.controlPoints.assign(aControls.begin(), aControls.end());
    merged.joinControlIndex = aControls.size() - 1;
    // The join span is fresh geometry with no producer yet; the default
    // global/lasagna descriptor matches the serializer's back-fill and
    // keeps in-memory span consumers valid until the re-fit replaces it.
    FiberTraceSegmentMetadata joinMetadata;
    joinMetadata.interpGoal = SegmentInterpolationGoal::Global;
    joinMetadata.interpMode = SegmentInterpolationMode::Lasagna;
    joinMetadata.message = "lasagna";
    merged.controlPoints[merged.joinControlIndex].segmentToNext =
        std::move(joinMetadata);
    merged.controlPoints.insert(merged.controlPoints.end(),
                                bControls.begin(), bControls.end());

    // Drop both extrapolated tails: line points past a's last CP and before
    // b's first CP would sit between the join CPs and break the strict
    // line-order mapping in the optimizer.
    merged.linePoints.assign(aLine.begin(),
                             aLine.begin() + aIndices->back() + 1);
    merged.linePoints.insert(merged.linePoints.end(),
                             bLine.begin() + bIndices->front(), bLine.end());
    return merged;
}

void invalidateSegmentSplitByInsertedControl(std::vector<LineControlPoint>& controls, size_t insertedIndex)
{
    if (insertedIndex >= controls.size()) {
        return;
    }
    std::vector<size_t> order(controls.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&controls](size_t lhs, size_t rhs) {
        return controls[lhs].linePosition < controls[rhs].linePosition;
    });
    const auto found = std::find(order.begin(), order.end(), insertedIndex);
    if (found != order.begin() && found != order.end()) {
        const size_t previous = *(found - 1);
        controls[insertedIndex].segmentToNext = controls[previous].segmentToNext;
    }
}

}  // namespace vc3d::line_annotation
