#include "LineAnnotationFiberSegments.hpp"

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

bool isAcceptedNativeTrace(const FiberTraceSegmentMetadata& metadata) noexcept
{
    return metadata.outcome ==
        FiberTraceSegmentMetadata::Outcome::AcceptedNative;
}

bool isAcceptedNativeTrace(
    const std::optional<FiberTraceSegmentMetadata>& metadata) noexcept
{
    return metadata.has_value() && isAcceptedNativeTrace(*metadata);
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
        if (std::isfinite(result.meetingErrorBaseVoxels))
            metadata.meetingErrorBaseVoxels = result.meetingErrorBaseVoxels;
        if (std::isfinite(result.meetingErrorRatio))
            metadata.meetingErrorRatio = result.meetingErrorRatio;
        metadata.meetingSource = result.meetingSource;
    } else {
        metadata.failureCode = result.reason.empty()
            ? "fusion_failed"
            : result.reason;
        metadata.failureDetail = result.detail;
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
    metadata.failureCode = "trace_exception";
    metadata.failureDetail = std::move(detail);
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
    if (!request.predictions || !request.baseNormalSampler ||
        !request.traceNormalSampler) {
        throw std::invalid_argument(
            "fiber-mode optimization requires prediction and normal samplers");
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
    if (request.retraceAll) {
        for (auto& control : request.controlPoints) {
            control.segmentToNext.reset();
        }
    }

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
        replaceOpenTailsWithNative(
            request, coordinates, controlIndex, controlIndex, output);
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
    std::vector<bool> nativeSpan(spans.size(), false);
    for (size_t spanIndex = 0; spanIndex < spans.size(); ++spanIndex) {
        auto& owner = request.controlPoints[spanIndex];
        const size_t first = originalControlIndices[spanIndex];
        const size_t last = originalControlIndices[spanIndex + 1];
        if (isAcceptedNativeTrace(owner.segmentToNext)) {
            spans[spanIndex] = inclusiveLineSpan(request.linePointsBase, first, last);
            nativeSpan[spanIndex] = true;
            ++output.nativeSegments;
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
        nativeSpan[spanIndex] = true;
        ++output.nativeSegments;
    }

    std::vector<cv::Vec3d> stitched;
    stitched.insert(stitched.end(),
                    request.linePointsBase.begin(),
                    request.linePointsBase.begin() +
                        static_cast<std::ptrdiff_t>(originalControlIndices.front()));
    std::vector<int> controlIndices;
    controlIndices.reserve(request.controlPoints.size());
    for (size_t spanIndex = 0; spanIndex < spans.size(); ++spanIndex) {
        const auto& span = spans[spanIndex];
        if (spanIndex == 0) {
            controlIndices.push_back(static_cast<int>(stitched.size()));
            stitched.insert(stitched.end(), span.begin(), span.end());
        } else {
            stitched.insert(stitched.end(), span.begin() + 1, span.end());
        }
        controlIndices.push_back(static_cast<int>(stitched.size()) - 1);
    }
    stitched.insert(
        stitched.end(),
        request.linePointsBase.begin() +
            static_cast<std::ptrdiff_t>(originalControlIndices.back() + 1),
        request.linePointsBase.end());
    for (size_t index = 0; index < request.controlPoints.size(); ++index) {
        request.controlPoints[index].optimizedIndex = controlIndices[index];
        request.controlPoints[index].linePosition =
            static_cast<double>(controlIndices[index]);
    }

    std::vector<std::pair<int, int>> protectedSpans;
    std::vector<vc::lasagna::LineControlPointHardDirectionConstraint>
        hardDirections;
    for (size_t index = 0; index < nativeSpan.size(); ++index) {
        if (nativeSpan[index]) {
            protectedSpans.emplace_back(
                static_cast<int>(index), static_cast<int>(index + 1));
            if (spans[index].size() < 2) {
                throw std::runtime_error(
                    "native fiber span has fewer than two dense points");
            }
            const cv::Vec3d leftIntoNative = normalizedFiberEndpointDirection(
                request.controlPoints[index].volumePoint,
                spans[index],
                true);
            const cv::Vec3d rightIntoNative = normalizedFiberEndpointDirection(
                request.controlPoints[index + 1].volumePoint,
                spans[index],
                false);
            if (index == 0 || !nativeSpan[index - 1]) {
                hardDirections.push_back({
                    static_cast<int>(index),
                    vc::lasagna::LineControlPointSide::Before,
                    -leftIntoNative,
                });
            }
            if (index + 1 == nativeSpan.size() || !nativeSpan[index + 1]) {
                hardDirections.push_back({
                    static_cast<int>(index + 1),
                    vc::lasagna::LineControlPointSide::After,
                    -rightIntoNative,
                });
            }
        }
    }
    auto optimizerControls = optimizerControlPoints(request.controlPoints);
    vc::lasagna::LineOptimizer optimizer(*request.baseNormalSampler);
    auto reinitialized = optimizer.reinitializeAndOptimizeExistingLine(
        std::move(stitched),
        std::move(optimizerControls),
        controlIndices,
        controlIndices[controlIndices.size() / 2],
        request.lasagnaConfig,
        std::move(protectedSpans),
        std::move(hardDirections));
    if (reinitialized.failed) {
        throw std::runtime_error(
            "Lasagna fallback failed: " + reinitialized.failureReason);
    }

    if (reinitialized.fixedPointIndices.size() != request.controlPoints.size()) {
        throw std::runtime_error(
            "Lasagna fallback returned an invalid control-point index map");
    }
    const int firstControl = reinitialized.fixedPointIndices.front();
    const int lastControl = reinitialized.fixedPointIndices.back();
    output.optimization = std::move(reinitialized.optimization);
    replaceOpenTailsWithNative(
        request, coordinates, firstControl, lastControl, output);
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
        {"outcome",
         acceptedNative
             ? "accepted_native"
             : "lasagna_fallback"},
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
    const bool previousVersion = metadataVersion == 1 && tracerVersion == 1;
    const bool currentVersion =
        metadataVersion == FiberTraceSegmentMetadata::MetadataVersion &&
        tracerVersion == FiberTraceSegmentMetadata::TracerVersion;
    if (!previousVersion && !currentVersion)
        throw std::runtime_error("unsupported segment_to_next metadata/tracer version");
    if (previousVersion) {
        rejectUnknownKeys(
            json,
            {"optimizer", "metadata_version", "tracer_version",
             "normal_manifest", "fiber_manifest", "trace_to_base_scale",
             "max_endpoint_error_base_voxels", "config"},
            "segment_to_next");
    } else {
        rejectUnknownKeys(
            json,
            {"optimizer", "metadata_version", "tracer_version", "outcome",
             "normal_manifest", "fiber_manifest", "trace_to_base_scale",
             "meeting_error_base_voxels", "meeting_error_ratio",
             "meeting_source", "failure_code", "failure_detail", "config"},
            "segment_to_next");
    }

    FiberTraceSegmentMetadata metadata;
    metadata.normalManifestLocation = json.at("normal_manifest").get<std::string>();
    metadata.fiberManifestLocation = json.at("fiber_manifest").get<std::string>();
    if (metadata.normalManifestLocation.empty() || metadata.fiberManifestLocation.empty()) {
        throw std::runtime_error("segment_to_next manifest locations must not be empty");
    }
    metadata.traceToBaseScale = json.at("trace_to_base_scale").get<double>();
    if (previousVersion) {
        metadata.outcome = FiberTraceSegmentMetadata::Outcome::AcceptedNative;
        metadata.meetingErrorBaseVoxels =
            json.at("max_endpoint_error_base_voxels").get<double>();
        metadata.meetingSource = "legacy_endpoint";
    } else {
        const std::string outcome = json.at("outcome").get<std::string>();
        if (outcome == "accepted_native") {
            metadata.outcome = FiberTraceSegmentMetadata::Outcome::AcceptedNative;
        } else if (outcome == "lasagna_fallback") {
            metadata.outcome = FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
        } else {
            throw std::runtime_error("unsupported segment_to_next outcome");
        }
        if (isAcceptedNativeTrace(metadata)) {
            if (!json.at("meeting_error_base_voxels").is_null()) {
                metadata.meetingErrorBaseVoxels =
                    json.at("meeting_error_base_voxels").get<double>();
            }
            if (!json.at("meeting_error_ratio").is_null()) {
                metadata.meetingErrorRatio =
                    json.at("meeting_error_ratio").get<double>();
            }
            metadata.meetingSource = json.at("meeting_source").get<std::string>();
        }
        metadata.failureCode = json.at("failure_code").get<std::string>();
        metadata.failureDetail = json.at("failure_detail").get<std::string>();
    }

    const auto& configJson = json.at("config");
    if (!configJson.is_object()) {
        throw std::runtime_error("segment_to_next config must be an object");
    }
    std::unordered_set<std::string> configKeys{
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
         "endpoint_accept_threshold_base_voxels"};
    configKeys.insert(previousVersion
        ? "fusion_gap_factor"
        : "meeting_accept_max_error_ratio");
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
    if (!previousVersion) {
        config.meetingAcceptMaxErrorRatio =
            configJson.at("meeting_accept_max_error_ratio").get<double>();
    }
    config.endpointAcceptThresholdBaseVoxels = configJson.at("endpoint_accept_threshold_base_voxels").get<double>();
    config.traceToBaseScale = metadata.traceToBaseScale;

    requireFinitePositive(metadata.traceToBaseScale, "trace_to_base_scale");
    if (metadata.meetingErrorBaseVoxels)
        requireFiniteNonNegative(*metadata.meetingErrorBaseVoxels, "meeting_error_base_voxels");
    if (metadata.meetingErrorRatio) {
        requireFiniteNonNegative(*metadata.meetingErrorRatio, "meeting_error_ratio");
    }
    if (!previousVersion) {
        if (metadata.outcome == FiberTraceSegmentMetadata::Outcome::AcceptedNative) {
            if (!metadata.meetingErrorBaseVoxels || !metadata.meetingErrorRatio ||
                metadata.meetingSource.empty() || !metadata.failureCode.empty() ||
                !metadata.failureDetail.empty()) {
                throw std::runtime_error("accepted segment_to_next outcome is inconsistent");
            }
        } else if (metadata.failureCode.empty()) {
            throw std::runtime_error("fallback segment_to_next requires a failure_code");
        }
        if (metadata.meetingErrorBaseVoxels.has_value() !=
            metadata.meetingErrorRatio.has_value()) {
            throw std::runtime_error(
                "segment_to_next meeting error and ratio must be present together");
        }
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
    if (fiberVersion != 2 || !json.is_object()) {
        throw std::runtime_error("version-2 control point entries must be objects");
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
    controls[controlIndex].segmentToNext.reset();
    if (found != order.begin() && found != order.end()) {
        controls[*(found - 1)].segmentToNext.reset();
    }
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
        controls[*(found - 1)].segmentToNext.reset();
    }
}

}  // namespace vc3d::line_annotation
