#pragma once

#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationGeneratedViews.hpp"

#include <optional>

namespace vc3d::line_annotation
{

// Fills a span label from the span's stored metadata: mode marker, status
// message, gap/damaged tags and trace meeting/failure diagnostics, none of
// which need normals. The normal-alignment measurement (the stored Lasagna
// metric) is filled only when `normalMeasurements` is set: a session without
// a normal source (inspection frame, e.g. cspline-only annotation) shows the
// descriptors but no normal-dependent number.
inline void fillSpanMetricFromSegment(GeneratedSpanAlignmentMetric& metric,
                                      const FiberTraceSegmentMetadata& segment,
                                      bool normalMeasurements)
{
    using Kind = GeneratedSpanAlignmentMetric::Kind;
    metric.modeMarker = segmentInterpolationModeMarker(segment.interpMode);
    metric.message = segment.message;
    const std::optional<FiberTraceSegmentMetadata> optionalSegment(segment);
    metric.gap = spanIsGap(optionalSegment);
    metric.damaged = spanIsDamaged(optionalSegment);
    if (segment.interpMode == SegmentInterpolationMode::Trace)
        metric.kind = Kind::NativeMeetingError;
    else if (segment.interpMode == SegmentInterpolationMode::Cspline)
        metric.kind = Kind::Cspline;
    else
        metric.kind = Kind::LasagnaNormalAlignment;
    if (segment.metric && normalMeasurements) {
        metric.available = true;
        metric.maxErrorDegrees = *segment.metric;
    }
    if (segment.meetingErrorBaseVoxels)
        metric.meetingErrorBaseVoxels = *segment.meetingErrorBaseVoxels;
    if (segment.meetingErrorRatio)
        metric.meetingErrorRatio = *segment.meetingErrorRatio;
    metric.meetingSource = segment.meetingSource;
    metric.failureCode = segment.failureCode;
    metric.failureDetail = segment.failureDetail;
}

}  // namespace vc3d::line_annotation
