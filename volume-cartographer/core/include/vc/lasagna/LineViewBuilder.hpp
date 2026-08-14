#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <memory>
#include <string>
#include <vector>

class PlaneSurface;
class QuadSurface;

namespace vc::lasagna {

struct LineViewConfig {
    // Derived ribbons are uniformly resampled in arclength. This is a view
    // parameter in level-0/base-volume voxels, independent of stored point or
    // optimizer spacing.
    double targetSpacingBaseVoxels = 50.0;
    // Non-positive values retain the legacy automatic strip height: cross-row
    // spacing matches the median optimized control-point step.
    double surfaceHalfWidth = 0.0;
    double sideSliceHalfDepth = 0.0;
    int crossSamples = 21;
    // Optional per-control-point oriented sheet normals, indexed like
    // LineModel::points (entries may be NaN/zero where unavailable).
    // When non-empty and size-matched, one global sign flip is applied so the
    // frame mesh normals AND the display up vectors agree with these on a
    // cosine-weighted majority. Empty/mismatched/all-invalid -> legacy signs.
    std::vector<cv::Vec3f> orientedPointNormals;
};

// Maps the original LineModel point-index coordinate to the uniformly sampled
// ribbon grid and back. Fractional original positions interpolate within an
// original segment. Consecutive duplicate points share one arclength; inversion
// at that arclength returns the first point in the duplicate run.
struct LineStripPositionMap {
    std::vector<double> originalArclengths;
    double totalArclength = 0.0;
    double stripGridSpacingBaseVoxels = 0.0;
    size_t stripGridColumnCount = 0;

    [[nodiscard]] bool valid() const;
    [[nodiscard]] double originalPositionToStripGridColumn(double originalPosition) const;
    [[nodiscard]] double stripGridColumnToOriginalPosition(double stripGridColumn) const;
};

struct LineViewSurfaces {
    std::shared_ptr<QuadSurface> lineSurface;
    std::shared_ptr<QuadSurface> lineSideSlice;
    std::vector<std::shared_ptr<PlaneSurface>> lineZSlices;
    std::vector<cv::Vec3f> lineUpVectors;
    LineStripPositionMap stripPositionMap;
};

struct LineViewFrameIssue {
    size_t index = 0;
    double rollDeltaRadians = 0.0;
    double normalContinuityDot = 1.0;
    double sideContinuityDot = 1.0;
    double sampledAxisContinuityDot = 1.0;
    double meshToSampledAxisDot = 1.0;
    double displayUpRollDeltaRadians = 0.0;
    double displayUpContinuityDot = 1.0;
    std::string reason;
};

struct LineViewFrameDiagnostics {
    size_t frameCount = 0;
    double maxAbsRollDeltaRadians = 0.0;
    double minNormalContinuityDot = 1.0;
    double minSideContinuityDot = 1.0;
    double minSampledAxisContinuityDot = 1.0;
    double minMeshToSampledAxisDot = 1.0;
    double maxAbsDisplayUpRollDeltaRadians = 0.0;
    double minDisplayUpContinuityDot = 1.0;
    std::vector<LineViewFrameIssue> issues;
};

LineViewSurfaces buildLineViewSurfaces(const LineModel& line,
                                       const LineViewConfig& config = {});

LineViewFrameDiagnostics diagnoseLineViewFrames(const LineModel& line,
                                                const LineViewConfig& config = {});

} // namespace vc::lasagna
