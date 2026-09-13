#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <memory>
#include <string>
#include <vector>

class PlaneSurface;
class QuadSurface;

namespace vc::lasagna {

// Along-strip resampling target and declared along-line grid density. Also the
// control-point collapse radius in VC3D (LineAnnotationController), which keeps
// every control span at least this long so no span is ever shorter than one
// strip column (the #1484 strip model: a span shorter than the target is drawn
// as one full column and would be stretched).
inline constexpr double kLineViewAlongSamplingDistanceBaseVoxels = 8.0;
// Cross-row spacing of the fixed seven-row ribbons (192 base voxels first to
// last row). Independent of the along-strip target; along <= cross.
inline constexpr double kLineViewCrossRowSpacingBaseVoxels = 32.0;
inline constexpr int kLineViewCrossSampleCount = 7;

struct LineViewConfig {
    // Derived ribbons retain every annotation control point and subdivide the
    // optimized polyline between adjacent controls as closely as possible to
    // this spacing. The declared along-strip scale always uses this target, so
    // a shorter control-point span occupies one full display interval.
    double targetSpacingBaseVoxels = kLineViewAlongSamplingDistanceBaseVoxels;
    // Fractional indices into LineModel::points. Line endpoints are always
    // retained as additional supports. Empty retains every line point for
    // callers that do not have separate annotation-control metadata.
    std::vector<double> controlPointLinePositions;
    // Optional per-line-point oriented sheet normals, indexed like
    // LineModel::points (entries may be NaN/zero where unavailable).
    // When non-empty and size-matched, one global sign flip is applied so the
    // frame mesh normals AND the display up vectors agree with these on a
    // cosine-weighted majority. Empty/mismatched/all-invalid -> legacy signs.
    std::vector<cv::Vec3f> orientedPointNormals;
    // Build one PlaneSurface per line point into lineZSlices. VC3D's line
    // annotation only consumes lineUpVectors, so it opts out - a 2000-point
    // fiber otherwise allocates 2000 shared_ptr planes per view rebuild for
    // nothing. Defaults on for the existing consumers/tests.
    bool buildLineZSlices = true;
};

// Maps the original LineModel point-index coordinate to the ribbon grid and
// back. Each configured annotation control point and each line endpoint is a
// grid support, with span-local subdivisions between supports. Fractional
// original positions interpolate within an optimized-line segment. Consecutive
// duplicate line points share one arclength; inversion at that arclength
// returns the first point in the duplicate run.
struct LineStripPositionMap {
    std::vector<double> originalArclengths;
    std::vector<double> stripGridArclengths;
    double totalArclength = 0.0;
    // Fixed target spacing used for the QuadSurface's scalar grid-density
    // metadata. Exact source mapping uses stripGridArclengths.
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
