#pragma once

// The pure geometry behind the Fiber Map's "Link <A> and <B>": where two
// drawn fibers meet on the map, the exact inversion of a drawn point to a
// position on the fiber's source line points, the closest 3D pass of the two
// fibers inside a bounded window around that position, and which control
// point each fiber links at (an existing one, or a new one placed clear of
// every gate). Everything here is a function of its inputs: the controller
// supplies line points, cumulative arclengths, control line indices, link
// flags, tag flags, gap ranges and a focus predicate; nothing reaches into
// sessions or the scene.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <QPointF>
#include <opencv2/core/types.hpp>

#include "FiberNetworkLayout.hpp"

namespace vc3d::fiber_map::autolink
{

// A meeting of the two DRAWN polylines: a crossing (the one nearest
// `preferNear` when there are several), else the closest approach of the two
// polylines when it is within `maxApproachVx`.
struct MeetingSide {
    std::size_t runIndex = 0;
    // The hit drawn segment [sample, sample + 1] of that run's display points
    // and the parameter along it.
    std::size_t sample = 0;
    double t = 0.0;
    // The hit's raw unrolled arclength, interpolated between the two samples'
    // arclengths (controlArclength where displayRunPoints replaced an overlap
    // sample by the control's drawn position).
    double arclength = 0.0;
};

struct MapMeeting {
    QPointF point;
    bool crossing = false;
    // Scene distance between the two polylines at the meeting (0 for a crossing).
    double distance = 0.0;
    MeetingSide a;
    MeetingSide b;
};

[[nodiscard]] std::optional<MapMeeting> findMapMeeting(const PlacedFiber& a,
                                                       const PlacedFiber& b,
                                                       QPointF preferNear,
                                                       double maxApproachVx);

// Exact inversion of a raw unrolled arclength to a fractional line position:
// `raw` is the fiber's cumulative unrolled arclength per source line point
// (unrolledLineArclengths, monotone non-decreasing). The FIRST exact match
// wins on a plateau (a radial run of the line resolves to its first point);
// otherwise the position interpolates between the predecessor of the first
// element >= s and that element. No tolerance.
[[nodiscard]] double sourcePositionAtArclength(const std::vector<double>& raw, double s);

// Fractional line position + its interpolated point.
[[nodiscard]] cv::Vec3d pointAtLinePosition(const std::vector<cv::Vec3d>& linePoints,
                                            double linePosition);

// Cumulative 3D arclength per line point (index 0 = 0). The controller can
// pass fiber_slice::cumulativePolylineArclengths' array; this is the same
// quantity for callers without that dependency.
[[nodiscard]] std::vector<double> cumulativeArclengths(const std::vector<cv::Vec3d>& linePoints);

// The closest pass of the two fibers, searched only inside the line-point
// ranges whose cumulative 3D arclength lies within +-windowVx of the
// arclength at each fiber's source position (D3: the only along-fiber bound,
// so a multi-winding fiber never searches another winding).
struct ClosestPass {
    double positionA = 0.0;
    double positionB = 0.0;
    cv::Vec3d pointA{0.0, 0.0, 0.0};
    cv::Vec3d pointB{0.0, 0.0, 0.0};
    double distance = 0.0;
    // The searched windows as fractional line positions [lo, hi] (exact
    // +-window clipping), which the placement search must stay inside, and
    // the enclosing whole line-index ranges for reporting.
    double windowLoA = 0.0, windowHiA = 0.0, windowLoB = 0.0, windowHiB = 0.0;
    std::size_t firstA = 0, lastA = 0, firstB = 0, lastB = 0;
};

struct FiberSide {
    const std::vector<cv::Vec3d>* linePoints = nullptr;
    const std::vector<double>* cumulative = nullptr;
    double position = 0.0;
    double windowVx = 0.0;
};

[[nodiscard]] std::optional<ClosestPass> findClosestPass(const FiberSide& a, const FiberSide& b);

// Which control point of one fiber the link uses.
struct ControlChoiceInput {
    const std::vector<cv::Vec3d>* linePoints = nullptr;
    const std::vector<double>* cumulative = nullptr;
    // The loader's ordered line index per control (orderedControlPointLineIndices).
    std::vector<std::size_t> controlLineIndex;
    // Per control: any branch link (ordinary or adjacent) on it.
    std::vector<bool> linked;
    // Per control: carries the kollesis termination tag (placement beyond
    // such an end is refused, as the interactive handler refuses it).
    std::vector<bool> kollesisTerminations;
    // Gap spans as [lower control, upper control] index pairs (line order):
    // nothing is placed inside one.
    std::vector<std::pair<std::size_t, std::size_t>> gapSpans;
    // The active focus bounds, if any: the predicate every candidate is
    // checked against, and the box itself so the exact entry/exit positions
    // of each line segment through it are candidates too.
    std::function<bool(const cv::Vec3d&)> inFocusBounds;
    std::optional<std::pair<cv::Vec3d, cv::Vec3d>> focusBox;
    // The pass on this fiber and the exact window (fractional line
    // positions) it was found in; nothing is placed outside the window.
    double passPosition = 0.0;
    double windowLo = 0.0;
    double windowHi = 0.0;
    // D4: an unlinked control within this arclength of the pass is reused.
    double reuseVx = 0.0;
    // A new control is never placed within this arclength of an existing one
    // (kMinimumControlPointSpacingBaseVoxels): no collapse, strict ordering.
    double minSpacingVx = 0.0;
    // The position-identity tolerance of the link format: a reused control
    // with another control within it, or a new control within it of any
    // control, is refused (the save canonicalizer identifies controls by
    // position under this tolerance).
    double ambiguityTolerance = 1.0e-6;
};

struct ControlChoice {
    enum class Kind { Reuse, Place, Refused };
    Kind kind = Kind::Refused;
    // Reuse: the stored control index. Place: the line position and point.
    int controlIndex = -1;
    double linePosition = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    // Refused: why, in words.
    std::string reason;
};

[[nodiscard]] ControlChoice chooseControlPoint(const ControlChoiceInput& input);

}  // namespace vc3d::fiber_map::autolink
