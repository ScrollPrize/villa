#pragma once

#include <QPointF>
#include <QString>

#include <opencv2/core/matx.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Extrinsic unroll of manually linked H/V fiber networks about the scroll
// umbilicus, ported from scripts/fiber_network_unroll.py.
//
//     x = unwrapped angle about the umbilicus * the network's median radius
//     y = z
//
// Each fiber's unwrapped angle carries its own arbitrary multiple of 2*pi, so
// the offsets are made mutually consistent by walking the link graph and
// snapping each fiber to the nearest whole turn of an already-placed
// neighbour (best-agreeing link first). Crossings then coincide and loops
// close by construction: there is no solver and no accumulated drift. A link
// whose endpoints still disagree by a large fraction of a turn afterwards was
// annotated on the wrong winding and is reported as suspect.
namespace vc3d::fiber_map
{

struct InputLink {
    int controlPointIndex = -1;
    uint64_t branchFiberId = 0;
    int branchControlPointIndex = -1;
    // The annotation keeps this in sync on both reciprocal refs; the layout
    // ORs the two sides on dedup anyway.
    bool pending = false;
};

struct InputFiber {
    uint64_t id = 0;
    // Stable identity, carried through so callers can act on a placed fiber
    // after the runtime ids have been reassigned.
    std::string fileName;
    QString label;
    char hvTag = '?';
    std::vector<cv::Vec3d> controlPoints;
    std::vector<cv::Vec3d> linePoints;
    // Per control-point-span "was fiber-model traced"; anything else is only
    // an interpolation. Empty or mismatched renders as a single traced run.
    std::vector<bool> tracedSegments;
    // Raw directed refs; the layout dedupes reciprocal pairs.
    std::vector<InputLink> links;
};

struct LayoutParams {
    double voxelSizeUm = 2.4;
    int minFibers = 3;
    int maxNetworks = 3;
    double suspectTurns = 0.25;
    // Gaussian arclength sigma in mm for de-bumping the drawn fibers; 0
    // disables smoothing.
    double smoothMm = 1.2;
    double panelTickCm = 5.0;
    double minGapCm = 1.0;
};

// One styling run of a fiber, in cm with +y = +z.
struct Run {
    bool traced = true;
    std::vector<QPointF> points;
};

struct PlacedFiber {
    uint64_t id = 0;
    std::string fileName;
    QString label;
    char hvTag = '?';
    std::vector<Run> runs;
    // Control-point positions read off the smoothed geometry, so they land
    // exactly on the drawn curve.
    std::vector<QPointF> controlPoints;
};

struct PlacedLink {
    uint64_t fiberA = 0;
    int cpA = -1;
    uint64_t fiberB = 0;
    int cpB = -1;
    QPointF a;
    QPointF b;
    // |dTheta| / 2pi after placement.
    double turnErr = 0.0;
    bool suspect = false;
    // True when either input ref of the deduped pair still awaits review.
    bool pending = false;
};

struct WindingMark {
    double xCm = 0.0;
    int number = 0;
};

struct PlacedNetwork {
    // 0-based index into the size-sorted list of networks with >= minFibers.
    int networkIndex = 0;
    double rRefCm = 0.0;
    double x0Cm = 0.0;
    double x1Cm = 0.0;
    std::vector<PlacedFiber> fibers;
    std::vector<PlacedLink> links;
    // Continuous numbering across panels.
    std::vector<WindingMark> windings;
};

struct Result {
    // Ordered inner -> outer by median umbilicus radius, panel offsets applied.
    std::vector<PlacedNetwork> networks;
    double widthCm = 0.0;
    double yMinCm = 0.0;
    double yMaxCm = 0.0;
    // Networks with >= minFibers, before the top-N cut.
    int qualifyingNetworkCount = 0;
    int suspectLinkCount = 0;
};

// umbilicusCenters are dense volume-frame centers (x, y, z), one per z slice;
// an empty list means the network cannot be unrolled and yields an empty
// result.
[[nodiscard]] Result buildLayout(const std::vector<InputFiber>& fibers,
                                 const std::vector<cv::Vec3f>& umbilicusCenters,
                                 const LayoutParams& params);

} // namespace vc3d::fiber_map
