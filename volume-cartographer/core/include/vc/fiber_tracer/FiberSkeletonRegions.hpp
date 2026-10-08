#pragma once

// Splits a reduced skeleton graph into unambiguous line segments and ambiguous
// regions, and places the split points ("ports") that separate them.
//
// An ambiguous region is one group of junction nodes, as labelled by graph
// reduction. Its interior structure stays in the graph, so a route through it
// is an ordinary graph path. Every edge leaving the group is cut at
// `entryOffsetBase` beyond the group's node, so the split point sits on clean
// line topology. An edge joining two regions that is too short to carry two
// cuts instead gets one shared port at its midpoint, which connects the two
// regions directly.

#include "vc/fiber_tracer/FiberSkeleton.hpp"
#include "vc/fiber_tracer/FiberSkeletonFields.hpp"
#include "vc/fiber_tracer/FiberSkeletonRefine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

inline constexpr std::size_t kFiberSkeletonNoIndex =
    static_cast<std::size_t>(-1);

struct FiberSkeletonPort {
    std::size_t id = 0;
    std::size_t edgeId = 0;
    // Regions this port guards. The second entry is used only by a shared
    // midpoint port on an edge too short for two cuts.
    std::array<std::size_t, 2> regionIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};
    // The region node its edge leaves from, one per guarded region, and the
    // arc from that node along the edge to this port.
    std::array<std::size_t, 2> nodeIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};
    std::array<double, 2> arcFromNodeBase{0.0, 0.0};
    cv::Vec3d positionBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d initialPositionBaseXYZ{0.0, 0.0, 0.0};
    // Where the fibre may run through this split point. The section of the
    // prediction here is a ribbon; presence fixes the fibre across it and says
    // nothing about where it runs along it. A narrow section gives one
    // candidate at its centre; a wide one, where fibres have merged, gives one
    // per fibre width spread across the ribbon. Which candidate is the fibre is
    // a routing choice, decided with the region's connections, and
    // `positionBaseXYZ` is the chosen one once that choice is made.
    std::vector<cv::Vec3d> candidatesBaseXYZ;
    // One flag per candidate: whether it stands for a fibre. The extremes of a
    // wide section are fibres; its centre is the seam between them, kept as a
    // fallback. A narrow section's single centre is the fibre.
    std::vector<std::uint8_t> candidateIsFibre;
    double sectionWidthBase = 0.0;
    // Unit edge tangent at the port, oriented away from `regionIds[0]`.
    cv::Vec3d tangentXYZ{0.0, 0.0, 0.0};
    bool shared = false;
};

// One refinement of a line between a particular candidate at each end.
struct FiberSkeletonLineVariant {
    std::array<std::size_t, 2> candidates{0, 0};
    std::vector<cv::Vec3d> pointsBaseXYZ;
    double lengthBase = 0.0;
    FiberSkeletonCostBreakdown cost;
    bool refined = false;
    int iterations = 0;
    bool converged = false;
};

// An unambiguous stretch: the part of one edge outside every region. It is
// refined once per pair of end candidates; `pointsBaseXYZ` and the fields
// beside it describe the selected variant.
struct FiberSkeletonLine {
    std::size_t id = 0;
    std::size_t edgeId = 0;
    std::vector<cv::Vec3d> initialPointsBaseXYZ;
    std::vector<cv::Vec3d> pointsBaseXYZ;
    double initialLengthBase = 0.0;
    double lengthBase = 0.0;
    // Port at each end, or `kFiberSkeletonNoIndex` for a free terminal end.
    std::array<std::size_t, 2> portIds{
        kFiberSkeletonNoIndex, kFiberSkeletonNoIndex};
    FiberSkeletonCostBreakdown cost;
    bool refined = false;
    int iterations = 0;
    bool converged = false;
    std::vector<FiberSkeletonLineVariant> variants;
    // Per end (0 start, 1 end) and per candidate of that end's port, the
    // cheapest variant with that candidate there: the route a fibre takes when
    // it enters the line at that sub-port. Nothing is chosen ahead of
    // tracing. `pointsBaseXYZ` and the fields beside it describe the variant
    // entered at the start port's first candidate.
    std::array<std::vector<std::size_t>, 2> entryVariants;

    [[nodiscard]] const FiberSkeletonLineVariant* variantFor(
        std::size_t startCandidate, std::size_t endCandidate) const noexcept
    {
        for (const auto& variant : variants) {
            if (variant.candidates[0] == startCandidate &&
                variant.candidates[1] == endCandidate) {
                return &variant;
            }
        }
        return nullptr;
    }
};

struct FiberSkeletonRegion {
    std::size_t id = 0;
    // The junction group this region is, and the graph it routes over.
    std::size_t group = 0;
    std::vector<std::size_t> nodeIds;
    std::vector<std::size_t> interiorEdgeIds;
    cv::Vec3d centerBaseXYZ{0.0, 0.0, 0.0};
    std::vector<std::size_t> portIds;
    // Sampled along the region's own geometry; written to the artifacts.
    std::vector<cv::Vec3d> supportPointsBaseXYZ;
};

struct FiberSkeletonTopology {
    std::vector<FiberSkeletonRegion> regions;
    std::vector<FiberSkeletonPort> ports;
    std::vector<FiberSkeletonLine> lines;
    std::size_t sharedMidpointPorts = 0;
    std::size_t droppedShortEdges = 0;
    // Line variants whose initial geometry came from the path search, and
    // those that fell back to the skeleton stretch with blended ends.
    std::size_t searchedInitials = 0;
    std::size_t unsearchedInitials = 0;
};

// The thresholded support in the plane normal to `tangent`, restricted to the
// blob the point lies in. The fibre section is a ribbon: narrow across the
// sheet and wide along it. `shortAxisXYZ` is the narrow direction, and it is
// the only one in which presence says where the fibre is.
struct FiberSkeletonSection {
    bool valid = false;
    cv::Vec3d centreBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d longAxisXYZ{0.0, 0.0, 0.0};
    cv::Vec3d shortAxisXYZ{0.0, 0.0, 0.0};
    double longSigmaBase = 0.0;
    double shortSigmaBase = 0.0;
    // Extent along the long axis, measured from the centroid.
    double longMinBase = 0.0;
    double longMaxBase = 0.0;
    double rmsRadiusBase = 0.0;
    double areaBase = 0.0;
};

[[nodiscard]] FiberSkeletonSection fiberSkeletonSectionAt(
    const FiberSkeletonCrop& crop, int thresholdU8, const cv::Vec3d& pointBaseXYZ,
    const cv::Vec3d& tangentXYZ, double searchRadiusBase);

// A few steps of a small presence-weighted centroid in the plane normal to
// `tangent`, which moves a point on the edge of the mask onto the fibre it
// belongs to.
[[nodiscard]] cv::Vec3d refineFiberSkeletonPointInPlane(
    const FiberSkeletonCrop& crop, int thresholdU8, const cv::Vec3d& pointBaseXYZ,
    const cv::Vec3d& tangentXYZ, double radiusBase, int iterations);

// Cheapest path through the thresholded prediction between two points,
// restricted to a corridor around the skeleton piece the stretch belongs to,
// under a simplified form of the tracing cost: the shared discrete metric's
// presence and direction-alignment terms, length-scaled, with the smoothness
// terms off, since a voxel search carries no incoming direction. Returns
// false when the mask does not connect the two points inside the corridor.
struct FiberSkeletonPathSearchConfig {
    double corridorRadiusBase = 16.0;
    int thresholdU8 = 128;
    float invalidPredictionCostPerVoxel = 4.0f;
};

bool fiberSkeletonSearchPath(
    FiberSkeletonFieldProvider& fields, const FiberSkeletonPathSearchConfig& config,
    const std::vector<cv::Vec3d>& corridorBaseXYZ, const cv::Vec3d& startBaseXYZ,
    const cv::Vec3d& endBaseXYZ, std::vector<cv::Vec3d>& pathBaseXYZ);

struct FiberSkeletonSplitConfig {
    double entryOffsetBase = 32.0;
    double sectionRadiusBase = 24.0;
    int thresholdU8 = 128;
    // An extreme of the section closer than this to its centre is the same
    // fibre and is not a candidate.
    double candidateMinOffsetBase = 6.0;
    // Local centroid applied to every candidate.
    double candidateRefineRadiusBase = 4.0;
    int candidateRefineIterations = 8;
};

// Splits the graph and places every port from the prediction's section rather
// than from the skeleton voxel: on the section's centre line across the sheet,
// with one candidate per fibre width along it. The skeleton decides which edge
// and at what arc; the prediction decides where in the section.
[[nodiscard]] FiberSkeletonTopology buildFiberSkeletonTopology(
    const FiberSkeletonGraph& graph, const FiberSkeletonCrop& crop,
    const FiberSkeletonSplitConfig& split, double supportSampleBase);

// Fallback initial geometry for a stretch between two chosen end positions
// when the path search finds no way: the stretch's own points with the end
// displacements blended linearly along its arc.
[[nodiscard]] std::vector<cv::Vec3d> fiberSkeletonBlendEnds(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, const cv::Vec3d& startBaseXYZ,
    const cv::Vec3d& endBaseXYZ);

// Refines every line once per pair of end candidates, ends held fixed, and
// records each sub-port's cheapest variant.
void refineFiberSkeletonLines(
    FiberSkeletonTopology& topology, FiberSkeletonFieldProvider& fields,
    const FiberSkeletonPathSearchConfig& search, bool routeThroughCentre,
    const FiberSkeletonCostWeights& weights,
    const FiberSkeletonSolverConfig& solver, double stepBase);

// Copies `variant` into the line's display geometry.
void applyFiberSkeletonLineVariant(FiberSkeletonLine& line, std::size_t variant);

}  // namespace vc::fiber_tracer
