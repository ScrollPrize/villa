#pragma once

// Presence thresholding, 3D thinning and skeleton-graph construction for the
// skeleton tracing backend. Everything here is pure geometry over a dense
// uint8 presence crop: no prediction source, solver or evaluation protocol.
//
// Index convention matches the Python pair-connectivity diagnostic:
// a prediction-grid index (z, y, x) maps to base XYZ as
// `(x + originX, y + originY, z + originZ) * predictionToBaseScale`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

// Dense presence crop on the prediction grid, row-major in Z, Y, X.
struct FiberSkeletonCrop {
    std::array<std::int64_t, 3> originZYX{0, 0, 0};
    std::array<std::size_t, 3> shapeZYX{0, 0, 0};
    double predictionToBaseScale = 1.0;
    std::vector<std::uint8_t> presence;
    // Coverage of the scoped region by the stored prediction cache.
    std::size_t scopedVoxels = 0;
    std::size_t missingVoxels = 0;

    [[nodiscard]] std::size_t voxelCount() const noexcept
    {
        return shapeZYX[0] * shapeZYX[1] * shapeZYX[2];
    }
    [[nodiscard]] std::size_t flatIndex(
        std::size_t z, std::size_t y, std::size_t x) const noexcept
    {
        return (z * shapeZYX[1] + y) * shapeZYX[2] + x;
    }
    [[nodiscard]] std::array<std::size_t, 3> voxelZYX(
        std::size_t flat) const noexcept
    {
        const std::size_t x = flat % shapeZYX[2];
        const std::size_t y = (flat / shapeZYX[2]) % shapeZYX[1];
        return {flat / (shapeZYX[2] * shapeZYX[1]), y, x};
    }
    [[nodiscard]] cv::Vec3d baseXYZ(const std::array<std::size_t, 3>& zyx) const
    {
        return {(static_cast<double>(zyx[2]) + static_cast<double>(originZYX[2])) * predictionToBaseScale,
                (static_cast<double>(zyx[1]) + static_cast<double>(originZYX[1])) * predictionToBaseScale,
                (static_cast<double>(zyx[0]) + static_cast<double>(originZYX[0])) * predictionToBaseScale};
    }
    [[nodiscard]] cv::Vec3d baseXYZ(std::size_t flat) const
    {
        return baseXYZ(voxelZYX(flat));
    }
};

// (26, 6) simple-point test on a 3x3x3 neighborhood in raster order
// (index = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1), center 13). A point is
// simple when its 26-neighborhood foreground forms exactly one 26-connected
// component and the background of its 18-neighborhood forms exactly one
// 6-connected component that is 6-adjacent to the center.
[[nodiscard]] bool fiberSkeletonSimplePoint(
    const std::array<std::uint8_t, 27>& neighborhood) noexcept;

// In-place 3D thinning of a 0/1 mask, following Lee, Kashyap and Chu as used
// by `skimage.morphology.skeletonize(method="lee")`: six directional border
// subiterations (-Z, +Z, +X, -X, -Y, +Y), preservation of endpoints (voxels
// with a single foreground neighbor), and sequential re-verification of every
// candidate before it is removed. Repeats until a full pass removes nothing.
void thinFiberSkeletonMask(
    const std::array<std::size_t, 3>& shapeZYX, std::vector<std::uint8_t>& mask);

// 26-connected labeling. Background is 0 and labels start at 1.
[[nodiscard]] std::vector<std::int32_t> labelFiberSkeletonComponents(
    const std::array<std::size_t, 3>& shapeZYX,
    const std::vector<std::uint8_t>& mask, std::size_t& componentCount);

// Thresholds `crop.presence` at `thresholdU8`, drops components smaller than
// `minComponentVoxels`, and returns the retained 0/1 mask.
[[nodiscard]] std::vector<std::uint8_t> thresholdFiberSkeletonCrop(
    const FiberSkeletonCrop& crop, int thresholdU8,
    std::size_t minComponentVoxels);

enum class FiberSkeletonNodeKind : std::uint8_t { Terminal, Junction, Other };

inline constexpr std::size_t kFiberSkeletonNoGroup =
    static_cast<std::size_t>(-1);

// One skeleton voxel. Junction voxels are never contracted together: the
// cluster's internal structure stays in the graph as ordinary short edges, so
// a route through an ambiguous region is a path in the graph like any other.
struct FiberSkeletonNode {
    std::size_t id = 0;
    cv::Vec3d positionBaseXYZ{0.0, 0.0, 0.0};
    std::vector<std::size_t> incidentEdges;
    std::size_t degree = 0;
    FiberSkeletonNodeKind kind = FiberSkeletonNodeKind::Other;
    // Junction nodes joined by an edge shorter than the merge distance share a
    // group; one group is one ambiguous region. `kFiberSkeletonNoGroup` for
    // every node that is not part of one.
    std::size_t regionGroup = kFiberSkeletonNoGroup;
    bool removed = false;
};

// The polyline is the edge's only geometry and passes through every skeleton
// voxel of the chain, starting and ending at the two node positions. There is
// no second, parallel voxel representation to fall out of step with it.
struct FiberSkeletonEdge {
    std::size_t id = 0;
    std::array<std::size_t, 2> nodeIds{0, 0};
    std::vector<cv::Vec3d> pointsBaseXYZ;
    double lengthBase = 0.0;
    bool removed = false;
};

struct FiberSkeletonStatistics {
    std::size_t thresholdVoxels = 0;
    std::size_t skeletonVoxels = 0;
    std::size_t skeletonComponents = 0;
    std::size_t rawNodes = 0;
    std::size_t rawEdges = 0;
    std::size_t graphNodes = 0;
    std::size_t graphEdges = 0;
    std::size_t terminalNodes = 0;
    std::size_t junctionNodes = 0;
    std::size_t graphComponents = 0;
    std::size_t cycleRank = 0;
    std::size_t regionGroups = 0;
    std::size_t groupedJunctions = 0;
    std::size_t prunedSpurs = 0;
    std::size_t suppressedDegreeTwoNodes = 0;
    double totalEdgeLengthBase = 0.0;
};

struct FiberSkeletonGraph {
    std::vector<FiberSkeletonNode> nodes;
    std::vector<FiberSkeletonEdge> edges;
    FiberSkeletonStatistics statistics;

    // Removed entries are compacted away; ids equal vector positions.
    void compact();
};

// Builds the 26-neighbour voxel graph of a thinned mask. Every voxel whose
// 26-neighbour degree differs from two becomes its own node, and the
// degree-two chains between them become edges whose polyline passes through
// each chain voxel. Adjacent non-degree-two voxels are joined by a short edge
// rather than contracted, so nothing of the skeleton's structure is lost.
// Isolated cycles without any non-degree-two voxel are reported through
// `skeletonComponents`/`cycleRank` and contribute no nodes or edges.
[[nodiscard]] FiberSkeletonGraph buildFiberSkeletonGraph(
    const FiberSkeletonCrop& crop, const std::vector<std::uint8_t>& skeleton);

// Reductions in order: prune terminal arms shorter than `spurPruneMaxBase`,
// suppress degree-two nodes by concatenating their two edges, then group
// junction nodes that are joined by an edge shorter than
// `junctionMergeMaxBase` into ambiguous regions by assigning `regionGroup`.
// Grouping is a labelling, never a contraction: no node, edge or incidence is
// removed by it, so the topology a region routes over stays intact. Lengths
// and geometry are preserved across concatenation.
void reduceFiberSkeletonGraph(
    FiberSkeletonGraph& graph, double junctionMergeMaxBase,
    double spurPruneMaxBase);

// Uniform resampling with both endpoints preserved exactly. Returns at least
// two points; `stepBase` must be finite and positive.
[[nodiscard]] std::vector<cv::Vec3d> resampleFiberPolyline(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double stepBase);

[[nodiscard]] double fiberPolylineLength(
    const std::vector<cv::Vec3d>& pointsBaseXYZ);

// Point at `arcBase` along the polyline, clamped to its ends.
[[nodiscard]] cv::Vec3d fiberPolylinePointAtArc(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double arcBase);

// Sub-polyline between two arc positions, with exact interpolated endpoints.
[[nodiscard]] std::vector<cv::Vec3d> fiberPolylineSlice(
    const std::vector<cv::Vec3d>& pointsBaseXYZ, double beginArcBase,
    double endArcBase);

// Writes `<path>` with one polyline per edge and `<stem>_markers.obj` with one
// point per node, both in base XYZ, through the shared polyline OBJ writer.
void writeFiberSkeletonGraphObj(
    const std::filesystem::path& path, const FiberSkeletonGraph& graph);

}  // namespace vc::fiber_tracer
