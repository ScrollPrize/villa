#pragma once
// Sign-free strip normals for a scaffold's fibers, and the anchors that fix
// their direction.
//
// A papyrus fiber is a flattened strip: at every point it carries one normal
// axis perpendicular to the fiber. Geometry alone decides that axis up to sign
// (vc/fiber_tracer/FiberStripNormals.hpp); the annotation fixes the sign
// wherever it states one, in this order of precedence:
//
//   1. a control point's `display_normal` with source `manual`. The fiber
//      format's display normals point outward (away from the umbilicus), so
//      the inward target is the negation of the stored direction;
//   2. the display normals of the spans approved for the normals aspect,
//      whatever source they carry;
//   3. an adjacent link: the V fiber lies one winding inside the H fiber, so
//      at the link's two anchors the H fiber's inward normal points towards
//      the V anchor and the V fiber's inward normal points away from the H
//      anchor. That is the geometry of the two anchors and assumes no
//      convention.
//
// A fiber with at least one anchor comes out in the inward gauge: its whole-
// fiber sign is decided here. A fiber with none keeps the arbitrary gauge of
// its geometry solve, and its sign is left to the sign solve
// (ScaffoldConstraints.hpp) - that is what the `normalSign` column of the
// winding assembly means.
//
// Nothing here samples a volume: variant A of the scaffold workflow reads its
// normals from the annotation and the fiber geometry only.
#include "vc/fiber_graph/ScaffoldGraph.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_graph {

// What fixed a fiber's gauge, in precedence order: a smaller value wins.
// `Crossings` is not set here - it is what the sign solve
// (ScaffoldConstraints.hpp) reports for the fibers it decided itself.
enum class ScaffoldSignSource : int { None = 0, Manual = 1, ApprovedPiece = 2, AdjacentLink = 3, Crossings = 4 };

const char* scaffoldSignSourceName(ScaffoldSignSource);

// One fixed normal direction on one fiber.
struct ScaffoldNormalAnchor {
    std::int64_t fiber = 0;
    double arc = 0;
    geometry::Point inward{};
    ScaffoldSignSource source = ScaffoldSignSource::None;
    // The link an adjacent-link anchor came from, or the control point a
    // display normal came from; -1 for the other kind.
    std::int64_t link = -1, control = -1;
    double weight = 1;
};

// Variant B: what a prediction volume states at one point - the sheet axis
// (sign-free) and the raw presence value there.
struct ScaffoldPredictedAxis {
    geometry::Point axis{};
    double presence = 0;
    bool valid = false;
};
// Samples the prediction at the fibers' own vertices, in order. The caller
// owns the dataset; the library only gathers what comes back.
using ScaffoldAxisSampler =
    std::function<std::vector<ScaffoldPredictedAxis>(std::span<const geometry::Point> points, std::size_t workers)>;

struct ScaffoldNormalOptions {
    fiber_tracer::FiberStripNormalOptions strip;
    // Variant B: map the prediction's presence to an axis weight, 0 at
    // `presenceFloor` and 1 at `presenceFull`. Ignored without a sampler.
    double presenceFloor = 0.0, presenceFull = 128.0;
    bool manualDisplayNormals = true;
    bool approvedNormalPieces = true;
    bool adjacentLinkAnchors = true;
    // Weight every anchor enters the strip solve's orientation vote with.
    double anchorWeight = 1.0;
    // Record approval states whose adjacent links may anchor a normal.
    std::vector<std::string> linkApprovalStates{"approved"};
    bool includeAutoLinks = false;
    std::size_t workers = 1;
};

struct ScaffoldNormalReport {
    std::int64_t fibers = 0, solved = 0, degenerate = 0, converged = 0;
    std::int64_t anchoredFibers = 0, unanchoredFibers = 0;
    std::int64_t manualAnchors = 0, approvedPieceAnchors = 0, adjacentLinkAnchors = 0;
    // Anchors dropped because the two link anchors coincide, or because the
    // stored display normal is parallel to the fiber there.
    std::int64_t degenerateAnchors = 0;
    std::int64_t skippedLinks = 0;
    // Variant B: the prediction's own contribution.
    std::int64_t predictedVertices = 0, verticesWithAxis = 0, chainNodes = 0, nodesWithAxis = 0;
    double meanAxisWeight = 0, predictionSeconds = 0;
    double seconds = 0;
};

class ScaffoldNormals {
public:
    // Without a sampler this is variant A: the normals come from the fibers'
    // own geometry and the annotation's anchors. With one, every chain node
    // is also pulled towards the sheet axis the prediction states there
    // (variant B); the anchors still win at the nodes they fix.
    ScaffoldNormals(const ScaffoldGraph&, const ScaffoldGeometry&, const ScaffoldNormalOptions& = {},
                    const ScaffoldAxisSampler& = {});

    // One gauge normal per line point, XYZ triples in ScaffoldGraph::linePoints
    // order; a degenerate fiber's points are zero.
    std::span<const double> normals() const { return normals_; }
    bool solved(std::size_t fiber) const { return solved_.at(fiber) != 0; }
    // The gauge normal at an arc, interpolated between the two line points
    // around it; nullopt for a fiber without normals.
    std::optional<geometry::Point> at(std::size_t fiber, double arc) const;

    const std::vector<ScaffoldNormalAnchor>& anchors() const { return anchors_; }
    const std::vector<ScaffoldSignSource>& signSources() const { return signSources_; }
    // Per fiber: the mean dot of its gauge normal with the outward radial
    // direction, weighted by the arc each line point represents, in [-1, 1].
    // The weak prior of the sign solve; 0 for a fiber without normals.
    const std::vector<double>& radialAgreement() const { return radialAgreement_; }
    // Per fiber: +1 for an anchored fiber (its gauge is the inward one), 0 for
    // a fiber whose sign the sign solve decides.
    const std::vector<int>& fixedSigns() const { return fixedSigns_; }
    const ScaffoldNormalReport& report() const { return report_; }

private:
    const ScaffoldGraph* graph_ = nullptr;
    const ScaffoldGeometry* geometry_ = nullptr;
    std::vector<double> normals_;
    std::vector<char> solved_;
    std::vector<ScaffoldNormalAnchor> anchors_;
    std::vector<ScaffoldSignSource> signSources_;
    std::vector<double> radialAgreement_;
    std::vector<int> fixedSigns_;
    ScaffoldNormalReport report_;
};

} // namespace vc::fiber_graph
