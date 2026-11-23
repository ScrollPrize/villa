#pragma once
// Sign-free strip normals for a scaffold's fibers, and the anchors that fix
// their direction.
//
// A papyrus fiber is a flattened strip: at every point it carries one normal
// axis perpendicular to the fiber. Geometry alone decides that axis up to sign
// (vc/fiber_tracer/FiberStripNormals.hpp).
//
// One convention for every signed normal, here and in the whole-volume
// pipeline: it points OUTWARD, away from the umbilicus. A fiber's solved
// normal times its whole-fiber sign is that outward normal. The winding
// relations need the direction of falling winding instead, which is the
// outward one in a scroll whose winding falls outward and the inward one
// otherwise; scaffoldFallingWindingSign() is that one conversion, and nothing
// else flips a normal.
//
// The annotation fixes the sign wherever it states one, in this order of
// precedence:
//
//   1. a control point's `display_normal` with source `manual`. The fiber
//      format's display normals point outward, so it is taken as stored;
//   2. the display normals of the spans approved for the normals aspect,
//      whatever source they carry;
//   3. an adjacent link: the V fiber lies one winding below the H fiber, so
//      the direction from the H anchor to the V anchor is the one of falling
//      winding, and the outward normal of both fibers there follows from it.
//
// A fiber with at least one anchor comes out in the outward gauge: its whole-
// fiber sign is decided here. A fiber with none keeps the arbitrary gauge of
// its geometry solve, and its sign is left to the sign solve
// (ScaffoldConstraints.hpp) - that is what the `normalSign` column of the
// winding assembly means.
//
// Nothing here samples a volume: variant A of the scaffold workflow reads its
// normals from the annotation and the fiber geometry only.
//
// Neighbour normals: where a signed normal field of an independently resolved
// fiber volume is given (ScaffoldNeighbourOptions), every fiber is also pulled
// towards what the automatic fibers around it state, sign included
// (vc/fiber_tracer/FiberNeighbourNormals.hpp). A fiber no anchor fixed and
// that the field covers takes its facing from it, and the sign solve holds it.
#include "vc/fiber_graph/ScaffoldGraph.hpp"
#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"
#include "vc/fiber_graph/ScaffoldRunMonitor.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_graph {

// What fixed a fiber's gauge, in precedence order: a smaller value wins.
// `Crossings` is not set here - it is what the sign solve
// (ScaffoldConstraints.hpp) reports for the fibers it decided itself.
// `Neighbours` is a fiber no anchor fixed, faced by the neighbour normals.
enum class ScaffoldSignSource : int {
    None = 0, Manual = 1, ApprovedPiece = 2, AdjacentLink = 3, Crossings = 4, Neighbours = 5
};

const char* scaffoldSignSourceName(ScaffoldSignSource);

// +1 where the scroll's winding falls outward (handedness -1), -1 where it
// falls inward: an outward normal times this points the way the winding
// falls, which is the direction the winding relations order two fibers by
// (geometry::fiberSignedOrder, windingEventRelation). Throws when the graph
// states no handedness.
double scaffoldFallingWindingSign(const ScaffoldGraph&);

// One fixed normal direction on one fiber.
struct ScaffoldNormalAnchor {
    std::int64_t fiber = 0;
    double arc = 0;
    geometry::Point outward{};
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

// The signed strip normals of a resolved fiber volume, opened from a signed
// strip samples file (`vc_fiber_constraints`, `signed_strip_samples_output`)
// and the statement next to it. Its normals are signed by the sign stage,
// which faces them the way this gauge faces an anchored fiber.
struct ScaffoldNeighbourSource {
    std::shared_ptr<const fiber_tracer::FiberNeighbourField> field;
    // What the field is, for the cache: the sign run and the sample count.
    std::string identity;
    // The frame its points are in: the volume's UUID, at scale 1.
    std::string volume;
    std::string coordinateSpace;
};
[[nodiscard]] ScaffoldNeighbourSource openScaffoldNeighbourSource(const std::filesystem::path& signedSamples,
                                                                  double radius = 384.0);

// How the neighbour normals enter a run. The defaults are the full-volume
// pipeline's own (fiber frame, crossing scales 128 / 32 / 128, parallel
// scales 192 / 192 / 64, radius 384, weight 300, sigma 32), with a parallel
// contact measured in the frame of the isotropic first estimate where the
// fiber carries no prediction axis.
struct ScaffoldNeighbourOptions {
    // Null: no neighbour normals.
    const fiber_tracer::FiberNeighbourField* field = nullptr;
    std::string identity;
    // The volume the field's points are voxels of. A run in another frame is
    // refused: nothing maps the field there.
    std::string volume;
    fiber_tracer::FiberNeighbourVariant variant{.estimatedParallelFrame = true};
    double weight = 300.0, sigma = 32.0;
    // Arc spacing of the samples the field is asked at, voxels.
    double sampleSpacing = 16.0;
    // The field faces a fiber no anchor fixed when it gave a target to at
    // least this share of the fiber's chain nodes; below it the fiber is left
    // to the sign solve.
    double minimumCoverage = 0.25;
    // The crossing term of the whole-volume strip solve: where another fiber
    // crosses, the normal lies along own tangent x partner tangent, both
    // tangents taken from the fibers themselves. `fieldCrossings` takes the
    // crossings by the field's fibers (found in the field's own grid),
    // `scaffoldCrossings` those by the scaffold's other fibers. The weighting
    // is the whole-volume run's (weight 100, sigma 64, Cauchy falloff of the
    // gap scaled 64 / 32 / 1024); a node without a prediction axis measures
    // the gap in the crossing's own frame.
    bool fieldCrossings = true, scaffoldCrossings = true;
    fiber_tracer::FiberCrossingTargetOptions crossing{.weight = 100.,
                                                      .sigma = 64.,
                                                      .falloff = ::GapFalloff::Cauchy,
                                                      .frame = fiber_tracer::FiberCrossingGapFrame::Prediction,
                                                      .scales = {64., 32., 1024.},
                                                      .predictionPower = 2.};
};

struct ScaffoldNormalOptions {
    // The strip solve, with the whole-volume pipeline's settings (twist 300,
    // smoothing and resampling 64 voxels): a scaffold fiber's normal is
    // solved the way the automatic fibers around it were. Measured on the
    // PHerc0332 scaffold with neighbour normals, twist 300 and the former
    // 1000 give the same result within one link (docs/fiber_map_scaffold_solve.md).
    fiber_tracer::FiberStripNormalOptions strip{.twistWeight = 300.0,
                                                .lateralWeight = 1.0,
                                                .smoothingSigma = 64.0,
                                                .resampleSpacing = 64.0,
                                                .axisWeight = 100.0,
                                                .maxIterations = 50};
    // Variant B: map the prediction's presence to an axis weight, 0 at
    // `presenceFloor` and 1 at `presenceFull`. Ignored without a sampler.
    double presenceFloor = 0.0, presenceFull = 128.0;
    ScaffoldNeighbourOptions neighbours;
    bool manualDisplayNormals = true;
    bool approvedNormalPieces = true;
    bool adjacentLinkAnchors = true;
    // Weight every anchor enters the strip solve's orientation vote with.
    double anchorWeight = 1.0;
    // Record approval states whose adjacent links may anchor a normal.
    std::vector<std::string> linkApprovalStates{"approved"};
    bool includeAutoLinks = false;
    std::size_t workers = 1;
    // Reuse: the normals of a fiber whose version and input digest are
    // already cached are read instead of solved, and what this run solves is
    // written into `sink` for the caller to commit. Both null is a run that
    // computes everything and keeps nothing.
    // What the axis sampler of a fused run reads, by the identity the store
    // or the catalog gives it (the prediction volume). A fused run without
    // one is solved every time: nothing says what its rows would be about.
    std::string predictionIdentity;
    const class ScaffoldCache* cache = nullptr;
    struct ScaffoldCacheSink* sink = nullptr;
    // Vertices per call into the prediction sampler. A sample depends on
    // nothing but its own point, so this changes no result; it only decides
    // how often a fused run reports its progress and can be stopped.
    // Measured on the PHerc0332 scaffold (851,285 line points, 32 workers,
    // one run each): 107.7 s of prediction sampling in 65,536-point chunks
    // against 107.6 s in a single call, so the chunking costs nothing and
    // the default is small enough to stop within seconds.
    std::size_t predictionChunkPoints = 65536;
    // Optional progress and cancellation (ScaffoldRunMonitor.hpp). The
    // prediction is sampled in chunks either way; with a monitor the stage
    // counts the vertices off and stops between chunks.
    ScaffoldRunMonitor* monitor = nullptr;
};

struct ScaffoldNormalReport {
    // Fibers whose normals came from the cache, and those this run solved
    // and wrote back.
    std::int64_t reused = 0, cached = 0;
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
    // Neighbour normals: the samples the field was asked at and those it
    // answered, the chain nodes that got a signed target, the fibers it
    // covers enough to face, and those it faced (no anchor did).
    std::int64_t neighbourSamples = 0, neighbourSamplesAnswered = 0, neighbourNodes = 0;
    std::int64_t neighbourCoveredFibers = 0, neighbourFacedFibers = 0;
    // The crossing term: crossings by the field's fibers and by the
    // scaffold's own, and the chain nodes that got a crossing axis.
    std::int64_t fieldCrossings = 0, scaffoldCrossings = 0, crossedNodes = 0;
    double neighbourSeconds = 0;
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
    // Per fiber: +1 for an anchored fiber (its gauge is the outward one), 0 for
    // a fiber whose sign the sign solve decides.
    const std::vector<int>& fixedSigns() const { return fixedSigns_; }
    const ScaffoldNormalReport& report() const { return report_; }
    // Per fiber: the identity of the cache row its normals are (read from it,
    // or written by this run), which the crossings extracted with them name.
    // Empty for a fiber whose normals cannot be cached.
    const std::vector<std::string>& cacheRows() const { return cacheRows_; }

private:
    const ScaffoldGraph* graph_ = nullptr;
    const ScaffoldGeometry* geometry_ = nullptr;
    std::vector<double> normals_;
    std::vector<char> solved_;
    std::vector<ScaffoldNormalAnchor> anchors_;
    std::vector<ScaffoldSignSource> signSources_;
    std::vector<double> radialAgreement_;
    std::vector<int> fixedSigns_;
    std::vector<std::string> cacheRows_;
    ScaffoldNormalReport report_;
};

} // namespace vc::fiber_graph
