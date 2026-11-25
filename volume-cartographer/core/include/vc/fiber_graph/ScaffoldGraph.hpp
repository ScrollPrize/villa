#pragma once
// The manually annotated scaffold of one vcobj store, read for a winding solve.
//
// A scaffold is a set of fiber objects with H/V families, their annotation
// links (`fiber_branch` edges), their piece approvals and the volume's
// umbilicus. This header reads that state into plain structs through any
// StoreReader, so the same loader serves VC3D and the command line. Nothing
// here samples a volume or solves anything; the winding solve itself lives in
// ScaffoldWindingSolve.hpp.
//
// Coordinates are voxels of the frame the fibers are annotated in, mapped to
// the requested target volume. The scroll's winding handedness is read from
// the umbilicus object and never guessed: a mirrored solve must not happen
// silently.
#include "vc/fiber_graph/ScaffoldRunMonitor.hpp"

#include "vc_object_store/geometry/FiberWinding.hpp"
#include "vc_object_store/objects/FiberPieceApprovalExtension.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

#include <cstdint>
#include <map>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vc::fiber_graph {

// The winding families of the annotation: H fibers run around the scroll, V
// fibers across it. `Untagged` fibers carry no usable family and take part in
// no winding relation.
enum class ScaffoldFamily : int { Horizontal = 0, Vertical = 1, Untagged = 2 };
// Which stored family labels a load accepts (the store's `manual_family` flag).
enum class ScaffoldFamilySource { Any, Manual, Automatic };

struct ScaffoldFiber {
    std::string uuid, name;
    ScaffoldFamily family = ScaffoldFamily::Untagged;
    bool manualFamily = false;
    std::optional<std::int64_t> version;
    // This fiber's line points in ScaffoldGraph::linePoints.
    std::int64_t pointFirst = 0, pointCount = 0;
    // Control point ids, coordinates, annotations and their line indices, as
    // the piece-approval extension reads them. Its `linePoints` are cleared
    // after loading: the polylines live once, in ScaffoldGraph::linePoints.
    objects::FiberPieceContent content;
    // The live piece-approval rows and the spans they currently cover.
    objects::FiberPieceApprovalStates approvals;
    // The normal approved at each of this fiber's line points: the stored
    // axis on the spans approved for the normals aspect, zero elsewhere.
    // Empty when no span is approved.
    std::vector<geometry::Point> approvedLineNormals;
};

// One annotation link between two fibers. Both anchors are the annotated
// positions; the loader does not project them onto the polylines (the solve
// does, so a coordinate-frame error is reported with its distance).
struct ScaffoldLink {
    std::string uuid, aUuid, bUuid;
    // The link record's version, as the store states it.
    std::optional<std::int64_t> version;
    // Fiber indices, or -1 when that endpoint is not a loaded fiber.
    std::int64_t a = -1, b = -1;
    std::optional<geometry::Point> anchorA, anchorB;
    // meta.adjacent: the V fiber sits one winding inside the H fiber.
    bool adjacent = false;
    // meta.auto: an unreviewed automatic link.
    bool autoLink = false;
    std::string approvalState;
};

// Where the winding handedness came from, so a run can state it.
enum class ScaffoldHandednessSource { Option, Store, Fallback };

struct ScaffoldLoadReport {
    std::int64_t fiberRecords = 0, fibers = 0, untagged = 0, manualFamily = 0, automaticFamily = 0,
                 filteredFamily = 0, shortFibers = 0, links = 0, linkRecords = 0, pieceApprovalRows = 0;
    // Fiber records the load's `fiberSubset` left out.
    std::int64_t filteredSubset = 0;
    ScaffoldHandednessSource handednessSource = ScaffoldHandednessSource::Store;
};

struct ScaffoldGraph {
    std::vector<ScaffoldFiber> fibers;
    // fibers.size() + 1 offsets into linePoints (in points, not doubles).
    std::vector<std::int64_t> offsets;
    std::vector<double> linePoints; // XYZ triples
    std::vector<ScaffoldLink> links;
    std::vector<geometry::Point> umbilicus;
    int handedness = 0;
    double seamAngle = -std::numbers::pi / 2;
    std::string volume;
    double scale = 1;
    objects::StoreStatus source;
    ScaffoldLoadReport report;

    // Fiber index by UUID, filled by the loader.
    std::map<std::string, std::size_t> index;

    std::optional<std::size_t> indexOf(const std::string& fiberUuid) const;
};

struct ScaffoldLoadOptions {
    // Target frame. Empty reads the frame the fibers declare
    // (meta.coordinates.volume_uuid); fibers declaring different volumes are
    // an error, never silently mixed.
    std::string volume;
    double scale = 1;
    // The scroll's winding direction. Unset reads `winding_handedness` from
    // the volume's umbilicus object; a store that states it nowhere falls
    // back to `fallbackHandedness` (the caller's own source, such as the
    // open-data catalog's orientation) and is an error without one. It is
    // never inferred from the fibers: a mirrored solve must not happen
    // silently.
    std::optional<int> handedness;
    std::optional<int> fallbackHandedness;
    double seamAngle = -std::numbers::pi / 2;
    ScaffoldFamilySource familySource = ScaffoldFamilySource::Any;
    // Load untagged fibers too (they carry no winding relation; the winding
    // solve leaves them out).
    bool includeUntagged = false;
    bool readPieceApprovals = true;
    // Load only these fibers (by UUID), empty for all of them. This is how a
    // run is restricted to one part of the annotation - the map's selected
    // network: the other fibers are not loaded at all, so no link, crossing
    // or normal of theirs enters the run, and its components are the
    // components of the subset.
    std::vector<std::string> fiberSubset;
    // Progress and cancellation of the read itself, which is a stage of its
    // own: a scaffold's line points are the run's first long wait.
    ScaffoldRunMonitor* monitor = nullptr;
};

// The store commit that last changed anything a scaffold run reads: a fiber
// record, one of their links, or an umbilicus. A run whose recorded input
// commit sequence is below this was computed from an annotation that has
// changed since. A run's own output is extension rows, which change no
// record, so saving a run does not move this.
std::int64_t scaffoldAnnotationCommit(const objects::StoreReader&);

// Throws std::invalid_argument when the store states no handedness, holds no
// umbilicus for the frame, or mixes annotation frames.
ScaffoldGraph loadScaffoldGraph(const objects::StoreReader&, const ScaffoldLoadOptions& = {});

// One fiber's piece content with its line points filled in again (the loader
// keeps them once, in ScaffoldGraph::linePoints). Fingerprinting a span needs
// them, so this is what an approval check is taken over.
objects::FiberPieceContent scaffoldFiberContent(const ScaffoldGraph&, std::size_t fiber);

// Seam transport and the packed arc/turn columns of a loaded scaffold. Holds
// its own columns and borrows the graph's polylines, so the graph must outlive
// it.
class ScaffoldGeometry {
public:
    explicit ScaffoldGeometry(const ScaffoldGraph&, std::size_t workers = 1);
    const geometry::PackedFiberGeometry& packed() const { return packed_; }
    const geometry::WindingSeam& seam() const { return packed_.seam(); }
    // Arc length of the polyline point nearest `point`, and its distance in
    // voxels (the anchor projection of the manual links).
    struct Projection { double arc = 0, distance = 0; };
    Projection project(std::size_t fiber, const geometry::Point&) const;
    // Arc length at a control point, or the fiber end when it has no line
    // index.
    double controlArc(std::size_t fiber, std::size_t control) const;
    // The fiber's own index of the line point nearest an arc position.
    std::int64_t lineIndexAt(std::size_t fiber, double arc) const;
    double arcLength(std::size_t fiber) const { return packed_.arcLength(std::int64_t(fiber)); }
    // The fiber's gap spans as arc ranges: a span whose two controls are
    // tagged `break` (the span's own `gap` tag). A gap is a bridge across
    // damaged papyrus, not fiber - like a tail it carries no constraint.
    std::span<const std::pair<double, double>> gaps(std::size_t fiber) const { return gaps_.at(fiber); }
    // Whether an encounter at `arc` is on a gap, or closer than `margin` to
    // one: an encounter needs fiber on both sides to be certified, exactly as
    // at a fiber's end.
    bool nearGap(std::size_t fiber, double arc, double margin) const;
    std::span<const double> arcs() const { return columns_.arc; }
    std::span<const std::int64_t> turns() const { return columns_.turns; }

private:
    const ScaffoldGraph* graph_;
    geometry::PackedWindingColumns columns_;
    geometry::PackedFiberGeometry packed_;
    std::vector<std::vector<std::pair<double, double>>> gaps_;
};

// The fiber an encounter of this kind needs on each side of it, in voxels:
// what the crossing extraction's endpoint certification asks for (a tangent
// window beyond its probes). The same distance is kept from a gap.
inline constexpr double kScaffoldPerpendicularSupport = 32.0, kScaffoldParallelSupport = 80.0;

} // namespace vc::fiber_graph
