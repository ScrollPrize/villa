#pragma once
// Approval layer of a fiber (docs/fiber-approval.md).
//
// A layer on top of a fiber's stored geometry and normals; it never changes
// the fiber. It holds stretches with a state per aspect (geometry, normals),
// normal points (a wanted normal at a position on the line) and the normals
// shown when it was saved. Every save writes a whole new generation of the
// layer; the newest generation of a fiber is its layer, older ones are
// history. Positions are indices into the fiber's stored line points (with a
// fraction); a stretch or normal point whose line points changed is stale.
#include "vc_object_store/objects/Extensions.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {
class StoreReader;

inline constexpr const char* kFiberApprovalExtension = "fiber-approval";
inline constexpr const char* kFiberApprovalTable = "ext_fiber_approval";
inline constexpr const char* kFiberApprovalStretchTable = "ext_fiber_approval_stretch";
inline constexpr const char* kFiberApprovalNormalPointTable = "ext_fiber_approval_normal_point";
inline constexpr int64_t kFiberApprovalFormat = 1;

enum class FiberApprovalAspect : int64_t { Geometry = 0, Normals = 1 };
// Undefined is the absence of a stretch; it is never stored.
enum class FiberStretchState : int64_t { Undefined = 0, Approved = 1, Broken = 2 };

struct FiberStretch {
    FiberApprovalAspect aspect = FiberApprovalAspect::Geometry;
    FiberStretchState state = FiberStretchState::Approved;
    double from = 0, to = 0;  // line positions, from < to
    uint64_t fingerprint = 0; // over the line points the stretch touches
    bool operator==(const FiberStretch&) const = default;
};
struct FiberNormalPoint {
    double position = 0;
    geometry::Point normal{}; // the wanted normal, an axis: its facing means nothing
    uint64_t fingerprint = 0; // over the two line points around the position
    bool operator==(const FiberNormalPoint&) const = default;
};
struct FiberApprovalLayer {
    std::string fiberUuid;
    int64_t generation = 0, fiberVersion = 0;
    std::string author, savedAt, host;
    // The scaffold solve whose normals the layer was shown on (may be empty).
    std::string baselineRun;
    std::vector<FiberStretch> stretches;      // per aspect sorted, never overlapping
    std::vector<FiberNormalPoint> normalPoints; // sorted by position
    // The normal shown at every stored line point when the layer was saved
    // (the solve's normals with the normal points' correction): what a
    // normals approval approves. Empty when nothing was shown.
    std::vector<geometry::Point> lineNormals;
    bool empty() const { return stretches.empty() && normalPoints.empty(); }
    bool operator==(const FiberApprovalLayer&) const = default;
};

std::vector<ExtensionTable> fiberApprovalTables();
struct FiberApprovalRows {
    ExtensionRow header;
    std::vector<ExtensionRow> stretches, normalPoints;
};
FiberApprovalRows fiberApprovalRows(const FiberApprovalLayer&);

// FNV-1a 64 over the line points floor(from) .. ceil(to), clamped to the line.
uint64_t fiberLineFingerprint(std::span<const geometry::Point> line, double from, double to);

// Paint `state` over [from, to] of one aspect: existing stretches of that
// aspect are cut back, Undefined only clears. Fingerprints are taken over
// `line`. Throws std::invalid_argument for an empty or reversed range.
void setFiberStretchState(std::vector<FiberStretch>&, FiberApprovalAspect, double from, double to, FiberStretchState,
                          std::span<const geometry::Point> line);
FiberStretchState fiberStretchStateAt(std::span<const FiberStretch>, FiberApprovalAspect, double position);
// Add or replace the normal point at `position` (within `tolerance` line
// points of an existing one); returns its index.
size_t setFiberNormalPoint(std::vector<FiberNormalPoint>&, double position, const geometry::Point& normal,
                           std::span<const geometry::Point> line, double tolerance = 0.5);

// Which items of a layer still describe `line`, the fiber's stored line points.
struct FiberApprovalStatus {
    std::vector<char> stretchCurrent, normalPointCurrent;
    bool lineNormalsUsable = false; // one stored normal per line point
    int64_t staleStretches = 0, staleNormalPoints = 0;
};
FiberApprovalStatus fiberApprovalStatus(const FiberApprovalLayer&, std::span<const geometry::Point> line);
// Per stored line point: the saved normal on the current stretches approved
// for normals and at the line point nearest each current normal point, zero
// everywhere else. Empty when there is none.
std::vector<geometry::Point> approvedFiberLineNormals(const FiberApprovalLayer&, const FiberApprovalStatus&,
                                                      size_t linePoints);
// The current stretches of one aspect and state as position ranges.
std::vector<std::pair<double, double>> fiberStretchRanges(const FiberApprovalLayer&, const FiberApprovalStatus&,
                                                          FiberApprovalAspect, FiberStretchState);

// The newest generation of a fiber's layer; nullopt when it has none. A store
// without the tables has no layers.
std::optional<FiberApprovalLayer> readFiberApproval(const StoreReader&, const std::string& fiberUuid);
// The fibers that have a layer, with its newest generation number.
std::map<std::string, int64_t> fiberApprovalGenerations(const StoreReader&);
} // namespace vc::objects
