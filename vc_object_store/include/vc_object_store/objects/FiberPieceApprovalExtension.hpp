#pragma once
// Piece-wise approval of fibers (docs/fiber-piece-approval.md).
//
// A piece is the range of one fiber between two lock control points, or from
// one lock control point to the fiber end. Lock control points are ordinary
// control points carrying the `lock` tag and an explicit direction. Approvals
// are immutable rows of `ext_fiber_piece_approval`; revoking or cropping an
// approval appends a superseding row. Live rows of one aspect never overlap.
#include "vc_object_store/objects/Extensions.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {
class StoreReader;
struct ObjectRevision;

inline constexpr const char* kFiberPieceApprovalTable = "ext_fiber_piece_approval";
inline constexpr const char* kFiberPieceApprovalExtension = "fiber-piece-approval";
inline constexpr int64_t kFiberPieceApprovalFormat = 1;
// Control-point tag of a lock control point (the `tags` array of its annotation).
inline constexpr const char* kFiberLockControlPointTag = "lock";

enum class FiberPieceAspect : int64_t { Geometry = 0, Normals = 1 };
enum class FiberPieceState : int64_t { Approved = 1, Revoked = 2 };

ExtensionTable fiberPieceApprovalTable();

struct FiberSpanFingerprint {
    int64_t idLow = 0, idHigh = 0; // the span's two control-point ids, ordered
    uint64_t hash = 0;
    bool operator==(const FiberSpanFingerprint&) const = default;
};
struct FiberControlFingerprint {
    int64_t id = 0;
    uint64_t hash = 0;
    bool operator==(const FiberControlFingerprint&) const = default;
};

// One row. `approvalUuid` names the approval event and is shared by the crop
// rows derived from it and by copies made when a fiber is split or merged;
// `rowUuid` is unique per row. Fingerprints walk from `lockCpA` towards
// `insideCp`; a revocation row carries none.
struct FiberPieceApproval {
    std::string fiberUuid, rowUuid, approvalUuid;
    FiberPieceAspect aspect = FiberPieceAspect::Geometry;
    FiberPieceState state = FiberPieceState::Approved;
    std::optional<std::string> supersedes;
    int64_t lockCpA = 0;
    std::optional<int64_t> lockCpB; // nullopt: end piece, open towards insideCp's side
    int64_t insideCp = 0;           // the control point next to lockCpA inside the piece
    int64_t fiberVersion = 0;
    std::string author, approvedAt, host;
    std::optional<std::string> changedBy, changedAt; // who wrote a crop/revocation row
    std::vector<FiberSpanFingerprint> spans;        // geometry aspect only
    std::vector<FiberControlFingerprint> controls;  // in walking order
    bool operator==(const FiberPieceApproval&) const = default;
};
ExtensionRow fiberPieceApprovalRow(const FiberPieceApproval&);
FiberPieceApproval decodeFiberPieceApprovalRow(const ExtensionRow&);
// Typed row blobs, exposed for other encoders (Python, tests).
std::vector<uint8_t> encodeFiberSpanFingerprints(std::span<const FiberSpanFingerprint>);
std::vector<FiberSpanFingerprint> decodeFiberSpanFingerprints(std::span<const uint8_t>);
std::vector<uint8_t> encodeFiberControlFingerprints(std::span<const FiberControlFingerprint>);
std::vector<FiberControlFingerprint> decodeFiberControlFingerprints(std::span<const uint8_t>);

// What the fingerprints are taken over: the fiber's control points with their
// stable ids and annotations, and its line points. Control point i's line
// index is the position of its coordinates in the line points (the ordered
// membership scan VC3D uses); -1 when the control is not on the line.
struct FiberPieceContent {
    std::vector<int64_t> controlIds;
    std::vector<geometry::Point> controlPoints;
    std::vector<nlohmann::json> annotations; // one object per control point
    std::vector<geometry::Point> linePoints;
    std::vector<int64_t> controlLineIndex;

    std::optional<size_t> indexOf(int64_t id) const;
    // Fill controlLineIndex from the coordinates (exact match, in order).
    void resolveLineIndices();
    bool isLockControl(size_t index) const;
};
FiberPieceContent fiberPieceContent(const StoreReader&, const ObjectRevision&);
FiberPieceContent fiberPieceContent(const geometry::Fiber&, const nlohmann::json& controlAnnotations);

// A resolved piece: inclusive control-point index range and walking direction
// (from `lockCpA` towards `insideCp`).
struct FiberPieceRange {
    size_t first = 0, last = 0; // first <= last
    bool forward = true;        // true: lockCpA is `first`
    bool openEnd = false;       // the far end is the fiber end, not a second lock
    size_t spanCount() const { return last - first; }
    size_t lockIndexA() const { return forward ? first : last; }
    // Control index at walking position k (0 = lockCpA).
    size_t at(size_t k) const { return forward ? first + k : last - k; }
};
// nullopt when an id is missing or `insideCp` does not lie next to `lockCpA`
// on the side of `lockCpB` (or of the open end).
std::optional<FiberPieceRange> fiberPieceRange(const FiberPieceContent&, int64_t lockCpA,
                                               std::optional<int64_t> lockCpB, int64_t insideCp);
std::optional<FiberPieceRange> fiberPieceRange(const FiberPieceContent&, const FiberPieceApproval&);

// FNV-1a 64 over canonical little-endian bytes (docs/fiber-piece-approval.md).
uint64_t fiberControlHash(const FiberPieceContent&, size_t index, FiberPieceAspect);
// Span between control index `lower` and `lower + 1`, its line points walked
// from the control with the lower id to the one with the higher id (stable
// under fiber reversal, independent of the piece's walking direction).
uint64_t fiberSpanHash(const FiberPieceContent&, size_t lower);
struct FiberPieceFingerprints {
    std::vector<FiberSpanFingerprint> spans;
    std::vector<FiberControlFingerprint> controls;
};
FiberPieceFingerprints fingerprintFiberPiece(const FiberPieceContent&, const FiberPieceRange&, FiberPieceAspect);

// Approved rows that no later row supersedes.
std::vector<FiberPieceApproval> liveFiberPieceApprovals(std::span<const FiberPieceApproval>);

struct FiberSpanApprovalState {
    std::optional<size_t> row; // index into FiberPieceApprovalStates::live covering this span
    bool current = false;      // the covering row's fingerprints match this span
};
struct FiberPieceApprovalStates {
    std::vector<FiberPieceApproval> live;
    // Per aspect, one state per span (owner control index).
    std::array<std::vector<FiberSpanApprovalState>, 2> spans;
    // Per aspect: the approved (covered and current) spans.
    std::array<std::vector<size_t>, 2> approvedSpans;
    // Per live row: every covered span is current (false when unresolvable).
    std::vector<bool> rowCurrent;
    std::set<int64_t> lockCpIds;           // referenced by live rows
    std::set<int64_t> releasableLockCpIds; // referenced by superseded rows only
};
FiberPieceApprovalStates fiberPieceApprovalStates(const FiberPieceContent&, std::span<const FiberPieceApproval>);

// The rows that crop every live row reaching into the opened control range
// [openedFirst, openedLast]: a row sharing spans with it is replaced by its
// part(s) outside, re-fingerprinted over `content` (unchanged outside the
// range at that moment); a row entirely inside is revoked; a row of exactly
// the opened range (the piece's own) is left alone. The opened range's end
// controls must already be lock control points. `now` is the ISO-8601 UTC
// stamp recorded on the new rows.
std::vector<FiberPieceApproval> cropFiberPieceApprovals(const FiberPieceContent&,
    std::span<const FiberPieceApproval> live, size_t openedFirst, size_t openedLast,
    const std::string& changedBy, const std::string& now);

// A fresh approval row for the piece, fingerprinted over `content`.
FiberPieceApproval approveFiberPiece(const FiberPieceContent&, const std::string& fiberUuid,
    const FiberPieceRange&, FiberPieceAspect, int64_t fiberVersion,
    const std::string& author, const std::string& approvedAt, const std::string& host);
FiberPieceApproval revokeFiberPieceApproval(const FiberPieceApproval& live,
    const std::string& changedBy, const std::string& now);

class ObjectStore;
class Txn;
std::vector<FiberPieceApproval> readFiberPieceApprovals(const ObjectStore&, const Txn&, const std::string& fiberUuid);
// Through any attachment's reader (ObjectStoreSession::readStore); a hosted
// reader has no extension access and throws.
std::vector<FiberPieceApproval> readFiberPieceApprovals(const StoreReader&, const std::string& fiberUuid);
} // namespace vc::objects
