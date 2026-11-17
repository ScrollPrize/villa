#pragma once
// Piece-wise approval of a fiber in Line Annotation (docs/line_annotation_fibers.md,
// "Piece approval and lock control points"). Pure session-side logic over the
// store's ext_fiber_piece_approval rows (vc_object_store
// FiberPieceApprovalExtension.hpp): which controls and spans an open session
// may edit, how a planned piece's lock controls enter the session, what the
// views dim, and what a save must verify.
#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationStoredFiber.hpp"
#include <vc_object_store/objects/FiberPieceApprovalExtension.hpp>
#include <opencv2/core.hpp>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vc3d::line_annotation {

using FiberPieceApproval = vc::objects::FiberPieceApproval;
using FiberPieceAspect = vc::objects::FiberPieceAspect;
using FiberPieceContent = vc::objects::FiberPieceContent;

// One end of a planned piece: an existing control (by stable id), a line
// index where a lock control is inserted, or the fiber end (no lock).
struct PiecePlanEnd {
    std::optional<int64_t> storeId;
    std::optional<int> lineIndex;
    bool fiberEnd = false;
    // The session identity (LineControlPoint::identity) of the resolved lock
    // control, assigned when the plan is applied to a session; 0 until then
    // and for a fiber end.
    uint64_t identity = 0;
    bool operator==(const PiecePlanEnd&) const = default;
};
// Ends in line order: `a` is the lower line position. A fiber-end end opens
// the piece to that side's end of the fiber.
struct PiecePlan {
    PiecePlanEnd a, b;
    bool operator==(const PiecePlan&) const = default;
};

// The fiber as the save writes it: stable ids, positions, line points and the
// control annotations (storedControlPointToJson). Fingerprints taken over this
// equal those the store computes over the stored revision.
FiberPieceContent pieceContentOfStoredFiber(const StoredFiber& fiber);

// Lock controls carry the lock tag and an explicit direction.
bool isLockControl(const LineControlPoint& control);
// Central chord tangent of the line at `index` (one-sided at the ends), unit length.
cv::Vec3d lineTangentAt(const std::vector<cv::Vec3d>& linePoints, int index);
// Give the control the lock tag and, when it has none, the line's tangent as
// its direction (so tracing against it is fixed).
void lockControl(LineControlPoint& control, const std::vector<cv::Vec3d>& linePoints);

struct LockControlInsertion {
    std::vector<LineControlPoint> controls;
    std::vector<size_t> oldToNewIndices;
    size_t index = 0; // of the inserted control
};
// A lock control exactly on line point `lineIndex`: no line splice, the
// enclosing span's descriptor copied to both halves, no dirty spans. Throws
// when the index is outside the line or a control already sits there.
LockControlInsertion insertLockControl(const std::vector<LineControlPoint>& controls,
                                       const std::vector<cv::Vec3d>& linePoints, int lineIndex);

struct PieceLockState {
    std::optional<PiecePlan> plan;
    // The plan's lock controls exist only in the session so far: nothing is
    // saved until the first edit, and leaving without one changes nothing.
    bool pendingFirstEdit = false;
    bool firstEditDone = false;
    // Inclusive session control index range of the open piece.
    std::optional<std::pair<size_t, size_t>> openedControls;
    std::vector<bool> lockedControls;    // per control: fixed in position, direction and tags
    std::vector<bool> lockedSpans;       // per span (owner control index): no geometry edit
    std::vector<bool> normalsLockedSpans; // per span: no display-normal edit
    // Per aspect and span: the live row covering it (index into states.live)
    // and whether that row is current there.
    std::array<std::vector<std::optional<size_t>>, 2> coveringRow;
    std::array<std::vector<bool>, 2> coveringCurrent;
    std::vector<FiberPieceApproval> rows; // every row of the fiber as last read
    std::optional<FiberPieceContent> storedContent; // the fiber as read from the store
    vc::objects::FiberPieceApprovalStates states;    // over storedContent
    bool pieceOpen() const { return openedControls.has_value(); }
    bool savesHeld() const { return pendingFirstEdit && !firstEditDone; }
    bool anyLocked() const;
};

// The lock set of a session. Without an open piece, locked spans are the
// approved ones (mapped from the stored content by stable id); with one
// (`plan`, its ends resolved to control identities), every span outside it.
// Lock-tagged controls are always locked. A plan whose lock controls are
// gone from the session opens nothing. `previous` keeps the first-edit flags.
PieceLockState computePieceLockState(const std::vector<LineControlPoint>& controls,
                                     std::vector<FiberPieceApproval> rows,
                                     std::optional<FiberPieceContent> storedContent,
                                     const std::optional<PiecePlan>& plan,
                                     const PieceLockState* previous = nullptr);
// Inclusive control index range of the plan in `controls`, by identity.
std::optional<std::pair<size_t, size_t>> plannedControlRange(const PiecePlan& plan,
                                                             const std::vector<LineControlPoint>& controls);
// Whether the open tails on the left / right stay as they are during a solve
// (that side is outside the open piece).
std::pair<bool, bool> pieceTailRetention(const PieceLockState& state, const std::vector<LineControlPoint>& controls);

// What the two approval checkboxes show for the open piece.
struct PieceApprovalBoxState {
    bool pieceOpen = false;
    std::array<bool, 2> approved{false, false}; // per aspect: every span of the piece covered and current
    std::array<bool, 2> stale{false, false};    // covered by a row that no longer matches
};
PieceApprovalBoxState pieceApprovalBoxState(const PieceLockState& state);

// Why an edit dirtying `dirtySpans` and touching `touchedControls` is refused,
// or nullopt when it is allowed.
std::optional<std::string> pieceLockRefusal(const PieceLockState& state, std::span<const size_t> dirtySpans,
                                            std::span<const size_t> touchedControls,
                                            FiberPieceAspect aspect = FiberPieceAspect::Geometry);

// The session's line-position ranges the views dim: everything outside the
// opened range, tails included. Empty without an open piece.
std::vector<std::pair<double, double>> dimmedLineRanges(const PieceLockState& state,
                                                        const std::vector<LineControlPoint>& controls,
                                                        size_t linePointCount);
struct PieceDisplayRanges {
    std::vector<std::pair<double, double>> dimmed;
    std::vector<std::pair<double, double>> approved; // geometry aspect, current
    std::vector<std::pair<double, double>> stale;    // covered by a row that no longer matches
};
PieceDisplayRanges pieceDisplayRanges(const PieceLockState& state, const std::vector<LineControlPoint>& controls,
                                      size_t linePointCount);

// What a save of the fiber must verify, taken from the session at snapshot
// time in stored (sorted) control indices.
struct PieceSaveCheck {
    bool pieceOpen = false;
    std::optional<std::pair<size_t, size_t>> openedStored; // inclusive stored control index range
    // Whether the fiber before / after the opened range is locked (false on
    // the open side of an end piece, whose tail may be re-traced).
    bool leftLocked = true, rightLocked = true;
    // Stable id pairs of the spans locked by approvals (normal open).
    std::vector<std::pair<int64_t, int64_t>> lockedSpanIds;
    // The fiber as read when the session opened (piece open).
    std::optional<FiberPieceContent> baseline;
};
PieceSaveCheck pieceSaveCheck(const PieceLockState& state, const std::vector<LineControlPoint>& controls,
                              const std::vector<int>& storedIndexForSessionIndex);
// The violation a save of `saved` would commit, or nullopt. Normal open: every
// locked span must still be current for the row covering it. Piece open: the
// line points and control annotations outside the opened range must equal the
// baseline.
std::optional<std::string> pieceSaveViolation(const FiberPieceContent& saved, const PieceSaveCheck& check,
                                              std::span<const FiberPieceApproval> rows);

// The approval range of the open piece in `content` (the saved fiber), or
// nullopt when the plan's controls are not in it yet.
std::optional<vc::objects::FiberPieceRange> openedPieceRange(const PieceLockState& state,
                                                             const std::vector<LineControlPoint>& controls,
                                                             const FiberPieceContent& content);

} // namespace vc3d::line_annotation
