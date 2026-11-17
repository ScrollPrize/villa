#pragma once
// Approval of one measured winding constraint (docs/extensions.md).
//
// A link between two fibers carries its own approval state on the edge
// record. A *measured* constraint - a crossing the extraction found, not an
// annotation - has no record to carry one, so its approval lives here: an
// immutable row naming the constraint and the two fiber stretches it was
// approved over. Revoking appends a superseding row; live rows of one
// constraint never overlap.
//
// A constraint is named by its two fibers in UUID order, its kind and both
// anchors quantized to a fixed arc step. A re-extraction places the same
// crossing within one step of where it was, so the approval survives it,
// while a different crossing of the same pair keeps its own identity. The two
// span fingerprints are the piece-approval ones (docs/fiber-piece-approval.md):
// editing either covering span invalidates the approval, exactly as it
// invalidates a piece approval.
#include "vc_object_store/objects/FiberPieceApprovalExtension.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {
class StoreReader;

inline constexpr const char* kFiberConstraintApprovalTable = "ext_fiber_constraint_approval";
inline constexpr const char* kFiberConstraintApprovalExtension = "fiber-constraint-approval";
inline constexpr int64_t kFiberConstraintApprovalFormat = 1;
// The arc step an anchor is quantized to, in base voxels: the extraction's
// own event spacing, so one crossing keeps one identity.
inline constexpr double kFiberConstraintArcQuantum = 64.0;

// The measured relation kinds, numbered as the solve's residual rows number
// them. A manual link is not one of these: its approval is its edge state.
enum class FiberConstraintKind : int64_t { Perpendicular = 1, Parallel = 2, Enclosure = 3 };
enum class FiberConstraintApprovalState : int64_t { Approved = 1, Revoked = 2 };

// The identity of one measured constraint. `fiberA` sorts before `fiberB`,
// and the arcs follow that order.
struct FiberConstraintIdentity {
    std::string fiberA, fiberB;
    FiberConstraintKind kind = FiberConstraintKind::Perpendicular;
    int64_t arcA = 0, arcB = 0; // quantized anchor arcs, in quanta
    bool operator==(const FiberConstraintIdentity&) const = default;
    auto operator<=>(const FiberConstraintIdentity&) const = default;
};

// The identity of a crossing of two fibers at two anchor arcs (base voxels),
// in whichever order the caller holds them.
FiberConstraintIdentity fiberConstraintIdentity(const std::string& fiberA, double arcA, const std::string& fiberB,
                                                double arcB, FiberConstraintKind,
                                                double quantum = kFiberConstraintArcQuantum);

struct FiberConstraintApproval {
    FiberConstraintIdentity identity;
    // `approvalUuid` names the approval event and is shared by the rows
    // derived from it; `rowUuid` is unique per row.
    std::string rowUuid, approvalUuid;
    FiberConstraintApprovalState state = FiberConstraintApprovalState::Approved;
    std::optional<std::string> supersedes;
    int64_t fiberVersionA = 0, fiberVersionB = 0;
    // The span of each fiber covering its anchor; a revocation carries none.
    std::optional<FiberSpanFingerprint> spanA, spanB;
    std::string author, approvedAt, host;
    std::optional<std::string> changedBy, changedAt;
    bool operator==(const FiberConstraintApproval&) const = default;
};

ExtensionTable fiberConstraintApprovalTable();
ExtensionRow fiberConstraintApprovalRow(const FiberConstraintApproval&);
FiberConstraintApproval decodeFiberConstraintApprovalRow(const ExtensionRow&);

// The span of a fiber that covers an arc position: the span between the two
// control points around it, by the line index of each control.
std::optional<size_t> fiberSpanCoveringLineIndex(const FiberPieceContent&, int64_t lineIndex);

// A fresh approval row for one constraint, fingerprinted over both fibers.
// Throws std::invalid_argument when a span index is outside its fiber.
FiberConstraintApproval approveFiberConstraint(const FiberConstraintIdentity&, const FiberPieceContent& a,
                                               size_t spanA, int64_t fiberVersionA, const FiberPieceContent& b,
                                               size_t spanB, int64_t fiberVersionB, const std::string& author,
                                               const std::string& approvedAt, const std::string& host);
FiberConstraintApproval revokeFiberConstraintApproval(const FiberConstraintApproval& live,
                                                      const std::string& changedBy, const std::string& now);

// Approved rows that no later row supersedes.
std::vector<FiberConstraintApproval> liveFiberConstraintApprovals(std::span<const FiberConstraintApproval>);

// Whether a live row still describes the store: both covering spans hash to
// what the row recorded.
bool fiberConstraintApprovalCurrent(const FiberConstraintApproval&, const FiberPieceContent& a, size_t spanA,
                                    const FiberPieceContent& b, size_t spanB);

// Every row of the table, in primary-key order. A store without the table
// yields nothing.
std::vector<FiberConstraintApproval> readFiberConstraintApprovals(const StoreReader&);
// One fiber's rows (either endpoint).
std::vector<FiberConstraintApproval> readFiberConstraintApprovals(const StoreReader&, const std::string& fiberUuid);
} // namespace vc::objects
