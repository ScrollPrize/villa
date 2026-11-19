#pragma once
// Scaffold solve runs as extension rows of the store they were solved in.
//
// A run is one solve: its settings, the winding and component of every fiber,
// its metrics, and the ranked residuals with the cycle pieces they accuse.
// The rows live in the scaffold store itself (`ext_fiber_graph_solve_*`), so
// a solve needs no result store and the fiber map, the command line and
// Python all read the same ranking.
//
// Nothing here opens or writes a store: the rows are built here and the
// caller submits them (VC3D through its checked request, the CLI through its
// own transaction). Reading goes through any StoreReader.
#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vc::fiber_graph {

struct ScaffoldRunRecord {
    // Caller-minted run UUID; the rows of one solve share it.
    std::string runUuid;
    // The store state that was solved, recorded as the run's input.
    objects::ExtensionSource input;
    std::string stage = "winding";
    // When the run was made, Unix seconds UTC, 0 when unstated. The caller
    // states it: the store records commits, not when a solve was computed.
    std::int64_t createdAtUnix = 0;
};

struct ScaffoldRunRows {
    // In creation order; a writer creates each table before appending to it.
    std::vector<objects::ExtensionTable> tables;
    std::vector<std::pair<std::string, std::vector<objects::ExtensionRow>>> rows;
};

// What a run did beyond the links: the normals it solved, the signs those
// decided and the measured crossings it assembled. Every member is optional;
// a links-only run passes none and the rows record only the link settings.
struct ScaffoldRunStages {
    const ScaffoldNormalOptions* normalOptions = nullptr;
    const ScaffoldNormalReport* normals = nullptr;
    const ScaffoldEventOptions* eventOptions = nullptr;
    const ScaffoldEventReport* events = nullptr;
    const ScaffoldMeasuredOptions* measuredOptions = nullptr;
    const ScaffoldMeasuredReport* measured = nullptr;
    const ScaffoldSignOptions* signOptions = nullptr;
    const ScaffoldSignResult* signs = nullptr;
    const ScaffoldApprovalReport* approvals = nullptr;
};

// The rows of one solve. `method` names the backend the result came from.
ScaffoldRunRows scaffoldSolveRunRows(const ScaffoldGraph&, const ScaffoldSolveResult&, const ScaffoldSolveOptions&,
                                     const ScaffoldRunRecord&, const ScaffoldRunStages& = {});

struct ScaffoldStoredRun {
    std::string runUuid, stage, method, settingsJson;
    // Read out of the settings, for the columns the docks show: when the run
    // was made (Unix seconds UTC, 0 when the run recorded none), which
    // normals it used ("geometry" or "fused") and which sign backend
    // ("irls", "ceres", or empty for a run that solved no signs).
    std::int64_t createdAtUnix = 0;
    std::string normalSource, signBackend;
    // The store state the run read, as it recorded it (role "input_0"). A run
    // whose generation or commit sequence is behind the store's own was
    // computed from an annotation that has changed since.
    objects::ExtensionSource input;
    // Metric rows of the run, by name (scope "all").
    std::map<std::string, double> metrics;
    std::map<std::string, std::int64_t> counts;
    std::int64_t fibers = 0, residuals = 0;
};

// Every solve run recorded in the store, newest commit last. `stage` filters
// (empty: all stages). A store without the tables yields nothing.
std::vector<ScaffoldStoredRun> readScaffoldRuns(const objects::StoreReader&, const std::string& stage = "winding");

struct ScaffoldStoredPiece {
    std::string fiberUuid;
    double arcFrom = 0, arcTo = 0, score = 0;
    std::optional<std::int64_t> controlFrom, controlTo;
};
struct ScaffoldStoredResidual {
    std::int64_t index = 0, kind = 0, target = 0;
    std::string fiberA, fiberB, edgeUuid;
    double arcA = 0, arcB = 0, weight = 0, residual = 0, cyclePieceLength = 0;
    std::optional<std::int64_t> cycleHops, redundancy;
    std::vector<ScaffoldStoredPiece> pieces;
};

// The ranked residuals of one run, in stored order (worst cost first).
std::vector<ScaffoldStoredResidual> readScaffoldResiduals(const objects::StoreReader&, const std::string& runUuid);

// Per-fiber windings of one run, by fiber UUID.
struct ScaffoldStoredWinding {
    std::int64_t winding = 0, component = 0;
    std::optional<std::int64_t> family, normalSign;
};
std::map<std::string, ScaffoldStoredWinding> readScaffoldWindings(const objects::StoreReader&,
                                                                  const std::string& runUuid);

} // namespace vc::fiber_graph
