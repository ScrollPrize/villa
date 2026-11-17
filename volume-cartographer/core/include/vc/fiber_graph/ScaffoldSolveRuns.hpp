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
};

struct ScaffoldRunRows {
    // In creation order; a writer creates each table before appending to it.
    std::vector<objects::ExtensionTable> tables;
    std::vector<std::pair<std::string, std::vector<objects::ExtensionRow>>> rows;
};

// The rows of one solve. `method` names the backend the result came from.
ScaffoldRunRows scaffoldSolveRunRows(const ScaffoldGraph&, const ScaffoldSolveResult&, const ScaffoldSolveOptions&,
                                     const ScaffoldRunRecord&);

struct ScaffoldStoredRun {
    std::string runUuid, stage, method, settingsJson;
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
