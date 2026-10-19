#pragma once

// Read-only access to a unified metadata graph database (vc_graph) from
// VC3D. Opt-in (Phase 6 of the graph DB plan): opened only when a graph
// DB path is configured via the --graph-db flag or the graph/db_path
// setting; with nothing configured VC3D behaves exactly as before.
// All queries go through vc_graph's generic record/edge APIs — this
// controller knows no per-kind query verbs and never touches
// overlapping.json.

#include <QString>

#include <optional>
#include <memory>
#include <string>
#include <vector>

#include "vc/core/graph/GraphDb.hpp"

class GraphDbController {
public:
    // Opens the graph database at path read-only. Returns false and
    // fills *errorMessage on failure (leaving any previously open DB
    // closed). An empty path just closes and returns true.
    bool open(const QString& path, QString* errorMessage = nullptr);
    void close();
    bool isOpen() const { return bool(db_); }
    QString path() const { return path_; }
    // Acquire on the GUI thread; workers retain the open DB through close/reopen.
    std::shared_ptr<vc::graph::GraphDb> readHandle() const { requireDb(); return db_; }

    // Query passthroughs. Each runs in its own short read transaction.
    // Throws std::runtime_error when no graph DB is open or on a
    // vc_graph error (unknown UUID, malformed UUID, ...).
    std::optional<vc::graph::ObjectRecord> objectByUuid(const std::string& uuid) const;
    std::vector<vc::graph::ObjectRecord> objectsByKind(const std::string& kind) const;
    std::vector<std::string> neighbours(const std::string& uuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const;
    std::vector<vc::graph::EdgeRecord> edgesOf(const std::string& uuid) const;
    std::vector<std::string> componentOf(const std::string& uuid,
                                         const std::vector<std::string>& edgeKindFilter = {}) const;

private:
    vc::graph::GraphDb& requireDb() const;
    std::shared_ptr<vc::graph::GraphDb> db_;
    QString path_;
};
