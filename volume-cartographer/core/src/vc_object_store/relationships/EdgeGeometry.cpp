#include "internal/RecordSql.hpp"

namespace vc::objects {
EdgeGeometryState ObjectStore::edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const
{
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, edgeUuid);
    if (!edge) throw std::invalid_argument("geometry: edge not found");
    Statement binding(db, "SELECT b.a_revision,b.b_revision FROM edge_geometry_bindings b JOIN edges e ON e.id=b.edge_id WHERE e.uuid=?");
    binding.bindBlob(1, parseUuid(edgeUuid));
    EdgeGeometryState result;
    if (binding.row()) {
        if (!binding.isNull(0)) result.aRevision = formatUuid(binding.blob16(0));
        if (!binding.isNull(1)) result.bRevision = formatUuid(binding.blob16(1));
    }
    result.stale = result.aRevision != geometryHead(db, edge->aUuid) || result.bRevision != geometryHead(db, edge->bUuid);
    return result;
}

void ObjectStore::revalidateEdgeGeometry(WriteTxn& txn, const std::string& edgeUuid)
{
    const auto state = edgeGeometryState(txn, edgeUuid);
    if (!state.stale) return;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = *edgeByUuid(txn, edgeUuid);
    const auto id = *edgeIdByUuid(db, parseUuid(edgeUuid));
    exec(db, "SAVEPOINT geometry_revalidate;");
    try {
        Statement s(db, "INSERT INTO edge_geometry_bindings VALUES(?,?,?) ON CONFLICT(edge_id) DO UPDATE SET a_revision=excluded.a_revision,b_revision=excluded.b_revision");
        s.bind(1, id);
        const auto a = geometryHead(db, edge.aUuid), b = geometryHead(db, edge.bUuid);
        if (a) s.bindBlob(2, parseUuid(*a)); else s.bindNull(2);
        if (b) s.bindBlob(3, parseUuid(*b)); else s.bindNull(3);
        s.stepDone();
        stampContentChange(db, author(), hostname(), "edges", id, parseUuid(edgeUuid), "geometry_revalidated");
        exec(db, "RELEASE geometry_revalidate;");
    } catch (...) {
        exec(db, "ROLLBACK TO geometry_revalidate; RELEASE geometry_revalidate;");
        throw;
    }
}

} // namespace vc::objects
