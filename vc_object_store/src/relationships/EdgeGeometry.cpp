#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "internal/History.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects {
using namespace detail;
void ObjectStore::repairEdgeGeometry(WriteTxn& txn, const EdgeGeometryRepair& repair)
{
    detail::DependencyScope dependencyScope;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, repair.edgeUuid);
    if (!edge) throw std::runtime_error("geometry repair: edge not found");
    auto canonical = [](const std::optional<std::string>& revision) -> std::optional<std::string> {
        return revision ? std::optional(canonicalGeometryToken(*revision)) : std::nullopt;
    };
    const auto aRevision = canonical(repair.aRevision), bRevision = canonical(repair.bRevision);
    if (geometryHead(db, edge->aUuid) != aRevision || geometryHead(db, edge->bUuid) != bRevision)
        historyConflict(db, "geometry repair: endpoint revision changed; recompute repair");
    if (edge->version != repair.expectedVersion)
        historyConflict(db, "geometry repair: edge version changed; recompute repair");
    auto a = repair.aAnchor, b = repair.bAnchor;
    for (const auto& [uuid, anchor] : {std::pair{edge->aUuid, a}, std::pair{edge->bUuid, b}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    auto normalized = *edge;
    normalized.aAnchor = a; normalized.bAnchor = b;
    canonicalizeEdge(normalized);
    if (normalized.aUuid != edge->aUuid) swapEndpoints(normalized);
    a = normalized.aAnchor; b = normalized.bAnchor;
    const auto state = edgeGeometryState(txn, repair.edgeUuid);
    if (state.aRevision == aRevision && state.bRevision == bRevision && a == edge->aAnchor && b == edge->bAnchor) {
        Statement changedGeometry(db,"SELECT 1 FROM objects WHERE uuid IN (?,?) AND commit_sequence=? LIMIT 1",
            parseUuid(edge->aUuid),parseUuid(edge->bUuid),writeCommitSequence(db));
        if (edge->commitSequence!=writeCommitSequence(db)&&changedGeometry.row())
            stampContentChange(db, author(), hostname(), "edges", 0,
                               parseUuid(edge->uuid), "geometry_revalidated");
        recordRelationshipValidation(db, *edge);
        return;
    }
    requireEdgeColumns(db,{"a_anchor_x","a_anchor_y","a_anchor_z","a_dir_x","a_dir_y","a_dir_z",
                           "b_anchor_x","b_anchor_y","b_anchor_z","b_dir_x","b_dir_y","b_dir_z","a_dependencies","b_dependencies"});
    Savepoint savepoint(db);
    Statement anchors(db, "UPDATE edges SET a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
        "b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,a_dependencies=?,b_dependencies=? WHERE uuid=?");
    bindAnchor(anchors, 1, a); bindAnchor(anchors, 7, b);
    anchors.bind(13, dependenciesToJson(a).dump()); anchors.bind(14, dependenciesToJson(b).dump());
    anchors.bind(15, parseUuid(repair.edgeUuid)); anchors.stepDone();
    stampContentChange(db, author(), hostname(), "edges", 0, parseUuid(repair.edgeUuid), "geometry_repaired");
    recordRelationshipValidation(db, normalized);
    savepoint.release();
}

EdgeGeometryState ObjectStore::edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const
{
    detail::DependencyScope dependencyScope;
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, edgeUuid);
    if (!edge) throw std::invalid_argument("geometry: edge not found");
    EdgeGeometryState state{geometryHead(db,edge->aUuid),geometryHead(db,edge->bUuid),false};
    if (!historyEnabled(db)) return state;
    Statement validated(db,"SELECT recorded_only FROM history_validated_edges WHERE uuid=?",parseUuid(edgeUuid));
    if (validated.row() && !validated.i64(0)) return state;
    Statement archived(db,"SELECT r.a_geometry,r.b_geometry FROM object_relationship_heads h "
        "JOIN relationship_revisions r ON r.uuid=h.relationship_uuid WHERE h.edge_uuid=? LIMIT 1",parseUuid(edgeUuid));
    if (!archived.row()) return state;
    const auto pin=[&](int column)->std::optional<std::string> {
        return archived.isNull(column)?std::nullopt:std::optional(readGeometryTokenColumn(archived,column));
    };
    const auto a=pin(0), b=pin(1);
    state.stale=endpointChanged(*this,txn,edge->aAnchor,a,state.aRevision) ||
                endpointChanged(*this,txn,edge->bAnchor,b,state.bRevision);
    state.aRevision=a;state.bRevision=b;
    return state;
}

} // namespace vc::objects
