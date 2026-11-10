#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "internal/History.hpp"
#include "internal/RecordOverrides.hpp"
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
    auto normalized = *edge;
    normalized.aAnchor = repair.aAnchor; normalized.bAnchor = repair.bAnchor;
    updateEdge(txn,normalized);
}

EdgeGeometryState ObjectStore::edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const
{
    detail::DependencyScope dependencyScope;
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, edgeUuid);
    if (!edge) throw std::invalid_argument("geometry: edge not found");
    EdgeGeometryState state{geometryHead(db,edge->aUuid),geometryHead(db,edge->bUuid),false};
    if(historyEnabled(db)){
        Statement validated(db,"SELECT recorded_only FROM history_validated_edges WHERE uuid=?",parseUuid(edgeUuid));
        if(validated.row()&&!validated.i64(0))return state;
    }
    const auto archived=recordedRelationship(db,edgeUuid);
    if(!archived)return state;
    const auto a=archived->aGeometryRevision,b=archived->bGeometryRevision;
    state.stale=endpointChanged(*this,txn,edge->aAnchor,a,state.aRevision) ||
                endpointChanged(*this,txn,edge->bAnchor,b,state.bRevision);
    state.aRevision=a;state.bRevision=b;
    return state;
}

} // namespace vc::objects
