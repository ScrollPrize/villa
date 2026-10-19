#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"

namespace vc::objects {
void ObjectStore::repairEdgeGeometry(WriteTxn& txn, const EdgeGeometryRepair& repair)
{
    detail::DependencyScope dependencyScope;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, repair.edgeUuid);
    if (!edge) throw std::runtime_error("geometry repair: edge not found");
    auto canonical = [](const std::optional<std::string>& revision) -> std::optional<std::string> {
        return revision ? std::optional(formatUuid(parseUuid(*revision))) : std::nullopt;
    };
    const auto aRevision = canonical(repair.aRevision), bRevision = canonical(repair.bRevision);
    if (geometryHead(db, edge->aUuid) != aRevision || geometryHead(db, edge->bUuid) != bRevision)
        throw std::runtime_error("geometry repair: endpoint revision changed; recompute repair");
    if (edge->generation != repair.expectedGeneration)
        throw std::runtime_error("geometry repair: edge generation changed; recompute repair");
    auto a = repair.aAnchor, b = repair.bAnchor;
    for (const auto& [uuid, anchor] : {std::pair{edge->aUuid, a}, std::pair{edge->bUuid, b}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    for (const auto* anchor : {&a, &b}) {
        for (const auto* vector : {&anchor->position, &anchor->direction})
            if (*vector) for (double value : **vector)
                if (!std::isfinite(value)) throw std::invalid_argument("geometry repair: anchors must be finite");
    }
    if (edge->aUuid == edge->bUuid) {
        if (a.points.empty() || b.points.empty() || a.points.front() == b.points.front())
            throw std::invalid_argument("geometry repair: self-edges require distinct point references");
        if (dependenciesToJson(b).dump() < dependenciesToJson(a).dump()) std::swap(a, b);
    }
    auto same = [](const Anchor& x, const Anchor& y) {
        return x.position == y.position && x.direction == y.direction &&
            x.points == y.points && x.wholeGeometry == y.wholeGeometry;
    };
    const auto state = edgeGeometryState(txn, repair.edgeUuid);
    if (state.aRevision == aRevision && state.bRevision == bRevision && same(a, edge->aAnchor) && same(b, edge->bAnchor)) return;
    const auto id = *edgeIdByUuid(db, parseUuid(repair.edgeUuid));
    exec(db, "SAVEPOINT geometry_repair;");
    try {
        Statement anchors(db, "UPDATE edges SET a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
            "b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,a_dependencies=?,b_dependencies=? WHERE id=?");
        bindAnchor(anchors, 1, a); bindAnchor(anchors, 7, b);
        anchors.bind(13, dependenciesToJson(a).dump()); anchors.bind(14, dependenciesToJson(b).dump());
        anchors.bind(15, id); anchors.stepDone();
        Statement binding(db, "INSERT INTO edge_geometry_bindings VALUES(?,?,?) ON CONFLICT(edge_id) DO UPDATE SET a_revision=excluded.a_revision,b_revision=excluded.b_revision");
        binding.bind(1, id);
        if (aRevision) binding.bindBlob(2, parseUuid(*aRevision)); else binding.bindNull(2);
        if (bRevision) binding.bindBlob(3, parseUuid(*bRevision)); else binding.bindNull(3);
        binding.stepDone();
        stampContentChange(db, author(), hostname(), "edges", id, parseUuid(repair.edgeUuid), "geometry_repaired");
        exec(db, "RELEASE geometry_repair;");
    } catch (...) {
        exec(db, "ROLLBACK TO geometry_repair; RELEASE geometry_repair;");
        throw;
    }
}

EdgeGeometryState ObjectStore::edgeGeometryState(const Txn& txn, const std::string& edgeUuid) const
{
    detail::DependencyScope dependencyScope;
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
    result.stale = detail::endpointChanged(*this, txn, edge->aAnchor, result.aRevision, geometryHead(db, edge->aUuid)) ||
        detail::endpointChanged(*this, txn, edge->bAnchor, result.bRevision, geometryHead(db, edge->bUuid));
    return result;
}

} // namespace vc::objects
