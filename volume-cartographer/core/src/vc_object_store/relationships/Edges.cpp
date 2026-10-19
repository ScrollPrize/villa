#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"

namespace vc::objects {
using namespace detail;
std::string ObjectStore::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    detail::DependencyScope dependencyScope;
    for (const auto& [uuid, anchor] : {std::pair{record.aUuid, record.aAnchor}, std::pair{record.bUuid, record.bAnchor}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    return addEdgeImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

void ObjectStore::updateEdge(WriteTxn& txn, const EdgeRecord& record)
{
    detail::DependencyScope dependencyScope;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto existing = edgeByUuid(txn, record.uuid);
    if (!existing) throw std::runtime_error("object store: edge not found: " + record.uuid);
    auto rec = record;
    canonicalizeEdge(rec);
    if (rec.aUuid != rec.bUuid && rec.aUuid == existing->bUuid && rec.bUuid == existing->aUuid)
        swapEndpoints(rec);
    if (rec.kind != existing->kind || rec.aUuid != existing->aUuid || rec.bUuid != existing->bUuid)
        throw std::runtime_error("object store: edge kind and endpoints cannot change");
    for (const auto& [uuid, anchor] : {std::pair{rec.aUuid, rec.aAnchor}, std::pair{rec.bUuid, rec.bAnchor}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    std::sort(rec.tags.begin(), rec.tags.end());
    rec.tags.erase(std::unique(rec.tags.begin(), rec.tags.end()), rec.tags.end());
    if (rec.aAnchor == existing->aAnchor && rec.bAnchor == existing->bAnchor &&
        rec.tags == existing->tags && nlohmann::json::parse(rec.meta) == nlohmann::json::parse(existing->meta)) return;
    const auto id = *edgeIdByUuid(db, parseUuid(rec.uuid));
    Savepoint savepoint(db);
    Statement s(db, "UPDATE edges SET a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
        "b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,meta=?,a_dependencies=?,b_dependencies=? WHERE id=?");
    bindAnchor(s, 1, rec.aAnchor); bindAnchor(s, 7, rec.bAnchor);
    s.bind(13, rec.meta);
    s.bind(14, dependenciesToJson(rec.aAnchor).dump()); s.bind(15, dependenciesToJson(rec.bAnchor).dump());
    s.bind(16, id); s.stepDone();
    writeTags(db, "edge_tags", "edge_id", id, rec.tags);
    stampContentChange(db, author(), hostname(), "edges", id, parseUuid(rec.uuid), "update_edge");
    savepoint.release();
}

std::optional<EdgeRecord> ObjectStore::edgeByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readEdgeRow(db, s);
}

void ObjectStore::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto bytes = parseUuid(uuid);
    auto id = edgeIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: edge not found: " + uuid);
    Statement s(db, "DELETE FROM edges WHERE id=?");
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge");
    savepoint.release();
}

std::vector<EdgeRecord> ObjectStore::edgesOf(const Txn& txn, const std::string& objectUuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE a_id=? OR b_id=? ORDER BY id");
    s.bind(1, *id);
    s.bind(2, *id);
    return readEdgeRows(db, s);
}

std::vector<EdgeRecord> ObjectStore::edgesByKind(const Txn& txn, const std::string& edgeKind) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE kind=? ORDER BY id");
    s.bind(1, edgeKind);
    return readEdgeRows(db, s);
}

} // namespace vc::objects
