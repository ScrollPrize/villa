#include "internal/EdgeSql.hpp"

namespace vc::objects {
std::string ObjectStore::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    return addEdgeImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

void ObjectStore::updateEdge(WriteTxn& txn, const EdgeRecord& record)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto existing = edgeByUuid(txn, record.uuid);
    if (!existing) throw std::runtime_error("object store: edge not found: " + record.uuid);
    auto rec = record;
    rec.aUuid = formatUuid(parseUuid(rec.aUuid));
    rec.bUuid = formatUuid(parseUuid(rec.bUuid));
    if (rec.aUuid == existing->bUuid && rec.bUuid == existing->aUuid) {
        std::swap(rec.aUuid, rec.bUuid); std::swap(rec.aAnchor, rec.bAnchor);
    }
    if (rec.kind != existing->kind || rec.aUuid != existing->aUuid || rec.bUuid != existing->bUuid)
        throw std::runtime_error("object store: edge kind and endpoints cannot change");
    auto sameAnchor = [](const Anchor& a, const Anchor& b) {
        return a.index == b.index && a.position == b.position && a.direction == b.direction;
    };
    std::sort(rec.tags.begin(), rec.tags.end());
    rec.tags.erase(std::unique(rec.tags.begin(), rec.tags.end()), rec.tags.end());
    if (sameAnchor(rec.aAnchor, existing->aAnchor) && sameAnchor(rec.bAnchor, existing->bAnchor) &&
        rec.tags == existing->tags && nlohmann::json::parse(rec.meta) == nlohmann::json::parse(existing->meta)) return;
    const auto id = *edgeIdByUuid(db, parseUuid(rec.uuid));
    exec(db, "SAVEPOINT edge_refresh;");
    try {
        Statement s(db, "UPDATE edges SET a_anchor_index=?,a_anchor_x=?,a_anchor_y=?,a_anchor_z=?,a_dir_x=?,a_dir_y=?,a_dir_z=?,"
            "b_anchor_index=?,b_anchor_x=?,b_anchor_y=?,b_anchor_z=?,b_dir_x=?,b_dir_y=?,b_dir_z=?,meta=? WHERE id=?");
        bindAnchor(s, 1, rec.aAnchor); bindAnchor(s, 8, rec.bAnchor);
        s.bind(15, rec.meta); s.bind(16, id); s.stepDone();
        writeTags(db, "edge_tags", "edge_id", id, rec.tags);
        stampContentChange(db, author(), hostname(), "edges", id, parseUuid(rec.uuid), "update_edge");
        exec(db, "RELEASE edge_refresh;");
    } catch (...) {
        exec(db, "ROLLBACK TO edge_refresh; RELEASE edge_refresh;");
        throw;
    }
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
    const auto bytes = parseUuid(uuid);
    auto id = edgeIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: edge not found: " + uuid);
    Statement s(db, "DELETE FROM edges WHERE id=?");
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge");
}

std::vector<EdgeRecord> ObjectStore::edgesOf(const Txn& txn, const std::string& objectUuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("object store: object not found: " + objectUuid);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE a_id=? OR b_id=? ORDER BY id");
    s.bind(1, *id);
    s.bind(2, *id);
    std::vector<EdgeRecord> out;
    while (s.row()) out.push_back(readEdgeRow(db, s));
    return out;
}

std::vector<EdgeRecord> ObjectStore::edgesByKind(const Txn& txn, const std::string& edgeKind) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE kind=? ORDER BY id");
    s.bind(1, edgeKind);
    std::vector<EdgeRecord> out;
    while (s.row()) out.push_back(readEdgeRow(db, s));
    return out;
}

} // namespace vc::objects
