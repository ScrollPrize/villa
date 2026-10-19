#pragma once
#include "ObjectStoreInternal.hpp"
namespace vc::objects { namespace {
const char* kEdgeCols =
    "id,uuid,a_id,b_id,kind,generation,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_generation,"
    "a_anchor_index,a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
    "b_anchor_index,b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta";

Anchor readAnchor(const Statement& s, int firstCol)
{
    Anchor a;
    if (!s.isNull(firstCol)) a.index = s.i64(firstCol);
    if (!s.isNull(firstCol + 1)) a.position = std::array<double, 3>{s.f64(firstCol + 1), s.f64(firstCol + 2), s.f64(firstCol + 3)};
    if (!s.isNull(firstCol + 4)) a.direction = std::array<double, 3>{s.f64(firstCol + 4), s.f64(firstCol + 5), s.f64(firstCol + 6)};
    return a;
}

void bindAnchor(Statement& s, int firstIdx, const Anchor& a)
{
    if (a.index) s.bind(firstIdx, *a.index); else s.bindNull(firstIdx);
    for (int i = 0; i < 3; ++i) {
        if (a.position) s.bind(firstIdx + 1 + i, (*a.position)[size_t(i)]); else s.bindNull(firstIdx + 1 + i);
    }
    for (int i = 0; i < 3; ++i) {
        if (a.direction) s.bind(firstIdx + 4 + i, (*a.direction)[size_t(i)]); else s.bindNull(firstIdx + 4 + i);
    }
}

std::string uuidTextByObjectId(sqlite3* db, int64_t id)
{
    Statement s(db, "SELECT uuid FROM objects WHERE id=?");
    s.bind(1, id);
    if (!s.row()) throw std::runtime_error("object store: dangling edge endpoint id");
    return formatUuid(s.blob16(0));
}

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s)
{
    EdgeRecord rec;
    const int64_t id = s.i64(0);
    const int64_t aId = s.i64(2);
    const int64_t bId = s.i64(3);
    readCommon(rec, s, 1, 4, 5);
    rec.aAnchor = readAnchor(s, 17);
    rec.bAnchor = readAnchor(s, 24);
    rec.meta = s.text(31);
    rec.aUuid = uuidTextByObjectId(db, aId);
    rec.bUuid = uuidTextByObjectId(db, bId);
    Statement tags(db, "SELECT tag FROM edge_tags WHERE edge_id=? ORDER BY tag");
    tags.bind(1, id);
    while (tags.row()) rec.tags.push_back(tags.text(0));
    return rec;
}

std::string addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input,
                        bool preserveApproval, bool preserveRecord = false)
{
    if (input.kind.empty())
        throw std::runtime_error("object store: edge kind must be non-empty");
    EdgeRecord rec = input;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    if (objectIdByUuid(db, parseUuid(rec.uuid)))
        throw std::runtime_error("object store: UUID already belongs to an object");
    const auto aUuid = parseUuid(rec.aUuid);
    const auto bUuid = parseUuid(rec.bUuid);
    auto aId = objectIdByUuid(db, aUuid);
    auto bId = objectIdByUuid(db, bUuid);
    if (!aId) throw std::runtime_error("object store: edge endpoint not found: " + rec.aUuid);
    if (!bId) throw std::runtime_error("object store: edge endpoint not found: " + rec.bUuid);
    if (*aId == *bId)
        throw std::runtime_error("object store: self-edges are not allowed");
    if (*aId > *bId) {
        std::swap(aId, bId);
        std::swap(rec.aUuid, rec.bUuid);
        std::swap(rec.aAnchor, rec.bAnchor);
    }
    const std::string now = nowIsoUtc();
    Statement s(db, "INSERT INTO edges(uuid,a_id,b_id,kind,generation,created_by,created_at,created_host,changed_by,changed_at,"
                    "changed_host,approval_state,approved_by,approved_at,approved_host,approved_generation,"
                    "a_anchor_index,a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
                    "b_anchor_index,b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    s.bindBlob(1, parseUuid(rec.uuid));
    s.bind(2, *aId); s.bind(3, *bId); s.bind(4, rec.kind); s.bind(5, int64_t(1));
    s.bind(6, author); s.bind(7, now); s.bind(8, host);
    s.bind(9, author); s.bind(10, now); s.bind(11, host);
    const std::string state = preserveApproval ? rec.approvalState : "pending";
    s.bind(12, state.empty() ? std::string("pending") : state);
    if (preserveApproval && rec.approvedBy) s.bind(13, *rec.approvedBy); else s.bindNull(13);
    if (preserveApproval && rec.approvedAt) s.bind(14, *rec.approvedAt); else s.bindNull(14);
    if (preserveApproval && rec.approvedHost) s.bind(15, *rec.approvedHost); else s.bindNull(15);
    if (preserveApproval && rec.approvedGeneration) s.bind(16, *rec.approvedGeneration); else s.bindNull(16);
    bindAnchor(s, 17, rec.aAnchor);
    bindAnchor(s, 24, rec.bAnchor);
    s.bind(31, rec.meta);
    s.stepDone();
    const int64_t id = sqlite3_last_insert_rowid(db);
    Statement binding(db, "INSERT INTO edge_geometry_bindings(edge_id,a_revision,b_revision) VALUES(?,"
        "(SELECT revision_uuid FROM geometry_heads WHERE object_uuid=?),"
        "(SELECT revision_uuid FROM geometry_heads WHERE object_uuid=?))");
    binding.bind(1, id); binding.bindBlob(2, parseUuid(rec.aUuid)); binding.bindBlob(3, parseUuid(rec.bUuid));
    binding.stepDone();
    writeTags(db, "edge_tags", "edge_id", id, rec.tags);
    if (preserveRecord) restoreCommon(db, "edges", rec);
    appendJournal(db, author, host, parseUuid(rec.uuid), preserveRecord ? rec.generation : 1, preserveRecord ? "import" : "add_edge");
    return rec.uuid;
}


} }
