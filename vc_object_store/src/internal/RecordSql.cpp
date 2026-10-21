#include "internal/RecordSql.hpp"
#include "internal/RecordFields.hpp"
#include "internal/GeometryCoordinates.hpp"

namespace vc::objects::detail {
const char* kObjectCols =
    "id,uuid,kind,name,generation,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_generation,format_version,storage_kind,"
    "storage_ref,data_generation,data_checksum,meta";

bool wants(const ObjectFields& fields, const std::string& name) {
    return !fields || fields->contains(name) || name == "uuid" || name == "kind";
}

std::string projectedObjectColumns(const ObjectFields& fields)
{
    if (!fields) return kObjectCols;
    const std::vector<std::pair<std::string, std::string>> columns{
        {"uuid", "NULL"}, {"kind", "''"}, {"name", "''"}, {"generation", "0"},
        {"created_by", "''"}, {"created_at", "''"}, {"created_host", "''"},
        {"changed_by", "''"}, {"changed_at", "''"}, {"changed_host", "''"},
        {"approval_state", "''"}, {"approved_by", "NULL"}, {"approved_at", "NULL"},
        {"approved_host", "NULL"}, {"approved_generation", "NULL"}, {"format_version", "0"},
        {"storage_kind", "''"}, {"storage_ref", "''"}, {"data_generation", "0"},
        {"data_checksum", "NULL"}, {"meta", "'{}'"}};
    std::set<std::string> valid{"tags", "fiber", "patch", "track", "retired", "lifecycle_epoch"};
    std::string sql = "id";
    for (const auto& [name, fallback] : columns) {
        valid.insert(name);
        sql += "," + (wants(fields, name) ? name : fallback);
    }
    for (const auto& field : *fields)
        if (!valid.contains(field)) throw std::invalid_argument("unknown object projection field: " + field);
    return sql;
}

void appendJournal(sqlite3* db, const std::string& author, const std::string& hostname,
                   const std::array<unsigned char, 16>& uuid, int64_t generation, const char* changeKind)
{
    Statement state(db, "SELECT approval_state,approved_by,approved_at,approved_host,approved_generation FROM objects WHERE uuid=? "
        "UNION ALL SELECT approval_state,approved_by,approved_at,approved_host,approved_generation FROM edges WHERE uuid=?");
    state.bindBlob(1, uuid); state.bindBlob(2, uuid);
    const bool hasState = state.row();
    Statement s(db, "INSERT INTO journal(record_uuid,generation,author,hostname,timestamp,change_kind,"
                    "approval_state,approved_by,approved_at,approved_host,approved_generation) VALUES(?,?,?,?,?,?,?,?,?,?,?)");
    s.bindBlob(1, uuid);
    s.bind(2, generation);
    s.bind(3, author);
    s.bind(4, hostname);
    s.bind(5, nowIsoUtc());
    s.bind(6, std::string(changeKind));
    for (int i = 0; i < 4; ++i) {
        if (hasState && !state.isNull(i)) s.bind(7 + i, state.text(i));
        else s.bindNull(7 + i);
    }
    if (hasState && !state.isNull(4)) s.bind(11, state.i64(4));
    else s.bindNull(11);
    s.stepDone();
}

void readCommon(RecordCommon& rec, const Statement& s, int uuidCol, int kindCol, int genCol)
{
    rec.uuid = formatUuid(s.blob16(uuidCol));
    rec.kind = s.text(kindCol);
    rec.generation = s.i64(genCol);
    rec.createdBy = s.text(genCol + 1);
    rec.createdAt = s.text(genCol + 2);
    rec.createdHost = s.text(genCol + 3);
    rec.changedBy = s.text(genCol + 4);
    rec.changedAt = s.text(genCol + 5);
    rec.changedHost = s.text(genCol + 6);
    rec.approvalState = s.text(genCol + 7);
    rec.approvedBy = s.optText(genCol + 8);
    rec.approvedAt = s.optText(genCol + 9);
    rec.approvedHost = s.optText(genCol + 10);
    if (!s.isNull(genCol + 11))
        rec.approvedGeneration = s.i64(genCol + 11);
}

std::optional<std::array<double, 6>> readBbox(const Statement& s, int firstCol)
{
    if (s.isNull(firstCol)) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = s.f64(firstCol + i);
    return box;
}

void bindBbox(Statement& s, int firstIdx, const std::optional<std::array<double, 6>>& box)
{
    for (int i = 0; i < 6; ++i) {
        if (box) s.bind(firstIdx + i, (*box)[size_t(i)]);
        else s.bindNull(firstIdx + i);
    }
}

namespace {
template<class T> T readDetail(const Statement& row)
{
    T result;
    int column = 0;
    detailFields(result, [&](const char*, auto& value) {
        using Field = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Field, std::string>) value = row.text(column++);
        else if constexpr (std::is_same_v<Field, double>) value = row.f64(column++);
        else if constexpr (std::is_integral_v<Field>) value = row.i64(column++);
        else if constexpr (std::is_same_v<Field, std::array<double, 2>>) {
            value = {row.f64(column), row.f64(column + 1)};
            column += 2;
        } else {
            value = readBbox(row, column);
            column += 6;
        }
    });
    return result;
}

template<class Visit> void detailTables(Visit visit)
{
    visit("fiber", "fibers", "family,manual_family,control_point_count,line_point_count,length,width,"
        "width_gap_fraction,optimization_mode,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::fiber, readDetail<FiberDetail>, 14);
    visit("patch", "patches", "grid_width,grid_height,scale,scale_y,"
        "bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::patch, readDetail<PatchDetail>, 10);
    visit("track", "tracks", "dataset_ref,external_key,family,point_count,length,"
        "bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::track, readDetail<TrackDetail>, 11);
}
}

ObjectRecord readObjectRow(sqlite3* db, const Statement& s, bool details)
{
    ObjectRecord rec;
    const int64_t id = s.i64(0);
    readCommon(rec, s, 1, 2, 4);
    rec.name = s.text(3);
    rec.formatVersion = s.i64(16);
    rec.storageKind = s.text(17);
    rec.storageRef = s.text(18);
    rec.dataGeneration = s.i64(19);
    rec.dataChecksum = s.optText(20);
    rec.meta = s.text(21);
    readLifecycle(db, id, rec);

    if (!details) return rec;
    {
        Statement tags(db, "SELECT tag FROM object_tags WHERE object_id=? ORDER BY tag");
        tags.bind(1, id);
        while (tags.row()) rec.tags.push_back(tags.text(0));
    }
    detailTables([&](const char* kind, const char* table, const char* columns, auto member, auto read, int) {
        if (rec.kind != kind) return;
        static const auto sql = std::string("SELECT ") + columns + " FROM " + table + " WHERE object_id=?";
        Statement detail(db, sql);
        detail.bind(1, id);
        if (detail.row()) rec.*member = read(detail);
    });
    return rec;
}

std::vector<ObjectRecord> readObjectRows(sqlite3* db, Statement& rows, const ObjectFields& fields)
{
    const bool hydrate = wants(fields, "tags") || wants(fields, "fiber") || wants(fields, "patch") || wants(fields, "track");
    std::vector<ObjectRecord> result;
    nlohmann::json ids = nlohmann::json::array();
    std::unordered_map<int64_t, size_t> positions;
    std::set<std::string> kinds;
    while (rows.row()) {
        if (hydrate) { positions.emplace(rows.i64(0), result.size()); ids.push_back(rows.i64(0)); }
        result.push_back(readObjectRow(db, rows, false));
        if (!wants(fields, "retired")) result.back().retired = false;
        if (!wants(fields, "lifecycle_epoch")) result.back().lifecycleEpoch.clear();
        if (hydrate) kinds.insert(result.back().kind);
    }
    if (result.empty() || !hydrate) return result;
    const auto selection = ids.dump();
    if (wants(fields, "tags")) {
        Statement tags(db, "SELECT object_id,tag FROM object_tags WHERE object_id IN (SELECT value FROM json_each(?)) ORDER BY object_id,tag");
        tags.bind(1, selection);
        while (tags.row()) result.at(positions.at(tags.i64(0))).tags.push_back(tags.text(1));
    }
    detailTables([&](const char* kind, const char* table, const char* columns, auto member, auto read, int idColumn) {
        if (!kinds.contains(kind) || !wants(fields, kind)) return;
        static const auto sql = std::string("SELECT ") + columns + ",object_id FROM " + table +
            " WHERE object_id IN (SELECT value FROM json_each(?))";
        Statement detail(db, sql);
        detail.bind(1, selection);
        while (detail.row()) result.at(positions.at(detail.i64(idColumn))).*member = read(detail);
    });
    return result;
}

std::optional<int64_t> objectIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM objects WHERE uuid=?");
    s.bindBlob(1, uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

namespace {
void validateObjectDetail(const ObjectRecord& rec)
{
    objectDetails(rec, [&](const char* kind, const auto& detail) {
        if (detail && rec.kind != kind)
            throw std::runtime_error(std::string("object store: ") + kind + " detail on object of kind " + rec.kind);
    });
}
}

void writeObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec)
{
    for (const char* table : {"fibers", "patches", "tracks"}) {
        Statement del(db, std::string("DELETE FROM ") + table + " WHERE object_id=?");
        del.bind(1, id);
        del.stepDone();
    }
    bool written = false;
    detailTables([&](const char*, const char* table, const char* columns, auto member, auto, int count) {
        const auto& detail = rec.*member;
        if (written || !detail) return;
        static const auto sql = [&] {
            std::string query = std::string("INSERT INTO ") + table + "(object_id," + columns + ") VALUES(?";
            for (int i = 0; i < count; ++i) query += ",?";
            return query + ")";
        }();
        Statement row(db, sql);
        row.bind(1, id);
        int next = 2;
        detailFields(*detail, [&](const char*, const auto& value) {
            using Field = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Field, bool>) row.bind(next++, int64_t(value));
            else if constexpr (std::is_same_v<Field, std::array<double, 2>>)
                for (auto axis : value) row.bind(next++, axis);
            else if constexpr (std::is_same_v<Field, std::optional<std::array<double, 6>>>) {
                bindBbox(row, next, value);
                next += 6;
            } else row.bind(next++, value);
        });
        row.stepDone();
        written = true;
    });
}

void writeTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags)
{
    {
        Statement del(db, std::string("DELETE FROM ") + table + " WHERE " + idColumn + "=?");
        del.bind(1, id);
        del.stepDone();
    }
    Statement ins(db, std::string("INSERT OR IGNORE INTO ") + table + "(" + idColumn + ",tag) VALUES(?,?)");
    for (const auto& tag : tags) {
        if (tag.empty()) continue;
        ins.reset();
        ins.bind(1, id);
        ins.bind(2, tag);
        ins.stepDone();
    }
}

void restoreCommon(sqlite3* db, const char* table, const RecordCommon& rec)
{
    Statement s(db, std::string("UPDATE ") + table + " SET generation=?,created_by=?,created_at=?,created_host=?,"
        "changed_by=?,changed_at=?,changed_host=?,approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_generation=? WHERE uuid=?");
    s.bind(1, rec.generation);
    s.bind(2, rec.createdBy); s.bind(3, rec.createdAt); s.bind(4, rec.createdHost);
    s.bind(5, rec.changedBy); s.bind(6, rec.changedAt); s.bind(7, rec.changedHost);
    s.bind(8, rec.approvalState);
    s.bind(9, rec.approvedBy);
    s.bind(10, rec.approvedAt);
    s.bind(11, rec.approvedHost);
    s.bind(12, rec.approvedGeneration);
    s.bindBlob(13, parseUuid(rec.uuid));
    s.stepDone();
}

std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input,
                             bool preserveRecord, const char* importOperation)
{
    validateObjectDetail(input);
    Savepoint savepoint(db);
    if (input.kind.empty())
        throw std::runtime_error("object store: object kind must be non-empty");
    if (input.patch)
        for (double axis : input.patch->scale)
            if (!std::isfinite(axis) || axis < 0)
                throw std::runtime_error("object store: patch scale must be finite and nonnegative");
    ObjectRecord rec = input;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    const auto uuid = parseUuid(rec.uuid);
    if (hasGeometryCoordinates(rec.kind)) {
        auto meta = nlohmann::json::parse(rec.meta);
        if (meta.contains("coordinates"))
            meta["coordinates"] = geometryCoordinates(meta["coordinates"]);
        else if (!preserveRecord && !objectIdByUuid(db, uuid))
            meta["coordinates"] = geometryCoordinates(nlohmann::json::object());
        rec.meta = meta.dump();
    }
    const auto bounds = detail::recordBounds(rec);
    if (bounds) detail::validateBounds(*bounds);
    Statement collision(db, "SELECT 1 FROM edges WHERE uuid=?");
    collision.bindBlob(1, uuid);
    if (collision.row()) throw std::runtime_error("object store: UUID already belongs to an edge");
    const std::string now = nowIsoUtc();
    Statement managed(db, "SELECT o.data_generation,o.data_checksum,o.storage_ref FROM objects o JOIN geometry_heads h ON h.object_uuid=o.uuid WHERE o.uuid=?");
    managed.bindBlob(1, uuid);
    const bool hasGeometry = managed.row();
    if (hasGeometry) {
        rec.storageKind = "sqlite";
        rec.storageRef = managed.text(2);
        rec.dataGeneration = managed.i64(0);
        rec.dataChecksum = managed.text(1);
    } else if (rec.storageKind == "sqlite") {
        throw std::invalid_argument("object store: managed geometry requires putGeometry, not a storage_kind override");
    }

    int64_t id = 0;
    int64_t generation = 1;
    if (auto existing = objectIdByUuid(db, uuid)) {
        id = *existing;
        Statement cur(db, "SELECT kind,generation FROM objects WHERE id=?");
        cur.bind(1, id);
        (void)cur.row();
        if (cur.text(0) != rec.kind)
            throw std::runtime_error("object store: object kind cannot change (" + cur.text(0) + " -> " + rec.kind + ")");
        generation = cur.i64(1) + 1;
        Statement s(db, "UPDATE objects SET name=?,generation=?,changed_by=?,changed_at=?,changed_host=?,"
                        "approval_state='pending',approved_by=NULL,approved_at=NULL,approved_host=NULL,approved_generation=NULL,"
                        "format_version=?,storage_kind=?,storage_ref=?,data_generation=?,data_checksum=?,meta=? WHERE id=?");
        s.bind(1, rec.name); s.bind(2, generation); s.bind(3, author); s.bind(4, now); s.bind(5, host);
        s.bind(6, rec.formatVersion); s.bind(7, rec.storageKind); s.bind(8, rec.storageRef); s.bind(9, rec.dataGeneration);
        s.bind(10, rec.dataChecksum);
        s.bind(11, rec.meta); s.bind(12, id);
        s.stepDone();
    } else {
        Statement s(db, "INSERT INTO objects(uuid,kind,name,generation,created_by,created_at,created_host,changed_by,changed_at,"
                        "changed_host,approval_state,approved_by,approved_at,approved_host,approved_generation,format_version,"
                        "storage_kind,storage_ref,data_generation,data_checksum,meta) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
        s.bindBlob(1, uuid); s.bind(2, rec.kind); s.bind(3, rec.name); s.bind(4, generation);
        s.bind(5, author); s.bind(6, now); s.bind(7, host);
        s.bind(8, author); s.bind(9, now); s.bind(10, host);
        const std::string state = preserveRecord ? rec.approvalState : "pending";
        s.bind(11, state.empty() ? std::string("pending") : state);
        if (preserveRecord && rec.approvedBy) s.bind(12, *rec.approvedBy); else s.bindNull(12);
        if (preserveRecord && rec.approvedAt) s.bind(13, *rec.approvedAt); else s.bindNull(13);
        if (preserveRecord && rec.approvedHost) s.bind(14, *rec.approvedHost); else s.bindNull(14);
        if (preserveRecord && rec.approvedGeneration) s.bind(15, *rec.approvedGeneration); else s.bindNull(15);
        s.bind(16, rec.formatVersion); s.bind(17, rec.storageKind); s.bind(18, rec.storageRef); s.bind(19, rec.dataGeneration);
        s.bind(20, rec.dataChecksum);
        s.bind(21, rec.meta);
        s.stepDone();
        id = sqlite3_last_insert_rowid(db);
    }
    writeObjectDetail(db, id, rec);
    if (!hasGeometry) detail::setSpatialBounds(db, id, bounds);
    writeTags(db, "object_tags", "object_id", id, rec.tags);
    if (preserveRecord) {
        restoreCommon(db, "objects", rec);
        writeLifecycle(db, id, rec);
    }
    appendJournal(db, author, host, uuid, preserveRecord ? rec.generation : generation, preserveRecord ? importOperation : "upsert");
    savepoint.release();
    return rec.uuid;
}

std::optional<int64_t> edgeIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM edges WHERE uuid=?");
    s.bindBlob(1, uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

void stampContentChange(sqlite3* db, const std::string& author, const std::string& host, const char* table,
                        int64_t id, const std::array<unsigned char, 16>& uuid, const char* changeKind)
{
    int64_t generation = 0;
    {
        Statement cur(db, std::string("SELECT generation FROM ") + table + " WHERE id=?");
        cur.bind(1, id);
        (void)cur.row();
        generation = cur.i64(0) + 1;
    }
    Statement s(db, std::string("UPDATE ") + table +
                        " SET generation=?,changed_by=?,changed_at=?,changed_host=?,approval_state='pending',"
                        "approved_by=NULL,approved_at=NULL,approved_host=NULL,approved_generation=NULL WHERE id=?");
    s.bind(1, generation);
    s.bind(2, author);
    s.bind(3, nowIsoUtc());
    s.bind(4, host);
    s.bind(5, id);
    s.stepDone();
    appendJournal(db, author, host, uuid, generation, changeKind);
}

std::string jsonPathForKey(const std::string& key)
{
    std::string out = "$.\"";
    for (char ch : key) {
        if (ch == '"' || ch == '\\') out.push_back('\\');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::optional<std::string> geometryHead(sqlite3* db, const std::string& objectUuid)
{
    Statement s(db, "SELECT revision_uuid FROM geometry_heads WHERE object_uuid=?");
    s.bindBlob(1, parseUuid(objectUuid));
    return s.row() ? std::optional(formatUuid(s.blob16(0))) : std::nullopt;
}
} // namespace vc::objects::detail
