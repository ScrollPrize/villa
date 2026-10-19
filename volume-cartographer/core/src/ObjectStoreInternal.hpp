#pragma once
#include "vc/core/objects/ObjectStore.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <ctime>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "vc/core/util/Uuid.hpp"

namespace vc::objects {
struct ObjectStore::Impl {
    std::filesystem::path path;
    OpenOptions options;
    std::string hostname;
    sqlite3* writer = nullptr;
    std::mutex writerMutex;
    std::vector<sqlite3*> readerPool;
    std::mutex poolMutex;
    std::condition_variable poolCv;
    bool closing = false;

    ~Impl()
    {
        for (sqlite3* c : readerPool)
            if (c) sqlite3_close(c);
        if (writer) sqlite3_close(writer);
    }

    sqlite3* checkoutReader()
    {
        std::unique_lock<std::mutex> lock(poolMutex);
        poolCv.wait(lock, [&] { return closing || !readerPool.empty(); });
        if (readerPool.empty())
            throw std::runtime_error("object store: handle is closing");
        sqlite3* conn = readerPool.back();
        readerPool.pop_back();
        return conn;
    }
    void checkinReader(sqlite3* conn)
    {
        {
            std::lock_guard<std::mutex> lock(poolMutex);
            readerPool.push_back(conn);
        }
        poolCv.notify_one();
    }
};

struct ObjectStoreImplAccess {
    static sqlite3* conn(const Txn& txn)
    {
        if (!txn.active())
            throw std::runtime_error("object store: use of finished transaction");
        return static_cast<sqlite3*>(txn.handle_);
    }
};

namespace {
[[noreturn]] void fail(sqlite3* db, const std::string& what)
{
    throw std::runtime_error(what + ": " + (db ? sqlite3_errmsg(db) : "no database"));
}

void check(int rc, sqlite3* db, const char* what = "sqlite")
{
    if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE)
        fail(db, what);
}

void exec(sqlite3* db, const char* sql)
{
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : sqlite3_errmsg(db);
        sqlite3_free(err);
        throw std::runtime_error("sqlite exec failed: " + msg);
    }
}

class Statement {
public:
    Statement(sqlite3* db, const char* sql) : db_(db)
    {
        check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr), db, "prepare");
    }
    Statement(sqlite3* db, const std::string& sql) : Statement(db, sql.c_str()) {}
    ~Statement() { sqlite3_finalize(value); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    void bind(int i, int64_t v) { check(sqlite3_bind_int64(value, i, v), db_, "bind"); }
    void bind(int i, double v) { check(sqlite3_bind_double(value, i, v), db_, "bind"); }
    void bind(int i, const std::string& v) { check(sqlite3_bind_text(value, i, v.c_str(), int(v.size()), SQLITE_TRANSIENT), db_, "bind"); }
    void bindNull(int i) { check(sqlite3_bind_null(value, i), db_, "bind"); }
    void bindBlob(int i, const std::array<unsigned char, 16>& v)
    {
        check(sqlite3_bind_blob(value, i, v.data(), 16, SQLITE_TRANSIENT), db_, "bind");
    }
    bool row()
    {
        const int rc = sqlite3_step(value);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        fail(db_, "step");
    }
    void stepDone()
    {
        if (row()) fail(db_, "unexpected row");
    }
    void reset()
    {
        check(sqlite3_reset(value), db_, "reset");
        check(sqlite3_clear_bindings(value), db_, "clear_bindings");
    }

    bool isNull(int c) const { return sqlite3_column_type(value, c) == SQLITE_NULL; }
    int64_t i64(int c) const { return sqlite3_column_int64(value, c); }
    double f64(int c) const { return sqlite3_column_double(value, c); }
    std::string text(int c) const
    {
        const auto* p = sqlite3_column_text(value, c);
        return p ? std::string(reinterpret_cast<const char*>(p), size_t(sqlite3_column_bytes(value, c))) : "";
    }
    std::optional<std::string> optText(int c) const
    {
        if (isNull(c)) return std::nullopt;
        return text(c);
    }
    std::array<unsigned char, 16> blob16(int c) const
    {
        if (sqlite3_column_bytes(value, c) != 16)
            throw std::runtime_error("object store: uuid column is not 16 bytes");
        std::array<unsigned char, 16> out{};
        std::memcpy(out.data(), sqlite3_column_blob(value, c), 16);
        return out;
    }

    sqlite3_stmt* value = nullptr;

private:
    sqlite3* db_;
};

std::array<unsigned char, 16> parseUuid(const std::string& uuid)
{
    if (!util::isUuid(uuid))
        throw std::runtime_error("object store: not a UUID: " + uuid);
    std::array<unsigned char, 16> out{};
    int nibble = 0;
    for (char ch : uuid) {
        if (ch == '-') continue;
        int v;
        if (ch >= '0' && ch <= '9') v = ch - '0';
        else if (ch >= 'a' && ch <= 'f') v = ch - 'a' + 10;
        else v = ch - 'A' + 10;
        out[size_t(nibble / 2)] = static_cast<unsigned char>((out[size_t(nibble / 2)] << 4) | v);
        ++nibble;
    }
    return out;
}

std::string formatUuid(const std::array<unsigned char, 16>& bytes)
{
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out.push_back('-');
        out.push_back(hex[bytes[i] >> 4]);
        out.push_back(hex[bytes[i] & 0x0f]);
    }
    return out;
}

std::string nowIsoUtc()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string detectHostname()
{
#ifdef _WIN32
    const char* name = std::getenv("COMPUTERNAME");
    return name && *name ? name : "unknown";
#else
    char buf[256]{};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0])
        return buf;
    return "unknown";
#endif
}

void applyPragmas(sqlite3* db, const OpenOptions& options, bool queryOnly)
{
    sqlite3_busy_timeout(db, options.busyTimeoutMs);
    exec(db, "PRAGMA foreign_keys=ON;");
    exec(db, "PRAGMA temp_store=MEMORY;");
    exec(db, "PRAGMA cache_size=-8000;");
    if (queryOnly)
        exec(db, "PRAGMA query_only=ON;");
}

sqlite3* openConnection(const std::filesystem::path& path, int flags, const OpenOptions& options, bool queryOnly)
{
    sqlite3* db = nullptr;
    const auto name = path.u8string();
    const int rc = sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()), &db, flags | SQLITE_OPEN_FULLMUTEX, nullptr);
    if (rc != SQLITE_OK) {
        std::string msg = db ? sqlite3_errmsg(db) : "open failed";
        if (db) sqlite3_close(db);
        throw std::runtime_error("object store: cannot open " + path.string() + ": " + msg);
    }
    try {
        applyPragmas(db, options, queryOnly);
    } catch (...) {
        sqlite3_close(db);
        throw;
    }
    return db;
}

int pragmaInt(sqlite3* db, const char* pragma)
{
    Statement s(db, (std::string("PRAGMA ") + pragma).c_str());
    if (!s.row())
        throw std::runtime_error(std::string("object store: PRAGMA ") + pragma + " returned no row");
    return int(s.i64(0));
}

const char* kObjectCols =
    "id,uuid,kind,name,generation,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_generation,format_version,storage_kind,"
    "storage_ref,data_generation,data_checksum,meta";

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

ObjectRecord readObjectRow(sqlite3* db, const Statement& s)
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

    {
        Statement tags(db, "SELECT tag FROM object_tags WHERE object_id=? ORDER BY tag");
        tags.bind(1, id);
        while (tags.row()) rec.tags.push_back(tags.text(0));
    }
    if (rec.kind == "fiber") {
        Statement d(db, "SELECT family,manual_family,control_point_count,line_point_count,length,width,"
                        "width_gap_fraction,optimization_mode,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z "
                        "FROM fibers WHERE object_id=?");
        d.bind(1, id);
        if (d.row()) {
            FiberDetail f;
            f.family = d.text(0);
            f.manualFamily = d.i64(1) != 0;
            f.controlPointCount = d.i64(2);
            f.linePointCount = d.i64(3);
            f.length = d.f64(4);
            f.width = d.f64(5);
            f.widthGapFraction = d.f64(6);
            f.optimizationMode = d.text(7);
            f.bbox = readBbox(d, 8);
            rec.fiber = f;
        }
    } else if (rec.kind == "patch") {
        Statement d(db, "SELECT grid_width,grid_height,scale,scale_y,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z "
                        "FROM patches WHERE object_id=?");
        d.bind(1, id);
        if (d.row()) {
            PatchDetail p;
            p.gridWidth = d.i64(0);
            p.gridHeight = d.i64(1);
            p.scale = {d.f64(2), d.f64(3)};
            p.bbox = readBbox(d, 4);
            rec.patch = p;
        }
    } else if (rec.kind == "track") {
        Statement d(db, "SELECT dataset_ref,external_key,family,point_count,length,bbox_min_x,bbox_min_y,bbox_min_z,"
                        "bbox_max_x,bbox_max_y,bbox_max_z FROM tracks WHERE object_id=?");
        d.bind(1, id);
        if (d.row()) {
            TrackDetail t;
            t.datasetRef = d.text(0);
            t.externalKey = d.text(1);
            t.family = d.text(2);
            t.pointCount = d.i64(3);
            t.length = d.f64(4);
            t.bbox = readBbox(d, 5);
            rec.track = t;
        }
    }
    return rec;
}

std::optional<int64_t> objectIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM objects WHERE uuid=?");
    s.bindBlob(1, uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

void writeObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec)
{
    for (const char* table : {"fibers", "patches", "tracks"}) {
        Statement del(db, std::string("DELETE FROM ") + table + " WHERE object_id=?");
        del.bind(1, id);
        del.stepDone();
    }
    if (rec.fiber && rec.kind != "fiber")
        throw std::runtime_error("object store: fiber detail on object of kind " + rec.kind);
    if (rec.patch && rec.kind != "patch")
        throw std::runtime_error("object store: patch detail on object of kind " + rec.kind);
    if (rec.track && rec.kind != "track")
        throw std::runtime_error("object store: track detail on object of kind " + rec.kind);
    if (rec.fiber) {
        Statement s(db, "INSERT INTO fibers(object_id,family,manual_family,control_point_count,line_point_count,length,width,"
                        "width_gap_fraction,optimization_mode,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
        const auto& f = *rec.fiber;
        s.bind(1, id); s.bind(2, f.family); s.bind(3, int64_t(f.manualFamily ? 1 : 0));
        s.bind(4, f.controlPointCount); s.bind(5, f.linePointCount); s.bind(6, f.length);
        s.bind(7, f.width); s.bind(8, f.widthGapFraction); s.bind(9, f.optimizationMode);
        bindBbox(s, 10, f.bbox);
        s.stepDone();
    } else if (rec.patch) {
        Statement s(db, "INSERT INTO patches(object_id,grid_width,grid_height,scale,scale_y,bbox_min_x,bbox_min_y,bbox_min_z,"
                        "bbox_max_x,bbox_max_y,bbox_max_z) VALUES(?,?,?,?,?,?,?,?,?,?,?)");
        const auto& p = *rec.patch;
        s.bind(1, id); s.bind(2, p.gridWidth); s.bind(3, p.gridHeight);
        s.bind(4, p.scale[0]); s.bind(5, p.scale[1]);
        bindBbox(s, 6, p.bbox);
        s.stepDone();
    } else if (rec.track) {
        Statement s(db, "INSERT INTO tracks(object_id,dataset_ref,external_key,family,point_count,length,bbox_min_x,bbox_min_y,"
                        "bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)");
        const auto& t = *rec.track;
        s.bind(1, id); s.bind(2, t.datasetRef); s.bind(3, t.externalKey); s.bind(4, t.family);
        s.bind(5, t.pointCount); s.bind(6, t.length);
        bindBbox(s, 7, t.bbox);
        s.stepDone();
    }
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
    if (rec.approvedBy) s.bind(9, *rec.approvedBy); else s.bindNull(9);
    if (rec.approvedAt) s.bind(10, *rec.approvedAt); else s.bindNull(10);
    if (rec.approvedHost) s.bind(11, *rec.approvedHost); else s.bindNull(11);
    if (rec.approvedGeneration) s.bind(12, *rec.approvedGeneration); else s.bindNull(12);
    s.bindBlob(13, parseUuid(rec.uuid));
    s.stepDone();
}

// preserveApproval is reserved for import paths seeding legacy records;
// the public API always starts/resets approval through the normal rules.
std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input,
                             bool preserveApproval, bool preserveRecord = false)
{
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
    Statement collision(db, "SELECT 1 FROM edges WHERE uuid=?");
    collision.bindBlob(1, uuid);
    if (collision.row()) throw std::runtime_error("object store: UUID already belongs to an edge");
    const std::string now = nowIsoUtc();
    Statement managed(db, "SELECT o.data_generation,o.data_checksum,o.storage_ref FROM objects o JOIN geometry_heads h ON h.object_uuid=o.uuid WHERE o.uuid=?");
    managed.bindBlob(1, uuid);
    if (managed.row()) {
        rec.storageKind = "sqlite";
        rec.storageRef = managed.text(2);
        rec.dataGeneration = managed.i64(0);
        rec.dataChecksum = managed.text(1);
    } else if (rec.storageKind == "sqlite" || rec.storageKind == "object_store") {
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
        if (rec.dataChecksum) s.bind(10, *rec.dataChecksum); else s.bindNull(10);
        s.bind(11, rec.meta); s.bind(12, id);
        s.stepDone();
        if (preserveApproval && rec.approvalState != "pending") {
            Statement a(db, "UPDATE objects SET approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_generation=? WHERE id=?");
            a.bind(1, rec.approvalState);
            if (rec.approvedBy) a.bind(2, *rec.approvedBy); else a.bindNull(2);
            if (rec.approvedAt) a.bind(3, *rec.approvedAt); else a.bindNull(3);
            if (rec.approvedHost) a.bind(4, *rec.approvedHost); else a.bindNull(4);
            if (rec.approvedGeneration) a.bind(5, *rec.approvedGeneration); else a.bindNull(5);
            a.bind(6, id);
            a.stepDone();
        }
    } else {
        Statement s(db, "INSERT INTO objects(uuid,kind,name,generation,created_by,created_at,created_host,changed_by,changed_at,"
                        "changed_host,approval_state,approved_by,approved_at,approved_host,approved_generation,format_version,"
                        "storage_kind,storage_ref,data_generation,data_checksum,meta) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
        s.bindBlob(1, uuid); s.bind(2, rec.kind); s.bind(3, rec.name); s.bind(4, generation);
        s.bind(5, author); s.bind(6, now); s.bind(7, host);
        s.bind(8, author); s.bind(9, now); s.bind(10, host);
        const std::string state = preserveApproval ? rec.approvalState : "pending";
        s.bind(11, state.empty() ? std::string("pending") : state);
        if (preserveApproval && rec.approvedBy) s.bind(12, *rec.approvedBy); else s.bindNull(12);
        if (preserveApproval && rec.approvedAt) s.bind(13, *rec.approvedAt); else s.bindNull(13);
        if (preserveApproval && rec.approvedHost) s.bind(14, *rec.approvedHost); else s.bindNull(14);
        if (preserveApproval && rec.approvedGeneration) s.bind(15, *rec.approvedGeneration); else s.bindNull(15);
        s.bind(16, rec.formatVersion); s.bind(17, rec.storageKind); s.bind(18, rec.storageRef); s.bind(19, rec.dataGeneration);
        if (rec.dataChecksum) s.bind(20, *rec.dataChecksum); else s.bindNull(20);
        s.bind(21, rec.meta);
        s.stepDone();
        id = sqlite3_last_insert_rowid(db);
    }
    writeObjectDetail(db, id, rec);
    writeTags(db, "object_tags", "object_id", id, rec.tags);
    if (preserveRecord) restoreCommon(db, "objects", rec);
    appendJournal(db, author, host, uuid, preserveRecord ? rec.generation : generation, preserveRecord ? "import" : "upsert");
    return rec.uuid;
}

std::optional<int64_t> edgeIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM edges WHERE uuid=?");
    s.bindBlob(1, uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

// Shared content-mutation stamp for setMeta/removeMeta: bumps
// generation, restamps changed_*, resets approval, journals.
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
} // namespace
} // namespace vc::objects
