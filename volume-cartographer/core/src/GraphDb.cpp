#include "vc/core/graph/GraphDb.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <climits>
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

namespace vc::graph {
namespace {

constexpr int kApplicationId = 0x56434752;  // VCGR
constexpr int kSchemaVersion = 1;

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
            throw std::runtime_error("graph db: uuid column is not 16 bytes");
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
        throw std::runtime_error("graph db: not a UUID: " + uuid);
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

const char* kSchemaSql = R"SQL(
CREATE TABLE objects(
  id INTEGER PRIMARY KEY,
  uuid BLOB NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK(length(kind)>0),
  name TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 1,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, created_host TEXT NOT NULL,
  changed_by TEXT NOT NULL, changed_at TEXT NOT NULL, changed_host TEXT NOT NULL,
  approval_state TEXT NOT NULL DEFAULT 'pending' CHECK(approval_state IN ('pending','approved','rejected')),
  approved_by TEXT, approved_at TEXT, approved_host TEXT, approved_generation INTEGER,
  format_version INTEGER NOT NULL DEFAULT 0,
  storage_kind TEXT NOT NULL DEFAULT '', storage_ref TEXT NOT NULL DEFAULT '',
  data_generation INTEGER NOT NULL DEFAULT 0, data_checksum TEXT,
  meta TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(meta) AND json_type(meta)='object')
);
CREATE INDEX objects_kind ON objects(kind);
CREATE INDEX objects_approval ON objects(approval_state);
CREATE TABLE fibers(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  family TEXT NOT NULL DEFAULT '', manual_family INTEGER NOT NULL DEFAULT 0,
  control_point_count INTEGER NOT NULL DEFAULT 0, line_point_count INTEGER NOT NULL DEFAULT 0,
  length REAL NOT NULL DEFAULT 0, width REAL NOT NULL DEFAULT 0, width_gap_fraction REAL NOT NULL DEFAULT 0.2,
  optimization_mode TEXT NOT NULL DEFAULT '',
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE TABLE patches(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  grid_width INTEGER NOT NULL DEFAULT 0, grid_height INTEGER NOT NULL DEFAULT 0, scale REAL NOT NULL DEFAULT 0,
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE TABLE tracks(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  dataset_ref TEXT NOT NULL DEFAULT '', external_key TEXT NOT NULL DEFAULT '',
  family TEXT NOT NULL DEFAULT '', point_count INTEGER NOT NULL DEFAULT 0, length REAL NOT NULL DEFAULT 0,
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE UNIQUE INDEX tracks_identity ON tracks(dataset_ref, external_key) WHERE external_key<>'';
CREATE TABLE object_tags(
  object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  tag TEXT NOT NULL, PRIMARY KEY(object_id, tag)
);
CREATE INDEX object_tags_tag ON object_tags(tag);
CREATE TABLE edges(
  id INTEGER PRIMARY KEY,
  uuid BLOB NOT NULL UNIQUE,
  a_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE RESTRICT,
  b_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE RESTRICT,
  kind TEXT NOT NULL CHECK(length(kind)>0),
  generation INTEGER NOT NULL DEFAULT 1,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, created_host TEXT NOT NULL,
  changed_by TEXT NOT NULL, changed_at TEXT NOT NULL, changed_host TEXT NOT NULL,
  approval_state TEXT NOT NULL DEFAULT 'pending' CHECK(approval_state IN ('pending','approved','rejected')),
  approved_by TEXT, approved_at TEXT, approved_host TEXT, approved_generation INTEGER,
  a_anchor_index INTEGER, a_anchor_x REAL, a_anchor_y REAL, a_anchor_z REAL,
  a_dir_x REAL, a_dir_y REAL, a_dir_z REAL,
  b_anchor_index INTEGER, b_anchor_x REAL, b_anchor_y REAL, b_anchor_z REAL,
  b_dir_x REAL, b_dir_y REAL, b_dir_z REAL,
  meta TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(meta) AND json_type(meta)='object'),
  CHECK(a_id < b_id)
);
CREATE INDEX edges_a ON edges(a_id, kind);
CREATE INDEX edges_b ON edges(b_id, kind);
CREATE UNIQUE INDEX edges_unique_unanchored ON edges(kind, a_id, b_id)
  WHERE a_anchor_index IS NULL AND b_anchor_index IS NULL;
CREATE UNIQUE INDEX edges_unique_anchored ON edges(kind, a_id, a_anchor_index, b_id, b_anchor_index)
  WHERE a_anchor_index IS NOT NULL AND b_anchor_index IS NOT NULL;
CREATE TABLE edge_tags(
  edge_id INTEGER NOT NULL REFERENCES edges(id) ON DELETE CASCADE,
  tag TEXT NOT NULL, PRIMARY KEY(edge_id, tag)
);
CREATE INDEX edge_tags_tag ON edge_tags(tag);
CREATE TABLE journal(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  record_uuid BLOB NOT NULL, generation INTEGER NOT NULL,
  author TEXT NOT NULL, hostname TEXT NOT NULL, timestamp TEXT NOT NULL,
  change_kind TEXT NOT NULL
);
CREATE INDEX journal_record ON journal(record_uuid, seq);
)SQL";

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
        throw std::runtime_error("graph db: cannot open " + path.string() + ": " + msg);
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
        throw std::runtime_error(std::string("graph db: PRAGMA ") + pragma + " returned no row");
    return int(s.i64(0));
}

}  // namespace

struct GraphDb::Impl {
    std::filesystem::path path;
    OpenOptions options;
    std::string hostname;
    sqlite3* writer = nullptr;
    std::mutex writerMutex;
    std::unique_lock<std::mutex> writerLock;  // held for the lifetime of a WriteTxn
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
            throw std::runtime_error("graph db: handle is closing");
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

GraphDb::GraphDb(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GraphDb::~GraphDb() = default;
GraphDb::GraphDb(GraphDb&&) noexcept = default;
GraphDb& GraphDb::operator=(GraphDb&&) noexcept = default;

std::unique_ptr<GraphDb::Impl> GraphDb::openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create)
{
    if (sqlite3_threadsafe() == 0)
        throw std::runtime_error("graph db: SQLite build is not thread-safe");
    if (!create && !options.readOnly && options.author.empty())
        throw std::runtime_error("graph db: read-write open requires an author (session identity)");
    if (create && options.author.empty())
        throw std::runtime_error("graph db: create requires an author (session identity)");

    auto impl = std::make_unique<GraphDb::Impl>();
    impl->path = path;
    impl->options = options;
    impl->options.readerPoolSize = std::max(1, options.readerPoolSize);
    impl->hostname = options.hostnameOverride.value_or(detectHostname());

    if (create) {
        std::error_code ec;
        if (std::filesystem::exists(path) && std::filesystem::file_size(path, ec) > 0)
            throw std::runtime_error("graph db: create refuses existing non-empty file: " + path.string());
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
        exec(impl->writer, "PRAGMA journal_mode=WAL;");
        exec(impl->writer, "PRAGMA synchronous=NORMAL;");
        exec(impl->writer, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
        exec(impl->writer, "BEGIN IMMEDIATE;");
        try {
            exec(impl->writer, kSchemaSql);
            exec(impl->writer, ("PRAGMA user_version=" + std::to_string(kSchemaVersion) + ";").c_str());
            exec(impl->writer, "COMMIT;");
        } catch (...) {
            exec(impl->writer, "ROLLBACK;");
            throw;
        }
    } else {
        if (!options.readOnly) {
            impl->writer = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
            exec(impl->writer, "PRAGMA journal_mode=WAL;");
            exec(impl->writer, "PRAGMA synchronous=NORMAL;");
            if (pragmaInt(impl->writer, "application_id") != kApplicationId)
                throw std::runtime_error("graph db: not a VC graph database: " + path.string());
            if (pragmaInt(impl->writer, "user_version") != kSchemaVersion)
                throw std::runtime_error("graph db: unsupported schema version in " + path.string());
        }
    }

    // WAL readers need write access to the -shm file, so pooled readers
    // are read-write connections pinned to query_only (true READONLY
    // only for a fully read-only open).
    const int readerFlags = options.readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE;
    for (int i = 0; i < impl->options.readerPoolSize; ++i) {
        sqlite3* conn = openConnection(path, readerFlags, options, true);
        if (options.readOnly && pragmaInt(conn, "application_id") != kApplicationId)
            throw std::runtime_error("graph db: not a VC graph database: " + path.string());
        impl->readerPool.push_back(conn);
    }
    return impl;
}

GraphDb GraphDb::create(const std::filesystem::path& path, const OpenOptions& options)
{
    return GraphDb(openImpl(path, options, true));
}

GraphDb GraphDb::open(const std::filesystem::path& path, const OpenOptions& options)
{
    return GraphDb(openImpl(path, options, false));
}

const std::filesystem::path& GraphDb::path() const { return impl_->path; }
std::string GraphDb::hostname() const { return impl_->hostname; }
std::string GraphDb::author() const { return impl_->options.author; }

Txn::Txn(Txn&& other) noexcept : db_(other.db_), handle_(other.handle_), write_(other.write_)
{
    other.handle_ = nullptr;
}

Txn::~Txn()
{
    if (handle_) {
        try {
            release(false);
        } catch (...) {
        }
    }
}

void Txn::commit()
{
    if (!handle_)
        throw std::runtime_error("graph db: commit on finished transaction");
    release(true);
}

void Txn::release(bool commit)
{
    auto* conn = static_cast<sqlite3*>(handle_);
    handle_ = nullptr;
    exec(conn, commit ? "COMMIT;" : "ROLLBACK;");
    if (write_) {
        if (db_->impl_->writerLock.owns_lock())
            db_->impl_->writerLock.unlock();
    } else {
        db_->impl_->checkinReader(conn);
    }
}

ReadTxn GraphDb::read()
{
    sqlite3* conn = impl_->checkoutReader();
    try {
        exec(conn, "BEGIN;");
        // Establish the snapshot now, so all reads in this transaction
        // see one consistent state even before the first real query.
        Statement pin(conn, "SELECT count(*) FROM objects");
        (void)pin.row();
    } catch (...) {
        impl_->checkinReader(conn);
        throw;
    }
    return ReadTxn(this, conn);
}

WriteTxn GraphDb::write()
{
    if (!impl_->writer)
        throw std::runtime_error("graph db: handle is read-only");
    impl_->writerLock = std::unique_lock<std::mutex>(impl_->writerMutex);
    try {
        exec(impl_->writer, "BEGIN IMMEDIATE;");
    } catch (...) {
        impl_->writerLock.unlock();
        throw;
    }
    return WriteTxn(this, impl_->writer);
}

void GraphDb::checkpoint()
{
    if (!impl_->writer)
        throw std::runtime_error("graph db: checkpoint on read-only handle");
    std::lock_guard<std::mutex> lock(impl_->writerMutex);
    check(sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_FULL, nullptr, nullptr), impl_->writer, "checkpoint");
}

struct GraphDbImplAccess {
    static sqlite3* conn(const Txn& txn)
    {
        if (!txn.active())
            throw std::runtime_error("graph db: use of finished transaction");
        return static_cast<sqlite3*>(txn.handle_);
    }
};

namespace {

const char* kObjectCols =
    "id,uuid,kind,name,generation,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_generation,format_version,storage_kind,"
    "storage_ref,data_generation,data_checksum,meta";

void appendJournal(sqlite3* db, const std::string& author, const std::string& hostname,
                   const std::array<unsigned char, 16>& uuid, int64_t generation, const char* changeKind)
{
    Statement s(db, "INSERT INTO journal(record_uuid,generation,author,hostname,timestamp,change_kind) VALUES(?,?,?,?,?,?)");
    s.bindBlob(1, uuid);
    s.bind(2, generation);
    s.bind(3, author);
    s.bind(4, hostname);
    s.bind(5, nowIsoUtc());
    s.bind(6, std::string(changeKind));
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
        Statement d(db, "SELECT grid_width,grid_height,scale,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z "
                        "FROM patches WHERE object_id=?");
        d.bind(1, id);
        if (d.row()) {
            PatchDetail p;
            p.gridWidth = d.i64(0);
            p.gridHeight = d.i64(1);
            p.scale = d.f64(2);
            p.bbox = readBbox(d, 3);
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
        throw std::runtime_error("graph db: fiber detail on object of kind " + rec.kind);
    if (rec.patch && rec.kind != "patch")
        throw std::runtime_error("graph db: patch detail on object of kind " + rec.kind);
    if (rec.track && rec.kind != "track")
        throw std::runtime_error("graph db: track detail on object of kind " + rec.kind);
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
        Statement s(db, "INSERT INTO patches(object_id,grid_width,grid_height,scale,bbox_min_x,bbox_min_y,bbox_min_z,"
                        "bbox_max_x,bbox_max_y,bbox_max_z) VALUES(?,?,?,?,?,?,?,?,?,?)");
        const auto& p = *rec.patch;
        s.bind(1, id); s.bind(2, p.gridWidth); s.bind(3, p.gridHeight); s.bind(4, p.scale);
        bindBbox(s, 5, p.bbox);
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

// preserveApproval is reserved for import paths seeding legacy records;
// the public API always starts/resets approval through the normal rules.
std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input, bool preserveApproval)
{
    if (input.kind.empty())
        throw std::runtime_error("graph db: object kind must be non-empty");
    ObjectRecord rec = input;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    const auto uuid = parseUuid(rec.uuid);
    const std::string now = nowIsoUtc();

    int64_t id = 0;
    int64_t generation = 1;
    if (auto existing = objectIdByUuid(db, uuid)) {
        id = *existing;
        Statement cur(db, "SELECT kind,generation FROM objects WHERE id=?");
        cur.bind(1, id);
        (void)cur.row();
        if (cur.text(0) != rec.kind)
            throw std::runtime_error("graph db: object kind cannot change (" + cur.text(0) + " -> " + rec.kind + ")");
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
    appendJournal(db, author, host, uuid, generation, "upsert");
    return rec.uuid;
}

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
    if (!s.row()) throw std::runtime_error("graph db: dangling edge endpoint id");
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

std::optional<int64_t> edgeIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM edges WHERE uuid=?");
    s.bindBlob(1, uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

std::string addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input, bool preserveApproval)
{
    if (input.kind.empty())
        throw std::runtime_error("graph db: edge kind must be non-empty");
    EdgeRecord rec = input;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    const auto aUuid = parseUuid(rec.aUuid);
    const auto bUuid = parseUuid(rec.bUuid);
    auto aId = objectIdByUuid(db, aUuid);
    auto bId = objectIdByUuid(db, bUuid);
    if (!aId) throw std::runtime_error("graph db: edge endpoint not found: " + rec.aUuid);
    if (!bId) throw std::runtime_error("graph db: edge endpoint not found: " + rec.bUuid);
    if (*aId == *bId)
        throw std::runtime_error("graph db: self-edges are not allowed");
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
    writeTags(db, "edge_tags", "edge_id", id, rec.tags);
    appendJournal(db, author, host, parseUuid(rec.uuid), 1, "add_edge");
    return rec.uuid;
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

}  // namespace

std::string GraphDb::upsertObject(WriteTxn& txn, const ObjectRecord& record)
{
    return upsertObjectImpl(GraphDbImplAccess::conn(txn), impl_->options.author, impl_->hostname, record, false);
}

std::optional<ObjectRecord> GraphDb::objectByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readObjectRow(db, s);
}

std::vector<ObjectRecord> GraphDb::objectsByKind(const Txn& txn, const std::string& kind,
                                                 const std::optional<std::string>& approvalState) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    std::string sql = std::string("SELECT ") + kObjectCols + " FROM objects WHERE kind=?";
    if (approvalState) sql += " AND approval_state=?";
    sql += " ORDER BY id";
    Statement s(db, sql);
    s.bind(1, kind);
    if (approvalState) s.bind(2, *approvalState);
    std::vector<ObjectRecord> out;
    while (s.row()) out.push_back(readObjectRow(db, s));
    return out;
}

std::vector<ObjectRecord> GraphDb::objectsByTag(const Txn& txn, const std::string& tag) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    std::string cols = kObjectCols;
    // Column names are ambiguous under the join only for id; qualify by selecting explicit object columns.
    Statement s(db, "SELECT o.id,o.uuid,o.kind,o.name,o.generation,o.created_by,o.created_at,o.created_host,o.changed_by,o.changed_at,"
                    "o.changed_host,o.approval_state,o.approved_by,o.approved_at,o.approved_host,o.approved_generation,o.format_version,"
                    "o.storage_kind,o.storage_ref,o.data_generation,o.data_checksum,o.meta FROM objects o "
                    "JOIN object_tags t ON t.object_id=o.id WHERE t.tag=? ORDER BY o.id");
    (void)cols;
    s.bind(1, tag);
    std::vector<ObjectRecord> out;
    while (s.row()) out.push_back(readObjectRow(db, s));
    return out;
}

void GraphDb::deleteObject(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    const auto bytes = parseUuid(uuid);
    auto id = objectIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("graph db: object not found: " + uuid);
    Statement s(db, "DELETE FROM objects WHERE id=?");  // FK RESTRICT on edges refuses while referenced
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete");
}

std::string GraphDb::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    return addEdgeImpl(GraphDbImplAccess::conn(txn), impl_->options.author, impl_->hostname, record, false);
}

std::optional<EdgeRecord> GraphDb::edgeByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readEdgeRow(db, s);
}

void GraphDb::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    const auto bytes = parseUuid(uuid);
    auto id = edgeIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("graph db: edge not found: " + uuid);
    Statement s(db, "DELETE FROM edges WHERE id=?");
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "remove_edge");
}

std::vector<EdgeRecord> GraphDb::edgesOf(const Txn& txn, const std::string& objectUuid) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    auto id = objectIdByUuid(db, parseUuid(objectUuid));
    if (!id) throw std::runtime_error("graph db: object not found: " + objectUuid);
    Statement s(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE a_id=? OR b_id=? ORDER BY id");
    s.bind(1, *id);
    s.bind(2, *id);
    std::vector<EdgeRecord> out;
    while (s.row()) out.push_back(readEdgeRow(db, s));
    return out;
}

std::vector<std::string> GraphDb::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind) const
{
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& e : edgesOf(txn, objectUuid)) {
        if (edgeKind && e.kind != *edgeKind) continue;
        const std::string& other = (e.aUuid == objectUuid) ? e.bUuid : e.aUuid;
        if (!seen.insert(other).second) continue;
        if (neighbourKind) {
            auto obj = objectByUuid(txn, other);
            if (!obj || obj->kind != *neighbourKind) continue;
        }
        out.push_back(other);
    }
    return out;
}

namespace {

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

std::vector<std::string> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel)
{
    auto checkCancelled = [&] {
        if (cancel && cancel->load())
            throw std::runtime_error("graph db: traversal cancelled");
    };
    checkCancelled();
    ProgressGuard guard{db};
    if (cancel)
        sqlite3_progress_handler(
            db, 1000, [](void* p) { return static_cast<const std::atomic_bool*>(p)->load() ? 1 : 0; },
            const_cast<std::atomic_bool*>(cancel));
    const auto startBytes = parseUuid(startUuid);
    if (!objectIdByUuid(db, startBytes))
        throw std::runtime_error("graph db: object not found: " + startUuid);
    std::string sql = "SELECT a_id,b_id FROM edges";
    if (!edgeKindFilter.empty()) {
        sql += " WHERE kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i) sql += (i ? ",?" : "?");
        sql += ")";
    }
    Statement s(db, sql);
    for (size_t i = 0; i < edgeKindFilter.size(); ++i) s.bind(int(i) + 1, edgeKindFilter[i]);
    std::unordered_map<int64_t, std::vector<int64_t>> adjacency;
    try {
        while (s.row()) {
            adjacency[s.i64(0)].push_back(s.i64(1));
            adjacency[s.i64(1)].push_back(s.i64(0));
        }
    } catch (const std::runtime_error&) {
        checkCancelled();  // SQL was interrupted by the progress handler
        throw;
    }
    checkCancelled();
    const int64_t startId = *objectIdByUuid(db, startBytes);
    std::vector<std::string> out;
    std::unordered_set<int64_t> visited{startId};
    std::queue<std::pair<int64_t, int>> queue;
    queue.emplace(startId, 0);
    while (!queue.empty()) {
        checkCancelled();
        auto [id, depth] = queue.front();
        queue.pop();
        out.push_back(uuidTextByObjectId(db, id));
        if (depth >= maxDepth) continue;
        for (int64_t next : adjacency[id])
            if (visited.insert(next).second) queue.emplace(next, depth + 1);
    }
    return out;
}

}  // namespace

std::vector<std::string> GraphDb::componentOf(const Txn& txn, const std::string& objectUuid,
                                              const std::vector<std::string>& edgeKindFilter,
                                              const std::atomic_bool* cancel) const
{
    return traverse(GraphDbImplAccess::conn(txn), objectUuid, INT_MAX, edgeKindFilter, cancel);
}

std::vector<std::string> GraphDb::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel) const
{
    if (maxDepth < 0) throw std::runtime_error("graph db: maxDepth must be >= 0");
    return traverse(GraphDbImplAccess::conn(txn), objectUuid, maxDepth, edgeKindFilter, cancel);
}

void GraphDb::setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state)
{
    if (state != "pending" && state != "approved" && state != "rejected")
        throw std::runtime_error("graph db: invalid approval state: " + state);
    sqlite3* db = GraphDbImplAccess::conn(txn);
    const auto bytes = parseUuid(recordUuid);
    const char* table = nullptr;
    int64_t id = 0;
    int64_t generation = 0;
    if (auto oid = objectIdByUuid(db, bytes)) {
        table = "objects";
        id = *oid;
    } else if (auto eid = edgeIdByUuid(db, bytes)) {
        table = "edges";
        id = *eid;
    } else {
        throw std::runtime_error("graph db: record not found: " + recordUuid);
    }
    {
        Statement cur(db, std::string("SELECT generation FROM ") + table + " WHERE id=?");
        cur.bind(1, id);
        (void)cur.row();
        generation = cur.i64(0);
    }
    Statement s(db, std::string("UPDATE ") + table +
                        " SET approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_generation=? WHERE id=?");
    s.bind(1, state);
    if (state == "pending") {
        s.bindNull(2); s.bindNull(3); s.bindNull(4); s.bindNull(5);
    } else {
        s.bind(2, impl_->options.author);
        s.bind(3, nowIsoUtc());
        s.bind(4, impl_->hostname);
        s.bind(5, generation);
    }
    s.bind(6, id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, generation, "approval");
}

namespace {

void mutateMeta(sqlite3* db, const std::string& author, const std::string& host, const std::string& recordUuid,
                const std::string& key, const std::optional<std::string>& jsonValue)
{
    if (key.empty()) throw std::runtime_error("graph db: meta key must be non-empty");
    const auto bytes = parseUuid(recordUuid);
    const char* table = nullptr;
    int64_t id = 0;
    if (auto oid = objectIdByUuid(db, bytes)) {
        table = "objects";
        id = *oid;
    } else if (auto eid = edgeIdByUuid(db, bytes)) {
        table = "edges";
        id = *eid;
    } else {
        throw std::runtime_error("graph db: record not found: " + recordUuid);
    }
    if (jsonValue) {
        Statement s(db, std::string("UPDATE ") + table + " SET meta=json_set(meta,?,json(?)) WHERE id=?");
        s.bind(1, jsonPathForKey(key));
        s.bind(2, *jsonValue);
        s.bind(3, id);
        s.stepDone();
    } else {
        Statement s(db, std::string("UPDATE ") + table + " SET meta=json_remove(meta,?) WHERE id=?");
        s.bind(1, jsonPathForKey(key));
        s.bind(2, id);
        s.stepDone();
    }
    stampContentChange(db, author, host, table, id, bytes, "meta");
}

}  // namespace

void GraphDb::setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue)
{
    mutateMeta(GraphDbImplAccess::conn(txn), impl_->options.author, impl_->hostname, recordUuid, key, jsonValue);
}

void GraphDb::removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key)
{
    mutateMeta(GraphDbImplAccess::conn(txn), impl_->options.author, impl_->hostname, recordUuid, key, std::nullopt);
}

std::vector<JournalEntry> GraphDb::journalSince(const Txn& txn, int64_t afterSeq) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    Statement s(db, "SELECT seq,record_uuid,generation,author,hostname,timestamp,change_kind FROM journal WHERE seq>? ORDER BY seq");
    s.bind(1, afterSeq);
    std::vector<JournalEntry> out;
    while (s.row()) {
        JournalEntry e;
        e.seq = s.i64(0);
        e.recordUuid = formatUuid(s.blob16(1));
        e.generation = s.i64(2);
        e.author = s.text(3);
        e.hostname = s.text(4);
        e.timestamp = s.text(5);
        e.changeKind = s.text(6);
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<std::string> GraphDb::sweepExternalData(const Txn& txn) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    std::vector<std::string> problems;
    Statement s(db, "SELECT uuid,storage_ref,data_generation,generation FROM objects WHERE storage_ref<>'' ORDER BY id");
    const auto root = impl_->path.parent_path();
    while (s.row()) {
        const std::string uuid = formatUuid(s.blob16(0));
        const std::string ref = s.text(1);
        const int64_t dataGen = s.i64(2);
        const int64_t gen = s.i64(3);
        if (dataGen > gen)
            problems.push_back(uuid + ": data_generation " + std::to_string(dataGen) + " ahead of generation " + std::to_string(gen));
        // Dataset URIs (containing "://") are not local files to check.
        if (ref.find("://") == std::string::npos) {
            std::error_code ec;
            if (!std::filesystem::exists(root / ref, ec))
                problems.push_back(uuid + ": storage file missing: " + ref);
        }
    }
    return problems;
}

namespace {

nlohmann::json anchorToJson(const Anchor& a)
{
    nlohmann::json j = nlohmann::json::object();
    if (a.index) j["index"] = *a.index;
    if (a.position) j["position"] = {(*a.position)[0], (*a.position)[1], (*a.position)[2]};
    if (a.direction) j["direction"] = {(*a.direction)[0], (*a.direction)[1], (*a.direction)[2]};
    return j;
}

Anchor anchorFromJson(const nlohmann::json& j)
{
    Anchor a;
    if (!j.is_object()) return a;
    if (j.contains("index") && j.at("index").is_number_integer()) a.index = j.at("index").get<int64_t>();
    for (const char* key : {"position", "direction"}) {
        if (j.contains(key) && j.at(key).is_array() && j.at(key).size() == 3) {
            std::array<double, 3> v{};
            for (int i = 0; i < 3; ++i) v[size_t(i)] = j.at(key).at(size_t(i)).get<double>();
            if (std::string(key) == "position") a.position = v; else a.direction = v;
        }
    }
    return a;
}

std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != 6) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = j.at(size_t(i)).get<double>();
    return box;
}

nlohmann::json bboxToJson(const std::optional<std::array<double, 6>>& box)
{
    if (!box) return nullptr;
    return {(*box)[0], (*box)[1], (*box)[2], (*box)[3], (*box)[4], (*box)[5]};
}

nlohmann::json objectToJson(const ObjectRecord& r)
{
    nlohmann::json j;
    j["uuid"] = r.uuid;
    j["kind"] = r.kind;
    j["name"] = r.name;
    j["format_version"] = r.formatVersion;
    j["storage_kind"] = r.storageKind;
    j["storage_ref"] = r.storageRef;
    j["data_generation"] = r.dataGeneration;
    if (r.dataChecksum) j["data_checksum"] = *r.dataChecksum;
    j["tags"] = r.tags;
    j["meta"] = nlohmann::json::parse(r.meta);
    j["approval_state"] = r.approvalState;
    if (r.fiber) {
        j["fiber"] = {{"family", r.fiber->family}, {"manual_family", r.fiber->manualFamily},
                      {"control_point_count", r.fiber->controlPointCount}, {"line_point_count", r.fiber->linePointCount},
                      {"length", r.fiber->length}, {"width", r.fiber->width},
                      {"width_gap_fraction", r.fiber->widthGapFraction}, {"optimization_mode", r.fiber->optimizationMode},
                      {"bbox", bboxToJson(r.fiber->bbox)}};
    }
    if (r.patch) {
        j["patch"] = {{"grid_width", r.patch->gridWidth}, {"grid_height", r.patch->gridHeight},
                      {"scale", r.patch->scale}, {"bbox", bboxToJson(r.patch->bbox)}};
    }
    if (r.track) {
        j["track"] = {{"dataset_ref", r.track->datasetRef}, {"external_key", r.track->externalKey},
                      {"family", r.track->family}, {"point_count", r.track->pointCount},
                      {"length", r.track->length}, {"bbox", bboxToJson(r.track->bbox)}};
    }
    return j;
}

ObjectRecord objectFromJson(const nlohmann::json& j)
{
    ObjectRecord r;
    r.uuid = j.value("uuid", "");
    r.kind = j.value("kind", "");
    r.name = j.value("name", "");
    r.formatVersion = j.value("format_version", int64_t(0));
    r.storageKind = j.value("storage_kind", "");
    r.storageRef = j.value("storage_ref", "");
    r.dataGeneration = j.value("data_generation", int64_t(0));
    if (j.contains("data_checksum") && j.at("data_checksum").is_string()) r.dataChecksum = j.at("data_checksum").get<std::string>();
    if (j.contains("tags")) r.tags = j.at("tags").get<std::vector<std::string>>();
    if (j.contains("meta")) r.meta = j.at("meta").dump();
    r.approvalState = j.value("approval_state", "pending");
    if (j.contains("fiber")) {
        const auto& f = j.at("fiber");
        FiberDetail d;
        d.family = f.value("family", "");
        d.manualFamily = f.value("manual_family", false);
        d.controlPointCount = f.value("control_point_count", int64_t(0));
        d.linePointCount = f.value("line_point_count", int64_t(0));
        d.length = f.value("length", 0.0);
        d.width = f.value("width", 0.0);
        d.widthGapFraction = f.value("width_gap_fraction", 0.2);
        d.optimizationMode = f.value("optimization_mode", "");
        if (f.contains("bbox")) d.bbox = bboxFromJson(f.at("bbox"));
        r.fiber = d;
    }
    if (j.contains("patch")) {
        const auto& p = j.at("patch");
        PatchDetail d;
        d.gridWidth = p.value("grid_width", int64_t(0));
        d.gridHeight = p.value("grid_height", int64_t(0));
        d.scale = p.value("scale", 0.0);
        if (p.contains("bbox")) d.bbox = bboxFromJson(p.at("bbox"));
        r.patch = d;
    }
    if (j.contains("track")) {
        const auto& t = j.at("track");
        TrackDetail d;
        d.datasetRef = t.value("dataset_ref", "");
        d.externalKey = t.value("external_key", "");
        d.family = t.value("family", "");
        d.pointCount = t.value("point_count", int64_t(0));
        d.length = t.value("length", 0.0);
        if (t.contains("bbox")) d.bbox = bboxFromJson(t.at("bbox"));
        r.track = d;
    }
    return r;
}

}  // namespace

std::string GraphDb::exportJson(const Txn& txn, const std::optional<std::string>& objectKind,
                                const std::optional<std::string>& tag) const
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    nlohmann::json out;
    out["format"] = "vc_graph_export";
    out["version"] = 1;
    out["objects"] = nlohmann::json::array();
    out["edges"] = nlohmann::json::array();
    std::unordered_set<std::string> included;
    const bool filtered = objectKind.has_value() || tag.has_value();
    std::vector<ObjectRecord> objects;
    if (tag) {
        objects = objectsByTag(txn, *tag);
    } else if (objectKind) {
        objects = objectsByKind(txn, *objectKind);
    } else {
        Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects ORDER BY id");
        while (s.row()) objects.push_back(readObjectRow(db, s));
    }
    if (tag && objectKind)
        std::erase_if(objects, [&](const ObjectRecord& o) { return o.kind != *objectKind; });
    for (const auto& o : objects) {
        included.insert(o.uuid);
        out["objects"].push_back(objectToJson(o));
    }
    Statement edges(db, std::string("SELECT ") + kEdgeCols + " FROM edges ORDER BY id");
    while (edges.row()) {
        EdgeRecord e = readEdgeRow(db, edges);
        if (filtered && (!included.contains(e.aUuid) || !included.contains(e.bUuid))) continue;
        nlohmann::json j;
        j["uuid"] = e.uuid;
        j["kind"] = e.kind;
        j["a"] = e.aUuid;
        j["b"] = e.bUuid;
        j["a_anchor"] = anchorToJson(e.aAnchor);
        j["b_anchor"] = anchorToJson(e.bAnchor);
        j["tags"] = e.tags;
        j["meta"] = nlohmann::json::parse(e.meta);
        j["approval_state"] = e.approvalState;
        out["edges"].push_back(std::move(j));
    }
    return out.dump(2);
}

void GraphDb::importJson(WriteTxn& txn, const std::string& jsonText)
{
    sqlite3* db = GraphDbImplAccess::conn(txn);
    const auto root = nlohmann::json::parse(jsonText);
    if (root.value("format", "") != "vc_graph_export")
        throw std::runtime_error("graph db: not a vc_graph_export document");
    for (const auto& j : root.value("objects", nlohmann::json::array()))
        upsertObjectImpl(db, impl_->options.author, impl_->hostname, objectFromJson(j), true);
    for (const auto& j : root.value("edges", nlohmann::json::array())) {
        EdgeRecord e;
        e.uuid = j.value("uuid", "");
        if (!e.uuid.empty() && edgeByUuid(txn, e.uuid)) continue;
        e.kind = j.value("kind", "");
        e.aUuid = j.value("a", "");
        e.bUuid = j.value("b", "");
        if (j.contains("a_anchor")) e.aAnchor = anchorFromJson(j.at("a_anchor"));
        if (j.contains("b_anchor")) e.bAnchor = anchorFromJson(j.at("b_anchor"));
        if (j.contains("tags")) e.tags = j.at("tags").get<std::vector<std::string>>();
        if (j.contains("meta")) e.meta = j.at("meta").dump();
        e.approvalState = j.value("approval_state", "pending");
        addEdgeImpl(db, impl_->options.author, impl_->hostname, e, true);
    }
}

namespace {

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string fnvHashHex(const std::string& text)
{
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : text) h = (h ^ c) * 1099511628211ULL;
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string relativeToDb(GraphDb& db, const std::filesystem::path& file)
{
    std::error_code ec;
    auto rel = std::filesystem::relative(file, db.path().parent_path(), ec);
    if (ec) return file.filename().string();
    return rel.string();
}

struct ParsedBranch {
    int localCp = -1;
    std::string targetFile;
    int targetCp = -1;
    bool pending = false;
    bool adjacent = false;
    Anchor local, target;
};

Anchor branchAnchor(const nlohmann::json& j, const char* indexKey, const char* posKey, const char* dirKey)
{
    Anchor a;
    if (j.contains(indexKey) && j.at(indexKey).is_number_integer()) a.index = j.at(indexKey).get<int64_t>();
    for (int i = 0; i < 2; ++i) {
        const char* key = i == 0 ? posKey : dirKey;
        if (j.contains(key) && j.at(key).is_array() && j.at(key).size() == 3) {
            std::array<double, 3> v{};
            for (int k = 0; k < 3; ++k) v[size_t(k)] = j.at(key).at(size_t(k)).get<double>();
            if (i == 0) a.position = v; else a.direction = v;
        }
    }
    return a;
}

}  // namespace

ImportReport importFibers(GraphDb& db, WriteTxn& txn, const std::filesystem::path& fiberDir)
{
    ImportReport report;
    struct Parsed {
        std::filesystem::path path;
        std::string fileName;
        ObjectRecord object;
        std::vector<ParsedBranch> branches;
        std::string checksum;
    };
    std::vector<Parsed> parsed;
    std::unordered_map<std::string, size_t> byFileName;

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(fiberDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        nlohmann::json root;
        std::string text;
        try {
            text = readFile(file);
            root = nlohmann::json::parse(text);
        } catch (const std::exception& e) {
            report.problems.push_back(file.filename().string() + ": unreadable/unparseable JSON: " + e.what());
            continue;
        }
        if (!root.is_object() || !root.contains("uuid") ||
            !(root.contains("control_points") || root.contains("line_points")))
            continue;  // not a vc3d_fiber file; other JSON in the dir is not our business
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(file.filename().string() + ": missing/invalid uuid, skipped");
            continue;
        }
        Parsed p;
        p.path = file;
        p.fileName = file.filename().string();
        p.checksum = fnvHashHex(text);
        ObjectRecord& o = p.object;
        o.uuid = uuid;
        o.kind = "fiber";
        o.name = file.stem().string();
        o.formatVersion = root.value("version", int64_t(1));
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, file);
        o.dataGeneration = 1;
        o.dataChecksum = p.checksum;
        if (root.contains("tags") && root.at("tags").is_array())
            for (const auto& t : root.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        FiberDetail f;
        if (root.contains("control_points") && root.at("control_points").is_array())
            f.controlPointCount = int64_t(root.at("control_points").size());
        if (root.contains("line_points") && root.at("line_points").is_array())
            f.linePointCount = int64_t(root.at("line_points").size());
        f.width = root.value("width", 0.0);
        f.widthGapFraction = root.value("width_gap_fraction", 0.2);
        f.optimizationMode = root.value("optimization_mode", "");
        if (root.contains("hv_classification") && root.at("hv_classification").is_object()) {
            const auto& hv = root.at("hv_classification");
            const std::string manual = hv.value("manual_tag", "");
            const std::string automatic = hv.value("automatic_tag", "");
            f.manualFamily = !manual.empty();
            f.family = f.manualFamily ? manual : automatic;
        }
        o.fiber = f;
        for (const auto* arrayKey : {"branches", "adjacent_branches"}) {
            if (!root.contains(arrayKey) || !root.at(arrayKey).is_array()) continue;
            for (const auto& bj : root.at(arrayKey)) {
                if (!bj.is_object()) continue;
                ParsedBranch b;
                b.adjacent = std::string(arrayKey) == "adjacent_branches";
                b.localCp = bj.value("control_point_index", -1);
                b.targetCp = bj.value("branch_control_point_index", -1);
                b.targetFile = std::filesystem::path(bj.value("branch_file", "")).filename().string();
                b.pending = bj.value("pending", false);
                b.local = branchAnchor(bj, "control_point_index", "control_point_position", "control_point_direction");
                b.target = branchAnchor(bj, "branch_control_point_index", "branch_control_point_position", "branch_control_point_direction");
                p.branches.push_back(std::move(b));
            }
        }
        byFileName[p.fileName] = parsed.size();
        parsed.push_back(std::move(p));
    }

    // Objects first (edges need their endpoints to exist). Unchanged
    // files (same uuid + content hash) are skipped: re-import is a no-op.
    for (const auto& p : parsed) {
        if (auto existing = db.objectByUuid(txn, p.object.uuid)) {
            if (existing->dataChecksum && *existing->dataChecksum == p.checksum)
                continue;
        }
        db.upsertObject(txn, p.object);
        ++report.objectsImported;
    }

    auto hasReciprocal = [&](const Parsed& owner, const ParsedBranch& b) {
        auto it = byFileName.find(b.targetFile);
        if (it == byFileName.end()) return false;
        for (const auto& other : parsed[it->second].branches)
            if (other.targetFile == owner.fileName && other.localCp == b.targetCp && other.targetCp == b.localCp &&
                other.adjacent == b.adjacent)
                return true;
        return false;
    };

    std::set<std::string> seenPairs;
    for (const auto& p : parsed) {
        for (const auto& b : p.branches) {
            auto it = byFileName.find(b.targetFile);
            if (it == byFileName.end()) {
                report.problems.push_back(p.fileName + ": branch target not in imported set: " + b.targetFile);
                continue;
            }
            const Parsed& target = parsed[it->second];
            std::string keyA = p.fileName + "#" + std::to_string(b.localCp);
            std::string keyB = b.targetFile + "#" + std::to_string(b.targetCp);
            const std::string pairKey = keyA < keyB ? keyA + "|" + keyB : keyB + "|" + keyA;
            const bool reciprocal = hasReciprocal(p, b);
            if (!reciprocal)
                report.problems.push_back(p.fileName + ": one-sided branch to " + b.targetFile +
                                          " (edge imported from this side only)");
            if (!seenPairs.insert(pairKey).second)
                continue;
            // Skip if an identical edge already exists (idempotent re-import).
            bool exists = false;
            for (const auto& e : db.edgesOf(txn, p.object.uuid)) {
                const std::string& other = e.aUuid == p.object.uuid ? e.bUuid : e.aUuid;
                const Anchor& mine = e.aUuid == p.object.uuid ? e.aAnchor : e.bAnchor;
                const Anchor& theirs = e.aUuid == p.object.uuid ? e.bAnchor : e.aAnchor;
                if (other == target.object.uuid && e.kind == "fiber_branch" && mine.index == b.local.index &&
                    theirs.index == b.target.index) {
                    exists = true;
                    break;
                }
            }
            if (exists) continue;
            EdgeRecord e;
            e.kind = "fiber_branch";
            e.aUuid = p.object.uuid;
            e.bUuid = target.object.uuid;
            e.aAnchor = b.local;
            e.bAnchor = b.target;
            e.meta = nlohmann::json{{"adjacent", b.adjacent}}.dump();
            e.approvalState = b.pending ? "pending" : "approved";
            addEdgeImpl(GraphDbImplAccess::conn(txn), db.author(), db.hostname(), e, true);
            ++report.edgesImported;
        }
    }
    return report;
}

ImportReport importPatches(GraphDb& db, WriteTxn& txn, const std::filesystem::path& segmentsDir)
{
    ImportReport report;
    std::vector<std::filesystem::path> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(segmentsDir))
        if (entry.is_directory()) dirs.push_back(entry.path());
    std::sort(dirs.begin(), dirs.end());
    for (const auto& dir : dirs) {
        const auto metaPath = dir / "meta.json";
        if (!std::filesystem::exists(metaPath)) continue;
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(readFile(metaPath));
        } catch (const std::exception& e) {
            report.problems.push_back(dir.filename().string() + ": unparseable meta.json: " + e.what());
            continue;
        }
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(dir.filename().string() + ": meta.json has no valid uuid, skipped");
            continue;
        }
        if (db.objectByUuid(txn, uuid)) continue;  // already registered
        ObjectRecord o;
        o.uuid = uuid;
        o.kind = "patch";
        o.name = dir.filename().string();
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, dir);
        o.dataGeneration = 1;
        PatchDetail p;
        p.gridWidth = root.value("grid_width", int64_t(0));
        p.gridHeight = root.value("grid_height", int64_t(0));
        p.scale = root.value("scale", 0.0);
        o.patch = p;
        db.upsertObject(txn, o);
        ++report.objectsImported;
    }
    return report;
}

ImportReport importTrackList(GraphDb& db, WriteTxn& txn, const std::filesystem::path& trackListJson)
{
    ImportReport report;
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(readFile(trackListJson));
    } catch (const std::exception& e) {
        throw std::runtime_error("graph db: cannot parse track list " + trackListJson.string() + ": " + e.what());
    }
    if (!root.is_array())
        throw std::runtime_error("graph db: track list must be a JSON array: " + trackListJson.string());

    // Index already-registered tracks by identity so re-registration
    // reuses the minted UUID instead of colliding with the UNIQUE index.
    std::map<std::pair<std::string, std::string>, ObjectRecord> registered;
    for (auto& existing : db.objectsByKind(txn, "track"))
        if (existing.track)
            registered[{existing.track->datasetRef, existing.track->externalKey}] = existing;

    for (size_t i = 0; i < root.size(); ++i) {
        const auto& entry = root.at(i);
        const std::string label = "entry " + std::to_string(i);
        if (!entry.is_object()) {
            report.problems.push_back(label + ": not an object, skipped");
            continue;
        }
        const std::string datasetRef = entry.value("dataset_ref", "");
        const std::string externalKey = entry.value("external_key", "");
        if (datasetRef.empty() || externalKey.empty()) {
            report.problems.push_back(label + ": missing dataset_ref/external_key, skipped");
            continue;
        }
        ObjectRecord o;
        o.kind = "track";
        o.name = entry.value("name", externalKey);
        o.storageKind = "dataset";
        o.storageRef = datasetRef;
        o.dataGeneration = 1;
        if (entry.contains("tags") && entry.at("tags").is_array())
            for (const auto& t : entry.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        if (entry.contains("meta")) o.meta = entry.at("meta").dump();
        TrackDetail t;
        t.datasetRef = datasetRef;
        t.externalKey = externalKey;
        t.family = entry.value("family", "");
        t.pointCount = entry.value("point_count", int64_t(0));
        t.length = entry.value("length", 0.0);
        if (entry.contains("bbox")) t.bbox = bboxFromJson(entry.at("bbox"));
        o.track = t;

        auto it = registered.find({datasetRef, externalKey});
        if (it != registered.end()) {
            const ObjectRecord& cur = it->second;
            const bool unchanged = cur.name == o.name && cur.tags == o.tags && cur.meta == o.meta && cur.track &&
                                   cur.track->family == t.family && cur.track->pointCount == t.pointCount &&
                                   cur.track->length == t.length && cur.track->bbox == t.bbox;
            if (unchanged)
                continue;  // idempotent re-registration
            o.uuid = cur.uuid;
            db.upsertObject(txn, o);
            ++report.objectsImported;
        } else {
            const std::string uuid = db.upsertObject(txn, o);
            o.uuid = uuid;
            registered[{datasetRef, externalKey}] = o;
            ++report.objectsImported;
        }
    }
    return report;
}

}  // namespace vc::graph





