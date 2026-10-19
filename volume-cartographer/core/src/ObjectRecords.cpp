#include "StorageSchemaInternal.hpp"
namespace vc::objects {
ObjectStore::ObjectStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ObjectStore::~ObjectStore() = default;
ObjectStore::ObjectStore(ObjectStore&&) noexcept = default;
ObjectStore& ObjectStore::operator=(ObjectStore&&) noexcept = default;

std::unique_ptr<ObjectStore::Impl> ObjectStore::openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create)
{
    const int kApplicationId = 0x56434f42;
    const int kSchemaVersion = 2;
    if (create && options.readOnly) throw std::invalid_argument("create requires writable options");
    if (sqlite3_threadsafe() == 0)
        throw std::runtime_error("object store: SQLite build is not thread-safe");
    if (!create && !options.readOnly && options.author.empty())
        throw std::runtime_error("object store: read-write open requires an author (session identity)");
    if (create && options.author.empty())
        throw std::runtime_error("object store: create requires an author (session identity)");

    auto impl = std::make_unique<ObjectStore::Impl>();
    impl->path = path;
    impl->options = options;
    impl->options.readerPoolSize = std::max(1, options.readerPoolSize);
    impl->hostname = options.hostnameOverride.value_or(detectHostname());

    if (create) {
        std::error_code ec;
        if (std::filesystem::exists(path) && std::filesystem::file_size(path, ec) > 0)
            throw std::runtime_error("object store: create refuses existing non-empty file: " + path.string());
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
        exec(impl->writer, "PRAGMA journal_mode=WAL;");
        exec(impl->writer, "PRAGMA synchronous=FULL;");
        exec(impl->writer, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
        exec(impl->writer, "BEGIN IMMEDIATE;");
        try {
            createSchema(impl->writer);
            exec(impl->writer, ("PRAGMA user_version=" + std::to_string(kSchemaVersion) + ";").c_str());
            exec(impl->writer, "COMMIT;");
        } catch (...) {
            exec(impl->writer, "ROLLBACK;");
            throw;
        }
    } else {
        if (!options.readOnly) {
            impl->writer = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
            if (pragmaInt(impl->writer, "application_id") != kApplicationId)
                throw std::runtime_error("object store: unsupported database format; recreate and re-ingest: " + path.string());
            if (pragmaInt(impl->writer, "user_version") != kSchemaVersion)
                throw std::runtime_error("object store: unsupported schema version; recreate and re-ingest");
            exec(impl->writer, "PRAGMA journal_mode=WAL;");
            exec(impl->writer, "PRAGMA synchronous=FULL;");
        }
    }

    // WAL readers need write access to the -shm file, so pooled readers
    // are read-write connections pinned to query_only (true READONLY
    // only for a fully read-only open).
    const int readerFlags = options.readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE;
    for (int i = 0; i < impl->options.readerPoolSize; ++i) {
        sqlite3* conn = openConnection(path, readerFlags, options, true);
        impl->readerPool.push_back(conn);
        if (options.readOnly) {
            if (pragmaInt(conn, "application_id") != kApplicationId)
                throw std::runtime_error("object store: unsupported database format; recreate and re-ingest: " + path.string());
            const auto version = pragmaInt(conn, "user_version");
            if (version != kSchemaVersion)
                throw std::runtime_error("object store: unsupported schema version; recreate and re-ingest: " + path.string());
        }
    }
    return impl;
}

const std::filesystem::path& ObjectStore::path() const { return impl_->path; }
std::string ObjectStore::hostname() const { return impl_->hostname; }
std::string ObjectStore::author() const { return impl_->options.author; }

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
        throw std::runtime_error("object store: commit on finished transaction");
    release(true);
}

void Txn::release(bool commit)
{
    auto* conn = static_cast<sqlite3*>(handle_);
    handle_ = nullptr;
    auto releaseConnection = [&] {
        if (write_) db_->impl_->writerMutex.unlock();
        else db_->impl_->checkinReader(conn);
    };
    try { exec(conn, commit ? "COMMIT;" : "ROLLBACK;"); }
    catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        releaseConnection(); throw;
    }
    releaseConnection();
}

void* ObjectStore::connection(const Txn& txn) const {
    if (txn.db_ != this) throw std::invalid_argument("transaction belongs to another database");
    return ObjectStoreImplAccess::conn(txn);
}

ReadTxn ObjectStore::read()
{
    sqlite3* conn = impl_->checkoutReader();
    try {
        exec(conn, "BEGIN;");
        // Establish the snapshot now, so all reads in this transaction
        // see one consistent state even before the first real query.
        Statement pin(conn, "SELECT count(*) FROM objects");
        (void)pin.row();
    } catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        impl_->checkinReader(conn);
        throw;
    }
    return ReadTxn(this, conn);
}

WriteTxn ObjectStore::write()
{
    if (!impl_->writer)
        throw std::runtime_error("object store: handle is read-only");
    std::unique_lock<std::mutex> lock(impl_->writerMutex);
    exec(impl_->writer, "BEGIN IMMEDIATE;");
    lock.release(); // WriteTxn releases the mutex on commit or rollback.
    return WriteTxn(this, impl_->writer);
}

void ObjectStore::checkpoint()
{
    if (!impl_->writer)
        throw std::runtime_error("object store: checkpoint on read-only handle");
    std::lock_guard<std::mutex> lock(impl_->writerMutex);
    check(sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_FULL, nullptr, nullptr), impl_->writer, "checkpoint");
}

std::string ObjectStore::upsertObject(WriteTxn& txn, const ObjectRecord& record)
{
    return upsertObjectImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

namespace {
void refreshLineMeasurements(sqlite3* db, const std::string& author, const std::string& host,
    const std::string& table, const std::string& uuid, double length,
    const std::optional<std::string>& family, const char* journalKind)
{
    if (!std::isfinite(length) || length < 0)
        throw std::invalid_argument("object store: line length must be finite and nonnegative");
    const auto key = parseUuid(uuid);
    Statement current(db, "SELECT o.id,o.generation,t.length,t.family FROM objects o JOIN " + table +
        " t ON t.object_id=o.id WHERE o.uuid=?");
    current.bindBlob(1, key);
    if (!current.row()) throw std::runtime_error("object store: " + table + " record not found: " + uuid);
    const auto previousFamily = current.text(3);
    if (family && !previousFamily.empty() && previousFamily != *family)
        throw std::runtime_error("object store: registered family differs from source: " + uuid);
    const auto newFamily = family.value_or(previousFamily);
    if (current.f64(2) == length && previousFamily == newFamily) return;
    Statement update(db, "UPDATE " + table + " SET length=?,family=? WHERE object_id=?");
    update.bind(1, length);
    update.bind(2, newFamily);
    update.bind(3, current.i64(0));
    update.stepDone();
    stampContentChange(db, author, host, "objects", current.i64(0), key, journalKind);
}
} // namespace

void ObjectStore::setTrackLength(WriteTxn& txn, const std::string& uuid, double length,
    const std::optional<std::string>& family)
{
    if (family && *family != "" && *family != "horizontal" && *family != "vertical")
        throw std::invalid_argument("object store: invalid packed track family");
    refreshLineMeasurements(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname,
        "tracks", uuid, length, family, family ? "track_measurements" : "track_length");
}

void ObjectStore::setFiberLength(WriteTxn& txn, const std::string& uuid, double length)
{
    refreshLineMeasurements(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname,
        "fibers", uuid, length, std::nullopt, "fiber_length");
}

std::optional<ObjectRecord> ObjectStore::objectByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readObjectRow(db, s);
}

std::vector<ObjectRecord> ObjectStore::objectsByKind(const Txn& txn, const std::string& kind,
                                                 const std::optional<std::string>& approvalState) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
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

int64_t ObjectStore::objectCount(const Txn& txn, const std::string& kind) const
{
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT count(*) FROM objects WHERE kind=?");
    s.bind(1, kind);
    (void)s.row();
    return s.i64(0);
}

std::vector<ObjectRecord> ObjectStore::objectsByTag(const Txn& txn, const std::string& tag) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
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

void ObjectStore::deleteObject(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto bytes = parseUuid(uuid);
    auto id = objectIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: object not found: " + uuid);
    Statement s(db, "DELETE FROM objects WHERE id=?");  // FK RESTRICT on edges refuses while referenced
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete");
}

void ObjectStore::setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state)
{
    if (state != "pending" && state != "approved" && state != "rejected")
        throw std::runtime_error("object store: invalid approval state: " + state);
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    if (state == "approved" && edgeByUuid(txn, recordUuid) && edgeGeometryState(txn, recordUuid).stale)
        throw std::runtime_error("object store: revalidate edge geometry before approving a stale edge");
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
        throw std::runtime_error("object store: record not found: " + recordUuid);
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
    if (key.empty()) throw std::runtime_error("object store: meta key must be non-empty");
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
        throw std::runtime_error("object store: record not found: " + recordUuid);
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

void ObjectStore::setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue)
{
    mutateMeta(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, recordUuid, key, jsonValue);
}

void ObjectStore::removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key)
{
    mutateMeta(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, recordUuid, key, std::nullopt);
}

std::vector<JournalEntry> ObjectStore::journalSince(const Txn& txn, int64_t afterSeq) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT seq,record_uuid,generation,author,hostname,timestamp,change_kind,"
                    "approval_state,approved_by,approved_at,approved_host,approved_generation FROM journal WHERE seq>? ORDER BY seq");
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
        e.approvalState = s.optText(7);
        e.approvedBy = s.optText(8);
        e.approvedAt = s.optText(9);
        e.approvedHost = s.optText(10);
        if (!s.isNull(11)) e.approvedGeneration = s.i64(11);
        out.push_back(std::move(e));
    }
    return out;
}

std::optional<std::filesystem::path> ObjectStore::resolveStoragePath(
    const Txn& txn, const std::string& objectUuid,
    const std::optional<std::filesystem::path>& datasetRoot) const
{
    namespace fs = std::filesystem;
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || (object->storageKind != "file" && object->storageKind != "sqlite") ||
        (object->kind != "patch" && object->kind != "fiber" && object->kind != "point_collection"))
        return std::nullopt;

    const bool patch = object->kind == "patch";
    const auto uuid = parseUuid(object->uuid);
    const auto matches = [&](const fs::path& candidate) {
        const auto metadata = patch ? candidate / "meta.json" : candidate;
        if (!patch && metadata.extension() != ".json")
            return false;
        std::ifstream input(metadata);
        if (!input)
            return false;
        try {
            const auto json = nlohmann::json::parse(input);
            if (object->kind == "point_collection") {
                if (!json.is_object() || json.value("vc_pointcollections_json_version", "") != "1" ||
                    !json.contains("collections") || !json.at("collections").is_object()) return false;
                for (const auto& collection : json.at("collections")) {
                    const auto sourceUuid = collection.value("uuid", "");
                    if (util::isUuid(sourceUuid) && parseUuid(sourceUuid) == uuid) return true;
                }
                return false;
            }
            if (!json.is_object() || !json.contains("uuid") || !json["uuid"].is_string())
                return false;
            if (!patch && !json.contains("control_points") && !json.contains("line_points"))
                return false;
            const auto sourceUuid = json["uuid"].get<std::string>();
            return util::isUuid(sourceUuid) && parseUuid(sourceUuid) == uuid;
        } catch (const nlohmann::json::exception&) {
            return false;
        }
    };

    const auto root = fs::absolute(datasetRoot.value_or(fs::absolute(impl_->path).parent_path())).lexically_normal();
    if (!object->storageRef.empty() && object->storageRef.find("://") == std::string::npos) {
        const auto hint = (root / object->storageRef).lexically_normal();
        if (matches(hint))
            return hint;
    }
    if (!fs::exists(root))
        return std::nullopt;
    // The caller may supply the patch directory itself as the search root.
    if (patch && matches(root))
        return root;

    std::optional<fs::path> found;
    for (fs::recursive_directory_iterator it(root), end; it != end; ++it) {
        const auto& entry = *it;
        fs::path candidate;
        if (entry.is_directory()) {
            const auto name = entry.path().filename().string();
            if (name.front() == '.' || entry.path().extension() == ".zarr") {
                it.disable_recursion_pending();
                continue;
            }
            if (!fs::is_regular_file(entry.path() / "meta.json"))
                continue;
            // Patch geometry can be large; only its metadata is relevant.
            it.disable_recursion_pending();
            if (!patch)
                continue;
            candidate = entry.path();
        } else {
            if (patch || entry.path().filename() == "meta.json" || !entry.is_regular_file())
                continue;
            candidate = entry.path();
        }
        if (!matches(candidate))
            continue;
        if (found) {
            // Multiple symlinks/hard links to the same object are harmless.
            if (fs::equivalent(*found, candidate))
                continue;
            throw std::runtime_error("ambiguous storage for UUID " + object->uuid + ": " +
                                     found->string() + " and " + candidate.string());
        }
        found = candidate;
    }
    return found;
}

std::vector<std::string> ObjectStore::sweepExternalData(const Txn& txn) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    std::vector<std::string> problems;
    Statement s(db, "SELECT uuid,storage_ref,data_generation,generation FROM objects WHERE storage_ref<>'' AND storage_kind<>'sqlite' ORDER BY id");
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


} // namespace vc::objects
