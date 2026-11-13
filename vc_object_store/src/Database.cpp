#include "internal/Schema.hpp"
#include "geometry/DependencyIndex.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "internal/History.hpp"
#include "internal/RecordSql.hpp"
#include "internal/Versions.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "internal/ImmutableImport.hpp"

namespace vc::objects {
thread_local std::unordered_map<sqlite3*, StatementPool*> activeStatementPools;
using namespace detail;
ObjectStore::ObjectStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ObjectStore::~ObjectStore() = default;
ObjectStore::ObjectStore(ObjectStore&&) noexcept = default;
ObjectStore& ObjectStore::operator=(ObjectStore&&) noexcept = default;

void detail::createStoreSchema(sqlite3* db, const std::string& generation)
{
    createProvisioning(db, true, true);
    createSchema(db, true, std::nullopt, false);
    execute(db, "UPDATE store_manifest SET generation=?,initial_generation=?", parseUuid(generation), parseUuid(generation));
    Statement owner(db, "SELECT uuid FROM store_identity"); owner.row();
    for (const char* name : {"objects"})
        execute(db, "INSERT INTO key_allocators VALUES(?,?,1,?,1)", std::string(name), owner.blob16(0), int64_t(INT64_MAX));
    createHistorySchema(db);
    createVersionSchema(db, generation);
    createGeometryHistoryIndex(db);
    createSourceIdentityIndex(db);
    createStorageDirectory(db);
    execute(db, "UPDATE database_identity SET schema_version=?", int64_t(kSchemaVersion));
    registerStorageTables(db);
    createChangeTracking(db);
}

std::unique_ptr<ObjectStore::Impl> ObjectStore::openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create, bool immutable)
{
    const int kApplicationId = 0x56434f42;
    if (create && options.readOnly) throw std::invalid_argument("create requires writable options");
    if (options.serviceOwner && options.readOnly) throw std::invalid_argument("hosting service requires writable access");
    if (create && (options.initialCommitSequence<0 || options.initialCommitSequence==INT64_MAX))
        throw std::invalid_argument("object store: invalid initial generation");
    if (create) for (const auto& range : {options.objectIdRange})
        if(range[0]<1 || range[1]<range[0]) throw std::invalid_argument("object store: invalid reserved ID range");
    if (sqlite3_threadsafe() == 0)
        throw std::runtime_error("object store: SQLite build is not thread-safe");
    if (!create && !options.readOnly && options.author.empty())
        throw std::runtime_error("object store: read-write open requires an author (session identity)");
    if (create && options.author.empty())
        throw std::runtime_error("object store: create requires an author (session identity)");

    auto impl = std::make_unique<ObjectStore::Impl>();
    impl->path = path;
    impl->options = options;
    impl->immutableInitial=immutable;
    if(immutable && (!create || options.history!=HistoryMode::Off || options.spatialIndex))
        throw std::invalid_argument("immutable construction requires a new history-off spatial-off database");
    impl->options.readerPoolSize = immutable ? 0 : std::max(1, options.readerPoolSize);
    impl->hostname = options.hostnameOverride.value_or(detectHostname());

    auto validate = [&](sqlite3* db) {
        if (pragmaInt(db, "application_id") != kApplicationId)
            throw std::runtime_error("object store: unsupported database format: " + path.string());
        if (pragmaInt(db, "user_version") != kSchemaVersion)
            throw std::runtime_error("object store: unsupported database schema (convert upstream stores with vc_object_convert): " + path.string());
        Statement manifest(db, "SELECT schema_version,history_enabled,sealed,spatial_enabled FROM store_manifest WHERE singleton=1");
        if (!manifest.row() || manifest.i64(0) != kSchemaVersion)
            throw std::runtime_error("object store: invalid store manifest");
        impl->options.history = manifest.i64(1) ? HistoryMode::On : HistoryMode::Off;
        impl->options.spatialIndex = manifest.i64(3) != 0;
        if (manifest.i64(2) && !options.readOnly)
            throw std::runtime_error("object store: sealed store requires a read-only open");
        Statement features(db, "SELECT name FROM store_required_features");
        bool currentState=false,binaryHashes=false,endpointKeys=false,stateLayout=false,binaryUuids=false;
        while (features.row()) {
            if(features.text(0)=="direct-geometry-v1"){currentState=true;continue;}
            if(features.text(0)=="binary-sha256-v1"){binaryHashes=true;continue;}
            if(features.text(0)=="direct-endpoint-keys-v1"){endpointKeys=true;continue;}
            if(features.text(0)=="object-state-v1"){stateLayout=true;continue;}
            if(features.text(0)=="binary-uuid-v1"){binaryUuids=true;continue;}
            if(features.text(0)=="uuid-hash-routing-v1")continue;
            if (features.text(0)=="collection-v1") {impl->composed=true;continue;}
            if (features.text(0)==immutableImportFeature) {
                if(!options.readOnly || !manifest.i64(2) || manifest.i64(1) || manifest.i64(3))
                    throw std::runtime_error("immutable initial store must be sealed, history/spatial-off and opened readonly");
                impl->immutableInitial=true;continue;
            }
            throw std::runtime_error("object store: unsupported required feature: " + features.text(0));
        }
        if(!currentState||!binaryHashes||!endpointKeys||!stateLayout||!binaryUuids)
            throw std::runtime_error("object store: required current-schema features are absent (direct geometry, binary hashes, direct endpoint keys, state or binary UUIDs)");
        Statement physical(db, "SELECT schema_version FROM database_identity WHERE singleton=1");
        if (!physical.row() || physical.i64(0) != kSchemaVersion)
            throw std::runtime_error("object store: invalid database identity/schema");
    };
    const bool buildingPacked = isBuildingPackedDatabasePath(path);
    const char* journalMode = buildingPacked ? "PRAGMA journal_mode=DELETE;" : "PRAGMA journal_mode=WAL;";
    if (!options.readOnly && !buildingPacked && !immutable)
        impl->ownership = std::make_unique<OwnershipGuard>(path,
            options.serviceOwner ? OwnershipGuard::Mode::Service : OwnershipGuard::Mode::Standalone);

    if (create) {
        std::error_code ec;
        if (!isPackedDatabasePath(path) && std::filesystem::exists(path) && std::filesystem::file_size(path, ec) > 0)
            throw std::runtime_error("object store: create refuses existing non-empty file: " + path.string());
        // Private initial-import writers are owned by one worker and never
        // exposed as editable stores. Keep mutex protection on normal stores.
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
            (immutable ? SQLITE_OPEN_NOMUTEX : 0), options, false);
        if (buildingPacked && pragmaInt(impl->writer, "page_count") != 0)
            throw std::runtime_error("object store: create refuses non-empty packed database: " + path.string());
        // Private bulk-built databases use large pages: dense sequential rows
        // and geometry payloads otherwise waste space in 4-KiB leaf pages.
        // Set before the first schema/application-id write; ordinary stores
        // and existing files retain their page size.
        if(immutable)exec(impl->writer,"PRAGMA page_size=65536;");
        exec(impl->writer, immutable?"PRAGMA journal_mode=OFF;PRAGMA synchronous=OFF;":journalMode);
        if(!immutable)exec(impl->writer, "PRAGMA synchronous=FULL;");
        exec(impl->writer, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
        exec(impl->writer, "BEGIN IMMEDIATE;");
        try {
            createProvisioning(impl->writer, options.history == HistoryMode::On, options.spatialIndex);
            createSchema(impl->writer, options.history == HistoryMode::On, options.creationStoreUuid,immutable,options);
            if(immutable)execute(impl->writer,"INSERT INTO store_required_features VALUES(?)",std::string(immutableImportFeature));
            const auto lineage = parseUuid(options.creationGeneration.value_or(util::newUuid()));
            execute(impl->writer,"UPDATE store_manifest SET commit_sequence=?,generation=?,initial_generation=?",
                options.initialCommitSequence,lineage,lineage);
            Statement owner(impl->writer,"SELECT uuid FROM store_identity"); owner.row();
            for (const auto& [name,range] : {std::pair{"objects",options.objectIdRange}})
                execute(impl->writer,"INSERT INTO key_allocators VALUES(?,?,?,?,?)",std::string(name),owner.blob16(0),range[0],range[1],range[0]);
            if (options.history == HistoryMode::On) {
                createHistorySchema(impl->writer);
                createVersionSchema(impl->writer, formatUuid(lineage));
            }
            if(!immutable)createGeometryHistoryIndex(impl->writer);
            createStorageDirectory(impl->writer, buildingPacked ? packedDatabaseUuid(path) : std::string{});
            exec(impl->writer, ("PRAGMA user_version=" + std::to_string(kSchemaVersion) + ";").c_str());
            execute(impl->writer, "UPDATE database_identity SET schema_version=? WHERE singleton=1", int64_t(kSchemaVersion));
            registerStorageTables(impl->writer);
            if(!immutable)createChangeTracking(impl->writer);
            exec(impl->writer, "COMMIT;");
        } catch (...) {
            exec(impl->writer, "ROLLBACK;");
            throw;
        }
    } else if (!options.readOnly) {
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
        validate(impl->writer);
        exec(impl->writer, "PRAGMA journal_mode=WAL;");
        exec(impl->writer, "PRAGMA synchronous=FULL;");
    }

    // WAL readers need write access to the -shm file, so pooled readers
    // are read-write connections pinned to query_only (true READONLY
    // only for a fully read-only open).
    const int readerFlags = options.readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE;
    for (int i = 0; i < impl->options.readerPoolSize; ++i) {
        sqlite3* conn = openConnection(path, readerFlags, options, true);
        impl->readerPool.push_back(conn);
        if (options.readOnly) validate(conn);
        auto pool = std::make_unique<StatementPool>();
        pool->historyEnabled = impl->options.history == HistoryMode::On;
        pool->spatialEnabled = impl->options.spatialIndex;
        pool->immutableInitial = impl->immutableInitial;
        impl->statementPools.emplace(conn, std::move(pool));
    }
    if (impl->writer) {
        auto pool = std::make_unique<StatementPool>();
        pool->historyEnabled = impl->options.history == HistoryMode::On;
        pool->spatialEnabled = impl->options.spatialIndex;
        pool->immutableInitial = impl->immutableInitial;
        impl->statementPools.emplace(impl->writer, std::move(pool));
    }
    return impl;
}

const std::filesystem::path& ObjectStore::path() const { return impl_->path; }
std::string ObjectStore::hostname() const { return impl_->hostname; }

ObjectStore ObjectStore::create(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,true)); }
ObjectStore ObjectStore::open(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,false)); }
std::string ObjectStore::uuid(const Txn& txn) const {
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM store_identity");
    if (!s.row()) throw std::runtime_error("object store: missing identity");
    return formatUuid(s.blob16(0));
}

} // namespace vc::objects
