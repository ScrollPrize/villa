#include "internal/Schema.hpp"
#include "geometry/DependencyIndex.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "internal/History.hpp"
#include "internal/RecordSql.hpp"
#include "hub/HubRules.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "internal/ImmutableImport.hpp"

namespace vc::objects {
thread_local std::unordered_map<sqlite3*, StatementPool*> activeStatementPools;
using namespace detail;
ObjectStore::ObjectStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ObjectStore::~ObjectStore() = default;
ObjectStore::ObjectStore(ObjectStore&&) noexcept = default;
ObjectStore& ObjectStore::operator=(ObjectStore&&) noexcept = default;

std::unique_ptr<ObjectStore::Impl> ObjectStore::openImpl(const std::filesystem::path& path, const OpenOptions& options, bool create, bool immutable)
{
    const int kApplicationId = 0x56434f42;
    const int kSchemaVersion = 18;
    if (create && options.readOnly) throw std::invalid_argument("create requires writable options");
    if (create && options.sharedFile) throw std::invalid_argument("object store: create a store locally, then initialize it as a hub");
    if (create && (options.initialGeneration<1 || options.initialGeneration==INT64_MAX))
        throw std::invalid_argument("object store: invalid initial generation");
    if (create) for (const auto& range : {options.objectIdRange,options.edgeIdRange})
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
    impl->options.readerPoolSize = options.sharedFile ? 1 : std::max(1, options.readerPoolSize);
    if (options.sharedFile) impl->options.geometryCacheBytes = impl->options.spatialCacheBytes = 0;
    impl->hostname = options.hostnameOverride.value_or(detectHostname());

    auto validate = [&](sqlite3* db) {
        if (pragmaInt(db, "application_id") != kApplicationId)
            throw std::runtime_error("object store: unsupported database format: " + path.string());
        if (pragmaInt(db, "user_version") != kSchemaVersion)
            throw std::runtime_error("object store: unsupported database schema (convert legacy stores with the separate v2 conversion tool): " + path.string());
        Statement manifest(db, "SELECT format_version,history_enabled,sealed,spatial_enabled FROM store_manifest WHERE singleton=1");
        if (!manifest.row() || manifest.i64(0) != 2)
            throw std::runtime_error("object store: invalid v2 manifest");
        impl->options.history = manifest.i64(1) ? HistoryMode::On : HistoryMode::Off;
        impl->options.spatialIndex = manifest.i64(3) != 0;
        if (manifest.i64(2) && !options.readOnly)
            throw std::runtime_error("object store: sealed store requires a read-only open");
        Statement features(db, "SELECT name FROM store_required_features");
        bool currentState=false,binaryHashes=false,endpointKeys=false;
        while (features.row()) {
            if(features.text(0)=="current-state-v1"){currentState=true;continue;}
            if(features.text(0)=="binary-sha256-v1"){binaryHashes=true;continue;}
            if(features.text(0)=="endpoint-keys-v1"){endpointKeys=true;continue;}
            if (features.text(0)=="collection-v1" && options.physicalPartitionOnly) continue;
            if (features.text(0)==immutableImportFeature) {
                if(!options.readOnly || !manifest.i64(2) || manifest.i64(1) || manifest.i64(3))
                    throw std::runtime_error("immutable initial store must be sealed, history/spatial-off and opened readonly");
                impl->immutableInitial=true;continue;
            }
            throw std::runtime_error("object store: unsupported required feature (use ObjectCollection for collections): " + features.text(0));
        }
        if(!currentState) throw std::runtime_error("object store: legacy current-state layout requires separate conversion; refusing mixed v2 semantics");
        if(!binaryHashes||!endpointKeys) throw std::runtime_error("object store: earlier experimental v2 layout requires rebuilding (binary hashes/qualified endpoint keys absent)");
        Statement physical(db, "SELECT schema_version FROM database_identity WHERE singleton=1");
        if (!physical.row() || physical.i64(0) != kSchemaVersion)
            throw std::runtime_error("object store: invalid database identity/schema");
        // A hub must keep its rollback journal: opening it with WAL from a
        // network mount would break locking for every other client.
        impl->hubRole = detectHubRole(db);
        if (impl->hubRole == HubRole::Hub && !options.sharedFile)
            throw std::runtime_error("object store: " + path.string() + " is a hosted hub; clone it to a local replica");
        if (impl->hubRole != HubRole::Hub && options.sharedFile)
            throw std::runtime_error("object store: " + path.string() + " is not a hosted hub");
    };
    const bool buildingPacked = isBuildingPackedDatabasePath(path);
    const char* journalMode = options.sharedFile || buildingPacked ? "PRAGMA journal_mode=DELETE;" : "PRAGMA journal_mode=WAL;";

    if (create) {
        std::error_code ec;
        if (!isPackedDatabasePath(path) && std::filesystem::exists(path) && std::filesystem::file_size(path, ec) > 0)
            throw std::runtime_error("object store: create refuses existing non-empty file: " + path.string());
        impl->writer = openConnection(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
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
            createSchema(impl->writer, options.history == HistoryMode::On, options.creationStoreUuid,immutable);
            if(immutable)execute(impl->writer,"INSERT INTO store_required_features VALUES(?)",std::string(immutableImportFeature));
            execute(impl->writer,"UPDATE store_manifest SET generation=?",options.initialGeneration);
            Statement owner(impl->writer,"SELECT uuid FROM store_identity"); owner.row();
            for (const auto& [name,range] : {std::pair{"objects",options.objectIdRange},std::pair{"edges",options.edgeIdRange}})
                execute(impl->writer,"INSERT INTO key_allocators VALUES(?,?,?,?,?)",std::string(name),owner.blob16(0),range[0],range[1],range[0]);
            if (options.history == HistoryMode::On) createHistorySchema(impl->writer);
            createLifecycleSchema(impl->writer);
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
        exec(impl->writer, journalMode);
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
std::string ObjectStore::author() const { return impl_->options.author; }

ObjectStore ObjectStore::create(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,true)); }
ObjectStore ObjectStore::open(const std::filesystem::path& p, const OpenOptions& o) { return ObjectStore(openImpl(p,o,false)); }
std::string ObjectStore::uuid(const Txn& txn) const {
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM store_identity");
    if (!s.row()) throw std::runtime_error("object store: missing identity");
    return formatUuid(s.blob16(0));
}

} // namespace vc::objects
