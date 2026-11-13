#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

#include "internal/Sqlite.hpp"
#include "internal/Ownership.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
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

#include "vc_object_store/Uuid.hpp"

namespace vc::objects {
namespace detail { class GeometryReadCache; class SpatialReadCache; }
// Only ordinary history-off edits need this transaction-local validation
// state. Savepoints retain copy-on-write snapshots; nothing is stored in SQL.
struct CurrentWriteState {
    std::map<std::string,nlohmann::json> before;
    std::map<std::string,std::pair<nlohmann::json,nlohmann::json>> validated;
    // Savepoint-aware receipt changes, never persisted as per-record history.
    std::set<std::string> changedObjects, changedRelationships, created;
};
// Connections are leased to one thread per transaction. Cache idle statements
// on the connection; the thread-local registry only identifies the active lease.
struct StatementPool {
    std::optional<std::set<std::string>> edgeColumnNames;
    // Installed only by a collection transaction with pinned dependencies.
    // Shared lookup for immutable records and payloads. Owns no data, aliases
    // or per-record directories; returned connections live for the transaction.
    std::function<sqlite3*(const std::function<bool(sqlite3*)>&)> referenceSource;
    // The optional final argument is a row already selected as live in this
    // same snapshot. Resolution still checks every higher-precedence source.
    std::function<sqlite3*(const std::string&,bool,bool,sqlite3*)> objectSource;
    std::function<sqlite3*(const std::string&,bool,bool,sqlite3*)> edgeSource;
    // Physical query sources in this pinned logical scope; visibility is
    // checked by key resolution before filtering/pagination is finalized.
    std::function<std::vector<sqlite3*>()> querySources;
    std::shared_ptr<CurrentWriteState> currentWrite;
    bool historyEnabled = true;
    bool spatialEnabled = true;
    bool immutableInitial = false;
    int startSchemaVersion = 0;
    std::map<std::string,double> transformedLengths;
    bool readSnapshot = false;
    bool checkedAction = false, actionConflict = false;
    VersionSelection checkedObjects;
    // Attribution and request identity of the active write (see
    // TransactionContext). actingAuthor overrides the open author for
    // mutation stamps; commit* attribute the store commit.
    std::optional<std::string> actingAuthor, requestUuid, requestDigest;
    std::string commitAuthor, commitHost;
    std::optional<int64_t> publishedSequence; // set by publishHistory
    // Components read on this connection, filed under every member's id (one
    // shared list per component; see detail::componentIds). Kept across read
    // snapshots while the connection's data_version is unchanged.
    struct Component {
        std::vector<int64_t> ids;
        std::optional<std::vector<std::string>> uuids; // filled on first ask
    };
    std::unordered_map<std::string, std::shared_ptr<Component>> components;
    size_t componentNodes = 0;
    std::optional<int> componentsVersion;
    std::shared_ptr<detail::GeometryReadCache> geometryCache;
    std::shared_ptr<detail::SpatialReadCache> spatialCache;
    std::unordered_multimap<std::string, sqlite3_stmt*> idle;
    ~StatementPool() { for (auto& [sql, stmt] : idle) sqlite3_finalize(stmt); }
};
// Defined in Database.cpp: MinGW GCC emits a non-COMDAT TLS init function for
// a dynamically initialized inline thread_local, so each TU would define it.
extern thread_local std::unordered_map<sqlite3*, StatementPool*> activeStatementPools;
namespace detail {
bool historyEnabled(sqlite3*);
void requireHistory(sqlite3*);
bool spatialEnabled(sqlite3*);
void requireSpatial(sqlite3*);
int64_t writeCommitSequence(sqlite3*);
int64_t writeObjectVersion(sqlite3*, const std::string& uuid);
int64_t writeRecordVersion(sqlite3*, const char* table, const std::string& uuid);
int64_t allocateRecordId(sqlite3*, const char* keySpace);
void createProvisioning(sqlite3*, bool history, bool spatial = true);
void createChangeTracking(sqlite3*);
void createStorageDirectory(sqlite3*, const std::string& databaseUuid = {});
void registerStorageTables(sqlite3*);
void forkStorageIdentity(sqlite3*);
void validateCurrentRelationships(sqlite3*);
// Called only after a read connection has pinned its new SQLite snapshot.
void beginReadCaches(sqlite3*, StatementPool&, size_t geometryBudget, size_t spatialBudget);
void beginSpatialRead(StatementPool&, size_t budget, int version);
void beforeCurrentObjectEdit(sqlite3*, const std::string& uuid);
// The current schema version and the whole schema of a new store (one
// history generation), inside the caller's transaction. The offline
// converter builds its destination with the same function.
inline constexpr int kSchemaVersion = 22;
void createStoreSchema(sqlite3*, const std::string& generation);
}
struct ObjectStore::Impl {
    // Declared first: released only after every connection has closed.
    std::unique_ptr<detail::OwnershipGuard> ownership;
    std::filesystem::path path;
    OpenOptions options;
    std::string hostname;
    bool immutableInitial = false;
    sqlite3* writer = nullptr;
    bool composed = false;
    std::mutex writerMutex;
    std::vector<sqlite3*> readerPool;
    std::mutex poolMutex;
    std::condition_variable poolCv;
    std::map<sqlite3*, std::unique_ptr<StatementPool>> statementPools;

    ~Impl()
    {
        statementPools.clear();
        for (sqlite3* c : readerPool)
            if (c) sqlite3_close(c);
        if (writer) sqlite3_close(writer);
    }

    sqlite3* checkoutReader()
    {
        if(options.readerPoolSize==0)throw std::logic_error("private initial construction has no reader connections");
        std::unique_lock<std::mutex> lock(poolMutex);
        poolCv.wait(lock, [&] { return !readerPool.empty(); });
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
    static ObjectStore createInitial(const std::filesystem::path& path,const OpenOptions& options) {
        return ObjectStore(ObjectStore::openImpl(path,options,true,true));
    }
    static sqlite3* conn(const Txn& txn)
    {
        if (!txn.active())
            throw std::runtime_error("object store: use of finished transaction");
        return static_cast<sqlite3*>(txn.handle_);
    }
    // Internal revision-UUID lookups (storage keys; see ObjectStore's private section).
    static std::optional<ObjectRevision> objectRevision(const ObjectStore& store, const Txn& txn, const std::string& revision)
    {
        return store.objectRevision(txn, revision);
    }
    static RevisionSelection revisionSelectionAt(const ObjectStore& store, const Txn& txn, int64_t publishedAtUs,
                                                 bool includeRetired)
    {
        return store.revisionSelectionAt(txn, publishedAtUs, includeRetired);
    }
};

} // namespace vc::objects
