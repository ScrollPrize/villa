#include "internal/Database.hpp"
#include "internal/History.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "hub/HubRules.hpp"
#include "internal/ExternalEndpoint.hpp"

namespace vc::objects {
using namespace detail;
Txn::Txn(Txn&& other) noexcept : db_(other.db_), handle_(other.handle_), write_(other.write_),
    expectedRevisions_(std::move(other.expectedRevisions_)), committedRevisions_(std::move(other.committedRevisions_))
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
        // Collection resolvers capture the transaction's pinned dependencies;
        // pooled connections must never retain those captures after release.
        if (const auto pool = activeStatementPools.find(conn); pool != activeStatementPools.end()) {
            pool->second->externalGeometryChunk = {};
            pool->second->externalGeometryChunkSize = {};
            pool->second->externalEndpointRecord = {};
            pool->second->externalEndpointVersion = {};
            pool->second->externalEndpointOwner = {};
            pool->second->externalEndpointValidateAnchor = {};
            pool->second->currentWrite.reset();
        }
        activeStatementPools.erase(conn);
        if (write_) db_->impl_->writerMutex.unlock();
        else db_->impl_->checkinReader(conn);
    };
    try {
        if (commit && write_) {
            if (pragmaInt(conn, "schema_version") != activeStatementPools.at(conn)->startSchemaVersion) {
                registerStorageTables(conn);
                exec(conn, "UPDATE store_write_state SET dirty=1 WHERE singleton=1;");
            }
            const bool immutable=activeStatementPools.at(conn)->immutableInitial;
            if(!immutable) {
                refreshSourceIdentityIndex(conn);
                if (historyEnabled(conn)) publishHistory(conn);
                else validateCurrentRelationships(conn);
            } else {refreshSourceIdentityIndex(conn);exec(conn,"UPDATE store_write_state SET dirty=1 WHERE singleton=1;");}
            exec(conn, "UPDATE store_manifest SET generation=generation+1 WHERE singleton=1 AND "
                       "(SELECT dirty FROM store_write_state WHERE singleton=1)=1;"
                       "UPDATE store_write_state SET dirty=0 WHERE singleton=1 AND dirty=1;");
            enforceHubRules(conn, *activeStatementPools.at(conn), db_->impl_->hubRole);
            for (const auto& [uuid, expected] : expectedRevisions_) {
                Statement head(conn, "SELECT revision_uuid FROM object_heads WHERE object_uuid=?",
                               parseUuid(uuid));
                if(head.row())committedRevisions_[uuid]=formatUuid(head.blob16(0));
                else committedRevisions_[uuid]=externalEndpointKey(conn,uuid)?std::optional(externalEndpointVersion(conn,uuid)):std::nullopt;
            }
        }
        exec(conn, commit ? "COMMIT;" : "ROLLBACK;");
    }
    catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        committedRevisions_.clear();
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
        activeStatementPools[conn] = impl_->statementPools.at(conn).get();
        auto& pool = *impl_->statementPools.at(conn);
        pool.readSnapshot = true;
        pool.components.clear(); pool.componentNodes = 0;
        exec(conn, "BEGIN;");
        // Establish the snapshot now, so all reads in this transaction
        // see one consistent state even before the first real query.
        Statement pin(conn, "SELECT uuid FROM store_identity LIMIT 1");
        (void)pin.row();
        detail::beginReadCaches(conn, *impl_->statementPools.at(conn),
                               impl_->options.geometryCacheBytes, impl_->options.spatialIndex ? impl_->options.spatialCacheBytes : 0);
    } catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        activeStatementPools.erase(conn);
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
    activeStatementPools[impl_->writer] = impl_->statementPools.at(impl_->writer).get();
    impl_->statementPools.at(impl_->writer)->readSnapshot = false;
    impl_->statementPools.at(impl_->writer)->checkedAction = false;
    impl_->statementPools.at(impl_->writer)->actionConflict = false;
    try {
        exec(impl_->writer, "BEGIN IMMEDIATE;");
        impl_->statementPools.at(impl_->writer)->startSchemaVersion = pragmaInt(impl_->writer, "schema_version");
        Statement manifest(impl_->writer, "SELECT sealed,generation,history_enabled FROM store_manifest WHERE singleton=1");
        if (!manifest.row() || manifest.i64(0)) throw std::runtime_error("object store: store is sealed");
        if (manifest.i64(1) == INT64_MAX) throw std::runtime_error("object store: generation exhausted");
        impl_->statementPools.at(impl_->writer)->historyEnabled = manifest.i64(2) != 0;
        beginCurrentWrite(impl_->writer);
        if (impl_->hubRole != HubRole::None) beginHubWrite(impl_->writer, *impl_->statementPools.at(impl_->writer));
    }
    catch (...) { sqlite3_exec(impl_->writer, "ROLLBACK;", nullptr, nullptr, nullptr); activeStatementPools.erase(impl_->writer); throw; }
    lock.release(); // WriteTxn releases the mutex on commit or rollback.
    return WriteTxn(this, impl_->writer);
}

WriteTxn ObjectStore::write(const RevisionSelection& expected)
{
    auto txn = write();
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    expectHistory(static_cast<sqlite3*>(connection(txn)), expected);
    txn.expectedRevisions_ = expected;
    return txn;
}

void ObjectStore::checkpoint()
{
    if (!impl_->writer)
        throw std::runtime_error("object store: checkpoint on read-only handle");
    std::lock_guard<std::mutex> lock(impl_->writerMutex);
    if(impl_->immutableInitial)return; // journal-off private build; publication owns persistence
    // Bulk-import callers checkpoint after publication. Maintain planner
    // statistics here, outside their atomic write transaction.
    exec(impl_->writer, "PRAGMA optimize;");
    check(sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_FULL, nullptr, nullptr), impl_->writer, "checkpoint");
}

} // namespace vc::objects
