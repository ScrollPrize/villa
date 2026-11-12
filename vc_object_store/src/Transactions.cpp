#include "internal/Database.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "internal/CollectionReads.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

namespace vc::objects {
using namespace detail;
Txn::Txn(Txn&& other) noexcept : db_(other.db_), handle_(other.handle_), write_(other.write_),
    referenceContext_(std::move(other.referenceContext_)),
    expectedRevisions_(std::move(other.expectedRevisions_)),
    receipt_(std::move(other.receipt_)), expectedObjectVersions_(std::move(other.expectedObjectVersions_)),
    expectedRelationshipVersions_(std::move(other.expectedRelationshipVersions_))
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
    auto releaseConnection = [&] {
        handle_ = nullptr;
        referenceContext_.reset();
        // Collection resolvers capture the transaction's pinned dependencies;
        // pooled connections must never retain those captures after release.
        if (const auto pool = activeStatementPools.find(conn); pool != activeStatementPools.end()) {
            // Readers can hold a pinned dependency connection. Never retain
            // such readers after releasing this collection transaction.
            if (pool->second->referenceSource) pool->second->geometryCache.reset();
            pool->second->referenceSource = {};
            pool->second->objectSource = {};
            pool->second->edgeSource = {};
            pool->second->querySources = {};
            pool->second->currentWrite.reset();
        }
        activeStatementPools.erase(conn);
        if (write_) db_->impl_->writerMutex.unlock();
        else db_->impl_->checkinReader(conn);
    };
    try {
        if (commit && write_) {
            auto& pool = *activeStatementPools.at(conn);
            if(pool.actionConflict) throw std::runtime_error("object store: checked action failed; all changes rolled back");
            if (pragmaInt(conn, "schema_version") != pool.startSchemaVersion) {
                registerStorageTables(conn);
                exec(conn, "UPDATE store_write_state SET dirty=1 WHERE singleton=1;");
            }
            refreshSourceIdentityIndex(conn);
            if (historyEnabled(conn)) publishHistory(conn);
            else if (!pool.immutableInitial) validateCurrentRelationships(conn);
            if (pool.immutableInitial) exec(conn, "UPDATE store_write_state SET dirty=1 WHERE singleton=1;");
            Statement dirty(conn, "SELECT dirty FROM store_write_state WHERE singleton=1");
            if (dirty.row() && dirty.i64(0)) {
                pool.publishedSequence = writeCommitSequence(conn);
                execute(conn, "UPDATE store_manifest SET commit_sequence=? WHERE singleton=1", *pool.publishedSequence);
                exec(conn, "UPDATE store_write_state SET dirty=0 WHERE singleton=1;");
            }
            auto expectedObjects = expectedObjectVersions_;
            for (const auto& [uuid, expected] : expectedRevisions_) expectedObjects.emplace(uuid, std::nullopt);
            receipt_ = finishReceipt(conn, *activeStatementPools.at(conn), expectedObjects, expectedRelationshipVersions_);
        }
        exec(conn, commit ? "COMMIT;" : "ROLLBACK;");
    }
    catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        receipt_.reset();
        releaseConnection(); throw;
    }
    releaseConnection();
}

void* ObjectStore::connection(const Txn& txn) const {
    if (txn.db_ != this) throw std::invalid_argument("transaction belongs to another database");
    return ObjectStoreImplAccess::conn(txn);
}

ReadTxn ObjectStore::read() const
{
    sqlite3* conn = impl_->checkoutReader();
    try {
        activeStatementPools[conn] = impl_->statementPools.at(conn).get();
        auto& pool = *impl_->statementPools.at(conn);
        pool.readSnapshot = true;
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
    ReadTxn txn(this,conn);
    if(impl_->composed&&!impl_->options.physicalPartitionOnly)
        txn.referenceContext_=attachCollectionReferenceResolver(conn,impl_->path,impl_->options);
    return txn;
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
    {
        auto& pool = *impl_->statementPools.at(impl_->writer);
        pool.actingAuthor.reset(); pool.requestUuid.reset(); pool.requestDigest.reset();
        pool.checkedObjects.clear();
        pool.publishedSequence.reset();
        pool.commitAuthor = impl_->options.author; pool.commitHost = impl_->hostname;
    }
    try {
        exec(impl_->writer, "BEGIN IMMEDIATE;");
        auto& pool = *impl_->statementPools.at(impl_->writer);
        pool.startSchemaVersion = pragmaInt(impl_->writer, "schema_version");
        Statement manifest(impl_->writer, "SELECT sealed,commit_sequence,history_enabled FROM store_manifest WHERE singleton=1");
        if (!manifest.row() || manifest.i64(0)) throw std::runtime_error("object store: store is sealed");
        if (manifest.i64(1)==INT64_MAX) throw std::overflow_error("object store: commit sequence exhausted");
        pool.historyEnabled = manifest.i64(2)!=0;
        beginCurrentWrite(impl_->writer);
    }
    catch (...) { sqlite3_exec(impl_->writer, "ROLLBACK;", nullptr, nullptr, nullptr); activeStatementPools.erase(impl_->writer); throw; }
    lock.release(); // WriteTxn releases the mutex on commit or rollback.
    WriteTxn txn(this,impl_->writer);
    if(impl_->composed&&!impl_->options.physicalPartitionOnly)
        txn.referenceContext_=attachCollectionReferenceResolver(impl_->writer,impl_->path,impl_->options);
    return txn;
}

WriteTxn ObjectStore::write(const RecordSelection& expectedObjects)
{
    TransactionContext context;
    context.expectedObjects = expectedObjects;
    return write(context);
}

WriteTxn ObjectStore::write(const TransactionContext& context)
{
    auto txn = write();
    configureWrite(txn, context);
    return txn;
}

WriteTxn ObjectStore::writeNumbered(const VersionSelection& numbers)
{
    auto txn = write();
    TransactionContext context;
    context.expectedObjects = selectCurrentExpectations(LocalStoreReader(*this, txn), numbers);
    configureWrite(txn, context);
    return txn;
}

void ObjectStore::configureWrite(WriteTxn& txn, const TransactionContext& context)
{
    if (context.author && context.author->empty()) throw std::invalid_argument("object store: empty transaction author");
    auto* db = static_cast<sqlite3*>(connection(txn));
    auto& pool = *activeStatementPools.at(db);
    if (context.requestUuid) {
        const auto request = util::canonicalUuid(*context.requestUuid);
        if (savedReceipt(db, request))
            throw std::invalid_argument("object store: request " + request + " already committed; recover its receipt");
        pool.requestUuid = request;
        pool.requestDigest = context.requestDigest;
    }
    if (context.author) pool.actingAuthor = *context.author;
    RevisionSelection expected;
    prepareCheckedWrite(db, context, expected);
    txn.expectedRevisions_ = expected;
    txn.expectedObjectVersions_ = numberedSelection(context.expectedObjects);
    txn.expectedRelationshipVersions_ = numberedSelection(context.expectedRelationships);
}

std::string ObjectStore::author() const
{
    // A write's transaction context attributes every stamp it makes, without
    // changing the handle's options for other threads/requests.
    if (impl_->writer)
        if (const auto pool = activeStatementPools.find(impl_->writer); pool != activeStatementPools.end() &&
            pool->second->actingAuthor) return *pool->second->actingAuthor;
    return impl_->options.author;
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

ObjectStore::WalCheckpoint ObjectStore::checkpointWal(WalCheckpointMode mode, std::chrono::milliseconds wait)
{
    if (!impl_->writer)
        throw std::runtime_error("object store: checkpoint on read-only handle");
    std::lock_guard<std::mutex> lock(impl_->writerMutex);
    WalCheckpoint result;
    int log = 0, copied = 0;
    int rc = SQLITE_OK;
    if (mode == WalCheckpointMode::Passive) {
        rc = sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_PASSIVE, &log, &copied);
    } else {
        // Truncate waits through the busy handler: bound it to `wait`.
        sqlite3_busy_timeout(impl_->writer, static_cast<int>(std::clamp<int64_t>(wait.count(), 0, INT32_MAX)));
        rc = sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_TRUNCATE, &log, &copied);
        sqlite3_busy_timeout(impl_->writer, impl_->options.busyTimeoutMs);
    }
    if (rc == SQLITE_BUSY) result.busy = true;
    else check(rc, impl_->writer, "checkpoint");
    result.logFrames = log < 0 ? 0 : log;
    result.checkpointedFrames = copied < 0 ? 0 : copied;
    std::error_code ec;
    const auto size = std::filesystem::file_size(impl_->path.string() + "-wal", ec);
    result.walBytes = ec ? 0 : size;
    return result;
}

} // namespace vc::objects
