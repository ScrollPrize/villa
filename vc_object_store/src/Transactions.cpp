#include "internal/Database.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "hub/HubRules.hpp"

namespace vc::objects {
using namespace detail;
Txn::Txn(Txn&& other) noexcept : db_(other.db_), handle_(other.handle_), write_(other.write_),
    expectedRevisions_(std::move(other.expectedRevisions_)), committedRevisions_(std::move(other.committedRevisions_)),
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
    handle_ = nullptr;
    auto releaseConnection = [&] {
        activeStatementPools.erase(conn);
        if (write_) db_->impl_->writerMutex.unlock();
        else db_->impl_->checkinReader(conn);
    };
    try {
        if (commit && write_) {
            refreshSourceIdentityIndex(conn);
            publishHistory(conn);
            enforceHubRules(conn, *activeStatementPools.at(conn), db_->impl_->hubRole);
            for (const auto& [uuid, expected] : expectedRevisions_) {
                Statement head(conn, "SELECT revision_uuid FROM object_heads WHERE object_uuid=?",
                               formatUuid(parseUuid(uuid)));
                committedRevisions_[uuid] = head.row() ? std::optional(head.text(0)) : std::nullopt;
            }
            auto expectedObjects = expectedObjectVersions_;
            for (const auto& [uuid, expected] : expectedRevisions_) expectedObjects.emplace(uuid, std::nullopt);
            receipt_ = finishReceipt(conn, *activeStatementPools.at(conn), expectedObjects, expectedRelationshipVersions_);
        }
        exec(conn, commit ? "COMMIT;" : "ROLLBACK;");
    }
    catch (...) {
        sqlite3_exec(conn, "ROLLBACK;", nullptr, nullptr, nullptr);
        committedRevisions_.clear(); receipt_.reset();
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
                               impl_->options.geometryCacheBytes, impl_->options.spatialCacheBytes);
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
    {
        auto& pool = *impl_->statementPools.at(impl_->writer);
        pool.actingAuthor.reset(); pool.requestUuid.reset(); pool.requestDigest.reset();
        pool.publishedSequence.reset();
        pool.commitAuthor = impl_->options.author; pool.commitHost = impl_->hostname;
    }
    try {
        exec(impl_->writer, "BEGIN IMMEDIATE;");
        if (impl_->hubRole != HubRole::None) beginHubWrite(impl_->writer, *impl_->statementPools.at(impl_->writer));
    }
    catch (...) { sqlite3_exec(impl_->writer, "ROLLBACK;", nullptr, nullptr, nullptr); activeStatementPools.erase(impl_->writer); throw; }
    lock.release(); // WriteTxn releases the mutex on commit or rollback.
    return WriteTxn(this, impl_->writer);
}

WriteTxn ObjectStore::write(const RevisionSelection& expected)
{
    auto txn = write();
    expectHistory(static_cast<sqlite3*>(connection(txn)), expected);
    txn.expectedRevisions_ = expected;
    return txn;
}

WriteTxn ObjectStore::write(const TransactionContext& context)
{
    if (context.author && context.author->empty()) throw std::invalid_argument("object store: empty transaction author");
    auto txn = write();
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
    checkTransactionContext(db, context, expected);
    expectHistory(db, expected);
    txn.expectedRevisions_ = expected;
    txn.expectedObjectVersions_ = context.expectedObjects;
    txn.expectedRelationshipVersions_ = context.expectedRelationships;
    return txn;
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
    // Bulk-import callers checkpoint after publication. Maintain planner
    // statistics here, outside their atomic write transaction.
    exec(impl_->writer, "PRAGMA optimize;");
    check(sqlite3_wal_checkpoint_v2(impl_->writer, nullptr, SQLITE_CHECKPOINT_FULL, nullptr, nullptr), impl_->writer, "checkpoint");
}

} // namespace vc::objects
