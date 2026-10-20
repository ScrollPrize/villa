#include "internal/Database.hpp"
#include "internal/History.hpp"

namespace vc::objects {
using namespace detail;
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
        activeStatementPools.erase(conn);
        if (write_) db_->impl_->writerMutex.unlock();
        else db_->impl_->checkinReader(conn);
    };
    try {
        if (commit && write_) publishHistory(conn);
        exec(conn, commit ? "COMMIT;" : "ROLLBACK;");
    }
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
    try { exec(impl_->writer, "BEGIN IMMEDIATE;"); }
    catch (...) { activeStatementPools.erase(impl_->writer); throw; }
    lock.release(); // WriteTxn releases the mutex on commit or rollback.
    return WriteTxn(this, impl_->writer);
}

WriteTxn ObjectStore::write(const RevisionSelection& expected)
{
    auto txn = write();
    expectHistory(static_cast<sqlite3*>(connection(txn)), expected);
    return txn;
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
