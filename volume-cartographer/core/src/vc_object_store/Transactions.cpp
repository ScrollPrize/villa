#include "internal/Database.hpp"

namespace vc::objects {
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

} // namespace vc::objects
