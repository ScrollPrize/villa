#include "internal/RecordSql.hpp"

namespace vc::objects {
using namespace detail;

std::string ObjectStore::replicaUuid(const Txn& txn) const {
    auto* db=static_cast<sqlite3*>(connection(txn));requireSyncReplica(db);
    Statement row(db, "SELECT uuid FROM sync_replica WHERE singleton=1");
    if (!row.row()) throw std::runtime_error("sync: missing replica identity");
    return formatUuid(row.blob16(0));
}
std::string ObjectStore::forkReplica() {
    {auto txn=read();requireSyncReplica(static_cast<sqlite3*>(connection(txn)));}
    auto txn = write(); auto* db = static_cast<sqlite3*>(connection(txn));
    requireSyncReplica(db);
    const auto id = util::newUuid();
    forkStorageIdentity(db);
    execute(db, "UPDATE sync_replica SET uuid=? WHERE singleton=1", parseUuid(id));
    txn.commit(); return id;
}
} // namespace vc::objects
