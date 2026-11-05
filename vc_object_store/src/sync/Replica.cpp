#include "internal/RecordSql.hpp"

namespace vc::objects {
using namespace detail;

std::string ObjectStore::replicaUuid(const Txn& txn) const {
    Statement row(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM sync_replica WHERE singleton=1");
    if (!row.row()) throw std::runtime_error("sync: missing replica identity");
    return row.text(0);
}
std::string ObjectStore::forkReplica() {
    auto txn = write(); auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = util::newUuid();
    execute(db, "UPDATE sync_replica SET uuid=? WHERE singleton=1", id);
    txn.commit(); return id;
}
} // namespace vc::objects
