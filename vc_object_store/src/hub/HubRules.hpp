#pragma once
#include "internal/Database.hpp"
#include <filesystem>
#include <optional>
#include <set>

// Hosted-store invariants shared by every write on a hub or a replica.
// Stores without hub_* tables (HubRole::None) never reach this code.
namespace vc::objects::detail {
HubRole detectHubRole(sqlite3*);
void createHubTables(sqlite3*);
void createReplicaTables(sqlite3*);
int64_t nowMicros();
std::string isoFromMicros(int64_t);
std::string replicaIdentity(sqlite3*);
// Version of a record on the hub: its latest journal sequence, 0 if never journaled.
int64_t hubVersion(sqlite3* hub, const std::string& record);
// The hub version this replica last installed or pushed for a record.
int64_t knownHubVersion(sqlite3* replica, const std::string& record);
// First column of every row, as a set.
std::set<std::string> uuidColumn(sqlite3* db, const char* sql);
// Lease management of a replica belongs to one owner at a time: a session
// worker holds an exclusive lock file beside the replica (keyed by the
// replica's canonical path) while it lives, and a direct lease call
// (acquire/renew/release) holds the same lock for its whole duration, so a
// worker cannot attach while one runs and none can run while a worker
// manages the replica. Calls made from inside the worker are exempt.
class ReplicaWorkerLock {
public:
    explicit ReplicaWorkerLock(const std::filesystem::path& replica);  // throws when another owner holds it
    ~ReplicaWorkerLock();
    ReplicaWorkerLock(ReplicaWorkerLock&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    ReplicaWorkerLock(const ReplicaWorkerLock&) = delete;
    ReplicaWorkerLock& operator=(const ReplicaWorkerLock&) = delete;
private:
    int fd_ = -1;
};
struct InsideWorkerScope {
    InsideWorkerScope();
    ~InsideWorkerScope();
};
// Suspends the exemption while application code (a trace callback) runs on
// the worker's thread.
struct OutsideWorkerScope {
    OutsideWorkerScope();
    ~OutsideWorkerScope();
private:
    int saved_;
};
bool insideWorker();
// Held by a direct lease call for its duration; empty inside the worker.
std::optional<ReplicaWorkerLock> requireLeaseOwnership(const std::filesystem::path& replica);
// Capture the cursors that delimit this write's changes; call after BEGIN.
void beginHubWrite(sqlite3*, StatementPool&);
// Before COMMIT: lease checks (hub and replica) and pending-change bookkeeping
// (replica writes that are not hub applies).
void enforceHubRules(sqlite3*, const StatementPool&, HubRole);
}
