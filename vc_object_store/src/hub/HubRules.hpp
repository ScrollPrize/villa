#pragma once
#include "internal/Database.hpp"

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
// Capture the cursors that delimit this write's changes; call after BEGIN.
void beginHubWrite(sqlite3*, StatementPool&);
// Before COMMIT: lease checks (hub and replica) and pending-change bookkeeping
// (replica writes that are not hub applies).
void enforceHubRules(sqlite3*, const StatementPool&, HubRole);
}
