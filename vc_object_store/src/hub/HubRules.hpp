#pragma once
#include "internal/Database.hpp"

// Hosted-store invariants shared by every write on a hub or a replica.
// Stores without hub_* tables (HubRole::None) never reach this code.
namespace vc::objects::detail {
// The ordinary schema.
constexpr int kSchemaVersion = 17;
// The schema of a store with a publication index (a hub, the replicas cloned
// from it, or a plain store with positions): the ordinary schema plus
// hub_publications. Binaries from before the index accept only the ordinary
// version, so they cannot write to an indexed store (their writes would
// journal without a publication, leaving every later position unreadable
// until the next indexed write).
constexpr int kIndexedHubSchemaVersion = 18;
HubRole detectHubRole(sqlite3*);
// Whether the store keeps a publication index.
bool hasPublications(sqlite3*);
// Start a plain store's publication index at its current state, if it has
// none; inside a write transaction. Returns the store's head.
int64_t ensurePublications(sqlite3*);
// Drop the publication index, leaving the ordinary schema.
void dropPublications(sqlite3*);
void createHubTables(sqlite3*);
void createReplicaTables(sqlite3*);
int64_t nowMicros();
std::string isoFromMicros(int64_t);
std::string replicaIdentity(sqlite3*);
// Version of a record on the hub: its latest journal sequence, 0 if never journaled.
int64_t hubVersion(sqlite3* hub, const std::string& record);
// The hub version this replica last installed or pushed for a record.
int64_t knownHubVersion(sqlite3* replica, const std::string& record);
// The hub publication a position (journal sequence) names: the last hub write
// at or before it. Throws unless nothing was journaled between that write and
// the position and the position is not beyond the hub's head.
struct HubPublication { int64_t position = 0, previous = 0, publishedAtUs = 0; std::string replica; };
HubPublication hubPublicationAt(sqlite3* hub, int64_t position);
// The publication that journaled a record version (a hubVersion), if recorded.
std::optional<HubPublication> hubPublicationOf(sqlite3* hub, int64_t version);
// The hub write that directly followed a position, if recorded.
std::optional<HubPublication> hubPublicationAfter(sqlite3* hub, int64_t previous);
int64_t hubHead(sqlite3* hub);
// Capture the cursors that delimit this write's changes; call after BEGIN
// (every store: a plain store may keep a publication index).
void beginHubWrite(sqlite3*, StatementPool&);
// Before COMMIT: lease checks (hub and replica), pending-change bookkeeping
// (replica writes that are not hub applies), and the publication of a write
// that journaled on a hub or an indexed plain store.
void enforceHubRules(sqlite3*, const StatementPool&, HubRole);
}
