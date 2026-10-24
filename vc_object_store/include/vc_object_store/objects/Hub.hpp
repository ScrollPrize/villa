#pragma once

// Hosted stores: one authoritative hub file on a network filesystem (sshfs,
// NFS, SMB) and full local replicas that every application opens normally.
// Only replicas are edited; changing the record or geometry of an object that
// exists on the hub requires this replica to hold its lease. Relationships are
// shared records protected by version checks, like every pushed record. Sync pulls hub changes and pushes local ones
// with per-record compare-and-set. Stores without hub tables are unaffected.

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Progress.hpp"
#include <stdexcept>

namespace vc::objects {

struct HubLease {
    std::string objectUuid;
    std::string holder;  // replica UUID
    std::string author, host;
    int64_t acquiredUs = 0, expiresUs = 0;  // UTC Unix microseconds
};

// A record changed both here and on the hub. Resolve with resolveHubConflict.
struct HubConflict {
    std::string uuid;
    int64_t hubVersion = 0;
    std::string hubRecord, localRecord;  // canonical sync JSON, "null" when absent
};

struct HubSyncReport {
    int64_t pulled = 0, pushed = 0;  // records installed here / published to the hub
    std::vector<HubConflict> conflicts;
};

struct HubStatus {
    std::filesystem::path hub;
    std::string replica;  // this replica's UUID, the lease holder identity
    int64_t pending = 0;  // records changed here and not yet pushed
    std::vector<HubConflict> conflicts;
    std::vector<HubLease> leases;  // as of the last hub contact
};

inline constexpr int64_t kDefaultLeaseSeconds = 600;

// Lease acquisition refused: objects another replica holds, and objects
// changed on the hub since this replica last pulled them (pull, then retry).
struct HubLeaseRefused : std::runtime_error {
    using std::runtime_error::runtime_error;
    std::vector<std::string> held, stale;
};

// Convert a local store (no other connections) into a hub, then place it on the share.
void initHub(const std::filesystem::path& store, const std::string& author);
// Copy the hub into a new local replica (refuses an existing file). The hub
// stays usable during the copy; progress counts copied database pages.
void cloneHub(const std::filesystem::path& hub, const std::filesystem::path& replica, const OpenOptions& options,
              const util::ProgressCallback& progress = {});
// Drop the replica's hub tables, leaving an ordinary standalone store.
void detachHub(const std::filesystem::path& replica, const std::string& author);

// Install hub changes. Records with unpushed local changes become conflicts.
HubSyncReport hubPull(ObjectStore& replica);
// Publish local changes atomically. Refuses while conflicts are unresolved;
// throws if a changed hub object is not leased by this replica.
HubSyncReport hubPush(ObjectStore& replica);
// Pull, then push.
HubSyncReport hubSync(ObjectStore& replica);

// All-or-nothing. Refuses objects another replica holds and objects this
// replica has not pulled the latest hub version of (pull, then retry).
// UUIDs not yet on the hub may be claimed in advance.
std::vector<HubLease> acquireLeases(ObjectStore& replica, const std::vector<std::string>& objects,
                                    int64_t seconds = kDefaultLeaseSeconds);
std::vector<HubLease> renewLeases(ObjectStore& replica, const std::vector<std::string>& objects,
                                  int64_t seconds = kDefaultLeaseSeconds);
// Refuses objects with unpushed changes. Leases of other holders are untouched.
void releaseLeases(ObjectStore& replica, const std::vector<std::string>& objects);

// Local state only; no hub contact.
std::optional<HubStatus> hubStatus(ObjectStore& store);
// keep = "hub": install the hub record here. keep = "local": push this
// replica's record over the conflicting hub version (requires the lease).
HubSyncReport resolveHubConflict(ObjectStore& replica, const std::string& uuid, const std::string& keep);

} // namespace vc::objects
