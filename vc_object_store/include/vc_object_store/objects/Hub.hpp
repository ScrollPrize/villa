#pragma once

// Hosted stores: one authoritative hub file, on a network filesystem (sshfs,
// NFS, SMB) or addressed directly as sftp://[user@]host[:port]/path (ssh://
// is accepted too), and full local replicas that every application opens normally.
// Only replicas are edited; changing the record or geometry of an object that
// exists on the hub requires this replica to hold its lease. Relationships are
// shared records protected by version checks, like every pushed record. Sync pulls hub changes and pushes local ones
// with per-record compare-and-set. Stores without hub tables are unaffected.

#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Progress.hpp"
#include <functional>
#include <stdexcept>

namespace vc::objects {

struct HubLease {
    std::string objectUuid;
    std::string holder;  // replica UUID
    std::string author, host;
    int64_t acquiredUs = 0, expiresUs = 0;  // UTC Unix microseconds
    bool operator==(const HubLease&) const = default;
};

// A record changed both here and on the hub. Resolve with resolveHubConflict.
struct HubConflict {
    std::string uuid;
    int64_t hubVersion = 0;
    std::string hubRecord, localRecord;  // canonical sync JSON, "null" when absent
};

// A hub position names one immutable hub state: the hub's journal sequence
// after a hub write. Every replica and the hub agree on it; read it with
// readHubPosition.
struct HubSyncReport {
    int64_t pulled = 0, pushed = 0;  // records installed here / published to the hub
    std::vector<HubConflict> conflicts;
    // The hub state a pull installed (0 without a pull).
    int64_t pulledPosition = 0;
    // The hub state holding this push: the position of its publication, or,
    // when it had nothing to publish under an expected head, that head (N=H).
    // 0 when nothing was published.
    int64_t publishedPosition = 0;
    // A push with an expected head found the hub elsewhere and published
    // nothing; hubHead is where the hub was.
    bool headMismatch = false;
    int64_t hubHead = 0;
};

struct HubStatus {
    std::filesystem::path hub;
    std::string hubReplica;  // the hub's own identity: positions belong to it
    int64_t position = 0;    // the hub state this replica last pulled
    std::string replica;  // this replica's UUID, the lease holder identity
    std::vector<std::string> pending;  // records changed here and not yet pushed
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
// One-time opt-in for an SSH-addressed hub, with all clients stopped. Creates
// a permanent guard against old dot-file clients; subsequent access must use
// SSH process locks. Needs remote flock, timeout and external sftp-server.
void enableHubProcessLocking(const std::filesystem::path& hub);
// Copy the hub into a new local replica (refuses an existing file). The hub
// is locked for the SSH copy; mounted copies verify an optimistic snapshot.
// Progress counts copied bytes.
void cloneHub(const std::filesystem::path& hub, const std::filesystem::path& replica, const OpenOptions& options,
              const util::ProgressCallback& progress = {});
// Drop the replica's hub tables, leaving an ordinary standalone store.
void detachHub(const std::filesystem::path& replica, const std::string& author);
// Legacy locks only: seconds since acquisition, or nullopt when unlocked.
// Process-lock hubs refuse age/removal: their permanent guard must remain.
std::optional<int64_t> hubLockAge(const std::filesystem::path& hub);
// Remove a stale lock left by a crashed client; only once no client is syncing.
// The next open rolls back any interrupted write from its journal.
void removeHubLock(const std::filesystem::path& hub);

// A hub address as replicas record it: an sftp:// or ssh:// address in its
// canonical sftp:// form, a path made absolute.
std::string hubAddress(const std::filesystem::path& hub);

// Install hub changes. Records with unpushed local changes become conflicts.
HubSyncReport hubPull(ObjectStore& replica);
// Publish local changes atomically. Refuses while conflicts are unresolved;
// throws if a changed hub object is not leased by this replica. With
// expectedHead, the hub write checks that the hub is still at that position
// and otherwise publishes nothing (headMismatch), even for unrelated hub
// commits; with nothing to publish it validates the head the same way. A
// retry whose earlier publication on top of expectedHead succeeded but whose
// response was lost reports that publication's position again.
HubSyncReport hubPush(ObjectStore& replica, std::optional<int64_t> expectedHead = std::nullopt);
// Pull, then push.
HubSyncReport hubSync(ObjectStore& replica);
// Discard every unpushed relationship change and relationship conflict: each
// becomes the hub's current record again (relationships the hub never had are
// removed). That record may be bound to endpoint geometry newer than this
// replica's, so the revert is a pull: everything the hub changed since the
// last one is installed with it (nothing is pulled when there is nothing to
// revert). Object changes stay pending. For a replica whose relationship
// writes are recomputable (a service's detector links), before it recomputes
// them. Returns the number of relationships reverted.
int64_t hubRevert(ObjectStore& replica);

// All-or-nothing. Refuses objects another replica holds and objects this
// replica has not pulled the latest hub version of (pull, then retry).
// UUIDs not yet on the hub may be claimed in advance.
std::vector<HubLease> acquireLeases(ObjectStore& replica, const std::vector<std::string>& objects,
                                    int64_t seconds = kDefaultLeaseSeconds);
std::vector<HubLease> renewLeases(ObjectStore& replica, const std::vector<std::string>& objects,
                                  int64_t seconds = kDefaultLeaseSeconds);
// Refuses objects with unpushed changes. Leases of other holders are untouched.
void releaseLeases(ObjectStore& replica, const std::vector<std::string>& objects);

// The authors this replica has written as since it was cloned, pulls
// included (a pull is journaled under the pulling session's author), sorted.
// Who a replica belongs to: a service refuses one that others write.
std::vector<std::string> hubWriters(ObjectStore& replica);

// Read the hub's state at a position through the hub's own history: `read`
// gets the hub, a read transaction and the hub's revision of every object
// present at that position, retired ones included. On a plain store that
// records positions (enableStorePositions), `read` gets the store itself. Records, tags, deletions
// and relationship membership/approval come from the hub; geometry revisions
// are shared and read from this replica, so a reader checks the objects whose
// geometry it reads with requireHubGeometry. Throws for a position that is
// not a hub state. A remote hub stays locked while `read` runs; it must not
// start another hub operation.
void readHubPosition(ObjectStore& replica, int64_t position,
                     const std::function<void(ObjectStore& hub, const ReadTxn&, const RevisionSelection&)>& read);
// Inside readHubPosition's `read`: throws unless this replica holds the
// geometry of every one of `objects` (hub revisions) at `position` (pull to
// the position, or publish it, first).
void requireHubGeometry(ObjectStore& replica, const ReadTxn& hubTxn, int64_t position,
                        const RevisionSelection& objects);

// The hub's journal entries after one position up to another: which
// records the hub states between them changed, by whom (a plain store's own
// entries, for a store that records positions).
std::vector<JournalEntry> hubJournal(ObjectStore& replica, int64_t after, int64_t upTo);

// Positions of a plain store, without a hub: processes on one machine share
// the store file, and a position names one of its states, read through its
// own history as readHubPosition reads a hub's. Start recording them (a
// write; nothing happens when the store already does): from then on every
// write records its position, whichever connection or process makes it.
// Refuses a hub or replica. Returns the store's head.
int64_t enableStorePositions(ObjectStore& store);
struct StorePositions {
    std::string store;     // the store's own identity: positions belong to it
    int64_t position = 0;  // its head now
};
// nullopt unless a plain store that records positions.
std::optional<StorePositions> storePositions(ObjectStore& store);

// A store's link marker: a client's note of what its last link refresh
// covered (a JSON object the client defines), so a later one can tell
// whether the links are still current. Not journaled: writing one adds no
// journal entry and no position, no hub sync carries it, and hubRevert
// leaves it. It belongs to the replica identity that wrote it, so a copy of
// the store (a clone, forkReplica) reads none. One per store; a write
// replaces it.
std::optional<std::string> linkMarker(ObjectStore& store);
void setLinkMarker(ObjectStore& store, const std::string& json);

// Local state only; no hub contact.
std::optional<HubStatus> hubStatus(ObjectStore& store);
// keep = "hub": install the hub record here. keep = "local": push this
// replica's record over the conflicting hub version (requires the lease).
HubSyncReport resolveHubConflict(ObjectStore& replica, const std::string& uuid, const std::string& keep);

} // namespace vc::objects
