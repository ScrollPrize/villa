#pragma once
// Numbered history, checked transactions, receipts and the change feed.
//
// Objects and relationships keep their stable UUIDs. Each published state of
// a record also has a version: 1 for its first publication, advancing once
// per transaction that changes the record (deletion included). A public
// historical reference is (store UUID, publication generation, record UUID,
// version). The publication generation is the history lineage the version was
// published in; restoring a backup starts a new active generation for new
// writes without relabelling what is already recorded, so a reference from a
// lost lineage never resolves to a same-numbered version of another one.
//
// Every committed transaction that publishes at least one record version
// advances the store's commit sequence by one and appends its changes to an
// ordered feed. A transaction that changes nothing advances nothing.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "vc_object_store/Progress.hpp"
#include "vc_object_store/objects/Geometry.hpp"
#include <vc_reader/registrations.hpp>

namespace vc::objects {

struct ObjectRecord;
struct EdgeRecord;
struct Anchor;

struct ObjectVersion {
    std::string storeUuid, generation, objectUuid;
    int64_t version = 0;
    bool operator==(const ObjectVersion&) const = default;
};

struct RelationshipVersion {
    std::string storeUuid, generation, relationshipUuid;
    int64_t version = 0;
    bool operator==(const RelationshipVersion&) const = default;
};

// Record UUID -> expected live version; nullopt expects no live record (never
// created, or deleted).
using VersionSelection = std::map<std::string, std::optional<int64_t>>;

struct StoreStatus {
    std::string storeUuid, generation;
    int64_t commitSequence = 0;
    // Changes after this sequence are retained in the feed; a cursor below it
    // cannot be continued and requires revalidating cached state.
    int64_t feedStart = 0;
};

// Writes are attributed to `author` (falling back to the store's open
// author). Expectations are checked when the transaction begins, under the
// writer lock, before any operation runs:
//  - storeUuid/generation, when given, must name this store and its active
//    history generation;
//  - expectedObjects are write preconditions. Every object the transaction
//    changes (including relationship endpoints) must be listed, and creation
//    lists the new UUID as nullopt;
//  - expectedRelationships are optional relationship preconditions (every
//    relationship change also versions both endpoints, which must be listed);
//  - readDependencies are current-state guards on records read but not
//    written. Checking them never creates a version;
//  - expectedCommitSequence conservatively guards a query's complete
//    membership/absence: any intervening commit is a conflict.
struct TransactionContext {
    std::optional<std::string> requestUuid;
    // Digest of the request contents, recorded with its receipt. Supplied by
    // submit(); raw write(context) callers may provide their own.
    std::optional<std::string> requestDigest;
    std::optional<std::string> storeUuid, generation;
    std::optional<std::string> author;
    VersionSelection expectedObjects, expectedRelationships, readDependencies;
    std::optional<int64_t> expectedCommitSequence;
};

struct TransactionReceipt {
    std::optional<std::string> requestUuid;
    std::string storeUuid, generation;
    // Absent for a successful transaction that published nothing.
    std::optional<int64_t> commitSequence;
    // Versions after commit of every expected and every changed record;
    // nullopt for a record that is absent (deleted).
    VersionSelection objects, relationships;
    // Objects/relationships this transaction created, in creation order.
    std::vector<std::string> created;
    // True when submit() returned the receipt saved by an earlier identical
    // request instead of executing it again.
    bool replayed = false;
};

struct ConflictDetail {
    std::string recordKind; // "object" | "relationship" | "store"
    std::string uuid;
    std::optional<int64_t> expected, actual;
};

// A precondition no longer holds. Nothing was written.
class TransactionConflict : public std::runtime_error {
public:
    enum class Reason { Store, Generation, Version, ReadDependency, CommitSequence };
    TransactionConflict(Reason reason, std::vector<ConflictDetail> details, const std::string& message)
        : std::runtime_error(message), reason_(reason), details_(std::move(details)) {}
    Reason reason() const { return reason_; }
    const std::vector<ConflictDetail>& details() const { return details_; }
private:
    Reason reason_;
    std::vector<ConflictDetail> details_;
};

// The request's binary payloads (by SHA-256) are not available, and no
// saved receipt of that exact request answers it: stage them and retry.
class MissingPayloads : public std::runtime_error {
public:
    explicit MissingPayloads(std::vector<std::string> hashes)
        : std::runtime_error("transaction: payloads not available (" + std::to_string(hashes.size()) + ")"),
          hashes_(std::move(hashes)) {}
    const std::vector<std::string>& hashes() const { return hashes_; }
private:
    std::vector<std::string> hashes_;
};

// Staged payload bytes by SHA-256; nullopt when not (or no longer) staged.
using PayloadLookup = std::function<std::optional<std::vector<uint8_t>>(const std::string& sha256)>;

// ---- Change feed ----------------------------------------------------------

struct FeedCursor {
    std::string storeUuid, generation;
    int64_t sequence = 0; // last completely processed commit
    // Changes of commit sequence+1 already delivered by a continuation page.
    int64_t offset = 0;
};

struct ChangeRecord {
    std::string recordKind; // "object" | "relationship"
    std::string uuid;
    int64_t version = 0;
    bool deleted = false;
};

struct FeedCommit {
    int64_t sequence = 0;
    std::optional<std::string> requestUuid;
    std::string author, host;
    int64_t publishedAtUs = 0;
    std::vector<ChangeRecord> changes;
    // False when this page holds only part of the commit (it alone exceeded
    // the page); `next` then continues inside it.
    bool complete = true;
};

struct ChangePage {
    // False: the cursor names another store/generation, a future sequence, or
    // a position before feedStart. Revalidate cached heads and rerun queries;
    // `next` is the current head.
    bool cursorValid = true;
    std::vector<FeedCommit> commits;
    FeedCursor next;
    bool more = false;
};

// ---- Typed commands -------------------------------------------------------
// Serializable operations. Local and hosted execution apply them through the
// same ObjectStore mutation code. Geometry operations name relationships by
// UUID; their expected edge state and observed geometry heads are taken from
// the checked current state rather than supplied as opaque revision UUIDs.

struct CreateObjectOp { std::string recordJson; };       // objectToJson form
struct UpdateObjectOp { std::string recordJson; };
struct SetMetaOp { std::string uuid, key, jsonValue; };
struct RemoveMetaOp { std::string uuid, key; };
struct SetApprovalOp { std::string uuid, state; };
struct RetireObjectOp { std::string uuid; };
struct RestoreObjectOp { std::string uuid; std::optional<ObjectVersion> source; };
struct DeleteObjectOp { std::string uuid; };
struct AddEdgeOp { std::string recordJson; };            // edgeToJson form
struct UpdateEdgeOp { std::string recordJson; };
struct RemoveEdgeOp { std::string uuid; };
struct RetainRelationshipOp { std::string uuid; };
struct KeepRecordedRelationshipOp { std::string uuid; };
// Recheck the mapping used to compute inverse geometry edits under the writer
// lock, before those edits. Includes registrations/route choices added since
// the view was read; unrelated commits do not invalidate the mapping.
struct CheckVolumeMappingOp {
    std::string objectUuid, targetVolume;
    double targetScale = 1;
    vcr::catalog::Affine mapping;
};
struct GeometryRelationshipEdit {
    std::string relationshipUuid;
    std::string action; // retain | update | remove | keep_recorded
    std::string aAnchorJson = "{}", bAnchorJson = "{}"; // update only
};
// keepAffectedRecorded: every relationship the geometry change affects and
// `relationships` does not decide is kept as recorded (its historical pins,
// no claim that it applies to the new geometry). The request must still
// expect both endpoints of each.
struct PutGeometryOp {
    std::string objectUuid;
    GeometryData data;
    std::vector<GeometryRelationshipEdit> relationships;
    bool keepAffectedRecorded = false;
};
struct EditGeometryOp {
    std::string objectUuid;
    std::vector<GeometryArrayEdit> edits;
    std::vector<GeometryRelationshipEdit> relationships;
    bool keepAffectedRecorded = false;
};
struct EditGeometryRectangleOp {
    std::string objectUuid, arrayName;
    int64_t row = 0, column = 0, rows = 0, columns = 0;
    std::vector<uint8_t> bytes;
    std::vector<GeometryRelationshipEdit> relationships;
    bool keepAffectedRecorded = false;
};
// Replace or add surface channels and remove others
// (ObjectStore::updateSurfaceChannels); "points" is not a channel.
struct SetSurfaceChannelsOp {
    std::string objectUuid;
    std::vector<GeometryArray> channels;
    std::vector<std::string> remove;
    std::vector<GeometryRelationshipEdit> relationships;
    bool keepAffectedRecorded = false;
};

// Make the object's current geometry arrays equal `data`
// (objects/GeometryEdits.hpp updateGeometryArrays): sparse deltas, sequences
// keeping their stable point identities, unchanged schema and grid shape.
// `annotations` (array name -> JSON array, one entry per row) replaces a
// sequence's per-point annotations; omitted ones follow stable IDs. Every
// relationship the change affects is decided by `affected`: "retain" (it
// still applies to the new geometry) or "keep_recorded". The request must
// expect both endpoints of each.
struct UpdateGeometryArraysOp {
    std::string objectUuid;
    GeometryData data;
    std::map<std::string, std::string> annotations;
    std::string affected = "keep_recorded";
};

// Publish catalog registrations used by shared queries (see
// ObjectStore::recordCatalogTransforms); expected versions name
// catalogTransformUuid(id) of each.
struct RecordCatalogTransformsOp { std::vector<vcr::catalog::Registration> registrations; };

// Create objects from decoded sources (objects/GeometryIngest.hpp,
// prepareGeometrySource) with their geometry and the native relationships
// their sources declare (fiber branches, point and winding links), in
// `coordinateContext` ({volume_uuid, scale_factor}). Every object is new
// (re-import is refused); the request expects each absent, and expects every
// existing object a declared relationship links to (ingestRequest).
struct IngestGeometryOp {
    struct Item {
        std::string recordJson; // objectToJson form
        GeometryData data;
        std::string sourceJson; // the decoded source document
    };
    std::vector<Item> items;
    std::string coordinateContext = "{}";
};

// Run relationship detectors over stored objects and publish their links with
// the store's link marker, as one commit. `planJson` is the canonical plan
// (object_links/LinkObjects.hpp: the ordered detectors with their parameters
// and the marker's content). This is a *prepared* operation: detection runs
// under a read snapshot outside the writer lock (ObjectStore::prepare), and
// the checked write expects every object the detectors read at the version
// they read, so a commit in between conflicts and nothing is written.
struct LinkObjectsOp { std::string planJson; };
inline constexpr const char* kLinkObjectsOperation = "link_objects"; // its wire name and preparer key

using TransactionOperation = std::variant<
    CreateObjectOp, UpdateObjectOp, SetMetaOp, RemoveMetaOp, SetApprovalOp,
    RetireObjectOp, RestoreObjectOp, DeleteObjectOp,
    AddEdgeOp, UpdateEdgeOp, RemoveEdgeOp, RetainRelationshipOp, KeepRecordedRelationshipOp,
    PutGeometryOp, EditGeometryOp, EditGeometryRectangleOp, RecordCatalogTransformsOp, SetSurfaceChannelsOp,
    UpdateGeometryArraysOp, IngestGeometryOp, CheckVolumeMappingOp, LinkObjectsOp>;

// Wire/command format version, part of every request digest.
inline constexpr int kTransactionCommandVersion = 1;

struct TransactionRequest {
    std::string requestUuid;       // required: identifies retries
    std::string storeUuid;         // required
    std::string generation;        // required: expected active generation
    std::string author;            // required: attribution, not authentication
    VersionSelection expectedObjects, expectedRelationships, readDependencies;
    std::optional<int64_t> expectedCommitSequence;
    std::vector<TransactionOperation> operations;
};

// ---- Prepared operations ---------------------------------------------------
// An operation whose work happens before the writer lock is taken (link
// detection over a read snapshot) and whose publication joins the request's
// checked write. ObjectStore::prepare runs the registered preparer of each
// such operation and returns what the write needs: the versions the
// preparation observed, added to the request's expectations and read
// dependencies at begin, and the publication to run in the operation's place.
// The request itself (and so its digest and receipt) is unchanged.

class ObjectStore;
class WriteTxn;

struct PreparedOperation {
    VersionSelection expectedObjects, readDependencies;
    std::function<void(ObjectStore&, WriteTxn&)> apply;
};
// By operation index in TransactionRequest::operations.
using PreparedOperations = std::map<std::size_t, PreparedOperation>;

struct PrepareOptions {
    util::ProgressCallback progress;          // the preparation's stages (detector progress)
    const std::atomic_bool* cancel = nullptr; // checked between steps; a cancelled preparation throws
    // Where a preparation may keep disposable files of its own (a hosted
    // service's strip cache); empty keeps everything in memory.
    std::filesystem::path scratchDirectory;
    // Whether the request may name files on this host (a standalone writer
    // runs where its author is; a hosted service refuses client paths).
    bool localPaths = true;
};

// The preparer of one operation type, registered by the library that
// implements it (object_links::enableLinkObjects for "link_objects"), so the
// core library stays free of detector dependencies. Called by
// ObjectStore::prepare with the operation to prepare.
using OperationPreparer = std::function<PreparedOperation(ObjectStore&, const TransactionOperation&, const PrepareOptions&)>;
void registerOperationPreparer(const std::string& operationName, OperationPreparer);

} // namespace vc::objects
