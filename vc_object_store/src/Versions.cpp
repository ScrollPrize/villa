#include "internal/Versions.hpp"
#include "internal/History.hpp"

#include <limits>

namespace vc::objects {
using namespace detail;
using Json = nlohmann::json;

void detail::createVersionSchema(sqlite3* db, const std::string& generation, const std::string& reason)
{
    exec(db, R"SQL(
CREATE TABLE store_history(
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  active_generation TEXT NOT NULL,
  commit_sequence INTEGER NOT NULL DEFAULT 0 CHECK(commit_sequence>=0),
  feed_start INTEGER NOT NULL DEFAULT 0 CHECK(feed_start>=0)
);
CREATE TABLE history_generations(
  generation TEXT PRIMARY KEY CHECK(length(generation)=36),
  after_sequence INTEGER NOT NULL,
  started_at_us INTEGER NOT NULL,
  reason TEXT NOT NULL
);
CREATE TABLE store_commits(
  seq INTEGER PRIMARY KEY CHECK(seq>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  request_uuid TEXT,
  author TEXT NOT NULL, host TEXT NOT NULL,
  published_at_us INTEGER NOT NULL
);
CREATE TABLE store_changes(
  seq INTEGER NOT NULL REFERENCES store_commits(seq),
  position INTEGER NOT NULL,
  record_kind TEXT NOT NULL CHECK(record_kind IN ('object','relationship')),
  record_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  deleted INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(seq,position)
);
CREATE TABLE object_versions(
  object_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid TEXT NOT NULL UNIQUE REFERENCES object_revisions(uuid),
  PRIMARY KEY(object_uuid,version)
);
CREATE TABLE relationship_versions(
  edge_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid TEXT UNIQUE REFERENCES relationship_revisions(uuid),
  deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted=(revision_uuid IS NULL)),
  PRIMARY KEY(edge_uuid,version)
);
CREATE TABLE transaction_receipts(
  request_uuid TEXT PRIMARY KEY CHECK(length(request_uuid)=36),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  digest TEXT NOT NULL,
  commit_seq INTEGER REFERENCES store_commits(seq),
  receipt TEXT NOT NULL CHECK(json_valid(receipt))
);
)SQL");
    for (const auto* table : {"history_generations", "store_commits", "store_changes", "object_versions",
                              "relationship_versions", "transaction_receipts"}) {
        for (const auto* operation : {"UPDATE", "DELETE"}) {
            const std::string name = std::string(table) + "_" + operation;
            exec(db, ("CREATE TRIGGER " + name + " BEFORE " + operation + " ON " + table +
                " BEGIN SELECT RAISE(ABORT,'numbered history is immutable and retained'); END;").c_str());
        }
    }
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    execute(db, "INSERT INTO history_generations VALUES(?,0,?,?)", generation, int64_t(now), reason);
    execute(db, "INSERT INTO store_history(singleton,active_generation) VALUES(1,?)", generation);
}

std::string detail::activeGeneration(sqlite3* db)
{
    Statement s(db, "SELECT active_generation FROM store_history");
    if (!s.row()) throw std::runtime_error("object store: missing history generation");
    return s.text(0);
}

int64_t detail::commitSequence(sqlite3* db)
{
    Statement s(db, "SELECT commit_sequence FROM store_history");
    if (!s.row()) throw std::runtime_error("object store: missing commit sequence");
    return s.i64(0);
}

std::string detail::storeUuidText(sqlite3* db)
{
    Statement s(db, "SELECT uuid FROM store_identity");
    if (!s.row()) throw std::runtime_error("object store: missing identity");
    return formatUuid(s.blob16(0));
}

std::optional<int64_t> detail::headVersion(sqlite3* db, const std::string& objectUuid)
{
    Statement s(db, "SELECT v.version FROM object_heads h JOIN object_versions v ON v.revision_uuid=h.revision_uuid "
                    "WHERE h.object_uuid=?", objectUuid);
    return s.row() ? std::optional(s.i64(0)) : std::nullopt;
}

std::optional<int64_t> detail::relationshipHeadVersion(sqlite3* db, const std::string& edgeUuid)
{
    Statement s(db, "SELECT version,deleted FROM relationship_versions WHERE edge_uuid=? ORDER BY version DESC LIMIT 1",
        edgeUuid);
    if (!s.row() || s.i64(1)) return std::nullopt;
    return s.i64(0);
}

int64_t detail::revisionVersion(sqlite3* db, const std::string& revisionUuid)
{
    Statement s(db, "SELECT version FROM object_versions WHERE revision_uuid=?", revisionUuid);
    return s.row() ? s.i64(0) : 0;
}

CommitPublisher::CommitPublisher(sqlite3* db, int64_t publishedAtUs)
    : db_(db), generation_(activeGeneration(db)), publishedAtUs_(publishedAtUs) {}

void CommitPublisher::begin()
{
    if (sequence_) return;
    const auto current = commitSequence(db_);
    if (current == std::numeric_limits<int64_t>::max()) throw std::overflow_error("object store: commit sequence exhausted");
    sequence_ = current + 1;
    std::optional<std::string> request;
    std::string author = "unknown", host = "unknown";
    if (const auto pool = activeStatementPools.find(db_); pool != activeStatementPools.end()) {
        request = pool->second->requestUuid;
        if (!pool->second->commitAuthor.empty()) author = pool->second->commitAuthor;
        if (pool->second->actingAuthor) author = *pool->second->actingAuthor;
        if (!pool->second->commitHost.empty()) host = pool->second->commitHost;
    }
    execute(db_, "INSERT INTO store_commits VALUES(?,?,?,?,?,?)", *sequence_, generation_, request, author, host,
        publishedAtUs_);
    execute(db_, "UPDATE store_history SET commit_sequence=?", *sequence_);
}

int64_t CommitPublisher::nextObject(const std::string& objectUuid)
{
    Statement s(db_, "SELECT coalesce(max(version),0) FROM object_versions WHERE object_uuid=?", objectUuid);
    s.row();
    return s.i64(0) + 1;
}

int64_t CommitPublisher::object(const std::string& objectUuid, const std::string& revisionUuid, bool deleted)
{
    begin();
    const auto version = nextObject(objectUuid);
    execute(db_, "INSERT INTO object_versions VALUES(?,?,?,?,?)", objectUuid, version, generation_, *sequence_, revisionUuid);
    execute(db_, "INSERT INTO store_changes VALUES(?,(SELECT count(*) FROM store_changes WHERE seq=?),'object',?,?,?)",
        *sequence_, *sequence_, objectUuid, version, int64_t(deleted));
    return version;
}

int64_t CommitPublisher::baseline(const std::string& objectUuid, const std::string& revisionUuid)
{
    const auto version = nextObject(objectUuid);
    execute(db_, "INSERT INTO object_versions(object_uuid,version,generation,revision_uuid) VALUES(?,?,?,?)",
        objectUuid, version, generation_, revisionUuid);
    return version;
}

int64_t CommitPublisher::relationship(const std::string& edgeUuid, const std::optional<std::string>& revisionUuid)
{
    begin();
    Statement next(db_, "SELECT coalesce(max(version),0)+1 FROM relationship_versions WHERE edge_uuid=?", edgeUuid);
    next.row();
    const auto version = next.i64(0);
    execute(db_, "INSERT INTO relationship_versions VALUES(?,?,?,?,?,?)", edgeUuid, version, generation_, *sequence_,
        revisionUuid, int64_t(!revisionUuid));
    execute(db_, "INSERT INTO store_changes VALUES(?,(SELECT count(*) FROM store_changes WHERE seq=?),'relationship',?,?,?)",
        *sequence_, *sequence_, edgeUuid, version, int64_t(!revisionUuid));
    return version;
}

namespace {
Json selectionToJson(const VersionSelection& selection)
{
    auto result = Json::object();
    for (const auto& [uuid, version] : selection) result[uuid] = version ? Json(*version) : Json(nullptr);
    return result;
}

VersionSelection selectionFromJson(const Json& j)
{
    VersionSelection result;
    for (const auto& [uuid, version] : j.items())
        result.emplace(uuid, version.is_null() ? std::nullopt : std::optional(version.get<int64_t>()));
    return result;
}

std::string canonical(const std::string& uuid) { return formatUuid(parseUuid(uuid)); }
}

Json detail::receiptToJson(const TransactionReceipt& r)
{
    return Json{{"request_uuid", r.requestUuid ? Json(*r.requestUuid) : Json(nullptr)},
                {"store_uuid", r.storeUuid}, {"generation", r.generation},
                {"commit_sequence", r.commitSequence ? Json(*r.commitSequence) : Json(nullptr)},
                {"objects", selectionToJson(r.objects)}, {"relationships", selectionToJson(r.relationships)},
                {"created", r.created}};
}

TransactionReceipt detail::receiptFromJson(const Json& j)
{
    TransactionReceipt r;
    if (!j.at("request_uuid").is_null()) r.requestUuid = j.at("request_uuid").get<std::string>();
    r.storeUuid = j.at("store_uuid").get<std::string>();
    r.generation = j.at("generation").get<std::string>();
    if (!j.at("commit_sequence").is_null()) r.commitSequence = j.at("commit_sequence").get<int64_t>();
    r.objects = selectionFromJson(j.at("objects"));
    r.relationships = selectionFromJson(j.at("relationships"));
    r.created = j.at("created").get<std::vector<std::string>>();
    return r;
}

TransactionReceipt detail::finishReceipt(sqlite3* db, const StatementPool& pool, const VersionSelection& expectedObjects,
                                         const VersionSelection& expectedRelationships)
{
    TransactionReceipt receipt;
    receipt.requestUuid = pool.requestUuid;
    receipt.storeUuid = storeUuidText(db);
    receipt.generation = activeGeneration(db);
    for (const auto& [uuid, version] : expectedObjects) receipt.objects[canonical(uuid)] = headVersion(db, canonical(uuid));
    for (const auto& [uuid, version] : expectedRelationships)
        receipt.relationships[canonical(uuid)] = relationshipHeadVersion(db, canonical(uuid));
    if (pool.publishedSequence) {
        receipt.commitSequence = pool.publishedSequence;
        Statement changes(db, "SELECT record_kind,record_uuid,version,deleted FROM store_changes WHERE seq=? ORDER BY position",
            *pool.publishedSequence);
        while (changes.row()) {
            const bool object = changes.text(0) == "object";
            const auto uuid = changes.text(1);
            const auto version = changes.i64(2);
            auto& target = object ? receipt.objects : receipt.relationships;
            target[uuid] = changes.i64(3) ? std::nullopt : std::optional(version);
            if (changes.i64(3)) continue;
            // Created: no earlier live version (first, or after a deletion).
            Statement previous(db, object ? "SELECT 0 FROM object_versions v JOIN object_revisions r ON r.uuid=v.revision_uuid "
                                            "WHERE v.object_uuid=? AND v.version=? AND r.deleted=0"
                                          : "SELECT 0 FROM relationship_versions WHERE edge_uuid=? AND version=? AND deleted=0",
                uuid, version - 1);
            if (!previous.row()) receipt.created.push_back(uuid);
        }
    }
    if (pool.requestUuid) {
        execute(db, "INSERT INTO transaction_receipts VALUES(?,?,?,?,?)", *pool.requestUuid, receipt.generation,
            pool.requestDigest.value_or(""), receipt.commitSequence, receiptToJson(receipt).dump());
    }
    return receipt;
}

std::optional<TransactionReceipt> detail::savedReceipt(sqlite3* db, const std::string& requestUuid, std::string* digest)
{
    Statement s(db, "SELECT receipt,digest FROM transaction_receipts WHERE request_uuid=?", canonical(requestUuid));
    if (!s.row()) return std::nullopt;
    if (digest) *digest = s.text(1);
    return receiptFromJson(Json::parse(s.text(0)));
}

void detail::checkTransactionContext(sqlite3* db, const TransactionContext& context, RevisionSelection& expectedRevisions)
{
    using Reason = TransactionConflict::Reason;
    if (context.storeUuid && canonical(*context.storeUuid) != storeUuidText(db))
        throw TransactionConflict(Reason::Store, {{"store", canonical(*context.storeUuid), std::nullopt, std::nullopt}},
            "object store: request names another store");
    if (context.generation && canonical(*context.generation) != activeGeneration(db))
        throw TransactionConflict(Reason::Generation, {{"store", activeGeneration(db), std::nullopt, std::nullopt}},
            "object store: request expects history generation " + canonical(*context.generation) +
            " but the active generation is " + activeGeneration(db) + "; revalidate before retrying");
    if (context.expectedCommitSequence && *context.expectedCommitSequence != commitSequence(db))
        throw TransactionConflict(Reason::CommitSequence,
            {{"store", storeUuidText(db), context.expectedCommitSequence, commitSequence(db)}},
            "object store: commit sequence changed since the guarded query");
    std::vector<ConflictDetail> conflicts;
    for (const auto& [uuid, expected] : context.expectedObjects) {
        const auto id = canonical(uuid);
        const auto actual = headVersion(db, id);
        if (actual != expected) conflicts.push_back({"object", id, expected, actual});
    }
    for (const auto& [uuid, expected] : context.expectedRelationships) {
        const auto id = canonical(uuid);
        const auto actual = relationshipHeadVersion(db, id);
        if (actual != expected) conflicts.push_back({"relationship", id, expected, actual});
    }
    if (!conflicts.empty())
        throw TransactionConflict(Reason::Version, conflicts,
            "object history: expected version conflict for " + conflicts.front().recordKind + " " + conflicts.front().uuid);
    for (const auto& [uuid, expected] : context.readDependencies) {
        const auto id = canonical(uuid);
        const auto actual = headVersion(db, id);
        if (actual != expected) conflicts.push_back({"object", id, expected, actual});
    }
    if (!conflicts.empty())
        throw TransactionConflict(Reason::ReadDependency, conflicts,
            "object history: read dependency changed: " + conflicts.front().uuid);
    for (const auto& [uuid, expected] : context.expectedObjects) {
        const auto id = canonical(uuid);
        Statement head(db, "SELECT revision_uuid FROM object_heads WHERE object_uuid=?", id);
        expectedRevisions[id] = head.row() ? std::optional(head.text(0)) : std::nullopt;
    }
}

// ---- Public reads ----------------------------------------------------------

StoreStatus ObjectStore::status(const Txn& txn) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT active_generation,commit_sequence,feed_start FROM store_history");
    if (!s.row()) throw std::runtime_error("object store: missing history state");
    return {storeUuidText(db), s.text(0), s.i64(1), s.i64(2)};
}

std::optional<TransactionReceipt> ObjectStore::receipt(const Txn& txn, const std::string& requestUuid) const
{
    return savedReceipt(static_cast<sqlite3*>(connection(txn)), requestUuid);
}

std::optional<int64_t> ObjectStore::currentVersion(const Txn& txn, const std::string& objectUuid) const
{
    return headVersion(static_cast<sqlite3*>(connection(txn)), canonical(objectUuid));
}

std::optional<int64_t> ObjectStore::currentRelationshipVersion(const Txn& txn, const std::string& uuid) const
{
    return relationshipHeadVersion(static_cast<sqlite3*>(connection(txn)), canonical(uuid));
}

std::optional<ObjectRevision> ObjectStore::objectAtVersion(const Txn& txn, const ObjectVersion& ref) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (canonical(ref.storeUuid) != storeUuidText(db)) return std::nullopt;
    Statement s(db, "SELECT revision_uuid FROM object_versions WHERE object_uuid=? AND version=? AND generation=?",
        canonical(ref.objectUuid), ref.version, canonical(ref.generation));
    return s.row() ? objectRevision(txn, s.text(0)) : std::nullopt;
}

ObjectVersion ObjectStore::versionReference(const Txn& txn, const ObjectRevision& revision) const
{
    if (revision.version <= 0) throw std::invalid_argument("object history: revision has no numbered version");
    return {storeUuidText(static_cast<sqlite3*>(connection(txn))), revision.generation, revision.objectUuid, revision.version};
}

std::vector<SelectedRelationship> ObjectStore::relationshipsAt(const Txn& txn, const std::vector<ObjectVersion>& versions) const
{
    RevisionSelection selection;
    for (const auto& reference : versions) {
        const auto revision = objectAtVersion(txn, reference);
        if (!revision)
            throw std::invalid_argument("object history: version " + std::to_string(reference.version) + " of " +
                                        reference.objectUuid + " is not recorded in this store and generation");
        if (!selection.emplace(revision->objectUuid, revision->uuid).second)
            throw std::invalid_argument("object history: object selected twice: " + revision->objectUuid);
    }
    return relationshipsAt(txn, selection);
}

ChangePage ObjectStore::changesSince(const Txn& txn, const FeedCursor& cursor, int64_t maxChanges) const
{
    if (maxChanges <= 0) throw std::invalid_argument("object store: feed page size must be positive");
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto state = status(txn);
    ChangePage page;
    const FeedCursor head{state.storeUuid, state.generation, state.commitSequence, 0};
    // Commits made before a generation change remain in the feed, but a cursor
    // from another lineage cannot be continued.
    if (cursor.storeUuid != state.storeUuid || cursor.generation != state.generation ||
        cursor.sequence < state.feedStart || cursor.sequence > state.commitSequence || cursor.offset < 0 ||
        (cursor.offset > 0 && cursor.sequence == state.commitSequence)) {
        page.cursorValid = false;
        page.next = head;
        return page;
    }
    page.next = cursor;
    int64_t delivered = 0;
    Statement commits(db, "SELECT seq,request_uuid,author,host,published_at_us FROM store_commits WHERE seq>? ORDER BY seq",
        cursor.sequence);
    while (commits.row()) {
        FeedCommit commit;
        commit.sequence = commits.i64(0); commit.requestUuid = commits.optText(1);
        commit.author = commits.text(2); commit.host = commits.text(3); commit.publishedAtUs = commits.i64(4);
        const int64_t skip = commit.sequence == cursor.sequence + 1 ? cursor.offset : 0;
        Statement count(db, "SELECT count(*) FROM store_changes WHERE seq=?", commit.sequence);
        count.row();
        const int64_t remaining = count.i64(0) - skip;
        if (delivered > 0 && delivered + remaining > maxChanges) { page.more = true; break; }
        const int64_t take = std::min(remaining, maxChanges - delivered);
        Statement changes(db, "SELECT record_kind,record_uuid,version,deleted FROM store_changes WHERE seq=? "
                              "ORDER BY position LIMIT ? OFFSET ?", commit.sequence, take, skip);
        while (changes.row())
            commit.changes.push_back({changes.text(0), changes.text(1), changes.i64(2), changes.i64(3) != 0});
        delivered += take;
        commit.complete = take == remaining;
        page.next.sequence = commit.complete ? commit.sequence : commit.sequence - 1;
        page.next.offset = commit.complete ? 0 : skip + take;
        page.commits.push_back(std::move(commit));
        if (!page.commits.back().complete || delivered >= maxChanges) {
            page.more = page.next.sequence < state.commitSequence;
            break;
        }
    }
    return page;
}

std::string ObjectStore::startHistoryGeneration(const std::string& reason)
{
    if (reason.empty()) throw std::invalid_argument("object store: a history generation needs a reason");
    auto txn = write();
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto generation = util::newUuid();
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    execute(db, "INSERT INTO history_generations VALUES(?,?,?,?)", generation, commitSequence(db), int64_t(now), reason);
    execute(db, "UPDATE store_history SET active_generation=?", generation);
    txn.commit();
    return generation;
}

} // namespace vc::objects
