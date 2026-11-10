#include "internal/Versions.hpp"
#include "internal/History.hpp"
#include "internal/Extensions.hpp"
#include "internal/RecordResolver.hpp"

#include <limits>

namespace vc::objects {
using namespace detail;
using Json = nlohmann::json;

void detail::createVersionSchema(sqlite3* db, const std::string& generation, const std::string& reason)
{
    exec(db, R"SQL(
CREATE TABLE history_generations(
  generation BLOB PRIMARY KEY CHECK(length(generation)=16),
  after_sequence INTEGER NOT NULL,
  started_at_us INTEGER NOT NULL,
  reason TEXT NOT NULL
);
CREATE TABLE store_commits(
  seq INTEGER PRIMARY KEY CHECK(seq>0),
  generation BLOB NOT NULL REFERENCES history_generations(generation),
  request_uuid BLOB,
  author TEXT NOT NULL, host TEXT NOT NULL,
  published_at_us INTEGER NOT NULL
);
CREATE TABLE store_changes(
  seq INTEGER NOT NULL REFERENCES store_commits(seq),
  position INTEGER NOT NULL,
  record_kind TEXT NOT NULL CHECK(record_kind IN ('object','relationship')),
  record_uuid BLOB NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  deleted INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(seq,position)
) WITHOUT ROWID;
CREATE TABLE object_versions(
  object_uuid BLOB NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation BLOB NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid BLOB NOT NULL UNIQUE REFERENCES object_revisions(uuid),
  PRIMARY KEY(object_uuid,version)
) WITHOUT ROWID;
CREATE TABLE relationship_versions(
  edge_uuid BLOB NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation BLOB NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid BLOB UNIQUE REFERENCES relationship_revisions(uuid),
  deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted=(revision_uuid IS NULL)),
  PRIMARY KEY(edge_uuid,version)
) WITHOUT ROWID;
)SQL");
    for (const auto* table : {"history_generations", "store_commits", "store_changes", "object_versions",
                              "relationship_versions"}) {
        for (const auto* operation : {"UPDATE", "DELETE"}) {
            const std::string name = std::string(table) + "_" + operation;
            exec(db, ("CREATE TRIGGER " + name + " BEFORE " + operation + " ON " + table +
                " BEGIN SELECT RAISE(ABORT,'numbered history is immutable and retained'); END;").c_str());
        }
    }
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    execute(db, "INSERT INTO history_generations VALUES(?,0,?,?)", parseUuid(generation), int64_t(now), reason);
    createReceiptSchema(db);
}

void detail::createReceiptSchema(sqlite3* db)
{
    exec(db, R"SQL(
CREATE TABLE IF NOT EXISTS transaction_receipts(
 request_uuid BLOB PRIMARY KEY CHECK(length(request_uuid)=16),
 generation BLOB NOT NULL CHECK(length(generation)=16),
 digest TEXT NOT NULL,
 commit_seq INTEGER,
 receipt TEXT NOT NULL CHECK(json_valid(receipt))
);
CREATE TRIGGER IF NOT EXISTS transaction_receipts_UPDATE BEFORE UPDATE ON transaction_receipts
 BEGIN SELECT RAISE(ABORT,'transaction receipts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS transaction_receipts_DELETE BEFORE DELETE ON transaction_receipts
 BEGIN SELECT RAISE(ABORT,'transaction receipts are immutable'); END;
)SQL");
}

std::string detail::activeGeneration(sqlite3* db)
{
    Statement s(db, "SELECT generation FROM store_manifest WHERE singleton=1");
    if (!s.row()) throw std::runtime_error("object store: missing history generation");
    return s.uuid(0);
}

void detail::clearTransactionReceipts(sqlite3* db)
{
    Statement table(db, "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='transaction_receipts'");
    if (!table.row()) return;
    exec(db, "DROP TRIGGER IF EXISTS transaction_receipts_DELETE; DELETE FROM transaction_receipts;");
    createReceiptSchema(db);
}

int64_t detail::commitSequence(sqlite3* db)
{
    Statement s(db, "SELECT commit_sequence FROM store_manifest WHERE singleton=1");
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
    db=objectSource(db,objectUuid);
    if(!db)return {};
    Statement s(db, "SELECT version FROM objects WHERE uuid=? AND state<>2", parseUuid(objectUuid));
    return s.row() ? std::optional(s.i64(0)) : std::nullopt;
}

void detail::attachObjectVersions(sqlite3* db, std::vector<ObjectRecord>& records)
{
    // A selected (live, published) object's head is its newest version: one
    // backward seek in the (object_uuid, version) key instead of a head join.
    Statement s(db, "SELECT version,coalesce(version_generation,(SELECT initial_generation FROM store_manifest)) FROM objects WHERE uuid=?");
    for (auto& record : records) {
        s.reset();
        s.bind(1, parseUuid(record.uuid));
        if (s.row()) {
            record.version = s.i64(0);
            record.versionGeneration = s.uuid(1);
        }
    }
}

void detail::attachRelationshipVersions(sqlite3* db, std::vector<EdgeRecord>& records)
{
    Statement s(db, "SELECT version,coalesce(version_generation,(SELECT initial_generation FROM store_manifest)),state=2 FROM edges WHERE uuid=?");
    for (auto& record : records) {
        s.reset();
        s.bind(1, parseUuid(record.uuid));
        if (s.row() && !s.i64(2)) {
            record.version = s.i64(0);
            record.versionGeneration = s.uuid(1);
        }
    }
}

std::optional<int64_t> detail::relationshipHeadVersion(sqlite3* db, const std::string& edgeUuid)
{
    db=edgeSource(db,edgeUuid);
    if(!db)return {};
    Statement s(db, "SELECT version,state=2 FROM edges WHERE uuid=?",
        parseUuid(edgeUuid));
    if (!s.row() || s.i64(1)) return std::nullopt;
    return s.i64(0);
}

int64_t detail::revisionVersion(sqlite3* db, const std::string& revisionUuid)
{
    Statement s(db, "SELECT version FROM object_versions WHERE revision_uuid=?", parseUuid(revisionUuid));
    return s.row() ? s.i64(0) : 0;
}

CommitPublisher::CommitPublisher(sqlite3* db, int64_t publishedAtUs, bool baseline)
    : db_(db), generation_(activeGeneration(db)), publishedAtUs_(publishedAtUs), baseline_(baseline) {}

void CommitPublisher::begin()
{
    if (sequence_) return;
    const auto current = commitSequence(db_);
    if (current == std::numeric_limits<int64_t>::max()) throw std::overflow_error("object store: commit sequence exhausted");
    sequence_ = writeCommitSequence(db_);
    std::optional<std::string> request;
    std::string author = "unknown", host = "unknown";
    if (const auto pool = activeStatementPools.find(db_); pool != activeStatementPools.end()) {
        request = pool->second->requestUuid;
        if (!pool->second->commitAuthor.empty()) author = pool->second->commitAuthor;
        if (pool->second->actingAuthor) author = *pool->second->actingAuthor;
        if (!pool->second->commitHost.empty()) host = pool->second->commitHost;
    }
    execute(db_, "INSERT INTO store_commits VALUES(?,?,?,?,?,?)", *sequence_, parseUuid(generation_),
        parseOptionalUuid(request), author, host, publishedAtUs_);
}

int64_t CommitPublisher::nextObject(const std::string& objectUuid)
{
    return writeObjectVersion(db_, objectUuid);
}

int64_t CommitPublisher::object(const std::string& objectUuid, const std::string& revisionUuid, bool deleted)
{
    if (baseline_) {
        Statement row(db_, "SELECT version,coalesce(version_generation,(SELECT initial_generation FROM store_manifest)) FROM objects WHERE uuid=?", parseUuid(objectUuid));
        if (!row.row()) throw std::runtime_error("baseline object is absent");
        execute(db_, "INSERT INTO object_versions VALUES(?,?,?,NULL,?)", parseUuid(objectUuid), row.i64(0), row.blob16(1), parseUuid(revisionUuid));
        return row.i64(0);
    }
    begin();
    const auto version = nextObject(objectUuid);
    execute(db_, "INSERT INTO object_versions VALUES(?,?,?,?,?)", parseUuid(objectUuid), version, parseUuid(generation_),
        *sequence_, parseUuid(revisionUuid));
    execute(db_, "INSERT INTO store_changes VALUES(?,(SELECT count(*) FROM store_changes WHERE seq=?),'object',?,?,?)",
        *sequence_, *sequence_, parseUuid(objectUuid), version, int64_t(deleted));
    return version;
}

int64_t CommitPublisher::relationship(const std::string& edgeUuid, const std::optional<std::string>& revisionUuid)
{
    if (baseline_) {
        Statement row(db_, "SELECT version,coalesce(version_generation,(SELECT initial_generation FROM store_manifest)) FROM edges WHERE uuid=?", parseUuid(edgeUuid));
        if (!row.row()) throw std::runtime_error("baseline relationship is absent");
        execute(db_, "INSERT INTO relationship_versions VALUES(?,?,?,NULL,?,?)", parseUuid(edgeUuid), row.i64(0), row.blob16(1), parseOptionalUuid(revisionUuid), int64_t(!revisionUuid));
        return row.i64(0);
    }
    begin();
    const auto version = writeRecordVersion(db_, "edges", edgeUuid);
    execute(db_, "INSERT INTO relationship_versions VALUES(?,?,?,?,?,?)", parseUuid(edgeUuid), version, parseUuid(generation_),
        *sequence_, parseOptionalUuid(revisionUuid), int64_t(!revisionUuid));
    execute(db_, "INSERT INTO store_changes VALUES(?,(SELECT count(*) FROM store_changes WHERE seq=?),'relationship',?,?,?)",
        *sequence_, *sequence_, parseUuid(edgeUuid), version, int64_t(!revisionUuid));
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
    Json references=Json::object();
    for(const auto& [uuid,ref]:r.objectReferences)references[uuid]=Json::parse(versionReferenceText(ref));
    return Json{{"request_uuid", r.requestUuid ? Json(*r.requestUuid) : Json(nullptr)},
                {"store_uuid", r.storeUuid}, {"generation", r.generation},
                {"commit_sequence", r.commitSequence ? Json(*r.commitSequence) : Json(nullptr)},
                {"objects", selectionToJson(r.objects)}, {"relationships", selectionToJson(r.relationships)},
                {"created", r.created}, {"object_references",references}};
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
    if(j.contains("object_references")){
        if(!j.at("object_references").is_object())throw std::invalid_argument("receipt object references must be a map");
        for(const auto& [uuid,value]:j.at("object_references").items()){
            const auto ref=parseVersionReference(value.dump());
            if(!ref)throw std::invalid_argument("invalid receipt object reference");
            r.objectReferences.emplace(canonical(uuid),*ref);
            (void)r.objectReference(uuid);
        }
    }
    return r;
}

std::optional<ObjectVersion> TransactionReceipt::objectReference(const std::string& objectUuid) const
{
    const auto uuid=canonical(objectUuid);
    const auto number=objects.find(uuid);
    if(number==objects.end()||!number->second)return {};
    const auto found=objectReferences.find(uuid);
    if(found==objectReferences.end())
        throw std::runtime_error("transaction receipt lacks complete object identities; reload before editing");
    if(found->second.objectUuid!=uuid||found->second.version!=*number->second)
        throw std::runtime_error("transaction receipt has inconsistent object identity");
    return found->second;
}

void detail::recordCurrentChange(sqlite3* db, const std::array<unsigned char,16>& key, const char* changeKind)
{
    const auto found=activeStatementPools.find(db);
    if(found==activeStatementPools.end() || found->second->immutableInitial) return;
    auto& pool=*found->second;
    auto& state=pool.currentWrite;
    if(!state) state=std::make_shared<CurrentWriteState>();
    else if(state.use_count()!=1) state=std::make_shared<CurrentWriteState>(*state);
    const auto uuid=formatUuid(key);
    const auto expectObject=[&](const std::string& object) {
        if(pool.checkedAction && !pool.checkedObjects.contains(object)) {
            pool.actionConflict=true;
            throw std::runtime_error("object store: missing expected validation endpoint " + object);
        }
    };
    Statement object(db,"SELECT version,state FROM objects WHERE uuid=?",key);
    if(object.row()) {
        expectObject(uuid);
        state->changedObjects.insert(uuid);
        const auto expected=pool.checkedObjects.find(uuid);
        if(object.i64(1)!=2 && (object.i64(0)==1 ||
           (expected!=pool.checkedObjects.end() && !expected->second))) state->created.insert(uuid);
        return;
    }
    Statement edge(db,"SELECT a_uuid,b_uuid,state FROM edges WHERE uuid=?",key);
    if(!edge.row()) throw std::logic_error("object store: changed record is absent: "+uuid);
    state->changedRelationships.insert(uuid);
    if(edge.i64(2)!=2 && std::string_view(changeKind)=="add_edge") state->created.insert(uuid);
    for(const auto column:{0,1}) {
        const auto endpoint=edge.uuid(column);
        // An immutable dependency cannot be versioned by this overlay.
        if(objectSource(db,endpoint)!=db) continue;
        expectObject(endpoint);
        writeObjectVersion(db,endpoint);
        state->changedObjects.insert(endpoint);
    }
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
    }
    if (pool.publishedSequence && historyEnabled(db)) {
        Statement changes(db, "SELECT record_kind,record_uuid,version,deleted FROM store_changes WHERE seq=? ORDER BY position",
            *pool.publishedSequence);
        while (changes.row()) {
            const bool object = changes.text(0) == "object";
            const auto uuid = changes.uuid(1);
            const auto version = changes.i64(2);
            auto& target = object ? receipt.objects : receipt.relationships;
            target[uuid] = changes.i64(3) ? std::nullopt : std::optional(version);
            if (changes.i64(3)) continue;
            // Created: no earlier live version (first, or after a deletion).
            Statement previous(db, object ? "SELECT 0 FROM object_versions v JOIN object_revisions r ON r.uuid=v.revision_uuid "
                                            "WHERE v.object_uuid=? AND v.version=? AND r.deleted=0"
                                          : "SELECT 0 FROM relationship_versions WHERE edge_uuid=? AND version=? AND deleted=0",
                parseUuid(uuid), version - 1);
            if (!previous.row()) receipt.created.push_back(uuid);
        }
    }
    if (!historyEnabled(db) && pool.currentWrite) {
        for(const auto& uuid:pool.currentWrite->changedObjects) receipt.objects[uuid]=headVersion(db,uuid);
        for(const auto& uuid:pool.currentWrite->changedRelationships) receipt.relationships[uuid]=relationshipHeadVersion(db,uuid);
        for(const auto& uuid:pool.currentWrite->created)
            if(headVersion(db,uuid) || relationshipHeadVersion(db,uuid)) receipt.created.push_back(uuid);
    }
    for(const auto& [uuid,version]:receipt.objects)if(version){
        auto* source=objectSource(db,uuid);
        if(!source)throw std::logic_error("receipt object no longer resolves");
        Statement row(source,"SELECT coalesce(version_generation,(SELECT initial_generation FROM store_manifest)) FROM objects WHERE uuid=?",parseUuid(uuid));
        if(!row.row())throw std::logic_error("receipt object no longer resolves");
        receipt.objectReferences.emplace(uuid,ObjectVersion{storeUuidText(source),row.uuid(0),uuid,*version});
    }
    if (pool.requestUuid) {
        execute(db, "INSERT INTO transaction_receipts VALUES(?,?,?,?,?)", parseUuid(*pool.requestUuid),
            parseUuid(receipt.generation), pool.requestDigest.value_or(""), receipt.commitSequence, receiptToJson(receipt).dump());
    }
    return receipt;
}

std::optional<TransactionReceipt> detail::savedReceipt(sqlite3* db, const std::string& requestUuid, std::string* digest)
{
    Statement table(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='transaction_receipts'");
    if (!table.row()) return std::nullopt;
    Statement s(db, "SELECT receipt,digest FROM transaction_receipts WHERE request_uuid=?", parseUuid(requestUuid));
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
    for (const auto& [uuid, ref] : context.expectedObjectReferences) {
        const auto id = canonical(uuid);
        const ObjectVersion expected{canonical(ref.storeUuid), canonical(ref.generation), canonical(ref.objectUuid), ref.version};
        if (id != expected.objectUuid || expected.version <= 0) throw std::invalid_argument("invalid expected object reference");
        const auto current = resolveObject(db, id);
        if (!current || objectVersion(*current) != expected)
            conflicts.push_back({"object", id, ref.version, current ? current->record.version : std::nullopt});
    }
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
    if (!historyEnabled(db)) return;
    for (const auto& [uuid, expected] : context.expectedObjects) {
        const auto id = canonical(uuid);
        expectedRevisions[id] = currentObjectReference(db,id);
    }
}

void detail::prepareCheckedWrite(sqlite3* db, const TransactionContext& context, RevisionSelection& expected)
{
    checkTransactionContext(db, context, expected);
    auto& pool = *activeStatementPools.at(db);
    if (historyEnabled(db)) expectHistory(db, expected);
    else pool.checkedAction = true;
    for (const auto& [uuid, version] : context.expectedObjects)
        pool.checkedObjects.emplace(util::canonicalUuid(uuid), version);
    if (pool.requestUuid) {
        Statement receipts(db, "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='transaction_receipts'");
        if (!receipts.row()) {
            // Operational deduplication storage is not a logical object edit.
            Statement dirty(db, "SELECT dirty FROM store_write_state WHERE singleton=1");
            dirty.row();
            const auto wasDirty = dirty.i64(0);
            createReceiptSchema(db);
            registerStorageTables(db);
            execute(db, "UPDATE store_write_state SET dirty=? WHERE singleton=1", wasDirty);
            pool.startSchemaVersion = pragmaInt(db, "schema_version");
        }
    }
}

// ---- Public reads ----------------------------------------------------------

StoreStatus ObjectStore::status(const Txn& txn) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT generation,commit_sequence,feed_start,history_enabled FROM store_manifest WHERE singleton=1");
    if (!s.row()) throw std::runtime_error("object store: missing history state");
    const auto sources = querySources(db);
    return {storeUuidText(db), s.uuid(0), s.i64(1), s.i64(2), s.i64(3)!=0,
        std::any_of(sources.begin(), sources.end(), hasExtensionTables)};
}

std::optional<TransactionReceipt> ObjectStore::receipt(const Txn& txn, const std::string& requestUuid) const
{
    return savedReceipt(static_cast<sqlite3*>(connection(txn)), requestUuid);
}

std::optional<int64_t> ObjectStore::currentVersion(const Txn& txn, const std::string& objectUuid) const
{
    auto* source=objectSource(static_cast<sqlite3*>(connection(txn)),canonical(objectUuid));
    return source?headVersion(source,canonical(objectUuid)):std::nullopt;
}

std::optional<int64_t> ObjectStore::currentRelationshipVersion(const Txn& txn, const std::string& uuid) const
{
    return relationshipHeadVersion(static_cast<sqlite3*>(connection(txn)), canonical(uuid));
}

std::optional<ObjectRevision> ObjectStore::objectAtVersion(const Txn& txn, const ObjectVersion& ref) const
{
    return std::move(objectsAtVersions(txn, {ref}).front());
}

ObjectVersion ObjectStore::versionReference(const Txn& txn, const ObjectRevision& revision) const
{
    if (revision.version <= 0) throw std::invalid_argument("object history: revision has no numbered version");
    if(!revision.storeUuid.empty()) {
        const auto reference=revision.reference();
        if(!resolveObjectVersion(static_cast<sqlite3*>(connection(txn)),reference))
            throw std::invalid_argument("object history: exact version is unavailable");
        return reference;
    }
    const auto resolved=resolveObjectRevision(static_cast<sqlite3*>(connection(txn)),revision.uuid);
    if(!resolved)throw std::invalid_argument("object history: revision is unavailable");
    return {storeUuidText(resolved->source), resolved->revision.generation, resolved->revision.objectUuid, resolved->revision.version};
}

ObjectVersion ObjectRevision::reference(std::optional<int64_t> selectedVersion,
                                       const std::optional<std::string>& selectedGeneration) const
{
    const auto number=selectedVersion.value_or(version);
    if(number<=0||storeUuid.empty()||generation.empty()||objectUuid.empty())
        throw std::invalid_argument("object reference: missing published identity");
    return {storeUuid,selectedGeneration.value_or(generation),objectUuid,number};
}

ObjectVersion ObjectRecord::reference() const
{
    if(!version||*version<=0||versionStoreUuid.empty()||versionGeneration.empty())
        throw std::invalid_argument("object reference: selected record lacks its published identity");
    return {versionStoreUuid,versionGeneration,uuid,*version};
}

std::vector<SelectedRelationship> ObjectStore::relationshipsAt(const Txn& txn, const std::vector<ObjectVersion>& versions) const
{
    RevisionSelection selection;
    for (const auto& reference : versions) {
        const auto revision = objectAtVersion(txn, reference);
        if (!revision)
            throw std::invalid_argument("object history: version " + std::to_string(reference.version) + " of " +
                                        reference.objectUuid + " is not recorded in this store and generation");
        if (!selection.emplace(revision->objectUuid, versionReferenceText(reference)).second)
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
    if (!state.historyEnabled) {
        page.cursorValid = cursor.storeUuid==state.storeUuid && cursor.generation==state.generation &&
            cursor.sequence==state.commitSequence && cursor.offset==0;
        page.next=head;
        return page;
    }
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
        commit.sequence = commits.i64(0); commit.requestUuid = commits.optUuid(1);
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
            commit.changes.push_back({changes.text(0), changes.uuid(1), changes.i64(2), changes.i64(3) != 0});
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
    // Provisioning commits and store_reference ordering can leave sequence gaps
    // without record changes. A completely consumed feed advances over them.
    if (!page.more) page.next = head;
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
    if (historyEnabled(db)) execute(db, "INSERT INTO history_generations VALUES(?,?,?,?)", parseUuid(generation), commitSequence(db), int64_t(now), reason);
    execute(db, "UPDATE store_manifest SET generation=?", parseUuid(generation));
    exec(db, "UPDATE store_write_state SET dirty=1 WHERE singleton=1;");
    txn.commit();
    return generation;
}

} // namespace vc::objects
