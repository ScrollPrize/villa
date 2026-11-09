#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "internal/Database.hpp"
#include "internal/Versions.hpp"
#include <algorithm>
#include <map>
#include <limits>
#include <tuple>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

std::string projectViewUuid(const std::string& author)
{
    return util::nameUuid(std::string(kProjectViewKind) + "/" + author);
}

std::optional<int64_t> currentRevision(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                       const std::string& uuid, std::optional<ObjectRecord>* record = nullptr)
{
    std::optional<int64_t> version;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        version = currentVersionOf(reader, uuid);
        if (record) {
            const auto head = reader.currentObject(uuid);
            *record = head ? std::optional(head->record) : std::nullopt;
        }
    });
    return version;
}

// Write `record` over the object's current version `version` (created when
// it is not in the store; restored first when `restore`), in one checked request.
void writeRecord(ObjectStoreSession& session, const AttachmentIdentity& attachment, const ObjectRecord& record,
                 const std::optional<int64_t>& version, bool present, bool restore = false)
{
    auto request = session.newRequest(attachment);
    request.expectedObjects[util::canonicalUuid(record.uuid)] = version;
    if (restore) request.operations.push_back(RestoreObjectOp{record.uuid, std::nullopt});
    if (present) request.operations.push_back(UpdateObjectOp{objectRecordJson(record)});
    else request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
    session.submit(attachment, request);
}
} // namespace

bool isProjectResourceKind(std::string_view kind)
{
    return kind == kVolumeResource || kind == kNormalGridResource || kind == kLasagnaDatasetResource;
}

void createProject(const std::filesystem::path& path, const std::string& name, const Json& meta,
                   const OpenOptions& options)
{
    if (std::filesystem::exists(path)) throw std::runtime_error("object store: " + path.string() + " already exists");
    auto store = ObjectStore::create(path, options);
    ObjectRecord record;
    record.uuid = util::newUuid();
    auto txn = store.write(VersionSelection{{record.uuid, std::nullopt}});
    record.kind = kProjectKind;
    record.name = name;
    record.meta = meta.is_object() ? meta.dump() : "{}";
    store.upsertObject(txn, record);
    txn.commit();
}

void copyProject(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                 const std::filesystem::path& destination, const std::string& name, const OpenOptions& options)
{
    namespace fs = std::filesystem;
    if (fs::exists(destination)) throw std::runtime_error("object store: " + destination.string() + " already exists");
    const auto source = session.path(attachment);
    // One consistent snapshot of the database, outside the session's connections.
    {
        sqlite3* from = nullptr;
        sqlite3* to = nullptr;
        const auto close = [&] { sqlite3_close(from); sqlite3_close(to); };
        try {
            from = detail::openConnection(source, SQLITE_OPEN_READONLY, options, true);
            to = detail::openConnection(destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
        } catch (...) {
            close();
            throw;
        }
        auto* backup = sqlite3_backup_init(to, "main", from, "main");
        const int rc = backup ? sqlite3_backup_step(backup, -1) : SQLITE_ERROR;
        const std::string error = sqlite3_errmsg(to);
        if (backup) sqlite3_backup_finish(backup);
        close();
        if (rc != SQLITE_DONE) {
            std::error_code ignored;
            fs::remove(destination, ignored);
            throw std::runtime_error("object store: copying " + source.string() + " failed: " + error);
        }
    }
    try {
        auto* db = detail::openConnection(destination, SQLITE_OPEN_READWRITE, options, false);
        try {
            detail::exec(db, "BEGIN IMMEDIATE;");
            detail::forkStorageIdentity(db);
            // A copy is an independent mutable publication, never an unseal of its source.
            detail::exec(db, "UPDATE store_manifest SET sealed=0;");
            const auto newStore = detail::parseUuid(util::newUuid());
            detail::exec(db, "DROP TRIGGER store_identity_immutable_update; DROP TRIGGER store_identity_immutable_delete;");
            detail::execute(db, "UPDATE store_identity SET uuid=?", newStore);
            detail::execute(db, "UPDATE key_allocators SET namespace_uuid=?", newStore);
            detail::clearTransactionReceipts(db);
            detail::exec(db, "CREATE TRIGGER store_identity_immutable_update BEFORE UPDATE ON store_identity "
                             "BEGIN SELECT RAISE(ABORT,'store identity is immutable'); END;"
                             "CREATE TRIGGER store_identity_immutable_delete BEFORE DELETE ON store_identity "
                             "BEGIN SELECT RAISE(ABORT,'store identity is retained'); END;");
            detail::Statement clock(db, "SELECT commit_sequence FROM store_manifest WHERE singleton=1");
            if (!clock.row() || clock.i64(0) == std::numeric_limits<int64_t>::max())
                throw std::overflow_error("object store: copied store commit sequence exhausted");
            detail::execute(db, "UPDATE store_manifest SET commit_sequence=? WHERE singleton=1", clock.i64(0) + 1);
            detail::exec(db, "UPDATE store_write_state SET dirty=0 WHERE singleton=1;");
            detail::exec(db, "COMMIT;");
        } catch (...) { sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr); sqlite3_close(db); throw; }
        sqlite3_close(db);
        const auto artifacts = artifactBlobDirectory(source).parent_path();
        if (fs::is_directory(artifacts))
            fs::copy(artifacts, fs::path(destination.string() + ".artifacts"), fs::copy_options::recursive);
        ObjectStoreSession copy;
        const auto copied = copy.attach(destination, options);
        setProjectName(copy, copied, name);
    } catch (...) {
        std::error_code ignored;
        fs::remove(destination, ignored);
        fs::remove_all(fs::path(destination.string() + ".artifacts"), ignored);
        throw;
    }
}

std::optional<ObjectRecord> projectRecord(const StoreReader& reader)
{
    ObjectSelection selection;
    selection.kind = kProjectKind;
    auto records = reader.selectObjects(selection);
    if (records.empty()) return std::nullopt;
    if (records.size() == 1) return std::move(records.front());
    // A store holds one; with several, the first added.
    const auto added = [&](const ObjectRecord& record) {
        const auto history = reader.status().historyEnabled ? reader.objectHistory(record.uuid) : std::vector<ObjectRevision>{};
        return history.empty() ? std::nullopt : history.front().publishedAtUs;
    };
    auto first = std::min_element(records.begin(), records.end(), [&](const auto& a, const auto& b) {
        return std::pair(added(a).value_or(0), a.uuid) < std::pair(added(b).value_or(0), b.uuid);
    });
    return std::move(*first);
}

void setProjectName(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& name)
{
    std::optional<ObjectRecord> record;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { record = projectRecord(reader); });
    if (record && record->name == name) return;
    if (!record) {
        record.emplace();
        record->uuid = util::newUuid();
        record->kind = kProjectKind;
    }
    record->name = name;
    std::optional<ObjectRecord> existing;
    const auto version = currentRevision(session, attachment, record->uuid, &existing);
    writeRecord(session, attachment, *record, version, existing.has_value());
}

std::vector<ProjectResource> projectResources(const StoreReader& reader, const std::string& kind,
                                              const std::optional<std::string>& viewer)
{
    if (!isProjectResourceKind(kind)) throw std::invalid_argument("not a project resource kind: " + kind);
    ObjectSelection selection;
    selection.kind = kind;
    auto records = reader.selectObjects(selection);
    // Added order: when the first revision was published (microseconds), or
    // created (seconds) for an imported baseline without its publication time.
    std::map<std::string, std::pair<std::string, int64_t>> added;
    for (const auto& record : records) {
        const auto history = reader.status().historyEnabled ? reader.objectHistory(record.uuid) : std::vector<ObjectRevision>{};
        const auto published = history.empty() ? std::nullopt : history.front().publishedAtUs;
        added[record.uuid] = published ? std::pair(std::string{}, *published) : std::pair(record.createdAt, int64_t(0));
    }
    std::stable_sort(records.begin(), records.end(), [&](const auto& a, const auto& b) {
        return std::tie(added[a.uuid], a.uuid) < std::tie(added[b.uuid], b.uuid);
    });
    std::vector<ProjectResource> resources;
    for (auto& record : records) {
        if (record.storageKind != kDatasetStorage) continue;
        auto meta = Json::parse(record.meta);
        if (!meta.is_object()) meta = Json::object();
        auto owner = meta.value("owner", std::string{});
        if (viewer && !owner.empty() && owner != *viewer) continue;
        meta.erase("owner");
        resources.push_back({std::move(record.uuid), kind, std::move(record.storageRef), std::move(owner),
                             std::move(record.tags), std::move(meta)});
    }
    return resources;
}

std::string projectResourceUuid(const std::string& kind, const std::string& location, const std::string& owner)
{
    return util::nameUuid("vc_project_resource/" + kind + "/" + location + (owner.empty() ? "" : "/owner/" + owner));
}

ObjectRecord projectResourceRecord(const std::string& kind, const std::string& location,
                                   const std::vector<std::string>& tags, const Json& meta, const std::string& owner)
{
    if (!isProjectResourceKind(kind)) throw std::invalid_argument("not a project resource kind: " + kind);
    if (location.empty()) throw std::invalid_argument("a project resource needs a location");
    ObjectRecord record;
    record.uuid = projectResourceUuid(kind, location, owner);
    record.kind = kind;
    record.name = location;
    record.storageKind = kDatasetStorage;
    record.storageRef = location;
    record.tags = tags;
    auto recorded = meta.is_object() ? meta : Json::object();
    if (!owner.empty()) recorded["owner"] = owner;
    record.meta = recorded.dump();
    return record;
}

bool addProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& kind,
                        const std::string& location, const std::vector<std::string>& tags, const Json& meta,
                        const std::string& owner)
{
    const auto record = projectResourceRecord(kind, location, tags, meta, owner);
    std::optional<ObjectRecord> existing;
    const auto revision = currentRevision(session, attachment, record.uuid, &existing);
    if (existing && !existing->retired && existing->tags == record.tags &&
        Json::parse(existing->meta) == Json::parse(record.meta))
        return false;
    writeRecord(session, attachment, record, revision, existing.has_value(), existing && existing->retired);
    return true;
}

void updateProjectResourceMeta(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                               const std::string& uuid, const Json& meta)
{
    std::optional<ObjectRecord> record;
    const auto revision = currentRevision(session, attachment, uuid, &record);
    if (!record || record->retired || !isProjectResourceKind(record->kind))
        throw std::runtime_error("not a project resource: " + uuid);
    auto merged = Json::parse(record->meta);
    const auto before = merged;
    merged.update(meta);
    if (merged == before) return;
    record->meta = merged.dump();
    writeRecord(session, attachment, *record, revision, true);
}

bool removeProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid)
{
    std::optional<ObjectRecord> record;
    const auto revision = currentRevision(session, attachment, uuid, &record);
    if (!record || record->retired || !isProjectResourceKind(record->kind)) return false;
    auto request = session.newRequest(attachment);
    request.expectedObjects[util::canonicalUuid(uuid)] = revision;
    request.operations.push_back(RetireObjectOp{uuid});
    session.submit(attachment, request);
    return true;
}

Json projectView(const StoreReader& reader, const std::string& author)
{
    const auto head = reader.currentObject(projectViewUuid(author));
    if (!head || head->record.retired) return Json::object();
    auto view = Json::parse(head->record.meta);
    return view.is_object() ? view : Json::object();
}

void setProjectView(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& author,
                    const Json& view)
{
    if (author.empty()) throw std::invalid_argument("a project view needs an author");
    ObjectRecord record;
    record.uuid = projectViewUuid(author);
    record.kind = kProjectViewKind;
    record.name = author;
    record.meta = (view.is_object() ? view : Json::object()).dump();
    std::optional<ObjectRecord> existing;
    const auto revision = currentRevision(session, attachment, record.uuid, &existing);
    if (existing && Json::parse(existing->meta) == Json::parse(record.meta)) return;
    writeRecord(session, attachment, record, revision, existing.has_value());
}

} // namespace vc::objects
