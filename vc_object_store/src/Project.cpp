#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <map>
#include <tuple>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

std::string projectViewUuid(const std::string& author)
{
    return util::nameUuid(std::string(kProjectViewKind) + "/" + author);
}

std::optional<std::string> currentRevision(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                           const std::string& uuid, std::optional<ObjectRecord>* record = nullptr)
{
    std::optional<std::string> revision;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        if (const auto head = store.currentObjectRevision(txn, uuid)) revision = head->uuid;
        if (record) *record = store.objectByUuid(txn, uuid);
    });
    return revision;
}

// Write `record` over the object's current revision (or as a new object), leased on a hosted replica.
void writeRecord(ObjectStoreSession& session, const AttachmentIdentity& attachment, const ObjectRecord& record,
                 const std::optional<std::string>& revision, bool restore = false)
{
    const auto lease = revision ? session.holdLeases(attachment, {record.uuid}) : LeaseHold{};
    session.act(attachment, {{record.uuid, revision}}, [&](ObjectStore& store, WriteTxn& txn) {
        if (restore) store.restoreObject(txn, record.uuid);
        store.upsertObject(txn, record);
    });
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
    auto txn = store.write({{record.uuid, std::nullopt}});
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
        if (sqlite3_open_v2(reinterpret_cast<const char*>(source.u8string().c_str()), &from, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK ||
            sqlite3_open_v2(reinterpret_cast<const char*>(destination.u8string().c_str()), &to, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
            const std::string error = from ? sqlite3_errmsg(to ? to : from) : "out of memory";
            close();
            throw std::runtime_error("object store: cannot copy " + source.string() + ": " + error);
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
            detail::exec(db, "DROP TRIGGER store_identity_immutable_update; DROP TRIGGER store_identity_immutable_delete;");
            detail::execute(db, "UPDATE store_identity SET uuid=?", detail::parseUuid(util::newUuid()));
            detail::exec(db, "CREATE TRIGGER store_identity_immutable_update BEFORE UPDATE ON store_identity "
                             "BEGIN SELECT RAISE(ABORT,'store identity is immutable'); END;"
                             "CREATE TRIGGER store_identity_immutable_delete BEFORE DELETE ON store_identity "
                             "BEGIN SELECT RAISE(ABORT,'store identity is retained'); END;");
            detail::execute(db, "UPDATE sync_replica SET uuid=?", util::newUuid());
            for (const auto* table : {"hub_upstream", "hub_versions", "hub_pending", "hub_inflight", "hub_conflicts", "hub_leases"})
                detail::exec(db, (std::string("DROP TABLE IF EXISTS ") + table + ";").c_str());
            detail::exec(db, "COMMIT;");
        } catch (...) { sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr); sqlite3_close(db); throw; }
        sqlite3_close(db);
        // The artifacts of a hosted replica are beside its hub.
        const auto artifacts = session.artifactDirectory(attachment, util::newUuid(), false).parent_path();
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

std::optional<ObjectRecord> projectRecord(const ObjectStore& store, const Txn& txn)
{
    auto records = store.objectsByKind(txn, kProjectKind);
    if (records.empty()) return std::nullopt;
    return std::move(records.front());
}

void setProjectName(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& name)
{
    std::optional<ObjectRecord> record;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) { record = projectRecord(store, txn); });
    if (record && record->name == name) return;
    if (!record) {
        record.emplace();
        record->uuid = util::newUuid();
        record->kind = kProjectKind;
    }
    record->name = name;
    writeRecord(session, attachment, *record, currentRevision(session, attachment, record->uuid));
}

std::vector<ProjectResource> projectResources(const ObjectStore& store, const Txn& txn, const std::string& kind,
                                              const std::optional<std::string>& viewer)
{
    if (!isProjectResourceKind(kind)) throw std::invalid_argument("not a project resource kind: " + kind);
    auto records = store.objectsByKind(txn, kind);
    // Added order: when the first revision was published (microseconds), or
    // created (seconds) for a hub baseline without its publication time.
    std::map<std::string, std::pair<std::string, int64_t>> added;
    for (const auto& record : records) {
        const auto history = store.objectHistory(txn, record.uuid);
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
    writeRecord(session, attachment, record, revision, existing && existing->retired);
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
    writeRecord(session, attachment, *record, revision);
}

bool removeProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid)
{
    std::optional<ObjectRecord> record;
    const auto revision = currentRevision(session, attachment, uuid, &record);
    if (!record || record->retired || !isProjectResourceKind(record->kind)) return false;
    const auto lease = session.holdLeases(attachment, {uuid});
    session.act(attachment, {{uuid, revision}}, [&](ObjectStore& store, WriteTxn& txn) { store.retireObject(txn, uuid); });
    return true;
}

Json projectView(const ObjectStore& store, const Txn& txn, const std::string& author)
{
    const auto record = store.objectByUuid(txn, projectViewUuid(author));
    if (!record || record->retired) return Json::object();
    auto view = Json::parse(record->meta);
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
    writeRecord(session, attachment, record, revision);
}

} // namespace vc::objects
