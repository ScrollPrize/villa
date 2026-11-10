#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <algorithm>
#include <tuple>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

void sortByAdded(std::vector<ObjectRecord>& records)
{
    // UTC creation stamps survive overrides. Legacy whole-second stamps sort
    // after precise stamps in the same second, then by UUID.
    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        return std::tuple(a.createdAt.empty(), std::string_view(a.createdAt), std::string_view(a.uuid)) <
               std::tuple(b.createdAt.empty(), std::string_view(b.createdAt), std::string_view(b.uuid));
    });
}

std::string projectViewUuid(const std::string& author)
{
    return util::nameUuid(std::string(kProjectViewKind) + "/" + author);
}

std::optional<int64_t> currentRevision(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                       const std::string& uuid, std::optional<ObjectRecord>* record = nullptr)
{
    std::optional<int64_t> version;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        const auto head = reader.currentObject(uuid);
        version = head ? std::optional(head->version) : currentVersionOf(reader, uuid);
        if (record) *record = head ? std::optional(head->record) : std::nullopt;
    });
    return version;
}

std::string legacyResourceUuid(const ObjectRecord& record, const std::string& owner)
{
    return util::nameUuid("vc_project_resource/" + record.kind + "/" + record.storageRef +
        (owner.empty() ? "" : "/owner/" + owner));
}

std::optional<ObjectRecord> resourceForEdit(const StoreReader& reader, const std::string& uuid)
{
    if (auto head = reader.currentObject(uuid)) return std::move(head->record);
    if (currentVersionOf(reader, uuid)) return std::nullopt;
    // The deterministic key for new resources can also address an exact legacy
    // resource in this API; generic object lookups always require a stored UUID.
    for (const auto* kind : {kVolumeResource, kNormalGridResource, kLasagnaDatasetResource})
        for (const auto& resource : projectResources(reader, kind))
            if (projectResourceUuid(kind, resource.location, resource.owner) == uuid) {
                auto head = reader.currentObject(resource.uuid);
                if (head && head->record.uuid == legacyResourceUuid(head->record, resource.owner))
                    return std::move(head->record);
            }
    return std::nullopt;
}

struct ResourceWrite {
    ObjectRecord record;
    std::optional<ObjectRecord> existing;
    bool unchanged() const {
        return existing && !existing->retired && existing->tags == record.tags &&
            Json::parse(existing->meta) == Json::parse(record.meta);
    }
    bool restore() const { return existing && existing->retired; }
};

ResourceWrite prepareResourceWrite(const StoreReader& reader, ObjectRecord record)
{
    record = resolveProjectResourceIdentity(reader, std::move(record));
    auto existing = reader.currentObject(record.uuid);
    return {std::move(record), existing ? std::optional(std::move(existing->record)) : std::nullopt};
}

// Write `record` over the object's current version `version` (created when
// it is not in the store; restored first when `restore`), in one checked request.
void appendRecordWrite(TransactionRequest& request, const ObjectRecord& record,
                       const std::optional<int64_t>& version, const std::optional<ObjectRecord>& existing,
                       bool restore)
{
    request.expectedObjects[util::canonicalUuid(record.uuid)] = version;
    if (existing) expectObject(request, existing->reference());
    if (restore) request.operations.push_back(RestoreObjectOp{record.uuid, std::nullopt});
    if (existing) request.operations.push_back(UpdateObjectOp{objectRecordJson(record)});
    else request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
}

void writeRecord(ObjectStoreSession& session, const AttachmentIdentity& attachment, const ObjectRecord& record,
                 const std::optional<int64_t>& version, const std::optional<ObjectRecord>& existing, bool restore = false)
{
    auto request = session.newRequest(attachment);
    appendRecordWrite(request, record, version, existing, restore);
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
    copyObjectCollection(session.path(attachment),destination,options);
    try {
        ObjectStoreSession copy;
        const auto copied = copy.attach(destination, options);
        setProjectName(copy, copied, name);
    } catch (...) {
        std::error_code ignored;
        for(const auto* suffix:{"","-owner","-wal","-shm","-journal"})fs::remove(destination.string()+suffix,ignored);
        fs::remove_all(fs::path(destination.string() + ".artifacts"), ignored);
        fs::remove_all(fs::path(destination.string() + ".stores"), ignored);
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
    sortByAdded(records);
    return std::move(records.front());
}

void setProjectName(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& name)
{
    std::optional<ObjectRecord> record;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { record = projectRecord(reader); });
    if (record && record->name == name) return;
    const auto existing = record;
    if (!record) {
        record.emplace();
        record->uuid = util::newUuid();
        record->kind = kProjectKind;
    }
    record->name = name;
    writeRecord(session, attachment, *record, existing ? existing->version : std::nullopt, existing);
}

std::vector<ProjectResource> projectResources(const StoreReader& reader, const std::string& kind,
                                              const std::optional<std::string>& viewer)
{
    if (!isProjectResourceKind(kind)) throw std::invalid_argument("not a project resource kind: " + kind);
    ObjectSelection selection;
    selection.kind = kind;
    auto records = reader.selectObjects(selection);
    sortByAdded(records);
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
    return util::nameUuid("vc_project_resource/v2:" + Json::array({kind, location, owner}).dump());
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
    recorded.erase("owner");
    if (!owner.empty()) recorded["owner"] = owner;
    record.meta = recorded.dump();
    return record;
}

ObjectRecord resolveProjectResourceIdentity(const StoreReader& reader, ObjectRecord record)
{
    const auto owner = Json::parse(record.meta).value("owner", std::string{});
    if (currentVersionOf(reader, record.uuid)) return record;
    const auto legacy = legacyResourceUuid(record, owner);
    if (const auto existing = reader.currentObject(legacy); existing &&
        existing->record.kind == record.kind && existing->record.storageKind == record.storageKind &&
        existing->record.storageRef == record.storageRef &&
        Json::parse(existing->record.meta).value("owner", std::string{}) == owner)
        record.uuid = legacy;
    return record;
}

std::string addProjectResource(ObjectStore& db, ObjectRecord record)
{
    TransactionRequest request;
    std::string uuid;
    {
        auto txn = db.read();
        LocalStoreReader reader(db, txn);
        auto write = prepareResourceWrite(reader, std::move(record));
        uuid = write.record.uuid;
        if (write.unchanged()) return uuid;
        const auto status = reader.status();
        request.requestUuid = util::newUuid();
        request.storeUuid = status.storeUuid;
        request.generation = status.generation;
        request.author = db.author();
        appendRecordWrite(request, write.record, write.existing ? write.existing->version :
                          currentVersionOf(reader, uuid), write.existing, write.restore());
    }
    db.submit(request);
    return uuid;
}

bool addProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& kind,
                        const std::string& location, const std::vector<std::string>& tags, const Json& meta,
                        const std::string& owner)
{
    ResourceWrite write;
    std::optional<int64_t> revision;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        write = prepareResourceWrite(reader, projectResourceRecord(kind, location, tags, meta, owner));
        revision = write.existing ? write.existing->version : currentVersionOf(reader, write.record.uuid);
    });
    if (write.unchanged()) return false;
    writeRecord(session, attachment, write.record, revision, write.existing, write.restore());
    return true;
}

void updateProjectResourceMeta(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                               const std::string& uuid, const Json& meta)
{
    std::optional<ObjectRecord> record;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        record = resourceForEdit(reader, util::canonicalUuid(uuid));
    });
    if (!record || record->retired || !isProjectResourceKind(record->kind))
        throw std::runtime_error("not a project resource: " + uuid);
    auto merged = Json::parse(record->meta);
    const auto before = merged;
    merged.update(meta);
    if (merged.value("owner", std::string{}) != before.value("owner", std::string{}))
        throw std::invalid_argument("project resource owner is part of its identity");
    if (merged == before) return;
    record->meta = merged.dump();
    writeRecord(session, attachment, *record, record->version, record);
}

bool removeProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid)
{
    std::optional<ObjectRecord> record;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        record = resourceForEdit(reader, util::canonicalUuid(uuid));
    });
    if (!record || record->retired || !isProjectResourceKind(record->kind)) return false;
    auto request = session.newRequest(attachment);
    expectObject(request, record->reference());
    request.operations.push_back(RetireObjectOp{record->uuid});
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
    writeRecord(session, attachment, record, revision, existing);
}

} // namespace vc::objects
