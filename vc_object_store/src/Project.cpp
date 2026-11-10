#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <map>
#include <tuple>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

std::optional<int64_t> addedAt(const StoreReader& reader, const ObjectRecord& record)
{
    const auto history = reader.objectHistory(record.uuid, true);
    if (!history.empty() && history.front().publishedAtUs) return history.front().publishedAtUs;
    const auto& s = record.createdAt;
    if (s.size() != 20 || s[4] != '-' || s[7] != '-' || s[10] != 'T' ||
        s[13] != ':' || s[16] != ':' || s[19] != 'Z') return {};
    auto number = [&](int offset, int length) {
        unsigned n = 0;
        const auto end = s.data() + offset + length;
        const auto [p, ec] = std::from_chars(s.data() + offset, end, n);
        return ec == std::errc{} && p == end ? n : 10000u;
    };
    using namespace std::chrono;
    const auto day = year(int(number(0,4))) / month(number(5,2)) / std::chrono::day(number(8,2));
    const auto h = number(11,2), m = number(14,2), sec = number(17,2);
    if (!day.ok() || h >= 24 || m >= 60 || sec >= 60) return {};
    return duration_cast<microseconds>((sys_days(day) + hours(h) + minutes(m) + seconds(sec)).time_since_epoch()).count();
}

void sortByAdded(const StoreReader& reader, std::vector<ObjectRecord>& records)
{
    std::map<std::string, int64_t> added;
    for (const auto& record : records)
        added.emplace(record.uuid, addedAt(reader, record).value_or(INT64_MAX));
    std::sort(records.begin(), records.end(), [&](const auto& a, const auto& b) {
        return std::tie(added.at(a.uuid), a.uuid) < std::tie(added.at(b.uuid), b.uuid);
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

// Write `record` over the object's current version `version` (created when
// it is not in the store; restored first when `restore`), in one checked request.
void writeRecord(ObjectStoreSession& session, const AttachmentIdentity& attachment, const ObjectRecord& record,
                 const std::optional<int64_t>& version, const std::optional<ObjectRecord>& existing, bool restore = false)
{
    auto request = session.newRequest(attachment);
    request.expectedObjects[util::canonicalUuid(record.uuid)] = version;
    if (existing) expectObject(request, existing->reference());
    if (restore) request.operations.push_back(RestoreObjectOp{record.uuid, std::nullopt});
    if (existing) request.operations.push_back(UpdateObjectOp{objectRecordJson(record)});
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
    sortByAdded(reader, records);
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
    sortByAdded(reader, records);
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
    const auto legacy = util::nameUuid("vc_project_resource/" + record.kind + "/" + record.storageRef +
        (owner.empty() ? "" : "/owner/" + owner));
    if (const auto existing = reader.currentObject(legacy); existing &&
        existing->record.kind == record.kind && existing->record.storageKind == record.storageKind &&
        existing->record.storageRef == record.storageRef &&
        Json::parse(existing->record.meta).value("owner", std::string{}) == owner)
        record.uuid = legacy;
    return record;
}

bool addProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& kind,
                        const std::string& location, const std::vector<std::string>& tags, const Json& meta,
                        const std::string& owner)
{
    auto record = projectResourceRecord(kind, location, tags, meta, owner);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        record = resolveProjectResourceIdentity(reader, std::move(record));
    });
    std::optional<ObjectRecord> existing;
    const auto revision = currentRevision(session, attachment, record.uuid, &existing);
    if (existing && !existing->retired && existing->tags == record.tags &&
        Json::parse(existing->meta) == Json::parse(record.meta))
        return false;
    writeRecord(session, attachment, record, revision, existing, existing && existing->retired);
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
    writeRecord(session, attachment, *record, revision, record);
}

bool removeProjectResource(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid)
{
    std::optional<ObjectRecord> record;
    const auto revision = currentRevision(session, attachment, uuid, &record);
    if (!record || record->retired || !isProjectResourceKind(record->kind)) return false;
    auto request = session.newRequest(attachment);
    expectObject(request, record->reference());
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
    writeRecord(session, attachment, record, revision, existing);
}

} // namespace vc::objects
