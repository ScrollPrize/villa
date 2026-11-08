#include "SpiralStoreEdits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>
#include <vc_object_store/objects/Transactions.hpp>

#include <QJsonArray>
#include <QJsonDocument>

#include <algorithm>
#include <stdexcept>

namespace vc3d::spiral {
namespace {
nlohmann::json toJson(const QJsonValue& value)
{
    return nlohmann::json::parse(QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact).toStdString())[0];
}

std::string changedSince(const std::string& what, const std::string& uuid)
{
    return what + " " + uuid + " was changed in the object store after this edit began (another editor saved "
           "it); nothing was overwritten: reload it and redo the edit";
}

// The current version of `uuid`; throws when it is gone (retired or deleted).
std::int64_t headVersion(const vc::objects::StoreReader& reader, const std::string& what, const std::string& uuid)
{
    const auto head = reader.currentObject(uuid);
    if (!head || head->record.retired) throw std::runtime_error(what + " " + uuid + " is no longer in the object store");
    return head->version;
}

// The version a receipt reports for `uuid` (0 when the receipt names none).
std::int64_t receiptVersion(const vc::objects::TransactionReceipt& receipt, const std::string& uuid)
{
    const auto found = receipt.objects.find(uuid);
    return found == receipt.objects.end() || !found->second ? 0 : *found->second;
}

// The refusal of a commit that found `uuid` moved on: its current version is
// read again for the message's "theirs".
StoreObjectChanged refused(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                           const std::string& message, const std::string& uuid)
{
    std::int64_t current = 0;
    try {
        session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            if (const auto head = reader.currentObject(uuid); head && !head->record.retired) current = head->version;
        });
    } catch (const std::exception&) {
    }
    return StoreObjectChanged(message, uuid, current);
}
}

std::vector<SavedBrushPatch> saveBrushPatches(vc::objects::ObjectStoreSession& session,
                                              const vc::objects::AttachmentIdentity& attachment,
                                              const std::string& volume, double previewScale,
                                              const std::vector<BrushPatchSave>& patches)
{
    std::vector<SavedBrushPatch> saved;
    const auto coordinates = vc::objects::volumeCoordinates(volume, previewScale);
    for (const auto& patch : patches) {
        if (patch.uuid.empty()) {
            if (patch.remove || !patch.surface) throw std::invalid_argument("a new brush patch needs a surface");
            const auto uuid = vc::objects::util::newUuid();
            vc::objects::SurfaceProvenance provenance;
            provenance.tool = "vc3d-spiral-brush";
            auto request = session.newRequest(attachment);
            request.expectedObjects = {{uuid, std::nullopt}};
            request.operations = create_quad_operations(uuid, *patch.surface, patch.name, {kBrushPatchTag},
                                                        coordinates, provenance);
            const auto receipt = session.submit(attachment, request);
            saved.push_back({uuid, receiptVersion(receipt, uuid), receipt.requestUuid.value_or("")});
            continue;
        }
        const auto uuid = vc::objects::util::canonicalUuid(patch.uuid);
        // Expected at the version this edit started from; its relationships
        // are kept as recorded. An unknown version (its last save could not
        // read it back) is refused like a changed one: the user saves again
        // knowingly.
        auto request = session.newRequest(attachment);
        session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            const auto head = headVersion(reader, "brush patch", uuid);
            if (head != patch.version)
                throw StoreObjectChanged(
                    "brush patch " + uuid + (patch.version == 0
                        ? std::string(" was saved, but its saved version could not be read back; nothing was "
                                      "overwritten. Save again to replace the stored version with this drawing")
                        : std::string(" was saved by another editor after this drawing's last save; nothing was "
                                      "overwritten. Save again to replace their version with this drawing, or "
                                      "remove the drawing to keep theirs")),
                    uuid, head);
            request.expectedObjects[uuid] = patch.version;
            vc::objects::expectRelationshipEndpoints(reader, request.expectedObjects);
        });
        if (patch.remove) {
            request.operations.push_back(vc::objects::RetireObjectOp{uuid});
        } else {
            if (!patch.surface) throw std::invalid_argument("a replaced brush patch needs a surface");
            request.operations.push_back(vc::objects::PutGeometryOp{uuid, quad_surface_geometry(*patch.surface), {}, true});
        }
        vc::objects::TransactionReceipt receipt;
        try {
            receipt = session.submit(attachment, request);
        } catch (const vc::objects::TransactionConflict& conflict) {
            throw refused(session, attachment, "brush patch " + uuid + " was saved by another editor after this "
                          "drawing's last save; nothing was overwritten. Save again to replace their version with "
                          "this drawing, or remove the drawing to keep theirs (" + conflict.what() + ")", uuid);
        }
        saved.push_back({uuid, patch.remove ? 0 : receiptVersion(receipt, uuid), receipt.requestUuid.value_or("")});
    }
    return saved;
}

std::optional<std::string> pointCollectionObject(const EditablePclDraft& draft)
{
    const QString uuid = draft.sourceCollection.value(QStringLiteral("metadata")).toObject()
                             .value(QStringLiteral("object_uuid")).toString();
    if (uuid.isEmpty() || !vc::objects::util::isUuid(uuid.toStdString())) return std::nullopt;
    return vc::objects::util::canonicalUuid(uuid.toStdString());
}

vc::objects::AttachedCollection storedPointCollection(const EditablePclDraft& draft, const std::string& uuid,
                                                      const nlohmann::json& source)
{
    vc::objects::AttachedCollection stored;
    stored.uuid = uuid;
    stored.name = draft.sourceCollection.value(QStringLiteral("name")).toString().toStdString();
    stored.source = source.is_object() ? source : nlohmann::json::object();
    stored.source.erase("point_attributes");
    const QJsonValue absolute = draft.sourceCollection.value(QStringLiteral("metadata")).toObject()
                                    .value(QStringLiteral("winding_is_absolute"));
    stored.source["winding_is_absolute"] = absolute.isBool() ? absolute.toBool() : false;
    qint64 next = 0;
    for (const auto& point : draft.points)
        if (point.sourceId) next = std::max(next, *point.sourceId + 1);
    for (const auto& point : draft.points) {
        const QJsonArray position = point.sourcePayload.value(QStringLiteral("p")).toArray();
        if (position.size() != 3) throw std::invalid_argument("a point-collection point has no position");
        stored.points.ids.push_back(point.sourceId ? *point.sourceId : next++);
        stored.points.points.push_back({position[0].toDouble(), position[1].toDouble(), position[2].toDouble()});
        auto annotation = nlohmann::json::object();
        for (auto it = point.sourcePayload.begin(); it != point.sourcePayload.end(); ++it)
            if (it.key() != QStringLiteral("p")) annotation[it.key().toStdString()] = toJson(it.value());
        stored.annotations.push_back(std::move(annotation));
    }
    return stored;
}

void setPointCollectionObject(EditablePclDraft& draft, const std::string& uuid, int64_t version)
{
    QJsonObject metadata = draft.sourceCollection.value(QStringLiteral("metadata")).toObject();
    metadata[QStringLiteral("object_uuid")] = QString::fromStdString(uuid);
    metadata[QLatin1String(kLocalObjectVersionKey)] = QJsonValue(qint64(version));
    draft.sourceCollection[QStringLiteral("metadata")] = metadata;
}

std::vector<SavedPointCollection> savePointCollections(vc::objects::ObjectStoreSession& session,
                                                       const vc::objects::AttachmentIdentity& attachment,
                                                       const std::string& volume,
                                                       const std::vector<EditablePclDraft>& drafts)
{
    std::vector<vc::objects::AttachedCollection> changed;
    std::vector<std::string> removed;
    std::vector<SavedPointCollection> saved(drafts.size());
    std::vector<std::pair<std::size_t, std::string>> replacements;
    // Object -> the version its draft was read at (0: unknown, refused below).
    std::map<std::string, int64_t> expected;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        const auto object = pointCollectionObject(drafts[i]);
        if (object) saved[i].uuid = *object;
        if (object)
            expected[*object] = drafts[i].sourceCollection.value(QStringLiteral("metadata")).toObject()
                                    .value(QLatin1String(kLocalObjectVersionKey)).toInteger(0);
        if (drafts[i].deleted) {
            if (object) removed.push_back(*object);
            continue;
        }
        if (object) {
            replacements.emplace_back(i, *object);
        } else {
            saved[i].uuid = vc::objects::util::newUuid();
            changed.push_back(storedPointCollection(drafts[i], saved[i].uuid, {}));
        }
    }
    // The meta.source kept is read at the expected version: the commit is
    // refused unless that is still current.
    vc::objects::VersionSelection pins;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
        for (const auto& [uuid, version] : expected) {
            const auto head = headVersion(reader, "point collection", uuid);
            if (head != version)
                throw StoreObjectChanged(version == 0
                    ? "point collection " + uuid + " was saved, but its saved version could not be read back; "
                      "nothing was overwritten. Save again to replace the stored version"
                    : changedSince("point collection", uuid), uuid, head);
            pins[uuid] = version;
        }
        for (const auto& [index, uuid] : replacements) {
            const auto object = reader.currentObject(uuid);
            if (!object || object->record.retired)
                throw std::runtime_error("point collection " + uuid + " is no longer in the object store");
            const auto meta = nlohmann::json::parse(object->record.meta.empty() ? std::string("{}") : object->record.meta);
            changed.push_back(storedPointCollection(drafts[index], uuid, meta.value("source", nlohmann::json::object())));
        }
    });
    vc::objects::TransactionReceipt receipt;
    try {
        receipt = vc::objects::writeVolumeCollections(session, attachment, vc::objects::VolumeVoxels{volume}, changed,
                                                      removed, pins);
    } catch (const vc::objects::TransactionConflict& conflict) {
        // The first pinned object the conflict names, else the first pinned.
        std::string uuid;
        for (const auto& detail : conflict.details())
            if (expected.contains(detail.uuid)) { uuid = detail.uuid; break; }
        if (uuid.empty() && !expected.empty()) uuid = expected.begin()->first;
        throw refused(session, attachment, changedSince("point collection", uuid) + " (" + conflict.what() + ")", uuid);
    }
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        saved[i].requestUuid = receipt.requestUuid.value_or("");
        if (!drafts[i].deleted) saved[i].version = receiptVersion(receipt, saved[i].uuid);
    }
    return saved;
}

} // namespace vc3d::spiral
