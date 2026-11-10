#include "SpiralStoreEdits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>
#include <vc_object_store/objects/Transactions.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>

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

// The refusal of a commit that found `uuid` moved on: its current version is
// read again for the message's "theirs".
StoreObjectChanged refused(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                           const std::string& message, const std::string& uuid)
{
    vc::objects::ObjectVersion current;
    try {
        session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            if (const auto head = reader.currentObject(uuid); head && !head->record.retired) current = head->reference();
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
            saved.push_back({uuid, receipt.objectReference(uuid).value(), receipt.requestUuid.value_or("")});
            continue;
        }
        const auto uuid = vc::objects::util::canonicalUuid(patch.uuid);
        // Guard the complete loaded identity atomically, including its owner.
        auto request = session.newRequest(attachment);
        if (patch.reference.objectUuid != uuid)
            throw refused(session, attachment, changedSince("brush patch", uuid), uuid);
        vc::objects::expectObject(request, patch.reference);
        session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            vc::objects::expectRelationshipEndpoints(reader, request);
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
        saved.push_back({uuid, patch.remove ? vc::objects::ObjectVersion{} : receipt.objectReference(uuid).value(), receipt.requestUuid.value_or("")});
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
    for (const auto& point : draft.points) {
        const QJsonArray position = point.sourcePayload.value(QStringLiteral("p")).toArray();
        if (position.size() != 3) throw std::invalid_argument("a point-collection point has no position");
        stored.points.ids.push_back(point.sourceId.value_or(-1));
        stored.points.points.push_back({position[0].toDouble(), position[1].toDouble(), position[2].toDouble()});
        auto annotation = nlohmann::json::object();
        for (auto it = point.sourcePayload.begin(); it != point.sourcePayload.end(); ++it)
            if (it.key() != QStringLiteral("p")) annotation[it.key().toStdString()] = toJson(it.value());
        stored.annotations.push_back(std::move(annotation));
    }
    return stored;
}

void setPointCollectionObject(EditablePclDraft& draft, const vc::objects::ObjectVersion& reference)
{
    QJsonObject metadata = draft.sourceCollection.value(QStringLiteral("metadata")).toObject();
    metadata[QStringLiteral("object_uuid")] = QString::fromStdString(reference.objectUuid);
    metadata[QLatin1String(kLocalObjectReferenceKey)] = QJsonDocument::fromJson(
        QByteArray::fromStdString(vc::objects::objectVersionsRequestToJson({reference}))).object()["versions"].toArray().first();
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
    std::map<std::string, vc::objects::ObjectVersion> expected;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        const auto object = pointCollectionObject(drafts[i]);
        if (object) saved[i].uuid = *object;
        if (object) {
            const auto value = drafts[i].sourceCollection.value(QStringLiteral("metadata")).toObject()
                                    .value(QLatin1String(kLocalObjectReferenceKey));
            if (!value.isObject())
                throw refused(session, attachment, changedSince("point collection", *object), *object);
            expected[*object] = vc::objects::objectVersionsRequestFromJson(
                QJsonDocument(QJsonObject{{"versions", QJsonArray{value}}}).toJson(QJsonDocument::Compact).toStdString()).at(0);
            if (expected[*object].objectUuid != *object)
                throw std::invalid_argument("point collection reference names another object");
        }
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
    std::vector<vc::objects::ObjectVersion> pins;
    for (const auto& [uuid, reference] : expected) pins.push_back(reference);
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
        for (const auto& [index, uuid] : replacements) {
            const auto object = reader.selectedObject(uuid, {}, {}, expected.at(uuid));
            if (!object || object->record.retired)
                throw std::runtime_error("point collection " + uuid + " is no longer in the object store");
            const auto meta = nlohmann::json::parse(object->record.meta.empty() ? std::string("{}") : object->record.meta);
            changed.push_back(storedPointCollection(drafts[index], uuid, meta.value("source", nlohmann::json::object())));
        }
        for (auto& collection : changed)
            collection.points.ids = reader.assignGeometryPointIds(collection.uuid, "points", collection.points.ids);
    });
    vc::objects::TransactionReceipt receipt;
    try {
        receipt = vc::objects::writeVolumeCollections(session, attachment, vc::objects::VolumeVoxels{volume}, changed,
                                                      removed, {}, pins);
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
        if (!drafts[i].deleted) saved[i].reference = receipt.objectReference(saved[i].uuid).value();
    }
    return saved;
}

} // namespace vc3d::spiral
