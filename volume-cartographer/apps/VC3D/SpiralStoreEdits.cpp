#include "SpiralStoreEdits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>

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
}

std::vector<std::string> saveBrushPatches(vc::objects::ObjectStoreSession& session,
                                          const vc::objects::AttachmentIdentity& attachment,
                                          const std::string& volume, double previewScale,
                                          const std::vector<BrushPatchSave>& patches)
{
    std::vector<std::string> saved;
    const auto coordinates = vc::objects::volumeCoordinates(volume, previewScale);
    for (const auto& patch : patches) {
        if (patch.uuid.empty()) {
            if (patch.remove || !patch.surface) throw std::invalid_argument("a new brush patch needs a surface");
            const auto uuid = vc::objects::util::newUuid();
            vc::objects::SurfaceProvenance provenance;
            provenance.tool = "vc3d-spiral-brush";
            session.act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                create_quad_in_store(db, txn, uuid, *patch.surface, patch.name, {kBrushPatchTag}, coordinates, provenance);
            });
            saved.push_back(uuid);
            continue;
        }
        const auto uuid = vc::objects::util::canonicalUuid(patch.uuid);
        // Leased (on a hosted replica) and pinned at its current revision;
        // its relationships are kept as recorded.
        auto hold = session.holdLeases(attachment, {uuid});
        vc::objects::RevisionSelection pins;
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto head = db.currentObjectRevision(txn, uuid);
            if (!head) throw std::runtime_error("brush patch " + uuid + " is no longer in the object store");
            pins[uuid] = head->uuid;
        });
        if (patch.remove) {
            session.act(attachment, pins, [&](auto& db, auto& txn) {
                if (const auto object = db.objectByUuid(txn, uuid); object && !object->retired) db.retireObject(txn, uuid);
            });
        } else {
            if (!patch.surface) throw std::invalid_argument("a replaced brush patch needs a surface");
            const auto geometry = quad_surface_geometry(*patch.surface);
            session.editKeepingRelationships(attachment, pins, {}, [&](auto& db, auto& txn, const auto& keep) {
                const auto current = db.currentGeometry(txn, uuid);
                db.putGeometry(txn, uuid, geometry, current ? std::optional(current->uuid) : std::nullopt,
                               keep(db.affectedGeometryEdges(txn, uuid, geometry)));
            });
        }
        saved.push_back(uuid);
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

std::vector<std::string> savePointCollections(vc::objects::ObjectStoreSession& session,
                                              const vc::objects::AttachmentIdentity& attachment,
                                              const std::string& volume,
                                              const std::vector<EditablePclDraft>& drafts)
{
    std::vector<vc::objects::AttachedCollection> changed;
    std::vector<std::string> removed, saved;
    std::vector<std::pair<std::size_t, std::string>> replacements;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        const auto object = pointCollectionObject(drafts[i]);
        if (drafts[i].deleted) {
            if (object) removed.push_back(*object);
            continue;
        }
        if (object) replacements.emplace_back(i, *object);
        else changed.push_back(storedPointCollection(drafts[i], vc::objects::util::newUuid(), {}));
    }
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& [index, uuid] : replacements) {
            const auto object = db.objectByUuid(txn, uuid);
            if (!object) throw std::runtime_error("point collection " + uuid + " is no longer in the object store");
            const auto meta = nlohmann::json::parse(object->meta.empty() ? std::string("{}") : object->meta);
            changed.push_back(storedPointCollection(drafts[index], uuid, meta.value("source", nlohmann::json::object())));
        }
    });
    vc::objects::writeVolumeCollections(session, attachment, volume, changed, removed);
    for (const auto& collection : changed) saved.push_back(collection.uuid);
    saved.insert(saved.end(), removed.begin(), removed.end());
    return saved;
}

} // namespace vc3d::spiral
