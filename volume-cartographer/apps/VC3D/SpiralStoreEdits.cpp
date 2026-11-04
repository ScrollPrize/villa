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

namespace {
std::string changedSince(const std::string& what, const std::string& uuid)
{
    return what + " " + uuid + " was changed in the object store after this edit began (another editor saved "
           "it); nothing was overwritten: reload it and redo the edit";
}

// The revision a save just committed, read after it (revisions are made at
// commit). Empty when it cannot be read (the store was detached meanwhile):
// the save stands, and the next save of the object, naming no revision, is
// refused with the current one, from which it can save again.
std::string committedRevision(vc::objects::ObjectStoreSession& session,
                              const vc::objects::AttachmentIdentity& attachment, const std::string& uuid)
{
    std::string revision;
    try {
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            if (const auto head = db.currentObjectRevision(txn, uuid)) revision = head->uuid;
        });
    } catch (const std::exception&) {
        return {};
    }
    return revision;
}

// The current revision of `uuid`; throws when it is gone.
template <typename Db, typename Txn>
std::string headRevision(const Db& db, const Txn& txn, const std::string& what, const std::string& uuid)
{
    const auto head = db.currentObjectRevision(txn, uuid);
    if (!head) throw std::runtime_error(what + " " + uuid + " is no longer in the object store");
    return head->uuid;
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
            session.act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                create_quad_in_store(db, txn, uuid, *patch.surface, patch.name, {kBrushPatchTag}, coordinates, provenance);
            });
            // Nobody else knows the new object yet: its head is this save.
            saved.push_back({uuid, committedRevision(session, attachment, uuid)});
            continue;
        }
        const auto uuid = vc::objects::util::canonicalUuid(patch.uuid);
        // Leased (on a hosted replica: taking it pulls), then pinned at the
        // revision this edit started from; its relationships are kept as
        // recorded. An unknown revision (its last save could not read it
        // back) is refused like a changed one: the user saves again knowingly.
        auto hold = session.holdLeases(attachment, {uuid});
        const vc::objects::RevisionSelection pins{{uuid, patch.revision}};
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto head = headRevision(db, txn, "brush patch", uuid);
            if (head != patch.revision)
                throw StoreObjectChanged(
                    "brush patch " + uuid + (patch.revision.empty()
                        ? std::string(" was saved, but its saved revision could not be read back; nothing was "
                                      "overwritten. Save again to replace the stored version with this drawing")
                        : std::string(" was saved by another editor after this drawing's last save; nothing was "
                                      "overwritten. Save again to replace their version with this drawing, or "
                                      "remove the drawing to keep theirs")),
                    uuid, head);
        });
        if (patch.remove) {
            session.act(attachment, pins, [&](auto& db, auto& txn) {
                if (const auto object = db.objectByUuid(txn, uuid); object && !object->retired) db.retireObject(txn, uuid);
            });
            saved.push_back({uuid, {}});
        } else {
            if (!patch.surface) throw std::invalid_argument("a replaced brush patch needs a surface");
            const auto geometry = quad_surface_geometry(*patch.surface);
            session.editKeepingRelationships(attachment, pins, {}, [&](auto& db, auto& txn, const auto& keep) {
                const auto current = db.currentGeometry(txn, uuid);
                db.putGeometry(txn, uuid, geometry, current ? std::optional(current->uuid) : std::nullopt,
                               keep(db.affectedGeometryEdges(txn, uuid, geometry)));
            });
            // Read under the lease: still this save's revision.
            saved.push_back({uuid, committedRevision(session, attachment, uuid)});
        }
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

void setPointCollectionObject(EditablePclDraft& draft, const std::string& uuid, const std::string& revision)
{
    QJsonObject metadata = draft.sourceCollection.value(QStringLiteral("metadata")).toObject();
    metadata[QStringLiteral("object_uuid")] = QString::fromStdString(uuid);
    metadata[QLatin1String(kLocalObjectRevisionKey)] = QString::fromStdString(revision);
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
    vc::objects::RevisionSelection expected;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        const auto object = pointCollectionObject(drafts[i]);
        if (object) saved[i].uuid = *object;
        if (object) {
            const std::string revision = drafts[i].sourceCollection.value(QStringLiteral("metadata")).toObject()
                                             .value(QLatin1String(kLocalObjectRevisionKey)).toString().toStdString();
            expected[*object] = revision;  // empty: unknown, refused below
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
    // The meta.source kept is read at the expected revision: the commit is
    // refused unless that is still current.
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& [uuid, revision] : expected) {
            const auto head = headRevision(db, txn, "point collection", uuid);
            if (head != *revision)
                throw StoreObjectChanged(revision->empty()
                    ? "point collection " + uuid + " was saved, but its saved revision could not be read back; "
                      "nothing was overwritten. Save again to replace the stored version"
                    : changedSince("point collection", uuid), uuid, head);
        }
        for (const auto& [index, uuid] : replacements) {
            const auto object = db.objectByUuid(txn, uuid);
            if (!object) throw std::runtime_error("point collection " + uuid + " is no longer in the object store");
            const auto meta = nlohmann::json::parse(object->meta.empty() ? std::string("{}") : object->meta);
            changed.push_back(storedPointCollection(drafts[index], uuid, meta.value("source", nlohmann::json::object())));
        }
    });
    vc::objects::writeVolumeCollections(session, attachment, vc::objects::VolumeVoxels{volume}, changed, removed, expected);
    // Each written head is this commit's: the objects are leased on a hosted
    // replica, and a new one is known to no one else yet.
    for (std::size_t i = 0; i < drafts.size(); ++i)
        if (!drafts[i].deleted) saved[i].revision = committedRevision(session, attachment, saved[i].uuid);
    return saved;
}

} // namespace vc3d::spiral
