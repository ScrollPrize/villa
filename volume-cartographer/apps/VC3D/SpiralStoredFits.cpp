#include "SpiralStoredFits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <stdexcept>

namespace vc3d::spiral {
namespace {

using Json = nlohmann::json;

bool committedFit(const vc::objects::ObjectRecord& record)
{
    return record.kind == "surface" && !record.retired
        && std::find(record.tags.begin(), record.tags.end(), kSpiralFitTag) != record.tags.end();
}

QJsonArray qtArray(const Json& value)
{
    return QJsonDocument::fromJson(QByteArray::fromStdString(
               Json{{"value", value}}.dump()))
        .object().value(QStringLiteral("value")).toArray();
}

}  // namespace

std::vector<StoredFitEntry> listStoredFits(vc::objects::ObjectStoreSession& session,
                                           const vc::objects::AttachmentIdentity& attachment)
{
    std::vector<StoredFitEntry> result;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& record : db.objectsByTag(txn, kSpiralFitTag)) {
            if (!committedFit(record)) continue;
            const Json meta = Json::parse(record.meta, nullptr, false);
            StoredFitEntry entry{record.uuid, record.name, record.createdAt, record.createdBy, std::nullopt};
            if (meta.is_object() && meta.value("source_fit_iteration", Json()).is_number_integer())
                entry.sourceIteration = meta["source_fit_iteration"].get<int64_t>();
            result.push_back(std::move(entry));
        }
    });
    std::sort(result.begin(), result.end(), [](const StoredFitEntry& a, const StoredFitEntry& b) {
        return a.createdAt != b.createdAt ? a.createdAt > b.createdAt : a.uuid < b.uuid;
    });
    return result;
}

StoredFit readStoredFit(const std::shared_ptr<vc::objects::ObjectStoreSession>& session,
                        const vc::objects::AttachmentIdentity& attachment, const std::string& uuid,
                        const std::optional<std::string>& geometryRevision,
                        std::chrono::milliseconds syncTimeout)
{
    auto present = [&]() {
        bool found = false;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            found = db.objectByUuid(txn, uuid).has_value();
        });
        return found;
    };
    if (!present() && session->hubStatus(attachment)) {
        const auto status = session->syncHubAndWait(attachment, syncTimeout);
        if (status && !status->error.empty())
            throw std::runtime_error("The committed fit is not in this replica, and syncing it failed: "
                                     + status->error);
    }
    if (!present()) throw std::runtime_error("The committed fit " + uuid + " is not in the attached object store");
    session->ensureGeometry(attachment, uuid);

    StoredFit fit;
    Json meta;
    session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto record = db.objectByUuid(txn, uuid);
        if (!record || !committedFit(*record))
            throw std::runtime_error(uuid + " is not a committed Spiral fit (a surface tagged spiral-fit)");
        meta = Json::parse(record->meta);
        fit.uuid = record->uuid;
        fit.name = record->name;
        const Json coordinates = vc::objects::geometryCoordinates(meta.value("coordinates", Json::object()));
        if (coordinates["volume_uuid"].is_string()) fit.volume = coordinates["volume_uuid"].get<std::string>();
        // Committed in its fit volume's own voxels: read as stored.
        if (fit.volume.empty() || vc::objects::geometryScaleToVolume(meta, fit.volume) != 1.0)
            throw std::runtime_error("The committed fit is not bound to its fit volume at scale 1");
        const auto revision = geometryRevision ? db.geometryRevision(txn, *geometryRevision)
                                               : db.currentGeometry(txn, uuid);
        if (!revision || revision->objectUuid != fit.uuid)
            throw std::runtime_error("The committed fit has no such geometry revision");
        fit.geometryRevision = revision->uuid;
        fit.surface = std::shared_ptr<QuadSurface>(load_quad_from_store(db, txn, uuid, revision->uuid));
    });
    fit.surface->setChannelLoader(store_channel_loader(session, attachment, fit.geometryRevision));

    const cv::Mat windings = fit.surface->channel("winding_id");
    const cv::Size grid = fit.surface->gridSize();
    if (windings.empty() || windings.type() != CV_32SC1 || windings.size() != grid)
        throw std::runtime_error("The committed fit has no int32 winding_id channel of its grid's size");
    fit.windingIds = windings.clone();

    QString error;
    auto components = parseWindingBounds(qtArray(meta.value("winding_bounds", Json::array())),
                                         qtArray(meta.value("component_winding_ids", Json::array())), &error);
    if (!components || !windingMapMatchesBounds(fit.windingIds, *components, &error))
        throw std::runtime_error(error.toStdString());
    for (const PreviewComponent& component : *components)
        if (component.rowEnd > grid.height || component.columnEnd > grid.width)
            throw std::runtime_error("Spiral winding bounds exceed the surface grid");
    fit.components = std::move(*components);
    return fit;
}

}  // namespace vc3d::spiral
